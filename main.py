import fitz
import requests
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, HTTPException, Depends, status, APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
import io
import os
from datetime import datetime
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
from dotenv import load_dotenv
import cohere
from openai import OpenAI
import re
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Load environment variables
load_dotenv()

# API Keys
API_KEY = "bfb8fabaf1ce137c1402366fb3d5a052836234c1ff376c326842f52e3164cc33"
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Clients
co = cohere.Client(COHERE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Constants
INDEX_NAME = "pdf"
security = HTTPBearer()
app = FastAPI(title="HackRX LLM API", version="1.0.0")
router = APIRouter(prefix="/api/v1")

# === Models ===
class HackRXRequest(BaseModel):
    documents: str  # URL to PDF
    questions: List[str]

class HackRXResponse(BaseModel):
    answers: List[str]

# === Auth ===
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token

# === PDF Parsing ===
def extract_text_from_pdf_url(pdf_url: str) -> str:
    response = requests.get(pdf_url, timeout=30)
    response.raise_for_status()
    pdf_content = io.BytesIO(response.content)
    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        text = " ".join(page.get_text().replace("\n", " ") for page in doc)
        doc.close()
        if len(text.strip()) > 50:
            return text
    except Exception:
        pass
    print("Falling back to OCR for image-based PDF")
    images = convert_from_bytes(pdf_content.getvalue())
    return " ".join(pytesseract.image_to_string(img) for img in images)

# === Text Splitting ===
def semantic_chunk_text(text: str) -> List[str]:
    """
    Enhanced semantic chunking that respects document structure and content boundaries
    """
    # Step 1: Clean and normalize text
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Step 2: Identify and preserve document structure markers
    structure_patterns = [
        # Headers and sections
        r'(?i)(?=\b(?:section|clause|article|chapter|part|subsection|paragraph|appendix)\s+[IVX\d]+[:.)])',
        r'(?i)(?=\b(?:section|clause|article|chapter|part|subsection|paragraph|appendix)\s+\d+[:.)])',
        r'(?=^\d+\.\s+[A-Z])',  # "1. Introduction"
        r'(?=^\d+\.\d+\s+[A-Z])',  # "1.1 Overview"
        r'(?=^[A-Z][A-Z\s]{2,}:)',  # "TERMS AND CONDITIONS:"
        r'(?=^[A-Z][^.!?]{5,50}\n)',  # Standalone headings
        
        # Legal document patterns
        r'(?i)(?=\bwhereas\b)',  # Contract clauses
        r'(?i)(?=\bnow,?\s+therefore\b)',  # Legal transitions
        r'(?i)(?=\bin\s+witness\s+whereof\b)',  # Contract endings
        r'(?i)(?=\bfor\s+the\s+avoidance\s+of\s+doubt\b)',  # Clarifications
        
        # Numbered/lettered lists
        r'(?=^\([a-z]\)\s+[A-Z])',  # "(a) First item"
        r'(?=^\([IVX]+\)\s+[A-Z])',  # "(i) Roman numerals"
        r'(?=^[a-z]\)\s+[A-Z])',  # "a) Item"
        
        # Special document sections
        r'(?i)(?=\b(?:definitions?|terms?|scope|purpose|background|summary|conclusion|recommendations?)\b:)',
        r'(?i)(?=\b(?:exhibit|schedule|attachment|addendum)\s+[A-Z\d])',
    ]
    
    # Combine all patterns
    combined_pattern = '|'.join(structure_patterns)
    
    # Step 3: Split on semantic boundaries
    semantic_sections = []
    if combined_pattern:
        sections = re.split(f'({combined_pattern})', text, flags=re.MULTILINE)
        # Filter out empty sections and combine split markers with following content
        current_section = ""
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # If this looks like a header/marker, start new section
            if re.match(combined_pattern, section, flags=re.MULTILINE | re.IGNORECASE):
                if current_section:
                    semantic_sections.append(current_section.strip())
                current_section = section
            else:
                current_section += " " + section if current_section else section
        
        # Add final section
        if current_section:
            semantic_sections.append(current_section.strip())
    else:
        semantic_sections = [text]
    
    # Step 4: Enhanced text splitter with better parameters
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Optimal size for semantic search
        chunk_overlap=150,  # Good context preservation
        separators=[
            "\n\n",    # Paragraph breaks (highest priority)
            "\n",      # Line breaks
            ". ",      # Sentence endings
            "; ",      # Clause separators
            ", ",      # Comma separations
            " ",       # Word boundaries
            ""         # Character level (last resort)
        ],
        length_function=len,
        is_separator_regex=False,
    )
    
    # Step 5: Process sections and create final chunks
    final_chunks = []
    
    for i, section in enumerate(semantic_sections):
        section = section.strip()
        
        # Skip very short sections
        if len(section) < 30:
            continue
        
        # If section is small enough, keep as single chunk
        if len(section) <= 800:
            final_chunks.append(section)
        else:
            # Split large sections while preserving context
            section_chunks = splitter.split_text(section)
            
            # Add section context to chunks (except first one which already has it)
            for j, chunk in enumerate(section_chunks):
                if j == 0:
                    final_chunks.append(chunk)
                else:
                    # Add a brief context from section start for continuity
                    section_start = section[:100] + "..." if len(section) > 100 else section
                    contextual_chunk = f"[Continued from: {section_start}] {chunk}"
                    final_chunks.append(contextual_chunk)
    
    # Step 6: Post-processing - merge very small chunks with neighbors
    processed_chunks = []
    i = 0
    
    while i < len(final_chunks):
        current_chunk = final_chunks[i]
        
        # If chunk is very small, try to merge with next one
        if len(current_chunk) < 200 and i + 1 < len(final_chunks):
            next_chunk = final_chunks[i + 1]
            merged = current_chunk + " " + next_chunk
            
            # Only merge if result isn't too large
            if len(merged) <= 1000:
                processed_chunks.append(merged)
                i += 2  # Skip next chunk as it's been merged
                continue
        
        processed_chunks.append(current_chunk)
        i += 1
    
    # Step 7: Final cleanup and validation
    final_processed_chunks = []
    for chunk in processed_chunks:
        chunk = chunk.strip()
        
        # Remove chunks that are too short or just whitespace/punctuation
        if len(chunk) >= 30 and not re.match(r'^[^\w]*$', chunk):
            # Clean up any residual formatting issues
            chunk = re.sub(r'\s+', ' ', chunk)
            final_processed_chunks.append(chunk)
    
    return final_processed_chunks


# === Embeddings ===
def preprocess(text):
    return text.replace("\n", " ").strip().lower()

def get_cohere_embeddings(texts: List[str]) -> List[List[float]]:
    preprocessed_texts = [preprocess(t) for t in texts]
    response = co.embed(
        texts=preprocessed_texts,
        model="embed-english-v3.0",
        input_type="search_document"
    )
    return response.embeddings


# === LLM Response using OpenAI ===
def ask_openai(question: str, context_chunks: List[str]) -> str:
    context = "\n\n".join(context_chunks)
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant who answers my questions precisely in one sentence."},
            {"role": "user", "content": f"The user asked a question based on a document. Use only the context below to answer.\n\nContext:\n{context}\n\nQuestion:\n{question}"}
        ]
        response = openai_client.chat.completions.create(
            model="gpt-5",
            messages=messages,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# === Pinecone Setup ===
def init_pinecone(index_name, api_key):
    pc = Pinecone(api_key=api_key)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1024,  # updated to match Cohere v3.0 dimensions
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    return pc.Index(index_name)

try:
    index = init_pinecone(INDEX_NAME, PINECONE_API_KEY)
except Exception as e:
    print(f"Pinecone connection error: {e}")
    index = None

# === Processing ===
def process_questions_with_model(document_text: str, questions: List[str]) -> List[str]:
    if index is None:
        return ["Pinecone index not available"] * len(questions)
    request_id = uuid.uuid4().hex[:8]
    chunks = semantic_chunk_text(document_text)
    embeddings = get_cohere_embeddings(chunks)
    pinecone_vectors = [
        (f"{request_id}-{i}", vec, {"text": chunks[i]})
        for i, vec in enumerate(embeddings)
    ]
    index.upsert(vectors=pinecone_vectors, namespace=request_id)
    answers = []
    for question in questions:
        try:
            query_embedding = get_cohere_embeddings([question])[0]
            results = index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True,
                namespace=request_id
            )
            context_chunks = [match['metadata']['text'] for match in results['matches']]
            answer = ask_openai(question, context_chunks)
            answers.append(answer if answer else "Unable to generate answer")
        except Exception as e:
            answers.append(f"Error processing question: {str(e)}")
    try:
        index.delete(delete_all=True, namespace=request_id)
    except:
        pass
    return answers

# === API Routes ===
@router.post("/hackrx/run", response_model=HackRXResponse)
async def process_hackrx_request(request: HackRXRequest, token: str = Depends(verify_token)):
    try:
        document_text = extract_text_from_pdf_url(request.documents)
        answers = process_questions_with_model(document_text, request.questions)
        return HackRXResponse(answers=answers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@router.get("/")
async def root():
    return {"message": "HackRX LLM API is running"}

# === Include router ===
app.include_router(router)

# === Entry Point ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
