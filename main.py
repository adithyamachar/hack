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
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Step 1: Split on strong semantic boundaries (headings, clause labels, numbered sections)
    semantic_sections = re.split(
        r'(?i)(?=\b(section|clause|article)\s+\d+[:.)])', 
        text
    )
    
    # Step 2: Further split sections if too long, but preserve sentence boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", "; "]
    )
    
    final_chunks = []
    for section in semantic_sections:
        if len(section.strip()) < 50:
            continue
        final_chunks.extend(splitter.split_text(section.strip()))
    
    return final_chunks

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
            model="gpt-4o",
            messages=messages,
            temperature=0.2
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
