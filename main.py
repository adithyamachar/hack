import fitz
import requests
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, HTTPException, Depends, status, APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Tuple
import io
import os
from datetime import datetime
import uuid
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
from dotenv import load_dotenv
import cohere
from openai import OpenAI
import spacy
from collections import Counter

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

# Load spaCy model for sentence segmentation
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

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

# === Enhanced Text Processing ===
class EnhancedTextProcessor:
    def __init__(self):
        # Common header/footer patterns
        self.header_footer_patterns = [
            r'^\d+\s*$',  # Page numbers only
            r'^Page \d+ of \d+\s*$',  # "Page X of Y"
            r'^\d+\s*/\s*\d+\s*$',  # "X/Y" page format
            r'^©.*\d{4}.*$',  # Copyright notices
            r'^.*confidential.*$',  # Confidential headers
            r'^.*proprietary.*$',  # Proprietary notices
            r'^\s*www\.[^\s]+\s*$',  # Website URLs
            r'^\s*https?://[^\s]+\s*$',  # Full URLs
            r'^.*\d{4}-\d{2}-\d{2}.*$',  # Dates in headers
        ]
        
        # OCR error correction patterns
        self.ocr_corrections = {
            r'\bl\s+(\w)': r'I\1',  # "l etter" -> "letter"
            r'\b0(\w)': r'O\1',     # "0ption" -> "Option"
            r'rn': 'm',             # "rn" -> "m"
            r'(\w)\s+([.,;:])': r'\1\2',  # Remove space before punctuation
            r'([.!?])\s*([a-z])': r'\1 \2',  # Ensure space after sentence endings
            r'\s{3,}': ' ',         # Multiple spaces to single space
            r'([a-z])([A-Z])': r'\1 \2',  # Add space between camelCase
        }
    
    def remove_headers_footers(self, pages_text: List[str]) -> List[str]:
        """Remove common headers and footers from each page"""
        cleaned_pages = []
        
        for page_text in pages_text:
            lines = page_text.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if line matches header/footer patterns
                is_header_footer = False
                for pattern in self.header_footer_patterns:
                    if re.match(pattern, line, re.IGNORECASE):
                        is_header_footer = True
                        break
                
                if not is_header_footer:
                    cleaned_lines.append(line)
            
            cleaned_pages.append('\n'.join(cleaned_lines))
        
        return cleaned_pages
    
    def clean_ocr_artifacts(self, text: str) -> str:
        """Clean common OCR errors and artifacts"""
        for pattern, replacement in self.ocr_corrections.items():
            text = re.sub(pattern, replacement, text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def preserve_structure(self, text: str) -> str:
        """Preserve document structure markers"""
        # Mark headings (lines that are all caps or start with numbers/bullets)
        lines = text.split('\n')
        structured_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                structured_lines.append(line)
                continue
            
            # Detect headings
            if (line.isupper() and len(line) > 3) or \
               re.match(r'^\d+\.?\s+[A-Z]', line) or \
               re.match(r'^[A-Z][^.!?]*$', line):
                structured_lines.append(f"HEADING: {line}")
            # Detect bullet points
            elif re.match(r'^[•·‣▪▫◦‣]\s+', line) or \
                 re.match(r'^[-*+]\s+', line) or \
                 re.match(r'^\d+\)\s+', line) or \
                 re.match(r'^[a-z]\)\s+', line):
                structured_lines.append(f"BULLET: {line}")
            # Detect numbered lists
            elif re.match(r'^\d+\.\s+', line):
                structured_lines.append(f"LIST_ITEM: {line}")
            else:
                structured_lines.append(line)
        
        return '\n'.join(structured_lines)
    
    def extract_tables(self, doc) -> Dict[int, List[str]]:
        """Extract table content from PDF"""
        tables_by_page = {}
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            tables = page.find_tables()
            
            if tables:
                page_tables = []
                for table in tables:
                    try:
                        table_data = table.extract()
                        # Convert table to structured text
                        table_text = "TABLE:\n"
                        for row in table_data:
                            if row:  # Skip empty rows
                                table_text += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
                        page_tables.append(table_text)
                    except:
                        continue
                
                if page_tables:
                    tables_by_page[page_num] = page_tables
        
        return tables_by_page

# === Enhanced PDF Parsing ===
def extract_text_from_pdf_url(pdf_url: str) -> str:
    response = requests.get(pdf_url, timeout=30)
    response.raise_for_status()
    pdf_content = io.BytesIO(response.content)
    
    processor = EnhancedTextProcessor()
    
    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        pages_text = []
        tables_by_page = processor.extract_tables(doc)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            
            # Add table content if exists
            if page_num in tables_by_page:
                page_text += "\n" + "\n".join(tables_by_page[page_num])
            
            pages_text.append(page_text)
        
        doc.close()
        
        # Remove headers/footers
        cleaned_pages = processor.remove_headers_footers(pages_text)
        
        # Combine all pages
        full_text = " ".join(cleaned_pages)
        
        # Clean OCR artifacts and preserve structure
        full_text = processor.clean_ocr_artifacts(full_text)
        full_text = processor.preserve_structure(full_text)
        
        if len(full_text.strip()) > 50:
            return full_text
            
    except Exception as e:
        print(f"PDF extraction error: {e}")
    
    # Fallback to OCR
    print("Falling back to OCR for image-based PDF")
    pdf_content.seek(0)
    images = convert_from_bytes(pdf_content.getvalue())
    ocr_text = " ".join(pytesseract.image_to_string(img) for img in images)
    
    # Clean OCR text
    ocr_text = processor.clean_ocr_artifacts(ocr_text)
    ocr_text = processor.preserve_structure(ocr_text)
    
    return ocr_text

# === Semantic Chunking ===
class SemanticChunker:
    def __init__(self, chunk_size=600, chunk_overlap=250):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy if available, otherwise use regex"""
        if nlp:
            doc = nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Fallback regex-based sentence splitting
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text_semantically(self, text: str) -> List[str]:
        """Create semantic chunks that respect document structure"""
        # Split into paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if this is a structural element
            is_heading = paragraph.startswith("HEADING:")
            is_bullet = paragraph.startswith("BULLET:")
            is_list = paragraph.startswith("LIST_ITEM:")
            is_table = paragraph.startswith("TABLE:")
            
            # For structural elements, try to keep them with following content
            if is_heading and current_chunk:
                # Finish current chunk before starting new section
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_size = 0
            
            paragraph_size = len(paragraph)
            
            # If adding this paragraph would exceed chunk size
            if current_size + paragraph_size > self.chunk_size and current_chunk:
                # Try to split paragraph by sentences
                sentences = self.split_by_sentences(paragraph)
                
                if len(sentences) > 1:
                    # Add sentences one by one
                    for sentence in sentences:
                        sentence_size = len(sentence)
                        
                        if current_size + sentence_size > self.chunk_size and current_chunk:
                            chunks.append(current_chunk.strip())
                            # Create overlap
                            overlap_text = self._create_overlap(current_chunk)
                            current_chunk = overlap_text + " " + sentence
                            current_size = len(current_chunk)
                        else:
                            current_chunk += " " + sentence
                            current_size += sentence_size + 1
                else:
                    # Single sentence too long, force split
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        overlap_text = self._create_overlap(current_chunk)
                        current_chunk = overlap_text + " " + paragraph
                        current_size = len(current_chunk)
                    else:
                        current_chunk = paragraph
                        current_size = paragraph_size
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n" + paragraph
                    current_size += paragraph_size + 1
                else:
                    current_chunk = paragraph
                    current_size = paragraph_size
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 20]
    
    def _create_overlap(self, text: str) -> str:
        """Create overlap from the end of previous chunk"""
        if len(text) <= self.chunk_overlap:
            return text
        
        # Try to find a good breaking point (sentence boundary)
        overlap_start = len(text) - self.chunk_overlap
        
        # Look for sentence boundaries
        for i in range(overlap_start, len(text)):
            if text[i] in '.!?':
                next_char_idx = i + 1
                if next_char_idx < len(text) and text[next_char_idx] == ' ':
                    return text[next_char_idx + 1:]
        
        # Fallback to character-based overlap
        return text[-self.chunk_overlap:]

# === Updated Text Splitting Function ===
def chunk_text(text: str) -> List[str]:
    """Enhanced chunking using semantic chunker"""
    chunker = SemanticChunker(chunk_size=650, chunk_overlap=250)
    return chunker.chunk_text_semantically(text)

# === Embeddings ===
def preprocess(text):
    # Enhanced preprocessing
    text = text.replace("\n", " ").strip()
    # Remove structure markers for embedding (but keep original for context)
    text = re.sub(r'(HEADING|BULLET|LIST_ITEM|TABLE):\s*', '', text)
    return text.lower()

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
            {
                "role": "system", 
                "content": """You are a helpful assistant who answers questions precisely based on the provided context. 
                Pay attention to document structure markers like HEADING:, BULLET:, LIST_ITEM:, and TABLE: to understand the content better.
                Provide accurate, concise answers in 1-2 sentences."""
            },
            {
                "role": "user", 
                "content": f"The user asked a question based on a document. Use only the context below to answer.\n\nContext:\n{context}\n\nQuestion:\n{question}"
            }
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
    chunks = chunk_text(document_text)
    
    print(f"Generated {len(chunks)} semantic chunks")
    
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