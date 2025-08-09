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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
from dotenv import load_dotenv
import cohere
from openai import OpenAI
import re
import numpy as np

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

# ============================================
# ENHANCED TEXT PROCESSING FOR PRECISE RETRIEVAL
# ============================================

def extract_numeric_facts(text: str) -> List[Dict]:
    """Extract explicit numeric facts from text"""
    patterns = [
        # Time periods
        (r'(\d+)\s*days?(?:\s+(?:grace\s+)?period)?', 'days'),
        (r'(\d+)\s*months?(?:\s+(?:waiting\s+)?period)?', 'months'),  
        (r'(\d+)\s*years?(?:\s+(?:waiting\s+)?period)?', 'years'),
        
        # Age ranges
        (r'(?:age\s+)?(\d+)(?:\s*[-–]\s*(\d+))?\s*years?', 'age_range'),
        
        # Coverage amounts
        (r'(?:up\s+to\s+)?(?:Rs\.?\s*|INR\s*)?(\d+(?:,\d+)*)', 'amount'),
        
        # Percentages  
        (r'(\d+)%', 'percentage'),
    ]
    
    facts = []
    for pattern, fact_type in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Get surrounding context (50 chars before/after)
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end].strip()
            
            facts.append({
                'value': match.group(),
                'type': fact_type,
                'context': context,
                'position': match.start()
            })
    
    return facts

def extract_keywords(text: str) -> List[str]:
    """Extract key terms from text"""
    keywords = []
    
    key_patterns = [
        r'\bgrace\s+period\b',
        r'\bwaiting\s+period\b', 
        r'\bmaternity\b',
        r'\bcataract\b',
        r'\bAYUSH\b',
        r'\bcoverage\b',
        r'\bexclusion\b',
        r'\bdeductible\b',
        r'\bco-?pay\b',
        r'\bsum\s+insured\b',
        r'\bpre-?existing\b'
    ]
    
    for pattern in key_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            keywords.append(pattern.strip('\\b'))
    
    return keywords

def enhanced_semantic_chunking(text: str) -> List[Dict]:
    """Enhanced chunking that preserves context and extracts key facts"""
    
    # Step 1: Extract numeric facts first
    numeric_facts = extract_numeric_facts(text)
    
    # Step 2: Create fact-aware chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,  # Smaller for precision
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", "; ", ", ", " "]
    )
    
    base_chunks = splitter.split_text(text)
    
    # Step 3: Enhance chunks with extracted facts
    enhanced_chunks = []
    for i, chunk in enumerate(base_chunks):
        # Find facts in this chunk
        chunk_facts = []
        for fact in numeric_facts:
            if fact['context'] in chunk or chunk in fact['context']:
                chunk_facts.append(fact)
        
        enhanced_chunks.append({
            'text': chunk,
            'chunk_id': i,
            'facts': chunk_facts,
            'has_numbers': bool(re.search(r'\d+', chunk)),
            'keywords': extract_keywords(chunk)
        })
    
    return enhanced_chunks

# ============================================
# IMPROVED RETRIEVAL SYSTEM
# ============================================

class PrecisionRetriever:
    """Retrieval system optimized for finding specific facts"""
    
    def __init__(self, cohere_client):
        self.co = cohere_client
        self.chunks = []
        
    def add_chunks(self, chunks: List[Dict]):
        """Add enhanced chunks to retriever"""
        self.chunks = chunks
        
        # Create embeddings
        chunk_texts = [chunk['text'] for chunk in chunks]
        response = self.co.embed(
            texts=chunk_texts,
            model="embed-english-v3.0",
            input_type="search_document"
        )
        
        for i, embedding in enumerate(response.embeddings):
            self.chunks[i]['embedding'] = embedding
    
    def retrieve_for_question(self, question: str, top_k: int = 5) -> List[Dict]:
        """Enhanced retrieval that prioritizes fact-containing chunks"""
        
        # Get question embedding
        question_response = self.co.embed(
            texts=[question],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        question_embedding = question_response.embeddings[0]
        
        # Calculate similarities
        similarities = []
        for chunk in self.chunks:
            # Base semantic similarity
            sim = self._cosine_similarity(question_embedding, chunk['embedding'])
            
            # Boost chunks with numbers if question asks for specific values
            if self._question_asks_for_number(question) and chunk['has_numbers']:
                sim *= 1.3
            
            # Boost chunks with relevant keywords
            question_lower = question.lower()
            for keyword in chunk['keywords']:
                if keyword.replace('\\b', '').replace('-?', '') in question_lower:
                    sim *= 1.2
            
            # Boost chunks with extracted facts
            if chunk['facts']:
                sim *= 1.1
            
            similarities.append((sim, chunk))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in similarities[:top_k]]
    
    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _question_asks_for_number(self, question: str) -> bool:
        """Check if question is asking for a specific numeric value"""
        numeric_question_patterns = [
            r'\bhow\s+many\b',
            r'\bhow\s+much\b', 
            r'\bhow\s+long\b',
            r'\bwhat\s+is\s+the\s+(?:period|duration|time|amount|limit)\b',
            r'\bdays?\b',
            r'\bmonths?\b',
            r'\byears?\b',
            r'\bamount\b',
            r'\blimit\b'
        ]
        
        return any(re.search(pattern, question, re.IGNORECASE) for pattern in numeric_question_patterns)

# ============================================
# ENHANCED ANSWER GENERATION
# ============================================

def generate_precise_answer(question: str, retrieved_chunks: List[Dict], openai_client) -> str:
    """Generate precise, fact-focused answers"""
    
    # Step 1: Extract the most relevant facts
    relevant_facts = []
    for chunk in retrieved_chunks:
        relevant_facts.extend(chunk.get('facts', []))
    
    # Step 2: Create focused context
    # Prioritize chunks with facts, then by similarity
    context_parts = []
    
    # First, add chunks with relevant facts
    fact_chunks = [chunk for chunk in retrieved_chunks if chunk.get('facts')]
    for chunk in fact_chunks[:2]:  # Top 2 fact-containing chunks
        context_parts.append(chunk['text'])
    
    # Then add other relevant chunks if needed
    remaining_chunks = [chunk for chunk in retrieved_chunks if not chunk.get('facts')]
    for chunk in remaining_chunks[:2]:  # Top 2 additional chunks
        context_parts.append(chunk['text'])
    
    context = "\n\n".join(context_parts)
    
    # Step 3: Create precise prompt
    prompt = f"""You are answering questions about insurance policy documents. Be precise and concise.

IMPORTANT INSTRUCTIONS:
1. Look for EXPLICIT numeric values (days, months, years, amounts) in the context
2. If you find a specific number related to the question, state it clearly
3. Do NOT say "not specified" if there's a clear value mentioned
4. Give the most direct answer possible
5. Only include essential details

Context:
{context}

Question: {question}

Answer (be precise and concise):"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": """You are a precise insurance policy assistant. Extract specific facts and numbers from the context. 
                    Never say 'not specified' if there's a clear value mentioned. Be concise but accurate."""
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.1,  # Low temperature for consistency
            max_tokens=150  # Force conciseness
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

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

# ============================================
# ENHANCED MAIN PROCESSING FUNCTION
# ============================================

def process_questions_with_model(document_text: str, questions: List[str]) -> List[str]:
    """Enhanced processing pipeline for better accuracy"""
    if index is None:
        return ["Pinecone index not available"] * len(questions)
    
    request_id = uuid.uuid4().hex[:8]
    
    # Step 1: Enhanced chunking with fact extraction
    enhanced_chunks = enhanced_semantic_chunking(document_text)
    
    # Step 2: Initialize precision retriever
    retriever = PrecisionRetriever(co)
    retriever.add_chunks(enhanced_chunks)
    
    # Step 3: Process each question
    answers = []
    
    try:
        # Store in Pinecone for backup retrieval if needed
        embeddings = [chunk['embedding'] for chunk in enhanced_chunks]
        pinecone_vectors = [
            (f"{request_id}-{i}", embedding, {"text": chunk['text']})
            for i, (chunk, embedding) in enumerate(zip(enhanced_chunks, embeddings))
        ]
        index.upsert(vectors=pinecone_vectors, namespace=request_id)
        
        for question in questions:
            try:
                # Step 4: Enhanced retrieval
                retrieved_chunks = retriever.retrieve_for_question(question, top_k=25)
                
                # Step 5: Generate precise answer
                answer = generate_precise_answer(question, retrieved_chunks, openai_client)
                answers.append(answer if answer else "Unable to generate answer")
                
            except Exception as e:
                answers.append(f"Error processing question: {str(e)}")
        
    finally:
        # Cleanup
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