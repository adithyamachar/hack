import fitz
import numpy as np
import cohere
import requests
from pinecone import Pinecone, ServerlessSpec  # v3 import
from fastapi import FastAPI, HTTPException, Depends, status, APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
import io
import os
from datetime import datetime
import uuid

# Initialize app and API router with prefix
app = FastAPI(title="HackRX LLM API", version="1.0.0")
router = APIRouter(prefix="/api/v1")

# Security
security = HTTPBearer()
PINECONE_API_KEY = "pcsk_7B3Z93_8WBKxheRs5H22N8LeMJTCWzjPR1wUZKE8oUJzHDyhMot6qbZ1JrfSkKM7kcLVu7"
INDEX_NAME = "pdf"

# Request/Response Models
class HackRXRequest(BaseModel):
    documents: str  # URL to PDF
    questions: List[str]

class HackRXResponse(BaseModel):
    answers: List[str]

API_KEY = "bfb8fabaf1ce137c1402366fb3d5a052836234c1ff376c326842f52e3164cc33"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token

def extract_text_from_pdf_url(pdf_url: str) -> str:
    response = requests.get(pdf_url, timeout=30)
    response.raise_for_status()
    pdf_content = io.BytesIO(response.content)
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    text = " ".join(page.get_text().replace("\n", " ") for page in doc)
    doc.close()
    return text

# Initialize Cohere Client
co = cohere.Client("ba9VI3VW1sXTxyIKhOZHWPA3326tAQzHGVVQ16aI")

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Initialize Pinecone (v3 style)
def init_pinecone(index_name, api_key):
    pc = Pinecone(api_key=api_key)
    if index_name not in pc.list_indexes().names():
        print(f"Creating index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print(f"Index {index_name} created successfully")
    return pc.Index(index_name)

try:
    index = init_pinecone(INDEX_NAME, PINECONE_API_KEY)
    print(f"Connected to Pinecone index: {INDEX_NAME}")
except Exception as e:
    print(f"Pinecone connection error: {e}")
    index = None

def ask_perplexity(query, context_chunks):
    api_key = "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"
    url = "https://api.perplexity.ai/chat/completions"
    short_chunks = [" ".join(chunk.split()[:100]) for chunk in context_chunks]
    context_text = "\n\n".join(short_chunks)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Strictly answer the user's question in only one sentence. Do not provide explanations or extra information and dont cite your answers"},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print("Error:", response.status_code, response.text)
        return None

def process_questions_with_model(document_text: str, questions: List[str]) -> List[str]:
    try:
        if index is None:
            return [f"Pinecone index not available"] * len(questions)
        request_id = uuid.uuid4().hex[:8]
        chunks = chunk_text(document_text)
        response = co.embed(
            texts=chunks,
            model="embed-english-v3.0",
            input_type="search_document"
        )
        embeddings = response.embeddings
        pinecone_vectors = [
            (f"{request_id}-{i}", vec, {"text": chunks[i]}) 
            for i, vec in enumerate(embeddings)
        ]
        index.upsert(vectors=pinecone_vectors, namespace=request_id)
        answers = []
        for question in questions:
            try:
                query_response = co.embed(
                    texts=[question],
                    model="embed-english-v3.0",
                    input_type="search_query"
                )
                query_vec = query_response.embeddings[0]
                results = index.query(
                    vector=query_vec, 
                    top_k=1, 
                    include_metadata=True,
                    namespace=request_id
                )
                context_chunks = [match['metadata']['text'] for match in results['matches']]
                answer = ask_perplexity(question, context_chunks)
                answers.append(answer if answer else "Unable to generate answer")
            except Exception as e:
                answers.append(f"Error processing question: {str(e)}")
        try:
            index.delete(delete_all=True, namespace=request_id)
        except:
            pass
        return answers
    except Exception as e:
        return [f"Error processing document: {str(e)}"] * len(questions)

# Route definitions under router
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

# Include router
app.include_router(router)

# Uvicorn runner for local development
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
