import fitz
import openai
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

# Load environment variables from .env file
load_dotenv()

# Load API Keys securely from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
API_KEY = os.getenv("API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize app and API router
app = FastAPI(title="HackRX LLM API", version="1.0.0")
router = APIRouter(prefix="/api/v1")

# Security
security = HTTPBearer()
INDEX_NAME = "pdf"

# Request/Response Models
class HackRXRequest(BaseModel):
    documents: str  # URL to PDF
    questions: List[str]

class HackRXResponse(BaseModel):
    answers: List[str]

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
    text = " ".join(pytesseract.image_to_string(img) for img in images)
    return text

def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def get_openai_embeddings(texts: List[str]) -> List[List[float]]:
    response = openai.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [e.embedding for e in response.data]

def ask_openai(question: str, context_chunks: List[str]) -> str:
    context_text = "\n\n".join(context_chunks[:3])
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Answer only using the given context. Reply in one sentence."},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion:\n{question}"}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def init_pinecone(index_name, api_key):
    pc = Pinecone(api_key=api_key)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    return pc.Index(index_name)

try:
    index = init_pinecone(INDEX_NAME, PINECONE_API_KEY)
except Exception as e:
    print(f"Pinecone connection error: {e}")
    index = None

def process_questions_with_model(document_text: str, questions: List[str]) -> List[str]:
    if index is None:
        return ["Pinecone index not available"] * len(questions)
    request_id = uuid.uuid4().hex[:8]
    chunks = chunk_text(document_text)
    embeddings = get_openai_embeddings(chunks)
    pinecone_vectors = [
        (f"{request_id}-{i}", vec, {"text": chunks[i]})
        for i, vec in enumerate(embeddings)
    ]
    index.upsert(vectors=pinecone_vectors, namespace=request_id)
    answers = []
    for question in questions:
        try:
            query_embedding = get_openai_embeddings([question])[0]
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

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
