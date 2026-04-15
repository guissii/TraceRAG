import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_engine import RAGEngine
from config import settings
from logging_config import setup_logging, logger

# Initialize logging
setup_logging()

rag_engine: Optional[RAGEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan manager for the FastAPI application.
    Initializes the RAG Engine on startup.
    """
    global rag_engine
    try:
        rag_engine = RAGEngine()
        logger.info(f"RAG Engine initialized successfully (Model: {settings.OLLAMA_MODEL})")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Engine: {e}")
        # We don't raise here to allow the app to start and show errors in health check
    yield
    logger.info("RAG Explorer shutting down")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Static files
STATIC_DIR = Path(__file__).parent / "frontend"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Models
class ChatRequest(BaseModel):
    question: str
    n_chunks: int = 5
    document_name: Optional[str] = None

class SourceChunk(BaseModel):
    chunk_text: str
    document: str
    score: float
    chunk_index: int
    token_count: Optional[int] = 0
    metadata: Optional[dict] = None

class PipelineStep(BaseModel):
    step: str
    duration_ms: int
    status: str
    chunks_found: Optional[int] = None

class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceChunk]
    pipeline_steps: List[PipelineStep]
    raw_prompt: str
    query_embedding_sample: List[float]
    total_tokens_approx: int

class DocumentInfo(BaseModel):
    document_name: str
    chunk_count: int
    file_type: str

class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    llm_model: str
    total_chunks: int

class ChunkDetail(BaseModel):
    text: str
    index: int
    token_count: int

class IndexResponse(BaseModel):
    document_name: str
    chunks_count: int
    char_count: int
    total_time_ms: int
    message: str
    all_chunks: List[ChunkDetail]

# Endpoints
@app.get("/")
async def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    raise HTTPException(status_code=404, detail="Frontend not found")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG Engine not initialized")
    
    stats = rag_engine.get_system_stats()
    return HealthResponse(
        status="healthy",
        embedding_model=stats["embedding_model"],
        llm_model=stats["llm_model"],
        total_chunks=stats["total_chunks"]
    )

@app.post("/upload", response_model=IndexResponse)
async def upload_document(file: UploadFile = File(...)):
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG Engine not initialized")

    allowed_types = ["application/pdf", "text/plain"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")

    file_ext = "pdf" if file.content_type == "application/pdf" else "txt"
    original_name = file.filename or f"doc_{uuid.uuid4().hex[:8]}.{file_ext}"

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, original_name)

    try:
        content = await file.read()
        char_count = len(content)
        await file.seek(0)
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = rag_engine.index_document(
            file_path=temp_path,
            file_name=original_name,
            file_type=file_ext
        )

        return IndexResponse(
            document_name=result["document_name"],
            chunks_count=result["chunks_count"],
            char_count=char_count,
            total_time_ms=result["total_time_ms"],
            message=f"Successfully indexed {result['chunks_count']} chunks",
            all_chunks=[ChunkDetail(**c) for c in result["all_chunks"]]
        )
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    if not rag_engine:
        return []
    return [DocumentInfo(**doc) for doc in rag_engine.list_documents()]

@app.delete("/documents/{document_name}")
async def delete_document(document_name: str):
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG Engine not initialized")
    
    success = rag_engine.delete_document(document_name)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete document")
    return {"message": f"Document {document_name} deleted"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG Engine not initialized")

    try:
        result = rag_engine.chat(
            question=request.question,
            n_chunks=request.n_chunks,
            document_name=request.document_name
        )
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import uuid # Needed for filename generation
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
