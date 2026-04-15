import re
import time 
import uuid
import httpx
import logging
import warnings
from typing import List, Dict, Any, Optional, Callable

# Filter out noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer
import fitz

from chroma_store import ChromaStore
from logging_config import logger
from config import settings

class RAGEngine:
    """
    Complete RAG Engine for document indexing and querying.
    Manages the full lifecycle: extraction → chunking → embedding → storage → retrieval → generation.
    """

    def __init__(
        self,
        chroma_path: str = settings.CHROMA_PATH,
        model_name: str = settings.EMBEDDING_MODEL_NAME,
        llm_model: str = settings.OLLAMA_MODEL,
        llm_base_url: str = settings.OLLAMA_BASE_URL
    ):
        """
        Initialize the RAG engine with its components.
        """
        self.chroma_store = ChromaStore(persist_directory=chroma_path)
        try:
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Embedding model '{model_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

        self.embedding_model_name = model_name
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.chunk_size = 500
        self.chunk_overlap = 50

    def count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text (~4 chars per token).
        """
        return len(text) // 4

    def clean_text(self, text: str) -> str:
        """
        Clean extracted text to improve embedding quality.
        """
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

    def chunk_text(self, text: str, doc_name: str) -> List[Dict[str, Any]]:
        """
        Split text into intelligent chunks, respecting paragraph boundaries.
        """
        text = self.clean_text(text)
        chunks = []
        
        paragraphs = text.split('\n')
        current_chunk = ""
        
        for p in paragraphs:
            if self.count_tokens(current_chunk + p) < self.chunk_size:
                current_chunk += p + "\n"
            else:
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "chunk_index": len(chunks),
                        "document_name": doc_name,
                        "token_count": self.count_tokens(current_chunk)
                    })
                
                if self.count_tokens(p) > self.chunk_size:
                    words = p.split()
                    for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
                        chunk_text = " ".join(words[i:i + self.chunk_size])
                        chunks.append({
                            "text": chunk_text,
                            "chunk_index": len(chunks),
                            "document_name": doc_name,
                            "token_count": self.count_tokens(chunk_text)
                        })
                    current_chunk = ""
                else:
                    current_chunk = p + "\n"
        
        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "chunk_index": len(chunks),
                "document_name": doc_name,
                "token_count": self.count_tokens(current_chunk)
            })

        return chunks

    def extract_text(self, file_path: str, file_type: str) -> str:
        """
        Extract raw text from PDF or TXT files.
        """
        if file_type.lower() == "pdf":
            text_parts = []
            try:
                doc = fitz.open(file_path)
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    if text.strip():
                        text_parts.append(f"[Page {page_num + 1}]\n{text}")
                doc.close()
                return "\n\n".join(text_parts)
            except Exception as e:
                logger.error(f"PDF extraction error: {e}")
                raise ValueError(f"Failed to read PDF: {e}")
        elif file_type.lower() == "txt":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"TXT extraction error: {e}")
                raise ValueError(f"Failed to read text file: {e}")
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def index_document(
        self,
        file_path: str,
        file_name: str,
        file_type: str
    ) -> Dict[str, Any]:
        """
        Index a complete document: extraction → chunking → embedding → storage.
        """
        start_time = time.time()
        logger.info(f"Indexing document: {file_name}")

        try:
            text = self.extract_text(file_path, file_type)
            logger.debug(f"Extracted {len(text)} characters")

            chunks = self.chunk_text(text, file_name)
            if not chunks:
                raise ValueError("No chunks generated from document")
            logger.debug(f"Generated {len(chunks)} chunks")

            texts_for_embedding = [c["text"] for c in chunks]
            embeddings = self.embedding_model.encode(
                texts_for_embedding,
                show_progress_bar=False,
                convert_to_numpy=True
            ).tolist()

            ids = [f"{file_name}_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "document_name": c["document_name"],
                    "chunk_index": c["chunk_index"],
                    "token_count": c["token_count"],
                    "file_type": file_type
                }
                for c in chunks
            ]

            success = self.chroma_store.add_documents(
                ids=ids,
                embeddings=embeddings,
                documents=texts_for_embedding,
                metadatas=metadatas
            )

            if not success:
                raise RuntimeError("Failed to store documents in ChromaDB")

            total_time = int((time.time() - start_time) * 1000)
            logger.info(f"Successfully indexed {file_name} in {total_time}ms")
            
            return {
                "document_name": file_name,
                "chunks_count": len(chunks),
                "total_time_ms": total_time,
                "all_chunks": [
                    {
                        "text": c["text"],
                        "index": c["chunk_index"],
                        "token_count": c["token_count"]
                    }
                    for c in chunks
                ],
                "char_count": len(text)
            }
        except Exception as e:
            logger.error(f"Failed to index {file_name}: {e}")
            raise

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a user query."""
        return self.embedding_model.encode(query, convert_to_numpy=True).tolist()

    def retrieve_chunks(
        self,
        query_embedding: List[float],
        n_chunks: int = 5,
        document_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for most similar chunks."""
        where_filter = {"document_name": document_name} if document_name else None
        results = self.chroma_store.search(
            query_embedding=query_embedding,
            n_results=n_chunks,
            where=where_filter
        )

        chunks = []
        if results["ids"] and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                chunks.append({
                    "chunk_text": results["documents"][0][i],
                    "document": results["metadatas"][0][i].get("document_name", "unknown"),
                    "score": 1 - results["distances"][0][i],
                    "chunk_index": results["metadatas"][0][i].get("chunk_index", 0),
                    "chunk_id": results["ids"][0][i],
                    "token_count": results["metadatas"][0][i].get("token_count", 0),
                    "metadata": results["metadatas"][0][i]
                })

        return chunks

    def generate_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Construct the prompt for the LLM."""
        context = "\n\n".join([
            f"[Document: {c['document']}, Chunk {c['chunk_index'] + 1}]\n{c['chunk_text']}"
            for c in chunks
        ])

        return f"""Tu es un assistant expert en analyse de documents. Réponds à la question en te basant UNIQUEMENT sur les documents fournis ci-dessous.

Si l'information n'est pas dans les documents, dis-le clairement.

---

CONTEXTE:
{context}

---

QUESTION: {query}

---

RÉPONSE (en français):"""

    def generate_with_ollama(self, prompt: str) -> str:
        """Generate response via Ollama."""
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{self.llm_base_url}/api/generate",
                    json={
                        "model": self.llm_model,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                response.raise_for_status()
                return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise RuntimeError(f"Ollama error: {e}")

    def chat(
        self,
        question: str,
        n_chunks: int = 5,
        document_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute full RAG pipeline."""
        pipeline_steps = []
        start_time = time.time()

        def add_step(step: str, status: str, duration_ms: int, **extra):
            pipeline_steps.append({"step": step, "duration_ms": duration_ms, "status": status, **extra})

        # Step 1: Embedding Query
        step_start = time.time()
        query_embedding = self.embed_query(question)
        add_step("embedding_query", "done", int((time.time() - step_start) * 1000))

        # Step 2: Retrieval
        step_start = time.time()
        chunks = self.retrieve_chunks(query_embedding, n_chunks, document_name)
        add_step("retrieval", "done", int((time.time() - step_start) * 1000), chunks_found=len(chunks))

        if not chunks:
            logger.warning(f"No context found for query: {question}")
            return {
                "question": question,
                "answer": "Aucun document pertinent trouvé.",
                "sources": [],
                "pipeline_steps": pipeline_steps,
                "raw_prompt": "",
                "query_embedding_sample": query_embedding[:10],
                "total_tokens_approx": 0
            }

        # Step 3: Generation
        step_start = time.time()
        prompt = self.generate_prompt(question, chunks)
        answer = self.generate_with_ollama(prompt)
        add_step("generation", "done", int((time.time() - step_start) * 1000))

        logger.info(f"Chat completed in {int((time.time() - start_time) * 1000)}ms")
        
        return {
            "question": question,
            "answer": answer,
            "sources": chunks,
            "pipeline_steps": pipeline_steps,
            "raw_prompt": prompt,
            "query_embedding_sample": query_embedding[:10],
            "total_tokens_approx": self.count_tokens(prompt) + self.count_tokens(answer)
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        return self.chroma_store.get_all_documents()

    def delete_document(self, document_name: str) -> bool:
        return self.chroma_store.delete_by_document_name(document_name)

    def get_system_stats(self) -> Dict[str, Any]:
        return {
            "embedding_model": self.embedding_model_name,
            "llm_model": self.llm_model,
            "total_chunks": self.chroma_store.get_collection_count()
        }
