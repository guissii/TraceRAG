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
    
    Pipeline éducatif en 4 étapes :
      1. Embedding Query   — Conversion de la question en vecteur sémantique
      2. Retrieval         — Recherche des chunks les plus similaires dans ChromaDB
      3. Augmentation      — Construction du prompt augmenté avec le contexte
      4. Generation        — Inférence LLM pour générer la réponse finale
    """

    def __init__(
        self,
        chroma_path: str = settings.CHROMA_PATH,
        model_name: str = settings.EMBEDDING_MODEL_NAME,
        llm_model: str = settings.OLLAMA_MODEL,
        llm_base_url: str = settings.OLLAMA_BASE_URL
    ):
        self.chroma_store = ChromaStore(persist_directory=chroma_path)
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedding_model = SentenceTransformer(model_name, device=device)
            logger.info(f"Embedding model '{model_name}' loaded successfully on {device.upper()}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

        self.embedding_model_name = model_name
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP

    def count_tokens(self, text: str) -> int:
        """Estimate token count using word-based approximation (~0.75 words per token)."""
        words = text.split()
        return int(len(words) * 1.3)

    def count_words(self, text: str) -> int:
        """Count actual words in text."""
        return len(text.split())

    def clean_text(self, text: str) -> str:
        """Clean extracted text: normalize whitespace, remove artifacts."""
        text = re.sub(r'\x00', '', text)
        text = re.sub(r'\f', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' *\n *', '\n', text)
        return text.strip()

    def chunk_text(self, text: str, doc_name: str) -> List[Dict[str, Any]]:
        """
        Split text into semantic chunks, respecting paragraph and sentence boundaries.
        Uses word count for sizing, with overlap for context continuity.
        """
        text = self.clean_text(text)
        if not text:
            return []

        chunks = []
        paragraphs = re.split(r'\n\n+', text)
        current_chunk = ""
        current_page = "1"
        chunk_page = "1"

        def _get_page(t, default):
            matches = re.findall(r'\[Page (\d+)\]', t)
            return matches[-1] if matches else default

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_page = _get_page(para, current_page)
            if not current_chunk.strip():
                chunk_page = _get_page(para, current_page)

            candidate = (current_chunk + "\n\n" + para).strip() if current_chunk else para

            if self.count_words(candidate) <= self.chunk_size:
                current_chunk = candidate
                current_page = para_page
            else:
                if current_chunk.strip():
                    chunks.append({
                        "text": re.sub(r'\[Page \d+\]\s*', '', current_chunk).strip(),
                        "chunk_index": len(chunks),
                        "document_name": doc_name,
                        "token_count": self.count_tokens(current_chunk),
                        "page_number": chunk_page
                    })

                if self.count_words(para) > self.chunk_size:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    current_chunk = ""
                    for sent in sentences:
                        sent_page = _get_page(sent, para_page)
                        if self.count_words(current_chunk + " " + sent) <= self.chunk_size:
                            if not current_chunk:
                                chunk_page = sent_page
                            current_chunk = (current_chunk + " " + sent).strip()
                        else:
                            if current_chunk:
                                chunks.append({
                                    "text": re.sub(r'\[Page \d+\]\s*', '', current_chunk).strip(),
                                    "chunk_index": len(chunks),
                                    "document_name": doc_name,
                                    "token_count": self.count_tokens(current_chunk),
                                    "page_number": chunk_page
                                })
                            current_chunk = sent
                            chunk_page = sent_page
                    current_page = para_page
                else:
                    if chunks:
                        prev_words = chunks[-1]["text"].split()
                        overlap_text = " ".join(prev_words[-self.chunk_overlap:]) if len(prev_words) > self.chunk_overlap else ""
                        current_chunk = (overlap_text + "\n\n" + para).strip() if overlap_text else para
                        chunk_page = _get_page(current_chunk, chunks[-1].get("page_number", "1"))
                    else:
                        current_chunk = para
                        chunk_page = para_page
                    current_page = para_page

        if current_chunk.strip():
            chunks.append({
                "text": re.sub(r'\[Page \d+\]\s*', '', current_chunk).strip(),
                "chunk_index": len(chunks),
                "document_name": doc_name,
                "token_count": self.count_tokens(current_chunk),
                "page_number": chunk_page
            })

        logger.debug(f"Chunked '{doc_name}' into {len(chunks)} segments (avg {sum(c['token_count'] for c in chunks)//max(len(chunks),1)} tokens/chunk)")
        return chunks

    def extract_text(self, file_path: str, file_type: str) -> str:
        """Extract raw text from PDF, TXT, or DOCX files."""
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
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"TXT extraction error: {e}")
                raise ValueError(f"Failed to read text file: {e}")
        elif file_type.lower() == "docx":
            try:
                import docx
                doc = docx.Document(file_path)
                return "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            except ImportError:
                logger.error("python-docx is not installed")
                raise ValueError("DOCX extraction requires 'python-docx' to be installed.")
            except Exception as e:
                logger.error(f"DOCX extraction error: {e}")
                raise ValueError(f"Failed to read docx file: {e}")
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def index_document(
        self,
        file_path: str,
        file_name: str,
        file_type: str,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Index a complete document through the 4-step ingestion pipeline:
          Step 1: Extraction  — Read raw text from file
          Step 2: Chunking    — Split into semantic segments  
          Step 3: Embedding   — Convert chunks to vectors
          Step 4: Storage     — Store in ChromaDB
        
        Returns detailed timing and chunk data for the frontend.
        """
        start_time = time.time()
        step_timings = {}
        logger.info(f"Indexing document: {file_name}")

        try:
            # Remove existing version if re-uploading
            existing = self.chroma_store.get_all_documents()
            for doc in existing:
                if doc["document_name"] == file_name:
                    logger.info(f"Document '{file_name}' already exists, replacing...")
                    self.chroma_store.delete_by_document_name(file_name)
                    break

            # Step 1: Extraction
            step_start = time.time()
            text = self.extract_text(file_path, file_type)
            if not text or len(text.strip()) < 10:
                raise ValueError("Document is empty or contains no extractable text")
            step_timings["extraction"] = int((time.time() - step_start) * 1000)
            logger.info(f"Step 1 - Extracted {len(text)} characters from {file_name} ({step_timings['extraction']}ms)")

            if progress_callback:
                progress_callback("extraction", 25, step_timings["extraction"])

            # Step 2: Chunking
            step_start = time.time()
            chunks = self.chunk_text(text, file_name)
            if not chunks:
                raise ValueError("No chunks generated - document may be empty")
            step_timings["chunking"] = int((time.time() - step_start) * 1000)
            logger.info(f"Step 2 - Generated {len(chunks)} chunks ({step_timings['chunking']}ms)")

            if progress_callback:
                progress_callback("chunking", 50, step_timings["chunking"])

            # Step 3: Embedding (batch with progress tracking)
            step_start = time.time()
            texts_for_embedding = [c["text"] for c in chunks]
            
            # Batch embedding for large documents
            batch_size = 32
            all_embeddings = []
            for i in range(0, len(texts_for_embedding), batch_size):
                batch = texts_for_embedding[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                ).tolist()
                all_embeddings.extend(batch_embeddings)
                
                # Report progress within embedding step
                embed_progress = min(75, 50 + int(25 * (i + len(batch)) / len(texts_for_embedding)))
                if progress_callback:
                    progress_callback("embedding", embed_progress, None)

            step_timings["embedding"] = int((time.time() - step_start) * 1000)
            logger.info(f"Step 3 - Generated {len(all_embeddings)} embeddings ({step_timings['embedding']}ms)")

            if progress_callback:
                progress_callback("embedding", 75, step_timings["embedding"])

            # Step 4: Storage in ChromaDB
            step_start = time.time()
            ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "document_name": c["document_name"],
                    "chunk_index": c["chunk_index"],
                    "token_count": c["token_count"],
                    "file_type": file_type,
                    "page_number": c.get("page_number", "Inconnu")
                }
                for c in chunks
            ]

            success = self.chroma_store.add_documents(
                ids=ids,
                embeddings=all_embeddings,
                documents=texts_for_embedding,
                metadatas=metadatas
            )

            if not success:
                raise RuntimeError("Failed to store documents in ChromaDB")

            step_timings["storage"] = int((time.time() - step_start) * 1000)
            logger.info(f"Step 4 - Stored in ChromaDB ({step_timings['storage']}ms)")

            if progress_callback:
                progress_callback("storage", 100, step_timings["storage"])

            total_time = int((time.time() - start_time) * 1000)
            logger.info(f"Successfully indexed {file_name}: {len(chunks)} chunks in {total_time}ms")

            return {
                "document_name": file_name,
                "chunks_count": len(chunks),
                "total_time_ms": total_time,
                "step_timings": step_timings,
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
        """Generate embedding vector for a user query."""
        return self.embedding_model.encode(
            query, 
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()

    def retrieve_chunks(
        self,
        query_embedding: List[float],
        n_chunks: int = 5,
        document_name: Optional[str] = None,
        min_score: float = 0.10
    ) -> List[Dict[str, Any]]:
        """Search for most similar chunks using cosine similarity."""
        where_filter = {"document_name": document_name} if document_name else None
        results = self.chroma_store.search(
            query_embedding=query_embedding,
            n_results=n_chunks,
            where=where_filter
        )

        chunks = []
        if results["ids"] and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                score = 1 - results["distances"][0][i]
                if score >= min_score:
                    chunks.append({
                        "chunk_text": results["documents"][0][i],
                        "document": results["metadatas"][0][i].get("document_name", "unknown"),
                        "score": score,
                        "chunk_index": results["metadatas"][0][i].get("chunk_index", 0),
                        "chunk_id": results["ids"][0][i],
                        "token_count": results["metadatas"][0][i].get("token_count", 0),
                        "metadata": results["metadatas"][0][i]
                    })

        return chunks

    def generate_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Construct a structured prompt with retrieved context for the LLM.
        This is the AUGMENTATION step of the RAG pipeline.
        Uses numbered reference markers [1],[2]... for traceable citations.
        """
        context_parts = []
        ref_legend = []
        for i, c in enumerate(chunks):
            ref_num = i + 1
            score_pct = f"{c.get('score', 0)*100:.1f}%"
            page_info = c.get('metadata', {}).get('page_number', 'Inconnu')
            doc_name = c['document']
            chunk_idx = c.get('chunk_index', '?')
            context_parts.append(
                f"[{ref_num}] Document: {doc_name} | Page: {page_info} | Chunk #{chunk_idx} (score: {score_pct})\n"
                f"{c['chunk_text']}"
            )
            ref_legend.append(
                f"[{ref_num}] {doc_name} — Page {page_info} — Chunk #{chunk_idx}"
            )
        context = "\n\n".join(context_parts)
        legend = "\n".join(ref_legend)

        return (
            "Tu es un assistant specialise dans l'analyse de documents. REGLES STRICTES:\n"
            "1. Reponds UNIQUEMENT avec les informations du CONTEXTE ci-dessous.\n"
            "2. Apres CHAQUE phrase ou affirmation, ecris le numero de reference [N] correspondant.\n"
            "3. A la fin, ecris OBLIGATOIREMENT:\n"
            "   === SOURCES UTILISEES ===\n"
            "   [N] Nom_du_document — Page X\n"
            "4. Reponds en francais.\n\n"
            f"=== CONTEXTE ===\n\n{context}\n\n"
            f"=== TABLE DES REFERENCES ===\n{legend}\n\n"
            f"=== QUESTION ===\n{query}\n\n"
            "=== REPONSE ==="
        )

    def generate_with_ollama(self, prompt: str, max_retries: int = 1) -> str:
        """Generate response via Ollama with retry logic and optimized parameters."""
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                with httpx.Client(timeout=180.0) as client:
                    response = client.post(
                        f"{self.llm_base_url}/api/generate",
                        json={
                            "model": self.llm_model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.2,
                                "top_p": 0.9,
                                "num_ctx": 4096,
                                "repeat_penalty": 1.1,
                                "num_predict": 1024
                            }
                        }
                    )
                    response.raise_for_status()
                    result = response.json().get("response", "")
                    if not result.strip():
                        return "Le modele n'a pas genere de reponse. Verifiez que le contexte contient des informations pertinentes."
                    return result.strip()
            except httpx.ConnectError as e:
                last_error = e
                logger.error(f"Cannot connect to Ollama (attempt {attempt+1}/{max_retries+1})")
                if attempt < max_retries:
                    time.sleep(1)
            except httpx.TimeoutException as e:
                last_error = e
                logger.error(f"Ollama timeout (attempt {attempt+1}/{max_retries+1})")
                if attempt < max_retries:
                    time.sleep(2)
            except Exception as e:
                logger.error(f"Ollama generation error: {e}")
                return f"[Erreur] Generation impossible: {str(e)}"

        if isinstance(last_error, httpx.ConnectError):
            return "[Erreur] Impossible de se connecter a Ollama. Lancez 'ollama serve' dans un terminal."
        elif isinstance(last_error, httpx.TimeoutException):
            return "[Erreur] Le modele a mis trop de temps a repondre. Reessayez avec une question plus courte."
        return f"[Erreur] {str(last_error)}"

    def chat(
        self,
        question: str,
        n_chunks: int = 5,
        document_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete RAG pipeline in 4 measured steps:
          1. Embedding Query   — Convert question to vector
          2. Retrieval         — Find similar chunks
          3. Augmentation      — Build the augmented prompt
          4. Generation        — LLM inference
        """
        pipeline_steps = []
        start_time = time.time()

        # Normalize query for better semantic matching
        question = question.strip().lower()

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
                "answer": "Aucun document pertinent trouve. Veuillez d'abord indexer un document dans l'etape 1 (Ingestion).",
                "sources": [],
                "pipeline_steps": pipeline_steps,
                "raw_prompt": "",
                "query_embedding_sample": query_embedding[:10],
                "total_tokens_approx": 0
            }

        # Step 3: Augmentation (prompt construction)
        step_start = time.time()
        prompt = self.generate_prompt(question, chunks)
        augmentation_time = int((time.time() - step_start) * 1000)
        add_step("augmentation", "done", augmentation_time, 
                 prompt_tokens=self.count_tokens(prompt),
                 context_chunks=len(chunks))

        # Step 4: Generation (LLM inference)
        step_start = time.time()
        answer = self.generate_with_ollama(prompt)
        add_step("generation", "done", int((time.time() - step_start) * 1000))

        total_time = int((time.time() - start_time) * 1000)
        logger.info(f"Chat completed in {total_time}ms (4 steps)")
        
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
            "total_chunks": self.chroma_store.get_collection_count(),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }
