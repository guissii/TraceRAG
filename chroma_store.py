import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import os
from logging_config import logger

class ChromaStore:
    """
    Wrapper class for ChromaDB to handle vector storage and retrieval.
    """

    def __init__(self, persist_directory: str = "./chroma_data"):
        """
        Initialize ChromaDB connection.

        Args:
            persist_directory: Path to store ChromaDB data.
        """
        self.persist_directory = persist_directory
        try:
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = self.client.get_or_create_collection(
                name="rag_documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"ChromaDB initialized at {persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> bool:
        """
        Add documents with embeddings to ChromaDB.

        Args:
            ids: List of unique IDs for each chunk.
            embeddings: Corresponding embedding vectors.
            documents: Original text of each chunk.
            metadatas: Metadata (filename, chunk index, etc.).

        Returns:
            True if addition was successful, False otherwise.
        """
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.debug(f"Successfully added {len(ids)} chunks to ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            return False

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for documents most similar to a query.

        Args:
            query_embedding: Embedding vector of the search query.
            n_results: Number of results to return.
            where: Optional metadata filter.

        Returns:
            Dictionary containing results, distances, and metadata.
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "distances", "metadatas"]
            )
            logger.debug(f"Found {len(results.get('ids', [[]])[0])} results in ChromaDB")
            return results
        except Exception as e:
            logger.error(f"Error during search in ChromaDB: {e}")
            return {"ids": [], "documents": [], "distances": [], "metadatas": []}

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        List all indexed documents with statistics.

        Returns:
            List of documents and their chunk counts.
        """
        try:
            results = self.collection.get(include=["metadatas"])
            if not results["metadatas"]:
                return []
            
            docs = {}
            for meta in results["metadatas"]:
                doc_name = meta["document_name"]
                if doc_name not in docs:
                    docs[doc_name] = {
                        "document_name": doc_name,
                        "chunk_count": 0,
                        "file_type": meta.get("file_type", "unknown")
                    }
                docs[doc_name]["chunk_count"] += 1
            
            return list(docs.values())
        except Exception as e:
            logger.error(f"Error getting documents: {e}")
            return []

    def delete_by_document_name(self, document_name: str) -> bool:
        """
        Delete a document and all its chunks from the index.

        Args:
            document_name: Name of the document to delete.

        Returns:
            True if deletion was successful.
        """
        try:
            self.collection.delete(where={"document_name": document_name})
            logger.info(f"Deleted document: {document_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {document_name}: {e}")
            return False

    def get_collection_count(self) -> int:
        """Get the total number of chunks in the collection."""
        return self.collection.count()
