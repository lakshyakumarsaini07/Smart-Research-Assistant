"""
Vector database creation and querying utilities
"""
import os
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS, Chroma
try:
    # Preferred import from langchain_huggingface (new)
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback to community import (deprecated)
    from langchain_community.embeddings import HuggingFaceEmbeddings

from app.utils.config import VECTOR_DB_PATH


class VectorStore:
    """
    Class for managing the vector store operations
    """

    def __init__(self, use_chroma: bool = False):
        """
        Initialize the vector store

        Args:
            use_chroma: Whether to use Chroma DB instead of FAISS
        """
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # ✅ Local embedding model
        self.use_chroma = use_chroma
        self.vector_store = None

    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Create a vector store from documents

        Args:
            documents: List of document chunks
        """
        if not documents:
            raise ValueError("No documents to embed. Skipping vector store creation.")

        if self.use_chroma:
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=VECTOR_DB_PATH
            )
        else:
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            # Save the vector store
            os.makedirs(VECTOR_DB_PATH, exist_ok=True)
            self.vector_store.save_local(VECTOR_DB_PATH)

    def load_vector_store(self) -> None:
        """
        Load an existing vector store
        """
        if self.use_chroma:
            self.vector_store = Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=self.embeddings
            )
        else:
            self.vector_store = FAISS.load_local(
                folder_path=VECTOR_DB_PATH,
                embeddings=self.embeddings
            )

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform a similarity search

        Args:
            query: The query string
            k: Number of documents to return

        Returns:
            List of relevant document chunks
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store or load_vector_store first.")
        return self.vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple[Document, float]]:
        """
        Perform a similarity search with relevance scores

        Args:
            query: The query string
            k: Number of documents to return

        Returns:
            List of tuples containing document chunks and their relevance scores
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store or load_vector_store first.")
        return self.vector_store.similarity_search_with_score(query, k=k)
