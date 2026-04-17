"""ChromaDB vector store manager for the RAG pipeline.

Handles document loading, text splitting, embedding, and retrieval.
Uses a module-level singleton to avoid re-initializing the embedding
model on every request.
"""

import os
import pathlib

from dotenv import load_dotenv
from typing import Optional

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.logger import logger

load_dotenv()

# Module-level singleton instance (lazy-initialized)
_instance: Optional["VectorStoreManager"] = None


def get_vector_store_manager() -> "VectorStoreManager":
    """Returns the singleton VectorStoreManager instance.

    Avoids re-initializing the embedding model on every call.
    The instance is created once and reused for the process lifetime.

    Returns:
        The shared VectorStoreManager instance.
    """
    global _instance
    if _instance is None:
        _instance = VectorStoreManager()
    return _instance


class VectorStoreManager:
    """Manages the ChromaDB vector store lifecycle: creation, loading, and retrieval.

    Attributes:
        persist_directory: File path where the ChromaDB database is stored.
        data_folder: Path to the knowledge documents folder.
        embedding_model: The HuggingFace embedding model used for vectorization.
        text_splitter: Splits documents into chunks for better retrieval quality.
        vector_store: The active ChromaDB vector store instance, if loaded.
    """

    def __init__(self, data_folder: str = "app/data/knowledge") -> None:
        self.persist_directory = str(
            pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "chroma_db"
        )
        self.data_folder = data_folder
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            length_function=len,
        )
        self.vector_store: Optional[Chroma] = None

    def get_retriever(self):
        """Returns a retriever backed by the vector store, auto-building if needed.

        If the ChromaDB folder is empty or missing, triggers an automatic
        build from the knowledge documents folder.

        Returns:
            A LangChain retriever configured to return the top 3 results.
        """
        if not os.path.exists(self.persist_directory) or not os.listdir(
            self.persist_directory
        ):
            logger.info("Database folder is empty! Triggering automatic build...")
            self.create_and_load_db()

        if self.vector_store is None:
            logger.info("Connecting to existing Vector Store...")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model,
            )

        retriever_k = int(os.getenv("RETRIEVER_K", "3"))
        return self.vector_store.as_retriever(search_kwargs={"k": retriever_k})

    def create_and_load_db(self) -> Optional[Chroma]:
        """Loads documents from the knowledge folder, splits them, and builds the vector store.

        Documents are split into chunks using ``RecursiveCharacterTextSplitter``
        (1000 chars with 200 overlap) before embedding. This ensures better
        retrieval quality compared to indexing whole documents.

        Returns:
            The created Chroma vector store, or None if no documents were found.
        """
        if not os.path.exists(self.data_folder):
            logger.warning(f"Data folder {self.data_folder} not found")
            os.makedirs(self.data_folder, exist_ok=True)
            logger.info(
                "Folder created! Please put some .pdf or .md files "
                "inside it and try again"
            )
            return None

        logger.info(f"Loading data from {self.data_folder}...")

        pdf_loader = DirectoryLoader(
            self.data_folder, glob="**/*.pdf", loader_cls=PyPDFLoader
        )
        md_loader = DirectoryLoader(
            self.data_folder, glob="**/*.md", loader_cls=TextLoader
        )
        txt_loader = DirectoryLoader(
            self.data_folder, glob="**/*.txt", loader_cls=TextLoader
        )

        raw_documents = []
        raw_documents.extend(pdf_loader.load())
        raw_documents.extend(md_loader.load())
        raw_documents.extend(txt_loader.load())

        if not raw_documents:
            logger.warning("No documents found to index!")
            return None

        # Split documents into chunks for better retrieval quality
        documents = self.text_splitter.split_documents(raw_documents)
        logger.info(
            f"Loaded {len(raw_documents)} documents → "
            f"split into {len(documents)} chunks. Creating vector store..."
        )

        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory,
        )

        logger.success("Vector Store created and saved securely")
        return self.vector_store
