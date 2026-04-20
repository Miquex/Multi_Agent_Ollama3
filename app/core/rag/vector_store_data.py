"""ChromaDB vector store manager for the RAG pipeline.

Handles document loading, text splitting, embedding, reranking, and retrieval.
Uses a module-level singleton to avoid re-initializing the embedding
model on every request.
"""

import os
import pathlib

from dotenv import load_dotenv
from typing import Optional

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
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


class RerankedRetriever(BaseRetriever):
    """Two-stage retriever: embedding similarity then cross-encoder reranking.

    Stage 1 fetches a broad candidate pool via cosine similarity.
    Stage 2 scores each candidate with a cross-encoder model that
    computes full attention between the query and chunk text, keeping
    only the ``top_n`` highest-scoring results.

    Attributes:
        base_retriever: The underlying embedding-based retriever.
        cross_encoder: The HuggingFace cross-encoder model for scoring.
        top_n: Number of documents to keep after reranking.
    """

    base_retriever: BaseRetriever
    cross_encoder: HuggingFaceCrossEncoder
    top_n: int = 3

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Retrieves and reranks documents for the given query.

        Args:
            query: The user's search query.
            run_manager: Callback manager provided by LangChain.

        Returns:
            The top-N most relevant documents after cross-encoder scoring.
        """
        # Stage 1: broad embedding retrieval
        candidates = self.base_retriever.invoke(query)
        if not candidates:
            return []

        # Stage 2: cross-encoder reranking
        pairs = [(query, doc.page_content) for doc in candidates]
        scores: list[float] = self.cross_encoder.score(pairs)

        scored = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )
        reranked = [doc for _, doc in scored[: self.top_n]]

        logger.debug(
            f"Reranker: {len(candidates)} candidates → "
            f"top {len(reranked)} (scores: "
            f"{[f'{s:.3f}' for s, _ in scored[:self.top_n]]})"
        )
        return reranked


class VectorStoreManager:
    """Manages the ChromaDB vector store lifecycle: creation, loading, and retrieval.

    Attributes:
        persist_directory: File path where the ChromaDB database is stored.
        data_folder: Path to the knowledge documents folder.
        embedding_model: The HuggingFace embedding model used for vectorization.
        cross_encoder: Cross-encoder model for second-stage relevance scoring.
        reranker_top_n: Number of chunks to keep after reranking.
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

        # Cross-encoder for two-stage retrieval
        reranker_model_name = os.getenv(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        logger.info(f"Loading cross-encoder reranker: {reranker_model_name}")
        self.cross_encoder = HuggingFaceCrossEncoder(
            model_name=reranker_model_name,
        )
        self.reranker_top_n = int(os.getenv("RERANKER_TOP_N", "3"))

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            length_function=len,
        )
        self.vector_store: Optional[Chroma] = None

    def get_retriever(self) -> RerankedRetriever:
        """Returns a two-stage retriever: broad embedding search then cross-encoder rerank.

        Stage 1 retrieves a broad candidate pool (``RETRIEVER_K`` chunks)
        via cosine similarity. Stage 2 passes those candidates through
        a cross-encoder model that computes full attention between the
        query and each chunk, keeping only the ``RERANKER_TOP_N`` best.

        If the ChromaDB folder is empty or missing, triggers an automatic
        build from the knowledge documents folder.

        Returns:
            A ``RerankedRetriever`` that yields the top reranked results.
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

        retriever_k = int(os.getenv("RETRIEVER_K", "10"))
        base_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": retriever_k}
        )

        logger.info(
            f"Retriever pipeline: fetch top-{retriever_k} → "
            f"rerank to top-{self.reranker_top_n}"
        )
        return RerankedRetriever(
            base_retriever=base_retriever,
            cross_encoder=self.cross_encoder,
            top_n=self.reranker_top_n,
        )

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
            self.data_folder,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,  # type: ignore[arg-type]
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
