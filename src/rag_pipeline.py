import time
from pathlib import Path
from typing import List, Dict, Any

from src.components.documents_loader import DocumentsLoader
from src.components.vector_store import VectorStore
from src.components.rag_chain import RAGChain
from src.utils.logger import logger
from config.settings import settings

class RAGPipeline:

    def __init__(self):
        self.documents_loader = DocumentsLoader()
        self.vector_store = VectorStore(settings)
        self.rag_chain = None
        self.is_initialized = False

        logger.info("RAG Pipeline initialized")

    def initialize(self, force_rebuild: bool = False) -> None:
        logger.info("Starting RAG pipeline initialization")
        start_time = time.time()

        settings.validate()

        if not force_rebuild and self._load_existing_index():
            logger.info("Using existing index")
        else:
            logger.info("Building new index")
            self._build_new_index()

        self.rag_chain = RAGChain(self.vector_store)
        self.is_initialized = True

        total_time = time.time() - start_time
        logger.info(f"RAG pipeline took {total_time:.2} seconds to initialize")

    def _load_existing_index(self) -> bool:
        try:
            return self.vector_store.load_index(
                settings.FAISS_INDEX_PATH,
                settings.METADATA_PATH
            )
        except Exception as e:
            logger.error(f"Failed to load existing index: {e}")
            return False

    def _build_new_index(self) -> None:
        logger.info("Building new index")

        documents = self.documents_loader.load_and_chunk(settings.DATA_PATH)

        if not documents:
            raise ValueError("No docs loaded. Check your data dir")

        self.vector_store.create_index(documents)

        self.vector_store.save_index(
            settings.FAISS_INDEX_PATH,
            settings.METADATA_PATH
        )

    def ask_question(self, question: str) -> Dict[str, Any]:
        if not self.is_initialized:
            raise RuntimeError("RAG pipeline is not initialized, call initialize() first")

        return self.rag_chain.ask(question)

    def ask_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        if not self.is_initialized:
            raise RuntimeError("RAG pipeline is not initialized, call initialize() first")

        return self.rag_chain.batch_ask(questions)

    def get_info(self) -> Dict[str, Any]:
        if not self.is_initialized:
            return {"status": "not_initialized"}

        return {
            "status": "initialized",
            "total_documents": len(self.vector_store.documents),
            "index_size": self.vector_store.index.ntotal if self.vector_store.index else 0,
            "data_path": str(settings.DATA_PATH),
            "storage_path": str(settings.STORAGE_PATH),
            "embedding_model": settings.EMBEDDING_MODEL,
            "chat_model": settings.CHAT_MODEL,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "top_k_results:": settings.TOP_K_RESULTS
        }

    def rebuild_index(self) -> None:
        logger.info("Rebuilding index")
        self._build_new_index()

        if self.rag_chain:
            self.rag_chain = RAGChain(self.vector_store)

        logger.info("Index rebuilt successfully")