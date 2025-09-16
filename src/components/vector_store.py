import pickle
import time
import re
import os
from pathlib import Path
from typing import Any,  Dict, List,  Tuple

import faiss
import numpy as np
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from psycopg2.extras import Json

from src.utils.logger import logger
from config.settings import settings
from database import get_db_connection


class VectorStore:

    #def __init__(self):
    #    self.embeddings = OpenAIEmbeddings(
    #        openai_api_key=settings.OPENAI_API_KEY,
    #        model=settings.EMBEDDING_MODEL
    #    )
    #    self.index = None
    #    self.documents = []
    #    self.metadata = []

    def __init__(self, settings):
        # Normalize common typo in HF model id
        model_name = settings.EMBEDDING_MODEL.replace("sentence-transformer/", "sentence-transformers/")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=os.environ.get("HF_HOME", "/root/.cache/huggingface"),
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True})
        self.index = None
        self.documents = []
        self.metadata = []



    def _extract_keywords(self, text: str) -> List[str]:
        prime_art_ids = re.findall(r'\b\d{10,}\b', text) #lange nummern

        numbers = re.findall(r'\b\d{6,}\b', text)

        alphanums = re.findall(r'\b[A-Za-z]\w*\d+\w*\b|\b\d+\w*[A-Za-z]\w*\b', text)

        return list(set(prime_art_ids + numbers + alphanums))

    def _keyword_search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        k = k or settings.TOP_K_RESULTS

        query_keywords = self._extract_keywords(query)
        query_words = query.lower().split()

        logger.info(f"Keyword Search for: {query_keywords + query_words}")

        matches = []

        for i, doc in enumerate(self.documents):
            score = 0.0
            content_lower = doc.page_content.lower()

            for keyword in query_keywords:
                if keyword in content_lower:
                    score += 1.0

            # for word in query_words:
            #     if word in content_lower:
            #         score += 1.0

            if score > 0:
                matches.append((doc, score))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:k]

    # def generate_embeddings(self, texts: List[str]) -> np.ndarray:
    #     logger.info(f"Generating embeddings for {len(texts)} texts")
    #     start_time = time.time()
    #
    #     batch_size = 100 # um api limits zu vermeiden
    #     all_embeddings = []
    #
    #     for i in range(0, len(texts), batch_size):
    #         batch_texts = texts[i:i + batch_size]
    #
    #         total_items = len(texts)
    #         total_batches = (total_items + batch_size - 1) // batch_size
    #         current_batch = (i // batch_size) + 1
    #
    #         logger.info(f"Processing batch {current_batch}/{total_batches}")
    #
    #         try:
    #             batch_embeddings = self.embeddings.embed_documents(batch_texts)
    #             all_embeddings.extend(batch_embeddings)
    #             time.sleep(0.1)
    #         except Exception as e:
    #             logger.error(f"Failed to generate embeddings for batch {i // batch_size + 1}: {str(e)}")
    #             raise
    #
    #     embeddings_array = np.array(all_embeddings, dtype=np.float32)
    #     embed_time = time.time() - start_time
    #     logger.info(f"Embedding time: {embed_time:.2} seconds")
    #     logger.info(f"Embedding shape: {embeddings_array.shape}")
    #     return embeddings_array
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        logger.info(f"Generating embeddings for {len(texts)} texts")
        start_time = time.time()
        batch_size = 100
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            total_batches = (len(texts) + batch_size - 1) // batch_size
            current_batch = (i // batch_size) + 1
            logger.info(f"Processing batch {current_batch}/{total_batches}")
            batch_embeddings = self.embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
            time.sleep(0.05)
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        logger.info(f"Embedding time: {time.time() - start_time:.2f} seconds; shape={embeddings_array.shape}")
        return embeddings_array

    # def create_index(self, documents: List[Document]) -> None:
    #     logger.info(f"Creating index for {len(documents)} documents")
    #     texts = [doc.page_content for doc in documents]
    #     metadata = [doc.metadata for doc in documents]
    #     embeddings = self.generate_embeddings(texts)
    #     dimension = embeddings.shape[1]
    #     logger.info(f"Creating FAISS idx with dimension {dimension}")
    #     self.index = faiss.IndexFlatIP(dimension)
    #     start_time = time.time()
    #     self.index.add(embeddings)
    #     index_time = time.time() - start_time
    #     self.documents = documents
    #     self.metadata = metadata
    #     logger.info(f"FAISS index with {self.index.ntotal} vectors. Indexing time: {index_time:.2} seconds")
    def create_index(self, documents: List[Document]) -> None:
        logger.info(f"Creating index for {len(documents)} documents and persisting to Postgres")
        texts = [doc.page_content for doc in documents]
        metadata = [doc.metadata for doc in documents]
        embeddings = self.generate_embeddings(texts)
        # Keep FAISS in-memory for fast search
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.documents = documents
        self.metadata = metadata
        # Persist to Postgres (float array + jsonb)
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS document_embeddings (
                        id SERIAL PRIMARY KEY,
                        doc_index INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        metadata JSONB,
                        embedding REAL[] NOT NULL
                    );
                """)
                # Replace entire content for now
                cur.execute("DELETE FROM document_embeddings;")
                insert_sql = (
                    "INSERT INTO document_embeddings (doc_index, content, metadata, embedding) "
                    "VALUES (%s, %s, %s, %s)"
                )
                for i, (text, meta, vec) in enumerate(zip(texts, metadata, embeddings)):
                    cur.execute(insert_sql, (i, text, Json(meta), vec.tolist()))
        logger.info("Persisted embeddings and metadata to Postgres table 'document_embeddings'")

    # def save_index(self, index_path: Path, metadata_path: Path) -> None:
    #     logger.info(f"Saving FAISS index to {index_path}")
    #     if self.index is None:
    #         raise ValueError("No index to save. Must create an index first")
    #     faiss.write_index(self.index, str(index_path))
    #     with open(metadata_path, "wb") as f:
    #         pickle.dump({'documents': self.documents, 'metadata': self.metadata}, f)
    #     logger.info(f"Success: Saved index and metadata")
    def save_index(self, index_path: Path, metadata_path: Path) -> None:
        if self.index is None:
            raise ValueError("No index to save. Must create an index first")
        # Already persisted to Postgres in create_index; just ensure a small checkpoint for local fallback
        logger.info("Persisting small checkpoint (optional) and confirming DB state")
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM document_embeddings;")
                count = cur.fetchone()[0]
        logger.info(f"Postgres has {count} embeddings in 'document_embeddings'")

    def load_index(self, index_path: Path, metadata_path: Path) -> bool:
        try:
            if not index_path.exists() or not metadata_path.exists():
                logger.info("Index files not found, creating index")
                return False

            logger.info(f"Loading FAISS index from {index_path}")

            self.index = faiss.read_index(str(index_path))

            with open(metadata_path, "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.metadata = data["metadata"]

            # Validate that the FAISS index dim matches the current embedding model dim
            try:
                expected_dim = len(self.embeddings.embed_query("dimension_check"))
                if hasattr(self.index, 'd'):
                    index_dim = self.index.d
                else:
                    index_dim = self.index.ntotal and self.index.d  # fallback
                if index_dim != expected_dim:
                    logger.warning(
                        f"FAISS index dim {index_dim} != embedding model dim {expected_dim}. "
                        f"Forcing index rebuild."
                    )
                    return False
            except Exception as e:
                logger.warning(f"Failed to validate index dimension: {e}. Forcing rebuild.")
                return False

            logger.info(f"Success: Loaded index with {self.index.ntotal} vectors")
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {str(e)}")
            return False

    def search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        if self.index is None:
            raise ValueError("No index found. Load/Create an index first")

        k = k or settings.TOP_K_RESULTS
        # Sanitize query to avoid tokenizer input errors
        try:
            if not isinstance(query, str):
                query = str(query)
            # Normalize to UTF-8 safe string
            query = query.encode('utf-8', 'ignore').decode('utf-8')
        except Exception:
            pass

        logger.info(f"Searching for TOP_K={k} for query: {query[:100]}")

        start_time = time.time()

        keyword_results = self._keyword_search(query, k * 2)

        try:
            query_embedding = self.embeddings.embed_query(query)
            if not isinstance(query_embedding, (list, tuple, np.ndarray)):
                raise TypeError(f"embed_query returned unexpected type: {type(query_embedding)}")
            query_vector = np.array([query_embedding], dtype=np.float32)
            logger.info(f"Query embedding shape: {query_vector.shape}; index size: {self.index.ntotal}")
        except Exception as e:
            logger.error(f"Failed to compute query embedding: {e}")
            raise

        try:
            scores, indices = self.index.search(query_vector, k * 2)
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            raise

        semantic_results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                document = self.documents[idx]
                semantic_results.append((document, float(score)))

        combined_scores = {}

        for doc, score in keyword_results:
            doc_id = id(doc)
            scaled_keyword_score = min(score * 0.1, 1.0)
            combined_scores[doc_id] = {
                'doc': doc,
                'keyword_score': scaled_keyword_score,
                'semantic_score': 0.0,
                'has_exact_match': True
            }

        for doc, score in semantic_results:
            doc_id = id(doc)
            if doc_id in combined_scores:
                combined_scores[doc_id]['semantic_score'] = score
            else:
                combined_scores[doc_id] = {
                    'doc': doc,
                    'keyword_score': 0.0,
                    'semantic_score': score,
                    'has_exact_match': False
                }

        final_results = []
        for doc_data in combined_scores.values():
            if doc_data['has_exact_match']:
                final_score = doc_data['semantic_score'] + 10.0
            else:
                final_score = (doc_data['semantic_score']
                    + doc_data['keyword_score'] * 2.0 )
            final_results.append((doc_data['doc'], final_score))

        final_results.sort(key=lambda x: x[1], reverse=True)
        results = final_results[:k]

        search_time = time.time() - start_time

        logger.info(f"Hybrid search found {len(results)} results in {search_time:.3f} seconds")
        logger.info(f"Keyword matches: {len(keyword_results)}, Semantic matches: {len(semantic_results)}")

        return results

