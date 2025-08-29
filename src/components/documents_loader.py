from pathlib import Path
from typing import List, Dict, Any
import time

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.utils.logger import logger
from config.settings import settings


class DocumentsLoader:

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_documents(self, data_path: Path) -> List[Document]:
        logger.info(f"Loading documents from {data_path}")

        pdf_files = list(data_path.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {data_path}")

        logger.info(f"Found {len(pdf_files)} PDF files in {data_path}")

        all_documents = []
        failed_files = []

        for pdf_file in pdf_files:
            try:
                start_time = time.time()
                logger.info(f"Processing {pdf_file.name}")

                loader = PyPDFLoader(str(pdf_file))
                documents = loader.load()

                for doc in documents:
                    doc.metadata.update({
                        "source_file": pdf_file.name,
                        "file_path": str(pdf_file),
                        "total_pages": len(documents)
                    })

                all_documents.extend(documents)
                load_time = time.time() - start_time
                logger.info(f"Processed {pdf_file.name} ({len(documents)} pages) in {load_time:.2} seconds")

            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {str(e)}")
                failed_files.append(pdf_file)

        if failed_files:
            logger.warning(f"Failed to process {len(failed_files)} PDF files: {failed_files}")

        logger.info(f"Success: Loaded {len(all_documents)} pages from {len(pdf_files) - len(failed_files)} files ")
        return all_documents

    def chunk_docs(self, docs: List[Document]) -> List[Document]:
        logger.info(f"Chunking {len(docs)} documents")
        start_time = time.time()

        chunks = self.text_splitter.split_documents(docs)

        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "chunk_size": len(chunk.page_content)
            })

        chunk_time = time.time() - start_time
        logger.info(f"Created {len(chunks)} chunks in {chunk_time:.2} seconds")

        avg_chunk_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
        logger.info(f"Average chunk size is {avg_chunk_size:.2} charakters")

        return chunks

    def load_and_chunk(self, data_path: Path) -> List[Document]:
        documents = self.load_documents(data_path)
        chunks = self.chunk_docs(documents)
        return chunks