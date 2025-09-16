from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
import time

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.utils.logger import logger
from config.settings import settings
from database import insert_extracted_entity


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
        documents = self._anonymize_documents(documents)
        chunks = self.chunk_docs(documents)
        return chunks

    def _anonymize_documents(self, docs: List[Document]) -> List[Document]:
        logger.info(f"Anonymizing {len(docs)} documents before embedding")

        # Counters per entity type to create stable placeholders
        name_counter = 1
        date_counter = 1
        place_counter = 1

        # Precompile regexes
        # Names like "Barack O." or "Mahatma G." (Firstname capitalized + space + capital initial + dot)
        name_initial_pattern = re.compile(r"\b([A-ZÄÖÜ][\wäöüß-]+)\s([A-ZÄÖÜ])\.(?!\w)")
        # Labeled name field: "Name: First Last" or "Name: First L."
        name_label_pattern = re.compile(r"\b(?:Name)\s*:\s*([A-ZÄÖÜ][\wäöüß-]+)\s+([A-ZÄÖÜ][\wäöüß-]+|[A-ZÄÖÜ])\.?\b")
        # Header form: "First X., <something>" (lookahead for comma)
        name_header_initial_pattern = re.compile(r"\b([A-ZÄÖÜ][\wäöüß-]+)\s([A-ZÄÖÜ])\.(?=\s*,)")
        # German/EN birth date variants
        date_pattern = re.compile(r"\b(?:Geburtsdatum[:]?\s*|Birth\s*date[:]?\s*)?((?:[0-3]?\d[\.\-/][01]?\d[\.\-/](?:19|20)\d\d)|(?:(?:19|20)\d\d-[01]?\d-[0-3]?\d))\b")
        # Birth place: words after Geburtsort/Birth place
        place_pattern = re.compile(r"\b(?:Geburtsort|Birth\s*place)[:]?\s*([A-ZÄÖÜ][\wäöüßÄÖÜ-]+(?:\s+[A-ZÄÖÜ][\wäöüßÄÖÜ-]+)*)")

        # Mapping cache so repeated occurrences use same placeholder
        value_to_placeholder: Dict[str, str] = {}

        for doc in docs:
            original_text = doc.page_content
            text = original_text

            # Names (Firstname + Initial.)
            def replace_name_initial(m: re.Match) -> str:
                nonlocal name_counter
                full = f"{m.group(1)} {m.group(2)}."
                if full not in value_to_placeholder:
                    placeholder = f"FirstName_{name_counter}"
                    value_to_placeholder[full] = placeholder
                    insert_extracted_entity('name', full, placeholder, 'regex_name')
                    name_counter += 1
                return value_to_placeholder[full]

            text = name_initial_pattern.sub(replace_name_initial, text)

            # Labeled names (Name: First Last or First L.)
            def replace_name_label(m: re.Match) -> str:
                nonlocal name_counter
                captured = f"{m.group(1)} {m.group(2)}"
                # Normalize trailing dot in second group
                if captured.endswith("."):
                    captured = captured[:-1]
                if captured not in value_to_placeholder:
                    placeholder = f"FirstName_{name_counter}"
                    value_to_placeholder[captured] = placeholder
                    insert_extracted_entity('name', captured, placeholder, 'regex_name_label')
                    name_counter += 1
                return f"Name: {value_to_placeholder[captured]}"

            text = name_label_pattern.sub(replace_name_label, text)

            # Header initial form before a comma (Firstname X., ...)
            def replace_name_header_initial(m: re.Match) -> str:
                nonlocal name_counter
                full = f"{m.group(1)} {m.group(2)}."
                if full not in value_to_placeholder:
                    placeholder = f"FirstName_{name_counter}"
                    value_to_placeholder[full] = placeholder
                    insert_extracted_entity('name', full, placeholder, 'regex_name_header')
                    name_counter += 1
                return value_to_placeholder[full]

            text = name_header_initial_pattern.sub(replace_name_header_initial, text)

            # Birth dates
            def replace_date(m: re.Match) -> str:
                nonlocal date_counter
                date_val = m.group(1)
                if date_val not in value_to_placeholder:
                    placeholder = f"BIRTHDATE_{date_counter}"
                    value_to_placeholder[date_val] = placeholder
                    insert_extracted_entity('birthdate', date_val, placeholder, 'regex_date')
                    date_counter += 1
                return text[m.start():m.start()] + value_to_placeholder[date_val]

            text = date_pattern.sub(lambda m: value_to_placeholder.get(m.group(1)) or (replace_date(m)), text)

            # Birth places
            def replace_place(m: re.Match) -> str:
                nonlocal place_counter
                place_val = m.group(1).strip()
                if place_val not in value_to_placeholder:
                    placeholder = f"BIRTHPLACE_{place_counter}"
                    value_to_placeholder[place_val] = placeholder
                    insert_extracted_entity('birthplace', place_val, placeholder, 'regex_place')
                    place_counter += 1
                prefix = m.group(0)[: m.group(0).find(place_val)]
                return f"{prefix}{value_to_placeholder[place_val]}"

            text = place_pattern.sub(replace_place, text)

            if text != original_text:
                doc.page_content = text

        logger.info("Completed anonymization")
        return docs