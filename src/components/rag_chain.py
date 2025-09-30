from typing import List, Dict, Any
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.schema import Document

from src.components.vector_store import VectorStore
from src.utils.logger import logger
from config.settings import settings
from database import get_entities_for_deanonymization


class RAGChain:

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=settings.CHAT_MODEL,
            temperature=0.1
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Du bist ein Assistent, der Stellenanfragen mit den am besten geeigneten Mitarbeitern abgleicht.

        Anweisungen:
        - Nutze NUR die Informationen aus den bereitgestellten Mitarbeiterprofilen.
        - Vergleiche die Anforderungen mit den Fähigkeiten, Erfahrungen und Sprachen der Mitarbeiter.
        - Antworte IMMER in diesem Format:

        Der beste Mitarbeiter für die Anfrage ist: [Name des Mitarbeiters]

        Begründung: [1-2 prägnante Sätze, die die relevanten Fähigkeiten oder Erfahrungen nennen]

        - Falls kein passender Mitarbeiter gefunden wird: "Kein geeigneter Mitarbeiter in den vorliegenden Profilen gefunden."
        - Sei präzise und auf den Punkt.

        Mitarbeiterprofile:
        {context}
        """),
            ("human", "Anfrage: {question}")
        ])
        # self.prompt = ChatPromptTemplate.from_messages([
        #     ("system", """You are an assistant that helps match job inquiries to the most suitable employees
        # based only on the provided context (employee profiles).
        #
        # Instructions:
        # - Use ONLY the information from the provided context to select employees.
        # - Compare the job requirements with the skills, experiences, and languages listed in the employee profiles.
        # - Rank or recommend the employee(s) that best fit the job requirements.
        # - If multiple employees fit, provide the top matches with reasoning.
        # - If no clear match is found, respond: "Context didn’t provide enough information to identify a suitable employee."
        # - Be concise but comprehensive: explain why the match is suitable (skills, languages, experience).
        # - When available, cite the employee name or identifier from the context so it’s clear who the recommendation is.
        #
        # Context Documents (Employee Profiles):
        # {context}
        # """),
        #     ("human", "Job Inquiry: {question}")
        # ])

        self.chain = (
            {
                "context": RunnableLambda(self._retrieve_and_format_docs),
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        logger.info("RAG Chain started.")

    def _retrieve_and_format_docs(self, question:str) -> str:
        logger.info(f"Retrieving docs for question: {question[:100]}")

        start_time = time.time()

        results = self.vector_store.search(question, k=settings.TOP_K_RESULTS)

        if not results:
            logger.warning("No relevant docs found")
            return "Not relevant docs found"

        context_parts = []
        for i, (document, score) in enumerate(results, 1):
            source = document.metadata.get("source_file", "Unknown")
            page = document.metadata.get("page", "Unknown")

            context_parts.append(
                f"Document {i}: (Source: {source}, Page: ({page}), Score: {score:.3f}):\n"
                f"{document.page_content}\n"
            )

        context = "\n" + "="*80 + "\n".join(context_parts)

        retrieve_time = time.time() - start_time
        logger.info(f"Retrieved {len(results)} docs in {retrieve_time:.2} seconds.")

        return context

    def ask(self, question:str) -> Dict[str, Any]:
        logger.info(f"Processing question: {question}")
        start_time = time.time()

        try:
            answer = self.chain.invoke(question)
            # De-anonymize the final answer from placeholders back to original values
            mapping = get_entities_for_deanonymization()
            if mapping:
                for anonymized, original in mapping.items():
                    if anonymized in answer:
                        answer = answer.replace(anonymized, original)

            relevant_docs = self.vector_store.search(question, k=settings.TOP_K_RESULTS)

            total_time = time.time() - start_time

            response = {
                "question": question,
                "answer": answer,
                "sources": [
                    {
                        "source_file" : doc.metadata.get("source_file", "Unknown"),
                        "page": doc.metadata.get("page", "Unknown"),
                        "chunk_id": doc.metadata.get("chunk_id", "Unknown"),
                        "relevance_score": score
                    } for doc, score in relevant_docs
                ],
                "response_time": round(total_time, 3),
                "num_sources": len(relevant_docs)
            }

            logger.info(f"Created response in {total_time:.3f} seconds.")
            return response

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Failed to process question: {str(e)}\n{tb}")
            return {
                "question": question,
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "response_time": time.time() - start_time,
                "num_sources": 0
            }

    def batch_ask(self, questions: List[str]) -> List[Dict[str, Any]]:
        logger.info(f"Processing {len(questions)} questions")

        results = []
        for idx, question in enumerate(questions, 1):
            logger.info(f"Processing question: {idx}/{len(questions)}") # zum beispiel Processing question 3/6
            result = self.ask(question)
            results.append(result)

        return results