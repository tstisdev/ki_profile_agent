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


class RAGChain:

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=settings.CHAT_MODEL,
            temperature=0.1
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are my helpful assistant. You answer only based on the provided context
            
        Instructions:
        - Use ONLY the information from the provided context to answer my questions
        - If the context doesnt contain enough information for you to answer my question, say "Context didnt provide enough information to answer your question"
        - If possible, provide references to source documents
        - Answer as concise but comprehensive as possible
        - Synthesize multiple relevant pieces of information if there are more than one 
        
        Context Documents: {context}
        """)
        , ("human", "Question: {question}")
        ])


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
            logger.error(f"Failed to process question: {str(e)}")
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