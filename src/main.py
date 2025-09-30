import sys
import argparse
import json
from typing import List

from src.rag_pipeline import RAGPipeline
from src.utils.logger import logger

def print_response(response:dict) -> None:
    print("\n" + "-"*80)
    print(f"QUESTION: {response['question']}")
    print("-"*80)
    print(f"ANSWER: {response['answer']}")
    print(f"\Response Time: {response['response_time']} seconds")
    print(f"Sources used: {response['num_sources']}")

    if response['sources']:
        print("\nSOURCES:")
        for i, source in enumerate(response['sources'], 1):
            print(f"  {i}. {source['source_file']} (Page {source['page']}) - Score: {source['relevance_score']:.3f}")

    print("-"*80)


# def run_hardcoded_questions(pipeline: RAGPipeline) -> None:
#     print("\n Running hardcoded questions")
#
#     questions = [
#         # hier können vordefinierte fragen stehen
#     ]
#
#     responses = pipeline.ask_questions(questions)
#     for response in responses:
#         print_response(response)
#

def run_interactive_mode(pipeline: RAGPipeline) -> None:
    print("Stelle deine Frage oder tippe 'quit' or 'exit'.\n")

    while True:
        try:
            question_in = input("Stelle deine Frage: ").strip()
            if question_in.lower() in ["q", "quit", "e", "exit"]:
                print("Closing...")
                break

            if not question_in:
                continue

            response = pipeline.ask_question(question_in)
            print_response(response)

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            print(f"Error: str{e}")

def print_pipeline_info(pipeline: RAGPipeline) -> None:
    info = pipeline.get_info()

    print(f"\n Pipeline info:")
    print(f" - Number of docs: {info['total_documents']}")
    print(f" - Index size: {info['index_size']}")
    print(f" - Embedding model: {info['embedding_model']}")
    print(f" - Chat model: {info['chat_model']}")


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Entry Point")
    parser.add_argument("--generate-sample-pdfs", action="store_true", help="Generate sample PDFs before starting the pipeline")
    args = parser.parse_args()

    try:

        if args.generate_sample_pdfs:
            try:
                from scripts.generate_sample_pdfs import create_pdfs, read_pdfs
                create_pdfs()
                read_pdfs()
            except Exception as e:
                logger.error(f"Failed generating sample PDFs: {e}")

        pipeline = RAGPipeline()
        pipeline.initialize()

        # print_pipeline_info(pipeline)
        print("-"*50)
        print("Hi, ich bin der KI_Profil BOT. Wen soll ich finden?\n")

        run_interactive_mode(pipeline)


    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()