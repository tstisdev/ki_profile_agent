import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY:str = os.getenv("OPENAI_API_KEY", "")
    # Default to a sentence-transformers model compatible with HuggingFaceEmbeddings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")

    DATA_PATH: Path = Path(os.getenv("DATA_PATH", "/app/data"))
    STORAGE_PATH: Path = Path(os.getenv("STORAGE_PATH", "/app/storage"))

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    FAISS_INDEX_PATH: Path = STORAGE_PATH / "faiss_index.pkl"
    METADATA_PATH: Path = STORAGE_PATH / "metadata.pkl"

    def validate(self) -> None:
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set")

        self.STORAGE_PATH.mkdir(parents=True, exist_ok=True)

        if not self.DATA_PATH.exists():
            raise FileNotFoundError(F"DATA_PATH: {self.DATA_PATH} - does not exist")

settings = Settings()
