"""Configuration management for document retrieval system."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Embedding model settings
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_device: str = "cuda"  # "cuda" or "cpu"

    # Ollama LLM settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_timeout: float = 120.0

    # Document processing settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    supported_extensions: list[str] = [".txt", ".pdf", ".md", ".docx", ".html"]

    # Vector store settings
    chroma_persist_dir: str = "./data/chroma_db"
    collection_name: str = "documents"

    # Retrieval settings
    top_k: int = 5

    @property
    def chroma_path(self) -> Path:
        """Get ChromaDB persistence path as Path object."""
        return Path(self.chroma_persist_dir)


# Global settings instance
settings = Settings()
