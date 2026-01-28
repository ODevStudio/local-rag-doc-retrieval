"""Configuration management for document retrieval system."""

import functools
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
    supported_extensions: tuple[str, ...] = (".txt", ".pdf", ".md", ".docx", ".html")

    # Vector store settings
    chroma_persist_dir: str = "./data/chroma_db"
    collection_name: str = "documents"

    # Retrieval settings
    top_k: int = 5

    # Security settings
    allowed_ingest_dir: str = ""  # Empty means allow any path (CLI-only use)
    gradio_username: str = ""
    gradio_password: str = ""

    @property
    def chroma_path(self) -> Path:
        """Get ChromaDB persistence path as Path object.

        Relative paths are resolved against the directory containing pyproject.toml
        (project root), falling back to cwd if not found.
        """
        p = Path(self.chroma_persist_dir)
        if p.is_absolute():
            return p
        # Try to find project root by looking for pyproject.toml
        candidate = Path(__file__).resolve().parent
        for _ in range(5):
            if (candidate / "pyproject.toml").exists():
                return (candidate / p).resolve()
            candidate = candidate.parent
        return p.resolve()


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get the application settings (lazily initialized, cached)."""
    return Settings()


class _SettingsProxy:
    """Proxy that delegates attribute access to the lazily-loaded Settings instance.

    This preserves backward compatibility so existing code using `from .config import settings`
    continues to work without changes, while deferring Settings() construction until first use.
    """

    def __getattr__(self, name: str):
        return getattr(get_settings(), name)


settings = _SettingsProxy()
