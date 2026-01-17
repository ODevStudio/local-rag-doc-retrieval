"""ChromaDB vector store operations."""

import gc
from pathlib import Path
from typing import Optional

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config import settings

console = Console()

# Global client instance for connection reuse
_chroma_client: Optional[chromadb.PersistentClient] = None


def get_chroma_client(persist_dir: Path) -> chromadb.PersistentClient:
    """Get or create the global ChromaDB client.

    This ensures only one client exists per persist directory,
    which is required for proper file locking on Windows.
    """
    global _chroma_client
    if _chroma_client is None:
        # Enable reset to allow clearing all data
        chroma_settings = chromadb.Settings(
            persist_directory=str(persist_dir),
            allow_reset=True,
            anonymized_telemetry=False,
        )
        _chroma_client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=chroma_settings,
        )
    return _chroma_client


def close_chroma_client() -> None:
    """Close the global ChromaDB client to allow reinitialization."""
    global _chroma_client
    if _chroma_client is not None:
        _chroma_client = None
        # Help garbage collection clean up resources
        gc.collect()


class ChromaStore:
    """Manages ChromaDB vector store operations."""

    def __init__(
        self,
        persist_dir: Optional[str | Path] = None,
        collection_name: Optional[str] = None,
    ):
        """Initialize the ChromaDB store.

        Args:
            persist_dir: Directory for persistent storage. Defaults to settings value.
            collection_name: Name of the collection. Defaults to settings value.
        """
        self.persist_dir = Path(persist_dir or settings.chroma_persist_dir)
        self.collection_name = collection_name or settings.collection_name

        # Ensure persist directory exists
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Use shared client
        self.client = get_chroma_client(self.persist_dir)

        # Initialize embedding model
        self._embed_model: Optional[HuggingFaceEmbedding] = None

    @property
    def embed_model(self) -> HuggingFaceEmbedding:
        """Lazy-load the embedding model."""
        if self._embed_model is None:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Loading embedding model...", total=None)
                self._embed_model = HuggingFaceEmbedding(
                    model_name=settings.embedding_model,
                    device=settings.embedding_device,
                )
        return self._embed_model

    def get_or_create_collection(self) -> chromadb.Collection:
        """Get or create the ChromaDB collection.

        Returns:
            ChromaDB collection instance.
        """
        return self.client.get_or_create_collection(name=self.collection_name)

    def create_index(self, nodes: list) -> VectorStoreIndex:
        """Create a vector store index from document nodes.

        Args:
            nodes: List of text nodes to index.

        Returns:
            VectorStoreIndex instance.
        """
        collection = self.get_or_create_collection()

        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Creating embeddings and indexing...", total=None)
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=self.embed_model,
            )

        console.print(f"[green]Indexed {len(nodes)} chunk(s) into collection '{self.collection_name}'[/green]")
        return index

    def load_index(self) -> Optional[VectorStoreIndex]:
        """Load an existing index from the vector store.

        Returns:
            VectorStoreIndex if collection exists and has data, None otherwise.
        """
        try:
            collection = self.client.get_collection(name=self.collection_name)
            if collection.count() == 0:
                return None
        except Exception:
            return None

        vector_store = ChromaVectorStore(chroma_collection=collection)

        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self.embed_model,
        )

        return index

    def get_document_count(self) -> int:
        """Get the number of documents in the collection.

        Returns:
            Number of indexed chunks.
        """
        try:
            collection = self.client.get_collection(name=self.collection_name)
            return collection.count()
        except Exception:
            return 0

    def clear(self) -> None:
        """Clear all data from the vector store.

        Uses ChromaDB's reset() method which clears all data without
        needing to delete files (which is problematic on Windows due
        to file locking).
        """
        try:
            # Use reset() to clear all data - this works while the app is running
            self.client.reset()
            console.print(f"[yellow]Cleared all collections and data[/yellow]")
        except Exception as e:
            console.print(f"[red]Error during reset: {e}[/red]")
            # Fallback: try to just delete the collection
            try:
                self.client.delete_collection(name=self.collection_name)
                console.print(f"[yellow]Cleared collection '{self.collection_name}'[/yellow]")
            except Exception:
                pass

        console.print("[dim]Note: Database files remain but will be reused. To reclaim disk space,[/dim]")
        console.print("[dim]stop the application and delete the data folder manually.[/dim]")
