"""Retrieval engine with Ollama LLM integration."""

from __future__ import annotations

from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config import settings
from ..vectorstore import ChromaStore

console = Console()

MAX_QUERY_LENGTH = 10_000


@dataclass
class QueryResult:
    """Container for query results."""

    answer: str
    sources: list[dict]


class RetrievalEngine:
    """Handles document retrieval and LLM-powered responses."""

    def __init__(
        self,
        store: ChromaStore | None = None,
        top_k: int | None = None,
    ):
        """Initialize the retrieval engine.

        Args:
            store: ChromaStore instance. Creates new one if not provided.
            top_k: Number of results to retrieve. Defaults to settings value.
        """
        self.store = store or ChromaStore()
        self.top_k = top_k or settings.top_k
        self._llm: Ollama | None = None
        self._index: VectorStoreIndex | None = None

    @property
    def llm(self) -> Ollama:
        """Lazy-load the Ollama LLM."""
        if self._llm is None:
            self._llm = Ollama(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                request_timeout=settings.ollama_timeout,
            )
        return self._llm

    @property
    def index(self) -> VectorStoreIndex | None:
        """Get the loaded index."""
        if self._index is None:
            self._index = self.store.load_index()
        return self._index

    def is_ready(self) -> bool:
        """Check if the engine has an index loaded.

        Returns:
            True if index is available, False otherwise.
        """
        return self.index is not None

    def query(self, question: str, show_sources: bool = True) -> QueryResult:
        """Query the document store.

        Args:
            question: The question to ask.
            show_sources: Whether to include source documents in result.

        Returns:
            QueryResult with answer and source documents.

        Raises:
            RuntimeError: If no index is loaded.
            ValueError: If the question is empty or exceeds MAX_QUERY_LENGTH.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if len(question) > MAX_QUERY_LENGTH:
            raise ValueError(
                f"Question exceeds maximum length of {MAX_QUERY_LENGTH} characters."
            )

        if not self.is_ready():
            raise RuntimeError(
                "No documents indexed. Run 'doc-retrieval ingest <path>' first."
            )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Querying...", total=None)

            query_engine = self.index.as_query_engine(
                llm=self.llm,
                similarity_top_k=self.top_k,
            )

            response = query_engine.query(question)

        # Extract source documents
        sources = []
        if show_sources and response.source_nodes:
            for node in response.source_nodes:
                source_info = {
                    "text": node.node.get_content()[:500],  # Truncate for display
                    "score": node.score,
                    "metadata": node.node.metadata,
                }
                sources.append(source_info)

        return QueryResult(
            answer=str(response),
            sources=sources,
        )

    def display_result(self, result: QueryResult) -> None:
        """Display query result in a formatted way.

        Args:
            result: QueryResult to display.
        """
        # Display answer
        console.print()
        console.print(Panel(result.answer, title="Answer", border_style="green"))

        # Display sources
        if result.sources:
            console.print()
            console.print("[bold]Sources:[/bold]")
            for i, source in enumerate(result.sources, 1):
                score = source.get("score", 0)
                metadata = source.get("metadata", {})
                file_name = metadata.get("file_name", "Unknown")

                console.print(f"\n[cyan]Source {i}[/cyan] (score: {score:.3f}) - {file_name}")
                console.print(Panel(
                    source["text"] + ("..." if len(source["text"]) >= 500 else ""),
                    border_style="dim",
                ))
