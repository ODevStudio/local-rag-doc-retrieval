"""Document processing pipeline for ingestion."""

from __future__ import annotations

import threading
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config import settings

console = Console()

# Lazy load marker models (they take time to initialize)
_marker_models = None
_marker_lock = threading.Lock()


def get_marker_models():
    """Lazy load marker-pdf models."""
    global _marker_models
    if _marker_models is not None:
        return _marker_models
    with _marker_lock:
        if _marker_models is None:
            console.print("[cyan]Loading marker-pdf models (first time may take a while)...[/cyan]")
            from marker.models import create_model_dict
            _marker_models = create_model_dict()
        return _marker_models


class DocumentProcessor:
    """Handles document loading and chunking."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        """Initialize the document processor.

        Args:
            chunk_size: Size of text chunks in tokens. Defaults to settings value.
            chunk_overlap: Overlap between chunks in tokens. Defaults to settings value.
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def _load_pdf(self, file_path: Path) -> list[Document]:
        """Load a PDF using marker-pdf.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of documents from the PDF.
        """
        try:
            from marker.converters.pdf import PdfConverter

            models = get_marker_models()
            converter = PdfConverter(artifact_dict=models)
            result = converter(str(file_path))

            markdown_text = result.markdown

            if markdown_text.strip():
                return [Document(
                    text=markdown_text,
                    metadata={
                        "file_name": file_path.name,
                        "file_path": str(file_path),
                        "file_type": "pdf",
                    }
                )]
            return []

        except Exception as e:
            console.print(f"[red]Error parsing PDF {file_path.name}: {e}[/red]")
            return []

    def _load_other_files(
        self,
        directory: Path,
        recursive: bool,
        exclude_pdfs: bool = True,
    ) -> list[Document]:
        """Load non-PDF files using SimpleDirectoryReader.

        Args:
            directory: Path to the directory.
            recursive: Whether to include subdirectories.
            exclude_pdfs: Whether to exclude PDF files.

        Returns:
            List of documents.
        """
        extensions = [ext for ext in settings.supported_extensions if ext != ".pdf"] if exclude_pdfs else settings.supported_extensions

        if not extensions:
            return []

        try:
            reader = SimpleDirectoryReader(
                input_dir=str(directory),
                recursive=recursive,
                required_exts=extensions,
            )
            return reader.load_data()
        except ValueError:
            # No files found with the given extensions
            return []

    def _find_pdfs(self, directory: Path, recursive: bool) -> list[Path]:
        """Find all PDF files in a directory.

        Args:
            directory: Path to the directory.
            recursive: Whether to include subdirectories.

        Returns:
            List of PDF file paths.
        """
        if recursive:
            return list(directory.rglob("*.pdf"))
        else:
            return list(directory.glob("*.pdf"))

    def load_documents(
        self,
        directory: str | Path,
        recursive: bool = True,
    ) -> list[Document]:
        """Load documents from a directory.

        Args:
            directory: Path to the directory containing documents.
            recursive: Whether to include subdirectories.

        Returns:
            List of loaded documents.
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory}")

        documents = []

        # Load PDFs with marker-pdf
        pdf_files = self._find_pdfs(directory, recursive)
        if pdf_files:
            console.print(f"[cyan]Processing {len(pdf_files)} PDF(s) with marker-pdf...[/cyan]")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                for pdf_path in pdf_files:
                    task = progress.add_task(f"Processing: {pdf_path.name}", total=None)
                    pdf_docs = self._load_pdf(pdf_path)
                    documents.extend(pdf_docs)
                    progress.remove_task(task)

        # Load other files with SimpleDirectoryReader
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Loading other documents...", total=None)
            other_docs = self._load_other_files(directory, recursive, exclude_pdfs=True)
            documents.extend(other_docs)

        # Log document details
        for doc in documents:
            file_name = doc.metadata.get("file_name", "unknown")
            text_len = len(doc.text)
            console.print(f"[dim]  - {file_name}: {text_len} characters[/dim]")

        console.print(f"[green]Loaded {len(documents)} document(s)[/green]")
        return documents

    def chunk_documents(self, documents: list[Document]) -> list:
        """Split documents into smaller chunks (nodes).

        Args:
            documents: List of documents to chunk.

        Returns:
            List of text nodes (chunks).
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Chunking documents...", total=None)
            nodes = self.splitter.get_nodes_from_documents(documents)

        console.print(f"[green]Created {len(nodes)} chunk(s)[/green]")
        return nodes

    def process(
        self,
        directory: str | Path,
        recursive: bool = True,
    ) -> list:
        """Load and chunk documents from a directory.

        Args:
            directory: Path to the directory containing documents.
            recursive: Whether to include subdirectories.

        Returns:
            List of text nodes (chunks) ready for embedding.
        """
        documents = self.load_documents(directory, recursive)
        nodes = self.chunk_documents(documents)
        return nodes
