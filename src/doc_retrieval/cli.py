"""CLI interface for document retrieval system."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from .config import settings
from .ingestion import DocumentProcessor
from .retrieval import RetrievalEngine
from .vectorstore import ChromaStore

app = typer.Typer(
    name="doc-retrieval",
    help="Document retrieval system using local LLMs and vector database.",
    add_completion=False,
)
console = Console()


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to the documents folder"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r", help="Include subdirectories"),
    clear_existing: bool = typer.Option(False, "--clear", "-c", help="Clear existing index before ingesting"),
) -> None:
    """Ingest documents from a folder into the vector store."""
    directory = Path(path)

    if not directory.exists():
        console.print(f"[red]Error: Directory not found: {directory}[/red]")
        raise typer.Exit(1)

    if not directory.is_dir():
        console.print(f"[red]Error: Path is not a directory: {directory}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Ingesting documents from:[/bold] {directory.absolute()}")
    console.print(f"[dim]Supported formats: {', '.join(settings.supported_extensions)}[/dim]")
    console.print()

    store = ChromaStore()

    if clear_existing:
        store.clear()
        console.print()

    # Process documents
    processor = DocumentProcessor()
    nodes = processor.process(directory, recursive=recursive)

    if not nodes:
        console.print("[yellow]No documents found to ingest.[/yellow]")
        raise typer.Exit(0)

    # Create index
    store.create_index(nodes)

    console.print()
    console.print("[bold green]Ingestion complete![/bold green]")


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask about the documents"),
    top_k: int | None = typer.Option(None, "--top-k", "-k", help="Number of results to retrieve"),
    no_sources: bool = typer.Option(False, "--no-sources", help="Don't show source documents"),
) -> None:
    """Query the document store with a question."""
    engine = RetrievalEngine(top_k=top_k)

    if not engine.is_ready():
        console.print("[red]Error: No documents indexed.[/red]")
        console.print("Run 'doc-retrieval ingest <path>' first.")
        raise typer.Exit(1)

    result = engine.query(question, show_sources=not no_sources)
    engine.display_result(result)


@app.command()
def interactive(
    top_k: int | None = typer.Option(None, "--top-k", "-k", help="Number of results to retrieve"),
) -> None:
    """Start interactive query mode."""
    engine = RetrievalEngine(top_k=top_k)

    if not engine.is_ready():
        console.print("[red]Error: No documents indexed.[/red]")
        console.print("Run 'doc-retrieval ingest <path>' first.")
        raise typer.Exit(1)

    console.print(Panel(
        "Interactive mode - Ask questions about your documents.\n"
        "Type 'quit' or 'exit' to leave.\n"
        "Type 'sources on/off' to toggle source display.",
        title="Document Retrieval",
        border_style="blue",
    ))
    console.print()

    show_sources = True

    while True:
        try:
            question = Prompt.ask("[bold cyan]Question[/bold cyan]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting...[/yellow]")
            break

        question = question.strip()

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            console.print("[yellow]Goodbye![/yellow]")
            break

        if question.lower() == "sources on":
            show_sources = True
            console.print("[green]Source display enabled[/green]")
            continue

        if question.lower() == "sources off":
            show_sources = False
            console.print("[yellow]Source display disabled[/yellow]")
            continue

        try:
            result = engine.query(question, show_sources=show_sources)
            engine.display_result(result)
        except (ValueError, RuntimeError, ConnectionError, TimeoutError) as e:
            console.print(f"[red]Error: {e}[/red]")

        console.print()


@app.command()
def status() -> None:
    """Show the status of the document store."""
    store = ChromaStore()

    count = store.get_document_count()

    console.print(Panel(
        f"[bold]Collection:[/bold] {settings.collection_name}\n"
        f"[bold]Indexed chunks:[/bold] {count}\n"
        f"[bold]Persist directory:[/bold] {store.persist_dir.absolute()}\n"
        f"[bold]Embedding model:[/bold] {settings.embedding_model}\n"
        f"[bold]LLM model:[/bold] {settings.ollama_model}",
        title="Document Store Status",
        border_style="blue",
    ))

    if count == 0:
        console.print("\n[yellow]No documents indexed. Run 'doc-retrieval ingest <path>' to add documents.[/yellow]")


@app.command()
def clear(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Clear the vector store (keeps database files)."""
    if not confirm:
        if not Confirm.ask("[yellow]Are you sure you want to clear all indexed documents?[/yellow]"):
            console.print("Cancelled.")
            raise typer.Exit(0)

    store = ChromaStore()
    store.clear()

    console.print("[green]Vector store cleared.[/green]")


@app.command()
def purge(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Completely delete the database files to reclaim disk space.

    This command should only be run when the Gradio interface is NOT running,
    as it needs exclusive access to delete the files.
    """
    import shutil

    if not confirm:
        console.print("[yellow]This will completely delete the database directory.[/yellow]")
        console.print("[yellow]Make sure the Gradio interface is NOT running.[/yellow]")
        if not Confirm.ask("Are you sure?"):
            console.print("Cancelled.")
            raise typer.Exit(0)

    persist_dir = settings.chroma_path

    if not persist_dir.exists():
        console.print("[yellow]Database directory does not exist.[/yellow]")
        raise typer.Exit(0)

    # Safety check: refuse to delete system-critical or suspiciously short paths
    dangerous_paths = {Path("/"), Path("/home"), Path("/etc"), Path("/var"), Path("/usr"), Path("/tmp")}
    if persist_dir in dangerous_paths or len(persist_dir.parts) <= 2:
        console.print(f"[red]Error: Refusing to delete '{persist_dir}' -- path looks dangerous.[/red]")
        console.print("[yellow]Check CHROMA_PERSIST_DIR in your .env file.[/yellow]")
        raise typer.Exit(1)

    try:
        shutil.rmtree(persist_dir)
        console.print(f"[green]Deleted: {persist_dir.absolute()}[/green]")
    except OSError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Make sure no other process is using the database.[/yellow]")
        console.print("[dim]Close Gradio/Python and try again.[/dim]")
        raise typer.Exit(1)


@app.command()
def config() -> None:
    """Show current configuration."""
    console.print(Panel(
        f"[bold]Embedding Model:[/bold] {settings.embedding_model}\n"
        f"[bold]Embedding Device:[/bold] {settings.embedding_device}\n"
        f"[bold]Ollama URL:[/bold] {settings.ollama_base_url}\n"
        f"[bold]Ollama Model:[/bold] {settings.ollama_model}\n"
        f"[bold]Chunk Size:[/bold] {settings.chunk_size}\n"
        f"[bold]Chunk Overlap:[/bold] {settings.chunk_overlap}\n"
        f"[bold]Top K:[/bold] {settings.top_k}\n"
        f"[bold]Supported Extensions:[/bold] {', '.join(settings.supported_extensions)}",
        title="Configuration",
        border_style="blue",
    ))


@app.command()
def gradio(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(7860, "--port", "-p", help="Port to listen on"),
    share: bool = typer.Option(False, "--share", help="Create a public Gradio link"),
) -> None:
    """Start the Gradio web interface."""
    from .ui import launch_app

    console.print(f"[bold]Starting Gradio interface...[/bold]")
    console.print(f"[dim]Host: {host}, Port: {port}[/dim]")

    if share:
        console.print("[yellow]Public link will be created[/yellow]")

    console.print()
    launch_app(host=host, port=port, share=share)


if __name__ == "__main__":
    app()
