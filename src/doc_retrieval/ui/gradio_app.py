"""Gradio web interface for document retrieval system."""

from __future__ import annotations

import gc
import threading
from pathlib import Path

import gradio as gr

from ..config import settings
from ..ingestion import DocumentProcessor
from ..retrieval import RetrievalEngine
from ..vectorstore import ChromaStore, close_chroma_client

# Global engine instance
_engine: RetrievalEngine | None = None
_engine_lock = threading.Lock()


def reset_engine() -> None:
    """Reset the global engine and close all ChromaDB connections."""
    global _engine
    with _engine_lock:
        if _engine is not None:
            _engine._index = None
            _engine._llm = None
            _engine.store = None
            _engine = None

    gc.collect()
    close_chroma_client()


def get_engine() -> RetrievalEngine:
    """Get or create the retrieval engine."""
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is None:
            _engine = RetrievalEngine()
        return _engine


def get_status() -> str:
    """Get current system status."""
    store = ChromaStore()
    count = store.get_document_count()

    status_lines = [
        f"| Setting | Value |",
        f"|---------|-------|",
        f"| Collection | {settings.collection_name} |",
        f"| Indexed chunks | {count} |",
        f"| Embedding model | {settings.embedding_model} |",
        f"| LLM model | {settings.ollama_model} |",
        f"| Ollama URL | {settings.ollama_base_url} |",
    ]

    if count == 0:
        status_lines.append("\n\n**No documents indexed.** Use the Ingest tab to add documents.")
    else:
        status_lines.append(f"\n\n**Ready.** {count} chunks available for querying.")

    return "\n".join(status_lines)


def _validate_ingest_path(folder_path: str) -> tuple[Path | None, str | None]:
    """Validate and resolve an ingest folder path.

    Returns:
        Tuple of (resolved_path, error_message). If error_message is set, path is None.
    """
    if not folder_path:
        return None, "Please enter a folder path."

    directory = Path(folder_path).resolve()

    if not directory.exists():
        return None, f"Directory not found: `{directory}`"

    if not directory.is_dir():
        return None, f"Path is not a directory: `{directory}`"

    allowed = settings.allowed_ingest_dir.strip()
    if allowed:
        allowed_path = Path(allowed).resolve()
        try:
            directory.relative_to(allowed_path)
        except ValueError:
            return None, (
                f"Access denied. Path `{directory}` is outside the allowed "
                f"ingest directory: `{allowed_path}`"
            )

    return directory, None


def ingest_documents(folder_path: str, recursive: bool, clear_existing: bool):
    """Ingest documents from a folder. Generator for immediate UI feedback."""
    directory, error = _validate_ingest_path(folder_path)
    if error is not None:
        yield f"Error: {error}"
        return

    # Show loading state immediately
    yield "Ingesting documents... This may take a while."

    try:
        if clear_existing:
            # Reset engine and close connections before clearing
            reset_engine()

        store = ChromaStore()

        if clear_existing:
            store.clear()

        processor = DocumentProcessor()
        nodes = processor.process(directory, recursive=recursive)

        if not nodes:
            yield "No supported documents found in the folder."
            return

        store.create_index(nodes)

        # Reset engine to reload index with new data
        reset_engine()

        yield f"Successfully ingested **{len(nodes)}** chunks from `{directory}`"

    except Exception as e:
        yield f"Error during ingestion: {str(e)}"


def query_documents(question: str, top_k: int, show_sources: bool):
    """Query the document store. Generator for immediate UI feedback."""
    if not question.strip():
        yield "Please enter a question.", ""
        return

    engine = get_engine()

    if not engine.is_ready():
        yield "No documents indexed. Use the Ingest tab to add documents first.", ""
        return

    # Show loading state immediately
    yield "Searching...", ""

    try:
        # Update top_k if different
        engine.top_k = top_k

        result = engine.query(question, show_sources=show_sources)

        # Format sources
        sources_text = ""
        if show_sources and result.sources:
            source_parts = []
            for i, source in enumerate(result.sources, 1):
                score = source.get("score", 0)
                metadata = source.get("metadata", {})
                file_name = metadata.get("file_name", "Unknown")
                text = source["text"]
                if len(text) >= 500:
                    text += "..."

                source_parts.append(
                    f"**Source {i}** -- `{file_name}` (score: {score:.3f})\n\n{text}"
                )

            sources_text = "\n\n---\n\n".join(source_parts)

        yield result.answer, sources_text

    except Exception as e:
        yield f"Error: {str(e)}", ""


def clear_index(confirm: bool) -> tuple[str, dict]:
    """Clear the vector store."""
    if not confirm:
        return "Please check the confirmation box to clear the index.", gr.update(value=False)

    try:
        # First, reset the engine so it reloads the index after clear
        reset_engine()

        # Clear all data
        store = ChromaStore()
        store.clear()

        msg = (
            "Vector store cleared.\n\n"
            "*Note: Database files remain but will be reused. "
            "To reclaim disk space, stop the app and run `doc-retrieval purge`.*"
        )
        return msg, gr.update(value=False)
    except Exception as e:
        return f"Error: {str(e)}", gr.update(value=False)


def create_app() -> gr.Blocks:
    """Create the Gradio application."""

    with gr.Blocks(
        title="Document Retrieval System",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            """
            # Document Retrieval System
            Ask questions about your documents using local LLMs and vector search.
            """
        )

        with gr.Tabs():
            # Query Tab
            with gr.Tab("Query"):
                gr.Markdown("### Ask a Question")

                question_input = gr.Textbox(
                    label="Question",
                    placeholder="What would you like to know about your documents?",
                    lines=2,
                )

                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=settings.top_k,
                        step=1,
                        label="Number of sources",
                    )
                    show_sources_checkbox = gr.Checkbox(
                        value=True,
                        label="Show sources",
                    )
                    query_button = gr.Button("Search", variant="primary")

                gr.Markdown("### Answer")
                answer_output = gr.Markdown()

                gr.Markdown("### Sources")
                sources_output = gr.Markdown()

                query_button.click(
                    fn=query_documents,
                    inputs=[question_input, top_k_slider, show_sources_checkbox],
                    outputs=[answer_output, sources_output],
                )

                question_input.submit(
                    fn=query_documents,
                    inputs=[question_input, top_k_slider, show_sources_checkbox],
                    outputs=[answer_output, sources_output],
                )

            # Ingest Tab
            with gr.Tab("Ingest"):
                gr.Markdown(
                    f"""
                    ### Add Documents
                    Supported formats: `{', '.join(settings.supported_extensions)}`
                    """
                )

                folder_input = gr.Textbox(
                    label="Folder path",
                    placeholder="C:/Documents/my-docs or /home/user/documents",
                )

                with gr.Row():
                    recursive_checkbox = gr.Checkbox(
                        value=True,
                        label="Include subfolders",
                    )
                    clear_checkbox = gr.Checkbox(
                        value=False,
                        label="Clear existing index",
                    )

                ingest_button = gr.Button("Ingest", variant="primary")

                gr.Markdown("### Result")
                ingest_output = gr.Markdown()

                ingest_button.click(
                    fn=ingest_documents,
                    inputs=[folder_input, recursive_checkbox, clear_checkbox],
                    outputs=[ingest_output],
                )

            # Status Tab
            with gr.Tab("Status"):
                gr.Markdown("### System Status")
                status_output = gr.Markdown()
                refresh_button = gr.Button("Refresh", variant="secondary")

                refresh_button.click(
                    fn=get_status,
                    outputs=[status_output],
                )

                gr.Markdown("### Danger Zone")
                with gr.Row():
                    clear_confirm_checkbox = gr.Checkbox(
                        value=False,
                        label="I want to delete all indexed documents",
                    )
                    clear_button = gr.Button("Clear Index", variant="stop")

                clear_output = gr.Markdown()

                clear_button.click(
                    fn=clear_index,
                    inputs=[clear_confirm_checkbox],
                    outputs=[clear_output, clear_confirm_checkbox],
                )

                # Load status on app start
                app.load(fn=get_status, outputs=[status_output])

        gr.Markdown(
            """
            ---
            *Powered by LlamaIndex, ChromaDB, and Ollama*
            """
        )

    return app


def launch_app(
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
) -> None:
    """Launch the Gradio application."""
    app = create_app()

    auth = None
    if settings.gradio_username and settings.gradio_password:
        auth = (settings.gradio_username, settings.gradio_password)

    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        auth=auth,
    )
