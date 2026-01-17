# 📑 Document Retrieval System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Backend](https://img.shields.io/badge/Backend-LlamaIndex-orange.svg?style=for-the-badge)](https://www.llamaindex.ai/)
[![LLM](https://img.shields.io/badge/LLM-Ollama-white.svg?style=for-the-badge&logo=ollama&logoColor=black)](https://ollama.com/)

**A local RAG (Retrieval-Augmented Generation) system for your documents.**
*Powered by LlamaIndex, ChromaDB, Marker-PDF & Ollama.*

[Features](#-features) • [Installation](#-installation) • [Hardware Guide](#-hardware--llm-guide) • [Usage](#-usage)

</div>

---

## 🖼️ Interface

![Gradio Web Interface](https://github.com/user-attachments/assets/a68432b6-a318-46be-9cc1-dd6385ad75cf)

## 🚀 Features

Build a local knowledge base and query it using LLMs without your data ever leaving your machine.

* **📄 Multi-Format Ingestion:** Supports `.pdf` (with deep OCR via `marker-pdf`), `.txt`, `.md`, `.docx`, `.html`.
* **🧠 Local Intelligence:** Uses **Ollama** for inference and **HuggingFace** embeddings locally on your GPU.
* **🔍 Vector Search:** High-performance retrieval using **ChromaDB**.
* **🖥️ Dual Interface:** Interact via a powerful **CLI** or a user-friendly **Web UI (Gradio)**.
* **⚡ GPU Accelerated:** Optimized for NVIDIA GPUs (CUDA) for fast indexing and response times.

---

## 🛠️ Installation

### 1. Prerequisites

* Python 3.10+
* [Ollama](https://ollama.com/) installed and running
* NVIDIA GPU (Highly recommended for OCR and LLM inference)

### 2. Setup

```bash
# Clone the repository
cd doc-retrieval

# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate    # Linux/Mac
# venv\Scripts\activate     # Windows

# Install the package
pip install -e .
3. Enable GPU Acceleration (IMPORTANT!)Standard installation often defaults to CPU versions of PyTorch. To utilize your GPU for OCR (marker-pdf) and Embeddings, you must reinstall PyTorch with CUDA support. Choose the command matching your hardware:CUDA VersionGPU Generation (Examples)CommandCUDA 12.4RTX 40 Series (4090, 4080, etc.)pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstallCUDA 12.1RTX 30 Series / Newerpip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstallCUDA 11.8RTX 20 Series / Older (GTX 10xx)pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --force-reinstallNote: If unsure, CUDA 12.1 is generally compatible with most modern cards.🎮 Hardware & LLM GuideSystem performance depends heavily on your available VRAM. Below are recommendations for models to pull via Ollama:VRAMRecommended LLM (Ollama)Embedding ModelNotes< 8 GBphi3:3.8b, gemma:2b, tinyllamaall-MiniLM-L6-v2OCR and inference will be slow. Use small models.8 GBllama3.1:8b (q4_0), qwen2.5:7bbge-base-en-v1.5Standard consumer cards (RTX 3060/4060). Solid performance.12-16 GBmistral-nemo:12b, llama3.1:8b (fp16)bge-large-en-v1.5High-End (RTX 3080/4080). Allows for larger context windows.24 GB+mixtral:8x7b, llama3.1:70b (quantized)bge-large-en-v1.5Enthusiast (RTX 3090/4090). Maximum quality.Pulling models:Bashollama pull llama3.1:8b
ollama pull qwen2.5:7b
⚙️ ConfigurationCreate a .env file in your root directory (see .env.example) to customize settings:Ini, TOML# .env Example
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
CHROMA_PERSIST_DIR=./data/chroma_db
💻 UsageThe system is controlled via the doc-retrieval command.1. Web Interface (Gradio)Starts a graphical interface in your browser.Bashdoc-retrieval gradio
# Optional with public sharing link:
# doc-retrieval gradio --share
2. Ingest DocumentsParses files, performs OCR (on PDFs), and stores vectors in the database.Bash# Ingest entire folder
doc-retrieval ingest ./my-documents

# Clear existing DB before ingesting
doc-retrieval ingest ./my-documents --clear
3. CLI QueryBash# Single question
doc-retrieval query "What does the contract say about termination?"

# Interactive Chat Mode
doc-retrieval interactive
4. MaintenanceBash# Check database status
doc-retrieval status

# Physically purge database files (reclaim disk space)
doc-retrieval purge
🏗️ Project StructureBashdoc-retrieval/
├── src/doc_retrieval/
│   ├── ingestion/       # PDF Parsing (marker-pdf) & Chunking
│   ├── retrieval/       # Vector Search & LLM Chain
│   ├── vectorstore/     # ChromaDB Management
│   └── ui/              # Gradio App Code
├── data/                # Vector Database Storage
└── pyproject.toml       # Dependencies & Build Config
```

<div align="center">Made using LlamaIndex & Ollama</div>
