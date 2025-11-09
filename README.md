# RAG Query System - Qdrant + Llama 3.1

A simplified RAG-powered system for querying documents and generating research reports using Qdrant vector database and Llama 3.1 embeddings.

## Features

- Document ingestion for PDF, DOCX, TXT, and MD files
- Multiple chunking methods (recursive, semantic, structural, agentic, cluster)
- Vector search using Qdrant with multiple search modes (semantic, BM25, mixed)
- Query legal documents with advanced filtering
- Simple web search functionality
- Research overview paper generation
- RAG evaluation with RAGAS
- Chunk exploration and visualization
- Five-tab interface with comprehensive features

## Quick Start

1. **Setup Python environment** (Conda recommended):
   ```bash
   # Create conda environment
   conda create -n rag_system python=3.11 -y
   conda activate rag_system
   
   # Or use venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   **Note:** This single `requirements.txt` file contains all dependencies needed for the entire project.

2. **Start Qdrant:**
   ```bash
   start_qdrant.bat
   ```
   Or manually:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

4. **Start Ollama (if not running):**
   ```bash
   ollama serve
   ```
   Then pull the model:
   ```bash
   ollama pull llama3.1
   ```

5. **Run the app:**
   ```bash
   streamlit run streamlit_app.py
   ```

See `documents/00_install.md` for detailed setup instructions.
See `documents/02_setup_ollama.md` for Ollama setup.
See `documents/03_qdrant_setup.md` for Qdrant setup.

## Documentation

- `documents/00_install.md` - Complete installation guide
- `documents/01_quickstart.md` - Quick start walkthrough
- `documents/02_setup_ollama.md` - Ollama setup
- `documents/03_qdrant_setup.md` - Qdrant setup
- `documents/04_readme_streamlit.md` - App documentation
- `documents/05_running_the_app.md` - Running the app
- `documents/06_startup_guide.md` - **Startup sequence after reboot** ⭐
- `documents/07_startup_quick_reference.md` - Quick reference card
- `documents/08_changelog.md` - Project changelog
- `documents/09_qdrant_performance_optimizations.md` - Qdrant optimizations
- `documents/10_research_assistant_architecture.md` - Research assistant architecture guide
- `documents/11_research_overview_workflow.md` - Research overview workflow guide

## System Requirements

- **Python 3.9+** (3.11+ recommended)
- **Conda** (optional but recommended) or Python venv
- **Docker Desktop** (for Qdrant)
- **Ollama** installed and running
- **8GB+ RAM** recommended

See `documents/00_install.md` for detailed setup instructions including conda environment setup.

## Architecture

- **Vector Database:** Qdrant (self-hosted)
- **Embeddings:** Llama 3.1 via Ollama
- **UI:** Streamlit
- **Web Search:** DuckDuckGo (free, no API key needed)

## License

Prototype for interview preparation.
"# Qdrant" 
