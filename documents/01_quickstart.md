# Quick Start Guide

Get up and running with the RAG Query System in 5 minutes.

## Prerequisites

- Python 3.11+ installed
- Docker Desktop installed and running
- Ollama installed with Llama 3.1 model

## Quick Start Steps

### 1. Start Qdrant (30 seconds)

```bash
start_qdrant.bat
```

Or manually:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 2. Verify Ollama is Running (10 seconds)

```bash
ollama list
```

If `llama3.1` is not listed:
```bash
ollama pull llama3.1
```

### 3. Start the Application (10 seconds)

```bash
streamlit run streamlit_app.py
```

The app will automatically open in your browser.

### 4. Upload and Query Documents (2 minutes)

1. **Upload a document:**
   - Go to sidebar "📚 Document Management"
   - Click "Upload documents"
   - Select a PDF, DOCX, TXT, or MD file
   - Click "Process"

2. **Query your documents:**
   - Go to the "⚖️ Legal Documents Query" tab
   - Enter your question
   - Click "Search Legal Documents"

3. **Try other features:**
   - **Simple Web Search:** Quick web search without documents
   - **Research Overview:** Generate comprehensive research papers
   - **Chunk Exploration:** Explore how documents are chunked

## First Query Example

Try this query after uploading a document:

```
What are the main topics discussed in this document?
```

## Common Issues

**Qdrant not connecting:**
- Ensure Docker Desktop is running
- Check: http://localhost:6333/dashboard

**Ollama not responding:**
- Ensure Ollama is running: `ollama serve`
- Check model: `ollama list`

**No results found:**
- Make sure documents are processed (click "Process" button)
- Check processed files list in sidebar

## Next Steps

- See `documents/02_setup_ollama.md` for Ollama details
- See `documents/03_qdrant_setup.md` for Qdrant details
- See `documents/05_running_the_app.md` for app usage

