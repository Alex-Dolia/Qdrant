# Quick Reference Card

Quick commands and URLs for the RAG Query System.

## Quick Commands

### Start Services

```bash
# Start Qdrant
start_qdrant.bat
# or
docker run -p 6333:6333 qdrant/qdrant

# Start Ollama (usually auto-starts)
ollama serve

# Start App
streamlit run streamlit_app.py
```

### Verify Services

```bash
# Check Qdrant
curl http://localhost:6333/health

# Check Ollama
ollama list

# Check Docker
docker ps
```

## Important URLs

- **Streamlit App:** http://localhost:8501
- **Qdrant Dashboard:** http://localhost:6333/dashboard
- **Qdrant API:** http://localhost:6333

## Ports

- **Streamlit:** 8501
- **Qdrant HTTP:** 6333
- **Qdrant gRPC:** 6334
- **Ollama:** 11434

## Environment Variables

```bash
# Qdrant (optional)
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# Ollama (optional)
OLLAMA_BASE_URL=http://localhost:11434
```

## File Locations

- **Logs:** `logs/`
- **Temp files:** `temp/`
- **Qdrant data:** Docker volume (if persistent)

## Common Commands

```bash
# Check Python version
python --version

# Install dependencies
pip install -r requirements.txt

# List Ollama models
ollama list

# Pull Llama 3.1
ollama pull llama3.1

# Check Docker containers
docker ps

# Stop Qdrant
docker stop $(docker ps -q --filter ancestor=qdrant/qdrant)
```

## Troubleshooting Quick Fixes

- **Qdrant not connecting:** Restart Docker, restart Qdrant container
- **Ollama not responding:** `ollama serve`, check port 11434
- **Import errors:** `pip install -r requirements.txt`
- **Port conflicts:** Change port in command or config

