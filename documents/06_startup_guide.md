# Startup Guide

Complete startup sequence after system reboot.

## Startup Sequence

Follow these steps in order:

### Step 1: Start Docker Desktop (1 minute)

1. Launch Docker Desktop
2. Wait for it to fully start (whale icon stable)
3. Verify: Docker icon shows "Docker Desktop is running"

### Step 2: Start Qdrant (30 seconds)

```bash
start_qdrant.bat
```

Or manually:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

Verify: Open http://localhost:6333/dashboard

### Step 3: Start Ollama (10 seconds)

Ollama usually starts automatically. If not:

```bash
ollama serve
```

Verify:
```bash
ollama list
```

Should show `llama3.1` model.

### Step 4: Start the Application (10 seconds)

```bash
streamlit run streamlit_app.py
```

Verify: Browser opens at http://localhost:8501

## Quick Startup Script

Create `start_all.bat`:

```batch
@echo off
echo Starting RAG System...

echo Starting Qdrant...
start /B docker run -p 6333:6333 qdrant/qdrant

timeout /t 5

echo Starting Ollama...
start /B ollama serve

timeout /t 5

echo Starting Streamlit...
streamlit run streamlit_app.py
```

## Verification Checklist

- [ ] Docker Desktop running
- [ ] Qdrant accessible at http://localhost:6333/dashboard
- [ ] Ollama responding: `ollama list` works
- [ ] Streamlit app running at http://localhost:8501
- [ ] Can upload documents
- [ ] Can query documents

## Troubleshooting

### Docker not starting
- Check Docker Desktop installation
- Restart computer if needed
- Check system requirements

### Qdrant not accessible
- Check Docker container: `docker ps`
- Check port 6333: `netstat -an | findstr 6333`
- Restart Qdrant container

### Ollama not responding
- Check Ollama service: `ollama serve`
- Check port 11434: `netstat -an | findstr 11434`
- Restart Ollama

## Next Steps

- See `documents/07_startup_quick_reference.md` for quick reference
- See `documents/05_running_the_app.md` for app usage

