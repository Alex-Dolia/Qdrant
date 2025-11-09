# Running the App

Guide to running the RAG Query System application.

## Prerequisites Check

Before running, ensure:

1. **Qdrant is running:**
   ```bash
   docker ps
   ```
   Should show Qdrant container running.

2. **Ollama is running:**
   ```bash
   ollama list
   ```
   Should show `llama3.1` model.

3. **Dependencies installed:**
   ```bash
   pip list | grep streamlit
   ```

## Starting the Application

### Standard Start

```bash
streamlit run streamlit_app.py
```

The app will:
- Start on http://localhost:8501
- Open automatically in your browser
- Show any errors in the terminal

### Custom Port

```bash
streamlit run streamlit_app.py --server.port 8502
```

### Custom Address

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0
```

## Application Workflow

1. **Start Qdrant** (if not running)
2. **Start Ollama** (if not running)
3. **Run Streamlit app**
4. **Upload documents** in sidebar
5. **Process documents** (click "Process")
6. **Query documents** or **Generate reports**

## Stopping the Application

- Press `Ctrl+C` in the terminal
- Or close the browser tab and stop the terminal process

## Common Issues

### Port 8501 already in use
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Qdrant connection error
- Check Qdrant is running: http://localhost:6333/dashboard
- Restart Qdrant if needed

### Ollama connection error
- Check Ollama is running: `ollama list`
- Start Ollama: `ollama serve`

### Import errors
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version`

## Performance Tips

- Close unused browser tabs
- Process documents one at a time
- Use smaller documents for faster processing
- Clear browser cache if app is slow

## Next Steps

- See `documents/06_startup_guide.md` for startup sequence
- See `documents/07_startup_quick_reference.md` for quick reference

