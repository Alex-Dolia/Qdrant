# Ollama Setup Guide

Setup guide for Ollama and Llama 3.1 embeddings.

## What is Ollama?

Ollama is a tool for running large language models locally. We use it to generate embeddings with Llama 3.1.

## Installation

### Windows

1. Download from [ollama.ai](https://ollama.ai/download)
2. Run the installer
3. Ollama will start automatically

### macOS

```bash
brew install ollama
```

Or download from [ollama.ai](https://ollama.ai/download)

### Linux

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

## Verify Installation

```bash
ollama --version
```

## Pull Llama 3.1 Model

```bash
ollama pull llama3.1
```

This downloads the model (approximately 4.7GB).

## Verify Model Installation

```bash
ollama list
```

You should see:
```
NAME        ID              SIZE    MODIFIED
llama3.1    abc123...       4.7GB   2 hours ago
```

## Start Ollama Service

Ollama usually starts automatically. If not:

```bash
ollama serve
```

## Test Ollama

```bash
ollama run llama3.1 "Hello, how are you?"
```

## Configuration

### Default Port

Ollama runs on port `11434` by default.

### Environment Variables

Set if needed:
```bash
export OLLAMA_BASE_URL=http://localhost:11434
```

Windows:
```cmd
set OLLAMA_BASE_URL=http://localhost:11434
```

## Troubleshooting

### Ollama not starting
- Check if port 11434 is available
- Restart Ollama service
- Check logs: `ollama serve` (run in terminal)

### Model not found
- Pull the model: `ollama pull llama3.1`
- Verify: `ollama list`

### Connection refused
- Ensure Ollama service is running
- Check firewall settings
- Verify port 11434 is not blocked

## Next Steps

- See `documents/03_qdrant_setup.md` for Qdrant setup
- See `documents/05_running_the_app.md` for using the app

