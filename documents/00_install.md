# Installation Guide

Complete installation guide for the RAG Query System using Qdrant and Llama 3.1.

## Prerequisites

- **Python 3.9+** (3.11+ recommended) - Download from [python.org](https://www.python.org/downloads/) or use Conda
- **Conda** (optional but recommended) - Download from [conda.io](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)
- **Docker Desktop** - Required for Qdrant (download from [docker.com](https://www.docker.com/products/docker-desktop))
- **Git** - For cloning the repository (optional)
- **8GB+ RAM** - Recommended for optimal performance

## Step 0: Setup Python Environment

### Option A: Using Conda (Recommended)

1. **Install Conda** (if not already installed):
   - Download Miniconda: https://docs.conda.io/en/latest/miniconda.html
   - Or download Anaconda: https://www.anaconda.com/
   - Follow the installation instructions for your operating system

2. **Create a new Conda environment**:
   ```bash
   conda create -n rag_system python=3.11 -y
   ```

3. **Activate the environment**:
   ```bash
   # On Windows
   conda activate rag_system
   
   # On Linux/Mac
   conda activate rag_system
   ```

4. **Verify Python version**:
   ```bash
   python --version
   ```
   Should show Python 3.11.x or higher.

### Option B: Using Python venv (Alternative)

1. **Verify Python version**:
   ```bash
   python --version
   ```
   Should be Python 3.9+ (3.11+ recommended).

2. **Create a virtual environment**:
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   
   # On Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Verify activation**:
   - Your terminal prompt should show `(venv)` or `(rag_system)` prefix

## Step 1: Install Python Dependencies

**Important:** Make sure your conda environment or virtual environment is activated before installing dependencies.

1. Navigate to the project directory
2. Install all dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

   **Note:** This single `requirements.txt` file contains all necessary dependencies for the entire project, including:
   - Core UI framework (Streamlit)
   - Vector database client (Qdrant)
   - LangChain components
   - Document processing libraries
   - Embeddings and ML models
   - Web search functionality (DeepSeek)
   - Research and academic tools
   - All other dependencies

3. **Verify installation** (optional):
   ```bash
   pip list | grep streamlit
   pip list | grep qdrant
   pip list | grep langchain
   ```

## Step 2: Install and Setup Ollama

See `documents/02_setup_ollama.md` for detailed Ollama installation and Llama 3.1 model setup.

Quick steps:
1. Download Ollama from [ollama.ai](https://ollama.ai)
2. Install and start Ollama
3. Pull the Llama 3.1 model:
   ```bash
   ollama pull llama3.1
   ```

## Step 3: Setup Qdrant

See `documents/03_qdrant_setup.md` for detailed Qdrant setup.

Quick steps:
1. Start Qdrant using Docker:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```
   Or use the provided script:
   ```bash
   start_qdrant.bat
   ```

2. Verify Qdrant is running:
   - Open browser: http://localhost:6333/dashboard
   - You should see the Qdrant dashboard

## Step 4: Verify Installation

1. **Check Python version**:
   ```bash
   python --version
   ```
   Should be 3.9+ (3.11+ recommended).

2. **Verify environment is active**:
   ```bash
   # For Conda
   conda info --envs
   # Should show asterisk (*) next to rag_system
   
   # For venv
   # Terminal prompt should show (venv) prefix
   ```

3. **Check Docker**:
   ```bash
   docker --version
   ```

4. **Check Ollama**:
   ```bash
   ollama list
   ```
   Should show `llama3.1` in the list.

## Step 5: Run the Application

**Important:** Make sure your conda environment or virtual environment is activated.

```bash
# Activate environment first (if not already active)
# For Conda:
conda activate rag_system

# For venv (Windows):
venv\Scripts\activate

# For venv (Linux/Mac):
source venv/bin/activate

# Then run the app
streamlit run streamlit_app.py
```

The app will open in your browser at http://localhost:8501

## Troubleshooting

### Qdrant Connection Issues
- Ensure Docker Desktop is running
- Check if port 6333 is available
- Verify Qdrant container is running: `docker ps`

### Ollama Connection Issues
- Ensure Ollama service is running
- Check if port 11434 is available
- Verify model is installed: `ollama list`

### Python Import Errors
- **Ensure environment is activated**: Check that your conda/venv environment is active
- **Ensure all dependencies are installed**: `pip install -r requirements.txt`
- **Check Python version**: `python --version` (should be 3.9+, preferably 3.11+)
- **Try reinstalling**: `pip install --upgrade -r requirements.txt`
- **For Conda users**: If issues persist, try: `conda install pip` then `pip install -r requirements.txt`

## Next Steps

- See `documents/01_quickstart.md` for a quick start guide
- See `documents/05_running_the_app.md` for running the app

