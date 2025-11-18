"""
Ollama Connection Health Check Utility

Provides functions to check Ollama connection status and health using the Python SDK.
"""

import os
import time
import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

def get_ollama_embedding(text: str, model: str = "llama3.1:latest") -> np.ndarray:
    """Get embedding using Ollama's Python SDK."""
    try:
        import ollama
        response = ollama.embeddings(model=model, prompt=text)
        return np.array(response["embedding"], dtype=np.float32)
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        raise

def generate_with_ollama(prompt: str, model: str = "llama3.1:latest") -> str:
    """Generate text using Ollama's Python SDK."""
    try:
        import ollama
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1}
        )
        return response["message"]["content"]
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise

def check_ollama_connection(
    model: str = "llama3.1:latest",
    timeout: float = 10.0
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Check if Ollama is accessible and responding using the Python SDK.
    
    Args:
        model: Model to check (default: llama3.1:latest)
        timeout: Maximum time to wait for the test to complete
    
    Returns:
        Tuple of (is_connected: bool, error_message: Optional[str], info: Optional[Dict])
    """
    start_time = time.time()
    
    try:
        import ollama
        
        # Test embedding
        test_text = "Test connection"
        start_emb = time.time()
        embedding = get_ollama_embedding(test_text, model)
        emb_time = (time.time() - start_emb) * 1000  # ms
        
        # Test generation
        test_prompt = "Generate a short test sentence."
        start_gen = time.time()
        generated = generate_with_ollama(test_prompt, model)
        gen_time = (time.time() - start_gen) * 1000  # ms
        
        total_time = (time.time() - start_time) * 1000  # ms
        
        info = {
            "embedding_time_ms": round(emb_time, 2),
            "generation_time_ms": round(gen_time, 2),
            "total_time_ms": round(total_time, 2),
            "embedding_shape": embedding.shape if hasattr(embedding, 'shape') else len(embedding),
            "generation_sample": generated[:100] + "..." if len(generated) > 100 else generated,
            "model": model,
            "status": "connected"
        }
        
        logger.info(f"Ollama connection test successful (total: {total_time:.2f}ms)")
        return True, None, info
        
    except ImportError:
        error_msg = "Ollama Python package not installed. Install with: pip install ollama"
        logger.error(error_msg)
        return False, error_msg, None
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Ollama connection test failed: {error_msg}")
        return False, error_msg, None

def display_ollama_status_in_ui(streamlit_module=None):
    """
    Display Ollama connection status in Streamlit UI.
    
    Args:
        streamlit_module: Streamlit module (usually `st`)
    """
    if streamlit_module is None:
        try:
            import streamlit as st
            streamlit_module = st
        except ImportError:
            logger.warning("Streamlit not available, cannot display status")
            return
    
    with streamlit_module.spinner("🔍 Testing Ollama connection..."):
        is_connected, error_msg, info = check_ollama_connection()
    
    if is_connected and info:
        streamlit_module.success("✅ Ollama is connected and ready!")
        with streamlit_module.expander("🔧 Ollama Status Details", expanded=False):
            streamlit_module.json({
                "Model": info.get("model", "llama3.1:latest"),
                "Embedding Time": f"{info.get('embedding_time_ms', 0):.2f} ms",
                "Generation Time": f"{info.get('generation_time_ms', 0):.2f} ms",
                "Total Test Time": f"{info.get('total_time_ms', 0):.2f} ms",
                "Embedding Shape": str(info.get("embedding_shape", "N/A")),
                "Status": "Connected and Functional"
            })
            
            # Show a sample generation
            if "generation_sample" in info:
                streamlit_module.markdown("**Test Generation:**")
                streamlit_module.info(f'"{info["generation_sample"]}"')
    else:
        streamlit_module.error(f"❌ Ollama is not available: {error_msg}")
        
        # More helpful error messages
        if "Connection refused" in error_msg or "Could not connect" in error_msg:
            streamlit_module.warning("""
            **Ollama server might not be running.**
            
            1. Make sure Ollama is installed and running
            2. Start the Ollama server in a terminal:
            ```bash
            ollama serve
            ```
            """)
        
        streamlit_module.info("💡 Installation and Setup:")
        streamlit_module.code("""
# Install the Ollama Python package
pip install ollama

# Pull the required models (in a separate terminal):
ollama pull llama3.1
ollama pull nomic-embed-text
        """, language="bash")

# For backward compatibility
get_ollama_status = check_ollama_connection
