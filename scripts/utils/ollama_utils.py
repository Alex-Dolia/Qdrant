"""
Utility functions for working with Ollama models.
"""
import os
import numpy as np
from typing import List, Optional, Dict, Any, Union
import logging
import ollama

logger = logging.getLogger(__name__)

# Default embedding models to use
DEFAULT_EMBEDDING_MODELS = ["llama3.1:latest", "nomic-embed-text:latest"]

def get_embedding(text: str, model: str) -> np.ndarray:
    """
    Get embedding vector from Ollama model.
    
    Args:
        text: Input text to embed
        model: Name of the Ollama model to use
        
    Returns:
        Numpy array containing the embedding vector
    """
    response = ollama.embeddings(model=model, prompt=text)
    return np.array(response["embedding"], dtype=np.float32)

def get_embeddings_batch(texts: List[str], model: str) -> List[np.ndarray]:
    """
    Get embeddings for a batch of texts.
    
    Args:
        texts: List of text strings to embed
        model: Name of the Ollama model to use
        
    Returns:
        List of numpy arrays containing the embedding vectors
    """
    return [get_embedding(text, model) for text in texts]

def is_ollama_model_available(model_name: str) -> bool:
    """
    Check if an Ollama model is available locally.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        bool: True if the model is available, False otherwise
    """
    try:
        # List all available models
        models = ollama.list()
        return any(m["name"] == model_name for m in models.get("models", []))
    except Exception as e:
        logger.error(f"Error checking for Ollama model {model_name}: {e}")
        return False

def ensure_ollama_model(model_name: str) -> bool:
    """
    Ensure the specified Ollama model is available, pulling it if necessary.
    
    Args:
        model_name: Name of the model to ensure is available
        
    Returns:
        bool: True if the model is available, False otherwise
    """
    if is_ollama_model_available(model_name):
        return True
        
    logger.info(f"Pulling Ollama model: {model_name}")
    try:
        ollama.pull(model_name)
        return True
    except Exception as e:
        logger.error(f"Failed to pull Ollama model {model_name}: {e}")
        return False
