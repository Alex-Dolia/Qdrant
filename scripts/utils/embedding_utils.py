"""
Utility functions for text embedding with Ollama models.
"""
import os
from typing import List, Optional, Union, Dict, Any
import logging
import numpy as np
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

class EmbeddingWrapper:
    """Wrapper class for handling Ollama embedding models."""
    
    def __init__(self, model_name: str):
        """
        Initialize the embedding wrapper.
        
        Args:
            model_name: Name of the Ollama model to use (e.g., 'llama3.1:latest')
        """
        self.model_name = model_name
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of text documents using the specified Ollama model.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors as lists of floats
        """
        if not texts:
            return []
            
        try:
            # Get embeddings for all texts
            embeddings = get_embeddings_batch(texts, self.model_name)
            
            # Convert numpy arrays to lists
            return [emb.tolist() for emb in embeddings]
            
        except Exception as e:
            logger.error(f"Error generating embeddings with {self.model_name}: {e}")
            raise

def get_embeddings(texts: List[str], model_name: str) -> List[List[float]]:
    """
    Get embeddings for a list of texts using the specified Ollama model.
    
    Args:
        texts: List of text strings to embed
        model_name: Name of the Ollama model to use (e.g., 'llama3.1:latest')
        
    Returns:
        List of embedding vectors as lists of floats
    """
    try:
        wrapper = EmbeddingWrapper(model_name)
        return wrapper.embed_documents(texts)
    except Exception as e:
        logger.error(f"Failed to get embeddings: {e}")
        raise
