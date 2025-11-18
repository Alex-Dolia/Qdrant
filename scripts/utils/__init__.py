"""
Utility functions for the legal document processing pipeline.
"""

from .model_utils import is_ollama_model, get_ollama_model_name, get_embedding_model_name
from .embedding_utils import get_embeddings, EmbeddingWrapper

__all__ = [
    'is_ollama_model',
    'get_ollama_model_name',
    'get_embedding_model_name',
    'get_embeddings',
    'EmbeddingWrapper'
]
