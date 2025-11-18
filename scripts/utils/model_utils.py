"""
Utility functions for handling Ollama models.
"""
import os
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)

def is_ollama_model(model_name: str) -> bool:
    """
    Check if the model is an Ollama model.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        bool: True if the model is an Ollama model, False otherwise
    """
    return "ollama" in model_name or ":" in model_name

def get_ollama_model_name(model_name: str) -> str:
    """
    Convert an Ollama model name to the format expected by Ollama API.
    
    Args:
        model_name: The input model name (can have 'ollama/' prefix)
        
    Returns:
        str: The cleaned model name
    """
    # Remove 'ollama/' prefix if present
    if "ollama/" in model_name:
        model_name = model_name.replace("ollama/", "")
    
    # Remove any remaining slashes
    return model_name.replace("/", "-")

def get_embedding_model_name(model_name: str) -> str:
    """
    Get the appropriate Ollama model name.
    
    Args:
        model_name: The input model name
        
    Returns:
        str: The cleaned Ollama model name
    """
    return get_ollama_model_name(model_name)
