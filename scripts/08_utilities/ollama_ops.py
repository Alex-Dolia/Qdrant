"""
Ollama Operations Module

Provides functions for working with Ollama models for embeddings and text generation.
"""

import os
import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

# Default models
DEFAULT_EMBEDDING_MODEL = "llama3.1:latest"
DEFAULT_GENERATION_MODEL = "llama3.1:latest"

# Ollama API settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_ENDPOINT = f"{OLLAMA_BASE_URL}/api/embeddings"
GENERATE_ENDPOINT = f"{OLLAMA_BASE_URL}/api/generate"
TAGS_ENDPOINT = f"{OLLAMA_BASE_URL}/api/tags"

# Cache for embeddings to avoid redundant API calls
_embedding_cache = {}

def get_embedding(text: str, model: str = DEFAULT_EMBEDDING_MODEL) -> List[float]:
    """
    Get embedding for a single text using Ollama.
    
    Args:
        text: The text to embed
        model: The Ollama model to use for embeddings
        
    Returns:
        List of floats representing the embedding vector
    """
    # Check cache first
    cache_key = f"{model}:{text}"
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]
    
    try:
        response = requests.post(
            EMBEDDING_ENDPOINT,
            json={"model": model, "prompt": text},
            timeout=60
        )
        response.raise_for_status()
        
        embedding = response.json().get("embedding")
        if not embedding:
            raise ValueError("No embedding returned from Ollama API")
            
        # Cache the result
        _embedding_cache[cache_key] = embedding
        return embedding
        
    except RequestException as e:
        error_msg = f"Error generating Ollama embeddings: {str(e)}"
        if hasattr(e.response, 'status_code') and e.response.status_code == 404:
            error_msg = f"Model '{model}' not found, try pulling it first (status code: {e.response.status_code})"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

def get_embeddings(texts: List[str], model: str = DEFAULT_EMBEDDING_MODEL) -> List[List[float]]:
    """
    Get embeddings for multiple texts using Ollama.
    
    Args:
        texts: List of texts to embed
        model: The Ollama model to use for embeddings
        
    Returns:
        List of embedding vectors (list of lists of floats)
    """
    return [get_embedding(text, model=model) for text in texts]

def generate_text(
    prompt: str, 
    model: str = DEFAULT_GENERATION_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    system_prompt: str = "You are a helpful AI assistant."
) -> str:
    """
    Generate text using Ollama's chat completion API.
    
    Args:
        prompt: The user's prompt
        model: The Ollama model to use for generation
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        system_prompt: System message to set the behavior of the assistant
        
    Returns:
        Generated text as a string
    """
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "options": {
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            },
            timeout=120
        )
        response.raise_for_status()
        
        return response.json()["message"]["content"]
        
    except RequestException as e:
        error_msg = f"Error generating text with Ollama: {str(e)}"
        if hasattr(e, 'response') and e.response.status_code == 404:
            error_msg = f"Model '{model}' not found, try pulling it first (status code: {e.response.status_code})"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

def check_ollama_models(required_models: List[str] = None) -> Dict[str, bool]:
    """
    Check if required Ollama models are available.
    
    Args:
        required_models: List of model names to check (default: [DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATION_MODEL])
        
    Returns:
        Dictionary mapping model names to availability status (True/False)
    """
    if required_models is None:
        required_models = [DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATION_MODEL]
    
    try:
        # Get list of available models
        response = requests.get(TAGS_ENDPOINT, timeout=10)
        response.raise_for_status()
        
        available_models = {model["name"] for model in response.json().get("models", [])}
        
        # Check which required models are available
        return {model: model in available_models for model in required_models}
        
    except RequestException as e:
        logger.error(f"Error checking Ollama models: {e}")
        return {model: False for model in required_models}

def run_self_test() -> Dict[str, Any]:
    """
    Run a self-test of Ollama functionality.
    
    Returns:
        Dictionary with test results including:
        - success: bool indicating if all tests passed
        - models_available: dict of model availability
        - embedding_test: dict with embedding test results
        - generation_test: dict with text generation test results
    """
    results = {
        "success": False,
        "models_available": {},
        "embedding_test": {"success": False, "error": None, "embedding": None},
        "generation_test": {"success": False, "error": None, "response": None},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        # Check model availability
        results["models_available"] = check_ollama_models([DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATION_MODEL])
        
        # Test embedding
        test_text = "This is a test for Ollama embeddings."
        embedding = get_embedding(test_text, model=DEFAULT_EMBEDDING_MODEL)
        results["embedding_test"]["embedding"] = f"Vector of length {len(embedding)}"
        results["embedding_test"]["success"] = True
        
        # Test text generation
        response = generate_text("Say 'Hello, Ollama!', but don't say anything else.", model=DEFAULT_GENERATION_MODEL)
        results["generation_test"]["response"] = response.strip()
        results["generation_test"]["success"] = True
        
        results["success"] = True
        
    except Exception as e:
        logger.error(f"Ollama self-test failed: {e}", exc_info=True)
        if not results["embedding_test"]["success"]:
            results["embedding_test"]["error"] = str(e)
        if not results["generation_test"]["success"]:
            results["generation_test"]["error"] = str(e)
    
    return results

def ensure_models_available(required_models: List[str] = None) -> bool:
    """
    Ensure required Ollama models are available, pulling them if necessary.
    
    Args:
        required_models: List of model names to check/pull
        
    Returns:
        bool: True if all models are available, False otherwise
    """
    if required_models is None:
        required_models = [DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATION_MODEL]
    
    # Check which models are already available
    available_models = check_ollama_models(required_models)
    missing_models = [model for model, available in available_models.items() if not available]
    
    if not missing_models:
        return True
    
    logger.warning(f"Missing Ollama models: {', '.join(missing_models)}")
    
    # Try to pull missing models
    for model in missing_models:
        try:
            logger.info(f"Pulling Ollama model: {model}")
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/pull",
                json={"name": model},
                timeout=300  # 5 minute timeout for model downloads
            )
            response.raise_for_status()
            
            # Wait for the model to be fully pulled
            for line in response.iter_lines():
                if line:
                    status = json.loads(line)
                    if status.get("status") == "success":
                        logger.info(f"Successfully pulled model: {model}")
                        break
                    
        except Exception as e:
            logger.error(f"Failed to pull model {model}: {e}")
            return False
    
    # Verify all models are now available
    available_models = check_ollama_models(required_models)
    return all(available_models.values())
