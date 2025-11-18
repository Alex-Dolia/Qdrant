"""
Production-Ready Advanced Reranking System for Legal Document RAG

This module provides a comprehensive reranking system specifically designed for legal documents,
integrating multiple retrieval methods, legal-specific optimizations, and ensemble ranking techniques.

Key Features:
- Hybrid search (semantic, BM25, n-gram)
- Legal clause type detection and weighting
- Obligation pattern recognition
- Cross-reference resolution
- Party-specific relevance boosting
- Llama 3 70B reranking via Together.ai
- Reciprocal Rank Fusion for ensemble methods
- Async API support with fallbacks
- Comprehensive caching and monitoring

Legal AI Best Practices:
- Accuracy prioritized over speed for contractual obligations
- Detailed logging for audit trails
- Configurable parameters for different legal contexts
- Safeguards against hallucination in legal interpretations
"""

import os
import re
import json
import time
import logging
import asyncio
import hashlib
from typing import List, Dict, Any, Tuple, Optional, Union, Literal
from dataclasses import dataclass, asdict, field
from enum import Enum
from functools import lru_cache
from datetime import datetime
from collections import defaultdict

import numpy as np
import ollama
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
from rank_bm25 import BM25Okapi

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except ImportError:
    nltk = None
    word_tokenize = lambda x: x.split()
    sent_tokenize = lambda x: [x]

try:
    from together import Together
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False
    Together = None

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    retry = lambda **kwargs: lambda f: f

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from pydantic import BaseModel, Field, validator

# Configure logging
logger = logging.getLogger(__name__)

class LegalDocumentProcessor:
    """Processes legal documents for RAG, with specialized handling for legal text.
    
    Includes legal-specific text cleaning, chunking, and metadata extraction.
    """
    
    def __init__(self):
        """Initialize the document processor with legal-specific patterns and configurations."""
        # Legal clause patterns for classification
        self.clause_patterns = {
            'definition': r'(?i)(?:definition|means|shall mean|refers to)[^.]*\b(?:the|a|an)?\s+[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*\b',
            'obligation': r'(?i)(?:shall|must|will|agrees? to|covenants? to|undertakes? to)\s+[A-Za-z]',
            'representation': r'(?i)(?:represents?|warrants?|certifies?|acknowledges?|confirms?)(?:\s+that)?\s+[A-Z]',
            'condition': r'(?i)(?:condition(?:s|ed)? upon|subject to|provided that|in the event(?: that)?|if)',
            'indemnity': r'(?i)indemn(?:ify|ification|ity)',
            'limitation': r'(?i)(?:limitation(?:s)? of liability|liability limit|cap(?:ped)? at|maximum liability)',
        }
        
        # Legal stopwords and phrases to remove
        self.legal_stopwords = {
            'hereby', 'herein', 'hereof', 'hereto', 'hereunder', 'thereby', 'therein', 'thereof',
            'thereto', 'thereunder', 'whereas', 'witnesseth', 'notwithstanding', 'pursuant to',
            'in accordance with', 'in the event of', 'in the event that', 'for the avoidance of doubt',
            'without limiting the foregoing', 'subject to the terms and conditions', 'including but not limited to',
        }
    
    def clean_legal_text(self, text: str) -> str:
        """Clean and normalize legal text."""
        # Remove multiple spaces and newlines
        text = ' '.join(text.split())
        # Remove section numbers (e.g., "1.1", "(a)")
        text = re.sub(r'\b\d+[.\-]\d+\b', ' ', text)
        text = re.sub(r'\s*\([a-zA-Z0-9]+\)\s*', ' ', text)
        # Remove legal stopwords
        for stopword in self.legal_stopwords:
            text = re.sub(r'\b' + re.escape(stopword) + r'\b', ' ', text, flags=re.IGNORECASE)
        return text.strip()
    
    def chunk_by_section(self, text: str, min_length: int = 100, max_length: int = 1000) -> List[Dict[str, Any]]:
        """
        Chunk text by sections (e.g., 1.1, 1.2, etc.)
        
        Args:
            text: Input text to chunk
            min_length: Minimum chunk length in characters
            max_length: Maximum chunk length in characters
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Split by common section patterns
        sections = re.split(r'(\n\s*\d+(?:\.\d+)*\s+[^\n]+)', text)
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
                
            # If this looks like a section header
            if i % 2 == 1 and len(section) < 100:  # Section headers are usually short
                if current_chunk and current_length >= min_length:
                    chunks.append({
                        'text': ' '.join(current_chunk),
                        'type': 'section',
                        'section_title': section.strip(),
                        'length': current_length
                    })
                current_chunk = [section]
                current_length = len(section)
            else:
                # Check if adding this section would exceed max_length
                if current_length + len(section) > max_length and current_chunk:
                    chunks.append({
                        'text': ' '.join(current_chunk),
                        'type': 'section',
                        'section_title': 'Continued',
                        'length': current_length
                    })
                    current_chunk = []
                    current_length = 0
                
                current_chunk.append(section)
                current_length += len(section)
        
        # Add the last chunk if not empty
        if current_chunk and current_length >= min_length:
            chunks.append({
                'text': ' '.join(current_chunk),
                'type': 'section',
                'section_title': 'Continued',
                'length': current_length
            })
            
        return chunks
    
    def chunk_by_sentence(self, text: str, min_sentences: int = 3, max_sentences: int = 10) -> List[Dict[str, Any]]:
        """
        Chunk text by sentences, combining them into chunks of similar size.
        
        Args:
            text: Input text to chunk
            min_sentences: Minimum number of sentences per chunk
            max_sentences: Maximum number of sentences per chunk
            
        Returns:
            List of chunks with metadata
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        
        for sentence in sentences:
            current_chunk.append(sentence)
            
            if len(current_chunk) >= min_sentences and (
                len(current_chunk) >= max_sentences or 
                len(' '.join(current_chunk)) > 500  # Approximate character limit
            ):
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'type': 'sentence',
                    'sentence_count': len(current_chunk),
                    'length': len(chunk_text)
                })
                current_chunk = []
        
        # Add any remaining sentences
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'type': 'sentence',
                'sentence_count': len(current_chunk),
                'length': len(chunk_text)
            })
            
        return chunks
    
    def chunk_by_paragraph(self, text: str, min_paragraphs: int = 1, max_paragraphs: int = 5) -> List[Dict[str, Any]]:
        """
        Chunk text by paragraphs, combining them into chunks of similar size.
        
        Args:
            text: Input text to chunk
            min_paragraphs: Minimum number of paragraphs per chunk
            max_paragraphs: Maximum number of paragraphs per chunk
            
        Returns:
            List of chunks with metadata
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            # If adding this paragraph would exceed max_length and we have at least min_paragraphs
            if (current_length + para_length > 1000 and 
                len(current_chunk) >= min_paragraphs or 
                len(current_chunk) >= max_paragraphs):
                
                chunks.append({
                    'text': '\n\n'.join(current_chunk),
                    'type': 'paragraph',
                    'paragraph_count': len(current_chunk),
                    'length': current_length
                })
                current_chunk = []
                current_length = 0
                
            current_chunk.append(para)
            current_length += para_length
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append({
                'text': '\n\n'.join(current_chunk),
                'type': 'paragraph',
                'paragraph_count': len(current_chunk),
                'length': current_length
            })
            
        return chunks
    
    def chunk_by_clause_type(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk text by identifying and extracting legal clauses.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of chunks with metadata about clause type
        """
        chunks = []
        
        # First, try to find clause boundaries
        clause_boundaries = []
        for i, (clause_type, pattern) in enumerate(self.clause_patterns.items()):
            for match in re.finditer(pattern, text):
                clause_boundaries.append((match.start(), clause_type, match.group(0)))
        
        # Sort boundaries by position
        clause_boundaries.sort()
        
        # Create chunks based on boundaries
        for i in range(len(clause_boundaries)):
            start_pos, clause_type, clause_title = clause_boundaries[i]
            end_pos = clause_boundaries[i+1][0] if i+1 < len(clause_boundaries) else len(text)
            
            chunk_text = text[start_pos:end_pos].strip()
            if len(chunk_text) < 20:  # Skip very short chunks
                continue
                
            chunks.append({
                'text': chunk_text,
                'type': 'clause',
                'clause_type': clause_type,
                'clause_title': clause_title.strip(),
                'length': len(chunk_text)
            })
        
        # If no clauses found, fall back to sentence chunking
        if not chunks:
            return self.chunk_by_sentence(text)
            
        return chunks
    
    def chunk_by_fixed_size(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Chunk text into fixed-size pieces with overlap.
        
        Args:
            text: Input text to chunk
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            # Try to end at a sentence boundary
            if end < text_length:
                # Look for the nearest sentence end
                sentence_end = text.rfind('. ', start, end) + 1
                if sentence_end > start + chunk_size * 0.7:  # Only if it's not too far back
                    end = sentence_end
            
            chunk_text = text[start:end].strip()
            if chunk_text:  # Only add non-empty chunks
                chunks.append({
                    'text': chunk_text,
                    'type': 'fixed_size',
                    'chunk_size': len(chunk_text),
                    'is_truncated': end < text_length
                })
            
            # Move start position, accounting for overlap
            start = end - overlap if end - overlap > start else end
            
            # Prevent infinite loop
            if start >= text_length - 1:
                break
                
        return chunks
    
    def chunk_by_semantic_units(self, text: str) -> List[Dict[str, Any]]:
        """
        Advanced chunking that combines multiple strategies.
        First tries to chunk by sections, then by clauses, then falls back to paragraphs.
        """
        # Try section-based chunking first
        section_chunks = self.chunk_by_section(text)
        if len(section_chunks) > 1:
            return section_chunks
            
        # Then try clause-based chunking
        clause_chunks = self.chunk_by_clause_type(text)
        if len(clause_chunks) > 1:
            return clause_chunks
            
        # Then try paragraph-based chunking
        para_chunks = self.chunk_by_paragraph(text)
        if len(para_chunks) > 1:
            return para_chunks
            
        # Finally, fall back to fixed-size chunks
        return self.chunk_by_fixed_size(text)

# ... (rest of the code remains the same)

class LegalRAGReranker:
    """Production-ready advanced reranking system for legal document RAG.
    
    Integrates multiple retrieval methods, legal-specific optimizations,
    and ensemble ranking for superior legal document search accuracy.
    """
    
    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str = "legal_documents",
        embedding_model: str = "llama3.1:latest",
        together_api_key: Optional[str] = None,
        enable_reranking: bool = True,
        enable_caching: bool = True
    ):
        """Initialize the legal RAG reranker.
        
        Args:
            qdrant_client: Qdrant client instance
            collection_name: Qdrant collection name
            embedding_model: Ollama model name (e.g., "llama3.1:latest")
            together_api_key: Together.ai API key for reranking
            enable_reranking: Whether to enable Llama 3 reranking
            enable_caching: Whether to enable query result caching
        """
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        
        # Initialize Ollama embedding model
        self.embedding_model = embedding_model
        try:
            # Test the embedding model
            test_embedding = self._get_embeddings(["test"])[0]
            if len(test_embedding) == 0:
                raise RuntimeError("Failed to generate test embedding")
            logger.info(f"Initialized Ollama embedding model: {self.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama model {self.embedding_model}: {e}")
            raise RuntimeError(
                f"Failed to initialize Ollama model {self.embedding_model}. "
                "Make sure Ollama is running and the model is available."
            )
            
        self.processor = LegalDocumentProcessor()
        self.enable_reranking = enable_reranking
        self.enable_caching = enable_caching
        
        # Initialize reranker if enabled
        if enable_reranking:
            self.reranker = Llama3Reranker(api_key=together_api_key)
        else:
            self.reranker = None
        
        # Cache for query results
        self.cache = {} if enable_caching else None
        
        logger.info(f"Initialized LegalRAGReranker with collection: {collection_name}")
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts using the Ollama model."""
        try:
            from langchain_ollama import OllamaEmbeddings
            import os
            
            # Initialize Ollama embeddings
            embeddings = OllamaEmbeddings(
                model=self.embedding_model,
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            )
            
            # Get embeddings for all texts
            if len(texts) == 1:
                # Single text
                return [embeddings.embed_query(texts[0])]
            else:
                # Multiple texts
                return embeddings.embed_documents(texts)
                
        except ImportError:
            raise ImportError("langchain-ollama package is required for embeddings. Install with: pip install langchain-ollama")
        except Exception as e:
            raise RuntimeError(f"Error generating embeddings: {str(e)}")
    
    # ... (rest of the code remains the same)

class LegalRAGConfig:
    """Configuration management for the legal RAG reranking system.
    
    Centralizes configuration via environment variables with sensible defaults.
    Makes the system more maintainable and easier to configure for different environments.
    """
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection_name = os.getenv("QDRANT_COLLECTION", "legal_documents")
        self.together_api_key = os.getenv("TOGETHER_API_KEY")
        self.together_model = os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3-70b-chat-hf")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "llama3.1:latest")
        self.enable_reranking = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
        self.enable_caching = os.getenv("ENABLE_CACHING", "true").lower() == "true"
        
        # RRF weights (can be overridden via environment)
        self.rrf_weights = {
            "semantic": float(os.getenv("WEIGHT_SEMANTIC", "0.4")),
            "bm25": float(os.getenv("WEIGHT_BM25", "0.35")),
            "ngram": float(os.getenv("WEIGHT_NGRAM", "0.25"))
        }
        
        # Normalize weights
        total = sum(self.rrf_weights.values())
        if total > 0:
            self.rrf_weights = {k: v / total for k, v in self.rrf_weights.items()}
    
    def get_reranker_weights(self, methods: List[str]) -> List[float]:
        """Get weights for specified retrieval methods."""
        weights = []
        for method in methods:
            if method in self.rrf_weights:
                weights.append(self.rrf_weights[method])
        # Normalize if needed
        if weights and sum(weights) > 0:
            total = sum(weights)
            weights = [w / total for w in weights]
        return weights


class LegalRAGSystem:
    """Main system class for legal document RAG with advanced reranking.
    
    Provides a high-level interface that wraps all components and simplifies usage.
    This is the recommended entry point for production use.
    """
    
    def __init__(self, config: Optional[LegalRAGConfig] = None, qdrant_client: Optional[QdrantClient] = None):
        """Initialize the Legal RAG System.
        
        Args:
            config: Configuration object (creates default if None)
            qdrant_client: Qdrant client instance (creates default if None)
        """
        self.config = config or LegalRAGConfig()
        
        # Initialize Qdrant client if not provided
        if qdrant_client is None:
            qdrant_client = QdrantClient(url=self.config.qdrant_url)
        
        # Initialize reranker
        self.reranker = LegalRAGReranker(
            qdrant_client=qdrant_client,
            collection_name=self.config.collection_name,
            embedding_model=self.config.embedding_model,
            together_api_key=self.config.together_api_key,
            enable_reranking=self.config.enable_reranking,
            enable_caching=self.config.enable_caching
        )
        
        logger.info("LegalRAGSystem initialized successfully")
    
    async def query_async(
        self,
        query_text: str,
        top_k: int = 10,
        methods: Optional[List[str]] = None,
        filter_condition: Optional[Filter] = None,
        use_reranking: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Async query method for better performance in async contexts.
        
        Args:
            query_text: Search query
            top_k: Number of results
            methods: Retrieval methods to use
            filter_condition: Qdrant filter
            use_reranking: Override reranking setting
        
        Returns:
            List of ranked chunks as dictionaries
        """
        if methods is None:
            methods = ["semantic", "bm25", "ngram"]
        
        ranked_chunks = await self.reranker.retrieve_and_rerank_async(
            query=query_text,
            top_k=top_k,
            methods=methods,
            filter_condition=filter_condition,
            use_reranking=use_reranking
        )
        
        return [chunk.to_dict() for chunk in ranked_chunks]
    
    def query(
        self,
        query_text: str,
        top_k: int = 10,
        methods: Optional[List[str]] = None,
        filter_condition: Optional[Filter] = None,
        use_reranking: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Sync query method (main entry point).
        
        Args:
            query_text: Search query
            top_k: Number of results
            methods: Retrieval methods to use
            filter_condition: Qdrant filter
            use_reranking: Override reranking setting
        
        Returns:
            List of ranked chunks as dictionaries
        """
        if methods is None:
            methods = ["semantic", "bm25", "ngram"]
        
        ranked_chunks = self.reranker.retrieve_and_rerank(
            query=query_text,
            top_k=top_k,
            methods=methods,
            filter_condition=filter_condition,
            use_reranking=use_reranking
        )
        
        return [chunk.to_dict() for chunk in ranked_chunks]


# Example usage function
def example_usage():
    """Example demonstrating retrieval and reranking of contract clauses."""
    from qdrant_client import QdrantClient
    
    # Initialize system using configuration (recommended approach)
    config = LegalRAGConfig()
    qdrant_client = QdrantClient(url=config.qdrant_url)
    system = LegalRAGSystem(config=config, qdrant_client=qdrant_client)
    
    # Example query
    query = "What are the termination conditions for this contract?"
    
    # Query using system wrapper
    results = system.query(
        query_text=query,
        top_k=10,
        methods=["semantic", "bm25", "ngram"],
        use_reranking=True
    )
    
    # Display results
    print(f"\nQuery: {query}\n")
    print(f"Found {len(results)} relevant clauses:\n")
    
    for i, result in enumerate(results, 1):
        print(f"Rank {i}:")
        print(f"  Final Score: {result['scores']['final']:.4f}")
        print(f"  Base Score: {result['scores']['base']:.4f}")
        if result['scores']['rerank'] > 0:
            print(f"  Rerank Score: {result['scores']['rerank']:.4f}")
        print(f"  Clause Type: {result['legal_context']['clause_type']}")
        print(f"  Section: {result['legal_context']['section_id']}")
        print(f"  Is Obligation: {result['legal_context']['is_obligation']}")
        print(f"  Retrieval Method: {result.get('retrieval_method', 'hybrid')}")
        print(f"  Content: {result['content'][:200]}...")
        print()


async def example_usage_async():
    """Example demonstrating async usage for better performance."""
    from qdrant_client import QdrantClient
    
    config = LegalRAGConfig()
    qdrant_client = QdrantClient(url=config.qdrant_url)
    system = LegalRAGSystem(config=config, qdrant_client=qdrant_client)
    
    query = "What are the termination conditions for this contract?"
    results = await system.query_async(
        query_text=query, 
        top_k=10, 
        methods=["semantic", "bm25", "ngram"],
        use_reranking=True
    )
    
    print(f"\nQuery: {query}\n")
    print(f"Found {len(results)} relevant clauses:\n")
    
    for i, result in enumerate(results, 1):
        print(f"Rank {i}: Score={result['scores']['final']:.4f}, "
              f"Type={result['legal_context']['clause_type']}, "
              f"Section={result['legal_context']['section_id']}, "
              f"Obligation={result['legal_context']['is_obligation']}")


if __name__ == "__main__":
    example_usage()

