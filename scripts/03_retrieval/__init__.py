"""
03_retrieval - Chunk Retrieval from Qdrant

This module handles:
- Semantic search in Qdrant
- BM25 keyword search
- Hybrid search (semantic + BM25)
"""

try:
    from .retrieval import RAGRetrievalEngine
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Retrieval module not available: {e}")

__all__ = [
    'RAGRetrievalEngine',
]

