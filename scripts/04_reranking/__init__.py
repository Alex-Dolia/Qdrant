"""
04_reranking - Result Reranking

This module handles:
- Legal document reranking
- Relevance scoring
- Result optimization
"""

try:
    from .legal_reranker import LegalReranker, rerank_results
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Reranking modules not available: {e}")

__all__ = [
    'LegalReranker',
    'rerank_results',
]

