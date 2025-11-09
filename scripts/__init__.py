"""
Scripts package for RAG Query System - Qdrant + Llama 3.1
"""

# Import from subdirectories
# Note: Use importlib for modules starting with numbers
import importlib

try:
    ingestion_module = importlib.import_module('scripts.01_data_ingestion_and_preprocessing.ingestion')
    DocumentIngestionPipeline = ingestion_module.DocumentIngestionPipeline
    
    retrieval_module = importlib.import_module('scripts.03_retrieval.retrieval')
    RAGRetrievalEngine = retrieval_module.RAGRetrievalEngine
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Some scripts modules not available: {e}")
    DocumentIngestionPipeline = None
    RAGRetrievalEngine = None

__all__ = [
    'DocumentIngestionPipeline',
    'RAGRetrievalEngine',
]

