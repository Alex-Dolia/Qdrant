"""
00_chunking - Document Splitting & Hierarchy Assignment

This module contains all chunking-related functionality:
- Structural chunking with hierarchy levels
- Legal document chunking
- Shared RAG pipeline for document ingestion
- Chunk metadata management
"""

# Import main chunking functions for easy access
try:
    from .qdrant_chunker import LegalDocumentChunker
    from .legal_chunker_integration import (
        ingest_legal_document,
        query_legal_documents,
        get_distinct_source_files,
        get_distinct_chunking_methods,
        get_available_chunking_methods,
        get_default_chunking_methods,
        delete_file_from_qdrant,
        delete_all_files_from_qdrant,
        get_file_statistics,
        get_chunks_for_exploration,
        get_max_chunk_number,
        EMBEDDING_MODELS,
        LEGAL_CHUNKER_AVAILABLE
    )
    from .structural_chunker import (
        structural_chunk_document,
        split_document,
        recursive_split,
        process_and_inject_document
    )
    from .rag_pipeline import (
        ingest_documents_to_qdrant,
        load_documents_from_directory
    )
except ImportError as e:
    # Handle import errors gracefully
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Some chunking modules not available: {e}")

__all__ = [
    'LegalDocumentChunker',
    'ingest_legal_document',
    'query_legal_documents',
    'structural_chunk_document',
    'ingest_documents_to_qdrant',
    'load_documents_from_directory',
]

