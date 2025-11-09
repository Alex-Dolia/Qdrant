# 00_chunking - Document Splitting & Hierarchy Assignment

This module contains all chunking-related functionality for the RAG system.

## Purpose

Handles document splitting, chunking strategies, and hierarchy assignment for legal and structured documents.

## Modules

- **qdrant_chunker.py**: `LegalDocumentChunker` class with multiple chunking strategies:
  - Recursive chunking
  - Semantic chunking
  - Structural chunking (hierarchical)
  - Agentic chunking
  - Cluster chunking
  - Hierarchical chunking

- **legal_chunker_integration.py**: Integration layer for legal document chunking:
  - Document ingestion into Qdrant
  - Query functions for legal documents
  - File management (upload, delete, statistics)
  - Chunk exploration utilities

- **structural_chunker.py**: Standalone structural chunking implementation:
  - Three-level hierarchy (Sections, Subclauses, Semantic Units)
  - Full document preservation
  - Recursive fallback for unstructured documents
  - Qdrant integration functions

- **rag_pipeline.py**: Shared RAG pipeline for document ingestion:
  - Document loading from directory
  - Chunking and embedding generation
  - Qdrant storage with metadata
  - Vectorstore creation

## Usage

```python
from utils.00_chunking import (
    LegalDocumentChunker,
    ingest_legal_document,
    structural_chunk_document,
    ingest_documents_to_qdrant
)
```

## Dependencies

- `qdrant-client`
- `langchain` / `langchain-community`
- `sentence-transformers` (optional)
- `langchain-ollama` (optional)

