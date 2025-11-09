# 03_retrieval - Chunk Retrieval from Qdrant

This module handles Qdrant-based retrieval functionality for the RAG system.

## Purpose

Provides semantic search, keyword search, and hybrid search capabilities for retrieving chunks from Qdrant.

## Modules

- **retrieval.py**: `RAGRetrievalEngine` class:
  - Semantic search (embedding-based)
  - BM25 keyword search
  - Hybrid search (semantic + BM25 with RRF)
  - Answer generation with LLM
  - Result deduplication and ranking

## Usage

```python
from scripts.03_retrieval import RAGRetrievalEngine
```

## Note

Web search, academic search, and caching functionality have been moved to `scripts/07_research_workflow/`.

## Dependencies

- `qdrant-client`
- `langchain-ollama`
- `requests` (for web search)
- `rank-bm25` (for BM25 search)

