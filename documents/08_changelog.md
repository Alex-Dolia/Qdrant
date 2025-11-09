# Changelog

All notable changes to the RAG Query System project.

## [Current] - 2025

### Project Structure
- **Modular Architecture:** Reorganized codebase into `scripts/` directory with numbered subdirectories:
  - `00_chunking/` - Document splitting and hierarchy assignment
  - `01_data_ingestion_and_preprocessing/` - Cleaning, preprocessing, embeddings
  - `02_query_completion/` - Query enhancement with LLaMA 3.1
  - `03_retrieval/` - Retrieve relevant chunks from Qdrant
  - `04_reranking/` - Rerank retrieved results
  - `05_output_generation/` - Produce final RAG output
  - `06_output_evaluation/` - Assess output quality
  - `07_research_workflow/` - Research-specific workflows
  - `08_utilities/` - General utilities

### Added
- **Four-tab interface:**
  - ⚖️ Legal Documents Query - Advanced document querying with multiple search modes
  - 🔍 Simple Web Search - Quick web search functionality
  - 📚 Research Overview - Generate comprehensive research survey papers
  - 🧩 Chunk Exploration - Explore how documents are chunked and stored

- **Multiple chunking methods:** recursive, semantic, structural, agentic, cluster
- **Multiple search modes:** semantic, BM25, mixed (hybrid search)
- **Chunk Exploration:** Visual inspection of chunking strategies and metadata
- **Research Overview:** Academic paper search and structured report generation
- **Consolidated requirements:** Single `requirements.txt` file for all dependencies

### Changed
- **Directory structure:** Moved from `utils/` to `scripts/` with organized subdirectories
- **Import paths:** Updated all imports to reflect new modular structure
- **Documentation:** Updated all documentation to match current project state
- **Requirements:** Consolidated `requirements.txt` and `requirements_rag_eval.txt` into single file

### Removed
- `utils/` directory (replaced with `scripts/`)
- `requirements_rag_eval.txt` (merged into `requirements.txt`)
- `install_dependencies.bat` (unnecessary wrapper script)
- Refactoring utility scripts (no longer needed)
- Temporary documentation files

### Fixed
- Updated all import paths to use new `scripts/` structure
- Fixed path references in documentation
- Updated file references from old `utils/` paths to new `scripts/` paths

## Future Improvements

- [ ] Add document preview
- [ ] Add export functionality for query results
- [ ] Add batch document processing
- [ ] Improve error messages
- [ ] Add progress indicators
- [ ] Add document metadata editing
- [ ] Enhanced chunk visualization
- [ ] Performance optimizations

