# Streamlit App Documentation

Complete guide to using the RAG Query System Streamlit application.

## Overview

The application provides four main features across multiple tabs:
1. **Legal Documents Query** - Search through your uploaded legal documents
2. **Simple Web Search** - Quick web search functionality
3. **Research Overview** - Generate comprehensive research survey papers
4. **Chunk Exploration** - Explore how documents are chunked and stored

## Interface Overview

### Sidebar

**Connection Status:**
- Qdrant connection status indicator

**Document Management:**
- Upload documents (PDF, DOCX, TXT, MD)
- Select embedding model
- Choose chunking methods
- View processed files
- Delete files or reset database
- View file statistics

### Main Tabs

**Tab 1: ⚖️ Legal Documents Query**
- Query uploaded legal documents
- Multiple search modes: semantic, BM25, mixed
- View search results with citations
- See relevance scores and metadata
- Filter by chunking method

**Tab 2: 🔍 Simple Web Search**
- Quick web search using DuckDuckGo
- View search results with snippets
- No document upload required

**Tab 3: 📚 Research Overview**
- Generate comprehensive research survey papers
- Academic paper search integration
- Structured report generation with multiple sections
- Download reports as Markdown or HTML

**Tab 4: 🧩 Chunk Exploration**
- Explore how documents are chunked
- View chunk metadata
- Navigate through chunks by document
- See chunk hierarchy levels

## Uploading Documents

1. Go to sidebar "📚 Document Management"
2. Select embedding model (default: Ollama Llama 3.1)
3. Choose chunking methods (multiple can be selected)
4. Click "Upload documents"
5. Select files (PDF, DOCX, TXT, or MD)
6. Click "Process" next to each file
7. Wait for processing to complete
8. Files appear in "Processed Files" list

## Querying Legal Documents

1. Go to "⚖️ Legal Documents Query" tab
2. Select search mode: semantic, BM25, or mixed
3. Choose chunking method filter (optional)
4. Enter your question
5. Click "Search Legal Documents"
6. View results with:
   - Chunk text
   - Relevance scores
   - Source file names
   - Chunking method
   - Metadata (page numbers, hierarchy levels, entities)

## Simple Web Search

1. Go to "🔍 Simple Web Search" tab
2. Enter your search query
3. Click "Search"
4. View web search results with snippets

## Generating Research Overview

1. Go to "📚 Research Overview" tab
2. Enter your research topic
3. Configure options:
   - Year range for papers
   - Max web results
   - Use web research (enabled by default)
4. Click "Generate Research Overview"
5. Wait for generation (may take several minutes)
6. Review the generated paper
7. Download as Markdown or HTML

## Chunk Exploration

1. Go to "🧩 Chunk Exploration" tab
2. Select a document from the list
3. Choose chunking method
4. Navigate through chunks
5. View chunk text and metadata

## Tips

- **Better queries:** Be specific and clear
- **Multiple documents:** Upload related documents for better context
- **Processing time:** Large documents take longer to process
- **Results quality:** More processed documents = better results

## Troubleshooting

See `documents/00_install.md` for installation issues.
See `documents/05_running_the_app.md` for running issues.

