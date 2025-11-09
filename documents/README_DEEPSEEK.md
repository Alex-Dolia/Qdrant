# DeepSeek: Open-Source Web Search System

A modular, open-source web search system with optional semantic reranking using BM25 and embeddings.

## Features

- **Open-Source Web Search**: Uses DuckDuckGo or Google HTML scraping (no API keys required)
- **Semantic Reranking**: Combines BM25 keyword search with embedding-based similarity
- **Caching**: File-based cache for faster repeated queries
- **Modular Design**: Easy to extend with new search engines or reranking methods
- **Multiple Interfaces**: CLI, Python API, and programmatic usage

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Interface

```bash
# Basic search
python deepseek.py "BBC"

# Search with more results
python deepseek.py "Python web scraping" -n 20

# Disable reranking
python deepseek.py "query" --no-rerank

# Use Google instead of DuckDuckGo
python deepseek.py "query" -e google

# Output as JSON
python deepseek.py "query" --json

# Verbose logging
python deepseek.py "query" -v
```

### Python API

```python
from deepseek import DeepSeek

# Initialize DeepSeek
deepseek = DeepSeek(
    search_engine="duckduckgo",  # or "google"
    use_cache=True,
    use_reranking=True,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# Perform search
results = deepseek.search("BBC", max_results=10)

# Format and display results
formatted = deepseek.format_results(results)
print(formatted)

# Access raw results
for result in results:
    print(f"Title: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"Relevance Score: {result.get('relevance_score', 'N/A')}")
```

## Output Format

Results are returned in a structured format:

```
1. Title (Relevance: 0.85 | BM25: 0.72 | Embedding: 0.91)
   🔗 https://example.com/article
   Article snippet or summary...
   Source: example.com
```

If no results are found:
```
⚠️ No results found.
```

## Architecture

### Components

1. **Search Engines** (`WebSearchEngine`):
   - `DuckDuckGoSearch`: HTML scraping of DuckDuckGo
   - `GoogleSearchScraper`: HTML scraping of Google

2. **Reranker** (`Reranker`):
   - BM25 keyword scoring (40% weight)
   - Embedding-based semantic similarity (60% weight)
   - Uses SentenceTransformers for embeddings

3. **Cache** (`SearchCache`):
   - File-based caching with TTL
   - MD5-based cache keys

4. **Main System** (`DeepSeek`):
   - Orchestrates search, reranking, and caching
   - Provides unified API

## Configuration

### Search Engines

- **DuckDuckGo** (default): Privacy-focused, no API key required
- **Google**: More comprehensive results, may be rate-limited

### Embedding Models

Default: `sentence-transformers/all-MiniLM-L6-v2` (fast, lightweight)

Other options:
- `sentence-transformers/all-mpnet-base-v2` (more accurate, slower)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingual)

### Reranking

Reranking combines:
- **BM25 Score** (40%): Keyword-based relevance
- **Embedding Score** (60%): Semantic similarity

Final relevance score is a weighted average.

## Examples

See `deepseek_example.py` for complete usage examples.

### Example 1: Basic Search

```python
from deepseek import DeepSeek

deepseek = DeepSeek(use_reranking=False)
results = deepseek.search("BBC", max_results=5)
print(deepseek.format_results(results))
```

### Example 2: Search with Reranking

```python
deepseek = DeepSeek(use_reranking=True)
results = deepseek.search("Kernel ellipsoidal trimming", max_results=10)
print(deepseek.format_results(results))
```

### Example 3: Programmatic Access

```python
results = deepseek.search("Alex Dolia", max_results=3)

for result in results:
    print(f"Title: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"Relevance: {result.get('relevance_score', 'N/A')}")
```

## Extending DeepSeek

### Adding a New Search Engine

```python
from deepseek import WebSearchEngine

class MySearchEngine(WebSearchEngine):
    def __init__(self):
        super().__init__("MyEngine")
    
    def search(self, query: str, max_results: int = 10) -> List[Dict]:
        # Implement your search logic
        results = []
        # ... your code ...
        return results
```

### Custom Reranking

Modify the `Reranker` class or create a subclass with your own scoring logic.

## Limitations

- HTML scraping may be blocked by some sites
- Rate limiting may apply (use caching to mitigate)
- Results quality depends on search engine availability
- Embedding models require initial download (~100MB)

## Troubleshooting

### No Results Found

- Check internet connection
- Try a different search engine (`-e google`)
- Verify the query is not too specific
- Check logs with `-v` flag

### Import Errors

Install missing dependencies:
```bash
pip install requests beautifulsoup4 numpy rank-bm25 sentence-transformers
```

### Slow Performance

- Disable reranking with `--no-rerank` for faster searches
- Use caching (enabled by default)
- Use a smaller embedding model

## License

Open-source, free to use and modify.

## Contributing

Contributions welcome! Areas for improvement:
- Additional search engines
- Better HTML parsing
- Advanced reranking methods
- API endpoint support
- Multi-source aggregation

