"""
DeepSeek Usage Examples
Demonstrates how to use the DeepSeek search system programmatically.
"""

from deepseek import DeepSeek

# Example 1: Basic search without reranking
print("=" * 60)
print("Example 1: Basic Search (No Reranking)")
print("=" * 60)

deepseek_basic = DeepSeek(
    search_engine="duckduckgo",
    use_cache=True,
    use_reranking=False
)

results = deepseek_basic.search("BBC", max_results=5)
formatted = deepseek_basic.format_results(results)
print(formatted)

# Example 2: Search with semantic reranking
print("\n" + "=" * 60)
print("Example 2: Search with Semantic Reranking")
print("=" * 60)

deepseek_rerank = DeepSeek(
    search_engine="duckduckgo",
    use_cache=True,
    use_reranking=True,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

results = deepseek_rerank.search("Kernel ellipsoidal trimming", max_results=5)
formatted = deepseek_rerank.format_results(results)
print(formatted)

# Example 3: Access raw results programmatically
print("\n" + "=" * 60)
print("Example 3: Access Raw Results")
print("=" * 60)

results = deepseek_rerank.search("Alex Dolia", max_results=3)

if results:
    print(f"Found {len(results)} results:\n")
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Title: {result.get('title', 'N/A')}")
        print(f"  URL: {result.get('url', 'N/A')}")
        print(f"  Snippet: {result.get('snippet', 'N/A')[:100]}...")
        if result.get('relevance_score'):
            print(f"  Relevance Score: {result['relevance_score']:.3f}")
        print()
else:
    print("⚠️ No results found.")

# Example 4: Using Google search engine
print("\n" + "=" * 60)
print("Example 4: Using Google Search Engine")
print("=" * 60)

deepseek_google = DeepSeek(
    search_engine="google",
    use_cache=True,
    use_reranking=True
)

results = deepseek_google.search("Python web scraping", max_results=3)
formatted = deepseek_google.format_results(results)
print(formatted)

