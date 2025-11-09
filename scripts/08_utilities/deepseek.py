"""
DeepSeek: Open-Source Web Search System
A modular web search system using open-source tools with optional semantic reranking.
"""

import os
import re
import json
import logging
import hashlib
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import time

# Core dependencies
try:
    import requests
    from bs4 import BeautifulSoup
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests and BeautifulSoup not available. Install with: pip install requests beautifulsoup4")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available. Install with: pip install numpy")

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("Warning: rank-bm25 not available. Install with: pip install rank-bm25")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

logger = logging.getLogger(__name__)


class SearchCache:
    """Simple file-based cache for search results."""
    
    def __init__(self, cache_dir: str = ".deepseek_cache", ttl_hours: int = 24):
        """
        Initialize search cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live for cache entries in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
    
    def _get_cache_key(self, query: str, engine: str) -> str:
        """Generate cache key from query and engine."""
        key_string = f"{query}_{engine}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, query: str, engine: str) -> Optional[List[Dict]]:
        """Retrieve cached results if available and not expired."""
        cache_key = self._get_cache_key(query, engine)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if cache is expired
            cache_time = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - cache_time > self.ttl:
                cache_file.unlink()  # Delete expired cache
                return None
            
            return data['results']
        except Exception as e:
            logger.warning(f"Error reading cache: {e}")
            return None
    
    def set(self, query: str, engine: str, results: List[Dict]):
        """Store results in cache."""
        cache_key = self._get_cache_key(query, engine)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'engine': engine,
                'results': results
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Error writing cache: {e}")


class WebSearchEngine:
    """Base class for web search engines."""
    
    def __init__(self, name: str):
        self.name = name
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def search(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Perform web search.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of dictionaries with keys: title, url, snippet, source
        """
        raise NotImplementedError("Subclasses must implement search method")


class DuckDuckGoSearch(WebSearchEngine):
    """DuckDuckGo search using HTML scraping."""
    
    def __init__(self):
        super().__init__("DuckDuckGo")
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests and BeautifulSoup required for DuckDuckGo search")
    
    def search(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search DuckDuckGo using HTML scraping."""
        results = []
        
        try:
            # Use DuckDuckGo HTML search
            search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            response = requests.get(search_url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"DuckDuckGo search returned status {response.status_code}")
                return results
            
            soup = BeautifulSoup(response.text, 'html.parser')
            result_divs = soup.find_all('div', class_='result')
            
            for div in result_divs[:max_results]:
                try:
                    title_elem = div.find('a', class_='result__a')
                    snippet_elem = div.find('a', class_='result__snippet')
                    
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        url = title_elem.get('href', '')
                        
                        # Clean URL (DuckDuckGo uses redirect URLs)
                        if url.startswith('//'):
                            url = 'https:' + url
                        elif url.startswith('/l/?kh='):
                            # Extract actual URL from DuckDuckGo redirect
                            import urllib.parse
                            parsed = urllib.parse.urlparse(url)
                            query_params = urllib.parse.parse_qs(parsed.query)
                            if 'uddg' in query_params:
                                url = query_params['uddg'][0]
                        
                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                        
                        # Extract domain
                        domain = ""
                        if url:
                            try:
                                from urllib.parse import urlparse
                                parsed = urlparse(url)
                                domain = parsed.netloc.replace("www.", "")
                            except:
                                pass
                        
                        if title and url:
                            results.append({
                                "title": title,
                                "url": url,
                                "snippet": snippet[:500] if snippet else "",
                                "source": domain,
                                "domain": domain
                            })
                except Exception as e:
                    logger.debug(f"Error parsing DuckDuckGo result: {e}")
                    continue
            
            logger.info(f"DuckDuckGo search returned {len(results)} results")
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
        
        return results


class GoogleSearchScraper(WebSearchEngine):
    """Google search using HTML scraping (fallback method)."""
    
    def __init__(self):
        super().__init__("Google")
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests and BeautifulSoup required for Google search")
    
    def search(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search Google using HTML scraping."""
        results = []
        
        try:
            # Use Google search
            search_url = f"https://www.google.com/search?q={requests.utils.quote(query)}&num={max_results}"
            response = requests.get(search_url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Google search returned status {response.status_code}")
                return results
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find search result containers
            result_divs = soup.find_all('div', class_='g')
            
            for div in result_divs[:max_results]:
                try:
                    # Extract title
                    title_elem = div.find('h3')
                    title = title_elem.get_text(strip=True) if title_elem else ""
                    
                    # Extract URL
                    link_elem = div.find('a')
                    url = link_elem.get('href', '') if link_elem else ""
                    
                    # Clean URL (remove Google redirect)
                    if url.startswith('/url?q='):
                        url = url.split('/url?q=')[1].split('&')[0]
                    
                    # Extract snippet
                    snippet_elem = div.find('span', class_='aCOpRe') or div.find('div', class_='VwiC3b')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    # Extract domain
                    domain = ""
                    if url:
                        try:
                            from urllib.parse import urlparse
                            parsed = urlparse(url)
                            domain = parsed.netloc.replace("www.", "")
                        except:
                            pass
                    
                    if title and url:
                        results.append({
                            "title": title,
                            "url": url,
                            "snippet": snippet[:500] if snippet else "",
                            "source": domain,
                            "domain": domain
                        })
                except Exception as e:
                    logger.debug(f"Error parsing Google result: {e}")
                    continue
            
            logger.info(f"Google search returned {len(results)} results")
            
        except Exception as e:
            logger.error(f"Google search error: {e}")
        
        return results


class Reranker:
    """Semantic reranker using BM25, embeddings, and RRF."""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", rrf_k: int = 60):
        """
        Initialize reranker.
        
        Args:
            embedding_model: SentenceTransformer model name
            rrf_k: Reciprocal Rank Fusion parameter (default: 60)
        """
        self.bm25_available = BM25_AVAILABLE
        self.embeddings = None
        self.rrf_k = rrf_k
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embeddings = SentenceTransformer(embedding_model)
                logger.info(f"Initialized embeddings model: {embedding_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize embeddings: {e}")
        else:
            logger.warning("SentenceTransformers not available, embedding reranking disabled")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def rerank(self, query: str, results: List[Dict], top_k: int = 10, 
               rank_mode: str = "rrf") -> List[Dict]:
        """
        Rerank search results using BM25, embeddings, or RRF.
        
        Args:
            query: Original search query
            results: List of search results to rerank
            top_k: Number of top results to return
            rank_mode: Ranking mode - "bm25", "embedding", or "rrf" (default: "rrf")
            
        Returns:
            Reranked list of results with scores
        """
        if not results:
            return results
        
        try:
            # Prepare texts for scoring
            result_texts = []
            for result in results:
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                combined_text = f"{title}. {snippet}".strip()
                if not combined_text:
                    combined_text = result.get("url", "")
                result_texts.append(combined_text)
            
            bm25_scores = []
            embedding_scores = []
            
            # BM25 scoring
            if self.bm25_available and BM25Okapi:
                try:
                    query_tokens = self._tokenize(query)
                    tokenized_docs = [self._tokenize(text) for text in result_texts]
                    bm25 = BM25Okapi(tokenized_docs)
                    bm25_scores = bm25.get_scores(query_tokens)
                    
                    # Normalize BM25 scores to [0, 1] for display
                    if len(bm25_scores) > 0:
                        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
                        bm25_scores_normalized = [score / max_bm25 for score in bm25_scores]
                    else:
                        bm25_scores_normalized = [0.0] * len(results)
                except Exception as e:
                    logger.warning(f"BM25 scoring failed: {e}")
                    bm25_scores = [0.0] * len(results)
                    bm25_scores_normalized = [0.0] * len(results)
            else:
                bm25_scores = [0.0] * len(results)
                bm25_scores_normalized = [0.0] * len(results)
            
            # Embedding-based similarity (using SentenceTransformers util for proper cosine similarity)
            if self.embeddings and SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    # Use SentenceTransformers encode method with convert_to_tensor
                    query_embedding = self.embeddings.encode(query, convert_to_tensor=True)
                    result_embeddings = self.embeddings.encode(result_texts, convert_to_tensor=True)
                    
                    # Use SentenceTransformers util for proper cosine similarity
                    from sentence_transformers import util as st_util
                    cosine_scores = st_util.cos_sim(query_embedding, result_embeddings)
                    
                    # Convert to numpy and flatten
                    if hasattr(cosine_scores, 'cpu'):
                        embedding_scores = cosine_scores.cpu().numpy().flatten()
                    else:
                        embedding_scores = np.array(cosine_scores).flatten()
                    
                    # Normalize to [0, 1] for display (cosine similarity is [-1, 1])
                    embedding_scores_normalized = [(score + 1) / 2 for score in embedding_scores]
                    
                    logger.debug(f"Embedding scores computed: min={min(embedding_scores):.3f}, max={max(embedding_scores):.3f}, mean={np.mean(embedding_scores):.3f}")
                except Exception as e:
                    logger.warning(f"Embedding scoring failed: {e}")
                    embedding_scores = np.array([0.0] * len(results))
                    embedding_scores_normalized = [0.0] * len(results)
            else:
                embedding_scores = np.array([0.0] * len(results))
                embedding_scores_normalized = [0.0] * len(results)
            
            # Apply ranking based on mode
            if rank_mode == "bm25":
                # Rank by BM25 only
                sorted_indices = np.argsort(bm25_scores)[::-1] if NUMPY_AVAILABLE else sorted(range(len(results)), key=lambda i: bm25_scores[i], reverse=True)
                final_scores = bm25_scores
                
            elif rank_mode == "embedding":
                # Rank by embedding similarity only
                sorted_indices = np.argsort(embedding_scores)[::-1] if NUMPY_AVAILABLE else sorted(range(len(results)), key=lambda i: embedding_scores[i], reverse=True)
                final_scores = embedding_scores
                
            elif rank_mode == "rrf":
                # Reciprocal Rank Fusion
                fused_scores = {}
                
                # Get BM25 ranking
                if self.bm25_available:
                    bm25_ranking = sorted(range(len(results)), key=lambda i: bm25_scores[i], reverse=True)
                    for rank, idx in enumerate(bm25_ranking):
                        doc_id = results[idx].get('url', str(idx))
                        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (self.rrf_k + rank + 1)
                
                # Get embedding ranking
                if self.embeddings:
                    emb_ranking = sorted(range(len(results)), key=lambda i: embedding_scores[i], reverse=True)
                    for rank, idx in enumerate(emb_ranking):
                        doc_id = results[idx].get('url', str(idx))
                        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (self.rrf_k + rank + 1)
                
                # Apply fused scores
                final_scores = [fused_scores.get(result.get('url', str(i)), 0.0) for i, result in enumerate(results)]
                sorted_indices = np.argsort(final_scores)[::-1] if NUMPY_AVAILABLE else sorted(range(len(results)), key=lambda i: final_scores[i], reverse=True)
                
            else:
                # Default: weighted combination (fallback)
                logger.warning(f"Unknown rank_mode '{rank_mode}', using weighted combination")
                hybrid_scores = []
                for i in range(len(results)):
                    hybrid_score = (0.4 * bm25_scores_normalized[i]) + (0.6 * embedding_scores_normalized[i])
                    hybrid_scores.append(hybrid_score)
                final_scores = hybrid_scores
                sorted_indices = np.argsort(hybrid_scores)[::-1] if NUMPY_AVAILABLE else sorted(range(len(results)), key=lambda i: hybrid_scores[i], reverse=True)
            
            # Create reranked results
            reranked_results = []
            for idx in sorted_indices:
                result = results[idx].copy()
                
                # Store all scores for display
                result["bm25_score"] = float(bm25_scores_normalized[idx] if self.bm25_available else 0.0)
                result["embedding_score"] = float(embedding_scores_normalized[idx] if self.embeddings else 0.0)
                
                # Set relevance score based on ranking mode
                if rank_mode == "bm25":
                    result["relevance_score"] = float(bm25_scores_normalized[idx])
                elif rank_mode == "embedding":
                    result["relevance_score"] = float(embedding_scores_normalized[idx])
                elif rank_mode == "rrf":
                    result["relevance_score"] = float(final_scores[idx])
                    result["rrf_score"] = float(final_scores[idx])
                else:
                    result["relevance_score"] = float(final_scores[idx])
                
                reranked_results.append(result)
                if len(reranked_results) >= top_k:
                    break
            
            logger.info(f"Reranked {len(results)} results using {rank_mode}, returning top {len(reranked_results)}")
            return reranked_results
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}, returning original results")
            import traceback
            logger.debug(traceback.format_exc())
            return results[:top_k]


class DeepSeek:
    """Main DeepSeek search system."""
    
    def __init__(self, 
                 search_engine: str = "duckduckgo",
                 use_cache: bool = True,
                 use_reranking: bool = True,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize DeepSeek search system.
        
        Args:
            search_engine: Search engine to use ("duckduckgo" or "google")
            use_cache: Enable caching for repeated queries
            use_reranking: Enable semantic reranking
            embedding_model: SentenceTransformer model for reranking
        """
        self.search_engine_name = search_engine.lower()
        self.use_cache = use_cache
        self.use_reranking = use_reranking
        
        # Initialize search engine
        if self.search_engine_name == "duckduckgo":
            self.search_engine = DuckDuckGoSearch()
        elif self.search_engine_name == "google":
            self.search_engine = GoogleSearchScraper()
        else:
            raise ValueError(f"Unknown search engine: {search_engine}")
        
        # Initialize cache
        self.cache = SearchCache() if use_cache else None
        
        # Initialize reranker
        self.reranker = Reranker(embedding_model, rrf_k=60) if use_reranking else None
    
    def search(self, query: str, max_results: int = 10, rerank: Optional[bool] = None, 
               rank_mode: str = "rrf") -> List[Dict]:
        """
        Perform web search with optional reranking.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return (after reranking)
            rerank: Override default reranking setting (None uses instance default)
            rank_mode: Ranking mode - "bm25", "embedding", or "rrf" (default: "rrf")
            
        Returns:
            List of search results with optional scores
        """
        if not query or not query.strip():
            return []
        
        query = query.strip()
        should_rerank = rerank if rerank is not None else self.use_reranking
        
        # Check cache (include rank_mode in cache key)
        cache_key = f"{query}_{self.search_engine_name}_{rank_mode}"
        if self.cache:
            cached_results = self.cache.get(cache_key, self.search_engine_name)
            if cached_results:
                logger.info(f"Retrieved {len(cached_results)} results from cache")
                return cached_results[:max_results]
        
        # Perform search
        logger.info(f"Searching {self.search_engine_name} for: '{query}'")
        results = self.search_engine.search(query, max_results=max_results * 3 if should_rerank else max_results)
        
        if not results:
            logger.warning(f"No results found for query: '{query}'")
            return []
        
        # Apply reranking if enabled
        if should_rerank and self.reranker:
            logger.info(f"Reranking {len(results)} results using {rank_mode}...")
            results = self.reranker.rerank(query, results, top_k=max_results, rank_mode=rank_mode)
        
        # Limit to max_results
        results = results[:max_results]
        
        # Cache results
        if self.cache:
            self.cache.set(cache_key, self.search_engine_name, results)
        
        return results
    
    def format_results(self, results: List[Dict], rank_mode: str = "rrf") -> str:
        """
        Format search results as a readable string.
        
        Args:
            results: List of search result dictionaries
            rank_mode: Ranking mode used (for display)
            
        Returns:
            Formatted string representation
        """
        if not results:
            return "⚠️ No results found."
        
        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            snippet = result.get("snippet", "")
            source = result.get("source", result.get("domain", ""))
            
            # Build score info - show BM25 and Embedding scores
            bm25_score = result.get("bm25_score", 0.0)
            embedding_score = result.get("embedding_score", 0.0)
            relevance_score = result.get("relevance_score", 0.0)
            
            # Format scores based on ranking mode
            if rank_mode == "bm25":
                score_str = f" (Relevance: {relevance_score:.2f} | BM25: {bm25_score:.2f} | Embedding: {embedding_score:.2f})"
            elif rank_mode == "embedding":
                score_str = f" (Relevance: {relevance_score:.2f} | BM25: {bm25_score:.2f} | Embedding: {embedding_score:.2f})"
            elif rank_mode == "rrf":
                rrf_score = result.get("rrf_score", relevance_score)
                score_str = f" (Relevance: {rrf_score:.2f} | BM25: {bm25_score:.2f} | Embedding: {embedding_score:.2f})"
            else:
                score_str = f" (Relevance: {relevance_score:.2f} | BM25: {bm25_score:.2f} | Embedding: {embedding_score:.2f})"
            
            formatted.append(f"{i}. {title}{score_str}")
            if url:
                formatted.append(f"🔗 {url}")
            if snippet:
                formatted.append(f"{snippet}")
            if source:
                formatted.append(f"Source: {source}")
            formatted.append("")  # Empty line between results
        
        return "\n".join(formatted)


def main():
    """CLI interface for DeepSeek."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepSeek: Open-Source Web Search System")
    parser.add_argument("query", help="Search query")
    parser.add_argument("-n", "--max-results", type=int, default=10, help="Maximum number of results (default: 10)")
    parser.add_argument("-e", "--engine", choices=["duckduckgo", "google"], default="duckduckgo", 
                       help="Search engine to use (default: duckduckgo)")
    parser.add_argument("--no-rerank", action="store_true", help="Disable semantic reranking")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Embedding model for reranking (default: sentence-transformers/all-MiniLM-L6-v2)")
    parser.add_argument("--rank-mode", choices=["bm25", "embedding", "rrf"], default="rrf",
                       help="Ranking mode: bm25 (keyword only), embedding (semantic only), rrf (fusion)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize DeepSeek
    try:
        deepseek = DeepSeek(
            search_engine=args.engine,
            use_cache=not args.no_cache,
            use_reranking=not args.no_rerank,
            embedding_model=args.embedding_model
        )
    except Exception as e:
        print(f"❌ Error initializing DeepSeek: {e}")
        return 1
    
    # Perform search
    try:
        results = deepseek.search(
            args.query, 
            max_results=args.max_results,
            rerank=not args.no_rerank,
            rank_mode=args.rank_mode
        )
        
        if args.json:
            # Output as JSON
            print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
        else:
            # Output formatted results
            formatted = deepseek.format_results(results, rank_mode=args.rank_mode)
            print(formatted)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error performing search: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

