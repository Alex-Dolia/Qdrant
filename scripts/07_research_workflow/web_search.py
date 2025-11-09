"""
Web Search Module
Provides web search capabilities for research augmentation with hybrid reranking.
"""

import os
import logging
from typing import List, Dict, Optional
import streamlit as st
import numpy as np

logger = logging.getLogger(__name__)

# Try to import web search libraries
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    try:
        import requests
        REQUESTS_AVAILABLE = True
    except ImportError:
        REQUESTS_AVAILABLE = False

# Try to import BM25 for keyword search
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    BM25Okapi = None


class WebSearch:
    """Web search interface for research augmentation with hybrid reranking."""
    
    def __init__(self, provider: str = "duckduckgo", use_cache: bool = True, 
                 use_reranking: bool = True, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize web search provider with optional hybrid reranking.
        
        Args:
            provider: "duckduckgo" (default) or "serpapi"
            use_cache: Enable caching for faster repeated searches
            use_reranking: Enable hybrid BM25 + embedding reranking
            embedding_model: Model for embeddings (e.g., "ollama/llama3.1:latest", "sentence-transformers/all-MiniLM-L6-v2")
        """
        self.provider = provider
        self.ddgs = None
        self.use_cache = use_cache
        self.use_reranking = use_reranking
        self.embedding_model = embedding_model
        self.embeddings = None
        
        # Initialize cache if enabled
        if use_cache:
            try:
                # Use importlib for modules starting with numbers
                import importlib
                web_cache_module = importlib.import_module('scripts.07_research_workflow.web_cache')
                WebSearchCache = web_cache_module.WebSearchCache
                self.cache = WebSearchCache()
            except Exception as e:
                logger.warning(f"Failed to initialize cache: {e}")
                self.cache = None
        else:
            self.cache = None
        
        if provider == "duckduckgo" and DDGS_AVAILABLE:
            try:
                # Initialize DuckDuckGo (timeout handled in search method)
                self.ddgs = DDGS()
                logger.info("Initialized DuckDuckGo search successfully")
                # Test search to verify it works
                try:
                    test_results = list(self.ddgs.text("test", max_results=1))
                    logger.info(f"DuckDuckGo test search successful: {len(test_results)} result(s)")
                except Exception as test_error:
                    logger.warning(f"DuckDuckGo test search failed: {test_error}")
            except Exception as e:
                logger.error(f"Failed to initialize DuckDuckGo: {e}", exc_info=True)
                self.ddgs = None
        
        elif provider == "serpapi":
            self.api_key = os.getenv("SERPAPI_API_KEY", "")
            if not self.api_key:
                logger.warning("SERPAPI_API_KEY not set, web search will be limited")
        
        # Initialize embeddings for reranking if enabled
        if self.use_reranking:
            self._initialize_embeddings()
            if not self.embeddings:
                logger.warning("Reranking disabled: embeddings not available")
                self.use_reranking = False
    
    def _initialize_embeddings(self):
        """Initialize embeddings model for reranking."""
        try:
            if self.embedding_model.startswith("ollama/"):
                from langchain_ollama import OllamaEmbeddings
                model_name = self.embedding_model.replace("ollama/", "")
                self.embeddings = OllamaEmbeddings(model=model_name)
                logger.info(f"Initialized Ollama embeddings: {model_name}")
            elif self.embedding_model.startswith("text-embedding-"):
                from langchain_openai import OpenAIEmbeddings
                self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
                logger.info(f"Initialized OpenAI embeddings: {self.embedding_model}")
            elif "sentence-transformers" in self.embedding_model or "/" in self.embedding_model:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
                logger.info(f"Initialized HuggingFace embeddings: {self.embedding_model}")
            else:
                # Default to sentence-transformers
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                logger.info("Initialized default HuggingFace embeddings")
        except ImportError as e:
            logger.warning(f"Failed to initialize embeddings for reranking: {e}")
            self.embeddings = None
            self.use_reranking = False
        except Exception as e:
            logger.warning(f"Error initializing embeddings: {e}")
            self.embeddings = None
            self.use_reranking = False
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        import re
        # Convert to lowercase and split on non-word characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _hybrid_rerank_results(self, query: str, results: List[Dict[str, str]], top_k: int = 10) -> List[Dict[str, str]]:
        """
        Rerank search results using hybrid BM25 + embedding similarity.
        
        Args:
            query: Original search query
            results: List of search results to rerank
            top_k: Number of top results to return
            
        Returns:
            Reranked list of results with relevance scores
        """
        if not self.use_reranking or not results:
            return results[:top_k]
        
        try:
            # Prepare texts for scoring
            result_texts = []
            for result in results:
                # Combine title and snippet for better context
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                combined_text = f"{title}. {snippet}".strip()
                if not combined_text:
                    combined_text = result.get("url", "")
                result_texts.append(combined_text)
            
            # Initialize scores
            bm25_scores = []
            embedding_scores = []
            
            # 1. BM25 keyword scoring
            if BM25_AVAILABLE and BM25Okapi:
                try:
                    # Tokenize query and documents
                    query_tokens = self._tokenize(query)
                    tokenized_docs = [self._tokenize(text) for text in result_texts]
                    
                    # Build BM25 index
                    bm25 = BM25Okapi(tokenized_docs)
                    
                    # Get BM25 scores
                    bm25_scores = bm25.get_scores(query_tokens)
                    
                    # Normalize BM25 scores to [0, 1]
                    if len(bm25_scores) > 0:
                        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
                        bm25_scores = [score / max_bm25 for score in bm25_scores]
                    
                    logger.debug(f"BM25 scores computed: {len(bm25_scores)} scores")
                except Exception as e:
                    logger.warning(f"BM25 scoring failed: {e}")
                    bm25_scores = [0.0] * len(results)
            else:
                bm25_scores = [0.0] * len(results)
            
            # 2. Embedding-based similarity scoring
            if self.embeddings:
                try:
                    # Generate embeddings
                    query_embedding = self.embeddings.embed_query(query)
                    result_embeddings = self.embeddings.embed_documents(result_texts)
                    
                    # Calculate cosine similarity
                    query_norm = np.linalg.norm(query_embedding)
                    
                    for result_emb in result_embeddings:
                        if query_norm > 0:
                            similarity = np.dot(query_embedding, result_emb) / (query_norm * np.linalg.norm(result_emb))
                        else:
                            similarity = 0.0
                        # Normalize to [0, 1] (cosine similarity is already in [-1, 1])
                        embedding_scores.append((similarity + 1) / 2)
                    
                    logger.debug(f"Embedding scores computed: {len(embedding_scores)} scores")
                except Exception as e:
                    logger.warning(f"Embedding scoring failed: {e}")
                    embedding_scores = [0.0] * len(results)
            else:
                embedding_scores = [0.0] * len(results)
            
            # 3. Combine scores (weighted average: 40% BM25, 60% embeddings)
            # BM25 is better for exact keyword matches, embeddings for semantic similarity
            hybrid_scores = []
            for i in range(len(results)):
                bm25_weight = 0.4
                embedding_weight = 0.6
                hybrid_score = (bm25_weight * bm25_scores[i]) + (embedding_weight * embedding_scores[i])
                hybrid_scores.append(hybrid_score)
            
            # 4. Sort by hybrid score (descending)
            sorted_indices = np.argsort(hybrid_scores)[::-1]
            
            # 5. Create reranked results with scores
            reranked_results = []
            
            for idx in sorted_indices:
                result = results[idx].copy()
                result["relevance_score"] = float(hybrid_scores[idx])
                result["bm25_score"] = float(bm25_scores[idx])
                result["embedding_score"] = float(embedding_scores[idx])
                
                # Add source type based on domain
                domain = result.get("domain", "")
                if domain:
                    if "linkedin.com" in domain:
                        result["source"] = "LinkedIn"
                    elif "researchgate.net" in domain or "sciencedirect.com" in domain or "arxiv.org" in domain:
                        result["source"] = "Academic"
                    elif any(news in domain for news in ["bbc.com", "cnn.com", "reuters.com", "nytimes.com"]):
                        result["source"] = "News"
                    elif domain.endswith(".pdf") or "arxiv.org" in domain:
                        result["source"] = "PDF/Paper"
                    else:
                        result["source"] = "Web"
                
                reranked_results.append(result)
                
                # Stop when we have enough results
                if len(reranked_results) >= top_k:
                    break
            
            logger.info(f"Hybrid reranked {len(results)} results, returning top {len(reranked_results)}")
            return reranked_results
            
        except Exception as e:
            logger.warning(f"Hybrid reranking failed: {e}, returning original results")
            return results[:top_k]
    
    def search(self, query: str, max_results: int = 5, initial_search_count: int = 50) -> List[Dict[str, str]]:
        """
        Perform web search with optional hybrid reranking and return results.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return after reranking
            initial_search_count: Number of results to retrieve before reranking (for hybrid search)
            
        Returns:
            List of dictionaries with keys: title, url, snippet, source, relevance_score
        """
        # Check cache first
        cache_key = f"{query}_{max_results}_{self.use_reranking}"
        if self.cache:
            cached_results = self.cache.get(cache_key)
            if cached_results:
                # Return cached results (limit to max_results)
                return cached_results[:max_results]
        
        results = []
        
        # Determine how many results to fetch initially
        # If reranking is enabled, fetch more results to rerank
        fetch_count = initial_search_count if self.use_reranking else max_results
        
        try:
            if self.provider == "duckduckgo" and self.ddgs:
                logger.info(f"Searching web for: '{query}' (fetching {fetch_count} results, returning top {max_results})")
                
                # Add retry logic with timeout handling
                max_retries = 3
                retry_delay = 2  # seconds
                search_results = []
                
                for attempt in range(max_retries):
                    try:
                        logger.info(f"Search attempt {attempt + 1}/{max_retries} for query: '{query}'")
                        
                        # Try different DuckDuckGo search methods
                        # Method 1: text() - standard text search
                        try:
                            search_results = list(self.ddgs.text(query, max_results=fetch_count))
                            logger.info(f"text() method returned {len(search_results)} raw results")
                        except Exception as text_error:
                            logger.warning(f"text() method failed: {text_error}")
                            search_results = []
                        
                        # If text() didn't work, try news() as fallback
                        if not search_results:
                            try:
                                logger.info("Trying news() method as fallback...")
                                news_results = list(self.ddgs.news(query, max_results=fetch_count))
                                # Convert news format to standard format
                                search_results = [
                                    {
                                        "title": r.get("title", ""),
                                        "href": r.get("url", ""),
                                        "body": r.get("body", r.get("snippet", ""))
                                    }
                                    for r in news_results
                                ]
                                logger.info(f"news() method returned {len(search_results)} results")
                            except Exception as news_error:
                                logger.warning(f"news() method also failed: {news_error}")
                        
                        # If still no results, try instant answer
                        if not search_results:
                            try:
                                logger.info("Trying instant answer as fallback...")
                                instant = self.ddgs.instant(query)
                                if instant:
                                    # Convert instant answer to result format
                                    search_results = [{
                                        "title": instant.get("heading", query),
                                        "href": instant.get("url", ""),
                                        "body": instant.get("abstract", instant.get("answer", ""))
                                    }]
                                    logger.info("instant() method returned 1 result")
                            except Exception as instant_error:
                                logger.warning(f"instant() method failed: {instant_error}")
                        
                        if search_results:
                            logger.info(f"Successfully retrieved {len(search_results)} results after attempt {attempt + 1}")
                            break  # Success, exit retry loop
                        else:
                            logger.warning(f"Search attempt {attempt + 1} returned empty results")
                            if attempt < max_retries - 1:
                                import time
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                            
                    except Exception as search_error:
                        error_str = str(search_error).lower()
                        logger.warning(f"Search error on attempt {attempt + 1}: {search_error}")
                        
                        # Check if it's a timeout error
                        if "timeout" in error_str or "timed out" in error_str or "bing" in error_str or "connection" in error_str:
                            if attempt < max_retries - 1:
                                logger.info(f"Timeout/connection error detected, retrying in {retry_delay}s...")
                                import time
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                                continue
                            else:
                                logger.error(f"Search failed after {max_retries} attempts due to timeout/connection")
                                search_results = []
                                break
                        else:
                            # Non-timeout error - log and try again
                            logger.error(f"Search error (non-timeout): {search_error}")
                            if attempt < max_retries - 1:
                                import time
                                time.sleep(retry_delay)
                                retry_delay *= 2
                                continue
                            search_results = []
                            break
                
                # Process and validate results
                if not search_results:
                    logger.warning(f"No search results available after {max_retries} attempts for query: '{query}'")
                    logger.warning(f"DDGS status: available={DDGS_AVAILABLE}, initialized={self.ddgs is not None}")
                    if self.ddgs:
                        logger.warning("DuckDuckGo client exists but returned no results - this may indicate API issues or rate limiting")
                else:
                    logger.info(f"Processing {len(search_results)} raw results...")
                    logger.debug(f"First result sample keys: {list(search_results[0].keys()) if search_results else 'N/A'}")
                
                for idx, result in enumerate(search_results):
                    try:
                        # Extract fields - DuckDuckGo returns different formats
                        title = result.get("title") or result.get("heading") or result.get("name") or ""
                        url = result.get("href") or result.get("url") or result.get("link") or ""
                        snippet = result.get("body") or result.get("snippet") or result.get("abstract") or result.get("description") or ""
                        
                        # Validate that we have at least a title or URL
                        if not title and not url:
                            logger.debug(f"Skipping result {idx}: no title or URL")
                            continue
                        
                        # Clean up snippet
                        if snippet:
                            snippet = snippet.strip()
                            # Limit snippet length
                            if len(snippet) > 500:
                                snippet = snippet[:500] + "..."
                        
                        # Try to extract domain for publication info
                        domain = ""
                        if url:
                            try:
                                from urllib.parse import urlparse
                                parsed = urlparse(url)
                                domain = parsed.netloc.replace("www.", "")
                                # Clean domain (remove port if present)
                                if ":" in domain:
                                    domain = domain.split(":")[0]
                            except Exception as e:
                                logger.debug(f"Error parsing URL {url}: {e}")
                        
                        # Only add if we have meaningful content
                        if title or (url and snippet):
                            results.append({
                                "title": title or url,  # Use URL as fallback title
                                "url": url,
                                "snippet": snippet,
                                "source": "web",
                                "source_type": "web_search",
                                "domain": domain,
                                "author": "",  # DuckDuckGo doesn't provide this
                                "date": "",  # DuckDuckGo doesn't provide this
                                "publication": domain  # Use domain as publication name
                            })
                            logger.debug(f"Added result {len(results)}: {title[:50]}...")
                        else:
                            logger.debug(f"Skipping result {idx}: insufficient content")
                            
                    except Exception as e:
                        logger.warning(f"Error processing result {idx}: {e}")
                        continue
                
                logger.info(f"Processed {len(results)} valid web results from {len(search_results)} raw results")
                
                # Log if we filtered out results
                if search_results and len(results) < len(search_results):
                    logger.info(f"Filtered {len(search_results) - len(results)} invalid results (missing title/URL)")
                
                # Apply hybrid reranking if enabled
                if self.use_reranking and results:
                    logger.info(f"Applying hybrid reranking to {len(results)} results...")
                    results = self._hybrid_rerank_results(query, results, top_k=max_results)
                
                # Cache results if we got any
                if results and self.cache:
                    self.cache.set(cache_key, results)
                elif not results:
                    logger.warning(f"Final result: No valid results returned for query: '{query}' (had {len(search_results)} raw results)")
                
            elif self.provider == "serpapi" and self.api_key:
                # SERP API implementation
                import requests
                params = {
                    "q": query,
                    "api_key": self.api_key,
                    "num": fetch_count
                }
                response = requests.get("https://serpapi.com/search", params=params)
                if response.status_code == 200:
                    data = response.json()
                    for result in data.get("organic_results", [])[:fetch_count]:
                        # SERP API provides more metadata
                        title = result.get("title", "")
                        url = result.get("link", "")
                        snippet = result.get("snippet", "")
                        date = result.get("date", "")
                        
                        # Extract domain
                        domain = ""
                        if url:
                            try:
                                from urllib.parse import urlparse
                                parsed = urlparse(url)
                                domain = parsed.netloc.replace("www.", "")
                            except:
                                pass
                        
                        results.append({
                            "title": title,
                            "url": url,
                            "snippet": snippet,
                            "source": "web",
                            "source_type": "web_search",
                            "domain": domain,
                            "author": "",  # SERP API may not always provide
                            "date": date,
                            "publication": domain
                        })
                    
                    # Apply hybrid reranking if enabled
                    if self.use_reranking and results:
                        logger.info(f"Applying hybrid reranking to {len(results)} results...")
                        results = self._hybrid_rerank_results(query, results, top_k=max_results)
                
                # Cache results
                if self.cache and results:
                    self.cache.set(cache_key, results)
            
            else:
                logger.warning("No web search provider available")
                # Return empty results or mock data for testing
                if st:
                    if not DDGS_AVAILABLE:
                        st.warning("⚠️ Web search not available. Install duckduckgo-search: `pip install duckduckgo-search`")
                    elif not self.ddgs:
                        st.warning("⚠️ DuckDuckGo search failed to initialize. Check your internet connection.")
                logger.warning(f"Web search provider status - DDGS_AVAILABLE: {DDGS_AVAILABLE}, ddgs initialized: {self.ddgs is not None}")
        
        except Exception as e:
            error_str = str(e)
            logger.error(f"Web search error: {error_str}")
            
            # Provide helpful error messages
            if "timeout" in error_str.lower() or "timed out" in error_str.lower() or "bing" in error_str.lower():
                error_msg = (
                    "⚠️ Web search timed out (DuckDuckGo backend issue). "
                    "The research overview will continue without web references. "
                    "You can try again later or proceed with the generated content."
                )
                logger.warning(error_msg)
                if st:
                    st.warning(error_msg)
            else:
                error_msg = f"Web search encountered an error: {error_str[:150]}. Continuing without web references."
                logger.warning(error_msg)
                if st:
                    st.warning(f"⚠️ {error_msg}")
        
        return results
    
    def is_available(self) -> bool:
        """Check if web search is available."""
        if self.provider == "duckduckgo":
            return DDGS_AVAILABLE and self.ddgs is not None
        elif self.provider == "serpapi":
            return bool(self.api_key)
        return False
