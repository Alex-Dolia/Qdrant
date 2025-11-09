"""
Academic Search Module
Provides integration with academic sources: arXiv, Semantic Scholar, ACL Anthology.

Features:
- arXiv API integration (free, no API key required)
- Semantic Scholar REST API (free tier)
- Metadata extraction: title, authors, year, venue, DOI, URL, abstract
- Relevance ranking and filtering by recency (2020-2025 prioritized)
"""

import os
import logging
import time
import json
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False
    logger.warning("arxiv library not available. Install with: pip install arxiv")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests library not available. Install with: pip install requests")


class AcademicSearch:
    """Academic search interface for arXiv, Semantic Scholar, and other sources."""
    
    def __init__(self, use_cache: bool = True):
        """
        Initialize academic search.
        
        Args:
            use_cache: Enable caching for faster repeated searches
        """
        self.use_cache = use_cache
        
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
        
        # Semantic Scholar API key (optional, free tier available)
        self.semantic_scholar_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", None)
        
        logger.info("AcademicSearch initialized")
    
    def normalize_authors(self, authors: List[str]) -> List[str]:
        """Normalize author names."""
        normalized = []
        for author in authors:
            if isinstance(author, str):
                normalized.append(author.strip())
            elif hasattr(author, 'name'):
                normalized.append(author.name.strip())
            elif isinstance(author, dict):
                name = author.get('name', '')
                if name:
                    normalized.append(name.strip())
        return [a for a in normalized if a]
    
    def search_arxiv(self, query: str, max_results: int = 50, sort_by: str = "relevance") -> List[Dict[str, Any]]:
        """
        Search arXiv using the arxiv Python library.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            sort_by: "relevance" or "lastUpdatedDate" or "submittedDate"
            
        Returns:
            List of paper metadata dictionaries
        """
        if not ARXIV_AVAILABLE:
            logger.warning("arXiv library not available. Install with: pip install arxiv")
            return []
        
        # Check cache first
        cache_key = f"arxiv:{query}:{max_results}:{sort_by}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                logger.info(f"Using cached arXiv results for: {query}")
                return cached
        
        results = []
        try:
            # Map sort_by to arxiv.SortCriterion
            if sort_by == "relevance":
                sort_criterion = arxiv.SortCriterion.Relevance
            elif sort_by == "lastUpdatedDate":
                sort_criterion = arxiv.SortCriterion.LastUpdatedDate
            else:
                sort_criterion = arxiv.SortCriterion.SubmittedDate
            
            # Search arXiv
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_criterion
            )
            
            logger.info(f"Searching arXiv for: {query} (max_results={max_results})")
            
            for result in search.results():
                # Extract DOI if available
                doi = None
                if hasattr(result, 'doi') and result.doi:
                    doi = result.doi
                elif hasattr(result, 'journal_ref') and result.journal_ref:
                    # Try to extract DOI from journal_ref
                    pass
                
                paper_metadata = {
                    "id": result.entry_id.split('/')[-1],  # arXiv ID
                    "title": result.title,
                    "authors": self.normalize_authors([a.name for a in result.authors]),
                    "year": result.published.year if result.published else None,
                    "venue": "arXiv",
                    "doi": doi,
                    "url": result.pdf_url,  # Direct PDF URL
                    "abstract": result.summary,
                    "source": "arxiv",
                    "published": result.published.isoformat() if result.published else None,
                    "arxiv_id": result.entry_id.split('/')[-1]
                }
                results.append(paper_metadata)
            
            logger.info(f"Found {len(results)} arXiv results")
            
            # Cache results
            if self.cache and results:
                self.cache.set(cache_key, results)
            
        except Exception as e:
            logger.error(f"arXiv search error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        return results
    
    def search_semantic_scholar(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Search Semantic Scholar using REST API (free tier).
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of paper metadata dictionaries
        """
        if not REQUESTS_AVAILABLE:
            logger.warning("requests library not available. Install with: pip install requests")
            return []
        
        # Check cache first
        cache_key = f"semantic_scholar:{query}:{max_results}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                logger.info(f"Using cached Semantic Scholar results for: {query}")
                return cached
        
        results = []
        try:
            # Semantic Scholar REST API endpoint
            api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            
            params = {
                "query": query,
                "limit": min(max_results, 100),  # API limit is 100
                "fields": "title,authors,year,venue,externalIds,url,abstract,citationCount"
            }
            
            headers = {}
            if self.semantic_scholar_api_key:
                headers["x-api-key"] = self.semantic_scholar_api_key
            
            logger.info(f"Searching Semantic Scholar for: {query} (max_results={max_results})")
            
            response = requests.get(api_url, params=params, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get("data", []):
                    external_ids = item.get("externalIds", {}) or {}
                    doi = external_ids.get("DOI")
                    
                    # Get authors
                    authors = []
                    for author in item.get("authors", []):
                        if isinstance(author, dict):
                            authors.append(author.get("name", ""))
                        else:
                            authors.append(str(author))
                    
                    paper_metadata = {
                        "id": item.get("paperId", ""),
                        "title": item.get("title", ""),
                        "authors": self.normalize_authors(authors),
                        "year": item.get("year"),
                        "venue": item.get("venue", ""),
                        "doi": doi,
                        "url": item.get("url", ""),
                        "abstract": item.get("abstract", ""),
                        "source": "semantic_scholar",
                        "citation_count": item.get("citationCount", 0)
                    }
                    results.append(paper_metadata)
                
                logger.info(f"Found {len(results)} Semantic Scholar results")
            else:
                logger.warning(f"Semantic Scholar API returned status {response.status_code}: {response.text[:200]}")
            
            # Cache results
            if self.cache and results:
                self.cache.set(cache_key, results)
            
            # Rate limiting: Semantic Scholar free tier allows 100 requests per 5 minutes
            time.sleep(0.1)  # Small delay to avoid rate limits
            
        except Exception as e:
            logger.error(f"Semantic Scholar search error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        return results
    
    def rank_and_filter(
        self,
        papers: List[Dict[str, Any]],
        topic: str,
        min_year: int = 2020,
        max_results: int = 40,
        prioritize_recent: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Rank and filter papers by relevance and recency.
        
        Args:
            papers: List of paper metadata dictionaries
            topic: Research topic for relevance scoring
            min_year: Minimum publication year (default: 2020)
            max_results: Maximum number of results to return
            prioritize_recent: Prioritize recent papers (2020-2025)
            
        Returns:
            Ranked and filtered list of papers
        """
        if not papers:
            return []
        
        scored_papers = []
        current_year = datetime.now().year
        
        for paper in papers:
            score = 0.0
            
            # Recency score (higher for recent papers)
            year = paper.get("year")
            if year:
                if prioritize_recent and year >= min_year:
                    # Recent papers get higher score
                    years_ago = current_year - year
                    recency_score = max(0, 1.0 - (years_ago / 10.0))  # Decay over 10 years
                    score += recency_score * 0.4  # 40% weight for recency
                elif year < min_year:
                    # Very old papers get penalized
                    score -= 0.2
            
            # Citation count score (if available)
            citation_count = paper.get("citation_count", 0)
            if citation_count > 0:
                # Normalize citation count (log scale)
                import math
                citation_score = min(1.0, math.log10(citation_count + 1) / 3.0)  # Max at ~1000 citations
                score += citation_score * 0.2  # 20% weight for citations
            
            # Relevance score based on title and abstract
            title = paper.get("title", "").lower()
            abstract = paper.get("abstract", "").lower()
            topic_lower = topic.lower()
            
            # Simple keyword matching (can be enhanced with embeddings)
            topic_words = set(topic_lower.split())
            title_words = set(title.split())
            abstract_words = set(abstract.split()[:100].split())  # First 100 words
            
            title_overlap = len(topic_words & title_words) / max(len(topic_words), 1)
            abstract_overlap = len(topic_words & abstract_words) / max(len(topic_words), 1)
            
            relevance_score = (title_overlap * 0.6 + abstract_overlap * 0.4)
            score += relevance_score * 0.4  # 40% weight for relevance
            
            # Venue quality bonus
            venue = paper.get("venue", "").lower()
            high_quality_venues = ["neurips", "icml", "iclr", "acl", "emnlp", "naacl", "aaai", "ijcai", "jmlr", "tacl"]
            if any(v in venue for v in high_quality_venues):
                score += 0.1
            
            scored_papers.append((score, paper))
        
        # Sort by score (descending)
        scored_papers.sort(key=lambda x: x[0], reverse=True)
        
        # Return top N papers
        ranked_papers = [paper for _, paper in scored_papers[:max_results]]
        
        logger.info(f"Ranked {len(papers)} papers, selected top {len(ranked_papers)}")
        
        return ranked_papers
    
    def search(
        self,
        topic: str,
        max_results: int = 40,
        sources: List[str] = None,
        min_year: int = 2020,
        prioritize_recent: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search across multiple academic sources.
        
        Args:
            topic: Research topic to search for
            max_results: Maximum total results to return
            sources: List of sources to search ("arxiv", "semantic_scholar"). If None, uses all available.
            min_year: Minimum publication year
            prioritize_recent: Prioritize recent papers
            
        Returns:
            List of ranked and filtered paper metadata dictionaries
        """
        if sources is None:
            sources = ["arxiv", "semantic_scholar"]
        
        all_papers = []
        seen_titles = set()
        
        # Form multiple queries for better coverage
        queries = [
            topic,
            f"{topic} explainability interpretability",
            f"{topic} methods techniques",
            f"{topic} survey review"
        ]
        
        for query in queries:
            # Search arXiv
            if "arxiv" in sources and ARXIV_AVAILABLE:
                try:
                    arxiv_results = self.search_arxiv(query, max_results=max_results // len(queries))
                    for paper in arxiv_results:
                        title = paper.get("title", "").lower()
                        if title not in seen_titles:
                            all_papers.append(paper)
                            seen_titles.add(title)
                except Exception as e:
                    logger.warning(f"arXiv search failed for query '{query}': {e}")
            
            # Search Semantic Scholar
            if "semantic_scholar" in sources and REQUESTS_AVAILABLE:
                try:
                    ss_results = self.search_semantic_scholar(query, max_results=max_results // len(queries))
                    for paper in ss_results:
                        title = paper.get("title", "").lower()
                        if title not in seen_titles:
                            all_papers.append(paper)
                            seen_titles.add(title)
                except Exception as e:
                    logger.warning(f"Semantic Scholar search failed for query '{query}': {e}")
            
            # Small delay between queries to avoid rate limits
            time.sleep(0.5)
        
        # Rank and filter results
        ranked_papers = self.rank_and_filter(
            all_papers,
            topic=topic,
            min_year=min_year,
            max_results=max_results,
            prioritize_recent=prioritize_recent
        )
        
        logger.info(f"Academic search completed: {len(ranked_papers)} papers found for topic '{topic}'")
        
        return ranked_papers
    
    def is_available(self) -> bool:
        """Check if academic search is available."""
        return ARXIV_AVAILABLE or REQUESTS_AVAILABLE

