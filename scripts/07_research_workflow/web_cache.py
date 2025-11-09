"""
Web Search Cache Module
Caches web search results to improve speed.
"""

import os
import json
import hashlib
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class WebSearchCache:
    """Cache for web search results."""
    
    def __init__(self, cache_file: str = "memory/web_search_cache.json", cache_duration_hours: int = 24):
        """
        Initialize web search cache.
        
        Args:
            cache_file: Path to cache file
            cache_duration_hours: How long to cache results (default: 24 hours)
        """
        self.cache_file = cache_file
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self.cache_dir = os.path.dirname(cache_file)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load cache from file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file."""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.md5(query.lower().encode()).hexdigest()
    
    def get(self, query: str) -> Optional[List[Dict]]:
        """Get cached results if available and not expired."""
        cache_key = self._get_cache_key(query)
        
        if cache_key in self.cache:
            cached_entry = self.cache[cache_key]
            cached_time = datetime.fromisoformat(cached_entry.get("timestamp", ""))
            
            if datetime.now() - cached_time < self.cache_duration:
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_entry.get("results", [])
            else:
                # Expired, remove from cache
                del self.cache[cache_key]
                self._save_cache()
        
        return None
    
    def set(self, query: str, results: List[Dict]):
        """Cache search results."""
        cache_key = self._get_cache_key(query)
        self.cache[cache_key] = {
            "query": query,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        self._save_cache()
        logger.info(f"Cached {len(results)} results for query: {query[:50]}...")
    
    def clear(self):
        """Clear all cache."""
        self.cache = {}
        self._save_cache()
        logger.info("Cache cleared")

