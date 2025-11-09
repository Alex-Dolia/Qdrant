"""
Hybrid Web Search with BM25 + Embeddings Reranking
This module adds hybrid search capabilities to web_search.py
"""

import numpy as np
import re
from typing import List, Dict

def add_hybrid_search_to_web_search():
    """
    This function adds hybrid search methods to the WebSearch class.
    Import and call this after importing WebSearch.
    """
    from scripts.07_research_workflow.web_search import WebSearch, BM25_AVAILABLE, BM25Okapi
    import logging
    logger = logging.getLogger(__name__)
    
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
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                logger.info("Initialized default HuggingFace embeddings")
        except ImportError as e:
            logger.warning(f"Failed to initialize embeddings: {e}")
            self.embeddings = None
            self.use_reranking = False
        except Exception as e:
            logger.warning(f"Error initializing embeddings: {e}")
            self.embeddings = None
            self.use_reranking = False
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _hybrid_rerank_results(self, query: str, results: List[Dict[str, str]], top_k: int = 10) -> List[Dict[str, str]]:
        """Rerank search results using hybrid BM25 + embedding similarity."""
        if not self.use_reranking or not results:
            return results[:top_k]
        
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
            if BM25_AVAILABLE and BM25Okapi:
                try:
                    query_tokens = self._tokenize(query)
                    tokenized_docs = [self._tokenize(text) for text in result_texts]
                    bm25 = BM25Okapi(tokenized_docs)
                    bm25_scores = bm25.get_scores(query_tokens)
                    if len(bm25_scores) > 0:
                        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
                        bm25_scores = [score / max_bm25 for score in bm25_scores]
                except Exception as e:
                    logger.warning(f"BM25 scoring failed: {e}")
                    bm25_scores = [0.0] * len(results)
            else:
                bm25_scores = [0.0] * len(results)
            
            # Embedding-based similarity
            if self.embeddings:
                try:
                    query_embedding = self.embeddings.embed_query(query)
                    result_embeddings = self.embeddings.embed_documents(result_texts)
                    query_norm = np.linalg.norm(query_embedding)
                    for result_emb in result_embeddings:
                        if query_norm > 0:
                            similarity = np.dot(query_embedding, result_emb) / (query_norm * np.linalg.norm(result_emb))
                        else:
                            similarity = 0.0
                        embedding_scores.append((similarity + 1) / 2)
                except Exception as e:
                    logger.warning(f"Embedding scoring failed: {e}")
                    embedding_scores = [0.0] * len(results)
            else:
                embedding_scores = [0.0] * len(results)
            
            # Combine scores (40% BM25, 60% embeddings)
            hybrid_scores = []
            for i in range(len(results)):
                hybrid_score = (0.4 * bm25_scores[i]) + (0.6 * embedding_scores[i])
                hybrid_scores.append(hybrid_score)
            
            # Sort by hybrid score
            sorted_indices = np.argsort(hybrid_scores)[::-1]
            
            # Create reranked results
            reranked_results = []
            for idx in sorted_indices:
                result = results[idx].copy()
                result["relevance_score"] = float(hybrid_scores[idx])
                result["bm25_score"] = float(bm25_scores[idx])
                result["embedding_score"] = float(embedding_scores[idx])
                
                domain = result.get("domain", "")
                if domain:
                    if "linkedin.com" in domain:
                        result["source"] = "LinkedIn"
                    elif "researchgate.net" in domain or "sciencedirect.com" in domain or "arxiv.org" in domain:
                        result["source"] = "Academic"
                    elif any(news in domain for news in ["bbc.com", "cnn.com", "reuters.com", "nytimes.com"]):
                        result["source"] = "News"
                    elif domain.endswith(".pdf") or "pdf" in domain.lower():
                        result["source"] = "PDF/Paper"
                    else:
                        result["source"] = "Web"
                
                reranked_results.append(result)
                if len(reranked_results) >= top_k:
                    break
            
            logger.info(f"Hybrid reranked {len(results)} results, returning top {len(reranked_results)}")
            return reranked_results
            
        except Exception as e:
            logger.warning(f"Hybrid reranking failed: {e}, returning original results")
            return results[:top_k]
    
    # Add methods to WebSearch class
    WebSearch._initialize_embeddings = _initialize_embeddings
    WebSearch._tokenize = _tokenize
    WebSearch._hybrid_rerank_results = _hybrid_rerank_results
    
    logger.info("Hybrid search methods added to WebSearch class")

