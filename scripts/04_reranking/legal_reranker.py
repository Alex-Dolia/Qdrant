"""
Production-Ready Advanced Reranking System for Legal Document RAG

This module provides a comprehensive reranking system specifically designed for legal documents,
integrating multiple retrieval methods, legal-specific optimizations, and ensemble ranking techniques.

Key Features:
- Hybrid search (semantic, BM25, n-gram)
- Legal clause type detection and weighting
- Obligation pattern recognition
- Cross-reference resolution
- Party-specific relevance boosting
- Llama 3 70B reranking via Together.ai
- Reciprocal Rank Fusion for ensemble methods
- Async API support with fallbacks
- Comprehensive caching and monitoring

Legal AI Best Practices:
- Accuracy prioritized over speed for contractual obligations
- Detailed logging for audit trails
- Configurable parameters for different legal contexts
- Safeguards against hallucination in legal interpretations
"""

import os
import re
import json
import time
import logging
import asyncio
import hashlib
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
from functools import lru_cache
from datetime import datetime
from collections import defaultdict

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except ImportError:
    nltk = None
    word_tokenize = lambda x: x.split()
    sent_tokenize = lambda x: [x]

try:
    from together import Together
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False
    Together = None

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    retry = lambda **kwargs: lambda f: f

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from pydantic import BaseModel, Field, validator

# Configure logging
logger = logging.getLogger(__name__)


class LegalClauseType(Enum):
    """Enum for legal clause types with their relative importance weights.
    
    Higher weights indicate clauses that are more critical for legal interpretation.
    These weights are used to boost relevance scores for important contractual elements.
    """
    DEFINITIONS = 1.3  # Important for understanding contract terms
    OBLIGATIONS = 1.7  # Critical - defines what parties must do
    TERMINATION = 1.9  # Very critical - contract exit conditions
    LIABILITY = 1.8    # Critical - risk allocation
    PAYMENT = 1.6      # Important - financial obligations
    CONFIDENTIALITY = 1.5  # Important - information protection
    INTELLECTUAL_PROPERTY = 1.7  # Critical - IP rights
    FORCE_MAJEURE = 1.4  # Important - exception clauses
    GOVERNING_LAW = 1.5  # Important - legal framework
    GENERAL = 1.0  # Default weight for unclassified clauses


@dataclass
class RankedChunk:
    """Data structure for ranked document chunks with legal context.
    
    Contains both retrieval scores and legal-specific metadata that influences
    final ranking for legal document queries.
    """
    id: str
    content: str
    metadata: Dict[str, Any]
    base_score: float
    rerank_score: float = 0.0
    final_score: float = 0.0
    clause_type: LegalClauseType = LegalClauseType.GENERAL
    section_id: str = ""
    party_relevance: float = 1.0
    is_obligation: bool = False
    retrieval_method: str = ""  # "semantic", "bm25", "ngram", "hybrid"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "scores": {
                "base": self.base_score,
                "rerank": self.rerank_score,
                "final": self.final_score
            },
            "legal_context": {
                "clause_type": self.clause_type.name,
                "section_id": self.section_id,
                "party_relevance": self.party_relevance,
                "is_obligation": self.is_obligation
            },
            "retrieval_method": self.retrieval_method
        }


class LegalDocumentProcessor:
    """Processes legal documents for optimal retrieval and reranking.
    
    Extracts legal-specific features including clause types, obligations,
    party references, and cross-references that are critical for accurate
    legal document ranking.
    """
    
    def __init__(self):
        # Critical legal terms that indicate important contractual language
        self.critical_terms = [
            "shall", "must", "will", "agrees", "undertakes", "warrants", "indemnifies",
            "terminat", "liabili", "responsib", "obligation", "breach", "penalt", "fee",
            "payment", "confidential", "intellectual property", "license", "assignment",
            "warranty", "representation", "indemnity", "limitation", "disclaimer"
        ]
        
        # Precompile regex patterns for performance
        self.section_pattern = re.compile(r'^\s*(\d+\.\d*\.?)\s+([A-Z][^\n]+)', re.MULTILINE)
        self.clause_reference_pattern = re.compile(r'clause\s+(\d+\.\d*)', re.IGNORECASE)
        self.definition_pattern = re.compile(r'"([^"]+)"\s+means\s+([^;]+);', re.IGNORECASE)
        self.party_pattern = re.compile(r'(Client|Agency|Consultancy|Contractor|Vendor|Supplier|Party)', re.IGNORECASE)
        self.obligation_pattern = re.compile(
            r'\b(Client|Agency|Consultancy|Contractor|Vendor|Supplier|Party)\s+(shall|must|will|agrees? to|undertakes? to|warrants? that)\b|'
            r'\b(shall|must|will|agrees? to|undertakes? to|warrants? that)\s+(Client|Agency|Consultancy|Contractor|Vendor|Supplier|Party)\b',
            re.IGNORECASE
        )
    
    def identify_clause_type(self, text: str) -> LegalClauseType:
        """Identify the type of legal clause based on content.
        
        This is critical for legal RAG as different clause types have
        different importance weights in ranking.
        """
        text_upper = text.upper()
        
        # Check for clause type keywords (order matters - more specific first)
        if any(term in text_upper for term in ["TERMINAT", "END OF AGREEMENT", "CEASE", "EXPIR"]):
            return LegalClauseType.TERMINATION
        elif any(term in text_upper for term in ["LIABILI", "INDEMN", "DAMAGES", "LOSS", "CLAIM"]):
            return LegalClauseType.LIABILITY
        elif any(term in text_upper for term in ["PAYMENT", "FEE", "CHARGE", "EXPENSE", "INVOICE", "COMPENSATION"]):
            return LegalClauseType.PAYMENT
        elif any(term in text_upper for term in ["CONFIDENTIAL", "NON-DISCLOSURE", "NDA", "PROPRIETARY"]):
            return LegalClauseType.CONFIDENTIALITY
        elif any(term in text_upper for term in ["INTELLECTUAL PROPERTY", "COPYRIGHT", "PATENT", "TRADEMARK", "LICENSE", "IP"]):
            return LegalClauseType.INTELLECTUAL_PROPERTY
        elif any(term in text_upper for term in ["FORCE MAJEURE", "ACT OF GOD", "UNFORESEEABLE", "BEYOND CONTROL"]):
            return LegalClauseType.FORCE_MAJEURE
        elif any(term in text_upper for term in ["GOVERNING LAW", "JURISDICTION", "DISPUTE", "ARBITRATION", "COURT"]):
            return LegalClauseType.GOVERNING_LAW
        elif any(term in text_upper for term in ["DEFINITION", "MEANS", "DEFINED AS", "REFERS TO"]):
            return LegalClauseType.DEFINITIONS
        elif any(term in text_upper for term in ["SHALL", "MUST", "WILL", "AGREES", "UNDERTAKES", "WARRANTS", "OBLIGATED"]):
            return LegalClauseType.OBLIGATIONS
        
        return LegalClauseType.GENERAL
    
    def extract_section_id(self, text: str) -> str:
        """Extract section ID from legal text (e.g., '3.1', '15.2').
        
        Section IDs are important for cross-reference resolution and
        maintaining document structure context.
        """
        match = self.section_pattern.search(text)
        if match:
            return match.group(1).rstrip('.')
        return "unknown"
    
    def detect_obligations(self, text: str) -> bool:
        """Detect if text contains obligations for parties.
        
        Obligations are critical in legal documents as they define
        what parties must do, making them highly relevant for queries.
        """
        return bool(self.obligation_pattern.search(text))
    
    def calculate_party_relevance(self, text: str, query: str) -> float:
        """Calculate relevance to specific parties mentioned in query.
        
        If a query mentions specific parties (e.g., "Client obligations"),
        boost chunks that mention those parties.
        """
        party_mentions = self.party_pattern.findall(text)
        query_parties = self.party_pattern.findall(query)
        
        if not query_parties or not party_mentions:
            return 1.0
        
        # Count matching parties
        matches = sum(1 for p in party_mentions if any(q.lower() in p.lower() for q in query_parties))
        return 1.0 + (0.3 * matches)  # Boost by 30% per matching party
    
    def extract_ngrams(self, text: str, n: int) -> List[str]:
        """Extract n-grams from text for legal terminology matching."""
        if nltk:
            tokens = word_tokenize(text.lower())
        else:
            tokens = text.lower().split()
        return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def get_legal_features(self, text: str, query: str = "") -> Dict[str, Any]:
        """Extract legal-specific features for ranking.
        
        These features are used to boost relevance scores for chunks
        that contain legally significant content.
        """
        return {
            "clause_type": self.identify_clause_type(text),
            "section_id": self.extract_section_id(text),
            "is_obligation": self.detect_obligations(text),
            "party_relevance": self.calculate_party_relevance(text, query),
            "critical_term_count": sum(1 for term in self.critical_terms if term in text.lower()),
            "definition_count": len(self.definition_pattern.findall(text)),
            "reference_count": len(self.clause_reference_pattern.findall(text))
        }


class NgramSearch:
    """N-gram based search for legal terminology matching.
    
    Legal documents often contain specific multi-word phrases that are
    better matched using n-grams than single words. This is especially
    important for legal terminology like "intellectual property rights"
    or "force majeure event".
    """
    
    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.ngram_indices = {n: defaultdict(list) for n in range(1, 6)}  # 1-5 gram indices
        self.doc_id_to_index = {doc["id"]: i for i, doc in enumerate(documents)}
        self._build_indices()
    
    def _build_indices(self):
        """Build n-gram indices for all documents."""
        processor = LegalDocumentProcessor()
        
        for doc in self.documents:
            doc_id = doc["id"]
            text = doc.get("content", doc.get("text", ""))
            
            for n in range(1, 6):
                ngrams = processor.extract_ngrams(text, n)
                for ng in ngrams:
                    self.ngram_indices[n][ng].append(doc_id)
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using n-gram matching with weighted scoring.
        
        Longer n-grams (phrases) get higher weight as they indicate
        more specific legal terminology matches.
        """
        processor = LegalDocumentProcessor()
        query_ngrams = {}
        
        # Extract n-grams from query
        for n in range(1, 6):
            query_ngrams[n] = processor.extract_ngrams(query, n)
        
        # Score documents based on n-gram matches
        doc_scores = defaultdict(float)
        for n, ngrams in query_ngrams.items():
            # Higher weight for longer n-grams (more specific matches)
            weight = n * 0.2  # 1-gram: 0.2, 2-gram: 0.4, ..., 5-gram: 1.0
            
            for ng in ngrams:
                if ng in self.ngram_indices[n]:
                    for doc_id in self.ngram_indices[n][ng]:
                        doc_scores[doc_id] += weight
        
        # Normalize scores
        if not doc_scores:
            return []
        
        max_score = max(doc_scores.values()) if doc_scores.values() else 1.0
        normalized_results = [
            (doc_id, score / max_score if max_score > 0 else 0) 
            for doc_id, score in doc_scores.items()
        ]
        
        # Sort and return top_k
        normalized_results.sort(key=lambda x: x[1], reverse=True)
        return normalized_results[:top_k]


class BM25Retriever:
    """BM25 implementation for sparse retrieval.
    
    BM25 is effective for keyword-based queries and complements
    semantic search by catching exact term matches.
    """
    
    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.doc_ids = [doc["id"] for doc in documents]
        if nltk:
            self.corpus = [word_tokenize(doc.get("content", doc.get("text", "")).lower()) for doc in documents]
        else:
            self.corpus = [doc.get("content", doc.get("text", "")).lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.corpus)
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using BM25 algorithm."""
        if nltk:
            tokenized_query = word_tokenize(query.lower())
        else:
            tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        max_score = max(scores) if max(scores) > 0 else 1.0
        
        results = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            normalized_score = scores[idx] / max_score if max_score > 0 else 0
            results.append((doc_id, float(normalized_score)))
        
        return results


class ReciprocalRankFusion:
    """Reciprocal Rank Fusion for combining multiple retrieval methods.
    
    RRF is a robust ensemble method that combines ranked lists from
    different retrieval methods without requiring score normalization.
    Supports weighted RRF for fine-tuning method contributions.
    """
    
    @staticmethod
    def fuse(
        ranked_lists: List[List[Tuple[str, float]]], 
        k: int = 60,
        weights: Optional[List[float]] = None
    ) -> List[Tuple[str, float]]:
        """Fuse multiple ranked lists using weighted RRF.
        
        Args:
            ranked_lists: List of ranked lists, each as [(doc_id, score), ...]
            k: RRF constant (default 60, standard value)
            weights: Optional weights for each ranked list (default: equal weights)
        
        Returns:
            Fused ranked list
        """
        if weights is None:
            weights = [1.0 / len(ranked_lists)] * len(ranked_lists)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(ranked_lists)] * len(ranked_lists)
        
        doc_scores = defaultdict(float)
        
        for ranked_list, weight in zip(ranked_lists, weights):
            for rank, (doc_id, _) in enumerate(ranked_list, start=1):
                doc_scores[doc_id] += weight * (1.0 / (k + rank))
        
        # Sort by RRF score
        fused_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return fused_results


class Llama3Reranker:
    """Reranker using Llama 3 70B via Together.ai API.
    
    Uses cross-encoder style reranking where the model sees both
    query and document together, allowing for deeper understanding
    of relevance in legal context. Supports async operations for better performance.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Llama 3 reranker.
        
        Args:
            api_key: Together.ai API key (or from TOGETHER_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            logger.warning("TOGETHER_API_KEY not set. Reranking will be disabled.")
            self.client = None
            return
        
        if not TOGETHER_AVAILABLE:
            logger.warning("together package not available. Install with: pip install together")
            self.client = None
            return
        
        try:
            self.client = Together(api_key=self.api_key)
            self.model = os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3-70b-chat-hf")
            logger.info(f"Initialized Llama 3 reranker with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Together.ai client: {e}")
            self.client = None
    
    def _create_legal_reranking_prompt(self, query: str, chunks: List[str]) -> str:
        """Create a legal-specific prompt for reranking with better legal context awareness.
        
        This prompt is designed to help the model understand legal hierarchy,
        contractual obligations, and the importance of different clause types.
        """
        chunk_block = "\n\n".join([
            f"CHUNK {i+1}:\n{chunk[:500]}..." if len(chunk) > 500 else f"CHUNK {i+1}:\n{chunk}"
            for i, chunk in enumerate(chunks)
        ])
        
        return f"""
QUERY: "{query}"

DOCUMENT CHUNKS:

{chunk_block}

INSTRUCTIONS:

1. Analyze each chunk's legal relevance to the query with emphasis on:
   - Contractual obligations and liabilities
   - Critical clauses (termination, payment, indemnity)
   - Definitions that affect interpretation
   - Party-specific responsibilities
   - Temporal aspects (effective dates, notice periods)

2. Consider legal hierarchy: 
   - Main clauses > sub-clauses
   - Definitions > general provisions
   - Obligations > procedural details
   - Termination clauses have high importance

3. Evaluate contextual completeness:
   - Prefer chunks that contain complete legal thoughts
   - Avoid fragments that lack critical context
   - Prioritize clauses with explicit party obligations

4. Output format: ONLY provide a comma-separated list of chunk numbers in order of relevance (most relevant first).
   Example: "3,1,5,2,4"

5. Be strictly factual and legally precise in your evaluation.
"""
    
    def _parse_reranking_response(self, response: str, total_chunks: int) -> List[int]:
        """Parse the reranking response to extract ordered chunk indices.
        
        More robust parsing that handles various response formats.
        """
        try:
            # Extract numbers from response
            numbers = re.findall(r'\d+', response)
            indices = [int(num) - 1 for num in numbers if 1 <= int(num) <= total_chunks]
            
            # Remove duplicates while preserving order
            seen = set()
            unique_indices = []
            for idx in indices:
                if idx not in seen:
                    seen.add(idx)
                    unique_indices.append(idx)
            
            # Add any missing indices
            for i in range(total_chunks):
                if i not in unique_indices:
                    unique_indices.append(i)
            
            return unique_indices[:total_chunks]
            
        except Exception as e:
            logger.error(f"Error parsing reranking response: {e}")
            return list(range(total_chunks))  # Default to original order
    
    async def rerank_chunks_async(
        self, 
        query: str, 
        chunks: List[RankedChunk], 
        top_k: Optional[int] = None
    ) -> List[RankedChunk]:
        """Async rerank chunks using Llama 3 with legal context awareness.
        
        This method processes all chunks together for better context understanding,
        which is more effective for legal document reranking.
        
        Args:
            query: Search query
            chunks: List of chunks to rerank
            top_k: Number of top results to return (None = all)
        
        Returns:
            Reranked chunks with updated rerank_score
        """
        if not self.client:
            logger.warning("Reranker not available. Returning original ranking.")
            return chunks
        
        if not chunks:
            return []
        
        logger.info(f"Reranking {len(chunks)} chunks with Llama 3 70B (async)")
        
        try:
            start_time = time.time()
            
            # Prepare chunk texts for reranking
            chunk_texts = [chunk.content for chunk in chunks]
            
            # Create legal-specific prompt
            prompt = self._create_legal_reranking_prompt(query, chunk_texts)
            
            # Call API (using asyncio.to_thread for async compatibility)
            if asyncio:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a legal expert specializing in contract analysis. Your task is to rerank "
                                "document chunks based on their relevance and importance to the query, with special "
                                "attention to contractual obligations, liabilities, and critical clauses. Consider "
                                "legal precision, context preservation, and the hierarchical importance of clauses."
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistent scoring
                    max_tokens=500,
                    top_p=0.9
                )
            else:
                # Fallback to sync if asyncio not available
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a legal expert specializing in contract analysis. Your task is to rerank "
                                "document chunks based on their relevance and importance to the query, with special "
                                "attention to contractual obligations, liabilities, and critical clauses."
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500,
                    top_p=0.9
                )
            
            processing_time = time.time() - start_time
            logger.info(f"Llama 3 reranking completed in {processing_time:.2f} seconds")
            
            # Parse response to get reranked order
            response_text = response.choices[0].message.content
            reranked_indices = self._parse_reranking_response(response_text, len(chunks))
            
            # Apply reranking
            reranked_chunks = []
            for idx in reranked_indices[:top_k if top_k else len(chunks)]:
                if idx < len(chunks):
                    chunk = chunks[idx]
                    # Calculate rerank score based on position
                    chunk.rerank_score = (len(reranked_indices) - reranked_indices.index(idx)) / len(reranked_indices)
                    reranked_chunks.append(chunk)
            
            logger.info(f"Reranking complete. Top score: {reranked_chunks[0].rerank_score if reranked_chunks else 0}")
            return reranked_chunks
            
        except Exception as e:
            logger.error(f"Error during Llama 3 reranking: {e}")
            # Fallback to original order with slight score decay
            for i, chunk in enumerate(chunks[:top_k if top_k else len(chunks)]):
                chunk.rerank_score = 1.0 - (i * 0.1)
            return chunks[:top_k if top_k else len(chunks)]
    
    def rerank(self, query: str, chunks: List[RankedChunk], top_k: Optional[int] = None) -> List[RankedChunk]:
        """Sync wrapper for rerank_chunks_async.
        
        For backward compatibility and when async is not needed.
        """
        if asyncio and hasattr(asyncio, '_get_running_loop'):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're in an async context, we need to handle differently
                    # For now, fall back to individual chunk processing
                    return self._rerank_chunks_sync(query, chunks, top_k)
            except RuntimeError:
                pass
        
        # Try async if possible
        try:
            if asyncio:
                return asyncio.run(self.rerank_chunks_async(query, chunks, top_k))
        except Exception:
            pass
        
        # Fallback to sync
        return self._rerank_chunks_sync(query, chunks, top_k)
    
    def _rerank_chunks_sync(self, query: str, chunks: List[RankedChunk], top_k: Optional[int] = None) -> List[RankedChunk]:
        """Synchronous reranking fallback (processes chunks individually)."""
        if not self.client:
            return chunks
        
        if not chunks:
            return []
        
        logger.info(f"Reranking {len(chunks)} chunks with Llama 3 70B (sync fallback)")
        
        reranked_chunks = []
        for chunk in chunks:
            try:
                # Use simpler prompt for individual chunk processing
                prompt = f"""You are a legal document analysis expert. Evaluate relevance (0.0-1.0) of this chunk to the query.

Query: {query}

Chunk: {chunk.content[:1500]}

Respond with ONLY a number between 0.0 and 1.0:"""
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a legal expert. Respond with only a number."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=50
                )
                
                # Extract score
                content = response.choices[0].message.content.strip()
                score_match = re.search(r'0?\.\d+|1\.0|0', content)
                if score_match:
                    chunk.rerank_score = float(score_match.group())
                else:
                    chunk.rerank_score = chunk.base_score
                
                reranked_chunks.append(chunk)
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error reranking chunk {chunk.id}: {e}")
                chunk.rerank_score = chunk.base_score
                reranked_chunks.append(chunk)
        
        reranked_chunks.sort(key=lambda x: x.rerank_score, reverse=True)
        return reranked_chunks[:top_k] if top_k else reranked_chunks


class LegalRAGReranker:
    """Production-ready advanced reranking system for legal document RAG.
    
    Integrates multiple retrieval methods, legal-specific optimizations,
    and ensemble ranking for superior legal document search accuracy.
    """
    
    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str = "legal_documents",
        embedding_model: str = "all-MiniLM-L6-v2",
        together_api_key: Optional[str] = None,
        enable_reranking: bool = True,
        enable_caching: bool = True
    ):
        """Initialize the legal RAG reranker.
        
        Args:
            qdrant_client: Qdrant client instance
            collection_name: Qdrant collection name
            embedding_model: Sentence transformer model for embeddings
            together_api_key: Together.ai API key for reranking
            enable_reranking: Whether to enable Llama 3 reranking
            enable_caching: Whether to enable query result caching
        """
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(embedding_model)
        self.processor = LegalDocumentProcessor()
        self.enable_reranking = enable_reranking
        self.enable_caching = enable_caching
        
        # Initialize reranker if enabled
        if enable_reranking:
            self.reranker = Llama3Reranker(api_key=together_api_key)
        else:
            self.reranker = None
        
        # Cache for query results
        self.cache = {} if enable_caching else None
        
        logger.info(f"Initialized LegalRAGReranker with collection: {collection_name}")
    
    def _get_cache_key(self, query: str, top_k: int, methods: List[str]) -> str:
        """Generate cache key for query."""
        key_str = f"{query}_{top_k}_{'_'.join(sorted(methods))}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        filter_condition: Optional[Filter] = None
    ) -> List[Tuple[str, float, Dict]]:
        """Perform semantic search using embeddings."""
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True
        ).tolist()
        
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k * 2,  # Get more for filtering
            query_filter=filter_condition,
            with_payload=True
        )
        
        return [
            (
                str(result.id),
                float(result.score),
                result.payload or {}
            ) for result in results
        ]
    
    def bm25_search(
        self,
        query: str,
        top_k: int = 10,
        filter_condition: Optional[Filter] = None
    ) -> List[Tuple[str, float]]:
        """Perform BM25 search on Qdrant collection."""
        # Retrieve all matching chunks
        all_chunks = []
        next_page_offset = None
        
        for _ in range(100):  # Safety limit
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                offset=next_page_offset,
                scroll_filter=filter_condition,
                with_payload=True,
                with_vectors=False
            )
            
            points, next_page_offset = scroll_result
            for point in points:
                payload = point.payload or {}
                text = payload.get("text", payload.get("content", ""))
                if text:
                    all_chunks.append({
                        "id": str(point.id),
                        "content": text
                    })
            
            if next_page_offset is None:
                break
        
        if not all_chunks:
            return []
        
        # Build BM25 index
        bm25_retriever = BM25Retriever(all_chunks)
        return bm25_retriever.search(query, top_k=top_k)
    
    def ngram_search(
        self,
        query: str,
        top_k: int = 10,
        filter_condition: Optional[Filter] = None
    ) -> List[Tuple[str, float]]:
        """Perform n-gram search on Qdrant collection."""
        # Retrieve all matching chunks
        all_chunks = []
        next_page_offset = None
        
        for _ in range(100):  # Safety limit
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                offset=next_page_offset,
                scroll_filter=filter_condition,
                with_payload=True,
                with_vectors=False
            )
            
            points, next_page_offset = scroll_result
            for point in points:
                payload = point.payload or {}
                text = payload.get("text", payload.get("content", ""))
                if text:
                    all_chunks.append({
                        "id": str(point.id),
                        "content": text
                    })
            
            if next_page_offset is None:
                break
        
        if not all_chunks:
            return []
        
        # Build n-gram index
        ngram_searcher = NgramSearch(all_chunks)
        return ngram_searcher.search(query, top_k=top_k)
    
    def _apply_legal_boosting(self, chunks: List[RankedChunk], query: str) -> List[RankedChunk]:
        """Apply legal-specific boosting to chunk scores.
        
        This is where legal domain knowledge is applied to improve
        ranking accuracy for legal documents. Uses sophisticated weighting
        based on clause types, obligations, definitions, and cross-references.
        """
        for chunk in chunks:
            # Get legal features
            legal_features = self.processor.get_legal_features(chunk.content, query)
            
            # Update chunk with legal context
            chunk.clause_type = legal_features["clause_type"]
            chunk.section_id = legal_features["section_id"]
            chunk.is_obligation = legal_features["is_obligation"]
            chunk.party_relevance = legal_features["party_relevance"]
            
            # Calculate legal boost factor
            boost = 1.0
            
            # Clause type weight (most important factor)
            boost *= legal_features["clause_type"].value
            
            # Obligation boost (obligations are highly relevant)
            if legal_features["is_obligation"]:
                boost *= 1.2
            
            # Party relevance boost
            boost *= legal_features["party_relevance"]
            
            # Critical term boost (indicates important legal language)
            if legal_features["critical_term_count"] > 0:
                boost *= 1.0 + (legal_features["critical_term_count"] * 0.05)
            
            # Definitions boost (important for contract interpretation)
            if legal_features["clause_type"] == LegalClauseType.DEFINITIONS:
                boost *= 1.5
            
            # Cross-reference boost (indicates importance and interconnectedness)
            if legal_features["reference_count"] > 0:
                boost *= 1.0 + (legal_features["reference_count"] * 0.1)
            
            # Apply boost to final score
            chunk.final_score = chunk.base_score * boost
        
        return chunks
    
    async def retrieve_and_rerank_async(
        self,
        query: str,
        top_k: int = 10,
        methods: Optional[List[str]] = None,
        filter_condition: Optional[Filter] = None,
        use_reranking: Optional[bool] = None
    ) -> List[RankedChunk]:
        """Async version of retrieve_and_rerank for better performance."""
        return await self._retrieve_and_rerank_impl(query, top_k, methods, filter_condition, use_reranking, async_mode=True)
    
    def retrieve_and_rerank(
        self,
        query: str,
        top_k: int = 10,
        methods: Optional[List[str]] = None,
        filter_condition: Optional[Filter] = None,
        use_reranking: Optional[bool] = None
    ) -> List[RankedChunk]:
        """Main method: retrieve and rerank legal documents (sync version).
        
        Args:
            query: Search query
            top_k: Number of results to return
            methods: List of retrieval methods to use ["semantic", "bm25", "ngram"]
            filter_condition: Qdrant filter condition
            use_reranking: Whether to use Llama 3 reranking (overrides init setting)
        
        Returns:
            List of ranked chunks with legal context
        """
        # Check if we're in an async context
        if asyncio:
            try:
                loop = asyncio.get_running_loop()
                # We're in async context, but called sync method - use sync implementation
                return self._retrieve_and_rerank_impl(query, top_k, methods, filter_condition, use_reranking, async_mode=False)
            except RuntimeError:
                # No running loop, can use async
                try:
                    return asyncio.run(self._retrieve_and_rerank_impl(query, top_k, methods, filter_condition, use_reranking, async_mode=True))
                except Exception:
                    pass
        
        return self._retrieve_and_rerank_impl(query, top_k, methods, filter_condition, use_reranking, async_mode=False)
    
    async def _retrieve_and_rerank_impl(
        self,
        query: str,
        top_k: int = 10,
        methods: Optional[List[str]] = None,
        filter_condition: Optional[Filter] = None,
        use_reranking: Optional[bool] = None,
        async_mode: bool = True
    ) -> List[RankedChunk]:
        """Internal implementation of retrieve and rerank (supports both sync and async)."""
        if methods is None:
            methods = ["semantic", "bm25", "ngram"]
        
        use_reranking = use_reranking if use_reranking is not None else self.enable_reranking
        
        # Check cache
        if self.cache is not None:
            cache_key = self._get_cache_key(query, top_k, methods)
            if cache_key in self.cache:
                logger.info("Returning cached results")
                return self.cache[cache_key]
        
        logger.info(f"Retrieving documents for query: {query[:100]}...")
        
        # Perform retrieval using multiple methods
        all_results = {}
        ranked_lists = []
        
        # Semantic search
        if "semantic" in methods:
            semantic_results = self.semantic_search(query, top_k * 2, filter_condition)
            ranked_lists.append([(doc_id, score) for doc_id, score, _ in semantic_results])
            for doc_id, score, payload in semantic_results:
                if doc_id not in all_results:
                    all_results[doc_id] = {
                        "content": payload.get("text", payload.get("content", "")),
                        "metadata": {k: v for k, v in payload.items() if k not in ["text", "content"]},
                        "scores": {"semantic": score}
                    }
        
        # BM25 search
        if "bm25" in methods:
            bm25_results = self.bm25_search(query, top_k * 2, filter_condition)
            ranked_lists.append(bm25_results)
            for doc_id, score in bm25_results:
                if doc_id not in all_results:
                    # Need to fetch content
                    result = self.qdrant_client.retrieve(
                        collection_name=self.collection_name,
                        ids=[doc_id],
                        with_payload=True
                    )
                    if result:
                        payload = result[0].payload or {}
                        all_results[doc_id] = {
                            "content": payload.get("text", payload.get("content", "")),
                            "metadata": {k: v for k, v in payload.items() if k not in ["text", "content"]},
                            "scores": {}
                        }
                if doc_id in all_results:
                    all_results[doc_id]["scores"]["bm25"] = score
        
        # N-gram search
        if "ngram" in methods:
            ngram_results = self.ngram_search(query, top_k * 2, filter_condition)
            ranked_lists.append(ngram_results)
            for doc_id, score in ngram_results:
                if doc_id not in all_results:
                    # Need to fetch content
                    result = self.qdrant_client.retrieve(
                        collection_name=self.collection_name,
                        ids=[doc_id],
                        with_payload=True
                    )
                    if result:
                        payload = result[0].payload or {}
                        all_results[doc_id] = {
                            "content": payload.get("text", payload.get("content", "")),
                            "metadata": {k: v for k, v in payload.items() if k not in ["text", "content"]},
                            "scores": {}
                        }
                if doc_id in all_results:
                    all_results[doc_id]["scores"]["ngram"] = score
        
        # Fuse results using weighted RRF
        if len(ranked_lists) > 1:
            # Default weights: semantic gets more weight, then BM25, then n-gram
            weights = []
            if "semantic" in methods:
                weights.append(0.4)
            if "bm25" in methods:
                weights.append(0.35)
            if "ngram" in methods:
                weights.append(0.25)
            
            # Normalize weights to match number of methods
            if len(weights) != len(ranked_lists):
                weights = [1.0 / len(ranked_lists)] * len(ranked_lists)
            
            fused_results = ReciprocalRankFusion.fuse(ranked_lists, weights=weights)
        else:
            fused_results = ranked_lists[0] if ranked_lists else []
        
        # Create RankedChunk objects
        chunks = []
        seen_content = set()
        
        for doc_id, rrf_score in fused_results[:top_k * 3]:  # Get more for deduplication
            if doc_id not in all_results:
                continue
            
            doc_data = all_results[doc_id]
            content = doc_data["content"]
            
            # Deduplicate by content
            content_hash = hashlib.md5(content.lower().strip().encode()).hexdigest()
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            
            # Determine retrieval method
            retrieval_method = "hybrid"
            if len(doc_data["scores"]) == 1:
                retrieval_method = list(doc_data["scores"].keys())[0]
            
            chunk = RankedChunk(
                id=doc_id,
                content=content,
                metadata=doc_data["metadata"],
                base_score=rrf_score,
                retrieval_method=retrieval_method
            )
            chunks.append(chunk)
            
            if len(chunks) >= top_k * 2:  # Get more for reranking
                break
        
        # Apply legal boosting
        chunks = self._apply_legal_boosting(chunks, query)
        
        # Sort by boosted score
        chunks.sort(key=lambda x: x.final_score, reverse=True)
        chunks = chunks[:top_k * 2]  # Limit for reranking
        
        # Rerank with Llama 3 if enabled
        if use_reranking and self.reranker:
            logger.info("Applying Llama 3 reranking...")
            try:
                if async_mode and asyncio:
                    chunks = await self.reranker.rerank_chunks_async(query, chunks, top_k=top_k)
                else:
                    chunks = self.reranker.rerank(query, chunks, top_k=top_k)
            except Exception as e:
                logger.warning(f"Reranking failed, using legal-boosted results: {e}")
                chunks = chunks[:top_k]
        else:
            # Just take top_k
            chunks = chunks[:top_k]
        
        # Final sort by final score (or rerank score if available)
        for chunk in chunks:
            if chunk.rerank_score > 0:
                chunk.final_score = chunk.rerank_score
            # Otherwise final_score already set by legal boosting
        
        chunks.sort(key=lambda x: x.final_score, reverse=True)
        chunks = chunks[:top_k]
        
        # Cache results
        if self.cache is not None:
            cache_key = self._get_cache_key(query, top_k, methods)
            self.cache[cache_key] = chunks
        
        logger.info(f"Retrieved and ranked {len(chunks)} chunks")
        return chunks


class LegalRAGConfig:
    """Configuration management for the legal RAG reranking system.
    
    Centralizes configuration via environment variables with sensible defaults.
    Makes the system more maintainable and easier to configure for different environments.
    """
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection_name = os.getenv("QDRANT_COLLECTION", "legal_documents")
        self.together_api_key = os.getenv("TOGETHER_API_KEY")
        self.together_model = os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3-70b-chat-hf")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.enable_reranking = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
        self.enable_caching = os.getenv("ENABLE_CACHING", "true").lower() == "true"
        
        # RRF weights (can be overridden via environment)
        self.rrf_weights = {
            "semantic": float(os.getenv("WEIGHT_SEMANTIC", "0.4")),
            "bm25": float(os.getenv("WEIGHT_BM25", "0.35")),
            "ngram": float(os.getenv("WEIGHT_NGRAM", "0.25"))
        }
        
        # Normalize weights
        total = sum(self.rrf_weights.values())
        if total > 0:
            self.rrf_weights = {k: v / total for k, v in self.rrf_weights.items()}
    
    def get_reranker_weights(self, methods: List[str]) -> List[float]:
        """Get weights for specified retrieval methods."""
        weights = []
        for method in methods:
            if method in self.rrf_weights:
                weights.append(self.rrf_weights[method])
        # Normalize if needed
        if weights and sum(weights) > 0:
            total = sum(weights)
            weights = [w / total for w in weights]
        return weights


class LegalRAGSystem:
    """Main system class for legal document RAG with advanced reranking.
    
    Provides a high-level interface that wraps all components and simplifies usage.
    This is the recommended entry point for production use.
    """
    
    def __init__(self, config: Optional[LegalRAGConfig] = None, qdrant_client: Optional[QdrantClient] = None):
        """Initialize the Legal RAG System.
        
        Args:
            config: Configuration object (creates default if None)
            qdrant_client: Qdrant client instance (creates default if None)
        """
        self.config = config or LegalRAGConfig()
        
        # Initialize Qdrant client if not provided
        if qdrant_client is None:
            qdrant_client = QdrantClient(url=self.config.qdrant_url)
        
        # Initialize reranker
        self.reranker = LegalRAGReranker(
            qdrant_client=qdrant_client,
            collection_name=self.config.collection_name,
            embedding_model=self.config.embedding_model,
            together_api_key=self.config.together_api_key,
            enable_reranking=self.config.enable_reranking,
            enable_caching=self.config.enable_caching
        )
        
        logger.info("LegalRAGSystem initialized successfully")
    
    async def query_async(
        self,
        query_text: str,
        top_k: int = 10,
        methods: Optional[List[str]] = None,
        filter_condition: Optional[Filter] = None,
        use_reranking: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Async query method for better performance in async contexts.
        
        Args:
            query_text: Search query
            top_k: Number of results
            methods: Retrieval methods to use
            filter_condition: Qdrant filter
            use_reranking: Override reranking setting
        
        Returns:
            List of ranked chunks as dictionaries
        """
        if methods is None:
            methods = ["semantic", "bm25", "ngram"]
        
        ranked_chunks = await self.reranker.retrieve_and_rerank_async(
            query=query_text,
            top_k=top_k,
            methods=methods,
            filter_condition=filter_condition,
            use_reranking=use_reranking
        )
        
        return [chunk.to_dict() for chunk in ranked_chunks]
    
    def query(
        self,
        query_text: str,
        top_k: int = 10,
        methods: Optional[List[str]] = None,
        filter_condition: Optional[Filter] = None,
        use_reranking: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Sync query method (main entry point).
        
        Args:
            query_text: Search query
            top_k: Number of results
            methods: Retrieval methods to use
            filter_condition: Qdrant filter
            use_reranking: Override reranking setting
        
        Returns:
            List of ranked chunks as dictionaries
        """
        if methods is None:
            methods = ["semantic", "bm25", "ngram"]
        
        ranked_chunks = self.reranker.retrieve_and_rerank(
            query=query_text,
            top_k=top_k,
            methods=methods,
            filter_condition=filter_condition,
            use_reranking=use_reranking
        )
        
        return [chunk.to_dict() for chunk in ranked_chunks]


# Example usage function
def example_usage():
    """Example demonstrating retrieval and reranking of contract clauses."""
    from qdrant_client import QdrantClient
    
    # Initialize system using configuration (recommended approach)
    config = LegalRAGConfig()
    qdrant_client = QdrantClient(url=config.qdrant_url)
    system = LegalRAGSystem(config=config, qdrant_client=qdrant_client)
    
    # Example query
    query = "What are the termination conditions for this contract?"
    
    # Query using system wrapper
    results = system.query(
        query_text=query,
        top_k=10,
        methods=["semantic", "bm25", "ngram"],
        use_reranking=True
    )
    
    # Display results
    print(f"\nQuery: {query}\n")
    print(f"Found {len(results)} relevant clauses:\n")
    
    for i, result in enumerate(results, 1):
        print(f"Rank {i}:")
        print(f"  Final Score: {result['scores']['final']:.4f}")
        print(f"  Base Score: {result['scores']['base']:.4f}")
        if result['scores']['rerank'] > 0:
            print(f"  Rerank Score: {result['scores']['rerank']:.4f}")
        print(f"  Clause Type: {result['legal_context']['clause_type']}")
        print(f"  Section: {result['legal_context']['section_id']}")
        print(f"  Is Obligation: {result['legal_context']['is_obligation']}")
        print(f"  Retrieval Method: {result.get('retrieval_method', 'hybrid')}")
        print(f"  Content: {result['content'][:200]}...")
        print()


async def example_usage_async():
    """Example demonstrating async usage for better performance."""
    from qdrant_client import QdrantClient
    
    config = LegalRAGConfig()
    qdrant_client = QdrantClient(url=config.qdrant_url)
    system = LegalRAGSystem(config=config, qdrant_client=qdrant_client)
    
    query = "What are the termination conditions for this contract?"
    results = await system.query_async(
        query_text=query, 
        top_k=10, 
        methods=["semantic", "bm25", "ngram"],
        use_reranking=True
    )
    
    print(f"\nQuery: {query}\n")
    print(f"Found {len(results)} relevant clauses:\n")
    
    for i, result in enumerate(results, 1):
        print(f"Rank {i}: Score={result['scores']['final']:.4f}, "
              f"Type={result['legal_context']['clause_type']}, "
              f"Section={result['legal_context']['section_id']}, "
              f"Obligation={result['legal_context']['is_obligation']}")


if __name__ == "__main__":
    example_usage()

