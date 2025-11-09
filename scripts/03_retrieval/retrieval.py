"""
Retrieval & Context Engine
Handles query processing, retrieval, and context ranking.
"""

import os
import re
from typing import List, Dict, Optional, Any, Callable
from pathlib import Path
from collections import Counter
import streamlit as st
import logging
import traceback
from datetime import datetime

# Import logging utility
try:
    from scripts.logging_config import setup_logger, write_to_log, get_current_log_file
except ImportError:
    # Fallback if logging_config not available
    # Use session state to get the same log file as streamlit_app.py
    try:
        import streamlit as st
        USE_STREAMLIT = True
    except ImportError:
        USE_STREAMLIT = False
    
    def get_current_log_file():
        """Get current log file from session state or create one."""
        if USE_STREAMLIT and 'log_file_path' in st.session_state:
            return st.session_state.log_file_path
        else:
            # Fallback: create timestamped file
            os.makedirs("logs", exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file = f"logs/logger_{timestamp}.log"
            if USE_STREAMLIT and 'log_file_path' not in st.session_state:
                st.session_state.log_file_path = log_file
            return log_file
    
    def setup_logger(name=None, level=logging.INFO):
        logger = logging.getLogger(name)
        if not logger.handlers:
            log_file = get_current_log_file()
            logging.basicConfig(
                level=level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file, encoding='utf-8', mode='a'),
                    logging.StreamHandler()
                ]
            )
        return logger
    
    def write_to_log(message: str, log_type: str = "INFO"):
        try:
            log_file = get_current_log_file()
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"{log_type} at {datetime.now().isoformat()}\n")
                f.write(f"{message}\n")
                f.write(f"{'='*80}\n\n")
        except:
            pass

logger = setup_logger(__name__)

# Import Qdrant client with fallback
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector, SearchParams
except ImportError:
    QdrantClient = None
    Filter = None
    FieldCondition = None
    MatchValue = None
    FilterSelector = None
    SearchParams = None

# Import OllamaEmbeddings for type checking
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    OllamaEmbeddings = None

# LangChain imports - handle different versions (Ollama preferred, OpenAI fallback)
Embeddings = None
DEFAULT_EMBEDDING_MODEL = None
EMBEDDING_DIMENSION = 1536

try:
    from langchain_ollama import OllamaEmbeddings
    Embeddings = OllamaEmbeddings
    DEFAULT_EMBEDDING_MODEL = "llama3.1:latest"  # Use latest tag to match installed model
    EMBEDDING_DIMENSION = 4096  # Llama 3.1 embedding dimension
except ImportError:
    OllamaEmbeddings = None
    try:
        from langchain_openai import OpenAIEmbeddings
        Embeddings = OpenAIEmbeddings
        DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"
        EMBEDDING_DIMENSION = 1536
    except ImportError:
        try:
            from langchain_community.embeddings import OpenAIEmbeddings
            Embeddings = OpenAIEmbeddings
            DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"
            EMBEDDING_DIMENSION = 1536
        except ImportError:
            st.warning("No embedding provider found. Please install langchain-ollama (recommended) or langchain-openai")
            Embeddings = None


class RAGRetrievalEngine:
    """
    Retrieval engine for RAG queries.
    
    Design Decisions:
    - Supports multiple retrieval strategies (summarization, comparison, synthesis)
    - Implements confidence thresholds for quality control
    - Returns comprehensive citation metadata
    - Handles graceful fallbacks when no relevant chunks found
    - Supports metadata filtering for multi-tenancy and cohort filtering
    """
    
    def __init__(self, project_id: str = "default", use_ollama: bool = True, vector_db_type: str = "Qdrant",
                 embedding_model: str = "llama3.1", multimodal_embeddings=None):
        self.project_id = project_id
        self.use_ollama = use_ollama
        self.vector_db_type = vector_db_type  # "Qdrant" (default), "ChromaDB", or "Pinecone"
        self.embedding_model = embedding_model
        self.multimodal_embeddings = multimodal_embeddings
        
        # CRITICAL: Initialize vector database clients FIRST to ensure they always exist
        self.qdrant_client = None
        self.chroma_client = None
        self.chroma_collection = None
        self.pinecone_index = None
        
        # Initialize embeddings FIRST (needed for retrieval)
        if self.multimodal_embeddings is not None:
            # Use multimodal embeddings
            self.embeddings = self.multimodal_embeddings
            self.embedding_dimension = self.multimodal_embeddings.get_dimension()
        elif embedding_model not in ["llama3.1", "openai-ada-002"]:
            # Try to load multimodal model
            try:
                from scripts.multimodal_embeddings import MultimodalEmbeddings
                self.multimodal_embeddings = MultimodalEmbeddings(
                    model_name=embedding_model,
                    use_ollama=use_ollama
                )
                self.embeddings = self.multimodal_embeddings
                self.embedding_dimension = self.multimodal_embeddings.get_dimension()
            except Exception as e:
                logger.warning(f"Failed to load multimodal model {embedding_model}: {e}, falling back to text-only")
                self._init_embeddings(use_ollama)
        else:
            # Text-only embeddings
            self._init_embeddings(use_ollama)
        
        # Initialize vector database based on user choice
        if self.vector_db_type == "Qdrant":
            self._init_qdrant()
        elif self.vector_db_type == "Pinecone":
            self._init_pinecone()
        else:  # Fallback to ChromaDB if Qdrant/Pinecone unavailable
            self._init_chroma_fallback()
    
    def get_processed_files(self) -> List[Dict[str, Any]]:
        """Get list of all processed files from the vector database."""
        processed_files = []
        
        try:
            if self.vector_db_type == "Qdrant" and self.qdrant_client:
                collection_name = f"project_{self.project_id}"
                # Check if collection exists
                collections = self.qdrant_client.get_collections().collections
                collection_names = [c.name for c in collections]
                
                if collection_name not in collection_names:
                    return []
                
                # Get all unique file names from the collection
                file_names = set()
                # Scroll through all points to get unique file names
                scroll_result = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=10000,  # Adjust if you have more points
                    with_payload=True
                )
                
                for point in scroll_result[0]:
                    file_name = point.payload.get("file_name")
                    if file_name:
                        file_names.add(file_name)
                
                # Count chunks per file
                for file_name in file_names:
                    from qdrant_client.models import Filter, FieldCondition, MatchValue
                    if Filter and FieldCondition and MatchValue:
                        file_filter = Filter(
                            must=[
                                FieldCondition(
                                    key="file_name",
                                    match=MatchValue(value=file_name)
                                )
                            ]
                        )
                        count_result = self.qdrant_client.scroll(
                            collection_name=collection_name,
                            scroll_filter=file_filter,
                            limit=10000
                        )
                        chunk_count = len(count_result[0])
                        
                        # Get file type from first chunk
                        file_type = "unknown"
                        if count_result[0]:
                            source_type = count_result[0][0].payload.get("source_type", "document")
                            if source_type == "transcript":
                                file_type = "json"
                            elif source_type == "video_subtitle":
                                file_type = "vtt"
                            else:
                                file_type = Path(file_name).suffix[1:].lower() or "txt"
                        
                        processed_files.append({
                            "name": file_name,
                            "type": file_type,
                            "chunks": chunk_count,
                            "status": "processed"
                        })
                        
            elif hasattr(self, 'chroma_collection') and self.chroma_collection:
                # Get all unique file names from ChromaDB
                try:
                    all_data = self.chroma_collection.get()
                    file_names = set()
                    if all_data.get('metadatas'):
                        for metadata in all_data['metadatas']:
                            file_name = metadata.get('file_name')
                            if file_name:
                                file_names.add(file_name)
                    
                    # Count chunks per file
                    for file_name in file_names:
                        results = self.chroma_collection.get(
                            where={"file_name": file_name}
                        )
                        chunk_count = len(results.get('ids', []))
                        file_type = Path(file_name).suffix[1:].lower() or "txt"
                        
                        processed_files.append({
                            "name": file_name,
                            "type": file_type,
                            "chunks": chunk_count,
                            "status": "processed"
                        })
                except Exception as e:
                    logger.warning(f"Error getting processed files from ChromaDB: {e}")
                    
        except Exception as e:
            logger.error(f"Error getting processed files: {e}")
        
        return processed_files
    
    def get_chunks_for_file(self, file_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all chunks for a specific file."""
        chunks = []
        
        try:
            if self.vector_db_type == "Qdrant" and self.qdrant_client:
                collection_name = f"project_{self.project_id}"
                # Check if collection exists
                collections = self.qdrant_client.get_collections().collections
                collection_names = [c.name for c in collections]
                
                if collection_name not in collection_names:
                    return []
                
                # Filter by file name
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                if Filter and FieldCondition and MatchValue:
                    file_filter = Filter(
                        must=[
                            FieldCondition(
                                key="file_name",
                                match=MatchValue(value=file_name)
                            )
                        ]
                    )
                    # Scroll through all chunks for this file
                    scroll_result = self.qdrant_client.scroll(
                        collection_name=collection_name,
                        scroll_filter=file_filter,
                        limit=limit,
                        with_payload=True,
                        with_vectors=False
                    )
                    
                    for point in scroll_result[0]:
                        chunks.append({
                            "id": str(point.id),
                            "text": point.payload.get("text", ""),
                            "chunk_index": point.payload.get("chunk_index", -1),
                            "metadata": {k: v for k, v in point.payload.items() if k != "text"}
                        })
                    
                    # Sort by chunk_index
                    chunks.sort(key=lambda x: x.get("chunk_index", 0))
                    
            elif hasattr(self, 'chroma_collection') and self.chroma_collection:
                # Get chunks from ChromaDB
                try:
                    results = self.chroma_collection.get(
                        where={"file_name": file_name},
                        limit=limit
                    )
                    
                    if results.get('ids'):
                        for i, doc_id in enumerate(results['ids']):
                            chunks.append({
                                "id": doc_id,
                                "text": results['documents'][i] if results.get('documents') else "",
                                "chunk_index": results['metadatas'][i].get('chunk_index', i) if results.get('metadatas') else i,
                                "metadata": results['metadatas'][i] if results.get('metadatas') else {}
                            })
                    
                    # Sort by chunk_index
                    chunks.sort(key=lambda x: x.get("chunk_index", 0))
                except Exception as e:
                    logger.warning(f"Error getting chunks from ChromaDB: {e}")
                    
        except Exception as e:
            logger.error(f"Error getting chunks for file {file_name}: {e}")
        
        return chunks
    
    def reset_database(self) -> bool:
        """Reset/clear the vector database for the current project."""
        try:
            if self.vector_db_type == "Qdrant" and self.qdrant_client:
                collection_name = f"project_{self.project_id}"
                # Delete the collection
                collections = self.qdrant_client.get_collections().collections
                collection_names = [c.name for c in collections]
                
                if collection_name in collection_names:
                    self.qdrant_client.delete_collection(collection_name)
                    logger.info(f"Deleted Qdrant collection: {collection_name}")
                    return True
                else:
                    logger.info(f"Collection {collection_name} does not exist")
                    return True
                    
            elif self.vector_db_type == "Pinecone" and self.pinecone_index:
                # Delete all vectors for this project
                # Note: Pinecone doesn't have a simple delete_all, so we'd need to query and delete
                # For now, return True (can be enhanced)
                logger.warning("Pinecone reset not fully implemented - vectors remain")
                return True
                
            elif hasattr(self, 'chroma_collection') and self.chroma_collection:
                # Delete collection in ChromaDB
                try:
                    self.chroma_client.delete_collection(f"project_{self.project_id}")
                    logger.info(f"Deleted ChromaDB collection: project_{self.project_id}")
                    # Recreate empty collection
                    self.chroma_collection = self.chroma_client.get_or_create_collection(
                        name=f"project_{self.project_id}"
                    )
                    return True
                except Exception as e:
                    logger.error(f"Error resetting ChromaDB: {e}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error resetting database: {e}")
            return False
        
        return False
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query to improve definition and concept matching."""
        query_lower = query.lower().strip()
        expanded = [query]  # Always include original query
        
        # Handle slash-separated terms (e.g., "outlier/novelty detection")
        query_normalized = query_lower.replace("/", " ").replace("-", " ")
        
        # Extract key terms (remove "what is", "define", etc.)
        key_terms = []
        question_words = ["what", "is", "are", "the", "a", "an", "define", "definition", "explain", "meaning", "of"]
        words = query_normalized.split()
        for word in words:
            clean_word = word.strip('.,!?;:()[]{}')
            if clean_word not in question_words and len(clean_word) > 2:
                key_terms.append(clean_word)
        
        if key_terms:
            main_term = " ".join(key_terms)
            
            # Add definition-style variations
            expanded.append(f"{main_term} is")
            expanded.append(f"{main_term} is a")
            expanded.append(f"{main_term} is an")
            expanded.append(f"{main_term} refers to")
            expanded.append(f"{main_term} definition")
            expanded.append(f"definition of {main_term}")
            expanded.append(f"{main_term} problem")
            expanded.append(f"{main_term} arises")
            
            # Add the term itself if it's a phrase
            if len(key_terms) > 1:
                expanded.append(main_term)
            
            # Handle slash variations specifically
            if "/" in query_lower:
                # Extract terms around slash
                parts = query_lower.split("/")
                if len(parts) == 2:
                    term1 = parts[0].strip().split()[-1] if parts[0].strip().split() else ""
                    term2 = parts[1].strip().split()[0] if parts[1].strip().split() else ""
                    if term1 and term2:
                        # Add both individual terms
                        expanded.append(f"{term1} detection")
                        expanded.append(f"{term2} detection")
                        expanded.append(f"{term1} {term2} detection")
                        expanded.append(f"{term1} {term2} detection is")
            
            # Add with "novelty" or "outlier" variations if present
            if "outlier" in query_lower or "outlier" in main_term:
                expanded.append("outlier detection")
                expanded.append("outlier detection is")
                expanded.append("outlier detection is a problem")
            if "novelty" in query_lower or "novelty" in main_term:
                expanded.append("novelty detection")
                expanded.append("novelty detection is")
                expanded.append("novelty detection is a problem")
            if "outlier" in query_lower and "novelty" in query_lower:
                expanded.append("outlier novelty detection")
                expanded.append("outlier novelty detection is")
                expanded.append("outlier novelty detection is a problem")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_expanded = []
        for q in expanded:
            q_normalized = q.lower().strip()
            if q_normalized not in seen and len(q_normalized) > 3:
                seen.add(q_normalized)
                unique_expanded.append(q)
        
        return unique_expanded[:8]  # Limit to 8 expanded queries
    
    def _rerank_results(self, results: List[Dict], original_query: str, expanded_queries: List[str]) -> List[Dict]:
        """Rerank results using keyword matching and semantic scores."""
        if not results:
            return []
        
        # Extract key terms from queries
        all_queries = [original_query] + expanded_queries
        key_terms = set()
        question_words = {"what", "is", "are", "the", "a", "an", "define", "definition", "explain", "meaning", "of"}
        
        for q in all_queries:
            words = q.lower().split()
            for word in words:
                clean_word = word.strip('.,!?;:()[]{}')
                if len(clean_word) > 2 and clean_word not in question_words:
                    key_terms.add(clean_word)
        
        # Score and rerank results
        scored_results = []
        for result in results:
            text = result.get("text", "").lower()
            score = result.get("score", 0)
            
            # Keyword boost: count how many key terms appear in the text
            keyword_matches = sum(1 for term in key_terms if term in text)
            keyword_boost = min(keyword_matches * 0.1, 0.3)  # Max 0.3 boost
            
            # Definition boost: boost chunks that look like definitions
            definition_indicators = [
                " is a ", " is an ", " refers to ", " means ", " defined as ",
                " definition ", " problem that ", " arises in ", " applications "
            ]
            definition_boost = 0.0
            for indicator in definition_indicators:
                if indicator in text:
                    definition_boost = 0.2
                    break
            
            # Exact phrase boost: boost chunks that contain the exact phrase or term
            exact_phrase_boost = 0.0
            main_phrases = []
            
            # Extract main phrases from queries
            for q in all_queries:
                q_words = [w for w in q.lower().split() if w not in question_words and len(w) > 2]
                if len(q_words) >= 2:
                    # Try different phrase lengths
                    main_phrases.append(" ".join(q_words[:2]))  # 2-word phrase
                    main_phrases.append(" ".join(q_words[:3]))  # 3-word phrase
                    if len(q_words) >= 4:
                        main_phrases.append(" ".join(q_words[:4]))  # 4-word phrase
            
            # Check for exact phrase matches (highest boost)
            for phrase in main_phrases:
                if phrase in text:
                    exact_phrase_boost = max(exact_phrase_boost, 0.3)  # Strong boost for exact match
                    break
            
            # Also boost if the phrase appears near definition indicators
            if exact_phrase_boost > 0:
                # Check if phrase appears near "is" or "arises"
                phrase_pos = text.find(phrase) if phrase in text else -1
                if phrase_pos >= 0:
                    context = text[max(0, phrase_pos-50):min(len(text), phrase_pos+len(phrase)+50)]
                    if any(ind in context for ind in [" is ", " is a ", " is an ", " arises ", " problem "]):
                        exact_phrase_boost = 0.4  # Even stronger boost
            
            # Combine scores
            final_score = score + keyword_boost + definition_boost + exact_phrase_boost
            scored_results.append({
                **result,
                "score": min(final_score, 1.0),  # Cap at 1.0
                "original_score": score,
                "keyword_matches": keyword_matches
            })
        
        # Sort by final score and remove duplicates (by text content)
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Deduplicate by text similarity
        seen_texts = set()
        deduplicated = []
        for result in scored_results:
            text_key = result.get("text", "")[:100].lower()  # Use first 100 chars as key
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                deduplicated.append(result)
        
        return deduplicated
    
    def _init_embeddings(self, use_ollama: bool):
        """Initialize embeddings - Ollama only, no OpenAI fallback."""
        if use_ollama and OllamaEmbeddings is not None:
            try:
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                logger.info(f"Attempting to initialize Ollama embeddings with model {DEFAULT_EMBEDDING_MODEL} at {base_url}")
                self.embeddings = OllamaEmbeddings(
                    model=DEFAULT_EMBEDDING_MODEL,
                    base_url=base_url
                )
                # Test the connection by trying to embed a small test string
                # If test fails, log warning but don't fail - might be temporary
                try:
                    _ = self.embeddings.embed_query("test")
                    logger.info(f"Successfully initialized Ollama with {DEFAULT_EMBEDDING_MODEL}")
                    # Show success message only once using session state
                    try:
                        if 'ollama_connected_message_shown' not in st.session_state:
                            st.info(f"✅ Using Ollama with {DEFAULT_EMBEDDING_MODEL}")
                            st.session_state.ollama_connected_message_shown = True
                    except (AttributeError, NameError):
                        # st might not be available in all contexts
                        pass
                except Exception as test_error:
                    # If test fails, log but don't fail completely - might be a temporary issue
                    logger.warning(f"Ollama connection test failed: {test_error}. Will try again on first use.")
                    # Still mark as initialized - the actual embedding will fail if Ollama is really down
                    try:
                        if 'ollama_connected_message_shown' not in st.session_state:
                            st.info(f"✅ Using Ollama with {DEFAULT_EMBEDDING_MODEL} (connection will be tested on first use)")
                            st.session_state.ollama_connected_message_shown = True
                    except (AttributeError, NameError):
                        # st might not be available in all contexts
                        pass
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Ollama initialization failed: {error_msg}")
                if "Connection" in error_msg or "refused" in error_msg.lower() or "not responding" in error_msg.lower():
                    error_message = (
                        f"⚠️ Ollama server not accessible at {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}. "
                        f"Please ensure Ollama is running. You can start it by running 'ollama serve' in your terminal."
                    )
                    st.error(error_message)
                    raise Exception(error_message)
                else:
                    error_message = f"⚠️ Failed to initialize Ollama embeddings: {error_msg}. Please ensure Ollama is installed and running."
                    st.error(error_message)
                    raise Exception(error_message)
        else:
            if OllamaEmbeddings is None:
                raise Exception(
                    "Ollama embeddings not available. Please install langchain-ollama: pip install langchain-ollama"
                )
            else:
                raise Exception(
                    "Ollama is required but not enabled. Please set use_ollama=True or ensure Ollama is properly configured."
                )
    
    def _init_openai_embeddings(self):
        """Initialize OpenAI embeddings as fallback - DEPRECATED, not used anymore."""
        # This method is kept for backward compatibility but should not be called
        # We only use Ollama now
        raise Exception(
            "OpenAI embeddings are not supported. Please use Ollama instead. "
            "Ensure Ollama is installed and running: pip install langchain-ollama && ollama serve"
        )
    
    def _init_qdrant(self):
        """Initialize Qdrant client with connection health check."""
        if QdrantClient is not None:
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
            
            try:
                # Import health check utility (use importlib for modules starting with numbers)
                import importlib
                qdrant_health_module = importlib.import_module('scripts.08_utilities.qdrant_health')
                check_qdrant_connection = qdrant_health_module.check_qdrant_connection
                verify_qdrant_before_operation = qdrant_health_module.verify_qdrant_before_operation
                
                logger.debug(f"Checking Qdrant connection at {qdrant_url}")
                is_connected, error_msg, info = check_qdrant_connection(
                    qdrant_url=qdrant_url,
                    qdrant_api_key=qdrant_api_key,
                    timeout=5.0
                )
                
                if not is_connected:
                    logger.error(f"Qdrant connection check failed: {error_msg}")
                    st.warning(f"⚠️ {error_msg}")
                    self.qdrant_client = None
                    self._init_chroma_fallback()
                    return
                
                # Connection check passed, create client
                logger.debug(f"Connecting to Qdrant at {qdrant_url}")
                self.qdrant_client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                    timeout=30
                )
                
                # Verify connection with actual operation
                verify_qdrant_before_operation(self.qdrant_client, "Qdrant initialization")
                
                # Test connection
                self.qdrant_client.get_collections()
                logger.info("Successfully connected to Qdrant")
                if info:
                    logger.info(f"Qdrant info: {info.get('collections_count', 0)} collections, "
                              f"response time: {info.get('response_time_ms', 'unknown')}ms")
                
                # Show success message only once using session state
                if 'qdrant_connected_message_shown' not in st.session_state:
                    st.success(f"✅ Connected to Qdrant ({info.get('url', qdrant_url) if info else qdrant_url})")
                    st.session_state.qdrant_connected_message_shown = True
            except Exception as e:
                st.warning(f"Qdrant connection failed: {e}. Falling back to ChromaDB.")
                st.info("💡 **To start Qdrant:** Run `docker run -p 6333:6333 qdrant/qdrant` or use `start_qdrant.bat`")
                self.qdrant_client = None
                self._init_chroma_fallback()
        else:
            st.warning("Qdrant not installed. Falling back to ChromaDB.")
            self._init_chroma_fallback()
    
    def _init_pinecone(self):
        """Initialize Pinecone client."""
        try:
            import pinecone
            from pinecone import Pinecone
            
            api_key = os.getenv("PINECONE_API_KEY", "")
            if not api_key:
                st.error("Pinecone API key not found. Please set PINECONE_API_KEY or use the sidebar.")
                self._init_chroma_fallback()
                return
            
            # Initialize Pinecone
            pc = Pinecone(api_key=api_key)
            
            # Get index
            index_name = f"deep-intellect-rag-{self.project_id}".lower().replace("_", "-")
            
            # Check if index exists
            existing_indexes = [idx.name for idx in pc.list_indexes()]
            
            if index_name in existing_indexes:
                self.pinecone_index = pc.Index(index_name)
                st.success(f"✅ Connected to Pinecone index: {index_name}")
            else:
                st.warning(f"Pinecone index '{index_name}' not found. Create it first or upload documents.")
                self._init_chroma_fallback()
            
        except ImportError:
            st.error("Pinecone not installed. Install with: pip install pinecone-client")
            self._init_chroma_fallback()
        except Exception as e:
            st.error(f"Pinecone initialization failed: {e}. Falling back to ChromaDB.")
            self._init_chroma_fallback()
    
    def _init_chroma_fallback(self):
        """Fallback to ChromaDB if Qdrant unavailable."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./chroma_db"
            ))
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name=f"project_{self.project_id}"
            )
        except Exception as e:
            st.error(f"ChromaDB fallback also failed: {e}")
            self.chroma_client = None
    
    def retrieve_hierarchical(
        self,
        query: str,
        query_type: str = "general",
        top_k: int = 5,
        top_documents: int = 5,
        confidence_threshold: float = 0.3,
        filters: Optional[Dict] = None,
        project_id: Optional[str] = None,
        search_type: str = "semantic",
        extract_query_filters: bool = True
    ) -> Dict[str, Any]:
        """
        Hierarchical retrieval: First find top documents, then top chunks from those documents.
        
        This approach:
        1. Retrieves chunks and groups by document/file
        2. Ranks documents by their best chunk scores
        3. Returns top chunks from top-ranked documents
        
        Args:
            query: User query text
            query_type: One of 'summarization', 'comparison', 'synthesis', 'general'
            top_k: Number of chunks to return
            top_documents: Number of top documents to consider
            confidence_threshold: Minimum similarity score (0-1)
            filters: Metadata filters
            project_id: Project ID for multi-tenancy
            
        Returns:
            Dict with chunks, metadata, and status
        """
        project_id = project_id or self.project_id
        
        try:
            logger.info(f"Hierarchical retrieval: query='{query[:100]}...', top_k={top_k}, top_documents={top_documents}")
            
            # Extract page/section/subsection filters from query if enabled
            if extract_query_filters:
                try:
                    from scripts.query_parser import extract_query_filters as parse_filters
                    query_filters = parse_filters(query, use_ollama=self.use_ollama)
                    # Merge with provided filters (query filters take precedence)
                    if query_filters:
                        if filters is None:
                            filters = {}
                        filters.update(query_filters)
                        logger.info(f"Extracted query filters: {query_filters}")
                except Exception as e:
                    logger.warning(f"Failed to extract query filters: {e}")
            
            # Check if embeddings are available
            if not hasattr(self, 'embeddings') or self.embeddings is None:
                error_msg = "Embeddings not initialized"
                logger.error(error_msg)
                return {
                    "chunks": [],
                    "status": "error",
                    "error": error_msg,
                    "message": "Please ensure Ollama is running or OpenAI API key is set."
                }
            
            # Expand query for better definition matching
            expanded_queries = self._expand_query(query)
            logger.debug(f"Expanded queries: {expanded_queries}")
            
            # Generate query embeddings
            query_embeddings = []
            for expanded_query in expanded_queries:
                try:
                    # Handle both multimodal and text-only embeddings
                    if hasattr(self.embeddings, 'embed_text'):
                        embedding = self.embeddings.embed_text(expanded_query)
                    elif hasattr(self.embeddings, 'embed_query'):
                        embedding = self.embeddings.embed_query(expanded_query)
                    else:
                        raise ValueError("Embeddings object does not have embed_query or embed_text method")
                    query_embeddings.append(embedding)
                except Exception as e:
                    logger.warning(f"Failed to embed query '{expanded_query}': {e}")
            
            # Use primary query embedding (first one)
            if query_embeddings and len(query_embeddings) > 0:
                query_embedding = query_embeddings[0]
            else:
                if hasattr(self.embeddings, 'embed_text'):
                    query_embedding = self.embeddings.embed_text(query)
                else:
                    query_embedding = self.embeddings.embed_query(query)
            
            # Increase initial retrieval to get more candidates for document ranking
            initial_top_k = max(top_k * 3, 20)  # Get 3x more chunks for document ranking
            
            # Build metadata filter
            search_filter = self._build_filter(filters, project_id)
            
            # Step 1: Retrieve candidate chunks (more than needed)
            all_results = []
            
            if self.vector_db_type == "Qdrant" and self.qdrant_client is not None:
                primary_results = self._search_qdrant(
                    query_embedding,
                    initial_top_k,
                    search_filter,
                    collection_name=f"project_{project_id}",
                    progress_callback=None  # Progress handled at higher level
                )
                all_results.extend(primary_results)
                
                # Also search with expanded queries
                for expanded_embedding in query_embeddings[1:]:
                    expanded_results = self._search_qdrant(
                        expanded_embedding,
                        top_k,
                        search_filter,
                        collection_name=f"project_{project_id}",
                        progress_callback=None
                    )
                    all_results.extend(expanded_results)
                    
            elif self.vector_db_type == "Pinecone" and self.pinecone_index is not None:
                primary_results = self._search_pinecone(
                    query_embedding,
                    initial_top_k,
                    filters,
                    project_id
                )
                all_results.extend(primary_results)
                
            elif hasattr(self, 'chroma_collection') and self.chroma_collection is not None:
                primary_results = self._search_chroma(
                    query_embedding,
                    initial_top_k,
                    filters
                )
                all_results.extend(primary_results)
            else:
                return {
                    "chunks": [],
                    "status": "error",
                    "error": "No vector database available",
                    "message": "Please ensure Qdrant, ChromaDB, or Pinecone is configured."
                }
            
            # Step 2: Rerank results
            results = self._rerank_results(all_results, query, expanded_queries)
            
            # Step 3: Group chunks by document/file and rank documents
            document_scores = {}  # file_name -> best_score
            document_chunks = {}  # file_name -> list of chunks
            
            for result in results:
                file_name = result.get("metadata", {}).get("file_name", "unknown")
                score = result.get("score", 0)
                
                if file_name not in document_chunks:
                    document_chunks[file_name] = []
                    document_scores[file_name] = 0
                
                document_chunks[file_name].append(result)
                # Track best score per document
                if score > document_scores[file_name]:
                    document_scores[file_name] = score
            
            # Step 4: Rank documents by their best chunk score
            ranked_documents = sorted(
                document_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_documents]
            
            logger.info(f"Top {len(ranked_documents)} documents: {[d[0] for d in ranked_documents]}")
            
            # Step 5: Select top chunks from top documents
            selected_chunks = []
            for file_name, doc_score in ranked_documents:
                chunks = document_chunks[file_name]
                # Sort chunks within document by score
                chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
                # Take top chunks from this document
                chunks_per_doc = max(1, top_k // top_documents)  # Distribute chunks across documents
                selected_chunks.extend(chunks[:chunks_per_doc])
            
            # Sort all selected chunks by score and take top_k
            selected_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
            final_chunks = selected_chunks[:top_k]
            
            # Filter by confidence threshold
            filtered_results = [
                r for r in final_chunks
                if r.get('score', 0) >= confidence_threshold
            ]
            
            # Fallback if no results above threshold
            if not filtered_results and final_chunks:
                logger.warning(f"No results above threshold {confidence_threshold}, returning top {min(3, len(final_chunks))} results")
                filtered_results = final_chunks[:3]
                for r in filtered_results:
                    r['low_confidence'] = True
            
            # Format response
            if not filtered_results:
                max_score = max([r.get('score', 0) for r in final_chunks]) if final_chunks else 0
                return {
                    "chunks": [],
                    "total_results": len(results),
                    "document_count": len(document_scores),
                    "filtered_count": 0,
                    "status": "no_results",
                    "max_score": max_score,
                    "message": f"No results found. Max score was {max_score:.3f}.",
                    "query_time_ms": 0
                }
            
            # Format chunks with citations
            formatted_chunks = []
            for result in filtered_results:
                chunk = {
                    "id": result.get("id", "unknown"),
                    "text": result.get("text", ""),
                    "score": result.get("score", 0),
                    "metadata": result.get("metadata", {})
                }
                formatted_chunks.append(chunk)
            
            return {
                "chunks": formatted_chunks,
                "total_results": len(results),
                "document_count": len(document_scores),
                "top_documents": [d[0] for d in ranked_documents],
                "filtered_count": len(filtered_results),
                "status": "success",
                "confidence_threshold": confidence_threshold,
                "hierarchical": True,
                "query_time_ms": 200
            }
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            error_msg = str(e)
            logger.error(f"Exception in hierarchical retrieve: {error_msg}")
            logger.error(f"Traceback:\n{error_traceback}")
            
            error_log_msg = f"""HIERARCHICAL RETRIEVAL ERROR
Query: {query}
Error: {error_msg}
Traceback:
{error_traceback}"""
            write_to_log(error_log_msg, "HIERARCHICAL RETRIEVAL ERROR")
            
            return {
                "chunks": [],
                "status": "error",
                "error": error_msg,
                "message": f"Hierarchical retrieval failed: {error_msg}"
            }
    
    def retrieve(
        self,
        query: str,
        query_type: str = "general",
        top_k: int = 5,
        confidence_threshold: float = 0.3,  # Lowered from 0.7 to be more lenient
        filters: Optional[Dict] = None,
        project_id: Optional[str] = None,
        use_hierarchical: bool = False,
        search_type: str = "semantic",  # "semantic", "lexical", or "hybrid"
        extract_query_filters: bool = True,  # Extract page/section filters from query
        progress_callback: Optional[Callable[[str, float], None]] = None  # Progress callback for UI updates
    ) -> Dict[str, Any]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query text
            query_type: One of 'summarization', 'comparison', 'synthesis', 'general'
            top_k: Number of chunks to retrieve
            confidence_threshold: Minimum similarity score (0-1)
            filters: Metadata filters (cohort_id, source_type, speaker_role, etc.)
            project_id: Project ID for multi-tenancy
            use_hierarchical: If True, use hierarchical search (document-level then chunk-level)
            search_type: "semantic" (default), "lexical", or "hybrid"
            
        Returns:
            Dict with chunks, metadata, and status
        """
        # Use hierarchical search if requested (currently only supports semantic)
        if use_hierarchical and search_type == "semantic":
            return self.retrieve_hierarchical(
                query=query,
                query_type=query_type,
                top_k=top_k,
                confidence_threshold=confidence_threshold,
                filters=filters,
                project_id=project_id
            )
        
        project_id = project_id or self.project_id
        
        try:
            logger.info(f"Retrieval request: query='{query[:100]}...', query_type={query_type}, top_k={top_k}, search_type={search_type}, project_id={project_id}")
            
            # For lexical search, embeddings are not needed
            query_embedding = None
            expanded_queries = []
            
            if search_type in ["semantic", "hybrid"]:
                # Check if embeddings are available
                if not hasattr(self, 'embeddings') or self.embeddings is None:
                    error_msg = "Embeddings not initialized"
                    logger.error(error_msg)
                    return {
                        "chunks": [],
                        "status": "error",
                        "error": error_msg,
                        "message": "Please ensure Ollama is running or OpenAI API key is set."
                    }
                
                # Expand query for better definition matching
                expanded_queries = self._expand_query(query)
                logger.debug(f"Expanded queries: {expanded_queries}")
                
                # Generate query embeddings for all expanded queries
                logger.debug("Generating query embeddings...")
                query_embeddings = []
                for expanded_query in expanded_queries:
                    try:
                        # Handle both multimodal and text-only embeddings
                        if hasattr(self.embeddings, 'embed_text'):
                            embedding = self.embeddings.embed_text(expanded_query)
                        elif hasattr(self.embeddings, 'embed_query'):
                            embedding = self.embeddings.embed_query(expanded_query)
                        else:
                            raise ValueError("Embeddings object does not have embed_query or embed_text method")
                        query_embeddings.append(embedding)
                    except Exception as e:
                        logger.warning(f"Failed to embed query '{expanded_query}': {e}")
                
                # Use primary query embedding (first one)
                if query_embeddings and len(query_embeddings) > 0:
                    query_embedding = query_embeddings[0]
                else:
                    if hasattr(self.embeddings, 'embed_text'):
                        query_embedding = self.embeddings.embed_text(query)
                    else:
                        query_embedding = self.embeddings.embed_query(query)
                logger.debug(f"Query embedding dimension: {len(query_embedding)}")
            
            # Adjust top_k based on query type
            if query_type == "summarization":
                top_k = max(top_k, 10)  # More chunks for summarization
            elif query_type == "comparison":
                top_k = max(top_k, 8)
            
            # Build metadata filter
            search_filter = self._build_filter(filters, project_id)
            
            # Perform search based on search_type
            all_results = []
            
            if search_type == "lexical":
                # Pure lexical search (no embeddings needed)
                all_results = self._lexical_search(
                    query,
                    top_k * 2,
                    search_filter,
                    project_id
                )
                
            elif search_type == "hybrid":
                # Hybrid search (semantic + lexical) with RRF
                hybrid_result = self._hybrid_search(
                    query,
                    query_embedding,
                    top_k * 2,
                    search_filter,
                    project_id,
                    progress_callback=progress_callback,
                    generate_answer=False  # Set to True if you want LLM-generated answers
                )
                # Extract chunks from hybrid result
                all_results = hybrid_result.get("chunks", [])
                
            else:  # semantic (default)
                # Semantic search with vector database
                if self.vector_db_type == "Qdrant" and self.qdrant_client is not None:
                    logger.info(f"Performing Qdrant semantic search for project '{project_id}'")
                    try:
                        primary_results = self._search_qdrant(
                            query_embedding,
                            top_k * 2,  # Get more results for reranking
                            search_filter,
                            collection_name=f"project_{project_id}",
                            progress_callback=progress_callback
                        )
                        logger.info(f"Qdrant search returned {len(primary_results)} primary results")
                        all_results.extend(primary_results)
                    except Exception as e:
                        logger.error(f"Qdrant search failed: {e}")
                        logger.error(f"Traceback:\n{traceback.format_exc()}")
                        # Continue with empty results rather than failing completely
                    
                    # Also search with expanded queries if available (only for semantic/hybrid)
                    if query_embeddings is not None and len(query_embeddings) > 1:
                        for expanded_embedding in query_embeddings[1:]:
                            expanded_results = self._search_qdrant(
                                expanded_embedding,
                                top_k,
                                search_filter,
                                collection_name=f"project_{project_id}",
                                progress_callback=progress_callback
                            )
                            all_results.extend(expanded_results)
                        
                elif self.vector_db_type == "Pinecone" and self.pinecone_index is not None:
                    primary_results = self._search_pinecone(
                        query_embedding,
                        top_k * 2,
                        filters,
                        project_id
                    )
                    all_results.extend(primary_results)
                    
                elif hasattr(self, 'chroma_collection') and self.chroma_collection is not None:
                    primary_results = self._search_chroma(
                        query_embedding,
                        top_k * 2,
                        filters
                    )
                    all_results.extend(primary_results)
                else:
                    # Mock results for demo when no vector DB available
                    st.info("⚠️ No vector database available. Using mock results for demonstration.")
                    all_results = self._get_mock_results(query, top_k)
            
            # Deduplicate and rerank results with keyword boosting (only for semantic/hybrid)
            if search_type in ["semantic", "hybrid"]:
                results = self._rerank_results(all_results, query, expanded_queries)
            else:  # lexical search already ranked
                results = all_results
            
            # Log scores for debugging
            if results:
                scores = [r.get('score', 0) for r in results]
                max_score = max(scores) if scores else 0
                min_score = min(scores) if scores else 0
                avg_score = sum(scores) / len(scores) if scores else 0
                logger.info(f"Retrieval scores - Max: {max_score:.3f}, Min: {min_score:.3f}, Avg: {avg_score:.3f}, Count: {len(results)}")
            
            # Filter by confidence threshold, but if no results above threshold, use top results anyway
            filtered_results = [
                r for r in results
                if r.get('score', 0) >= confidence_threshold
            ]
            
            # If no results above threshold but we have results, use top 3 results anyway
            if not filtered_results and results:
                logger.warning(f"No results above threshold {confidence_threshold}, but returning top {min(3, len(results))} results")
                filtered_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)[:3]
                # Mark these as low confidence
                for r in filtered_results:
                    r['low_confidence'] = True
            
            # Format response
            if not filtered_results:
                max_score = max([r.get('score', 0) for r in results]) if results else 0
                return {
                    "chunks": [],
                    "total_results": len(results),
                    "filtered_count": 0,
                    "status": "no_results",
                    "max_score": max_score,
                    "message": f"No results found. Max score was {max_score:.3f}. Ensure documents are processed and indexed.",
                    "query_time_ms": 0
                }
            
            # Format chunks with citations
            formatted_chunks = []
            for result in filtered_results[:top_k]:
                chunk = {
                    "id": result.get("id", "unknown"),
                    "text": result.get("text", ""),
                    "score": result.get("score", 0),
                    "metadata": result.get("metadata", {})
                }
                formatted_chunks.append(chunk)
            
            return {
                "chunks": formatted_chunks,
                "total_results": len(results),
                "filtered_count": len(filtered_results),
                "status": "success",
                "confidence_threshold": confidence_threshold,
                "search_type": search_type,
                "query_time_ms": 150  # Mock query time
            }
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            error_msg = str(e)
            logger.error(f"Exception in retrieve: {error_msg}")
            logger.error(f"Traceback:\n{error_traceback}")
            
            # Write detailed error to log file
            error_log_msg = f"""RETRIEVAL ERROR
Query: {query}
Query Type: {query_type}
Project ID: {project_id}
Vector DB Type: {self.vector_db_type}
Error: {error_msg}
Traceback:
{error_traceback}"""
            write_to_log(error_log_msg, "RETRIEVAL ERROR")
            
            return {
                "chunks": [],
                "status": "error",
                "error": error_msg,
                "message": f"Retrieval failed: {error_msg}",
                "traceback": error_traceback
            }
    
    def _build_filter(self, filters: Optional[Dict], project_id: str):
        """Build Qdrant filter from metadata filters. Supports page/section/subsection filtering."""
        # Return None if Qdrant is not available
        if Filter is None or FieldCondition is None or MatchValue is None:
            return None
        
        from qdrant_client.models import Range
            
        if not filters:
            # Always filter by project_id for multi-tenancy
            return Filter(
                must=[
                    FieldCondition(
                        key="project_id",
                        match=MatchValue(value=project_id)
                    )
                ]
            )
        
        conditions = [
            FieldCondition(
                key="project_id",
                match=MatchValue(value=project_id)
            )
        ]
        
        # Page number filtering
        if filters.get("page_number") is not None:
            conditions.append(
                FieldCondition(
                    key="page_number",
                    match=MatchValue(value=filters["page_number"])
                )
            )
        elif filters.get("page_range"):
            # Page range filtering
            page_range = filters["page_range"]
            conditions.append(
                FieldCondition(
                    key="page_number",
                    range=Range(
                        gte=page_range.get("start"),
                        lte=page_range.get("end")
                    )
                )
            )
        
        # Section number filtering
        if filters.get("section_number"):
            conditions.append(
                FieldCondition(
                    key="section_number",
                    match=MatchValue(value=filters["section_number"])
                )
            )
        
        # Subsection number filtering
        if filters.get("subsection_number"):
            conditions.append(
                FieldCondition(
                    key="subsection_number",
                    match=MatchValue(value=filters["subsection_number"])
                )
            )
        
        # Other filters
        if filters.get("cohort_id"):
            conditions.append(
                FieldCondition(
                    key="cohort_id",
                    match=MatchValue(value=filters["cohort_id"])
                )
            )
        
        if filters.get("source_type"):
            conditions.append(
                FieldCondition(
                    key="source_type",
                    match=MatchValue(value=filters["source_type"])
                )
            )
        
        if filters.get("speaker_role"):
            conditions.append(
                FieldCondition(
                    key="speaker_role",
                    match=MatchValue(value=filters["speaker_role"])
                )
            )
        
        return Filter(must=conditions) if conditions else None
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query for lexical search."""
        # Remove stop words and extract meaningful terms
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
            "of", "with", "by", "from", "as", "is", "are", "was", "were", "be", 
            "been", "being", "have", "has", "had", "do", "does", "did", "will", 
            "would", "should", "could", "may", "might", "must", "can", "this", 
            "that", "these", "those", "what", "which", "who", "when", "where", 
            "why", "how", "if", "then", "than", "so", "such", "very", "just", 
            "only", "also", "more", "most", "all", "each", "every", "some", 
            "any", "no", "not", "too", "much", "many", "few", "little"
        }
        
        # Convert to lowercase and split into words
        words = re.findall(r'\b\w+\b', query.lower())
        # Filter out stop words and short words
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords
    
    def _lexical_search(
        self,
        query: str,
        top_k: int,
        search_filter: Optional[Filter] = None,
        project_id: Optional[str] = None
    ) -> List[Dict]:
        """Perform lexical (keyword-based) search."""
        project_id = project_id or self.project_id
        keywords = self._extract_keywords(query)
        
        if not keywords:
            return []
        
        logger.info(f"Lexical search: keywords={keywords}, top_k={top_k}")
        
        results = []
        
        try:
            if self.vector_db_type == "Qdrant" and self.qdrant_client:
                collection_name = f"project_{project_id}"
                # Check if collection exists
                collections = self.qdrant_client.get_collections().collections
                collection_names = [c.name for c in collections]
                
                if collection_name not in collection_names:
                    return []
                
                # Scroll through all chunks (or a large sample)
                scroll_result = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    scroll_filter=search_filter,
                    limit=10000,  # Get many chunks for keyword matching
                    with_payload=True,
                    with_vectors=False
                )
                
                # Score chunks based on keyword matches
                for point in scroll_result[0]:
                    text = point.payload.get("text", "").lower()
                    if not text:
                        continue
                    
                    # Count keyword matches
                    keyword_matches = sum(1 for keyword in keywords if keyword in text)
                    if keyword_matches == 0:
                        continue
                    
                    # Calculate score based on:
                    # 1. Number of keywords matched
                    # 2. Keyword frequency in text
                    # 3. Position of keywords (earlier = better)
                    keyword_count = sum(text.count(keyword) for keyword in keywords)
                    total_words = len(text.split())
                    
                    # TF-like score: keyword frequency normalized by text length
                    tf_score = keyword_count / max(total_words, 1)
                    
                    # Match ratio: how many keywords matched
                    match_ratio = keyword_matches / len(keywords)
                    
                    # Position bonus: keywords appearing early get bonus
                    first_positions = []
                    for keyword in keywords:
                        pos = text.find(keyword)
                        if pos >= 0:
                            first_positions.append(pos)
                    
                    position_bonus = 0.0
                    if first_positions:
                        avg_position = sum(first_positions) / len(first_positions)
                        # Normalize by text length (earlier = higher bonus)
                        position_bonus = max(0, (len(text) - avg_position) / len(text)) * 0.2
                    
                    # Combined score
                    score = (match_ratio * 0.5) + (tf_score * 0.3) + (position_bonus)
                    
                    results.append({
                        "id": str(point.id),
                        "text": point.payload.get("text", ""),
                        "score": min(score, 1.0),  # Cap at 1.0
                        "metadata": {k: v for k, v in point.payload.items() if k != "text"},
                        "search_type": "lexical",
                        "keyword_matches": keyword_matches,
                        "matched_keywords": [k for k in keywords if k in text]
                    })
            
            elif hasattr(self, 'chroma_collection') and self.chroma_collection:
                # Get all chunks from ChromaDB
                all_data = self.chroma_collection.get(
                    limit=10000,
                    where={"project_id": self.project_id} if search_filter is None else None
                )
                
                if all_data.get('documents'):
                    for i, text in enumerate(all_data['documents']):
                        text_lower = text.lower()
                        keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
                        if keyword_matches == 0:
                            continue
                        
                        keyword_count = sum(text_lower.count(keyword) for keyword in keywords)
                        total_words = len(text.split())
                        tf_score = keyword_count / max(total_words, 1)
                        match_ratio = keyword_matches / len(keywords)
                        
                        score = (match_ratio * 0.5) + (tf_score * 0.5)
                        
                        results.append({
                            "id": all_data['ids'][i] if all_data.get('ids') else str(i),
                            "text": text,
                            "score": min(score, 1.0),
                            "metadata": all_data['metadatas'][i] if all_data.get('metadatas') else {},
                            "search_type": "lexical",
                            "keyword_matches": keyword_matches,
                            "matched_keywords": [k for k in keywords if k in text_lower]
                        })
            
            # Sort by score and return top_k
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Lexical search error: {e}")
            return []
    
    def _hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int,
        search_filter: Optional[Filter] = None,
        project_id: Optional[str] = None,
        semantic_weight: float = 0.7,
        lexical_weight: float = 0.3,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        generate_answer: bool = False
    ) -> Dict[str, Any]:
        """
        Perform hybrid search using Reciprocal Rank Fusion (RRF) and optionally generate answer.
        
        Returns:
            Dict with 'chunks' (ranked passages) and optionally 'answer' (LLM-generated answer)
        """
        project_id = project_id or self.project_id
        
        logger.info(f"Hybrid search with RRF: top_k={top_k}, generate_answer={generate_answer}")
        
        if progress_callback:
            progress_callback("Performing semantic search...", 0.2)
        
        # Perform both searches
        semantic_results = []
        lexical_results = []
        
        # Semantic search
        if self.vector_db_type == "Qdrant" and self.qdrant_client:
            semantic_results = self._search_qdrant(
                query_embedding,
                top_k * 2,  # Get more for combining
                search_filter,
                f"project_{project_id}",
                progress_callback=progress_callback
            )
        elif hasattr(self, 'chroma_collection') and self.chroma_collection:
            semantic_results = self._search_chroma(
                query_embedding,
                top_k * 2,
                None  # Filters handled in method
            )
        
        if progress_callback:
            progress_callback("Performing keyword search...", 0.4)
        
        # Lexical search
        lexical_results = self._lexical_search(
            query,
            top_k * 2,
            search_filter,
            project_id
        )
        
        if progress_callback:
            progress_callback("Applying RRF ranking...", 0.6)
        
        # Step 1: Normalize scores to 0-1 range
        def normalize_scores(results: List[Dict]) -> List[Dict]:
            """Normalize scores to 0-1 range."""
            if not results:
                return []
            
            scores = [r.get("score", 0) for r in results]
            if not scores:
                return results
            
            min_score = min(scores)
            max_score = max(scores)
            
            if max_score == min_score:
                # All scores are the same, set to 1.0
                for r in results:
                    r["normalized_score"] = 1.0
            else:
                # Normalize to 0-1
                for r in results:
                    original_score = r.get("score", 0)
                    r["normalized_score"] = (original_score - min_score) / (max_score - min_score)
            
            return results
        
        semantic_results = normalize_scores(semantic_results)
        lexical_results = normalize_scores(lexical_results)
        
        # Step 2: Reciprocal Rank Fusion (RRF)
        k = 60  # RRF constant
        
        # Create rank dictionaries
        semantic_ranks = {}
        lexical_ranks = {}
        
        for rank, result in enumerate(semantic_results, start=1):
            chunk_id = result.get("id") or result.get("chunk_id") or str(result.get("metadata", {}).get("chunk_index", rank))
            semantic_ranks[chunk_id] = rank
        
        for rank, result in enumerate(lexical_results, start=1):
            chunk_id = result.get("id") or result.get("chunk_id") or str(result.get("metadata", {}).get("chunk_index", rank))
            lexical_ranks[chunk_id] = rank
        
        # Combine results with RRF scores
        combined_results = {}
        
        # Add semantic results
        for result in semantic_results:
            chunk_id = result.get("id") or result.get("chunk_id") or str(result.get("metadata", {}).get("chunk_index", 0))
            rank_semantic = semantic_ranks.get(chunk_id, len(semantic_results) + 1)
            rrf_semantic = 1.0 / (k + rank_semantic)
            
            if chunk_id not in combined_results:
                combined_results[chunk_id] = result.copy()
                combined_results[chunk_id]["rrf_semantic"] = rrf_semantic
                combined_results[chunk_id]["rrf_lexical"] = 0.0
                combined_results[chunk_id]["rank_semantic"] = rank_semantic
                combined_results[chunk_id]["rank_lexical"] = None
            else:
                combined_results[chunk_id]["rrf_semantic"] = rrf_semantic
                combined_results[chunk_id]["rank_semantic"] = rank_semantic
        
        # Add lexical results
        for result in lexical_results:
            chunk_id = result.get("id") or result.get("chunk_id") or str(result.get("metadata", {}).get("chunk_index", 0))
            rank_lexical = lexical_ranks.get(chunk_id, len(lexical_results) + 1)
            rrf_lexical = 1.0 / (k + rank_lexical)
            
            if chunk_id not in combined_results:
                combined_results[chunk_id] = result.copy()
                combined_results[chunk_id]["rrf_semantic"] = 0.0
                combined_results[chunk_id]["rrf_lexical"] = rrf_lexical
                combined_results[chunk_id]["rank_semantic"] = None
                combined_results[chunk_id]["rank_lexical"] = rank_lexical
            else:
                combined_results[chunk_id]["rrf_lexical"] = rrf_lexical
                combined_results[chunk_id]["rank_lexical"] = rank_lexical
        
        # Step 3: Calculate final RRF scores and remove duplicates
        final_results = []
        seen_texts = set()  # For duplicate detection
        
        for chunk_id, result in combined_results.items():
            rrf_score = result.get("rrf_semantic", 0.0) + result.get("rrf_lexical", 0.0)
            
            # Check for duplicates by text content
            text = result.get("text", "") or result.get("metadata", {}).get("text", "")
            text_hash = hash(text[:200])  # Use first 200 chars for duplicate detection
            
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                result["score"] = rrf_score
                result["original_semantic_score"] = result.get("normalized_score", result.get("score", 0))
                result["original_lexical_score"] = result.get("normalized_score", 0) if chunk_id in lexical_ranks else 0
                result["search_type"] = "hybrid_rrf"
                final_results.append(result)
            else:
                # Duplicate found - keep the one with higher score
                for i, existing in enumerate(final_results):
                    existing_text = existing.get("text", "") or existing.get("metadata", {}).get("text", "")
                    if hash(existing_text[:200]) == text_hash and existing.get("score", 0) < rrf_score:
                        # Replace with higher scoring duplicate
                        result["score"] = rrf_score
                        result["original_semantic_score"] = result.get("normalized_score", result.get("score", 0))
                        result["original_lexical_score"] = result.get("normalized_score", 0) if chunk_id in lexical_ranks else 0
                        result["search_type"] = "hybrid_rrf"
                        final_results[i] = result
                        break
        
        # Step 4: Sort by final RRF score and return top N
        final_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        top_results = final_results[:top_k]
        
        result_dict = {
            "chunks": top_results,
            "status": "success",
            "search_type": "hybrid_rrf",
            "total_results": len(final_results)
        }
        
        # Step 5: Generate answer if requested
        if generate_answer and top_results:
            if progress_callback:
                progress_callback("Generating answer with LLM...", 0.8)
            
            answer = self._generate_answer_with_rrf(query, semantic_results, lexical_results, top_results)
            result_dict["answer"] = answer
        
        if progress_callback:
            progress_callback("Hybrid search complete", 1.0)
        
        return result_dict
    
    def _generate_answer_with_rrf(
        self,
        query: str,
        semantic_results: List[Dict],
        lexical_results: List[Dict],
        top_passages: List[Dict]
    ) -> str:
        """
        Generate answer using LLM based on RRF-ranked passages.
        
        Uses the prompt template specified by the user.
        """
        try:
            # Try to import LLM
            try:
                from langchain_ollama import ChatOllama
                use_ollama = True
            except ImportError:
                use_ollama = False
                try:
                    from langchain_openai import ChatOpenAI
                    use_openai = True
                except ImportError:
                    use_openai = False
                    logger.warning("No LLM available for answer generation")
                    return "LLM not available for answer generation."
            
            # Initialize LLM
            if use_ollama:
                llm = ChatOllama(
                    model=os.getenv("OLLAMA_MODEL", "llama3.1:latest"),
                    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                    temperature=0.3
                )
            elif use_openai:
                llm = ChatOpenAI(
                    model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                    temperature=0.3
                )
            else:
                return "LLM not available for answer generation."
            
            # Format semantic results
            semantic_formatted = []
            for i, result in enumerate(semantic_results[:10], 1):  # Top 10 for context
                score = result.get("normalized_score", result.get("score", 0))
                text = result.get("text", "") or result.get("metadata", {}).get("text", "")
                metadata = result.get("metadata", {})
                semantic_formatted.append({
                    "score": round(score, 3),
                    "text": text[:500],  # Limit text length
                    "metadata": {k: v for k, v in metadata.items() if k != "text"}
                })
            
            # Format lexical results
            lexical_formatted = []
            for i, result in enumerate(lexical_results[:10], 1):  # Top 10 for context
                score = result.get("normalized_score", result.get("score", 0))
                text = result.get("text", "") or result.get("metadata", {}).get("text", "")
                metadata = result.get("metadata", {})
                lexical_formatted.append({
                    "score": round(score, 3),
                    "text": text[:500],  # Limit text length
                    "metadata": {k: v for k, v in metadata.items() if k != "text"}
                })
            
            # Format top passages
            passages_formatted = []
            for i, passage in enumerate(top_passages[:5], 1):  # Top 5 passages
                score = passage.get("score", 0)
                text = passage.get("text", "") or passage.get("metadata", {}).get("text", "")
                passages_formatted.append(f"{i}. (score: {round(score, 2)}) \"{text[:300]}...\"")
            
            # Build prompt
            prompt = f"""You are a retrieval and ranking assistant. Your job is to combine results from both semantic search and keyword search to produce an improved ranked list of passages.

User Query:

{query}

Semantic Search Results (with relevance scores):

{self._format_results_for_prompt(semantic_formatted)}

Keyword Search Results (with relevance scores):

{self._format_results_for_prompt(lexical_formatted)}

Step 1: Normalize the Relevance Scores

- Convert all scores to a comparable scale between 0 and 1.

- Do NOT discard results with lower scores unless they are irrelevant or duplicates.

Step 2: Perform Rank Fusion

Use the "Reciprocal Rank Fusion (RRF)" method:

Final Score for a document = 1 / (k + rank_semantic) + 1 / (k + rank_keyword)

Use k = 60 (or the value already used in retrieval pipeline).

Step 3: Remove duplicates

- If the same document appears in both lists, merge and keep the passage with the highest combined score.

Step 4: Return the Top N (N=5) Ranked Passages

- Sort them by final combined score from highest to lowest.

Step 5: Produce Final Answer

Using only the selected top passages, write an answer to the User Query.

- Be concise and factual.

- If the answer is not available in the passages, respond: "Insufficient information to answer."

Output Format:

1) <Ranked Passages List with scores>

2) Final Answer

Top Ranked Passages:

{chr(10).join(passages_formatted)}

Final Answer:"""
            
            # Generate answer
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # Extract final answer if the response includes both passages and answer
            if "Final Answer:" in answer:
                answer = answer.split("Final Answer:")[-1].strip()
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer with LLM: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return f"Error generating answer: {str(e)}"
    
    def _format_results_for_prompt(self, results: List[Dict]) -> str:
        """Format results for LLM prompt."""
        if not results:
            return "No results found."
        
        formatted = []
        for i, result in enumerate(results, 1):
            score = result.get("score", 0)
            text = result.get("text", "")
            metadata = result.get("metadata", {})
            formatted.append(f"{i}. Score: {score}, Text: \"{text}\", Metadata: {metadata}")
        
        return "\n".join(formatted)
    
    def _search_qdrant(
        self,
        query_embedding: List[float],
        top_k: int,
        search_filter: Optional[Filter],
        collection_name: str,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> List[Dict]:
        """Search Qdrant vector database with progress tracking."""
        if self.qdrant_client is None:
            logger.error("Qdrant client is None - cannot perform search")
            if progress_callback:
                progress_callback("Error: Qdrant client not initialized", 0.0)
            return []
            
        try:
            # Progress: Checking collection
            if progress_callback:
                progress_callback("Checking Qdrant collection...", 0.1)
            logger.debug(f"Checking Qdrant collection: {collection_name}")
            
            # Check if collection exists
            try:
                collections = self.qdrant_client.get_collections().collections
                collection_names = [c.name for c in collections]
                logger.debug(f"Available collections: {collection_names}")
            except Exception as e:
                logger.error(f"Failed to get collections: {e}")
                error_msg = f"Failed to connect to Qdrant: {e}"
                if progress_callback:
                    progress_callback(f"Error: {error_msg}", 0.0)
                return []
            
            if collection_name not in collection_names:
                warning_msg = f"Collection '{collection_name}' not found. Available: {', '.join(collection_names)}"
                logger.warning(warning_msg)
                if progress_callback:
                    progress_callback(f"Warning: {warning_msg}", 0.0)
                return []
            
            # Progress: Performing search
            if progress_callback:
                progress_callback("Searching Qdrant database...", 0.3)
            logger.info(f"Searching Qdrant collection '{collection_name}' with top_k={top_k}, filter={search_filter}")
            logger.debug(f"Query embedding dimension: {len(query_embedding) if query_embedding else 0}")
            
            # Perform search with optimized parameters
            try:
                if progress_callback:
                    progress_callback("Executing optimized vector search...", 0.5)
                
                # ✅ Optimization 4: Query-time HNSW parameters
                # hnsw_ef: Trade-off between speed and accuracy (20-200 range)
                # Lower values = faster search, higher values = more accurate
                # For most use cases, 64 provides good balance
                search_params = None
                if SearchParams is not None:
                    search_params = SearchParams(hnsw_ef=64)  # Lower than ef_construct for faster queries
                else:
                    # Fallback: use dict format if SearchParams class not available
                    search_params = {"hnsw_ef": 64}
                
                search_results = self.qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    query_filter=search_filter,
                    limit=top_k,
                    score_threshold=None,  # Get all results, filter by confidence later
                    search_params=search_params  # HNSW optimization
                )
                logger.info(f"Qdrant search returned {len(search_results)} results (hnsw_ef=64)")
            except Exception as search_error:
                logger.error(f"Qdrant search failed: {search_error}")
                logger.error(f"Collection: {collection_name}, Top_k: {top_k}, Filter: {search_filter}")
                error_traceback = traceback.format_exc()
                logger.error(f"Traceback:\n{error_traceback}")
                
                # Try without filter as fallback
                try:
                    logger.warning("Retrying search without filter...")
                    if progress_callback:
                        progress_callback("Retrying search without filter...", 0.4)
                    
                    # Use optimized search params for retry as well
                    search_params = None
                    if SearchParams is not None:
                        search_params = SearchParams(hnsw_ef=64)
                    else:
                        search_params = {"hnsw_ef": 64}
                    
                    search_results = self.qdrant_client.search(
                        collection_name=collection_name,
                        query_vector=query_embedding,
                        query_filter=None,
                        limit=top_k,
                        score_threshold=None,
                        search_params=search_params  # HNSW optimization
                    )
                    logger.info(f"Retry search returned {len(search_results)} results")
                except Exception as retry_error:
                    error_msg = f"Qdrant search error: {retry_error}"
                    logger.error(f"Retry also failed: {retry_error}")
                    logger.error(f"Traceback:\n{traceback.format_exc()}")
                    if progress_callback:
                        progress_callback(f"Error: {error_msg}", 0.0)
                    return []
            
            # Progress: Processing results
            if progress_callback:
                progress_callback("Processing search results...", 0.7)
            
            results = []
            result_count = 0
            for result in search_results:
                try:
                    result_dict = {
                        "id": str(result.id),
                        "text": result.payload.get("text", ""),
                        "score": float(result.score) if hasattr(result, 'score') else 0.0,
                        "metadata": {k: v for k, v in result.payload.items() if k != "text"}
                    }
                    results.append(result_dict)
                    result_count += 1
                except Exception as e:
                    logger.warning(f"Failed to process result {result.id}: {e}")
                    continue
            
            logger.info(f"Qdrant search returned {result_count} results from collection '{collection_name}'")
            if result_count == 0:
                logger.warning(f"No results found. Check if collection '{collection_name}' has indexed documents.")
            
            # Progress: Complete
            if progress_callback:
                progress_callback(f"Found {result_count} results", 1.0)
            
            return results
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"Qdrant search exception: {e}")
            logger.error(f"Traceback:\n{error_traceback}")
            error_msg = f"Qdrant search error: {e}"
            if progress_callback:
                progress_callback(f"Error: {error_msg}", 0.0)
            return []
    
    def _search_chroma(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict]
    ) -> List[Dict]:
        """Search ChromaDB vector database. Supports page/section/subsection filtering."""
        try:
            where = {"project_id": self.project_id}
            if filters:
                # Handle page number filtering
                if filters.get("page_number") is not None:
                    where["page_number"] = filters["page_number"]
                elif filters.get("page_range"):
                    # ChromaDB doesn't support range queries directly, so we'll filter after
                    page_range = filters["page_range"]
                    where["page_number"] = {"$gte": page_range.get("start"), "$lte": page_range.get("end")}
                
                # Handle section/subsection filtering
                if filters.get("section_number"):
                    where["section_number"] = filters["section_number"]
                if filters.get("subsection_number"):
                    where["subsection_number"] = filters["subsection_number"]
                
                # Other filters
                if filters.get("cohort_id"):
                    where["cohort_id"] = filters["cohort_id"]
                if filters.get("source_type"):
                    where["source_type"] = filters["source_type"]
                if filters.get("speaker_role"):
                    where["speaker_role"] = filters["speaker_role"]
            
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2,  # Get more results for post-filtering if needed
                where=where if where else None,
                include=['documents', 'metadatas', 'distances']  # Include distances for scoring
            )
            
            # Post-filter for page ranges if ChromaDB doesn't support it
            if filters and filters.get("page_range") and results.get('ids') is not None and len(results.get('ids', [[]])[0]) > 0:
                page_range = filters["page_range"]
                filtered_results = []
                for i, metadata in enumerate(results.get('metadatas', [[]])[0]):
                    page_num = metadata.get('page_number')
                    if page_num and page_range.get("start") <= page_num <= page_range.get("end"):
                        filtered_results.append(i)
                
                if len(filtered_results) > 0:
                    # Rebuild results with filtered indices
                    filtered_ids = [results['ids'][0][i] for i in filtered_results]
                    filtered_docs = [results['documents'][0][i] for i in filtered_results]
                    filtered_metas = [results['metadatas'][0][i] for i in filtered_results]
                    filtered_dists = [results['distances'][0][i] for i in filtered_results]
                    results = {
                        'ids': [filtered_ids],
                        'documents': [filtered_docs],
                        'metadatas': [filtered_metas],
                        'distances': [filtered_dists]
                    }
            
            formatted_results = []
            if results.get('ids') is not None and len(results.get('ids', [[]])[0]) > 0:
                distances = results.get('distances', [[]])
                for i in range(len(results['ids'][0])):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    # Cosine distance: 0 = identical, 2 = opposite
                    # Convert to similarity: score = 1 - (distance / 2)
                    distance = distances[0][i] if distances is not None and len(distances[0]) > i else 1.0
                    similarity_score = max(0.0, 1.0 - (distance / 2.0))  # Convert distance to similarity
                    
                    # Safely get metadata - ChromaDB returns metadatas as list of lists
                    metadata = {}
                    if results.get('metadatas') is not None and len(results.get('metadatas', [[]])) > 0:
                        if len(results['metadatas'][0]) > i:
                            metadata = results['metadatas'][0][i]
                    
                    formatted_results.append({
                        "id": results['ids'][0][i],
                        "text": results['documents'][0][i],
                        "score": similarity_score,
                        "metadata": metadata
                    })
            
            return formatted_results
        except Exception as e:
            st.error(f"ChromaDB search error: {e}")
            return []
    
    def _search_pinecone(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict],
        project_id: str
    ) -> List[Dict]:
        """Search Pinecone vector database."""
        if self.pinecone_index is None:
            return []
        
        try:
            # Build metadata filter for Pinecone
            metadata_filter = {"project_id": {"$eq": project_id}}
            
            if filters:
                # Page number filtering
                if filters.get("page_number") is not None:
                    metadata_filter["page_number"] = {"$eq": filters["page_number"]}
                elif filters.get("page_range"):
                    page_range = filters["page_range"]
                    metadata_filter["page_number"] = {
                        "$gte": page_range.get("start"),
                        "$lte": page_range.get("end")
                    }
                
                # Section/subsection filtering
                if filters.get("section_number"):
                    metadata_filter["section_number"] = {"$eq": filters["section_number"]}
                if filters.get("subsection_number"):
                    metadata_filter["subsection_number"] = {"$eq": filters["subsection_number"]}
                
                # Other filters
                if filters.get("cohort_id"):
                    metadata_filter["cohort_id"] = {"$eq": filters["cohort_id"]}
                if filters.get("source_type"):
                    metadata_filter["source_type"] = {"$eq": filters["source_type"]}
                if filters.get("speaker_role"):
                    metadata_filter["speaker_role"] = {"$eq": filters["speaker_role"]}
            
            # Search Pinecone
            search_results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=metadata_filter if metadata_filter else None
            )
            
            results = []
            for match in search_results.matches:
                metadata = match.metadata or {}
                results.append({
                    "id": str(match.id),
                    "text": metadata.get("text", ""),
                    "score": match.score,
                    "metadata": {k: v for k, v in metadata.items() if k != "text"}
                })
            
            return results
        except Exception as e:
            st.error(f"Pinecone search error: {e}")
            return []
    
    def _get_mock_results(self, query: str, top_k: int) -> List[Dict]:
        """Generate mock results for demo when no vector DB available."""
        mock_results = [
            {
                "id": "chunk_1",
                "text": "Sample retrieved text content related to the query. This demonstrates how the RAG system retrieves relevant context.",
                "score": 0.85,
                "metadata": {
                    "file_name": "interview_001.pdf",
                    "source_type": "transcript",
                    "speaker_role": "interviewee",
                    "cohort_id": "cohort_a",
                    "start_time": "00:05:30",
                    "end_time": "00:06:15"
                }
            },
            {
                "id": "chunk_2",
                "text": "Another relevant snippet that matches the query context. This shows citation metadata.",
                "score": 0.78,
                "metadata": {
                    "file_name": "notes_2024.txt",
                    "source_type": "note",
                    "cohort_id": "cohort_b"
                }
            }
        ]
        return mock_results[:top_k]
    
    def retrieve_for_summarization(
        self,
        query: str,
        top_k: int = 15,
        filters: Optional[Dict] = None,
        use_hierarchical: bool = True,
        search_type: str = "semantic",
        extract_query_filters: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, Any]:
        """Specialized retrieval for summarization tasks."""
        return self.retrieve(
            query=query,
            query_type="summarization",
            top_k=top_k,
            confidence_threshold=0.6,  # Lower threshold for summarization
            filters=filters,
            use_hierarchical=use_hierarchical,
            search_type=search_type,
            extract_query_filters=extract_query_filters,
            progress_callback=progress_callback
        )
    
    def retrieve_for_comparison(
        self,
        query: str,
        cohort_ids: List[str],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Specialized retrieval for comparison tasks across cohorts."""
        results = {}
        for cohort_id in cohort_ids:
            cohort_results = self.retrieve(
                query=query,
                query_type="comparison",
                top_k=top_k,
                filters={"cohort_id": cohort_id}
            )
            results[cohort_id] = cohort_results
        
        return {
            "comparison_results": results,
            "status": "success"
        }
    
    def retrieve_for_synthesis(
        self,
        query: str,
        top_k: int = 10,
        iterations: int = 2,
        use_hierarchical: bool = True,
        search_type: str = "semantic",
        extract_query_filters: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, Any]:
        """Multi-hop retrieval for synthesis tasks."""
        # Initial retrieval
        initial_results = self.retrieve(
            query=query,
            query_type="synthesis",
            top_k=top_k,
            use_hierarchical=use_hierarchical,
            search_type=search_type,
            extract_query_filters=extract_query_filters,
            progress_callback=progress_callback
        )
        
        # Refined retrieval (preserve filters from first query)
        refined_query = query  # Placeholder
        refined_results = self.retrieve(
            query=refined_query,
            query_type="synthesis",
            top_k=top_k,
            use_hierarchical=use_hierarchical,
            search_type=search_type,
            extract_query_filters=extract_query_filters,
            progress_callback=progress_callback
        )
        
        # Extract key terms from initial results for refinement
        # (Simplified - in production, use LLM to extract key concepts)
        refined_query = query  # Placeholder
        
        # Refined retrieval
        refined_results = self.retrieve(
            query=refined_query,
            query_type="synthesis",
            top_k=top_k,
            use_hierarchical=use_hierarchical,
            search_type=search_type
        )
        
        # Combine and deduplicate
        combined_chunks = {}
        for chunk in initial_results.get("chunks", []):
            combined_chunks[chunk["id"]] = chunk
        for chunk in refined_results.get("chunks", []):
            combined_chunks[chunk["id"]] = chunk
        
        return {
            "chunks": list(combined_chunks.values()),
            "status": "success",
            "iterations": iterations
        }

