"""
Legal Document Chunker Integration
Integrates LegalDocumentChunker with Qdrant and Streamlit app.
Supports multiple search modes: semantic, BM25, and mixed (hybrid).
Also provides integration with advanced legal reranking system.
"""

import logging
import re
import os
from typing import List, Dict, Optional, Tuple, Callable, Any
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Chunking log file path
CHUNKING_LOG_FILE = "chunking_log.txt"

# Import normalization utilities
try:
    from scripts.normalization import (
        TextNormalizer,
        ScoreNormalizer,
        QueryNormalizer,
        normalize_search_results,
        deduplicate_by_normalized_content
    )
    NORMALIZATION_AVAILABLE = True
except ImportError:
    NORMALIZATION_AVAILABLE = False
    # Fallback to simple normalization
    class TextNormalizer:
        @staticmethod
        def normalize_for_deduplication(text: str) -> str:
            return re.sub(r'\s+', ' ', text.lower().strip())

# Try to import rank-bm25 for BM25 scoring (optional dependency)
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    BM25Okapi = None

try:
    # Use importlib for modules starting with numbers
    import importlib
    qdrant_chunker_module = importlib.import_module('scripts.00_chunking.qdrant_chunker')
    LegalDocumentChunker = qdrant_chunker_module.LegalDocumentChunker
    read_pdf = qdrant_chunker_module.read_pdf
    read_text = qdrant_chunker_module.read_text
    read_docx = qdrant_chunker_module.read_docx
    embed_texts = qdrant_chunker_module.embed_texts
    create_collection = qdrant_chunker_module.create_collection
    upsert = qdrant_chunker_module.upsert
    extract_ner_entities = qdrant_chunker_module.extract_ner_entities
    from qdrant_client import QdrantClient
    LEGAL_CHUNKER_AVAILABLE = True
except ImportError as e:
    LEGAL_CHUNKER_AVAILABLE = False
    print(f"Warning: Legal chunker not available: {e}")

logger = logging.getLogger(__name__)

# Default chunking methods (can be overridden)
_DEFAULT_CHUNKING_METHODS = [
    "hierarchical",  # Default primary method
    "semantic",
    "structural"
]

def get_available_chunking_methods() -> List[str]:
    """
    Dynamically detect available chunking methods from LegalDocumentChunker.
    
    Returns:
        List of available chunking method names
    """
    if not LEGAL_CHUNKER_AVAILABLE:
        return []
    
    try:
        # LegalDocumentChunker already imported at top using importlib
        
        # Get all methods from LegalDocumentChunker that are chunking methods
        # Exclude private methods and special methods like __init__, chunk, preprocess, etc.
        chunker = LegalDocumentChunker(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
        
        # List of known chunking methods (these are the actual chunking strategies)
        available_methods = []
        
        # Check each method by trying to call it or checking if it exists
        potential_methods = [
            "recursive",
            "semantic", 
            "structural",
            "agentic",
            "cluster",
            "hierarchical",
        ]
        
        for method in potential_methods:
            if hasattr(chunker, method) and callable(getattr(chunker, method)):
                # Skip if it's a private method or helper
                if not method.startswith('_'):
                    # Special check for semantic: verify SemanticChunker is available
                    if method == "semantic":
                        try:
                            from langchain_experimental.text_splitter import SemanticChunker
                            if SemanticChunker is None:
                                logger.debug("Semantic chunking not available: langchain-experimental not installed")
                                continue
                        except (ImportError, AttributeError):
                            logger.debug("Semantic chunking not available: langchain-experimental not installed")
                            continue
                    available_methods.append(method)
                    logger.debug(f"Added chunking method: {method}")
        
        # Ensure hierarchical is first (default)
        if "hierarchical" in available_methods:
            available_methods.remove("hierarchical")
            available_methods.insert(0, "hierarchical")
        
        return available_methods if available_methods else _DEFAULT_CHUNKING_METHODS
    except Exception as e:
        logger.warning(f"Could not detect chunking methods dynamically: {e}")
        # Fallback to default list
        return _DEFAULT_CHUNKING_METHODS.copy()

# Available chunking methods (dynamically detected)
CHUNKING_METHODS = get_available_chunking_methods()

def _write_chunking_log(
    timestamp: str,
    document_name: str,
    chunk_id: str,
    chunk_index: int,
    chunking_method: str,
    status: str,
    error_message: Optional[str] = None
) -> None:
    """
    Write chunking operation to log file.
    
    Args:
        timestamp: ISO timestamp of the operation
        document_name: Name of the document being chunked
        chunk_id: Unique ID of the chunk
        chunk_index: Index of the chunk within the method
        chunking_method: Chunking method applied
        status: Status of the operation ("APPLIED", "SKIPPED", "ERROR")
        error_message: Optional error message if status is ERROR
    """
    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / CHUNKING_LOG_FILE
        
        with open(log_file, "a", encoding="utf-8") as f:
            log_entry = (
                f"{timestamp}|{document_name}|{chunk_id}|{chunk_index}|"
                f"{chunking_method}|{status}"
            )
            if error_message:
                log_entry += f"|{error_message}"
            log_entry += "\n"
            f.write(log_entry)
    except Exception as e:
        logger.warning(f"Could not write to chunking log file: {e}")


def get_max_chunk_number(
    qdrant_client: Any,
    collection_name: str
) -> int:
    """
    Query Qdrant to find the current highest chunk_number value among all stored chunks.
    
    Args:
        qdrant_client: Qdrant client instance
        collection_name: Qdrant collection name
        
    Returns:
        Maximum chunk_number found, or 0 if no chunks exist or error occurs
    """
    try:
        max_chunk_number = 0
        
        # Scroll through all chunks to find max chunk_number
        # We'll process in batches to avoid memory issues
        offset = None
        while True:
            scroll_result = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=None,  # Get all chunks
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            if not points:
                break
            
            # Check chunk_number in each point's payload
            for point in points:
                payload = point.payload or {}
                chunk_num = payload.get("chunk_number")
                
                if chunk_num is not None:
                    try:
                        chunk_num_int = int(chunk_num)
                        # Only consider valid chunk_numbers (not Qdrant Point IDs)
                        if 0 <= chunk_num_int <= 1000000:
                            max_chunk_number = max(max_chunk_number, chunk_num_int)
                    except (ValueError, TypeError):
                        pass
            
            # Get next offset
            offset = scroll_result[1]
            if offset is None:
                break
        
        logger.info(f"Found max chunk_number: {max_chunk_number}")
        return max_chunk_number
        
    except Exception as e:
        logger.warning(f"Error getting max chunk_number, defaulting to 0: {e}")
        return 0


def get_default_chunking_methods() -> List[str]:
    """
    Get default chunking methods for upload.
    
    Returns:
        List of default chunking method names (includes hierarchical)
    """
    available = get_available_chunking_methods()
    # Default: hierarchical + semantic + structural if available
    defaults = []
    for method in ["hierarchical", "semantic", "structural"]:
        if method in available:
            defaults.append(method)
    # If no defaults found, use all available
    return defaults if defaults else available

# Available embedding models (open-source first, Ollama included)
EMBEDDING_MODELS = [
    "ollama/llama3.1:latest",  # Llama 3.1 via Ollama (local)
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/bge-base-en-v1.5",
    "sentence-transformers/nomic-embed-text-v1",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]


def ingest_legal_document(
    file_path: str,
    qdrant_client: QdrantClient,
    collection_name: str,
    embedding_model: str,
    chunking_methods: Optional[List[str]] = None,
    file_id: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None
) -> Dict:
    """
    Ingest legal document using all specified chunking methods.
    
    Args:
        file_path: Path to PDF or TXT file
        qdrant_client: Qdrant client instance
        collection_name: Qdrant collection name
        embedding_model: Embedding model name
        chunking_methods: List of chunking methods (default: all methods)
        file_id: Optional file identifier
        progress_callback: Optional callback function(current_method, method_index, total_methods) for progress updates
        
    Returns:
        Dictionary with ingestion results
    """
    if not LEGAL_CHUNKER_AVAILABLE:
        raise ImportError("Legal chunker not available. Install dependencies.")
    
    if chunking_methods is None:
        # Use default chunking methods if none specified
        chunking_methods = get_default_chunking_methods()
    
    # Validate methods against available methods
    available_methods = get_available_chunking_methods()
    for method in chunking_methods:
        if method not in available_methods:
            raise ValueError(f"Unknown chunking method: {method}. Available methods: {available_methods}")
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read file and convert to Markdown
    logger.info(f"Reading file: {file_path}")
    suffix = path.suffix.lower()
    
    if suffix == '.pdf':
        text, page_numbers = read_pdf(path)
    elif suffix in ['.docx', '.doc']:
        text, page_numbers = read_docx(path)
    else:
        # Handles .md (read directly) and other text files (convert to Markdown)
        text, page_numbers = read_text(path)
    
    logger.info(f"Converted to Markdown. Text length: {len(text)} characters")
    
    # Extract NER entities once from the full document (for reference)
    # Note: Entities will be extracted per chunk to ensure chunk-specific entities
    logger.info("Extracting Named Entities (NER) from document...")
    # extract_ner_entities already imported at top using importlib
    doc_ner_entities = extract_ner_entities(text)
    logger.info(f"Extracted document-level NER entities: {', '.join([f'{k}={v}' for k, v in doc_ner_entities.items() if v])}")
    
    # Initialize chunker
    chunker = LegalDocumentChunker(embedding_model=embedding_model)
    
    # Get document name and timestamp for logging and document_id generation
    document_name = path.name
    upload_timestamp = datetime.now().isoformat()
    
    # Get current max chunk_number from Qdrant before starting
    # This ensures global sequential numbering across all files
    logger.info("Querying Qdrant for current max chunk_number...")
    max_id = get_max_chunk_number(qdrant_client, collection_name)
    
    # If Qdrant is empty (max_id = 0), start from 0, otherwise start from max_id + 1
    if max_id == 0:
        # Empty Qdrant: first chunk gets chunk_number = 0
        start_chunk_number = -1
        logger.info(f"Qdrant is empty. New chunks will start from chunk_number = 0")
    else:
        # Non-empty: continue from max_id + 1
        start_chunk_number = max_id
        logger.info(f"Current max chunk_number in Qdrant: {max_id}. New chunks will start from {max_id + 1}")
    
    # Generate unique document_id for this file
    # Use filename + timestamp for uniqueness
    import hashlib
    document_id_string = f"{document_name}::{upload_timestamp}"
    document_id = hashlib.md5(document_id_string.encode('utf-8')).hexdigest()[:16]
    
    # Apply all chunking methods
    all_chunks = []
    method_counts = {}
    total_methods = len(chunking_methods)
    
    # Track current chunk_number (will increment for each chunk across all methods)
    current_chunk_number = start_chunk_number
    
    # Log start of chunking process
    logger.info(f"Starting chunking process for document: {document_name}")
    logger.info(f"Document ID: {document_id}")
    logger.info(f"Selected chunking methods: {', '.join(chunking_methods)}")
    
    for method_index, method in enumerate(chunking_methods):
        method_start_time = datetime.now().isoformat()
        method_status = "APPLIED"
        method_error = None
        
        try:
            # Call progress callback if provided
            if progress_callback:
                progress_callback(method, method_index, total_methods)
            
            logger.info(f"[{method_index + 1}/{total_methods}] Chunking with method: {method}")
            
            # Check if semantic chunking is available
            if method == "semantic":
                try:
                    from langchain_experimental.text_splitter import SemanticChunker
                    if SemanticChunker is None:
                        raise ImportError("SemanticChunker not available")
                except (ImportError, AttributeError):
                    method_status = "SKIPPED"
                    method_error = "langchain-experimental not installed or SemanticChunker unavailable"
                    logger.warning(f"Semantic chunking skipped: {method_error}")
                    _write_chunking_log(
                        method_start_time,
                        document_name,
                        "N/A",
                        0,
                        method,
                        method_status,
                        method_error
                    )
                    method_counts[method] = 0
                    continue
            
            # First chunk without entities - we'll add chunk-specific entities later
            chunks = chunker.chunk(text, method, page_numbers, ner_entities=None)
            
            # Verify chunks were created and have correct chunking_method
            if not chunks:
                logger.warning(f"No chunks generated for method: {method}")
                method_status = "SKIPPED"
                method_error = "No chunks generated"
                _write_chunking_log(
                    method_start_time,
                    document_name,
                    "N/A",
                    0,
                    method,
                    method_status,
                    method_error
                )
                method_counts[method] = 0
                continue
            
            # First pass: assign chunk_numbers and build clause_number to chunk_number mapping for structural chunks
            clause_to_chunk_number = {}  # Maps clause_number -> chunk_number for parent lookups
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = chunk.get("id", f"chunk_{chunk_idx}")
                
                # Ensure chunking_method is set correctly
                if chunk.get("chunking_method") != method:
                    logger.warning(f"Chunk {chunk_id} has incorrect chunking_method: expected '{method}', got '{chunk.get('chunking_method')}'. Fixing...")
                    chunk["chunking_method"] = method
                
                # Assign global sequential chunk_number (unique across all files)
                # Increment for each chunk to ensure uniqueness
                current_chunk_number += 1
                chunk["chunk_number"] = current_chunk_number
                
                # For structural chunks, map clause_number to chunk_number for parent lookups
                if method == "structural" and chunk.get("clause_number"):
                    clause_to_chunk_number[chunk.get("clause_number")] = current_chunk_number
                
                # Log each chunk
                _write_chunking_log(
                    method_start_time,
                    document_name,
                    str(chunk_id),
                    chunk_idx,
                    method,
                    "APPLIED"
                )
            
            # Second pass: set parent_chunk_number and chunk_level for structural chunks
            for chunk in chunks:
                # Set chunk_level (same as hierarchy_level for structural chunks, or infer from level)
                hierarchy_level = chunk.get("hierarchy_level")
                if hierarchy_level is not None:
                    chunk["chunk_level"] = hierarchy_level
                elif method == "structural":
                    # Infer from level if hierarchy_level not set
                    chunk["chunk_level"] = chunk.get("level", 1)
                else:
                    # Non-structural chunks default to level 1
                    chunk["chunk_level"] = 1
                
                # Set parent_chunk_number for structural chunks
                if method == "structural":
                    parent_clause = chunk.get("parent")
                    if parent_clause and parent_clause in clause_to_chunk_number:
                        chunk["parent_chunk_number"] = clause_to_chunk_number[parent_clause]
                    else:
                        chunk["parent_chunk_number"] = None  # Level 1 chunks have no parent
                
                # Add text_preview (first 200 characters)
                chunk_text = chunk.get("text", "")
                chunk["text_preview"] = chunk_text[:200] + ("..." if len(chunk_text) > 200 else "")
            
            # Extract entities per chunk and filter to only include entities present in chunk text
            for chunk in chunks:
                chunk_text = chunk.get("text", "")
                # Extract entities from this specific chunk
                chunk_ner_entities = extract_ner_entities(chunk_text)
                
                # Filter: only include entities that actually appear in this chunk's text
                filtered_entities = {}
                chunk_text_lower = chunk_text.lower()
                
                for key, value in chunk_ner_entities.items():
                    if value:
                        # Check if entity value appears in chunk text (case-insensitive)
                        if value.lower() in chunk_text_lower:
                            filtered_entities[key] = value
                
                # If no chunk-specific entities found, use document-level entities as fallback
                # but only if they appear in the chunk
                if not any(filtered_entities.values()):
                    for key, value in doc_ner_entities.items():
                        if value and value.lower() in chunk_text_lower:
                            filtered_entities[key] = value
                
                # Add filtered entities to chunk
                if filtered_entities:
                    for key, value in filtered_entities.items():
                        if value:
                            chunk[key] = value
            
            # Add file metadata to all chunks
            # Use full filename with extension to ensure uniqueness (e.g., contract.md vs contract.pdf)
            source_filename = path.name  # Includes extension: e.g., "contract_restored_original.pdf"
            
            for chunk in chunks:
                chunk["source_file"] = source_filename  # REQUIRED metadata field - includes extension for uniqueness
                chunk["source_filename"] = source_filename  # Also store as source_filename for consistency
                chunk["document_id"] = document_id  # Unique document identifier
                chunk["upload_timestamp"] = upload_timestamp  # Track when file was uploaded
                chunk["upload_time"] = upload_timestamp  # Also add upload_time for backward compatibility
                if file_id:
                    chunk["file_id"] = file_id
                chunk["file_name"] = source_filename  # For backward compatibility
                
                # Ensure chunking_method is always set
                if not chunk.get("chunking_method"):
                    chunk["chunking_method"] = method
                    logger.warning(f"Chunk {chunk.get('id', 'unknown')} missing chunking_method, set to '{method}'")
            
            all_chunks.extend(chunks)
            method_counts[method] = len(chunks)
            logger.info(f"  ✅ Generated {len(chunks)} chunks for {method}")
            
            # Verify chunks have chunking_method set
            for chunk in chunks:
                if chunk.get("chunking_method") != method:
                    logger.error(f"ERROR: Chunk missing or incorrect chunking_method! Expected '{method}', got '{chunk.get('chunking_method')}'")
                    chunk["chunking_method"] = method  # Fix it
            
            # Log method completion
            logger.info(f"  Method '{method}': {len(chunks)} chunks created successfully")
            
        except Exception as e:
            method_status = "ERROR"
            method_error = str(e)
            logger.error(f"❌ Error chunking with method '{method}': {e}", exc_info=True)
            
            # Log the error
            _write_chunking_log(
                method_start_time,
                document_name,
                "N/A",
                0,
                method,
                method_status,
                method_error
            )
            
            # Continue with other methods instead of failing completely
            method_counts[method] = 0
            logger.warning(f"Skipping method '{method}' due to error. Other methods will continue.")
    
    logger.info(f"Total chunks generated: {len(all_chunks)}")
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    texts = [c['text'] for c in all_chunks]
    embeddings = embed_texts(texts, embedding_model)
    
    # Create collection if needed
    vector_dim = len(embeddings[0])
    try:
        create_collection(qdrant_client, collection_name, vector_dim)
    except Exception as e:
        logger.warning(f"Collection creation warning: {e}")
    
    # Upsert to Qdrant
    logger.info("Upserting to Qdrant...")
    upsert(qdrant_client, collection_name, all_chunks, embeddings)
    
    return {
        "success": True,
        "file_name": path.name,
        "total_chunks": len(all_chunks),
        "method_counts": method_counts,
        "embedding_model": embedding_model,
        "chunking_methods": chunking_methods
    }


def get_distinct_source_files(
    qdrant_client: QdrantClient,
    collection_name: str
) -> List[str]:
    """
    Get distinct source_file values from Qdrant collection.
    
    Args:
        qdrant_client: Qdrant client instance
        collection_name: Collection name
        
    Returns:
        List of distinct source file names
    """
    try:
        # Use scroll to get all points and extract distinct source_file values
        # Handle pagination for large collections
        source_files = set()
        next_page_offset = None
        max_iterations = 100  # Safety limit
        
        for _ in range(max_iterations):
            scroll_result = qdrant_client.scroll(
                collection_name=collection_name,
                limit=1000,  # Process in batches
                offset=next_page_offset,
                with_payload=True,
                with_vectors=False
            )
            
            points, next_page_offset = scroll_result
            
            # Extract source_file from each point
            for point in points:
                payload = point.payload or {}
                source_file = payload.get("source_file") or payload.get("file_name")
                if source_file:
                    source_files.add(source_file)
            
            # Break if no more pages
            if next_page_offset is None:
                break
        
        return sorted(list(source_files))
    except Exception as e:
        logger.error(f"Error getting distinct source files: {e}")
        return []


def get_distinct_chunking_methods(
    qdrant_client: QdrantClient,
    collection_name: str
) -> List[str]:
    """
    Get distinct chunking_method values from Qdrant collection metadata.
    This is used for retrieval to show only methods that actually exist in stored data.
    
    Args:
        qdrant_client: Qdrant client instance
        collection_name: Collection name
        
    Returns:
        List of distinct chunking method names found in Qdrant
    """
    try:
        # Use scroll to get all points and extract distinct chunking_method values
        # Handle pagination for large collections
        chunking_methods = set()
        next_page_offset = None
        max_iterations = 100  # Safety limit
        
        for _ in range(max_iterations):
            scroll_result = qdrant_client.scroll(
                collection_name=collection_name,
                limit=1000,  # Process in batches
                offset=next_page_offset,
                with_payload=True,
                with_vectors=False
            )
            
            points, next_page_offset = scroll_result
            
            # Extract chunking_method from each point
            for point in points:
                payload = point.payload or {}
                chunking_method = payload.get("chunking_method")
                if chunking_method:
                    chunking_methods.add(chunking_method)
                # Debug: log if chunking_method is missing
                elif logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Point {point.id} missing chunking_method in payload: {list(payload.keys())}")
            
            # Break if no more pages
            if next_page_offset is None:
                break
        
        # Log found methods for debugging
        found_methods = sorted(list(chunking_methods))
        logger.info(f"Found chunking methods in Qdrant: {found_methods}")
        
        # Also log available methods from codebase for comparison
        try:
            available_methods = get_available_chunking_methods()
            logger.info(f"Available chunking methods in codebase: {sorted(available_methods)}")
            
            # Check for case sensitivity or naming mismatches
            found_methods_lower = {m.lower() for m in found_methods}
            available_methods_lower = {m.lower() for m in available_methods}
            
            missing_methods = set(available_methods) - set(found_methods)
            if missing_methods:
                logger.warning(f"Available chunking methods not found in Qdrant: {sorted(missing_methods)}. "
                             f"This may indicate they weren't used during ingestion or failed silently.")
            
            # Check for case-insensitive matches that might indicate a naming issue
            case_mismatches = []
            for avail in available_methods:
                for found in found_methods:
                    if avail.lower() == found.lower() and avail != found:
                        case_mismatches.append((avail, found))
            if case_mismatches:
                logger.warning(f"Case sensitivity mismatches found: {case_mismatches}")
        except Exception as e:
            logger.debug(f"Could not compare with available methods: {e}")
        
        # Sort and ensure hierarchical is first if available (for default selection)
        methods_list = found_methods.copy()
        if "hierarchical" in methods_list:
            methods_list.remove("hierarchical")
            methods_list.insert(0, "hierarchical")
        
        return methods_list
    except Exception as e:
        logger.error(f"Error getting distinct chunking methods: {e}", exc_info=True)
        return []


def _tokenize_text(text: str, use_ngrams: bool = True) -> List[str]:
    """
    Tokenize text for BM25 scoring with n-gram support.
    Generates unigrams, bigrams, trigrams, 4-grams, and 5-grams.
    
    Args:
        text: Input text
        use_ngrams: If True, generate n-grams (1-5). If False, only unigrams.
        
    Returns:
        List of tokens (n-grams)
    """
    # Extract words (unigrams)
    words = re.findall(r'\b\w+\b', text.lower())
    
    if not use_ngrams or len(words) < 2:
        return words
    
    # Generate n-grams (1-5 grams)
    tokens = []
    
    # Unigrams (already have words)
    tokens.extend(words)
    
    # Bigrams (2-grams)
    for i in range(len(words) - 1):
        tokens.append(f"{words[i]} {words[i+1]}")
    
    # Trigrams (3-grams)
    if len(words) >= 3:
        for i in range(len(words) - 2):
            tokens.append(f"{words[i]} {words[i+1]} {words[i+2]}")
    
    # 4-grams
    if len(words) >= 4:
        for i in range(len(words) - 3):
            tokens.append(f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}")
    
    # 5-grams
    if len(words) >= 5:
        for i in range(len(words) - 4):
            tokens.append(f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]} {words[i+4]}")
    
    return tokens


def _highlight_query_terms(text: str, query_text: str) -> str:
    """
    Highlight query terms in the text using HTML mark tags.
    Uses n-gram matching (1-5 grams) to find all matching phrases.
    
    Args:
        text: Text to highlight
        query_text: Query text containing terms to highlight
        
    Returns:
        Text with highlighted terms wrapped in <mark> tags
    """
    if not query_text or not text:
        return text
    
    # Extract query n-grams (1-5 grams)
    query_words = re.findall(r'\b\w+\b', query_text.lower())
    
    if not query_words:
        return text
    
    # Generate all n-grams from query (1-5 grams)
    query_ngrams = []
    
    # Unigrams
    query_ngrams.extend(query_words)
    
    # Bigrams
    if len(query_words) >= 2:
        for i in range(len(query_words) - 1):
            query_ngrams.append(f"{query_words[i]} {query_words[i+1]}")
    
    # Trigrams
    if len(query_words) >= 3:
        for i in range(len(query_words) - 2):
            query_ngrams.append(f"{query_words[i]} {query_words[i+1]} {query_words[i+2]}")
    
    # 4-grams
    if len(query_words) >= 4:
        for i in range(len(query_words) - 3):
            query_ngrams.append(f"{query_words[i]} {query_words[i+1]} {query_words[i+2]} {query_words[i+3]}")
    
    # 5-grams
    if len(query_words) >= 5:
        for i in range(len(query_words) - 4):
            query_ngrams.append(f"{query_words[i]} {query_words[i+1]} {query_words[i+2]} {query_words[i+3]} {query_words[i+4]}")
    
    # Sort by length (longest first) to match longer phrases first
    query_ngrams.sort(key=len, reverse=True)
    
    # Create case-insensitive regex patterns for each n-gram
    highlighted_text = text
    used_positions = set()  # Track positions already highlighted
    
    # Process longest n-grams first to avoid partial matches
    for ngram in query_ngrams:
        # Create regex pattern that matches the n-gram as whole words/phrases
        # Escape special regex characters
        escaped_ngram = re.escape(ngram)
        # Match as whole phrase (word boundaries or spaces)
        pattern = r'\b' + escaped_ngram.replace(r'\ ', r'\s+') + r'\b'
        
        # Find all matches
        for match in re.finditer(pattern, highlighted_text, re.IGNORECASE):
            start, end = match.span()
            
            # Check if this position is already highlighted
            if not any(start <= pos < end for pos in used_positions):
                # Mark positions as used
                used_positions.update(range(start, end))
                
                # Replace with highlighted version
                matched_text = highlighted_text[start:end]
                highlighted_text = (
                    highlighted_text[:start] +
                    f'<mark style="background-color: yellow; padding: 2px 4px; border-radius: 3px;">{matched_text}</mark>' +
                    highlighted_text[end:]
                )
                # Update used_positions after insertion (account for HTML tag length)
                tag_length = len('<mark style="background-color: yellow; padding: 2px 4px; border-radius: 3px;"></mark>')
                used_positions = {pos + tag_length if pos >= start else pos for pos in used_positions}
                break  # Only highlight first occurrence of each n-gram to avoid over-highlighting
    
    return highlighted_text


def _bm25_search(
    qdrant_client: QdrantClient,
    collection_name: str,
    query_text: str,
    filter_condition: Any,
    top_k: int = 5
) -> List[Dict]:
    """
    Perform BM25 keyword-based search on Qdrant collection using n-grams (1-5 grams).
    
    Args:
        qdrant_client: Qdrant client instance
        collection_name: Collection name
        query_text: Query text for keyword matching
        filter_condition: Qdrant filter condition
        top_k: Number of results
        
    Returns:
        List of retrieved chunks with BM25 scores and highlighted text
    """
    if not BM25_AVAILABLE:
        raise ImportError("rank-bm25 required for BM25 search. Install with: pip install rank-bm25")
    
    # Tokenize query with n-grams (1-5 grams)
    query_tokens = _tokenize_text(query_text, use_ngrams=True)
    if not query_tokens:
        return []
    
    # Retrieve all matching chunks (with filter applied)
    # We need to get chunks to build BM25 index, then score them
    all_chunks = []
    all_texts = []
    chunk_metadata = []
    
    try:
        # Scroll through filtered chunks
        next_page_offset = None
        max_iterations = 100
        
        for _ in range(max_iterations):
            scroll_result = qdrant_client.scroll(
                collection_name=collection_name,
                limit=1000,
                offset=next_page_offset,
                scroll_filter=filter_condition,
                with_payload=True,
                with_vectors=False
            )
            
            points, next_page_offset = scroll_result
            
            for point in points:
                payload = point.payload or {}
                chunk_text = payload.get("text", "").strip()
                
                if chunk_text:
                    # Tokenize with n-grams for BM25
                    all_texts.append(_tokenize_text(chunk_text, use_ngrams=True))
                    all_chunks.append({
                        "id": point.id,
                        "payload": payload,
                        "text": chunk_text
                    })
                    chunk_metadata.append(payload)
            
            if next_page_offset is None:
                break
        
        if not all_texts:
            return []
        
        # Build BM25 index
        bm25 = BM25Okapi(all_texts)
        
        # Score documents
        scores_raw = bm25.get_scores(query_tokens)
        
        # Convert numpy array to list to avoid boolean ambiguity errors
        scores = list(scores_raw) if scores_raw is not None and hasattr(scores_raw, '__iter__') else []
        
        # Normalize BM25 scores to [0, 1] range for consistent comparison with cosine similarity
        if len(scores) > 0:
            min_score = min(scores)
            max_score = max(scores)
            if max_score > min_score:
                # Normalize to [0, 1]
                normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
            else:
                # All scores are the same
                normalized_scores = [1.0] * len(scores)
        else:
            normalized_scores = []
        
        # Combine scores with chunks and sort (use normalized scores for ranking)
        scored_chunks = list(zip(normalized_scores, scores, all_chunks, chunk_metadata))  # (normalized, original, chunk, metadata)
        scored_chunks.sort(key=lambda x: x[0], reverse=True)  # Sort by normalized score
        
        # Format results
        formatted_results = []
        seen_content = set()
        
        for norm_score, orig_score, chunk, metadata in scored_chunks[:top_k * 2]:  # Get more for deduplication
            chunk_text = chunk["text"]
            # Normalize text more aggressively: lowercase, strip, normalize whitespace
            normalized_text = re.sub(r'\s+', ' ', chunk_text.lower().strip())
            
            if normalized_text in seen_content:
                continue
            
            seen_content.add(normalized_text)
            
            # Highlight query terms in the text
            highlighted_text = _highlight_query_terms(chunk_text, query_text)
            
            formatted_results.append({
                "text": chunk_text,  # Original text
                "highlighted_text": highlighted_text,  # Text with highlighted query terms
                "score": float(norm_score),  # Normalized BM25 score [0, 1]
                "original_score": float(orig_score),  # Original BM25 score for reference
                "chunking_method": metadata.get("chunking_method"),
                "embedding_model": metadata.get("embedding_model"),
                "hierarchy_level": metadata.get("hierarchy_level"),
                "clause_number": metadata.get("clause_number"),
                "title": metadata.get("title"),
                "page": metadata.get("page"),
                "parent": metadata.get("parent"),
                "source_file": metadata.get("source_file") or metadata.get("file_name"),
                "file_id": metadata.get("file_id"),
                "file_name": metadata.get("file_name"),
                "individual_name": metadata.get("individual_name"),
                "company_name": metadata.get("company_name"),
                "address": metadata.get("address"),
                "email": metadata.get("email"),
                "phone": metadata.get("phone")
            })
            
            if len(formatted_results) >= top_k:
                break
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error in BM25 search: {e}", exc_info=True)
        raise


def _normalize_scores(results: List[Dict], score_key: str = "score") -> List[Dict]:
    """
    Normalize scores to [0, 1] range using min-max normalization.
    
    This ensures BM25 scores (typically 0-20+) and cosine similarity scores 
    (typically -1 to 1 or 0 to 1) are on the same scale before fusion.
    
    Args:
        results: List of result dictionaries with scores
        score_key: Key in dictionary containing the score
    
    Returns:
        Results with normalized scores (new key: "normalized_score")
    """
    if not results:
        return results
    
    # Extract scores
    scores = [result.get(score_key, 0.0) for result in results]
    
    if not scores:
        return results
    
    # Find min and max
    min_score = min(scores)
    max_score = max(scores)
    
    # Normalize scores
    if max_score == min_score:
        # All scores are the same, set normalized to 1.0
        normalized_scores = [1.0] * len(scores)
    else:
        # Min-max normalization to [0, 1]
        normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
    
    # Add normalized scores to results
    normalized_results = []
    for result, norm_score in zip(results, normalized_scores):
        new_result = result.copy()
        new_result["normalized_score"] = norm_score
        new_result["original_score"] = result.get(score_key, 0.0)  # Keep original for reference
        normalized_results.append(new_result)
    
    return normalized_results


def _reciprocal_rank_fusion(
    semantic_results: List[Dict],
    bm25_results: List[Dict],
    k: int = 60,
    normalize_scores: bool = True
) -> List[Dict]:
    """
    Combine semantic and BM25 results using Reciprocal Rank Fusion (RRF).
    
    Before fusion, normalizes both BM25 and cosine similarity scores to [0, 1] range
    to ensure fair combination of different score scales.
    
    RRF score = sum(1 / (k + rank)) for each result list
    
    Args:
        semantic_results: Results from semantic search with 'text' and 'score' fields
        bm25_results: Results from BM25 search with 'text' and 'score' fields
        k: RRF constant (default: 60, standard value)
        normalize_scores: Whether to normalize scores before fusion (default: True)
        
    Returns:
        Combined and re-ranked results with normalized scores
    """
    # Step 1: Normalize scores to [0, 1] range for fair comparison
    if normalize_scores:
        semantic_results = _normalize_scores(semantic_results, score_key="score")
        bm25_results = _normalize_scores(bm25_results, score_key="score")
        logger.debug(f"Normalized {len(semantic_results)} semantic scores and {len(bm25_results)} BM25 scores")
    
    # Step 2: Create content-to-result mapping for deduplication
    content_to_results = {}
    
    # Process semantic results
    for rank, result in enumerate(semantic_results, start=1):
        # Normalize text more aggressively for deduplication
        text_key = re.sub(r'\s+', ' ', result.get("text", "").lower().strip())
        if text_key:
            if text_key not in content_to_results:
                content_to_results[text_key] = {
                    **result,
                    "semantic_rank": rank,
                    "semantic_normalized_score": result.get("normalized_score", result.get("score", 0.0)),
                    "bm25_rank": None,
                    "bm25_normalized_score": None,
                    "rrf_score": 0.0
                }
            else:
                # Update if this rank is better (lower rank number)
                if content_to_results[text_key].get("semantic_rank", float('inf')) > rank:
                    content_to_results[text_key]["semantic_rank"] = rank
                    content_to_results[text_key]["semantic_normalized_score"] = result.get("normalized_score", result.get("score", 0.0))
    
    # Process BM25 results
    for rank, result in enumerate(bm25_results, start=1):
        # Normalize text more aggressively for deduplication
        text_key = re.sub(r'\s+', ' ', result.get("text", "").lower().strip())
        if text_key:
            if text_key not in content_to_results:
                content_to_results[text_key] = {
                    **result,
                    "semantic_rank": None,
                    "semantic_normalized_score": None,
                    "bm25_rank": rank,
                    "bm25_normalized_score": result.get("normalized_score", result.get("score", 0.0)),
                    "rrf_score": 0.0
                }
            else:
                # Update BM25 rank
                if content_to_results[text_key].get("bm25_rank") is None or \
                   content_to_results[text_key].get("bm25_rank", float('inf')) > rank:
                    content_to_results[text_key]["bm25_rank"] = rank
                    content_to_results[text_key]["bm25_normalized_score"] = result.get("normalized_score", result.get("score", 0.0))
    
    # Step 3: Calculate RRF scores
    for text_key, result in content_to_results.items():
        rrf_score = 0.0
        
        # Add semantic contribution (based on rank, not score)
        if result.get("semantic_rank") is not None:
            rrf_score += 1.0 / (k + result["semantic_rank"])
        
        # Add BM25 contribution (based on rank, not score)
        if result.get("bm25_rank") is not None:
            rrf_score += 1.0 / (k + result["bm25_rank"])
        
        result["rrf_score"] = rrf_score
        
        # Also calculate weighted score fusion using normalized scores (optional enhancement)
        weighted_score = 0.0
        weight_sum = 0.0
        
        if result.get("semantic_normalized_score") is not None:
            weighted_score += 0.6 * result["semantic_normalized_score"]  # 60% weight for semantic
            weight_sum += 0.6
        
        if result.get("bm25_normalized_score") is not None:
            weighted_score += 0.4 * result["bm25_normalized_score"]  # 40% weight for BM25
            weight_sum += 0.4
        
        if weight_sum > 0:
            result["weighted_score"] = weighted_score / weight_sum
        else:
            result["weighted_score"] = 0.0
        
        # Use RRF score as primary, but keep weighted score for reference
        result["score"] = rrf_score
    
    # Step 4: Sort by RRF score (descending)
    fused_results = list(content_to_results.values())
    fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
    
    # Step 5: Clean up temporary fields
    for result in fused_results:
        result.pop("semantic_rank", None)
        result.pop("bm25_rank", None)
        result.pop("rrf_score", None)
        result.pop("semantic_normalized_score", None)
        result.pop("bm25_normalized_score", None)
        result.pop("weighted_score", None)
        # Keep "normalized_score" and "original_score" for debugging if present
    
    return fused_results


def query_legal_documents(
    qdrant_client: QdrantClient,
    collection_name: str,
    query_text: str,
    embedding_model: str,
    chunking_method: str,
    source_file: Optional[str] = None,
    top_k: int = 5,
    search_mode: str = "semantic"
) -> List[Dict]:
    """
    Query legal documents with filtering by chunking method, embedding model, and optionally source file.
    Supports multiple search modes: semantic, BM25, and mixed (hybrid).
    
    Args:
        qdrant_client: Qdrant client instance
        collection_name: Collection name
        query_text: Query text
        embedding_model: Embedding model name (must match stored embeddings for semantic/mixed modes)
        chunking_method: Chunking method to filter by
        source_file: Optional source file name to filter by (None = all files)
        top_k: Number of results
        search_mode: Search mode - "semantic" (default), "bm25", or "mixed"
        
    Returns:
        List of retrieved chunks with metadata
    """
    if not LEGAL_CHUNKER_AVAILABLE:
        raise ImportError("Legal chunker not available")
    
    # Validate search mode
    if search_mode not in ["semantic", "bm25", "mixed"]:
        raise ValueError(f"Invalid search_mode: {search_mode}. Must be 'semantic', 'bm25', or 'mixed'")
    
    # Build filter dynamically (used by all search modes)
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    filter_conditions = [
        FieldCondition(
            key="chunking_method",
            match=MatchValue(value=chunking_method)
        )
    ]
    
    # For semantic/mixed modes, also filter by embedding model
    if search_mode in ["semantic", "mixed"]:
        filter_conditions.append(
            FieldCondition(
                key="embedding_model",
                match=MatchValue(value=embedding_model)
            )
        )
    
    # Add source_file filter if specified (not "All Files")
    if source_file and source_file != "All Files":
        filter_conditions.append(
            FieldCondition(
                key="source_file",
                match=MatchValue(value=source_file)
            )
        )
    
    filter_condition = Filter(must=filter_conditions)
    
    try:
        if search_mode == "bm25":
            # Pure BM25 keyword search
            if not BM25_AVAILABLE:
                raise ImportError("rank-bm25 required for BM25 search. Install with: pip install rank-bm25")
            
            return _bm25_search(
                qdrant_client=qdrant_client,
                collection_name=collection_name,
                query_text=query_text,
                filter_condition=filter_condition,
                top_k=top_k
            )
        
        elif search_mode == "semantic":
            # Pure semantic search (existing logic)
            # Generate query embedding
            if embedding_model.startswith("ollama/"):
                try:
                    from langchain_ollama import OllamaEmbeddings
                    import os
                    
                    ollama_model = embedding_model.replace("ollama/", "")
                    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                    
                    embeddings = OllamaEmbeddings(
                        model=ollama_model,
                        base_url=base_url
                    )
                    
                    query_vector = embeddings.embed_query(query_text)
                except ImportError:
                    raise ImportError("langchain-ollama required for Ollama embeddings. Install with: pip install langchain-ollama")
                except Exception as e:
                    raise RuntimeError(f"Error generating Ollama query embedding: {e}")
            else:
                # Use SentenceTransformer
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(embedding_model)
                query_vector = model.encode(query_text, convert_to_numpy=True, show_progress_bar=False).tolist()
            
            # Semantic search
            results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=filter_condition,
                limit=top_k * 2  # Get more for deduplication
            )
            
            # Normalize cosine similarity scores to [0, 1] range for consistency
            cosine_scores = [r.score for r in results]
            if cosine_scores:
                min_cosine = min(cosine_scores)
                max_cosine = max(cosine_scores)
                if max_cosine > min_cosine:
                    normalized_cosine = [(score - min_cosine) / (max_cosine - min_cosine) for score in cosine_scores]
                else:
                    normalized_cosine = [1.0] * len(cosine_scores)
            else:
                normalized_cosine = cosine_scores
            
            # Format results and deduplicate
            formatted_results = []
            seen_content = set()
            
            for result, norm_score in zip(results, normalized_cosine):
                payload = result.payload or {}
                chunk_text = payload.get("text", "").strip()
                
                # Normalize text more aggressively: lowercase, strip, normalize whitespace
                normalized_text = re.sub(r'\s+', ' ', chunk_text.lower().strip())
                if normalized_text in seen_content:
                    continue
                
                seen_content.add(normalized_text)
                
                formatted_results.append({
                    "text": chunk_text,
                    "score": float(norm_score),  # Normalized cosine similarity [0, 1]
                    "original_score": float(result.score),  # Original cosine similarity for reference
                    "chunking_method": payload.get("chunking_method"),
                    "embedding_model": payload.get("embedding_model"),
                    "hierarchy_level": payload.get("hierarchy_level"),
                    "clause_number": payload.get("clause_number"),
                    "title": payload.get("title"),
                    "page": payload.get("page"),
                    "parent": payload.get("parent"),
                    "source_file": payload.get("source_file") or payload.get("file_name"),
                    "file_id": payload.get("file_id"),
                    "file_name": payload.get("file_name"),
                    "individual_name": payload.get("individual_name"),
                    "company_name": payload.get("company_name"),
                    "address": payload.get("address"),
                    "email": payload.get("email"),
                    "phone": payload.get("phone")
                })
                
                if len(formatted_results) >= top_k:
                    break
            
            return formatted_results
        
        elif search_mode == "mixed":
            # Hybrid search: combine semantic + BM25 using RRF
            if not BM25_AVAILABLE:
                raise ImportError("rank-bm25 required for mixed search. Install with: pip install rank-bm25")
            
            # Perform both searches
            # 1. Semantic search
            if embedding_model.startswith("ollama/"):
                try:
                    from langchain_ollama import OllamaEmbeddings
                    import os
                    
                    ollama_model = embedding_model.replace("ollama/", "")
                    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                    
                    embeddings = OllamaEmbeddings(
                        model=ollama_model,
                        base_url=base_url
                    )
                    
                    query_vector = embeddings.embed_query(query_text)
                except ImportError:
                    raise ImportError("langchain-ollama required for Ollama embeddings. Install with: pip install langchain-ollama")
                except Exception as e:
                    raise RuntimeError(f"Error generating Ollama query embedding: {e}")
            else:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(embedding_model)
                query_vector = model.encode(query_text, convert_to_numpy=True, show_progress_bar=False).tolist()
            
            semantic_results_raw = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=filter_condition,
                limit=top_k * 2  # Get more for fusion
            )
            
            # Format semantic results with highlighting for mixed mode
            semantic_results = []
            cosine_scores = [r.score for r in semantic_results_raw]
            
            # Normalize cosine similarity scores to [0, 1] range
            # Cosine similarity is typically [-1, 1] or [0, 1] depending on normalization
            if cosine_scores:
                min_cosine = min(cosine_scores)
                max_cosine = max(cosine_scores)
                if max_cosine > min_cosine:
                    # Normalize to [0, 1]
                    normalized_cosine = [(score - min_cosine) / (max_cosine - min_cosine) for score in cosine_scores]
                else:
                    # All scores are the same
                    normalized_cosine = [1.0] * len(cosine_scores)
            else:
                normalized_cosine = cosine_scores
            
            for result, norm_score in zip(semantic_results_raw, normalized_cosine):
                payload = result.payload or {}
                chunk_text = payload.get("text", "").strip()
                
                # Highlight query terms for mixed mode display
                highlighted_text = _highlight_query_terms(chunk_text, query_text)
                
                semantic_results.append({
                    "text": chunk_text,  # Original text
                    "highlighted_text": highlighted_text,  # Highlighted version for display
                    "score": float(norm_score),  # Normalized cosine similarity [0, 1]
                    "original_score": float(result.score),  # Original cosine similarity for reference
                    "chunking_method": payload.get("chunking_method"),
                    "embedding_model": payload.get("embedding_model"),
                    "hierarchy_level": payload.get("hierarchy_level"),
                    "clause_number": payload.get("clause_number"),
                    "title": payload.get("title"),
                    "page": payload.get("page"),
                    "parent": payload.get("parent"),
                    "source_file": payload.get("source_file") or payload.get("file_name"),
                    "file_id": payload.get("file_id"),
                    "file_name": payload.get("file_name"),
                    "individual_name": payload.get("individual_name"),
                    "company_name": payload.get("company_name"),
                    "address": payload.get("address"),
                    "email": payload.get("email"),
                    "phone": payload.get("phone")
                })
            
            # BM25 search
            bm25_results = _bm25_search(
                qdrant_client=qdrant_client,
                collection_name=collection_name,
                query_text=query_text,
                filter_condition=filter_condition,
                top_k=top_k * 2  # Get more for fusion
            )
            
            # Fuse results using RRF
            fused_results = _reciprocal_rank_fusion(
                semantic_results=semantic_results,
                bm25_results=bm25_results,
                k=60  # Standard RRF constant
            )
            
            # Return top_k results
            return fused_results[:top_k]
        
    except Exception as e:
        logger.error(f"Error querying Qdrant with mode {search_mode}: {e}", exc_info=True)
        raise


def delete_file_from_qdrant(
    qdrant_client: QdrantClient,
    collection_name: str,
    source_file: str
) -> Dict[str, Any]:
    """
    Delete all chunks for a specific source file from Qdrant collection.
    
    Args:
        qdrant_client: Qdrant client instance
        collection_name: Collection name
        source_file: Source file name to delete
        
    Returns:
        Dictionary with deletion results: {"success": bool, "deleted_count": int, "message": str}
    """
    if not LEGAL_CHUNKER_AVAILABLE:
        return {
            "success": False,
            "deleted_count": 0,
            "message": "Legal chunker not available"
        }
    
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # Create filter for the specific file
        file_filter = Filter(
            must=[
                FieldCondition(
                    key="source_file",
                    match=MatchValue(value=source_file)
                )
            ]
        )
        
        # Count points before deletion (for reporting)
        scroll_result = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=file_filter,
            limit=10000,  # Large limit to count all
            with_payload=False,
            with_vectors=False
        )
        
        points_before, _ = scroll_result
        deleted_count = len(points_before) if points_before else 0
        
        if deleted_count == 0:
            return {
                "success": True,
                "deleted_count": 0,
                "message": f"No chunks found for file '{source_file}'"
            }
        
        # Delete points using filter (more efficient than collecting IDs)
        qdrant_client.delete(
            collection_name=collection_name,
            points_selector=file_filter,
            wait=True
        )
        
        logger.info(f"Deleted chunks for file '{source_file}' from collection '{collection_name}'")
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "message": f"Successfully deleted {deleted_count} chunk(s) for file '{source_file}'"
        }
        
    except Exception as e:
        logger.error(f"Error deleting file '{source_file}' from Qdrant: {e}")
        return {
            "success": False,
            "deleted_count": 0,
            "message": f"Error deleting file: {e}"
        }


def delete_all_files_from_qdrant(
    qdrant_client: QdrantClient,
    collection_name: str
) -> Dict[str, Any]:
    """
    Delete all chunks from Qdrant collection (delete all files).
    
    Args:
        qdrant_client: Qdrant client instance
        collection_name: Collection name
        
    Returns:
        Dictionary with deletion results: {"success": bool, "deleted_count": int, "message": str}
    """
    if not LEGAL_CHUNKER_AVAILABLE:
        return {
            "success": False,
            "deleted_count": 0,
            "message": "Legal chunker not available"
        }
    
    try:
        # Count points before deletion (for reporting)
        scroll_result = qdrant_client.scroll(
            collection_name=collection_name,
            limit=10000,  # Large limit to count
            with_payload=False,
            with_vectors=False
        )
        
        points, _ = scroll_result
        if not points:
            return {
                "success": True,
                "deleted_count": 0,
                "message": "Collection is already empty"
            }
        
        deleted_count = len(points)
        
        # Delete all points - scroll through and collect IDs, then delete
        # More efficient: use a filter that matches everything, or delete collection
        # For now, we'll scroll and delete by IDs in batches
        point_ids = []
        next_page_offset = None
        max_iterations = 100  # Safety limit
        
        for _ in range(max_iterations):
            scroll_result = qdrant_client.scroll(
                collection_name=collection_name,
                limit=1000,  # Process in batches
                offset=next_page_offset,
                with_payload=False,
                with_vectors=False
            )
            
            batch_points, next_page_offset = scroll_result
            
            if batch_points:
                batch_ids = [point.id for point in batch_points]
                # Delete this batch
                qdrant_client.delete(
                    collection_name=collection_name,
                    points_selector=batch_ids,
                    wait=True
                )
                point_ids.extend(batch_ids)
            
            # Break if no more pages
            if next_page_offset is None:
                break
        
        logger.info(f"Deleted all {len(point_ids)} chunks from collection '{collection_name}'")
        
        return {
            "success": True,
            "deleted_count": len(point_ids),
            "message": f"Successfully deleted all {len(point_ids)} chunk(s) from collection"
        }
        
    except Exception as e:
        logger.error(f"Error deleting all files from Qdrant: {e}")
        return {
            "success": False,
            "deleted_count": 0,
            "message": f"Error deleting all files: {e}"
        }


def query_legal_documents_with_reranking(
    qdrant_client: QdrantClient,
    collection_name: str,
    query_text: str,
    embedding_model: str,
    chunking_method: str,
    source_file: Optional[str] = None,
    top_k: int = 5,
    use_reranking: bool = True,
    retrieval_methods: Optional[List[str]] = None
) -> List[Dict]:
    """
    Query legal documents using advanced reranking system.
    
    This function uses the production-ready LegalRAGReranker which includes:
    - Multiple retrieval methods (semantic, BM25, n-gram)
    - Legal-specific optimizations (clause type detection, obligation patterns)
    - Llama 3 70B reranking via Together.ai (optional)
    - Ensemble methods (Reciprocal Rank Fusion)
    
    Args:
        qdrant_client: Qdrant client instance
        collection_name: Collection name
        query_text: Query text
        embedding_model: Embedding model name (for semantic search)
        chunking_method: Chunking method to filter by
        source_file: Optional source file name to filter by
        top_k: Number of results
        use_reranking: Whether to use Llama 3 reranking (requires TOGETHER_API_KEY)
        retrieval_methods: List of methods to use ["semantic", "bm25", "ngram"]
    
    Returns:
        List of retrieved chunks with advanced ranking and legal context
    """
    try:
        # Use importlib for modules starting with numbers
        import importlib
        reranker_module = importlib.import_module('scripts.04_reranking.legal_reranker')
        LegalRAGReranker = reranker_module.LegalRAGReranker
        from qdrant_client.models import Filter, FieldCondition, MatchValue
    except ImportError as e:
        logger.error(f"Advanced reranker not available: {e}. Falling back to standard query.")
        return query_legal_documents(
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            query_text=query_text,
            embedding_model=embedding_model,
            chunking_method=chunking_method,
            source_file=source_file,
            top_k=top_k,
            search_mode="mixed"
        )
    
    # Build filter condition
    filter_conditions = [
        FieldCondition(
            key="chunking_method",
            match=MatchValue(value=chunking_method)
        )
    ]
    
    if source_file and source_file != "All Files":
        filter_conditions.append(
            FieldCondition(
                key="source_file",
                match=MatchValue(value=source_file)
            )
        )
    
    filter_condition = Filter(must=filter_conditions) if filter_conditions else None
    
    # Initialize reranker
    together_api_key = os.getenv("TOGETHER_API_KEY")
    reranker = LegalRAGReranker(
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        embedding_model=embedding_model,
        together_api_key=together_api_key,
        enable_reranking=use_reranking and together_api_key is not None,
        enable_caching=True
    )
    
    # Set default retrieval methods
    if retrieval_methods is None:
        retrieval_methods = ["semantic", "bm25", "ngram"]
    
    # Retrieve and rerank
    ranked_chunks = reranker.retrieve_and_rerank(
        query=query_text,
        top_k=top_k,
        methods=retrieval_methods,
        filter_condition=filter_condition,
        use_reranking=use_reranking
    )
    
    # Convert RankedChunk objects to dict format compatible with existing code
    results = []
    for chunk in ranked_chunks:
        result_dict = {
            "text": chunk.content,
            "score": chunk.final_score,
            "base_score": chunk.base_score,
            "rerank_score": chunk.rerank_score,
            "chunking_method": chunk.metadata.get("chunking_method", chunking_method),
            "embedding_model": chunk.metadata.get("embedding_model", embedding_model),
            "hierarchy_level": chunk.metadata.get("hierarchy_level"),
            "clause_number": chunk.metadata.get("clause_number"),
            "title": chunk.metadata.get("title"),
            "page": chunk.metadata.get("page"),
            "parent": chunk.metadata.get("parent"),
            "source_file": chunk.metadata.get("source_file") or chunk.metadata.get("file_name"),
            "file_id": chunk.metadata.get("file_id"),
            "file_name": chunk.metadata.get("file_name"),
            "individual_name": chunk.metadata.get("individual_name"),
            "company_name": chunk.metadata.get("company_name"),
            "address": chunk.metadata.get("address"),
            "email": chunk.metadata.get("email"),
            "phone": chunk.metadata.get("phone"),
            # Legal context information
            "legal_context": {
                "clause_type": chunk.clause_type.name,
                "section_id": chunk.section_id,
                "party_relevance": chunk.party_relevance,
                "is_obligation": chunk.is_obligation,
                "retrieval_method": chunk.retrieval_method
            }
        }
        results.append(result_dict)
    
    logger.info(f"Retrieved {len(results)} chunks using advanced reranking")
    return results


def get_file_statistics(
    qdrant_client: QdrantClient,
    collection_name: str
) -> Dict[str, Any]:
    """
    Get comprehensive statistics about all uploaded files in Qdrant.
    
    Args:
        qdrant_client: Qdrant client instance
        collection_name: Qdrant collection name
        
    Returns:
        Dictionary with file statistics including:
        - total_files: Number of distinct files
        - total_chunks: Total number of chunks across all files
        - files: List of file statistics with:
            - source_file: File name (with extension for uniqueness)
            - chunk_count: Number of chunks for this file
            - chunking_methods: List of chunking methods used
            - chunking_method_counts: Dictionary mapping chunking method to chunk count
            - upload_timestamp: When the file was uploaded (ISO format)
            - upload_date: Human-readable upload date/time
            - embedding_model: Embedding model used
    """
    if not LEGAL_CHUNKER_AVAILABLE:
        raise ImportError("Legal chunker not available")
    
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # Scroll through all points to collect statistics
        all_files_stats = defaultdict(lambda: {
            "chunk_count": 0,
            "chunking_methods": set(),
            "chunking_method_counts": defaultdict(int),  # Track counts per chunking method
            "upload_timestamps": set(),
            "embedding_models": set()
        })
        
        # Scroll through all chunks
        offset = None
        while True:
            scroll_result = qdrant_client.scroll(
                collection_name=collection_name,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            if not points:
                break
            
            for point in points:
                payload = point.payload or {}
                source_file = payload.get("source_file") or payload.get("file_name")
                
                if not source_file:
                    continue
                
                # Update statistics for this file
                file_stats = all_files_stats[source_file]
                file_stats["chunk_count"] += 1
                
                # Track chunking methods and counts per method
                chunking_method = payload.get("chunking_method")
                if chunking_method:
                    file_stats["chunking_methods"].add(chunking_method)
                    file_stats["chunking_method_counts"][chunking_method] += 1
                
                # Track upload timestamp
                upload_timestamp = payload.get("upload_timestamp")
                if upload_timestamp:
                    file_stats["upload_timestamps"].add(upload_timestamp)
                
                # Track embedding model
                embedding_model = payload.get("embedding_model")
                if embedding_model:
                    file_stats["embedding_models"].add(embedding_model)
            
            # Get next offset
            offset = scroll_result[1]
            if offset is None:
                break
        
        # Format results
        files_list = []
        total_chunks = 0
        
        for source_file, stats in sorted(all_files_stats.items()):
            # Get the earliest upload timestamp (first upload)
            upload_timestamps = sorted(stats["upload_timestamps"]) if stats["upload_timestamps"] else []
            upload_timestamp = upload_timestamps[0] if upload_timestamps else None
            
            # Format upload date/time
            upload_date = "Unknown"
            if upload_timestamp:
                try:
                    # Handle ISO format with or without timezone
                    dt_str = upload_timestamp.replace('Z', '+00:00')
                    dt = datetime.fromisoformat(dt_str)
                    upload_date = dt.strftime("%Y-%m-%d %H:%M:%S")
                except (ValueError, AttributeError):
                    upload_date = upload_timestamp
            
            # Get embedding model (use first one found)
            embedding_model = list(stats["embedding_models"])[0] if stats["embedding_models"] else "Unknown"
            
            # Format chunking methods as comma-separated list
            chunking_methods_list = sorted(list(stats["chunking_methods"]))
            chunking_methods_str = ", ".join(chunking_methods_list) if chunking_methods_list else "Unknown"
            
            # Convert chunking_method_counts defaultdict to regular dict for JSON serialization
            chunking_method_counts_dict = dict(stats["chunking_method_counts"])
            
            files_list.append({
                "source_file": source_file,
                "chunk_count": stats["chunk_count"],
                "chunking_methods": chunking_methods_list,  # Keep as list for programmatic access
                "chunking_methods_str": chunking_methods_str,  # Add string version for display
                "chunking_method_counts": chunking_method_counts_dict,  # Dict: {method: count}
                "upload_timestamp": upload_timestamp,
                "upload_date": upload_date,
                "embedding_model": embedding_model
            })
            
            total_chunks += stats["chunk_count"]
        
        return {
            "total_files": len(files_list),
            "total_chunks": total_chunks,
            "files": files_list
        }
        
    except Exception as e:
        logger.error(f"Error getting file statistics: {e}", exc_info=True)
        raise


def get_chunks_for_exploration(
    qdrant_client: QdrantClient,
    collection_name: str,
    source_file: str,
    chunking_method: str,
    chunk_level: Optional[int] = None,
    limit: int = 10000
) -> List[Dict[str, Any]]:
    """
    Get all chunks for a specific document and chunking method for exploration.
    
    Args:
        qdrant_client: Qdrant client instance
        collection_name: Qdrant collection name
        source_file: Source file name (with extension for uniqueness)
        chunking_method: Chunking method to filter by
        chunk_level: Optional chunk level filter (1, 2, or 3). If None, returns all levels.
        limit: Maximum number of chunks to retrieve (default: 10000)
        
    Returns:
        List of chunk dictionaries with all metadata, sorted by chunk_number
    """
    if not LEGAL_CHUNKER_AVAILABLE:
        raise ImportError("Legal chunker not available")
    
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # Build filter for source_file and chunking_method
        filter_conditions = [
            FieldCondition(
                key="source_file",
                match=MatchValue(value=source_file)
            ),
            FieldCondition(
                key="chunking_method",
                match=MatchValue(value=chunking_method)
            )
        ]
        
        # Add chunk_level filter if specified
        if chunk_level is not None:
            filter_conditions.append(
                FieldCondition(
                    key="chunk_level",
                    match=MatchValue(value=chunk_level)
                )
            )
        
        filter_condition = Filter(must=filter_conditions)
        
        # Scroll through all matching chunks
        chunks = []
        offset = None
        
        while True:
            scroll_result = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=filter_condition,
                limit=min(1000, limit - len(chunks)),
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            if not points:
                break
            
            for point in points:
                payload = point.payload or {}
                
                # Extract chunk information
                chunk_data = {
                    "id": str(point.id),
                    "text": payload.get("text", ""),
                    "chunk_number": payload.get("chunk_number"),  # May not exist for older chunks
                    "chunk_index": payload.get("chunk_index"),  # Fallback
                    "chunk_level": payload.get("chunk_level", payload.get("hierarchy_level", 1)),  # Level 1, 2, or 3
                    "parent_chunk_number": payload.get("parent_chunk_number"),  # Parent chunk's chunk_number
                    "text_preview": payload.get("text_preview", ""),  # First 200 chars for quick preview
                    "preserved_full_text": payload.get("preserved_full_text", False),  # True if chunk came from fallback/intro/outro
                    "chunking_method": payload.get("chunking_method", "unknown"),
                    "embedding_model": payload.get("embedding_model", "unknown"),
                    "source_file": payload.get("source_file") or payload.get("file_name", "unknown"),
                    "source_filename": payload.get("source_filename") or payload.get("source_file") or payload.get("file_name", "unknown"),
                    "document_id": payload.get("document_id"),  # Unique document identifier
                    "hierarchy_level": payload.get("hierarchy_level"),
                    "clause_number": payload.get("clause_number"),
                    "title": payload.get("title"),
                    "page": payload.get("page"),
                    "parent": payload.get("parent"),  # Parent clause ID (for backward compatibility)
                    "char_count": payload.get("char_count"),
                    "upload_timestamp": payload.get("upload_timestamp") or payload.get("upload_time"),
                    "file_id": payload.get("file_id"),
                    "file_name": payload.get("file_name"),
                    # NER entities
                    "individual_name": payload.get("individual_name"),
                    "company_name": payload.get("company_name"),
                    "address": payload.get("address"),
                    "email": payload.get("email"),
                    "phone": payload.get("phone"),
                    # All other metadata
                    "metadata": {k: v for k, v in payload.items() if k not in [
                        "text", "chunk_number", "chunk_index", "chunk_level", "parent_chunk_number",
                        "text_preview", "preserved_full_text", "chunking_method", "embedding_model", 
                        "source_file", "source_filename", "document_id",
                        "file_name", "hierarchy_level", "clause_number", "title", "page", 
                        "parent", "char_count", "upload_timestamp", "upload_time", "file_id", 
                        "individual_name", "company_name", "address", "email", "phone"
                    ]}
                }
                
                # Use chunk_number if available, otherwise use chunk_index
                # Do NOT use Qdrant Point ID as fallback - it's a hash, not a sequential number
                # Also check if chunk_number looks like a Qdrant Point ID (very large number > 1M)
                chunk_num = chunk_data.get("chunk_number")
                if chunk_num is not None:
                    try:
                        # If chunk_number is unreasonably large, it's probably a Qdrant Point ID
                        # Chunk numbers should be small sequential integers (0, 1, 2, ...)
                        if int(chunk_num) > 1000000:
                            chunk_data["chunk_number"] = None  # Treat as missing
                    except (ValueError, TypeError):
                        pass
                
                if chunk_data["chunk_number"] is None:
                    if chunk_data["chunk_index"] is not None:
                        chunk_index_val = chunk_data["chunk_index"]
                        # Also check chunk_index isn't a Qdrant Point ID
                        try:
                            if int(chunk_index_val) <= 1000000:
                                chunk_data["chunk_number"] = chunk_index_val
                        except (ValueError, TypeError):
                            pass
                    # If both are None, leave it as None - we'll set it based on sorted position later
                
                chunks.append(chunk_data)
            
            # Check if we've reached the limit
            if len(chunks) >= limit:
                break
            
            # Get next offset
            offset = scroll_result[1]
            if offset is None:
                break
        
        # Sort by chunk_number globally (unique across all files)
        # This ensures chunks are ordered by import order and position within documents
        try:
            chunks.sort(key=lambda x: (
                x.get("chunk_number") if x.get("chunk_number") is not None else float('inf'),
                x.get("page") if x.get("page") is not None else float('inf'),
                x.get("id")
            ))
        except (TypeError, ValueError):
            # If sorting fails, just sort by id
            chunks.sort(key=lambda x: x.get("id", ""))
        
        # For chunks missing chunk_number, assign sequential numbers based on sorted position
        # This handles legacy chunks that were uploaded before chunk_number was added
        # Find the maximum chunk_number to continue numbering from there
        # Skip chunk_numbers that look like Qdrant Point IDs (> 1M)
        max_chunk_number = -1
        for chunk in chunks:
            chunk_num = chunk.get("chunk_number")
            if chunk_num is not None and isinstance(chunk_num, (int, float)):
                try:
                    chunk_num_int = int(chunk_num)
                    # Only consider valid chunk_numbers (small sequential integers)
                    if chunk_num_int <= 1000000:
                        max_chunk_number = max(max_chunk_number, chunk_num_int)
                except (ValueError, TypeError):
                    pass
        
        # Assign sequential numbers to chunks missing chunk_number
        next_number = max_chunk_number + 1
        for chunk in chunks:
            if chunk.get("chunk_number") is None:
                chunk["chunk_number"] = next_number
                next_number += 1
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error getting chunks for exploration: {e}", exc_info=True)
        raise

