"""
Shared RAG Pipeline Module
Canonical implementation for document ingestion, chunking, and Qdrant indexing.
Used by both streamlit_app.py and app.py to ensure consistency.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any, TYPE_CHECKING
from datetime import datetime

logger = logging.getLogger(__name__)

# Import legal chunker components
try:
    # Use importlib for modules starting with numbers
    import importlib
    qdrant_chunker_module = importlib.import_module('scripts.00_chunking.qdrant_chunker')
    LegalDocumentChunker = qdrant_chunker_module.LegalDocumentChunker
    read_pdf = qdrant_chunker_module.read_pdf
    read_text = qdrant_chunker_module.read_text
    read_docx = qdrant_chunker_module.read_docx
    extract_ner_entities = qdrant_chunker_module.extract_ner_entities
    embed_texts = qdrant_chunker_module.embed_texts
    create_collection = qdrant_chunker_module.create_collection
    upsert = qdrant_chunker_module.upsert
    from qdrant_client import QdrantClient
    LEGAL_CHUNKER_AVAILABLE = True
except ImportError as e:
    LEGAL_CHUNKER_AVAILABLE = False
    logger.warning(f"Legal chunker not available: {e}")
    # Define QdrantClient for type hints if import fails
    if TYPE_CHECKING:
        from typing import Any as QdrantClient
    else:
        QdrantClient = Any

# Import chunking utilities
try:
    # Use importlib for modules starting with numbers
    legal_chunker_module = importlib.import_module('scripts.00_chunking.legal_chunker_integration')
    get_available_chunking_methods = legal_chunker_module.get_available_chunking_methods
    get_default_chunking_methods = legal_chunker_module.get_default_chunking_methods
    get_max_chunk_number = legal_chunker_module.get_max_chunk_number
    CHUNKING_LOG_FILE = legal_chunker_module.CHUNKING_LOG_FILE
except ImportError:
    logger.warning("Could not import chunking utilities")
    get_available_chunking_methods = lambda: ["recursive"]
    get_default_chunking_methods = lambda: ["recursive"]
    get_max_chunk_number = lambda client, collection: 0
    CHUNKING_LOG_FILE = "chunking_log.txt"


def _write_chunking_log(
    timestamp: str,
    document_name: str,
    chunk_id: str,
    chunk_index: int,
    chunking_method: str,
    status: str,
    error_message: Optional[str] = None
) -> None:
    """Write chunking operation to log file."""
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


def ingest_documents_to_qdrant(
    file_paths: List[str],
    qdrant_client: QdrantClient,
    collection_name: str,
    embedding_model: str,
    chunking_methods: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None
) -> Dict:
    """
    Ingest multiple documents to Qdrant using the same pipeline as streamlit_app.py.
    
    This is the canonical ingestion function that ensures consistency across apps.
    
    Args:
        file_paths: List of paths to documents (PDF, DOCX, MD, TXT)
        qdrant_client: Qdrant client instance
        collection_name: Qdrant collection name
        embedding_model: Embedding model name (with or without "ollama/" prefix)
        chunking_methods: List of chunking methods to use (default: all available)
        progress_callback: Optional callback(file_name, file_index, total_files, status)
        
    Returns:
        Dictionary with ingestion results:
        {
            "success": bool,
            "total_files": int,
            "total_chunks": int,
            "files_processed": List[Dict],
            "method_counts": Dict[str, int],
            "errors": List[str]
        }
    """
    if not LEGAL_CHUNKER_AVAILABLE:
        raise ImportError("Legal chunker not available. Install dependencies.")
    
    # Handle "ollama/" prefix
    model_for_chunker = embedding_model.replace("ollama/", "") if embedding_model.startswith("ollama/") else embedding_model
    
    # Get chunking methods
    if chunking_methods is None:
        chunking_methods = get_default_chunking_methods()
    
    # Validate methods
    available_methods = get_available_chunking_methods()
    for method in chunking_methods:
        if method not in available_methods:
            logger.warning(f"Unknown chunking method: {method}. Available: {available_methods}")
            # Remove invalid method instead of failing
            chunking_methods = [m for m in chunking_methods if m in available_methods]
    
    if not chunking_methods:
        raise ValueError(f"No valid chunking methods. Available: {available_methods}")
    
    # Ensure collection exists (we'll create it after generating embeddings to get the dimension)
    # For now, we'll generate a test embedding to get the dimension
    try:
        # Generate a test embedding to determine vector dimension
        test_embedding = embed_texts(["test"], model_for_chunker)
        vector_dim = len(test_embedding[0]) if test_embedding else 384  # Default to 384 if unknown
        
        create_collection(
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            dim=vector_dim
        )
    except Exception as e:
        logger.warning(f"Collection creation/verification failed (may already exist): {e}")
    
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
    
    # Track current chunk_number (will increment for each chunk across all files and methods)
    current_chunk_number = start_chunk_number
    
    # Process all files
    all_chunks = []
    method_counts = {}
    files_processed = []
    errors = []
    total_files = len(file_paths)
    
    for file_index, file_path in enumerate(file_paths):
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            error_msg = f"File not found: {file_path}"
            errors.append(error_msg)
            logger.error(error_msg)
            continue
        
        file_name = file_path_obj.name
        
        if progress_callback:
            progress_callback(file_name, file_index, total_files, "reading")
        
        try:
            # Read file using same methods as streamlit_app.py
            suffix = file_path_obj.suffix.lower()
            if suffix == '.pdf':
                text, page_numbers = read_pdf(file_path_obj)
            elif suffix in ['.docx', '.doc']:
                text, page_numbers = read_docx(file_path_obj)
            else:
                text, page_numbers = read_text(file_path_obj)
            
            logger.info(f"Read {file_name}: {len(text)} characters")
            
            # Extract document-level NER entities
            doc_ner_entities = extract_ner_entities(text)
            
            # Initialize chunker
            chunker = LegalDocumentChunker(embedding_model=model_for_chunker)
            
            # Generate unique document_id for this file
            import hashlib
            upload_timestamp = datetime.now().isoformat()
            document_id_string = f"{file_name}::{upload_timestamp}"
            document_id = hashlib.md5(document_id_string.encode('utf-8')).hexdigest()[:16]
            
            # Process each chunking method
            file_chunks = []
            
            for method_index, method in enumerate(chunking_methods):
                if progress_callback:
                    progress_callback(file_name, file_index, total_files, f"chunking_{method}")
                
                method_start_time = datetime.now().isoformat()
                method_status = "APPLIED"
                method_error = None
                
                try:
                    # Check if semantic chunking is available
                    if method == "semantic":
                        try:
                            from langchain_experimental.text_splitter import SemanticChunker
                            if SemanticChunker is None:
                                raise ImportError("SemanticChunker not available")
                        except (ImportError, AttributeError):
                            method_status = "SKIPPED"
                            method_error = "langchain-experimental not installed"
                            logger.warning(f"Semantic chunking skipped: {method_error}")
                            _write_chunking_log(
                                method_start_time,
                                file_name,
                                "N/A",
                                0,
                                method,
                                method_status,
                                method_error
                            )
                            continue
                    
                    # Chunk document
                    chunks = chunker.chunk(text, method, page_numbers, ner_entities=None)
                    
                    if not chunks:
                        logger.warning(f"No chunks generated for {file_name} with method: {method}")
                        method_status = "SKIPPED"
                        method_error = "No chunks generated"
                        _write_chunking_log(
                            method_start_time,
                            file_name,
                            "N/A",
                            0,
                            method,
                            method_status,
                            method_error
                        )
                        continue
                    
                    # First pass: assign chunk_numbers and build clause_number to chunk_number mapping for structural chunks
                    clause_to_chunk_number = {}  # Maps clause_number -> chunk_number for parent lookups
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        chunk_id = chunk.get("id", f"chunk_{chunk_idx}")
                        
                        # Ensure chunking_method is set
                        if chunk.get("chunking_method") != method:
                            chunk["chunking_method"] = method
                        
                        # Assign global sequential chunk_number (unique across all files)
                        # Increment for each chunk to ensure uniqueness
                        current_chunk_number += 1
                        chunk["chunk_number"] = current_chunk_number
                        
                        # For structural chunks, map clause_number to chunk_number for parent lookups
                        if method == "structural" and chunk.get("clause_number"):
                            clause_to_chunk_number[chunk.get("clause_number")] = current_chunk_number
                    
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
                    
                    # Process each chunk: extract entities and set metadata
                    for chunk_idx, chunk in enumerate(chunks):
                        chunk_id = chunk.get("id", f"chunk_{chunk_idx}")
                        
                        # Extract chunk-specific entities
                        chunk_text = chunk.get("text", "")
                        chunk_ner_entities = extract_ner_entities(chunk_text)
                        
                        # Filter entities to only those present in chunk text
                        filtered_entities = {}
                        chunk_text_lower = chunk_text.lower()
                        
                        for key, value in chunk_ner_entities.items():
                            if value and value.lower() in chunk_text_lower:
                                filtered_entities[key] = value
                        
                        # Fallback to document-level entities if no chunk-specific entities
                        if not any(filtered_entities.values()):
                            for key, value in doc_ner_entities.items():
                                if value and value.lower() in chunk_text_lower:
                                    filtered_entities[key] = value
                        
                        # Add entities to chunk
                        if filtered_entities:
                            for key, value in filtered_entities.items():
                                if value:
                                    chunk[key] = value
                        
                        # Set file metadata (same as streamlit_app.py)
                        chunk["source_file"] = file_name
                        chunk["source_filename"] = file_name  # Also store as source_filename for consistency
                        chunk["document_id"] = document_id  # Unique document identifier
                        chunk["upload_timestamp"] = upload_timestamp
                        chunk["upload_time"] = upload_timestamp
                        chunk["file_name"] = file_name
                        chunk["chunking_method"] = method
                        
                        # Log chunk
                        _write_chunking_log(
                            method_start_time,
                            file_name,
                            str(chunk_id),
                            chunk_idx,
                            method,
                            "APPLIED"
                        )
                    
                    file_chunks.extend(chunks)
                    method_counts[method] = method_counts.get(method, 0) + len(chunks)
                    logger.info(f"  ✅ {file_name}: {len(chunks)} chunks with {method}")
                    
                except Exception as e:
                    method_status = "ERROR"
                    method_error = str(e)
                    logger.error(f"Error chunking {file_name} with {method}: {e}")
                    _write_chunking_log(
                        method_start_time,
                        file_name,
                        "N/A",
                        0,
                        method,
                        method_status,
                        method_error
                    )
            
            # Store chunks in Qdrant (same as streamlit_app.py)
            if file_chunks:
                if progress_callback:
                    progress_callback(file_name, file_index, total_files, "storing")
                
                try:
                    # Generate embeddings for chunks
                    logger.info(f"Generating embeddings for {len(file_chunks)} chunks from {file_name}...")
                    texts = [c['text'] for c in file_chunks]
                    embeddings = embed_texts(texts, model_for_chunker)
                    
                    # Upsert to Qdrant (using positional arguments as per qdrant_chunker.py signature)
                    upsert(qdrant_client, collection_name, file_chunks, embeddings)
                    
                    files_processed.append({
                        "file_name": file_name,
                        "chunks": len(file_chunks),
                        "methods": chunking_methods,
                        "method_counts": {m: method_counts.get(m, 0) for m in chunking_methods}
                    })
                    
                    all_chunks.extend(file_chunks)
                    logger.info(f"✅ Stored {len(file_chunks)} chunks for {file_name}")
                    
                except Exception as e:
                    error_msg = f"Error storing {file_name} in Qdrant: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            else:
                error_msg = f"No chunks generated for {file_name}"
                errors.append(error_msg)
                logger.warning(error_msg)
                
        except Exception as e:
            error_msg = f"Error processing {file_name}: {e}"
            errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
    
    return {
        "success": len(files_processed) > 0,
        "total_files": total_files,
        "total_chunks": len(all_chunks),
        "files_processed": files_processed,
        "method_counts": method_counts,
        "errors": errors,
        "embedding_model": embedding_model,
        "chunking_methods": chunking_methods
    }


def load_documents_from_directory(
    source_dir: str,
    file_extensions: List[str] = [".pdf", ".md", ".docx", ".txt"]
) -> List[str]:
    """
    Get list of document file paths from directory.
    Returns paths that can be passed to ingest_documents_to_qdrant().
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    file_paths = []
    for ext in file_extensions:
        for file_path in source_path.rglob(f"*{ext}"):
            if file_path.is_file():
                file_paths.append(str(file_path))
    
    return sorted(file_paths)

