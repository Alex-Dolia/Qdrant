"""
Structural Chunking Module
Standalone implementation for hierarchical document chunking.

Features:
- Three-level hierarchy (Level 1: Sections, Level 2: Subclauses, Level 3: Semantic Units)
- Preserves all text (intro, clauses, outro)
- Recursive fallback for unstructured documents
- Parent-child relationships between chunks
- Ready for Qdrant insertion with metadata
"""

import re
import logging
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# Qdrant imports
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest_models
    QDRANT_AVAILABLE = True
except ImportError:
    QdrantClient = None
    rest_models = None
    QDRANT_AVAILABLE = False
    logger.warning("qdrant-client not available. Qdrant functions will be disabled.")

# Constants
TARGET_CHUNK_WORDS = 300
MAX_SUBCLAUSE_WORDS = 500
CHUNK_OVERLAP_PERCENT = 0.15  # 15% overlap for recursive fallback
MIN_CHUNK_SIZE = 50  # Minimum chunk size in characters
MAX_CHUNK_LENGTH = 500  # Maximum chunk length in characters for recursive splitting
DEFAULT_VECTOR_DIM = 1536  # Default vector dimension for placeholder embeddings


def approx_word_count(text: str) -> int:
    """Approximate word count."""
    return len(text.split())


def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Simple sentence splitting by periods, exclamation, question marks
    sentences = re.split(r'[.!?]+\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def split_document(document_text: str) -> List[str]:
    """
    Initial document splitting by double newlines (\n\n).
    
    Args:
        document_text: Full document text
    
    Returns:
        List of initial text segments
    """
    # Split by double newlines (paragraphs)
    segments = re.split(r'\n\n+', document_text)
    
    # Clean up segments (remove empty/whitespace-only)
    cleaned_segments = [s.strip() for s in segments if s.strip()]
    
    return cleaned_segments


def recursive_split(chunk_text: str, max_length: int = MAX_CHUNK_LENGTH) -> List[str]:
    """
    Recursively split chunks that exceed max_length.
    
    Strategy:
    1. If chunk <= max_length, return as-is
    2. Otherwise, split by paragraphs (\n\n)
    3. If still too long, split by sentences
    4. If still too long, split by words with overlap
    
    Args:
        chunk_text: Text chunk to split
        max_length: Maximum chunk length in characters
    
    Returns:
        List of split chunk texts
    """
    if len(chunk_text) <= max_length:
        return [chunk_text]
    
    # Step 1: Try splitting by paragraphs
    paragraphs = re.split(r'\n\n+', chunk_text)
    if len(paragraphs) > 1:
        chunks = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(para) <= max_length:
                chunks.append(para)
            else:
                # Paragraph too long, split further
                chunks.extend(recursive_split(para, max_length))
        return chunks
    
    # Step 2: Try splitting by sentences
    sentences = split_sentences(chunk_text)
    if len(sentences) > 1:
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length <= max_length or not current_chunk:
                current_chunk.append(sentence)
                current_length += sentence_length + 1  # +1 for space
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
        
        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # If any chunk is still too long, recursively split it
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > max_length:
                final_chunks.extend(recursive_split(chunk, max_length))
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    # Step 3: Split by words with overlap (last resort)
    return recursive_chunk_with_overlap(chunk_text, chunk_size=approx_word_count(chunk_text[:max_length]))


def recursive_chunk_with_overlap(
    text: str,
    chunk_size: int = 500,
    overlap_percent: float = 0.15
) -> List[str]:
    """
    Recursive chunking with overlap for unstructured text.
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in words
        overlap_percent: Overlap percentage (0.15 = 15%)
    
    Returns:
        List of chunk texts
    """
    words = text.split()
    if not words:
        return []
    
    chunks = []
    overlap_size = int(chunk_size * overlap_percent)
    step_size = chunk_size - overlap_size
    
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        if chunk_words:
            chunk_text = " ".join(chunk_words)
            chunks.append(chunk_text)
        i += step_size
    
    return chunks


def detect_level1_sections(text: str) -> List[Dict[str, Any]]:
    """
    Detect Level 1 sections (top-level clauses/sections).
    
    Patterns:
    - Numbered clauses: "1. Title", "2. Title"
    - Section headers: "SECTION 1", "Clause 1"
    - Uppercase titles (fallback)
    
    Returns:
        List of dicts with 'start', 'end', 'number', 'title'
    """
    matches = []
    
    # Pattern 1: Numbered clauses at start of line (1., 2., etc.)
    pat1 = re.compile(r'^\s*(\d{1,2})\.\s+(.+?)(?=\n|$)', re.MULTILINE)
    for m in pat1.finditer(text):
        matches.append({
            "start": m.start(),
            "number": m.group(1),
            "title": m.group(2).strip(),
            "pattern": 1
        })
    
    # Pattern 2: SECTION/Clause headers
    pat2 = re.compile(r'^\s*(SECTION|Clause|CLAUSE)\s+(\d+)[:\.]?\s*(.+?)(?=\n|$)', re.MULTILINE | re.IGNORECASE)
    for m in pat2.finditer(text):
        matches.append({
            "start": m.start(),
            "number": m.group(2),
            "title": m.group(3).strip() if m.group(3) else "",
            "pattern": 2
        })
    
    # Pattern 3: All-caps headings (only if no other matches)
    if not matches:
        pat3 = re.compile(r'^([A-Z][A-Z\s]{5,80})(?=\n|$)', re.MULTILINE)
        for m in pat3.finditer(text):
            line = m.group(1).strip()
            # Only consider if reasonable length and doesn't look like regular text
            if 10 <= len(line) <= 80 and not line.endswith(('.', '!', '?', ':', ';')):
                matches.append({
                    "start": m.start(),
                    "number": None,
                    "title": line,
                    "pattern": 3
                })
    
    # If no matches, return fallback covering entire document
    if not matches:
        return [{"start": 0, "end": len(text), "number": None, "title": None}]
    
    # Sort by position
    matches.sort(key=lambda x: x["start"])
    
    # Build result with proper boundaries
    result = []
    for i, m in enumerate(matches):
        start = m["start"]
        end = matches[i+1]["start"] if i+1 < len(matches) else len(text)
        result.append({
            "start": start,
            "end": end,
            "number": m["number"],
            "title": m["title"]
        })
    
    return result


def detect_level2_subclauses(text: str) -> List[Dict[str, Any]]:
    """
    Detect Level 2 subclauses within a Level 1 section.
    
    Patterns:
    - Numbered subclauses: "1.1", "1.2", "2.1.3"
    - Lettered subclauses: "(a)", "(b)", "(i)"
    
    Returns:
        List of dicts with 'start', 'end', 'label'
    """
    matches = []
    
    # Pattern 1: Numbered subclauses (1.1, 1.2, 2.1.3, etc.)
    pat_num = re.compile(r'^\s*(\d+\.\d+(?:\.\d+)*)\s+', re.MULTILINE)
    for m in pat_num.finditer(text):
        matches.append((m.start(), m.group(1)))
    
    # Pattern 2: Lettered subclauses ((a), (b), (i), etc.)
    pat_letter = re.compile(r'^\s*\(([a-zA-Z0-9]+)\)\s+', re.MULTILINE)
    for m in pat_letter.finditer(text):
        matches.append((m.start(), f"({m.group(1)})"))
    
    if not matches:
        return [{"start": 0, "end": len(text), "label": None}]
    
    # Sort by position
    matches.sort(key=lambda x: x[0])
    
    # Build result with boundaries
    subs = []
    for i, (s, label) in enumerate(matches):
        e = matches[i+1][0] if i+1 < len(matches) else len(text)
        subs.append({"start": s, "end": e, "label": label})
    
    return subs


def semantic_split_level3(text: str, target_words: int = TARGET_CHUNK_WORDS) -> List[str]:
    """
    Split long text into Level 3 semantic units (sentences/paragraphs).
    
    Args:
        text: Text to split
        target_words: Target words per chunk
    
    Returns:
        List of semantic unit texts
    """
    sentences = split_sentences(text)
    if not sentences:
        return [text] if text.strip() else []
    
    units = []
    current_unit = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = approx_word_count(sentence)
        
        if current_word_count + sentence_words <= target_words or not current_unit:
            current_unit.append(sentence)
            current_word_count += sentence_words
        else:
            # Save current unit and start new one
            if current_unit:
                units.append(" ".join(current_unit))
            current_unit = [sentence]
            current_word_count = sentence_words
    
    # Add final unit
    if current_unit:
        units.append(" ".join(current_unit))
    
    return units if units else [text]


def structural_chunk_document(
    text: str,
    document_id: str,
    source_filename: str,
    existing_max_chunk_number: int = 0
) -> List[Dict[str, Any]]:
    """
    Perform structural chunking on a document with three-level hierarchy.
    
    Args:
        text: Full document text
        document_id: Unique document identifier (UUID)
        source_filename: Original filename
        existing_max_chunk_number: Maximum existing chunk_number from Qdrant (for global sequencing)
    
    Returns:
        List of chunk dictionaries ready for Qdrant insertion:
        {
            "chunk_number": int,              # Unique, sequential globally
            "chunk_level": int,                # 1, 2, or 3
            "parent_chunk_number": int|None,   # Parent's chunk_number (None for Level 1)
            "text": str,                       # Full chunk text
            "metadata": {
                "document_id": str,
                "source_filename": str,
                "text_preview": str            # First 120 chars
            }
        }
    """
    chunks = []
    current_chunk_number = existing_max_chunk_number
    
    # Step 1: Detect Level 1 sections
    level1_sections = detect_level1_sections(text)
    
    # Check if we have real structure (numbered sections)
    has_real_structure = any(s.get('number') is not None for s in level1_sections)
    
    # Step 2: If no structure, use recursive fallback
    if not has_real_structure:
        logger.info("No structural markers found, using recursive chunking fallback")
        recursive_chunks = recursive_chunk_with_overlap(
            text,
            chunk_size=500,
            overlap_percent=CHUNK_OVERLAP_PERCENT
        )
        
        for chunk_text in recursive_chunks:
            if not chunk_text.strip() or len(chunk_text.strip()) < MIN_CHUNK_SIZE:
                continue
            
            current_chunk_number += 1
            chunks.append({
                "chunk_number": current_chunk_number,
                "chunk_level": 1,
                "parent_chunk_number": None,
                "text": chunk_text.strip(),
                "metadata": {
                    "document_id": document_id,
                    "source_filename": source_filename,
                    "text_preview": chunk_text.strip()[:120]
                }
            })
        
        return chunks
    
    # Step 3: Process intro text (before first section)
    first_section_start = level1_sections[0]['start'] if level1_sections else len(text)
    if first_section_start > 0:
        intro_text = text[:first_section_start].strip()
        if intro_text and len(intro_text) >= MIN_CHUNK_SIZE:
            current_chunk_number += 1
            chunks.append({
                "chunk_number": current_chunk_number,
                "chunk_level": 1,
                "parent_chunk_number": None,
                "text": intro_text,
                "metadata": {
                    "document_id": document_id,
                    "source_filename": source_filename,
                    "text_preview": intro_text[:120]
                }
            })
    
    # Step 4: Process each Level 1 section
    for section_idx, section in enumerate(level1_sections):
        section_text = text[section['start']:section['end']].strip()
        if not section_text or len(section_text) < MIN_CHUNK_SIZE:
            continue
        
        section_number = section.get('number') or f"section_{section_idx + 1}"
        section_title = section.get('title', '')
        
        # Create Level 1 chunk for the entire section
        current_chunk_number += 1
        level1_chunk_number = current_chunk_number
        
        level1_chunk = {
            "chunk_number": level1_chunk_number,
            "chunk_level": 1,
            "parent_chunk_number": None,
            "text": section_text,
            "metadata": {
                "document_id": document_id,
                "source_filename": source_filename,
                "text_preview": section_text[:120],
                "section_number": section_number,
                "section_title": section_title
            }
        }
        chunks.append(level1_chunk)
        
        # Step 5: Detect Level 2 subclauses within this section
        level2_subclauses = detect_level2_subclauses(section_text)
        
        # Check if we have real subclauses (not just fallback)
        has_real_subclauses = any(s.get('label') is not None for s in level2_subclauses)
        
        if has_real_subclauses and len(level2_subclauses) > 1:
            # Process each subclause
            for sub_idx, subclause in enumerate(level2_subclauses):
                sub_text = section_text[subclause['start']:subclause['end']].strip()
                if not sub_text or len(sub_text) < MIN_CHUNK_SIZE:
                    continue
                
                sub_label = subclause.get('label') or f"{section_number}.{sub_idx + 1}"
                word_count = approx_word_count(sub_text)
                
                if word_count <= MAX_SUBCLAUSE_WORDS:
                    # Keep as Level 2 chunk
                    current_chunk_number += 1
                    chunks.append({
                        "chunk_number": current_chunk_number,
                        "chunk_level": 2,
                        "parent_chunk_number": level1_chunk_number,
                        "text": sub_text,
                        "metadata": {
                            "document_id": document_id,
                            "source_filename": source_filename,
                            "text_preview": sub_text[:120],
                            "subclause_label": sub_label
                        }
                    })
                else:
                    # Split into Level 3 semantic units
                    level3_units = semantic_split_level3(sub_text, TARGET_CHUNK_WORDS)
                    for unit_idx, unit_text in enumerate(level3_units):
                        if not unit_text.strip() or len(unit_text.strip()) < MIN_CHUNK_SIZE:
                            continue
                        
                        current_chunk_number += 1
                        chunks.append({
                            "chunk_number": current_chunk_number,
                            "chunk_level": 3,
                            "parent_chunk_number": level1_chunk_number,  # Parent is Level 1, not Level 2
                            "text": unit_text.strip(),
                            "metadata": {
                                "document_id": document_id,
                                "source_filename": source_filename,
                                "text_preview": unit_text.strip()[:120],
                                "subclause_label": sub_label,
                                "unit_index": unit_idx + 1
                            }
                        })
        
        # If no subclauses or section is short, keep Level 1 chunk as-is
    
    # Step 6: Process outro text (after last section)
    if level1_sections:
        last_section_end = level1_sections[-1]['end']
        if last_section_end < len(text):
            outro_text = text[last_section_end:].strip()
            if outro_text and len(outro_text) >= MIN_CHUNK_SIZE:
                current_chunk_number += 1
                chunks.append({
                    "chunk_number": current_chunk_number,
                    "chunk_level": 1,
                    "parent_chunk_number": None,
                    "text": outro_text,
                    "metadata": {
                        "document_id": document_id,
                        "source_filename": source_filename,
                        "text_preview": outro_text[:120]
                    }
                })
    
    # Step 7: Verify all text is covered (safety check)
    covered_positions = set()
    for chunk in chunks:
        chunk_text = chunk["text"]
        # Find position in original text
        start_pos = text.find(chunk_text)
        if start_pos >= 0:
            for i in range(start_pos, start_pos + len(chunk_text)):
                covered_positions.add(i)
    
    # Check for gaps
    uncovered_ranges = []
    in_gap = False
    gap_start = 0
    
    for i in range(len(text)):
        if i not in covered_positions:
            if not in_gap:
                gap_start = i
                in_gap = True
        else:
            if in_gap:
                uncovered_ranges.append((gap_start, i))
                in_gap = False
    
    if in_gap:
        uncovered_ranges.append((gap_start, len(text)))
    
    # Create chunks for uncovered ranges
    for gap_start, gap_end in uncovered_ranges:
        gap_text = text[gap_start:gap_end].strip()
        if gap_text and len(gap_text) >= MIN_CHUNK_SIZE:
            logger.warning(f"Gap detected in structural chunking (positions {gap_start}-{gap_end}), creating chunk")
            current_chunk_number += 1
            
            gap_chunk = {
                "chunk_number": current_chunk_number,
                "chunk_level": 1,
                "parent_chunk_number": None,
                "text": gap_text,
                "metadata": {
                    "document_id": document_id,
                    "source_filename": source_filename,
                    "text_preview": gap_text[:120]
                }
            }
            
            # Insert gap chunk in correct position
            insert_pos = 0
            for idx, existing_chunk in enumerate(chunks):
                existing_text = existing_chunk["text"]
                existing_pos = text.find(existing_text)
                if existing_pos > gap_start:
                    insert_pos = idx
                    break
                insert_pos = idx + 1
            
            chunks.insert(insert_pos, gap_chunk)
    
    return chunks


# Example usage function
def example_usage():
    """Example of how to use structural_chunk_document."""
    import uuid
    
    sample_document = """
CONTRACTOR TERMS 2021
THIS AGREEMENT is made on the date on which it has been executed by both parties.

IT IS AGREED:

1. ACCEPTANCE OF TERMS AND CONDITIONS
   1.1 Definitions
   The following terms shall have the meanings set forth below.
   
   1.2 Scope
   This agreement applies to all services provided by the contractor.
   
2. PAYMENT TERMS
   Payment shall be made within 30 days of invoice receipt.
   
[SIGNATURE BLOCK]
Alex Doyle
Date: 2024-01-01
"""
    
    doc_id = str(uuid.uuid4())
    filename = "contract_2021.txt"
    max_chunk = 0  # Start from 0 if Qdrant is empty
    
    chunks = structural_chunk_document(
        text=sample_document,
        document_id=doc_id,
        source_filename=filename,
        existing_max_chunk_number=max_chunk
    )
    
    print(f"Generated {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"\nChunk #{chunk['chunk_number']} (Level {chunk['chunk_level']}):")
        print(f"  Parent: {chunk['parent_chunk_number']}")
        print(f"  Preview: {chunk['metadata']['text_preview']}")
        print(f"  Text length: {len(chunk['text'])} chars")


# ============================================================================
# Qdrant Integration Functions
# ============================================================================

def get_next_chunk_id(
    qdrant_client: QdrantClient,
    collection_name: str
) -> int:
    """
    Get the next available chunk_id by querying Qdrant for the maximum existing ID.
    
    Args:
        qdrant_client: Qdrant client instance
        collection_name: Qdrant collection name
    
    Returns:
        Next chunk_id (max existing ID + 1, or 0 if collection is empty)
    """
    if not QDRANT_AVAILABLE:
        raise ImportError("qdrant-client not available. Install with: pip install qdrant-client")
    
    try:
        # Scroll through all points to find max chunk_id
        max_id = 0
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
            
            # Check chunk_id in payload
            for point in points:
                payload = point.payload or {}
                chunk_id = payload.get("chunk_id") or payload.get("chunk_number")
                if chunk_id is not None:
                    try:
                        chunk_id = int(chunk_id)
                        max_id = max(max_id, chunk_id)
                    except (ValueError, TypeError):
                        pass
            
            offset = scroll_result[1]
            if offset is None:
                break
        
        return max_id + 1 if max_id > 0 else 0
    
    except Exception as e:
        logger.warning(f"Error getting max chunk_id from Qdrant: {e}. Starting from 0.")
        return 0


def assign_hierarchy(
    chunk_text: str,
    previous_chunks: List[Dict[str, Any]],
    rules: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Assign hierarchy level and context metadata to a chunk.
    
    Args:
        chunk_text: Text of the chunk
        previous_chunks: List of previously processed chunks (for context)
        rules: Optional rules dict (for future extensibility)
    
    Returns:
        Dict with hierarchy_level, context_influence, fallback_applied
    """
    hierarchy_level = 1  # Default to Level 1
    fallback_applied = False
    
    # Check for Level 1 patterns (top-level clauses)
    if re.search(r'^\s*\d+\.\s+', chunk_text, re.MULTILINE):
        hierarchy_level = 1
    # Check for Level 2 patterns (subclauses)
    elif re.search(r'^\s*\d+\.\d+(\.\d+)?\s+', chunk_text, re.MULTILINE):
        hierarchy_level = 2
    # Check for Level 3 patterns (deep subclauses or semantic units)
    elif re.search(r'^\s*\d+\.\d+\.\d+', chunk_text, re.MULTILINE):
        hierarchy_level = 3
    # If no structure detected and chunk is long, mark as fallback
    elif len(chunk_text) > MAX_CHUNK_LENGTH:
        fallback_applied = True
        hierarchy_level = 3  # Long unstructured chunks are Level 3
    
    # Calculate context influence based on previous chunks
    context_influence = 0.0
    if previous_chunks:
        # Simple heuristic: more previous chunks = more context
        context_influence = min(len(previous_chunks) * 0.1, 1.0)
    
    return {
        "hierarchy_level": hierarchy_level,
        "context_influence": context_influence,
        "fallback_applied": fallback_applied
    }


def inject_chunks_to_qdrant(
    chunks: List[Dict[str, Any]],
    document_name: str,
    qdrant_client: QdrantClient,
    collection_name: str,
    vector_dim: int = DEFAULT_VECTOR_DIM,
    batch_size: int = 50,
    rules: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Inject chunks into Qdrant with full metadata and placeholder vectors.
    
    Args:
        chunks: List of chunk dictionaries from structural_chunk_document()
        document_name: Name of the document
        qdrant_client: Qdrant client instance
        collection_name: Qdrant collection name
        vector_dim: Vector dimension for placeholder embeddings (default: 1536)
        batch_size: Number of points to upsert per batch
        rules: Optional rules dict (for future extensibility)
    
    Returns:
        Dict with injection results:
        {
            "success": bool,
            "chunks_inserted": int,
            "errors": List[str]
        }
    """
    if not QDRANT_AVAILABLE:
        raise ImportError("qdrant-client not available. Install with: pip install qdrant-client")
    
    if not chunks:
        logger.warning("No chunks to inject")
        return {"success": False, "chunks_inserted": 0, "errors": ["No chunks provided"]}
    
    errors = []
    chunks_inserted = 0
    
    try:
        # Ensure collection exists
        try:
            collections = qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if collection_name not in collection_names:
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=rest_models.VectorParams(
                        size=vector_dim,
                        distance=rest_models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {collection_name}")
        except Exception as e:
            logger.warning(f"Collection creation/verification failed: {e}")
        
        # Prepare points for Qdrant
        points = []
        previous_chunks = []
        
        for chunk in chunks:
            try:
                # Assign hierarchy and context metadata
                hierarchy_info = assign_hierarchy(chunk["text"], previous_chunks, rules)
                
                # Build payload with all metadata
                payload = {
                    "chunk_id": chunk["chunk_number"],
                    "chunk_number": chunk["chunk_number"],  # For compatibility
                    "document_name": document_name,
                    "hierarchy_level": hierarchy_info["hierarchy_level"],
                    "chunk_level": chunk.get("chunk_level", hierarchy_info["hierarchy_level"]),
                    "parent_chunk_number": chunk.get("parent_chunk_number"),
                    "text": chunk["text"],
                    "context_influence": hierarchy_info["context_influence"],
                    "fallback_applied": hierarchy_info["fallback_applied"],
                    # Include all metadata from chunk
                    **chunk.get("metadata", {})
                }
                
                # Create placeholder vector (all zeros)
                vector = [0.0] * vector_dim
                
                # Create point
                point = rest_models.PointStruct(
                    id=chunk["chunk_number"],  # Use chunk_number as point ID
                    vector=vector,
                    payload=payload
                )
                
                points.append(point)
                previous_chunks.append(chunk)
                
            except Exception as e:
                error_msg = f"Error preparing chunk {chunk.get('chunk_number', 'unknown')}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        # Upsert in batches
        total_batches = (len(points) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(points))
            batch_points = points[start_idx:end_idx]
            
            try:
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=batch_points
                )
                chunks_inserted += len(batch_points)
                logger.info(f"Inserted batch {batch_idx + 1}/{total_batches}: {len(batch_points)} chunks")
            
            except Exception as e:
                error_msg = f"Error inserting batch {batch_idx + 1}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        success = chunks_inserted > 0 and len(errors) == 0
        
        return {
            "success": success,
            "chunks_inserted": chunks_inserted,
            "errors": errors
        }
    
    except Exception as e:
        error_msg = f"Fatal error in inject_chunks_to_qdrant: {e}"
        logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "chunks_inserted": chunks_inserted,
            "errors": errors + [error_msg]
        }


def process_and_inject_document(
    document_text: str,
    document_name: str,
    document_id: str,
    qdrant_client: QdrantClient,
    collection_name: str,
    vector_dim: int = DEFAULT_VECTOR_DIM,
    max_chunk_length: int = MAX_CHUNK_LENGTH,
    rules: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Complete pipeline: Split document, chunk structurally, and inject into Qdrant.
    
    This is the main entry point that combines all steps:
    1. Get next chunk_id from Qdrant (query for max existing ID)
    2. Split document by double newlines (\n\n)
    3. Recursively split long segments (paragraphs → sentences → words)
    4. Rejoin and perform structural chunking (detect hierarchy: Level 1/2/3)
    5. Apply recursive splitting to structural chunks that exceed max_length
    6. Inject chunks into Qdrant with placeholder vectors and metadata
    
    Args:
        document_text: Full document text
        document_name: Name of the document (filename)
        document_id: Unique document identifier (UUID)
        qdrant_client: Qdrant client instance
        collection_name: Qdrant collection name
        vector_dim: Vector dimension for embeddings
        max_chunk_length: Maximum chunk length for recursive splitting
        rules: Optional rules dict
    
    Returns:
        Dict with processing results:
        {
            "success": bool,
            "chunks_generated": int,
            "chunks_inserted": int,
            "max_chunk_id": int,
            "errors": List[str]
        }
    """
    errors = []
    
    try:
        # Step 1: Get next chunk_id from Qdrant
        logger.info("Querying Qdrant for next chunk_id...")
        next_chunk_id = get_next_chunk_id(qdrant_client, collection_name)
        logger.info(f"Starting chunk numbering from: {next_chunk_id}")
        
        # Step 2: Initial document splitting by double newlines
        logger.info("Splitting document by double newlines...")
        initial_segments = split_document(document_text)
        logger.info(f"Split into {len(initial_segments)} initial segments")
        
        # Step 3: Recursively split long segments
        logger.info(f"Recursively splitting segments longer than {max_chunk_length} chars...")
        processed_segments = []
        for segment in initial_segments:
            if len(segment) > max_chunk_length:
                split_segments = recursive_split(segment, max_chunk_length)
                processed_segments.extend(split_segments)
            else:
                processed_segments.append(segment)
        
        logger.info(f"After recursive splitting: {len(processed_segments)} segments")
        
        # Step 4: Rejoin segments and perform structural chunking
        # Note: We rejoin because structural chunking needs full document structure
        # to detect hierarchy (e.g., "1. Section" followed by "1.1 Subsection")
        rejoined_text = "\n\n".join(processed_segments)
        
        logger.info("Performing structural chunking on full document...")
        chunks = structural_chunk_document(
            text=rejoined_text,
            document_id=document_id,
            source_filename=document_name,
            existing_max_chunk_number=next_chunk_id - 1  # -1 because function increments
        )
        
        # Step 5: Apply recursive splitting to chunks that exceed max_length
        logger.info(f"Applying recursive splitting to chunks longer than {max_chunk_length} chars...")
        final_chunks = []
        for chunk in chunks:
            chunk_text = chunk["text"]
            if len(chunk_text) > max_chunk_length:
                # Recursively split this chunk
                split_texts = recursive_split(chunk_text, max_chunk_length)
                for split_idx, split_text in enumerate(split_texts):
                    if not split_text.strip() or len(split_text.strip()) < MIN_CHUNK_SIZE:
                        continue
                    
                    # Create new chunk for each split
                    new_chunk = chunk.copy()
                    new_chunk["text"] = split_text.strip()
                    new_chunk["chunk_number"] = next_chunk_id + len(final_chunks)
                    new_chunk["metadata"]["text_preview"] = split_text.strip()[:120]
                    # Mark as recursively split
                    new_chunk["metadata"]["recursively_split"] = True
                    new_chunk["metadata"]["original_chunk_number"] = chunk["chunk_number"]
                    new_chunk["metadata"]["split_index"] = split_idx + 1
                    final_chunks.append(new_chunk)
            else:
                # Update chunk_number to continue global sequence
                chunk["chunk_number"] = next_chunk_id + len(final_chunks)
                final_chunks.append(chunk)
        
        chunks = final_chunks
        
        logger.info(f"Generated {len(chunks)} final chunks (after structural chunking and recursive splitting)")
        
        # Step 6: Inject chunks into Qdrant
        logger.info("Injecting chunks into Qdrant...")
        injection_result = inject_chunks_to_qdrant(
            chunks=chunks,
            document_name=document_name,
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            vector_dim=vector_dim,
            rules=rules
        )
        
        # Calculate max chunk_id
        max_chunk_id = max([c["chunk_number"] for c in chunks]) if chunks else next_chunk_id - 1
        
        return {
            "success": injection_result["success"],
            "chunks_generated": len(chunks),
            "chunks_inserted": injection_result["chunks_inserted"],
            "max_chunk_id": max_chunk_id,
            "errors": errors + injection_result.get("errors", [])
        }
    
    except Exception as e:
        error_msg = f"Fatal error in process_and_inject_document: {e}"
        logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "chunks_generated": 0,
            "chunks_inserted": 0,
            "max_chunk_id": 0,
            "errors": errors + [error_msg]
        }


# ============================================================================
# Example Usage with Qdrant
# ============================================================================

def example_usage_with_qdrant():
    """Example of complete pipeline with Qdrant injection."""
    import uuid
    
    if not QDRANT_AVAILABLE:
        print("Qdrant not available. Install with: pip install qdrant-client")
        return
    
    # Sample legal document
    sample_document = """
CONTRACTOR TERMS 2021
THIS AGREEMENT is made on the date on which it has been executed by both parties.

IT IS AGREED:

1. ACCEPTANCE OF TERMS AND CONDITIONS
   1.1 Definitions
   The following terms shall have the meanings set forth below.
   
   1.2 Scope
   This agreement applies to all services provided by the contractor.
   
2. PAYMENT TERMS
   Payment shall be made within 30 days of invoice receipt.
   
[SIGNATURE BLOCK]
Alex Doyle
Date: 2024-01-01
"""
    
    # Initialize Qdrant client
    try:
        qdrant_client = QdrantClient(url="http://localhost:6333")
        collection_name = "legal_documents"
        document_name = "contract_2021.txt"
        document_id = str(uuid.uuid4())
        
        # Process and inject document
        result = process_and_inject_document(
            document_text=sample_document,
            document_name=document_name,
            document_id=document_id,
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            vector_dim=1536
        )
        
        print("\n" + "="*60)
        print("Processing Results:")
        print("="*60)
        print(f"Success: {result['success']}")
        print(f"Chunks Generated: {result['chunks_generated']}")
        print(f"Chunks Inserted: {result['chunks_inserted']}")
        print(f"Max Chunk ID: {result['max_chunk_id']}")
        
        if result['errors']:
            print(f"\nErrors ({len(result['errors'])}):")
            for error in result['errors']:
                print(f"  - {error}")
        
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Example usage error: {e}", exc_info=True)


if __name__ == "__main__":
    # Run basic example
    example_usage()
    
    # Run Qdrant example (if available)
    print("\n" + "="*60)
    print("Qdrant Integration Example:")
    print("="*60)
    example_usage_with_qdrant()

