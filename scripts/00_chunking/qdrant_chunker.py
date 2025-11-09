#!/usr/bin/env python3
"""
qdrant_chunker.py

LegalDocumentChunker with selectable chunking strategies and Qdrant ingestion.

Features:
- Methods: recursive, semantic, structural, agentic, cluster, hierarchical
- Metadata fields: chunking_method, embedding_model (REQUIRED)
- Hierarchical chunking: Page → Clause → Subclause → Semantic Units
- Storage of multiple chunking strategies for the same document
- All chunking methods stored with metadata for filtering
"""

import re
import json
import uuid
import hashlib
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# PDF extraction
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# DOCX extraction
try:
    from docx import Document
    DOCX_AVAILABLE = True
except Exception:
    Document = None
    DOCX_AVAILABLE = False

try:
    import pypandoc
except Exception:
    pypandoc = None

try:
    import docx2txt
except Exception:
    docx2txt = None

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# LangChain splitters
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    RecursiveCharacterTextSplitter = None

try:
    from langchain_experimental.text_splitter import SemanticChunker
except Exception:
    SemanticChunker = None

# Qdrant
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest_models
except Exception:
    QdrantClient = None
    rest_models = None

# Clustering
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import numpy as np
except Exception:
    KMeans = None
    silhouette_score = None
    np = None

# Constants
MIN_PAGE_NUMBER = 1
MAX_PAGE_NUMBER = 200
MAX_SUBCLAUSE_WORDS = 500
TARGET_CHUNK_WORDS = 300


def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    parts = re.split(r"(?<=[\.\?\!])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def approx_word_count(text: str) -> int:
    """Approximate word count."""
    return len(re.findall(r"\w+", text))


def extract_ner_entities(text: str) -> Dict[str, Optional[str]]:
    """
    Extract Named Entities (NER) from legal/contract documents.
    
    Extracts:
    - individual_name: Full name of person
    - company_name: Company name
    - address: Company or client address
    - email: Email address
    - phone: Phone number
    
    Args:
        text: Full document text
        
    Returns:
        Dictionary with extracted entities (keys: individual_name, company_name, address, email, phone)
    """
    entities = {
        "individual_name": None,
        "company_name": None,
        "address": None,
        "email": None,
        "phone": None
    }
    
    # Pattern for email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    if emails:
        entities["email"] = emails[0]  # Take first email found
    
    # Pattern for UK phone numbers (with or without spaces/dashes)
    phone_patterns = [
        r'\b0\d{2}\s?\d{3}\s?\d{3}\s?\d{3}\b',  # 077916851111 or 07791 685 1111
        r'\b0\d{2}\s?\d{4}\s?\d{4}\b',          # 01234 567890
        r'\b\+44\s?\d{2}\s?\d{4}\s?\d{4}\b',   # +44 12 3456 7890
    ]
    for pattern in phone_patterns:
        phones = re.findall(pattern, text)
        if phones:
            entities["phone"] = phones[0].replace(' ', '').replace('-', '')
            break
    
    # Look for structured contract sections
    # Pattern: "Consultancy Personnel", "Agency", "Contractor Information", etc.
    section_patterns = [
        r'(?:Consultancy Personnel|Agency|Contractor Information|Client Information|Hiring Manager|Timesheet Approver)[\s:]*\n?([^\n]+(?:\n[^\n]+)*?)(?:Company address|Company E-mail|Phone number|$)',
        r'(?:Company|Client)[\s:]*\n?([^\n]+(?:\n[^\n]+)*?)(?:address|E-mail|Phone|$)',
    ]
    
    for pattern in section_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            section_text = match.group(0)
            
            # Extract company name (usually first line after section header)
            lines = section_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and not any(keyword in line.lower() for keyword in ['address', 'email', 'phone', 'company', 'information']):
                    # Check if it looks like a company name (contains Ltd, Ltd., LLC, Inc, etc.)
                    if re.search(r'\b(Ltd|Limited|LLC|Inc|Corporation|Corp|Company|Co\.)\b', line, re.IGNORECASE):
                        entities["company_name"] = line.strip()
                        break
                    # Or if it's a person name (two words, capitalized)
                    elif re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+', line) and not entities["individual_name"]:
                        entities["individual_name"] = line.strip()
            
            # Extract address (pattern: "Company address: ...")
            address_match = re.search(r'Company address:\s*([^\n]+(?:\n[^\n]+)*?)(?:Company E-mail|Phone number|$)', section_text, re.IGNORECASE)
            if address_match:
                address = address_match.group(1).strip()
                # Clean up address
                address = re.sub(r'\s+', ' ', address)
                entities["address"] = address
            
            # Extract email from section if not already found
            if not entities["email"]:
                email_match = re.search(r'[Ee]-?mail address:\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})', section_text)
                if email_match:
                    entities["email"] = email_match.group(1)
            
            # Extract phone from section if not already found
            if not entities["phone"]:
                phone_match = re.search(r'Phone number:\s*([0-9\s+\-]+)', section_text)
                if phone_match:
                    phone = phone_match.group(1).strip().replace(' ', '').replace('-', '')
                    if len(phone) >= 10:
                        entities["phone"] = phone
    
    # Fallback: Try to find company names in the document (Ltd, Limited, etc.)
    if not entities["company_name"]:
        company_pattern = r'\b([A-Z][A-Za-z0-9\s&]+(?:Ltd|Limited|LLC|Inc|Corporation|Corp|Company|Co\.))\b'
        companies = re.findall(company_pattern, text)
        if companies:
            entities["company_name"] = companies[0].strip()
    
    return entities


def make_chunk(
    chunking_method: str,
    id: str,
    level: int,
    clause_number: Optional[str],
    title: Optional[str],
    page: Optional[int],
    text: str,
    parent: Optional[str],
    embedding_model: str,
    source_file: Optional[str] = None,
    ner_entities: Optional[Dict[str, Optional[str]]] = None
) -> Dict[str, Any]:
    """
    Create chunk dictionary with required metadata.
    
    Args:
        chunking_method: REQUIRED - method used to create chunk
        id: Unique chunk ID
        level: Hierarchy level (0-3)
        clause_number: Optional clause identifier
        title: Optional title
        page: Optional page number
        text: Chunk text content
        parent: Optional parent clause ID
        embedding_model: REQUIRED - embedding model name
        source_file: Optional source filename
        ner_entities: Optional NER entities dictionary (individual_name, company_name, address, email, phone)
        
    Returns:
        Chunk dictionary with all required fields
    """
    chunk = {
        "id": id,
        "hierarchy_level": level,
        "clause_number": clause_number,
        "title": title,
        "page": page,
        "text": text,
        "parent": parent,
        "char_count": len(text),
        "chunking_method": chunking_method,  # REQUIRED
        "embedding_model": embedding_model   # REQUIRED
    }
    
    # Add source_file if provided
    if source_file:
        chunk["source_file"] = source_file
    
    # Add NER entities if provided
    if ner_entities:
        if ner_entities.get("individual_name"):
            chunk["individual_name"] = ner_entities["individual_name"]
        if ner_entities.get("company_name"):
            chunk["company_name"] = ner_entities["company_name"]
        if ner_entities.get("address"):
            chunk["address"] = ner_entities["address"]
        if ner_entities.get("email"):
            chunk["email"] = ner_entities["email"]
        if ner_entities.get("phone"):
            chunk["phone"] = ner_entities["phone"]
    
    return chunk


def _text_to_markdown(text: str) -> str:
    """
    Convert plain text to Markdown format.
    Preserves structure, headings, and paragraph breaks.
    
    Args:
        text: Plain text content
        
    Returns:
        Markdown-formatted text
    """
    # Clean up extra whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
    text = re.sub(r' {2,}', ' ', text)  # Multiple spaces to single
    text = re.sub(r' *\n *', '\n', text)  # Clean line breaks
    
    # Detect potential headings (lines that are short and end without punctuation)
    lines = text.split('\n')
    markdown_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            markdown_lines.append('')
            continue
        
        # Check if line looks like a heading (short, no period, might be numbered)
        # Pattern: "1. Title" or "Title" (short line)
        if len(line) < 100 and not line.endswith(('.', '!', '?', ':', ';')):
            # Check if it starts with a number pattern (e.g., "1.", "1.1", "1.1.1")
            heading_match = re.match(r'^(\d+(?:\.\d+)*)\.\s+(.+)$', line)
            if heading_match:
                # It's a numbered heading - convert to Markdown heading
                title = heading_match.group(2).strip()
                level = len(heading_match.group(1).split('.'))
                # Use appropriate heading level (max ##)
                heading_level = min(level, 2)
                markdown_lines.append('#' * heading_level + ' ' + title)
            elif len(line.split()) <= 10:  # Short line might be heading
                markdown_lines.append('## ' + line)
            else:
                markdown_lines.append(line)
        else:
            markdown_lines.append(line)
    
    # Join and clean up
    markdown_text = '\n'.join(markdown_lines)
    # Ensure proper paragraph separation
    markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)
    
    return markdown_text.strip()


def read_pdf(path: Path) -> Tuple[str, List[int]]:
    """
    Read PDF and extract text, converting to Markdown format.
    Preserves page boundaries and structure.
    
    Returns:
        Tuple of (Markdown text, list of page numbers)
    """
    if PdfReader is None:
        raise RuntimeError("Install: pip install pypdf")
    
    reader = PdfReader(str(path))
    pages = []
    page_numbers = []
    
    for i, p in enumerate(reader.pages, start=1):
        page_text = p.extract_text() or ""
        
        # Convert page text to Markdown
        markdown_page = _text_to_markdown(page_text)
        
        # Add page marker as Markdown comment
        pages.append(f"\n<!-- Page {i} -->\n")
        pages.append(markdown_page)
        
        # Track page numbers (approximate - one per line)
        lines = markdown_page.split('\n')
        page_numbers.extend([i] * len(lines))
    
    # Join all pages and clean up
    markdown_text = '\n'.join(pages)
    # Clean up excessive newlines
    markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)
    
    return markdown_text.strip(), page_numbers


def read_docx(path: Path) -> Tuple[str, List[int]]:
    """
    Read DOCX/DOC file and convert to Markdown format.
    Preserves headings, numbered clauses, and structure.
    
    Returns:
        Tuple of (Markdown text, empty page numbers list)
    """
    # Try pypandoc first (best quality)
    if pypandoc is not None:
        try:
            markdown_text = pypandoc.convert_file(
                str(path),
                'md',
                format='docx'
            )
            # Clean up
            markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)
            return markdown_text.strip(), []
        except Exception as e:
            # Fall back to docx2txt or python-docx
            pass
    
    # Try docx2txt
    if docx2txt is not None:
        try:
            text = docx2txt.process(str(path))
            # Convert to Markdown format
            markdown_text = _text_to_markdown(text)
            return markdown_text.strip(), []
        except Exception as e:
            pass
    
    # Try python-docx
    if Document is not None:
        try:
            doc = Document(str(path))
            paragraphs = []
            
            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    paragraphs.append('')
                    continue
                
                # Check paragraph style for headings
                style = para.style.name.lower() if para.style else ''
                if 'heading' in style:
                    # Extract heading level
                    level_match = re.search(r'heading\s*(\d+)', style)
                    level = int(level_match.group(1)) if level_match else 1
                    level = min(level, 6)  # Max heading level
                    paragraphs.append('#' * level + ' ' + text)
                elif para.style and 'list' in para.style.name.lower():
                    # Handle lists
                    paragraphs.append('- ' + text)
                else:
                    paragraphs.append(text)
            
            markdown_text = '\n'.join(paragraphs)
            markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)
            return markdown_text.strip(), []
        except Exception as e:
            raise RuntimeError(f"Failed to read DOCX file. Install: pip install pypandoc OR docx2txt OR python-docx. Error: {e}")
    
    raise RuntimeError("No DOCX reader available. Install: pip install pypandoc OR docx2txt OR python-docx")


def read_text(path: Path) -> Tuple[str, List[int]]:
    """
    Read text file.
    If file is .md, read directly without conversion.
    Otherwise, convert to Markdown format.
    
    Returns:
        Tuple of (Markdown text, empty page numbers list)
    """
    text = path.read_text(encoding="utf-8")
    
    # If already Markdown, return as-is
    if path.suffix.lower() == '.md':
        # Just clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip(), []
    
    # Convert plain text to Markdown
    markdown_text = _text_to_markdown(text)
    return markdown_text.strip(), []


def preprocess_text(text: str) -> str:
    """
    Preprocess text: remove footer page numbers and normalize whitespace.
    """
    # Remove footer page numbers (standalone numbers on their own line)
    text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
    # Remove "Page X of Y" patterns
    text = re.sub(r'Page\s+\d+(?:\s+of\s+\d+)?', '', text, flags=re.IGNORECASE)
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def detect_top_level_clauses(txt: str) -> List[Dict[str, Any]]:
    """
    Detect top-level clauses/sections (e.g., 1., 2., 3., or SECTION 1, Clause 1).
    
    Enhanced to detect multiple patterns:
    - Numbered clauses: "1. Title", "2. Title"
    - Section headers: "SECTION 1", "Clause 1"
    - All-caps headings that might be sections
    """
    # Pattern 1: Numbered clauses at start of line (1., 2., etc.)
    pat1 = re.compile(r"^(\d{1,2})\.\s+(.+?)(?=\n|$)", re.M)
    
    # Pattern 2: SECTION/Clause headers
    pat2 = re.compile(r"^(SECTION|Clause|CLAUSE)\s+(\d+)[:\.]?\s*(.+?)(?=\n|$)", re.M | re.I)
    
    # Pattern 3: All-caps lines that might be section headers (short lines, no punctuation)
    pat3 = re.compile(r"^([A-Z][A-Z\s]{5,50})(?=\n|$)", re.M)
    
    matches = []
    
    # Collect matches from all patterns
    for m in pat1.finditer(txt):
        matches.append({
            "start": m.start(),
            "number": m.group(1),
            "title": m.group(2).strip(),
            "pattern": 1
        })
    
    for m in pat2.finditer(txt):
        matches.append({
            "start": m.start(),
            "number": m.group(2),
            "title": m.group(3).strip() if m.group(3) else "",
            "pattern": 2
        })
    
    # Only use pattern 3 if no other matches found
    if not matches:
        for m in pat3.finditer(txt):
            line = m.group(1).strip()
            # Only consider if it's a reasonable length and doesn't look like regular text
            if 10 <= len(line) <= 80 and not line.endswith(('.', '!', '?', ':', ';')):
                matches.append({
                    "start": m.start(),
                    "number": None,
                    "title": line,
                    "pattern": 3
                })
    
    if not matches:
        return [{"start": 0, "end": len(txt), "number": None, "title": None}]
    
    # Sort by position
    matches.sort(key=lambda x: x["start"])
    
    # Build result with proper boundaries
    result = []
    for i, m in enumerate(matches):
        start = m["start"]
        end = matches[i+1]["start"] if i+1 < len(matches) else len(txt)
        result.append({
            "start": start,
            "end": end,
            "number": m["number"],
            "title": m["title"]
        })
    
    return result


def detect_subclauses(txt: str) -> List[Dict[str, Any]]:
    """Detect subclauses (e.g., 1.1, 1.2, (a), (b))."""
    pat_num = re.compile(r"^(\d+\.\d+(?:\.\d+)*)\s+", re.M)
    pat_letter = re.compile(r"^\(([a-zA-Z0-9]+)\)\s+", re.M)
    
    matches = []
    for m in pat_num.finditer(txt):
        matches.append((m.start(), m.group(1)))
    for m in pat_letter.finditer(txt):
        matches.append((m.start(), f"({m.group(1)})"))
    
    if not matches:
        return [{"start": 0, "end": len(txt), "label": None}]
    
    matches.sort(key=lambda x: x[0])
    subs = []
    for i, (s, label) in enumerate(matches):
        e = matches[i+1][0] if i+1 < len(matches) else len(txt)
        subs.append({"start": s, "end": e, "label": label})
    return subs


def semantic_split(text: str) -> List[str]:
    """Split long text into semantic units."""
    sents = split_sentences(text)
    out, cur, wc = [], [], 0
    
    for s in sents:
        c = approx_word_count(s)
        if wc + c <= TARGET_CHUNK_WORDS or not cur:
            cur.append(s)
            wc += c
        else:
            out.append(" ".join(cur))
            cur, wc = [s], c
    
    if cur:
        out.append(" ".join(cur))
    return out


class LegalDocumentChunker:
    """
    Legal document chunker with multiple strategies.
    All methods return chunks with chunking_method and embedding_model metadata.
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize chunker.
        
        Args:
            embedding_model: Embedding model name (sentence-transformers/* or ollama/llama3.1:latest)
        """
        self.embedding_model = embedding_model
        
        # Initialize embedding model based on type
        if embedding_model.startswith("ollama/"):
            # Ollama model - will be initialized on-demand
            self.model = None
            try:
                from langchain_ollama import OllamaEmbeddings
                import os
                ollama_model = embedding_model.replace("ollama/", "")
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                self.ollama_embeddings = OllamaEmbeddings(
                    model=ollama_model,
                    base_url=base_url
                )
            except ImportError:
                self.ollama_embeddings = None
        else:
            # SentenceTransformer model
            self.model = SentenceTransformer(embedding_model) if SentenceTransformer else None
            self.ollama_embeddings = None
    
    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Helper method to encode texts using either SentenceTransformer or Ollama.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        if self.embedding_model.startswith("ollama/"):
            if self.ollama_embeddings is None:
                raise RuntimeError("Ollama embeddings not initialized. Install langchain-ollama.")
            return self.ollama_embeddings.embed_documents(texts)
        else:
            if self.model is None:
                raise RuntimeError("SentenceTransformer model not initialized.")
            return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()
    
    def preprocess(self, txt: str) -> str:
        """Preprocess text."""
        return preprocess_text(txt)
    
    def recursive(self, text: str, page_numbers: Optional[List[int]] = None, ner_entities: Optional[Dict[str, Optional[str]]] = None) -> List[Dict[str, Any]]:
        """
        Recursive chunking using LangChain RecursiveCharacterTextSplitter.
        
        Returns:
            List of chunks with chunking_method="recursive"
        """
        if RecursiveCharacterTextSplitter is None:
            raise RuntimeError("install langchain")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " "]
        )
        parts = splitter.split_text(text)
        
        chunks = []
        current_offset = 0
        
        for i, p in enumerate(parts, 1):
            start_offset = text.find(p, current_offset)
            end_offset = start_offset + len(p)
            current_offset = start_offset
            
            page = None
            if page_numbers:
                # Simple heuristic: find page based on text position
                if start_offset < len(text):
                    ratio = start_offset / len(text)
                    page_index = int(ratio * len(page_numbers))
                    page = page_numbers[min(page_index, len(page_numbers) - 1)]
            
            chunks.append(make_chunk(
                chunking_method="recursive",
                id=f"r_{uuid.uuid4().hex[:8]}_{i}",
                level=0,
                clause_number=None,
                title=None,
                page=page,
                text=p,
                parent=None,
                embedding_model=self.embedding_model,
                ner_entities=ner_entities
            ))
        
        return chunks
    
    def semantic(self, text: str, page_numbers: Optional[List[int]] = None, ner_entities: Optional[Dict[str, Optional[str]]] = None) -> List[Dict[str, Any]]:
        """
        Semantic chunking using LangChain SemanticChunker.
        
        Returns:
            List of chunks with chunking_method="semantic"
        """
        if SemanticChunker is None:
            raise RuntimeError("install langchain-experimental")
        
        if self.model is None and self.ollama_embeddings is None:
            raise RuntimeError("install sentence-transformers or langchain-ollama")
        
        # SemanticChunker expects an embeddings object with embed_documents method
        # Use OllamaEmbeddings directly if available, otherwise create a wrapper
        if self.ollama_embeddings is not None:
            # OllamaEmbeddings already has embed_documents method
            embeddings_obj = self.ollama_embeddings
        else:
            # For SentenceTransformer, create a wrapper class
            class SentenceTransformerEmbeddings:
                def __init__(self, model):
                    self.model = model
                
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    """Embed documents using SentenceTransformer."""
                    if self.model is None:
                        raise RuntimeError("SentenceTransformer model not initialized")
                    return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()
            
            embeddings_obj = SentenceTransformerEmbeddings(self.model)
        
        splitter = SemanticChunker(
            embeddings=embeddings_obj,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95.0
        )
        parts = splitter.split_text(text)
        
        chunks = []
        current_offset = 0
        
        for i, p in enumerate(parts, 1):
            start_offset = text.find(p, current_offset)
            end_offset = start_offset + len(p)
            current_offset = start_offset
            
            page = None
            if page_numbers:
                if start_offset < len(text):
                    ratio = start_offset / len(text)
                    page_index = int(ratio * len(page_numbers))
                    page = page_numbers[min(page_index, len(page_numbers) - 1)]
            
            chunks.append(make_chunk(
                chunking_method="semantic",
                id=f"s_{uuid.uuid4().hex[:8]}_{i}",
                level=0,
                clause_number=None,
                title=None,
                page=page,
                text=p,
                parent=None,
                embedding_model=self.embedding_model,
                ner_entities=ner_entities
            ))
        
        return chunks
    
    def structural(self, text: str, page_numbers: Optional[List[int]] = None, ner_entities: Optional[Dict[str, Optional[str]]] = None) -> List[Dict[str, Any]]:
        """
        Structural chunking by clause numbering patterns.
        Hierarchical: Clause → Subclause → Semantic Units.
        
        Ensures full document integrity:
        - Preserves text before first structural marker (intro/title)
        - Preserves text after last structural marker (outro/signature)
        - Falls back to recursive chunking if no structural markers found
        
        Returns:
            List of chunks with chunking_method="structural"
        """
        out = []
        clauses = detect_top_level_clauses(text)
        
        # Check if we found any real structural markers
        # detect_top_level_clauses returns a fallback clause if no markers found:
        # {"start": 0, "end": len(txt), "number": None, "title": None}
        has_real_structure = False
        if clauses:
            # Check if any clause has a number (real structural marker)
            for clause in clauses:
                if clause.get('number') is not None:
                    has_real_structure = True
                    break
        
        # If no structural markers found (only fallback clause), use recursive chunking
        if not has_real_structure:
            # No structural markers detected - use recursive chunking as fallback
            logger.debug("No structural markers found, falling back to recursive chunking")
            recursive_chunks = self.recursive(text, page_numbers, ner_entities)
            # Mark all chunks as preserved_full_text since they came from fallback
            for chunk in recursive_chunks:
                chunk["preserved_full_text"] = True
                # Change chunking_method to structural but keep the recursive structure
                chunk["chunking_method"] = "structural"
                chunk["chunk_level"] = 1  # All fallback chunks are Level 1
            return recursive_chunks
        
        # Process intro text (before first clause)
        first_clause_start = clauses[0]['start'] if clauses else len(text)
        if first_clause_start > 0:
            intro_text = text[:first_clause_start].strip()
            if intro_text:
                # Determine page for intro
                page = None
                if page_numbers:
                    ratio = 0 / len(text) if len(text) > 0 else 0
                    page_index = int(ratio * len(page_numbers))
                    page = page_numbers[min(page_index, len(page_numbers) - 1)] if page_numbers else None
                
                out.append(make_chunk(
                    chunking_method="structural",
                    id="struct_intro",
                    level=1,
                    clause_number=None,
                    title=None,
                    page=page,
                    text=intro_text,
                    parent=None,
                    embedding_model=self.embedding_model,
                    ner_entities=ner_entities
                ))
                # Mark as preserved (intro text)
                out[-1]["preserved_full_text"] = True
        
        # Process structural clauses
        for ci, c in enumerate(clauses, 1):
            c_text = text[c['start']:c['end']].strip()
            if not c_text:
                continue
            
            cid = c['number'] or f"cl{ci}"
            
            # Determine page
            page = None
            if page_numbers:
                if c['start'] < len(text):
                    ratio = c['start'] / len(text) if len(text) > 0 else 0
                    page_index = int(ratio * len(page_numbers))
                    page = page_numbers[min(page_index, len(page_numbers) - 1)]
            
            # Top-level clause
            chunk = make_chunk(
                chunking_method="structural",
                id=f"struct_{cid}",
                level=1,
                clause_number=cid,
                title=c['title'],
                page=page,
                text=c_text,
                parent=None,
                embedding_model=self.embedding_model,
                ner_entities=ner_entities
            )
            chunk["preserved_full_text"] = False  # Structural chunk, not preserved
            out.append(chunk)
            
            # Subclauses
            subs = detect_subclauses(c_text)
            if subs:
                for si, s in enumerate(subs, 1):
                    s_text = c_text[s['start']:s['end']].strip()
                    if not s_text:
                        continue
                    
                    sid = f"{cid}.{si}"
                    
                    if approx_word_count(s_text) <= MAX_SUBCLAUSE_WORDS:
                        # Keep as subclause
                        sub_chunk = make_chunk(
                            chunking_method="structural",
                            id=f"struct_{sid}",
                            level=2,
                            clause_number=sid,
                            title=None,
                            page=page,
                            text=s_text,
                            parent=cid,
                            embedding_model=self.embedding_model,
                            ner_entities=ner_entities
                        )
                        sub_chunk["preserved_full_text"] = False
                        out.append(sub_chunk)
                    else:
                        # Split into semantic units
                        units = semantic_split(s_text)
                        for ui, u in enumerate(units, 1):
                            if not u.strip():
                                continue
                            uid = f"{sid}.{ui}"
                            unit_chunk = make_chunk(
                                chunking_method="structural",
                                id=f"struct_{uid}",
                                level=3,
                                clause_number=uid,
                                title=None,
                                page=page,
                                text=u,
                                parent=sid,
                                embedding_model=self.embedding_model,
                                ner_entities=ner_entities
                            )
                            unit_chunk["preserved_full_text"] = False
                            out.append(unit_chunk)
        
        # Process trailing text (after last clause)
        if clauses:
            last_clause_end = clauses[-1]['end']
            if last_clause_end < len(text):
                outro_text = text[last_clause_end:].strip()
                if outro_text:
                    # Determine page for outro
                    page = None
                    if page_numbers:
                        ratio = last_clause_end / len(text) if len(text) > 0 else 0
                        page_index = int(ratio * len(page_numbers))
                        page = page_numbers[min(page_index, len(page_numbers) - 1)] if page_numbers else None
                    
                    outro_chunk = make_chunk(
                        chunking_method="structural",
                        id="struct_outro",
                        level=1,
                        clause_number=None,
                        title=None,
                        page=page,
                        text=outro_text,
                        parent=None,
                        embedding_model=self.embedding_model,
                        ner_entities=ner_entities
                    )
                    outro_chunk["preserved_full_text"] = True  # Trailing text, preserved
                    out.append(outro_chunk)
        
        # Verify we covered all text (safety check)
        # Build a set of all character positions covered by chunks
        covered_positions = set()
        for chunk in out:
            chunk_text = chunk.get("text", "")
            # Find all occurrences of chunk text in original document
            start = 0
            while True:
                pos = text.find(chunk_text, start)
                if pos == -1:
                    break
                # Mark positions as covered
                for i in range(pos, pos + len(chunk_text)):
                    covered_positions.add(i)
                start = pos + 1
        
        # Check for uncovered positions (gaps)
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
        
        # Handle gap at end of document
        if in_gap:
            uncovered_ranges.append((gap_start, len(text)))
        
        # Create chunks for uncovered ranges
        for gap_start, gap_end in uncovered_ranges:
            gap_text = text[gap_start:gap_end].strip()
            if gap_text and len(gap_text) > 10:  # Only create chunk if substantial text
                logger.warning(f"Gap detected in structural chunking (positions {gap_start}-{gap_end}), creating chunk for missing text (length: {len(gap_text)})")
                page = None
                if page_numbers:
                    ratio = gap_start / len(text) if len(text) > 0 else 0
                    page_index = int(ratio * len(page_numbers))
                    page = page_numbers[min(page_index, len(page_numbers) - 1)] if page_numbers else None
                
                gap_chunk = make_chunk(
                    chunking_method="structural",
                    id=f"struct_gap_{gap_start}",
                    level=1,
                    clause_number=None,
                    title=None,
                    page=page,
                    text=gap_text,
                    parent=None,
                    embedding_model=self.embedding_model,
                    ner_entities=ner_entities
                )
                gap_chunk["preserved_full_text"] = True
                # Insert gap chunk in correct position (sorted by position in document)
                insert_pos = 0
                for idx, existing_chunk in enumerate(out):
                    existing_text = existing_chunk.get("text", "")
                    existing_pos = text.find(existing_text)
                    if existing_pos > gap_start:
                        insert_pos = idx
                        break
                    insert_pos = idx + 1
                out.insert(insert_pos, gap_chunk)
        
        return out
    
    def agentic(self, text: str, page_numbers: Optional[List[int]] = None, ner_entities: Optional[Dict[str, Optional[str]]] = None) -> List[Dict[str, Any]]:
        """
        Agentic chunking: intelligent concept-based grouping that preserves document structure.
        
        Strategy:
        1. Detect document structure (sections, clauses, subsections)
        2. Group semantically similar content within structural boundaries
        3. Preserve hierarchical relationships and context
        4. Ensure chunks have sufficient context for retrieval
        
        Returns:
            List of chunks with chunking_method="agentic"
        """
        if self.model is None and self.ollama_embeddings is None:
            # Fallback to recursive
            return self.recursive(text, page_numbers, ner_entities)
        
        # Step 1: Detect document structure (top-level clauses/sections)
        top_level_clauses = detect_top_level_clauses(text)
        
        # If no structure detected, use improved semantic grouping
        if len(top_level_clauses) <= 1 or not top_level_clauses[0].get("number"):
            return self._agentic_semantic_grouping(text, page_numbers, ner_entities)
        
        # Step 2: Process each structural section with semantic grouping
        chunks = []
        chunk_counter = 0
        
        for clause_idx, clause in enumerate(top_level_clauses):
            clause_text = text[clause['start']:clause['end']].strip()
            if not clause_text:
                continue
            
            clause_number = clause.get('number')
            clause_title = clause.get('title', '')
            
            # Determine page for this clause
            page = None
            if page_numbers:
                if clause['start'] < len(text):
                    ratio = clause['start'] / len(text)
                    page_index = int(ratio * len(page_numbers))
                    page = page_numbers[min(page_index, len(page_numbers) - 1)]
            
            # Detect subclauses within this clause
            subclauses = detect_subclauses(clause_text)
            
            # If clause is small enough, keep as single chunk
            word_count = approx_word_count(clause_text)
            if word_count <= TARGET_CHUNK_WORDS * 2:  # Allow up to 2x target for context
                # Single chunk for this clause
                chunk_counter += 1
                chunks.append(make_chunk(
                    chunking_method="agentic",
                    id=f"a_{uuid.uuid4().hex[:8]}_{chunk_counter}",
                    level=1,
                    clause_number=clause_number,
                    title=clause_title,
                    page=page,
                    text=clause_text,
                    parent=None,
                    embedding_model=self.embedding_model,
                    ner_entities=ner_entities
                ))
            else:
                # Process subclauses or semantic units
                if len(subclauses) > 1 and subclauses[0].get('label'):
                    # Has subclauses - process each
                    for sub_idx, subclause in enumerate(subclauses):
                        sub_text = clause_text[subclause['start']:subclause['end']].strip()
                        if not sub_text:
                            continue
                        
                        sub_label = subclause.get('label', '')
                        sub_number = f"{clause_number}.{sub_idx+1}" if clause_number else None
                        
                        sub_word_count = approx_word_count(sub_text)
                        if sub_word_count <= TARGET_CHUNK_WORDS * 2:
                            # Keep subclause as single chunk
                            chunk_counter += 1
                            chunks.append(make_chunk(
                                chunking_method="agentic",
                                id=f"a_{uuid.uuid4().hex[:8]}_{chunk_counter}",
                                level=2,
                                clause_number=sub_number,
                                title=sub_label if sub_label else None,
                                page=page,
                                text=sub_text,
                                parent=clause_number,
                                embedding_model=self.embedding_model,
                                ner_entities=ner_entities
                            ))
                        else:
                            # Split subclause into semantic units
                            semantic_units = self._agentic_semantic_grouping(
                                sub_text, 
                                page_numbers, 
                                ner_entities,
                                parent_clause=clause_number,
                                base_level=2
                            )
                            # Update IDs and metadata for semantic units
                            for unit in semantic_units:
                                chunk_counter += 1
                                unit['id'] = f"a_{uuid.uuid4().hex[:8]}_{chunk_counter}"
                                unit['clause_number'] = f"{sub_number}.{chunk_counter}" if sub_number else None
                                unit['parent'] = clause_number
                                unit['page'] = page
                                chunks.append(unit)
                else:
                    # No subclauses - use semantic grouping
                    semantic_units = self._agentic_semantic_grouping(
                        clause_text,
                        page_numbers,
                        ner_entities,
                        parent_clause=clause_number,
                        base_level=1,
                        clause_title=clause_title
                    )
                    # Update IDs and metadata
                    for unit in semantic_units:
                        chunk_counter += 1
                        unit['id'] = f"a_{uuid.uuid4().hex[:8]}_{chunk_counter}"
                        unit['clause_number'] = f"{clause_number}.{chunk_counter}" if clause_number else None
                        unit['title'] = clause_title if not unit.get('title') else unit['title']
                        unit['page'] = page
                        chunks.append(unit)
        
        return chunks if chunks else self._agentic_semantic_grouping(text, page_numbers, ner_entities)
    
    def _agentic_semantic_grouping(
        self, 
        text: str, 
        page_numbers: Optional[List[int]] = None, 
        ner_entities: Optional[Dict[str, Optional[str]]] = None,
        parent_clause: Optional[str] = None,
        base_level: int = 0,
        clause_title: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Improved semantic grouping that preserves context and uses adaptive similarity thresholds.
        
        Args:
            text: Text to chunk
            page_numbers: Page numbers mapping
            ner_entities: NER entities
            parent_clause: Parent clause identifier
            base_level: Base hierarchy level
            clause_title: Optional clause title
            
        Returns:
            List of chunk dictionaries
        """
        if np is None:
            # Fallback to recursive
            return self.recursive(text, page_numbers, ner_entities)
        
        sentences = split_sentences(text)
        if len(sentences) < 2:
            page = page_numbers[0] if page_numbers else None
            return [make_chunk(
                chunking_method="agentic",
                id=f"a_{uuid.uuid4().hex[:8]}_1",
                level=base_level,
                clause_number=None,
                title=clause_title,
                page=page,
                text=text,
                parent=parent_clause,
                embedding_model=self.embedding_model,
                ner_entities=ner_entities
            )]
        
        # Embed sentences
        embeddings = self._encode_texts(sentences)
        
        # Adaptive similarity threshold based on document characteristics
        # Lower threshold for longer documents to create more cohesive chunks
        base_similarity = 0.65  # Lowered from 0.7 for better grouping
        min_chunk_words = 100  # Minimum words per chunk for context
        max_chunk_words = 800  # Maximum words per chunk
        
        chunks = []
        current_group = [sentences[0]]
        current_embeddings = [embeddings[0]]
        current_offset = 0
        
        for i in range(1, len(sentences)):
            current_text = ' '.join(current_group)
            current_word_count = approx_word_count(current_text)
            
            # Calculate similarity
            current_center = np.mean(current_embeddings, axis=0)
            similarity = np.dot(current_center, embeddings[i]) / (
                np.linalg.norm(current_center) * np.linalg.norm(embeddings[i]) + 1e-8
            )
            
            # Adaptive threshold: lower threshold if chunk is small (needs more content)
            adaptive_threshold = base_similarity
            if current_word_count < min_chunk_words:
                adaptive_threshold = base_similarity - 0.1  # More lenient for small chunks
            elif current_word_count > max_chunk_words * 0.8:
                adaptive_threshold = base_similarity + 0.05  # Stricter when approaching max
            
            # Check if we should add sentence to current group
            next_text = current_text + ' ' + sentences[i]
            next_word_count = approx_word_count(next_text)
            
            should_group = (
                similarity > adaptive_threshold and 
                next_word_count <= max_chunk_words
            )
            
            if should_group:
                current_group.append(sentences[i])
                current_embeddings.append(embeddings[i])
            else:
                # Finalize current chunk if it meets minimum size
                if current_word_count >= min_chunk_words or len(chunks) == 0:
                    chunk_text = ' '.join(current_group)
                    start_off = text.find(chunk_text, current_offset)
                    end_off = start_off + len(chunk_text) if start_off >= 0 else current_offset + len(chunk_text)
                    
                    page = None
                    if page_numbers:
                        if start_off >= 0 and start_off < len(text):
                            ratio = start_off / len(text)
                            page_index = int(ratio * len(page_numbers))
                            page = page_numbers[min(page_index, len(page_numbers) - 1)]
                    
                    chunks.append(make_chunk(
                        chunking_method="agentic",
                        id=f"a_{uuid.uuid4().hex[:8]}_{len(chunks)+1}",
                        level=base_level,
                        clause_number=None,
                        title=clause_title if len(chunks) == 0 else None,
                        page=page,
                        text=chunk_text,
                        parent=parent_clause,
                        embedding_model=self.embedding_model,
                        ner_entities=ner_entities
                    ))
                
                # Start new group
                current_group = [sentences[i]]
                current_embeddings = [embeddings[i]]
                current_offset = end_off if start_off >= 0 else current_offset + len(' '.join(current_group[:-1]))
        
        # Final group - ensure it meets minimum size or merge with previous
        if current_group:
            chunk_text = ' '.join(current_group)
            current_word_count = approx_word_count(chunk_text)
            
            if current_word_count >= min_chunk_words or len(chunks) == 0:
                start_off = text.find(chunk_text, current_offset)
                
                page = None
                if page_numbers:
                    if start_off >= 0 and start_off < len(text):
                        ratio = start_off / len(text)
                        page_index = int(ratio * len(page_numbers))
                        page = page_numbers[min(page_index, len(page_numbers) - 1)]
                
                chunks.append(make_chunk(
                    chunking_method="agentic",
                    id=f"a_{uuid.uuid4().hex[:8]}_{len(chunks)+1}",
                    level=base_level,
                    clause_number=None,
                    title=None,
                    page=page,
                    text=chunk_text,
                    parent=parent_clause,
                    embedding_model=self.embedding_model,
                    ner_entities=ner_entities
                ))
            elif len(chunks) > 0:
                # Merge with last chunk if too small
                last_chunk = chunks[-1]
                last_chunk['text'] = last_chunk['text'] + ' ' + chunk_text
        
        return chunks if chunks else [make_chunk(
            chunking_method="agentic",
            id=f"a_{uuid.uuid4().hex[:8]}_1",
            level=base_level,
            clause_number=None,
            title=clause_title,
            page=page_numbers[0] if page_numbers else None,
            text=text,
            parent=parent_clause,
            embedding_model=self.embedding_model,
            ner_entities=ner_entities
        )]
    
    def cluster(self, text: str, n_clusters: int = 10, page_numbers: Optional[List[int]] = None, ner_entities: Optional[Dict[str, Optional[str]]] = None) -> List[Dict[str, Any]]:
        """
        Clustering-based chunking using KMeans.
        
        Returns:
            List of chunks with chunking_method="cluster"
        """
        if KMeans is None or np is None:
            raise RuntimeError("install scikit-learn")
        
        if self.model is None and self.ollama_embeddings is None:
            raise RuntimeError("install sentence-transformers or langchain-ollama")
        
        sentences = split_sentences(text)
        if len(sentences) < n_clusters:
            return self.recursive(text, page_numbers)
        
        embeddings = self._encode_texts(sentences)
        kmeans = KMeans(n_clusters=min(n_clusters, len(sentences)), random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        cluster_groups = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            cluster_groups[cluster_id].append(i)
        
        chunks = []
        current_offset = 0
        
        for cluster_id, sentence_indices in sorted(cluster_groups.items()):
            cluster_sentences = [sentences[i] for i in sorted(sentence_indices)]
            chunk_text = ' '.join(cluster_sentences)
            
            start_off = text.find(chunk_text, current_offset)
            if start_off == -1:
                start_off = current_offset
            end_off = start_off + len(chunk_text)
            current_offset = end_off
            
            page = None
            if page_numbers:
                if start_off < len(text):
                    ratio = start_off / len(text)
                    page_index = int(ratio * len(page_numbers))
                    page = page_numbers[min(page_index, len(page_numbers) - 1)]
            
            chunks.append(make_chunk(
                chunking_method="cluster",
                id=f"c_{uuid.uuid4().hex[:8]}_{cluster_id}",
                level=0,
                clause_number=None,
                title=f"Cluster {cluster_id}",
                page=page,
                text=chunk_text,
                parent=None,
                embedding_model=self.embedding_model,
                ner_entities=ner_entities
            ))
        
        return chunks
    
    def hierarchical(self, text: str, page_numbers: Optional[List[int]] = None, ner_entities: Optional[Dict[str, Optional[str]]] = None) -> List[Dict[str, Any]]:
        """
        Hierarchical chunking: Page → Clause → Subclause → Semantic Units.
        Combines structural and semantic chunking.
        
        Returns:
            List of chunks with chunking_method="hierarchical"
        """
        # Use structural as base, which already implements hierarchy
        # But update chunking_method to "hierarchical"
        chunks = self.structural(text, page_numbers, ner_entities)
        
        # Update all chunks to have chunking_method="hierarchical"
        for chunk in chunks:
            chunk["chunking_method"] = "hierarchical"
        
        return chunks
    
    
    def chunk(self, text: str, method: str, page_numbers: Optional[List[int]] = None, ner_entities: Optional[Dict[str, Optional[str]]] = None) -> List[Dict[str, Any]]:
        """
        Chunk document using specified method.
        
        Args:
            text: Document text
            method: Chunking method name
            page_numbers: Optional list of page numbers
            ner_entities: Optional NER entities dictionary (individual_name, company_name, address, email, phone)
            
        Returns:
            List of chunk dictionaries
        """
        t = self.preprocess(text)
        
        if method == "recursive":
            return self.recursive(t, page_numbers, ner_entities)
        elif method == "semantic":
            return self.semantic(t, page_numbers, ner_entities)
        elif method == "structural":
            return self.structural(t, page_numbers, ner_entities)
        elif method == "agentic":
            return self.agentic(t, page_numbers, ner_entities)
        elif method == "cluster":
            return self.cluster(t, n_clusters=10, page_numbers=page_numbers, ner_entities=ner_entities)
        elif method == "hierarchical":
            return self.hierarchical(t, page_numbers, ner_entities)
        else:
            raise ValueError(f"Unknown method: {method}")


def embed_texts(texts: List[str], model_name: str) -> List[List[float]]:
    """
    Embed texts using SentenceTransformer or Ollama.
    
    Args:
        texts: List of text strings
        model_name: Embedding model name (sentence-transformers/* or ollama/llama3.1:latest)
        
    Returns:
        List of embedding vectors
    """
    # Check if Ollama model
    if model_name.startswith("ollama/"):
        try:
            from langchain_ollama import OllamaEmbeddings
            import os
            
            # Extract model name (e.g., "llama3.1:latest" from "ollama/llama3.1:latest")
            ollama_model = model_name.replace("ollama/", "")
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            
            embeddings = OllamaEmbeddings(
                model=ollama_model,
                base_url=base_url
            )
            
            # Generate embeddings
            vectors = embeddings.embed_documents(texts)
            return vectors
        except ImportError:
            raise RuntimeError("langchain-ollama required for Ollama embeddings. Install with: pip install langchain-ollama")
        except Exception as e:
            raise RuntimeError(f"Error generating Ollama embeddings: {e}")
    else:
        # Use SentenceTransformer
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers required")
        
        model = SentenceTransformer(model_name)
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()


def create_collection(client: QdrantClient, name: str, dim: int):
    """
    Create Qdrant collection if it doesn't exist.
    
    Args:
        client: Qdrant client
        name: Collection name
        dim: Vector dimension
    """
    try:
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if name not in collection_names:
            client.create_collection(
                collection_name=name,
                vectors_config=rest_models.VectorParams(
                    size=dim,
                    distance=rest_models.Distance.COSINE
                )
            )
            print(f"Created collection: {name}")
        else:
            print(f"Collection {name} already exists")
    except Exception as e:
        raise RuntimeError(f"Error creating collection: {e}")


def upsert(client: QdrantClient, name: str, chunks: List[Dict[str, Any]], vectors: List[List[float]], batch_size: int = 50):
    """
    Upsert chunks to Qdrant with metadata in batches to avoid payload size limits.
    
    Args:
        client: Qdrant client
        name: Collection name
        chunks: List of chunk dictionaries (must include chunking_method and embedding_model)
        vectors: List of embedding vectors
        batch_size: Number of points to upsert per batch (default: 100)
    """
    if not chunks or not vectors:
        return
    
    if len(chunks) != len(vectors):
        raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(vectors)} vectors")
    
    total_points = len(chunks)
    total_batches = (total_points + batch_size - 1) // batch_size  # Ceiling division
    
    print(f"Upserting {total_points} points in {total_batches} batch(es) of up to {batch_size} points each...")
    
    # Process in batches
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_points)
        batch_chunks = chunks[start_idx:end_idx]
        batch_vectors = vectors[start_idx:end_idx]
        
        points = []
        for i, (chunk, vector) in enumerate(zip(batch_chunks, batch_vectors), start=start_idx):
            # Ensure required metadata fields
            payload = {
                "text": chunk["text"],
                "hierarchy_level": chunk["hierarchy_level"],
                "char_count": chunk["char_count"],
                "chunking_method": chunk.get("chunking_method", "unknown"),  # REQUIRED
                "embedding_model": chunk.get("embedding_model", "unknown")   # REQUIRED
            }
            
            # Add chunk_number (index within the file for this chunking method)
            # Use chunk's own chunk_number if available, otherwise use batch index
            # This is useful for exploration and understanding chunk order
            chunk_number = chunk.get("chunk_number")
            if chunk_number is None:
                # Use batch index (i starts from start_idx, which is 0 for first batch of each file)
                chunk_number = i
            payload["chunk_number"] = chunk_number
            payload["chunk_index"] = chunk_number  # Also store as chunk_index for backward compatibility
            
            # Add chunk_level (1, 2, or 3 for structural chunks)
            if chunk.get("chunk_level") is not None:
                payload["chunk_level"] = chunk.get("chunk_level")
            elif chunk.get("hierarchy_level") is not None:
                payload["chunk_level"] = chunk.get("hierarchy_level")
            else:
                payload["chunk_level"] = 1  # Default to level 1
            
            # Add parent_chunk_number (for hierarchical chunks)
            if chunk.get("parent_chunk_number") is not None:
                payload["parent_chunk_number"] = chunk.get("parent_chunk_number")
            
            # Add text_preview (first N characters for quick UI preview)
            if chunk.get("text_preview"):
                payload["text_preview"] = chunk.get("text_preview")
            else:
                # Generate preview if not provided
                chunk_text = chunk.get("text", "")
                payload["text_preview"] = chunk_text[:200] + ("..." if len(chunk_text) > 200 else "")
            
            # Add preserved_full_text flag (indicates if chunk came from fallback/intro/outro)
            if chunk.get("preserved_full_text") is not None:
                payload["preserved_full_text"] = chunk.get("preserved_full_text")
            else:
                # Default to False if not set (structural chunks are not preserved by default)
                payload["preserved_full_text"] = False
            
            # Add optional fields
            if chunk.get("clause_number"):
                payload["clause_number"] = chunk["clause_number"]
            if chunk.get("title"):
                payload["title"] = chunk["title"]
            if chunk.get("page"):
                payload["page"] = chunk["page"]
            if chunk.get("parent"):
                payload["parent"] = chunk["parent"]
            
            # Add source_file (from source_file or file_name for backward compatibility)
            source_file = chunk.get("source_file") or chunk.get("file_name")
            if source_file:
                payload["source_file"] = source_file
            
            # Add upload_timestamp if present (for tracking when file was uploaded)
            upload_timestamp = chunk.get("upload_timestamp")
            if upload_timestamp:
                payload["upload_timestamp"] = upload_timestamp
                # Also add upload_time for backward compatibility
                payload["upload_time"] = upload_timestamp
            
            # Add document_id if present (unique document identifier)
            if chunk.get("document_id"):
                payload["document_id"] = chunk.get("document_id")
            
            # Add source_filename if present (for consistency with requirements)
            if chunk.get("source_filename"):
                payload["source_filename"] = chunk.get("source_filename")
            
            # Add NER entities if present
            if chunk.get("individual_name"):
                payload["individual_name"] = chunk["individual_name"]
            if chunk.get("company_name"):
                payload["company_name"] = chunk["company_name"]
            if chunk.get("address"):
                payload["address"] = chunk["address"]
            if chunk.get("email"):
                payload["email"] = chunk["email"]
            if chunk.get("phone"):
                payload["phone"] = chunk["phone"]
            
            # Generate unique point ID based on source_file, chunking_method, and index
            # This ensures multiple files don't overwrite each other
            chunking_method = chunk.get("chunking_method", "unknown")
            file_id_part = source_file or "unknown_file"
            
            # Create a unique ID: deterministic hash of (source_file + chunking_method + index)
            # Using hashlib for deterministic hashing across Python runs
            unique_id_string = f"{file_id_part}::{chunking_method}::{i}"
            # Use MD5 hash and convert first 8 bytes to int64 (Qdrant accepts int)
            hash_obj = hashlib.md5(unique_id_string.encode('utf-8'))
            # Take first 8 bytes and convert to signed int64
            hash_bytes = hash_obj.digest()[:8]
            point_id = int.from_bytes(hash_bytes, byteorder='big', signed=True)
            # Ensure positive (Qdrant prefers positive IDs)
            if point_id < 0:
                point_id = abs(point_id)
            
            points.append(
                rest_models.PointStruct(
                    id=point_id,  # Unique ID per file+method+index combination
                    vector=vector,
                    payload=payload
                )
            )
        
        try:
            client.upsert(collection_name=name, points=points)
            print(f"  Batch {batch_idx + 1}/{total_batches}: Upserted {len(points)} points (indices {start_idx + 1}-{end_idx})")
        except Exception as e:
            raise RuntimeError(f"Error upserting batch {batch_idx + 1}/{total_batches} (indices {start_idx + 1}-{end_idx}): {e}")
    
    print(f"✅ Successfully upserted all {total_points} points to {name}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Legal Document Chunking + Qdrant Ingestion"
    )
    parser.add_argument('--input', required=True, help='Input file (PDF or TXT)')
    parser.add_argument(
        '--methods',
        nargs='+',
        default=['hierarchical'],
        choices=['recursive', 'semantic', 'structural', 'agentic', 'cluster', 'hierarchical'],
        help='Chunking methods to apply (can specify multiple)'
    )
    parser.add_argument(
        '--embed-model',
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Embedding model name'
    )
    parser.add_argument('--qdrant-host', help='Qdrant host URL')
    parser.add_argument('--collection', default='legal_documents', help='Collection name')
    parser.add_argument('--output', help='Output JSONL file')
    
    args = parser.parse_args()
    
    # Read input
    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {args.input}")
    
    if path.suffix.lower() == '.pdf':
        text, page_numbers = read_pdf(path)
    else:
        text, page_numbers = read_text(path)
    
    # Initialize chunker
    chunker = LegalDocumentChunker(args.embed_model)
    
    # Apply all specified chunking methods
    all_chunks = []
    for method in args.methods:
        print(f"Chunking with method: {method}")
        chunks = chunker.chunk(text, method, page_numbers)
        all_chunks.extend(chunks)
        print(f"  Generated {len(chunks)} chunks")
    
    print(f"\nTotal chunks: {len(all_chunks)}")
    
    # Save to JSONL if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for c in all_chunks:
                f.write(json.dumps(c, ensure_ascii=False) + '\n')
        print(f"Saved to {args.output}")
    
    # Upsert to Qdrant if requested
    if args.qdrant_host:
        if QdrantClient is None:
            raise RuntimeError("qdrant-client required")
        
        print(f"\nConnecting to Qdrant: {args.qdrant_host}")
        client = QdrantClient(url=args.qdrant_host)
        
        print("Generating embeddings...")
        texts = [c['text'] for c in all_chunks]
        embs = embed_texts(texts, args.embed_model)
        
        print(f"Creating collection: {args.collection}")
        create_collection(client, args.collection, len(embs[0]))
        
        print("Upserting to Qdrant...")
        upsert(client, args.collection, all_chunks, embs)
        
        print("Done!")


if __name__ == '__main__':
    main()
