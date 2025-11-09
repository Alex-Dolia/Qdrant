"""
PDF Handler Module
Handles downloading PDFs from URLs and extracting text content.

Features:
- PDF downloading from URLs (arXiv, DOI links, direct PDFs)
- Text extraction using pdfplumber
- Metadata preservation
- Error handling and retry logic
"""

import os
import logging
import time
from typing import Dict, Optional, Any
from pathlib import Path
import traceback

logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests library not available. Install with: pip install requests")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber library not available. Install with: pip install pdfplumber")


class PDFHandler:
    """Handle PDF downloading and text extraction."""
    
    def __init__(self, output_dir: str = "output/papers", parsed_dir: str = "output/parsed_txt"):
        """
        Initialize PDF handler.
        
        Args:
            output_dir: Directory to save downloaded PDFs
            parsed_dir: Directory to save extracted text
        """
        self.output_dir = Path(output_dir)
        self.parsed_dir = Path(parsed_dir)
        
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"PDFHandler initialized: PDFs -> {self.output_dir}, Text -> {self.parsed_dir}")
    
    def download_pdf(self, url: str, paper_id: str, retries: int = 2) -> Optional[str]:
        """
        Download PDF from URL.
        
        Args:
            url: URL to PDF or paper page
            paper_id: Unique identifier for the paper (used as filename)
            retries: Number of retry attempts
            
        Returns:
            Path to downloaded PDF file, or None if download failed
        """
        if not REQUESTS_AVAILABLE:
            logger.warning("requests library not available. Cannot download PDFs.")
            return None
        
        # Check if already downloaded
        pdf_path = self.output_dir / f"{paper_id}.pdf"
        if pdf_path.exists():
            logger.info(f"PDF already exists: {pdf_path}")
            return str(pdf_path)
        
        # Try downloading
        for attempt in range(retries + 1):
            try:
                logger.info(f"Downloading PDF (attempt {attempt + 1}/{retries + 1}): {url}")
                
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                
                response = requests.get(url, stream=True, timeout=60, headers=headers)
                
                if response.status_code == 200:
                    # Check if content is actually a PDF
                    content_type = response.headers.get('content-type', '').lower()
                    if 'pdf' in content_type or url.lower().endswith('.pdf'):
                        with open(pdf_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        
                        # Verify file was written
                        if pdf_path.exists() and pdf_path.stat().st_size > 0:
                            logger.info(f"Successfully downloaded PDF: {pdf_path} ({pdf_path.stat().st_size} bytes)")
                            return str(pdf_path)
                        else:
                            logger.warning(f"Downloaded file is empty or missing: {pdf_path}")
                    else:
                        logger.warning(f"URL does not point to PDF (content-type: {content_type})")
                        # Try to resolve DOI if URL is not a direct PDF
                        if 'doi.org' in url or 'doi' in url.lower():
                            return self._resolve_doi_to_pdf(url, paper_id)
                else:
                    logger.warning(f"HTTP {response.status_code} for URL: {url}")
                    
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < retries:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error(f"Failed to download PDF after {retries + 1} attempts: {url}")
        return None
    
    def _resolve_doi_to_pdf(self, doi_url: str, paper_id: str) -> Optional[str]:
        """Try to resolve DOI to PDF URL."""
        try:
            # Follow redirects to find PDF
            response = requests.get(doi_url, allow_redirects=True, timeout=30)
            final_url = response.url
            
            # Check if final URL is a PDF
            if final_url.lower().endswith('.pdf') or 'pdf' in response.headers.get('content-type', '').lower():
                return self.download_pdf(final_url, paper_id, retries=1)
            
            logger.info(f"DOI resolved to: {final_url} (not a direct PDF)")
        except Exception as e:
            logger.warning(f"DOI resolution failed: {e}")
        
        return None
    
    def extract_text(self, pdf_path: str, paper_id: str = None) -> str:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            paper_id: Optional paper ID (used for cached text file)
            
        Returns:
            Extracted text content
        """
        if not PDFPLUMBER_AVAILABLE:
            logger.warning("pdfplumber library not available. Cannot extract text from PDFs.")
            return ""
        
        # Check if text already extracted
        if paper_id:
            text_path = self.parsed_dir / f"{paper_id}.txt"
            if text_path.exists():
                logger.info(f"Using cached extracted text: {text_path}")
                try:
                    with open(text_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    logger.warning(f"Failed to read cached text: {e}")
        
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return ""
        
        try:
            logger.info(f"Extracting text from PDF: {pdf_path}")
            text_parts = []
            
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"Processing {total_pages} pages...")
                
                for i, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {i + 1}: {e}")
                        continue
            
            full_text = "\n\n".join(text_parts)
            
            # Save extracted text if paper_id provided
            if paper_id and full_text:
                text_path = self.parsed_dir / f"{paper_id}.txt"
                try:
                    with open(text_path, 'w', encoding='utf-8') as f:
                        f.write(full_text)
                    logger.info(f"Saved extracted text: {text_path} ({len(full_text)} chars)")
                except Exception as e:
                    logger.warning(f"Failed to save extracted text: {e}")
            
            logger.info(f"Extracted {len(full_text)} characters from PDF")
            return full_text
            
        except Exception as e:
            logger.error(f"PDF text extraction error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return ""
    
    def process_paper(self, paper_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Download PDF and extract text for a paper.
        
        Args:
            paper_metadata: Paper metadata dictionary with 'id', 'url', 'doi', etc.
            
        Returns:
            Updated paper metadata with 'pdf_path' and 'full_text' fields
        """
        paper_id = paper_metadata.get("id", "unknown")
        url = paper_metadata.get("url", "")
        doi = paper_metadata.get("doi")
        
        # Try to download PDF
        pdf_path = None
        if url:
            pdf_path = self.download_pdf(url, paper_id)
        
        # If download failed and DOI available, try DOI resolution
        if not pdf_path and doi:
            doi_url = f"https://doi.org/{doi}"
            pdf_path = self.download_pdf(doi_url, paper_id)
        
        # Extract text if PDF downloaded
        full_text = ""
        if pdf_path:
            full_text = self.extract_text(pdf_path, paper_id)
        
        # Update metadata
        updated_metadata = paper_metadata.copy()
        updated_metadata["pdf_path"] = pdf_path
        updated_metadata["full_text"] = full_text
        updated_metadata["text_length"] = len(full_text)
        
        return updated_metadata
    
    def is_available(self) -> bool:
        """Check if PDF handling is available."""
        return REQUESTS_AVAILABLE and PDFPLUMBER_AVAILABLE

