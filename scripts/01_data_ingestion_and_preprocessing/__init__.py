"""
01_data_ingestion_and_preprocessing - Data Loading & Preprocessing

This module handles:
- Document loading (PDF, DOCX, TXT, MD)
- Text preprocessing and cleaning
- PDF extraction and parsing
- Document ingestion pipelines
"""

try:
    from .ingestion import DocumentIngestionPipeline
    from .pdf_handler import PDFHandler
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Some ingestion modules not available: {e}")
    DocumentIngestionPipeline = None
    PDFHandler = None

# Create convenience functions if PDFHandler is available
if PDFHandler is not None:
    _pdf_handler_instance = None
    
    def _get_pdf_handler():
        """Get or create PDFHandler instance."""
        global _pdf_handler_instance
        if _pdf_handler_instance is None:
            _pdf_handler_instance = PDFHandler()
        return _pdf_handler_instance
    
    def read_pdf(file_path: str) -> str:
        """Read PDF file and extract text."""
        handler = _get_pdf_handler()
        return handler.extract_text(file_path) if hasattr(handler, 'extract_text') else ""
    
    def read_docx(file_path: str) -> str:
        """Read DOCX file (placeholder - implement if needed)."""
        # TODO: Implement DOCX reading if needed
        return ""
    
    def read_text(file_path: str) -> str:
        """Read text file."""
        from pathlib import Path
        return Path(file_path).read_text(encoding='utf-8', errors='ignore')
else:
    def read_pdf(file_path: str) -> str:
        """Read PDF file (not available)."""
        return ""
    
    def read_docx(file_path: str) -> str:
        """Read DOCX file (not available)."""
        return ""
    
    def read_text(file_path: str) -> str:
        """Read text file."""
        from pathlib import Path
        return Path(file_path).read_text(encoding='utf-8', errors='ignore')

__all__ = [
    'DocumentIngestionPipeline',
    'PDFHandler',
    'read_pdf',
    'read_docx',
    'read_text',
]

