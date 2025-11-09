"""
05_output_generation - Output Formatting & Generation

This module handles:
- Report formatting
- HTML export
- Graph building
- Output export utilities
"""

try:
    from .report_formatter import ReportFormatter
    from .html_exporter import save_research_overview_html, sanitize_topic_name
    from .exporter import export_results
    from .graph_builder import GraphBuilder
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Some output generation modules not available: {e}")
    ReportFormatter = None
    save_research_overview_html = None
    sanitize_topic_name = None
    export_results = None
    GraphBuilder = None

# Create HTMLExporter class wrapper for backward compatibility
if save_research_overview_html is not None:
    class HTMLExporter:
        """Wrapper class for HTML export functions."""
        
        @staticmethod
        def save_research_overview(report: dict, topic: str, base_dir: str = "html"):
            """Save research overview as HTML."""
            return save_research_overview_html(report, topic, base_dir)
        
        @staticmethod
        def sanitize_topic(topic: str) -> str:
            """Sanitize topic name."""
            return sanitize_topic_name(topic)
else:
    HTMLExporter = None

__all__ = [
    'ReportFormatter',
    'HTMLExporter',
    'save_research_overview_html',
    'sanitize_topic_name',
    'export_results',
    'GraphBuilder',
]

