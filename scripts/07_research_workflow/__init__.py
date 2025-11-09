"""
07_research_workflow - Research-Specific Workflows

This module handles:
- Research overview generation
- Academic paper summarization
- Literature review workflows
- JMLR-specific processing
- Research assistant agents
- Web search integration
- Academic paper search
- Search result caching
"""

try:
    from .research_overview_workflow import ResearchOverviewWorkflow
    from .paper_summarizer import PaperSummarizer
    from .jmlr_literature_review import JMLRLiteratureReviewAgent
    from .research_assistant import ResearchAssistant
    from .research_agents import ResearchAgent, EnhancedResearchWorkflow
    from .web_search import WebSearch
    from .academic_search import AcademicSearch
    from .web_cache import WebSearchCache, WebCache
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Some research workflow modules not available: {e}")
    # Set defaults to None for missing imports
    ResearchOverviewWorkflow = None
    PaperSummarizer = None
    JMLRLiteratureReviewAgent = None
    ResearchAssistant = None
    ResearchAgent = None
    EnhancedResearchWorkflow = None
    WebSearch = None
    AcademicSearch = None
    WebSearchCache = None
    WebCache = None

# Create alias for backward compatibility
JMLRLiteratureReview = JMLRLiteratureReviewAgent

__all__ = [
    'ResearchOverviewWorkflow',
    'PaperSummarizer',
    'JMLRLiteratureReview',
    'JMLRLiteratureReviewAgent',  # Also export the actual class name
    'ResearchAssistant',
    'ResearchAgent',
    'EnhancedResearchWorkflow',
    'WebSearch',
    'AcademicSearch',
    'WebSearchCache',
    'WebCache',
]

