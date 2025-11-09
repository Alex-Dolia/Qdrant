"""
08_utilities - General Utilities

This module contains utility functions:
- Qdrant health checks
- Reproducibility logging
- General helper functions
"""

try:
    from .qdrant_health import check_qdrant_health
    from .reproducibility_logger import ReproducibilityLogger
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Some utility modules not available: {e}")

try:
    from .deepseek import DeepSeek
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"DeepSeek not available: {e}")
    DeepSeek = None

__all__ = [
    'check_qdrant_health',
    'ReproducibilityLogger',
    'DeepSeek',
]

