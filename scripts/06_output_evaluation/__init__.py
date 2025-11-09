"""
06_output_evaluation - RAG Output Quality Assessment

This module handles:
- RAGAS evaluation
- Metric calculation
- Evaluation dataset generation
- Performance analysis
"""

try:
    from .ragas_evaluation import (
        load_documents_from_directory,
        create_rag_chain,
        evaluate_rag_with_ragas,
        evaluate_all_combinations
    )
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Evaluation modules not available: {e}")

try:
    from .ollama_report import (
        generate_report_with_ollama31,
        save_report,
        create_summary_data,
        get_save_root as get_save_root_ollama
    )
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Ollama report module not available: {e}")

try:
    from .run_rag_evaluation import (
        run_rag_evaluation,
        get_save_root,
        load_documents_from_directory as load_documents_from_directory_eval
    )
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"RAG evaluation runner not available: {e}")

__all__ = [
    'load_documents_from_directory',
    'create_rag_chain',
    'evaluate_rag_with_ragas',
    'evaluate_all_combinations',
    'generate_report_with_ollama31',
    'save_report',
    'create_summary_data',
    'get_save_root',
    'run_rag_evaluation',
    'load_documents_from_directory_eval',
]

