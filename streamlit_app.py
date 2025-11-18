"""
Simplified RAG System - Qdrant + Llama 3.1 Only
Query documents or perform web search to generate reports.
"""

import streamlit as st
import os
import sys
import time
import json
import uuid
import base64
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import re
import logging
import hashlib
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import uuid
from uuid import uuid4

# Caching configuration
CACHE_DIR = Path("test_embedding")
CACHE_DIR.mkdir(exist_ok=True)

def get_model_cache_path(model_name: str) -> Path:
    """Get the cache path for a specific model."""
    # Create a safe filename from model name
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)
    return CACHE_DIR / safe_name

def generate_interpretation(results: Dict[str, Any]) -> str:
    """Generate interpretation text for the test results.
    
    Args:
        results: Dictionary containing test results and metrics
        
    Returns:
        str: Formatted interpretation text
    """
    metrics = results.get('metrics', {})
    
    # Basic metrics
    accuracy = metrics.get('accuracy', 0)
    precision = metrics.get('precision', 0)
    recall = metrics.get('recall', 0)
    f1 = metrics.get('f1_score', 0)
    roc_auc = metrics.get('roc_auc', 0)
    pr_auc = metrics.get('pr_auc', 0)
    threshold = metrics.get('optimal_threshold', 0.5)
    
    # Generate interpretation
    interpretation = f"""# Model Performance Interpretation

## 📊 Key Metrics
- **Accuracy**: {accuracy:.2%}
- **Precision**: {precision:.2f}
- **Recall**: {recall:.2f}
- **F1 Score**: {f1:.2f}
- **ROC AUC**: {roc_auc:.3f}
- **PR AUC**: {pr_auc:.3f}
- **Optimal Threshold**: {threshold:.3f}

## 📈 Performance Analysis
"""
    
    # Add performance assessment
    if roc_auc >= 0.9:
        interpretation += "- 🎉 Excellent discrimination ability (ROC AUC ≥ 0.9)\n"
    elif roc_auc >= 0.8:
        interpretation += "- 👍 Good discrimination ability (0.8 ≤ ROC AUC < 0.9)\n"
    elif roc_auc >= 0.7:
        interpretation += "- ℹ️ Fair discrimination ability (0.7 ≤ ROC AUC < 0.8)\n"
    else:
        interpretation += "- ⚠️ Poor discrimination ability (ROC AUC < 0.7)\n"
        
    # Add threshold interpretation
    interpretation += f"\n## 🎯 Threshold Analysis\n"
    if threshold > 0.7:
        interpretation += f"- The model uses a high similarity threshold ({threshold:.3f}), indicating it requires strong evidence to classify pairs as similar.\n"
    elif threshold < 0.3:
        interpretation += f"- The model uses a low similarity threshold ({threshold:.3f}), making it more likely to classify pairs as similar.\n"
    else:
        interpretation += f"- The model uses a moderate similarity threshold ({threshold:.3f}).\n"
    
    # Add recommendations
    interpretation += "\n## 💡 Recommendations\n"
    if f1 < 0.7:
        interpretation += "- Consider fine-tuning the embedding model or trying a different architecture.\n"
    if abs(recall - precision) > 0.3:
        if recall > precision:
            interpretation += "- The model has higher recall than precision, meaning it may be too permissive in classifying pairs as similar.\n"
        else:
            interpretation += "- The model has higher precision than recall, meaning it may be too conservative in classifying pairs as similar.\n"
    
    interpretation += "- For production use, consider adjusting the threshold based on your specific needs for precision vs. recall.\n"
    interpretation += "- Monitor the model's performance on new data to ensure it generalizes well.\n"
    
    return interpretation

def save_test_results(model_name: str, results: Dict[str, Any]) -> None:
    """Save test results to disk for caching.
    
    Saves the following components:
    1. Test metrics (accuracy, precision, recall, etc.)
    2. Similarity analysis plots
    3. Confusion matrix
    4. Sample predictions
    5. Interpretation text
    """
    try:
        cache_path = get_model_cache_path(model_name)
        
        # 1. Save metrics to JSON
        metrics = {
            'model_name': model_name,
            'best_threshold': results.get('metrics', {}).get('optimal_threshold'),
            'accuracy': results.get('metrics', {}).get('accuracy'),
            'precision': results.get('metrics', {}).get('precision'),
            'recall': results.get('metrics', {}).get('recall'),
            'f1': results.get('metrics', {}).get('f1_score'),
            'roc_auc': results.get('metrics', {}).get('roc_auc'),
            'pr_auc': results.get('metrics', {}).get('pr_auc'),
            'confusion_matrix': results.get('confusion_matrix', []).tolist() if 'confusion_matrix' in results else [],
            
            # Add confusion matrix values
            'true_negatives': int(results.get('confusion_matrix', [[0, 0], [0, 0]])[0][0]) if 'confusion_matrix' in results else 0,
            'false_positives': int(results.get('confusion_matrix', [[0, 0], [0, 0]])[0][1]) if 'confusion_matrix' in results else 0,
            'false_negatives': int(results.get('confusion_matrix', [[0, 0], [0, 0]])[1][0]) if 'confusion_matrix' in results else 0,
            'true_positives': int(results.get('confusion_matrix', [[0, 0], [0, 0]])[1][1]) if 'confusion_matrix' in results else 0
        }
        
        # Save DataFrame with predictions if it exists
        if 'data' in results and results['data'] is not None:
            df = results['data']
            metrics['sample_predictions'] = df.sample(min(10, len(df))).to_dict(orient='records')
        
        # Save metrics to JSON file
        with open(f"{cache_path}.json", 'w') as f:
            json.dump(metrics, f)
            
        # 2. Save similarity analysis plots
        if 'figure' in results and results['figure'] is not None:
            fig = results['figure']
            for i, ax in enumerate(fig.axes, 1):
                # Special handling for histogram subplot
                if 'Distribution of Cosine Similarities' in ax.get_title():
                    # Create a new figure for the histogram with threshold
                    fig_hist = plt.figure(figsize=(10, 6))
                    ax_hist = fig_hist.add_subplot(111)
                    
                    # Recreate the histogram with the same data
                    if 'data' in results and isinstance(results['data'], pd.DataFrame):
                        df = results['data']
                        # Plot histogram for similar and non-similar pairs
                        sns.histplot(
                            data=df, x='similarity', hue='similar', 
                            bins=30, alpha=0.6, palette={True: 'orange', False: 'blue'},
                            ax=ax_hist
                        )
                        
                        # Add threshold line
                        threshold = results.get('metrics', {}).get('optimal_threshold', 0.5)
                        ax_hist.axvline(x=threshold, color='red', linestyle='--', 
                                     label=f'Decision Threshold: {threshold:.2f}')
                        
                        # Set labels and title
                        ax_hist.set_title('Distribution of Cosine Similarities')
                        ax_hist.set_xlabel('Cosine Similarity')
                        ax_hist.set_ylabel('Count')
                        ax_hist.legend()
                        
                        # Save the histogram
                        plt.savefig(f"{cache_path}_distribution_of_cosine_similarities.png", 
                                  dpi=300, bbox_inches='tight')
                        plt.close(fig_hist)
                else:
                    # Handle other plot types (ROC, PR curves)
                    fig2 = plt.figure(figsize=(8, 6))
                    ax2 = fig2.add_subplot(111)
                    
                    # Copy the content from the original subplot
                    for line in ax.get_lines():
                        ax2.plot(line.get_xdata(), line.get_ydata(), 
                                color=line.get_color(), 
                                linestyle=line.get_linestyle(),
                                linewidth=line.get_linewidth(),
                                label=line.get_label())
                    
                    # Copy other plot elements
                    ax2.set_title(ax.get_title())
                    ax2.set_xlabel(ax.get_xlabel())
                    ax2.set_ylabel(ax.get_ylabel())
                    if ax.get_legend() is not None:
                        ax2.legend()
                    
                    # Save individual plot
                    plot_type = ax.get_title().lower().replace(' ', '_')
                    if plot_type:  # Only save if we have a valid plot type
                        plt.savefig(f"{cache_path}_{plot_type}.png", dpi=300, bbox_inches='tight')
                    plt.close(fig2)
            
            plt.close(fig)
        
        # 3. Save confusion matrix if it exists
        if 'confusion_matrix' in results and results['confusion_matrix'] is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            try:
                from sklearn.metrics import ConfusionMatrixDisplay
                ConfusionMatrixDisplay(
                    confusion_matrix=results['confusion_matrix'],
                    display_labels=["Not Similar", "Similar"]
                ).plot(ax=ax, cmap='Blues')
                plt.title("Confusion Matrix")
                plt.savefig(f"{cache_path}_confusion_matrix.png", dpi=300, bbox_inches='tight')
                plt.close(fig)
            except ImportError:
                pass
        
        # 4. Save sample predictions as CSV
        if 'data' in results and isinstance(results['data'], pd.DataFrame):
            df = results['data']
            sample_df = df.sample(min(100, len(df)))
            sample_df.to_csv(f"{cache_path}_sample_predictions.csv", index=False)
        
        logger.info(f"Saved test results for model: {model_name} to {cache_path}_*")
        
    except Exception as e:
        logger.error(f"Error saving test results: {e}", exc_info=True)
        raise

def load_test_results(model_name: str) -> Optional[Dict[str, Any]]:
    """Load test results from disk if available.
    
    Returns:
        Dictionary with keys:
        - 'metrics': dict of evaluation metrics
        - 'data': DataFrame with predictions
        - 'figure': matplotlib figure (if available)
        - 'confusion_matrix': confusion matrix (if available)
        - 'cached': bool indicating if results were loaded from cache
        - 'missing_components': list of missing components if any
        - 'cache_status': dict with detailed status of each cache file
    """
    try:
        cache_path = get_model_cache_path(model_name)
        base_path = str(cache_path)
        
        # Define all possible cache files with their actual names
        cache_files = {
            'metrics': f"{base_path}.json",
            'sample_predictions': f"{base_path}_sample_predictions.csv",
            'distribution_plot': f"{base_path}_distribution_of_cosine_similarities.png",
            'roc_curve': f"{base_path}_receiver_operating_characteristic_(roc)_curve.png",
            'pr_curve': f"{base_path}_precision-recall_curve.png",
            'confusion_matrix_plot': f"{base_path}_confusion_matrix.png"
        }
        
        # Check which files exist and their status
        cache_status = {}
        for name, path in cache_files.items():
            exists = os.path.exists(path)
            cache_status[name] = {
                'exists': exists,
                'path': path,
                'size': os.path.getsize(path) if exists else 0
            }
        
        # Get missing files
        missing_files = [name for name, status in cache_status.items() if not status['exists']]
        
        # Critical files that must exist for cache to be valid
        critical_files = {'metrics', 'sample_predictions'}
        missing_critical = [f for f in critical_files if f in missing_files]
        
        # If we're missing critical files, return None
        if missing_critical:
            logger.info(f"Cache miss for {model_name}: Missing critical files: {', '.join(missing_critical)}")
            return {
                'cached': False,
                'missing_components': missing_files,
                'missing_critical': missing_critical,
                'cache_status': cache_status
            }
        
        # Log cache hit
        logger.info(f"Loading cached results for model: {model_name}")
        
        # Load metrics
        try:
            with open(cache_files['metrics'], 'r') as f:
                metrics = json.load(f)
            
            # Ensure all required metrics exist with default values if missing
            required_metrics = {
                'accuracy': 0.5,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'roc_auc': 0.5,
                'pr_auc': 0.5,
                'optimal_threshold': 0.5
            }
            
            # Update with any existing metrics, keeping defaults for missing ones
            for key, default in required_metrics.items():
                metrics[key] = metrics.get(key, default)
                
            logger.info("✓ Loaded metrics from cache")
        except Exception as e:
            logger.error(f"Error loading metrics from cache: {e}")
            return {
                'cached': False,
                'missing_components': ['metrics (corrupted)'] + missing_files,
                'cache_status': cache_status
            }
        
        # Load sample predictions
        try:
            data = pd.read_csv(cache_files['sample_predictions'])
            logger.info("✓ Loaded sample predictions from cache")
            
            # Calculate metrics from the data if not in metrics
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score, 
                roc_auc_score, average_precision_score, confusion_matrix, roc_curve
            )
            
            try:
                # Calculate metrics from the data
                y_true = data['label']
                y_scores = data['similarity']
                
                # Calculate ROC curve and find optimal threshold using Youden's J statistic
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                if len(thresholds) > 0:  # Ensure we have thresholds
                    j_scores = tpr - fpr
                    optimal_idx = np.argmax(j_scores)
                    optimal_threshold = float(thresholds[optimal_idx])
                else:
                    optimal_threshold = 0.5
                
                # Use optimal threshold for predictions
                y_pred = (y_scores >= optimal_threshold).astype(int)
                
                # Calculate all metrics with error handling
                metrics = {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, zero_division=0),
                    'recall': recall_score(y_true, y_pred, zero_division=0),
                    'f1_score': f1_score(y_true, y_pred, zero_division=0),
                    'roc_auc': roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.5,
                    'pr_auc': average_precision_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.5,
                    'optimal_threshold': optimal_threshold,
                    'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
                }
            except Exception as e:
                logger.error(f"Error calculating metrics: {e}")
                # Fallback to default metrics if calculation fails
                metrics = {
                    'accuracy': 0.5,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'roc_auc': 0.5,
                    'pr_auc': 0.5,
                    'optimal_threshold': 0.5,
                    'confusion_matrix': [[0, 0], [0, 0]]
                }
        except Exception as e:
            logger.error(f"Error loading sample predictions from cache: {e}")
            return {
                'cached': False,
                'missing_components': ['sample_predictions (corrupted)'] + missing_files,
                'cache_status': cache_status
            }
        
        # Create a figure with subplots for the plots
        fig = plt.figure(figsize=(20, 6))
        
        # Create subplots for each visualization
        ax1 = plt.subplot(1, 3, 1)  # Distribution plot
        ax2 = plt.subplot(1, 3, 2)  # ROC curve
        ax3 = plt.subplot(1, 3, 3)  # PR curve
        
        # Load distribution plot if available
        if cache_status['distribution_plot']['exists']:
            try:
                img = plt.imread(cache_files['distribution_plot'])
                ax1.imshow(img)
                ax1.axis('off')
                ax1.set_title('Distribution of Cosine Similarities')
            except Exception as e:
                logger.warning(f"Could not load distribution plot: {e}")
        
        # Load ROC curve if available
        if cache_status['roc_curve']['exists']:
            try:
                img = plt.imread(cache_files['roc_curve'])
                ax2.imshow(img)
                ax2.axis('off')
                ax2.set_title('ROC Curve')
            except Exception as e:
                logger.warning(f"Could not load ROC curve: {e}")
        
        # Load PR curve if available
        if cache_status['pr_curve']['exists']:
            try:
                img = plt.imread(cache_files['pr_curve'])
                ax3.imshow(img)
                ax3.axis('off')
                ax3.set_title('Precision-Recall Curve')
            except Exception as e:
                logger.warning(f"Could not load PR curve: {e}")
        
        plt.tight_layout()
        
        # Get confusion matrix from metrics or calculate it
        cm = None
        if 'confusion_matrix' in metrics:
            try:
                cm = np.array(metrics['confusion_matrix'])
            except Exception as e:
                logger.warning(f"Could not load confusion matrix: {e}")
        
        # Check for any missing non-critical files
        non_critical_missing = [f for f in missing_files if f not in critical_files]
        
        result = {
            'metrics': metrics,
            'data': data,
            'figure': fig,
            'confusion_matrix': cm,
            'cached': True,
            'missing_components': non_critical_missing,
            'cache_status': cache_status
        }
        
        if non_critical_missing:
            logger.info(f"Partially loaded from cache. Missing non-critical files: {', '.join(non_critical_missing)}")
        else:
            logger.info("✓ Successfully loaded all data from cache")
            
        return result
        
    except Exception as e:
        logger.error(f"Error loading cached results: {e}", exc_info=True)
        return None

# Import pandas for dataframes (used in file statistics and chunk exploration)

# Setup logging (lightweight - only console by default, file logging deferred)
logger = logging.getLogger(__name__)
if not logger.handlers:  # Only setup if not already configured
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]  # Console only for faster startup
    )
    # File logging can be added later if needed
    # os.makedirs("logs", exist_ok=True)
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # log_file = f"logs/logger_{timestamp}.log"
    # logger.addHandler(logging.FileHandler(log_file, encoding='utf-8', mode='a'))

# Setup signal handlers for graceful shutdown
def signal_handler(sig, frame):
    """Handle Ctrl-C gracefully."""
    logger.info("Received interrupt signal (Ctrl-C). Shutting down gracefully...")
    print("\n\n⚠️  Interrupt received. Shutting down gracefully...")
    print("Please wait for current operations to complete...")
    
    # Set a flag to stop processing
    if 'interrupt_flag' not in st.session_state:
        st.session_state.interrupt_flag = True
    
    # Try to stop any running operations
    try:
        # Cancel any running async operations if possible
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Try to cancel all tasks
            tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for task in tasks:
                task.cancel()
    except Exception:
        pass
    
    # Exit the application
    os._exit(0)  # Use os._exit() for immediate termination

# Register signal handler for SIGINT (Ctrl-C)
# This allows the app to be stopped with Ctrl-C in the terminal
if os.name != 'nt':  # Only set up signal handlers on Unix-like systems
    try:
        import signal
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    except Exception as e:
        logger.warning(f"Could not set up signal handlers: {e}")
else:
    logger.info("Running on Windows - signal handlers not set up")
    logger.warning("Signal handlers might not work in all environments (e.g., some IDEs)")
    logger.info("You can still stop the app with Ctrl-C in the terminal where Streamlit is running")
    
    # On Windows, try to handle SIGTERM if available
    try:
        import signal
        signal.signal(signal.SIGTERM, signal_handler)
    except (ImportError, AttributeError, ValueError, OSError) as e:
        # SIGTERM might not be available or signal module might not be fully functional
        logger.debug(f"Could not set up SIGTERM handler on Windows: {e}")
        pass

# Initialize interrupt flag in session state
if 'interrupt_flag' not in st.session_state:
    st.session_state.interrupt_flag = False

# Lazy import helper functions - only import modules when needed
@st.cache_resource
def _get_legal_chunker_module():
    """Lazy load legal chunker module (cached)."""
    import importlib
    return importlib.import_module('scripts.00_chunking.legal_chunker_integration')

@st.cache_resource
def _get_research_modules():
    """Lazy load research workflow modules (cached)."""
    import importlib
    modules = {}
    try:
        modules['retrieval'] = importlib.import_module('scripts.03_retrieval.retrieval')
        modules['web_search'] = importlib.import_module('scripts.07_research_workflow.web_search')
        modules['research_assistant'] = importlib.import_module('scripts.07_research_workflow.research_assistant')
        modules['research_overview'] = importlib.import_module('scripts.07_research_workflow.research_overview_workflow')
        modules['synthesis'] = importlib.import_module('scripts.02_query_completion.synthesis')
        modules['report_formatter'] = importlib.import_module('scripts.05_output_generation.report_formatter')
    except ImportError as e:
        logger.warning(f"Some research modules not available: {e}")
    return modules

@st.cache_resource
def _get_deepseek_module():
    """Lazy load DeepSeek module (cached)."""
    import importlib
    try:
        return importlib.import_module('scripts.08_utilities.deepseek'), True
    except ImportError:
        return None, False

# Load prompt templates lazily (only when needed)
@st.cache_resource
def _get_prompt_templates():
    """Lazy load prompt templates (cached)."""
    try:
        from prompts import load_prompt_template, PROMPTS_DIR
        return load_prompt_template, PROMPTS_DIR, True
    except ImportError:
        return None, None, False

# Import only essential modules for Legal Documents (most commonly used)
import importlib
try:
    chunking_module = importlib.import_module('scripts.00_chunking.legal_chunker_integration')
    
    def process_legal_document(
        file_path: str,
        qdrant_client,
        collection_name: str,
        embedding_model: str,
        chunking_method: str,
        file_metadata: dict
    ) -> dict:
        """
        Process a legal document and store it in Qdrant.
        
        Args:
            file_path: Path to the document file
            qdrant_client: Qdrant client instance
            collection_name: Name of the collection to store the document
            embedding_model: Name of the embedding model to use
            chunking_method: Chunking method to apply
            file_metadata: Additional metadata for the document
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Call the ingest_legal_document function from the legal chunker module
            result = chunking_module.ingest_legal_document(
                file_path=file_path,
                qdrant_client=qdrant_client,
                collection_name=collection_name,
                embedding_model=embedding_model,
                chunking_methods=[chunking_method],
                file_id=file_metadata.get('file_id')
            )
            
            # Add file metadata to the result
            result['file_metadata'] = file_metadata
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'file_metadata': file_metadata
            }
    ingest_legal_document = chunking_module.ingest_legal_document
    query_legal_documents = chunking_module.query_legal_documents
    query_legal_documents_with_reranking = chunking_module.query_legal_documents_with_reranking
    get_distinct_source_files = chunking_module.get_distinct_source_files
    get_distinct_chunking_methods = chunking_module.get_distinct_chunking_methods
    get_distinct_chunking_methods_for_file = chunking_module.get_distinct_chunking_methods_for_file
    get_distinct_embedding_models = chunking_module.get_distinct_embedding_models
    get_available_chunking_methods = chunking_module.get_available_chunking_methods
    get_default_chunking_methods = chunking_module.get_default_chunking_methods
    delete_file_from_qdrant = chunking_module.delete_file_from_qdrant
    delete_all_files_from_qdrant = chunking_module.delete_all_files_from_qdrant
    get_file_statistics = chunking_module.get_file_statistics
    get_chunks_for_exploration = chunking_module.get_chunks_for_exploration
    get_embedding_models_for_file_and_method = chunking_module.get_embedding_models_for_file_and_method
    compute_chunk_similarities = chunking_module.compute_chunk_similarities
    EMBEDDING_MODELS = chunking_module.EMBEDDING_MODELS
    LEGAL_CHUNKER_AVAILABLE = chunking_module.LEGAL_CHUNKER_AVAILABLE
    
    # Import embedding functions
    qdrant_chunker_module = importlib.import_module('scripts.00_chunking.qdrant_chunker')
    embed_texts = qdrant_chunker_module.embed_texts
    create_collection = qdrant_chunker_module.create_collection
except ImportError as e:
    logger.error(f"Failed to import legal chunker modules: {e}")
    LEGAL_CHUNKER_AVAILABLE = False
    EMBEDDING_MODELS = []

def test_embedding_quality(embedding_model: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Test embedding model quality using word pairs from a CSV file.
    
    Args:
        embedding_model: Name of the embedding model to test
        use_cache: Whether to use cached results if available
        
    Returns:
        Dictionary containing test results including metrics and visualizations
    """
    # Check cache first if enabled
    if use_cache:
        logger.info(f"🔍 Checking for cached results for model: {embedding_model}")
        with st.spinner("Checking for cached results..."):
            cached_results = load_test_results(embedding_model)
            
        if cached_results and cached_results.get('cached', False):
            missing_components = cached_results.get('missing_components', [])
            cache_status = cached_results.get('cache_status', {})
            
            # Show detailed cache status
            cache_info = []
            for name, status in cache_status.items():
                status_emoji = "✅" if status['exists'] else "❌"
                size_mb = status['size'] / (1024 * 1024)
                cache_info.append(f"{status_emoji} {name}: {size_mb:.2f} MB")
            
            st.info("\n".join(["**Cache Status:**"] + cache_info))
            
            if not missing_components:
                st.success(f"✅ Using fully cached results for model: {embedding_model}")
                logger.info(f"Using fully cached results for model: {embedding_model}")
                return cached_results
            else:
                missing_str = ", ".join(missing_components)
                st.warning(f"⚠️ Found partial cache for {embedding_model}. Missing: {missing_str}")
                st.info("Computing missing components and updating cache...")
                logger.info(f"Found partial cache for {embedding_model}, missing: {missing_components}")
        elif cached_results is not None:  # Cache exists but is incomplete
            missing_critical = cached_results.get('missing_critical', [])
            st.warning(f"❌ Cache incomplete. Missing critical files: {', '.join(missing_critical)}")
            st.info("Proceeding with full computation...")
        else:
            st.info("ℹ️ No valid cache found. Starting full computation...")
    else:
        cached_results = None
        st.info("ℹ️ Cache disabled, performing full computation...")
        logger.info("Cache disabled, forcing recomputation")
    
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        roc_curve, auc, precision_recall_curve, average_precision_score,
        confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
    )
    from sklearn.metrics import roc_auc_score
    
    # Load the word pairs CSV file
    dataset_path = "data/small_data/word_pairs.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Word pairs file not found at: {dataset_path}")
    
    # Load and preprocess the data
    df = pd.read_csv(dataset_path)
    
    # Convert boolean 'similar' column to numeric (1 for True, 0 for False)
    df['label'] = df['similar'].astype(int)
    
    # Get unique words and their embeddings
    unique_words = list(set(df['word1'].tolist() + df['word2'].tolist()))
    
    # Generate embeddings for all unique words
    word_embeddings = {}
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Generate embeddings using Ollama
    try:
        from langchain_ollama import OllamaEmbeddings
        import os
        
        # Remove 'ollama/' prefix if present
        ollama_model = embedding_model.replace("ollama/", "")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        embeddings = OllamaEmbeddings(
            model=ollama_model,
            base_url=base_url
        )
        
        for idx, word in enumerate(unique_words):
            if idx % 10 == 0:
                progress = (idx + 1) / len(unique_words)
                progress_bar.progress(progress)
                status_text.info(f"Computing embeddings: {idx + 1}/{len(unique_words)} words...")
            word_embeddings[word] = np.array(embeddings.embed_query(word))
    except ImportError:
        raise ImportError("langchain-ollama package is required for Ollama embeddings")
    except Exception as e:
        raise RuntimeError(f"Error generating embeddings with Ollama: {str(e)}")
    
    progress_bar.progress(1.0)
    status_text.empty()
    progress_bar.empty()
    
    # Compute cosine similarities for each pair
    similarities = []
    for _, row in df.iterrows():
        word1 = row['word1']
        word2 = row['word2']
        
        # Get embeddings
        emb1 = word_embeddings[word1]
        emb2 = word_embeddings[word2]
        
        # Compute cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        similarities.append(similarity)
    
    df['similarity'] = similarities
    
    # Find optimal threshold using F1 score
    thresholds = np.linspace(0, 1, 100)
    f1_scores = [f1_score(df['label'], df['similarity'] >= t) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    # Compute predictions using the best threshold
    y_pred = (df['similarity'] >= best_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(df['label'], y_pred)
    precision = precision_score(df['label'], y_pred)
    recall = recall_score(df['label'], y_pred)
    f1 = f1_score(df['label'], y_pred)
    
    # Calculate ROC AUC and PR AUC
    fpr, tpr, _ = roc_curve(df['label'], df['similarity'])
    roc_auc = roc_auc_score(df['label'], df['similarity'])
    precision_curve, recall_curve, _ = precision_recall_curve(df['label'], df['similarity'])
    pr_auc = average_precision_score(df['label'], df['similarity'])
    
    # Create visualizations - 1x3 subplot
    fig = plt.figure(figsize=(20, 6))
    
    # 1. Histogram of similarities (left)
    ax1 = plt.subplot(1, 3, 1)
    sns.histplot(
        data=df, x='similarity', hue='similar', 
        bins=30, alpha=0.6, palette={True: 'orange', False: 'blue'}, 
        ax=ax1
    )
    ax1.axvline(x=best_threshold, color='red', linestyle='--', 
               label=f'Optimal Threshold: {best_threshold:.2f}')
    ax1.set_title('Distribution of Cosine Similarities')
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Count')
    ax1.legend()
    
    # 2. ROC Curve (middle)
    ax2 = plt.subplot(1, 3, 2)
    fpr, tpr, _ = roc_curve(df['label'], df['similarity'])
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve (right)
    ax3 = plt.subplot(1, 3, 3)
    precision_curve, recall_curve, _ = precision_recall_curve(df['label'], df['similarity'])
    pr_auc = average_precision_score(df['label'], df['similarity'])
    
    ax3.plot(recall_curve, precision_curve, color='green', lw=2,
            label=f'PR Curve (AP = {pr_auc:.3f})')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall Curve')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.legend(loc='lower left')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Confusion matrix
    cm = confusion_matrix(df['label'], y_pred)
    
    # Create results dictionary
    results = {
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'optimal_threshold': best_threshold
        },
        'confusion_matrix': cm,
        'figure': fig,
        'data': df
    }
    
    # Close the figure to free memory
    plt.close(fig)
    
    # Save results to cache
    try:
        save_test_results(embedding_model, results)
    except Exception as e:
        logger.warning(f"Failed to save results to cache: {e}")
    
    return results
    
    # Define test word pairs
    # Positive pairs: words that should be similar
    positive_pairs = [
        ("cat", "dog"),  # Both animals
        ("apple", "orange"),  # Both fruits
        ("chair", "table"),  # Both furniture
        ("car", "vehicle"),  # Related concepts
        ("happy", "joyful"),  # Similar emotions
        ("book", "novel"),  # Related items
        ("water", "liquid"),  # Related concepts
        ("run", "sprint"),  # Similar actions
    ]
    
    # Negative pairs: words that should be dissimilar
    negative_pairs = [
        ("cat", "chair"),  # Animal vs furniture
        ("apple", "car"),  # Fruit vs vehicle
        ("happy", "table"),  # Emotion vs furniture
        ("book", "water"),  # Object vs substance
        ("run", "orange"),  # Action vs fruit
        ("dog", "novel"),  # Animal vs object
        ("chair", "joyful"),  # Furniture vs emotion
        ("car", "sprint"),  # Vehicle vs action
    ]
    
    # Collect all unique words
    all_words = set()
    for pair in positive_pairs + negative_pairs:
        all_words.add(pair[0])
        all_words.add(pair[1])
    all_words = sorted(list(all_words))
    
    # Generate embeddings for all words
    try:
        embeddings = embed_texts(all_words, embedding_model)
    except Exception as e:
        raise RuntimeError(f"Failed to generate embeddings: {e}")
    
    # Create a test collection (separate from production)
    test_collection_name = f"{collection_name}_embedding_test"
    vector_dim = len(embeddings[0])
    
    try:
        # Create test collection if it doesn't exist
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if test_collection_name not in collection_names:
            qdrant_client.create_collection(
                collection_name=test_collection_name,
                vectors_config=VectorParams(
                    size=vector_dim,
                    distance=Distance.COSINE
                )
            )
        else:
            # Clear existing test collection
            qdrant_client.delete_collection(collection_name=test_collection_name)
            qdrant_client.create_collection(
                collection_name=test_collection_name,
                vectors_config=VectorParams(
                    size=vector_dim,
                    distance=Distance.COSINE
                )
            )
        
        # Insert test words into Qdrant
        points = []
        for i, (word, embedding) in enumerate(zip(all_words, embeddings)):
            # Generate unique ID
            unique_id_string = f"test_{word}_{i}"
            hash_obj = hashlib.md5(unique_id_string.encode('utf-8'))
            hash_bytes = hash_obj.digest()[:8]
            point_id = int.from_bytes(hash_bytes, byteorder='big', signed=True)
            if point_id < 0:
                point_id = abs(point_id)
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={"word": word, "test": True}
                )
            )
        
        qdrant_client.upsert(collection_name=test_collection_name, points=points)
        
        # Wait a moment for indexing
        import time
        time.sleep(0.5)
        
        # Test positive pairs (should have high similarity)
        positive_similarities = []
        for word1, word2 in positive_pairs:
            # Get embeddings for both words
            idx1 = all_words.index(word1)
            idx2 = all_words.index(word2)
            emb1 = np.array(embeddings[idx1])
            emb2 = np.array(embeddings[idx2])
            
            # Compute cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            positive_similarities.append(similarity)
        
        # Test negative pairs (should have low similarity)
        negative_similarities = []
        for word1, word2 in negative_pairs:
            # Get embeddings for both words
            idx1 = all_words.index(word1)
            idx2 = all_words.index(word2)
            emb1 = np.array(embeddings[idx1])
            emb2 = np.array(embeddings[idx2])
            
            # Compute cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            negative_similarities.append(similarity)
        
        # Compute confusion matrix
        # Threshold: if similarity > 0.5, consider as "similar"
        threshold = 0.5
        
        TP = sum(1 for sim in positive_similarities if sim > threshold)  # True Positive: similar words correctly identified as similar
        FN = sum(1 for sim in positive_similarities if sim <= threshold)  # False Negative: similar words incorrectly identified as dissimilar
        TN = sum(1 for sim in negative_similarities if sim <= threshold)  # True Negative: dissimilar words correctly identified as dissimilar
        FP = sum(1 for sim in negative_similarities if sim > threshold)  # False Positive: dissimilar words incorrectly identified as similar
        
        # Calculate accuracy
        total = TP + TN + FP + FN
        accuracy = (TP + TN) / total if total > 0 else 0.0
        
        # Test exact word retrieval (query same words that were stored)
        exact_match_results = []
        test_words = ["cat", "dog", "apple", "chair"]  # Sample words to test exact retrieval
        
        for test_word in test_words:
            if test_word in all_words:
                # Get embedding for the test word
                word_idx = all_words.index(test_word)
                query_embedding = embeddings[word_idx]
                
                # Query Qdrant for nearest neighbors
                search_results = qdrant_client.search(
                    collection_name=test_collection_name,
                    query_vector=query_embedding,
                    limit=5,  # Get top 5 results
                    with_payload=True
                )
                
                # Find the exact match in results
                exact_match_found = False
                exact_match_score = 0.0
                for result in search_results:
                    if result.payload.get("word") == test_word:
                        exact_match_found = True
                        exact_match_score = result.score
                        break
                
                exact_match_results.append({
                    "word": test_word,
                    "found": exact_match_found,
                    "similarity_score": exact_match_score,
                    "rank": next((i+1 for i, r in enumerate(search_results) if r.payload.get("word") == test_word), None)
                })
        
        # Calculate average similarity for exact matches
        exact_match_scores = [r["similarity_score"] for r in exact_match_results if r["found"]]
        avg_exact_match_similarity = float(np.mean(exact_match_scores)) if exact_match_scores else 0.0
        
        # Clean up test collection
        try:
            qdrant_client.delete_collection(collection_name=test_collection_name)
        except Exception:
            pass  # Ignore cleanup errors
        
        # Prepare results
        results = {
            "confusion_matrix": {
                "TP": TP,
                "TN": TN,
                "FP": FP,
                "FN": FN
            },
            "avg_positive_similarity": float(np.mean(positive_similarities)) if positive_similarities else 0.0,
            "avg_negative_similarity": float(np.mean(negative_similarities)) if negative_similarities else 0.0,
            "positive_pairs": list(zip(positive_pairs, positive_similarities)),
            "negative_pairs": list(zip(negative_pairs, negative_similarities)),
            "accuracy": accuracy,
            "threshold": threshold,
            "exact_match_results": exact_match_results,
            "avg_exact_match_similarity": avg_exact_match_similarity
        }
        
        return results
        
    except Exception as e:
        # Clean up test collection on error
        try:
            qdrant_client.delete_collection(collection_name=test_collection_name)
        except Exception:
            pass
        raise RuntimeError(f"Error during embedding test: {e}")

# Page configuration
st.set_page_config(
    page_title="RAG Query System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Lazy initialization for Research Overview (only when needed)
@st.cache_resource
def _get_research_overview_workflow():
    """Lazy initialize research overview workflow (cached)."""
    try:
        research_modules = _get_research_modules()
        if not research_modules:
            return None
        
        # Initialize retrieval engine only for Research Overview workflow
        RAGRetrievalEngine = research_modules['retrieval'].RAGRetrievalEngine
        ResearchSynthesizer = research_modules['synthesis'].ResearchSynthesizer
        ReportFormatter = research_modules['report_formatter'].ReportFormatter
        ResearchOverviewWorkflow = research_modules['research_overview'].ResearchOverviewWorkflow
        
        retrieval_engine = RAGRetrievalEngine(
            use_ollama=True,
            vector_db_type="Qdrant",
            embedding_model="llama3.1"
        )
        synthesizer = ResearchSynthesizer(use_ollama=True)
        report_formatter = ReportFormatter()
        workflow = ResearchOverviewWorkflow(
            synthesizer=synthesizer,
            report_formatter=report_formatter,
            retrieval_engine=retrieval_engine
        )
        logger.info("Research overview workflow initialized")
        return workflow
    except Exception as e:
        logger.warning(f"Could not initialize research overview workflow: {e}")
        return None

# Main UI
st.title("⚖️ Legal Documents RAG System")
st.caption("Upload and query legal documents, perform web search, or generate research overview papers")

# Info about stopping the app
with st.expander("ℹ️ How to Stop the App"):
    st.markdown("""
    **To stop this Streamlit app:**
    
    1. **Press Ctrl-C** in the terminal/command prompt where you started the app
    2. The app will shut down gracefully
    3. Wait for any ongoing operations to complete
    
    **Note:** If the app is running in a background process, you may need to:
    - Find the process ID (PID) and terminate it
    - Close the terminal window
    - Use Task Manager (Windows) or Activity Monitor (Mac) to end the process
    """)

# Sidebar Navigation
with st.sidebar:
    # Main Mode Selection
    st.header("📋 Navigation")
    
    # Initialize main mode in session state
    if 'main_mode' not in st.session_state:
        st.session_state.main_mode = "⚖️ Legal Documents Query"
    
    main_mode_options = [
        "⚖️ Legal Documents Query",
        "🔍 Simple Web Search",
        "📚 Research Overview",
        "🧩 Chunk Exploration",
        "🧪 Test Embedding Model"
    ]
    
    selected_main_mode = st.radio(
        "Select Mode:",
        options=main_mode_options,
        index=main_mode_options.index(st.session_state.main_mode) if st.session_state.main_mode in main_mode_options else 0,
        key="main_mode_selector"
    )
    
    # Update session state
    st.session_state.main_mode = selected_main_mode
    
    st.markdown("---")
    
    # Sub-options for Legal Documents Query
    if selected_main_mode == "⚖️ Legal Documents Query":
        st.markdown("### ⚖️ Legal Documents")
        
        # Initialize sub-mode in session state
        if 'legal_sub_mode' not in st.session_state:
            st.session_state.legal_sub_mode = "📤 Upload Legal Document"
        
        legal_sub_options = [
            "📤 Upload Legal Document",
            "🔍 Query Legal Documents"
        ]
        
        selected_sub_mode = st.radio(
            "Select Action:",
            options=legal_sub_options,
            index=legal_sub_options.index(st.session_state.legal_sub_mode) if st.session_state.legal_sub_mode in legal_sub_options else 0,
            key="legal_sub_mode_selector"
        )
        
        # Update session state
        st.session_state.legal_sub_mode = selected_sub_mode
    
    st.markdown("---")
    
    # Display connection statuses
    st.markdown("### 🔌 Connection Status")
    
    # Create columns for side-by-side status
    col1, col2 = st.columns(2)
    
    # Qdrant status
    with col1:
        st.markdown("#### 🗄️ Qdrant")
        try:
            qdrant_health_module = importlib.import_module('scripts.08_utilities.qdrant_health')
            display_qdrant_status_in_ui = qdrant_health_module.display_qdrant_status_in_ui
            display_qdrant_status_in_ui(st)
        except Exception as e:
            logger.warning(f"Could not display Qdrant status: {e}")
            st.error("⚠️ Qdrant status check failed")
    
    # Ollama status
    with col2:
        st.markdown("#### 🤖 Ollama")
        try:
            ollama_health_module = importlib.import_module('scripts.08_utilities.ollama_health')
            display_ollama_status_in_ui = ollama_health_module.display_ollama_status_in_ui
            display_ollama_status_in_ui(st)
        except Exception as e:
            logger.warning(f"Could not display Ollama status: {e}")
            st.error("⚠️ Ollama status check failed")
    st.markdown("---")
    
    st.header("📚 Document Management")
    
    # Embedding Model Selection
    # EMBEDDING_MODELS, get_available_chunking_methods, get_default_chunking_methods already imported above
    
    # Initialize legal embedding model in session state (default to ollama/llama3.1:latest)
    if 'legal_embedding_model' not in st.session_state:
        # Set default to ollama/llama3.1:latest if available, otherwise first in list
        default_model = "ollama/llama3.1:latest"
        if default_model in EMBEDDING_MODELS:
            st.session_state.legal_embedding_model = default_model
        else:
            st.session_state.legal_embedding_model = EMBEDDING_MODELS[0] if EMBEDDING_MODELS else "ollama/llama3.1:latest"
    
    legal_embedding_model = st.selectbox(
        "Embedding Model",
        options=EMBEDDING_MODELS,
        index=EMBEDDING_MODELS.index(st.session_state.legal_embedding_model) if st.session_state.legal_embedding_model in EMBEDDING_MODELS else (EMBEDDING_MODELS.index("ollama/llama3.1:latest") if "ollama/llama3.1:latest" in EMBEDDING_MODELS else 0),
        key="sidebar_legal_embedding_model",
        help="Select embedding model for legal document processing. This will be stored in Qdrant metadata."
    )
    
    # Update session state
    st.session_state.legal_embedding_model = legal_embedding_model
    
    st.markdown("---")
    
# Main Content Area - Dynamic based on selected mode
if st.session_state.main_mode == "⚖️ Legal Documents Query":
    # Legal Documents Query Mode
    if not LEGAL_CHUNKER_AVAILABLE:
        st.error("⚠️ Legal chunker not available. Please install dependencies: pip install qdrant-client ollama langchain langchain-experimental scikit-learn pypdf")
        st.stop()
    
    # Initialize Qdrant client for legal documents (cached resource) - shared across all sub-modes
    @st.cache_resource
    def get_qdrant_client():
        """Get or create Qdrant client (cached resource)."""
        try:
            from qdrant_client import QdrantClient
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            client = QdrantClient(url=qdrant_url)
            logger.info("Legal Qdrant client initialized")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            return None
    
    # Initialize Qdrant client in session state
    if 'legal_qdrant_client' not in st.session_state:
        st.session_state.legal_qdrant_client = get_qdrant_client()
        st.session_state.legal_collection_name = "legal_documents"
    
    # Cache expensive data fetching operations - shared across all sub-modes
    @st.cache_data(ttl=60)  # Cache for 60 seconds
    def get_cached_distinct_files(_qdrant_client, collection_name):
        """Get distinct source files from Qdrant (cached)."""
        if _qdrant_client is None:
            return []
        try:
            return get_distinct_source_files(_qdrant_client, collection_name)
        except Exception as e:
            logger.error(f"Error fetching distinct files: {e}")
            return []
    
    @st.cache_data(ttl=60)  # Cache for 60 seconds
    def get_cached_chunking_methods(_qdrant_client, collection_name, source_file=None):
        """Get distinct chunking methods from Qdrant (cached)."""
        if _qdrant_client is None:
            return []
        try:
            if source_file:
                return get_distinct_chunking_methods_for_file(_qdrant_client, collection_name, source_file)
            else:
                return get_distinct_chunking_methods(_qdrant_client, collection_name)
        except Exception as e:
            logger.error(f"Error fetching chunking methods: {e}")
            return []
    
    @st.cache_data(ttl=60)  # Cache for 60 seconds
    def get_cached_embedding_models(_qdrant_client, collection_name, source_file=None):
        """Get distinct embedding models from Qdrant (cached)."""
        if _qdrant_client is None:
            return []
        try:
            return get_distinct_embedding_models(_qdrant_client, collection_name, source_file)
        except Exception as e:
            logger.error(f"Error fetching embedding models: {e}")
            return []
    
    # Initialize Qdrant client if not exists
    if 'legal_qdrant_client' not in st.session_state:
        st.session_state.legal_qdrant_client = get_qdrant_client()
        st.session_state.legal_collection_name = "legal_documents"
            
    # Initialize legal_query_params if not exists
    if 'legal_query_params' not in st.session_state:
        st.session_state.legal_query_params = {}
            
    # Initialize legal_query_form if not exists
    if 'legal_query_form' not in st.session_state:
        st.session_state.legal_query_form = {
            'query': '',
            'selected_file': 'All Files',
            'chunking_method': 'All Methods',
            'embedding_model': '',
            'llm_model': 'llama3.1:latest',
            'reranker_model': 'BAAI/bge-reranker-large',
            'enable_reranking': False,
            'top_k': 5,
            'rerank_top_n': 20,
            'rerank_top_k': 5
        }
        
    # Initialize legal_query_params if not exists
    if 'legal_query_params' not in st.session_state:
        st.session_state.legal_query_params = {
            'query_text': '',
            'source_file': 'All Files',
            'chunking_method': 'All Methods',
            'embedding_model': '',
            'search_type': 'semantic',
            'llm_model': 'llama3.1:latest',
            'reranker_model': 'BAAI/bge-reranker-large',
            'enable_reranking': False,
            'top_k': 5,
            'rerank_top_n': 20,
            'rerank_top_k': 5
        }

    # Fetch available files
    with st.spinner("Loading documents..."):
        distinct_files = get_cached_distinct_files(
            st.session_state.legal_qdrant_client,
            st.session_state.legal_collection_name
        )
    
    # Handle Upload Legal Document mode
    if st.session_state.legal_sub_mode == "📤 Upload Legal Document":
        st.subheader("📤 Upload Legal Document")
        st.markdown("Upload and process legal documents to add them to the Qdrant vector database with multiple embedding models and chunking methods.")
        
        # Initialize session state for upload form
        if 'upload_form' not in st.session_state:
            st.session_state.upload_form = {
                'embedding_models': [],
                'chunking_methods': [],
                'file_metadata': {}
            }
        
        # Two columns layout
        col1, col2 = st.columns(2)
        
        with col1:
            # File uploader for legal documents
            uploaded_file = st.file_uploader(
                "Upload Legal Document (PDF, DOCX, DOC, MD, or TXT)",
                type=["pdf", "docx", "doc", "md", "txt"],
                accept_multiple_files=False,
                key="file_uploader"
            )
            
            # Get available embedding models and chunking methods
            available_embedding_models = EMBEDDING_MODELS
            available_chunking_methods = get_available_chunking_methods()
            
            # Multi-select for embedding models
            selected_embedding_models = st.multiselect(
                "Select Embedding Models",
                options=available_embedding_models,
                default=available_embedding_models[:1] if available_embedding_models else [],
                help="Select one or more embedding models to use for this document"
            )
            
            # Multi-select for chunking methods
            selected_chunking_methods = st.multiselect(
                "Select Chunking Methods",
                options=available_chunking_methods,
                default=get_default_chunking_methods(),
                help="Select one or more chunking methods to apply to this document"
            )
            
            # File metadata
            st.subheader("📝 File Metadata")
            file_metadata = {}
            file_metadata['source'] = st.text_input("Document Source", "")
            file_metadata['category'] = st.text_input("Document Category", "")
            file_metadata['language'] = st.selectbox("Document Language", ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"], index=0)
            
            # Update session state
            st.session_state.upload_form = {
                'embedding_models': selected_embedding_models,
                'chunking_methods': selected_chunking_methods,
                'file_metadata': file_metadata
            }
        
        with col2:
            # Display upload summary
            st.subheader("📋 Upload Summary")
            
            if uploaded_file:
                st.markdown(f"**File:** {uploaded_file.name}")
                st.markdown(f"**Size:** {len(uploaded_file.getvalue()) / 1024:.2f} KB")
                
                st.markdown("**Selected Embedding Models:**")
                for model in selected_embedding_models:
                    st.markdown(f"- {model}")
                
                st.markdown("**Selected Chunking Methods:**")
                for method in selected_chunking_methods:
                    st.markdown(f"- {method}")
                
                st.markdown("**Metadata:**")
                for key, value in file_metadata.items():
                    if value:  # Only show non-empty metadata
                        st.markdown(f"- **{key.title()}:** {value}")
            else:
                st.info("Upload a file and configure processing options")
        
        # Process button
        if uploaded_file and selected_embedding_models and selected_chunking_methods:
            if st.button("🚀 Process Document", type="primary", use_container_width=True):
                try:
                    # Save the uploaded file temporarily
                    file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process the file with all selected models and methods
                    with st.spinner("Processing document (this may take a few minutes)..."):
                        # Create a progress bar
                        progress_bar = st.progress(0)
                        total_steps = len(selected_embedding_models) * len(selected_chunking_methods)
                        current_step = 0
                        
                        file_id = str(uuid.uuid4())
                        
                        for model in selected_embedding_models:
                            for method in selected_chunking_methods:
                                # Update progress
                                current_step += 1
                                progress = current_step / total_steps
                                progress_bar.progress(progress, text=f"Processing with {model} and {method}...")
                                
                                # Process the document with the current model and method
                                result = process_legal_document(
                                    file_path=file_path,
                                    qdrant_client=st.session_state.legal_qdrant_client,
                                    collection_name=st.session_state.legal_collection_name,
                                    embedding_model=model,
                                    chunking_method=method,
                                    file_metadata={
                                        'file_id': file_id,
                                        'file_name': uploaded_file.name,
                                        'file_size': len(uploaded_file.getvalue()),
                                        'upload_date': datetime.now().isoformat(),
                                        'embedding_model': model,
                                        'chunking_method': method,
                                        **file_metadata
                                    }
                                )
                                
                                if not result.get('success', False):
                                    st.error(f"Error processing with {model} and {method}: {result.get('error', 'Unknown error')}")
                                    logger.error(f"Error processing document: {result.get('error')}")
                                    continue
                        
                        # Clean up temporary file
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            logger.warning(f"Could not remove temporary file {file_path}: {e}")
                        
                        progress_bar.progress(1.0, "Processing complete!")
                        st.success(f"✅ Successfully processed and added {uploaded_file.name} to the database!")
                        
                        # Refresh the file list
                        distinct_files = get_cached_distinct_files(
                            st.session_state.legal_qdrant_client,
                            st.session_state.legal_collection_name
                        )
                        
                        # Show success message with details
                        st.balloons()
                        
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    logger.error(f"Error processing file: {str(e)}", exc_info=True)
        
        # File management section
        st.markdown("---")
        st.subheader("📂 Uploaded Files")
        
        # Get all files from the database
        try:
            stats = get_file_statistics(
                st.session_state.legal_qdrant_client,
                st.session_state.legal_collection_name
            )
            
            if not stats or 'files' not in stats or not stats['files']:
                st.info("No files have been uploaded yet.")
            else:
                # Create a summary DataFrame
                file_data = []
                for file_info in stats['files']:
                    source_file = file_info.get('source_file', 'Unknown')
                    chunking_methods = file_info.get('chunking_methods', [])
                    
                    # If no specific chunking methods, just add one row
                    if not chunking_methods:
                        file_data.append({
                            'File ID': source_file,  # Using source_file as ID if no file_id available
                            'File Name': source_file,
                            'Upload Date': file_info.get('upload_date', 'Unknown'),
                            'Chunking Method': 'N/A',
                            'Embedding Model': file_info.get('embedding_model', 'Unknown'),
                            'Chunk Count': file_info.get('chunk_count', 0),
                            'Total Size (KB)': round(file_info.get('total_chunk_size', 0) / 1024, 2) if file_info.get('chunk_count', 0) > 0 else 0,
                            'Avg Chunk Size': file_info.get('avg_chunk_size', 0),
                            'Min Chunk Size': file_info.get('min_chunk_size', 0),
                            'Max Chunk Size': file_info.get('max_chunk_size', 0)
                        })
                    else:
                        # Add a row for each chunking method
                        for method in chunking_methods:
                            method_stats = file_info.get('chunking_method_stats', {}).get(method, {})
                            method_count = method_stats.get('chunk_count', 0)
                            method_avg = method_stats.get('avg_size', 0)
                            method_min = method_stats.get('min_size', 0)
                            method_max = method_stats.get('max_size', 0)
                            method_total_size = method_stats.get('total_size', 0)
                            
                            file_data.append({
                                'File ID': source_file,
                                'File Name': source_file,
                                'Upload Date': file_info.get('upload_date', 'Unknown'),
                                'Chunking Method': method,
                                'Embedding Model': file_info.get('embedding_model', 'Unknown'),
                                'Chunk Count': method_count,
                                'Total Size (KB)': round(method_total_size / 1024, 2) if method_count > 0 else 0,
                                'Avg Chunk Size': round(method_avg, 2),
                                'Min Chunk Size': method_min,
                                'Max Chunk Size': method_max
                            })
                
                if file_data:
                    # Display the files in a table
                    df = pd.DataFrame(file_data)
                    st.dataframe(
                        df,
                        column_config={
                            'Total Size (KB)': st.column_config.NumberColumn(
                                "Size (KB)",
                                format="%.2f"
                            ),
                            'Upload Date': st.column_config.DatetimeColumn(
                                "Uploaded",
                                format="YYYY-MM-DD HH:mm"
                            ),
                            'Avg Chunk Size': st.column_config.NumberColumn(
                                "Avg Chars/Chunk",
                                format="%.0f"
                            ),
                            'Min Chunk Size': st.column_config.NumberColumn(
                                "Min Chars/Chunk",
                                format="%.0f"
                            ),
                            'Max Chunk Size': st.column_config.NumberColumn(
                                "Max Chars/Chunk",
                                format="%.0f"
                            )
                        },
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Add download button for the table
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Export to CSV",
                        data=csv,
                        file_name="legal_documents_export.csv",
                        mime="text/csv"
                    )
                    
                    # File deletion options
                    st.markdown("### 🗑️ File Management")
                    
                    # Get unique source files from the stats
                    unique_files = {}
                    for file_info in stats['files']:
                        source_file = file_info.get('source_file', 'Unknown')
                        if source_file not in unique_files:
                            unique_files[source_file] = {
                                'chunk_count': file_info.get('chunk_count', 0),
                                'upload_date': file_info.get('upload_date', 'Unknown')
                            }
                    
                    if unique_files:
                        # Create a sorted list of files (newest first)
                        sorted_files = sorted(
                            unique_files.items(),
                            key=lambda x: x[1].get('upload_date', ''),
                            reverse=True
                        )
                        
                        # Create display names with metadata
                        file_options = [
                            f"{file_name} (chunks: {info['chunk_count']}, uploaded: {info['upload_date']})" 
                            for file_name, info in sorted_files
                        ]
                        
                        selected_display = st.selectbox(
                            "Select a file to delete:",
                            options=[""] + file_options,
                            index=0,
                            help="Select a file to delete from the Qdrant collection"
                        )
                        
                        # Delete specific file with a clear warning
                        if st.button("🗑️ Delete Selected File", type="secondary",
                                   help="Delete the selected file from Qdrant"):
                            # Extract the original file name from the display string
                            file_to_delete = selected_display.split(' (chunks: ')[0]
                            
                            with st.spinner(f"Deleting {file_to_delete}..."):
                                try:
                                    result = delete_file_from_qdrant(
                                        st.session_state.legal_qdrant_client,
                                        st.session_state.legal_collection_name,
                                        file_to_delete
                                    )
                                    
                                    if result.get('success'):
                                        st.success(result.get('message', f"Successfully deleted {file_to_delete}"))
                                        # Clear the file cache and refresh the page
                                        if 'file_statistics' in st.session_state:
                                            del st.session_state.file_statistics
                                        st.rerun()
                                    else:
                                        st.error(result.get('message', f"Failed to delete {file_to_delete}"))
                                        logger.error(f"Failed to delete file: {result.get('message')}")
                                        
                                except Exception as e:
                                    error_msg = f"Error deleting file: {str(e)}"
                                    st.error(error_msg)
                                    logger.error(error_msg, exc_info=True)
                    
                    # Delete all files with a single button click
                    if st.button("🗑️ Delete All Files", type="primary", 
                               help="WARNING: This will immediately delete ALL files from Qdrant. This action cannot be undone!",
                               key="delete_all_btn"):
                        with st.spinner("Deleting all files from Qdrant..."):
                            try:
                                result = delete_all_files_from_qdrant(
                                    st.session_state.legal_qdrant_client,
                                    st.session_state.legal_collection_name
                                )
                                
                                if result.get('success'):
                                    st.success("✅ Successfully deleted all files from the database")
                                    # Clear the file cache and refresh the page
                                    if 'file_statistics' in st.session_state:
                                        del st.session_state.file_statistics
                                    time.sleep(1)  # Let the user see the success message
                                    st.rerun()
                                else:
                                    st.error(f"❌ {result.get('message', 'Failed to delete all files')}")
                                    logger.error(f"Failed to delete all files: {result.get('message')}")
                                    
                            except Exception as e:
                                error_msg = f"❌ Error deleting all files: {str(e)}"
                                st.error(error_msg)
                                logger.error(error_msg, exc_info=True)
                
                else:
                    st.info("No file statistics available.")
                    
        except Exception as e:
            st.error(f"Error retrieving file information: {str(e)}")
            logger.error(f"Error retrieving file information: {str(e)}", exc_info=True)
    
    # Handle Query Legal Documents mode
    elif st.session_state.legal_sub_mode == "🔍 Query Legal Documents":
        st.subheader("🔍 Query Legal Documents")
        st.markdown("Search through your legal documents using semantic search with configurable models and optional reranking.")
        
        if not distinct_files:
            st.warning("No documents found. Upload files first.")
            st.stop()
            
        # Initialize session state for form (only for query mode)
        if 'legal_query_form' not in st.session_state:
            st.session_state.legal_query_form = {
                'query': '',
                'selected_file': 'All Files',
                'chunking_method': 'All Methods',
                'embedding_model': '',
                'llm_model': 'llama3.1:latest',
                'reranker_model': 'BAAI/bge-reranker-large',
                'enable_reranking': False,
                'top_k': 5,
                'rerank_top_n': 20,
                'rerank_top_k': 5,
                'search_type': 'semantic'
            }
            
        # Create the search form
        with st.form(key='query_form'):
            # Create form columns
            col1, col2 = st.columns(2)
            
            # Initialize form values from session state or defaults
            form_values = {
                'query': st.session_state.legal_query_form.get('query', ''),
                'selected_file': st.session_state.legal_query_form.get('selected_file', 'All Files'),
                'chunking_method': st.session_state.legal_query_form.get('chunking_method', 'All Methods'),
                'embedding_model': st.session_state.legal_query_form.get('embedding_model', ''),
                'llm_model': st.session_state.legal_query_form.get('llm_model', 'llama3.1:latest'),
                'reranker_model': st.session_state.legal_query_form.get('reranker_model', 'BAAI/bge-reranker-large'),
                'enable_reranking': st.session_state.legal_query_form.get('enable_reranking', False),
                'top_k': st.session_state.legal_query_form.get('top_k', 5),
                'rerank_top_n': st.session_state.legal_query_form.get('rerank_top_n', 20),
                'rerank_top_k': st.session_state.legal_query_form.get('rerank_top_k', 5),
                'search_type': st.session_state.legal_query_form.get('search_type', 'semantic')
            }
            
            with col1:
                # Initialize legal_query_params if it doesn't exist
                if 'legal_query_params' not in st.session_state:
                    st.session_state.legal_query_params = {}
                
                # File selection with "All Files" option
                selected_file = st.selectbox(
                    "Document (optional)",
                    options=["All Files"] + distinct_files,
                    index=0  # Always default to first option
                )
                
                # Get available chunking methods for the selected file
                with st.spinner("Loading chunking methods..."):
                    if selected_file == "All Files":
                        chunking_methods = get_cached_chunking_methods(
                            st.session_state.legal_qdrant_client,
                            st.session_state.legal_collection_name
                        )
                    else:
                        chunking_methods = get_cached_chunking_methods(
                            st.session_state.legal_qdrant_client,
                            st.session_state.legal_collection_name,
                            selected_file
                        )
                
                chunking_method = st.selectbox(
                    "Chunking Method (optional)",
                    options=["All Methods"] + (chunking_methods if chunking_methods else []),
                    index=0
                )
                
                # Search type
                search_mode = st.selectbox(
                    "Search Type",
                    ["Semantic", "BM25", "Hybrid"],
                    index=["semantic", "bm25", "hybrid"].index(form_values['search_type'].lower()) 
                           if form_values['search_type'].lower() in ["semantic", "bm25", "hybrid"] 
                           else 0
                )
                
                # Reranking options
                with st.expander("⚙️ Reranking Options", expanded=form_values['enable_reranking']):
                    enable_reranking = st.checkbox(
                        "Enable Reranking",
                        value=form_values['enable_reranking'],
                        key='enable_reranking_checkbox'
                    )
                    
                    if enable_reranking:
                        reranker_models = {
                            "BAAI/bge-reranker-large": "Best overall accuracy (recommended)",
                            "BAAI/bge-reranker-base": "Good balance of speed and accuracy",
                            "cross-encoder/ms-marco-MiniLM-L-6-v2": "Fastest option, CPU-friendly"
                        }
                        
                        reranker_model = st.selectbox(
                            "Reranker Model",
                            options=list(reranker_models.keys()),
                            index=list(reranker_models.keys()).index(form_values['reranker_model']) 
                                   if form_values['reranker_model'] in reranker_models 
                                   else 0,
                            format_func=lambda x: f"{x} ({reranker_models[x].split(')')[0]})",
                            key='reranker_model_select'
                        )
                        
                        rerank_top_n = st.slider(
                            "Initial results to consider for reranking",
                            min_value=10,
                            max_value=100,
                            value=form_values['rerank_top_n'],
                            step=5,
                            key='rerank_top_n_slider'
                        )
                        
                        rerank_top_k = st.slider(
                            "Final results after reranking",
                            min_value=1,
                            max_value=20,
                            value=form_values['rerank_top_k'],
                            step=1,
                            key='rerank_top_k_slider'
                        )
                
            with col2:
                # Get available embedding models
                with st.spinner("Loading embedding models..."):
                    if selected_file == "All Files":
                        embedding_models = get_cached_embedding_models(
                            st.session_state.legal_qdrant_client,
                            st.session_state.legal_collection_name
                        )
                    else:
                        embedding_models = get_cached_embedding_models(
                            st.session_state.legal_qdrant_client,
                            st.session_state.legal_collection_name,
                            selected_file
                        )
                
                # Only use the two specified models
                embedding_models = ["llama3.1:latest", "nomic-embed-text:latest"]
                default_index = 0  # Default to llama3.1:latest
                
                embedding_model = st.selectbox(
                    "Embedding Model",
                    options=embedding_models,
                    index=default_index,
                    key='embedding_model_select'
                )
                
                # Log the selected model for debugging
                logger.info(f"Selected embedding model: {embedding_model}")
                
                # Number of results slider
                top_k = st.slider(
                    "Number of results",
                    min_value=1,
                    max_value=20,
                    value=5,  # Default value
                    step=1
                )
            
            # Text input for search query
            query_text = st.text_area(
                "Enter your search query",
                value=st.session_state.legal_query_form.get('query', ''),
                placeholder="Search legal documents...",
                key='search_query_input'
            )
            
            # Form submission buttons - must be at the end of the form
            button_col1, button_col2 = st.columns(2)
            with button_col1:
                search_clicked = st.form_submit_button("🔍 Search Legal Documents", type="primary", use_container_width=True)
            with button_col2:
                clear_clicked = st.form_submit_button("🔄 Clear Form", type="secondary", use_container_width=True)
            
            # Handle form submission
            if search_clicked or clear_clicked:
                if clear_clicked:
                    st.session_state.legal_query_form = {
                        'query': '', 
                        'selected_file': 'All Files', 
                        'chunking_method': 'All Methods',
                        'embedding_model': embedding_models[0] if 'embedding_models' in locals() and embedding_models else '',
                        'llm_model': 'llama3.1:latest',
                        'reranker_model': 'BAAI/bge-reranker-large', 
                        'enable_reranking': False, 
                        'top_k': 5,
                        'rerank_top_n': 20,
                        'rerank_top_k': 5,
                        'search_type': 'semantic'
                    }
                    if 'legal_query_results' in st.session_state:
                        del st.session_state.legal_query_results
                    st.rerun()
                else:
                    # Clear any previous results
                    if 'legal_query_results' in st.session_state:
                        del st.session_state.legal_query_results
                        
                    # Update form values in session state
                    st.session_state.legal_query_form = {
                        'query': query_text,
                        'selected_file': selected_file,
                        'chunking_method': chunking_method if 'chunking_method' in locals() else 'All Methods',
                        'embedding_model': embedding_model if 'embedding_model' in locals() else '',
                        'llm_model': 'llama3.1:latest',
                        'reranker_model': reranker_model if 'reranker_model' in locals() else 'BAAI/bge-reranker-large',
                        'enable_reranking': enable_reranking if 'enable_reranking' in locals() else False,
                        'top_k': top_k if 'top_k' in locals() else 5,
                        'rerank_top_n': rerank_top_n if 'rerank_top_n' in locals() else 20,
                        'rerank_top_k': rerank_top_k if 'rerank_top_k' in locals() else 5,
                        'search_type': search_mode.lower() if 'search_mode' in locals() else 'semantic'
                    }
                    
                    # Set flag to process the search in the next run
                    st.session_state.process_search = True
                    st.session_state.should_run_query = True
                    
                    # Force a rerun to process the search
                    st.rerun()
            
        # Initialize selected_llm_model with a default value (outside the form)
        selected_llm_model = 'llama3.1:latest'
        
        # Handle search form submission and state management
        search_triggered = 'process_search' in st.session_state and st.session_state.process_search
        query = st.session_state.legal_query_form.get('query', '')
        
        if search_triggered and query and isinstance(query, str) and query.strip():
            # Reset the flag first to prevent multiple submissions
            st.session_state.process_search = False
            
            # Clear any previous results to ensure fresh search
            if 'legal_query_results' in st.session_state:
                del st.session_state.legal_query_results
            
            # Initialize with default values
            source_file = None if selected_file == "All Files" else selected_file
            chunking = None if chunking_method == "All Methods" else chunking_method
            
            # Log the search initiation
            query_text = st.session_state.legal_query_form['query']
            logger.info("\n" + "="*80)
            logger.info("🔍 INITIATING LEGAL DOCUMENT SEARCH")
            logger.info("="*80)
            logger.info(f"📝 Search Query: '{query_text}'")
            logger.info(f"📂 Selected File: {selected_file}")
            logger.info(f"✂️  Chunking Method: {chunking_method}")
            logger.info(f"🔤 Search Type: {search_mode}")
            logger.info(f"🤖 Embedding Model: {embedding_model}")
            
            # Store search parameters in session state
            st.session_state.search_params = {
                'query_text': query_text,
                'source_file': source_file,
                'chunking': chunking,
                'search_mode': search_mode,
                'embedding_model': embedding_model,
                'enable_reranking': st.session_state.legal_query_form.get('enable_reranking', False),
                'reranker_model': st.session_state.legal_query_form.get('reranker_model'),
                'rerank_top_n': int(st.session_state.legal_query_form.get('rerank_top_n', 20)),
                'rerank_top_k': int(st.session_state.legal_query_form.get('rerank_top_k', 5)),
                'top_k': int(st.session_state.legal_query_form.get('top_k', 5))
            }
        
        # Only show search results if we have valid search parameters
        if 'search_params' in st.session_state and st.session_state.search_params.get('query_text'):
            # Get search parameters from session state
            params = st.session_state.search_params
            
            # Only perform search if we don't have results yet
            if 'legal_query_results' not in st.session_state:
                with st.spinner("Searching documents..."):
                    # Use parameters from session state
                    params = st.session_state.search_params
                    query_text = params['query_text']
                    source_file = params['source_file']
                    chunking = params['chunking']
                    search_mode_lower = params['search_mode'].lower()
                    enable_reranking = params['enable_reranking']
                    rerank_top_n = params['rerank_top_n']
                    rerank_top_k = params['rerank_top_k']
                    top_k = params['top_k']
                    embedding_model = params['embedding_model']
                    
                    # Log the search parameters for debugging
                    logger.info("\n🔧 SEARCH PARAMETERS")
                    logger.info("-" * 30)
                    logger.info(f"Query: '{query_text}'")
                    logger.info(f"Chunking: {chunking}")
                    logger.info(f"Search Mode: {search_mode_lower}")
                    logger.info(f"Source File: {source_file}")
                    logger.info(f"Top K: {top_k}")
                    logger.info(f"Reranking: {'Enabled' if enable_reranking else 'Disabled'}")
                    if enable_reranking:
                        logger.info(f"Rerank Top N: {rerank_top_n}")
                        logger.info(f"Rerank Top K: {rerank_top_k}")
                    logger.info("-" * 30 + "\n")
                
                # Use the advanced reranking function with validated parameters
                try:
                    # Log search start
                    search_start_time = datetime.now()
                    logger.info("🚀 Starting document search...")
                    
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(progress, status):
                        progress_bar.progress(progress)
                        status_text.text(status)
                    
                    # Execute the search with progress updates
                    results = []
                    
                    # Only get total chunks if we don't already have them or if the query has changed
                    cache_key = f"total_chunks_{query_text}_{source_file}_{chunking}"
                    if cache_key not in st.session_state:
                        try:
                            from qdrant_client.models import Filter, FieldCondition, MatchValue
                            
                            # Build filter conditions
                            filter_conditions = []
                            if chunking and chunking != 'All Methods':
                                filter_conditions.append(
                                    FieldCondition(
                                        key="chunking_method",
                                        match=MatchValue(value=chunking)
                                    )
                                )
                            if source_file and source_file != 'All Files':
                                filter_conditions.append(
                                    FieldCondition(
                                        key="source_file",
                                        match=MatchValue(value=source_file)
                                    )
                                )
                            
                            filter_condition = Filter(must=filter_conditions) if filter_conditions else None
                            
                            # Get total count of matching documents
                            st.session_state.total_chunks = st.session_state.legal_qdrant_client.count(
                                collection_name=st.session_state.legal_collection_name,
                                count_filter=filter_condition
                            ).count
                            
                            # Store the total chunks count in the session state with a unique key
                            st.session_state[cache_key] = total_chunks
                            logger.info(f"Cached total chunks for query: {query_text} - {total_chunks} chunks")
                            
                            update_progress(0, f"Found {total_chunks} chunks to search through...")
                            
                        except Exception as e:
                            logger.warning(f"Could not get total chunk count: {str(e)}")
                            st.session_state[cache_key] = 0
                    else:
                        total_chunks = st.session_state[cache_key]
                        update_progress(0, f"Searching through {total_chunks} chunks...")
                    
                    # Execute the search
                    update_progress(10, "Generating query embeddings...")
                    
                    results = query_legal_documents_with_reranking(
                        qdrant_client=st.session_state.legal_qdrant_client,
                        collection_name=st.session_state.legal_collection_name,
                        query_text=query_text.strip(),
                        embedding_model=embedding_model,
                        chunking_method=chunking or 'semantic',
                        source_file=source_file,
                        top_k=min(int(rerank_top_n if enable_reranking else top_k), 100),
                        use_reranking=enable_reranking,
                        retrieval_methods=[search_mode_lower],
                        progress_callback=update_progress if st.session_state.get(cache_key, 0) > 0 else None,
                        total_chunks=st.session_state.get(cache_key, 0)
                    )
                    
                    # Clear progress bar when done
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Log search completion
                    search_duration = (datetime.now() - search_start_time).total_seconds()
                    logger.info(f"✅ Search completed in {search_duration:.2f} seconds")
                    
                    # Process and log results
                    if results:
                        logger.info(f"📊 Found {len(results)} results")
                        if enable_reranking and len(results) > rerank_top_k:
                            results = results[:rerank_top_k]
                            logger.info(f"🔝 Keeping top {len(results)} results after reranking")
                        
                        # Log top results metadata
                        logger.info("\n🔍 TOP SEARCH RESULTS")
                        logger.info("-" * 30)
                        for i, result in enumerate(results[:3], 1):  # Log details of top 3 results
                            score = result.get('score', 0)
                            source = result.get('source', 'unknown')
                            chunk_id = result.get('chunk_id', 'N/A')
                            logger.info(f"{i}. Score: {score:.4f} | Source: {source} | Chunk: {chunk_id}")
                        
                        # Store the results in session state and clear the progress
                        st.session_state.legal_query_results = results
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Clear the search params to prevent re-running the search
                        if 'search_params' in st.session_state:
                            del st.session_state.search_params
                            st.session_state.legal_query_used_params = st.session_state.legal_query_form.copy()
                            
                            # Log successful completion
                            logger.info("\n✨ Search completed successfully")
                            logger.info("=" * 80 + "\n")
                        
                        # Rerun to update the UI
                        st.rerun()
                    else:
                        logger.warning("⚠️ No results found for the query")
                        st.session_state.legal_query_results = []
                        st.session_state.legal_query_used_params = st.session_state.legal_query_form.copy()
                        st.rerun()
                    
                except Exception as e:
                    error_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    error_msg = f"❌ SEARCH FAILED at {error_time}"
                    logger.error("\n" + "!"*80)
                    logger.error(error_msg)
                    logger.error("!"*80)
                    logger.error(f"Error Type: {type(e).__name__}")
                    logger.error(f"Error Message: {str(e)}")
                    logger.error("\nStack Trace:", exc_info=True)
                    logger.error("-"*80 + "\n")
                    
                    st.error(f"An error occurred during search: {str(e)}")
                    st.session_state.legal_query_results = []
                    st.stop()

    # === Display Results ===
    if 'legal_query_results' in st.session_state:
        # Check if we have results or not
        if not st.session_state.legal_query_results:
            st.warning("⚠️ No results found for your query. Please try different search terms or filters.")
            # Show search tips to help user refine their search
            with st.expander("💡 Search Tips", expanded=True):
                st.markdown("""
                - Try using more specific or different keywords
                - Check your spelling
                - Try using fewer filters
                - Make sure the selected source file contains relevant content
                - Try a different search mode (e.g., switch between semantic and BM25)
                """)
            # Don't proceed with the rest of the results display
            st.stop()
            
        # If we have results, display them
        params = st.session_state.legal_query_used_params
        
        # Display search parameters
        st.markdown("## 📋 Search Results")
        st.caption(
            f"**Query:** \"{params['query']}\" | **Embedding:** `{params['embedding_model']}` | "
            f"**Chunking:** {params['chunking_method']} | **LLM:** `{params['llm_model']}`" +
            (f" | **Reranker:** `{params['reranker_model']}`" if params.get('enable_reranking') else "")
        )

        # Generate LLM answer
        try:
            from langchain_ollama import ChatOllama
            import time
            
            # Prepare context from top chunks
            context_chunks = []
            for i, res in enumerate(st.session_state.legal_query_results[:3]):  # Use top 3 chunks for context
                chunk_text = res.get('text', '').strip()
                if chunk_text:
                    context_chunks.append(f"### Chunk {i+1} (Score: {res.get('score', 0):.4f})\n{chunk_text[:2000]}")
            
            context = "\n\n".join(context_chunks)
            
            # Create prompt with context and query
            prompt = f"""You are a legal assistant. Answer the question based ONLY on the following context. 
If the answer isn't in the context, say "Not found in documents." Be precise and cite sources.

CONTEXT:
{context}

QUESTION: {params['query']}

ANSWER:"""

            # Generate answer with timing
            with st.spinner(f"Generating answer with {params['llm_model']}..."):
                start_time = time.time()
                
                # Initialize LLM
                llm = ChatOllama(
                    model=params['llm_model'].split(':')[0],  # Remove version if present
                    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                    temperature=0.1,
                    num_ctx=4096  # Larger context window
                )
                
                # Generate response
                response = llm.invoke(prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
                gen_time = time.time() - start_time

            # Display answer
            st.markdown("### 💬 Generated Answer")
            st.markdown(f"*Generated in {gen_time:.1f}s using {params['llm_model']}*")
            st.info(answer)
            st.markdown("---")

            # Show sources
            st.markdown("### 📚 Source Chunks")
            for i, res in enumerate(st.session_state.legal_query_results[:5]):  # Show top 5 chunks
                with st.expander(f"📄 Chunk {i+1} | Score: {res.get('score', 0):.4f} | Source: {res.get('source_file', 'Unknown')}"):
                    # Display chunk text with syntax highlighting
                    st.markdown(f"```\n{res.get('text', 'N/A')}\n```")
                    
                    # Display metadata
                    meta = {k: v for k, v in res.items() if k not in ['text', 'score']}
                    with st.expander("View Metadata"):
                        st.json(meta)

        except Exception as e:
            st.error(f"❌ Error generating answer: {str(e)}")
            logger.exception("LLM generation error")

    elif st.session_state.legal_query_form['query']:
        st.warning("No results found. Try adjusting your search parameters.")
    
    # Display results if available and in query mode
    if (st.session_state.legal_sub_mode == "🔍 Query Legal Documents" and 
        'legal_query_results' in st.session_state and 
        st.session_state.legal_query_results):
        st.markdown("## 📋 Search Results")
        
        # Display search mode info
        search_mode_display = {
            'semantic': '🔍 Semantic Search',
            'bm25': '🔤 Keyword Search',
            'mixed': '🤖 Hybrid Search'
        }.get(search_mode, 'Search')
        
        st.caption(f"Showing results using: {search_mode_display} | Model: {embedding_model} | " +
                  f"Chunking: {chunking_method if 'chunking_method' in locals() and chunking_method != 'All Methods' else 'All Methods'}")
        
        # Display each search result
        for i, result in enumerate(st.session_state.legal_query_results, 1):
            # Extract score and handle different score formats
            score = result.get('score', 0)
            if isinstance(score, (list, tuple)) and len(score) > 0:
                score = score[0] if isinstance(score[0], (int, float)) else 0
            
            # Create expander for each result
            with st.expander(f"📄 Result {i} - Score: {score:.4f}", expanded=i==1):
                # Display chunk content with syntax highlighting
                st.markdown("### 📝 Content")
                st.markdown(f"```\n{result.get('text', 'No content available')}\n```")
                
                # Display metadata in a clean format
                st.markdown("### 📋 Metadata")
                
                # Create columns for better layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Source File:**  \n`{result.get('source_file', result.get('file_name', 'N/A'))}`")
                    if 'chunking_method' in result:
                        st.markdown(f"**Chunking Method:**  \n`{result.get('chunking_method', 'N/A')}`")
                
                with col2:
                    if 'page' in result and result['page'] is not None:
                        st.markdown(f"**Page:**  \n{result.get('page', 'N/A')}")
                    if 'chunk_number' in result and result['chunk_number'] is not None:
                        st.markdown(f"**Chunk #:**  \n{result.get('chunk_number')}")
                
                # Display score and other metrics
                if 'score' in result:
                    # Add score interpretation
                    score_interpretation = ""
                    if search_mode == 'semantic':
                        if score >= 0.9:
                            score_interpretation = " (Very High Relevance)"
                        elif score >= 0.7:
                            score_interpretation = " (High Relevance)"
                        elif score >= 0.5:
                            score_interpretation = " (Moderate Relevance)"
                        else:
                            score_interpretation = " (Low Relevance)"
                    
                    st.markdown(f"**Relevance Score:** {score:.4f}{score_interpretation}")
                
                # Display additional metadata if available
                if 'metadata' in result and isinstance(result['metadata'], dict):
                    with st.expander("View All Metadata"):
                        st.json(result['metadata'])
                
                # Add a separator between results
                if i < len(st.session_state.legal_query_results):
                    st.markdown("---")
    
    # Show message if no results but search was performed
    elif 'legal_query_params' in st.session_state and st.session_state.legal_query_params.get('query_text'):
        st.info("No results found for your query. Try adjusting your search parameters or using a different search mode.")
    
    # Show help message if no search has been performed yet
    elif (st.session_state.legal_sub_mode == "🔍 Query Legal Documents" and 
          ('legal_query_params' not in st.session_state or not st.session_state.legal_query_params.get('query_text'))):
        st.info("💡 Enter a query and click 'Search' to find relevant legal documents.")
        
        # Add search tips
        with st.expander("🔍 Search Tips", expanded=True):
            st.markdown("""
            ### Search Modes:
            - **🔍 Semantic Search**: Finds documents based on meaning and context
            - **🔤 Keyword Search**: Traditional keyword-based search
            - **🤖 Hybrid Search**: Combines both semantic and keyword search
            
            ### Tips for Better Results:
            1. Use natural language queries (e.g., "Find documents about data privacy")
            2. Try different search modes for different types of queries
            3. Filter by document or chunking method to narrow down results
            4. Adjust the number of results as needed
            
            ### Advanced:
            - Use the chunking method filter to see how different chunking strategies affect results
            - Try different embedding models for potentially better semantic understanding
            """)
            st.session_state.legal_llm_model = selected_llm_model
        
        st.markdown("---")
        
            # Initialize query results in session state
        if 'legal_query_results' not in st.session_state:
            st.session_state.legal_query_results = None
            st.session_state.legal_query_retrieval_timing = None
            st.session_state.legal_query_params = None
        
        # Query Input with blue search button
        legal_query = st.text_area(
            "Enter your query",
            height=100,
            placeholder="Search legal documents...",
            key="legal_query_input"
        )
        
        # Ensure query variables are defined (may be None if not available)
        if 'query_chunking_method' not in locals():
            query_chunking_method = st.session_state.get('legal_query_chunking_method', None)
        
        if 'query_embedding' not in locals():
            query_embedding = st.session_state.get('legal_query_embedding_model', None)
        
        if 'query_source_file' not in locals():
            query_source_file = st.session_state.get('legal_query_source_file', "All Files")
        
        if 'search_mode' not in locals():
            search_mode = st.session_state.get('legal_query_search_mode', "semantic")
        
        if 'selected_llm_model' not in locals():
            selected_llm_model = st.session_state.get('legal_llm_model', "llama3.1:latest")
        
        # Check if query parameters have changed
        current_query_params = {
            'query': legal_query,
            'embedding_model': query_embedding,
            'chunking_method': query_chunking_method,
            'source_file': query_source_file if query_source_file != "All Files" else None,
            'search_mode': search_mode,
            'llm_model': selected_llm_model
        }
        
        # Determine if we should run query
        should_run_query = st.session_state.get('should_run_query', False)
        
        # Also check if query parameters changed and we have cached results
        if (st.session_state.legal_query_params != current_query_params and 
            st.session_state.legal_query_results is not None):
            # Parameters changed, clear cached results
            st.session_state.legal_query_results = None
            st.session_state.legal_query_retrieval_timing = None
        
        # Run query if button clicked
        if should_run_query:
            if not legal_query:
                st.warning("⚠️ Please enter a query")
            elif st.session_state.legal_qdrant_client is None:
                st.error("⚠️ Qdrant client not available")
            elif query_chunking_method is None:
                st.warning("⚠️ No chunking method available. Please upload documents first.")
            else:
                import time
                
                try:
                    with st.spinner("Searching legal documents..."):
                        # Query with timing enabled
                        query_result = query_legal_documents(
                            qdrant_client=st.session_state.legal_qdrant_client,
                            collection_name=st.session_state.legal_collection_name,
                            query_text=legal_query,
                            embedding_model=query_embedding,
                            chunking_method=query_chunking_method,
                            source_file=query_source_file if query_source_file != "All Files" else None,
                        top_k=5,
                            search_mode=search_mode,
                            return_timing=True
                        )
                        
                        # Unpack results and timing
                        results, retrieval_timing = query_result
                        
                        # Store in session state
                        st.session_state.legal_query_results = results
                        st.session_state.legal_query_retrieval_timing = retrieval_timing
                        st.session_state.legal_query_params = current_query_params.copy()
                except Exception as e:
                    st.error(f"❌ Error searching documents: {e}")
                    logger.error(f"Legal document query error: {e}", exc_info=True)
        
        # Display results if available
        if st.session_state.legal_query_results is not None:
            results = st.session_state.legal_query_results
            retrieval_timing = st.session_state.legal_query_retrieval_timing
            
            if results:
                st.success(f"✅ Found {len(results)} unique results")
                
                # Generate LLM answer from reranked results
                try:
                    from langchain_ollama import ChatOllama
                    import os
                    
                    with st.spinner(f"Generating answer using {st.session_state.legal_query_params['llm_model']}..."):
                        # Initialize LLM
                        llm = ChatOllama(
                            model=st.session_state.legal_query_params['llm_model'],
                            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                            temperature=0.1  # Low temperature for factual answers
                        )
                        
                        # Format context from search results with metadata
                        import re
                        retrieved_chunks_formatted = []
                        
                        for i, result in enumerate(results, 1):
                            # Get text content (remove HTML tags if present)
                            text = result.get('text', result.get('highlighted_text', ''))
                            text_clean = re.sub(r'<[^>]+>', '', text) if isinstance(text, str) else str(text)
                            
                            # Collect extracted entities
                            entities = []
                            if result.get('individual_name'):
                                entities.append(f"Person: {result.get('individual_name')}")
                            if result.get('company_name'):
                                entities.append(f"Company: {result.get('company_name')}")
                            if result.get('email'):
                                entities.append(f"Email: {result.get('email')}")
                            if result.get('phone'):
                                entities.append(f"Phone: {result.get('phone')}")
                            if result.get('address'):
                                entities.append(f"Address: {result.get('address')}")
                            
                            entities_str = ", ".join(entities) if entities else "None"
                            
                            # Format chunk with metadata (limit to 1500 chars for context)
                            chunk_info = f"""Chunk {i}:
  Content: {text_clean[:1500]}
  Metadata:
    - Source file: {result.get('source_file', result.get('file_name', 'Unknown'))}
    - Chunking method: {result.get('chunking_method', 'Unknown')}
    - Page: {result.get('page', 'N/A')}
    - Hierarchy level: {result.get('hierarchy_level', 'N/A')}
    - Extracted entities: {entities_str}"""
                            
                            # Add additional metadata if available
                            if result.get('clause_number'):
                                chunk_info += f"\n    - Clause number: {result.get('clause_number')}"
                            if result.get('title'):
                                chunk_info += f"\n    - Title: {result.get('title')}"
                            if result.get('parent'):
                                chunk_info += f"\n    - Parent: {result.get('parent')}"
                            
                            retrieved_chunks_formatted.append(chunk_info)
                        
                            # Join all chunks
                            retrieved_chunks_context = "\n\n".join(retrieved_chunks_formatted)
                            
                            # Check if we have cached answer for these exact parameters
                            answer_cache_key = f"answer_{hash(str(st.session_state.legal_query_params))}_{hash(retrieved_chunks_context[:1000])}"
                            
                            if answer_cache_key in st.session_state:
                                # Use cached answer
                                llm_answer = st.session_state[answer_cache_key]['answer']
                                answer_time = st.session_state[answer_cache_key]['time']
                        else:
                            # Create prompt using the new template
                            prompt = f"""You are an expert legal assistant. Your task is to answer the user query based on the following retrieved document chunks. Use the content and metadata to provide a precise, accurate, and concise answer.

Guidelines:

1. Use the retrieved chunks as the only source of truth. Do not make up information.

2. For each chunk, you have access to:

   - Content: The text content of the chunk

   - Metadata:

       - Source file: The name of the source document file

       - Chunking method: The method used to create this chunk

       - Page: The page number where this chunk appears

       - Hierarchy level: The hierarchical level of this chunk in the document structure

       - Extracted entities: Person, Company, Email, Phone, Address (if present in the chunk)

3. If multiple chunks contain the same information, avoid repeating it in your answer.

4. If the query specifically relates to a person, company, email, phone, or address, focus on chunks where these entities are present.

5. If the query is general, synthesize the answer from relevant chunks, maintaining legal accuracy.

6. Cite the source chunk file and page for each fact used in your answer when possible.

7. If no relevant information exists in the retrieved chunks, respond: "The information is not available in the provided documents."

User Query:

{st.session_state.legal_query_params['query']}

Retrieved Chunks (Context):

{retrieved_chunks_context}

Your Answer:"""
                            
                            # Generate answer with timing
                            answer_start = time.time()
                            response = llm.invoke(prompt)
                            answer_time = time.time() - answer_start
                            llm_answer = response.content if hasattr(response, 'content') else str(response)
                            
                            # Cache the answer
                            st.session_state[answer_cache_key] = {
                                'answer': llm_answer,
                                'time': answer_time
                            }
                            
                            # Display LLM answer prominently
                            st.markdown("### 💬 Generated Answer")
                            st.info(llm_answer)
                            st.markdown("---")
                                
                        # Display timing information
                        st.markdown("### ⏱️ Performance Metrics")
                        timing_cols = st.columns(4)
                        
                        with timing_cols[0]:
                            if retrieval_timing.get("embedding_time", 0) > 0:
                                st.metric("Query Embedding", f"{retrieval_timing['embedding_time']:.3f}s")
                            else:
                                st.metric("Query Embedding", "N/A")
                        
                        with timing_cols[1]:
                            if retrieval_timing.get("vector_search_time", 0) > 0:
                                st.metric("Vector Search", f"{retrieval_timing['vector_search_time']:.3f}s")
                            elif retrieval_timing.get("bm25_search_time", 0) > 0:
                                st.metric("BM25 Search", f"{retrieval_timing['bm25_search_time']:.3f}s")
                            else:
                                st.metric("Search", "N/A")
                        
                        with timing_cols[2]:
                            if retrieval_timing.get("reranking_time", 0) > 0:
                                st.metric("Reranking", f"{retrieval_timing['reranking_time']:.3f}s")
                            else:
                                st.metric("Reranking", "N/A")
                        
                        with timing_cols[3]:
                            st.metric("Answer Generation", f"{answer_time:.3f}s")
                        
                        # Show detailed timing breakdown
                        with st.expander("📊 Detailed Timing Breakdown", expanded=False):
                            timing_data = []
                            
                            if retrieval_timing.get("embedding_time", 0) > 0:
                                timing_data.append({
                                    "Step": "Query Embedding",
                                    "Time (s)": f"{retrieval_timing['embedding_time']:.4f}",
                                    "Percentage": f"{(retrieval_timing['embedding_time'] / retrieval_timing.get('total_retrieval_time', 1) * 100):.1f}%"
                                })
                            
                            if retrieval_timing.get("vector_search_time", 0) > 0:
                                timing_data.append({
                                    "Step": "Vector Search",
                                    "Time (s)": f"{retrieval_timing['vector_search_time']:.4f}",
                                    "Percentage": f"{(retrieval_timing['vector_search_time'] / retrieval_timing.get('total_retrieval_time', 1) * 100):.1f}%"
                                })
                            
                            if retrieval_timing.get("bm25_search_time", 0) > 0:
                                timing_data.append({
                                    "Step": "BM25 Search",
                                    "Time (s)": f"{retrieval_timing['bm25_search_time']:.4f}",
                                    "Percentage": f"{(retrieval_timing['bm25_search_time'] / retrieval_timing.get('total_retrieval_time', 1) * 100):.1f}%"
                                })
                            
                            if retrieval_timing.get("reranking_time", 0) > 0:
                                timing_data.append({
                                    "Step": "Reranking/Fusion",
                                    "Time (s)": f"{retrieval_timing['reranking_time']:.4f}",
                                    "Percentage": f"{(retrieval_timing['reranking_time'] / retrieval_timing.get('total_retrieval_time', 1) * 100):.1f}%"
                                })
                            
                            timing_data.append({
                                "Step": "Answer Generation",
                                "Time (s)": f"{answer_time:.4f}",
                                "Percentage": f"{(answer_time / (retrieval_timing.get('total_retrieval_time', 0) + answer_time) * 100):.1f}%"
                            })
                            
                            total_time = retrieval_timing.get('total_retrieval_time', 0) + answer_time
                            timing_data.append({
                                "Step": "**Total**",
                                "Time (s)": f"**{total_time:.4f}**",
                                "Percentage": "100.0%"
                            })
                            
                            if timing_data and pd is not None:
                                df_timing = pd.DataFrame(timing_data)
                                st.dataframe(
                                    df_timing,
                                    use_container_width=True,
                                    hide_index=True
                                )
                            else:
                                # Fallback: display as markdown table
                                st.markdown("| Step | Time (s) | Percentage |")
                                st.markdown("|------|----------|------------|")
                                for item in timing_data:
                                    st.markdown(f"| {item['Step']} | {item['Time (s)']} | {item['Percentage']} |")
                        
                                st.markdown("---")
                                
                except ImportError:
                    st.warning("⚠️ langchain-ollama not available. Install with: pip install langchain-ollama")
                except Exception as e:
                    logger.error(f"Error generating LLM answer: {e}", exc_info=True)
                    st.warning(f"⚠️ Could not generate answer: {e}")
                
                st.info("💡 Click '🔍 Search Legal Documents' to run a query")
                
                # Explanation about search modes and results
                with st.expander("ℹ️ About Search Modes & Results", expanded=False):
                            st.markdown(f"""
                            **Search Mode: {search_mode.upper()}**
                            
                            - **Semantic**: Pure embedding-based similarity search using vector embeddings
                            - **BM25**: Keyword-based search using BM25 ranking algorithm (no embeddings needed)
                            - **Mixed**: Hybrid search combining semantic + BM25 using Reciprocal Rank Fusion (RRF)
                            
                            **Deduplication:**
                            - Duplicate chunks with identical content are automatically removed
                            - Only unique chunks are displayed, even if they were created by different chunking methods
        - Deduplication is based on normalized text content (lowercase, whitespace-normalized)
                            
                            **Scores:**
        - **Semantic mode**: 
          - **Cosine Similarity**: Actual cosine similarity between query and chunk embeddings (0-1, where 1.0 = identical)
          - **Normalized Score**: Relative score scaled to [0, 1] range (best result = 1.0, even if actual similarity is low)
          - \u26A0\uFE0F **Important**: A normalized score of 1.0 doesn't mean perfect match - check the Cosine Similarity value!
          - Good matches typically have cosine similarity > 0.7, excellent matches > 0.9
                            - **BM25 mode**: BM25 relevance scores (higher = more relevant keywords)
                            - **Mixed mode**: RRF fusion scores combining both semantic and keyword relevance
        
        **Why Same Content Shows Different Scores:**
        - Different chunking methods may create chunks with the same text but different metadata
        - Scores reflect the relevance of the **entire chunk** (including context/metadata), not just the text
        - Semantic scores depend on the embedding model and query embedding
        - BM25 scores depend on keyword frequency and document length
        - Mixed mode combines both, so scores can vary based on which search method contributes more
        - Deduplication removes duplicates **after** scoring, so you may see different scores for identical text if they came from different search methods or contexts
                            
                            **Filtering:**
                            - Results are filtered by the selected chunking method from Qdrant metadata
                            - Only chunks matching the selected method are returned
                            - For semantic/mixed modes, embedding model must match the one used during upload
        - Filtering happens **before** deduplication, ensuring only relevant chunks are considered
                            """)
                        
        # Display search results if available
        if st.session_state.legal_query_results is not None and len(st.session_state.legal_query_results) > 0:
            st.markdown("### 📋 Search Results")
            st.markdown("---")
            
            for i, result in enumerate(st.session_state.legal_query_results, 1):
                # Display result in a card-like format with expandable sections
                # Show both normalized score and original cosine similarity if available
                score_display = f"{result['score']:.4f}"
                if result.get('original_score') is not None:
                    score_display = f"{result['score']:.4f} (Cosine: {result['original_score']:.4f})"
                
                with st.expander(f"📄 Result {i} - Score: {score_display} | {result.get('source_file', result.get('file_name', 'Unknown'))}", expanded=True):
                    # Display the actual text content prominently
                    st.markdown("### 📄 Chunk Content")
                    
                    # For BM25 and mixed modes, show highlighted text if available; otherwise show plain text
                    if st.session_state.legal_query_params['search_mode'] in ["bm25", "mixed"] and result.get('highlighted_text'):
                        st.markdown(result.get('highlighted_text'), unsafe_allow_html=True)
                    else:
                        text_content = result.get('text', 'No text content available')
                        st.markdown(f"```\n{text_content}\n```")
                    
                    st.markdown("---")
                    
                    # Display all metadata in organized sections
                    st.markdown("### 📋 Metadata")
                    
                    # Core metadata in columns
                    metadata_cols = st.columns(4)
                    with metadata_cols[0]:
                        st.markdown(f"**Source File:**\n{result.get('source_file', result.get('file_name', 'Unknown'))}")
                    with metadata_cols[1]:
                        st.markdown(f"**Chunking Method:**\n{result.get('chunking_method', 'Unknown')}")
                    with metadata_cols[2]:
                        page = result.get('page')
                        st.markdown(f"**Page:**\n{page if page is not None else 'N/A'}")
                    with metadata_cols[3]:
                        st.markdown(f"**Hierarchy Level:**\n{result.get('hierarchy_level', 'N/A')}")
                    
                    # Additional metadata
                    additional_metadata = []
                    if result.get('clause_number'):
                        additional_metadata.append(("Clause Number", result.get('clause_number')))
                    if result.get('title'):
                        additional_metadata.append(("Title", result.get('title')))
                    if result.get('parent'):
                        additional_metadata.append(("Parent", result.get('parent')))
                    if result.get('embedding_model'):
                        additional_metadata.append(("Embedding Model", result.get('embedding_model')))
                    if result.get('file_id'):
                        additional_metadata.append(("File ID", result.get('file_id')))
                    
                    if additional_metadata:
                        st.markdown("**Additional Information:**")
                        for key, value in additional_metadata:
                            st.caption(f"**{key}:** {value}")
                    
                    # Extracted entities section
                    extracted_entities = []
                    if result.get('individual_name'):
                        extracted_entities.append(("👤 Person", result.get('individual_name')))
                    if result.get('company_name'):
                        extracted_entities.append(("🏢 Company", result.get('company_name')))
                    if result.get('email'):
                        extracted_entities.append(("📧 Email", result.get('email')))
                    if result.get('phone'):
                        extracted_entities.append(("📞 Phone", result.get('phone')))
                    if result.get('address'):
                        extracted_entities.append(("📍 Address", result.get('address')))
                    
                    if extracted_entities:
                        st.markdown("### 🏷️ Extracted Entities")
                        entity_cols = st.columns(len(extracted_entities) if len(extracted_entities) <= 5 else 5)
                        for idx, (entity_type, entity_value) in enumerate(extracted_entities):
                            with entity_cols[idx % len(entity_cols)]:
                                st.caption(f"**{entity_type}:**\n{entity_value}")
                    else:
                        st.caption("_No extracted entities found in this chunk_")
                    
                    # Score information
                    st.markdown("### 📊 Score Information")
                    score_info = []
                    
                    # Show original cosine similarity prominently (this is the actual similarity)
                    if result.get('original_score') is not None:
                        original_score = result['original_score']
                        score_info.append(f"**Cosine Similarity:** {original_score:.4f}")
                        # Add interpretation
                        if original_score >= 0.9:
                            interpretation = "Very High Similarity"
                        elif original_score >= 0.7:
                            interpretation = "High Similarity"
                        elif original_score >= 0.5:
                            interpretation = "Moderate Similarity"
                        elif original_score >= 0.3:
                            interpretation = "Low Similarity"
                        else:
                            interpretation = "Very Low Similarity"
                        score_info.append(f"({interpretation})")
                    
                    # Show normalized score if different from original
                    if result.get('original_score') is not None and abs(result['score'] - result['original_score']) > 0.01:
                        score_info.append(f"**Normalized Score:** {result['score']:.4f}")
                        score_info.append("(relative to other results)")
                    
                    # Show other scores if available
                    if result.get('bm25_score') is not None:
                        score_info.append(f"**BM25 Score:** {result.get('bm25_score'):.4f}")
                    if result.get('semantic_score') is not None:
                        score_info.append(f"**Semantic Score:** {result.get('semantic_score'):.4f}")
                    if result.get('rrf_score') is not None:
                        score_info.append(f"**RRF Score:** {result.get('rrf_score'):.4f}")
                    
                    if score_info:
                        st.caption(" | ".join(score_info))
                    else:
                        st.caption(f"**Relevance Score:** {result['score']:.4f}")

elif st.session_state.main_mode == "🔍 Simple Web Search":
    # Simple Web Search Mode
    st.subheader("🔍 Simple Web Search (DeepSeek)")
    st.markdown("Search the web for any topic using DeepSeek - open-source web search with semantic reranking.")
    
    # Search input
    search_query = st.text_input(
        "Enter search topic",
        placeholder="e.g., Kernel ellipsoidal trimming, Machine Learning, BBC, etc.",
        key="simple_search_input"
    )
    
    # Configuration options
    col1, col2 = st.columns([2, 1])
    with col1:
        # Number of results selector
        max_results = st.slider(
            "Number of results",
            min_value=5,
            max_value=20,
            value=10,
            step=5,
            help="Select how many search results to retrieve (default: 10). Hybrid BM25 + embedding reranking is enabled by default."
        )
    with col2:
        # Search engine selector
        search_engine = st.selectbox(
            "Search Engine",
            options=["duckduckgo", "google"],
            index=0,
            help="Choose search engine (DuckDuckGo is privacy-focused, Google may have more results)"
        )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            use_reranking = st.checkbox(
                "Enable Semantic Reranking",
                value=True,
                help="Use BM25 + embedding reranking for better relevance"
            )
        with col2:
            use_cache = st.checkbox(
                "Enable Caching",
                value=True,
                help="Cache search results for faster repeated queries"
            )
        
        # Ranking mode selector
        rank_mode = st.selectbox(
            "Ranking Mode",
            options=["rrf", "bm25", "embedding"],
            index=0,
            help="""Ranking method:
            - RRF: Reciprocal Rank Fusion (combines BM25 + embeddings)
            - BM25: Keyword-based ranking only
            - Embedding: Semantic similarity ranking only"""
        )
        
        # Embedding model selector
        try:
            legal_chunker_module = importlib.import_module('scripts.00_chunking.legal_chunker_integration')
            available_models = legal_chunker_module.EMBEDDING_MODELS if hasattr(legal_chunker_module, 'EMBEDDING_MODELS') else ["llama3.1:latest"]
        except:
            available_models = ["llama3.1:latest"]
        
        embedding_model = st.selectbox(
            "Embedding Model",
            options=available_models,
            index=0,
            help="Select embedding model for semantic similarity calculation"
        )
    
    # Search button
    if st.button("🔍 Search", type="primary", use_container_width=True, key="simple_web_search"):
        if search_query and search_query.strip():
            try:
                # Lazy load DeepSeek module
                deepseek_module, DEEPSEEK_AVAILABLE = _get_deepseek_module()
                if not DEEPSEEK_AVAILABLE or deepseek_module is None:
                    st.error("❌ DeepSeek is not available. Please install dependencies:")
                    st.code("pip install -r requirements.txt")
                    st.stop()
                
                DeepSeek = deepseek_module.DeepSeek
                
                # Initialize DeepSeek if not already done or if settings changed
                cache_key = f"deepseek_{search_engine}_{use_reranking}_{use_cache}_{rank_mode}_{embedding_model}"
                if cache_key not in st.session_state:
                    try:
                            st.session_state[cache_key] = DeepSeek(
                                search_engine=search_engine,
                                use_cache=use_cache,
                                use_reranking=use_reranking,
                                embedding_model=embedding_model
                            )
                            logger.info(f"Initialized DeepSeek with engine={search_engine}, reranking={use_reranking}, rank_mode={rank_mode}")
                    except Exception as e:
                        st.error(f"❌ Failed to initialize DeepSeek: {e}")
                        logger.error(f"DeepSeek initialization error: {e}", exc_info=True)
                        st.stop()
                
                deepseek = st.session_state[cache_key]
                
                # Perform search
                results = []
                search_time = 0
                
                with st.spinner(f"🔍 DeepSeek searching '{search_query}' (mode: {rank_mode.upper()})..."):
                    import time
                    start_time = time.time()
                    
                    # Execute search with selected ranking mode
                    results = deepseek.search(
                        query=search_query.strip(),
                        max_results=max_results,
                        rerank=use_reranking,
                        rank_mode=rank_mode
                    )
                    
                    search_time = time.time() - start_time
                    
                    # Debug: Log search status
                    logger.info(f"DeepSeek search completed: query='{search_query}', results={len(results) if results else 0}, time={search_time:.2f}s")
                        
                        # Display results
                    if results:
                        st.success(f"✅ Found {len(results)} results in {search_time:.2f}s")
                        st.markdown("---")
                        
                        # Display each result
                        for i, result in enumerate(results, 1):
                            # Extract core fields
                            title = result.get('title', 'Untitled')
                            url = result.get('url', '')
                            snippet = result.get('snippet', '')
                            source = result.get('source', result.get('domain', ''))
                            relevance_score = result.get('relevance_score', 0.0)
                            bm25_score = result.get('bm25_score', 0.0)
                            embedding_score = result.get('embedding_score', 0.0)
                            
                            # Display result card
                            with st.container():
                                # Show relevance score based on ranking mode
                                score_info = []
                            if rank_mode == "rrf":
                                rrf_score = result.get('rrf_score', relevance_score)
                                score_info.append(f"**Relevance (RRF):** {rrf_score:.2f}")
                            elif rank_mode == "bm25":
                                score_info.append(f"**Relevance (BM25):** {relevance_score:.2f}")
                            elif rank_mode == "embedding":
                                score_info.append(f"**Relevance (Embedding):** {relevance_score:.2f}")
                            else:
                                score_info.append(f"**Relevance:** {relevance_score:.2f}")
                            
                            # Always show BM25 and Embedding scores for transparency
                            score_info.append(f"BM25: {bm25_score:.2f}")
                            score_info.append(f"Embedding: {embedding_score:.2f}")
                            
                            header = f"### {i}. {title}"
                            if score_info:
                                header += f" ({' | '.join(score_info)})"
                            st.markdown(header)
                            
                            if url:
                                st.markdown(f"🔗 [{url}]({url})")
                            
                            if snippet:
                                st.write(snippet)
                            
                            # Show source type and domain
                            source_info = []
                            if source:
                                source_info.append(f"**Source:** {source}")
                            domain = result.get('domain', '')
                            if domain and domain != source:
                                source_info.append(f"**Domain:** {domain}")
                            
                            if source_info:
                                st.caption(" | ".join(source_info))
                            
                                if i < len(results):
                                    st.markdown("---")
                        
                        # Download results as JSON
                        import json
                        results_json = json.dumps(results, indent=2, ensure_ascii=False, default=str)
                        st.download_button(
                            "📥 Download Results (JSON)",
                            data=results_json,
                                file_name=f"deepseek_results_{search_query[:30].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    else:
                            st.warning("⚠️ No results found.")
                    
                    # Show detailed troubleshooting
                    with st.expander("🔍 Troubleshooting"):
                        st.markdown("""
                        **Possible reasons:**
                        1. **Search engine may be temporarily unavailable** - Try switching to a different engine
                        2. **Search query too specific** - Try broader or simpler terms
                        3. **Network/connection issues** - Check your internet connection
                        4. **Rate limiting** - Wait a few moments and try again
                        
                        **Tips:**
                        - Try a simple test query like "Python" or "Machine Learning"
                        - Switch between DuckDuckGo and Google search engines
                        - Check the console/logs for detailed error messages
                        """)
                        
            except Exception as e:
                st.error(f"❌ Error searching: {e}")
                logger.error(f"DeepSeek search error: {e}", exc_info=True)
                st.info("💡 **Troubleshooting:**\n- Make sure dependencies are installed: `pip install -r requirements.txt`\n- Check your internet connection\n- Try a different search query or engine")
        else:
            st.warning("⚠️ Please enter a search topic")

elif st.session_state.main_mode == "📚 Research Overview":
    # Research Overview Mode
    st.subheader("Generate Research Overview Paper")
    st.markdown("""
    Generate a comprehensive scholarly survey paper with:
    - Abstract & Introduction
    - Background & Foundations
    - Taxonomy & Classification
    - Recent Advances (select date range below)
    - Applications & Use Cases
    - Comparative Analysis
    - Challenges & Limitations
    - Future Research Directions
    - Conclusion
    """)
    
    research_topic = st.text_area(
        "Enter research topic for overview paper",
        height=100,
        placeholder="e.g., Explainability of Large Language Models, Novelty Detection in Deep Learning, etc."
    )
    
    # Date interval selection
    st.markdown("### 📅 Date Range Selection")
    current_year = datetime.now().year
    default_start_year = current_year - 5
    
    # Use sliders for date range
    col_date1, col_date2 = st.columns([1, 1])
    with col_date1:
        start_year = st.slider(
            "Start Year",
            min_value=1990,
            max_value=current_year,
            value=default_start_year,
            step=1,
            key="research_start_year",
            help="Start year for paper search (default: 5 years back from current year)"
        )
    with col_date2:
        end_year = st.slider(
            "End Year",
            min_value=1990,
            max_value=current_year + 1,
            value=current_year,
            step=1,
            key="research_end_year",
            help="End year for paper search (default: current year)"
        )
    
    # Validate date range
    if start_year > end_year:
        st.error("⚠️ Start year must be less than or equal to end year. Please adjust the date range.")
        st.stop()
    
    # Show date range info
    year_span = end_year - start_year + 1
    st.info(f"📊 Will search for papers published between **{start_year}** and **{end_year}** ({year_span} year{'s' if year_span != 1 else ''})")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        use_web_research = st.checkbox("Use web research", value=True, help="Perform web search to gather references")
    with col2:
        max_web_results = st.slider(
            "Max web results",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            key="max_web_results_slider",
            help="Research Overview uses parallelization and can handle 100+ papers"
        )
    with col3:
        use_academic_sources = st.checkbox("Use academic sources", value=False, 
                                          help="Search arXiv and Semantic Scholar (can handle 40+ papers with parallel processing)")
    
    if st.button("📚 Generate Research Overview", type="primary", use_container_width=True, key="generate_research_overview"):
        if research_topic:
            # Lazy load research overview workflow
            research_overview_workflow = _get_research_overview_workflow()
            if research_overview_workflow is None:
                st.error("Research overview workflow not initialized. Please check logs.")
            else:
                try:
                    with st.spinner("Generating comprehensive research overview... This may take several minutes."):
                        # First, gather research context
                        context = ""
                        references = []
                        
                        if use_web_research:
                            # Lazy load research modules
                            research_modules = _get_research_modules()
                            if not research_modules:
                                st.error("Research modules not available")
                                st.stop()
                            
                            ResearchAssistant = research_modules['research_assistant'].ResearchAssistant
                            
                            # Use research assistant to gather web research
                            if 'research_assistant' not in st.session_state:
                                # Get retrieval engine from workflow
                                retrieval_engine = research_overview_workflow.retrieval_engine
                                st.session_state.research_assistant = ResearchAssistant(
                                    retrieval_engine=retrieval_engine,
                                    use_ollama=True,
                                    framework="LangChain"
                                )
                            
                            # Perform web search with error handling
                            if st.session_state.research_assistant.web_search:
                                try:
                                    web_results = st.session_state.research_assistant.web_search.search(
                                        research_topic,
                                        max_results=max_web_results
                                    )
                                    references = web_results if web_results else []
                                    
                                    # Synthesize context from web results if available
                                    if web_results and st.session_state.research_assistant.synthesizer:
                                        try:
                                            context = st.session_state.research_assistant.synthesizer.synthesize(
                                                query=research_topic,
                                                retrieved_chunks=[],
                                                web_results=web_results,
                                                memory_context=""
                                            )
                                        except Exception as synth_error:
                                            logger.warning(f"Context synthesis failed: {synth_error}, using empty context")
                                            context = ""
                                    
                                    if not web_results:
                                        st.info("ℹ️ No web search results available. Generating overview without web references.")
                                except Exception as search_error:
                                    logger.error(f"Web search error in research overview: {search_error}")
                                    st.warning(f"⚠️ Web search failed: {str(search_error)[:200]}. Continuing without web references.")
                                    references = []
                                    context = ""
                        else:
                            # No web research requested
                            context = ""
                        
                        # Generate research overview
                        result = research_overview_workflow.execute(
                            topic=research_topic,
                            context=context,
                            references=references,
                            use_web_research=use_web_research,
                            max_academic_papers=40 if use_academic_sources else 0,
                            min_year=start_year,
                            max_year=end_year
                        )
                        
                        if result.get("report") and not result.get("report", {}).get("error"):
                            report = result["report"]
                            
                            st.success("✅ Research overview generated successfully!")
                            
                            # Show errors if any
                            if result.get("errors"):
                                st.warning(f"⚠️ {len(result['errors'])} errors encountered during generation")
                                with st.expander("View Errors"):
                                    for error in result["errors"]:
                                        st.error(error)
                        
                        # Display report
                            st.markdown("## Generated Research Overview")
                            st.markdown(report.get("markdown", report.get("plain_text", "No content")))
                            
                            # Save HTML and show download buttons
                            report_text = report.get("markdown", report.get("plain_text", ""))
                            html_path = None
                            
                            # Save HTML automatically
                            try:
                                html_exporter_module = importlib.import_module('scripts.05_output_generation.html_exporter')
                                save_research_overview_html = html_exporter_module.save_research_overview_html
                                html_path = save_research_overview_html(report, research_topic)
                                st.success(f"🌐 HTML report saved to: `{html_path}`")
                                
                                # Show link to HTML file
                                html_path_obj = Path(html_path)
                                if html_path_obj.exists():
                                    # Convert to relative path for display
                                    try:
                                        rel_path = html_path_obj.relative_to(Path.cwd())
                                        st.info(f"📄 HTML file location: `{rel_path}`")
                                    except:
                                        st.info(f"📄 HTML file location: `{html_path}`")
                            except Exception as html_error:
                                logger.warning(f"HTML export error: {html_error}")
                                st.warning(f"⚠️ Could not save HTML: {html_error}")
                            
                            # Download buttons
                            col_dl1, col_dl2 = st.columns([1, 1])
                            with col_dl1:
                                            st.download_button(
                                    "📥 Download Markdown",
                                    data=report_text,
                                    file_name=f"research_overview_{research_topic[:30].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                        mime="text/markdown",
                                    key="download_markdown"
                                )
                            with col_dl2:
                                # Read HTML file if it was saved
                                if html_path and Path(html_path).exists():
                                    try:
                                        with open(html_path, 'r', encoding='utf-8') as f:
                                            html_content = f.read()
                                        st.download_button(
                                            "📥 Download HTML",
                                            data=html_content,
                                            file_name=Path(html_path).name,
                                            mime="text/html",
                                            key="download_html"
                                        )
                                    except Exception as dl_error:
                                        logger.warning(f"HTML download button error: {dl_error}")
                                else:
                                    st.info("HTML file not available")
                            
                            # Show sections summary
                            sections = report.get("sections", {})
                            if sections:
                                st.markdown("---")
                                st.subheader("📋 Generated Sections")
                                for section_name, section_content in sections.items():
                                    with st.expander(f"{section_name.replace('_', ' ').title()} ({len(section_content)} chars)"):
                                        st.markdown(section_content[:500] + "..." if len(section_content) > 500 else section_content)
                            
                            # Show references
                            if report.get("references"):
                                st.markdown("---")
                                st.subheader(f"📚 References ({len(report['references'])} sources)")
                                for i, ref in enumerate(report["references"], 1):
                                    with st.expander(f"{i}. {ref.get('title', 'Untitled')}"):
                                        if ref.get('url'):
                                            st.markdown(f"**URL:** [{ref['url']}]({ref['url']})")
                                        if ref.get('author'):
                                            st.write(f"**Author:** {ref['author']}")
                                        if ref.get('date'):
                                            st.write(f"**Date:** {ref['date']}")
                                        if ref.get('publication'):
                                            st.write(f"**Publication:** {ref['publication']}")
                                        if ref.get('snippet'):
                                            st.write(f"**Summary:** {ref['snippet'][:300]}...")
                        else:
                            error_msg = result.get("report", {}).get("error", "Unknown error")
                            st.error(f"Failed to generate research overview: {error_msg}")
                            if result.get("errors"):
                                with st.expander("View Errors"):
                                    for error in result["errors"]:
                                        st.error(error)
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.error(f"Research overview generation error: {e}", exc_info=True)
        else:
            st.warning("Please enter a research topic")

elif st.session_state.main_mode == "🧩 Chunk Exploration":
    # Chunk Exploration Mode
    st.subheader("🧩 Chunk Exploration")
    st.markdown("""
    **Explore how documents are chunked and stored in Qdrant.**
    
    Navigate through chunks with Previous/Next buttons, slider, or arrow keys (↑/↓).
    """)
    
    # Initialize session state for chunk navigation
    if "chunk_exploration_current_index" not in st.session_state:
        st.session_state.chunk_exploration_current_index = 0
    if "chunk_exploration_chunks" not in st.session_state:
        st.session_state.chunk_exploration_chunks = []
    if "chunk_exploration_file" not in st.session_state:
        st.session_state.chunk_exploration_file = None
    if "chunk_exploration_method" not in st.session_state:
        st.session_state.chunk_exploration_method = None
    if "chunk_exploration_level" not in st.session_state:
        st.session_state.chunk_exploration_level = None  # None = all levels
    
    # Check if Qdrant is available
    if not LEGAL_CHUNKER_AVAILABLE:
        st.error("❌ Legal chunker not available. Please install required dependencies.")
    elif "legal_qdrant_client" not in st.session_state or st.session_state.legal_qdrant_client is None:
        st.warning("⚠️ Qdrant client not initialized. Please configure Qdrant connection in the sidebar.")
    else:
        try:
            # Get available files and chunking methods
            source_files = get_distinct_source_files(
                st.session_state.legal_qdrant_client,
                st.session_state.legal_collection_name
            )
            
            if not source_files:
                st.info("📭 No documents found in Qdrant. Please upload documents first using the 'Legal Documents Query' tab.")
            else:
                # Two-panel layout
                left_panel, right_panel = st.columns([1, 2])
                
                with left_panel:
                    st.markdown("### 📄 Document Selector")
                    
                    # Document selection
                    selected_file = st.selectbox(
                        "Choose a document:",
                        options=source_files,
                        key="chunk_exploration_file_selector",
                        help="Select a document to view its chunks",
                        index=source_files.index(st.session_state.chunk_exploration_file) if st.session_state.chunk_exploration_file in source_files else 0
                    )
                    
                    # Reset current index if file changed
                    if selected_file != st.session_state.chunk_exploration_file:
                        st.session_state.chunk_exploration_file = selected_file
                        st.session_state.chunk_exploration_current_index = 0
                        st.session_state.chunk_exploration_chunks = []
                        st.session_state.chunk_exploration_method = None
                    
                    if selected_file:
                        # Get chunking methods for this file
                        all_chunking_methods = get_distinct_chunking_methods(
                            st.session_state.legal_qdrant_client,
                            st.session_state.legal_collection_name
                        )
                        
                        # Filter to only methods that exist for this file
                        available_methods_for_file = []
                        for method in all_chunking_methods:
                            try:
                                test_chunks = get_chunks_for_exploration(
                                    qdrant_client=st.session_state.legal_qdrant_client,
                                    collection_name=st.session_state.legal_collection_name,
                                    source_file=selected_file,
                                    chunking_method=method,
                                    limit=1
                                )
                                if test_chunks:
                                    available_methods_for_file.append(method)
                            except Exception:
                                continue
                        
                        if not available_methods_for_file:
                            st.warning(f"⚠️ No chunks found for document '{selected_file}'.")
                        else:
                            # Chunking method selection
                            st.markdown("### ✂️ Chunking Method")
                            selected_method = st.selectbox(
                                "Select method:",
                                options=available_methods_for_file,
                                key="chunk_exploration_method_selector",
                                help="Select a chunking method",
                                index=available_methods_for_file.index(st.session_state.chunk_exploration_method) if st.session_state.chunk_exploration_method in available_methods_for_file else 0
                            )
                            
                            # Reset index if method changed
                            if selected_method != st.session_state.chunk_exploration_method:
                                st.session_state.chunk_exploration_method = selected_method
                                st.session_state.chunk_exploration_current_index = 0
                                st.session_state.chunk_exploration_chunks = []
                                st.session_state.chunk_exploration_level = None  # Reset level filter
                                st.session_state.chunk_exploration_embedding_model = None  # Reset embedding model
                                st.session_state.chunk_exploration_similarity_results = None  # Clear similarity results
                            
                            # Embedding Model Selection (for similarity computation)
                            if selected_method:
                                # Get available embedding models for this file+method combination
                                available_embedding_models = get_embedding_models_for_file_and_method(
                                    st.session_state.legal_qdrant_client,
                                    st.session_state.legal_collection_name,
                                    selected_file,
                                    selected_method
                                )
                                
                                if available_embedding_models:
                                    st.markdown("### 🔤 Embedding Model")
                                    # Initialize embedding model in session state
                                    if 'chunk_exploration_embedding_model' not in st.session_state:
                                        st.session_state.chunk_exploration_embedding_model = available_embedding_models[0]
                                    
                                    # Ensure current selection is valid
                                    current_embedding_model = st.session_state.chunk_exploration_embedding_model
                                    if current_embedding_model not in available_embedding_models:
                                        current_embedding_model = available_embedding_models[0]
                                        st.session_state.chunk_exploration_embedding_model = current_embedding_model
                                    
                                    selected_embedding_model = st.selectbox(
                                        "Select embedding model:",
                                        options=available_embedding_models,
                                        key="chunk_exploration_embedding_model_selector",
                                        help="Select embedding model for similarity computation",
                                        index=available_embedding_models.index(current_embedding_model) if current_embedding_model in available_embedding_models else 0
                                    )
                                    
                                    # Update session state and clear similarity results if model changed
                                    if selected_embedding_model != st.session_state.chunk_exploration_embedding_model:
                                        st.session_state.chunk_exploration_embedding_model = selected_embedding_model
                                        st.session_state.chunk_exploration_similarity_results = None  # Clear similarity results
                                    else:
                                        st.session_state.chunk_exploration_embedding_model = selected_embedding_model
                                else:
                                    st.warning("⚠️ No embedding models found for this file+method combination.")
                                    st.session_state.chunk_exploration_embedding_model = None
                            
                            # Chunk Level Switcher (only for structural chunking)
                            if selected_method:
                                if selected_method == "structural":
                                    st.markdown("### 📊 Chunk Level")
                                    level_options = {
                                        "All Levels": None,
                                        "Level 1 - Sections": 1,
                                        "Level 2 - Subclauses": 2,
                                        "Level 3 - Semantic Units": 3
                                    }
                                    
                                    level_labels = list(level_options.keys())
                                    current_level_idx = 0
                                    if st.session_state.chunk_exploration_level in level_options.values():
                                        current_level_idx = list(level_options.values()).index(st.session_state.chunk_exploration_level)
                                    
                                    selected_level_label = st.selectbox(
                                        "View Chunking Level:",
                                        options=level_labels,
                                        key="chunk_exploration_level_selector",
                                        help="Filter chunks by structural level (Level 1 = top sections, Level 2 = subclauses, Level 3 = semantic units)",
                                        index=current_level_idx
                                    )
                                    
                                    selected_level = level_options[selected_level_label]
                                    
                                    # Update session state if level changed
                                    if selected_level != st.session_state.chunk_exploration_level:
                                        st.session_state.chunk_exploration_level = selected_level
                                        st.session_state.chunk_exploration_current_index = 0
                                        st.session_state.chunk_exploration_chunks = []
                                    
                                    # Use selected_level for current operations
                                    current_level = selected_level
                                else:
                                    # Non-structural methods: no level filtering
                                    st.session_state.chunk_exploration_level = None
                                    current_level = None
                                
                                # Load chunks if not already loaded or if file/method/level changed
                                if (not st.session_state.chunk_exploration_chunks or 
                                    st.session_state.chunk_exploration_file != selected_file or
                                    st.session_state.chunk_exploration_method != selected_method or
                                    (selected_method == "structural" and st.session_state.chunk_exploration_level != current_level)):
                                    
                                    with st.spinner(f"Loading chunks..."):
                                        try:
                                            chunks = get_chunks_for_exploration(
                                                qdrant_client=st.session_state.legal_qdrant_client,
                                                collection_name=st.session_state.legal_collection_name,
                                                source_file=selected_file,
                                                chunking_method=selected_method,
                                                chunk_level=current_level,
                                                limit=10000
                                            )
                                            st.session_state.chunk_exploration_chunks = chunks
                                            st.session_state.chunk_exploration_current_index = 0
                                        except Exception as e:
                                            st.error(f"❌ Error loading chunks: {e}")
                                            st.session_state.chunk_exploration_chunks = []
                                
                                chunks = st.session_state.chunk_exploration_chunks
                                
                                if chunks:
                                    # Document metadata summary
                                    st.markdown("---")
                                    st.markdown("### 📊 Summary")
                                    
                                    total_chunks = len(chunks)
                                    total_chars = sum(chunk.get("char_count", len(chunk.get("text", ""))) for chunk in chunks)
                                    avg_chunk_size = total_chars / total_chunks if total_chunks > 0 else 0
                                    pages = set(chunk.get("page") for chunk in chunks if chunk.get("page") is not None)
                                    
                                    st.metric("Total Chunks", total_chunks)
                                    st.metric("Total Characters", f"{total_chars:,}")
                                    st.metric("Avg Size", f"{avg_chunk_size:.0f} chars")
                                    st.metric("Pages", len(pages) if pages else "N/A")
                                    
                                    # Export button
                                    st.markdown("---")
                                    export_data = []
                                    for chunk in chunks:
                                        export_data.append({
                                            "chunk_number": chunk.get("chunk_number", ""),
                                            "chunk_id": chunk.get("id", ""),
                                            "text": chunk.get("text", "")[:500],
                                            "char_count": chunk.get("char_count", len(chunk.get("text", ""))),
                                            "chunking_method": chunk.get("chunking_method", ""),
                                            "page": chunk.get("page", ""),
                                            "hierarchy_level": chunk.get("hierarchy_level", ""),
                                            "clause_number": chunk.get("clause_number", ""),
                                            "title": chunk.get("title", ""),
                                            "parent": chunk.get("parent", ""),
                                            "source_file": chunk.get("source_file", "")
                                        })
                                    
                                    if pd is not None:
                                        df_export = pd.DataFrame(export_data)
                                        csv = df_export.to_csv(index=False)
                                    else:
                                        # Fallback: create CSV manually if pandas not available
                                        import csv as csv_module
                                        import io
                                        output = io.StringIO()
                                        if export_data:
                                            writer = csv_module.DictWriter(output, fieldnames=export_data[0].keys())
                                            writer.writeheader()
                                            writer.writerows(export_data)
                                        csv = output.getvalue()
                                    
                                    st.download_button(
                                        label="📥 Export CSV",
                                        data=csv,
                                        file_name=f"{selected_file}_{selected_method}_chunks.csv",
                                        mime="text/csv",
                                        key="download_chunks_csv_left",
                                        use_container_width=True
                                    )
                                
                                # Similarity Query Section
                                if selected_method and st.session_state.chunk_exploration_embedding_model:
                                    st.markdown("---")
                                    st.markdown("### 🔍 Similarity Search")
                                    st.caption("Compute cosine similarity between a query and chunks using the selected embedding model")
                                    
                                    similarity_query = st.text_input(
                                        "Enter query for similarity computation:",
                                        key="chunk_exploration_similarity_query",
                                        placeholder="e.g., Bee Associates Ltd",
                                        help="Enter text to find most similar chunks"
                                    )
                                    
                                    top_n_similar = st.slider(
                                        "Number of top results:",
                                        min_value=1,
                                        max_value=50,
                                        value=10,
                                        key="chunk_exploration_top_n",
                                        help="Number of most similar chunks to display"
                                    )
                                    
                                    compute_similarity_button = st.button(
                                        "🔍 Compute Similarity",
                                        key="chunk_exploration_compute_similarity",
                                        type="primary",
                                        use_container_width=True
                                    )
                                    
                                    # Initialize similarity results in session state
                                    if 'chunk_exploration_similarity_results' not in st.session_state:
                                        st.session_state.chunk_exploration_similarity_results = None
                                    
                                    # Store the current query in a separate session state variable
                                    if 'current_similarity_query' not in st.session_state:
                                        st.session_state.current_similarity_query = ""
                                    
                                    # Compute similarity if button clicked
                                    if compute_similarity_button:
                                        if not similarity_query:
                                            st.warning("⚠️ Please enter a query")
                                        else:
                                            with st.spinner(f"Computing similarity for '{similarity_query}'..."):
                                                try:
                                                    similarity_results = compute_chunk_similarities(
                                                        qdrant_client=st.session_state.legal_qdrant_client,
                                                        collection_name=st.session_state.legal_collection_name,
                                                        query_text=similarity_query,
                                                        embedding_model=st.session_state.chunk_exploration_embedding_model,
                                                        source_file=selected_file,
                                                        chunking_method=selected_method,
                                                        top_n=top_n_similar
                                                    )
                                                    # Store results and current query in session state
                                                    st.session_state.chunk_exploration_similarity_results = similarity_results
                                                    st.session_state.current_similarity_query = similarity_query
                                                    st.success(f"✅ Found {len(similarity_results)} most similar chunks")
                                                except Exception as e:
                                                    st.error(f"❌ Error computing similarity: {e}")
                                                    logger.error(f"Similarity computation error: {e}", exc_info=True)
                                                    st.session_state.chunk_exploration_similarity_results = None
                                                    st.session_state.current_similarity_query = ""
                                    
                                    # Display similarity results if available
                                    if st.session_state.chunk_exploration_similarity_results:
                                        st.markdown("---")
                                        st.markdown("### 📊 Similarity Results")
                                        st.caption(f"Query: '{st.session_state.get('current_similarity_query', 'N/A')}' | Model: {st.session_state.chunk_exploration_embedding_model}")
                                        
                                        similarity_results = st.session_state.chunk_exploration_similarity_results
                                        
                                        for idx, result in enumerate(similarity_results, 1):
                                            similarity_score = result.get("similarity_score", 0.0)
                                            chunk_num = result.get("chunk_number", result.get("chunk_index", idx - 1))
                                            chunk_num_display = chunk_num + 1 if chunk_num is not None else idx
                                            
                                            with st.expander(
                                                f"#{idx} - Chunk {chunk_num_display} | Similarity: {similarity_score:.4f}",
                                                expanded=(idx == 1)  # Expand first result
                                            ):
                                                # Similarity interpretation
                                                if similarity_score >= 0.9:
                                                    similarity_label = "Very High Similarity"
                                                    similarity_color = "🟢"
                                                elif similarity_score >= 0.7:
                                                    similarity_label = "High Similarity"
                                                    similarity_color = "🟡"
                                                elif similarity_score >= 0.5:
                                                    similarity_label = "Moderate Similarity"
                                                    similarity_color = "🟠"
                                                elif similarity_score >= 0.3:
                                                    similarity_label = "Low Similarity"
                                                    similarity_color = "🔴"
                                                else:
                                                    similarity_label = "Very Low Similarity"
                                                    similarity_color = "⚫"
                                                
                                                st.markdown(f"**{similarity_color} {similarity_label}** (Cosine: {similarity_score:.4f})")
                                                
                                                # Chunk text
                                                st.markdown("#### 📄 Chunk Text")
                                                chunk_text = result.get("text", "No text content")
                                                st.text_area(
                                                    "Content:",
                                                    value=chunk_text,
                                                    height=200,
                                                    key=f"similarity_chunk_text_{result.get('id')}",
                                                    label_visibility="collapsed"
                                                )
                                                
                                                # Metadata
                                                st.markdown("#### 📋 Metadata")
                                                meta_col1, meta_col2 = st.columns(2)
                                                with meta_col1:
                                                    st.caption(f"**Chunk Number:** {chunk_num_display}")
                                                    st.caption(f"**Chunk ID:** `{result.get('id', 'N/A')}`")
                                                    st.caption(f"**Page:** {result.get('page', 'N/A')}")
                                                    st.caption(f"**Character Count:** {result.get('char_count', len(chunk_text))}")
                                                with meta_col2:
                                                    st.caption(f"**Chunk Level:** {result.get('chunk_level', 'N/A')}")
                                                    st.caption(f"**Clause Number:** {result.get('clause_number', 'N/A')}")
                                                    st.caption(f"**Title:** {result.get('title', 'N/A')}")
                                                
                                                # Jump to chunk button
                                                if st.button(
                                                    f"📍 Jump to Chunk {chunk_num_display}",
                                                    key=f"jump_to_similar_chunk_{idx}",
                                                    use_container_width=True
                                                ):
                                                    # Find the chunk index in the main chunks list
                                                    chunks = st.session_state.chunk_exploration_chunks
                                                    chunk_id_to_find = result.get('id')
                                                    for i, chunk in enumerate(chunks):
                                                        if str(chunk.get('id')) == str(chunk_id_to_find):
                                                            st.session_state.chunk_exploration_current_index = i
                                                            st.rerun()
                                                            break
                
                with right_panel:
                    if (st.session_state.chunk_exploration_file and 
                        st.session_state.chunk_exploration_method and 
                        st.session_state.chunk_exploration_chunks):
                        
                        chunks = st.session_state.chunk_exploration_chunks
                        total_chunks = len(chunks)
                        current_index = st.session_state.chunk_exploration_current_index
                        
                        # Ensure index is within bounds
                        if current_index >= total_chunks:
                            current_index = total_chunks - 1
                        if current_index < 0:
                            current_index = 0
                        st.session_state.chunk_exploration_current_index = current_index
                        
                        if total_chunks > 0:
                            current_chunk = chunks[current_index]
                            # chunk_number is 0-based, display as 1-based
                            chunk_num_raw = current_chunk.get("chunk_number", current_index)
                            chunk_num = chunk_num_raw + 1 if chunk_num_raw is not None else current_index + 1
                            
                            # Navigation controls
                            st.markdown("### 🧭 Navigation")
                            
                            # Chunk counter display (prominent)
                            st.markdown(f"### Chunk {chunk_num} / {total_chunks}")
                            
                            # Navigation buttons and slider
                            nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
                            
                            with nav_col1:
                                prev_disabled = (current_index == 0)
                                if st.button("← Previous", key="prev_chunk", disabled=prev_disabled, use_container_width=True):
                                    st.session_state.chunk_exploration_current_index = max(0, current_index - 1)
                                    st.rerun()
                                if prev_disabled:
                                    st.caption("_First chunk_")
                            
                            with nav_col2:
                                # Slider for jumping to any chunk
                                st.markdown(f"**Jump to chunk:** (1-{total_chunks})")
                                new_index = st.slider(
                                    "Chunk number",
                                    min_value=1,
                                    max_value=total_chunks,
                                    value=current_index + 1,
                                    key="chunk_slider",
                                    help=f"Drag to jump to any chunk",
                                    label_visibility="collapsed"
                                )
                                if new_index - 1 != current_index:
                                    st.session_state.chunk_exploration_current_index = new_index - 1
                                    st.rerun()
                            
                            with nav_col3:
                                next_disabled = (current_index >= total_chunks - 1)
                                if st.button("Next →", key="next_chunk", disabled=next_disabled, use_container_width=True):
                                    st.session_state.chunk_exploration_current_index = min(total_chunks - 1, current_index + 1)
                                    st.rerun()
                                if next_disabled:
                                    st.caption("_Last chunk_")
                            
                            # Quick jump input
                            jump_col1, jump_col2 = st.columns([2, 1])
                            with jump_col1:
                                jump_to = st.number_input(
                                    "Jump to chunk number:",
                                    min_value=1,
                                    max_value=total_chunks,
                                    value=current_index + 1,
                                    key="chunk_jump_input",
                                    help="Enter chunk number to jump to"
                                )
                            with jump_col2:
                                st.markdown("<br>", unsafe_allow_html=True)  # Spacing
                                if st.button("Go", key="chunk_jump_button", use_container_width=True):
                                    if 1 <= jump_to <= total_chunks:
                                        st.session_state.chunk_exploration_current_index = jump_to - 1
                                        st.rerun()
                            
                            # Keyboard shortcuts info
                            with st.expander("⌨️ Keyboard Navigation Tips"):
                                st.markdown("""
                                **Navigation Methods:**
                                1. **← Previous / Next →** buttons: Click to navigate
                                2. **Slider**: Drag to jump to any chunk (1-{total_chunks})
                                3. **Jump input**: Enter chunk number and click "Go"
                                4. **Arrow Keys**: Use ↑/↓ or ←/→ keys when focused on the page
                                
                                **Tip:** The slider provides the fastest way to navigate through chunks!
                                """.format(total_chunks=total_chunks))
                            
                            st.markdown("---")
                            
                            # Current chunk display
                            st.markdown("### 📄 Current Chunk")
                            
                            # Chunk text
                            chunk_text = current_chunk.get("text", "No text content")
                            st.markdown("#### 📝 Chunk Text")
                            st.text_area(
                                "Content:",
                                value=chunk_text,
                                height=300,
                                key=f"chunk_text_display_{current_chunk.get('id')}",
                                label_visibility="collapsed",
                                help="Chunk text content"
                            )
                            
                            st.markdown("---")
                            
                            # Metadata display
                            st.markdown("#### 📋 Metadata")
                            meta_col1, meta_col2 = st.columns(2)
                            
                            with meta_col1:
                                st.markdown("**Core Information:**")
                                st.caption(f"**Chunk Number:** {chunk_num} (0-based index: {chunk_num_raw})")
                                st.caption(f"**Chunk ID:** `{current_chunk.get('id', 'N/A')}`")
                                chunk_level = current_chunk.get('chunk_level', current_chunk.get('hierarchy_level', 'N/A'))
                                st.caption(f"**Chunk Level:** {chunk_level} {'(Top-Level Section)' if chunk_level == 1 else '(Subclause)' if chunk_level == 2 else '(Semantic Unit)' if chunk_level == 3 else ''}")
                                parent_chunk_num = current_chunk.get('parent_chunk_number')
                                if parent_chunk_num is not None:
                                    st.caption(f"**Parent Chunk Number:** {parent_chunk_num + 1} (0-based: {parent_chunk_num})")
                                else:
                                    st.caption(f"**Parent Chunk Number:** None (Top-level)")
                                st.caption(f"**Chunking Method:** {current_chunk.get('chunking_method', 'N/A')}")
                                st.caption(f"**Embedding Model:** {current_chunk.get('embedding_model', 'N/A')}")
                                st.caption(f"**Source File:** {current_chunk.get('source_file', 'N/A')}")
                                st.caption(f"**Character Count:** {current_chunk.get('char_count', len(chunk_text))}")
                            
                            with meta_col2:
                                st.markdown("**Document Structure:**")
                                st.caption(f"**Hierarchy Level:** {current_chunk.get('hierarchy_level', 'N/A')}")
                                st.caption(f"**Clause Number:** {current_chunk.get('clause_number', 'N/A')}")
                                st.caption(f"**Title:** {current_chunk.get('title', 'N/A')}")
                                st.caption(f"**Page:** {current_chunk.get('page', 'N/A')}")
                                parent_clause = current_chunk.get('parent')
                                if parent_clause:
                                    st.caption(f"**Parent Clause:** {parent_clause}")
                                else:
                                    st.caption(f"**Parent Clause:** None")
                                text_preview = current_chunk.get('text_preview', '')
                                if text_preview:
                                    st.caption(f"**Preview:** {text_preview[:100]}...")
                            
                            # NER Entities
                            entities = []
                            if current_chunk.get("individual_name"):
                                entities.append(("👤 Person", current_chunk.get("individual_name")))
                            if current_chunk.get("company_name"):
                                entities.append(("🏢 Company", current_chunk.get("company_name")))
                            if current_chunk.get("email"):
                                entities.append(("📧 Email", current_chunk.get("email")))
                            if current_chunk.get("phone"):
                                entities.append(("📞 Phone", current_chunk.get("phone")))
                            if current_chunk.get("address"):
                                entities.append(("📍 Address", current_chunk.get("address")))
                            
                            if entities:
                                st.markdown("#### 🏷️ Extracted Entities")
                                entity_cols = st.columns(len(entities) if len(entities) <= 5 else 5)
                                for idx, (entity_type, entity_value) in enumerate(entities):
                                    with entity_cols[idx % len(entity_cols)]:
                                        st.caption(f"**{entity_type}:** {entity_value}")
                            
                            # Additional metadata
                            if current_chunk.get("upload_timestamp"):
                                st.caption(f"**Uploaded:** {current_chunk.get('upload_timestamp')}")
                            
                            # Additional metadata expander
                            if current_chunk.get("metadata"):
                                with st.expander("🔍 Additional Metadata"):
                                    st.json(current_chunk.get("metadata"))
                            
                            # JavaScript for keyboard navigation
                            # Note: Keyboard navigation works best when the page is focused
                            st.markdown(f"""
                            <script>
                            (function() {{
                                // Store the current chunk index in a data attribute for JavaScript access
                                const chunkNav = document.createElement('div');
                                chunkNav.id = 'chunk-nav-data';
                                chunkNav.setAttribute('data-current-index', '{current_index}');
                                chunkNav.setAttribute('data-total-chunks', '{total_chunks}');
                                chunkNav.style.display = 'none';
                                document.body.appendChild(chunkNav);
                                
                                // Keyboard event listener
                                function handleKeyPress(e) {{
                                    // Only handle if not typing in an input field
                                    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {{
                                        return;
                                    }}
                                    
                                    const currentIdx = parseInt(chunkNav.getAttribute('data-current-index')) || 0;
                                    const totalChunks = parseInt(chunkNav.getAttribute('data-total-chunks')) || 1;
                                    
                                    if ((e.key === 'ArrowUp' || e.key === 'ArrowLeft') && currentIdx > 0) {{
                                        e.preventDefault();
                                        // Find and click previous button
                                        const buttons = Array.from(document.querySelectorAll('button'));
                                        const prevBtn = buttons.find(btn => btn.textContent.includes('Previous'));
                                        if (prevBtn && !prevBtn.disabled) {{
                                            prevBtn.click();
                                        }}
                                    }} else if ((e.key === 'ArrowDown' || e.key === 'ArrowRight') && currentIdx < totalChunks - 1) {{
                                        e.preventDefault();
                                        // Find and click next button
                                        const buttons = Array.from(document.querySelectorAll('button'));
                                        const nextBtn = buttons.find(btn => btn.textContent.includes('Next'));
                                        if (nextBtn && !nextBtn.disabled) {{
                                            nextBtn.click();
                                        }}
                                    }}
                                }}
                                
                                document.addEventListener('keydown', handleKeyPress);
                            }})();
                            </script>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("No chunks to display.")
                    else:
                        st.info("👈 Select a document and chunking method from the left panel to start exploring chunks.")
                                    
        except Exception as e:
            st.error(f"❌ Error in Chunk Exploration: {e}")
            logger.error(f"Chunk exploration error: {e}", exc_info=True)
            import traceback
            st.code(traceback.format_exc())

elif st.session_state.main_mode == "🧪 Test Embedding Model":
    # Test Embedding Model Mode
    st.subheader("🧪 Test Embedding Model")
    st.markdown("Test the quality of embedding models using word pairs from a CSV file.")
    
    # Model selection
    st.markdown("### 💻 Model Selection")
    
    # Check if EMBEDDING_MODELS is available
    if not EMBEDDING_MODELS:
        st.warning("⚠️ No embedding models found. Please ensure the legal chunker is properly configured.")
        st.stop()
    
    # Get available embedding models - handle both string and dict formats
    available_models = []
    if EMBEDDING_MODELS and isinstance(EMBEDDING_MODELS[0], dict):
        available_models = [model.get('name', '') for model in EMBEDDING_MODELS if isinstance(model, dict) and 'name' in model]
    else:
        available_models = [str(model) for model in EMBEDDING_MODELS if isinstance(model, str)]
    
    # Add Ollama models if available
    if 'ollama_models' in st.session_state and st.session_state.ollama_models:
        ollama_models = [f"ollama/{model}" for model in st.session_state.ollama_models if model not in available_models]
        available_models.extend(ollama_models)
    
    # Initialize test_embedding_model in session state if not set
    if 'test_embedding_model' not in st.session_state:
        st.session_state.test_embedding_model = available_models[0] if available_models else ""
    
    # Select model
    selected_embedding_model = st.selectbox(
        "Select Embedding Model",
        options=available_models,
        index=available_models.index(st.session_state.test_embedding_model) 
            if st.session_state.test_embedding_model in available_models 
            else 0,
        key="test_embedding_model_select",
        help="Select the embedding model to test"
    )
    
    # Update session state
    st.session_state.test_embedding_model = selected_embedding_model
    
    # Dataset info
    st.markdown("### 📂 Dataset")
    st.info(f"Using word pairs from: `data/small_data/word_pairs.csv`")
    
    # Show sample data
    try:
        import pandas as pd
        dataset_path = "data/small_data/word_pairs.csv"
        if os.path.exists(dataset_path):
            sample_df = pd.read_csv(dataset_path).head(5)
            st.dataframe(sample_df, hide_index=True)
        else:
            st.warning("⚠️ Word pairs file not found. Please ensure the file exists at the specified path.")
    except Exception as e:
        st.warning(f"⚠️ Could not load sample data: {e}")
    
    # Cache toggle and test button
    cache_col1, cache_col2 = st.columns([1, 4])
    with cache_col1:
        use_cache = st.checkbox("Use Cached Results", value=True, 
                              help="Use previously computed results to save time")
    
    # Test Button
    test_col1, test_col2 = st.columns([1, 4])
    with test_col1:
        test_embedding_button = st.button("🧪 Test Embedding Model", key="test_embedding_button", type="primary", use_container_width=True)
    
    with test_col2:
        if test_embedding_button:
            with st.spinner("Running embedding test..."):
                try:
                    # Run the test with cache preference
                    results = test_embedding_quality(selected_embedding_model, use_cache=use_cache)
                    
                    # Show cache status
                    if results.get('cached', False):
                        st.info("ℹ️ Using cached results. Uncheck 'Use Cached Results' to force recomputation.")
                    
                    # Display results
                    st.success("✅ Embedding test completed successfully!")
                    
                    # Show metrics
                    st.markdown("### 📊 Test Results")
                    
                    # Get confusion matrix values
                    cm = results['confusion_matrix']
                    TN, FP = cm[0][0], cm[0][1]
                    FN, TP = cm[1][0], cm[1][1]
                    
                    # Create a metrics table with confusion matrix values
                    metrics = results['metrics']
                    metrics_data = {
                        'Metric': [
                            'Optimal Threshold', 
                            'F1 Score', 
                            'Accuracy', 
                            'Precision', 
                            'Recall', 
                            'ROC AUC', 
                            'PR AUC',
                            'True Negatives (TN)',
                            'False Positives (FP)',
                            'False Negatives (FN)',
                            'True Positives (TP)'
                        ],
                        'Value': [
                            f"{metrics['optimal_threshold']:.4f}",
                            f"{metrics['f1_score']:.4f}",
                            f"{metrics['accuracy']:.4f}",
                            f"{metrics['precision']:.4f}",
                            f"{metrics['recall']:.4f}",
                            f"{metrics['roc_auc']:.4f}",
                            f"{metrics['pr_auc']:.4f}",
                            f"{TN}",
                            f"{FP}",
                            f"{FN}",
                            f"{TP}"
                        ]
                    }
                    
                    # Add color coding for metrics
                    def color_metric(row):
                        # Check if the row contains a metric name (first column)
                        if row.iloc[0] in ['F1 Score', 'Accuracy', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']:
                            return ['background-color: #f0f8ff'] * len(row)  # Light blue for metric names
                        return [''] * len(row)
                    
                    # Create and style the metrics DataFrame
                    import matplotlib.pyplot as plt  # Import matplotlib here to avoid circular imports
                    import seaborn as sns  # Import seaborn for visualization
                    metrics_df = pd.DataFrame(metrics_data)
                    styled_metrics = metrics_df.style.apply(color_metric, axis=1)
                    
                    # Display the styled DataFrame with enough height to show all rows
                    st.dataframe(
                        styled_metrics,
                        hide_index=True,
                        width='stretch',  # Use full width
                        height=(len(metrics_data['Metric']) + 1) * 35 + 3  # Calculate height based on number of rows
                    )
                    
                    # Show the figure from results (contains both histograms)
                    st.markdown("### 📊 Similarity Analysis")
                    st.pyplot(results['figure'])
                    plt.close(results['figure'])
                    
                    # Confusion matrix values are now shown in the metrics table above
                    # Show sample predictions
                    st.markdown("### 📊 Sample Predictions")
                    sample_data = results['data'].copy()
                    sample_data['prediction'] = (sample_data['similarity'] >= metrics['optimal_threshold']).astype(int)
                    sample_data['correct'] = (sample_data['label'] == sample_data['prediction']).map({True: '✅', False: '❌'})
                    sample_data['similarity'] = sample_data['similarity'].round(4)
                    
                    # Reorder columns for better display
                    sample_data = sample_data[['word1', 'word2', 'similar', 'similarity', 'prediction', 'correct']]
                    
                    # Add color coding for correct/incorrect predictions
                    def highlight_correct(row):
                        if row['correct'] == '✅':
                            return ['background-color: #e6ffe6'] * len(row)  # Light green for correct
                        else:
                            return ['background-color: #ffe6e6'] * len(row)  # Light red for incorrect
                    
                    # Display the table with styling
                    st.dataframe(
                        sample_data.style.apply(highlight_correct, axis=1),
                        column_config={
                            'word1': 'Word 1',
                            'word2': 'Word 2',
                            'similar': 'Similar (Ground Truth)',
                            'similarity': 'Cosine Similarity',
                            'prediction': 'Predicted',
                            'correct': 'Correct?'
                        },
                        hide_index=True,
                        use_container_width=True,
                        height=min(400, len(sample_data) * 35 + 3)  # Dynamic height based on number of rows
                    )
                    
                    # Add interpretation of results
                    st.markdown("### 📝 Interpretation")
                    
                    if metrics['f1_score'] > 0.8:
                        st.success(f"✅ **Excellent performance**: The model shows strong ability to distinguish between similar and dissimilar word pairs (F1 Score: {metrics['f1_score']:.3f})")
                    elif metrics['f1_score'] > 0.6:
                        st.warning(f"⚠️ **Moderate performance**: The model shows some ability to distinguish between similar and dissimilar word pairs (F1 Score: {metrics['f1_score']:.3f})")
                    else:
                        st.error(f"❌ **Poor performance**: The model struggles to distinguish between similar and dissimilar word pairs (F1 Score: {metrics['f1_score']:.3f})")
                    
                    st.markdown(f"- **Optimal Similarity Threshold**: {metrics['optimal_threshold']:.4f} (automatically determined to maximize F1 score)")
                    st.markdown(f"- **ROC AUC**: {metrics['roc_auc']:.4f} (closer to 1.0 is better)")
                    st.markdown(f"- **Precision-Recall AUC**: {metrics['pr_auc']:.4f} (better for imbalanced datasets)")
                    
                except FileNotFoundError as e:
                    st.error(f"❌ Error: {e}. Please ensure the word pairs file exists at 'data/small_data/word_pairs.csv'")
                except Exception as e:
                    st.error(f"❌ Error running embedding test: {e}")
                    logger.error(f"Embedding test error: {e}", exc_info=True)
        else:
            st.info("👈 Click the '🧪 Test Embedding Model' button to evaluate the selected model")

elif st.session_state.main_mode == "🧪 Test Embedding Model":
    # Test Embedding Model Mode
    st.subheader("🧪 Test Embedding Model")
    st.markdown("Test the quality of embedding models by checking similarity between related and unrelated words.")
    
    # Initialize Qdrant client for legal documents (cached resource)
    @st.cache_resource
    def get_qdrant_client():
        """Get or create Qdrant client (cached resource)."""
        try:
            from qdrant_client import QdrantClient
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            client = QdrantClient(url=qdrant_url)
            logger.info("Legal Qdrant client initialized")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            return None
    
    # Initialize Qdrant client in session state
    if 'legal_qdrant_client' not in st.session_state:
        st.session_state.legal_qdrant_client = get_qdrant_client()
        st.session_state.legal_collection_name = "legal_documents"
    
    # Embedding Model Selection
    st.markdown("### 🔤 Select Embedding Model")
    if EMBEDDING_MODELS:
        # Initialize embedding model in session state if not set
        if 'test_embedding_model' not in st.session_state:
            st.session_state.test_embedding_model = EMBEDDING_MODELS[0]
        
        selected_embedding_model = st.selectbox(
            "Choose an embedding model to test:",
            options=EMBEDDING_MODELS,
            index=EMBEDDING_MODELS.index(st.session_state.test_embedding_model) if st.session_state.test_embedding_model in EMBEDDING_MODELS else 0,
            key="test_embedding_model_select",
            help="Select the embedding model you want to test"
        )
        
        # Update session state
        st.session_state.test_embedding_model = selected_embedding_model
    else:
        st.error("⚠️ No embedding models available")
        st.stop()
    
    st.markdown("---")
    
    # Test Button
    test_col1, test_col2 = st.columns([1, 4])
    with test_col1:
        test_embedding_button = st.button("🧪 Test Embedding Model", key="test_embedding_button", type="primary", use_container_width=True)
    
    with test_col2:
        if test_embedding_button:
            if st.session_state.legal_qdrant_client is None:
                st.error("⚠️ Qdrant client not available. Please ensure Qdrant is running.")
            elif not selected_embedding_model:
                st.error("⚠️ No embedding model selected. Please select an embedding model above.")
            else:
                # Run embedding test
                with st.spinner("Running embedding quality test..."):
                    try:
                        test_results = test_embedding_quality(
                            qdrant_client=st.session_state.legal_qdrant_client,
                            embedding_model=selected_embedding_model,
                            collection_name=st.session_state.legal_collection_name
                        )
                        
                        # Display results
                        st.markdown("#### 📊 Test Results")
                        
                        # Confusion Matrix
                        st.markdown("**Confusion Matrix:**")
                        cm_data = {
                            "Metric": ["True Positive", "True Negative", "False Positive", "False Negative"],
                            "Count": [
                                test_results["confusion_matrix"]["TP"],
                                test_results["confusion_matrix"]["TN"],
                                test_results["confusion_matrix"]["FP"],
                                test_results["confusion_matrix"]["FN"]
                            ]
                        }
                        if pd is not None:
                            cm_df = pd.DataFrame(cm_data)
                            st.dataframe(cm_df, use_container_width=True, hide_index=True)
                        else:
                            st.markdown("| Metric | Count |")
                            st.markdown("|--------|-------|")
                            for metric, count in zip(cm_data["Metric"], cm_data["Count"]):
                                st.markdown(f"| {metric} | {count} |")
                        
                        # Similarity Scores
                        st.markdown("**Average Similarity Scores:**")
                        sim_col1, sim_col2, sim_col3 = st.columns(3)
                        with sim_col1:
                            st.metric("Positive Pairs (Similar)", f"{test_results['avg_positive_similarity']:.4f}")
                        with sim_col2:
                            st.metric("Negative Pairs (Dissimilar)", f"{test_results['avg_negative_similarity']:.4f}")
                        with sim_col3:
                            st.metric("Exact Matches", f"{test_results['avg_exact_match_similarity']:.4f}")
                        
                        # Exact Match Results
                        st.markdown("**Exact Word Retrieval Test:**")
                        st.caption("Testing retrieval of exact same words (should have similarity ≈ 1.0)")
                        
                        exact_match_data = []
                        for result in test_results["exact_match_results"]:
                            status = "✅ Found" if result["found"] else "❌ Not Found"
                            rank_info = f"Rank {result['rank']}" if result["rank"] else "N/A"
                            exact_match_data.append({
                                "Word": result["word"],
                                "Status": status,
                                "Similarity Score": f"{result['similarity_score']:.4f}",
                                "Rank": rank_info
                            })
                        
                        if pd is not None:
                            exact_df = pd.DataFrame(exact_match_data)
                            st.dataframe(exact_df, use_container_width=True, hide_index=True)
                        else:
                            st.markdown("| Word | Status | Similarity Score | Rank |")
                            st.markdown("|------|--------|------------------|------|")
                            for item in exact_match_data:
                                st.markdown(f"| {item['Word']} | {item['Status']} | {item['Similarity Score']} | {item['Rank']} |")
                        
                        # Accuracy
                        accuracy = test_results["accuracy"]
                        st.metric("Overall Accuracy", f"{accuracy:.2%}")
                        
                        # Detailed Results
                        with st.expander("📋 Detailed Test Results", expanded=False):
                            st.markdown("**Positive Pairs (Should be similar):**")
                            for pair, similarity in test_results["positive_pairs"]:
                                st.caption(f"`{pair[0]}` ↔ `{pair[1]}`: {similarity:.4f}")
                            
                            st.markdown("**Negative Pairs (Should be dissimilar):**")
                            for pair, similarity in test_results["negative_pairs"]:
                                st.caption(f"`{pair[0]}` ↔ `{pair[1]}`: {similarity:.4f}")
                        
                        # Interpretation
                        if accuracy >= 0.8:
                            st.success(f"✅ Embedding model quality is good (Accuracy: {accuracy:.2%})")
                        elif accuracy >= 0.6:
                            st.warning(f"⚠️ Embedding model quality is moderate (Accuracy: {accuracy:.2%})")
                        else:
                            st.error(f"❌ Embedding model quality is poor (Accuracy: {accuracy:.2%})")
                    except Exception as e:
                        st.error(f"❌ Error running embedding test: {e}")
                        logger.error(f"Embedding test error: {e}", exc_info=True)
        else:
            st.info("👈 Click the '🧪 Test Embedding Model' button to run the test")
        
        # WordSim353 Dataset Evaluation
        st.markdown("---")
        st.markdown("### 📊 WordSim353 Dataset Evaluation")
        st.markdown("Evaluate embedding model performance using the WordSim353 dataset with ROC curves and precision-recall analysis.")
        
        wordsim_button_col1, wordsim_button_col2 = st.columns([1, 4])
        with wordsim_button_col1:
            run_wordsim_eval = st.button("📊 Run WordSim353 Evaluation", key="wordsim_eval_button", type="primary", use_container_width=True)
        
        with wordsim_button_col2:
            if run_wordsim_eval:
                if not selected_embedding_model:
                    st.error("⚠️ No embedding model selected. Please select an embedding model above.")
                else:
                    try:
                        import numpy as np
                        from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
                        import seaborn as sns
                        import matplotlib.pyplot as plt
                        from sklearn.metrics.pairwise import cosine_similarity
                        
                        # Load WordSim353 dataset
                        dataset_path = "data/wordsim353crowd/wordsim353crowd.csv"
                        if not os.path.exists(dataset_path):
                            st.error(f"⚠️ Dataset not found at: {dataset_path}")
                        else:
                            with st.spinner("Loading WordSim353 dataset and computing embeddings..."):
                                # Load CSV
                                if pd is not None:
                                    df = pd.read_csv(dataset_path)
                                else:
                                    st.error("⚠️ Pandas is required for WordSim353 evaluation. Please install: pip install pandas")
                                    st.stop()
                                
                                # Derive label from Human (Mean) - use fixed threshold of 6.0
                                # Score ≥ 6.0 → similar (label = 1)
                                # Score < 6.0 → not similar (label = 0)
                                threshold = 6.0
                                df['label'] = (df['Human (Mean)'] >= threshold).astype(int)
                                
                                st.info(f"📊 Dataset loaded: {len(df)} word pairs. Using threshold {threshold} for labels (≥{threshold} = similar, <{threshold} = not similar).")
                                
                                # Compute embeddings for all words
                                unique_words = set(df['Word 1'].tolist() + df['Word 2'].tolist())
                                word_embeddings = {}
                                
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # Generate embeddings based on model type
                                if selected_embedding_model.startswith("ollama/"):
                                    try:
                                        from langchain_ollama import OllamaEmbeddings
                                        import os
                                        
                                        ollama_model = selected_embedding_model.replace("ollama/", "")
                                        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                                        
                                        embeddings = OllamaEmbeddings(
                                            model=ollama_model,
                                            base_url=base_url
                                        )
                                        
                                        for idx, word in enumerate(unique_words):
                                            if idx % 10 == 0:
                                                progress_bar.progress((idx + 1) / len(unique_words))
                                                status_text.info(f"Computing embeddings: {idx + 1}/{len(unique_words)} words...")
                                            word_embeddings[word] = np.array(embeddings.embed_query(word))
                                    except ImportError:
                                        st.error("⚠️ langchain-ollama required for Ollama embeddings")
                                        st.stop()
                                else:
                                    # Use Ollama for embeddings
                                    from scripts.utils.ollama_utils import get_embeddings_batch
                                    
                                    # Batch embed all words at once for efficiency
                                    words_list = list(unique_words)
                                    embeddings = get_embeddings_batch(words_list, selected_embedding_model)
                                    progress_bar.progress(0.3)
                                    status_text.info("Computing embeddings for all words...")
                                    embeddings_batch = model.encode(words_list, convert_to_numpy=True, show_progress_bar=False)
                                    
                                    for word, embedding in zip(words_list, embeddings_batch):
                                        word_embeddings[word] = np.array(embedding)
                                
                                progress_bar.progress(1.0)
                                status_text.empty()
                                progress_bar.empty()
                                
                                # Compute cosine similarity for each pair
                                cosine_similarities = []
                                for _, row in df.iterrows():
                                    word1_emb = word_embeddings[row['Word 1']]
                                    word2_emb = word_embeddings[row['Word 2']]
                                    
                                    # Compute cosine similarity
                                    similarity = np.dot(word1_emb, word2_emb) / (
                                        np.linalg.norm(word1_emb) * np.linalg.norm(word2_emb)
                                    )
                                    cosine_similarities.append(float(similarity))
                                
                                df['cosine_similarity'] = cosine_similarities
                                
                                # Calculate statistics
                                mean_sim_label1 = df[df['label'] == 1]['cosine_similarity'].mean()
                                mean_sim_label0 = df[df['label'] == 0]['cosine_similarity'].mean()
                                separation_ratio = mean_sim_label1 - mean_sim_label0
                                
                                # Display statistics
                                st.markdown("#### 📈 Evaluation Statistics")
                                stats_col1, stats_col2, stats_col3 = st.columns(3)
                                with stats_col1:
                                    st.metric("Mean Similarity (label=1)", f"{mean_sim_label1:.4f}")
                                with stats_col2:
                                    st.metric("Mean Similarity (label=0)", f"{mean_sim_label0:.4f}")
                                with stats_col3:
                                    st.metric("Separation Ratio", f"{separation_ratio:.4f}")
                                
                                # ROC Curve 1: Human (Mean) vs label
                                st.markdown("#### 📊 ROC Curves")
                                
                                fig_roc1, ax_roc1 = plt.subplots(figsize=(8, 6))
                                fpr_human, tpr_human, _ = roc_curve(df['label'], df['Human (Mean)'])
                                auc_human = auc(fpr_human, tpr_human)
                                ax_roc1.plot(fpr_human, tpr_human, label=f'Human (Mean) (AUC = {auc_human:.3f})', linewidth=2)
                                ax_roc1.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
                                ax_roc1.set_xlabel('False Positive Rate', fontsize=12)
                                ax_roc1.set_ylabel('True Positive Rate', fontsize=12)
                                ax_roc1.set_title('ROC Curve: Human (Mean) vs Label', fontsize=14, fontweight='bold')
                                ax_roc1.legend(loc='lower right', fontsize=10)
                                ax_roc1.grid(True, alpha=0.3)
                                st.pyplot(fig_roc1)
                                plt.close(fig_roc1)
                                
                                # ROC Curve 2: cosine_similarity vs label
                                fig_roc2, ax_roc2 = plt.subplots(figsize=(8, 6))
                                fpr_cosine, tpr_cosine, _ = roc_curve(df['label'], df['cosine_similarity'])
                                auc_cosine = auc(fpr_cosine, tpr_cosine)
                                ax_roc2.plot(fpr_cosine, tpr_cosine, label=f'Cosine Similarity (AUC = {auc_cosine:.3f})', linewidth=2, color='green')
                                ax_roc2.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
                                ax_roc2.set_xlabel('False Positive Rate', fontsize=12)
                                ax_roc2.set_ylabel('True Positive Rate', fontsize=12)
                                ax_roc2.set_title('ROC Curve: Cosine Similarity vs Label', fontsize=14, fontweight='bold')
                                ax_roc2.legend(loc='lower right', fontsize=10)
                                ax_roc2.grid(True, alpha=0.3)
                                st.pyplot(fig_roc2)
                                plt.close(fig_roc2)
                                
                                # Precision-Recall Curve 1: Human (Mean) vs label
                                st.markdown("#### 📊 Precision-Recall Curves")
                                
                                fig_pr1, ax_pr1 = plt.subplots(figsize=(8, 6))
                                precision_human, recall_human, _ = precision_recall_curve(df['label'], df['Human (Mean)'])
                                ap_human = average_precision_score(df['label'], df['Human (Mean)'])
                                ax_pr1.plot(recall_human, precision_human, label=f'Human (Mean) (AP = {ap_human:.3f})', linewidth=2)
                                ax_pr1.set_xlabel('Recall', fontsize=12)
                                ax_pr1.set_ylabel('Precision', fontsize=12)
                                ax_pr1.set_title('Precision-Recall Curve: Human (Mean) vs Label', fontsize=14, fontweight='bold')
                                ax_pr1.legend(loc='lower left', fontsize=10)
                                ax_pr1.grid(True, alpha=0.3)
                                st.pyplot(fig_pr1)
                                plt.close(fig_pr1)
                                
                                # Precision-Recall Curve 2: cosine_similarity vs label
                                fig_pr2, ax_pr2 = plt.subplots(figsize=(8, 6))
                                precision_cosine, recall_cosine, _ = precision_recall_curve(df['label'], df['cosine_similarity'])
                                ap_cosine = average_precision_score(df['label'], df['cosine_similarity'])
                                ax_pr2.plot(recall_cosine, precision_cosine, label=f'Cosine Similarity (AP = {ap_cosine:.3f})', linewidth=2, color='green')
                                ax_pr2.set_xlabel('Recall', fontsize=12)
                                ax_pr2.set_ylabel('Precision', fontsize=12)
                                ax_pr2.set_title('Precision-Recall Curve: Cosine Similarity vs Label', fontsize=14, fontweight='bold')
                                ax_pr2.legend(loc='lower left', fontsize=10)
                                ax_pr2.grid(True, alpha=0.3)
                                st.pyplot(fig_pr2)
                                plt.close(fig_pr2)
                                
                                # Histograms
                                st.markdown("#### 📊 Distribution Histograms")
                                
                                # Histogram 1: Human (Mean)
                                fig_hist1, ax_hist1 = plt.subplots(figsize=(10, 6))
                                sns.histplot(data=df, x='Human (Mean)', hue='label', bins=30, alpha=0.6, palette={1: 'red', 0: 'blue'}, ax=ax_hist1)
                                ax_hist1.set_xlabel('Human (Mean) Score', fontsize=12)
                                ax_hist1.set_ylabel('Frequency', fontsize=12)
                                ax_hist1.set_title('Distribution of Human (Mean) Scores by Label', fontsize=14, fontweight='bold')
                                ax_hist1.legend(title='Label', labels=['Not Similar (0)', 'Similar (1)'], fontsize=10)
                                ax_hist1.grid(True, alpha=0.3)
                                st.pyplot(fig_hist1)
                                plt.close(fig_hist1)
                                
                                # Histogram 2: cosine_similarity
                                fig_hist2, ax_hist2 = plt.subplots(figsize=(10, 6))
                                sns.histplot(data=df, x='cosine_similarity', hue='label', bins=30, alpha=0.6, palette={1: 'red', 0: 'blue'}, ax=ax_hist2)
                                ax_hist2.set_xlabel('Cosine Similarity', fontsize=12)
                                ax_hist2.set_ylabel('Frequency', fontsize=12)
                                ax_hist2.set_title('Distribution of Cosine Similarity Scores by Label', fontsize=14, fontweight='bold')
                                ax_hist2.legend(title='Label', labels=['Not Similar (0)', 'Similar (1)'], fontsize=10)
                                ax_hist2.grid(True, alpha=0.3)
                                st.pyplot(fig_hist2)
                                plt.close(fig_hist2)
                                
                                # Summary metrics
                                st.markdown("#### 📋 Summary Metrics")
                                metrics_data = {
                                    "Metric": [
                                        "Mean Cosine Similarity (label=1)",
                                        "Mean Cosine Similarity (label=0)",
                                        "Separation Ratio",
                                        "ROC AUC (Human Mean)",
                                        "ROC AUC (Cosine Similarity)",
                                        "Average Precision (Human Mean)",
                                        "Average Precision (Cosine Similarity)"
                                    ],
                                    "Value": [
                                        f"{mean_sim_label1:.4f}",
                                        f"{mean_sim_label0:.4f}",
                                        f"{separation_ratio:.4f}",
                                        f"{auc_human:.4f}",
                                        f"{auc_cosine:.4f}",
                                        f"{ap_human:.4f}",
                                        f"{ap_cosine:.4f}"
                                    ]
                                }
                                
                                if pd is not None:
                                    metrics_df = pd.DataFrame(metrics_data)
                                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                                else:
                                    st.markdown("| Metric | Value |")
                                    st.markdown("|--------|-------|")
                                    for metric, value in zip(metrics_data["Metric"], metrics_data["Value"]):
                                        st.markdown(f"| {metric} | {value} |")
                                
                                # Interpretation
                                st.markdown("#### 💡 Interpretation")
                                if separation_ratio > 0.1:
                                    st.success(f"✅ Good separation: Embedding model distinguishes similar vs non-similar words well (separation ratio: {separation_ratio:.4f})")
                                elif separation_ratio > 0.05:
                                    st.warning(f"⚠️ Moderate separation: Embedding model shows some ability to distinguish similar vs non-similar words (separation ratio: {separation_ratio:.4f})")
                                else:
                                    st.error(f"❌ Poor separation: Embedding model struggles to distinguish similar vs non-similar words (separation ratio: {separation_ratio:.4f})")
                                
                                if auc_cosine > 0.7:
                                    st.success(f"✅ Good ROC AUC: {auc_cosine:.4f} (closer to 1.0 is better)")
                                elif auc_cosine > 0.6:
                                    st.warning(f"⚠️ Moderate ROC AUC: {auc_cosine:.4f}")
                                else:
                                    st.error(f"❌ Poor ROC AUC: {auc_cosine:.4f}")
                                
                    except FileNotFoundError:
                        st.error(f"⚠️ Dataset file not found at: {dataset_path}")
                    except Exception as e:
                        st.error(f"❌ Error running WordSim353 evaluation: {e}")
                        logger.error(f"WordSim353 evaluation error: {e}", exc_info=True)
                        import traceback
                        st.code(traceback.format_exc())
            else:
                st.info("👈 Click the '📊 Run WordSim353 Evaluation' button to evaluate the embedding model using the WordSim353 dataset")


