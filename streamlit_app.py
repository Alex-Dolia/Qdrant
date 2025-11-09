"""
Simplified RAG System - Qdrant + Llama 3.1 Only
Query documents or perform web search to generate reports.
"""

import streamlit as st
import os
import signal
import sys
import asyncio
from pathlib import Path
from datetime import datetime
import logging

# Import pandas for dataframes (used in file statistics and chunk exploration)
try:
    import pandas as pd
except ImportError:
    pd = None

# Setup logging
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"logs/logger_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
try:
    signal.signal(signal.SIGINT, signal_handler)
    logger.info("Signal handler registered for Ctrl-C")
except (ValueError, OSError) as e:
    # Signal handling might not work in all environments (e.g., some IDEs)
    logger.warning(f"Could not register signal handler: {e}")
    logger.info("You can still stop the app with Ctrl-C in the terminal where Streamlit is running")

# On Windows, also handle SIGTERM if available
if sys.platform == "win32":
    try:
        signal.signal(signal.SIGTERM, signal_handler)
    except (AttributeError, ValueError, OSError):
        # SIGTERM might not be available on all Windows versions
        pass

# Initialize interrupt flag in session state
if 'interrupt_flag' not in st.session_state:
    st.session_state.interrupt_flag = False

# Load prompt templates at startup
try:
    from prompts import load_prompt_template, PROMPTS_DIR
    logger.info(f"Prompts directory found: {PROMPTS_DIR}")
    # Verify prompts are available
    try:
        test_prompt = load_prompt_template("abstract_introduction")
        logger.info("✓ Prompt templates loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load test prompt: {e}")
except ImportError as e:
    logger.warning(f"Prompts module not available: {e}")

# Imports
try:
    # Only import what's needed for Research Overview (Tab 3)
    # Note: Use importlib for modules starting with numbers
    import importlib
    retrieval_module = importlib.import_module('scripts.03_retrieval.retrieval')
    RAGRetrievalEngine = retrieval_module.RAGRetrievalEngine
    
    # Import DeepSeek for Simple Web Search (Tab 2)
    try:
        deepseek_module = importlib.import_module('scripts.08_utilities.deepseek')
        DeepSeek = deepseek_module.DeepSeek
        DEEPSEEK_AVAILABLE = True
    except ImportError:
        DEEPSEEK_AVAILABLE = False
        logger.warning("DeepSeek not available, falling back to WebSearch")
        research_workflow_module = importlib.import_module('scripts.07_research_workflow.web_search')
        WebSearch = research_workflow_module.WebSearch
    
    # Keep WebSearch for Research Overview (Tab 3) compatibility
    research_workflow_module = importlib.import_module('scripts.07_research_workflow.web_search')
    WebSearch = research_workflow_module.WebSearch
    
    research_assistant_module = importlib.import_module('scripts.07_research_workflow.research_assistant')
    ResearchAssistant = research_assistant_module.ResearchAssistant
    
    research_overview_module = importlib.import_module('scripts.07_research_workflow.research_overview_workflow')
    ResearchOverviewWorkflow = research_overview_module.ResearchOverviewWorkflow
    
    synthesis_module = importlib.import_module('scripts.02_query_completion.synthesis')
    ResearchSynthesizer = synthesis_module.ResearchSynthesizer
    
    output_gen_module = importlib.import_module('scripts.05_output_generation.report_formatter')
    ReportFormatter = output_gen_module.ReportFormatter
    
    chunking_module = importlib.import_module('scripts.00_chunking.legal_chunker_integration')
    ingest_legal_document = chunking_module.ingest_legal_document
    query_legal_documents = chunking_module.query_legal_documents
    get_distinct_source_files = chunking_module.get_distinct_source_files
    get_distinct_chunking_methods = chunking_module.get_distinct_chunking_methods
    get_available_chunking_methods = chunking_module.get_available_chunking_methods
    get_default_chunking_methods = chunking_module.get_default_chunking_methods
    delete_file_from_qdrant = chunking_module.delete_file_from_qdrant
    delete_all_files_from_qdrant = chunking_module.delete_all_files_from_qdrant
    get_file_statistics = chunking_module.get_file_statistics
    get_chunks_for_exploration = chunking_module.get_chunks_for_exploration
    EMBEDDING_MODELS = chunking_module.EMBEDDING_MODELS
    LEGAL_CHUNKER_AVAILABLE = chunking_module.LEGAL_CHUNKER_AVAILABLE
except ImportError as e:
    st.error(f"❌ Failed to import required modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="RAG Query System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state for Research Overview (Tab 3)
# Note: Legal Documents (Tab 1) uses its own Qdrant client initialized in the tab
if 'research_overview_workflow' not in st.session_state:
    try:
        # Initialize retrieval engine only for Research Overview workflow
        retrieval_engine = RAGRetrievalEngine(
            use_ollama=True,
            vector_db_type="Qdrant",
            embedding_model="llama3.1"
        )
        # Store retrieval engine in session state for reuse
        st.session_state.retrieval_engine = retrieval_engine
        synthesizer = ResearchSynthesizer(use_ollama=True)
        report_formatter = ReportFormatter()
        st.session_state.research_overview_workflow = ResearchOverviewWorkflow(
            synthesizer=synthesizer,
            report_formatter=report_formatter,
            retrieval_engine=retrieval_engine
        )
        logger.info("Research overview workflow initialized")
    except Exception as e:
        logger.warning(f"Could not initialize research overview workflow: {e}")
        st.session_state.research_overview_workflow = None

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

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["⚖️ Legal Documents Query", "🔍 Simple Web Search", "📚 Research Overview", "🧩 Chunk Exploration"])

# Sidebar for Legal Documents Configuration
with st.sidebar:
    # Display Qdrant connection status at the top
    st.markdown("### 🔌 Connection Status")
    try:
        qdrant_health_module = importlib.import_module('scripts.08_utilities.qdrant_health')
        display_qdrant_status_in_ui = qdrant_health_module.display_qdrant_status_in_ui
        display_qdrant_status_in_ui(st)
    except Exception as e:
        logger.warning(f"Could not display Qdrant status: {e}")
        st.info("⚠️ Qdrant status check unavailable")
    st.markdown("---")
    
    st.header("📚 Document Management")
    
    # Embedding Model Selection
    # EMBEDDING_MODELS, get_available_chunking_methods, get_default_chunking_methods already imported above
    
    # Initialize legal embedding model in session state
    if 'legal_embedding_model' not in st.session_state:
        st.session_state.legal_embedding_model = EMBEDDING_MODELS[0] if EMBEDDING_MODELS else "ollama/llama3.1:latest"
    
    legal_embedding_model = st.selectbox(
        "Embedding Model",
        options=EMBEDDING_MODELS,
        index=EMBEDDING_MODELS.index(st.session_state.legal_embedding_model) if st.session_state.legal_embedding_model in EMBEDDING_MODELS else 0,
        key="sidebar_legal_embedding_model",
        help="Select embedding model for legal document processing"
    )
    
    # Update session state
    st.session_state.legal_embedding_model = legal_embedding_model
    
    st.markdown("---")
    
    # Chunking Methods Selection
    st.subheader("⚙️ Chunking Methods")
    st.caption("Select which chunking methods to use when storing documents")
    
    # Get available chunking methods
    available_chunking_methods = get_available_chunking_methods()
    default_chunking_methods = get_default_chunking_methods()
    
    # Initialize session state for chunking methods (default: all methods)
    if 'sidebar_selected_chunking_methods' not in st.session_state:
        st.session_state.sidebar_selected_chunking_methods = available_chunking_methods.copy()
    
    # Create checkboxes for each chunking method
    selected_chunking_methods = []
    
    for method in available_chunking_methods:
        # Check if method is in current selection (default to True for all)
        is_checked = method in st.session_state.sidebar_selected_chunking_methods
        checked = st.checkbox(
            method.capitalize(),
            value=is_checked,
            key=f"sidebar_chunking_{method}",
            help=f"Use {method} chunking method when storing documents"
        )
        if checked:
            selected_chunking_methods.append(method)
    
    # Update session state
    st.session_state.sidebar_selected_chunking_methods = selected_chunking_methods
    
    # Show summary
    if selected_chunking_methods:
        st.caption(f"✅ {len(selected_chunking_methods)} method(s) selected")
    else:
        st.warning("⚠️ No chunking methods selected. Please select at least one.")
    
    # Info about chunking method consistency
    st.info("💡 **Note:** Both upload and retrieval use the same chunking methods from codebase. Retrieval filters to show only methods that exist in Qdrant.")
    
# Tab 1: Legal Documents Query
with tab1:
    st.subheader("⚖️ Legal Document RAG Pipeline")
    st.markdown("Upload legal documents with multiple chunking strategies. Query with selected embedding model and chunking method.")
    
    if not LEGAL_CHUNKER_AVAILABLE:
        st.error("⚠️ Legal chunker not available. Please install dependencies: pip install qdrant-client sentence-transformers langchain langchain-experimental scikit-learn pypdf")
        st.stop()
    
    # Check if semantic chunking is available
    available_chunking_methods_for_display = get_available_chunking_methods()
    if "semantic" not in available_chunking_methods_for_display:
        st.info("💡 **Note:** Semantic chunking is not available. Install `langchain-experimental` to enable semantic chunking: `pip install langchain-experimental`")
    
    # Initialize Qdrant client for legal documents
    if 'legal_qdrant_client' not in st.session_state:
        try:
            from qdrant_client import QdrantClient
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            st.session_state.legal_qdrant_client = QdrantClient(url=qdrant_url)
            st.session_state.legal_collection_name = "legal_documents"
            logger.info("Legal Qdrant client initialized")
        except Exception as e:
            st.error(f"Failed to initialize Qdrant client: {e}")
            st.session_state.legal_qdrant_client = None
    
    # Legal Document Upload Section
    st.markdown("### 📤 Upload Legal Document")
    
    legal_file = st.file_uploader(
        "Upload legal document (PDF, DOCX, DOC, MD, or TXT)",
        type=["pdf", "docx", "doc", "md", "txt"],
        key="legal_file_upload",
        help="All files will be converted to Markdown format before chunking (except .md files)"
    )
    
    if legal_file:
        # Use embedding model and chunking methods from sidebar
        selected_embedding = st.session_state.legal_embedding_model
        selected_chunking_methods = st.session_state.sidebar_selected_chunking_methods
        
        # Show current settings from sidebar
        st.info(f"""📌 **Settings from sidebar:**
- **Embedding Model:** `{selected_embedding}`
- **Chunking Methods:** {', '.join(selected_chunking_methods) if selected_chunking_methods else 'None selected'}

💡 Change these settings in the sidebar (📚 Document Management)""")
        
        # Warn if no chunking methods selected
        if not selected_chunking_methods:
            st.error("⚠️ No chunking methods selected in sidebar. Please select at least one method in the sidebar before uploading.")
        
        if st.button("📥 Upload & Process", type="primary", key="legal_upload_button"):
            if not selected_chunking_methods:
                st.error("⚠️ Please select at least one chunking method")
            elif st.session_state.legal_qdrant_client is None:
                st.error("⚠️ Qdrant client not available")
            else:
                try:
                    # Save uploaded file temporarily
                    temp_path = f"./temp/legal_{legal_file.name}"
                    os.makedirs("./temp", exist_ok=True)
                    
                    with open(temp_path, "wb") as f:
                        f.write(legal_file.getbuffer())
                    
                    # Create progress container
                    progress_container = st.container()
                    
                    with progress_container:
                        st.info(f"🔄 Processing **{legal_file.name}** with embedding model: `{selected_embedding}`")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Track progress callback
                        def update_progress(current_method: str, method_index: int, total_methods: int):
                            """Update progress indicator for chunking method."""
                            progress = (method_index + 1) / total_methods
                            progress_bar.progress(progress)
                            status_text.info(f"📦 Processing chunking method **{current_method}** ({method_index + 1}/{total_methods})...")
                        
                        # Process with progress tracking
                        try:
                            result = ingest_legal_document(
                                file_path=temp_path,
                                qdrant_client=st.session_state.legal_qdrant_client,
                                collection_name=st.session_state.legal_collection_name,
                                embedding_model=selected_embedding,
                                chunking_methods=selected_chunking_methods,
                                file_id=legal_file.name,
                                progress_callback=update_progress
                            )
                        except Exception as e:
                            progress_bar.empty()
                            status_text.empty()
                            raise e
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        if result.get("success"):
                            st.success(f"✅ Document processed successfully!")
                            st.info(f"""
                            **Processing Summary:**
                            - File: {result['file_name']}
                            - Total chunks: {result['total_chunks']}
                            - Embedding model: {result['embedding_model']}
                            - Chunking methods: {', '.join(result['chunking_methods'])}
                            
                            **Chunks per method:**
                            """)
                            for method, count in result['method_counts'].items():
                                st.write(f"  - {method}: {count} chunks")
                            
                            # Refresh file list by rerunning
                            st.rerun()
                        else:
                            st.error(f"❌ Processing failed")
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.error(f"Legal document ingestion error: {e}", exc_info=True)
    
    st.markdown("---")
    
    # Display stored files section with deletion options
    st.markdown("### 📋 Stored Files in Qdrant")
    
    # Refresh button for file list
    col_refresh, col_info = st.columns([1, 4])
    with col_refresh:
        refresh_files = st.button("🔄 Refresh File List", key="refresh_legal_files", help="Reload list of files from Qdrant")
    
    if refresh_files:
            st.rerun()
        
    # Get distinct source files from Qdrant
    if st.session_state.legal_qdrant_client is not None:
        try:
            distinct_files = get_distinct_source_files(
                st.session_state.legal_qdrant_client,
                st.session_state.legal_collection_name
            )
            
            if distinct_files:
                st.success(f"✅ Found {len(distinct_files)} file(s) in collection: `{st.session_state.legal_collection_name}`")
                
                # Display files in a nice format with delete buttons
                with st.expander(f"📁 View All Files ({len(distinct_files)} files)", expanded=False):
                    for i, filename in enumerate(distinct_files, 1):
                        col_file, col_delete = st.columns([4, 1])
                        with col_file:
                            st.write(f"{i}. 📄 **{filename}**")
                        with col_delete:
                            if st.button("🗑️ Delete", key=f"delete_file_{i}_{filename}", help=f"Delete all chunks for {filename}"):
                                with st.spinner(f"Deleting {filename}..."):
                                    result = delete_file_from_qdrant(
                                        qdrant_client=st.session_state.legal_qdrant_client,
                                        collection_name=st.session_state.legal_collection_name,
                                        source_file=filename
                                    )
                                    if result.get("success"):
                                        st.success(result.get("message", "File deleted successfully"))
                                        st.rerun()
                                    else:
                                        st.error(result.get("message", "Failed to delete file"))
                
                # Store for use in query dropdown
                source_files_list = ["All Files"] + distinct_files
                
                # File Statistics Button
                st.markdown("---")
                st.markdown("### 📊 File Statistics")
                
                if st.button("📊 Show File Statistics", type="secondary", key="show_file_statistics", help="Display detailed statistics about all uploaded files"):
                    try:
                        with st.spinner("Gathering file statistics..."):
                            stats = get_file_statistics(
                                qdrant_client=st.session_state.legal_qdrant_client,
                                collection_name=st.session_state.legal_collection_name
                            )
                            
                            # Display summary statistics
                            st.success(f"✅ **Statistics Generated**")
                            st.markdown(f"""
                            **Summary:**
                            - **Total Files:** {stats['total_files']}
                            - **Total Chunks:** {stats['total_chunks']}
                            """)
                            
                            # Display detailed file information
                            if stats['files']:
                                st.markdown("### 📋 Detailed File Information")
                                
                                for file_info in stats['files']:
                                    with st.expander(f"📄 {file_info['source_file']}", expanded=False):
                                        # Get chunking methods string (use string version if available, otherwise join list)
                                        chunking_methods_display = file_info.get('chunking_methods_str') or (
                                            ', '.join(file_info['chunking_methods']) if file_info.get('chunking_methods') else 'Unknown'
                                        )
                                        
                                        st.markdown(f"""
                                        **File:** `{file_info['source_file']}`
                                        
                                        **Total Chunks:** {file_info['chunk_count']}
                                        
                                        **Chunking Methods:** {chunking_methods_display}
                                        
                                        **Upload Date/Time:** {file_info['upload_date']}
                                        
                                        **Embedding Model:** `{file_info['embedding_model']}`
                                        """)
                                        
                                        # Show chunking method breakdown with counts
                                        chunking_method_counts = file_info.get('chunking_method_counts', {})
                                        if chunking_method_counts:
                                            st.markdown("#### 📊 Chunks per Chunking Method")
                                            
                                            # Create a table showing method and count
                                            method_data = []
                                            for method in sorted(chunking_method_counts.keys()):
                                                count = chunking_method_counts[method]
                                                method_data.append({
                                                    "Chunking Method": method,
                                                    "Chunk Count": count,
                                                    "Percentage": f"{(count / file_info['chunk_count'] * 100):.1f}%"
                                                })
                                            
                                            if method_data:
                                                if pd is not None:
                                                    df_methods = pd.DataFrame(method_data)
                                                    st.dataframe(
                                                        df_methods,
                                                        use_container_width=True,
                                                        hide_index=True
                                                    )
                                                else:
                                                    # Fallback: display as markdown table if pandas not available
                                                    st.markdown("| Chunking Method | Chunk Count | Percentage |")
                                                    st.markdown("|----------------|-------------|------------|")
                                                    for item in method_data:
                                                        st.markdown(f"| {item['Chunking Method']} | {item['Chunk Count']} | {item['Percentage']} |")
                                            
                                            # Show summary if multiple methods
                                            if len(chunking_method_counts) > 1:
                                                st.caption(f"💡 This file uses {len(chunking_method_counts)} different chunking methods")
                                        else:
                                            # Fallback if chunking_method_counts is not available
                                            if len(file_info['chunking_methods']) > 1:
                                                st.caption(f"💡 This file uses {len(file_info['chunking_methods'])} different chunking methods")
                            else:
                                st.info("No files found in the collection.")
                                
                    except Exception as e:
                        st.error(f"❌ Error getting file statistics: {e}")
                        logger.error(f"Error getting file statistics: {e}", exc_info=True)
                
                # Delete all files button
                st.markdown("---")
                st.markdown("### 🗑️ Delete All Files")
                st.warning("⚠️ This will delete ALL files and chunks from the collection. This action cannot be undone.")
                
                col_delete_all, col_cancel = st.columns([1, 4])
                with col_delete_all:
                    if st.button("🗑️ Delete All Files", type="primary", key="delete_all_files", help="Delete all files from Qdrant collection"):
                        with st.spinner("Deleting all files..."):
                            result = delete_all_files_from_qdrant(
                                qdrant_client=st.session_state.legal_qdrant_client,
                                collection_name=st.session_state.legal_collection_name
                            )
                            if result.get("success"):
                                st.success(result.get("message", "All files deleted successfully"))
                                st.rerun()
                            else:
                                st.error(result.get("message", "Failed to delete all files"))
            else:
                st.info("ℹ️ No files found in Qdrant collection. Upload documents to get started.")
                source_files_list = ["All Files"]
        except Exception as e:
            st.warning(f"⚠️ Could not fetch files from Qdrant: {e}")
            logger.warning(f"Could not fetch source files: {e}")
            source_files_list = ["All Files"]
    else:
        st.warning("⚠️ Qdrant client not available")
        source_files_list = ["All Files"]
    
    st.markdown("---")
    
    # Legal Document Query Section
    st.markdown("### 🔍 Query Legal Documents")
    st.markdown("Select search mode, embedding model, chunking method, and source file to query stored documents.")
    st.caption("💡 Chunking methods shown are consistent with upload options, filtered to methods available in Qdrant")
    
    query_col1, query_col2, query_col3, query_col4, query_col5, query_col6 = st.columns([2, 1, 1, 1, 1, 1])
    
    with query_col1:
        legal_query = st.text_area(
            "Enter your query",
            height=100,
            placeholder="Search legal documents...",
            key="legal_query_input"
        )
    
    with query_col2:
        # Search mode selection
        search_mode = st.selectbox(
            "Search Mode",
            options=["semantic", "bm25", "mixed"],
            index=0,
            key="legal_query_search_mode",
            help="semantic: Embedding similarity | bm25: Keyword search | mixed: Hybrid (RRF)"
        )
    
    with query_col3:
        query_embedding = st.selectbox(
            "Embedding Model",
            options=EMBEDDING_MODELS,
            index=0,
            key="legal_query_embedding",
            help="Required for semantic/mixed modes. Must match ingestion model."
        )
    
    with query_col4:
        # Use same chunking methods as upload, but filter to only show methods that exist in Qdrant
        # This ensures consistency between upload and retrieval while supporting legacy data
        if st.session_state.legal_qdrant_client is not None:
            try:
                # Get available methods from codebase (same as upload)
                available_methods = get_available_chunking_methods()
                
                # Get methods that actually exist in Qdrant
                qdrant_chunking_methods = get_distinct_chunking_methods(
                    st.session_state.legal_qdrant_client,
                    st.session_state.legal_collection_name
                )
                
                # Intersection: only show methods that are both available AND in Qdrant
                if qdrant_chunking_methods:
                    # Filter available_methods to only include those found in Qdrant
                    filtered_methods = [m for m in available_methods if m in qdrant_chunking_methods]
                    
                    # Also include any methods from Qdrant that might not be in available_methods
                    # (for legacy support and to catch any methods that were added but not detected)
                    for qdrant_method in qdrant_chunking_methods:
                        if qdrant_method not in filtered_methods:
                            filtered_methods.append(qdrant_method)
                            logger.info(f"Added Qdrant method '{qdrant_method}' to dropdown (not in available_methods)")
                    
                    # If no intersection, fall back to Qdrant methods (for legacy support)
                    if not filtered_methods:
                        filtered_methods = qdrant_chunking_methods
                        st.warning("⚠️ Some methods in Qdrant are not in codebase (legacy data)")
                    
                    # Debug: Show what methods were found (always log for troubleshooting)
                    logger.info(f"Available methods from codebase: {available_methods}")
                    logger.info(f"Methods found in Qdrant: {qdrant_chunking_methods}")
                    logger.info(f"Final filtered methods for dropdown: {filtered_methods}")
                    
                    # Find index of hierarchical (default) or use 0
                    default_index = 0
                    if "hierarchical" in filtered_methods:
                        default_index = filtered_methods.index("hierarchical")
                    
                    query_chunking_method = st.selectbox(
                        "Chunking Method",
                        options=filtered_methods,
                        index=default_index,
                        key="legal_query_chunking",
                        help=f"Filter results by chunking method. Available: {', '.join(filtered_methods)}"
                    )
                else:
                    # No chunking methods found in Qdrant
                    st.warning("⚠️ No chunking methods found in Qdrant. Please upload documents first.")
                    st.selectbox(
                        "Chunking Method",
                        options=["No methods found"],
                        index=0,
                        key="legal_query_chunking",
                        disabled=True,
                        help="No chunking methods found in Qdrant. Upload documents first."
                    )
                    query_chunking_method = None
            except Exception as e:
                st.warning(f"⚠️ Could not fetch chunking methods: {e}")
                logger.warning(f"Could not fetch chunking methods: {e}", exc_info=True)
                query_chunking_method = None
        else:
            st.selectbox(
                "Chunking Method",
                options=["Qdrant not available"],
                index=0,
                key="legal_query_chunking",
                disabled=True,
                help="Qdrant client not available"
            )
            query_chunking_method = None
    
    with query_col5:
        if len(source_files_list) > 1:
            query_source_file = st.selectbox(
                "Source File",
                options=source_files_list,
                index=0,  # Default to "All Files"
                key="legal_query_source_file",
                help="Filter results by source file (or search all files)"
            )
        else:
            st.selectbox(
                "Source File",
                options=["All Files"],
                index=0,
                key="legal_query_source_file",
                disabled=True,
                help="No files available. Upload documents first."
            )
            query_source_file = "All Files"
    
    # LLM Model Selection for Answer Generation
    st.markdown("**Answer Generation Model:**")
    llm_model_col1, llm_model_col2 = st.columns([1, 3])
    with llm_model_col1:
        # Initialize session state for LLM model
        if 'legal_llm_model' not in st.session_state:
            st.session_state.legal_llm_model = "llama3.1:latest"
        
        llm_model_options = ["llama3.1:latest", "llama3.2:latest", "llama3:latest", "mistral:latest", "mixtral:latest"]
        selected_llm_model = st.selectbox(
            "LLM Model",
            options=llm_model_options,
            index=llm_model_options.index(st.session_state.legal_llm_model) if st.session_state.legal_llm_model in llm_model_options else 0,
            key="legal_llm_model_select",
            help="Model for generating final answer from search results"
        )
        st.session_state.legal_llm_model = selected_llm_model
    
    st.markdown("---")
    
    if st.button("🔍 Search Legal Documents", type="primary", key="legal_query_button"):
        if not legal_query:
            st.warning("⚠️ Please enter a query")
        elif st.session_state.legal_qdrant_client is None:
            st.error("⚠️ Qdrant client not available")
        elif query_chunking_method is None:
            st.warning("⚠️ No chunking method available. Please upload documents first.")
        else:
            try:
                with st.spinner("Searching legal documents..."):
                    results = query_legal_documents(
                        qdrant_client=st.session_state.legal_qdrant_client,
                        collection_name=st.session_state.legal_collection_name,
                        query_text=legal_query,
                        embedding_model=query_embedding,
                        chunking_method=query_chunking_method,
                        source_file=query_source_file if query_source_file != "All Files" else None,
                        top_k=5,
                        search_mode=search_mode
                    )
                    
                    if results:
                        st.success(f"✅ Found {len(results)} unique results")
                        
                        # Generate LLM answer from reranked results
                        try:
                            from langchain_ollama import ChatOllama
                            import os
                            
                            with st.spinner(f"Generating answer using {selected_llm_model}..."):
                                # Initialize LLM
                                llm = ChatOllama(
                                    model=selected_llm_model,
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

{legal_query}

Retrieved Chunks (Context):

{retrieved_chunks_context}

Your Answer:"""
                                
                                # Generate answer
                                response = llm.invoke(prompt)
                                llm_answer = response.content if hasattr(response, 'content') else str(response)
                                
                                # Display LLM answer prominently
                                st.markdown("### 💬 Generated Answer")
                                st.info(llm_answer)
                                st.markdown("---")
                                
                        except ImportError:
                            st.warning("⚠️ langchain-ollama not available. Install with: pip install langchain-ollama")
                        except Exception as e:
                            logger.error(f"Error generating LLM answer: {e}", exc_info=True)
                            st.warning(f"⚠️ Could not generate answer: {e}")
                        
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
                            - **Semantic mode**: Normalized cosine similarity scores (0-1, higher = more similar)
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
                        
                        st.markdown("### 📋 Search Results")
                        st.markdown("---")
                        
                        for i, result in enumerate(results, 1):
                            # Display result in a card-like format with expandable sections
                            with st.expander(f"📄 Result {i} - Score: {result['score']:.4f} | {result.get('source_file', result.get('file_name', 'Unknown'))}", expanded=True):
                                # Display the actual text content prominently
                                st.markdown("### 📄 Chunk Content")
                                
                                # For BM25 and mixed modes, show highlighted text if available; otherwise show plain text
                                if search_mode in ["bm25", "mixed"] and result.get('highlighted_text'):
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
                                score_info = [f"**Relevance Score:** {result['score']:.4f}"]
                                if result.get('original_score') is not None and result.get('original_score') != result['score']:
                                    score_info.append(f"**Original Score:** {result.get('original_score'):.4f}")
                                if result.get('bm25_score') is not None:
                                    score_info.append(f"**BM25 Score:** {result.get('bm25_score'):.4f}")
                                if result.get('semantic_score') is not None:
                                    score_info.append(f"**Semantic Score:** {result.get('semantic_score'):.4f}")
                                if result.get('rrf_score') is not None:
                                    score_info.append(f"**RRF Score:** {result.get('rrf_score'):.4f}")
                                
                                st.caption(" | ".join(score_info))
                    else:
                        st.info("No results found. Try a different query or check if documents are uploaded.")
            except Exception as e:
                st.error(f"Error searching documents: {e}")
                logger.error(f"Legal document query error: {e}", exc_info=True)

# Tab 2: Simple Web Search (DeepSeek)
with tab2:
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
            available_models = legal_chunker_module.EMBEDDING_MODELS if hasattr(legal_chunker_module, 'EMBEDDING_MODELS') else ["sentence-transformers/all-MiniLM-L6-v2"]
        except:
            available_models = ["sentence-transformers/all-MiniLM-L6-v2"]
        
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
                # Initialize DeepSeek if not already done or if settings changed
                cache_key = f"deepseek_{search_engine}_{use_reranking}_{use_cache}_{rank_mode}_{embedding_model}"
                if cache_key not in st.session_state:
                    if DEEPSEEK_AVAILABLE:
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
                    else:
                        st.error("❌ DeepSeek is not available. Please install dependencies:")
                        st.code("pip install -r requirements.txt")
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

# Tab 3: Research Overview
with tab3:
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
            if st.session_state.research_overview_workflow is None:
                st.error("Research overview workflow not initialized. Please check logs.")
            else:
                try:
                    with st.spinner("Generating comprehensive research overview... This may take several minutes."):
                        # First, gather research context
                        context = ""
                        references = []
                        
                        if use_web_research:
                            # Use research assistant to gather web research
                            if 'research_assistant' not in st.session_state:
                                st.session_state.research_assistant = ResearchAssistant(
                                    retrieval_engine=st.session_state.retrieval_engine,
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
                        result = st.session_state.research_overview_workflow.execute(
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

# Tab 4: Chunk Exploration
with tab4:
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
                            
                            if selected_method:
                                # Chunk Level Switcher (only for structural chunking)
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


