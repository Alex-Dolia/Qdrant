"""
LangGraph Builder Module
Defines the research workflow using LangGraph.
"""

import os
import logging
import traceback
from typing import Dict, List, Optional, Any, Annotated
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st

# Import logging utility
try:
    from scripts.logging_config import setup_logger, write_to_log, get_current_log_file
except ImportError:
    # Fallback if logging_config not available
    # Use session state to get the same log file as streamlit_app.py
    try:
        import streamlit as st
        USE_STREAMLIT = True
    except ImportError:
        USE_STREAMLIT = False
    
    def get_current_log_file():
        """Get current log file from session state or create one."""
        if USE_STREAMLIT and 'log_file_path' in st.session_state:
            return st.session_state.log_file_path
        else:
            # Fallback: create timestamped file
            os.makedirs("logs", exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file = f"logs/logger_{timestamp}.log"
            if USE_STREAMLIT and 'log_file_path' not in st.session_state:
                st.session_state.log_file_path = log_file
            return log_file
    
    def setup_logger(name=None, level=logging.INFO):
        logger = logging.getLogger(name)
        if not logger.handlers:
            log_file = get_current_log_file()
            logging.basicConfig(
                level=level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file, encoding='utf-8', mode='a'),
                    logging.StreamHandler()
                ]
            )
        return logger
    
    def write_to_log(message: str, log_type: str = "INFO"):
        try:
            log_file = get_current_log_file()
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"{log_type} at {datetime.now().isoformat()}\n")
                f.write(f"{message}\n")
                f.write(f"{'='*80}\n\n")
        except:
            pass

logger = setup_logger(__name__)

# Import agent configuration
try:
    from agents.agent_config import ResearchState, WORKFLOW_CONFIG, PARALLEL_GENERATION_CONFIG
except ImportError:
    # Fallback to inline definition if agents module not available
    from typing import TypedDict
    class ResearchState(TypedDict):
        """State for the research workflow."""
        query: str
        task_type: str
        retrieved_chunks: List[Dict]
        web_results: List[Dict]
        memory_context: str
        synthesized_context: str
        sections: Dict[str, str]
        references: List[Dict]
        report: Dict[str, Any]
        metadata: Dict[str, Any]
    
    WORKFLOW_CONFIG = {
        "entry_point": "input",
        "nodes": ["input", "retrieve", "synthesize", "generate_report", "format_output"],
        "edges": [
            ("input", "retrieve"),
            ("retrieve", "synthesize"),
            ("synthesize", "generate_report"),
            ("generate_report", "format_output"),
            ("format_output", "END")
        ]
    }
    
    PARALLEL_GENERATION_CONFIG = {
        "max_workers": 3,
        "sections": ["Abstract", "Introduction", "Research Findings", "Conclusion"]
    }

logger = logging.getLogger(__name__)

# Try to import LangGraph
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    try:
        # Try alternative import path
        from langgraph.graph.state import StateGraph
        from langgraph.graph import END
        LANGGRAPH_AVAILABLE = True
    except ImportError:
        LANGGRAPH_AVAILABLE = False
        logger.warning("LangGraph not available. Install with: pip install langgraph")


class ResearchGraphBuilder:
    """Builds and executes LangGraph workflow for research."""
    
    def __init__(
        self,
        retrieval_engine,
        web_search,
        synthesizer,
        report_formatter,
        use_ollama: bool = True
    ):
        """
        Initialize the graph builder.
        
        Args:
            retrieval_engine: RAGRetrievalEngine instance
            web_search: WebSearch instance
            synthesizer: ResearchSynthesizer instance
            report_formatter: ReportFormatter instance
            use_ollama: Whether to use Ollama for LLM
        """
        self.retrieval_engine = retrieval_engine
        self.web_search = web_search
        self.synthesizer = synthesizer
        self.report_formatter = report_formatter
        self.use_ollama = use_ollama
        self.graph = None
        
        if LANGGRAPH_AVAILABLE:
            self._build_graph()
        else:
            logger.warning("LangGraph not available, using fallback workflow")
    
    def _build_graph(self):
        """Build the LangGraph workflow using configuration from agents/agent_config.py."""
        if not LANGGRAPH_AVAILABLE:
            return
        
        workflow = StateGraph(ResearchState)
        
        # Add nodes from configuration
        for node_name in WORKFLOW_CONFIG["nodes"]:
            node_method = getattr(self, f"_{node_name}_node", None)
            if node_method:
                workflow.add_node(node_name, node_method)
            else:
                logger.warning(f"Node method _{node_name}_node not found")
        
        # Define flow from configuration
        workflow.set_entry_point(WORKFLOW_CONFIG["entry_point"])
        for from_node, to_node in WORKFLOW_CONFIG["edges"]:
            if to_node == "END":
                workflow.add_edge(from_node, END)
            else:
                workflow.add_edge(from_node, to_node)
        
        self.graph = workflow.compile()
        logger.info("Built LangGraph workflow")
    
    def _input_node(self, state: ResearchState) -> ResearchState:
        """Input node - accepts research topic and task type."""
        logger.info(f"Input node: query={state.get('query')}, task_type={state.get('task_type')}")
        return state
    
    def _retrieve_node(self, state: ResearchState) -> ResearchState:
        """Retrieve node - performs web-only retrieval (no vector DB) with caching."""
        query = state.get("query", "")
        
        logger.info("-" * 80)
        logger.info("Stage 2: Web Research (Basic Workflow)")
        logger.info("-" * 80)
        logger.info(f"  Query: {query}")
        
        # Skip vector DB retrieval - web-only research
        retrieved_chunks = []
        logger.info("  Vector DB retrieval: Skipped (web-only research mode)")
        
        # Retrieve from web (with caching for speed)
        web_results = []
        if self.web_search and self.web_search.is_available():
            try:
                # Use max_results from configuration
                try:
                    from agents.agent_config import RETRIEVAL_CONFIG
                    max_results = RETRIEVAL_CONFIG["web_search"]["max_results"]
                except ImportError:
                    max_results = 10  # Increased for better coverage
                logger.info(f"  Max results: {max_results}")
                logger.info(f"  Performing web search...")
                web_results = self.web_search.search(query, max_results=max_results)
                logger.info(f"  Retrieved {len(web_results)} web results")
                if web_results:
                    logger.info("  Sample results:")
                    for i, result in enumerate(web_results[:3], 1):
                        logger.info(f"    [{i}] {result.get('title', 'N/A')[:60]}...")
                        logger.info(f"        URL: {result.get('url', 'N/A')[:80]}")
                else:
                    logger.warning("  WARNING: Web search returned no results")
            except Exception as e:
                logger.error(f"  ERROR: Web search error: {e}")
                logger.error(f"  Traceback: {traceback.format_exc()}")
        else:
            logger.warning("  Web search provider: NOT AVAILABLE")
            logger.warning("  Install duckduckgo-search: pip install duckduckgo-search")
        
        logger.info(f"Stage 2 Complete: {len(web_results)} web results collected")
        logger.info("-" * 80)
        
        state["retrieved_chunks"] = retrieved_chunks
        state["web_results"] = web_results
        return state
    
    def _synthesize_node(self, state: ResearchState) -> ResearchState:
        """Synthesize node - combines retrieved information."""
        query = state.get("query", "")
        retrieved_chunks = state.get("retrieved_chunks", [])
        web_results = state.get("web_results", [])
        memory_context = state.get("memory_context", "")
        
        logger.info("-" * 80)
        logger.info("Stage 3: Technical Analysis & Synthesis (Basic Workflow)")
        logger.info("-" * 80)
        logger.info(f"  Query: {query}")
        logger.info(f"  Web Results Count: {len(web_results)}")
        logger.info(f"  Retrieved Chunks Count: {len(retrieved_chunks)}")
        logger.info(f"  Memory Context Length: {len(memory_context)} chars")
        logger.info("  Starting synthesis...")
        
        try:
            synthesized = self.synthesizer.synthesize(
                query=query,
                retrieved_chunks=retrieved_chunks,
                web_results=web_results,
                memory_context=memory_context
            )
            logger.info(f"  Synthesis complete")
            logger.info(f"  Synthesized context length: {len(synthesized)} chars")
            logger.info(f"  Synthesized context preview: {synthesized[:200]}...")
            state["synthesized_context"] = synthesized
        except Exception as e:
            logger.error(f"  ERROR: Synthesis error: {e}")
            logger.error(f"  Traceback: {traceback.format_exc()}")
            state["synthesized_context"] = f"Error during synthesis: {e}"
        
        logger.info("Stage 3 Complete: Context synthesized")
        logger.info("-" * 80)
        
        return state
    
    def _generate_report_node(self, state: ResearchState) -> ResearchState:
        """Generate report node - creates structured report sections (parallel)."""
        query = state.get("query", "")
        synthesized_context = state.get("synthesized_context", "")
        retrieved_chunks = state.get("retrieved_chunks", [])
        web_results = state.get("web_results", [])
        
        logger.info("-" * 80)
        logger.info("Stage 4: Report Writing (Basic Workflow)")
        logger.info("-" * 80)
        logger.info(f"  Sections to generate: {PARALLEL_GENERATION_CONFIG['sections']}")
        
        sections = {}
        section_names = PARALLEL_GENERATION_CONFIG["sections"]
        
        # Collect references FIRST (before generating sections)
        references = []
        
        # Add vector DB references
        for chunk in retrieved_chunks:
            ref = {
                "id": chunk.get("id", ""),
                "file_name": chunk.get("metadata", {}).get("file_name", "Unknown"),
                "source_type": "vector_db"
            }
            references.append(ref)
        
        # Add web search references (prioritize these for research reports)
        logger.info(f"Collecting {len(web_results)} web results as references")
        for web_result in web_results:
            ref = {
                "title": web_result.get("title", ""),
                "url": web_result.get("url", ""),
                "snippet": web_result.get("snippet", ""),
                "author": web_result.get("author", ""),
                "date": web_result.get("date", ""),
                "publication": web_result.get("publication", web_result.get("domain", "")),
                "domain": web_result.get("domain", ""),
                "source_type": "web_search"
            }
            references.append(ref)
            logger.debug(f"Added reference: {ref.get('title', 'Untitled')} - {ref.get('url', 'No URL')}")
        
        logger.info(f"  References collected: {len(references)} (Vector DB: {len(retrieved_chunks)}, Web: {len(web_results)})")
        
        if len(references) == 0:
            logger.warning("  WARNING: No references collected! Web search may have returned no results.")
        else:
            logger.info("  Reference details:")
            for i, ref in enumerate(references[:5], 1):  # Show first 5
                logger.info(f"    [{i}] {ref.get('title', 'No title')[:60]}...")
                logger.info(f"        URL: {ref.get('url', 'No URL')[:80]}")
        
        logger.info("  Generating sections in parallel...")
        
        # Generate sections in parallel for speed (with references)
        def generate_section_parallel(section_name: str) -> tuple:
            try:
                section_start_time = datetime.now()
                section_content = self.synthesizer.generate_section(
                    section_name=section_name,
                    query=query,
                    context=synthesized_context,
                    references=references  # Pass references to section generation
                )
                section_time = (datetime.now() - section_start_time).total_seconds()
                logger.info(f"  ✓ {section_name} generated ({len(section_content)} chars, {section_time:.2f}s)")
                return (section_name, section_content)
            except Exception as e:
                # Get full traceback
                tb_str = traceback.format_exc()
                
                # Create detailed error message
                error_msg = f"""EXCEPTION in ResearchGraphBuilder._generate_report_node.generate_section_parallel
Timestamp: {datetime.now().isoformat()}
Section Name: {section_name}
Error Type: {type(e).__name__}
Error Message: {str(e)}
Query: {query[:200] if query else 'N/A'}...
Context Length: {len(synthesized_context) if synthesized_context else 0} characters
Traceback:
{tb_str}"""
                
                # Log to file
                try:
                    write_to_log(error_msg, "EXCEPTION")
                    logger.error(f"Error generating {section_name}: {type(e).__name__}: {str(e)}")
                    logger.error(f"Traceback:\n{tb_str}")
                except Exception as log_error:
                    # Fallback logging if write_to_log fails
                    logger.error(f"Error generating {section_name}: {type(e).__name__}: {str(e)}")
                    logger.error(f"Traceback:\n{tb_str}")
                    print(f"Failed to write exception log: {log_error}")
                
                return (section_name, f"[Error generating {section_name}: {type(e).__name__} - {str(e)}]")
        
        # Use ThreadPoolExecutor for parallel generation
        with ThreadPoolExecutor(max_workers=PARALLEL_GENERATION_CONFIG["max_workers"]) as executor:
            future_to_section = {
                executor.submit(generate_section_parallel, section_name): section_name 
                for section_name in section_names
            }
            
            for future in as_completed(future_to_section):
                section_name, section_content = future.result()
                sections[section_name] = section_content
        
        logger.info(f"Stage 4 Complete: All {len(sections)} sections generated")
        logger.info("-" * 80)
        
        state["sections"] = sections
        state["references"] = references
        return state
    
    def _format_output_node(self, state: ResearchState) -> ResearchState:
        """Format output node - formats final report."""
        query = state.get("query", "")
        sections = state.get("sections", {})
        references = state.get("references", [])
        
        logger.info("-" * 80)
        logger.info("Stage 5: Report Formatting (Basic Workflow)")
        logger.info("-" * 80)
        logger.info(f"  Formatting report with {len(sections)} sections and {len(references)} references...")
        
        try:
            report = self.report_formatter.format_report(
                query=query,
                sections=sections,
                references=references,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "task_type": state.get("task_type", "research")
                }
            )
            logger.info(f"  Report formatted successfully")
            logger.info(f"  Report markdown length: {len(report.get('markdown', ''))} chars")
            logger.info(f"  Report sections: {list(sections.keys())}")
            logger.info(f"  Report references: {len(references)}")
            logger.info("Stage 5 Complete: Report formatted")
            logger.info("-" * 80)
            state["report"] = report
        except Exception as e:
            logger.error(f"  ERROR: Report formatting error: {e}")
            logger.error(f"  Traceback: {traceback.format_exc()}")
            state["report"] = {"error": str(e)}
        
        return state
    
    def run_research(
        self,
        query: str,
        task_type: str = "research",
        use_memory: bool = False,
        memory_context: str = ""
    ) -> Dict[str, Any]:
        """
        Run research workflow using LangGraph or fallback.
        
        Uses configuration from agents/agent_config.py
        
        Args:
            query: Research query
            task_type: Task type (default: "research")
            use_memory: Whether to use conversation memory
            memory_context: Memory context string
            
        Returns:
            Final state dictionary with report
        """
        if not LANGGRAPH_AVAILABLE:
            # Fallback: run workflow without LangGraph
            return self._run_fallback(query, task_type, use_memory, memory_context)
        
        initial_state: ResearchState = {
            "query": query,
            "task_type": task_type,
            "retrieved_chunks": [],
            "web_results": [],
            "memory_context": memory_context if use_memory else "",
            "synthesized_context": "",
            "sections": {},
            "references": [],
            "report": {},
            "metadata": {}
        }
        
        try:
            final_state = self.graph.invoke(initial_state)
            return final_state
        except Exception as e:
            logger.error(f"Graph execution error: {e}")
            return self._run_fallback(query, task_type, use_memory, memory_context)
    
    def _run_fallback(
        self, 
        query: str, 
        task_type: str,
        use_memory: bool = False,
        memory_context: str = ""
    ) -> Dict[str, Any]:
        """Fallback workflow without LangGraph."""
        logger.info("Running fallback workflow (no LangGraph)")
        
        state: ResearchState = {
            "query": query,
            "task_type": task_type,
            "retrieved_chunks": [],
            "web_results": [],
            "memory_context": memory_context if use_memory else "",
            "synthesized_context": "",
            "sections": {},
            "references": [],
            "report": {},
            "metadata": {}
        }
        
        # Run nodes manually
        state = self._retrieve_node(state)
        state = self._synthesize_node(state)
        state = self._generate_report_node(state)
        state = self._format_output_node(state)
        
        return state
    
    def visualize(self) -> Optional[str]:
        """
        Generate visualization of the graph.
        
        Returns:
            Mermaid diagram string or None
        """
        if not LANGGRAPH_AVAILABLE or not self.graph:
            return None
        
        try:
            # Try to get graph visualization
            mermaid_diagram = self.graph.get_graph().draw_mermaid()
            return mermaid_diagram
        except Exception as e:
            logger.warning(f"Could not generate graph visualization: {e}")
            # Return a simple Mermaid diagram
            return """graph TD
    A[Input] --> B[Retrieve]
    B --> C[Synthesize]
    C --> D[Generate Report]
    D --> E[Format Output]
    E --> F[END]
    
    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#e8f5e9
    style D fill:#f3e5f5
    style E fill:#fff9c4
    style F fill:#ffebee"""

