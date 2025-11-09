"""
Research Assistant Agent Module
Handles autonomous and human-in-loop research workflows with multiple use cases.

Key Features:
- Multiple use cases: academic review, news aggregation, documentation analysis, product comparison
- Two operation modes: autonomous and human-in-loop
- Framework-agnostic design (LangChain, AutoGen support)
- Multiple report formats: Markdown, JSON, PDF
- Uses free/open-source retrieval/search only (no paid APIs)

LLM Backend Configuration:
- Supports Ollama (Llama 3.1) - default, free, local
- Supports OpenAI (GPT-3.5-turbo) - requires API key
- Configuration via use_ollama parameter in __init__

Note: Do not call paid proprietary research-agent APIs; use open-source or free retrieval/search options only.
"""

import os
import json
import logging
import traceback
import functools
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from pathlib import Path

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

# Initialize logger
logger = setup_logger(__name__)


def log_exception(func: Callable) -> Callable:
    """
    Decorator to capture and log all exceptions with function name and traceback.
    Logs to file before re-raising the exception.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Get function name and class name if applicable
            func_name = func.__name__
            class_name = ""
            if args and hasattr(args[0], '__class__'):
                class_name = args[0].__class__.__name__ + "."
            
            full_func_name = f"{class_name}{func_name}"
            
            # Get full traceback
            tb_str = traceback.format_exc()
            
            # Get function arguments (safe representation)
            try:
                args_repr = [repr(a) for a in args[1:]]  # Skip self
                kwargs_repr = [f"{k}={repr(v)}" for k, v in kwargs.items()]
                args_str = ", ".join(args_repr + kwargs_repr)
                if len(args_str) > 500:
                    args_str = args_str[:500] + "... (truncated)"
            except:
                args_str = "Unable to represent arguments"
            
            # Create detailed error message
            error_msg = f"""EXCEPTION in {full_func_name}
Timestamp: {datetime.now().isoformat()}
Error Type: {type(e).__name__}
Error Message: {str(e)}
Function Arguments: {args_str}
Traceback:
{tb_str}"""
            
            # Log to file
            try:
                write_to_log(error_msg, "EXCEPTION")
                logger.error(f"Exception in {full_func_name}: {type(e).__name__}: {str(e)}")
            except Exception as log_error:
                # Fallback logging if write_to_log fails
                print(f"Failed to write exception log: {log_error}")
                logger.error(f"Exception in {full_func_name}: {type(e).__name__}: {str(e)}")
                logger.error(f"Traceback:\n{tb_str}")
            
            # Re-raise the exception
            raise
    
    return wrapper


# Import research components
try:
    # Use importlib for modules starting with numbers
    import importlib
    graph_builder_module = importlib.import_module('scripts.05_output_generation.graph_builder')
    ResearchGraphBuilder = graph_builder_module.ResearchGraphBuilder
    
    web_search_module = importlib.import_module('scripts.07_research_workflow.web_search')
    WebSearch = web_search_module.WebSearch
    
    synthesis_module = importlib.import_module('scripts.02_query_completion.synthesis')
    ResearchSynthesizer = synthesis_module.ResearchSynthesizer
    
    report_formatter_module = importlib.import_module('scripts.05_output_generation.report_formatter')
    ReportFormatter = report_formatter_module.ReportFormatter
    
    web_cache_module = importlib.import_module('scripts.07_research_workflow.web_cache')
    WebSearchCache = web_cache_module.WebSearchCache
    
    exporter_module = importlib.import_module('scripts.05_output_generation.exporter')
    ReportExporter = exporter_module.ReportExporter
    
    research_agents_module = importlib.import_module('scripts.07_research_workflow.research_agents')
    EnhancedResearchWorkflow = research_agents_module.EnhancedResearchWorkflow
except ImportError as e:
    logger.warning(f"Research components not available: {e}")
    ResearchGraphBuilder = None
    WebSearch = None
    ResearchSynthesizer = None
    ReportFormatter = None
    WebSearchCache = None
    ReportExporter = None
    EnhancedResearchWorkflow = None


class ResearchAssistant:
    """
    Research Assistant Agent supporting multiple use cases and operation modes.
    
    Use Cases:
    - Academic paper review
    - News aggregation
    - Technical documentation analysis
    - Product comparison
    
    Operation Modes:
    - Autonomous: Plans and executes without human intervention
    - Human-in-loop: Asks for confirmation at key steps
    """
    
    USE_CASES = {
        "academic_paper_review": {
            "name": "Academic Paper Review",
            "description": "Analyze and summarize academic papers with citations",
            "sections": ["Abstract", "Introduction", "Research Findings", "Conclusion", "References"]
        },
        "news_aggregation": {
            "name": "News Aggregation",
            "description": "Gather and synthesize news from multiple sources",
            "sections": ["Abstract", "Introduction", "Research Findings", "Conclusion", "References"]
        },
        "technical_documentation_summary": {
            "name": "Technical Documentation Summary",
            "description": "Analyze technical docs and extract key information",
            "sections": ["Abstract", "Introduction", "Research Findings", "Conclusion", "References"]
        },
        "product_comparison": {
            "name": "Product Comparison",
            "description": "Compare products, features, and specifications",
            "sections": ["Abstract", "Introduction", "Research Findings", "Conclusion", "References"]
        }
    }
    
    REPORT_FORMATS = ["markdown", "json", "pdf", "docx", "html"]
    
    @log_exception
    def __init__(
        self,
        retrieval_engine,
        use_ollama: bool = True,
        framework: str = "LangChain"
    ):
        """
        Initialize research assistant.
        
        Args:
            retrieval_engine: RAGRetrievalEngine instance (can be None for web-only research)
            use_ollama: Whether to use Ollama for LLM (True) or OpenAI (False)
            framework: Framework to use ("LangChain" or "AutoGen" - AutoGen not implemented yet)
        
        LLM Backend Configuration:
        - If use_ollama=True: Uses Ollama with Llama 3.1 (local, free)
        - If use_ollama=False: Uses OpenAI GPT-3.5-turbo (requires API key)
        - LLM is configured in ResearchSynthesizer initialization
        """
        self.retrieval_engine = retrieval_engine
        self.use_ollama = use_ollama
        self.framework = framework
        
        # Initialize components based on framework
        self.web_search = None
        self.synthesizer = None
        self.report_formatter = None
        self.graph_builder = None
        self.enhanced_workflow = None
        self.exporter = None
        
        # Framework-specific initialization
        try:
            if framework == "LangChain":
                self._initialize_langchain_components()
            elif framework == "AutoGen":
                # Placeholder for AutoGen initialization
                # TODO: Implement AutoGen agent initialization
                logger.warning("AutoGen framework not yet implemented. Using LangChain fallback.")
                self._initialize_langchain_components()
            else:
                logger.warning(f"Unknown framework {framework}. Using LangChain.")
                self._initialize_langchain_components()
        except Exception as e:
            # Exception is already logged by decorator, re-raise
            raise
    
    @log_exception
    def _initialize_langchain_components(self):
        """
        Initialize LangChain-based research components.
        
        This method sets up:
        - WebSearch: Free DuckDuckGo search (no API key needed)
        - ResearchSynthesizer: LLM backend (Ollama or OpenAI)
        - ReportFormatter: Report structure and formatting
        - ResearchGraphBuilder: LangGraph workflow orchestration
        """
        try:
            # Web search (always available, uses free DuckDuckGo)
            # NOTE: Do not use paid proprietary research-agent APIs
            # We use DuckDuckGo which is free and open-source
            if WebSearch:
                # WebSearch will create its own cache internally if use_cache=True
                self.web_search = WebSearch(provider="duckduckgo", use_cache=True)
                logger.info("WebSearch initialized (DuckDuckGo)")
            
            # Synthesizer (uses Ollama or OpenAI)
            # LLM Backend Configuration:
            # - If use_ollama=True: Uses Ollama with Llama 3.1 (local, free)
            # - If use_ollama=False: Uses OpenAI GPT-3.5-turbo (requires OPENAI_API_KEY)
            if ResearchSynthesizer:
                self.synthesizer = ResearchSynthesizer(use_ollama=self.use_ollama)
                llm_backend = "Ollama (Llama 3.1)" if self.use_ollama else "OpenAI (GPT-3.5-turbo)"
                logger.info(f"ResearchSynthesizer initialized with {llm_backend}")
            
            # Report formatter
            if ReportFormatter:
                self.report_formatter = ReportFormatter()
                logger.info("ReportFormatter initialized")
            
            # Exporter
            if ReportExporter:
                self.exporter = ReportExporter()
                logger.info("ReportExporter initialized")
            
            # Graph builder (LangGraph workflow)
            # This orchestrates the research workflow: retrieve -> synthesize -> generate_report
            if ResearchGraphBuilder and self.web_search and self.synthesizer and self.report_formatter:
                self.graph_builder = ResearchGraphBuilder(
                    retrieval_engine=self.retrieval_engine,
                    web_search=self.web_search,
                    synthesizer=self.synthesizer,
                    report_formatter=self.report_formatter,
                    use_ollama=self.use_ollama
                )
                logger.info("ResearchGraphBuilder initialized (LangGraph workflow)")
            
            # Enhanced workflow with specialized agents and validation
            if EnhancedResearchWorkflow and self.web_search and self.synthesizer and self.report_formatter:
                self.enhanced_workflow = EnhancedResearchWorkflow(
                    web_search=self.web_search,
                    synthesizer=self.synthesizer,
                    report_formatter=self.report_formatter,
                    retrieval_engine=self.retrieval_engine,
                    max_iterations=3
                )
                logger.info("EnhancedResearchWorkflow initialized (with validation)")
        except Exception as e:
            # Exception is already logged by decorator, just re-raise
            raise
    
    @log_exception
    def _initialize_autogen_components(self):
        """
        Initialize AutoGen-based research components.
        
        Placeholder for future AutoGen implementation.
        TODO: Implement AutoGen agent initialization when AutoGen is integrated.
        """
        logger.warning("AutoGen initialization not implemented yet")
        # Placeholder for AutoGen agent setup
        # Would initialize AutoGen agents here
    
    @log_exception
    def generate_research_plan(
        self,
        query: str,
        use_case: str
    ) -> Dict[str, Any]:
        """
        Generate a research plan based on query and use case.
        
        Args:
            query: Research query
            use_case: Use case type (academic_review, news_aggregation, etc.)
            
        Returns:
            Research plan dictionary
        """
        use_case_config = self.USE_CASES.get(use_case, self.USE_CASES["academic_paper_review"])
        
        plan = {
            "query": query,
            "use_case": use_case,
            "use_case_name": use_case_config["name"],
            "steps": [
                "1. Query formulation and expansion",
                "2. Web search execution",
                "3. Source validation and filtering",
                "4. Information synthesis",
                "5. Report generation"
            ],
            "sections": use_case_config["sections"],
            "estimated_sources": 8,
            "estimated_time": "2-5 minutes"
        }
        
        return plan
    
    @log_exception
    def run_autonomous_research(
        self,
        query: str,
        use_case: str,
        use_memory: bool = False,
        memory_context: str = ""
    ) -> Dict[str, Any]:
        """
        Run research in autonomous mode - no human intervention.
        
        Args:
            query: Research query
            use_case: Use case type
            use_memory: Whether to use conversation memory
            memory_context: Memory context string
            
        Returns:
            Research result dictionary
        """
        # ========== STAGE 1: USER QUERY RECEIVED ==========
        logger.info("=" * 80)
        logger.info("REPORT GENERATION STARTED")
        logger.info("=" * 80)
        logger.info(f"Stage 1: User Query Received")
        logger.info(f"  Query: {query}")
        logger.info(f"  Use Case: {use_case}")
        logger.info(f"  Use Memory: {use_memory}")
        logger.info(f"  Memory Context Length: {len(memory_context)} chars")
        logger.info(f"  Timestamp: {datetime.now().isoformat()}")
        logger.info("-" * 80)
        
        # Use enhanced workflow if available, otherwise fall back to graph builder
        if self.enhanced_workflow:
            logger.info("Stage 1: Workflow Selection: Enhanced Workflow (with validation)")
            try:
                # Generate research plan
                logger.info("Stage 1: Generating research plan...")
                plan = self.generate_research_plan(query, use_case)
                logger.info(f"Stage 1: Research plan generated: {plan.get('name', 'Unknown')}")
                logger.info(f"  Estimated sources: {plan.get('estimated_sources', 'N/A')}")
                logger.info(f"  Estimated time: {plan.get('estimated_time', 'N/A')}")
                logger.info("-" * 80)
                
                # Execute enhanced workflow with validation
                logger.info("Stage 1: Executing enhanced workflow...")
                result = self.enhanced_workflow.execute(
                    query=query,
                    task_type=use_case,
                    use_memory=use_memory,
                    memory_context=memory_context
                )
                
                logger.info("=" * 80)
                logger.info("REPORT GENERATION COMPLETED")
                logger.info("=" * 80)
                logger.info(f"  Success: True")
                logger.info(f"  Validation Passed: {result.get('validation_passed', False)}")
                logger.info(f"  Iterations: {result.get('iterations', 0)}")
                logger.info(f"  Report Sections: {list(result.get('report', {}).get('sections', {}).keys())}")
                logger.info(f"  References Count: {len(result.get('references', []))}")
                logger.info(f"  Timestamp: {datetime.now().isoformat()}")
                logger.info("=" * 80)
                
                return {
                    "success": True,
                    "plan": plan,
                    "result": result,
                    "report": result.get("report", {}),
                    "validation_passed": result.get("validation_passed", False),
                    "iterations": result.get("iterations", 0)
                }
            except Exception as e:
                logger.error("=" * 80)
                logger.error("ENHANCED WORKFLOW ERROR")
                logger.error("=" * 80)
                logger.error(f"  Error: {str(e)}")
                logger.error(f"  Traceback: {traceback.format_exc()}")
                logger.error("  Falling back to basic workflow...")
                logger.error("=" * 80)
                # Fall back to basic workflow
                if self.graph_builder:
                    return self._run_basic_workflow(query, use_case, use_memory, memory_context)
                else:
                    logger.error("  No fallback workflow available!")
                    return {
                        "success": False,
                        "error": str(e)
                    }
        elif self.graph_builder:
            logger.info("Stage 1: Workflow Selection: Basic Workflow (no validation)")
            return self._run_basic_workflow(query, use_case, use_memory, memory_context)
        else:
            return {
                "success": False,
                "error": "Research components not initialized"
            }
    
    def _run_basic_workflow(self, query: str, use_case: str, use_memory: bool, memory_context: str) -> Dict[str, Any]:
        """Run basic workflow as fallback."""
        try:
            logger.info("Stage 1: Generating research plan (basic workflow)...")
            # Generate research plan
            plan = self.generate_research_plan(query, use_case)
            logger.info(f"Stage 1: Research plan generated: {plan.get('name', 'Unknown')}")
            logger.info("-" * 80)
            
            # Execute research workflow
            logger.info("Stage 1: Executing basic workflow...")
            result = self.graph_builder.run_research(
                query=query,
                task_type=use_case,
                use_memory=use_memory,
                memory_context=memory_context
            )
            
            logger.info("=" * 80)
            logger.info("REPORT GENERATION COMPLETED (Basic Workflow)")
            logger.info("=" * 80)
            logger.info(f"  Success: True")
            logger.info(f"  Report Sections: {list(result.get('report', {}).get('sections', {}).keys())}")
            logger.info(f"  References Count: {len(result.get('references', []))}")
            logger.info(f"  Timestamp: {datetime.now().isoformat()}")
            logger.info("=" * 80)
            
            return {
                "success": True,
                "plan": plan,
                "result": result,
                "report": result.get("report", {})
            }
        except Exception as e:
            # Exception is already logged by decorator
            return {
                "success": False,
                "error": str(e)
            }
    
    @log_exception
    def run_human_in_loop_research(
        self,
        query: str,
        use_case: str,
        ask_user_confirmation: Callable,
        use_memory: bool = False,
        memory_context: str = "",
        pending_confirmations: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run research in human-in-loop mode - asks for confirmation at key steps.
        
        This method supports Streamlit's stateless execution by accepting
        pending_confirmations dict to track confirmation state across reruns.
        
        Args:
            query: Research query
            use_case: Use case type
            ask_user_confirmation: Function to ask user for confirmation (returns True/False or None)
            use_memory: Whether to use conversation memory
            memory_context: Memory context string
            pending_confirmations: Dict tracking confirmation state (for Streamlit stateless execution)
            
        Returns:
            Research result dictionary with status and any pending confirmations
        """
        if not self.graph_builder:
            return {
                "success": False,
                "error": "Research components not initialized"
            }
        
        pending_confirmations = pending_confirmations or {}
        
        try:
            # Step 1: Generate and confirm research plan
            if "plan_approved" not in pending_confirmations:
                plan = self.generate_research_plan(query, use_case)
                confirmation = ask_user_confirmation("plan", plan)
                if confirmation is None:
                    # Confirmation pending - return state for user to approve
                    return {
                        "success": False,
                        "status": "pending_confirmation",
                        "step": "plan",
                        "plan": plan,
                        "pending_confirmations": pending_confirmations
                    }
                elif not confirmation:
                    return {
                        "success": False,
                        "error": "User rejected research plan"
                    }
                pending_confirmations["plan_approved"] = True
            
            # Step 2: Confirm retrieval strategy
            if "retrieval_approved" not in pending_confirmations:
                plan = self.generate_research_plan(query, use_case)
                confirmation = ask_user_confirmation("retrieval", {
                    "query": query,
                    "sources": plan["estimated_sources"],
                    "use_case": use_case
                })
                if confirmation is None:
                    return {
                        "success": False,
                        "status": "pending_confirmation",
                        "step": "retrieval",
                        "pending_confirmations": pending_confirmations
                    }
                elif not confirmation:
                    return {
                        "success": False,
                        "error": "User rejected retrieval step"
                    }
                pending_confirmations["retrieval_approved"] = True
            
            # Step 3: Execute research workflow
            result = self.graph_builder.run_research(
                query=query,
                task_type=use_case,
                use_memory=use_memory,
                memory_context=memory_context
            )
            
            # Step 4: Review draft report
            draft_report = result.get("report", {})
            if "draft_approved" not in pending_confirmations:
                confirmation = ask_user_confirmation("draft", draft_report)
                if confirmation is None:
                    return {
                        "success": False,
                        "status": "pending_confirmation",
                        "step": "draft",
                        "draft": draft_report,
                        "pending_confirmations": pending_confirmations
                    }
                elif not confirmation:
                    return {
                        "success": False,
                        "error": "User rejected draft report",
                        "draft": draft_report
                    }
                pending_confirmations["draft_approved"] = True
            
            return {
                "success": True,
                "plan": self.generate_research_plan(query, use_case),
                "result": result,
                "report": draft_report
            }
        except Exception as e:
            # Exception is already logged by decorator
            return {
                "success": False,
                "error": str(e)
            }
    
    @log_exception
    def format_report(
        self,
        report: Dict[str, Any],
        format_type: str = "markdown"
    ) -> str:
        """
        Format report in requested format.
        
        Args:
            report: Report dictionary
            format_type: Format type (markdown, json, pdf, docx, html)
            
        Returns:
            Formatted report string or file path for binary formats
        """
        if format_type == "json":
            return json.dumps(report, indent=2, ensure_ascii=False)
        elif format_type == "markdown":
            return report.get("markdown", report.get("plain_text", ""))
        elif format_type in ["pdf", "docx", "html"]:
            # For binary formats, return the file path
            if not self.exporter:
                logger.warning(f"Exporter not available for {format_type} format")
                return report.get("markdown", report.get("plain_text", ""))
            
            try:
                if format_type == "docx":
                    file_path = self.exporter.export_docx(report)
                elif format_type == "pdf":
                    file_path = self.exporter.export_pdf(report)
                elif format_type == "html":
                    file_path = self.exporter.export_html(report)
                else:
                    return report.get("markdown", report.get("plain_text", ""))
                
                logger.info(f"Exported {format_type} to {file_path}")
                return file_path
            except Exception as e:
                logger.error(f"Export error for {format_type}: {e}")
                # Fallback to markdown
                return report.get("markdown", report.get("plain_text", ""))
        else:
            return report.get("markdown", report.get("plain_text", ""))
    
    @log_exception
    def generate_pdf_report(
        self,
        report: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate PDF report from markdown content.
        
        Args:
            report: Report dictionary
            output_path: Optional output file path
            
        Returns:
            Path to generated PDF file
        """
        if not self.exporter:
            logger.warning("Exporter not available for PDF generation")
            return ""
        
        try:
            return self.exporter.export_pdf(report, output_path)
        except Exception as e:
            logger.error(f"PDF generation error: {e}")
            return ""


def ask_user_confirmation_placeholder(
    step: str,
    data: Any,
    st_session=None
) -> bool:
    """
    Placeholder function for asking user confirmation.
    
    Actual implementation in streamlit_app.py uses Streamlit UI elements.
    This placeholder is for testing or non-Streamlit environments.
    
    Args:
        step: Step name (plan, retrieval, draft)
        data: Data to show user
        st_session: Streamlit session (optional, not used in placeholder)
        
    Returns:
        True if user confirms, False otherwise
    """
    # This is a placeholder - actual implementation in streamlit_app.py
    logger.info(f"User confirmation requested for step: {step}")
    logger.debug(f"Confirmation data: {data}")
    # Default to True for testing
    return True

