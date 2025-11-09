"""
Synthesis Module
Handles LLM-based synthesis and reasoning for research.
"""

import os
import logging
import traceback
from typing import List, Dict, Optional, Any
from datetime import datetime
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

# Try to import LLM providers
try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ResearchSynthesizer:
    """Synthesizes research information using LLMs."""
    
    def __init__(self, use_ollama: bool = True, model_name: Optional[str] = None):
        """
        Initialize the synthesizer.
        
        Args:
            use_ollama: If True, use Ollama (local), else use OpenAI
            model_name: Model name (defaults based on provider)
        """
        try:
            self.use_ollama = use_ollama
            self.llm = None
            
            if use_ollama and OLLAMA_AVAILABLE:
                try:
                    model = model_name or "llama3.1:latest"
                    self.llm = ChatOllama(
                        model=model,
                        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                        temperature=0.7,
                        num_ctx=4096  # Limit context window for speed
                    )
                    logger.info(f"Initialized Ollama synthesizer with {model}")
                except Exception as e:
                    tb_str = traceback.format_exc()
                    error_msg = f"""EXCEPTION in ResearchSynthesizer.__init__ (Ollama initialization)
Timestamp: {datetime.now().isoformat()}
Error Type: {type(e).__name__}
Error Message: {str(e)}
Model: {model}
Traceback:
{tb_str}"""
                    try:
                        write_to_log(error_msg, "EXCEPTION")
                        logger.warning(f"Failed to initialize Ollama: {e}")
                    except:
                        logger.warning(f"Failed to initialize Ollama: {e}")
                    self._init_openai()
            else:
                self._init_openai()
        except Exception as e:
            tb_str = traceback.format_exc()
            error_msg = f"""EXCEPTION in ResearchSynthesizer.__init__
Timestamp: {datetime.now().isoformat()}
Error Type: {type(e).__name__}
Error Message: {str(e)}
Traceback:
{tb_str}"""
            try:
                write_to_log(error_msg, "EXCEPTION")
                logger.error(f"Failed to initialize ResearchSynthesizer: {e}")
            except:
                logger.error(f"Failed to initialize ResearchSynthesizer: {e}")
            raise
    
    def _init_openai(self):
        """Initialize OpenAI as fallback."""
        try:
            if OPENAI_AVAILABLE:
                try:
                    api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, 'secrets') else "")
                    if api_key:
                        self.llm = ChatOpenAI(
                            model="gpt-3.5-turbo",  # Faster than gpt-4, still good quality
                            temperature=0.7,
                            openai_api_key=api_key,
                            max_tokens=1000  # Limit tokens for speed
                        )
                        logger.info("Initialized OpenAI synthesizer (gpt-3.5-turbo for speed)")
                    else:
                        logger.warning("OpenAI API key not found")
                except Exception as e:
                    tb_str = traceback.format_exc()
                    error_msg = f"""EXCEPTION in ResearchSynthesizer._init_openai
Timestamp: {datetime.now().isoformat()}
Error Type: {type(e).__name__}
Error Message: {str(e)}
Traceback:
{tb_str}"""
                    try:
                        write_to_log(error_msg, "EXCEPTION")
                        logger.error(f"Failed to initialize OpenAI: {e}")
                    except:
                        logger.error(f"Failed to initialize OpenAI: {e}")
        except Exception as e:
            tb_str = traceback.format_exc()
            error_msg = f"""EXCEPTION in ResearchSynthesizer._init_openai (outer)
Timestamp: {datetime.now().isoformat()}
Error Type: {type(e).__name__}
Error Message: {str(e)}
Traceback:
{tb_str}"""
            try:
                write_to_log(error_msg, "EXCEPTION")
                logger.error(f"Failed to initialize OpenAI (outer): {e}")
            except:
                logger.error(f"Failed to initialize OpenAI (outer): {e}")
    
    def synthesize(self, query: str, retrieved_chunks: List[Dict], web_results: List[Dict] = None, memory_context: str = "") -> str:
        """
        Synthesize information from retrieved chunks and web results.
        
        Args:
            query: Research query
            retrieved_chunks: Retrieved chunks from vector DB
            web_results: Web search results (optional)
            memory_context: Memory context string
            
        Returns:
            Synthesized text
        """
        try:
            if not self.llm:
                return "LLM not available. Please configure Ollama or OpenAI."
            
            # Import synthesis config for context limits
            try:
                from agents.agent_config import SYNTHESIS_CONFIG
                config = SYNTHESIS_CONFIG["context_limits"]
            except ImportError:
                # Fallback to hardcoded values
                config = {
                    "memory_context": 500,
                    "vector_db_chunks": 3,
                    "vector_db_chunk_length": 250,
                    "web_results": 5,
                    "web_snippet_length": 150
                }
            
            # Prepare context (optimized - limit size for speed)
            context_parts = []
            
            # Add memory context if provided (truncated)
            if memory_context:
                # Limit memory context to configured length
                memory_limit = config.get("memory_context", 500)
                memory_truncated = memory_context[-memory_limit:] if len(memory_context) > memory_limit else memory_context
                context_parts.append("## Conversation Context:\n" + memory_truncated + "\n")
            
            # Add vector DB results (limit to configured number)
            if retrieved_chunks:
                context_parts.append("## Internal Knowledge Base:\n")
                chunk_limit = config.get("vector_db_chunks", 3)
                chunk_length = config.get("vector_db_chunk_length", 250)
                for i, chunk in enumerate(retrieved_chunks[:chunk_limit], 1):
                    text = chunk.get("text", "")
                    metadata = chunk.get("metadata", {})
                    source = metadata.get("file_name", "Unknown")
                    context_parts.append(f"{i}. [{source}] {text[:chunk_length]}...\n")
            
            # Add web results (limit to configured number and length)
            if web_results:
                context_parts.append("\n## Web Sources:\n")
                web_limit = config.get("web_results", 5)
                snippet_length = config.get("web_snippet_length", 150)
                for i, result in enumerate(web_results[:web_limit], 1):
                    title = result.get("title", "")
                    snippet = result.get("snippet", "")
                    url = result.get("url", "")
                    # Truncate snippet to configured length
                    snippet_short = snippet[:snippet_length] + "..." if len(snippet) > snippet_length else snippet
                    context_parts.append(f"{i}. **{title}** ({url})\n   {snippet_short}\n")
            
            context = "\n".join(context_parts)
            
            # Import synthesis prompt template
            try:
                from agents.prompts import SYNTHESIS_PROMPT_TEMPLATE
            except ImportError:
                # Fallback prompt template
                logger.warning("Failed to import SYNTHESIS_PROMPT_TEMPLATE, using fallback")
                SYNTHESIS_PROMPT_TEMPLATE = """Research Question: {query}

Context:
{context}

Synthesize key findings, insights, and patterns. Be concise but comprehensive. Synthesis:"""
            
            # Create synthesis prompt using template
            prompt = SYNTHESIS_PROMPT_TEMPLATE.format(query=query, context=context)
            
            # Use streaming if available for faster perceived performance
            # For now, use invoke (can be optimized with streaming later)
            response = self.llm.invoke(prompt)
            if hasattr(response, 'content'):
                return response.content
            return str(response)
        except Exception as e:
            # Get full traceback
            tb_str = traceback.format_exc()
            
            # Create detailed error message
            error_msg = f"""EXCEPTION in ResearchSynthesizer.synthesize
Timestamp: {datetime.now().isoformat()}
Error Type: {type(e).__name__}
Error Message: {str(e)}
Query: {query[:200] if query else 'N/A'}...
Retrieved Chunks: {len(retrieved_chunks) if retrieved_chunks else 0}
Web Results: {len(web_results) if web_results else 0}
Memory Context Length: {len(memory_context) if memory_context else 0} characters
Traceback:
{tb_str}"""
            
            # Log to file
            try:
                write_to_log(error_msg, "EXCEPTION")
                logger.error(f"Error in synthesize: {type(e).__name__}: {str(e)}")
                logger.error(f"Traceback:\n{tb_str}")
            except Exception as log_error:
                # Fallback logging if write_to_log fails
                logger.error(f"Error in synthesize: {type(e).__name__}: {str(e)}")
                logger.error(f"Traceback:\n{tb_str}")
                print(f"Failed to write exception log: {log_error}")
            
            # Return error message
            return f"[Error during synthesis: {type(e).__name__} - {str(e)}]"
    
    def generate_section(self, section_name: str, query: str, context: str, references: Optional[List[Dict]] = None) -> str:
        """
        Generate a specific section of a research report.
        
        Args:
            section_name: Name of the section (e.g., "Abstract", "Introduction")
            query: Research query
            context: Context information
            references: Optional list of references to cite
            
        Returns:
            Generated section text
        """
        references = references or []
        try:
            if not self.llm:
                return f"[{section_name} section - LLM not available]"
            
            # Import prompt templates and context limits
            try:
                from agents.prompts import (
                    SECTION_PROMPTS, 
                    SECTION_CONTEXT_MAP, 
                    CONTEXT_LIMITS, 
                    DEFAULT_SECTION_PROMPT
                )
            except ImportError:
                # Fallback if prompts module not available
                logger.warning("Failed to import prompts from agents.prompts, using fallback")
                
                # Format references for prompt
                references_text = ""
                if references:
                    ref_list = []
                    for i, ref in enumerate(references, 1):
                        title = ref.get("title", "Untitled")
                        url = ref.get("url", "")
                        ref_list.append(f"[{i}] {title} - {url}")
                    references_text = "\n\nAvailable References:\n" + "\n".join(ref_list) + "\n\nIMPORTANT: Only cite references that are listed above. Do NOT create fake citations like [1], [2] if no references are available. If no references are available, write without citations."
                else:
                    references_text = "\n\nIMPORTANT: No references are available. Write the section WITHOUT any citations like [1], [2], etc. Do not create fake citations."
                
                SECTION_PROMPTS = {
                    "Abstract": "Write an Abstract section for: {query}\n\nContext: {context}{references_text}\n\nAbstract:",
                    "Introduction": "Write an Introduction section for: {query}\n\nContext: {context}{references_text}\n\nIntroduction:",
                    "Research Findings": "Write a Research Findings section for: {query}\n\nContext: {context}{references_text}\n\nResearch Findings:",
                    "Conclusion": "Write a Conclusion section for: {query}\n\nContext: {context}{references_text}\n\nConclusion:"
                }
                SECTION_CONTEXT_MAP = {"Abstract": "short", "Introduction": "short", "Research Findings": "long", "Conclusion": "short"}
                CONTEXT_LIMITS = {"short": 600, "long": 1200}
                DEFAULT_SECTION_PROMPT = "Write a {section_name} section for: {query}\n\nContext: {context}{references_text}\n\n{section_name}:"
            
            # Get context limit for this section type
            context_type = SECTION_CONTEXT_MAP.get(section_name, "short")
            context_limit = CONTEXT_LIMITS.get(context_type, 600)
            
            # Truncate context based on section type
            if context_limit > 0:
                truncated_context = context[:context_limit]
            else:
                truncated_context = ""  # No context for Method Overview
            
            # Format references for prompt
            references_text = ""
            if references:
                ref_list = []
                for i, ref in enumerate(references, 1):
                    title = ref.get("title", "Untitled")
                    url = ref.get("url", "")
                    author = ref.get("author", "")
                    date = ref.get("date", "")
                    ref_info = f"[{i}] {title}"
                    if url:
                        ref_info += f" - {url}"
                    if author:
                        ref_info += f" (Author: {author})"
                    if date:
                        ref_info += f" ({date})"
                    ref_list.append(ref_info)
                references_text = "\n\nAvailable References:\n" + "\n".join(ref_list) + "\n\nIMPORTANT: Only cite references that are listed above using [N] format. Do NOT create fake citations."
            else:
                references_text = "\n\nIMPORTANT: No references are available. Write the section WITHOUT any citations like [1], [2], etc. Do not create fake citations."
            
            # Get prompt template for this section
            prompt_template = SECTION_PROMPTS.get(section_name, DEFAULT_SECTION_PROMPT)
            
            # Format prompt with query, context, section_name, and references
            if context_limit > 0:
                prompt = prompt_template.format(
                    section_name=section_name, 
                    query=query, 
                    context=truncated_context,
                    references_text=references_text
                )
            else:
                # Method Overview doesn't use context
                prompt = prompt_template.format(
                    section_name=section_name, 
                    query=query,
                    references_text=references_text
                )
            
            # Log prompt details (truncated for readability)
            logger.debug(f"  Generating {section_name} section")
            logger.debug(f"    Prompt length: {len(prompt)} chars")
            logger.debug(f"    References provided: {len(references)}")
            logger.debug(f"    Context length: {len(truncated_context)} chars")
            
            # Optimize: shorter timeout, faster response
            response = self.llm.invoke(prompt)
            if hasattr(response, 'content'):
                result = response.content
                logger.debug(f"    Section generated: {len(result)} chars")
                return result
            result = str(response)
            logger.debug(f"    Section generated: {len(result)} chars")
            return result
        except Exception as e:
            # Get full traceback
            tb_str = traceback.format_exc()
            
            # Create detailed error message
            error_msg = f"""EXCEPTION in ResearchSynthesizer.generate_section
Timestamp: {datetime.now().isoformat()}
Section Name: {section_name}
Error Type: {type(e).__name__}
Error Message: {str(e)}
Query: {query[:200] if query else 'N/A'}...
Context Length: {len(context) if context else 0} characters
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
            
            return f"[Error generating {section_name} section: {type(e).__name__} - {str(e)}]"

