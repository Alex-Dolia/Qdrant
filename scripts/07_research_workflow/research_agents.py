"""
Enhanced Research Agents Module
Specialized agents for the research workflow with validation and quality control.
"""

import os
import logging
import traceback
from typing import Dict, List, Optional, Any, Literal
from abc import ABC, abstractmethod
from datetime import datetime
import re

logger = logging.getLogger(__name__)

# Research State Definition
from typing import TypedDict

class ResearchState(TypedDict):
    """Enhanced state for research workflow with validation."""
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
    current_step: Literal["web_research", "analysis", "writing", "validation", "complete"]
    report_draft: str
    feedback: Optional[str]
    iterations: int
    validation_passed: bool


class BaseAgent(ABC):
    """Base class for all research agents."""
    
    @abstractmethod
    def run(self, state: ResearchState) -> ResearchState:
        """Process state and return updated state."""
        pass
    
    def _log_step(self, step_name: str, state: ResearchState):
        """Log agent step execution."""
        logger.info(f"{step_name}: query='{state.get('query', '')[:50]}...', step={state.get('current_step')}")


class WebResearchAgent(BaseAgent):
    """Agent for web research using DuckDuckGo."""
    
    def __init__(self, web_search, max_results: int = 10):
        """
        Initialize web research agent.
        
        Args:
            web_search: WebSearch instance
            max_results: Maximum number of results to retrieve
        """
        self.web_search = web_search
        self.max_results = max_results
    
    def run(self, state: ResearchState) -> ResearchState:
        """Execute targeted web searches with relevance filtering."""
        self._log_step("WebResearchAgent", state)
        
        query = state.get("query", "")
        web_results = []
        
        logger.info("-" * 80)
        logger.info("Stage 2: Web Research")
        logger.info("-" * 80)
        logger.info(f"  Query: {query}")
        logger.info(f"  Max Results: {self.max_results}")
        
        try:
            if self.web_search and self.web_search.is_available():
                logger.info("  Web search provider: Available")
                # Perform web search
                logger.info("  Performing web search...")
                raw_results = self.web_search.search(query, max_results=self.max_results)
                logger.info(f"  Raw results retrieved: {len(raw_results)}")
                
                # Filter and enhance results
                logger.info("  Filtering and enhancing results...")
                web_results = self._filter_results(raw_results)
                
                logger.info(f"  Filtered results: {len(web_results)}")
                if web_results:
                    logger.info("  Sample results:")
                    for i, result in enumerate(web_results[:3], 1):
                        logger.info(f"    [{i}] {result.get('title', 'No title')[:60]}...")
                        logger.info(f"        URL: {result.get('url', 'No URL')[:80]}")
                else:
                    logger.warning("  WARNING: No web results after filtering!")
            else:
                logger.warning("  Web search provider: NOT AVAILABLE")
                logger.warning("  Install duckduckgo-search: pip install duckduckgo-search")
                state["feedback"] = "Web search not available. Install duckduckgo-search."
        
        except Exception as e:
            error_msg = f"Web search failed: {str(e)}"
            logger.error(f"  ERROR: {error_msg}")
            logger.error(f"  Traceback: {traceback.format_exc()}")
            state["feedback"] = error_msg
        
        logger.info(f"Stage 2 Complete: {len(web_results)} web results collected")
        logger.info("-" * 80)
        
        state["web_results"] = web_results
        state["current_step"] = "analysis"
        return state
    
    def _filter_results(self, raw_results: List[Dict]) -> List[Dict]:
        """Remove low-quality sources and extract key insights."""
        filtered = []
        
        for result in raw_results:
            # Quality checks
            url = result.get("url", "")
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            
            # Skip if missing essential data
            if not url or not title:
                continue
            
            # Skip low-quality domains (optional - can be configured)
            low_quality_domains = ["spam.com", "advertising.com"]
            if any(domain in url.lower() for domain in low_quality_domains):
                continue
            
            # Require minimum snippet length
            if snippet and len(snippet) < 20:
                continue
            
            filtered.append(result)
        
        return filtered


class TechnicalAnalystAgent(BaseAgent):
    """Agent for synthesizing web insights and technical analysis."""
    
    def __init__(self, synthesizer):
        """
        Initialize technical analyst agent.
        
        Args:
            synthesizer: ResearchSynthesizer instance
        """
        self.synthesizer = synthesizer
    
    def run(self, state: ResearchState) -> ResearchState:
        """Synthesize web results into coherent technical analysis."""
        self._log_step("TechnicalAnalystAgent", state)
        
        query = state.get("query", "")
        web_results = state.get("web_results", [])
        retrieved_chunks = state.get("retrieved_chunks", [])
        memory_context = state.get("memory_context", "")
        
        logger.info("-" * 80)
        logger.info("Stage 3: Technical Analysis & Synthesis")
        logger.info("-" * 80)
        logger.info(f"  Query: {query}")
        logger.info(f"  Web Results Count: {len(web_results)}")
        logger.info(f"  Retrieved Chunks Count: {len(retrieved_chunks)}")
        logger.info(f"  Memory Context Length: {len(memory_context)} chars")
        
        try:
            logger.info("  Starting synthesis...")
            # Synthesize information
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
            state["current_step"] = "writing"
            logger.info("Stage 3 Complete: Context synthesized")
            logger.info("-" * 80)
        
        except Exception as e:
            error_msg = f"Synthesis failed: {str(e)}"
            logger.error(f"  ERROR: {error_msg}")
            logger.error(f"  Traceback: {traceback.format_exc()}")
            state["feedback"] = error_msg
            state["current_step"] = "validation"
        
        return state


class ReportWriterAgent(BaseAgent):
    """Agent for generating structured reports."""
    
    def __init__(self, synthesizer, report_formatter):
        """
        Initialize report writer agent.
        
        Args:
            synthesizer: ResearchSynthesizer instance
            report_formatter: ReportFormatter instance
        """
        self.synthesizer = synthesizer
        self.report_formatter = report_formatter
    
    def run(self, state: ResearchState) -> ResearchState:
        """Generate report sections in parallel."""
        self._log_step("ReportWriterAgent", state)
        
        query = state.get("query", "")
        synthesized_context = state.get("synthesized_context", "")
        web_results = state.get("web_results", [])
        retrieved_chunks = state.get("retrieved_chunks", [])
        
        sections = {}
        section_names = ["Abstract", "Introduction", "Research Findings", "Conclusion"]
        
        # Collect references FIRST (before generating sections)
        references = []
        logger.info(f"ReportWriterAgent: Processing {len(web_results)} web results for references")
        
        for i, web_result in enumerate(web_results):
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
            logger.debug(f"Reference {i+1}: {ref.get('title', 'No title')} - {ref.get('url', 'No URL')}")
        
        logger.info(f"  References collected: {len(references)}")
        
        if len(references) == 0:
            logger.warning("  WARNING: No references collected! Web search may have returned no results.")
        else:
            logger.info("  Reference details:")
            for i, ref in enumerate(references[:5], 1):  # Show first 5
                logger.info(f"    [{i}] {ref.get('title', 'No title')[:60]}...")
                logger.info(f"        URL: {ref.get('url', 'No URL')[:80]}")
        
        logger.info("-" * 80)
        logger.info("Stage 4: Report Writing")
        logger.info("-" * 80)
        logger.info(f"  Sections to generate: {section_names}")
        logger.info(f"  References available: {len(references)}")
        
        try:
            # Generate sections with reference information
            for section_name in section_names:
                try:
                    logger.info(f"  Generating {section_name} section...")
                    section_start_time = datetime.now()
                    
                    section_content = self.synthesizer.generate_section(
                        section_name=section_name,
                        query=query,
                        context=synthesized_context,
                        references=references  # Pass references to section generation
                    )
                    
                    section_time = (datetime.now() - section_start_time).total_seconds()
                    sections[section_name] = section_content
                    logger.info(f"  ✓ {section_name} generated ({len(section_content)} chars, {section_time:.2f}s)")
                except Exception as e:
                    logger.error(f"  ✗ Error generating {section_name}: {e}")
                    sections[section_name] = f"[Error generating {section_name}: {str(e)}]"
            
            logger.info(f"Stage 4 Complete: All {len(sections)} sections generated")
            logger.info("-" * 80)
            
            # Format report
            logger.info("Stage 5: Report Formatting")
            logger.info("-" * 80)
            logger.info("  Formatting report with sections and references...")
            
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
            
            state["sections"] = sections
            state["references"] = references
            state["report"] = report
            state["report_draft"] = report.get("markdown", "")
            state["current_step"] = "validation"
        
        except Exception as e:
            error_msg = f"Report generation failed: {str(e)}"
            logger.error(f"ReportWriterAgent error: {error_msg}")
            state["feedback"] = error_msg
            state["current_step"] = "validation"
        
        return state


class ValidationAgent(BaseAgent):
    """Agent for quality control and validation."""
    
    def __init__(self, synthesizer, max_iterations: int = 3):
        """
        Initialize validation agent.
        
        Args:
            synthesizer: ResearchSynthesizer instance (for re-generation if needed)
            max_iterations: Maximum validation iterations
        """
        self.synthesizer = synthesizer
        self.max_iterations = max_iterations
    
    def run(self, state: ResearchState) -> ResearchState:
        """Validate report quality and provide feedback."""
        self._log_step("ValidationAgent", state)
        
        iterations = state.get("iterations", 0)
        report_draft = state.get("report_draft", "")
        sections = state.get("sections", {})
        references = state.get("references", [])
        
        logger.info("-" * 80)
        logger.info(f"Stage 6: Validation (Iteration {iterations + 1}/{self.max_iterations})")
        logger.info("-" * 80)
        logger.info(f"  Report draft length: {len(report_draft)} chars")
        logger.info(f"  Sections: {list(sections.keys())}")
        logger.info(f"  References count: {len(references)}")
        
        # Validation checks
        logger.info("  Running validation checks...")
        validation_results = self._validate_report(
            report_draft=report_draft,
            sections=sections,
            references=references
        )
        
        logger.info(f"  Validation passed: {validation_results['passed']}")
        if validation_results.get("issues"):
            logger.info(f"  Issues found: {len(validation_results['issues'])}")
            for issue in validation_results["issues"]:
                logger.warning(f"    - {issue}")
        else:
            logger.info("  ✓ All validation checks passed")
        
        state["validation_passed"] = validation_results["passed"]
        state["iterations"] = iterations + 1
        
        if validation_results["passed"]:
            state["current_step"] = "complete"
            logger.info("Stage 6 Complete: Report passed validation")
            logger.info("-" * 80)
        elif iterations >= self.max_iterations:
            state["current_step"] = "complete"
            state["feedback"] = f"Validation failed after {iterations} iterations. Proceeding with current draft."
            logger.warning(f"Stage 6 Complete: Max iterations ({self.max_iterations}) reached, proceeding anyway")
            logger.info("-" * 80)
        else:
            # Provide feedback for improvement
            state["feedback"] = validation_results["feedback"]
            state["current_step"] = "writing"  # Loop back to writing
            logger.info(f"Stage 6: Report needs improvement - {validation_results['feedback']}")
            logger.info("  Looping back to Stage 4 (Report Writing) for improvement...")
            logger.info("-" * 80)
        
        return state
    
    def _validate_report(
        self,
        report_draft: str,
        sections: Dict[str, str],
        references: List[Dict]
    ) -> Dict[str, Any]:
        """Perform quality checks on the report."""
        issues = []
        
        # Check 1: All sections present
        required_sections = ["Abstract", "Introduction", "Research Findings", "Conclusion"]
        missing_sections = [s for s in required_sections if s not in sections or not sections[s].strip()]
        if missing_sections:
            issues.append(f"Missing sections: {', '.join(missing_sections)}")
        
        # Check 2: Sections not empty
        for section_name, content in sections.items():
            if not content or content.strip() == "" or content.startswith("[Error"):
                issues.append(f"Section '{section_name}' is empty or has errors")
        
        # Check 3: References present
        if not references:
            issues.append("No references found")
        
        # Check 4: Report length (minimum content)
        if len(report_draft) < 500:
            issues.append("Report is too short (less than 500 characters)")
        
        # Check 5: Check for placeholder text
        placeholder_patterns = ["[insert", "[Error", "TODO", "FIXME", "Not generated"]
        for pattern in placeholder_patterns:
            if pattern.lower() in report_draft.lower():
                issues.append(f"Report contains placeholder text: {pattern}")
        
        # Check 6: URL validation for references
        for ref in references:
            url = ref.get("url", "")
            if url and not self._is_valid_url(url):
                issues.append(f"Invalid URL in references: {url}")
        
        passed = len(issues) == 0
        feedback = "; ".join(issues) if issues else "All validation checks passed"
        
        return {
            "passed": passed,
            "feedback": feedback,
            "issues": issues
        }
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format."""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return url_pattern.match(url) is not None


class EnhancedResearchWorkflow:
    """Enhanced research workflow with specialized agents and validation."""
    
    def __init__(
        self,
        web_search,
        synthesizer,
        report_formatter,
        retrieval_engine=None,
        max_iterations: int = 3
    ):
        """
        Initialize enhanced research workflow.
        
        Args:
            web_search: WebSearch instance
            synthesizer: ResearchSynthesizer instance
            report_formatter: ReportFormatter instance
            retrieval_engine: Optional RAGRetrievalEngine for document queries
            max_iterations: Maximum validation iterations
        """
        self.web_research_agent = WebResearchAgent(web_search, max_results=10)
        self.technical_analyst_agent = TechnicalAnalystAgent(synthesizer)
        self.report_writer_agent = ReportWriterAgent(synthesizer, report_formatter)
        self.validation_agent = ValidationAgent(synthesizer, max_iterations=max_iterations)
        self.retrieval_engine = retrieval_engine
    
    def execute(self, query: str, task_type: str = "research", use_memory: bool = False, memory_context: str = "") -> Dict[str, Any]:
        """
        Execute the enhanced research workflow.
        
        Args:
            query: Research query
            task_type: Task type (e.g., "academic_paper_review")
            use_memory: Whether to use conversation memory
            memory_context: Memory context string
            
        Returns:
            Final state dictionary with report
        """
        # Initialize state
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
            "metadata": {},
            "current_step": "web_research",
            "report_draft": "",
            "feedback": None,
            "iterations": 0,
            "validation_passed": False
        }
        
        logger.info(f"Starting enhanced research workflow for: {query}")
        
        # Execute workflow with validation loop
        while state["current_step"] != "complete":
            try:
                if state["current_step"] == "web_research":
                    state = self.web_research_agent.run(state)
                
                elif state["current_step"] == "analysis":
                    state = self.technical_analyst_agent.run(state)
                
                elif state["current_step"] == "writing":
                    state = self.report_writer_agent.run(state)
                
                elif state["current_step"] == "validation":
                    state = self.validation_agent.run(state)
                
                else:
                    logger.warning(f"Unknown step: {state['current_step']}")
                    state["current_step"] = "complete"
            
            except Exception as e:
                logger.error(f"Workflow error at step {state.get('current_step')}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                state["current_step"] = "complete"
                state["report"] = {"error": str(e)}
                break
        
        logger.info(f"Research workflow complete. Iterations: {state.get('iterations', 0)}, Validation passed: {state.get('validation_passed', False)}")
        
        return state

