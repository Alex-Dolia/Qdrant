"""
Research Overview Workflow Module
LangGraph-based workflow for generating comprehensive research overview papers.
Uses specialized agents for each section with detailed prompt templates.
"""

import os
import logging
import traceback
import re
from typing import Dict, List, Optional, Any, Literal, Set, Tuple
from datetime import datetime
from pathlib import Path

# Import LangGraph components
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logging.warning("LangGraph not available. Install with: pip install langgraph")

# Import prompt loader
try:
    from prompts import load_prompt_template
    PROMPTS_AVAILABLE = True
except ImportError:
    PROMPTS_AVAILABLE = False
    logging.warning("Prompts module not available")

logger = logging.getLogger(__name__)

# Research Overview State Definition
from typing import TypedDict

class ResearchOverviewState(TypedDict):
    """State for research overview generation workflow."""
    topic: str
    context: str  # Synthesized research context
    references: List[Dict]  # Available references
    # Section outputs
    abstract_introduction: str
    background_foundations: str
    taxonomy_classification: str
    recent_advances: str
    applications_use_cases: str
    comparative_analysis: str
    challenges_limitations: str
    future_directions: str
    conclusion: str
    # Workflow state
    current_step: Literal[
        "planning",
        "abstract_intro",
        "background",
        "taxonomy",
        "recent_advances",
        "applications",
        "comparative",
        "challenges",
        "future",
        "conclusion",
        "formatting",
        "complete"
    ]
    sections: Dict[str, str]  # Final compiled sections
    report: Dict[str, Any]  # Final formatted report
    errors: List[str]  # Any errors encountered


class ResearchOverviewAgent:
    """Base agent for research overview generation."""
    
    def __init__(self, synthesizer, prompt_template_name: str):
        """
        Initialize research overview agent.
        
        Args:
            synthesizer: ResearchSynthesizer instance
            prompt_template_name: Name of prompt template file (without .txt)
        """
        self.synthesizer = synthesizer
        self.prompt_template_name = prompt_template_name
        self.prompt_template = None
        self._load_prompt_template()
    
    def _load_prompt_template(self):
        """Load prompt template from prompts directory."""
        try:
            if PROMPTS_AVAILABLE:
                self.prompt_template = load_prompt_template(self.prompt_template_name)
                logger.info(f"Loaded prompt template: {self.prompt_template_name}")
            else:
                logger.warning(f"Prompts not available, using fallback for {self.prompt_template_name}")
                self.prompt_template = self._get_fallback_template()
        except Exception as e:
            logger.error(f"Failed to load prompt template {self.prompt_template_name}: {e}")
            self.prompt_template = self._get_fallback_template()
    
    def _get_fallback_template(self) -> str:
        """Fallback template if file loading fails."""
        return f"Write a comprehensive {self.prompt_template_name.replace('_', ' ')} section for: {{topic}}\n\nContext: {{context}}\n\nReferences: {{references}}\n\nSection:"
    
    def _format_prompt(self, state: ResearchOverviewState, **kwargs) -> str:
        """Format prompt template with state and additional kwargs."""
        if not self.prompt_template:
            return self._get_fallback_template()
        
        # Format references
        references_text = ""
        if state.get("references"):
            ref_list = []
            for i, ref in enumerate(state["references"], 1):
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
            references_text = "\n".join(ref_list)
        else:
            references_text = "No references available. Use placeholder format: [Author et al., Year]"
        
        # Get context from previous sections for cohesion
        previous_sections = self._get_previous_sections_context(state)
        
        # Format prompt
        try:
            prompt = self.prompt_template.format(
                topic=state.get("topic", ""),
                context=state.get("context", "") + previous_sections,
                references=references_text,
                **kwargs
            )
            return prompt
        except KeyError as e:
            logger.warning(f"Missing key in prompt template: {e}. Using fallback.")
            return self._get_fallback_template().format(
                topic=state.get("topic", ""),
                context=state.get("context", "") + previous_sections,
                references=references_text
            )
    
    def _get_previous_sections_context(self, state: ResearchOverviewState) -> str:
        """Get context from previously generated sections to maintain cohesion."""
        sections = state.get("sections", {})
        current_section_key = self._get_section_key()
        
        # Define section order
        section_order = [
            "abstract_introduction",
            "background_foundations",
            "taxonomy_classification",
            "recent_advances",
            "applications_use_cases",
            "comparative_analysis",
            "challenges_limitations",
            "future_directions",
            "conclusion"
        ]
        
        # Find current section index
        try:
            current_idx = section_order.index(current_section_key)
        except ValueError:
            return ""
        
        # Get previous sections (limit to last 2 for context length)
        previous_context = []
        for i in range(max(0, current_idx - 2), current_idx):
            section_key = section_order[i]
            if section_key in sections:
                section_content = sections[section_key]
                # Get first 200 chars as context
                preview = section_content[:200].replace("\n", " ")
                previous_context.append(f"Previous section '{section_key}': {preview}...")
        
        if previous_context:
            return "\n\n**Previous Sections Context (for cohesion):**\n" + "\n".join(previous_context) + "\n\n**IMPORTANT:** Ensure this section flows naturally from previous sections and maintains consistency with the overall survey paper structure."
        
        return ""
    
    def run(self, state: ResearchOverviewState) -> ResearchOverviewState:
        """Run agent to generate section."""
        logger.info(f"Running {self.__class__.__name__} for step: {state.get('current_step')}")
        
        try:
            if not self.synthesizer or not self.synthesizer.llm:
                error_msg = f"LLM not available for {self.__class__.__name__}"
                logger.error(error_msg)
                state["errors"].append(error_msg)
                return state
            
            # Format prompt with context from previous sections
            prompt = self._format_prompt(state, **self._get_additional_kwargs(state))
            
            # Add instruction to maintain cohesion
            cohesion_instruction = "\n\n**CRITICAL:** This section is part of a SINGLE cohesive survey paper. Ensure it flows naturally from previous sections and maintains consistency. Do NOT create separate introductions or summaries - this is one unified document."
            prompt = prompt + cohesion_instruction
            
            # Generate section
            logger.info(f"Generating section with prompt length: {len(prompt)} chars")
            response = self.synthesizer.llm.invoke(prompt)
            
            if hasattr(response, 'content'):
                section_content = response.content
            else:
                section_content = str(response)
            
            # Clean up section content - remove any duplicate headers that might have been generated
            section_content = self._clean_section_content(section_content)
            
            # Store section
            section_key = self._get_section_key()
            state[section_key] = section_content
            state["sections"][section_key] = section_content
            
            logger.info(f"Generated {section_key}: {len(section_content)} chars")
            
        except Exception as e:
            error_msg = f"Error in {self.__class__.__name__}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            state["errors"].append(error_msg)
            # Set empty section on error
            section_key = self._get_section_key()
            state[section_key] = f"[Error generating {section_key}: {str(e)}]"
        
        return state
    
    def _clean_section_content(self, content: str) -> str:
        """Clean section content to remove duplicate headers or unwanted subsections."""
        lines = content.split('\n')
        cleaned_lines = []
        skip_next = False
        
        for i, line in enumerate(lines):
            # Skip duplicate "## Abstract" or "## Introduction" headers (except in abstract_introduction section)
            if self._get_section_key() != "abstract_introduction":
                if line.strip().startswith("## Abstract") or line.strip().startswith("## Introduction"):
                    logger.warning(f"Removing duplicate header: {line.strip()}")
                    continue
            
            # Skip standalone "### Introduction" or "### Summary" subsections
            if line.strip() == "### Introduction" or line.strip() == "### Summary":
                logger.warning(f"Removing unwanted subsection header: {line.strip()}")
                skip_next = True
                continue
            
            if skip_next and (not line.strip() or line.strip().startswith("###")):
                skip_next = False
            
            if not skip_next:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _clean_final_section(self, content: str, section_key: str) -> str:
        """Clean section content during final formatting to ensure cohesion."""
        lines = content.split('\n')
        cleaned_lines = []
        
        # Remove duplicate main headers (## Abstract, ## Introduction) except in abstract_introduction section
        if section_key != "abstract_introduction":
            for line in lines:
                if line.strip().startswith("## Abstract") or line.strip().startswith("## Introduction"):
                    logger.warning(f"Removing duplicate header in {section_key}: {line.strip()}")
                    continue
                cleaned_lines.append(line)
        else:
            cleaned_lines = lines
        
        return '\n'.join(cleaned_lines)
    
    def _get_section_key(self) -> str:
        """Get the state key for this section."""
        return self.prompt_template_name
    
    def _get_additional_kwargs(self, state: ResearchOverviewState) -> Dict[str, Any]:
        """Get additional kwargs for prompt formatting (override in subclasses)."""
        return {}


class AbstractIntroductionAgent(ResearchOverviewAgent):
    """Agent for generating Abstract and Introduction sections."""
    
    def __init__(self, synthesizer):
        super().__init__(synthesizer, "abstract_introduction")
    
    def _get_section_key(self) -> str:
        return "abstract_introduction"


class BackgroundFoundationsAgent(ResearchOverviewAgent):
    """Agent for generating Background / Foundations section."""
    
    def __init__(self, synthesizer):
        super().__init__(synthesizer, "background_foundations")
    
    def _get_section_key(self) -> str:
        return "background_foundations"


class TaxonomyClassificationAgent(ResearchOverviewAgent):
    """Agent for generating Taxonomy / Classification section."""
    
    def __init__(self, synthesizer):
        super().__init__(synthesizer, "taxonomy_classification")
    
    def _get_section_key(self) -> str:
        return "taxonomy_classification"


class RecentAdvancesAgent(ResearchOverviewAgent):
    """Agent for generating Recent Advances section."""
    
    def __init__(self, synthesizer):
        super().__init__(synthesizer, "recent_advances")
    
    def _get_section_key(self) -> str:
        return "recent_advances"


class ApplicationsUseCasesAgent(ResearchOverviewAgent):
    """Agent for generating Applications and Use Cases section."""
    
    def __init__(self, synthesizer):
        super().__init__(synthesizer, "applications_use_cases")
    
    def _get_section_key(self) -> str:
        return "applications_use_cases"


class ComparativeAnalysisAgent(ResearchOverviewAgent):
    """Agent for generating Comparative Analysis / Synthesis section."""
    
    def __init__(self, synthesizer):
        # Use the new comparative analysis table prompt
        super().__init__(synthesizer, "comparative_analysis_table")
    
    def _get_section_key(self) -> str:
        return "comparative_analysis"
    
    def _get_additional_kwargs(self, state: ResearchOverviewState) -> Dict[str, Any]:
        """Include taxonomy and recent advances in context."""
        return {
            "taxonomy": state.get("taxonomy_classification", ""),
            "recent_advances": state.get("recent_advances", "")
        }


class ChallengesLimitationsAgent(ResearchOverviewAgent):
    """Agent for generating Challenges and Limitations section."""
    
    def __init__(self, synthesizer):
        super().__init__(synthesizer, "challenges_limitations")
    
    def _get_section_key(self) -> str:
        return "challenges_limitations"


class FutureDirectionsAgent(ResearchOverviewAgent):
    """Agent for generating Future Research Directions section."""
    
    def __init__(self, synthesizer):
        super().__init__(synthesizer, "future_directions")
    
    def _get_section_key(self) -> str:
        return "future_directions"
    
    def _get_additional_kwargs(self, state: ResearchOverviewState) -> Dict[str, Any]:
        """Include challenges in context."""
        return {
            "challenges": state.get("challenges_limitations", "")
        }


class ConclusionAgent(ResearchOverviewAgent):
    """Agent for generating Conclusion section."""
    
    def __init__(self, synthesizer):
        super().__init__(synthesizer, "conclusion")
    
    def _get_section_key(self) -> str:
        return "conclusion"
    
    def _get_additional_kwargs(self, state: ResearchOverviewState) -> Dict[str, Any]:
        """Include all key findings in context."""
        return {
            "key_findings": self._summarize_findings(state),
            "contributions": self._summarize_contributions(state),
            "challenges": state.get("challenges_limitations", ""),
            "future_directions": state.get("future_directions", "")
        }
    
    def _summarize_findings(self, state: ResearchOverviewState) -> str:
        """Summarize key findings from all sections."""
        findings = []
        if state.get("recent_advances"):
            findings.append("Recent advances identified in the field")
        if state.get("taxonomy_classification"):
            findings.append("Comprehensive taxonomy and classification scheme developed")
        if state.get("applications_use_cases"):
            findings.append("Diverse applications and use cases documented")
        return "; ".join(findings) if findings else "Comprehensive survey of the field"
    
    def _summarize_contributions(self, state: ResearchOverviewState) -> str:
        """Summarize main contributions."""
        contributions = [
            "Organizational taxonomy and classification scheme",
            "Synthesis of recent advances (2020-2025)",
            "Comparative analysis of approaches",
            "Identification of challenges and limitations",
            "Future research directions"
        ]
        return "; ".join(contributions)


class ResearchOverviewWorkflow:
    """LangGraph workflow for generating comprehensive research overview papers."""
    
    def __init__(
        self,
        synthesizer,
        report_formatter,
        retrieval_engine=None,
        use_academic_sources: bool = False,
        use_pdf_processing: bool = False,
        enable_checkpoint: bool = False
    ):
        """
        Initialize research overview workflow.
        
        Args:
            synthesizer: ResearchSynthesizer instance
            report_formatter: ReportFormatter instance
            retrieval_engine: Optional RAGRetrievalEngine for document queries
            use_academic_sources: Enable arXiv/Semantic Scholar integration
            use_pdf_processing: Enable PDF downloading and text extraction
            enable_checkpoint: Enable human-in-the-loop checkpoint after paper collection
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph is required. Install with: pip install langgraph")
        
        self.synthesizer = synthesizer
        self.report_formatter = report_formatter
        self.retrieval_engine = retrieval_engine
        self.use_academic_sources = use_academic_sources
        self.use_pdf_processing = use_pdf_processing
        self.enable_checkpoint = enable_checkpoint
        
        # Initialize academic search if enabled
        if use_academic_sources:
            try:
                # Use importlib for modules starting with numbers
                import importlib
                academic_search_module = importlib.import_module('scripts.07_research_workflow.academic_search')
                AcademicSearch = academic_search_module.AcademicSearch
                self.academic_search = AcademicSearch(use_cache=True)
                logger.info("Academic search enabled (arXiv, Semantic Scholar)")
            except Exception as e:
                logger.warning(f"Failed to initialize academic search: {e}")
                self.academic_search = None
                self.use_academic_sources = False
        else:
            self.academic_search = None
        
        # Initialize PDF handler if enabled
        if use_pdf_processing:
            try:
                # Use importlib for modules starting with numbers
                pdf_handler_module = importlib.import_module('scripts.01_data_ingestion_and_preprocessing.pdf_handler')
                PDFHandler = pdf_handler_module.PDFHandler
                self.pdf_handler = PDFHandler()
                logger.info("PDF processing enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize PDF handler: {e}")
                self.pdf_handler = None
                self.use_pdf_processing = False
        else:
            self.pdf_handler = None
        
        # Initialize paper summarizer if PDF processing enabled
        if use_pdf_processing:
            try:
                # Use importlib for modules starting with numbers
                paper_summarizer_module = importlib.import_module('scripts.07_research_workflow.paper_summarizer')
                PaperSummarizer = paper_summarizer_module.PaperSummarizer
                self.paper_summarizer = PaperSummarizer(synthesizer=synthesizer)
                logger.info("Paper summarization enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize paper summarizer: {e}")
                self.paper_summarizer = None
        else:
            self.paper_summarizer = None
        
        # Initialize reproducibility logger
        try:
            # Use importlib for modules starting with numbers
            import importlib
            repro_logger_module = importlib.import_module('scripts.08_utilities.reproducibility_logger')
            ReproducibilityLogger = repro_logger_module.ReproducibilityLogger
            self.repro_logger = ReproducibilityLogger()
            logger.info("Reproducibility logging enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize reproducibility logger: {e}")
            self.repro_logger = None
        
        # Initialize agents
        self.abstract_intro_agent = AbstractIntroductionAgent(synthesizer)
        self.background_agent = BackgroundFoundationsAgent(synthesizer)
        self.taxonomy_agent = TaxonomyClassificationAgent(synthesizer)
        self.recent_advances_agent = RecentAdvancesAgent(synthesizer)
        self.applications_agent = ApplicationsUseCasesAgent(synthesizer)
        self.comparative_agent = ComparativeAnalysisAgent(synthesizer)
        self.challenges_agent = ChallengesLimitationsAgent(synthesizer)
        self.future_agent = FutureDirectionsAgent(synthesizer)
        self.conclusion_agent = ConclusionAgent(synthesizer)
        
        # Build workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow."""
        workflow = StateGraph(ResearchOverviewState)
        
        # Add nodes
        workflow.add_node("abstract_intro", self.abstract_intro_agent.run)
        workflow.add_node("background", self.background_agent.run)
        workflow.add_node("taxonomy", self.taxonomy_agent.run)
        workflow.add_node("recent_advances", self.recent_advances_agent.run)
        workflow.add_node("applications", self.applications_agent.run)
        workflow.add_node("comparative", self.comparative_agent.run)
        workflow.add_node("challenges", self.challenges_agent.run)
        workflow.add_node("future", self.future_agent.run)
        workflow.add_node("conclusion", self.conclusion_agent.run)
        workflow.add_node("formatting", self._format_report)
        
        # Set entry point
        workflow.set_entry_point("abstract_intro")
        
        # Define edges (sequential flow)
        workflow.add_edge("abstract_intro", "background")
        workflow.add_edge("background", "taxonomy")
        workflow.add_edge("taxonomy", "recent_advances")
        workflow.add_edge("recent_advances", "applications")
        workflow.add_edge("applications", "comparative")
        workflow.add_edge("comparative", "challenges")
        workflow.add_edge("challenges", "future")
        workflow.add_edge("future", "conclusion")
        workflow.add_edge("conclusion", "formatting")
        workflow.add_edge("formatting", END)
        
        return workflow.compile()
    
    def _extract_citations(self, text: str) -> Set[int]:
        """
        Extract citation numbers from text (e.g., [1], [2], [3]).
        
        Args:
            text: Text to search for citations
            
        Returns:
            Set of citation numbers found
        """
        citations = set()
        
        # Pattern 1: [1], [2], [3], etc.
        pattern1 = r'\[(\d+)\]'
        matches = re.findall(pattern1, text)
        for match in matches:
            try:
                citations.add(int(match))
            except ValueError:
                pass
        
        # Pattern 2: [1, 2, 3] (multiple citations)
        pattern2 = r'\[(\d+(?:\s*,\s*\d+)+)\]'
        matches = re.findall(pattern2, text)
        for match in matches:
            numbers = re.findall(r'\d+', match)
            for num in numbers:
                try:
                    citations.add(int(num))
                except ValueError:
                    pass
        
        return citations
    
    def _clean_final_section(self, content: str, section_key: str) -> str:
        """Clean section content during final formatting to ensure cohesion."""
        try:
            logger.debug(f"Cleaning section: {section_key}")
            lines = content.split('\n')
            cleaned_lines = []
            
            # Remove duplicate main headers (## Abstract, ## Introduction) except in abstract_introduction section
            if section_key != "abstract_introduction":
                for line in lines:
                    if line.strip().startswith("## Abstract") or line.strip().startswith("## Introduction"):
                        logger.warning(f"Removing duplicate header in {section_key}: {line.strip()}")
                        continue
                    cleaned_lines.append(line)
            else:
                cleaned_lines = lines
            
            result = '\n'.join(cleaned_lines)
            logger.debug(f"Section {section_key} cleaned: {len(result)} chars")
            return result
        except Exception as e:
            logger.error(f"Error cleaning section {section_key}: {e}", exc_info=True)
            return content  # Return original content on error
    
    def _generate_bibtex_key(self, author: str, year: str, title: str) -> str:
        """Generate a BibTeX key from author, year, and title."""
        # Extract first author's last name
        if author:
            first_author = author.split(",")[0].split()[0] if "," in author else author.split()[0]
            first_author = first_author.lower().replace(".", "")
        else:
            first_author = "unknown"
        
        # Extract year (last 4 digits)
        year_part = ""
        if year:
            year_digits = re.findall(r'\d{4}', year)
            if year_digits:
                year_part = year_digits[-1]
        
        # Extract first significant word from title
        title_part = ""
        if title:
            words = title.split()
            for word in words[:3]:
                # Skip common words
                if word.lower() not in ["the", "a", "an", "on", "in", "for", "of", "and", "or"]:
                    title_part = word.lower()[:8]
                    break
        
        # Combine: author_year_title
        bibtex_key = f"{first_author}{year_part}{title_part}".replace(" ", "").replace("-", "")
        return bibtex_key[:50]  # Limit length
    
    def _format_bibtex_entry(self, bibtex_key: str, author: str, title: str, year: str, venue: str, url: str) -> str:
        """Format a BibTeX entry."""
        # Determine entry type
        if "arxiv" in venue.lower() or "arxiv" in (url or "").lower():
            entry_type = "@article"
        elif any(word in venue.lower() for word in ["conference", "workshop", "proceedings"]):
            entry_type = "@inproceedings"
        else:
            entry_type = "@article"
        
        bibtex = f"@{entry_type[1:]}{{{bibtex_key},\n"
        bibtex += f"  title={{{title}}},\n"
        
        if author:
            bibtex += f"  author={{{author}}},\n"
        
        if year:
            year_digits = re.findall(r'\d{4}', year)
            if year_digits:
                bibtex += f"  year={{{year_digits[-1]}}},\n"
        
        if venue:
            if entry_type == "@inproceedings":
                bibtex += f"  booktitle={{{venue}}},\n"
            else:
                bibtex += f"  journal={{{venue}}},\n"
        
        if url:
            bibtex += f"  url={{{url}}},\n"
        
        # Remove trailing comma
        bibtex = bibtex.rstrip(",\n") + "\n"
        bibtex += "}"
        
        return bibtex
    
    def _format_human_citation(self, author: str, title: str, venue: str, year: str) -> str:
        """Format human-readable citation."""
        parts = []
        
        if author:
            parts.append(author)
        
        if year:
            year_digits = re.findall(r'\d{4}', year)
            if year_digits:
                parts.append(f"({year_digits[-1]})")
        
        if title:
            parts.append(f'"{title}"')
        
        if venue:
            parts.append(f", {venue}")
        
        return ", ".join(parts) + "."
    
    def _extract_author_year_citations(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract author-year citations (e.g., [Author et al., 2023], [Smith, 2022]).
        
        Args:
            text: Text to search for citations
            
        Returns:
            List of (author, year) tuples
        """
        citations = []
        
        # Pattern: [Author et al., Year] or [Author, Year]
        pattern = r'\[([^,\]]+(?:\s+et\s+al\.?)?)\s*,\s*(\d{4})\]'
        matches = re.findall(pattern, text, re.IGNORECASE)
        for author, year in matches:
            citations.append((author.strip(), year.strip()))
        
        return citations
    
    def _format_report(self, state: ResearchOverviewState) -> ResearchOverviewState:
        """Format final report from all sections."""
        logger.info("=" * 80)
        logger.info("STAGE: Report Formatting")
        logger.info("=" * 80)
        logger.info(f"State keys: {list(state.keys())}")
        logger.info(f"Sections available: {list(state.get('sections', {}).keys())}")
        logger.info(f"References count: {len(state.get('references', []))}")
        
        try:
            # Compile all sections
            sections = state.get("sections", {})
            
            # Collect all text to extract citations
            all_text = " ".join(sections.values())
            
            # Extract citations from text
            numeric_citations = self._extract_citations(all_text)
            author_year_citations = self._extract_author_year_citations(all_text)
            
            logger.info(f"Found {len(numeric_citations)} numeric citations: {sorted(numeric_citations)}")
            logger.info(f"Found {len(author_year_citations)} author-year citations")
            
            # Get provided references
            provided_references = state.get("references", [])
            
            # Build references list from citations
            used_references = []
            seen_refs = set()  # Track references we've already added (by title+author)
            citation_to_ref = {}  # Map citation number to reference index in used_references
            
            # Process numeric citations [1], [2], etc.
            for citation_num in sorted(numeric_citations):
                ref_index = citation_num - 1  # Convert to 0-based index
                if 0 <= ref_index < len(provided_references):
                    ref = provided_references[ref_index]
                    # Create unique key for this reference
                    ref_key = (ref.get("title", "").lower(), ref.get("author", "").lower())
                    if ref_key not in seen_refs:
                        used_references.append(ref)
                        seen_refs.add(ref_key)
                        citation_to_ref[citation_num] = len(used_references) - 1
                    else:
                        # Reference already added, just map citation to existing index
                        for idx, existing_ref in enumerate(used_references):
                            if (existing_ref.get("title", "").lower(), existing_ref.get("author", "").lower()) == ref_key:
                                citation_to_ref[citation_num] = idx
                                break
            
            # Process author-year citations
            for author, year in author_year_citations:
                # Try to match with provided references
                matched = False
                for ref in provided_references:
                    ref_author = ref.get("author", "").lower()
                    ref_date = ref.get("date", "")
                    
                    # Check if author matches (partial match or contains)
                    author_match = (
                        author.lower() in ref_author or 
                        ref_author in author.lower() or
                        any(word.lower() in ref_author for word in author.split() if len(word) > 3)
                    )
                    
                    # Check if year matches
                    year_match = year in ref_date or ref_date in year or not ref_date
                    
                    if author_match and year_match:
                        ref_key = (ref.get("title", "").lower(), ref.get("author", "").lower())
                        if ref_key not in seen_refs:
                            used_references.append(ref)
                            seen_refs.add(ref_key)
                            matched = True
                            break
                        else:
                            matched = True  # Already added, skip
                            break
                
                # If no match found, create placeholder
                if not matched:
                    placeholder_ref = {
                        "title": f"Research paper by {author}",
                        "author": author,
                        "date": year,
                        "url": "",
                        "publication": "",
                        "domain": "",
                        "source_type": "placeholder"
                    }
                    ref_key = (placeholder_ref["title"].lower(), author.lower())
                    if ref_key not in seen_refs:
                        used_references.append(placeholder_ref)
                        seen_refs.add(ref_key)
            
            # If no citations found but references provided, use all provided references
            if not numeric_citations and not author_year_citations and provided_references:
                logger.info("No citations found in text, using all provided references")
                used_references = provided_references.copy()
            
            # Create markdown report
            markdown_parts = []
            
            # Title
            markdown_parts.append(f"# Research Overview: {state.get('topic', 'Unknown Topic')}\n\n")
            markdown_parts.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            markdown_parts.append("---\n\n")
            
            # Add sections in order
            section_order = [
                "abstract_introduction",
                "background_foundations",
                "taxonomy_classification",
                "recent_advances",
                "applications_use_cases",
                "comparative_analysis",
                "challenges_limitations",
                "future_directions",
                "conclusion"
            ]
            
            for i, section_key in enumerate(section_order):
                if section_key in sections:
                    section_content = sections[section_key]
                    logger.info(f"Processing section {i+1}/{len(section_order)}: {section_key} ({len(section_content)} chars)")
                    
                    # Clean section content to remove any duplicate headers
                    try:
                        section_content = self._clean_final_section(section_content, section_key)
                        logger.debug(f"Section {section_key} cleaned successfully")
                    except Exception as clean_error:
                        logger.error(f"Error cleaning section {section_key}: {clean_error}", exc_info=True)
                        # Continue with original content
                        section_content = sections[section_key]
                    
                    markdown_parts.append(section_content)
                    
                    # Add separator only if not the last section
                    if i < len(section_order) - 1:
                        markdown_parts.append("\n\n---\n\n")
            
            # Add references section with BibTeX format
            if used_references:
                markdown_parts.append("## References\n\n")
                
                for i, ref in enumerate(used_references, 1):
                    title = ref.get("title", "Untitled")
                    url = ref.get("url", "")
                    author = ref.get("author", "")
                    date = ref.get("date", "")
                    publication = ref.get("publication", ref.get("domain", ""))
                    
                    # Generate BibTeX key from author and year
                    bibtex_key = self._generate_bibtex_key(author, date, title)
                    
                    # Format BibTeX entry
                    bibtex_entry = self._format_bibtex_entry(
                        bibtex_key=bibtex_key,
                        author=author,
                        title=title,
                        year=date,
                        venue=publication,
                        url=url
                    )
                    
                    # Format human-readable citation
                    human_citation = self._format_human_citation(
                        author=author,
                        title=title,
                        venue=publication,
                        year=date
                    )
                    
                    # Combine BibTeX, human-readable, and URL
                    ref_block = f"**[{i}]**\n\n"
                    ref_block += "```bibtex\n"
                    ref_block += bibtex_entry
                    ref_block += "\n```\n\n"
                    ref_block += f"{human_citation}"
                    if url:
                        ref_block += f"\n\nAvailable at: {url}"
                    ref_block += "\n\n---\n\n"
                    
                    markdown_parts.append(ref_block)
            else:
                markdown_parts.append("## References\n\nNo references available.\n\n")
            
            # Compile markdown
            markdown = "".join(markdown_parts)
            
            # Create report dict
            report = {
                "markdown": markdown,
                "plain_text": markdown,  # For compatibility
                "sections": sections,
                "references": used_references,  # Use extracted references
                "metadata": {
                    "topic": state.get("topic", ""),
                    "timestamp": datetime.now().isoformat(),
                    "type": "research_overview"
                }
            }
            
            logger.info(f"References included: {len(used_references)} (from {len(provided_references)} provided)")
            
            state["report"] = report
            state["current_step"] = "complete"
            
            logger.info(f"Report formatted: {len(markdown)} chars, {len(sections)} sections")
            
        except Exception as e:
            error_msg = f"Report formatting error: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            state["errors"].append(error_msg)
            state["report"] = {"error": error_msg}
        
        return state
    
    def execute(
        self,
        topic: str,
        context: str,
        references: List[Dict],
        use_web_research: bool = True,
        max_academic_papers: int = 40,
        min_year: int = None,
        max_year: int = None
    ) -> Dict[str, Any]:
        """
        Execute research overview generation workflow.
        
        Args:
            topic: Research topic
            context: Synthesized research context
            references: List of references
            use_web_research: Whether to use web research
            max_academic_papers: Maximum papers to collect from academic sources
            min_year: Minimum publication year for papers (default: 5 years back from current year)
            max_year: Maximum publication year for papers (default: current year)
            
        Returns:
            Final state dictionary with report
        """
        # Set default year range if not provided
        current_year = datetime.now().year
        if min_year is None:
            min_year = current_year - 5
        if max_year is None:
            max_year = current_year
        
        logger.info("=" * 80)
        logger.info("RESEARCH OVERVIEW GENERATION STARTED")
        logger.info("=" * 80)
        logger.info(f"Topic: {topic}")
        logger.info(f"Context length: {len(context)} chars")
        logger.info(f"References: {len(references)}")
        logger.info(f"Year range: {min_year}-{max_year}")
        logger.info(f"Academic sources: {self.use_academic_sources}")
        logger.info(f"PDF processing: {self.use_pdf_processing}")
        logger.info("-" * 80)
        
        # Collect academic papers if enabled
        academic_papers = []
        if self.use_academic_sources and self.academic_search:
            logger.info("Collecting papers from academic sources...")
            try:
                # Log query
                if self.repro_logger:
                    self.repro_logger.log_query(
                        query=topic,
                        source="academic_search",
                        max_results=max_academic_papers,
                        parameters={"min_year": min_year, "max_year": max_year}
                    )
                
                academic_papers = self.academic_search.search(
                    topic=topic,
                    max_results=max_academic_papers,
                    min_year=min_year,
                    prioritize_recent=True
                )
                
                # Filter by max_year if provided
                if max_year:
                    academic_papers = [p for p in academic_papers if p.get("year") is None or p.get("year") <= max_year]
                    logger.info(f"Filtered to {len(academic_papers)} papers within year range {min_year}-{max_year}")
                
                logger.info(f"Collected {len(academic_papers)} academic papers")
                
                # Log paper collection
                if self.repro_logger:
                    self.repro_logger.log_paper_collection(
                        papers=academic_papers,
                        topic=topic,
                        filters={"min_year": min_year}
                    )
                
                # Human-in-the-loop checkpoint
                if self.enable_checkpoint:
                    checkpoint_file = Path("output/user_confirm.true")
                    candidates_file = Path("output/candidates.json")
                    
                    # Save candidates for review
                    import json
                    with open(candidates_file, 'w', encoding='utf-8') as f:
                        json.dump(academic_papers, f, indent=2, default=str)
                    
                    logger.info(f"Checkpoint: Saved {len(academic_papers)} candidates to {candidates_file}")
                    logger.info(f"Waiting for user confirmation: create {checkpoint_file} to proceed")
                    
                    # Wait for confirmation (in Streamlit, this would be handled differently)
                    import time
                    max_wait = 3600  # 1 hour max wait
                    waited = 0
                    while not checkpoint_file.exists() and waited < max_wait:
                        time.sleep(5)
                        waited += 5
                    
                    if not checkpoint_file.exists():
                        logger.warning("Checkpoint timeout - proceeding without confirmation")
                    else:
                        logger.info("User confirmation received - proceeding")
                
                # Process PDFs if enabled
                if self.use_pdf_processing and self.pdf_handler:
                    logger.info("Processing PDFs...")
                    processed_papers = []
                    for i, paper in enumerate(academic_papers, 1):
                        logger.info(f"Processing PDF {i}/{len(academic_papers)}: {paper.get('title', 'Unknown')[:60]}")
                        processed = self.pdf_handler.process_paper(paper)
                        processed_papers.append(processed)
                    academic_papers = processed_papers
                    
                    # Summarize papers if summarizer available
                    if self.paper_summarizer:
                        logger.info("Summarizing papers...")
                        academic_papers = self.paper_summarizer.summarize_batch(academic_papers)
                
                # Merge academic papers into references
                for paper in academic_papers:
                    ref = {
                        "title": paper.get("title", ""),
                        "url": paper.get("url", ""),
                        "doi": paper.get("doi", ""),
                        "author": ", ".join(paper.get("authors", [])),
                        "date": str(paper.get("year", "")) if paper.get("year") else "",
                        "publication": paper.get("venue", ""),
                        "abstract": paper.get("abstract", ""),
                        "source_type": "academic_search",
                        "summary": paper.get("summary", ""),
                        "contributions": paper.get("contributions", []),
                        "keywords": paper.get("keywords", []),
                        "taxonomy": paper.get("taxonomy", "")
                    }
                    references.append(ref)
                
                logger.info(f"Added {len(academic_papers)} academic papers to references")
                
            except Exception as e:
                error_msg = f"Academic source collection error: {e}"
                logger.error("=" * 80)
                logger.error("ACADEMIC SOURCE COLLECTION FAILED")
                logger.error("=" * 80)
                logger.error(error_msg)
                logger.error(f"Traceback: {traceback.format_exc()}")
                logger.error("Continuing with existing references...")
                logger.error("=" * 80)
                # Continue with existing references
        
        # Add date range to context if provided
        enhanced_context = context
        if min_year and max_year:
            year_range_note = f"\n\n**Date Range for Research:** Papers published between {min_year} and {max_year} (inclusive). Focus on recent advances from {min_year}-{max_year}."
            enhanced_context = context + year_range_note if context else year_range_note
        
        # Initialize state
        initial_state: ResearchOverviewState = {
            "topic": topic,
            "context": enhanced_context,
            "references": references,
            "abstract_introduction": "",
            "background_foundations": "",
            "taxonomy_classification": "",
            "recent_advances": "",
            "applications_use_cases": "",
            "comparative_analysis": "",
            "challenges_limitations": "",
            "future_directions": "",
            "conclusion": "",
            "current_step": "abstract_intro",
            "sections": {},
            "report": {},
            "errors": []
        }
        
        # Execute workflow
        logger.info("Executing LangGraph workflow...")
        try:
            final_state = self.workflow.invoke(initial_state)
            
            logger.info("=" * 80)
            logger.info("WORKFLOW EXECUTION COMPLETED")
            logger.info("=" * 80)
            logger.info(f"Final step: {final_state.get('current_step')}")
            logger.info(f"Sections generated: {list(final_state.get('sections', {}).keys())}")
            logger.info(f"Errors encountered: {len(final_state.get('errors', []))}")
            
            if final_state.get("errors"):
                logger.warning("Errors during workflow execution:")
                for i, error in enumerate(final_state["errors"], 1):
                    logger.warning(f"  {i}. {error}")
            
            # Check if report was generated
            report = final_state.get("report", {})
            if report and not report.get("error"):
                logger.info(f"Report generated successfully: {len(report.get('markdown', ''))} chars")
            else:
                error_msg = report.get("error", "Unknown error") if report else "No report generated"
                logger.error(f"Report generation failed: {error_msg}")
            
            logger.info("=" * 80)
            
            return {
                "report": report,
                "errors": final_state.get("errors", []),
                "sections": final_state.get("sections", {}),
                "references": final_state.get("references", []),
                "metadata": {
                    "topic": topic,
                    "timestamp": datetime.now().isoformat(),
                    "year_range": f"{min_year}-{max_year}",
                    "academic_sources_used": self.use_academic_sources,
                    "pdf_processing_used": self.use_pdf_processing
                }
            }
            
        except Exception as e:
            error_msg = f"Workflow execution error: {e}"
            logger.error("=" * 80)
            logger.error("WORKFLOW EXECUTION FAILED")
            logger.error("=" * 80)
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.error("=" * 80)
            
            return {
                "report": {"error": error_msg},
                "errors": [error_msg],
                "sections": {},
                "references": references,
                "metadata": {
                    "topic": topic,
                    "timestamp": datetime.now().isoformat(),
                    "year_range": f"{min_year}-{max_year}",
                    "error": str(e)
                }
            }

