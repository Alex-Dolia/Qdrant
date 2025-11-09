"""
JMLR-Style Literature Review Automation
Agentic RAG workflow using LangGraph, Qdrant, and web search for automated literature review.

Adapted to use:
- Ollama/Llama 3.1 (instead of OpenAI)
- Existing Qdrant setup (unchanged)
- DuckDuckGo web search (instead of BraveSearch)
- Academic sources (arXiv, Semantic Scholar)
"""

import os
import json
import logging
from datetime import datetime
from typing import TypedDict, Annotated, List, Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Import LangGraph components
try:
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph not available. Install with: pip install langgraph")

# Import LLM (Ollama/Llama 3.1)
try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama not available. Install with: pip install langchain-ollama")

# Import existing modules
try:
    # Use importlib for modules starting with numbers
    import importlib
    academic_search_module = importlib.import_module('scripts.07_research_workflow.academic_search')
    AcademicSearch = academic_search_module.AcademicSearch
    
    web_search_module = importlib.import_module('scripts.07_research_workflow.web_search')
    WebSearch = web_search_module.WebSearch
    
    retrieval_module = importlib.import_module('scripts.03_retrieval.retrieval')
    RAGRetrievalEngine = retrieval_module.RAGRetrievalEngine
    
    synthesis_module = importlib.import_module('scripts.02_query_completion.synthesis')
    ResearchSynthesizer = synthesis_module.ResearchSynthesizer
except ImportError as e:
    logger.warning(f"Failed to import utility modules: {e}")


# Define LangGraph State
class LiteratureReviewState(TypedDict):
    """State for literature review generation."""
    paper_title: str
    method_description: str
    start_year: int
    end_year: int
    messages: Annotated[List, "add_messages"]
    retrieved_papers: List[Dict[str, Any]]
    qdrant_results: List[Dict[str, Any]]
    web_results: List[Dict[str, Any]]
    academic_results: List[Dict[str, Any]]
    organized_papers: Dict[str, List[Dict[str, Any]]]
    latex_section: str
    bibtex_entries: List[str]
    current_step: str


class JMLRLiteratureReviewAgent:
    """Agent for generating JMLR-style literature review sections."""
    
    def __init__(
        self,
        retrieval_engine: Optional[Any] = None,
        use_ollama: bool = True,
        model_name: str = "llama3.1:latest"
    ):
        """
        Initialize JMLR literature review agent.
        
        Args:
            retrieval_engine: RAGRetrievalEngine instance (for Qdrant queries)
            use_ollama: Use Ollama (True) or OpenAI (False)
            model_name: Model name for Ollama
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph is required. Install with: pip install langgraph")
        
        if not OLLAMA_AVAILABLE and use_ollama:
            raise ImportError("Ollama is required. Install with: pip install langchain-ollama")
        
        self.retrieval_engine = retrieval_engine
        self.use_ollama = use_ollama
        
        # Initialize LLM (Ollama/Llama 3.1)
        if use_ollama and OLLAMA_AVAILABLE:
            self.llm = ChatOllama(
                model=model_name,
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=0.3,  # Lower temperature for more focused academic writing
                num_ctx=4096
            )
            logger.info(f"Initialized Ollama LLM with {model_name}")
        else:
            raise ValueError("Ollama is required for this agent")
        
        # Initialize search tools
        self.web_search = WebSearch(provider="duckduckgo", use_cache=True) if WebSearch else None
        self.academic_search = AcademicSearch(use_cache=True) if AcademicSearch else None
        
        # Initialize synthesizer for final generation
        self.synthesizer = ResearchSynthesizer(use_ollama=use_ollama, model_name=model_name)
        
        # Build workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow for literature review."""
        workflow = StateGraph(LiteratureReviewState)
        
        # Add nodes
        workflow.add_node("plan_search", self._plan_search_node)
        workflow.add_node("search_sources", self._search_sources_node)
        workflow.add_node("organize_papers", self._organize_papers_node)
        workflow.add_node("generate_review", self._generate_review_node)
        workflow.add_node("format_output", self._format_output_node)
        
        # Set entry point
        workflow.set_entry_point("plan_search")
        
        # Define edges
        workflow.add_edge("plan_search", "search_sources")
        workflow.add_edge("search_sources", "organize_papers")
        workflow.add_edge("organize_papers", "generate_review")
        workflow.add_edge("generate_review", "format_output")
        workflow.add_edge("format_output", END)
        
        return workflow.compile()
    
    def _plan_search_node(self, state: LiteratureReviewState) -> LiteratureReviewState:
        """Plan which sources to search."""
        logger.info("Planning search strategy...")
        
        paper_title = state.get("paper_title", "")
        start_year = state.get("start_year", datetime.now().year - 5)
        end_year = state.get("end_year", datetime.now().year)
        
        # Determine search strategy
        plan_prompt = f"""You are planning a literature review search strategy for the paper: "{paper_title}"

The review should cover papers from {start_year} to {end_year} from top ML venues (NeurIPS, ICML, ICLR, JMLR, PMLR, IEEE TPAMI).

Determine which sources to search:
1. Qdrant vector store (if available) - for previously ingested papers
2. Academic sources (arXiv, Semantic Scholar) - for recent papers
3. Web search (DuckDuckGo) - for very recent or specific topics

Output: JSON with keys: use_qdrant (bool), use_academic (bool), use_web (bool), search_queries (list of strings)

Be specific with search queries - include venue names and year ranges where relevant."""

        try:
            response = self.llm.invoke(plan_prompt)
            plan_text = response.content if hasattr(response, 'content') else str(response)
            
            # Try to parse JSON from response
            try:
                # Extract JSON if wrapped in markdown code blocks
                if "```json" in plan_text:
                    plan_text = plan_text.split("```json")[1].split("```")[0].strip()
                elif "```" in plan_text:
                    plan_text = plan_text.split("```")[1].split("```")[0].strip()
                
                plan = json.loads(plan_text)
            except json.JSONDecodeError:
                # Fallback: use defaults
                logger.warning("Could not parse search plan JSON, using defaults")
                plan = {
                    "use_qdrant": True,
                    "use_academic": True,
                    "use_web": True,
                    "search_queries": [
                        f"{paper_title} NeurIPS ICML ICLR {start_year} {end_year}",
                        f"{paper_title} survey review",
                        f"{paper_title} recent advances"
                    ]
                }
            
            state["current_step"] = "search_sources"
            state["search_plan"] = plan
            
            logger.info(f"Search plan: {plan}")
            
        except Exception as e:
            logger.error(f"Planning error: {e}")
            # Use default plan
            state["search_plan"] = {
                "use_qdrant": True,
                "use_academic": True,
                "use_web": True,
                "search_queries": [paper_title]
            }
        
        return state
    
    def _search_sources_node(self, state: LiteratureReviewState) -> LiteratureReviewState:
        """Search across multiple sources."""
        logger.info("Searching sources...")
        
        paper_title = state.get("paper_title", "")
        start_year = state.get("start_year", datetime.now().year - 5)
        end_year = state.get("end_year", datetime.now().year)
        search_plan = state.get("search_plan", {})
        
        qdrant_results = []
        academic_results = []
        web_results = []
        
        # Search Qdrant if available and planned
        if search_plan.get("use_qdrant", True) and self.retrieval_engine:
            try:
                logger.info("Searching Qdrant vector store...")
                # Use the paper title as query
                chunks = self.retrieval_engine.retrieve(
                    query=paper_title,
                    top_k=20,
                    use_reranking=False
                )
                
                # Convert chunks to paper format
                for chunk in chunks:
                    metadata = chunk.get("metadata", {})
                    qdrant_results.append({
                        "title": metadata.get("file_name", "Unknown"),
                        "content": chunk.get("text", ""),
                        "source": "qdrant",
                        "metadata": metadata
                    })
                
                logger.info(f"Found {len(qdrant_results)} results from Qdrant")
            except Exception as e:
                logger.warning(f"Qdrant search error: {e}")
        
        # Search academic sources if planned
        if search_plan.get("use_academic", True) and self.academic_search:
            try:
                logger.info("Searching academic sources (arXiv, Semantic Scholar)...")
                
                queries = search_plan.get("search_queries", [paper_title])
                for query in queries[:3]:  # Limit to 3 queries
                    papers = self.academic_search.search(
                        topic=query,
                        max_results=15,
                        min_year=start_year,
                        prioritize_recent=True
                    )
                    academic_results.extend(papers)
                
                # Remove duplicates by title
                seen_titles = set()
                unique_academic = []
                for paper in academic_results:
                    title = paper.get("title", "").lower()
                    if title not in seen_titles:
                        unique_academic.append(paper)
                        seen_titles.add(title)
                
                academic_results = unique_academic[:30]  # Limit to 30 papers
                logger.info(f"Found {len(academic_results)} unique academic papers")
            except Exception as e:
                logger.warning(f"Academic search error: {e}")
        
        # Search web if planned
        if search_plan.get("use_web", True) and self.web_search:
            try:
                logger.info("Searching web (DuckDuckGo)...")
                
                queries = search_plan.get("search_queries", [paper_title])
                for query in queries[:2]:  # Limit to 2 queries
                    results = self.web_search.search(query, max_results=10)
                    web_results.extend(results)
                
                # Remove duplicates
                seen_urls = set()
                unique_web = []
                for result in web_results:
                    url = result.get("url", "")
                    if url and url not in seen_urls:
                        unique_web.append(result)
                        seen_urls.add(url)
                
                web_results = unique_web[:15]  # Limit to 15 results
                logger.info(f"Found {len(web_results)} unique web results")
            except Exception as e:
                logger.warning(f"Web search error: {e}")
        
        state["qdrant_results"] = qdrant_results
        state["academic_results"] = academic_results
        state["web_results"] = web_results
        state["current_step"] = "organize_papers"
        
        return state
    
    def _organize_papers_node(self, state: LiteratureReviewState) -> LiteratureReviewState:
        """Organize papers thematically."""
        logger.info("Organizing papers thematically...")
        
        paper_title = state.get("paper_title", "")
        method_description = state.get("method_description", "")
        academic_results = state.get("academic_results", [])
        web_results = state.get("web_results", [])
        
        # Combine all papers
        all_papers = []
        
        # Add academic papers
        for paper in academic_results:
            all_papers.append({
                "title": paper.get("title", ""),
                "authors": ", ".join(paper.get("authors", [])),
                "year": paper.get("year"),
                "venue": paper.get("venue", ""),
                "abstract": paper.get("abstract", ""),
                "url": paper.get("url", ""),
                "doi": paper.get("doi", ""),
                "source": "academic"
            })
        
        # Add web results (convert to paper format)
        for result in web_results:
            all_papers.append({
                "title": result.get("title", ""),
                "authors": result.get("author", ""),
                "year": None,
                "venue": result.get("publication", ""),
                "abstract": result.get("snippet", ""),
                "url": result.get("url", ""),
                "doi": None,
                "source": "web"
            })
        
        # Organize thematically using LLM
        organize_prompt = f"""You are organizing papers for a literature review on: "{paper_title}"

Our method: {method_description}

Papers to organize:
{json.dumps(all_papers[:50], indent=2, default=str)}

Organize papers into these thematic categories:
1. Foundational Approaches (early/classic work)
2. Scalability & Efficiency (methods focusing on efficiency)
3. Theoretical Guarantees (work with theoretical analysis)
4. Recent Advances ({state.get('end_year', datetime.now().year) - 1}-{state.get('end_year', datetime.now().year)}) (very recent work)

For each paper, provide:
- Core contribution (1-2 sentences)
- Key limitation or gap (1 sentence)
- How our method "{method_description}" addresses it (1 sentence)

Output: JSON with keys matching category names, each containing a list of papers with: title, authors, year, venue, contribution, limitation, how_our_method_addresses"""

        try:
            response = self.llm.invoke(organize_prompt)
            organize_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON
            try:
                if "```json" in organize_text:
                    organize_text = organize_text.split("```json")[1].split("```")[0].strip()
                elif "```" in organize_text:
                    organize_text = organize_text.split("```")[1].split("```")[0].strip()
                
                organized = json.loads(organize_text)
            except json.JSONDecodeError:
                logger.warning("Could not parse organization JSON, using simple grouping")
                # Fallback: simple grouping by year
                organized = {
                    "Foundational Approaches": [],
                    "Scalability & Efficiency": [],
                    "Theoretical Guarantees": [],
                    "Recent Advances": []
                }
                for paper in all_papers[:20]:
                    year = paper.get("year")
                    if year and year >= state.get("end_year", datetime.now().year) - 1:
                        organized["Recent Advances"].append(paper)
                    else:
                        organized["Foundational Approaches"].append(paper)
            
            state["organized_papers"] = organized
            logger.info(f"Organized {sum(len(papers) for papers in organized.values())} papers into {len(organized)} categories")
            
        except Exception as e:
            logger.error(f"Organization error: {e}")
            state["organized_papers"] = {"All Papers": all_papers[:20]}
        
        state["current_step"] = "generate_review"
        return state
    
    def _generate_review_node(self, state: LiteratureReviewState) -> LiteratureReviewState:
        """Generate LaTeX literature review section."""
        logger.info("Generating LaTeX literature review section...")
        
        paper_title = state.get("paper_title", "")
        method_description = state.get("method_description", "")
        organized_papers = state.get("organized_papers", {})
        
        # Generate LaTeX section
        latex_prompt = f"""You are writing a literature review section for a JMLR-style paper.

Paper Title: "{paper_title}"
Our Method: {method_description}

Organized Papers:
{json.dumps(organized_papers, indent=2, default=str)}

Write a formal academic literature review section in LaTeX format following JMLR style:

1. Use formal academic English, passive voice, critical tone
2. Organize by the thematic categories provided
3. For each paper mentioned, use citation format: \\cite{{AuthorYear}} (e.g., \\cite{{Smith2023}})
4. Include core contributions, limitations, and how our method addresses gaps
5. Do NOT include introduction or conclusion - only the literature review section
6. Use proper LaTeX formatting: \\section{{...}}, \\subsection{{...}}, \\paragraph{{...}}
7. Be concise but comprehensive - aim for 1500-2500 words total

Output ONLY the LaTeX section content (no markdown, no explanations)."""

        try:
            response = self.llm.invoke(latex_prompt)
            latex_content = response.content if hasattr(response, 'content') else str(response)
            
            # Clean up LaTeX (remove markdown code blocks if present)
            if "```latex" in latex_content:
                latex_content = latex_content.split("```latex")[1].split("```")[0].strip()
            elif "```" in latex_content:
                latex_content = latex_content.split("```")[1].split("```")[0].strip()
            
            state["latex_section"] = latex_content
            logger.info(f"Generated LaTeX section: {len(latex_content)} chars")
            
        except Exception as e:
            logger.error(f"LaTeX generation error: {e}")
            state["latex_section"] = "\\section{Literature Review}\n\n[Error generating literature review]"
        
        state["current_step"] = "format_output"
        return state
    
    def _format_output_node(self, state: LiteratureReviewState) -> LiteratureReviewState:
        """Format final output with BibTeX."""
        logger.info("Formatting output with BibTeX...")
        
        organized_papers = state.get("organized_papers", {})
        latex_section = state.get("latex_section", "")
        
        # Generate BibTeX entries
        bibtex_entries = []
        all_papers = []
        
        for category, papers in organized_papers.items():
            all_papers.extend(papers)
        
        for paper in all_papers:
            title = paper.get("title", "Untitled")
            authors = paper.get("authors", "Unknown")
            year = paper.get("year", "")
            venue = paper.get("venue", "")
            url = paper.get("url", "")
            doi = paper.get("doi", "")
            
            # Generate BibTeX key
            if authors and year:
                first_author = authors.split(",")[0].split()[0] if "," in authors else authors.split()[0]
                bibtex_key = f"{first_author.lower()}{year}"
            else:
                bibtex_key = f"paper{len(bibtex_entries) + 1}"
            
            # Determine entry type
            if "arxiv" in venue.lower() or "arxiv" in (url or "").lower():
                entry_type = "article"
                bibtex = f"@article{{{bibtex_key},\n"
                bibtex += f"  title={{{title}}},\n"
                if authors:
                    bibtex += f"  author={{{authors}}},\n"
                if year:
                    bibtex += f"  year={{{year}}},\n"
                bibtex += f"  journal={{arXiv preprint}},\n"
                if url:
                    bibtex += f"  url={{{url}}},\n"
                if doi:
                    bibtex += f"  doi={{{doi}}},\n"
            else:
                entry_type = "inproceedings" if any(v in venue.lower() for v in ["neurips", "icml", "iclr", "conference"]) else "article"
                bibtex = f"@{entry_type}{{{bibtex_key},\n"
                bibtex += f"  title={{{title}}},\n"
                if authors:
                    bibtex += f"  author={{{authors}}},\n"
                if year:
                    bibtex += f"  year={{{year}}},\n"
                if venue:
                    if entry_type == "inproceedings":
                        bibtex += f"  booktitle={{{venue}}},\n"
                    else:
                        bibtex += f"  journal={{{venue}}},\n"
                if url:
                    bibtex += f"  url={{{url}}},\n"
                if doi:
                    bibtex += f"  doi={{{doi}}},\n"
            
            # Remove trailing comma
            bibtex = bibtex.rstrip(",\n") + "\n"
            bibtex += "}"
            
            bibtex_entries.append(bibtex)
        
        state["bibtex_entries"] = bibtex_entries
        state["current_step"] = "complete"
        
        logger.info(f"Generated {len(bibtex_entries)} BibTeX entries")
        
        return state
    
    def generate_literature_review(
        self,
        paper_title: str,
        method_description: str,
        start_year: int = None,
        end_year: int = None
    ) -> Dict[str, Any]:
        """
        Generate JMLR-style literature review.
        
        Args:
            paper_title: Title of the paper
            method_description: Brief description of the method
            start_year: Start year for search (default: 5 years ago)
            end_year: End year for search (default: current year)
        
        Returns:
            Dictionary with latex_section, bibtex_entries, organized_papers, etc.
        """
        if start_year is None:
            start_year = datetime.now().year - 5
        if end_year is None:
            end_year = datetime.now().year
        
        # Initialize state
        initial_state: LiteratureReviewState = {
            "paper_title": paper_title,
            "method_description": method_description,
            "start_year": start_year,
            "end_year": end_year,
            "messages": [],
            "retrieved_papers": [],
            "qdrant_results": [],
            "web_results": [],
            "academic_results": [],
            "organized_papers": {},
            "latex_section": "",
            "bibtex_entries": [],
            "current_step": "plan_search"
        }
        
        # Execute workflow
        logger.info("=" * 80)
        logger.info("JMLR LITERATURE REVIEW GENERATION STARTED")
        logger.info("=" * 80)
        logger.info(f"Paper Title: {paper_title}")
        logger.info(f"Method: {method_description}")
        logger.info(f"Year Range: {start_year}-{end_year}")
        logger.info("-" * 80)
        
        try:
            final_state = self.workflow.invoke(initial_state)
            
            logger.info("=" * 80)
            logger.info("LITERATURE REVIEW GENERATION COMPLETED")
            logger.info("=" * 80)
            
            return {
                "success": True,
                "latex_section": final_state.get("latex_section", ""),
                "bibtex_entries": final_state.get("bibtex_entries", []),
                "organized_papers": final_state.get("organized_papers", {}),
                "academic_results": final_state.get("academic_results", []),
                "web_results": final_state.get("web_results", []),
                "qdrant_results": final_state.get("qdrant_results", [])
            }
        
        except Exception as e:
            logger.error(f"Literature review generation error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "latex_section": "",
                "bibtex_entries": []
            }

