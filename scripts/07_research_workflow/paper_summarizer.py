"""
Paper Summarizer Module
Provides structured summarization of academic papers using LLM.

Features:
- 150-300 word summaries
- Contribution extraction (3-6 bullet points)
- Keyword extraction
- Taxonomy categorization
- Metadata preservation
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Import synthesis module for LLM access
try:
    # Use importlib for modules starting with numbers
    import importlib
    synthesis_module = importlib.import_module('scripts.02_query_completion.synthesis')
    ResearchSynthesizer = synthesis_module.ResearchSynthesizer
except ImportError:
    ResearchSynthesizer = None
    logger.warning("ResearchSynthesizer not available")


PROMPT_SUMMARY = """You are an expert ML researcher. Given the extracted full text (or abstract) below, produce a structured summary.

**Paper Metadata:**
- Title: {title}
- Authors: {authors}
- Year: {year}
- Venue: {venue}
- Abstract: {abstract}

**Full Text (if available):**
{fulltext}

**Task:** Produce a structured summary with:

1. **Summary (150-250 words):** Concise summary emphasizing contributions and methods
2. **Contributions (3-6 bullet points):** Each contribution should be 1 line, focusing on:
   - Core innovation
   - Problem solved
   - Novelty compared to previous work
   - Technical details (at appropriate level)
   - Impact on the field
3. **Keywords (3-6):** Relevant keywords for this paper
4. **Taxonomy Category:** Assign to ONE of these categories:
   - Local Explanations (LIME, SHAP variants, local surrogate models)
   - Attribution & Attention Methods (attention analysis, gradient-based attributions)
   - Representation & Probing (concept activation vectors, probing classifiers)
   - Mechanistic Interpretability (neuron/module analysis, circuit analysis)
   - Behavioral / Intervention Methods (counterfactuals, behavioral tests)
   - Evaluation & Metrics (evaluation frameworks, metrics for explainability)
   - Other (if none of the above fit)

**Output Format:** Return valid JSON with these exact keys:
{{
  "summary": "150-250 word summary...",
  "contributions": ["contribution 1", "contribution 2", ...],
  "keywords": ["keyword1", "keyword2", ...],
  "taxonomy": "category name",
  "taxonomy_justification": "brief explanation of why this category",
  "note_source": "fulltext" or "abstract_only"
}}

Be concise and accurate. Do not invent information not present in the text."""


class PaperSummarizer:
    """Summarize academic papers into structured format."""
    
    def __init__(self, synthesizer: Optional[Any] = None, use_ollama: bool = True):
        """
        Initialize paper summarizer.
        
        Args:
            synthesizer: ResearchSynthesizer instance (optional, will create if None)
            use_ollama: Use Ollama for LLM (if synthesizer not provided)
        """
        if synthesizer:
            self.synthesizer = synthesizer
        elif ResearchSynthesizer:
            self.synthesizer = ResearchSynthesizer(use_ollama=use_ollama)
        else:
            self.synthesizer = None
            logger.warning("No synthesizer available. Paper summarization will not work.")
        
        logger.info("PaperSummarizer initialized")
    
    def summarize_paper(self, paper_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize a paper into structured format.
        
        Args:
            paper_metadata: Paper metadata dictionary with:
                - title, authors, year, venue, abstract
                - full_text (optional, extracted from PDF)
                - id, url, doi, etc.
        
        Returns:
            Updated paper metadata with summary fields:
                - summary: 150-250 word summary
                - contributions: List of contribution strings
                - keywords: List of keywords
                - taxonomy: Taxonomy category
                - taxonomy_justification: Why this category
                - note_source: "fulltext" or "abstract_only"
        """
        if not self.synthesizer or not self.synthesizer.llm:
            logger.error("LLM not available for summarization")
            return paper_metadata
        
        title = paper_metadata.get("title", "")
        authors = ", ".join(paper_metadata.get("authors", []))
        year = paper_metadata.get("year", "")
        venue = paper_metadata.get("venue", "")
        abstract = paper_metadata.get("abstract", "")
        full_text = paper_metadata.get("full_text", "")
        
        # Determine if we have full text or just abstract
        has_fulltext = bool(full_text and len(full_text) > 500)  # At least 500 chars
        note_source = "fulltext" if has_fulltext else "abstract_only"
        
        # Truncate full text if too long (keep first 10000 chars for context)
        if has_fulltext:
            text_to_use = full_text[:10000] + "..." if len(full_text) > 10000 else full_text
        else:
            text_to_use = abstract
        
        # Format prompt
        prompt = PROMPT_SUMMARY.format(
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            abstract=abstract[:1000] if abstract else "No abstract available",  # Limit abstract length
            fulltext=text_to_use
        )
        
        try:
            logger.info(f"Summarizing paper: {title[:60]}...")
            
            # Call LLM
            response = self.synthesizer.llm.invoke(prompt)
            
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Parse JSON response
            try:
                # Try to extract JSON from response (might have markdown code blocks)
                response_text = response_text.strip()
                if "```json" in response_text:
                    # Extract JSON from markdown code block
                    start = response_text.find("```json") + 7
                    end = response_text.find("```", start)
                    response_text = response_text[start:end].strip()
                elif "```" in response_text:
                    # Extract from generic code block
                    start = response_text.find("```") + 3
                    end = response_text.find("```", start)
                    response_text = response_text[start:end].strip()
                
                summary_data = json.loads(response_text)
                
                # Validate required fields
                if "summary" not in summary_data:
                    summary_data["summary"] = ""
                if "contributions" not in summary_data:
                    summary_data["contributions"] = []
                if "keywords" not in summary_data:
                    summary_data["keywords"] = []
                if "taxonomy" not in summary_data:
                    summary_data["taxonomy"] = "Other"
                if "taxonomy_justification" not in summary_data:
                    summary_data["taxonomy_justification"] = ""
                if "note_source" not in summary_data:
                    summary_data["note_source"] = note_source
                
                # Update paper metadata
                updated_metadata = paper_metadata.copy()
                updated_metadata.update(summary_data)
                
                logger.info(f"Summary generated: {len(summary_data.get('summary', ''))} chars, "
                          f"{len(summary_data.get('contributions', []))} contributions, "
                          f"taxonomy: {summary_data.get('taxonomy', 'Unknown')}")
                
                return updated_metadata
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                logger.debug(f"Response text: {response_text[:500]}")
                
                # Fallback: create basic summary structure
                return {
                    **paper_metadata,
                    "summary": response_text[:500],  # Use first 500 chars as summary
                    "contributions": [],
                    "keywords": [],
                    "taxonomy": "Other",
                    "taxonomy_justification": "Could not parse LLM response",
                    "note_source": note_source
                }
        
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return metadata with error note
            return {
                **paper_metadata,
                "summary": f"[Error during summarization: {str(e)}]",
                "contributions": [],
                "keywords": [],
                "taxonomy": "Other",
                "taxonomy_justification": "Error during categorization",
                "note_source": note_source
            }
    
    def summarize_batch(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Summarize multiple papers.
        
        Args:
            papers: List of paper metadata dictionaries
        
        Returns:
            List of updated paper metadata dictionaries with summaries
        """
        summarized_papers = []
        
        for i, paper in enumerate(papers, 1):
            logger.info(f"Summarizing paper {i}/{len(papers)}: {paper.get('title', 'Unknown')[:60]}")
            summarized = self.summarize_paper(paper)
            summarized_papers.append(summarized)
        
        logger.info(f"Summarized {len(summarized_papers)} papers")
        return summarized_papers

