"""
Reproducibility Logger Module
Tracks queries, data sources, versions, and LLM calls for reproducibility.

Features:
- Log all queries and search parameters
- Track data sources and versions
- Log LLM prompts and responses
- Save environment and dependency versions
- Generate reproducibility report
"""

import os
import json
import logging
import subprocess
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ReproducibilityLogger:
    """Log research process for reproducibility."""
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize reproducibility logger.
        
        Args:
            output_dir: Output directory for logs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log files
        self.queries_log = self.output_dir / "queries.json"
        self.llm_calls_log = self.output_dir / "llm_calls.log"
        self.requirements_log = self.output_dir / "requirements.txt"
        self.reproducibility_report = self.output_dir / "reproducibility_report.md"
        
        # Initialize logs
        self.queries = []
        self.llm_calls = []
        
        # Log environment info
        self._log_environment()
        
        logger.info(f"ReproducibilityLogger initialized: {self.output_dir}")
    
    def _log_environment(self):
        """Log environment and dependency versions."""
        try:
            # Try to get pip freeze output
            result = subprocess.run(
                ["pip", "freeze"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                with open(self.requirements_log, 'w', encoding='utf-8') as f:
                    f.write(result.stdout)
                logger.info(f"Saved requirements to {self.requirements_log}")
        except Exception as e:
            logger.warning(f"Failed to capture requirements: {e}")
        
        # Log Python version
        import sys
        env_info = {
            "python_version": sys.version,
            "platform": sys.platform,
            "timestamp": datetime.now().isoformat()
        }
        
        env_log_path = self.output_dir / "environment.json"
        with open(env_log_path, 'w', encoding='utf-8') as f:
            json.dump(env_info, f, indent=2)
    
    def log_query(
        self,
        query: str,
        source: str,
        max_results: int,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Log a search query.
        
        Args:
            query: Search query string
            source: Source name ("arxiv", "semantic_scholar", "duckduckgo", etc.)
            max_results: Maximum results requested
            parameters: Additional parameters (filters, date ranges, etc.)
        """
        query_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "source": source,
            "max_results": max_results,
            "parameters": parameters or {}
        }
        
        self.queries.append(query_entry)
        
        # Save to file immediately
        try:
            with open(self.queries_log, 'w', encoding='utf-8') as f:
                json.dump(self.queries, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save query log: {e}")
        
        logger.debug(f"Logged query: {source} - {query[:50]}...")
    
    def log_llm_call(
        self,
        prompt: str,
        response: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log an LLM call for reproducibility.
        
        Args:
            prompt: Input prompt
            response: LLM response
            model: Model name/identifier
            temperature: Temperature setting
            max_tokens: Max tokens setting
            metadata: Additional metadata
        """
        call_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "prompt": prompt[:1000] + "..." if len(prompt) > 1000 else prompt,  # Truncate for storage
            "response": response[:2000] + "..." if len(response) > 2000 else response,  # Truncate
            "metadata": metadata or {}
        }
        
        self.llm_calls.append(call_entry)
        
        # Append to log file
        try:
            with open(self.llm_calls_log, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Timestamp: {call_entry['timestamp']}\n")
                f.write(f"Model: {model}\n")
                f.write(f"Temperature: {temperature}, Max Tokens: {max_tokens}\n")
                f.write(f"Prompt ({call_entry['prompt_length']} chars):\n{call_entry['prompt']}\n")
                f.write(f"Response ({call_entry['response_length']} chars):\n{call_entry['response']}\n")
                f.write(f"{'='*80}\n")
        except Exception as e:
            logger.warning(f"Failed to save LLM call log: {e}")
        
        logger.debug(f"Logged LLM call: {model} - {len(prompt)} chars prompt, {len(response)} chars response")
    
    def log_paper_collection(
        self,
        papers: List[Dict[str, Any]],
        topic: str,
        filters: Optional[Dict[str, Any]] = None
    ):
        """
        Log paper collection results.
        
        Args:
            papers: List of collected paper metadata
            topic: Research topic
            filters: Applied filters (year range, venue, etc.)
        """
        collection_log = {
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "paper_count": len(papers),
            "filters": filters or {},
            "papers": [
                {
                    "id": p.get("id"),
                    "title": p.get("title"),
                    "year": p.get("year"),
                    "venue": p.get("venue"),
                    "source": p.get("source")
                }
                for p in papers
            ]
        }
        
        collection_log_path = self.output_dir / "paper_collection.json"
        try:
            with open(collection_log_path, 'w', encoding='utf-8') as f:
                json.dump(collection_log, f, indent=2)
            logger.info(f"Logged paper collection: {len(papers)} papers for topic '{topic}'")
        except Exception as e:
            logger.warning(f"Failed to save paper collection log: {e}")
    
    def generate_report(self) -> str:
        """
        Generate reproducibility report.
        
        Returns:
            Markdown report string
        """
        report_lines = [
            "# Reproducibility Report",
            "",
            f"**Generated:** {datetime.now().isoformat()}",
            "",
            "## Environment",
            "",
            f"- Python version: Logged in `environment.json`",
            f"- Dependencies: See `requirements.txt`",
            "",
            "## Queries",
            "",
            f"Total queries logged: {len(self.queries)}",
            ""
        ]
        
        # Add query details
        for i, query in enumerate(self.queries, 1):
            report_lines.extend([
                f"### Query {i}",
                f"- **Timestamp:** {query['timestamp']}",
                f"- **Source:** {query['source']}",
                f"- **Query:** {query['query']}",
                f"- **Max Results:** {query['max_results']}",
                ""
            ])
        
        report_lines.extend([
            "## LLM Calls",
            "",
            f"Total LLM calls logged: {len(self.llm_calls)}",
            ""
        ])
        
        # Add LLM call summary
        if self.llm_calls:
            models_used = set(call['model'] for call in self.llm_calls)
            report_lines.extend([
                f"**Models used:** {', '.join(models_used)}",
                "",
                "See `llm_calls.log` for full details.",
                ""
            ])
        
        report_lines.extend([
            "## Data Sources",
            "",
            "- See `paper_collection.json` for collected papers",
            "- See `queries.json` for all search queries",
            "",
            "## Reproducibility Checklist",
            "",
            "- [x] All queries logged",
            "- [x] LLM calls logged with prompts and responses",
            "- [x] Environment and dependencies recorded",
            "- [x] Paper collection metadata saved",
            ""
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save report
        try:
            with open(self.reproducibility_report, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Generated reproducibility report: {self.reproducibility_report}")
        except Exception as e:
            logger.warning(f"Failed to save reproducibility report: {e}")
        
        return report_text

