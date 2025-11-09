"""
Ollama Report Generator
Generate summary reports using LLaMA 3.1 via Ollama (HTTP API or CLI).
"""

import os
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)


def get_save_root() -> Path:
    """Get the save root directory: RAG_performatnece/"""
    # Get project root (two levels up from scripts/06_output_evaluation/)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    save_root = project_root / "RAG_performatnece"
    save_root.mkdir(parents=True, exist_ok=True)
    return save_root


def generate_report_with_ollama31(
    summary_data: Dict[str, Any],
    model: str = "llama3.1:latest",
    base_url: str = "http://localhost:11434"
) -> str:
    """
    Generate summary report using LLaMA 3.1 via Ollama.
    
    Args:
        summary_data: Dictionary containing evaluation results summary
        model: Ollama model name (default: "llama3.1:latest")
        base_url: Ollama API base URL
        
    Returns:
        Generated report text
    """
    # Build prompt from summary data
    prompt = build_report_prompt(summary_data)
    
    # Try HTTP API first
    report_text = None
    try:
        report_text = generate_via_http_api(prompt, model, base_url)
    except Exception as e:
        logger.warning(f"HTTP API failed: {e}, trying CLI fallback")
    
    # Fallback to CLI
    if report_text is None:
        try:
            report_text = generate_via_cli(prompt, model)
        except Exception as e:
            logger.error(f"CLI generation failed: {e}")
            report_text = f"Error generating report: {e}\n\nPrompt was:\n{prompt}"
    
    return report_text


def build_report_prompt(summary_data: Dict[str, Any]) -> str:
    """Build prompt for report generation."""
    prompt = f"""You are an expert data analyst. Analyze the following RAG evaluation results and write a comprehensive summary report.

Evaluation Summary Data:
{json.dumps(summary_data, indent=2)}

Please provide a detailed analysis covering:
1. Overall performance assessment
2. Best performing combinations (embedding × chunker × search mode)
3. Weakest performing combinations
4. Key insights and patterns
5. Recommendations for improvement

Write the report in a clear, professional format suitable for stakeholders.
"""
    return prompt


def generate_via_http_api(
    prompt: str,
    model: str = "llama3.1:latest",
    base_url: str = "http://localhost:11434"
) -> str:
    """Generate report via Ollama HTTP API."""
    # Try /api/generate first
    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")
    except Exception:
        # Try /api/v1/generate fallback
        url_v1 = f"{base_url}/api/v1/generate"
        try:
            response = requests.post(url_v1, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            raise Exception(f"Both API endpoints failed: {e}")


def generate_via_cli(
    prompt: str,
    model: str = "llama3.1:latest"
) -> str:
    """Generate report via Ollama CLI."""
    # Escape prompt for shell
    import shlex
    escaped_prompt = shlex.quote(prompt)
    
    # Build command
    cmd = f'ollama run {model} --no-stream --prompt {escaped_prompt}'
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            return result.stdout
        else:
            raise Exception(f"CLI command failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise Exception("CLI command timed out after 300 seconds")
    except Exception as e:
        raise Exception(f"CLI execution failed: {e}")


def save_report(
    report_text: str,
    summary_data: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Save report to RAG_performatnece/report/summary_report.txt
    
    Args:
        report_text: Generated report text
        summary_data: Optional summary data to save as JSON
        
    Returns:
        Path to saved report file
    """
    save_root = get_save_root()
    report_dir = save_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Save text report
    report_path = report_dir / "summary_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    logger.info(f"Saved report to {report_path}")
    
    # Save JSON summary if provided
    if summary_data:
        json_path = report_dir / "summary_data.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        logger.info(f"Saved summary data to {json_path}")
    
    return report_path


def create_summary_data(results_df) -> Dict[str, Any]:
    """Create summary data dictionary from results DataFrame."""
    import pandas as pd
    import numpy as np
    
    summary = {
        "total_combinations": len(results_df),
        "metrics_summary": {},
        "best_combinations": {},
        "worst_combinations": {}
    }
    
    # Calculate summary statistics for each metric
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_similarity"]
    for metric in metrics:
        if metric in results_df.columns:
            summary["metrics_summary"][metric] = {
                "mean": float(results_df[metric].mean()),
                "std": float(results_df[metric].std()),
                "min": float(results_df[metric].min()),
                "max": float(results_df[metric].max())
            }
    
    # Find best combinations for each metric
    for metric in metrics:
        if metric in results_df.columns:
            best_idx = results_df[metric].idxmax()
            best_row = results_df.loc[best_idx]
            summary["best_combinations"][metric] = {
                "embedding": best_row["embedding_model"],
                "chunker": best_row["chunking_method"],
                "search": best_row["search_mode"],
                "score": float(best_row[metric])
            }
            
            worst_idx = results_df[metric].idxmin()
            worst_row = results_df.loc[worst_idx]
            summary["worst_combinations"][metric] = {
                "embedding": worst_row["embedding_model"],
                "chunker": worst_row["chunking_method"],
                "search": worst_row["search_mode"],
                "score": float(worst_row[metric])
            }
    
    return summary


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create dummy data
    dummy_data = {
        "embedding_model": ["llama3.1:latest", "llama3.1:latest"],
        "chunking_method": ["recursive", "semantic"],
        "search_mode": ["semantic", "bm25"],
        "faithfulness": [0.8, 0.7],
        "answer_relevancy": [0.75, 0.65],
        "context_precision": [0.85, 0.7]
    }
    df = pd.DataFrame(dummy_data)
    
    summary_data = create_summary_data(df)
    report = generate_report_with_ollama31(summary_data)
    save_report(report, summary_data)
    print("Report generated successfully!")

