"""
Prompts package for research overview generation.
All prompt templates are loaded when the Streamlit app starts.
"""

import os
from pathlib import Path

# Get the prompts directory
PROMPTS_DIR = Path(__file__).parent

# Load all prompt templates
def load_prompt_template(template_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    template_path = PROMPTS_DIR / f"{template_name}.txt"
    if template_path.exists():
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise FileNotFoundError(f"Prompt template not found: {template_name}")

__all__ = ['load_prompt_template', 'PROMPTS_DIR']

