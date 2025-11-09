"""
HTML Exporter Module
Handles saving research overview reports as HTML web pages in organized directory structure.
"""

import os
import re
import logging
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import markdown library
try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    logger.warning("markdown not available. Install with: pip install markdown")


def sanitize_topic_name(topic: str) -> str:
    """
    Sanitize topic name for use as directory/filename.
    
    Args:
        topic: Research topic string
        
    Returns:
        Sanitized string safe for filesystem
    """
    # Remove special characters, keep alphanumeric, spaces, hyphens, underscores
    sanitized = re.sub(r'[^\w\s-]', '', topic)
    # Replace spaces with underscores
    sanitized = re.sub(r'\s+', '_', sanitized)
    # Remove multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    # Limit length
    sanitized = sanitized[:100]
    return sanitized.lower() if sanitized else "research_overview"


def save_research_overview_html(
    report: Dict,
    topic: str,
    base_dir: str = "html"
) -> str:
    """
    Save research overview report as HTML webpage.
    
    Args:
        report: Report dictionary with markdown content
        topic: Research topic (used for subdirectory name)
        base_dir: Base directory for HTML files (default: "html")
        
    Returns:
        Path to saved HTML file
    """
    # Sanitize topic name for directory
    topic_dir = sanitize_topic_name(topic)
    
    # Create directory structure: html/{topic}/
    html_base = Path(base_dir)
    topic_path = html_base / topic_dir
    topic_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"research_overview_{timestamp}.html"
    filepath = topic_path / filename
    
    # Get markdown content
    markdown_content = report.get("markdown", report.get("plain_text", ""))
    
    # Convert markdown to HTML
    if MARKDOWN_AVAILABLE:
        try:
            # Try with extensions for better formatting
            try:
                html_content = markdown.markdown(
                    markdown_content,
                    extensions=['extra', 'codehilite', 'tables', 'fenced_code']
                )
            except:
                # Fallback to basic markdown
                html_content = markdown.markdown(markdown_content)
        except Exception as e:
            logger.warning(f"Markdown conversion error: {e}")
            html_content = _markdown_to_html_fallback(markdown_content)
    else:
        html_content = _markdown_to_html_fallback(markdown_content)
    
    # Get metadata - use topic parameter as fallback
    metadata = report.get("metadata", {})
    report_title = metadata.get("topic", topic) or topic
    timestamp_str = metadata.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Create full HTML document with modern styling
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.7;
            color: #2c3e50;
            background-color: #f8f9fa;
            padding: 20px;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background-color: #ffffff;
            padding: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        header {{
            border-bottom: 3px solid #3498db;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        .metadata {{
            color: #7f8c8d;
            font-size: 0.9em;
            font-style: italic;
            margin-top: 10px;
        }}
        h2 {{
            color: #34495e;
            font-size: 1.8em;
            margin-top: 40px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
        }}
        h3 {{
            color: #555;
            font-size: 1.4em;
            margin-top: 30px;
            margin-bottom: 12px;
        }}
        h4 {{
            color: #666;
            font-size: 1.2em;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        p {{
            margin-bottom: 15px;
            text-align: justify;
            color: #34495e;
        }}
        ul, ol {{
            margin-left: 30px;
            margin-bottom: 15px;
        }}
        li {{
            margin-bottom: 8px;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', 'Monaco', monospace;
            font-size: 0.9em;
            color: #e74c3c;
        }}
        pre {{
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 20px 0;
            border-left: 4px solid #3498db;
        }}
        pre code {{
            background-color: transparent;
            color: #ecf0f1;
            padding: 0;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            padding-left: 20px;
            margin: 20px 0;
            color: #555;
            font-style: italic;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ecf0f1;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .references {{
            margin-top: 50px;
            padding-top: 30px;
            border-top: 2px solid #ecf0f1;
        }}
        .reference-item {{
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-left: 3px solid #3498db;
            border-radius: 4px;
        }}
        .reference-item strong {{
            color: #2c3e50;
        }}
        hr {{
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 40px 0;
        }}
        @media print {{
            body {{
                background-color: white;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
        @media (max-width: 768px) {{
            .container {{
                padding: 20px;
            }}
            h1 {{
                font-size: 2em;
            }}
            h2 {{
                font-size: 1.5em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{report_title}</h1>
            <div class="metadata">Generated: {timestamp_str}</div>
        </header>
        <main>
            {html_content}
        </main>
    </div>
</body>
</html>"""
    
    # Write HTML file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        logger.info(f"Saved HTML report to {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Error saving HTML report: {e}")
        raise


def _markdown_to_html_fallback(markdown_text: str) -> str:
    """
    Fallback markdown to HTML converter (basic implementation).
    
    Args:
        markdown_text: Markdown text
        
    Returns:
        HTML text
    """
    html = markdown_text
    
    # Convert headers
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    
    # Convert bold
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    
    # Convert italic
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
    
    # Convert code blocks
    html = re.sub(r'```([^`]+)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
    
    # Convert links
    html = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2">\1</a>', html)
    
    # Convert line breaks to paragraphs
    paragraphs = html.split('\n\n')
    html = '\n'.join(f'<p>{p.strip()}</p>' if p.strip() and not p.strip().startswith('<') else p for p in paragraphs)
    
    return html

