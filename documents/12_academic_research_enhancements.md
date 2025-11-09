# Academic Research Enhancements

This document describes the enhancements made to the research assistant system based on ideas from the LangGraph JMLR survey pipeline specification.

## Overview

The research assistant has been enhanced with academic source integration, PDF processing, structured paper summarization, and reproducibility logging capabilities. These features enable the system to generate higher-quality research overview papers with verified references and comprehensive documentation.

## New Modules

### 1. `scripts/07_research_workflow/academic_search.py`

**Purpose:** Integration with academic sources (arXiv, Semantic Scholar)

**Features:**
- arXiv API integration (free, no API key required)
- Semantic Scholar REST API (free tier, optional API key)
- Metadata extraction: title, authors, year, venue, DOI, URL, abstract
- Relevance ranking and filtering by recency (2020-2025 prioritized)
- Citation count consideration
- Caching for faster repeated searches

**Usage:**
```python
from scripts.07_research_workflow.academic_search import AcademicSearch

academic_search = AcademicSearch(use_cache=True)
papers = academic_search.search(
    topic="Explainability of Large Language Models",
    max_results=40,
    min_year=2020,
    prioritize_recent=True
)
```

**Dependencies:**
- `arxiv` (pip install arxiv)
- `requests` (pip install requests)

### 2. `scripts/01_data_ingestion_and_preprocessing/pdf_handler.py`

**Purpose:** Download PDFs and extract text content

**Features:**
- PDF downloading from URLs (arXiv, DOI links, direct PDFs)
- Text extraction using pdfplumber
- Metadata preservation
- Error handling and retry logic
- Automatic caching of extracted text

**Usage:**
```python
from scripts.01_data_ingestion_and_preprocessing.pdf_handler import PDFHandler

pdf_handler = PDFHandler(output_dir="output/papers", parsed_dir="output/parsed_txt")
processed_paper = pdf_handler.process_paper(paper_metadata)
# Returns: paper_metadata with 'pdf_path', 'full_text', 'text_length' fields
```

**Dependencies:**
- `requests` (pip install requests)
- `pdfplumber` (pip install pdfplumber)

### 3. `scripts/07_research_workflow/paper_summarizer.py`

**Purpose:** Structured summarization of academic papers using LLM

**Features:**
- 150-250 word summaries
- Contribution extraction (3-6 bullet points)
- Keyword extraction (3-6 keywords)
- Taxonomy categorization (7 categories)
- Handles both full-text and abstract-only papers

**Taxonomy Categories:**
- Local Explanations (LIME, SHAP variants)
- Attribution & Attention Methods
- Representation & Probing
- Mechanistic Interpretability
- Behavioral / Intervention Methods
- Evaluation & Metrics
- Other

**Usage:**
```python
from scripts.07_research_workflow.paper_summarizer import PaperSummarizer
from scripts.02_query_completion.synthesis import ResearchSynthesizer

synthesizer = ResearchSynthesizer(use_ollama=True)
summarizer = PaperSummarizer(synthesizer=synthesizer)
summarized_paper = summarizer.summarize_paper(paper_metadata)
```

### 4. `scripts/08_utilities/reproducibility_logger.py`

**Purpose:** Track queries, data sources, versions, and LLM calls for reproducibility

**Features:**
- Log all search queries and parameters
- Track data sources and versions
- Log LLM prompts and responses
- Save environment and dependency versions
- Generate reproducibility report

**Output Files:**
- `output/queries.json` - All search queries
- `output/llm_calls.log` - LLM prompts and responses
- `output/requirements.txt` - Python dependencies
- `output/environment.json` - Environment info
- `output/paper_collection.json` - Collected papers metadata
- `output/reproducibility_report.md` - Comprehensive report

**Usage:**
```python
from scripts.08_utilities.reproducibility_logger import ReproducibilityLogger

repro_logger = ReproducibilityLogger(output_dir="output")
repro_logger.log_query(query="...", source="arxiv", max_results=40)
repro_logger.log_llm_call(prompt="...", response="...", model="llama3.1")
repro_logger.generate_report()
```

## Enhanced Workflow

### Research Overview Workflow Enhancements

The `ResearchOverviewWorkflow` class has been enhanced with:

1. **Academic Source Integration:**
   - Optional arXiv and Semantic Scholar search
   - Automatic ranking and filtering by relevance and recency
   - Metadata extraction and normalization

2. **PDF Processing:**
   - Optional PDF downloading and text extraction
   - Full-text analysis for better summarization
   - Caching of extracted text

3. **Structured Summarization:**
   - Automatic paper summarization with contributions and keywords
   - Taxonomy categorization
   - Enhanced context for report generation

4. **Human-in-the-Loop Checkpoint:**
   - After collecting papers, save to `output/candidates.json`
   - Wait for user confirmation via `output/user_confirm.true`
   - Prevents blind hallucination by allowing review before synthesis

5. **Reproducibility Logging:**
   - Automatic logging of all queries and LLM calls
   - Environment and dependency tracking
   - Comprehensive reproducibility report generation

**Usage:**
```python
from scripts.07_research_workflow.research_overview_workflow import ResearchOverviewWorkflow
from scripts.02_query_completion.synthesis import ResearchSynthesizer
from scripts.05_output_generation.report_formatter import ReportFormatter

synthesizer = ResearchSynthesizer(use_ollama=True)
formatter = ReportFormatter()

workflow = ResearchOverviewWorkflow(
    synthesizer=synthesizer,
    report_formatter=formatter,
    use_academic_sources=True,      # Enable arXiv/Semantic Scholar
    use_pdf_processing=True,         # Enable PDF download/extraction
    enable_checkpoint=True           # Enable human-in-the-loop
)

result = workflow.execute(
    topic="Explainability of Large Language Models",
    context="...",
    references=[],
    max_academic_papers=40,
    min_year=2020
)
```

## Workflow Steps

1. **Paper Collection:**
   - Search arXiv and Semantic Scholar
   - Rank and filter by relevance and recency
   - Save candidates to `output/candidates.json`

2. **Checkpoint (if enabled):**
   - Wait for user to review `output/candidates.json`
   - User creates `output/user_confirm.true` to proceed

3. **PDF Processing (if enabled):**
   - Download PDFs for collected papers
   - Extract text content
   - Save to `output/papers/` and `output/parsed_txt/`

4. **Summarization (if PDF processing enabled):**
   - Generate structured summaries
   - Extract contributions and keywords
   - Categorize by taxonomy

5. **Report Generation:**
   - Use enhanced references (with summaries and contributions)
   - Generate cohesive survey paper
   - Format with BibTeX references

6. **Reproducibility:**
   - Generate reproducibility report
   - Save all logs and metadata

## Configuration

### Environment Variables

- `SEMANTIC_SCHOLAR_API_KEY` (optional): Semantic Scholar API key for higher rate limits
- `OPENAI_API_KEY` (optional): If using OpenAI instead of Ollama

### Dependencies

Add to `requirements.txt`:
```
arxiv>=2.1.0
pdfplumber>=0.10.0
requests>=2.31.0
```

## Benefits

1. **Higher Quality References:**
   - Verified DOIs and URLs from academic sources
   - No hallucinated citations
   - Proper metadata (authors, year, venue)

2. **Better Context:**
   - Full-text analysis when PDFs available
   - Structured summaries with contributions
   - Taxonomy categorization for organization

3. **Reproducibility:**
   - Complete audit trail of queries and LLM calls
   - Environment and dependency tracking
   - Easy to reproduce results

4. **Human Control:**
   - Review papers before synthesis
   - Prevent hallucination
   - Quality assurance checkpoint

## Limitations

1. **PDF Download:**
   - Some papers may not have publicly available PDFs
   - DOI resolution may fail for paywalled content
   - Large PDFs may take time to process

2. **Rate Limits:**
   - Semantic Scholar free tier: 100 requests per 5 minutes
   - arXiv: No official limit, but be respectful
   - Implemented delays and caching to mitigate

3. **Checkpoint:**
   - Currently uses file-based checkpoint (works in CLI)
   - In Streamlit, checkpoint would need UI implementation

## Future Enhancements

1. **Additional Sources:**
   - ACL Anthology integration
   - IEEE Xplore (requires API key)
   - ACM Digital Library (requires API key)

2. **Enhanced Ranking:**
   - Semantic similarity using embeddings
   - Citation network analysis
   - Impact factor consideration

3. **Streamlit UI Integration:**
   - Checkpoint UI in Streamlit app
   - Paper review interface
   - Progress tracking

4. **BibTeX Enhancement:**
   - Better DOI resolution
   - Venue normalization
   - Author name disambiguation

## Example Output Structure

```
output/
├── candidates.json              # Papers for review (checkpoint)
├── user_confirm.true           # User confirmation file
├── queries.json                # All search queries
├── llm_calls.log               # LLM prompts and responses
├── requirements.txt            # Python dependencies
├── environment.json             # Environment info
├── paper_collection.json        # Collected papers metadata
├── reproducibility_report.md   # Comprehensive report
├── papers/                     # Downloaded PDFs
│   ├── 2301.01234.pdf
│   └── ...
├── parsed_txt/                 # Extracted text
│   ├── 2301.01234.txt
│   └── ...
└── draft.md                    # Generated survey paper
```

## Integration with Existing System

These enhancements are **optional** and **backward compatible**. The existing workflow continues to work as before. To enable new features:

1. Install additional dependencies: `pip install arxiv pdfplumber`
2. Set flags when initializing `ResearchOverviewWorkflow`:
   - `use_academic_sources=True`
   - `use_pdf_processing=True`
   - `enable_checkpoint=True`

The system gracefully handles missing dependencies and falls back to web search if academic sources are unavailable.

