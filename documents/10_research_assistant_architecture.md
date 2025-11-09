# Research Assistant Architecture

Complete guide to the enhanced research assistant system with specialized agents and validation.

## Overview

The Research Assistant uses a multi-agent workflow orchestrated by LangGraph to generate comprehensive research reports. The system combines web research with document analysis to produce high-quality technical reports.

## Architecture Components

### 1. Specialized Agents

#### WebResearchAgent
- **Purpose:** Performs web searches using DuckDuckGo
- **Features:**
  - Relevance filtering
  - Quality checks (URL validation, content filtering)
  - Result enhancement
- **Input:** Research query
- **Output:** Filtered web search results

#### TechnicalAnalystAgent
- **Purpose:** Synthesizes information from multiple sources
- **Features:**
  - Combines web results and document chunks
  - Creates coherent technical analysis
  - Context-aware synthesis
- **Input:** Web results + document chunks
- **Output:** Synthesized context

#### ReportWriterAgent
- **Purpose:** Generates structured report sections
- **Features:**
  - Parallel section generation
  - Reference collection
  - Report formatting
- **Input:** Synthesized context
- **Output:** Complete report draft

#### ValidationAgent
- **Purpose:** Quality control and validation
- **Features:**
  - Section completeness checks
  - Content quality validation
  - URL validation
  - Iterative improvement loop (max 3 iterations)
- **Input:** Report draft
- **Output:** Validation status + feedback

### 2. Workflow Orchestration

The workflow follows this pattern:

```
Web Research → Analysis → Writing → Validation → Complete
                                    ↓ (if failed)
                                 Writing (retry)
```

**State Management:**
- Uses `ResearchState` TypedDict for type safety
- Tracks current step, iterations, and validation status
- Preserves context across iterations

### 3. Validation Checks

The ValidationAgent performs:

1. **Section Completeness:** All required sections present
2. **Content Quality:** Sections not empty or error-filled
3. **Reference Validation:** References present and URLs valid
4. **Length Checks:** Report meets minimum length requirements
5. **Placeholder Detection:** No placeholder text remains

## Usage

### Basic Usage

```python
from scripts.07_research_workflow.research_assistant import ResearchAssistant
from scripts.03_retrieval.retrieval import RAGRetrievalEngine

# Initialize
retrieval_engine = RAGRetrievalEngine(
    use_ollama=True,
    vector_db_type="Qdrant",
    embedding_model="llama3.1"
)

assistant = ResearchAssistant(
    retrieval_engine=retrieval_engine,
    use_ollama=True,
    framework="LangChain"
)

# Generate report
result = assistant.run_autonomous_research(
    query="Recent advances in novelty detection",
    use_case="academic_paper_review",
    use_memory=False,
    memory_context=""
)

if result.get("success"):
    report = result.get("report", {})
    print(report.get("markdown", ""))
```

### Enhanced Workflow Features

The enhanced workflow automatically:
- Uses specialized agents for each step
- Validates report quality
- Retries up to 3 times if validation fails
- Provides detailed feedback on issues

## Configuration

### Web Search Settings

- **Max Results:** 10 (configurable)
- **Provider:** DuckDuckGo (free, no API key)
- **Caching:** Enabled for faster repeated searches

### Validation Settings

- **Max Iterations:** 3
- **Required Sections:** Abstract, Introduction, Research Findings, Conclusion
- **Minimum Report Length:** 500 characters

## Error Handling

All agents include comprehensive error handling:
- API failures are caught and logged
- Fallback to basic workflow if enhanced workflow fails
- Detailed error messages for debugging

## Security Features

- URL validation for all web sources
- Content sanitization
- Rate limiting considerations
- Safe error messages (no sensitive data exposure)

## Performance Optimizations

- Parallel section generation
- Web search result caching
- Context truncation for large inputs
- Efficient state management

## Troubleshooting

### No References Found

1. Check internet connection
2. Verify `duckduckgo-search` is installed: `pip install duckduckgo-search`
3. Check logs for web search errors
4. Try a different query

### Validation Failing

1. Check if all sections generated successfully
2. Review validation feedback in logs
3. Ensure query is specific enough
4. Check LLM (Ollama) is running

### Report Generation Errors

1. Check Ollama is running: `ollama list`
2. Verify model is installed: `ollama pull llama3.1`
3. Check logs for detailed error messages
4. Ensure sufficient system resources

## Next Steps

- See `documents/05_running_the_app.md` for running the app
- See `documents/04_readme_streamlit.md` for UI usage

