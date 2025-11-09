# Research Overview Workflow

Complete guide to the LangGraph-based research overview generation system.

## Overview

The Research Overview Workflow generates comprehensive scholarly survey papers using specialized agents for each section. It uses detailed prompt templates stored in the `prompts/` directory and follows a sequential LangGraph workflow.

## Architecture

### Workflow Structure

The workflow consists of 9 specialized agents, each responsible for generating a specific section:

1. **AbstractIntroductionAgent** - Abstract & Introduction (300-500 words)
2. **BackgroundFoundationsAgent** - Background & Foundations
3. **TaxonomyClassificationAgent** - Taxonomy & Classification Scheme
4. **RecentAdvancesAgent** - Recent Advances (2020-2025)
5. **ApplicationsUseCasesAgent** - Applications & Use Cases
6. **ComparativeAnalysisAgent** - Comparative Analysis & Synthesis
7. **ChallengesLimitationsAgent** - Open Challenges & Limitations
8. **FutureDirectionsAgent** - Future Research Directions
9. **ConclusionAgent** - Conclusion

### Sequential Flow

```
Abstract/Intro → Background → Taxonomy → Recent Advances → 
Applications → Comparative → Challenges → Future → Conclusion → Formatting
```

## Prompt Templates

All prompt templates are stored in the `prompts/` directory and loaded when the Streamlit app starts.

### Available Templates

- `abstract_introduction.txt` - Abstract and Introduction generation
- `background_foundations.txt` - Background and foundational concepts
- `taxonomy_classification.txt` - Taxonomy and classification scheme
- `recent_advances.txt` - Recent advances (2020-2025)
- `applications_use_cases.txt` - Applications and use cases
- `comparative_analysis.txt` - Comparative analysis and synthesis
- `challenges_limitations.txt` - Challenges and limitations
- `future_directions.txt` - Future research directions
- `conclusion.txt` - Conclusion section

### Template Format

Templates use Python string formatting with placeholders:
- `{topic}` - Research topic
- `{context}` - Synthesized research context
- `{references}` - Formatted list of references
- Additional placeholders for section-specific context

## Usage

### In Streamlit App

1. Navigate to the "📚 Research Overview" tab
2. Enter your research topic
3. Configure options:
   - Use web research (enabled by default)
   - Max web results (5-20, default: 10)
4. Click "Generate Research Overview"
5. Wait for generation (may take several minutes)
6. Review and download the generated paper

### Programmatic Usage

```python
from scripts.07_research_workflow.research_overview_workflow import ResearchOverviewWorkflow
from scripts.02_query_completion.synthesis import ResearchSynthesizer
from scripts.05_output_generation.report_formatter import ReportFormatter

# Initialize components
synthesizer = ResearchSynthesizer(use_ollama=True)
report_formatter = ReportFormatter()

# Create workflow
workflow = ResearchOverviewWorkflow(
    synthesizer=synthesizer,
    report_formatter=report_formatter,
    retrieval_engine=retrieval_engine  # Optional
)

# Execute
result = workflow.execute(
    topic="Explainability of Large Language Models",
    context=synthesized_context,
    references=web_references,
    use_web_research=True
)

# Access report
report = result["report"]
markdown = report["markdown"]
sections = report["sections"]
```

## Features

### 1. Comprehensive Structure

Generates all sections required for a scholarly survey paper:
- Abstract and Introduction
- Background and Foundations
- Taxonomy and Classification
- Recent Advances (2020-2025)
- Applications and Use Cases
- Comparative Analysis
- Challenges and Limitations
- Future Research Directions
- Conclusion

### 2. Academic Quality

- Formal academic tone
- Proper citation formatting
- Structured organization
- Critical analysis and synthesis
- Reference management

### 3. Reference Handling

- Formats references with titles, URLs, authors, dates
- Supports academic citation style
- Handles missing information gracefully
- Includes placeholder format when details unavailable

### 4. Error Handling

- Continues generation even if one section fails
- Logs errors for debugging
- Provides fallback templates
- Reports errors in final output

## Configuration

### Prompt Customization

Edit prompt templates in `prompts/` directory to customize:
- Writing style
- Section structure
- Required information
- Formatting requirements

### Workflow Customization

Modify `ResearchOverviewWorkflow` class to:
- Change section order
- Add/remove sections
- Modify agent behavior
- Add validation steps

## Integration

### With Web Research

The workflow integrates with the web search system:
1. Performs web search for topic
2. Synthesizes context from results
3. Extracts references
4. Uses context and references in generation

### With Document Retrieval

Can optionally use RAG retrieval engine:
- Query local documents
- Combine with web research
- Use document chunks as context

## Output Format

### Markdown Structure

```markdown
# Research Overview: [Topic]

*Generated: [Timestamp]*

---

## Abstract

[Abstract content]

## Introduction

[Introduction content]

## Background / Foundations

[Background content]

...

## References

[Formatted references]
```

### Section Details

Each section includes:
- Proper heading hierarchy
- Academic formatting
- Citations and references
- Structured content
- Logical flow

## Performance

### Generation Time

- Full overview: 5-15 minutes (depending on LLM speed)
- Per section: 30 seconds - 2 minutes
- Web research: 10-30 seconds

### Resource Requirements

- LLM: Ollama with Llama 3.1 (local) or OpenAI (API)
- Memory: Sufficient for context and references
- Network: Required for web research

## Troubleshooting

### No References Found

1. Check web search is enabled
2. Verify internet connection
3. Check DuckDuckGo search availability
4. Review logs for search errors

### Section Generation Fails

1. Check LLM is running (Ollama)
2. Verify prompt templates are loaded
3. Review error messages in logs
4. Check context length limits

### Prompt Template Errors

1. Verify templates exist in `prompts/` directory
2. Check template formatting syntax
3. Ensure all placeholders are provided
4. Review template loading logs

## Best Practices

1. **Topic Selection**: Use specific, well-defined topics
2. **Web Research**: Enable for better references and context
3. **Review Output**: Always review and edit generated content
4. **Citation Check**: Verify references are accurate
5. **Customization**: Adjust prompts for your specific needs

## Next Steps

- See `documents/10_research_assistant_architecture.md` for general architecture
- See `documents/05_running_the_app.md` for running the app
- Edit `prompts/*.txt` to customize templates

