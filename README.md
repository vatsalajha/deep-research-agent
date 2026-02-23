# Deep Research Agent

An intelligent research agent built with [LangGraph](https://github.com/langchain-ai/langgraph) that performs iterative web searches and synthesizes findings into comprehensive, cited reports.

## Features

- Intelligent query analysis — breaks complex questions into targeted search queries
- Iterative web search with automatic gap detection via Tavily API
- LLM-powered synthesis that decides when research is sufficient
- Comprehensive report generation with inline citations and source lists
- Configurable research depth (number of iterations, results per query)

## Architecture

The agent uses a LangGraph state graph with four nodes connected in a loop:

```
analyze_query → search → synthesize ─┬→ generate_report → END
                  ^                   │
                  └───────────────────┘  (needs more research)
```

1. **Query Analyzer** — Breaks the user question into 3-5 specific search queries
2. **Web Searcher** — Executes searches via Tavily, skips duplicates
3. **Synthesizer** — Evaluates coverage, identifies gaps, decides whether to loop back or proceed
4. **Report Generator** — Produces a structured, cited report from all gathered research

## Prerequisites

- Python 3.10+
- [Anthropic API key](https://console.anthropic.com/)
- [Tavily API key](https://tavily.com/) (free tier: 1,000 searches/month)

## Installation

```bash
git clone https://github.com/vatsalajha/deep-research-agent.git
cd deep-research-agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## API Keys Setup

1. **Anthropic** — Sign up at [console.anthropic.com](https://console.anthropic.com/) and create an API key.
2. **Tavily** — Sign up at [tavily.com](https://tavily.com/) and grab your API key (free tier available).

Then create your `.env` file:

```bash
cp .env.template .env
```

Edit `.env` and paste your keys:

```
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
```

## Usage

### Run the example script

```bash
python examples/run_agent.py
```

This runs three built-in example queries. Reports are saved to `reports/`.

### Custom query via CLI

```bash
python examples/run_agent.py "What is the current state of nuclear fusion research?"
```

### Use in your own code

```python
from src.agent import DeepResearchAgent
from src.utils import load_environment

env = load_environment()

agent = DeepResearchAgent(
    anthropic_key=env["anthropic_api_key"],
    tavily_key=env["tavily_api_key"],
    max_iterations=3,
)

result = agent.run("What are the latest AI trends?")
report = agent.get_report(result)
print(report)

agent.save_report(report, "my_report.md")
```

### Choose a report style

Three built-in templates are available: `detailed` (default), `summary`, and `academic`.

```python
# Quick executive briefing (under 500 words)
result = agent.run("Impact of AI on healthcare", report_style="summary")

# Formal academic-style report
result = agent.run("Impact of AI on healthcare", report_style="academic")
```

## Project Structure

```
deep-research-agent/
├── src/
│   ├── agent.py        # DeepResearchAgent class + LangGraph wiring
│   ├── state.py        # AgentState TypedDict definition
│   ├── nodes.py        # 4 node functions (analyze, search, synthesize, report)
│   ├── tools.py        # WebSearchTool wrapper around Tavily
│   └── utils.py        # Environment loader + helpers
├── examples/
│   └── run_agent.py    # Example usage script
├── test_agent.py       # Validation tests
├── requirements.txt
├── .env.template
└── README.md
```

## Configuration

### Report style

```python
# "detailed" (default) — full report with executive summary, findings, analysis
# "summary"            — concise briefing under 500 words
# "academic"           — formal paper with abstract, lit review, discussion

result = agent.run("my query", report_style="academic")
```

### Research depth

```python
agent = DeepResearchAgent(
    anthropic_key=key,
    tavily_key=key,
    max_iterations=5,   # more cycles = deeper research (default: 3)
)
```

### Search results per query

In `src/nodes.py`, the web searcher calls `batch_search(queries, max_results=5)`. Adjust `max_results` for more or fewer sources per query.

### LLM model

Each node in `src/nodes.py` uses `claude-3-5-sonnet-20241022`. Change the `model` parameter to use a different Claude model.

## How It Works

1. **User submits a query** — e.g. "How is AI being used to combat climate change?"
2. **Query analysis** — Claude breaks this into specific searches like "AI climate change mitigation 2024", "machine learning carbon emissions reduction", etc.
3. **Web search** — Tavily searches the web and returns titles, URLs, and content snippets.
4. **Synthesis** — Claude reviews all results, checks for coverage gaps, and either requests more searches or marks research as complete.
5. **Iteration** — Steps 3-4 repeat until coverage is sufficient or `max_iterations` is reached.
6. **Report generation** — Claude synthesizes everything into a structured report with inline citations.

## Example Report Structure

```
# Research Report: [Topic]

## Executive Summary
[2-3 sentence overview]

## Key Findings
1. Finding one [1][3]
2. Finding two [2][5]
3. Finding three [4]

## Detailed Analysis
[Expanded sections with evidence and citations]

## Trends & Implications
[Forward-looking analysis]

## Sources
[1] Article Title — https://example.com/...
[2] Article Title — https://example.com/...
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ANTHROPIC_API_KEY not found` | Check `.env` exists in project root with the key set |
| `TAVILY_API_KEY not found` | Get a free key at [tavily.com](https://tavily.com/) |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` with venv activated |
| Empty search results | Check internet connection and Tavily API quota |
| Poor report quality | Increase `max_iterations` or `max_results` for more sources |

## Performance Notes

- Each search-synthesize cycle takes ~5-10 seconds (API calls + LLM reasoning)
- A full 3-iteration research run typically completes in 30-60 seconds
- Tavily free tier allows 1,000 searches/month

## Future Improvements

- Streaming report generation for real-time output
- Search result caching to avoid duplicate API calls
- Support for additional search providers (Exa, SerpAPI)
- Custom report templates (academic, executive summary, technical brief)
- CLI interface with configurable options
- Web UI for interactive research sessions

## License

MIT
