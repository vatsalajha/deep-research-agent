# Deep Research Agent

An intelligent research agent built with [LangGraph](https://github.com/langchain-ai/langgraph) that performs iterative web searches and synthesizes findings into comprehensive, cited reports.

## Features

- **Multi-provider LLM support** — Groq, OpenAI, Anthropic, Google (Gemini), with easy extensibility for more
- Intelligent query analysis — breaks complex questions into targeted search queries
- Iterative web search with automatic gap detection via Tavily API
- LLM-powered synthesis that decides when research is sufficient
- Comprehensive report generation with inline citations and source lists
- Fully configurable: LLM provider, model, temperature, system prompt, search results per query, max iterations
- Multiple report styles: detailed, summary, academic

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
- **At least one LLM API key** from a supported provider (see below)
- [Tavily API key](https://tavily.com/) (free tier: 1,000 searches/month)

### Supported LLM Providers

| Provider | Sign-up | Default Model |
|----------|---------|---------------|
| Groq | [console.groq.com](https://console.groq.com/keys) | llama-3.3-70b-versatile |
| OpenAI | [platform.openai.com](https://platform.openai.com/api-keys) | gpt-4o |
| Anthropic | [console.anthropic.com](https://console.anthropic.com/settings/keys) | claude-sonnet-4-20250514 |
| Google (Gemini) | [aistudio.google.com](https://aistudio.google.com/apikey) | gemini-2.0-flash |

## Installation

```bash
git clone https://github.com/vatsalajha/deep-research-agent.git
cd deep-research-agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## API Keys Setup

Create your `.env` file from the template:

```bash
cp .env.template .env
```

Edit `.env` and add at least one LLM key and the Tavily search key:

```
# At least one LLM provider key
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...

# Search provider key (required)
TAVILY_API_KEY=tvly-...
```

You only need **one** LLM provider key — the agent will auto-detect which providers are available.

## Usage

### Run the example script

```bash
python examples/run_agent.py
```

This auto-detects the first available provider, runs three built-in example queries, and saves reports to `reports/`.

### Custom query via CLI

```bash
python examples/run_agent.py "What is the current state of nuclear fusion research?"
```

### Use in your own code

```python
from src.agent import DeepResearchAgent
from src.utils import load_environment

env = load_environment()

# Explicit provider selection
agent = DeepResearchAgent(
    llm_provider="openai",
    llm_api_key=env["openai_api_key"],
    llm_model="gpt-4o",
    search_api_key=env["tavily_api_key"],
    max_iterations=3,
    max_results=5,
    temperature=0.7,
    system_prompt="You are an expert technology analyst.",
)

result = agent.run("What are the latest AI trends?")
report = agent.get_report(result)
print(report)

agent.save_report(report, "my_report.md")
```

### Backward-compatible API

The old constructor still works:

```python
agent = DeepResearchAgent(
    groq_api_key=env["groq_api_key"],
    tavily_key=env["tavily_api_key"],
    model="llama-3.3-70b-versatile",
)
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
│   ├── agent.py        # DeepResearchAgent class + multi-provider LLM factory
│   ├── state.py        # AgentState TypedDict definition
│   ├── nodes.py        # 4 node functions (analyze, search, synthesize, report)
│   ├── tools.py        # WebSearchTool wrapper around Tavily (+ commented alternatives)
│   ├── templates.py    # Report templates (detailed, summary, academic)
│   └── utils.py        # Multi-key environment loader + helpers
├── examples/
│   └── run_agent.py    # Example usage script (auto-detects provider)
├── test_agent.py       # Validation tests
├── streamlit_app.py    # Streamlit web UI with provider selection
├── requirements.txt
├── .env.template
└── README.md
```

## Configuration

### LLM Provider

```python
from src.agent import DeepResearchAgent, PROVIDER_MODELS

# See all supported providers and their models
print(PROVIDER_MODELS)
# {'groq': ['llama-3.3-70b-versatile', ...], 'openai': ['gpt-4o', ...], ...}

agent = DeepResearchAgent(
    llm_provider="anthropic",
    llm_api_key="sk-ant-...",
    llm_model="claude-sonnet-4-20250514",
    search_api_key="tvly-...",
)
```

### Temperature

```python
agent = DeepResearchAgent(
    llm_provider="groq",
    llm_api_key=key,
    search_api_key=tavily_key,
    temperature=0.3,  # lower = more focused, higher = more creative
)
```

### System Prompt

```python
agent = DeepResearchAgent(
    llm_provider="openai",
    llm_api_key=key,
    search_api_key=tavily_key,
    system_prompt="You are a senior financial analyst. Focus on quantitative data and market trends.",
)
```

### Search Results Per Query

```python
agent = DeepResearchAgent(
    llm_provider="groq",
    llm_api_key=key,
    search_api_key=tavily_key,
    max_results=5,  # default: 3
)
```

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
    llm_provider="groq",
    llm_api_key=key,
    search_api_key=tavily_key,
    max_iterations=5,   # more cycles = deeper research (default: 3)
)
```

## How It Works

1. **User submits a query** — e.g. "How is AI being used to combat climate change?"
2. **Query analysis** — The LLM breaks this into specific searches like "AI climate change mitigation 2024", "machine learning carbon emissions reduction", etc.
3. **Web search** — Tavily searches the web and returns titles, URLs, and content snippets.
4. **Synthesis** — The LLM reviews all results, checks for coverage gaps, and either requests more searches or marks research as complete.
5. **Iteration** — Steps 3-4 repeat until coverage is sufficient or `max_iterations` is reached.
6. **Report generation** — The LLM synthesizes everything into a structured report with inline citations.

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
| `No LLM provider key found` | Set at least one of GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY in `.env` |
| `No search provider key found` | Set TAVILY_API_KEY in `.env` |
| `Unknown LLM provider` | Check spelling; supported: groq, openai, anthropic, google |
| `No API key for provider` | Set the corresponding env var or pass `llm_api_key=` explicitly |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` with venv activated |
| Empty search results | Check internet connection and Tavily API quota |
| Poor report quality | Increase `max_iterations` or `max_results` for more sources |
| Token limit errors | Reduce `max_results` or `max_iterations` for shorter prompts |

## Performance Notes

- Each search-synthesize cycle takes ~5-10 seconds (API calls + LLM reasoning)
- A full 2-iteration research run typically completes in 20-40 seconds
- Tavily free tier allows 1,000 searches/month
- Groq free tier has rate limits — reduce `max_iterations` if hitting TPM limits

## License

MIT
