"""Report templates for different output formats."""

TEMPLATES: dict[str, str] = {
    "detailed": (
        "Write a well-structured, comprehensive report with:\n"
        "1. Executive Summary (2-3 sentences)\n"
        "2. Key Findings (3-5 main points, each cited)\n"
        "3. Detailed Analysis (expand on key findings with evidence)\n"
        "4. Trends & Implications (what this means going forward)\n"
        "5. Sources (list all sources used with [citation number])\n"
        "\n"
        "Format:\n"
        "- Use clear sections with markdown headers\n"
        "- Cite sources as [1], [2], etc. when referencing information\n"
        "- Be objective and comprehensive\n"
        "- Include specific data points and examples"
    ),
    "summary": (
        "Write a concise executive briefing with:\n"
        "1. One-paragraph summary (4-5 sentences covering the essentials)\n"
        "2. Top 3 takeaways (one sentence each, cited)\n"
        "3. Sources\n"
        "\n"
        "Format:\n"
        "- Keep the total length under 500 words\n"
        "- Cite sources as [1], [2], etc.\n"
        "- Focus on actionable insights, skip background detail"
    ),
    "academic": (
        "Write a report in academic style with:\n"
        "1. Abstract (single paragraph summarizing the research)\n"
        "2. Introduction (context and research question)\n"
        "3. Literature Review (synthesize findings from sources)\n"
        "4. Discussion (analysis, compare perspectives, identify consensus and debate)\n"
        "5. Conclusion (key takeaways and open questions)\n"
        "6. References (all sources in numbered format)\n"
        "\n"
        "Format:\n"
        "- Use formal, objective academic tone\n"
        "- Cite sources as [1], [2], etc. inline\n"
        "- Include specific data and evidence\n"
        "- Acknowledge limitations and conflicting findings"
    ),
}

VALID_STYLES = list(TEMPLATES.keys())
