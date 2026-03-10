# Code Summarization Feature

## Overview

LLM-based code summarization service for AST-RAG that generates structured summaries of functions, methods, and classes.

## Files

- `ast_rag/summarizer.py` - Main service implementation
- `ast_rag/cli.py` - CLI command (`ast-rag summarize`)
- `ast_rag/ast_rag_mcp.py` - MCP tool (`summarize_code`)
- `docs/SUMMARIZATION.md` - Full documentation
- `tests/test_summarizer.py` - Unit tests
- `demo_summarizer.py` - Demo script
- `summarizer_config_example.json` - Example configuration

## Quick Start

### 1. Install Ollama

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5-coder:14b
```

### 2. Summarize Code

```bash
# CLI usage
ast-rag summarize com.example.MyFunction

# JSON output
ast-rag summarize MyFunction --output json

# Force regeneration
ast-rag summarize MyFunction --force
```

### 3. Programmatic Usage

```python
from ast_rag.summarizer import SummarizerService

summarizer = SummarizerService(
    base_url="http://localhost:11434/v1",
    model="qwen2.5-coder:14b",
)

summary = summarizer.summarize_node(node_id="abc123", api=rag_api)
print(summary.summary)
print(summary.to_markdown())
```

## Output Format

```json
{
  "node_id": "abc123",
  "summary": "Processes HTTP requests...",
  "inputs": [{"name": "request", "type": "HttpRequest", "description": "..."}],
  "outputs": [{"name": "return", "type": "HttpResponse", "description": "..."}],
  "side_effects": ["Database write", "Logging"],
  "calls": ["com.example.repo.save"],
  "called_by": ["com.example.controller.handle"],
  "complexity": "medium",
  "tags": ["async", "io"]
}
```

## Features

- ✅ Structured JSON output
- ✅ Caller/callee context
- ✅ Complexity estimation
- ✅ Auto-tagging
- ✅ Summary caching
- ✅ Multiple output formats (Markdown, JSON, text)
- ✅ CLI and MCP integration
- ✅ Batch summarization

## Configuration

See `summarizer_config_example.json` for configuration options.

Key settings:
- `llm.base_url`: OpenAI-compatible API endpoint
- `llm.model`: Model name for summarization
- `cache.enabled`: Enable/disable caching
- `context.max_callers/callees`: Context size

## Testing

```bash
pytest tests/test_summarizer.py -v
```

## Documentation

See `docs/SUMMARIZATION.md` for full documentation.
