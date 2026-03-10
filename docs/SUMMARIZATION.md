# Code Summarization with AST-RAG

This document describes the LLM-based code summarization feature for AST-RAG.

## Overview

The summarization service uses a local OpenAI-compatible LLM (Ollama, vLLM, etc.) to generate structured summaries of code entities (functions, methods, classes).

## Features

- **Structured JSON output**: Summary, inputs, outputs, side effects, call graph
- **Context-aware**: Includes callers and callees for better understanding
- **Caching**: Avoids regenerating summaries for unchanged code
- **Multiple output formats**: Markdown, JSON, plain text
- **Complexity estimation**: Automatic complexity assessment (low/medium/high)
- **Tagging**: Auto-generated tags (async, pure, deprecated, etc.)

## Quick Start

### 1. Set Up Ollama (Recommended for Local Use)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a code-focused model (recommended: 14B or larger)
ollama pull qwen2.5-coder:14b

# Ollama automatically starts on http://localhost:11434
# Verify it's running:
curl http://localhost:11434/api/tags
```

### 2. Summarize a Function

```bash
# Basic usage
ast-rag summarize com.example.MyService.processRequest

# With JSON output
ast-rag summarize MyFunction --output json

# Force regeneration (ignore cache)
ast-rag summarize MyFunction --force

# Use more context
ast-rag summarize MyFunction --max-callers 10 --max-callees 10
```

### 3. Programmatic Usage

```python
from ast_rag.ast_rag_api import ASTRagAPI
from ast_rag.graph_schema import create_driver
from ast_rag.embeddings import EmbeddingManager
from ast_rag.summarizer import SummarizerService, NodeSummary
from ast_rag.models import ProjectConfig

# Load configuration
cfg = ProjectConfig.model_validate_json("ast_rag_config.json")

# Initialize API
driver = create_driver(cfg.neo4j)
embed = EmbeddingManager(cfg.qdrant, cfg.embedding, neo4j_driver=driver)
api = ASTRagAPI(driver, embed)

# Initialize summarizer
summarizer = SummarizerService(
    base_url="http://localhost:11434/v1",  # Ollama
    model="qwen2.5-coder:14b",
)

# Find a node
nodes = api.find_definition("MyFunction", kind="Function")
if nodes:
    node = nodes[0]
    
    # Generate summary
    summary = summarizer.summarize_node(
        node_id=node.id,
        api=api,
        max_callers=5,
        max_callees=5,
    )
    
    # Access summary fields
    print(summary.summary)
    print(summary.inputs)
    print(summary.outputs)
    print(summary.side_effects)
    print(f"Complexity: {summary.complexity.value}")
    print(f"Tags: {summary.tags}")
    
    # Render as Markdown
    print(summary.to_markdown())
```

## Output Format

### JSON Structure

```json
{
  "node_id": "abc123def456",
  "summary": "Processes HTTP requests by validating input, calling external services, and returning formatted responses.",
  "inputs": [
    {
      "name": "request",
      "type": "HttpRequest",
      "description": "The incoming HTTP request object"
    },
    {
      "name": "context",
      "type": "Context",
      "description": "Request context with user session and metadata"
    }
  ],
  "outputs": [
    {
      "name": "return",
      "type": "HttpResponse",
      "description": "Formatted HTTP response with status code and body"
    }
  ],
  "side_effects": [
    "Writes to database via UserRepository",
    "Logs request metadata",
    "May throw ValidationException on invalid input"
  ],
  "calls": [
    "com.example.validator.RequestValidator.validate",
    "com.example.repository.UserRepository.save",
    "com.example.logger.InfoLogger.log"
  ],
  "called_by": [
    "com.example.controller.UserController.handleRequest",
    "com.example.service.BatchProcessor.processBatch"
  ],
  "complexity": "medium",
  "tags": ["async", "io"],
  "generated_at": "2026-03-11T10:30:00Z",
  "model_used": "qwen2.5-coder:14b"
}
```

### Markdown Output

```markdown
## Summary: `abc123def456...`

### Description
Processes HTTP requests by validating input, calling external services, and returning formatted responses.

### Inputs
- `request` (HttpRequest): The incoming HTTP request object
- `context` (Context): Request context with user session and metadata

### Outputs
- `return` (HttpResponse): Formatted HTTP response with status code and body

### Side Effects
- Writes to database via UserRepository
- Logs request metadata
- May throw ValidationException on invalid input

### Calls
- `com.example.validator.RequestValidator.validate`
- `com.example.repository.UserRepository.save`
- `com.example.logger.InfoLogger.log`

### Called By
- `com.example.controller.UserController.handleRequest`
- `com.example.service.BatchProcessor.processBatch`

### Metadata
- **Complexity:** medium
- **Tags:** async, io
- **Model:** qwen2.5-coder:14b
- **Generated:** 2026-03-11T10:30:00Z
```

## Configuration

### LLM Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_url` | `http://localhost:11434/v1` | OpenAI-compatible API endpoint |
| `model` | `qwen2.5-coder:14b` | Model name for summarization |
| `api_key` | `ollama` | API key (not needed for Ollama) |
| `timeout` | `120` | Request timeout in seconds |

### Supported LLM Backends

#### Ollama (Recommended)

```bash
# Install and run
ollama pull qwen2.5-coder:14b
ollama serve

# Use in CLI
ast-rag summarize MyFunction --llm-url http://localhost:11434/v1 --llm-model qwen2.5-coder:14b
```

#### vLLM

```bash
# Run vLLM with OpenAI-compatible API
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-14B-Instruct \
    --port 8000

# Use in CLI
ast-rag summarize MyFunction --llm-url http://localhost:8000/v1 --llm-model Qwen/Qwen2.5-Coder-14B-Instruct
```

#### Other OpenAI-Compatible APIs

Any OpenAI-compatible API works:

```bash
ast-rag summarize MyFunction \
    --llm-url http://your-server:8000/v1 \
    --llm-model your-model \
    --llm-api-key your-api-key
```

### Cache Configuration

Summaries are cached by default to avoid regenerating for unchanged code.

```python
summarizer = SummarizerService(
    cache_enabled=True,
    cache_path=".ast_rag_summary_cache.json",
)
```

Cache is invalidated when:
- Source code changes (detected via hash)
- `force_regenerate=True` is passed
- Cache is manually cleared

```python
# Clear cache
summarizer.clear_cache()

# Get cache stats
stats = summarizer.get_cache_stats()
print(stats)  # {"entries": 42, "cache_path": "...", "cache_enabled": true}
```

## Prompt Template

The service uses a structured prompt template that includes:

1. **Code signature**: Function/method signature
2. **Source code**: Full source code with syntax highlighting
3. **Context**: Callers and callees from the call graph
4. **Instructions**: Structured JSON output requirements

See `ast_rag/summarizer.py` for the full `SUMMARY_PROMPT_TEMPLATE`.

### Customizing the Prompt

You can modify the prompt template by editing `SUMMARY_PROMPT_TEMPLATE` in `summarizer.py`:

```python
from ast_rag.summarizer import SUMMARY_PROMPT_TEMPLATE

# View current template
print(SUMMARY_PROMPT_TEMPLATE)

# Modify as needed for your use case
```

## Model Recommendations

For best results, use code-focused models with at least 14B parameters:

| Model | Size | Quality | Speed | Recommended |
|-------|------|---------|-------|-------------|
| Qwen2.5-Coder | 14B | Excellent | Fast | ✅ Yes |
| Qwen2.5-Coder | 32B | Excellent | Medium | ✅ Yes |
| CodeLlama | 34B | Very Good | Medium | ✅ Yes |
| DeepSeek-Coder | 33B | Very Good | Medium | ✅ Yes |
| StarCoder2 | 15B | Good | Fast | ⚠️ OK |
| Phi-2 | 2.7B | Fair | Very Fast | ❌ Not recommended |

## Troubleshooting

### "LLM API request failed"

- Ensure Ollama/vLLM is running
- Check the `--llm-url` parameter
- Verify network connectivity

```bash
# Test Ollama
curl http://localhost:11434/api/tags

# Test vLLM
curl http://localhost:8000/v1/models
```

### "Node not found"

- Verify the qualified name is correct
- Try with `--lang` or `--kind` filters
- Use `ast-rag goto <name>` to find the correct name

### "Could not retrieve source code"

- Check that the source file exists at the indexed path
- Verify file permissions
- Re-index if files have moved

### Slow summary generation

- Use a smaller model (7B-14B)
- Reduce context size (`--max-callers`, `--max-callees`)
- Enable caching to avoid regeneration

## API Reference

### SummarizerService

```python
class SummarizerService:
    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5-coder:14b",
        api_key: Optional[str] = None,
        timeout: int = 120,
        cache_enabled: bool = True,
        cache_path: str = ".ast_rag_summary_cache.json",
    )
    
    def summarize_node(
        self,
        node_id: str,
        api: ASTRagAPI,
        max_callers: int = 5,
        max_callees: int = 5,
        force_regenerate: bool = False,
    ) -> NodeSummary
    
    def summarize_nodes(
        self,
        node_ids: list[str],
        api: ASTRagAPI,
        max_callers: int = 3,
        max_callees: int = 3,
        force_regenerate: bool = False,
    ) -> list[NodeSummary]
    
    def clear_cache(self) -> None
    
    def get_cache_stats(self) -> dict[str, Any]
```

### NodeSummary

```python
class NodeSummary(BaseModel):
    node_id: str
    summary: str
    inputs: list[dict[str, str]]
    outputs: list[dict[str, str]]
    side_effects: list[str]
    calls: list[str]
    called_by: list[str]
    complexity: ComplexityLevel  # "low" | "medium" | "high"
    tags: list[str]
    generated_at: str
    model_used: Optional[str]
    
    def to_dict(self) -> dict[str, Any]
    def to_markdown(self) -> str
```

## Integration Examples

### Batch Summarization

```python
# Summarize all functions in a class
class_nodes = api.find_definition("MyService", kind="Class")
if class_nodes:
    class_id = class_nodes[0].id
    
    # Get all methods (via call graph or other means)
    methods = api.find_callees(class_id, max_depth=1)
    method_ids = [m.id for m in methods if m.kind in ("Method", "Function")]
    
    # Generate summaries in batch
    summaries = summarizer.summarize_nodes(method_ids, api)
    
    # Process results
    for summary in summaries:
        if summary:
            print(f"{summary.node_id}: {summary.summary[:100]}...")
```

### Documentation Generation

```python
# Generate documentation for a module
nodes = api.find_definition("my_module", kind="Module")
if nodes:
    summary = summarizer.summarize_node(nodes[0].id, api)
    
    # Generate Markdown documentation
    doc = f"""
# Module Documentation

## {nodes[0].qualified_name}

{summary.summary}

### API Reference

#### Inputs
{chr(10).join(f"- `{inp['name']}`: {inp['description']}" for inp in summary.inputs)}

#### Outputs
{chr(10).join(f"- `{out['name']}`: {out['description']}" for out in summary.outputs)}

#### Side Effects
{chr(10).join(f"- {effect}" for effect in summary.side_effects)}
"""
    print(doc)
```

### Code Review Assistant

```python
# Analyze complexity and side effects for code review
def analyze_for_review(node_id: str) -> dict:
    summary = summarizer.summarize_node(node_id, api)
    
    review_notes = []
    
    if summary.complexity == "high":
        review_notes.append("⚠️ High complexity - consider refactoring")
    
    if len(summary.side_effects) > 3:
        review_notes.append("⚠️ Many side effects - ensure proper testing")
    
    if "async" in summary.tags and not any("timeout" in s for s in summary.side_effects):
        review_notes.append("💡 Async function - consider adding timeout handling")
    
    return {
        "summary": summary.summary,
        "complexity": summary.complexity.value,
        "side_effects_count": len(summary.side_effects),
        "review_notes": review_notes,
    }
```

## License

This feature is part of AST-RAG and follows the same LGPL-3.0 license.
