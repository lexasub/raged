"""
summarizer.py - LLM-based code summarization service for AST-RAG.

Provides:
- NodeSummary: Pydantic model for structured summary output
- SummarizerService: Service for generating summaries via local OpenAI-compatible LLM
- Prompt templates for code summarization

Usage::

    summarizer = SummarizerService(base_url="http://localhost:11434/v1")
    summary = summarizer.summarize_node(node_id="abc123", api=rag_api)
    print(summary.summary)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field

from ast_rag.ast_rag_api import ASTRagAPI
from ast_rag.models import ASTNode, NodeKind

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Summary data models
# ---------------------------------------------------------------------------

class ComplexityLevel(str, Enum):
    """Estimated complexity level of a function/class."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class NodeSummary(BaseModel):
    """Structured summary of a code node (function, method, class, etc.).

    This model captures the essential information about a code entity
    in a structured format suitable for RAG, documentation, or code analysis.

    Attributes:
        node_id: Unique identifier of the AST node
        summary: Natural language summary of what the code does
        inputs: List of input parameters/arguments with descriptions
        outputs: List of return values/outputs with descriptions
        side_effects: List of side effects (IO, mutations, exceptions, etc.)
        calls: List of functions/methods called by this node
        called_by: List of functions/methods that call this node
        complexity: Estimated complexity level (low/medium/high)
        tags: List of relevant tags (e.g., "async", "pure", "deprecated")
        generated_at: ISO timestamp when summary was generated
        model_used: Name of the LLM model used for generation
    """
    node_id: str
    summary: str = Field(..., description="Natural language summary of the code")
    inputs: list[dict[str, str]] = Field(
        default_factory=list,
        description="Input parameters: [{name, type, description}, ...]"
    )
    outputs: list[dict[str, str]] = Field(
        default_factory=list,
        description="Return values: [{name, type, description}, ...]"
    )
    side_effects: list[str] = Field(
        default_factory=list,
        description="Side effects: IO operations, state mutations, exceptions thrown"
    )
    calls: list[str] = Field(
        default_factory=list,
        description="Functions/methods called by this node (qualified names)"
    )
    called_by: list[str] = Field(
        default_factory=list,
        description="Functions/methods that call this node (qualified names)"
    )
    complexity: ComplexityLevel = Field(
        default=ComplexityLevel.MEDIUM,
        description="Estimated complexity level"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Relevant tags: async, pure, deprecated, thread-safe, etc."
    )
    generated_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        description="ISO timestamp of summary generation"
    )
    model_used: Optional[str] = Field(
        default=None,
        description="LLM model name used for generation"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dictionary."""
        return self.model_dump()

    def to_markdown(self) -> str:
        """Render summary as Markdown for display."""
        lines = [
            f"## Summary: `{self.node_id[:12]}...`",
            "",
            "### Description",
            self.summary,
            "",
        ]

        if self.inputs:
            lines.append("### Inputs")
            for inp in self.inputs:
                type_str = inp.get("type", "any")
                desc = inp.get("description", "")
                lines.append(f"- `{inp.get('name', 'unknown')}` ({type_str}): {desc}")
            lines.append("")

        if self.outputs:
            lines.append("### Outputs")
            for out in self.outputs:
                type_str = out.get("type", "any")
                desc = out.get("description", "")
                lines.append(f"- `{out.get('name', 'return')}` ({type_str}): {desc}")
            lines.append("")

        if self.side_effects:
            lines.append("### Side Effects")
            for effect in self.side_effects:
                lines.append(f"- {effect}")
            lines.append("")

        if self.calls:
            lines.append("### Calls")
            for call in self.calls[:10]:  # Limit to 10
                lines.append(f"- `{call}`")
            if len(self.calls) > 10:
                lines.append(f"- ... and {len(self.calls) - 10} more")
            lines.append("")

        if self.called_by:
            lines.append("### Called By")
            for caller in self.called_by[:10]:  # Limit to 10
                lines.append(f"- `{caller}`")
            if len(self.called_by) > 10:
                lines.append(f"- ... and {len(self.called_by) - 10} more")
            lines.append("")

        lines.append("### Metadata")
        lines.append(f"- **Complexity:** {self.complexity.value}")
        lines.append(f"- **Tags:** {', '.join(self.tags) if self.tags else 'none'}")
        lines.append(f"- **Model:** {self.model_used or 'unknown'}")
        lines.append(f"- **Generated:** {self.generated_at}")

        return "\n".join(lines)


class SummaryCacheEntry(BaseModel):
    """Cache entry for a generated summary."""
    node_id: str
    code_hash: str  # Hash of the code to detect changes
    summary: NodeSummary
    created_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SUMMARY_PROMPT_TEMPLATE = """You are an expert code analyst. Your task is to analyze the provided code and generate a structured summary.

## Code to Analyze

### Signature
```
{signature}
```

### Source Code
```{lang}
{code}
```

### Context

#### Functions/Methods Called by This Code
{calls_context}

#### Functions/Methods That Call This Code
{callers_context}

## Instructions

Analyze the code and provide a structured response in JSON format with the following fields:

1. **summary** (string): A concise 2-4 sentence description of what this code does. Focus on the purpose and behavior, not implementation details.

2. **inputs** (array of objects): List all input parameters. Each object should have:
   - "name": parameter name
   - "type": parameter type (if available, otherwise "any")
   - "description": brief description of what this parameter represents

3. **outputs** (array of objects): List all return values. Each object should have:
   - "name": "return" (or name for multiple outputs)
   - "type": return type
   - "description": what this return value represents

4. **side_effects** (array of strings): List any side effects such as:
   - File I/O operations
   - Network requests
   - Database operations
   - State mutations
   - Exception throwing
   - Logging
   - If no side effects, include "None (pure function)"

5. **calls** (array of strings): List qualified names of functions/methods called by this code (from the provided context).

6. **called_by** (array of strings): List qualified names of functions/methods that call this code (from the provided context).

7. **complexity** (string): Estimate complexity as one of: "low", "medium", "high"
   - low: Simple logic, few branches, < 20 lines
   - medium: Moderate logic, some branches, 20-100 lines
   - high: Complex logic, many branches, > 100 lines, or complex algorithms

8. **tags** (array of strings): Relevant tags such as:
   - "async" if asynchronous
   - "pure" if no side effects
   - "deprecated" if marked as deprecated
   - "thread-safe" if explicitly thread-safe
   - "recursive" if recursive
   - "getter"/"setter" for accessors
   - "constructor"/"initializer" for constructors
   - "factory" for factory functions
   - "validator" for validation logic
   - "transformer" for data transformation
   - "handler" for event/request handlers

## Response Format

Respond ONLY with a valid JSON object. Do not include any explanatory text outside the JSON.

Example response structure:
```json
{{
  "summary": "...",
  "inputs": [...],
  "outputs": [...],
  "side_effects": [...],
  "calls": [...],
  "called_by": [...],
  "complexity": "medium",
  "tags": [...]
}}
```
"""


# ---------------------------------------------------------------------------
# Summarizer Service
# ---------------------------------------------------------------------------

class SummarizerService:
    """Service for generating code summaries using local OpenAI-compatible LLM.

    This service integrates with local LLM servers like Ollama or vLLM
    to generate structured summaries of code entities.

    Features:
    - Generates structured JSON summaries with inputs, outputs, side effects
    - Includes caller/callee context for better understanding
    - Caches summaries to avoid regenerating for unchanged code
    - Supports multiple LLM backends via OpenAI-compatible API

    Usage::

        # Initialize with Ollama (default)
        summarizer = SummarizerService()

        # Or with custom endpoint
        summarizer = SummarizerService(
            base_url="http://localhost:8000/v1",
            model="my-code-model",
            api_key="optional-key"
        )

        # Generate summary
        summary = summarizer.summarize_node(
            node_id="abc123",
            api=rag_api,
            max_callers=5,
            max_callees=5
        )

        print(summary.summary)
        print(summary.to_markdown())

    Args:
        base_url: Base URL of OpenAI-compatible API (default: Ollama local)
        model: Model name to use for summarization
        api_key: Optional API key (not needed for local Ollama)
        timeout: Request timeout in seconds
        cache_enabled: Whether to cache generated summaries
        cache_path: Path to store summary cache (JSON file)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5-coder:14b",
        api_key: Optional[str] = None,
        timeout: int = 120,
        cache_enabled: bool = True,
        cache_path: str = ".ast_rag_summary_cache.json",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key or "ollama"  # Ollama accepts any key
        self._timeout = timeout
        self._cache_enabled = cache_enabled
        self._cache_path = Path(cache_path)
        self._cache: dict[str, SummaryCacheEntry] = {}
        self._client: Optional[httpx.AsyncClient] = None

        # Load cache if exists
        if cache_enabled and self._cache_path.exists():
            self._load_cache()

    def _load_cache(self) -> None:
        """Load summary cache from disk."""
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            self._cache = {
                k: SummaryCacheEntry.model_validate(v)
                for k, v in data.items()
            }
            logger.info("Loaded %d summary cache entries", len(self._cache))
        except Exception as exc:
            logger.warning("Failed to load summary cache: %s", exc)
            self._cache = {}

    def _save_cache(self) -> None:
        """Save summary cache to disk."""
        if not self._cache_enabled:
            return
        try:
            data = {
                k: v.model_dump()
                for k, v in self._cache.items()
            }
            self._cache_path.write_text(
                json.dumps(data, indent=2),
                encoding="utf-8"
            )
            logger.debug("Saved %d summary cache entries", len(self._cache))
        except Exception as exc:
            logger.warning("Failed to save summary cache: %s", exc)

    def _compute_code_hash(self, code: str) -> str:
        """Compute SHA-256 hash of code for cache validation."""
        return hashlib.sha256(code.encode("utf-8")).hexdigest()[:24]

    def _get_cached_summary(
        self,
        node_id: str,
        code_hash: str,
    ) -> Optional[NodeSummary]:
        """Get cached summary if available and code hasn't changed."""
        if not self._cache_enabled:
            return None

        entry = self._cache.get(node_id)
        if entry and entry.code_hash == code_hash:
            logger.debug("Cache hit for node %s", node_id[:12])
            return entry.summary

        return None

    def _cache_summary(
        self,
        node_id: str,
        code_hash: str,
        summary: NodeSummary,
    ) -> None:
        """Cache a generated summary."""
        if not self._cache_enabled:
            return

        self._cache[node_id] = SummaryCacheEntry(
            node_id=node_id,
            code_hash=code_hash,
            summary=summary,
        )
        self._save_cache()

    async def _call_llm_async(self, prompt: str) -> str:
        """Call LLM API asynchronously."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert code analyst specializing in understanding and summarizing code across multiple programming languages."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,  # Low temperature for consistent structured output
            "stream": False,
        }

        try:
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPError as exc:
            logger.error("LLM API request failed: %s", exc)
            raise RuntimeError(f"LLM API error: {exc}")
        except (KeyError, IndexError) as exc:
            logger.error("Invalid LLM API response: %s", exc)
            raise RuntimeError(f"Invalid LLM API response format: {exc}")

    def _build_context(
        self,
        node: ASTNode,
        api: ASTRagAPI,
        max_callers: int = 5,
        max_callees: int = 5,
    ) -> tuple[str, str]:
        """Build caller and callee context strings."""
        # Get callers
        callers = api.find_callers(node.id, max_depth=1)
        callers_context = "\n".join(
            f"- {c.qualified_name}" for c in callers[:max_callers]
        ) if callers else "- (none)"

        # Get callees
        callees = api.find_callees(node.id, max_depth=1)
        calls_context = "\n".join(
            f"- {c.qualified_name}" for c in callees[:max_callees]
        ) if callees else "- (none)"

        return calls_context, callers_context

    def _parse_llm_response(self, response_text: str) -> dict[str, Any]:
        """Parse LLM response to extract JSON."""
        # Try to extract JSON from response (may be wrapped in markdown)
        text = response_text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse LLM JSON: %s", exc)
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Could not parse JSON from LLM response: {response_text[:200]}")

    def summarize_node(
        self,
        node_id: str,
        api: ASTRagAPI,
        max_callers: int = 5,
        max_callees: int = 5,
        force_regenerate: bool = False,
    ) -> NodeSummary:
        """Generate a structured summary for a code node.

        This method retrieves the code node, gathers context (callers/callees),
        prompts the LLM for analysis, and returns a structured summary.

        Args:
            node_id: Unique identifier of the AST node to summarize
            api: ASTRagAPI instance for retrieving code and context
            max_callers: Maximum number of callers to include in context
            max_callees: Maximum number of callees to include in context
            force_regenerate: If True, ignore cache and regenerate

        Returns:
            NodeSummary with structured analysis of the code

        Raises:
            ValueError: If node not found or not summarizable
            RuntimeError: If LLM API call fails
        """
        # Get the node
        node = api.get_node(node_id)
        if node is None:
            raise ValueError(f"Node not found: {node_id}")

        # Check if node kind is summarizable
        summarizable_kinds = {
            NodeKind.FUNCTION, NodeKind.METHOD,
            NodeKind.CONSTRUCTOR, NodeKind.DESTRUCTOR,
            NodeKind.CLASS, NodeKind.INTERFACE,
            NodeKind.STRUCT, NodeKind.TRAIT,
        }
        if node.kind not in summarizable_kinds:
            raise ValueError(
                f"Node kind {node.kind.value} is not summarizable. "
                f"Supported kinds: {[k.value for k in summarizable_kinds]}"
            )

        # Get source code
        code = api.get_code_snippet(
            node.file_path,
            node.start_line,
            node.end_line,
        )
        if not code:
            raise ValueError(f"Could not retrieve source code for node {node_id}")

        code_hash = self._compute_code_hash(code)

        # Check cache
        if not force_regenerate:
            cached = self._get_cached_summary(node_id, code_hash)
            if cached:
                return cached

        # Build context
        calls_context, callers_context = self._build_context(
            node, api, max_callers, max_callees
        )

        # Build prompt
        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            signature=node.signature or "(no signature)",
            lang=node.lang.value,
            code=code,
            calls_context=calls_context,
            callers_context=callers_context,
        )

        logger.info(
            "Generating summary for %s (%s) - code lines: %d",
            node.qualified_name,
            node.kind.value,
            node.end_line - node.start_line + 1
        )

        # Call LLM (synchronous wrapper for async)
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        llm_response = loop.run_until_complete(self._call_llm_async(prompt))

        # Parse response
        try:
            parsed = self._parse_llm_response(llm_response)
        except ValueError as exc:
            logger.error("Failed to parse LLM response: %s", exc)
            # Fallback: create minimal summary
            parsed = {
                "summary": f"Code entity: {node.qualified_name}",
                "inputs": [],
                "outputs": [],
                "side_effects": ["Unknown"],
                "calls": [c.split("- ")[1] for c in calls_context.split("\n") if c.startswith("- ") and c != "- (none)"],
                "called_by": [c.split("- ")[1] for c in callers_context.split("\n") if c.startswith("- ") and c != "- (none)"],
                "complexity": "medium",
                "tags": [],
            }

        # Build NodeSummary
        summary = NodeSummary(
            node_id=node_id,
            summary=parsed.get("summary", "No summary generated"),
            inputs=parsed.get("inputs", []),
            outputs=parsed.get("outputs", []),
            side_effects=parsed.get("side_effects", []),
            calls=parsed.get("calls", []),
            called_by=parsed.get("called_by", []),
            complexity=ComplexityLevel(
                parsed.get("complexity", "medium")
            ),
            tags=parsed.get("tags", []),
            model_used=self._model,
        )

        # Cache the summary
        self._cache_summary(node_id, code_hash, summary)

        return summary

    def summarize_nodes(
        self,
        node_ids: list[str],
        api: ASTRagAPI,
        max_callers: int = 3,
        max_callees: int = 3,
        force_regenerate: bool = False,
    ) -> list[NodeSummary]:
        """Generate summaries for multiple nodes.

        Args:
            node_ids: List of node IDs to summarize
            api: ASTRagAPI instance
            max_callers: Maximum callers per node
            max_callees: Maximum callees per node
            force_regenerate: Force regeneration ignoring cache

        Returns:
            List of NodeSummary objects (may include errors as None)
        """
        results = []
        for node_id in node_ids:
            try:
                summary = self.summarize_node(
                    node_id, api, max_callers, max_callees, force_regenerate
                )
                results.append(summary)
            except Exception as exc:
                logger.warning("Failed to summarize node %s: %s", node_id[:12], exc)
                results.append(None)
        return results

    def clear_cache(self) -> None:
        """Clear the summary cache."""
        self._cache = {}
        if self._cache_path.exists():
            self._cache_path.unlink()
        logger.info("Summary cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "entries": len(self._cache),
            "cache_path": str(self._cache_path),
            "cache_enabled": self._cache_enabled,
        }
