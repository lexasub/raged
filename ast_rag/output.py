"""
output.py - Output formatters for CLI.

Abstraction layer for formatting CLI output. Supports:
- JSON (AI-friendly, default)
- Human-readable (Rich tables, with --humanize flag)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Optional

from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax

from ast_rag.models import ASTNode, SearchResult


console = Console()


class OutputFormatter(ABC):
    """Abstract base class for output formatters."""

    @abstractmethod
    def format_search_results(self, results: list[SearchResult], query: str) -> None:
        """Format semantic search results."""
        pass

    @abstractmethod
    def format_definitions(
        self, nodes: list[ASTNode], api: Optional[Any] = None, snippet: bool = False
    ) -> None:
        """Format definition lookup results."""
        pass

    @abstractmethod
    def format_callers(self, qualified_name: str, nodes: list[ASTNode]) -> None:
        """Format callers results."""
        pass


class JSONFormatter(OutputFormatter):
    """AI-friendly JSON output (default)."""

    def format_search_results(self, results: list[SearchResult], query: str) -> None:
        output = {
            "query": query,
            "count": len(results),
            "results": [
                {
                    "score": round(r.score, 3),
                    "kind": r.node.kind.value,
                    "lang": r.node.lang.value,
                    "qualified_name": r.node.qualified_name,
                    "file": r.node.file_path,
                    "line": r.node.start_line,
                }
                for r in results
            ],
        }
        print(json.dumps(output, indent=2))

    def format_definitions(
        self, nodes: list[ASTNode], api: Optional[Any] = None, snippet: bool = False
    ) -> None:
        output = {
            "count": len(nodes),
            "definitions": [
                {
                    "kind": n.kind.value,
                    "qualified_name": n.qualified_name,
                    "file": n.file_path,
                    "start_line": n.start_line,
                    "end_line": n.end_line,
                    "lang": n.lang.value,
                    "signature": n.signature,
                }
                for n in nodes
            ],
        }
        print(json.dumps(output, indent=2))

    def format_callers(self, qualified_name: str, nodes: list[ASTNode]) -> None:
        output = {
            "target": qualified_name,
            "count": len(nodes),
            "callers": [
                {
                    "kind": n.kind.value,
                    "lang": n.lang.value,
                    "qualified_name": n.qualified_name,
                    "file": n.file_path,
                    "line": n.start_line,
                }
                for n in nodes
            ],
        }
        print(json.dumps(output, indent=2))


class HumanFormatter(OutputFormatter):
    """Human-readable output using Rich (table format)."""

    # Default color mapping (structured for future configurability)
    COLOR_MAP = {
        "class": "blue",
        "function": "green",
        "method": "cyan",
        "field": "yellow",
    }

    def _colorize_kind(self, kind: str) -> str:
        """Apply color styling to node kind based on COLOR_MAP."""
        color = self.COLOR_MAP.get(kind.lower())
        if not color:
            return kind
        return f"[{color}]{kind}[/{color}]"

    def format_search_results(self, results: list[SearchResult], query: str) -> None:
        table = Table(
            "Score",
            "Kind",
            "Lang",
            "Qualified Name",
            "File",
            "Line",
            title=f"Search: [italic]{query}[/italic]",
        )
        for r in results:
            n = r.node
            table.add_row(
                f"{r.score:.3f}",
                self._colorize_kind(n.kind.value),
                n.lang.value,
                n.qualified_name,
                n.file_path,
                str(n.start_line),
            )
        console.print(table)

    def format_definitions(
        self, nodes: list[ASTNode], api: Optional[Any] = None, snippet: bool = False
    ) -> None:
        for node in nodes:
            console.print(
                f"{self._colorize_kind(node.kind.value)} "
                f"[bold]{node.qualified_name}[/bold]\n"
                f"  File:      {node.file_path}\n"
                f"  Lines:     {node.start_line}–{node.end_line}\n"
                f"  Language:  {node.lang.value}\n"
                + (f"  Signature: {node.signature}\n" if node.signature else "")
            )

            if snippet and api:
                code = api.get_code_snippet(
                    node.file_path, node.start_line, node.end_line
                )
                if code:
                    lang_map = {
                        "java": "java",
                        "cpp": "cpp",
                        "rust": "rust",
                        "python": "python",
                        "typescript": "typescript",
                    }
                    console.print(
                        Syntax(
                            code,
                            lang_map.get(node.lang.value, "text"),
                            line_numbers=True,
                            start_line=node.start_line,
                        )
                    )

    def format_callers(self, qualified_name: str, nodes: list[ASTNode]) -> None:
        table = Table(
            "Kind",
            "Lang",
            "Qualified Name",
            "File",
            "Line",
            title=f"Callers of {qualified_name}",
        )
        for n in nodes:
            table.add_row(
                self._colorize_kind(n.kind.value),
                n.lang.value,
                n.qualified_name,
                n.file_path,
                str(n.start_line),
            )
        console.print(table)


def get_formatter(humanize: bool = False) -> OutputFormatter:
    """Get the appropriate formatter based on flags."""
    if humanize:
        return HumanFormatter()
    return JSONFormatter()
