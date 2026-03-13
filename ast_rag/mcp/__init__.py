"""
MCP server module for AST-RAG.

This module provides the MCP (Model Context Protocol) server implementation
for AST-RAG code analysis tools.
"""

from ast_rag.mcp.server import mcp, main, VERSION, SCHEMA_VERSION

__all__ = ["mcp", "main", "VERSION", "SCHEMA_VERSION"]
