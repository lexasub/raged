"""Compatibility shim for the legacy MCP module path."""

from ast_rag._optional import missing_dependency_entry_point, module_available

if module_available("mcp"):
    from ast_rag.mcp.server import *  # noqa: F403
    from ast_rag.mcp.server import main  # noqa: F401
else:  # the `mcp` extra is not installed; exercised in a subprocess by the tests
    main = missing_dependency_entry_point("mcp", feature="The AST-RAG MCP server", extra="mcp")
