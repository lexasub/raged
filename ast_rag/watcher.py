"""Compatibility shim for the legacy watcher module path."""

from ast_rag._optional import missing_dependency_entry_point, module_available

if module_available("watchdog"):
    from ast_rag.services.watcher_service import *  # noqa: F403
    from ast_rag.services.watcher_service import main  # noqa: F401
else:  # the `mcp` extra is not installed; exercised in a subprocess by the tests
    main = missing_dependency_entry_point(
        "watchdog", feature="The AST-RAG file watcher", extra="mcp"
    )
