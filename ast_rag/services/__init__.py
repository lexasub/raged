"""AST-RAG Services.

This package provides high-level business logic services:
- ParsingService: Code parsing and AST extraction
- EmbeddingManager: Vector embeddings management
- GraphService: Neo4j graph operations
- SearchService: Semantic and keyword search
- SummarizerService: LLM-based code summarization
- WorkspaceWatcher: File system watcher for incremental updates
"""

from typing import TYPE_CHECKING, Any

from ast_rag.services.config import ServiceConfig, LLMConfig
from ast_rag.services.embedding_manager import EmbeddingManager

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ast_rag.services.watcher_service import WorkspaceWatcher

__all__ = [
    "ServiceConfig",
    "LLMConfig",
    "EmbeddingManager",
    "WorkspaceWatcher",
]


def __getattr__(name: str) -> Any:
    """Load watcher support lazily.

    ``watcher_service`` imports ``watchdog``, which ships in the optional
    ``mcp``/``dev`` extras rather than the base dependencies. Importing it
    eagerly here made every ``ast_rag.services`` import -- and therefore the
    whole CLI -- fail with ModuleNotFoundError after a plain ``pip install -e .``.
    """
    if name == "WorkspaceWatcher":
        from ast_rag.services.watcher_service import WorkspaceWatcher

        return WorkspaceWatcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
