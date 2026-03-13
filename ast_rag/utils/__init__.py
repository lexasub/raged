"""AST-RAG Utilities.

Common utilities and helpers:
- metrics: Prometheus metrics and latency tracking
- output: Output formatters (JSON, Rich tables)
- file_cache: File content caching
- parse_cache: Parse tree caching
"""

from ast_rag.utils.metrics import (
    track_latency,
    SEARCH_LATENCY,
    FIND_DEFINITION_LATENCY,
    FIND_REFERENCES_LATENCY,
    SEARCH_TOTAL,
    UPDATE_LATENCY,
    INDEX_FILES_TOTAL,
    INDEX_ERRORS_TOTAL,
)
from ast_rag.utils.output import OutputFormatter, get_formatter
from ast_rag.utils.file_cache import FileCache
from ast_rag.utils.parse_cache import ParseCache, SQLiteParseCache

__all__ = [
    # Metrics
    "track_latency",
    "SEARCH_LATENCY",
    "FIND_DEFINITION_LATENCY",
    "FIND_REFERENCES_LATENCY",
    "SEARCH_TOTAL",
    "UPDATE_LATENCY",
    "INDEX_FILES_TOTAL",
    "INDEX_ERRORS_TOTAL",
    # Output
    "OutputFormatter",
    "get_formatter",
    # Caches
    "FileCache",
    "ParseCache",
    "SQLiteParseCache",
]
