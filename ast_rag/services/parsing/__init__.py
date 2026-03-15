"""AST-RAG Parsing Services.

Code parsing and AST extraction using Tree-sitter:
- ParserManager: Main parser orchestrator
- BlockExtractor: Extract code blocks from functions
- language_queries: Tree-sitter query definitions (per-language modules)
"""

from ast_rag.services.parsing.java import JAVA_QUERIES
from ast_rag.services.parsing.cpp import CPP_QUERIES
from ast_rag.services.parsing.rust import RUST_QUERIES
from ast_rag.services.parsing.python import PYTHON_QUERIES
from ast_rag.services.parsing.typescript import TYPESCRIPT_QUERIES

LANGUAGE_QUERIES: dict[str, dict[str, str]] = {
    "java": JAVA_QUERIES,
    "cpp": CPP_QUERIES,
    "rust": RUST_QUERIES,
    "python": PYTHON_QUERIES,
    "typescript": TYPESCRIPT_QUERIES,
}

from ast_rag.services.parsing.block_extractor import BlockExtractor
from ast_rag.services.parsing.parser_manager import ParserManager
from ast_rag.services.parsing.node_extractor import NodeExtractor
from ast_rag.services.parsing.edge_extractor import EdgeExtractor

__all__ = [
    "LANGUAGE_QUERIES",
    "JAVA_QUERIES",
    "CPP_QUERIES",
    "RUST_QUERIES",
    "PYTHON_QUERIES",
    "TYPESCRIPT_QUERIES",
    "BlockExtractor",
    "ParserManager",
    "NodeExtractor",
    "EdgeExtractor",
]
