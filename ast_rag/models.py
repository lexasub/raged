"""
models.py - Backward compatibility module.

This module re-exports all DTOs for backward compatibility.
New code should import directly from ast_rag.dto.
"""

from ast_rag.dto import (
    NodeKind,
    EdgeKind,
    Language,
    BlockType,
    ASTNode,
    ASTEdge,
    ASTBlock,
    DiffResult,
    SubGraph,
    SearchResult,
    StandardResult,
    Neo4jConfig,
    QdrantConfig,
    EmbeddingConfig,
    ProjectConfig,
)

__all__ = [
    "NodeKind",
    "EdgeKind",
    "Language",
    "BlockType",
    "ASTNode",
    "ASTEdge",
    "ASTBlock",
    "DiffResult",
    "SubGraph",
    "SearchResult",
    "StandardResult",
    "Neo4jConfig",
    "QdrantConfig",
    "EmbeddingConfig",
    "ProjectConfig",
]
