"""
api/__init__.py - Public API exports for AST-RAG.

This module provides the main entry point for the AST-RAG API.

Usage:
    from ast_rag.api import ASTRagAPI
    # or
    from ast_rag.api.ast_rag_api import ASTRagAPI
"""

from ast_rag.api.ast_rag_api import ASTRagAPI

__all__ = ["ASTRagAPI"]
