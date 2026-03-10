"""
stack_trace - Smart Stack Trace Mapping for AST-RAG.

This module provides stack trace parsing, AST mapping, and analysis
for multiple programming languages (Python, C++, Java, Rust).

Main components:
- parsers: Stack trace parsers for each language
- models: Data models (StackFrame, StackTraceReport, etc.)
- service: StackTraceService for analysis
"""

from .models import (
    Language,
    FrameType,
    StackFrame,
    RootCause,
    SimilarIssue,
    StackTraceReport,
)
from .parsers import (
    StackTraceParser,
    PythonParser,
    CppParser,
    JavaParser,
    RustParser,
    StackTraceParserFactory,
)
from .service import StackTraceService

__all__ = [
    # Models
    "Language",
    "FrameType",
    "StackFrame",
    "RootCause",
    "SimilarIssue",
    "StackTraceReport",
    # Parsers
    "StackTraceParser",
    "PythonParser",
    "CppParser",
    "JavaParser",
    "RustParser",
    "StackTraceParserFactory",
    # Service
    "StackTraceService",
]
