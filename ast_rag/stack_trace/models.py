"""
models.py - Data models for Smart Stack Trace Mapping.

Defines data structures for:
- Stack trace frames
- Root cause analysis
- Call chain
- Similar issues
- StackTraceReport
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Language(str, Enum):
    """Supported languages for stack trace parsing."""
    PYTHON = "python"
    CPP = "cpp"
    JAVA = "java"
    RUST = "rust"
    UNKNOWN = "unknown"


class FrameType(str, Enum):
    """Type of stack frame."""
    FUNCTION_CALL = "function_call"
    METHOD_CALL = "method_call"
    CONSTRUCTOR = "constructor"
    LAMBDA = "lambda"
    EXCEPTION_HANDLER = "exception_handler"
    ASYNC_CALLBACK = "async_callback"


class StackFrame(BaseModel):
    """Represents a single frame in a stack trace.
    
    Attributes:
        frame_index: Position in the call stack (0 = topmost/innermost)
        function_name: Name of the function/method
        class_name: Class name (for OOP languages)
        file_path: Source file path
        line_number: Line number where the call occurred
        column: Column number (if available)
        language: Detected language for this frame
        frame_type: Type of frame (function, method, etc.)
        raw_line: Original raw line from stack trace
        module: Module/namespace/package name
        arguments: Function arguments (if parseable)
        is_native: True if this is a native/system frame
        is_async: True if this is an async frame
        code_snippet: Source code snippet (populated after AST mapping)
        ast_node_id: AST node ID (populated after AST mapping)
    """
    frame_index: int
    function_name: str
    class_name: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column: Optional[int] = None
    language: Language = Language.UNKNOWN
    frame_type: FrameType = FrameType.FUNCTION_CALL
    raw_line: str = ""
    module: Optional[str] = None
    arguments: list[str] = Field(default_factory=list)
    is_native: bool = False
    is_async: bool = False
    code_snippet: Optional[str] = None
    ast_node_id: Optional[str] = None
    ast_node_qualified_name: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "frame_index": self.frame_index,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column": self.column,
            "language": self.language.value,
            "frame_type": self.frame_type.value,
            "raw_line": self.raw_line,
            "module": self.module,
            "arguments": self.arguments,
            "is_native": self.is_native,
            "is_async": self.is_async,
            "code_snippet": self.code_snippet,
            "ast_node_id": self.ast_node_id,
            "ast_node_qualified_name": self.ast_node_qualified_name,
        }


class RootCause(BaseModel):
    """Analysis of the root cause of the error.
    
    Attributes:
        error_type: Type/class of the error (e.g., NullPointerException)
        error_message: The error message text
        likely_cause: Human-readable explanation of the likely cause
        severity: Severity level (critical, high, medium, low)
        category: Error category (null_pointer, out_of_bounds, type_error, etc.)
        suggested_fix: Suggested fix for the root cause
        confidence: Confidence score (0.0-1.0) in the analysis
        related_frames: Indices of frames related to the root cause
    """
    error_type: str
    error_message: str
    likely_cause: Optional[str] = None
    severity: str = "medium"  # critical, high, medium, low
    category: Optional[str] = None  # null_pointer, out_of_bounds, type_error, etc.
    suggested_fix: Optional[str] = None
    confidence: float = 0.0
    related_frames: list[int] = Field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "likely_cause": self.likely_cause,
            "severity": self.severity,
            "category": self.category,
            "suggested_fix": self.suggested_fix,
            "confidence": self.confidence,
            "related_frames": self.related_frames,
        }


class SimilarIssue(BaseModel):
    """Represents a similar issue found in the codebase or knowledge base.
    
    Attributes:
        issue_id: Unique identifier for the issue
        title: Issue title/description
        location: File path or symbol where similar issue occurred
        similarity_score: How similar this issue is (0.0-1.0)
        resolution: How the issue was resolved (if known)
        source: Source of the issue (internal, stackoverflow, github, etc.)
        url: Link to the issue (if available)
    """
    issue_id: str
    title: str
    location: Optional[str] = None
    similarity_score: float = 0.0
    resolution: Optional[str] = None
    source: str = "internal"
    url: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "issue_id": self.issue_id,
            "title": self.title,
            "location": self.location,
            "similarity_score": self.similarity_score,
            "resolution": self.resolution,
            "source": self.source,
            "url": self.url,
        }


class StackTraceReport(BaseModel):
    """Complete analysis report for a stack trace.
    
    This is the main output model for StackTraceService.analyze().
    
    Attributes:
        error_type: Primary error type (e.g., NullPointerException)
        message: Full error message
        language: Detected language of the stack trace
        root_cause: Root cause analysis
        call_chain: Ordered list of stack frames (top to bottom)
        similar_issues: List of similar issues found
        summary: Human-readable summary of the analysis
        total_frames: Total number of frames parsed
        mapped_frames: Number of frames successfully mapped to AST
        analysis_metadata: Additional metadata about the analysis
    """
    error_type: str
    message: str
    language: Language = Language.UNKNOWN
    root_cause: Optional[RootCause] = None
    call_chain: list[StackFrame] = Field(default_factory=list)
    similar_issues: list[SimilarIssue] = Field(default_factory=list)
    summary: Optional[str] = None
    total_frames: int = 0
    mapped_frames: int = 0
    analysis_metadata: dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "language": self.language.value,
            "root_cause": self.root_cause.to_dict() if self.root_cause else None,
            "call_chain": [frame.to_dict() for frame in self.call_chain],
            "similar_issues": [issue.to_dict() for issue in self.similar_issues],
            "summary": self.summary,
            "total_frames": self.total_frames,
            "mapped_frames": self.mapped_frames,
            "analysis_metadata": self.analysis_metadata,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def to_markdown(self) -> str:
        """Render as Markdown for chat display."""
        lines = []
        lines.append(f"# Stack Trace Analysis Report\n")
        lines.append(f"## Error: `{self.error_type}`\n")
        lines.append(f"**Message:** {self.message}\n")
        lines.append(f"**Language:** {self.language.value}\n")
        lines.append(f"**Total Frames:** {self.total_frames} ({self.mapped_frames} mapped to AST)\n")
        
        if self.root_cause:
            lines.append("\n## Root Cause Analysis\n")
            lines.append(f"- **Type:** {self.root_cause.error_type}")
            lines.append(f"- **Category:** {self.root_cause.category or 'N/A'}")
            lines.append(f"- **Severity:** {self.root_cause.severity}")
            lines.append(f"- **Confidence:** {self.root_cause.confidence:.0%}")
            if self.root_cause.likely_cause:
                lines.append(f"- **Likely Cause:** {self.root_cause.likely_cause}")
            if self.root_cause.suggested_fix:
                lines.append(f"- **Suggested Fix:** {self.root_cause.suggested_fix}")
        
        if self.call_chain:
            lines.append("\n## Call Chain\n")
            for frame in self.call_chain[:10]:  # Show first 10
                loc = ""
                if frame.file_path and frame.line_number:
                    loc = f" at {frame.file_path}:{frame.line_number}"
                lines.append(f"{frame.frame_index}. `{frame.function_name}()`{loc}")
            if len(self.call_chain) > 10:
                lines.append(f"... and {len(self.call_chain) - 10} more frames")
        
        if self.similar_issues:
            lines.append("\n## Similar Issues\n")
            for issue in self.similar_issues[:5]:  # Show first 5
                lines.append(f"- [{issue.similarity_score:.0%}] {issue.title}")
                if issue.resolution:
                    lines.append(f"  - Resolution: {issue.resolution}")
        
        if self.summary:
            lines.append(f"\n## Summary\n")
            lines.append(self.summary)
        
        return "\n".join(lines)
