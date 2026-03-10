"""
service.py - StackTraceService for Smart Stack Trace Mapping.

Main service class that:
1. Parses stack traces for multiple languages
2. Maps frames to AST nodes
3. Retrieves code snippets
4. Analyzes root cause
5. Finds similar issues
6. Generates comprehensive reports
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

from neo4j import Driver

from ast_rag.ast_rag_api import ASTRagAPI
from ast_rag.embeddings import EmbeddingManager
from ast_rag.models import ASTNode, Language as ModelLanguage, NodeKind

from .models import (
    Language,
    StackFrame,
    RootCause,
    SimilarIssue,
    StackTraceReport,
)
from .parsers import (
    StackTraceParser,
    StackTraceParserFactory,
    PythonParser,
    CppParser,
    JavaParser,
    RustParser,
)

logger = logging.getLogger(__name__)


# Error category mappings for root cause analysis
ERROR_CATEGORIES: dict[str, list[str]] = {
    "null_pointer": [
        "NullPointerException", "NullReferenceException", "NoneType", 
        "null pointer", "None has no attribute", "undefined is not a function"
    ],
    "out_of_bounds": [
        "IndexOutOfBoundsException", "IndexError", "out of range", 
        "out of bounds", "array index", "slice index", "bounds check"
    ],
    "type_error": [
        "TypeError", "ClassCastException", "type mismatch", 
        "cannot convert", "incompatible type", "no matching function"
    ],
    "value_error": [
        "ValueError", "IllegalArgumentException", "invalid argument",
        "invalid value", "invalid literal", "format error"
    ],
    "key_error": [
        "KeyError", "NoSuchElementException", "key not found",
        "missing key", "map key", "hash key"
    ],
    "attribute_error": [
        "AttributeError", "MissingPropertyException", "has no attribute",
        "property not found", "member not found"
    ],
    "file_error": [
        "FileNotFoundError", "IOException", "file not found",
        "cannot open file", "access denied", "permission denied"
    ],
    "memory_error": [
        "MemoryError", "OutOfMemoryError", "out of memory",
        "memory allocation", "heap space", "stack overflow"
    ],
    "concurrency": [
        "ConcurrentModificationException", "Deadlock", "RaceCondition",
        "thread", "lock", "synchronization", "mutex"
    ],
    "panic": [
        "panic", "unreachable", "assertion failed", "unwrap failed"
    ],
}

# Severity mappings
ERROR_SEVERITY: dict[str, str] = {
    "null_pointer": "high",
    "out_of_bounds": "high",
    "memory_error": "critical",
    "concurrency": "critical",
    "panic": "critical",
    "file_error": "medium",
    "type_error": "medium",
    "value_error": "medium",
    "key_error": "low",
    "attribute_error": "low",
}


class StackTraceService:
    """Service for analyzing stack traces and mapping to AST.
    
    This service provides comprehensive stack trace analysis:
    1. Parse stack traces from multiple languages
    2. Map each frame to AST nodes in the graph database
    3. Retrieve code snippets for context
    4. Analyze root cause and suggest fixes
    5. Find similar issues in the codebase
    
    Usage:
        service = StackTraceService(driver, embedding_manager)
        report = service.analyze(stacktrace_text)
        print(report.to_markdown())
    """
    
    def __init__(
        self,
        driver: Driver,
        embedding_manager: EmbeddingManager,
        codebase_root: Optional[str] = None,
    ) -> None:
        """Initialize the stack trace service.
        
        Args:
            driver: Neo4j driver for graph database access
            embedding_manager: EmbeddingManager for semantic search
            codebase_root: Optional root path for the codebase
        """
        self._driver = driver
        self._api = ASTRagAPI(driver, embedding_manager)
        self._embed = embedding_manager
        self._codebase_root = Path(codebase_root) if codebase_root else None
        
        # Initialize parsers
        self._parsers = StackTraceParserFactory.get_all_parsers()
    
    def analyze(self, stacktrace: str) -> StackTraceReport:
        """Analyze a stack trace and generate a comprehensive report.
        
        This is the main entry point for stack trace analysis.
        
        Args:
            stacktrace: Raw stack trace text
            
        Returns:
            StackTraceReport with full analysis
        """
        # Step 1: Detect language and parse frames
        parser, frames, detected_language = StackTraceParserFactory.detect_and_parse(stacktrace)
        
        # Step 2: Extract error info
        error_type, error_message = parser.extract_error_info(stacktrace)
        
        # Step 3: Create initial report
        report = StackTraceReport(
            error_type=error_type,
            message=error_message,
            language=detected_language,
            call_chain=frames,
            total_frames=len(frames),
        )
        
        # Step 4: Map frames to AST nodes
        mapped_count = self._map_frames_to_ast(frames)
        report.mapped_frames = mapped_count
        
        # Step 5: Retrieve code snippets for mapped frames
        self._retrieve_code_snippets(frames)
        
        # Step 6: Analyze root cause
        report.root_cause = self._analyze_root_cause(error_type, error_message, frames)
        
        # Step 7: Find similar issues
        report.similar_issues = self._find_similar_issues(error_type, error_message, frames)
        
        # Step 8: Generate summary
        report.summary = self._generate_summary(report)
        
        # Step 9: Analyze text with existing API for additional context
        text_results = self._analyze_with_text_api(stacktrace)
        if text_results and not report.similar_issues:
            report.similar_issues = self._convert_text_results_to_issues(text_results)
        
        return report
    
    def _map_frames_to_ast(self, frames: list[StackFrame]) -> int:
        """Map stack frames to AST nodes in the graph database.
        
        For each frame, attempts to find the corresponding AST node
        using file path, line number, and function name.
        
        Args:
            frames: List of StackFrame objects to map
            
        Returns:
            Number of successfully mapped frames
        """
        mapped_count = 0
        
        for frame in frames:
            if not frame.file_path and not frame.function_name:
                continue
            
            # Strategy 1: Find by file path and line number
            if frame.file_path and frame.line_number:
                node = self._find_node_by_location(
                    frame.file_path,
                    frame.line_number,
                    frame.language,
                )
                if node:
                    frame.ast_node_id = node.id
                    frame.ast_node_qualified_name = node.qualified_name
                    mapped_count += 1
                    continue
            
            # Strategy 2: Find by function/class name
            if frame.function_name:
                node = self._find_node_by_name(
                    frame.function_name,
                    frame.class_name,
                    frame.language,
                )
                if node:
                    frame.ast_node_id = node.id
                    frame.ast_node_qualified_name = node.qualified_name
                    mapped_count += 1
                    continue
            
            # Strategy 3: Semantic search by frame context
            if frame.function_name or frame.raw_line:
                node = self._find_node_by_semantic_search(frame)
                if node:
                    frame.ast_node_id = node.id
                    frame.ast_node_qualified_name = node.qualified_name
                    mapped_count += 1
        
        return mapped_count
    
    def _find_node_by_location(
        self,
        file_path: str,
        line_number: int,
        language: Language,
    ) -> Optional[ASTNode]:
        """Find AST node by file path and line number.
        
        Args:
            file_path: Source file path
            line_number: Line number (1-indexed)
            language: Programming language
            
        Returns:
            ASTNode if found, None otherwise
        """
        # Convert our Language enum to model's Language enum
        lang_value = self._convert_language(language)
        
        cypher = """
MATCH (n)
WHERE n.valid_to IS NULL
  AND n.file_path ENDS WITH $file_path
  AND n.start_line <= $line_number
  AND n.end_line >= $line_number
  AND n.lang = $lang
RETURN n
ORDER BY n.start_line DESC
LIMIT 1
"""
        # Normalize file path for matching
        normalized_path = file_path.replace('\\', '/')
        
        with self._driver.session() as session:
            result = session.run(
                cypher,
                file_path=normalized_path,
                line_number=line_number,
                lang=lang_value,
            )
            record = result.single()
            if record:
                return self._record_to_node(dict(record["n"]))
        
        return None
    
    def _find_node_by_name(
        self,
        function_name: str,
        class_name: Optional[str],
        language: Language,
    ) -> Optional[ASTNode]:
        """Find AST node by function/class name.
        
        Args:
            function_name: Function or method name
            class_name: Optional class name for methods
            language: Programming language
            
        Returns:
            ASTNode if found, None otherwise
        """
        lang_value = self._convert_language(language)
        
        # Build qualified name if class is provided
        qualified_name = f"{class_name}.{function_name}" if class_name else function_name
        
        # Try exact match first
        cypher = """
MATCH (n)
WHERE n.valid_to IS NULL
  AND (n.name = $name OR n.qualified_name = $qualified_name)
  AND n.lang = $lang
  AND n.kind IN ['Function', 'Method', 'Constructor']
RETURN n
LIMIT 1
"""
        with self._driver.session() as session:
            result = session.run(
                cypher,
                name=function_name,
                qualified_name=qualified_name,
                lang=lang_value,
            )
            record = result.single()
            if record:
                return self._record_to_node(dict(record["n"]))
        
        # Try contains match
        cypher = """
MATCH (n)
WHERE n.valid_to IS NULL
  AND n.qualified_name CONTAINS $name
  AND n.lang = $lang
  AND n.kind IN ['Function', 'Method', 'Constructor']
RETURN n
ORDER BY n.qualified_name
LIMIT 1
"""
        with self._driver.session() as session:
            result = session.run(
                cypher,
                name=function_name,
                lang=lang_value,
            )
            record = result.single()
            if record:
                return self._record_to_node(dict(record["n"]))
        
        return None
    
    def _find_node_by_semantic_search(self, frame: StackFrame) -> Optional[ASTNode]:
        """Find AST node by semantic search on frame context.
        
        Args:
            frame: StackFrame to search for
            
        Returns:
            ASTNode if found, None otherwise
        """
        # Build search query from frame context
        query_parts = []
        if frame.function_name:
            query_parts.append(frame.function_name)
        if frame.class_name:
            query_parts.append(frame.class_name)
        if frame.module:
            query_parts.append(frame.module)
        
        if not query_parts:
            return None
        
        query = " ".join(query_parts)
        
        # Filter by language
        lang_filter = self._convert_language(frame.language).value
        
        try:
            results = self._api.search_semantic(query, limit=5, lang=lang_filter)
            if results:
                return results[0].node
        except Exception as e:
            logger.warning(f"Semantic search failed for frame {frame.function_name}: {e}")
        
        return None
    
    def _retrieve_code_snippets(self, frames: list[StackFrame]) -> None:
        """Retrieve code snippets for mapped frames.
        
        Populates the code_snippet field for each frame that has
        been mapped to an AST node.
        
        Args:
            frames: List of StackFrame objects
        """
        for frame in frames:
            if frame.ast_node_id and frame.file_path and frame.line_number:
                # Get snippet from the API
                # Use a reasonable context: 5 lines before and after
                start_line = max(1, frame.line_number - 5)
                end_line = frame.line_number + 5
                snippet = self._api.get_code_snippet(
                    frame.file_path,
                    start_line,
                    end_line,
                )
                frame.code_snippet = snippet
    
    def _analyze_root_cause(
        self,
        error_type: str,
        error_message: str,
        frames: list[StackFrame],
    ) -> RootCause:
        """Analyze the root cause of the error.
        
        Args:
            error_type: Type of error (e.g., NullPointerException)
            error_message: Error message text
            frames: List of stack frames
            
        Returns:
            RootCause analysis object
        """
        # Determine error category
        category = self._categorize_error(error_type, error_message)
        severity = ERROR_SEVERITY.get(category, "medium")
        
        # Find frames related to the error (usually the first few)
        related_frames = [f.frame_index for f in frames[:3]]
        
        # Generate likely cause explanation
        likely_cause = self._generate_likely_cause(category, error_type, error_message, frames)
        
        # Generate suggested fix
        suggested_fix = self._generate_suggested_fix(category, error_type, frames)
        
        # Calculate confidence based on how well we could categorize
        confidence = 0.5  # Base confidence
        if category and category != "unknown":
            confidence += 0.2
        if frames and frames[0].ast_node_id:
            confidence += 0.2
        if error_message and len(error_message) > 10:
            confidence += 0.1
        
        return RootCause(
            error_type=error_type,
            error_message=error_message,
            likely_cause=likely_cause,
            severity=severity,
            category=category,
            suggested_fix=suggested_fix,
            confidence=min(confidence, 1.0),
            related_frames=related_frames,
        )
    
    def _categorize_error(self, error_type: str, error_message: str) -> str:
        """Categorize the error type.
        
        Args:
            error_type: Error type string
            error_message: Error message string
            
        Returns:
            Category string (e.g., 'null_pointer', 'out_of_bounds')
        """
        text = f"{error_type} {error_message}".lower()
        
        for category, keywords in ERROR_CATEGORIES.items():
            if any(keyword.lower() in text for keyword in keywords):
                return category
        
        return "unknown"
    
    def _generate_likely_cause(
        self,
        category: str,
        error_type: str,
        error_message: str,
        frames: list[StackFrame],
    ) -> str:
        """Generate human-readable likely cause explanation.
        
        Args:
            category: Error category
            error_type: Error type
            error_message: Error message
            frames: Stack frames
            
        Returns:
            Human-readable explanation
        """
        explanations = {
            "null_pointer": (
                f"A null/None reference was accessed. "
                f"This typically happens when a variable expected to hold an object "
                f"contains null/None instead. Check the value at {frames[0].file_path}:{frames[0].line_number if frames and frames[0].line_number else 'unknown location'}."
            ),
            "out_of_bounds": (
                f"An array/list index was out of valid range. "
                f"The index used was likely beyond the array bounds or negative. "
                f"Verify array length before accessing elements."
            ),
            "type_error": (
                f"An operation was performed on an incompatible type. "
                f"Check that the types of operands match the expected types for this operation."
            ),
            "value_error": (
                f"A function received an argument with the correct type but invalid value. "
                f"Validate input values before processing."
            ),
            "key_error": (
                f"A dictionary/map key was not found. "
                f"The key being accessed does not exist in the collection. "
                f"Use .get() method or check key existence before access."
            ),
            "attribute_error": (
                f"An attribute or method was accessed on an object that doesn't have it. "
                f"Verify the object type and available methods."
            ),
            "file_error": (
                f"A file operation failed. The file may not exist, or there may be "
                f"permission issues. Check file paths and permissions."
            ),
            "memory_error": (
                f"The program ran out of memory. This could be due to a memory leak, "
                f"large data structures, or insufficient system memory."
            ),
            "concurrency": (
                f"A threading or synchronization issue occurred. "
                f"This could be a race condition, deadlock, or concurrent modification."
            ),
            "panic": (
                f"The program encountered an unrecoverable error and panicked. "
                f"This is often due to an assertion failure or explicit panic call."
            ),
            "unknown": (
                f"An error of type '{error_type}' occurred: {error_message}. "
                f"Review the stack trace and code context for more details."
            ),
        }
        
        return explanations.get(category, explanations["unknown"])
    
    def _generate_suggested_fix(
        self,
        category: str,
        error_type: str,
        frames: list[StackFrame],
    ) -> str:
        """Generate suggested fix for the error.
        
        Args:
            category: Error category
            error_type: Error type
            frames: Stack frames
            
        Returns:
            Suggested fix description
        """
        fixes = {
            "null_pointer": (
                "1. Add null checks before accessing object properties\n"
                "2. Use Optional/Maybe types where appropriate\n"
                "3. Initialize variables before use\n"
                "4. Review the call chain for missing initializations"
            ),
            "out_of_bounds": (
                "1. Check array/list length before accessing by index\n"
                "2. Use safe access methods (e.g., .get() with default)\n"
                "3. Validate input indices\n"
                "4. Consider using iterators instead of index-based access"
            ),
            "type_error": (
                "1. Add type hints/annotations\n"
                "2. Use type checking tools (mypy, mypy, etc.)\n"
                "3. Validate types at function boundaries\n"
                "4. Review API contracts"
            ),
            "value_error": (
                "1. Add input validation\n"
                "2. Use try-catch for expected invalid inputs\n"
                "3. Document valid value ranges\n"
                "4. Provide clear error messages"
            ),
            "key_error": (
                "1. Use .get() method with default values\n"
                "2. Check key existence with 'in' operator\n"
                "3. Use defaultdict for missing keys\n"
                "4. Validate keys before access"
            ),
            "attribute_error": (
                "1. Verify object type before method calls\n"
                "2. Use hasattr() or getattr() with defaults\n"
                "3. Review class hierarchy and inheritance\n"
                "4. Check for typos in attribute names"
            ),
            "file_error": (
                "1. Check file existence before operations\n"
                "2. Use absolute paths or resolve relative paths\n"
                "3. Verify file permissions\n"
                "4. Handle missing files gracefully"
            ),
            "memory_error": (
                "1. Profile memory usage to find leaks\n"
                "2. Process data in chunks instead of loading all at once\n"
                "3. Release resources explicitly\n"
                "4. Consider increasing memory limits"
            ),
            "concurrency": (
                "1. Use proper synchronization primitives\n"
                "2. Avoid shared mutable state\n"
                "3. Use thread-safe collections\n"
                "4. Review lock ordering to prevent deadlocks"
            ),
            "panic": (
                "1. Handle errors explicitly instead of panicking\n"
                "2. Use Result/Option types for recoverable errors\n"
                "3. Add proper error handling in callers\n"
                "4. Review assertion conditions"
            ),
            "unknown": (
                "1. Review the stack trace carefully\n"
                "2. Check the code at the error location\n"
                "3. Add logging to understand the state\n"
                "4. Search for similar issues in the codebase"
            ),
        }
        
        return fixes.get(category, fixes["unknown"])
    
    def _find_similar_issues(
        self,
        error_type: str,
        error_message: str,
        frames: list[StackFrame],
    ) -> list[SimilarIssue]:
        """Find similar issues in the codebase.
        
        Uses semantic search to find similar error patterns or
        related code sections.
        
        Args:
            error_type: Error type
            error_message: Error message
            frames: Stack frames
            
        Returns:
            List of SimilarIssue objects
        """
        similar = []
        
        # Search by error type
        query = f"{error_type} error handling exception"
        try:
            results = self._api.search_semantic(query, limit=5)
            for i, result in enumerate(results[:3]):
                node = result.node
                similar.append(SimilarIssue(
                    issue_id=f"similar_{i}",
                    title=f"Similar code pattern in {node.qualified_name}",
                    location=f"{node.file_path}:{node.start_line}",
                    similarity_score=result.score,
                    source="internal",
                ))
        except Exception as e:
            logger.warning(f"Similar issue search failed: {e}")
        
        return similar
    
    def _generate_summary(self, report: StackTraceReport) -> str:
        """Generate a human-readable summary of the analysis.
        
        Args:
            report: StackTraceReport object
            
        Returns:
            Summary string
        """
        parts = []
        
        # Error summary
        parts.append(f"Error: {report.error_type}")
        if report.message:
            parts.append(f"Message: {report.message}")
        
        # Language info
        parts.append(f"Language: {report.language.value}")
        
        # Frame summary
        parts.append(f"Stack frames: {report.total_frames} total, {report.mapped_frames} mapped to AST")
        
        # Root cause
        if report.root_cause:
            rc = report.root_cause
            parts.append(f"Root cause: {rc.likely_cause or 'Analysis pending'}")
            if rc.suggested_fix:
                parts.append(f"Suggested fix: {rc.suggested_fix[:100]}...")
        
        # Similar issues
        if report.similar_issues:
            parts.append(f"Found {len(report.similar_issues)} similar issues in codebase")
        
        return " | ".join(parts)
    
    def _analyze_with_text_api(self, stacktrace: str) -> list[Any]:
        """Use existing analyze_text API for additional context.
        
        Args:
            stacktrace: Raw stack trace text
            
        Returns:
            List of search results
        """
        try:
            results = self._api.analyze_text(stacktrace, limit=10)
            return results
        except Exception as e:
            logger.warning(f"analyze_text failed: {e}")
            return []
    
    def _convert_text_results_to_issues(
        self,
        results: list[Any],
    ) -> list[SimilarIssue]:
        """Convert analyze_text results to SimilarIssue objects.
        
        Args:
            results: Results from analyze_text API
            
        Returns:
            List of SimilarIssue objects
        """
        issues = []
        for i, result in enumerate(results[:5]):
            node = result.node if hasattr(result, 'node') else result
            if hasattr(node, 'file_path'):
                issues.append(SimilarIssue(
                    issue_id=f"text_result_{i}",
                    title=f"Related code: {node.qualified_name if hasattr(node, 'qualified_name') else 'Unknown'}",
                    location=f"{node.file_path}:{node.start_line if hasattr(node, 'start_line') else '?'}",
                    similarity_score=result.score if hasattr(result, 'score') else 0.5,
                    source="semantic_search",
                ))
        return issues
    
    def _convert_language(self, language: Language) -> ModelLanguage:
        """Convert stack_trace.Language to models.Language.
        
        Args:
            language: Language from stack_trace module
            
        Returns:
            Language from models module
        """
        mapping = {
            Language.PYTHON: ModelLanguage.PYTHON,
            Language.CPP: ModelLanguage.CPP,
            Language.JAVA: ModelLanguage.JAVA,
            Language.RUST: ModelLanguage.RUST,
            Language.UNKNOWN: ModelLanguage.PYTHON,  # Default fallback
        }
        return mapping.get(language, ModelLanguage.PYTHON)
    
    def _record_to_node(self, record: dict) -> ASTNode:
        """Convert Neo4j record to ASTNode.
        
        Args:
            record: Neo4j record dict
            
        Returns:
            ASTNode object
        """
        r = record
        return ASTNode(
            id=r.get("id", ""),
            kind=NodeKind(r.get("kind", "Function")),
            name=r.get("name", ""),
            qualified_name=r.get("qualified_name", ""),
            lang=ModelLanguage(r.get("lang", "python")),
            file_path=r.get("file_path", ""),
            start_line=int(r.get("start_line", 0)),
            end_line=int(r.get("end_line", 0)),
            start_byte=int(r.get("start_byte", 0)),
            end_byte=int(r.get("end_byte", 0)),
            code_hash=r.get("code_hash", ""),
            signature=r.get("signature") or None,
            valid_from=r.get("valid_from", "INIT"),
            valid_to=r.get("valid_to"),
        )
    
    def analyze_from_file(self, file_path: str) -> StackTraceReport:
        """Analyze a stack trace from a file.
        
        Args:
            file_path: Path to file containing stack trace
            
        Returns:
            StackTraceReport object
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Stack trace file not found: {file_path}")
        
        stacktrace = path.read_text(encoding="utf-8", errors="replace")
        return self.analyze(stacktrace)
