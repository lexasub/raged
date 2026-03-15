"""AST-RAG Parsing Service.

Service layer wrapper for ParserManager providing code parsing functionality.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from ast_rag.dto import ASTNode, ASTEdge, Language
from ast_rag.services.parsing.parser_manager import ParserManager, walk_source_files

logger = logging.getLogger(__name__)


class ParsingService:
    """Service for parsing source code into AST nodes and edges.

    This service wraps ParserManager to provide a clean interface for
    parsing code files and extracting AST structures. It handles:
    - Single file parsing
    - Batch file parsing
    - Language detection
    - Parse result caching

    Example:
        >>> parsing_service = ParsingService()
        >>> nodes, edges = parsing_service.parse_file("src/main.py")
    """

    def __init__(
        self,
        cache_enabled: bool = True,
        exclude_patterns: Optional[list[str]] = None,
        project_id: str = "default",
    ) -> None:
        """Initialize the ParsingService.

        Args:
            cache_enabled: Whether to enable parse result caching
            exclude_patterns: Patterns to exclude during parsing
            project_id: Project identifier for cache isolation
        """
        self._cache_enabled = cache_enabled
        self._exclude_patterns = exclude_patterns or []
        self._parser_manager = ParserManager(project_id=project_id)

    def parse_file(
        self,
        file_path: str | Path,
        lang: Optional[str] = None,
        commit_hash: str = "INIT",
    ) -> tuple[list[ASTNode], list[ASTEdge]]:
        """Parse a single source file into AST nodes and edges.

        Args:
            file_path: Path to the source file to parse
            lang: Optional language hint (auto-detected if not provided)
            commit_hash: Version hash for the parsed content

        Returns:
            Tuple of (nodes, edges) extracted from the file

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the language is not supported
        """
        file_path_str = str(file_path)
        
        # Check if file exists
        if not os.path.exists(file_path_str):
            raise FileNotFoundError(f"File not found: {file_path_str}")

        # Detect language if not provided
        if lang is None:
            lang = self._parser_manager.detect_language(file_path_str)
            if lang is None:
                raise ValueError(f"Could not detect language for file: {file_path_str}")

        # Parse the file
        tree = self._parser_manager.parse_file(file_path_str)
        if tree is None:
            raise ValueError(f"Failed to parse file: {file_path_str}")

        # Read source for extraction
        try:
            with open(file_path_str, "rb") as f:
                source = f.read()
        except OSError as exc:
            raise ValueError(f"Could not read file {file_path_str}: {exc}") from exc

        # Extract nodes and edges
        nodes = self._parser_manager.extract_nodes(
            tree=tree,
            file_path=file_path_str,
            lang=lang,
            source=source,
            commit_hash=commit_hash,
        )

        edges = self._parser_manager.extract_edges(
            tree=tree,
            nodes=nodes,
            file_path=file_path_str,
            lang=lang,
            source=source,
            commit_hash=commit_hash,
        )

        logger.debug(
            "Parsed %s: %d nodes, %d edges",
            file_path_str,
            len(nodes),
            len(edges),
        )

        return nodes, edges

    def parse_directory(
        self,
        dir_path: str | Path,
        lang: Optional[str] = None,
        recursive: bool = True,
        commit_hash: str = "INIT",
    ) -> tuple[list[ASTNode], list[ASTEdge]]:
        """Parse all source files in a directory.

        Args:
            dir_path: Path to the directory to parse
            lang: Optional language filter (parses all languages if not provided)
            recursive: Whether to parse subdirectories recursively
            commit_hash: Version hash for the parsed content

        Returns:
            Tuple of (nodes, edges) extracted from all files
        """
        dir_path_str = str(dir_path)
        
        if not os.path.isdir(dir_path_str):
            raise ValueError(f"Directory not found: {dir_path_str}")

        # Get all source files
        if recursive:
            source_files = walk_source_files(dir_path_str)
        else:
            source_files = []
            for fname in os.listdir(dir_path_str):
                fpath = os.path.join(dir_path_str, fname)
                if os.path.isfile(fpath):
                    detected_lang = self._parser_manager.detect_language(fpath)
                    if detected_lang:
                        source_files.append((fpath, detected_lang))

        # Filter by language if specified
        if lang:
            source_files = [(f, l) for f, l in source_files if l == lang]

        # Apply exclude patterns
        if self._exclude_patterns:
            import fnmatch
            filtered_files = []
            for fpath, file_lang in source_files:
                excluded = False
                for pattern in self._exclude_patterns:
                    if fnmatch.fnmatch(fpath, pattern) or fnmatch.fnmatch(os.path.basename(fpath), pattern):
                        excluded = True
                        break
                if not excluded:
                    filtered_files.append((fpath, file_lang))
            source_files = filtered_files

        # Parse all files
        all_nodes: list[ASTNode] = []
        all_edges: list[ASTEdge] = []

        for fpath, file_lang in source_files:
            try:
                nodes, edges = self.parse_file(
                    file_path=fpath,
                    lang=file_lang,
                    commit_hash=commit_hash,
                )
                all_nodes.extend(nodes)
                all_edges.extend(edges)
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", fpath, exc)

        logger.info(
            "Parsed directory %s: %d files, %d nodes, %d edges",
            dir_path_str,
            len(source_files),
            len(all_nodes),
            len(all_edges),
        )

        return all_nodes, all_edges

    def detect_language(self, file_path: str | Path) -> Optional[Language]:
        """Detect the programming language of a file.

        Args:
            file_path: Path to the file

        Returns:
            Detected Language enum value or None if unknown
        """
        lang_str = self._parser_manager.detect_language(str(file_path))
        if lang_str is None:
            return None

        # Map string language to Language enum
        lang_map = {
            "java": Language.JAVA,
            "cpp": Language.CPP,
            "rust": Language.RUST,
            "python": Language.PYTHON,
            "typescript": Language.TYPESCRIPT,
        }
        return lang_map.get(lang_str)

    def clear_cache(self) -> None:
        """Clear the parse result cache."""
        self._parser_manager.clear_tree_cache()
        logger.info("Parse cache cleared")

    def get_cache_stats(self) -> dict:
        """Get parse cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return self._parser_manager.tree_cache_stats()
