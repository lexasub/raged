"""
ast_parser.py - Tree-sitter based parser for multi-language AST extraction.

Responsibilities:
- Load and cache tree-sitter Language objects for all supported languages.
- Parse files with optional incremental parsing support.
- Extract ASTNode and ASTEdge objects via per-language tree-sitter queries.

Full extraction: Java, C++.
Skeletal extraction: Rust, Python, TypeScript.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

import tree_sitter_cpp as tscpp
import tree_sitter_java as tsjava
import tree_sitter_rust as tsrust
import tree_sitter_python as tspython
import tree_sitter_typescript as tsts
import tree_sitter as ts
from tree_sitter import Parser, Query, Tree

from ast_rag.models import (
    ASTNode,
    ASTEdge,
    ASTBlock,
)
from ast_rag.services.parsing import LANGUAGE_QUERIES
from ast_rag.services.parsing.node_extractor import NodeExtractor
from ast_rag.services.parsing.edge_extractor import EdgeExtractor
from ast_rag.utils.parse_cache import ParseCache, SQLiteParseCache

logger = logging.getLogger(__name__)

EXT_TO_LANG: dict[str, str] = {
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cc": "cpp",
    ".c": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".hh": "cpp",
    ".h": "cpp",
    ".java": "java",
    ".rs": "rust",
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
}


class ParserManager:
    """Loads and caches tree-sitter parsers and compiled queries per language.

    Usage::

        pm = ParserManager()
        tree = pm.parse_file("/path/to/Foo.java")
        nodes = pm.extract_nodes(tree, "/path/to/Foo.java", "java")
        edges = pm.extract_edges(tree, nodes, "/path/to/Foo.java", "java")
    """

    def __init__(
        self,
        cache: Optional[Union[ParseCache, SQLiteParseCache]] = None,
        config: Optional[dict] = None,
        project_id: str = "default",
    ) -> None:
        self._languages: dict[str, object] = {}
        self._parsers: dict[str, Parser] = {}
        self._compiled_queries: dict[str, dict[str, object]] = {}
        self._project_id: str = project_id

        self._node_extractor = NodeExtractor(project_id=project_id)
        self._edge_extractor = EdgeExtractor(project_id=project_id)

        if cache is not None:
            self._cache: Union[ParseCache, SQLiteParseCache] = cache
        else:
            pc_cfg: dict = (config or {}).get("parse_cache", {})
            if pc_cfg.get("persistence_enabled", False):
                db_path = pc_cfg.get("db_path", ".ast_rag_parse_cache.sqlite")
                self._cache = SQLiteParseCache(db_path)
                logger.info("ParserManager: using SQLiteParseCache at %s", db_path)
            else:
                self._cache = ParseCache()
        self._init_languages()

    def _init_languages(self) -> None:
        lang_defs = {
            "cpp": ts.Language(tscpp.language()),
            "java": ts.Language(tsjava.language()),
            "rust": ts.Language(tsrust.language()),
            "python": ts.Language(tspython.language()),
            "typescript": ts.Language(tsts.language_typescript()),
        }
        for name, lang in lang_defs.items():
            self._languages[name] = lang
            parser = Parser(lang)
            self._parsers[name] = parser
            queries = LANGUAGE_QUERIES.get(name, {})
            compiled: dict[str, object] = {}
            for qname, qstr in queries.items():
                try:
                    compiled[qname] = Query(lang, qstr)
                except Exception as exc:
                    logger.warning("Failed to compile query '%s' for '%s': %s", qname, name, exc)
            self._compiled_queries[name] = compiled

    def detect_language(self, file_path: str) -> Optional[str]:
        ext = Path(file_path).suffix.lower()
        return EXT_TO_LANG.get(ext)

    def parse_file(
        self,
        file_path: str,
        old_tree: Optional[Tree] = None,
        source: Optional[bytes] = None,
        resolve: bool = False,
    ) -> Optional[Tree]:
        lang = self.detect_language(file_path)
        if lang is None:
            return None

        abs_path = os.path.abspath(file_path)

        if source is None:
            try:
                with open(abs_path, "rb") as fh:
                    source = fh.read()
            except OSError as exc:
                logger.error("Cannot read '%s': %s", file_path, exc)
                return None

        if old_tree is None:
            _lang = lang
            _src = source
            lazy = self._cache.get(
                abs_path,
                source,
                loader=lambda: self._parsers[_lang].parse(_src),
            )
            if lazy is not None:
                return lazy.resolve() if resolve else lazy

        parser = self._parsers[lang]
        tree = parser.parse(source, old_tree) if old_tree is not None else parser.parse(source)
        self._cache.put(abs_path, source, tree)
        return tree

    def clear_tree_cache(self) -> None:
        self._cache.clear()

    def tree_cache_stats(self) -> dict:
        return self._cache.stats()

    def extract_nodes(
        self,
        tree: Tree,
        file_path: str,
        lang: str,
        source: Optional[bytes] = None,
        commit_hash: str = "INIT",
    ) -> list[ASTNode]:
        return self._node_extractor.extract_nodes(
            tree=tree,
            file_path=file_path,
            lang=lang,
            compiled_queries=self._compiled_queries.get(lang, {}),
            source=source,
            commit_hash=commit_hash,
        )

    def extract_edges(
        self,
        tree: Tree,
        nodes: list[ASTNode],
        file_path: str,
        lang: str,
        source: Optional[bytes] = None,
        commit_hash: str = "INIT",
    ) -> list[ASTEdge]:
        return self._edge_extractor.extract_edges(
            tree=tree,
            nodes=nodes,
            file_path=file_path,
            lang=lang,
            compiled_queries=self._compiled_queries.get(lang, {}),
            source=source,
            commit_hash=commit_hash,
        )

    def extract_blocks(
        self,
        tree: Tree,
        nodes: list[ASTNode],
        file_path: str,
        lang: str,
        source: Optional[bytes] = None,
        commit_hash: str = "INIT",
    ) -> tuple[list[ASTBlock], list[ASTEdge]]:
        from ast_rag.models import ASTBlock, ASTEdge, EdgeKind, NodeKind

        if lang not in ("python", "rust"):
            return [], []

        if source is None:
            try:
                with open(file_path, "rb") as fh:
                    source = fh.read()
            except OSError:
                return [], []

        function_nodes = [
            n
            for n in nodes
            if n.kind
            in (NodeKind.FUNCTION, NodeKind.METHOD, NodeKind.CONSTRUCTOR, NodeKind.DESTRUCTOR)
        ]

        if not function_nodes:
            return [], []

        from ast_rag.services.parsing.block_extractor import BlockExtractor

        extractor = BlockExtractor()
        blocks = extractor.extract_blocks(tree, source, function_nodes, lang, commit_hash)

        edges: list[ASTEdge] = []
        for block in blocks:
            edge = ASTEdge(
                kind=EdgeKind.CONTAINS_BLOCK,
                from_id=block.parent_function_id,
                to_id=block.id,
                label=block.block_type.value,
                valid_from=commit_hash,
            )
            edges.append(edge)

        logger.debug(
            "Extracted %d blocks and %d CONTAINS_BLOCK edges from %s (%s)",
            len(blocks),
            len(edges),
            file_path,
            lang,
        )

        return blocks, edges


def walk_source_files(
    root: str,
    exclude_dirs: Optional[list[str]] = None,
) -> list[tuple[str, str]]:
    if exclude_dirs is None:
        exclude_dirs = [
            ".git",
            "__pycache__",
            "node_modules",
            "target",
            "build",
            "dist",
            ".gradle",
            ".idea",
            ".vscode",
        ]
    result: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs and not d.startswith(".")]
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            lang = EXT_TO_LANG.get(ext)
            if lang:
                result.append((os.path.join(dirpath, fname), lang))
    return result
