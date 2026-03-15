"""
node_extractor.py - AST node extraction from parsed trees.

Extracts meaningful AST nodes (classes, methods, functions, fields, etc.)
from tree-sitter parse trees using compiled queries.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

from tree_sitter import Node, QueryCursor, Tree

from ast_rag.models import (
    ASTNode,
    Language,
    NodeKind,
)

logger = logging.getLogger(__name__)


# Query name → NodeKind mapping
QUERY_KIND_MAP: dict[str, NodeKind] = {
    "class_defs": NodeKind.CLASS,
    "interface_defs": NodeKind.INTERFACE,
    "struct_defs": NodeKind.STRUCT,
    "enum_defs": NodeKind.ENUM,
    "trait_defs": NodeKind.TRAIT,
    "annotation_type_defs": NodeKind.INTERFACE,
    "namespace_defs": NodeKind.NAMESPACE,
    "impl_defs": NodeKind.CLASS,  # Rust impl block mapped to class
    "function_defs": NodeKind.FUNCTION,
    "method_defs": NodeKind.METHOD,
    "constructor_defs": NodeKind.CONSTRUCTOR,
    "destructor_defs": NodeKind.DESTRUCTOR,
    "field_defs": NodeKind.FIELD,
}


class NodeExtractor:
    """Extracts AST nodes from parsed trees using tree-sitter queries."""

    def __init__(self, project_id: str = "default") -> None:
        self._project_id = project_id

    def extract_nodes(
        self,
        tree: Tree,
        file_path: str,
        lang: str,
        compiled_queries: dict[str, object],
        source: Optional[bytes] = None,
        commit_hash: str = "INIT",
    ) -> list[ASTNode]:
        """Extract meaningful AST nodes from a parsed tree.

        Returns a list of ASTNode instances with fully populated fields.
        """
        if source is None:
            try:
                with open(file_path, "rb") as fh:
                    source = fh.read()
            except OSError:
                source = b""

        lang_enum = Language(lang)
        nodes: list[ASTNode] = []

        # Track seen (name, kind, start_line) tuples to avoid duplicates
        seen: set[tuple[str, str, int]] = set()

        for qname, kind in QUERY_KIND_MAP.items():
            query = compiled_queries.get(qname)
            if query is None:
                continue
            matches = QueryCursor(query).matches(tree.root_node)
            for _, match_dict in matches:
                node_ts = match_dict.get("node")
                name_ts = match_dict.get("name")
                if node_ts is None or name_ts is None:
                    continue
                if isinstance(node_ts, list):
                    node_ts = node_ts[0]
                if isinstance(name_ts, list):
                    name_ts = name_ts[0]

                name_text = _node_text(name_ts, source)
                if not name_text:
                    continue

                start_line = node_ts.start_point[0] + 1
                end_line = node_ts.end_point[0] + 1
                start_byte = node_ts.start_byte
                end_byte = node_ts.end_byte

                dedup_key = (name_text, kind.value, start_line)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                # Build qualified_name based on file path + class context
                qname_str = _build_qualified_name(file_path, name_text, lang)
                src_text = source[start_byte:end_byte].decode("utf-8", errors="replace")
                code_hash = hashlib.sha256(src_text.encode()).hexdigest()[:24]

                # Build signature for callables
                signature: Optional[str] = None
                if kind in (
                    NodeKind.FUNCTION,
                    NodeKind.METHOD,
                    NodeKind.CONSTRUCTOR,
                    NodeKind.DESTRUCTOR,
                ):
                    params_ts = match_dict.get("params") or match_dict.get("parameters")
                    if params_ts:
                        if isinstance(params_ts, list):
                            params_ts = params_ts[0]
                        params_text = _node_text(params_ts, source)
                    else:
                        params_text = "()"
                    signature = f"{name_text}{params_text}"

                ast_node = ASTNode(
                    kind=kind,
                    name=name_text,
                    qualified_name=qname_str,
                    lang=lang_enum,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    start_byte=start_byte,
                    end_byte=end_byte,
                    code_hash=code_hash,
                    signature=signature,
                    source_text=src_text,
                    project_id=self._project_id,
                    valid_from=commit_hash,
                )
                nodes.append(ast_node)

        return nodes


# ---------------------------------------------------------------------------
# Helper functions (module-private)
# ---------------------------------------------------------------------------


def _node_text(node: Node, source: bytes) -> str:
    """Extract the UTF-8 text for a tree-sitter node."""
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace").strip()


def _build_qualified_name(file_path: str, name: str, lang: str) -> str:
    """Build a best-effort qualified name from file path and simple name.

    For PoC we use <module_path>.<name>.  A real implementation would
    walk the AST to collect the enclosing namespace / package chain.
    """
    module = Path(file_path).stem
    return f"{module}.{name}"
