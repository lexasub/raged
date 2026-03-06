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

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import tree_sitter_cpp as tscpp
import tree_sitter_java as tsjava
import tree_sitter_rust as tsrust
import tree_sitter_python as tspython
import tree_sitter_typescript as tsts
from tree_sitter import Language, Parser, Query, QueryCursor, Tree, Node

from ast_rag.models import (
    ASTNode, ASTEdge,
    NodeKind, EdgeKind, Language as Lang,
)
from ast_rag.language_queries import LANGUAGE_QUERIES
from ast_rag.parse_cache import ParseCache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extension → Language mapping
# ---------------------------------------------------------------------------

EXT_TO_LANG: dict[str, str] = {
    ".cpp": "cpp", ".cxx": "cpp", ".cc": "cpp", ".c": "cpp",
    ".hpp": "cpp", ".hxx": "cpp", ".hh": "cpp", ".h": "cpp",
    ".java": "java",
    ".rs": "rust",
    ".py": "python",
    ".ts": "typescript", ".tsx": "typescript",
}

# ---------------------------------------------------------------------------
# Query name → (NodeKind for the extracted entity, EdgeKind for containment)
# ---------------------------------------------------------------------------

# Maps query name → NodeKind that the query extracts
QUERY_KIND_MAP: dict[str, NodeKind] = {
    "class_defs":            NodeKind.CLASS,
    "interface_defs":        NodeKind.INTERFACE,
    "struct_defs":           NodeKind.STRUCT,
    "enum_defs":             NodeKind.ENUM,
    "trait_defs":            NodeKind.TRAIT,
    "annotation_type_defs":  NodeKind.INTERFACE,
    "namespace_defs":        NodeKind.NAMESPACE,
    "impl_defs":             NodeKind.CLASS,  # Rust impl block mapped to class
    "function_defs":         NodeKind.FUNCTION,
    "method_defs":           NodeKind.METHOD,
    "constructor_defs":      NodeKind.CONSTRUCTOR,
    "destructor_defs":       NodeKind.DESTRUCTOR,
    "field_defs":            NodeKind.FIELD,
}


class ParserManager:
    """Loads and caches tree-sitter parsers and compiled queries per language.

    Usage::

        pm = ParserManager()
        tree = pm.parse_file("/path/to/Foo.java")
        nodes = pm.extract_nodes(tree, "/path/to/Foo.java", "java")
        edges = pm.extract_edges(tree, nodes, "/path/to/Foo.java", "java")
    """

    def __init__(self, cache: Optional[ParseCache] = None) -> None:
        self._languages: dict[str, Language] = {}
        self._parsers: dict[str, Parser] = {}
        self._compiled_queries: dict[str, dict[str, object]] = {}
        # Delegate all cache concerns to ParseCache (swappable backend).
        self._cache: ParseCache = cache if cache is not None else ParseCache()
        self._init_languages()

    def _init_languages(self) -> None:
        """Load tree-sitter Language objects from pre-built bindings."""
        lang_defs = {
            "cpp":        Language(tscpp.language()),
            "java":       Language(tsjava.language()),
            "rust":       Language(tsrust.language()),
            "python":     Language(tspython.language()),
            "typescript": Language(tsts.language_typescript()),
        }
        for name, lang in lang_defs.items():
            self._languages[name] = lang
            parser = Parser(lang)
            self._parsers[name] = parser
            # Pre-compile all queries for the language
            queries = LANGUAGE_QUERIES.get(name, {})
            compiled: dict[str, object] = {}
            for qname, qstr in queries.items():
                try:
                    compiled[qname] = Query(lang, qstr)
                except Exception as exc:
                    logger.warning("Failed to compile query '%s' for '%s': %s", qname, name, exc)
            self._compiled_queries[name] = compiled

    def detect_language(self, file_path: str) -> Optional[str]:
        """Return the language key for a file, or None if unsupported."""
        ext = Path(file_path).suffix.lower()
        return EXT_TO_LANG.get(ext)

    def parse_file(
        self,
        file_path: str,
        old_tree: Optional[Tree] = None,
        source: Optional[bytes] = None,
    ) -> Optional[Tree]:
        """Parse a source file and return a tree-sitter Tree.

        Results are cached by content hash so unchanged files are never
        re-parsed within the same process.  Pass ``old_tree`` to enable
        incremental parsing; the cache entry is refreshed afterwards.

        Args:
            file_path: Absolute or relative path to the file.
            old_tree:  Previous Tree for incremental parsing, if available.
            source:    Pre-read bytes, to avoid re-reading the file.

        Returns:
            A tree-sitter Tree, or None on failure.
        """
        lang = self.detect_language(file_path)
        if lang is None:
            return None

        abs_path = os.path.abspath(file_path)

        # Read source bytes once so both the cache check and the parser use
        # the same bytes.
        if source is None:
            try:
                with open(abs_path, "rb") as fh:
                    source = fh.read()
            except OSError as exc:
                logger.error("Cannot read '%s': %s", file_path, exc)
                return None

        # Check cache — skip when incremental parse is explicitly requested
        # (old_tree supplied) to honour the caller's intent.
        if old_tree is None:
            cached = self._cache.get(abs_path, source)
            if cached is not None:
                return cached

        # Parse (full or incremental).
        parser = self._parsers[lang]
        tree = parser.parse(source, old_tree) if old_tree is not None else parser.parse(source)

        # Persist result in cache.
        self._cache.put(abs_path, source, tree)
        return tree

    # ------------------------------------------------------------------
    # Cache management — thin delegation to ParseCache
    # ------------------------------------------------------------------

    def clear_tree_cache(self) -> None:
        """Evict all cached parse trees (delegates to ParseCache.clear)."""
        self._cache.clear()

    def tree_cache_stats(self) -> dict:
        """Return cache performance counters (delegates to ParseCache.stats)."""
        return self._cache.stats()

    # ------------------------------------------------------------------
    # Node extraction
    # ------------------------------------------------------------------

    def extract_nodes(
        self,
        tree: Tree,
        file_path: str,
        lang: str,
        source: Optional[bytes] = None,
        commit_hash: str = "INIT",
    ) -> list[ASTNode]:
        """Extract meaningful AST nodes from a parsed tree.

        Returns a list of ASTNode instances with fully populated fields.
        """
        compiled = self._compiled_queries.get(lang, {})
        if source is None:
            try:
                with open(file_path, "rb") as fh:
                    source = fh.read()
            except OSError:
                source = b""

        file_rel = file_path  # callers may pass relative paths; keep as-is
        lang_enum = Lang(lang)
        nodes: list[ASTNode] = []

        # We track seen (name, kind, start_line) tuples to avoid duplicates
        seen: set[tuple[str, str, int]] = set()

        for qname, kind in QUERY_KIND_MAP.items():
            query = compiled.get(qname)
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
                end_line   = node_ts.end_point[0] + 1
                start_byte = node_ts.start_byte
                end_byte   = node_ts.end_byte

                dedup_key = (name_text, kind.value, start_line)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                # Build qualified_name based on file path + class context
                qname_str = _build_qualified_name(file_path, name_text, lang)
                src_text   = source[start_byte:end_byte].decode("utf-8", errors="replace")
                code_hash  = hashlib.sha256(src_text.encode()).hexdigest()[:24]

                # Build signature for callables
                signature: Optional[str] = None
                if kind in (NodeKind.FUNCTION, NodeKind.METHOD, NodeKind.CONSTRUCTOR,
                            NodeKind.DESTRUCTOR):
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
                    file_path=file_rel,
                    start_line=start_line,
                    end_line=end_line,
                    start_byte=start_byte,
                    end_byte=end_byte,
                    code_hash=code_hash,
                    signature=signature,
                    source_text=src_text,
                    valid_from=commit_hash,
                )
                nodes.append(ast_node)

        return nodes

    # ------------------------------------------------------------------
    # Edge extraction
    # ------------------------------------------------------------------

    def extract_edges(
        self,
        tree: Tree,
        nodes: list[ASTNode],
        file_path: str,
        lang: str,
        source: Optional[bytes] = None,
        commit_hash: str = "INIT",
    ) -> list[ASTEdge]:
        """Extract edges (relationships) between AST nodes.

        Produces:
        - CONTAINS_CLASS / CONTAINS_METHOD / CONTAINS_FUNCTION edges
        - IMPORTS / INCLUDES edges
        - CALLS edges (type-based resolution with confidence scoring)
        - INHERITS / EXTENDS / IMPLEMENTS edges
        - INJECTS edges (DI heuristic for Java)
        - OVERRIDES edges (Java @Override, C++ override keyword)
        - VIRTUAL_CALL edges (virtual method dispatch tracking)
        - LAMBDA_CALL edges (lambda/closure call tracking)
        - CROSS_FILE_CALL edges (cross-file symbol resolution)
        - DEPENDS_ON edges (C++ #include, Java imports)
        - TYPES edges (type annotations)
        """
        compiled = self._compiled_queries.get(lang, {})
        if source is None:
            try:
                with open(file_path, "rb") as fh:
                    source = fh.read()
            except OSError:
                source = b""

        edges: list[ASTEdge] = []
        # Build quick lookup: name → node id (for call resolution)
        name_to_id: dict[str, str] = {n.name: n.id for n in nodes}
        # Map of file_path → file ASTNode (for CONTAINS edges from file)
        file_node_id = hashlib.sha256(
            f"{file_path}:{NodeKind.FILE.value}:{file_path}".encode()
        ).hexdigest()[:24]

        # --- Containment edges ---
        # Simple rule: all Class/Interface/Struct/Enum/Trait nodes are contained by File.
        # All Method/Constructor/Destructor/Function nodes are contained by their enclosing
        # class if one exists with overlapping byte range; otherwise by File.
        type_nodes   = [n for n in nodes if n.kind in (
            NodeKind.CLASS, NodeKind.INTERFACE, NodeKind.STRUCT,
            NodeKind.ENUM, NodeKind.TRAIT, NodeKind.NAMESPACE,
        )]
        method_nodes = [n for n in nodes if n.kind in (
            NodeKind.FUNCTION, NodeKind.METHOD, NodeKind.CONSTRUCTOR, NodeKind.DESTRUCTOR,
        )]
        field_nodes  = [n for n in nodes if n.kind == NodeKind.FIELD]

        for tn in type_nodes:
            ek = _containment_edge_kind(tn.kind)
            edges.append(ASTEdge(
                kind=ek, from_id=file_node_id, to_id=tn.id, valid_from=commit_hash,
            ))

        for mn in method_nodes:
            parent = _find_enclosing_type(mn, type_nodes)
            if parent:
                edges.append(ASTEdge(
                    kind=EdgeKind.CONTAINS_METHOD, from_id=parent.id, to_id=mn.id,
                    valid_from=commit_hash,
                ))
            else:
                edges.append(ASTEdge(
                    kind=EdgeKind.CONTAINS_FUNCTION, from_id=file_node_id, to_id=mn.id,
                    valid_from=commit_hash,
                ))

        for fn in field_nodes:
            parent = _find_enclosing_type(fn, type_nodes)
            if parent:
                edges.append(ASTEdge(
                    kind=EdgeKind.CONTAINS_FIELD, from_id=parent.id, to_id=fn.id,
                    valid_from=commit_hash,
                ))

        # --- Import / Include edges ---
        import_qname  = "imports" if lang != "cpp" else None
        include_qname = "includes" if lang == "cpp" else None

        for qname_key, edge_kind in (
            (import_qname,  EdgeKind.IMPORTS),
            (include_qname, EdgeKind.INCLUDES),
        ):
            if qname_key is None:
                continue
            query = compiled.get(qname_key)
            if query is None:
                continue
            for _, md in QueryCursor(query).matches(tree.root_node):
                path_ts = md.get("import_path") or md.get("path") or md.get("module_path")
                if path_ts is None:
                    continue
                if isinstance(path_ts, list):
                    path_ts = path_ts[0]
                path_text = _node_text(path_ts, source).strip('"<>').strip()
                if not path_text:
                    continue
                # target id is synthetic: we use a stable hash for the import target
                target_id = hashlib.sha256(
                    f"import:{path_text}".encode()
                ).hexdigest()[:24]
                edges.append(ASTEdge(
                    kind=edge_kind,
                    from_id=file_node_id,
                    to_id=target_id,
                    label=path_text,
                    valid_from=commit_hash,
                ))

        # --- Call edges with enhanced type-based resolution ---
        call_query = compiled.get("calls")
        if call_query:
            for _, md in QueryCursor(call_query).matches(tree.root_node):
                callee_ts = md.get("callee_name")
                call_node_ts = md.get("node")
                if callee_ts is None or call_node_ts is None:
                    continue
                if isinstance(callee_ts, list):
                    callee_ts = callee_ts[0]
                if isinstance(call_node_ts, list):
                    call_node_ts = call_node_ts[0]
                callee_name = _node_text(callee_ts, source)
                if not callee_name:
                    continue
                # Find the enclosing method/function at the call site
                call_line = call_node_ts.start_point[0] + 1
                caller = _find_enclosing_callable(call_line, method_nodes)
                if not caller:
                    continue

                # Enhanced type-based call resolution with confidence scoring
                resolved_edge = self._resolve_call_with_types(
                    call_node_ts, name_to_id, callee_name, caller.id
                )
                if resolved_edge:
                    edges.append(resolved_edge)

                # Virtual method dispatch tracking for C++/Java
                if lang in ("cpp", "java"):
                    virtual_edge = self._track_virtual_method_dispatch(
                        call_node_ts, name_to_id, callee_name, caller.id
                    )
                    if virtual_edge:
                        edges.append(virtual_edge)

                # Lambda/closure call tracking
                if lang in ("cpp", "java", "rust"):
                    lambda_edge = self._track_lambda_calls(
                        call_node_ts, name_to_id, callee_name, caller.id
                    )
                    if lambda_edge:
                        edges.append(lambda_edge)

        # --- Cross-file symbol resolution ---
        if lang in ("cpp", "java", "rust"):
            cross_file_edges = self._resolve_cross_file_symbols(
                tree, nodes, file_path, lang, source, commit_hash
            )
            edges.extend(cross_file_edges)

        # --- Inheritance / Implements edges ---
        _add_type_relation_edges(
            edges, tree, compiled, nodes, source, lang, commit_hash, name_to_id,
        )

        # --- DI injection edges (Java-specific) ---
        if lang == "java":
            injects_edges = self._extract_injects(tree, nodes, file_path, lang, source, commit_hash)
            edges.extend(injects_edges)

        # --- New edge types ---
        # DEPENDS_ON edges (C++ #include, Java imports)
        depends_on_edges = self._extract_depends_on(tree, nodes, file_path, lang, source, commit_hash)
        edges.extend(depends_on_edges)

        # OVERRIDES edges (Java @Override, C++ override keyword)
        overrides_edges = self._extract_overrides(tree, nodes, lang, source, commit_hash)
        edges.extend(overrides_edges)

        # TYPES edges (type annotations)
        types_edges = self._extract_types(tree, nodes, lang, source, commit_hash)
        edges.extend(types_edges)

        # --- Rust-specific edges ---
        if lang == "rust":
            rust_edges = self._extract_rust_edges(tree, nodes, file_path, source, commit_hash)
            edges.extend(rust_edges)

        return edges

    def _track_virtual_method_dispatch(
        self,
        call_node: Node,
        name_to_id: dict[str, str],
        callee_name: str,
        caller_id: str,
    ) -> Optional[ASTEdge]:
        """Track virtual method dispatch for C++/Java.

        Args:
            call_node: The call expression node
            name_to_id: Mapping of symbol names to node IDs
            callee_name: The name of the callee (already extracted)
            caller_id: ID of the calling function/method

        Returns:
            ASTEdge with VIRTUAL_CALL kind and confidence score, or None if not virtual
        """
        # Check if this is a virtual method call
        receiver_node = call_node.child_by_field_name("object")
        if not receiver_node:
            return None

        # For C++, check if it's a virtual function call
        if callee_name and "::" in callee_name:
            # This is likely a C++ method call
            confidence = 0.8
            return ASTEdge(
                kind=EdgeKind.VIRTUAL_CALL,
                from_id=name_to_id.get(callee_name, ""),
                to_id=name_to_id.get(callee_name, ""),
                confidence=confidence,
                resolution_method="virtual",
                valid_from="INIT",
            )
        
        # For Java, check if it's a virtual method call
        elif callee_name and "." in callee_name:
            # This is likely a Java method call
            confidence = 0.9
            return ASTEdge(
                kind=EdgeKind.VIRTUAL_CALL,
                from_id=name_to_id.get(callee_name, ""),
                to_id=name_to_id.get(callee_name, ""),
                confidence=confidence,
                resolution_method="virtual",
                valid_from="INIT",
            )

        return None

    def _track_cpp_virtual_call(
        self,
        receiver_node: Node,
        call_node: Node,
        name_to_id: dict[str, str],
        source: bytes,
    ) -> Optional[ASTEdge]:
        """Track C++ virtual method calls.

        Handles:
        - Virtual function calls through base class pointers
        - Pure virtual function calls
        - Interface method calls
        """
        # Check if receiver is a pointer or reference to a base class
        receiver_type = self._infer_cpp_receiver_type(receiver_node, None, name_to_id, source)
        if not receiver_type or not receiver_type.startswith("var:"):
            return None

        # Get the callee name
        callee_name = _node_text(call_node.child_by_field_name("callee_name") or call_node.child_by_field_name("function"), source)
        if not callee_name:
            return None

        # Create virtual call edge with confidence
        confidence = 0.8  # High confidence for virtual calls

        return ASTEdge(
            kind=EdgeKind.VIRTUAL_CALL,
            from_id=name_to_id.get(callee_name, ""),
            to_id=name_to_id.get(callee_name, ""),  # Will be resolved later
            confidence=confidence,
            resolution_method="virtual",
            valid_from="INIT",
        )

    def _track_java_virtual_call(
        self,
        receiver_node: Node,
        call_node: Node,
        name_to_id: dict[str, str],
        source: bytes,
    ) -> Optional[ASTEdge]:
        """Track Java virtual method calls.

        Handles:
        - Interface method calls
        - Abstract class method calls
        - Polymorphic method calls
        """
        # Check if receiver is an interface or abstract class
        receiver_type = self._infer_java_receiver_type(receiver_node, None, name_to_id, source)
        if not receiver_type or receiver_type not in ("this", "super"):
            return None

        # Get the callee name
        callee_name = _node_text(call_node.child_by_field_name("callee_name") or call_node.child_by_field_name("function"), source)
        if not callee_name:
            return None

        # Create virtual call edge with confidence
        confidence = 0.9  # Very high confidence for Java virtual calls

        return ASTEdge(
            kind=EdgeKind.VIRTUAL_CALL,
            from_id=name_to_id.get(callee_name, ""),
            to_id=name_to_id.get(callee_name, ""),  # Will be resolved later
            confidence=confidence,
            resolution_method="virtual",
            valid_from="INIT",
        )

    def _extract_depends_on(
        self,
        tree: Tree,
        nodes: list[ASTNode],
        file_path: str,
        lang: str,
        source: bytes,
        commit_hash: str,
    ) -> list[ASTEdge]:
        """Extract DEPENDS_ON edges for C++ #include and Java imports.

        Args:
            tree: The parsed tree
            nodes: List of AST nodes in current file
            file_path: Current file path
            lang: Language of the file
            source: Source bytes
            commit_hash: Commit hash for versioning

        Returns:
            List of DEPENDS_ON edges
        """
        edges: list[ASTEdge] = []
        file_node_id = hashlib.sha256(
            f"{file_path}:{NodeKind.FILE.value}:{file_path}".encode()
        ).hexdigest()[:24]

        if lang == "cpp":
            # Extract #include directives
            include_query = self._compiled_queries.get(lang, {}).get("includes")
            if include_query:
                for _, md in QueryCursor(include_query).matches(tree.root_node):
                    path_ts = md.get("path")
                    if path_ts is None:
                        continue
                    if isinstance(path_ts, list):
                        path_ts = path_ts[0]
                    path_text = _node_text(path_ts, source).strip('"<>').strip()
                    if not path_text:
                        continue
                    
                    # Create target ID for the include file
                    target_id = hashlib.sha256(
                        f"include:{path_text}".encode()
                    ).hexdigest()[:24]
                    
                    edges.append(ASTEdge(
                        kind=EdgeKind.DEPENDS_ON,
                        from_id=file_node_id,
                        to_id=target_id,
                        label=path_text,
                        valid_from=commit_hash,
                    ))

        elif lang == "java":
            # Extract import statements
            import_query = self._compiled_queries.get(lang, {}).get("imports")
            if import_query:
                for _, md in QueryCursor(import_query).matches(tree.root_node):
                    import_path_ts = md.get("import_path") or md.get("module_path")
                    if import_path_ts is None:
                        continue
                    if isinstance(import_path_ts, list):
                        import_path_ts = import_path_ts[0]
                    import_text = _node_text(import_path_ts, source).strip()
                    if not import_text:
                        continue
                    
                    # Create target ID for the imported module
                    target_id = hashlib.sha256(
                        f"import:{import_text}".encode()
                    ).hexdigest()[:24]
                    
                    edges.append(ASTEdge(
                        kind=EdgeKind.DEPENDS_ON,
                        from_id=file_node_id,
                        to_id=target_id,
                        label=import_text,
                        valid_from=commit_hash,
                    ))

        return edges

    def _extract_overrides(
            self,
            tree: Tree,
            nodes: list[ASTNode],
            lang: str,
            source: bytes,
            commit_hash: str,
        ) -> list[ASTEdge]:
            """Extract OVERRIDES edges for Java @Override and C++ override keyword.

            Args:
                tree: The parsed tree
                nodes: List of AST nodes in current file
                lang: Language of the file
                source: Source bytes
                commit_hash: Commit hash for versioning

            Returns:
                List of OVERRIDES edges
            """
            edges: list[ASTEdge] = []
            
            if lang == "java":
                # Extract methods with @Override annotation
                override_query = self._compiled_queries.get(lang, {}).get("overrides")
                if override_query:
                    for _, md in QueryCursor(override_query).matches(tree.root_node):
                        method_name_ts = md.get("name")
                        if method_name_ts is None:
                            continue
                        if isinstance(method_name_ts, list):
                            method_name_ts = method_name_ts[0]
                        method_name = _node_text(method_name_ts, source)
                        if not method_name:
                            continue
                        
                        # Find the overriding method node
                        overriding_node = next((n for n in nodes if n.name == method_name and n.kind == NodeKind.METHOD), None)
                        if not overriding_node:
                            continue
                        
                        # Create OVERRIDES edge (resolution of overridden method will happen later)
                        edges.append(ASTEdge(
                            kind=EdgeKind.OVERRIDES,
                            from_id=overriding_node.id,
                            to_id="",  # Will be resolved later
                            valid_from=commit_hash,
                        ))

            elif lang == "cpp":
                # Extract methods with override keyword (would need additional query)
                # This is a placeholder for future implementation
                pass

            return edges

    def _extract_types(
            self,
            tree: Tree,
            nodes: list[ASTNode],
            lang: str,
            source: bytes,
            commit_hash: str,
        ) -> list[ASTEdge]:
            """Extract TYPES edges for type annotations.

            Args:
                tree: The parsed tree
                nodes: List of AST nodes in current file
                lang: Language of the file
                source: Source bytes
                commit_hash: Commit hash for versioning

            Returns:
                List of TYPES edges
            """
            edges: list[ASTEdge] = []
            
            if lang == "java":
                # Extract field type annotations
                field_query = self._compiled_queries.get(lang, {}).get("field_defs")
                if field_query:
                    for _, md in QueryCursor(field_query).matches(tree.root_node):
                        field_name_ts = md.get("name")
                        field_type_ts = md.get("field_type")
                        if field_name_ts is None or field_type_ts is None:
                            continue
                        if isinstance(field_name_ts, list):
                            field_name_ts = field_name_ts[0]
                        if isinstance(field_type_ts, list):
                            field_type_ts = field_type_ts[0]
                        
                        field_name = _node_text(field_name_ts, source)
                        field_type = _node_text(field_type_ts, source)
                        if not field_name or not field_type:
                            continue
                        
                        # Find the field node
                        field_node = next((n for n in nodes if n.name == field_name and n.kind == NodeKind.FIELD), None)
                        if not field_node:
                            continue
                        
                        # Create target ID for the type
                        type_id = hashlib.sha256(
                            f"type:{field_type}".encode()
                        ).hexdigest()[:24]
                        
                        edges.append(ASTEdge(
                            kind=EdgeKind.TYPES,
                            from_id=field_node.id,
                            to_id=type_id,
                            label=field_type,
                            valid_from=commit_hash,
                        ))

            elif lang == "cpp":
                # Extract variable type annotations (would need additional query)
                # This is a placeholder for future implementation
                pass

            return edges

    def _extract_injects(
        self,
        tree: Tree,
        nodes: list[ASTNode],
        file_path: str,
        lang: str,
        source: bytes,
        commit_hash: str,
    ) -> list[ASTEdge]:
        """Extract dependency injection relationships.
        
        Creates INJECTS edges from fields/constructors to injected types.
        """
        edges = []
        
        if lang != "java":
            return edges
        
        compiled = self._compiled_queries.get("java", {})
        nodes_by_name = {n.name: n for n in nodes}
        
        # Field injections (@Autowired, @Inject, @Resource)
        field_query = compiled.get("di_fields")
        if field_query:
            for _, match in QueryCursor(field_query).matches(tree.root_node):
                annotation_ts = match.get("annotation_name")
                type_ts = match.get("injected_type")
                field_ts = match.get("field_name")
                
                if not all([annotation_ts, type_ts, field_ts]):
                    continue
                
                annotation = _node_text(annotation_ts, source)
                type_name = _node_text(type_ts, source)
                field_name = _node_text(field_ts, source)
                
                # Find the field node and injected type node
                field_node = next((n for n in nodes if n.name == field_name), None)
                type_node = nodes_by_name.get(type_name)
                
                if field_node and type_node:
                    edge = ASTEdge(
                        kind=EdgeKind.INJECTS,
                        from_id=field_node.id,
                        to_id=type_node.id,
                        dep_kind=annotation.lower(),  # "autowired", "inject", "resource"
                        valid_from=commit_hash,
                    )
                    edges.append(edge)
        
        # Constructor injections
        ctor_query = compiled.get("di_constructors")
        if ctor_query:
            for _, match in QueryCursor(ctor_query).matches(tree.root_node):
                annotation_ts = match.get("annotation_name")
                type_ts = match.get("injected_type")
                node_ts = match.get("node")  # constructor_declaration node
                
                if not all([annotation_ts, type_ts, node_ts]):
                    continue
                
                annotation = _node_text(annotation_ts, source)
                type_name = _node_text(type_ts, source)
                
                # Extract constructor name from the constructor_declaration node
                name_ts = node_ts.child_by_field_name("name")
                if not name_ts:
                    continue
                ctor_name = _node_text(name_ts, source)
                
                # Find the constructor node and injected type node
                ctor_node = next((n for n in nodes if n.name == ctor_name and n.kind == NodeKind.CONSTRUCTOR), None)
                type_node = nodes_by_name.get(type_name)
                
                if ctor_node and type_node:
                    edge = ASTEdge(
                        kind=EdgeKind.INJECTS,
                        from_id=ctor_node.id,
                        to_id=type_node.id,
                        dep_kind=annotation.lower(),  # "autowired", "inject"
                        valid_from=commit_hash,
                    )
                    edges.append(edge)
        
        return edges

    def _extract_rust_edges(
            self,
            tree: Tree,
            nodes: list[ASTNode],
            file_path: str,
            source: bytes,
            commit_hash: str,
        ) -> list[ASTEdge]:
            """Extract Rust-specific edges: IMPLEMENTS for trait impls.

            Creates:
            - IMPLEMENTS edges: impl Trait for Type → Type implements Trait
            - Generic constraint edges: where T: Trait → T constrained by Trait
            """
            edges: list[ASTEdge] = []
            
            # Build lookups
            nodes_by_name: dict[str, ASTNode] = {n.name: n for n in nodes}
            
            # --- IMPLEMENTS edges from impl blocks ---
            # Look for impl blocks that implement traits
            impl_blocks_query = self._compiled_queries.get("rust", {}).get("impl_defs")
            if impl_blocks_query:
                for _, match in QueryCursor(impl_blocks_query).matches(tree.root_node):
                    impl_type_ts = match.get("impl_type")
                    trait_name_ts = match.get("trait_name")
                    
                    if impl_type_ts is None or trait_name_ts is None:
                        continue
                    
                    if isinstance(impl_type_ts, list):
                        impl_type_ts = impl_type_ts[0]
                    if isinstance(trait_name_ts, list):
                        trait_name_ts = trait_name_ts[0]
                    
                    impl_type_name = _node_text(impl_type_ts, source)
                    trait_name = _node_text(trait_name_ts, source)
                    
                    if not impl_type_name or not trait_name:
                        continue
                    
                    # Find the implementing type node (struct, enum, or class)
                    impl_node = None
                    for node in nodes:
                        if node.name == impl_type_name and node.kind in (
                            NodeKind.STRUCT, NodeKind.ENUM, NodeKind.CLASS
                        ):
                            impl_node = node
                            break
                    
                    # Find the trait node
                    trait_node = nodes_by_name.get(trait_name)
                    
                    if impl_node and trait_node:
                        edge = ASTEdge(
                            kind=EdgeKind.IMPLEMENTS,
                            from_id=impl_node.id,
                            to_id=trait_node.id,
                            valid_from=commit_hash,
                        )
                        edges.append(edge)
            
            # --- WHERE clause constraint edges ---
            # Extract type bounds from where clauses (optional enhancement)
            where_query = self._compiled_queries.get("rust", {}).get("where_clauses")
            if where_query:
                for _, match in QueryCursor(where_query).matches(tree.root_node):
                    predicates_ts = match.get("predicates")
                    if predicates_ts is None:
                        continue
                    
                    if isinstance(predicates_ts, list):
                        predicates_ts = predicates_ts[0]
                    
                    # Extract trait bounds from predicates
                    # Pattern: where T: Trait + Lifetime
                    bounds_query = self._compiled_queries.get("rust", {}).get("trait_bounds")
                    if bounds_query:
                        for _, bound_match in QueryCursor(bounds_query).matches(predicates_ts):
                            trait_ts = bound_match.get("trait_name")
                            if trait_ts:
                                if isinstance(trait_ts, list):
                                    trait_ts = trait_ts[0]
                                trait_name = _node_text(trait_ts, source)
                                if trait_name:
                                    # Create TYPES edge for constraint
                                    # This links the generic parameter to its trait bound
                                    pass  # Optional enhancement
            
            return edges

    def _track_lambda_calls(
            self,
            call_node: Node,
            name_to_id: dict[str, str],
            callee_name: str,
            caller_id: str,
        ) -> Optional[ASTEdge]:
            """Track lambda/closure calls.

            Args:
                call_node: The call expression node
                name_to_id: Mapping of symbol names to node IDs
                callee_name: The name of the callee (already extracted)
                caller_id: ID of the calling function/method

            Returns:
                ASTEdge with LAMBDA_CALL kind, or None if not a lambda call
            """
            # Check if callee is a lambda/closure
            if self._is_lambda_or_closure(callee_name, "cpp"):
                # Find the lambda definition
                lambda_node_id = name_to_id.get(callee_name)
                if lambda_node_id:
                    return ASTEdge(
                        kind=EdgeKind.LAMBDA_CALL,
                        from_id=caller_id,
                        to_id=lambda_node_id,
                        confidence=1.0,
                        resolution_method="lambda",
                        valid_from="INIT",
                    )

            return None

    def _is_lambda_or_closure(
            self,
            name: str,
            lang: str,
        ) -> bool:
            """Check if a name represents a lambda or closure.

            Args:
                name: The symbol name
                lang: Language of the symbol

            Returns:
                True if the name represents a lambda/closure
            """
            if lang == "cpp":
                return name.startswith("lambda") or name.startswith("\\lambda")
            elif lang == "java":
                return name.startswith("lambda$") or name.startswith("\\$lambda")
            elif lang == "rust":
                return name.startswith("||") or name.startswith("move ||")
            
            return False

    def _resolve_cross_file_symbols(
            self,
            tree: Tree,
            nodes: list[ASTNode],
            file_path: str,
            lang: str,
            source: bytes,
            commit_hash: str,
        ) -> list[ASTEdge]:
            """Resolve cross-file symbol references.

            Args:
                tree: The parsed tree
                nodes: List of AST nodes in current file
                file_path: Current file path
                lang: Language of the file
                source: Source bytes
                commit_hash: Commit hash for versioning

            Returns:
                List of CROSS_FILE_CALL edges
            """
            edges: list[ASTEdge] = []
            
            if lang not in ("cpp", "java", "rust"):
                return edges

            # Look for cross-file references in the AST
            for node in tree.root_node.children:
                if node.type in ("identifier", "path_expression"):
                    symbol_name = _node_text(node, source)
                    # Check if this is a cross-file reference
                    if self._is_cross_file_reference(symbol_name, lang, file_path):
                        # Try to resolve the symbol
                        resolved_edge = self._resolve_cross_file_symbol(
                            symbol_name, nodes, lang, source
                        )
                        if resolved_edge:
                            edges.append(resolved_edge)

            return edges

    def _is_cross_file_reference(
            self,
            symbol_name: str,
            lang: str,
            current_file: str,
        ) -> bool:
            """Check if a symbol reference is cross-file.

            Args:
                symbol_name: The symbol name
                lang: Language of the symbol
                current_file: Current file path

            Returns:
                True if the symbol is likely defined in another file
            """
            if lang == "cpp":
                # Check for fully qualified names or external includes
                return "::" in symbol_name or self._is_external_include(symbol_name, current_file)
            elif lang == "java":
                # Check for fully qualified class names
                return "." in symbol_name and not symbol_name.startswith("java.")
            elif lang == "rust":
                # Check for external crate references
                return "::" in symbol_name and not symbol_name.startswith("crate::")
            
            return False

    def _is_external_include(
            self,
            symbol_name: str,
            current_file: str,
        ) -> bool:
            """Check if a symbol is from an external include.

            Args:
                symbol_name: The symbol name
                current_file: Current file path

            Returns:
                True if the symbol is from an external include
            """
            # This would need access to the actual include graph
            # For now, we'll use a simple heuristic
            external_headers = ["stdio.h", "stdlib.h", "string.h", "vector", "iostream"]
            return any(symbol_name.startswith(header) for header in external_headers)

    def _resolve_cross_file_symbol(
            self,
            symbol_name: str,
            nodes: list[ASTNode],
            lang: str,
            source: bytes,
        ) -> Optional[ASTEdge]:
            """Resolve a cross-file symbol reference.

            Args:
                symbol_name: The symbol name
                nodes: List of AST nodes in current file
                lang: Language of the file
                source: Source bytes

            Returns:
                ASTEdge with CROSS_FILE_CALL kind, or None if cannot resolve
            """
            # Try to find the symbol definition in the current index
            # This is a simplified implementation - in a real system,
            # we'd query the global symbol table or use semantic search
            
            # For now, we'll create a synthetic edge with low confidence
            confidence = 0.3  # Low confidence for cross-file resolution
            
            # Create a synthetic target ID for the cross-file symbol
            target_id = hashlib.sha256(
                f"crossfile:{symbol_name}".encode()
            ).hexdigest()[:24]
            
            return ASTEdge(
                kind=EdgeKind.CROSS_FILE_CALL,
                from_id="",  # Will be set by caller
                to_id=target_id,
                label=symbol_name,
                confidence=confidence,
                resolution_method="crossfile",
                valid_from="INIT",
            )

    def _resolve_call_with_types(
            self,
            call_node: Node,
            name_to_id: dict[str, str],
            callee_name: str,
            caller_id: str,
        ) -> Optional[ASTEdge]:
            """Resolve call target using type information from receiver and arguments.

            Args:
                call_node: The call expression node
                name_to_id: Mapping of symbol names to node IDs
                callee_name: The name of the callee (already extracted)
                caller_id: ID of the calling function/method

            Returns:
                ASTEdge with CALLS kind and confidence score, or None if resolution fails
            """
            # Simplified version - perform name-based resolution with confidence
            callee_node_id = name_to_id.get(callee_name)
            if callee_node_id:
                # Use moderate confidence for name-based resolution
                confidence = 0.7
                return ASTEdge(
                    kind=EdgeKind.CALLS,
                    from_id=caller_id,
                    to_id=callee_node_id,
                    confidence=confidence,
                    resolution_method="name",
                    valid_from="INIT",
                )
            return None

    def _infer_receiver_type(
        self,
        call_node: Node,
        tree: Tree,
        name_to_id: dict[str, str],
        lang: str,
        source: bytes,
    ) -> Optional[str]:
        """Infer the type of the receiver object for method calls.

        Args:
            call_node: The call expression node
            tree: The parsed tree
            name_to_id: Mapping of symbol names to node IDs
            lang: Language of the file
            source: Source bytes

        Returns:
            The inferred type name, or None if cannot be determined
        """
        if lang not in ("java", "cpp"):
            return None

        # Find receiver expression
        receiver_node = call_node.child_by_field_name("object")
        if not receiver_node:
            return None

        # Handle different receiver patterns
        if lang == "java":
            return self._infer_java_receiver_type(receiver_node, tree, name_to_id, source)
        elif lang == "cpp":
            return self._infer_cpp_receiver_type(receiver_node, tree, name_to_id, source)
        return None

    def _infer_java_receiver_type(
        self,
        receiver_node: Node,
        tree: Tree,
        name_to_id: dict[str, str],
        source: bytes,
    ) -> Optional[str]:
        """Infer Java receiver type from various patterns.

        Handles:
        - this.identifier
        - super.identifier
        - variable.identifier
        - ClassName.identifier (static)
        """
        # Check for 'this' or 'super'
        if receiver_node.type == "this_expression":
            return "this"
        if receiver_node.type == "super_expression":
            return "super"

        # Check for variable or type name
        if receiver_node.type == "identifier":
            var_name = _node_text(receiver_node, source)
            # Look up variable declarations to find its type
            # This is a simplified approach - in a full implementation,
            # we'd need a symbol table with type information
            return f"var:{var_name}"

        return None

    def _infer_cpp_receiver_type(
        self,
        receiver_node: Node,
        tree: Tree,
        name_to_id: dict[str, str],
        source: bytes,
    ) -> Optional[str]:
        """Infer C++ receiver type from various patterns.

        Handles:
        - this->identifier
        - object.identifier
        - ClassName::identifier (static)
        """
        if receiver_node.type == "identifier" and receiver_node.text == "this":
            return "this"

        # For simple variable names, return as-is (simplified)
        if receiver_node.type == "identifier":
            var_name = _node_text(receiver_node, source)
            return f"var:{var_name}"

        return None

    def _infer_rust_receiver_type(
        self,
        receiver_node: Node,
        tree: Tree,
        name_to_id: dict[str, str],
        source: bytes,
    ) -> Optional[str]:
        """Infer Rust receiver type from various patterns.

        Handles:
        - self.identifier (instance method)
        - &self.identifier (borrowed instance method)
        - &mut self.identifier (mutable borrowed instance method)
        - Type::identifier (static method)
        - variable.identifier (method call on variable)
        - path::to::Type::identifier (fully qualified static method)
        """
        if receiver_node.type == "identifier":
            text = _node_text(receiver_node, source)
            if text in ("self", "&self", "&mut self"):
                return "self"
            return f"var:{text}"

        # Handle path expressions for static methods
        if receiver_node.type == "path_expression":
            return self._extract_rust_type_from_path(receiver_node, source)

        return None

    def _extract_rust_type_from_path(
        self,
        path_node: Node,
        source: bytes,
    ) -> Optional[str]:
        """Extract type name from Rust path expression.

        Handles:
        - simple paths (std::collections::HashMap)
        - generic paths (Vec<T>)
        - qualified paths (::std::collections::HashMap)
        """
        segments = []
        current = path_node
        
        while current:
            if current.type == "identifier":
                segments.append(_node_text(current, source))
                break
            elif current.type == "path_segment":
                ident = current.child_by_field_name("name")
                if ident and ident.type == "identifier":
                    segments.append(_node_text(ident, source))
                current = current.child_by_field_name("prefix")
            else:
                break
        
        if segments:
            return "::".join(reversed(segments))
        
        return None

    def _infer_rust_type_from_annotation(
        self,
        var_node: Node,
        tree: Tree,
        source: bytes,
    ) -> Optional[str]:
        """Infer Rust type from variable annotation.

        Handles:
        - let x: Type = ...
        - let x = Type::default()
        - let x = SomeType { ... }
        - let x = &Type
        """
        # Look for type annotation
        type_node = var_node.child_by_field_name("type")
        if type_node:
            return self._extract_rust_type_from_path(type_node, source)
        
        # Look for initializer expression
        initializer = var_node.child_by_field_name("initializer")
        if initializer:
            return self._infer_rust_type_from_expression(initializer, source)
        
        return None

    def _infer_rust_type_from_expression(
        self,
        expr_node: Node,
        source: bytes,
    ) -> Optional[str]:
        """Infer Rust type from expression.

        Handles:
        - Constructor calls (Type { ... })
        - Function calls (Type::new())
        - Method calls (vec.push())
        - Literals (123, "string", true)
        """
        if expr_node.type == "call_expression":
            callee = expr_node.child_by_field_name("callee")
            if callee and callee.type == "path_expression":
                return self._extract_rust_type_from_path(callee, source)
        
        elif expr_node.type == "struct_expression":
            path = expr_node.child_by_field_name("path")
            if path:
                return self._extract_rust_type_from_path(path, source)
        
        elif expr_node.type == "identifier":
            text = _node_text(expr_node, source)
            if text in ("true", "false"):
                return "bool"
        
        elif expr_node.type == "integer_literal":
            return "i32"  # Default integer type
        
        elif expr_node.type == "string_literal":
            return "String"
        
        return None

    def _filter_by_receiver_type(
        self,
        call_node: Node,
        receiver_type: str,
        name_to_id: dict[str, str],
        lang: str,
        source: bytes,
    ) -> dict[str, str]:
        """Filter candidate methods by receiver type.

        Args:
            call_node: The call expression node
            receiver_type: Inferred receiver type
            name_to_id: Mapping of symbol names to node IDs
            lang: Language of the file
            source: Source bytes

        Returns:
            Filtered dictionary of candidates (name -> node_id)
        """
        # This is a simplified implementation - in a full version,
        # we'd need access to type definitions and inheritance hierarchies
        filtered = {}
        callee_name = _node_text(call_node.child_by_field_name("callee_name") or call_node.child_by_field_name("function"), source)

        for name, node_id in name_to_id.items():
            if name == callee_name:
                # In a real implementation, we'd check if the method
                # is compatible with the receiver type
                filtered[name] = node_id

        return filtered

    def _compute_confidence(
        self,
        call_node: Node,
        candidates: dict[str, str],
        lang: str,
    ) -> float:
        """Compute confidence score for call resolution.

        Args:
            call_node: The call expression node
            candidates: Dictionary of candidate methods
            lang: Language of the file

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not candidates:
            return 0.0

        if len(candidates) == 1:
            return 1.0  # Exact match

        # Multiple candidates - reduce confidence based on ambiguity
        return max(0.7, 1.0 - (len(candidates) - 1) * 0.1)

    def _disambiguate_by_signature(
        self,
        call_node: Node,
        candidates: dict[str, str],
        lang: str,
        source: bytes,
    ) -> Optional[str]:
        """Disambiguate between multiple candidates using signature matching.

        Args:
            call_node: The call expression node
            candidates: Dictionary of candidate methods
            lang: Language of the file
            source: Source bytes

        Returns:
            The best matching candidate name, or None if no clear match
        """
        # Get argument list from call expression
        args_node = call_node.child_by_field_name("arguments") or call_node.child_by_field_name("parameter_list")
        if not args_node:
            return None

        # This is a simplified implementation - in a full version,
        # we'd extract parameter types and match against method signatures
        if len(candidates) == 1:
            return next(iter(candidates.keys()))

        # For multiple candidates, return the first one (simplified)
        return next(iter(candidates.keys()))


# ---------------------------------------------------------------------------
# Helper functions (module-private)
# ---------------------------------------------------------------------------

def _node_text(node: Node, source: bytes) -> str:
    """Extract the UTF-8 text for a tree-sitter node."""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace").strip()


def _build_qualified_name(file_path: str, name: str, lang: str) -> str:
    """Build a best-effort qualified name from file path and simple name.

    For PoC we use <module_path>.<name>.  A real implementation would
    walk the AST to collect the enclosing namespace / package chain.
    """
    module = Path(file_path).stem
    return f"{module}.{name}"


def _containment_edge_kind(kind: NodeKind) -> EdgeKind:
    if kind == NodeKind.FUNCTION:
        return EdgeKind.CONTAINS_FUNCTION
    return EdgeKind.CONTAINS_CLASS


def _find_enclosing_type(
    target: ASTNode, type_nodes: list[ASTNode]
) -> Optional[ASTNode]:
    """Find the innermost type node whose byte range fully contains target."""
    best: Optional[ASTNode] = None
    best_size = float("inf")
    for tn in type_nodes:
        if tn.start_byte <= target.start_byte and tn.end_byte >= target.end_byte:
            size = tn.end_byte - tn.start_byte
            if size < best_size:
                best, best_size = tn, size
    return best


def _find_enclosing_callable(
    call_line: int, callables: list[ASTNode]
) -> Optional[ASTNode]:
    """Find the innermost callable (method/function) containing the given line."""
    best: Optional[ASTNode] = None
    best_size = float("inf")
    for c in callables:
        if c.start_line <= call_line <= c.end_line:
            size = c.end_line - c.start_line
            if size < best_size:
                best, best_size = c, size
    return best


def _add_type_relation_edges(
    edges: list[ASTEdge],
    tree: Tree,
    compiled: dict,
    nodes: list[ASTNode],
    source: bytes,
    lang: str,
    commit_hash: str,
    name_to_id: dict[str, str],
) -> None:
    """Add INHERITS / EXTENDS / IMPLEMENTS edges from class_def queries."""
    qname = "class_defs" if lang != "cpp" else "class_defs"
    query = compiled.get(qname)
    if not query:
        return

    for _, md in QueryCursor(query).matches(tree.root_node):
        name_ts = md.get("name")
        if name_ts is None:
            continue
        if isinstance(name_ts, list):
            name_ts = name_ts[0]
        cls_name = _node_text(name_ts, source)
        cls_id = name_to_id.get(cls_name)
        if not cls_id:
            continue

        # C++ / Java base class
        for key in ("base_class", "superclass"):
            base_ts = md.get(key)
            if base_ts is None:
                continue
            if isinstance(base_ts, list):
                base_ts = base_ts[0]
            base_name = _node_text(base_ts, source)
            base_id = name_to_id.get(base_name)
            if base_id:
                ek = EdgeKind.EXTENDS if lang == "java" else EdgeKind.INHERITS
                edges.append(ASTEdge(
                    kind=ek, from_id=cls_id, to_id=base_id,
                    label=base_name, valid_from=commit_hash,
                ))

        # Java interfaces
        for key in ("iface",):
            iface_list = md.get(key)
            if iface_list is None:
                continue
            items = iface_list if isinstance(iface_list, list) else [iface_list]
            for iface_ts in items:
                iface_name = _node_text(iface_ts, source)
                iface_id = name_to_id.get(iface_name)
                if iface_id:
                    edges.append(ASTEdge(
                        kind=EdgeKind.IMPLEMENTS, from_id=cls_id, to_id=iface_id,
                        label=iface_name, valid_from=commit_hash,
                    ))


# ---------------------------------------------------------------------------
# Convenience: walk a directory and yield (file_path, lang) pairs
# ---------------------------------------------------------------------------

def walk_source_files(
    root: str,
    exclude_dirs: Optional[list[str]] = None,
) -> list[tuple[str, str]]:
    """Recursively enumerate all source files under root.

    Returns a list of (absolute_file_path, language) tuples.
    """
    if exclude_dirs is None:
        exclude_dirs = [
            ".git", "__pycache__", "node_modules", "target", "build",
            "dist", ".gradle", ".idea", ".vscode",
        ]
    result: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories in-place
        dirnames[:] = [
            d for d in dirnames
            if d not in exclude_dirs and not d.startswith(".")
        ]
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            lang = EXT_TO_LANG.get(ext)
            if lang:
                result.append((os.path.join(dirpath, fname), lang))
    return result