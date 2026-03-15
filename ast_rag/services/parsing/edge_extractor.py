"""
edge_extractor.py - AST edge extraction from parsed trees.

Extracts edges (relationships) between AST nodes including:
- CONTAINS_* edges (containment)
- IMPORTS / INCLUDES edges
- CALLS edges
- INHERITS / EXTENDS / IMPLEMENTS edges
- INJECTS edges (DI heuristic for Java)
- OVERRIDES edges
- VIRTUAL_CALL edges
- LAMBDA_CALL edges
- CROSS_FILE_CALL edges
- DEPENDS_ON edges
- TYPES edges
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from tree_sitter import Node, QueryCursor, Tree

from ast_rag.models import ASTEdge, ASTNode, EdgeKind, NodeKind

logger = logging.getLogger(__name__)


class EdgeExtractor:
    """Extracts edges from parsed trees and existing nodes."""

    def __init__(self, project_id: str = "default") -> None:
        self._project_id = project_id

    def extract_edges(
        self,
        tree: Tree,
        nodes: list[ASTNode],
        file_path: str,
        lang: str,
        compiled_queries: dict[str, object],
        source: Optional[bytes] = None,
        commit_hash: str = "INIT",
    ) -> list[ASTEdge]:
        """Extract edges (relationships) between AST nodes."""
        if source is None:
            try:
                with open(file_path, "rb") as fh:
                    source = fh.read()
            except OSError:
                source = b""

        edges: list[ASTEdge] = []
        name_to_id: dict[str, str] = {n.name: n.id for n in nodes}

        file_node_id = hashlib.sha256(
            f"{file_path}:{NodeKind.FILE.value}:{file_path}".encode()
        ).hexdigest()[:24]

        edges.extend(self._extract_containment_edges(nodes, file_node_id, commit_hash))

        import_qname = "imports" if lang != "cpp" else None
        include_qname = "includes" if lang == "cpp" else None
        for qname_key, edge_kind in (
            (import_qname, EdgeKind.IMPORTS),
            (include_qname, EdgeKind.INCLUDES),
        ):
            if qname_key is not None:
                edges.extend(
                    self._extract_import_edges(
                        tree,
                        compiled_queries,
                        file_node_id,
                        qname_key,
                        edge_kind,
                        source,
                        commit_hash,
                    )
                )

        edges.extend(
            self._extract_call_edges(tree, compiled_queries, name_to_id, source, commit_hash, lang)
        )

        if lang in ("cpp", "java", "rust"):
            edges.extend(
                self._resolve_cross_file_symbols(tree, nodes, file_path, lang, source, commit_hash)
            )

        _add_type_relation_edges(
            edges,
            tree,
            compiled_queries,
            nodes,
            source,
            lang,
            commit_hash,
            name_to_id,
        )

        if lang == "java":
            edges.extend(
                self._extract_injects(tree, nodes, lang, compiled_queries, source, commit_hash)
            )

        edges.extend(
            self._extract_depends_on(tree, compiled_queries, file_path, lang, source, commit_hash)
        )

        edges.extend(
            self._extract_overrides(tree, nodes, lang, compiled_queries, source, commit_hash)
        )

        edges.extend(self._extract_types(tree, nodes, lang, compiled_queries, source, commit_hash))

        if lang == "rust":
            edges.extend(
                self._extract_rust_edges(tree, nodes, source, commit_hash, compiled_queries)
            )

        return edges

    def _extract_containment_edges(
        self,
        nodes: list[ASTNode],
        file_node_id: str,
        commit_hash: str,
    ) -> list[ASTEdge]:
        edges = []
        type_nodes = [
            n
            for n in nodes
            if n.kind
            in (
                NodeKind.CLASS,
                NodeKind.INTERFACE,
                NodeKind.STRUCT,
                NodeKind.ENUM,
                NodeKind.TRAIT,
                NodeKind.NAMESPACE,
            )
        ]
        method_nodes = [
            n
            for n in nodes
            if n.kind
            in (
                NodeKind.FUNCTION,
                NodeKind.METHOD,
                NodeKind.CONSTRUCTOR,
                NodeKind.DESTRUCTOR,
            )
        ]
        field_nodes = [n for n in nodes if n.kind == NodeKind.FIELD]

        for tn in type_nodes:
            ek = (
                EdgeKind.CONTAINS_CLASS
                if tn.kind != NodeKind.FUNCTION
                else EdgeKind.CONTAINS_FUNCTION
            )
            edges.append(
                ASTEdge(
                    kind=ek,
                    from_id=file_node_id,
                    to_id=tn.id,
                    valid_from=commit_hash,
                )
            )

        for mn in method_nodes:
            parent = _find_enclosing_type(mn, type_nodes)
            if parent:
                edges.append(
                    ASTEdge(
                        kind=EdgeKind.CONTAINS_METHOD,
                        from_id=parent.id,
                        to_id=mn.id,
                        valid_from=commit_hash,
                    )
                )
            else:
                edges.append(
                    ASTEdge(
                        kind=EdgeKind.CONTAINS_FUNCTION,
                        from_id=file_node_id,
                        to_id=mn.id,
                        valid_from=commit_hash,
                    )
                )

        for fn in field_nodes:
            parent = _find_enclosing_type(fn, type_nodes)
            if parent:
                edges.append(
                    ASTEdge(
                        kind=EdgeKind.CONTAINS_FIELD,
                        from_id=parent.id,
                        to_id=fn.id,
                        valid_from=commit_hash,
                    )
                )

        return edges

    def _extract_import_edges(
        self,
        tree: Tree,
        compiled: dict,
        file_node_id: str,
        qname_key: str,
        edge_kind: EdgeKind,
        source: bytes,
        commit_hash: str,
    ) -> list[ASTEdge]:
        edges = []
        query = compiled.get(qname_key)
        if query is None:
            return edges

        for _, md in QueryCursor(query).matches(tree.root_node):
            path_ts = md.get("import_path") or md.get("path") or md.get("module_path")
            if path_ts is None:
                continue
            if isinstance(path_ts, list):
                path_ts = path_ts[0]
            path_text = _node_text(path_ts, source).strip('"<>').strip()
            if not path_text:
                continue
            target_id = hashlib.sha256(f"import:{path_text}".encode()).hexdigest()[:24]
            edges.append(
                ASTEdge(
                    kind=edge_kind,
                    from_id=file_node_id,
                    to_id=target_id,
                    label=path_text,
                    valid_from=commit_hash,
                )
            )
        return edges

    def _extract_call_edges(
        self,
        tree: Tree,
        compiled: dict,
        name_to_id: dict[str, str],
        source: bytes,
        commit_hash: str,
        lang: str,
    ) -> list[ASTEdge]:
        edges = []
        call_query = compiled.get("calls")
        if not call_query:
            return edges

        method_nodes = [
            n
            for n in []
            if n.kind
            in (NodeKind.METHOD, NodeKind.FUNCTION, NodeKind.CONSTRUCTOR, NodeKind.DESTRUCTOR)
        ]

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

            call_line = call_node_ts.start_point[0] + 1
            caller = _find_enclosing_callable(call_line, method_nodes)
            if not caller:
                continue

            callee_node_id = name_to_id.get(callee_name)
            if callee_node_id:
                edges.append(
                    ASTEdge(
                        kind=EdgeKind.CALLS,
                        from_id=caller.id,
                        to_id=callee_node_id,
                        confidence=0.7,
                        resolution_method="name",
                        valid_from=commit_hash,
                    )
                )

        return edges

    def _extract_depends_on(
        self,
        tree: Tree,
        compiled: dict,
        file_path: str,
        lang: str,
        source: bytes,
        commit_hash: str,
    ) -> list[ASTEdge]:
        edges = []
        file_node_id = hashlib.sha256(
            f"{file_path}:{NodeKind.FILE.value}:{file_path}".encode()
        ).hexdigest()[:24]

        if lang == "cpp":
            include_query = compiled.get("includes")
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
                    target_id = hashlib.sha256(f"include:{path_text}".encode()).hexdigest()[:24]
                    edges.append(
                        ASTEdge(
                            kind=EdgeKind.DEPENDS_ON,
                            from_id=file_node_id,
                            to_id=target_id,
                            label=path_text,
                            valid_from=commit_hash,
                        )
                    )

        elif lang == "java":
            import_query = compiled.get("imports")
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
                    target_id = hashlib.sha256(f"import:{import_text}".encode()).hexdigest()[:24]
                    edges.append(
                        ASTEdge(
                            kind=EdgeKind.DEPENDS_ON,
                            from_id=file_node_id,
                            to_id=target_id,
                            label=import_text,
                            valid_from=commit_hash,
                        )
                    )

        return edges

    def _extract_overrides(
        self,
        tree: Tree,
        nodes: list[ASTNode],
        lang: str,
        compiled: dict,
        source: bytes,
        commit_hash: str,
    ) -> list[ASTEdge]:
        edges = []

        if lang == "java":
            override_query = compiled.get("overrides")
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

                    overriding_node = next(
                        (n for n in nodes if n.name == method_name and n.kind == NodeKind.METHOD),
                        None,
                    )
                    if not overriding_node:
                        continue

                    edges.append(
                        ASTEdge(
                            kind=EdgeKind.OVERRIDES,
                            from_id=overriding_node.id,
                            to_id="",
                            valid_from=commit_hash,
                        )
                    )

        elif lang == "cpp":
            logger.debug("C++ OVERRIDES not yet implemented - requires query")

        return edges

    def _extract_types(
        self,
        tree: Tree,
        nodes: list[ASTNode],
        lang: str,
        compiled: dict,
        source: bytes,
        commit_hash: str,
    ) -> list[ASTEdge]:
        edges = []

        if lang == "java":
            field_query = compiled.get("field_defs")
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

                    field_node = next(
                        (n for n in nodes if n.name == field_name and n.kind == NodeKind.FIELD),
                        None,
                    )
                    if not field_node:
                        continue

                    type_id = hashlib.sha256(f"type:{field_type}".encode()).hexdigest()[:24]
                    edges.append(
                        ASTEdge(
                            kind=EdgeKind.TYPES,
                            from_id=field_node.id,
                            to_id=type_id,
                            label=field_type,
                            valid_from=commit_hash,
                        )
                    )

        elif lang == "cpp":
            logger.debug("C++ TYPES not yet implemented - requires query")

        return edges

    def _extract_injects(
        self,
        tree: Tree,
        nodes: list[ASTNode],
        lang: str,
        compiled: dict,
        source: bytes,
        commit_hash: str,
    ) -> list[ASTEdge]:
        edges = []

        if lang != "java":
            return edges

        nodes_by_name = {n.name: n for n in nodes}

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

                field_node = next((n for n in nodes if n.name == field_name), None)
                type_node = nodes_by_name.get(type_name)

                if field_node and type_node:
                    edges.append(
                        ASTEdge(
                            kind=EdgeKind.INJECTS,
                            from_id=field_node.id,
                            to_id=type_node.id,
                            dep_kind=annotation.lower(),
                            valid_from=commit_hash,
                        )
                    )

        ctor_query = compiled.get("di_constructors")
        if ctor_query:
            for _, match in QueryCursor(ctor_query).matches(tree.root_node):
                annotation_ts = match.get("annotation_name")
                type_ts = match.get("injected_type")
                node_ts = match.get("node")

                if not all([annotation_ts, type_ts, node_ts]):
                    continue

                annotation = _node_text(annotation_ts, source)
                type_name = _node_text(type_ts, source)

                name_ts = node_ts.child_by_field_name("name")
                if not name_ts:
                    continue
                ctor_name = _node_text(name_ts, source)

                ctor_node = next(
                    (n for n in nodes if n.name == ctor_name and n.kind == NodeKind.CONSTRUCTOR),
                    None,
                )
                type_node = nodes_by_name.get(type_name)

                if ctor_node and type_node:
                    edges.append(
                        ASTEdge(
                            kind=EdgeKind.INJECTS,
                            from_id=ctor_node.id,
                            to_id=type_node.id,
                            dep_kind=annotation.lower(),
                            valid_from=commit_hash,
                        )
                    )

        return edges

    def _extract_rust_edges(
        self,
        tree: Tree,
        nodes: list[ASTNode],
        source: bytes,
        commit_hash: str,
        compiled: dict,
    ) -> list[ASTEdge]:
        edges = []
        nodes_by_name: dict[str, ASTNode] = {n.name: n for n in nodes}

        impl_blocks_query = compiled.get("rust", {}).get("impl_defs")
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

                impl_node = None
                for node in nodes:
                    if node.name == impl_type_name and node.kind in (
                        NodeKind.STRUCT,
                        NodeKind.ENUM,
                        NodeKind.CLASS,
                    ):
                        impl_node = node
                        break

                trait_node = nodes_by_name.get(trait_name)

                if impl_node and trait_node:
                    edges.append(
                        ASTEdge(
                            kind=EdgeKind.IMPLEMENTS,
                            from_id=impl_node.id,
                            to_id=trait_node.id,
                            valid_from=commit_hash,
                        )
                    )

        return edges

    def _resolve_cross_file_symbols(
        self,
        tree: Tree,
        nodes: list[ASTNode],
        file_path: str,
        lang: str,
        source: bytes,
        commit_hash: str,
    ) -> list[ASTEdge]:
        edges = []

        if lang not in ("cpp", "java", "rust"):
            return edges

        for node in tree.root_node.children:
            if node.type in ("identifier", "path_expression"):
                symbol_name = _node_text(node, source)
                if self._is_cross_file_reference(symbol_name, lang, file_path):
                    target_id = hashlib.sha256(f"crossfile:{symbol_name}".encode()).hexdigest()[:24]
                    edges.append(
                        ASTEdge(
                            kind=EdgeKind.CROSS_FILE_CALL,
                            from_id="",
                            to_id=target_id,
                            label=symbol_name,
                            confidence=0.3,
                            resolution_method="crossfile",
                            valid_from=commit_hash,
                        )
                    )

        return edges

    def _is_cross_file_reference(self, symbol_name: str, lang: str, current_file: str) -> bool:
        if lang == "cpp":
            return "::" in symbol_name or symbol_name in (
                "stdio.h",
                "stdlib.h",
                "string.h",
                "vector",
                "iostream",
            )
        elif lang == "java":
            return "." in symbol_name and not symbol_name.startswith("java.")
        elif lang == "rust":
            return "::" in symbol_name and not symbol_name.startswith("crate::")
        return False


def _node_text(node: Node, source: bytes) -> str:
    """Extract the UTF-8 text for a tree-sitter node."""
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace").strip()


def _containment_edge_kind(kind: NodeKind) -> EdgeKind:
    if kind == NodeKind.FUNCTION:
        return EdgeKind.CONTAINS_FUNCTION
    return EdgeKind.CONTAINS_CLASS


def _find_enclosing_type(target: ASTNode, type_nodes: list[ASTNode]) -> Optional[ASTNode]:
    """Find the innermost type node whose byte range fully contains target."""
    best: Optional[ASTNode] = None
    best_size = float("inf")
    for tn in type_nodes:
        if tn.start_byte <= target.start_byte and tn.end_byte >= target.end_byte:
            size = tn.end_byte - tn.start_byte
            if size < best_size:
                best, best_size = tn, size
    return best


def _find_enclosing_callable(call_line: int, callables: list[ASTNode]) -> Optional[ASTNode]:
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
                edges.append(
                    ASTEdge(
                        kind=ek,
                        from_id=cls_id,
                        to_id=base_id,
                        label=base_name,
                        valid_from=commit_hash,
                    )
                )

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
                    edges.append(
                        ASTEdge(
                            kind=EdgeKind.IMPLEMENTS,
                            from_id=cls_id,
                            to_id=iface_id,
                            label=iface_name,
                            valid_from=commit_hash,
                        )
                    )

        if lang == "rust":
            edges.extend(
                self._extract_rust_edges(tree, nodes, source, commit_hash, compiled_queries)
            )

        return edges

    def _extract_containment_edges(
        self,
        nodes: list[ASTNode],
        file_node_id: str,
        commit_hash: str,
    ) -> list[ASTEdge]:
        edges = []
        type_nodes = [
            n
            for n in nodes
            if n.kind
            in (
                NodeKind.CLASS,
                NodeKind.INTERFACE,
                NodeKind.STRUCT,
                NodeKind.ENUM,
                NodeKind.TRAIT,
                NodeKind.NAMESPACE,
            )
        ]
        method_nodes = [
            n
            for n in nodes
            if n.kind
            in (
                NodeKind.FUNCTION,
                NodeKind.METHOD,
                NodeKind.CONSTRUCTOR,
                NodeKind.DESTRUCTOR,
            )
        ]
        field_nodes = [n for n in nodes if n.kind == NodeKind.FIELD]

        for tn in type_nodes:
            ek = (
                EdgeKind.CONTAINS_CLASS
                if tn.kind != NodeKind.FUNCTION
                else EdgeKind.CONTAINS_FUNCTION
            )
            edges.append(
                ASTEdge(
                    kind=ek,
                    from_id=file_node_id,
                    to_id=tn.id,
                    valid_from=commit_hash,
                )
            )

        for mn in method_nodes:
            parent = _find_enclosing_type(mn, type_nodes)
            if parent:
                edges.append(
                    ASTEdge(
                        kind=EdgeKind.CONTAINS_METHOD,
                        from_id=parent.id,
                        to_id=mn.id,
                        valid_from=commit_hash,
                    )
                )
            else:
                edges.append(
                    ASTEdge(
                        kind=EdgeKind.CONTAINS_FUNCTION,
                        from_id=file_node_id,
                        to_id=mn.id,
                        valid_from=commit_hash,
                    )
                )

        for fn in field_nodes:
            parent = _find_enclosing_type(fn, type_nodes)
            if parent:
                edges.append(
                    ASTEdge(
                        kind=EdgeKind.CONTAINS_FIELD,
                        from_id=parent.id,
                        to_id=fn.id,
                        valid_from=commit_hash,
                    )
                )

        return edges

    def _extract_import_edges(
        self,
        tree: Tree,
        compiled: dict,
        file_node_id: str,
        qname_key: str,
        edge_kind: EdgeKind,
        source: bytes,
        commit_hash: str,
    ) -> list[ASTEdge]:
        edges = []
        query = compiled.get(qname_key)
        if query is None:
            return edges

        for _, md in QueryCursor(query).matches(tree.root_node):
            path_ts = md.get("import_path") or md.get("path") or md.get("module_path")
            if path_ts is None:
                continue
            if isinstance(path_ts, list):
                path_ts = path_ts[0]
            path_text = _node_text(path_ts, source).strip('"<>').strip()
            if not path_text:
                continue
            target_id = hashlib.sha256(f"import:{path_text}".encode()).hexdigest()[:24]
            edges.append(
                ASTEdge(
                    kind=edge_kind,
                    from_id=file_node_id,
                    to_id=target_id,
                    label=path_text,
                    valid_from=commit_hash,
                )
            )
        return edges

    def _extract_call_edges(
        self,
        tree: Tree,
        compiled: dict,
        name_to_id: dict[str, str],
        source: bytes,
        commit_hash: str,
        lang: str,
    ) -> list[ASTEdge]:
        edges = []
        call_query = compiled.get("calls")
        if not call_query:
            return edges

        method_nodes = [
            n
            for n in []
            if n.kind
            in (NodeKind.METHOD, NodeKind.FUNCTION, NodeKind.CONSTRUCTOR, NodeKind.DESTRUCTOR)
        ]

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

            call_line = call_node_ts.start_point[0] + 1
            caller = _find_enclosing_callable(call_line, method_nodes)
            if not caller:
                continue

            callee_node_id = name_to_id.get(callee_name)
            if callee_node_id:
                edges.append(
                    ASTEdge(
                        kind=EdgeKind.CALLS,
                        from_id=caller.id,
                        to_id=callee_node_id,
                        confidence=0.7,
                        resolution_method="name",
                        valid_from=commit_hash,
                    )
                )

        return edges

    def _extract_depends_on(
        self,
        tree: Tree,
        compiled: dict,
        file_path: str,
        lang: str,
        source: bytes,
        commit_hash: str,
    ) -> list[ASTEdge]:
        edges = []
        file_node_id = hashlib.sha256(
            f"{file_path}:{NodeKind.FILE.value}:{file_path}".encode()
        ).hexdigest()[:24]

        if lang == "cpp":
            include_query = compiled.get("includes")
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
                    target_id = hashlib.sha256(f"include:{path_text}".encode()).hexdigest()[:24]
                    edges.append(
                        ASTEdge(
                            kind=EdgeKind.DEPENDS_ON,
                            from_id=file_node_id,
                            to_id=target_id,
                            label=path_text,
                            valid_from=commit_hash,
                        )
                    )

        elif lang == "java":
            import_query = compiled.get("imports")
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
                    target_id = hashlib.sha256(f"import:{import_text}".encode()).hexdigest()[:24]
                    edges.append(
                        ASTEdge(
                            kind=EdgeKind.DEPENDS_ON,
                            from_id=file_node_id,
                            to_id=target_id,
                            label=import_text,
                            valid_from=commit_hash,
                        )
                    )

        return edges

    def _extract_overrides(
        self,
        tree: Tree,
        nodes: list[ASTNode],
        lang: str,
        compiled: dict,
        source: bytes,
        commit_hash: str,
    ) -> list[ASTEdge]:
        edges = []

        if lang == "java":
            override_query = compiled.get("overrides")
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

                    overriding_node = next(
                        (n for n in nodes if n.name == method_name and n.kind == NodeKind.METHOD),
                        None,
                    )
                    if not overriding_node:
                        continue

                    edges.append(
                        ASTEdge(
                            kind=EdgeKind.OVERRIDES,
                            from_id=overriding_node.id,
                            to_id="",
                            valid_from=commit_hash,
                        )
                    )

        elif lang == "cpp":
            logger.debug("C++ OVERRIDES not yet implemented - requires query")

        return edges

    def _extract_types(
        self,
        tree: Tree,
        nodes: list[ASTNode],
        lang: str,
        compiled: dict,
        source: bytes,
        commit_hash: str,
    ) -> list[ASTEdge]:
        edges = []

        if lang == "java":
            field_query = compiled.get("field_defs")
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

                    field_node = next(
                        (n for n in nodes if n.name == field_name and n.kind == NodeKind.FIELD),
                        None,
                    )
                    if not field_node:
                        continue

                    type_id = hashlib.sha256(f"type:{field_type}".encode()).hexdigest()[:24]
                    edges.append(
                        ASTEdge(
                            kind=EdgeKind.TYPES,
                            from_id=field_node.id,
                            to_id=type_id,
                            label=field_type,
                            valid_from=commit_hash,
                        )
                    )

        elif lang == "cpp":
            logger.debug("C++ TYPES not yet implemented - requires query")

        return edges

    def _extract_injects(
        self,
        tree: Tree,
        nodes: list[ASTNode],
        lang: str,
        compiled: dict,
        source: bytes,
        commit_hash: str,
    ) -> list[ASTEdge]:
        edges = []

        if lang != "java":
            return edges

        nodes_by_name = {n.name: n for n in nodes}

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

                field_node = next((n for n in nodes if n.name == field_name), None)
                type_node = nodes_by_name.get(type_name)

                if field_node and type_node:
                    edges.append(
                        ASTEdge(
                            kind=EdgeKind.INJECTS,
                            from_id=field_node.id,
                            to_id=type_node.id,
                            dep_kind=annotation.lower(),
                            valid_from=commit_hash,
                        )
                    )

        ctor_query = compiled.get("di_constructors")
        if ctor_query:
            for _, match in QueryCursor(ctor_query).matches(tree.root_node):
                annotation_ts = match.get("annotation_name")
                type_ts = match.get("injected_type")
                node_ts = match.get("node")

                if not all([annotation_ts, type_ts, node_ts]):
                    continue

                annotation = _node_text(annotation_ts, source)
                type_name = _node_text(type_ts, source)

                name_ts = node_ts.child_by_field_name("name")
                if not name_ts:
                    continue
                ctor_name = _node_text(name_ts, source)

                ctor_node = next(
                    (n for n in nodes if n.name == ctor_name and n.kind == NodeKind.CONSTRUCTOR),
                    None,
                )
                type_node = nodes_by_name.get(type_name)

                if ctor_node and type_node:
                    edges.append(
                        ASTEdge(
                            kind=EdgeKind.INJECTS,
                            from_id=ctor_node.id,
                            to_id=type_node.id,
                            dep_kind=annotation.lower(),
                            valid_from=commit_hash,
                        )
                    )

        return edges

    def _extract_rust_edges(
        self,
        tree: Tree,
        nodes: list[ASTNode],
        source: bytes,
        commit_hash: str,
        compiled: dict,
    ) -> list[ASTEdge]:
        edges = []
        nodes_by_name: dict[str, ASTNode] = {n.name: n for n in nodes}

        impl_blocks_query = compiled.get("rust", {}).get("impl_defs")
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

                impl_node = None
                for node in nodes:
                    if node.name == impl_type_name and node.kind in (
                        NodeKind.STRUCT,
                        NodeKind.ENUM,
                        NodeKind.CLASS,
                    ):
                        impl_node = node
                        break

                trait_node = nodes_by_name.get(trait_name)

                if impl_node and trait_node:
                    edges.append(
                        ASTEdge(
                            kind=EdgeKind.IMPLEMENTS,
                            from_id=impl_node.id,
                            to_id=trait_node.id,
                            valid_from=commit_hash,
                        )
                    )

        return edges

    def _resolve_cross_file_symbols(
        self,
        tree: Tree,
        nodes: list[ASTNode],
        file_path: str,
        lang: str,
        source: bytes,
        commit_hash: str,
    ) -> list[ASTEdge]:
        edges = []

        if lang not in ("cpp", "java", "rust"):
            return edges

        for node in tree.root_node.children:
            if node.type in ("identifier", "path_expression"):
                symbol_name = _node_text(node, source)
                if self._is_cross_file_reference(symbol_name, lang, file_path):
                    target_id = hashlib.sha256(f"crossfile:{symbol_name}".encode()).hexdigest()[:24]
                    edges.append(
                        ASTEdge(
                            kind=EdgeKind.CROSS_FILE_CALL,
                            from_id="",
                            to_id=target_id,
                            label=symbol_name,
                            confidence=0.3,
                            resolution_method="crossfile",
                            valid_from=commit_hash,
                        )
                    )

        return edges

    def _is_cross_file_reference(self, symbol_name: str, lang: str, current_file: str) -> bool:
        if lang == "cpp":
            return "::" in symbol_name or symbol_name in (
                "stdio.h",
                "stdlib.h",
                "string.h",
                "vector",
                "iostream",
            )
        elif lang == "java":
            return "." in symbol_name and not symbol_name.startswith("java.")
        elif lang == "rust":
            return "::" in symbol_name and not symbol_name.startswith("crate::")
        return False


def _node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace").strip()


def _find_enclosing_type(target: ASTNode, type_nodes: list[ASTNode]) -> Optional[ASTNode]:
    best: Optional[ASTNode] = None
    best_size = float("inf")
    for tn in type_nodes:
        if tn.start_byte <= target.start_byte and tn.end_byte >= target.end_byte:
            size = tn.end_byte - tn.start_byte
            if size < best_size:
                best, best_size = tn, size
    return best


def _find_enclosing_callable(call_line: int, callables: list[ASTNode]) -> Optional[ASTNode]:
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
                edges.append(
                    ASTEdge(
                        kind=ek,
                        from_id=cls_id,
                        to_id=base_id,
                        label=base_name,
                        valid_from=commit_hash,
                    )
                )

        for key in ("iface",):
            iface_list = md.get(key)
            if iface_list is None:
                continue
            items = iface_list if isinstance(iface_list, list) else [iface_list]
            for iface_ts in items:
                iface_name = _node_text(iface_ts, source)
                iface_id = name_to_id.get(iface_name)
                if iface_id:
                    edges.append(
                        ASTEdge(
                            kind=EdgeKind.IMPLEMENTS,
                            from_id=cls_id,
                            to_id=iface_id,
                            label=iface_name,
                            valid_from=commit_hash,
                        )
                    )
