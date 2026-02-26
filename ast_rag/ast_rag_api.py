"""
ast_rag_api.py - High-level query API for AST-RAG.

Combines Neo4j graph queries with Qdrant vector search.

API surface:
- find_definition(name, kind, lang) -> list[ASTNode]
- find_callers(node_id, max_depth) -> list[ASTNode]
- find_callees(node_id, max_depth) -> list[ASTNode]
- expand_neighbourhood(node_id, depth, edge_types) -> SubGraph
- search_semantic(query, limit) -> list[SearchResult]
- get_code_snippet(file_path, start_line, end_line) -> str
- analyze_text(text, limit) -> list[SearchResult]
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from neo4j import Driver

from ast_rag.metrics import (
    track_latency,
    SEARCH_LATENCY,
    FIND_DEFINITION_LATENCY,
    FIND_REFERENCES_LATENCY,
    SEARCH_TOTAL,
)
from ast_rag.models import (
    ASTNode,
    ASTEdge,
    SubGraph,
    SearchResult,
    NodeKind,
    EdgeKind,
    Language,
)
from ast_rag.embeddings import EmbeddingManager
from ast_rag.graph_schema import _KIND_TO_LABEL
from ast_rag.graph_updater import compute_diff_for_commits

logger = logging.getLogger(__name__)

# The active version filter used in all Cypher queries
_ACTIVE_FILTER = "n.valid_to IS NULL"
_ACTIVE_FILTER_R = "r.valid_to IS NULL"


def _record_to_node(record: dict) -> ASTNode:
    """Convert a Neo4j record map to an ASTNode model."""
    r = record
    return ASTNode(
        id=r.get("id", ""),
        kind=NodeKind(r.get("kind", "Function")),
        name=r.get("name", ""),
        qualified_name=r.get("qualified_name", ""),
        lang=Language(r.get("lang", "java")),
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


def _record_to_edge(record: dict) -> ASTEdge:
    r = record
    return ASTEdge(
        id=r.get("id", ""),
        kind=EdgeKind(r.get("kind", "CALLS")),
        from_id=r.get("from_id", ""),
        to_id=r.get("to_id", ""),
        label=r.get("label") or None,
        valid_from=r.get("valid_from", "INIT"),
        valid_to=r.get("valid_to"),
    )


class ASTRagAPI:
    """Unified API combining graph queries and vector search.

    Usage::

        api = ASTRagAPI(driver, embedding_manager)
        nodes = api.find_definition("MyService", kind="Class", lang="java")
        callers = api.find_callers(nodes[0].id, max_depth=2)
    """

    def __init__(
        self,
        driver: Driver,
        embedding_manager: EmbeddingManager,
    ) -> None:
        self._driver = driver
        self._embed = embedding_manager

    # ------------------------------------------------------------------
    # Definition lookup
    # ------------------------------------------------------------------

    @track_latency(FIND_DEFINITION_LATENCY)
    def find_definition(
        self,
        name: str,
        kind: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> list[ASTNode]:
        """Find active AST nodes by name, with optional kind and lang filters.

        Uses exact match on `name` or tries qualified_name contains.
        Falls back to full-text index if no exact match found.
        """
        SEARCH_TOTAL.labels(lang=lang or "any", kind=kind or "any").inc()
        label_filter = _KIND_TO_LABEL.get(kind, "") if kind else ""
        label_clause = f":{label_filter}" if label_filter else ""

        lang_clause = "AND n.lang = $lang" if lang else ""

        cypher = f"""
MATCH (n{label_clause})
WHERE n.valid_to IS NULL
  AND (n.name = $name OR n.qualified_name CONTAINS $name)
  {lang_clause}
RETURN n
ORDER BY n.qualified_name
LIMIT 50
"""
        params: dict = {"name": name}
        if lang:
            params["lang"] = lang

        results: list[ASTNode] = []
        with self._driver.session() as session:
            for record in session.run(cypher, **params):
                node_data = dict(record["n"])
                results.append(_record_to_node(node_data))

        return results

    # ------------------------------------------------------------------
    # Call graph traversal
    # ------------------------------------------------------------------

    def find_callers(
        self,
        node_id: str,
        max_depth: int = 1,
    ) -> list[ASTNode]:
        """Find all callers of the given node up to max_depth hops.

        Traverses CALLS edges in reverse direction.
        """
        max_depth = min(max_depth, 5)  # safety cap
        cypher = f"""
MATCH (caller)-[:CALLS|VIRTUAL_CALL|LAMBDA_CALL|CROSS_FILE_CALL*1..{max_depth}]->(target {{id: $node_id}})
WHERE caller.valid_to IS NULL AND target.valid_to IS NULL
RETURN DISTINCT caller
ORDER BY caller.qualified_name
LIMIT 200
"""
        results: list[ASTNode] = []
        with self._driver.session() as session:
            for record in session.run(cypher, node_id=node_id):
                results.append(_record_to_node(dict(record["caller"])))
        return results

    def find_callees(
        self,
        node_id: str,
        max_depth: int = 1,
    ) -> list[ASTNode]:
        """Find all callees of the given node up to max_depth hops.

        Traverses CALLS edges forward.
        """
        max_depth = min(max_depth, 5)
        cypher = f"""
MATCH (source {{id: $node_id}})-[:CALLS|VIRTUAL_CALL|LAMBDA_CALL|CROSS_FILE_CALL*1..{max_depth}]->(callee)
WHERE source.valid_to IS NULL AND callee.valid_to IS NULL
RETURN DISTINCT callee
ORDER BY callee.qualified_name
LIMIT 200
"""
        results: list[ASTNode] = []
        with self._driver.session() as session:
            for record in session.run(cypher, node_id=node_id):
                results.append(_record_to_node(dict(record["callee"])))
        return results

    # ------------------------------------------------------------------
    # Neighbourhood expansion
    # ------------------------------------------------------------------

    def expand_neighbourhood(
        self,
        node_id: str,
        depth: int = 1,
        edge_types: Optional[list[str]] = None,
    ) -> SubGraph:
        """Return a SubGraph of nodes and edges within `depth` hops.

        edge_types: list of relationship type names to follow, e.g.
            ["CALLS", "INHERITS"].  None = all types.
        """
        depth = min(depth, 4)
        if edge_types:
            rel_clause = f"[*1..{depth}]"  # can't use dynamic types in variable-length easily
            # Build the Cypher with explicit types
            rel_parts = " | ".join(f":{et}" for et in edge_types)
            rel_clause = f"[{rel_parts}*1..{depth}]"
        else:
            rel_clause = f"[*1..{depth}]"

        cypher = f"""
MATCH path = (start {{id: $node_id}})-{rel_clause}-(neighbour)
WHERE start.valid_to IS NULL AND neighbour.valid_to IS NULL
UNWIND relationships(path) AS r
RETURN DISTINCT
  start,
  neighbour,
  r.id      AS r_id,
  r.kind     AS r_kind,
  startNode(r).id AS r_from,
  endNode(r).id   AS r_to,
  r.label    AS r_label,
  r.valid_from AS r_vf,
  r.valid_to   AS r_vt
LIMIT 500
"""
        nodes_seen: dict[str, ASTNode] = {}
        edges_seen: dict[str, ASTEdge] = {}

        with self._driver.session() as session:
            for record in session.run(cypher, node_id=node_id):
                for key in ("start", "neighbour"):
                    nd = dict(record[key])
                    nid = nd.get("id", "")
                    if nid and nid not in nodes_seen:
                        nodes_seen[nid] = _record_to_node(nd)
                rid = record["r_id"]
                if rid and rid not in edges_seen:
                    edges_seen[rid] = ASTEdge(
                        id=rid,
                        kind=EdgeKind(record["r_kind"]) if record["r_kind"] else EdgeKind.CALLS,
                        from_id=record["r_from"] or "",
                        to_id=record["r_to"] or "",
                        label=record["r_label"],
                        valid_from=record["r_vf"] or "INIT",
                        valid_to=record["r_vt"],
                    )

        return SubGraph(
            nodes=list(nodes_seen.values()),
            edges=list(edges_seen.values()),
        )

    # ------------------------------------------------------------------
    # Semantic (vector) search
    # ------------------------------------------------------------------

    @track_latency(SEARCH_LATENCY)
    def search_semantic(
        self,
        query: str,
        limit: int = 10,
        lang: Optional[str] = None,
        kind: Optional[str] = None,
        vector_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
    ) -> list[SearchResult]:
        """Semantic search using Qdrant + bge-m3 embeddings.

        Uses hybrid search (vector + keyword) by default for better relevance.
        Optionally filter by lang and/or kind.
        Auto-fallback: if filtered results < limit, searches across all languages.

        Args:
            query: Natural language or code query
            limit: Maximum results (default 10, max 50)
            lang: Optional language filter
            kind: Optional node kind filter
            vector_weight: Override config vector weight (None = use config)
            keyword_weight: Override config keyword weight (None = use config)

        Returns:
            List of SearchResult ordered by fused score (descending)
        """
        SEARCH_TOTAL.labels(lang=lang or "any", kind=kind or "any").inc()
        # Use hybrid search when Neo4j driver is available
        if self._driver is not None:
            return self._embed.hybrid_search(
                query=query,
                limit=limit,
                lang_filter=lang,
                kind_filter=kind,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
            )
        else:
            # Fallback to pure vector search
            return self._embed.search(
                query=query,
                limit=limit,
                lang_filter=lang,
                kind_filter=kind,
                auto_fallback=True,
            )

    # ------------------------------------------------------------------
    # Source code snippet retrieval
    # ------------------------------------------------------------------

    def get_code_snippet(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
    ) -> str:
        """Read and return the source lines [start_line, end_line] (1-indexed).

        Returns an empty string if the file cannot be read.
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning("File not found: %s", file_path)
                return ""
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            # Clamp to valid range (1-indexed)
            s = max(0, start_line - 1)
            e = min(len(lines), end_line)
            return "\n".join(lines[s:e])
        except OSError as exc:
            logger.error("Cannot read snippet from %s: %s", file_path, exc)
            return ""

    # ------------------------------------------------------------------
    # Convenience: look up a node by id from the graph
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> Optional[ASTNode]:
        """Fetch a single active node by its id."""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (n {id: $node_id})
                WHERE n.valid_to IS NULL
                RETURN n LIMIT 1
                """,
                node_id=node_id,
            )
            record = result.single()
            if record:
                return _record_to_node(dict(record["n"]))
        return None

    # ------------------------------------------------------------------
    # Inheritance hierarchy helpers
    # ------------------------------------------------------------------

    def find_subclasses(self, node_id: str, max_depth: int = 3) -> list[ASTNode]:
        """Find all subclasses/implementors of the given type."""
        max_depth = min(max_depth, 5)
        cypher = f"""
MATCH (child)-[:INHERITS|EXTENDS|IMPLEMENTS*1..{max_depth}]->(parent {{id: $node_id}})
WHERE child.valid_to IS NULL
RETURN DISTINCT child
ORDER BY child.qualified_name
LIMIT 100
"""
        results: list[ASTNode] = []
        with self._driver.session() as session:
            for record in session.run(cypher, node_id=node_id):
                results.append(_record_to_node(dict(record["child"])))
        return results

    def find_superclasses(self, node_id: str, max_depth: int = 3) -> list[ASTNode]:
        """Find all parent classes/interfaces of the given type."""
        max_depth = min(max_depth, 5)
        cypher = f"""
MATCH (child {{id: $node_id}})-[:INHERITS|EXTENDS|IMPLEMENTS*1..{max_depth}]->(parent)
WHERE parent.valid_to IS NULL
RETURN DISTINCT parent
ORDER BY parent.qualified_name
LIMIT 100
"""
        results: list[ASTNode] = []
        with self._driver.session() as session:
            for record in session.run(cypher, node_id=node_id):
                results.append(_record_to_node(dict(record["parent"])))
        return results

    def find_overrides(self, method_node_id: str, max_depth: int = 3) -> list[ASTNode]:
        """Find all override implementations of a virtual method.

        Traverses OVERRIDES edges to find all methods that override
        the given method (directly or transitively).

        Args:
            method_node_id: ID of the base/virtual method
            max_depth: Maximum override chain depth (default 3, max 5)

        Returns:
            List of overriding method nodes ordered by qualified_name

        Example:
            Base.method() → Child.method() → Grandchild.method()
            Returns [Child.method(), Grandchild.method()]
        """
        max_depth = min(max_depth, 5)

        cypher = f"""
MATCH (base {{id: $method_node_id}})
WHERE base.kind IN ['Method', 'Function']
MATCH (overrider)-[:OVERRIDES*1..{max_depth}]->(base)
WHERE overrider.valid_to IS NULL
RETURN DISTINCT overrider
ORDER BY overrider.qualified_name
LIMIT 100
"""
        results: list[ASTNode] = []

        with self._driver.session() as session:
            for record in session.run(cypher, method_node_id=method_node_id):
                node_data = dict(record["overrider"])
                results.append(_record_to_node(node_data))

        return results

    def get_call_confidence(self, call_edge_id: str) -> Optional[float]:
        """Get confidence score for a CALLS edge.

        Confidence indicates how certain the type-based resolution was:
        - 1.0: Exact type match, unambiguous receiver
        - 0.8-0.9: Type inferred with high certainty
        - 0.5-0.7: Name-based fallback or partial type info
        - <0.5: Heuristic or speculative resolution

        Args:
            call_edge_id: ID of the CALLS edge

        Returns:
            Confidence score (0.0-1.0) or None if edge not found

        Example:
            confidence = api.get_call_confidence("edge123")
            if confidence < 0.7:
                print("Low confidence - may be incorrect resolution")
        """
        cypher = """
MATCH ()-[r:CALLS {id: $call_edge_id}]->()
RETURN r.confidence AS confidence
"""
        with self._driver.session() as session:
            result = session.run(cypher, call_edge_id=call_edge_id)
            record = result.single()
            if record and record["confidence"] is not None:
                return float(record["confidence"])
        return None

    # ------------------------------------------------------------------
    # Text analysis — the "cheat" tool for agents
    # ------------------------------------------------------------------

    def analyze_text(
        self,
        text: str,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Find code nodes relevant to arbitrary text input.

        Designed for the "cheat" scenario: pass in any text — a stack trace,
        error message, CLI output, linter warning, or natural language
        description — and get back the most relevant code nodes.

        Strategy:
        1. Truncate the text to a reasonable length for embedding.
        2. Extract identifiers that look like code symbols (CamelCase, snake_case)
           from the first 2000 chars and prepend them to the query.
        3. Run semantic search with the enriched query.

        Args:
            text: Arbitrary text to analyze (stack traces, logs, descriptions).
            limit: Maximum number of results to return (default 10, max 50).

        Returns:
            List of SearchResult ordered by relevance score.
        """
        import re

        limit = min(limit, 50)

        # Extract code-like identifiers from the text (CamelCase, snake_case, dotted)
        # to boost the semantic query with concrete symbol names.
        identifier_pattern = re.compile(
            r"\b([A-Z][a-zA-Z0-9]{2,}|[a-z][a-z0-9_]{2,}[A-Z][a-zA-Z0-9]*"
            r"|[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*)\b"
        )
        # Scan first 2000 chars for identifiers
        scan_text = text[:2000]
        identifiers = identifier_pattern.findall(scan_text)
        # Deduplicate, keep order, limit to 20 most frequent
        seen: set[str] = set()
        unique_ids: list[str] = []
        for ident in identifiers:
            if ident not in seen:
                seen.add(ident)
                unique_ids.append(ident)
            if len(unique_ids) >= 20:
                break

        # Build enriched query: identifiers first, then truncated text
        enriched_query = " ".join(unique_ids)
        if enriched_query:
            enriched_query += " | " + text[:500]
        else:
            enriched_query = text[:500]

        logger.debug(
            "analyze_text: extracted %d identifiers, query length=%d",
            len(unique_ids),
            len(enriched_query),
        )

        return self._embed.search(query=enriched_query, limit=limit)

    # ------------------------------------------------------------------
    # Signature-based search
    # ------------------------------------------------------------------

    def search_by_signature(
        self,
        pattern: str,
        lang: Optional[str] = None,
        limit: int = 20,
    ) -> list[ASTNode]:
        """Find functions/methods matching a signature pattern.

        Pattern syntax:
        - name: optional function name (wildcard * supported)
        - params: comma-separated type list in parentheses
        - return: arrow followed by return type

        Examples:
        - "*(int, String)" — any function with 2 params (int, String)
        - "map<T> -> T" — generic function map returning T
        - "toString() -> String" — exact signature
        - "process*" — any function starting with "process"

        Args:
            pattern: Signature pattern to match
            lang: Optional language filter (java, cpp, rust, etc.)
            limit: Maximum number of results (default 20, max 100)

        Returns:
            List of matching ASTNode (Function or Method kinds)
        """
        limit = min(limit, 100)
        parsed = self._parse_signature_pattern(pattern)

        # Build Cypher query
        lang_clause = "AND n.lang = $lang" if lang else ""

        # Build signature regex based on parsed pattern
        sig_regex = self._build_signature_regex(parsed)

        cypher = f"""
MATCH (n)
WHERE n.kind IN ['Function', 'Method']
  AND n.valid_to IS NULL
  AND n.signature =~ $sig_regex
  {lang_clause}
RETURN n
ORDER BY n.qualified_name
LIMIT $limit
"""
        params: dict = {
            "sig_regex": sig_regex,
            "limit": limit,
        }
        if lang:
            params["lang"] = lang

        results: list[ASTNode] = []
        with self._driver.session() as session:
            for record in session.run(cypher, **params):
                results.append(_record_to_node(dict(record["n"])))

        return results

    def _parse_signature_pattern(self, pattern: str) -> dict:
        """Parse a signature pattern string into components.

        Returns dict with:
        - name: function name pattern (may include wildcards)
        - params: list of parameter type patterns
        - return_type: return type pattern (may be None)
        - generics: generic type parameters (may be None)
        """
        result: dict = {
            "name": ".*",
            "params": [],
            "return_type": None,
            "generics": None,
        }

        # Extract return type (everything after ->)
        return_match = re.search(r"\s*->\s*(.+)$", pattern)
        if return_match:
            result["return_type"] = return_match.group(1).strip()
            pattern = pattern[: return_match.start()]

        # Extract function name and parameters
        # Pattern: name(params) - need to find matching parentheses
        pattern = pattern.strip()
        paren_start = pattern.find("(")
        paren_end = pattern.rfind(")")

        if paren_start != -1 and paren_end > paren_start:
            name_part = pattern[:paren_start].strip()
            params_part = pattern[paren_start + 1 : paren_end].strip()

            if name_part:
                if name_part == "*":
                    result["name"] = ".*"
                elif "*" in name_part:
                    result["name"] = name_part
                else:
                    result["name"] = name_part

            if params_part:
                # Split by comma, but respect nested generics like Map<String, List<Integer>>
                result["params"] = self._split_params_respecting_generics(params_part)
        elif pattern and pattern != "*":
            # No parentheses - might be just a name pattern like "process*"
            result["name"] = pattern if pattern != "*" else ".*"

        return result

    def _split_params_respecting_generics(self, params_str: str) -> list[str]:
        """Split parameter types by comma, respecting nested generics."""
        params: list[str] = []
        current: list[str] = []
        depth = 0

        for char in params_str:
            if char == "<":
                depth += 1
                current.append(char)
            elif char == ">":
                depth -= 1
                current.append(char)
            elif char == "," and depth == 0:
                param = "".join(current).strip()
                if param:
                    params.append(param)
                current = []
            else:
                current.append(char)

        # Don't forget the last parameter
        param = "".join(current).strip()
        if param:
            params.append(param)

        return params

    def _build_signature_regex(self, parsed: dict) -> str:
        """Build a Neo4j-compatible regex from parsed signature components.

        Neo4j uses Java regex syntax via =~ operator.
        """

        # Convert wildcard pattern to regex
        def wildcard_to_regex(pattern: str) -> str:
            # Escape regex special chars except *
            escaped = re.escape(pattern)
            # Convert \\* to .* for wildcard matching
            return escaped.replace(r"\*", ".*")

        name_pattern = wildcard_to_regex(parsed.get("name", ".*"))

        # Build parameter pattern
        params = parsed.get("params", [])
        if params:
            param_patterns = [wildcard_to_regex(p.strip()) for p in params]
            # Match params list: (param1, param2, ...)
            # Use flexible matching to allow for extra chars between params
            params_pattern = r"\(" + r",\s*".join(param_patterns) + r".*\)"
        else:
            # Match either no params () or any params like (int x, ...)
            params_pattern = r"\(.*\)"

        # Build return type pattern
        return_type = parsed.get("return_type")
        if return_type:
            return_pattern = r"->\s*" + wildcard_to_regex(return_type.strip())
        else:
            return_pattern = r"(?:->\s*.+)?"  # Optional return type

        # Combine: name(params) -> return_type
        # Signature format varies by language, but typically:
        # - Java: "public void methodName(int x, String y)"
        # - C++: "void methodName(int x, String y)"
        # - Rust: "fn method_name(&self, x: i32) -> Result<T>"
        full_pattern = f".*{name_pattern}{params_pattern}.*{return_pattern}"

        return full_pattern

    # ------------------------------------------------------------------
    # Find references (usages)
    # ------------------------------------------------------------------

    @track_latency(FIND_REFERENCES_LATENCY)
    def find_references(
        self,
        name: str,
        kind: Optional[str] = None,
        lang: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """Find all references/usages of a symbol (with pagination).

        Returns:
            dict with:
              - references: list of reference locations with context snippets
              - total: total count of references (for pagination)
              - limit: limit used
              - offset: offset used
              - has_more: boolean indicating more results exist
        """
        SEARCH_TOTAL.labels(lang=lang or "any", kind=kind or "any").inc()
        # 1. Find definition(s)
        definitions = self.find_definition(name, kind=kind, lang=lang)
        if not definitions:
            return {
                "references": [],
                "total": 0,
                "limit": limit,
                "offset": offset,
                "has_more": False,
            }

        # 2. For each definition, find references (CALLS, TYPES, etc.)
        all_refs = []
        for definition in definitions:
            refs = self._find_usages_of_node(definition.id)
            all_refs.extend(refs)

        # 3. Apply pagination
        total = len(all_refs)
        paginated_refs = all_refs[offset : offset + limit]

        return {
            "references": paginated_refs,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": total > offset + limit,
        }

    def _find_usages_of_node(self, node_id: str) -> list[dict]:
        """Find all usages/references of a given node.

        Searches for:
        - CALLS edges pointing to this node (function/method calls)
        - TYPES edges pointing to this node (type annotations)
        - INHERITS/EXTENDS/IMPLEMENTS edges (for classes/traits)

        Returns:
            List of reference dicts with location and context
        """
        node = self.get_node(node_id)
        if not node:
            return []

        references = []

        # Query for incoming CALLS edges (who calls this function/method)
        calls_cypher = """
MATCH (caller)-[r:CALLS]->(target {id: $node_id})
WHERE caller.valid_to IS NULL AND r.valid_to IS NULL
RETURN caller, r
ORDER BY caller.qualified_name
LIMIT 200
"""
        with self._driver.session() as session:
            for record in session.run(calls_cypher, node_id=node_id):
                caller_data = dict(record["caller"])
                edge_data = dict(record["r"])
                references.append(
                    {
                        "type": "CALLS",
                        "from_node": _record_to_node(caller_data),
                        "edge": {
                            "id": edge_data.get("id", ""),
                            "kind": edge_data.get("kind", "CALLS"),
                            "confidence": edge_data.get("confidence"),
                        },
                    }
                )

        # Query for incoming TYPES edges (who uses this as a type)
        types_cypher = """
MATCH (user)-[r:TYPES]->(target {id: $node_id})
WHERE user.valid_to IS NULL AND r.valid_to IS NULL
RETURN user, r
ORDER BY user.qualified_name
LIMIT 200
"""
        with self._driver.session() as session:
            for record in session.run(types_cypher, node_id=node_id):
                user_data = dict(record["user"])
                edge_data = dict(record["r"])
                references.append(
                    {
                        "type": "TYPES",
                        "from_node": _record_to_node(user_data),
                        "edge": {
                            "id": edge_data.get("id", ""),
                            "kind": edge_data.get("kind", "TYPES"),
                            "raw_type_string": edge_data.get("raw_type_string"),
                        },
                    }
                )

        # Query for incoming INHERITS/EXTENDS/IMPLEMENTS edges (for classes)
        inherits_cypher = """
MATCH (child)-[r:INHERITS|EXTENDS|IMPLEMENTS]->(parent {id: $node_id})
WHERE child.valid_to IS NULL AND r.valid_to IS NULL
RETURN child, r, type(r) as rel_type
ORDER BY child.qualified_name
LIMIT 200
"""
        with self._driver.session() as session:
            for record in session.run(inherits_cypher, node_id=node_id):
                child_data = dict(record["child"])
                edge_data = dict(record["r"])
                references.append(
                    {
                        "type": record["rel_type"],
                        "from_node": _record_to_node(child_data),
                        "edge": {
                            "id": edge_data.get("id", ""),
                            "kind": edge_data.get("kind", record["rel_type"]),
                        },
                    }
                )

        # Convert references to serializable dicts
        result = []
        for ref in references:
            from_node = ref["from_node"]
            result.append(
                {
                    "reference_type": ref["type"],
                    "node": {
                        "id": from_node.id,
                        "kind": from_node.kind.value,
                        "name": from_node.name,
                        "qualified_name": from_node.qualified_name,
                        "lang": from_node.lang.value,
                        "file_path": from_node.file_path,
                        "start_line": from_node.start_line,
                        "end_line": from_node.end_line,
                    },
                    "edge": ref["edge"],
                }
            )

        return result

    # ------------------------------------------------------------------
    # Git diff - code evolution analysis
    # ------------------------------------------------------------------

    def get_diff(
        self,
        repo_path: str,
        from_commit: str,
        to_commit: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        """Compute AST-level diff between two git commits (with pagination).

        This is a read-only operation that returns structured diff information
        without modifying the database. Useful for understanding code evolution
        and change impact.

        Args:
            repo_path: Path to the git repository
            from_commit: Starting commit hash (old)
            to_commit: Ending commit hash (new). Defaults to HEAD if not provided.
            limit: Maximum number of changes to return per page (default 100, max 1000)
            offset: Number of changes to skip for pagination (default 0)

        Returns:
            Dictionary with keys:
              - added_count, deleted_count, updated_count: Total counts
              - added: list of added node dicts
              - deleted: list of deleted node dicts (with id)
              - updated: list of updated node dicts
              - added_edges, deleted_edges, updated_edges: Edge change counts
              - limit, offset: Pagination parameters used
              - has_more: boolean indicating more results exist
              - from_commit, to_commit: Commit hashes used
        """
        import git

        # Validate and resolve commits
        limit = min(limit, 1000)  # Enforce max limit
        limit = max(1, limit)  # Ensure positive
        offset = max(0, offset)  # Ensure non-negative

        try:
            repo = git.Repo(repo_path)
        except (git.InvalidGitRepositoryError, OSError):
            logger.error(f"Not a valid git repository: {repo_path}")
            return {
                "error": f"Not a valid git repository: {repo_path}",
                "added_count": 0,
                "deleted_count": 0,
                "updated_count": 0,
                "added_edges": 0,
                "deleted_edges": 0,
                "updated_edges": 0,
                "has_more": False,
            }

        # Resolve to_commit to HEAD if not provided
        if to_commit is None or to_commit == "":
            try:
                to_commit = repo.head.commit.hexsha
            except (git.InvalidGitRepositoryError, ValueError):
                logger.error("Could not resolve HEAD commit")
                return {
                    "error": "Could not resolve HEAD commit. Repository may be empty.",
                    "added_count": 0,
                    "deleted_count": 0,
                    "updated_count": 0,
                    "added_edges": 0,
                    "deleted_edges": 0,
                    "updated_edges": 0,
                    "has_more": False,
                }

        # Validate commits exist
        try:
            repo.commit(from_commit)
        except (git.BadObject, git.GitCommandError):
            logger.error(f"Invalid from_commit: {from_commit}")
            return {
                "error": f"Invalid from_commit: {from_commit}",
                "added_count": 0,
                "deleted_count": 0,
                "updated_count": 0,
                "added_edges": 0,
                "deleted_edges": 0,
                "updated_edges": 0,
                "has_more": False,
            }

        try:
            repo.commit(to_commit)
        except (git.BadObject, git.GitCommandError):
            logger.error(f"Invalid to_commit: {to_commit}")
            return {
                "error": f"Invalid to_commit: {to_commit}",
                "added_count": 0,
                "deleted_count": 0,
                "updated_count": 0,
                "added_edges": 0,
                "deleted_edges": 0,
                "updated_edges": 0,
                "has_more": False,
            }

        # Compute the diff
        diff = compute_diff_for_commits(repo_path, from_commit, to_commit)

        # Helper to convert node to dict
        def node_to_dict(node: ASTNode) -> dict:
            return {
                "id": node.id,
                "kind": node.kind.value,
                "name": node.name,
                "qualified_name": node.qualified_name,
                "lang": node.lang.value,
                "file_path": node.file_path,
                "start_line": node.start_line,
                "end_line": node.end_line,
                "signature": node.signature or "",
            }

        # Apply pagination
        total_added = len(diff.added_nodes)
        total_deleted = len(diff.deleted_node_ids)
        total_updated = len(diff.updated_nodes)

        paginated_added = diff.added_nodes[offset : offset + limit]
        paginated_deleted_ids = diff.deleted_node_ids[offset : offset + limit]
        paginated_updated = diff.updated_nodes[offset : offset + limit]

        # Calculate has_more
        has_more = (
            total_added > offset + limit
            or total_deleted > offset + limit
            or total_updated > offset + limit
        )

        return {
            "added_count": total_added,
            "deleted_count": total_deleted,
            "updated_count": total_updated,
            "added_edges": len(diff.added_edges),
            "deleted_edges": len(diff.deleted_edge_ids),
            "updated_edges": len(diff.updated_edges),
            "added": [node_to_dict(n) for n in paginated_added],
            "deleted": [{"id": nid} for nid in paginated_deleted_ids],
            "updated": [node_to_dict(n) for n in paginated_updated],
            "limit": limit,
            "offset": offset,
            "has_more": has_more,
            "from_commit": from_commit,
            "to_commit": to_commit,
        }
