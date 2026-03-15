"""AST-RAG Search Service.

Service layer for code search operations combining graph and vector search.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from neo4j import Driver

from ast_rag.dto import ASTNode, SearchResult, NodeKind, Language
from ast_rag.services.embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)


class SearchService:
    """Service for unified code search operations.

    This service provides a high-level interface for searching code
    using multiple strategies. It combines:
    - Semantic (vector) search
    - Exact name matching
    - Signature-based search
    - Text analysis (stack traces, errors)

    Example:
        >>> search_service = SearchService(driver, embedding_manager)
        >>> results = search_service.semantic_search("batch upsert nodes")
        >>> definitions = search_service.find_by_name("UserService")
    """

    def __init__(
        self,
        driver: Driver,
        embedding_manager: EmbeddingManager,
    ) -> None:
        """Initialize the SearchService.

        Args:
            driver: Neo4j driver for graph queries
            embedding_manager: EmbeddingManager for vector search
        """
        self._driver = driver
        self._embedding_manager = embedding_manager

    def semantic_search(
        self,
        query: str,
        limit: int = 10,
        lang_filter: Optional[str] = None,
        kind_filter: Optional[str] = None,
    ) -> list[SearchResult]:
        """Search code using semantic similarity.

        Args:
            query: Natural language or code query
            limit: Maximum results to return (default 10, max 50)
            lang_filter: Optional language filter
            kind_filter: Optional node kind filter

        Returns:
            List of SearchResult ordered by relevance score
        """
        # Delegate to EmbeddingManager's search method
        return self._embedding_manager.search(
            query=query,
            limit=min(limit, 50),  # Cap at 50
            lang_filter=lang_filter,
            kind_filter=kind_filter,
            auto_fallback=True,
        )

    def find_by_name(
        self,
        name: str,
        kind: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> list[ASTNode]:
        """Find nodes by exact or partial name match.

        Args:
            name: Node name to search for
            kind: Optional node kind filter
            lang: Optional language filter

        Returns:
            List of matching ASTNode
        """
        if not name:
            return []

        # Build Cypher query for name matching
        conditions = ["n.valid_to IS NULL"]
        params: dict[str, str] = {"name": name}

        if kind:
            conditions.append("n.kind = $kind")
            params["kind"] = kind

        if lang:
            conditions.append("n.lang = $lang")
            params["lang"] = lang

        where_clause = " AND ".join(conditions)

        cypher = f"""
MATCH (n)
WHERE {where_clause}
  AND (n.name = $name OR n.qualified_name CONTAINS $name)
RETURN n
ORDER BY n.name
LIMIT 100
"""

        results: list[ASTNode] = []
        try:
            with self._driver.session() as session:
                for record in session.run(cypher, **params):
                    node_data = dict(record["n"])
                    results.append(self._record_to_node(node_data))
        except Exception as exc:
            logger.warning("Name search failed: %s", exc)

        return results

    def search_by_signature(
        self,
        pattern: str,
        lang: Optional[str] = None,
        limit: int = 20,
    ) -> list[ASTNode]:
        """Search for functions/methods by signature pattern.

        Pattern syntax:
        - name: optional function name (wildcard * supported)
        - params: comma-separated type list in parentheses
        - return: arrow followed by return type

        Examples:
        - "*(int, String)" — any function with 2 params
        - "map<T> -> T" — generic function returning T
        - "toString() -> String" — exact signature
        - "process*" — any function starting with "process"

        Args:
            pattern: Signature pattern to match
            lang: Optional language filter
            limit: Maximum results to return

        Returns:
            List of matching ASTNode (Function or Method kinds)
        """
        if not pattern:
            return []

        # Parse pattern to extract name and signature parts
        # Simple implementation: treat as name pattern with optional signature
        name_pattern = pattern.split("(")[0].strip() if "(" in pattern else pattern
        signature_part = pattern[len(name_pattern):].strip() if "(" in pattern else ""

        conditions = ["n.valid_to IS NULL", "n.kind IN ['Function', 'Method', 'Constructor']"]
        params: dict[str, str] = {}

        if lang:
            conditions.append("n.lang = $lang")
            params["lang"] = lang

        # Handle name pattern (support * wildcard)
        if name_pattern and name_pattern != "*":
            if "*" in name_pattern:
                # Convert wildcard to regex
                regex_pattern = "^" + name_pattern.replace("*", ".*") + "$"
                conditions.append("n.name =~ $name_regex")
                params["name_regex"] = regex_pattern
            else:
                conditions.append("n.name = $name")
                params["name"] = name_pattern

        # Handle signature part if present
        if signature_part:
            # Simple signature matching - look for signature containing the pattern
            conditions.append("n.signature CONTAINS $signature")
            params["signature"] = signature_part

        where_clause = " AND ".join(conditions)

        cypher = f"""
MATCH (n)
WHERE {where_clause}
RETURN n
ORDER BY n.name
LIMIT $limit
"""
        params["limit"] = str(limit)

        results: list[ASTNode] = []
        try:
            with self._driver.session() as session:
                for record in session.run(cypher, **params):
                    node_data = dict(record["n"])
                    results.append(self._record_to_node(node_data))
        except Exception as exc:
            logger.warning("Signature search failed: %s", exc)

        return results

    def analyze_text(
        self,
        text: str,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Analyze arbitrary text and find relevant code.

        Designed for stack traces, error messages, logs, or any
        text output. Extracts identifiers and performs semantic search.

        Args:
            text: Arbitrary text to analyze (stack trace, error, etc.)
            limit: Maximum results to return

        Returns:
            List of SearchResult with relevant code nodes
        """
        if not text:
            return []

        # Extract identifiers from text (camelCase, snake_case, etc.)
        identifiers = self._extract_identifiers(text)

        if not identifiers:
            # Fallback to semantic search with full text
            return self.semantic_search(text, limit=limit)

        # Search for each identifier and combine results
        all_results: list[SearchResult] = []
        seen_ids: set[str] = set()

        for identifier in identifiers[:5]:  # Limit to top 5 identifiers
            results = self.find_by_name(identifier, limit=limit // 2)
            for node in results:
                if node.id not in seen_ids:
                    seen_ids.add(node.id)
                    # Create SearchResult with moderate score for name matches
                    all_results.append(SearchResult(node=node, score=0.7))

        # If we have few results, supplement with semantic search
        if len(all_results) < limit:
            semantic_results = self.semantic_search(text, limit=limit - len(all_results))
            for result in semantic_results:
                if result.node.id not in seen_ids:
                    seen_ids.add(result.node.id)
                    all_results.append(result)

        # Sort by score and return top results
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:limit]

    def _extract_identifiers(self, text: str) -> list[str]:
        """Extract identifiers from text (camelCase, snake_case, etc.)."""
        # Match camelCase, snake_case, and simple identifiers
        pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        identifiers = re.findall(pattern, text)

        # Filter out common words and short identifiers
        common_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who',
            'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but',
            'and', 'or', 'if', 'while', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
            'error', 'exception', 'failed', 'failure', 'success', 'warning', 'info',
            'debug', 'trace', 'log', 'message', 'code', 'line', 'file', 'function',
            'method', 'class', 'module', 'package', 'import', 'return', 'value',
        }

        # Filter and deduplicate
        filtered = []
        seen = set()
        for ident in identifiers:
            if (len(ident) > 2 and
                ident.lower() not in common_words and
                ident not in seen):
                filtered.append(ident)
                seen.add(ident)

        return filtered[:10]  # Return top 10 identifiers

    def search_blocks(
        self,
        block_type: Optional[str] = None,
        lang: Optional[str] = None,
        min_nesting: int = 0,
        limit: int = 50,
    ) -> list[dict]:
        """Search for code blocks (if/for/while/try/lambda).

        Args:
            block_type: Optional block type filter (if/for/while/try/lambda/with)
            lang: Optional language filter
            min_nesting: Minimum nesting depth
            limit: Maximum results to return

        Returns:
            List of block dictionaries with metadata
        """
        conditions = ["b.valid_to IS NULL"]
        params: dict[str, str] = {}

        if block_type:
            conditions.append("b.block_type = $block_type")
            params["block_type"] = block_type

        if lang:
            conditions.append("b.lang = $lang")
            params["lang"] = lang

        if min_nesting > 0:
            conditions.append("b.nesting_depth >= $min_nesting")
            params["min_nesting"] = str(min_nesting)

        where_clause = " AND ".join(conditions)

        cypher = f"""
MATCH (b:Block)
WHERE {where_clause}
RETURN b
ORDER BY b.file_path, b.start_line
LIMIT $limit
"""
        params["limit"] = str(limit)

        results: list[dict] = []
        try:
            with self._driver.session() as session:
                for record in session.run(cypher, **params):
                    block_data = dict(record["b"])
                    results.append({
                        "id": block_data.get("id", ""),
                        "block_type": block_data.get("block_type", ""),
                        "lang": block_data.get("lang", ""),
                        "file_path": block_data.get("file_path", ""),
                        "start_line": block_data.get("start_line", 0),
                        "end_line": block_data.get("end_line", 0),
                        "nesting_depth": block_data.get("nesting_depth", 0),
                        "parent_function_id": block_data.get("parent_function_id", ""),
                    })
        except Exception as exc:
            logger.warning("Block search failed: %s", exc)

        return results

    def find_lambdas(
        self,
        lang: Optional[str] = None,
        with_captured_vars: bool = True,
        limit: int = 50,
    ) -> list[dict]:
        """Find all lambda/closure expressions.

        Args:
            lang: Optional language filter (python/rust/typescript)
            with_captured_vars: Include captured variables
            limit: Maximum results to return

        Returns:
            List of lambda blocks with captured variables
        """
        conditions = ["b.valid_to IS NULL", "b.block_type = 'lambda'"]
        params: dict[str, str] = {}

        if lang:
            conditions.append("b.lang = $lang")
            params["lang"] = lang

        where_clause = " AND ".join(conditions)

        if with_captured_vars:
            cypher = f"""
MATCH (b:Block)
WHERE {where_clause}
OPTIONAL MATCH (b)-[:CAPTURES]->(v:Variable)
WITH b, collect(v.name) as captured_vars
RETURN b, captured_vars
ORDER BY b.file_path, b.start_line
LIMIT $limit
"""
        else:
            cypher = f"""
MATCH (b:Block)
WHERE {where_clause}
RETURN b
ORDER BY b.file_path, b.start_line
LIMIT $limit
"""
        params["limit"] = str(limit)

        results: list[dict] = []
        try:
            with self._driver.session() as session:
                for record in session.run(cypher, **params):
                    block_data = dict(record["b"])
                    result = {
                        "id": block_data.get("id", ""),
                        "lang": block_data.get("lang", ""),
                        "file_path": block_data.get("file_path", ""),
                        "start_line": block_data.get("start_line", 0),
                        "end_line": block_data.get("end_line", 0),
                        "nesting_depth": block_data.get("nesting_depth", 0),
                        "parent_function_id": block_data.get("parent_function_id", ""),
                    }
                    if with_captured_vars:
                        result["captured_vars"] = record.get("captured_vars", [])
                    results.append(result)
        except Exception as exc:
            logger.warning("Lambda search failed: %s", exc)

        return results

    def _record_to_node(self, node_data: dict) -> ASTNode:
        """Convert Neo4j node record to ASTNode."""
        return ASTNode(
            id=node_data.get("id", ""),
            kind=NodeKind(node_data.get("kind", "Function")),
            name=node_data.get("name", ""),
            qualified_name=node_data.get("qualified_name", ""),
            lang=Language(node_data.get("lang", "java")),
            file_path=node_data.get("file_path", ""),
            start_line=int(node_data.get("start_line", 0)),
            end_line=int(node_data.get("end_line", 0)),
            start_byte=int(node_data.get("start_byte", 0)),
            end_byte=int(node_data.get("end_byte", 0)),
            signature=node_data.get("signature") or None,
            valid_from=node_data.get("valid_from", "INIT"),
            valid_to=node_data.get("valid_to"),
        )
