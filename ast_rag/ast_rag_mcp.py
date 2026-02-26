"""
ast_rag_mcp.py - MCP Server for AST-RAG code analysis.

Provides tools for semantic search, definition lookup, call graph traversal,
and text analysis via the Model Context Protocol (MCP).

Tools:
- ping: Returns server version and status
- index_project: Full indexing of a codebase
- update_project: Incremental update from git diff
- get_diff: Compute AST-level diff between two git commits
- find_definition: Find definition by name
- find_references: Find all references/usages of a symbol
- find_callers: Find callers of a function
- find_callees: Find callees of a function
- semantic_search: Search by natural language
- get_code_snippet: Get source code snippet
- expand_neighbourhood: Get subgraph around a node
- analyze_text: Analyze arbitrary text (stack traces, errors, etc.)
- search_by_signature: Search functions/methods by signature pattern
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from ast_rag.models import ProjectConfig, StandardResult
from ast_rag.graph_schema import create_driver, apply_schema
from ast_rag.graph_updater import (
    full_index,
    update_from_git,
    compute_diff_for_commits,
)
from ast_rag.embeddings import EmbeddingManager
from ast_rag.ast_rag_api import ASTRagAPI
from ast_rag.ast_parser import ParserManager, walk_source_files

logger = logging.getLogger(__name__)

# MCP server version
VERSION = "0.2.0"
SCHEMA_VERSION = "1.0"

# Create MCP server
mcp = FastMCP("AST-RAG")

# Global API instance (lazy-initialized)
_api: Optional[ASTRagAPI] = None
_config: Optional[ProjectConfig] = None


def _get_api(config_path: Optional[str] = None) -> ASTRagAPI:
    """Get or create the API instance."""
    global _api, _config

    if _api is not None:
        return _api

    # Load config
    if config_path and Path(config_path).exists():
        _config = ProjectConfig.model_validate_json(Path(config_path).read_text())
    else:
        default = Path("ast_rag_config.json")
        if default.exists():
            _config = ProjectConfig.model_validate_json(default.read_text())
        else:
            _config = ProjectConfig()

    # Create driver and embedding manager
    driver = create_driver(_config.neo4j)
    embed = EmbeddingManager(_config.qdrant, _config.embedding, neo4j_driver=driver)
    _api = ASTRagAPI(driver, embed)

    return _api


@mcp.tool()
def ping() -> dict:
    """Return server version and status.

    Returns:
        dict with version, schema_version, and status
    """
    return {
        "version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
    }


@mcp.tool()
def index_project(
    path: str = ".",
    commit: str = "INIT",
    config_path: Optional[str] = None,
) -> dict:
    """Perform a full initial indexing of a codebase.

    Args:
        path: Root directory of the codebase to index
        commit: Commit hash label for this index
        config_path: Path to config JSON file

    Returns:
        dict with counts of files, nodes, and edges
    """
    cfg = _config or ProjectConfig()
    root = os.path.abspath(path)

    # Apply schema
    driver = create_driver(cfg.neo4j)
    apply_schema(driver)

    # Parse all source files
    pm = ParserManager()
    files = list(walk_source_files(root, exclude_dirs=cfg.exclude_patterns))

    all_nodes = []
    all_edges = []

    for fp, lang in files:
        tree = pm.parse_file(fp)
        if tree is None:
            continue
        with open(fp, "rb") as fh:
            source = fh.read()
        nodes = pm.extract_nodes(tree, fp, lang, source, commit)
        edges = pm.extract_edges(tree, nodes, fp, lang, source, commit)
        all_nodes.extend(nodes)
        all_edges.extend(edges)

    # Write to Neo4j
    full_index(driver, all_nodes, all_edges, commit_hash=commit)

    # Build embeddings
    embed = EmbeddingManager(cfg.qdrant, cfg.embedding, neo4j_driver=driver)
    embed.build_embeddings(all_nodes)

    return {
        "files": len(files),
        "nodes": len(all_nodes),
        "edges": len(all_edges),
        "status": "ok",
    }


@mcp.tool()
def update_project(
    path: str = ".",
    from_commit: str = "",
    to_commit: str = "",
    max_changed_nodes: int = 100000,
    config_path: Optional[str] = None,
) -> dict:
    """Incrementally update the index from git diff.

    Args:
        path: Root directory of the codebase
        from_commit: Old commit hash
        to_commit: New commit hash
        max_changed_nodes: Safety limit - abort if diff too large
        config_path: Path to config JSON file

    Returns:
        dict with counts of added, updated, deleted nodes/edges
        or error if limit exceeded
    """
    from ast_rag.graph_updater import compute_diff_for_commits

    dry_result = compute_diff_for_commits(
        path,
        from_commit,
        to_commit,
        dry_run=True,
        max_changed_nodes=max_changed_nodes,
    )

    if dry_result["exceeds_limit"]:
        return {
            "error": "Changes exceed max_changed_nodes limit",
            "stats": dry_result["stats"],
            "warning": dry_result.get("warning"),
        }

    cfg = _config or ProjectConfig()
    driver = create_driver(cfg.neo4j)

    diff = update_from_git(driver, path, from_commit, to_commit)

    # Update embeddings
    embed = EmbeddingManager(cfg.qdrant, cfg.embedding, neo4j_driver=driver)
    embed.update_embeddings(
        diff.added_nodes,
        diff.updated_nodes,
        diff.deleted_node_ids,
    )

    return {
        "added_nodes": len(diff.added_nodes),
        "updated_nodes": len(diff.updated_nodes),
        "deleted_nodes": len(diff.deleted_node_ids),
        "added_edges": len(diff.added_edges),
        "deleted_edges": len(diff.deleted_edge_ids),
        "status": "ok",
    }


@mcp.tool()
def update_project_dry_run(
    path: str = ".",
    from_commit: str = "",
    to_commit: str = "",
    max_changed_nodes: int = 100000,
    config_path: Optional[str] = None,
) -> dict:
    """Safety check: compute diff stats without applying changes.
    
    Use this before running update_project to ensure changes are reasonable.
    
    Args:
        path: Root directory of the codebase
        from_commit: Old commit hash
        to_commit: New commit hash
        max_changed_nodes: Maximum allowed node changes (default 100k)
        config_path: Path to config JSON file
    
    Returns:
        Dict with stats and warning if limit exceeded.
    """
    from ast_rag.graph_updater import compute_diff_for_commits
    
    result = compute_diff_for_commits(
        path, from_commit, to_commit,
        dry_run=True,
        max_changed_nodes=max_changed_nodes,
    )
    
    if result["exceeds_limit"]:
        result["warning"] = (
            f"⚠️ Estimated changes ({result['stats']['estimated_nodes']}) "
            f"exceed max_changed_nodes ({max_changed_nodes}). "
            f"Manual review required."
        )
    
    return result


@mcp.tool()
def get_diff(
    repo_path: str,
    from_commit: str,
    to_commit: str,
    limit: int = 100,
    offset: int = 0,
    config_path: Optional[str] = None,
) -> dict:
    """Compute AST-level diff between two git commits.

    This is a read-only operation that analyzes code changes between commits
    without modifying the database. Useful for understanding code evolution
    and change impact.

    Args:
        repo_path: Path to git repository
        from_commit: Starting commit hash (old)
        to_commit: Ending commit hash (new)
        limit: Maximum number of changes to return per page (default 100, max 1000)
        offset: Number of changes to skip for pagination (default 0)
        config_path: Path to config JSON file (unused, kept for compatibility)

    Returns:
        Dict with standardized results:
        {
            "added": List[StandardResult],
            "deleted": List[dict],  # Only IDs for deleted
            "updated": List[dict],  # old/new StandardResult
            "stats": {...}
        }
    """
    from ast_rag.models import StandardResult

    diff = compute_diff_for_commits(repo_path, from_commit, to_commit)

    # Convert added nodes
    added = [
        StandardResult(
            id=n.id,
            name=n.name,
            qualified_name=n.qualified_name,
            kind=n.kind.value,
            lang=n.lang.value,
            file_path=n.file_path,
            start_line=n.start_line,
            end_line=n.end_line,
            edge_type="ADDED",
        ).dict()
        for n in diff.added_nodes[offset : offset + limit]
    ]

    # Convert updated nodes
    updated = [
        {
            "old": {"id": n.id, "edge_type": "UPDATED_OLD"},
            "new": StandardResult(
                id=n.id,
                name=n.name,
                qualified_name=n.qualified_name,
                kind=n.kind.value,
                lang=n.lang.value,
                file_path=n.file_path,
                start_line=n.start_line,
                end_line=n.end_line,
                edge_type="UPDATED_NEW",
            ).dict(),
        }
        for n in diff.updated_nodes[offset : offset + limit]
    ]

    # Convert deleted node IDs
    deleted = [
        {"id": nid, "edge_type": "DELETED"}
        for nid in diff.deleted_node_ids[offset : offset + limit]
    ]

    return {
        "added": added,
        "deleted": deleted,
        "updated": updated,
        "stats": {
            "added_count": len(diff.added_nodes),
            "deleted_count": len(diff.deleted_node_ids),
            "updated_count": len(diff.updated_nodes),
            "added_edges": len(diff.added_edges),
            "deleted_edges": len(diff.deleted_edge_ids),
            "updated_edges": len(diff.updated_edges),
        },
    }


@mcp.tool()
def find_definition(
    name: str,
    kind: Optional[str] = None,
    lang: Optional[str] = None,
    config_path: Optional[str] = None,
) -> list[dict]:
    """Find definition of a symbol by name.

    Args:
        name: Symbol name to look up
        kind: Optional filter by node kind (Class, Function, Method, etc.)
        lang: Optional filter by language (python, java, etc.)
        config_path: Path to config JSON file

    Returns:
        List of matching definitions with file path and line numbers
    """
    api = _get_api(config_path)
    nodes = api.find_definition(name, kind=kind, lang=lang)

    return [
        {
            "id": n.id,
            "kind": n.kind.value,
            "name": n.name,
            "qualified_name": n.qualified_name,
            "lang": n.lang.value,
            "file_path": n.file_path,
            "start_line": n.start_line,
            "end_line": n.end_line,
        }
        for n in nodes
    ]


@mcp.tool()
def find_references(
    name: str,
    kind: Optional[str] = None,
    lang: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    config_path: Optional[str] = None,
) -> list[dict]:
    """Find all references/usages of a symbol.

    Returns: List[StandardResult] with edge_type field populated.
    """
    from ast_rag.models import NodeKind, StandardResult

    api = _get_api(config_path)

    # Enforce max limit
    if limit > 1000:
        limit = 1000

    kind_enum = NodeKind(kind) if kind else None
    results = api.find_references(name, kind=kind_enum, lang=lang, limit=limit, offset=offset)

    # Convert to StandardResult format
    standard_results = []
    for ref in results.get("references", []):
        node = ref.get("node", {})
        standard_results.append(
            StandardResult(
                id=node.get("id", ""),
                name=node.get("name", ""),
                qualified_name=node.get("qualified_name", ""),
                kind=node.get("kind", ""),
                lang=node.get("lang", ""),
                file_path=node.get("file_path", ""),
                start_line=node.get("start_line", 0),
                end_line=node.get("end_line", 0),
                edge_type=ref.get("reference_type"),
            ).dict()
        )

    return standard_results


@mcp.tool()
def find_callers(
    name: str,
    depth: int = 1,
    kind: Optional[str] = None,
    lang: Optional[str] = None,
    config_path: Optional[str] = None,
) -> list[dict]:
    """Find all callers of a function or method.

    Args:
        name: Function/method name
        depth: Call depth to traverse (1-5)
        kind: Optional filter by node kind
        lang: Optional filter by language
        config_path: Path to config JSON file

    Returns:
        List of caller nodes
    """
    api = _get_api(config_path)

    defs = api.find_definition(name, kind=kind, lang=lang)
    if not defs:
        return []

    callers = api.find_callers(defs[0].id, max_depth=depth)

    return [
        StandardResult(
            id=c.id,
            name=c.name,
            qualified_name=c.qualified_name,
            kind=c.kind.value,
            lang=c.lang.value,
            file_path=c.file_path,
            start_line=c.start_line,
            end_line=c.end_line,
            edge_type="CALLS",
        ).dict()
        for c in callers
    ]


@mcp.tool()
def find_callees(
    name: str,
    depth: int = 1,
    kind: Optional[str] = None,
    lang: Optional[str] = None,
    config_path: Optional[str] = None,
) -> list[dict]:
    """Find all functions called by a function.

    Args:
        name: Function/method name
        depth: Call depth to traverse (1-5)
        kind: Optional filter by node kind
        lang: Optional filter by language
        config_path: Path to config JSON file

    Returns:
        List of callee nodes
    """
    api = _get_api(config_path)

    defs = api.find_definition(name, kind=kind, lang=lang)
    if not defs:
        return []

    callees = api.find_callees(defs[0].id, max_depth=depth)

    return [
        StandardResult(
            id=c.id,
            name=c.name,
            qualified_name=c.qualified_name,
            kind=c.kind.value,
            lang=c.lang.value,
            file_path=c.file_path,
            start_line=c.start_line,
            end_line=c.end_line,
            edge_type="CALLS",
        ).dict()
        for c in callees
    ]


@mcp.tool()
def semantic_search(
    query: str,
    limit: int = 10,
    lang: Optional[str] = None,
    kind: Optional[str] = None,
    config_path: Optional[str] = None,
) -> list[dict]:
    """Semantic search for code by natural language.

    Returns: List[StandardResult] with score field populated.
    """
    from ast_rag.models import StandardResult

    api = _get_api(config_path)
    limit = min(limit, 50)
    results = api.search_semantic(query, limit=limit, lang=lang, kind=kind)

    # Convert to StandardResult format
    return [
        StandardResult(
            id=r.node.id,
            name=r.node.name,
            qualified_name=r.node.qualified_name,
            kind=r.node.kind.value,
            lang=r.node.lang.value,
            file_path=r.node.file_path,
            start_line=r.node.start_line,
            end_line=r.node.end_line,
            score=r.score,
        ).dict()
        for r in results
    ]


@mcp.tool()
def get_code_snippet(
    file_path: str,
    start_line: int,
    end_line: int,
    config_path: Optional[str] = None,
) -> str:
    """Get source code snippet from a file.

    Args:
        file_path: Path to source file
        start_line: Start line (1-indexed)
        end_line: End line (1-indexed)
        config_path: Path to config JSON file

    Returns:
        Source code snippet as string
    """
    api = _get_api(config_path)
    return api.get_code_snippet(file_path, start_line, end_line)


@mcp.tool()
def expand_neighbourhood(
    node_id: str,
    depth: int = 1,
    edge_types: Optional[list[str]] = None,
    config_path: Optional[str] = None,
) -> dict:
    """Get subgraph around a node (neighbourhood expansion).

    Args:
        node_id: Node ID to expand from
        depth: Number of hops (1-4)
        edge_types: Optional list of edge types to follow
        config_path: Path to config JSON file

    Returns:
        dict with nodes and edges in the neighbourhood
    """
    api = _get_api(config_path)
    subgraph = api.expand_neighbourhood(node_id, depth=depth, edge_types=edge_types)

    return {
        "nodes": [
            {
                "id": n.id,
                "kind": n.kind.value,
                "name": n.name,
                "qualified_name": n.qualified_name,
                "lang": n.lang.value,
                "file_path": n.file_path,
            }
            for n in subgraph.nodes
        ],
        "edges": [
            {
                "id": e.id,
                "kind": e.kind.value,
                "from_id": e.from_id,
                "to_id": e.to_id,
            }
            for e in subgraph.edges
        ],
    }


@mcp.tool()
def analyze_text(
    text: str,
    limit: int = 10,
    config_path: Optional[str] = None,
) -> list[dict]:
    """Analyze arbitrary text (stack traces, errors, logs) and find relevant code.

    Extracts code identifiers from the text and performs semantic search.
    Useful for debugging: paste an error message and get relevant code locations.

    Args:
        text: Arbitrary text to analyze (stack trace, error message, etc.)
        limit: Maximum results (default 10, max 50)
        config_path: Path to config JSON file

    Returns:
        List of relevant code nodes
    """
    api = _get_api(config_path)
    limit = min(limit, 50)
    results = api.analyze_text(text, limit=limit)

    return [
        {
            "id": r.node.id,
            "kind": r.node.kind.value,
            "name": r.node.name,
            "qualified_name": r.node.qualified_name,
            "lang": r.node.lang.value,
            "file_path": r.node.file_path,
            "start_line": r.node.start_line,
            "score": r.score,
        }
        for r in results
    ]


@mcp.tool()
def search_by_signature(
    signature: str,
    lang: Optional[str] = None,
    limit: int = 20,
    config_path: Optional[str] = None,
) -> list[dict]:
    """Search for functions/methods by signature pattern.

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
        signature: Pattern like "*(int, String)" or "map<T> -> T"
        lang: Optional language filter (java, cpp, rust, python, etc.)
        limit: Maximum results to return (default 20, max 100)
        config_path: Path to config JSON file

    Returns:
        List of matching function/method nodes with:
        - id, kind, name, qualified_name, lang, file_path, start_line, signature
    """
    from ast_rag.models import StandardResult
    
    api = _get_api(config_path)
    limit = min(limit, 100)
    results = api.search_by_signature(signature, lang=lang, limit=limit)

    return [
        StandardResult(
            id=n.id,
            name=n.name,
            qualified_name=n.qualified_name,
            kind=n.kind.value,
            lang=n.lang.value,
            file_path=n.file_path,
            start_line=n.start_line,
            end_line=n.end_line,
            metadata={"signature": n.signature},
        ).dict()
        for n in results
    ]


@mcp.tool()
def find_overrides(
    name: str,
    lang: Optional[str] = None,
    config_path: Optional[str] = None,
) -> list[dict]:
    """Find all override implementations of a virtual method.

    Traverses OVERRIDES edges to find all methods that override the given
    method. This is useful for understanding inheritance hierarchies and
    polymorphic behavior.

    Args:
        name: Method name to find overrides for
        lang: Optional language filter (java, cpp, etc.)
        config_path: Path to config JSON file

    Returns:
        List of overriding method nodes with metadata
    """
    api = _get_api(config_path)
    results = api.find_overrides(name, lang=lang)

    return [
        {
            "id": n.id,
            "kind": n.kind.value,
            "name": n.name,
            "qualified_name": n.qualified_name,
            "lang": n.lang.value,
            "file_path": n.file_path,
            "start_line": n.start_line,
            "end_line": n.end_line,
        }
        for n in results
    ]


@mcp.tool()
def get_call_confidence(
    from_name: str,
    to_name: str,
    config_path: Optional[str] = None,
) -> dict:
    """Get the confidence score for a function call relationship.

    Returns the confidence score (0.0-1.0) indicating how certain the
    call resolution is. Higher scores indicate more reliable call targets.

    Args:
        from_name: Calling function/method name
        to_name: Called function/method name
        config_path: Path to config JSON file

    Returns:
        Dictionary with:
        - confidence: float between 0.0 and 1.0
        - from_id, to_id: node IDs
        - edge_id: CALLS edge ID
    """
    api = _get_api(config_path)
    result = api.get_call_confidence(from_name, to_name)
    return result


def main():
    """Run the MCP server with stdio transport."""
    import sys
    import traceback

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )
    logger.info("Starting AST-RAG MCP Server v%s", VERSION)
    logger.info("Python: %s", sys.executable)
    logger.info("CWD: %s", os.getcwd())

    try:
        # Pre-initialize API to catch config errors early
        logger.info("Pre-initializing API...")
        _get_api()
        logger.info("API initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize API: %s", e)
        logger.error("Traceback: %s", traceback.format_exc())
        # Don't exit - let the server start anyway, tools will fail gracefully

    # Run with stdio transport
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
