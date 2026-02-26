"""
graph_updater.py - MVCC-based incremental graph update engine.

Key responsibilities:
1. compute_diff: compare old vs new ASTNode/ASTEdge lists → DiffResult
2. apply_diff:   apply DiffResult to Neo4j in a single transaction (MVCC semantics)
3. full_index:   initial bulk load of all nodes and edges
4. update_from_git: orchestrate git diff → re-parse → compute_diff → apply_diff

MVCC semantics:
- Active records have valid_to = NULL.
- Expired records have valid_to = commit_hash that made them obsolete.
- New/updated records have valid_from = new_commit_hash.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import git

from neo4j import Driver, Session

from ast_rag.models import ASTNode, ASTEdge, DiffResult
from ast_rag.graph_schema import (
    batch_upsert_nodes,
    batch_expire_nodes,
    batch_upsert_edges,
    batch_expire_edges,
    ensure_current_version,
    _KIND_TO_LABEL,
)
from ast_rag.ast_parser import ParserManager, walk_source_files
from ast_rag.file_cache import (
    init_file_cache,
    file_changed_since_last_index,
    update_file_cache,
    save_file_cache,
)
from ast_rag.metrics import (
    track_latency,
    UPDATE_LATENCY,
    UPDATE_TOTAL,
    SKIP_RATIO,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Diff computation
# ---------------------------------------------------------------------------


def compute_diff(
    old_nodes: list[ASTNode],
    new_nodes: list[ASTNode],
    old_edges: list[ASTEdge],
    new_edges: list[ASTEdge],
    new_commit_hash: str,
) -> DiffResult:
    """Compare old and new extraction results and compute a DiffResult.

    Matching is done by stable node id.  A node is 'updated' if its id
    matches but code_hash differs.
    """
    old_by_id: dict[str, ASTNode] = {n.id: n for n in old_nodes}
    new_by_id: dict[str, ASTNode] = {n.id: n for n in new_nodes}

    old_edge_by_id: dict[str, ASTEdge] = {e.id: e for e in old_edges}
    new_edge_by_id: dict[str, ASTEdge] = {e.id: e for e in new_edges}

    added_nodes: list[ASTNode] = []
    deleted_node_ids: list[str] = []
    updated_nodes: list[ASTNode] = []
    old_updated_ids: list[str] = []

    for nid, new_node in new_by_id.items():
        if nid not in old_by_id:
            # Brand-new node
            n = new_node.model_copy(update={"valid_from": new_commit_hash})
            added_nodes.append(n)
        elif old_by_id[nid].code_hash != new_node.code_hash:
            # Updated node: expire old, create new version
            old_updated_ids.append(nid)
            n = new_node.model_copy(update={"valid_from": new_commit_hash})
            updated_nodes.append(n)

    for nid in old_by_id:
        if nid not in new_by_id:
            deleted_node_ids.append(nid)

    # Edge diffing (by id only — edges don't have code_hash)
    added_edges: list[ASTEdge] = []
    deleted_edge_ids: list[str] = []
    updated_edges: list[ASTEdge] = []
    old_updated_edge_ids: list[str] = []

    for eid, new_edge in new_edge_by_id.items():
        if eid not in old_edge_by_id:
            e = new_edge.model_copy(update={"valid_from": new_commit_hash})
            added_edges.append(e)
    for eid in old_edge_by_id:
        if eid not in new_edge_by_id:
            deleted_edge_ids.append(eid)

    return DiffResult(
        added_nodes=added_nodes,
        deleted_node_ids=deleted_node_ids,
        updated_nodes=updated_nodes,
        old_updated_node_ids=old_updated_ids,
        added_edges=added_edges,
        deleted_edge_ids=deleted_edge_ids,
        updated_edges=updated_edges,
        old_updated_edge_ids=old_updated_edge_ids,
    )


# ---------------------------------------------------------------------------
# Apply diff to Neo4j (MVCC transaction)
# ---------------------------------------------------------------------------


def _nodes_to_batch_by_label(nodes: list[ASTNode]) -> dict[str, list[dict]]:
    """Group node property dicts by their Neo4j label."""
    by_label: dict[str, list[dict]] = {}
    for node in nodes:
        label = _KIND_TO_LABEL.get(node.kind.value, "Class")
        by_label.setdefault(label, []).append(node.to_neo4j_props())
    return by_label


def _expired_nodes_by_label(ids: list[str], existing_nodes: list[ASTNode]) -> dict[str, list[str]]:
    """Build {label: [id, ...]} for nodes that need to be expired."""
    id_to_node: dict[str, ASTNode] = {n.id: n for n in existing_nodes}
    by_label: dict[str, list[str]] = {}
    for nid in ids:
        node = id_to_node.get(nid)
        if node:
            label = _KIND_TO_LABEL.get(node.kind.value, "Class")
            by_label.setdefault(label, []).append(nid)
    return by_label


def apply_diff(
    session: Session,
    diff: DiffResult,
    new_commit_hash: str,
    all_old_nodes: Optional[list[ASTNode]] = None,
    all_old_edges: Optional[list[ASTEdge]] = None,
) -> None:
    """Apply a DiffResult to Neo4j in a single session (caller manages transaction).

    For expiration of nodes/edges we need to know their label,
    so all_old_nodes / all_old_edges should be provided when possible.
    """
    if diff.is_empty:
        logger.debug("Diff is empty; nothing to apply.")
        return

    # 1. Expire deleted nodes
    if diff.deleted_node_ids:
        if all_old_nodes:
            by_label = _expired_nodes_by_label(diff.deleted_node_ids, all_old_nodes)
            batch_expire_nodes(session, by_label, new_commit_hash)
        else:
            # Fallback: expire without label filtering (slower but works)
            _expire_nodes_generic(session, diff.deleted_node_ids, new_commit_hash)

    # 2. Expire old versions of updated nodes
    if diff.old_updated_node_ids:
        if all_old_nodes:
            by_label = _expired_nodes_by_label(diff.old_updated_node_ids, all_old_nodes)
            batch_expire_nodes(session, by_label, new_commit_hash)
        else:
            _expire_nodes_generic(session, diff.old_updated_node_ids, new_commit_hash)

    # 3. Create new nodes (added + updated versions)
    new_nodes = diff.added_nodes + diff.updated_nodes
    if new_nodes:
        batch_upsert_nodes(session, _nodes_to_batch_by_label(new_nodes))

    # 4. Expire deleted edges
    if diff.deleted_edge_ids:
        batch_expire_edges(session, diff.deleted_edge_ids, new_commit_hash)
    if diff.old_updated_edge_ids:
        batch_expire_edges(session, diff.old_updated_edge_ids, new_commit_hash)

    # 5. Create new edges
    new_edges = diff.added_edges + diff.updated_edges
    if new_edges:
        edge_dicts = [e.to_neo4j_props() for e in new_edges]
        batch_upsert_edges(session, edge_dicts)


def _expire_nodes_generic(session: Session, ids: list[str], commit_hash: str) -> None:
    """Expire nodes across all labels (label-agnostic fallback)."""
    session.run(
        """
        UNWIND $ids AS nid
        MATCH (n {id: nid})
        WHERE n.valid_to IS NULL
        SET n.valid_to = $commit_hash
        """,
        ids=ids,
        commit_hash=commit_hash,
    )


# ---------------------------------------------------------------------------
# Full initial index
# ---------------------------------------------------------------------------


def full_index(
    driver: Driver,
    nodes: list[ASTNode],
    edges: list[ASTEdge],
    commit_hash: str = "INIT",
    batch_size: int = 500,
    file_path: Optional[str] = None,
) -> None:
    """Bulk-load all nodes and edges into an empty (or cleared) Neo4j graph.

    This is called during `ast-rag init`.  It does NOT apply MVCC expiration
    because we assume a clean slate.

    Args:
        driver: Neo4j driver
        nodes: List of ASTNode to index
        edges: List of ASTEdge to index
        commit_hash: Commit hash label
        batch_size: Batch size for Neo4j operations
        file_path: Optional file path for cache update
    """
    logger.info("Full index: %d nodes, %d edges", len(nodes), len(edges))

    # Set valid_from on all nodes
    stamped_nodes = [n.model_copy(update={"valid_from": commit_hash}) for n in nodes]
    stamped_edges = [e.model_copy(update={"valid_from": commit_hash}) for e in edges]

    with driver.session() as session:
        # Nodes in batches
        by_label = _nodes_to_batch_by_label(stamped_nodes)
        for label, props_list in by_label.items():
            for i in range(0, len(props_list), batch_size):
                chunk = props_list[i : i + batch_size]
                batch_upsert_nodes(session, {label: chunk})
                logger.debug("Upserted %d %s nodes", len(chunk), label)

        # Edges in batches
        all_edge_dicts = [e.to_neo4j_props() for e in stamped_edges]
        for i in range(0, len(all_edge_dicts), batch_size):
            chunk = all_edge_dicts[i : i + batch_size]
            batch_upsert_edges(session, chunk)
            logger.debug("Upserted %d edges (batch %d)", len(chunk), i // batch_size)

        ensure_current_version(session, commit_hash)

    # Update file cache
    if file_path:
        update_file_cache(file_path)

    logger.info("Full index complete.")


# ---------------------------------------------------------------------------
# Git-diff based incremental update
# ---------------------------------------------------------------------------


@track_latency(UPDATE_LATENCY)
def update_from_git(
    driver: Driver,
    repo_path: str,
    old_commit: str,
    new_commit: str,
    exclude_dirs: Optional[list[str]] = None,
) -> DiffResult:
    """Orchestrate an incremental update from git diff OLD..NEW.

    Returns the aggregated DiffResult (for use by the embeddings layer).

    Skips files that haven't changed since last index (optimization).
    """
    import git  # GitPython

    try:
        # Initialize file cache
        init_file_cache(repo_path)

        repo = git.Repo(repo_path)
        old_rev = repo.commit(old_commit)
        new_rev = repo.commit(new_commit)

        # Collect changed files
        diff_index = old_rev.diff(new_rev)
        changed_paths: list[str] = []
        for diff_item in diff_index:
            # a_path = old, b_path = new
            if diff_item.b_path:
                changed_paths.append(os.path.join(repo_path, diff_item.b_path))
            elif diff_item.a_path:
                changed_paths.append(os.path.join(repo_path, diff_item.a_path))

        if not changed_paths:
            logger.info("No changed files between %s and %s", old_commit[:8], new_commit[:8])
            UPDATE_TOTAL.labels(status="success").inc()
            return DiffResult()

        # Filter out unchanged files (optimization)
        filtered_paths = []
        for file_path in changed_paths:
            if file_changed_since_last_index(file_path):
                filtered_paths.append(file_path)
            else:
                logger.debug("Skipping unchanged file: %s", file_path)

        if not filtered_paths:
            logger.info("All %d changed files are unchanged in cache - skipping", len(changed_paths))
            UPDATE_TOTAL.labels(status="success").inc()
            return DiffResult()

        # Update skip ratio gauge
        skip_ratio = 1 - len(filtered_paths) / len(changed_paths)
        SKIP_RATIO.set(skip_ratio)

        logger.info(
            f"Changed files: {len(changed_paths)} total, "
            f"{len(filtered_paths)} after cache filter "
            f"({100 - len(filtered_paths) / len(changed_paths) * 100:.1f}% skipped)"
        )

        pm = ParserManager()
        agg_diff = DiffResult()

        for file_path in filtered_paths:
            lang = pm.detect_language(file_path)
            if lang is None:
                continue

            # Load new content (may no longer exist if deleted)
            if not os.path.exists(file_path):
                # File was deleted: all old nodes from this file need expiration.
                # For PoC: we mark them by querying Neo4j.
                _expire_file_nodes(driver, file_path, new_commit)
                continue

            with open(file_path, "rb") as fh:
                new_source = fh.read()

            # Parse the NEW version
            new_tree = pm.parse_file(file_path, source=new_source)
            if new_tree is None:
                continue
            new_nodes = pm.extract_nodes(new_tree, file_path, lang, new_source, new_commit)
            new_edges = pm.extract_edges(new_tree, new_nodes, file_path, lang, new_source, new_commit)

            # Load old version from git at old_commit
            rel_path = os.path.relpath(file_path, repo_path)
            old_source_bytes = _read_blob(repo, old_commit, rel_path)
            old_nodes: list[ASTNode] = []
            old_edges: list[ASTEdge] = []
            if old_source_bytes is not None:
                old_tree = pm.parse_file(file_path, source=old_source_bytes)
                if old_tree:
                    old_nodes = pm.extract_nodes(
                        old_tree, file_path, lang, old_source_bytes, old_commit
                    )
                    old_edges = pm.extract_edges(
                        old_tree, old_nodes, file_path, lang, old_source_bytes, old_commit
                    )

            file_diff = compute_diff(old_nodes, new_nodes, old_edges, new_edges, new_commit)

            # Accumulate
            agg_diff.added_nodes.extend(file_diff.added_nodes)
            agg_diff.deleted_node_ids.extend(file_diff.deleted_node_ids)
            agg_diff.updated_nodes.extend(file_diff.updated_nodes)
            agg_diff.old_updated_node_ids.extend(file_diff.old_updated_node_ids)
            agg_diff.added_edges.extend(file_diff.added_edges)
            agg_diff.deleted_edge_ids.extend(file_diff.deleted_edge_ids)
            agg_diff.updated_edges.extend(file_diff.updated_edges)
            agg_diff.old_updated_edge_ids.extend(file_diff.old_updated_edge_ids)

            # Update cache for this file
            update_file_cache(file_path)

        # Save cache
        save_file_cache()

        # Apply the accumulated diff in one transaction + update CurrentVersion
        _apply_agg_diff(driver, agg_diff, new_commit)

        logger.info(
            f"Update complete: "
            f"{len(agg_diff.added_nodes)} added, "
            f"{len(agg_diff.updated_nodes)} updated, "
            f"{len(agg_diff.deleted_node_ids)} deleted"
        )
        UPDATE_TOTAL.labels(status="success").inc()
        return agg_diff

    except Exception:
        UPDATE_TOTAL.labels(status="error").inc()
        raise


def compute_diff_for_commits(
    repo_path: str,
    from_commit: str,
    to_commit: str,
    exclude_dirs: Optional[list[str]] = None,
    dry_run: bool = False,
    max_changed_nodes: int = 100000,
) -> DiffResult | dict:
    """Compute AST-level diff between two git commits without applying to database.

    This is a read-only operation that returns a DiffResult containing:
    - added_nodes: New AST nodes in to_commit
    - deleted_node_ids: Node IDs that exist in from_commit but not in to_commit
    - updated_nodes: Modified nodes (new versions)
    - old_updated_node_ids: Old IDs of updated nodes (for expiration if needed)
    - added_edges: New edges in to_commit
    - deleted_edge_ids: Edge IDs that exist in from_commit but not in to_commit
    - updated_edges: Modified edges (new versions)
    - old_updated_edge_ids: Old IDs of updated edges

    Args:
        repo_path: Path to git repository
        from_commit: Starting commit hash (old)
        to_commit: Ending commit hash (new)
        exclude_dirs: Optional list of directories to exclude
        dry_run: If True, only compute stats without full diff
        max_changed_nodes: Safety limit for estimation check

    Returns:
        If dry_run=True: {"stats": {...}, "exceeds_limit": bool}
        If dry_run=False: DiffResult with all changes between the two commits
    """
    import git  # GitPython

    repo = git.Repo(repo_path)
    old_rev = repo.commit(from_commit)
    new_rev = repo.commit(to_commit)

    # Collect changed files
    diff_index = old_rev.diff(new_rev)
    changed_paths: list[str] = []
    for diff_item in diff_index:
        if diff_item.b_path:
            changed_paths.append(os.path.join(repo_path, diff_item.b_path))
        elif diff_item.a_path:
            changed_paths.append(os.path.join(repo_path, diff_item.a_path))

    if not changed_paths:
        logger.info("No changed files between %s and %s", from_commit[:8], to_commit[:8])
        if dry_run:
            return {
                "stats": {
                    "changed_files": 0,
                    "estimated_nodes": 0,
                    "from_commit": from_commit[:8],
                    "to_commit": to_commit[:8],
                },
                "exceeds_limit": False,
            }
        return DiffResult()

    logger.info(
        "%d changed files between %s..%s",
        len(changed_paths),
        from_commit[:8],
        to_commit[:8],
    )

    if dry_run:
        # Quick estimate (without full parsing)
        estimated_changes = len(changed_paths) * 50  # Rough estimate: 50 nodes/file
        result = {
            "stats": {
                "changed_files": len(changed_paths),
                "estimated_nodes": estimated_changes,
                "from_commit": from_commit[:8],
                "to_commit": to_commit[:8],
            },
            "exceeds_limit": estimated_changes > max_changed_nodes,
        }
        if estimated_changes > max_changed_nodes:
            logger.warning(
                "Estimated %d nodes exceeds limit of %d", estimated_changes, max_changed_nodes
            )
        return result

    pm = ParserManager()
    agg_diff = DiffResult()

    for file_path in changed_paths:
        lang = pm.detect_language(file_path)
        if lang is None:
            continue

        # Load new content (may no longer exist if deleted)
        if not os.path.exists(file_path):
            # File was deleted - we still process it to mark nodes for expiration
            # Load old version from git
            rel_path = os.path.relpath(file_path, repo_path)
            old_source_bytes = _read_blob(repo, from_commit, rel_path)
            if old_source_bytes is not None:
                old_tree = pm.parse_file(file_path, source=old_source_bytes)
                if old_tree:
                    old_nodes = pm.extract_nodes(
                        old_tree, file_path, lang, old_source_bytes, from_commit
                    )
                    old_edges = pm.extract_edges(
                        old_tree, old_nodes, file_path, lang, old_source_bytes, from_commit
                    )
                    # All old nodes are deleted
                    file_diff = compute_diff([], [], [], [], to_commit)
                    file_diff.deleted_node_ids = [n.id for n in old_nodes]
                    file_diff.deleted_edge_ids = [e.id for e in old_edges]
                    agg_diff.deleted_node_ids.extend(file_diff.deleted_node_ids)
                    agg_diff.deleted_edge_ids.extend(file_diff.deleted_edge_ids)
            continue

        with open(file_path, "rb") as fh:
            new_source = fh.read()

        # Parse the NEW version
        new_tree = pm.parse_file(file_path, source=new_source)
        if new_tree is None:
            continue
        new_nodes = pm.extract_nodes(new_tree, file_path, lang, new_source, to_commit)
        new_edges = pm.extract_edges(new_tree, new_nodes, file_path, lang, new_source, to_commit)

        # Load old version from git at from_commit
        rel_path = os.path.relpath(file_path, repo_path)
        old_source_bytes = _read_blob(repo, from_commit, rel_path)
        old_nodes: list[ASTNode] = []
        old_edges: list[ASTEdge] = []
        if old_source_bytes is not None:
            old_tree = pm.parse_file(file_path, source=old_source_bytes)
            if old_tree:
                old_nodes = pm.extract_nodes(
                    old_tree, file_path, lang, old_source_bytes, from_commit
                )
                old_edges = pm.extract_edges(
                    old_tree, old_nodes, file_path, lang, old_source_bytes, from_commit
                )

        file_diff = compute_diff(old_nodes, new_nodes, old_edges, new_edges, to_commit)

        # Accumulate
        agg_diff.added_nodes.extend(file_diff.added_nodes)
        agg_diff.deleted_node_ids.extend(file_diff.deleted_node_ids)
        agg_diff.updated_nodes.extend(file_diff.updated_nodes)
        agg_diff.old_updated_node_ids.extend(file_diff.old_updated_node_ids)
        agg_diff.added_edges.extend(file_diff.added_edges)
        agg_diff.deleted_edge_ids.extend(file_diff.deleted_edge_ids)
        agg_diff.updated_edges.extend(file_diff.updated_edges)
        agg_diff.old_updated_edge_ids.extend(file_diff.old_updated_edge_ids)

    return agg_diff


def _apply_agg_diff(driver: Driver, diff: DiffResult, new_commit: str) -> None:
    """Write the aggregated diff and update CurrentVersion atomically."""
    with driver.session() as session:
        # For PoC use auto-commit session calls:
        apply_diff(session, diff, new_commit)
        ensure_current_version(session, new_commit)


def _expire_file_nodes(driver: Driver, file_path: str, commit_hash: str) -> None:
    """Expire all active nodes that belong to a deleted file."""
    with driver.session() as session:
        session.run(
            """
            MATCH (n)
            WHERE n.file_path = $file_path AND n.valid_to IS NULL
            SET n.valid_to = $commit_hash
            """,
            file_path=file_path,
            commit_hash=commit_hash,
        )


def _read_blob(repo: "git.Repo", commit_sha: str, rel_path: str) -> Optional[bytes]:
    """Read a file's content at a given commit from the git object store."""
    try:
        blob = repo.commit(commit_sha).tree / rel_path
        return blob.data_stream.read()
    except (KeyError, AttributeError):
        return None  # file did not exist at that commit


# ---------------------------------------------------------------------------
# Working tree diff (uncommitted changes)
# ---------------------------------------------------------------------------


def get_workspace_diff(
    driver: Driver,
    repo_path: str,
    exclude_dirs: Optional[list[str]] = None,
) -> DiffResult:
    """
    Compute diff between HEAD commit and current working tree.

    This allows seeing uncommitted changes in the graph without committing.
    Returns a DiffResult that can be applied with --apply flag.
    """
    import git

    repo = git.Repo(repo_path)

    # Get HEAD commit
    try:
        head_commit = repo.commit("HEAD")
        head_hash = head_commit.hexsha
    except (git.InvalidGitRepositoryError, ValueError):
        # Not a git repo or no commits yet - treat all files as new
        head_hash = "INIT"
        head_commit = None

    pm = ParserManager()
    agg_diff = DiffResult()

    # Get all changed files (staged + unstaged)
    changed_paths: set[str] = set()

    if head_commit:
        # Compare working tree against HEAD
        diff_index = head_commit.diff(None)  # None = working tree
        for diff_item in diff_index:
            if diff_item.b_path:
                changed_paths.add(os.path.join(repo_path, diff_item.b_path))
            if diff_item.a_path:
                changed_paths.add(os.path.join(repo_path, diff_item.a_path))
    else:
        # No commits yet - all files are new
        for file_path, lang in walk_source_files(repo_path, exclude_dirs=exclude_dirs):
            changed_paths.add(file_path)

    if not changed_paths:
        logger.info("No uncommitted changes in working tree.")
        return DiffResult()

    logger.info("%d files with uncommitted changes", len(changed_paths))

    # Process each changed file
    for file_path in changed_paths:
        lang = pm.detect_language(file_path)
        if lang is None:
            continue

        # Check if file exists in working tree
        if not os.path.exists(file_path):
            # File was deleted - mark nodes for expiration
            _expire_file_nodes(driver, file_path, "WORKSPACE")
            continue

        # Parse new version from working tree
        with open(file_path, "rb") as fh:
            new_source = fh.read()

        new_tree = pm.parse_file(file_path, source=new_source)
        if new_tree is None:
            continue

        new_nodes = pm.extract_nodes(new_tree, file_path, lang, new_source, "WORKSPACE")
        new_edges = pm.extract_edges(new_tree, new_nodes, file_path, lang, new_source, "WORKSPACE")

        # Load old version from HEAD
        old_nodes: list[ASTNode] = []
        old_edges: list[ASTEdge] = []

        if head_commit:
            rel_path = os.path.relpath(file_path, repo_path)
            old_source_bytes = _read_blob(repo, head_hash, rel_path)
            if old_source_bytes is not None:
                old_tree = pm.parse_file(file_path, source=old_source_bytes)
                if old_tree:
                    old_nodes = pm.extract_nodes(
                        old_tree, file_path, lang, old_source_bytes, head_hash
                    )
                    old_edges = pm.extract_edges(
                        old_tree, old_nodes, file_path, lang, old_source_bytes, head_hash
                    )

        file_diff = compute_diff(old_nodes, new_nodes, old_edges, new_edges, "WORKSPACE")

        # Accumulate
        agg_diff.added_nodes.extend(file_diff.added_nodes)
        agg_diff.deleted_node_ids.extend(file_diff.deleted_node_ids)
        agg_diff.updated_nodes.extend(file_diff.updated_nodes)
        agg_diff.old_updated_node_ids.extend(file_diff.old_updated_node_ids)
        agg_diff.added_edges.extend(file_diff.added_edges)
        agg_diff.deleted_edge_ids.extend(file_diff.deleted_edge_ids)
        agg_diff.updated_edges.extend(file_diff.updated_edges)
        agg_diff.old_updated_edge_ids.extend(file_diff.old_updated_edge_ids)

    return agg_diff


def apply_workspace_diff(
    driver: Driver,
    repo_path: str,
    exclude_dirs: Optional[list[str]] = None,
) -> DiffResult:
    """
    Compute and apply workspace diff to the graph.
    Uses WORKSPACE label instead of a commit hash.
    """
    diff = get_workspace_diff(driver, repo_path, exclude_dirs)

    if diff.is_empty:
        return diff

    # Apply the diff
    with driver.session() as session:
        apply_diff(session, diff, "WORKSPACE")

    logger.info(
        "Workspace diff applied: %d nodes added/updated",
        len(diff.added_nodes) + len(diff.updated_nodes),
    )
    return diff
