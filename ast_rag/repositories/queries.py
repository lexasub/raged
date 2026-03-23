"""
Repository query helpers for the MVCC graph updater.

These helpers were split out during refactoring and are still imported by
``graph_updater_service``. Restoring them keeps the current architecture intact
without changing the graph updater call sites.
"""

from __future__ import annotations

from neo4j import Session


def batch_upsert_nodes(session: Session, by_label: dict[str, list[dict]]) -> None:
    """MERGE nodes grouped by label."""
    for label, props_list in by_label.items():
        if not props_list:
            continue
        session.run(
            f"""
            UNWIND $props AS p
            MERGE (n:{label} {{id: p.id}})
            SET n += p
            """,
            props=props_list,
        )


def batch_expire_nodes(
    session: Session,
    by_label: dict[str, list[str]],
    commit_hash: str,
) -> None:
    """Set ``valid_to`` on nodes to mark them expired."""
    for label, ids in by_label.items():
        if not ids:
            continue
        session.run(
            f"""
            UNWIND $ids AS nid
            MATCH (n:{label} {{id: nid}})
            WHERE n.valid_to IS NULL
            SET n.valid_to = $commit_hash
            """,
            ids=ids,
            commit_hash=commit_hash,
        )


def batch_upsert_edges(session: Session, edge_dicts: list[dict]) -> None:
    """MERGE edges keyed by ``id``."""
    if not edge_dicts:
        return
    session.run(
        """
        UNWIND $edges AS e
        MATCH (a {id: e.from_id}), (b {id: e.to_id})
        MERGE (a)-[r:EDGE {id: e.id}]->(b)
        SET r += e
        """,
        edges=edge_dicts,
    )


def batch_expire_edges(
    session: Session,
    edge_ids: list[str],
    commit_hash: str,
) -> None:
    """Set ``valid_to`` on edges to mark them expired."""
    if not edge_ids:
        return
    session.run(
        """
        UNWIND $ids AS eid
        MATCH ()-[r:EDGE {id: eid}]->()
        WHERE r.valid_to IS NULL
        SET r.valid_to = $commit_hash
        """,
        ids=edge_ids,
        commit_hash=commit_hash,
    )


def ensure_current_version(session: Session, commit_hash: str) -> None:
    """Create or update the ``CurrentVersion`` singleton node."""
    session.run(
        """
        MERGE (v:CurrentVersion {id: 'current'})
        SET v.commit_hash = $commit_hash
        """,
        commit_hash=commit_hash,
    )
