"""
graph_schema.py - Apply Neo4j schema and provide Cypher helper functions.

Reads schema/graph_schema.cql and executes each statement against Neo4j.
Also provides utility functions used by graph_updater.py and ast_rag_api.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from neo4j import Driver, GraphDatabase, Session

from ast_rag.models import Neo4jConfig

logger = logging.getLogger(__name__)

# Path to the CQL schema file relative to the project root
_SCHEMA_FILE = Path(__file__).parent / "schema" / "graph_schema.cql"

# Labels that correspond to callable code entities (for query filtering)
CALLABLE_LABELS = frozenset(["Function", "Method", "Constructor", "Destructor"])

# Labels that correspond to type definitions
TYPE_LABELS = frozenset(["Class", "Interface", "Struct", "Enum", "Trait"])

# All AST entity labels (excludes Project/Package/File/CurrentVersion)
ALL_ENTITY_LABELS = CALLABLE_LABELS | TYPE_LABELS | frozenset(["Field", "Parameter", "Namespace"])


def create_driver(config: Neo4jConfig) -> Driver:
    """Create and return a Neo4j driver instance."""
    return GraphDatabase.driver(
        config.uri,
        auth=(config.user, config.password),
        database=config.database,
    )


def apply_schema(driver: Driver, schema_path: Optional[Path] = None) -> None:
    """Execute all statements in the CQL schema file against Neo4j.

    Skips CALL statements that may fail on older Neo4j versions (e.g.
    full-text index creation) with a warning.
    """
    path = schema_path or _SCHEMA_FILE
    if not path.exists():
        logger.warning("Schema file not found at %s; skipping schema apply.", path)
        return

    raw = path.read_text(encoding="utf-8")
    # Split on semicolons (crude but works for our simple schema)
    statements = [s.strip() for s in raw.split(";") if s.strip()]

    with driver.session() as session:
        for stmt in statements:
            # Strip comments
            clean = "\n".join(
                line for line in stmt.splitlines()
                if not line.strip().startswith("//")
            ).strip()
            if not clean:
                continue
            try:
                session.run(clean)
                logger.debug("Applied schema statement: %s...", clean[:60])
            except Exception as exc:
                # Full-text index creation may fail on Community Edition; log and continue.
                logger.warning("Schema statement failed (skipping): %s | %s", clean[:80], exc)

    logger.info("Schema applied from %s", path)


def ensure_current_version(session: Session, commit_hash: str) -> None:
    """Create or update the singleton CurrentVersion node."""
    session.run(
        """
        MERGE (v:CurrentVersion {id: 'singleton'})
        SET v.hash = $hash, v.updated_at = datetime()
        """,
        hash=commit_hash,
    )


def get_current_version(driver: Driver) -> Optional[str]:
    """Return the current graph version hash, or None if not set."""
    with driver.session() as session:
        result = session.run(
            "MATCH (v:CurrentVersion {id: 'singleton'}) RETURN v.hash AS hash"
        )
        record = result.single()
        return record["hash"] if record else None


# ---------------------------------------------------------------------------
# Cypher helpers for MERGE / UPDATE operations
# ---------------------------------------------------------------------------

# Generic labels that can carry an AST node
_KIND_TO_LABEL: dict[str, str] = {
    "Project":     "Project",
    "Package":     "Package",
    "Namespace":   "Namespace",
    "Module":      "Module",
    "File":        "File",
    "Class":       "Class",
    "Interface":   "Interface",
    "Struct":      "Struct",
    "Enum":        "Enum",
    "Trait":       "Trait",
    "Function":    "Function",
    "Method":      "Method",
    "Constructor": "Constructor",
    "Destructor":  "Destructor",
    "Field":       "Field",
    "Variable":    "Variable",
    "Parameter":   "Parameter",
}


def upsert_node_cypher(label: str) -> str:
    """Return a Cypher MERGE statement that creates or updates an AST node.

    Parameters bound: props (map with all node fields).
    """
    return f"""
MERGE (n:{label} {{id: $props.id}})
SET n += $props
"""


def expire_node_cypher(label: str) -> str:
    """Return a Cypher statement that sets valid_to on an existing node."""
    return f"""
MATCH (n:{label} {{id: $node_id}})
WHERE n.valid_to IS NULL
SET n.valid_to = $commit_hash
"""


def upsert_edge_cypher(from_label: str, to_label: str, edge_type: str) -> str:
    """Return a Cypher MERGE statement for a directed relationship.

    Parameters bound: from_id, to_id, props (map with edge fields).
    """
    return f"""
MATCH (a:{from_label} {{id: $from_id}})
MATCH (b:{to_label} {{id: $to_id}})
MERGE (a)-[r:{edge_type} {{id: $props.id}}]->(b)
SET r += $props
"""


def expire_edge_cypher() -> str:
    """Expire all currently active edges whose id is in $edge_ids."""
    return """
MATCH ()-[r]->()
WHERE r.id IN $edge_ids AND r.valid_to IS NULL
SET r.valid_to = $commit_hash
"""


def batch_upsert_nodes(session: Session, nodes_by_label: dict[str, list[dict]]) -> None:
    """Batch-upsert a dict of {label: [props_dict]} into Neo4j.

    Uses UNWIND for efficiency.
    """
    for label, props_list in nodes_by_label.items():
        if not props_list:
            continue
        cypher = f"""
UNWIND $batch AS props
MERGE (n:{label} {{id: props.id}})
SET n += props
"""
        session.run(cypher, batch=props_list)


def batch_expire_nodes(
    session: Session, ids_by_label: dict[str, list[str]], commit_hash: str
) -> None:
    """Set valid_to = commit_hash on all active nodes whose ids are given."""
    for label, ids in ids_by_label.items():
        if not ids:
            continue
        cypher = f"""
UNWIND $ids AS nid
MATCH (n:{label} {{id: nid}})
WHERE n.valid_to IS NULL
SET n.valid_to = $commit_hash
"""
        session.run(cypher, ids=ids, commit_hash=commit_hash)


def batch_upsert_edges(session: Session, edges: list[dict[str, Any]]) -> None:
    """Upsert a list of edges represented as dicts.

    Each dict must have: from_id, to_id, kind (edge type), and the full props map.
    We use a generic APOC-free approach with UNWIND and dynamic labels avoided by
    batching per edge type.
    """
    # Group by edge kind to avoid dynamic relationship type syntax
    by_kind: dict[str, list[dict]] = {}
    for e in edges:
        by_kind.setdefault(e["kind"], []).append(e)

    for kind, batch in by_kind.items():
        cypher = f"""
UNWIND $batch AS e
MATCH (a {{id: e.from_id}})
MATCH (b {{id: e.to_id}})
MERGE (a)-[r:{kind} {{id: e.id}}]->(b)
SET r += e
"""
        session.run(cypher, batch=batch)


def batch_expire_edges(session: Session, edge_ids: list[str], commit_hash: str) -> None:
    """Expire a list of edges by their id."""
    if not edge_ids:
        return
    session.run(expire_edge_cypher(), edge_ids=edge_ids, commit_hash=commit_hash)
