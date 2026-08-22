"""
Neo4j connection and schema helpers.

For new code, import directly from ast_rag.repositories.queries:
    from ast_rag.repositories.queries import batch_upsert_nodes, batch_upsert_edges
"""

from pathlib import Path
from typing import Optional

from neo4j import Driver, GraphDatabase

from ast_rag.models import Neo4jConfig

_SCHEMA_FILE = Path(__file__).parent.parent / "schema" / "graph_schema.cql"

CALLABLE_LABELS = frozenset(["Function", "Method", "Constructor", "Destructor"])
TYPE_LABELS = frozenset(["Class", "Interface", "Struct", "Enum", "Trait"])
ALL_ENTITY_LABELS = CALLABLE_LABELS | TYPE_LABELS | frozenset(["Field", "Parameter", "Namespace"])

# Re-export for backward compat
KIND_TO_LABEL: dict[str, str] = {
    "Project": "Project",
    "Package": "Package",
    "Namespace": "Namespace",
    "Module": "Module",
    "File": "File",
    "Class": "Class",
    "Interface": "Interface",
    "Struct": "Struct",
    "Enum": "Enum",
    "Trait": "Trait",
    "Function": "Function",
    "Method": "Method",
    "Constructor": "Constructor",
    "Destructor": "Destructor",
    "Field": "Field",
    "Variable": "Variable",
    "Parameter": "Parameter",
}


def create_driver(config: Neo4jConfig, create_if_not_exists: bool = True) -> Driver:
    driver = GraphDatabase.driver(
        config.uri,
        auth=(config.user, config.password),
        database="neo4j",
    )

    if create_if_not_exists and config.database != "neo4j":
        try:
            with driver.session(database="neo4j") as session:
                result = session.run(
                    "SHOW DATABASES WHERE name = $name",
                    name=config.database,
                )
                db_info = result.single()
                if db_info is None:
                    pass
        except Exception:
            pass

    driver.close()
    try:
        return GraphDatabase.driver(
            config.uri,
            auth=(config.user, config.password),
            database=config.database,
        )
    except Exception:
        return GraphDatabase.driver(
            config.uri,
            auth=(config.user, config.password),
            database="neo4j",
        )


def apply_schema(driver: Driver, schema_path: Optional[Path] = None) -> None:
    path = schema_path or _SCHEMA_FILE
    if not path.exists():
        return

    raw = path.read_text(encoding="utf-8")
    statements = [s.strip() for s in raw.split(";") if s.strip()]

    with driver.session() as session:
        for stmt in statements:
            clean = "\n".join(
                line for line in stmt.splitlines() if not line.strip().startswith("//")
            ).strip()
            if not clean:
                continue
            try:
                session.run(clean)
            except Exception:
                pass
