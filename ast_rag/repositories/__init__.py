"""AST-RAG Repositories.

Data access layer for database operations:
- Neo4jRepository: Graph database operations
- QdrantRepository: Vector database operations
- SchemaManager: Schema management and migrations
- neo4j_helpers: Neo4j utilities (create_driver, apply_schema, Cypher helpers)
"""

from ast_rag.repositories.neo4j_repository import Neo4jRepository
from ast_rag.repositories.qdrant_repository import QdrantRepository
from ast_rag.repositories.schema_manager import SchemaManager
from ast_rag.repositories.neo4j_helpers import (
    create_driver,
    apply_schema,
    ensure_current_version,
    get_current_version,
    CALLABLE_LABELS,
    TYPE_LABELS,
    ALL_ENTITY_LABELS,
    KIND_TO_LABEL,
    upsert_node_cypher,
    expire_node_cypher,
    upsert_edge_cypher,
    expire_edge_cypher,
    batch_upsert_nodes,
    batch_expire_nodes,
    batch_upsert_edges,
    batch_expire_edges,
)

__all__ = [
    "Neo4jRepository",
    "QdrantRepository",
    "SchemaManager",
    # Helpers
    "create_driver",
    "apply_schema",
    "ensure_current_version",
    "get_current_version",
    # Constants
    "CALLABLE_LABELS",
    "TYPE_LABELS",
    "ALL_ENTITY_LABELS",
    "KIND_TO_LABEL",
    # Cypher helpers
    "upsert_node_cypher",
    "expire_node_cypher",
    "upsert_edge_cypher",
    "expire_edge_cypher",
    # Batch operations
    "batch_upsert_nodes",
    "batch_expire_nodes",
    "batch_upsert_edges",
    "batch_expire_edges",
]
