"""
neo4j_repository.py - Neo4j graph database repository layer.

Provides Neo4jRepository class for all graph database operations:
- Node CRUD operations (get, create, update, delete)
- Edge operations (create, delete, query)
- Cypher query execution
- Schema management helpers

This repository abstracts Neo4j driver operations and provides
a clean API for the rest of the application.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from collections.abc import Iterable

from neo4j import Driver, GraphDatabase, Session, Result

from ast_rag.dto import ASTNode, ASTEdge, Neo4jConfig

logger = logging.getLogger(__name__)


class Neo4jRepository:
    """Repository for Neo4j graph database operations.

    This class provides a high-level API for interacting with Neo4j,
    abstracting away the low-level driver operations and Cypher query
    construction.

    Usage::

        config = Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="password")
        repo = Neo4jRepository(config)
        
        # Create a node
        node = ASTNode(...)
        repo.create_node(node)
        
        # Query nodes
        results = repo.execute_query("MATCH (n:Function) RETURN n LIMIT 10")

    Attributes:
        config: Neo4j configuration
        driver: Neo4j driver instance (lazy-initialized)
    """

    def __init__(self, config: Neo4jConfig) -> None:
        """Initialize Neo4j repository with configuration.

        Args:
            config: Neo4j connection configuration
        """
        self._config = config
        self._driver: Optional[Driver] = None

    @property
    def driver(self) -> Driver:
        """Get or create Neo4j driver instance (lazy initialization)."""
        if self._driver is None:
            self._driver = self._create_driver()
        return self._driver

    def _create_driver(self) -> Driver:
        """Create Neo4j driver from configuration."""
        return GraphDatabase.driver(
            self._config.uri,
            auth=(self._config.user, self._config.password),
        )

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    # ------------------------------------------------------------------
    # Node Operations
    # ------------------------------------------------------------------

    def get_node(self, node_id: str, label: Optional[str] = None) -> Optional[ASTNode]:
        """Retrieve a single node by its ID.

        Args:
            node_id: Unique identifier of the node
            label: Optional Neo4j label to filter by (e.g., "Function", "Class")

        Returns:
            ASTNode if found, None otherwise
        """
        label_clause = f":{label}" if label else ""
        query = f"""
MATCH (n{label_clause} {{id: $node_id}})
RETURN n
"""
        with self.driver.session() as session:
            result = session.run(query, node_id=node_id)
            record = result.single()
            if record is None:
                return None
            return self._record_to_node(record["n"])

    def get_nodes_by_ids(
        self,
        node_ids: list[str],
        label: Optional[str] = None,
    ) -> list[ASTNode]:
        """Retrieve multiple nodes by their IDs.

        Args:
            node_ids: List of unique node identifiers
            label: Optional Neo4j label to filter by

        Returns:
            List of ASTNode objects (may be fewer than requested if some not found)
        """
        if not node_ids:
            return []

        label_clause = f":{label}" if label else ""
        query = f"""
MATCH (n{label_clause})
WHERE n.id IN $node_ids
RETURN n
"""
        with self.driver.session() as session:
            result = session.run(query, node_ids=node_ids)
            return [self._record_to_node(record["n"]) for record in result]

    def get_node_by_property(
        self,
        label: str,
        property_name: str,
        property_value: Any,
    ) -> Optional[ASTNode]:
        """Retrieve a single node by a property match.

        Args:
            label: Neo4j node label (e.g., "Function", "Class")
            property_name: Property name to match on
            property_value: Property value to match

        Returns:
            ASTNode if found, None otherwise
        """
        query = f"""
MATCH (n:{label} {{{property_name}: $property_value}})
RETURN n
LIMIT 1
"""
        with self.driver.session() as session:
            result = session.run(query, property_value=property_value)
            record = result.single()
            if record is None:
                return None
            return self._record_to_node(record["n"])

    def create_node(self, node: ASTNode, label: Optional[str] = None) -> ASTNode:
        """Create a new node in the graph.

        Args:
            node: ASTNode to create
            label: Optional Neo4j label (defaults to node.kind.value)

        Returns:
            The created ASTNode
        """
        node_label = label or node.kind.value
        props = self._node_to_props(node)
        query = f"""
CREATE (n:{node_label} $props)
RETURN n
"""
        with self.driver.session() as session:
            result = session.run(query, props=props)
            record = result.single()
            return self._record_to_node(record["n"])

    def create_nodes(self, nodes: Iterable[ASTNode]) -> list[ASTNode]:
        """Batch create multiple nodes.

        Args:
            nodes: Iterable of ASTNode objects to create

        Returns:
            List of created ASTNode objects
        """
        nodes_list = list(nodes)
        if not nodes_list:
            return []

        # Group nodes by label for batch creation
        nodes_by_label: dict[str, list[dict[str, Any]]] = {}
        for node in nodes_list:
            label = node.kind.value
            if label not in nodes_by_label:
                nodes_by_label[label] = []
            nodes_by_label[label].append(self._node_to_props(node))

        created_nodes: list[ASTNode] = []
        with self.driver.session() as session:
            for label, props_list in nodes_by_label.items():
                query = f"""
UNWIND $batch AS props
CREATE (n:{label} $props)
RETURN n
"""
                result = session.run(query, batch=props_list)
                for record in result:
                    created_nodes.append(self._record_to_node(record["n"]))

        return created_nodes

    def update_node(
        self,
        node_id: str,
        properties: dict[str, Any],
        label: Optional[str] = None,
    ) -> Optional[ASTNode]:
        """Update properties of an existing node.

        Args:
            node_id: Unique identifier of the node to update
            properties: Dictionary of properties to update
            label: Optional Neo4j label to filter by

        Returns:
            Updated ASTNode if found, None otherwise
        """
        label_clause = f":{label}" if label else ""
        query = f"""
MATCH (n{label_clause} {{id: $node_id}})
SET n += $properties
RETURN n
"""
        with self.driver.session() as session:
            result = session.run(query, node_id=node_id, properties=properties)
            record = result.single()
            if record is None:
                return None
            return self._record_to_node(record["n"])

    def delete_node(self, node_id: str, label: Optional[str] = None) -> bool:
        """Delete a single node by its ID.

        Args:
            node_id: Unique identifier of the node to delete
            label: Optional Neo4j label to filter by

        Returns:
            True if node was deleted, False if not found
        """
        label_clause = f":{label}" if label else ""
        query = f"""
MATCH (n{label_clause} {{id: $node_id}})
DETACH DELETE n
RETURN count(n) as deleted
"""
        with self.driver.session() as session:
            result = session.run(query, node_id=node_id)
            record = result.single()
            return record["deleted"] > 0

    def delete_nodes(self, node_ids: list[str], label: Optional[str] = None) -> int:
        """Batch delete multiple nodes by their IDs.

        Args:
            node_ids: List of node IDs to delete
            label: Optional Neo4j label to filter by

        Returns:
            Number of nodes actually deleted
        """
        if not node_ids:
            return 0

        label_clause = f":{label}" if label else ""
        query = f"""
MATCH (n{label_clause})
WHERE n.id IN $node_ids
DETACH DELETE n
RETURN count(n) as deleted
"""
        with self.driver.session() as session:
            result = session.run(query, node_ids=node_ids)
            record = result.single()
            return record["deleted"]

    def node_exists(self, node_id: str, label: Optional[str] = None) -> bool:
        """Check if a node exists by its ID.

        Args:
            node_id: Unique identifier to check
            label: Optional Neo4j label to filter by

        Returns:
            True if node exists, False otherwise
        """
        label_clause = f":{label}" if label else ""
        query = f"""
MATCH (n{label_clause} {{id: $node_id}})
RETURN count(n) > 0 as exists
"""
        with self.driver.session() as session:
            result = session.run(query, node_id=node_id)
            record = result.single()
            return record["exists"]

    # ------------------------------------------------------------------
    # Edge Operations
    # ------------------------------------------------------------------

    def create_edge(self, edge: ASTEdge) -> ASTEdge:
        """Create a new edge (relationship) in the graph.

        Args:
            edge: ASTEdge to create

        Returns:
            The created ASTEdge
        """
        props = self._edge_to_props(edge)
        query = """
MATCH (a {id: $from_id})
MATCH (b {id: $to_id})
CREATE (a)-[r:RELATES $props]->(b)
RETURN r, a.id as from_id, b.id as to_id
"""
        with self.driver.session() as session:
            result = session.run(
                query,
                from_id=edge.from_id,
                to_id=edge.to_id,
                props=props,
            )
            record = result.single()
            return self._record_to_edge(record["r"], record["from_id"], record["to_id"])

    def create_edges(self, edges: Iterable[ASTEdge]) -> list[ASTEdge]:
        """Batch create multiple edges.

        Args:
            edges: Iterable of ASTEdge objects to create

        Returns:
            List of created ASTEdge objects
        """
        edges_list = list(edges)
        if not edges_list:
            return []

        batch = [self._edge_to_props(e) for e in edges_list]
        query = """
UNWIND $batch AS props
MATCH (a {id: props.from_id})
MATCH (b {id: props.to_id})
CREATE (a)-[r:RELATES]->(b)
SET r += props
RETURN r, a.id as from_id, b.id as to_id
"""
        created_edges: list[ASTEdge] = []
        with self.driver.session() as session:
            result = session.run(query, batch=batch)
            for record in result:
                created_edges.append(
                    self._record_to_edge(record["r"], record["from_id"], record["to_id"])
                )
        return created_edges

    def get_edge(self, edge_id: str, edge_type: Optional[str] = None) -> Optional[ASTEdge]:
        """Retrieve a single edge by its ID.

        Args:
            edge_id: Unique identifier of the edge
            edge_type: Optional edge type to filter by

        Returns:
            ASTEdge if found, None otherwise
        """
        query = """
MATCH (a)-[r {id: $edge_id}]->(b)
RETURN r, a.id as from_id, b.id as to_id
"""
        with self.driver.session() as session:
            result = session.run(query, edge_id=edge_id)
            record = result.single()
            if record is None:
                return None
            return self._record_to_edge(record["r"], record["from_id"], record["to_id"])

    def get_edges_by_ids(
        self,
        edge_ids: list[str],
        edge_type: Optional[str] = None,
    ) -> list[ASTEdge]:
        """Retrieve multiple edges by their IDs.

        Args:
            edge_ids: List of edge identifiers
            edge_type: Optional edge type to filter by

        Returns:
            List of ASTEdge objects
        """
        if not edge_ids:
            return []

        query = """
MATCH (a)-[r]->(b)
WHERE r.id IN $edge_ids
RETURN r, a.id as from_id, b.id as to_id
"""
        with self.driver.session() as session:
            result = session.run(query, edge_ids=edge_ids)
            return [
                self._record_to_edge(record["r"], record["from_id"], record["to_id"])
                for record in result
            ]

    def get_edges_between(
        self,
        from_node_id: str,
        to_node_id: str,
        edge_type: Optional[str] = None,
    ) -> list[ASTEdge]:
        """Retrieve all edges between two nodes.

        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            edge_type: Optional edge type to filter by

        Returns:
            List of edges connecting the two nodes
        """
        query = """
MATCH (a {id: $from_id})-[r]->(b {id: $to_id})
RETURN r, a.id as from_id, b.id as to_id
"""
        with self.driver.session() as session:
            result = session.run(query, from_id=from_node_id, to_id=to_node_id)
            return [
                self._record_to_edge(record["r"], record["from_id"], record["to_id"])
                for record in result
            ]

    def get_outgoing_edges(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[ASTEdge]:
        """Retrieve all outgoing edges from a node.

        Args:
            node_id: Source node ID
            edge_type: Optional edge type to filter by
            limit: Optional maximum number of edges to return

        Returns:
            List of outgoing edges
        """
        limit_clause = f"LIMIT {limit}" if limit else ""
        query = f"""
MATCH (a {{id: $node_id}})-[r]->(b)
RETURN r, a.id as from_id, b.id as to_id
{limit_clause}
"""
        with self.driver.session() as session:
            result = session.run(query, node_id=node_id)
            return [
                self._record_to_edge(record["r"], record["from_id"], record["to_id"])
                for record in result
            ]

    def get_incoming_edges(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[ASTEdge]:
        """Retrieve all incoming edges to a node.

        Args:
            node_id: Target node ID
            edge_type: Optional edge type to filter by
            limit: Optional maximum number of edges to return

        Returns:
            List of incoming edges
        """
        limit_clause = f"LIMIT {limit}" if limit else ""
        query = f"""
MATCH (a)-[r]->(b {{id: $node_id}})
RETURN r, a.id as from_id, b.id as to_id
{limit_clause}
"""
        with self.driver.session() as session:
            result = session.run(query, node_id=node_id)
            return [
                self._record_to_edge(record["r"], record["from_id"], record["to_id"])
                for record in result
            ]

    def update_edge(
        self,
        edge_id: str,
        properties: dict[str, Any],
        edge_type: Optional[str] = None,
    ) -> Optional[ASTEdge]:
        """Update properties of an existing edge.

        Args:
            edge_id: Unique identifier of the edge to update
            properties: Dictionary of properties to update
            edge_type: Optional edge type to filter by

        Returns:
            Updated ASTEdge if found, None otherwise
        """
        query = """
MATCH (a)-[r {id: $edge_id}]->(b)
SET r += $properties
RETURN r, a.id as from_id, b.id as to_id
"""
        with self.driver.session() as session:
            result = session.run(query, edge_id=edge_id, properties=properties)
            record = result.single()
            if record is None:
                return None
            return self._record_to_edge(record["r"], record["from_id"], record["to_id"])

    def delete_edge(self, edge_id: str, edge_type: Optional[str] = None) -> bool:
        """Delete a single edge by its ID.

        Args:
            edge_id: Unique identifier of the edge to delete
            edge_type: Optional edge type to filter by

        Returns:
            True if edge was deleted, False if not found
        """
        query = """
MATCH ()-[r {id: $edge_id}]->()
DELETE r
RETURN count(r) as deleted
"""
        with self.driver.session() as session:
            result = session.run(query, edge_id=edge_id)
            record = result.single()
            return record["deleted"] > 0

    def delete_edges(self, edge_ids: list[str], edge_type: Optional[str] = None) -> int:
        """Batch delete multiple edges by their IDs.

        Args:
            edge_ids: List of edge IDs to delete
            edge_type: Optional edge type to filter by

        Returns:
            Number of edges actually deleted
        """
        if not edge_ids:
            return 0

        query = """
MATCH ()-[r]->()
WHERE r.id IN $edge_ids
DELETE r
RETURN count(r) as deleted
"""
        with self.driver.session() as session:
            result = session.run(query, edge_ids=edge_ids)
            record = result.single()
            return record["deleted"]

    def edge_exists(self, edge_id: str, edge_type: Optional[str] = None) -> bool:
        """Check if an edge exists by its ID.

        Args:
            edge_id: Unique identifier to check
            edge_type: Optional edge type to filter by

        Returns:
            True if edge exists, False otherwise
        """
        query = """
MATCH ()-[r {id: $edge_id}]->()
RETURN count(r) > 0 as exists
"""
        with self.driver.session() as session:
            result = session.run(query, edge_id=edge_id)
            record = result.single()
            return record["exists"]

    # ------------------------------------------------------------------
    # Query Execution
    # ------------------------------------------------------------------

    def execute_query(
        self,
        query: str,
        parameters: Optional[dict[str, Any]] = None,
        read_only: bool = True,
    ) -> Result:
        """Execute a Cypher query and return the result.

        Args:
            query: Cypher query string
            parameters: Query parameters (named parameters in Cypher)
            read_only: If True, use read transaction; otherwise write transaction

        Returns:
            Neo4j Result object

        Example::

            result = repo.execute_query(
                "MATCH (n:Function) WHERE n.name = $name RETURN n",
                {"name": "process_data"}
            )
        """
        with self.driver.session() as session:
            if read_only:
                return session.run(query, parameters or {})
            else:
                return session.write_transaction(lambda tx: tx.run(query, parameters or {}))

    def execute_write(
        self,
        query: str,
        parameters: Optional[dict[str, Any]] = None,
    ) -> Result:
        """Execute a write Cypher query (convenience method).

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            Neo4j Result object
        """
        return self.execute_query(query, parameters, read_only=False)

    def execute_read(
        self,
        query: str,
        parameters: Optional[dict[str, Any]] = None,
    ) -> Result:
        """Execute a read-only Cypher query (convenience method).

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            Neo4j Result object
        """
        return self.execute_query(query, parameters, read_only=True)

    def execute_in_transaction(
        self,
        queries: list[tuple[str, Optional[dict[str, Any]]]],
    ) -> list[Result]:
        """Execute multiple queries within a single transaction.

        All queries succeed or fail together. Use for atomic operations.

        Args:
            queries: List of (query_string, parameters) tuples

        Returns:
            List of Result objects for each query

        Example::

            queries = [
                ("CREATE (n:Node {id: $id})", {"id": "123"}),
                ("MATCH (n:Node {id: $id}) SET n.updated = true", {"id": "123"}),
            ]
            results = repo.execute_in_transaction(queries)
        """
        results: list[Result] = []
        with self.driver.session() as session:
            with session.begin_transaction() as tx:
                for query, params in queries:
                    result = tx.run(query, params or {})
                    results.append(result)
                tx.commit()
        return results

    def count_nodes(self, label: Optional[str] = None) -> int:
        """Count nodes in the graph, optionally filtered by label.

        Args:
            label: Optional Neo4j label to filter by

        Returns:
            Number of matching nodes
        """
        label_clause = f":{label}" if label else ""
        query = f"""
MATCH (n{label_clause})
RETURN count(n) as count
"""
        with self.driver.session() as session:
            result = session.run(query)
            record = result.single()
            return record["count"]

    def count_edges(self, edge_type: Optional[str] = None) -> int:
        """Count edges in the graph, optionally filtered by type.

        Args:
            edge_type: Optional edge type to filter by

        Returns:
            Number of matching edges
        """
        query = """
MATCH ()-[r]->()
RETURN count(r) as count
"""
        with self.driver.session() as session:
            result = session.run(query)
            record = result.single()
            return record["count"]

    # ------------------------------------------------------------------
    # Schema Management
    # ------------------------------------------------------------------

    def create_index(
        self,
        label: str,
        property_name: str,
        index_name: Optional[str] = None,
        if_not_exists: bool = True,
    ) -> None:
        """Create a B-tree index on a node property.

        Args:
            label: Neo4j node label
            property_name: Property to index
            index_name: Optional custom index name
            if_not_exists: If True, skip if index already exists
        """
        name = index_name or f"idx_{label}_{property_name}"
        if_not_exists_clause = "IF NOT EXISTS" if if_not_exists else ""
        query = f"""
CREATE INDEX {name} {if_not_exists_clause}
FOR (n:{label})
ON (n.{property_name})
"""
        with self.driver.session() as session:
            session.run(query)
        logger.info("Created index %s on %s.%s", name, label, property_name)

    def create_fulltext_index(
        self,
        index_name: str,
        labels: list[str],
        properties: list[str],
        if_not_exists: bool = True,
    ) -> None:
        """Create a full-text index for text search.

        Args:
            index_name: Name of the full-text index
            labels: List of node labels to index
            properties: List of properties to include in index
            if_not_exists: If True, skip if index already exists
        """
        if_not_exists_clause = "IF NOT EXISTS" if if_not_exists else ""
        labels_str = ", ".join([f"`{l}`" for l in labels])
        props_str = ", ".join([f"n.{p}" for p in properties])
        query = f"""
CREATE FULLTEXT INDEX {index_name} {if_not_exists_clause}
FOR (n)
ON EACH [{props_str}]
OPTIONS {{
  indexConfig: {{
    `fulltext.analyzer`: 'standard',
    `fulltext.eventually_consistent`: true
  }}
}}
"""
        with self.driver.session() as session:
            session.run(query)
        logger.info("Created full-text index %s for labels %s", index_name, labels)

    def create_constraint(
        self,
        constraint_name: str,
        label: str,
        property_name: str,
        constraint_type: str = "UNIQUE",
        if_not_exists: bool = True,
    ) -> None:
        """Create a constraint (e.g., UNIQUE, NOT NULL).

        Args:
            constraint_name: Name of the constraint
            label: Neo4j node label
            property_name: Property to constrain
            constraint_type: Type of constraint ("UNIQUE", "NOT_NULL", etc.)
            if_not_exists: If True, skip if constraint already exists
        """
        if_not_exists_clause = "IF NOT EXISTS" if if_not_exists else ""

        if constraint_type.upper() == "UNIQUE":
            query = f"""
CREATE CONSTRAINT {constraint_name} {if_not_exists_clause}
FOR (n:{label})
REQUIRE n.{property_name} IS UNIQUE
"""
        elif constraint_type.upper() == "NOT_NULL":
            query = f"""
CREATE CONSTRAINT {constraint_name} {if_not_exists_clause}
FOR (n:{label})
REQUIRE n.{property_name} IS NOT NULL
"""
        else:
            raise ValueError(f"Unsupported constraint type: {constraint_type}")

        with self.driver.session() as session:
            session.run(query)
        logger.info("Created constraint %s on %s.%s", constraint_name, label, property_name)

    def drop_index(self, index_name: str, if_exists: bool = True) -> None:
        """Drop an index by name.

        Args:
            index_name: Name of the index to drop
            if_exists: If True, skip if index doesn't exist
        """
        if_exists_clause = "IF EXISTS" if if_exists else ""
        query = f"DROP INDEX {index_name} {if_exists_clause}"
        with self.driver.session() as session:
            session.run(query)
        logger.info("Dropped index %s", index_name)

    def drop_constraint(self, constraint_name: str, if_exists: bool = True) -> None:
        """Drop a constraint by name.

        Args:
            constraint_name: Name of the constraint to drop
            if_exists: If True, skip if constraint doesn't exist
        """
        if_exists_clause = "IF EXISTS" if if_exists else ""
        query = f"DROP CONSTRAINT {constraint_name} {if_exists_clause}"
        with self.driver.session() as session:
            session.run(query)
        logger.info("Dropped constraint %s", constraint_name)

    def list_indexes(self) -> list[dict[str, Any]]:
        """List all indexes in the database.

        Returns:
            List of index metadata dictionaries
        """
        query = "SHOW INDEXES"
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]

    def list_constraints(self) -> list[dict[str, Any]]:
        """List all constraints in the database.

        Returns:
            List of constraint metadata dictionaries
        """
        query = "SHOW CONSTRAINTS"
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]

    def get_schema_info(self) -> dict[str, Any]:
        """Get database schema information.

        Returns:
            Dictionary with schema metadata including labels,
            relationship types, and property keys
        """
        with self.driver.session() as session:
            labels_result = session.run("CALL db.labels()")
            labels = [record["label"] for record in labels_result]

            rel_types_result = session.run("CALL db.relationshipTypes()")
            rel_types = [record["relationshipType"] for record in rel_types_result]

            prop_keys_result = session.run("CALL db.propertyKeys()")
            prop_keys = [record["propertyKey"] for record in prop_keys_result]

        return {
            "labels": labels,
            "relationship_types": rel_types,
            "property_keys": prop_keys,
        }

    def apply_cql_file(self, file_path: str) -> list[Result]:
        """Apply Cypher statements from a file.

        Reads a .cql file and executes each statement. Statements
        are split by semicolons.

        Args:
            file_path: Path to the CQL file

        Returns:
            List of Result objects for each statement
        """
        with open(file_path, "r") as f:
            content = f.read()

        # Split by semicolons and filter out empty statements
        statements = [s.strip() for s in content.split(";") if s.strip()]

        results: list[Result] = []
        with self.driver.session() as session:
            for statement in statements:
                try:
                    result = session.run(statement)
                    results.append(result)
                except Exception as exc:
                    logger.warning("Failed to execute statement: %s", exc)
                    raise

        return results

    # ------------------------------------------------------------------
    # Context Manager Support
    # ------------------------------------------------------------------

    def __enter__(self) -> Neo4jRepository:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and close connection."""
        self.close()

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _record_to_node(self, record: Any) -> ASTNode:
        """Convert a Neo4j node record to ASTNode."""
        from ast_rag.dto import NodeKind, Language

        props = dict(record)
        return ASTNode(
            id=props.get("id", ""),
            kind=NodeKind(props.get("kind", "Function")),
            name=props.get("name", ""),
            qualified_name=props.get("qualified_name", ""),
            lang=Language(props.get("lang", "java")),
            file_path=props.get("file_path", ""),
            start_line=props.get("start_line", 0),
            end_line=props.get("end_line", 0),
            start_byte=props.get("start_byte", 0),
            end_byte=props.get("end_byte", 0),
            signature=props.get("signature"),
            docstring=props.get("docstring"),
            valid_from=props.get("valid_from", "INIT"),
            valid_to=props.get("valid_to"),
        )

    def _node_to_props(self, node: ASTNode) -> dict[str, Any]:
        """Convert ASTNode to Neo4j properties dictionary."""
        return {
            "id": node.id,
            "kind": node.kind.value,
            "name": node.name,
            "qualified_name": node.qualified_name,
            "lang": node.lang.value,
            "file_path": node.file_path,
            "start_line": node.start_line,
            "end_line": node.end_line,
            "start_byte": node.start_byte,
            "end_byte": node.end_byte,
            "signature": node.signature,
            "docstring": node.docstring,
            "valid_from": node.valid_from,
            "valid_to": node.valid_to,
        }

    def _record_to_edge(self, record: Any, from_id: str, to_id: str) -> ASTEdge:
        """Convert a Neo4j edge record to ASTEdge."""
        from ast_rag.dto import EdgeKind

        props = dict(record)
        return ASTEdge(
            id=props.get("id", ""),
            kind=EdgeKind(props.get("kind", "CALLS")),
            from_id=from_id,
            to_id=to_id,
            label=props.get("label"),
            valid_from=props.get("valid_from", "INIT"),
            valid_to=props.get("valid_to"),
        )

    def _edge_to_props(self, edge: ASTEdge) -> dict[str, Any]:
        """Convert ASTEdge to Neo4j properties dictionary."""
        return {
            "id": edge.id,
            "kind": edge.kind.value,
            "from_id": edge.from_id,
            "to_id": edge.to_id,
            "label": edge.label,
            "valid_from": edge.valid_from,
            "valid_to": edge.valid_to,
        }
