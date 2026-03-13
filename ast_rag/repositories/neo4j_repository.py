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

from typing import Any, Optional
from collections.abc import Iterable

from neo4j import Driver, Session, Result

from ast_rag.dto import ASTNode, ASTEdge, Neo4jConfig


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
        raise NotImplementedError("Subclasses must implement _create_driver")

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
        raise NotImplementedError("Subclasses must implement get_node")

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
        raise NotImplementedError("Subclasses must implement get_nodes_by_ids")

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
        raise NotImplementedError("Subclasses must implement get_node_by_property")

    def create_node(self, node: ASTNode, label: Optional[str] = None) -> ASTNode:
        """Create a new node in the graph.

        Args:
            node: ASTNode to create
            label: Optional Neo4j label (defaults to node.kind.value)

        Returns:
            The created ASTNode
        """
        raise NotImplementedError("Subclasses must implement create_node")

    def create_nodes(self, nodes: Iterable[ASTNode]) -> list[ASTNode]:
        """Batch create multiple nodes.

        Args:
            nodes: Iterable of ASTNode objects to create

        Returns:
            List of created ASTNode objects
        """
        raise NotImplementedError("Subclasses must implement create_nodes")

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
        raise NotImplementedError("Subclasses must implement update_node")

    def delete_node(self, node_id: str, label: Optional[str] = None) -> bool:
        """Delete a single node by its ID.

        Args:
            node_id: Unique identifier of the node to delete
            label: Optional Neo4j label to filter by

        Returns:
            True if node was deleted, False if not found
        """
        raise NotImplementedError("Subclasses must implement delete_node")

    def delete_nodes(self, node_ids: list[str], label: Optional[str] = None) -> int:
        """Batch delete multiple nodes by their IDs.

        Args:
            node_ids: List of node IDs to delete
            label: Optional Neo4j label to filter by

        Returns:
            Number of nodes actually deleted
        """
        raise NotImplementedError("Subclasses must implement delete_nodes")

    def node_exists(self, node_id: str, label: Optional[str] = None) -> bool:
        """Check if a node exists by its ID.

        Args:
            node_id: Unique identifier to check
            label: Optional Neo4j label to filter by

        Returns:
            True if node exists, False otherwise
        """
        raise NotImplementedError("Subclasses must implement node_exists")

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
        raise NotImplementedError("Subclasses must implement create_edge")

    def create_edges(self, edges: Iterable[ASTEdge]) -> list[ASTEdge]:
        """Batch create multiple edges.

        Args:
            edges: Iterable of ASTEdge objects to create

        Returns:
            List of created ASTEdge objects
        """
        raise NotImplementedError("Subclasses must implement create_edges")

    def get_edge(self, edge_id: str, edge_type: Optional[str] = None) -> Optional[ASTEdge]:
        """Retrieve a single edge by its ID.

        Args:
            edge_id: Unique identifier of the edge
            edge_type: Optional edge type to filter by

        Returns:
            ASTEdge if found, None otherwise
        """
        raise NotImplementedError("Subclasses must implement get_edge")

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
        raise NotImplementedError("Subclasses must implement get_edges_by_ids")

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
        raise NotImplementedError("Subclasses must implement get_edges_between")

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
        raise NotImplementedError("Subclasses must implement get_outgoing_edges")

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
        raise NotImplementedError("Subclasses must implement get_incoming_edges")

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
        raise NotImplementedError("Subclasses must implement update_edge")

    def delete_edge(self, edge_id: str, edge_type: Optional[str] = None) -> bool:
        """Delete a single edge by its ID.

        Args:
            edge_id: Unique identifier of the edge to delete
            edge_type: Optional edge type to filter by

        Returns:
            True if edge was deleted, False if not found
        """
        raise NotImplementedError("Subclasses must implement delete_edge")

    def delete_edges(self, edge_ids: list[str], edge_type: Optional[str] = None) -> int:
        """Batch delete multiple edges by their IDs.

        Args:
            edge_ids: List of edge IDs to delete
            edge_type: Optional edge type to filter by

        Returns:
            Number of edges actually deleted
        """
        raise NotImplementedError("Subclasses must implement delete_edges")

    def edge_exists(self, edge_id: str, edge_type: Optional[str] = None) -> bool:
        """Check if an edge exists by its ID.

        Args:
            edge_id: Unique identifier to check
            edge_type: Optional edge type to filter by

        Returns:
            True if edge exists, False otherwise
        """
        raise NotImplementedError("Subclasses must implement edge_exists")

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
        raise NotImplementedError("Subclasses must implement execute_query")

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
        raise NotImplementedError("Subclasses must implement execute_write")

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
        raise NotImplementedError("Subclasses must implement execute_read")

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
        raise NotImplementedError("Subclasses must implement execute_in_transaction")

    def count_nodes(self, label: Optional[str] = None) -> int:
        """Count nodes in the graph, optionally filtered by label.

        Args:
            label: Optional Neo4j label to filter by

        Returns:
            Number of matching nodes
        """
        raise NotImplementedError("Subclasses must implement count_nodes")

    def count_edges(self, edge_type: Optional[str] = None) -> int:
        """Count edges in the graph, optionally filtered by type.

        Args:
            edge_type: Optional edge type to filter by

        Returns:
            Number of matching edges
        """
        raise NotImplementedError("Subclasses must implement count_edges")

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
        raise NotImplementedError("Subclasses must implement create_index")

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
        raise NotImplementedError("Subclasses must implement create_fulltext_index")

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
        raise NotImplementedError("Subclasses must implement create_constraint")

    def drop_index(self, index_name: str, if_exists: bool = True) -> None:
        """Drop an index by name.

        Args:
            index_name: Name of the index to drop
            if_exists: If True, skip if index doesn't exist
        """
        raise NotImplementedError("Subclasses must implement drop_index")

    def drop_constraint(self, constraint_name: str, if_exists: bool = True) -> None:
        """Drop a constraint by name.

        Args:
            constraint_name: Name of the constraint to drop
            if_exists: If True, skip if constraint doesn't exist
        """
        raise NotImplementedError("Subclasses must implement drop_constraint")

    def list_indexes(self) -> list[dict[str, Any]]:
        """List all indexes in the database.

        Returns:
            List of index metadata dictionaries
        """
        raise NotImplementedError("Subclasses must implement list_indexes")

    def list_constraints(self) -> list[dict[str, Any]]:
        """List all constraints in the database.

        Returns:
            List of constraint metadata dictionaries
        """
        raise NotImplementedError("Subclasses must implement list_constraints")

    def get_schema_info(self) -> dict[str, Any]:
        """Get database schema information.

        Returns:
            Dictionary with schema metadata including labels,
            relationship types, and property keys
        """
        raise NotImplementedError("Subclasses must implement get_schema_info")

    def apply_cql_file(self, file_path: str) -> list[Result]:
        """Apply Cypher statements from a file.

        Reads a .cql file and executes each statement. Statements
        are split by semicolons.

        Args:
            file_path: Path to the CQL file

        Returns:
            List of Result objects for each statement
        """
        raise NotImplementedError("Subclasses must implement apply_cql_file")

    # ------------------------------------------------------------------
    # Context Manager Support
    # ------------------------------------------------------------------

    def __enter__(self) -> Neo4jRepository:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and close connection."""
        self.close()
