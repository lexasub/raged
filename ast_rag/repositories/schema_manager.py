"""
schema_manager.py - Schema management for Neo4j graph database.

Provides SchemaManager class for database schema operations:
- Applying CQL schema files
- Creating and managing indexes
- Creating and managing constraints
- Schema validation and introspection

This module handles the database schema lifecycle, ensuring that
the graph structure matches the expected schema for AST-RAG.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from neo4j import Driver, Session, Result

from ast_rag.dto import Neo4jConfig

logger = logging.getLogger(__name__)


class SchemaManager:
    """Manager for Neo4j database schema operations.

    This class provides methods for applying schema definitions,
    creating indexes and constraints, and validating the database
    schema.

    Usage::

        config = Neo4jConfig(uri="bolt://localhost:7687")
        driver = GraphDatabase.driver(config.uri, auth=(config.user, config.password))
        
        schema_mgr = SchemaManager(driver)
        
        # Apply schema from file
        schema_mgr.apply_schema_file("schema/graph_schema.cql")
        
        # Create indexes
        schema_mgr.create_index("Function", "name")
        
        # Validate schema
        is_valid = schema_mgr.validate_schema()

    Attributes:
        driver: Neo4j driver instance
        config: Neo4j configuration (optional)
    """

    # Default schema file path relative to project root
    DEFAULT_SCHEMA_PATH = Path(__file__).parent / "schema" / "graph_schema.cql"

    # Standard indexes for AST-RAG schema
    STANDARD_INDEXES = [
        # Node property indexes
        ("ASTNode", "id", "ast_node_id_idx"),
        ("ASTNode", "qualified_name", "ast_node_qualified_name_idx"),
        ("ASTNode", "file_path", "ast_node_file_path_idx"),
        ("Function", "name", "function_name_idx"),
        ("Function", "signature", "function_signature_idx"),
        ("Class", "name", "class_name_idx"),
        # Full-text indexes
        ("ast_symbol_fulltext", ["Function", "Class", "Method"], ["name", "qualified_name"]),
    ]

    # Standard constraints for AST-RAG schema
    STANDARD_CONSTRAINTS = [
        ("ast_node_id_unique", "ASTNode", "id", "UNIQUE"),
        ("function_id_unique", "Function", "id", "UNIQUE"),
        ("class_id_unique", "Class", "id", "UNIQUE"),
    ]

    def __init__(
        self,
        driver: Driver,
        config: Optional[Neo4jConfig] = None,
    ) -> None:
        """Initialize schema manager.

        Args:
            driver: Neo4j driver instance
            config: Optional Neo4j configuration
        """
        self._driver = driver
        self._config = config

    # ------------------------------------------------------------------
    # Schema Application
    # ------------------------------------------------------------------

    def apply_schema_file(
        self,
        schema_path: Optional[Path] = None,
        skip_failed: bool = True,
    ) -> dict[str, Any]:
        """Apply Cypher schema from a file.

        Reads a .cql file and executes each statement. Statements
        are split by semicolons. Comments (// style) are stripped.

        Args:
            schema_path: Path to CQL file (defaults to DEFAULT_SCHEMA_PATH)
            skip_failed: If True, log warnings and continue on errors

        Returns:
            Dictionary with statistics:
            - statements_executed: Number of statements executed
            - statements_failed: Number of statements that failed
            - errors: List of error messages if skip_failed=True

        Example::

            result = schema_mgr.apply_schema_file()
            print(f"Applied {result['statements_executed']} statements")
        """
        raise NotImplementedError("Subclasses must implement apply_schema_file")

    def apply_schema_statements(
        self,
        statements: list[str],
        skip_failed: bool = True,
    ) -> dict[str, Any]:
        """Apply a list of Cypher statements.

        Args:
            statements: List of Cypher statement strings
            skip_failed: If True, log warnings and continue on errors

        Returns:
            Dictionary with execution statistics
        """
        raise NotImplementedError("Subclasses must implement apply_schema_statements")

    def apply_single_statement(
        self,
        statement: str,
        parameters: Optional[dict[str, Any]] = None,
    ) -> Result:
        """Apply a single Cypher statement.

        Args:
            statement: Cypher statement string
            parameters: Optional query parameters

        Returns:
            Neo4j Result object
        """
        raise NotImplementedError("Subclasses must implement apply_single_statement")

    # ------------------------------------------------------------------
    # Index Management
    # ------------------------------------------------------------------

    def create_index(
        self,
        label: str,
        property_name: str,
        index_name: Optional[str] = None,
        if_not_exists: bool = True,
    ) -> bool:
        """Create a B-tree index on a node property.

        Args:
            label: Neo4j node label (e.g., "Function", "Class")
            property_name: Property to index
            index_name: Optional custom index name
            if_not_exists: If True, skip if index already exists

        Returns:
            True if index was created, False if it already existed
        """
        raise NotImplementedError("Subclasses must implement create_index")

    def create_fulltext_index(
        self,
        index_name: str,
        labels: list[str],
        properties: list[str],
        if_not_exists: bool = True,
        analyzer: Optional[str] = None,
    ) -> bool:
        """Create a full-text index for text search.

        Full-text indexes enable efficient text search across
        multiple labels and properties.

        Args:
            index_name: Name of the full-text index
            labels: List of node labels to index
            properties: List of properties to include in index
            if_not_exists: If True, skip if index already exists
            analyzer: Optional custom analyzer (default: "standard")

        Returns:
            True if index was created, False if it already existed

        Example::

            schema_mgr.create_fulltext_index(
                index_name="ast_symbol_fulltext",
                labels=["Function", "Class", "Method"],
                properties=["name", "qualified_name"]
            )
        """
        raise NotImplementedError("Subclasses must implement create_fulltext_index")

    def create_vector_index(
        self,
        index_name: str,
        label: str,
        property_name: str,
        dimensions: int = 1024,
        similarity_fn: str = "cosine",
        if_not_exists: bool = True,
    ) -> bool:
        """Create a vector index for approximate nearest neighbor search.

        Requires Neo4j 5.16+ with vector search plugin.

        Args:
            index_name: Name of the vector index
            label: Node label to index
            property_name: Property containing vector embeddings
            dimensions: Vector dimension (e.g., 1024 for bge-m3)
            similarity_fn: Similarity function ("cosine", "euclidean", "dot")
            if_not_exists: If True, skip if index already exists

        Returns:
            True if index was created, False if it already existed
        """
        raise NotImplementedError("Subclasses must implement create_vector_index")

    def drop_index(
        self,
        index_name: str,
        if_exists: bool = True,
    ) -> bool:
        """Drop an index by name.

        Args:
            index_name: Name of the index to drop
            if_exists: If True, skip if index doesn't exist

        Returns:
            True if index was dropped, False if it didn't exist
        """
        raise NotImplementedError("Subclasses must implement drop_index")

    def list_indexes(self) -> list[dict[str, Any]]:
        """List all indexes in the database.

        Returns:
            List of index metadata dictionaries with keys:
            - name: Index name
            - type: Index type (RANGE, FULLTEXT, VECTOR)
            - label: Node label (if applicable)
            - properties: Indexed properties
            - state: Index state (ONLINE, POPULATING, etc.)
        """
        raise NotImplementedError("Subclasses must implement list_indexes")

    def index_exists(self, index_name: str) -> bool:
        """Check if an index exists by name.

        Args:
            index_name: Name of the index

        Returns:
            True if index exists, False otherwise
        """
        raise NotImplementedError("Subclasses must implement index_exists")

    def create_standard_indexes(self) -> dict[str, Any]:
        """Create all standard indexes for AST-RAG schema.

        Creates indexes defined in STANDARD_INDEXES class attribute.

        Returns:
            Dictionary with creation statistics
        """
        raise NotImplementedError("Subclasses must implement create_standard_indexes")

    # ------------------------------------------------------------------
    # Constraint Management
    # ------------------------------------------------------------------

    def create_constraint(
        self,
        constraint_name: str,
        label: str,
        property_name: str,
        constraint_type: str = "UNIQUE",
        if_not_exists: bool = True,
    ) -> bool:
        """Create a constraint on a node property.

        Supported constraint types:
        - UNIQUE: Ensures property value is unique across nodes
        - NOT_NULL: Ensures property value is not null (Neo4j 5+)
        - PROPERTY_EXISTENCE: Ensures property exists

        Args:
            constraint_name: Name of the constraint
            label: Node label
            property_name: Property to constrain
            constraint_type: Type of constraint
            if_not_exists: If True, skip if constraint already exists

        Returns:
            True if constraint was created, False if it already existed
        """
        raise NotImplementedError("Subclasses must implement create_constraint")

    def drop_constraint(
        self,
        constraint_name: str,
        if_exists: bool = True,
    ) -> bool:
        """Drop a constraint by name.

        Args:
            constraint_name: Name of the constraint
            if_exists: If True, skip if constraint doesn't exist

        Returns:
            True if constraint was dropped, False if it didn't exist
        """
        raise NotImplementedError("Subclasses must implement drop_constraint")

    def list_constraints(self) -> list[dict[str, Any]]:
        """List all constraints in the database.

        Returns:
            List of constraint metadata dictionaries with keys:
            - name: Constraint name
            - type: Constraint type (UNIQUE, NOT_NULL, etc.)
            - label: Node label
            - properties: Constrained properties
        """
        raise NotImplementedError("Subclasses must implement list_constraints")

    def constraint_exists(self, constraint_name: str) -> bool:
        """Check if a constraint exists by name.

        Args:
            constraint_name: Name of the constraint

        Returns:
            True if constraint exists, False otherwise
        """
        raise NotImplementedError("Subclasses must implement constraint_exists")

    def create_standard_constraints(self) -> dict[str, Any]:
        """Create all standard constraints for AST-RAG schema.

        Creates constraints defined in STANDARD_CONSTRAINTS class attribute.

        Returns:
            Dictionary with creation statistics
        """
        raise NotImplementedError("Subclasses must implement create_standard_constraints")

    # ------------------------------------------------------------------
    # Schema Validation
    # ------------------------------------------------------------------

    def validate_schema(self) -> dict[str, Any]:
        """Validate that the database schema matches expected state.

        Checks for:
        - Required indexes exist
        - Required constraints exist
        - Expected labels are present
        - Schema version matches (if versioning is enabled)

        Returns:
            Dictionary with validation results:
            - is_valid: Overall validation status
            - missing_indexes: List of missing index names
            - missing_constraints: List of missing constraint names
            - warnings: List of warning messages
            - errors: List of error messages
        """
        raise NotImplementedError("Subclasses must implement validate_schema")

    def validate_indexes(self) -> dict[str, Any]:
        """Validate that all required indexes exist.

        Returns:
            Dictionary with:
            - is_valid: True if all indexes exist
            - missing: List of missing index names
            - present: List of present index names
        """
        raise NotImplementedError("Subclasses must implement validate_indexes")

    def validate_constraints(self) -> dict[str, Any]:
        """Validate that all required constraints exist.

        Returns:
            Dictionary with:
            - is_valid: True if all constraints exist
            - missing: List of missing constraint names
            - present: List of present constraint names
        """
        raise NotImplementedError("Subclasses must implement validate_constraints")

    def get_schema_version(self) -> Optional[str]:
        """Get the current schema version from the database.

        Returns:
            Schema version string or None if not set
        """
        raise NotImplementedError("Subclasses must implement get_schema_version")

    def set_schema_version(self, version: str) -> None:
        """Set the schema version in the database.

        Args:
            version: Schema version string (e.g., "1.0.0")
        """
        raise NotImplementedError("Subclasses must implement set_schema_version")

    # ------------------------------------------------------------------
    # Schema Introspection
    # ------------------------------------------------------------------

    def get_schema_info(self) -> dict[str, Any]:
        """Get comprehensive database schema information.

        Returns:
            Dictionary with:
            - labels: List of node labels with counts
            - relationship_types: List of relationship types with counts
            - property_keys: List of all property keys
            - indexes: List of index metadata
            - constraints: List of constraint metadata
        """
        raise NotImplementedError("Subclasses must implement get_schema_info")

    def get_node_labels(self) -> list[str]:
        """Get all node labels in the database.

        Returns:
            List of label names
        """
        raise NotImplementedError("Subclasses must implement get_node_labels")

    def get_relationship_types(self) -> list[str]:
        """Get all relationship types in the database.

        Returns:
            List of relationship type names
        """
        raise NotImplementedError("Subclasses must implement get_relationship_types")

    def get_property_keys(self) -> list[str]:
        """Get all property keys in the database.

        Returns:
            List of property key names
        """
        raise NotImplementedError("Subclasses must implement get_property_keys")

    def get_label_counts(self) -> dict[str, int]:
        """Get node counts per label.

        Returns:
            Dictionary mapping label names to node counts
        """
        raise NotImplementedError("Subclasses must implement get_label_counts")

    def get_relationship_counts(self) -> dict[str, int]:
        """Get relationship counts per type.

        Returns:
            Dictionary mapping relationship types to counts
        """
        raise NotImplementedError("Subclasses must implement get_relationship_counts")

    # ------------------------------------------------------------------
    # Schema Migration
    # ------------------------------------------------------------------

    def migrate_schema(
        self,
        from_version: str,
        to_version: str,
        migration_script: Path,
    ) -> dict[str, Any]:
        """Run a schema migration script.

        Args:
            from_version: Starting schema version
            to_version: Target schema version
            migration_script: Path to migration CQL file

        Returns:
            Dictionary with migration results:
            - success: True if migration succeeded
            - statements_executed: Number of statements run
            - errors: List of errors if any
        """
        raise NotImplementedError("Subclasses must implement migrate_schema")

    def rollback_migration(
        self,
        from_version: str,
        to_version: str,
        rollback_script: Path,
    ) -> dict[str, Any]:
        """Rollback a schema migration.

        Args:
            from_version: Current schema version
            to_version: Target version after rollback
            rollback_script: Path to rollback CQL file

        Returns:
            Dictionary with rollback results
        """
        raise NotImplementedError("Subclasses must implement rollback_migration")

    # ------------------------------------------------------------------
    # Context Manager Support
    # ------------------------------------------------------------------

    def __enter__(self) -> SchemaManager:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        # SchemaManager doesn't own the driver, so we don't close it
        pass
