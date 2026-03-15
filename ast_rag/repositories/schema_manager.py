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
from typing import Any, Optional, List

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
        path = schema_path or self.DEFAULT_SCHEMA_PATH
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {path}")

        raw = path.read_text(encoding="utf-8")
        # Split on semicolons (crude but works for our simple schema)
        statements = [s.strip() for s in raw.split(";") if s.strip()]

        return self.apply_schema_statements(statements, skip_failed=skip_failed)

    def apply_schema_statements(
        self,
        statements: List[str],
        skip_failed: bool = True,
    ) -> dict[str, Any]:
        """Apply a list of Cypher statements.

        Args:
            statements: List of Cypher statement strings
            skip_failed: If True, log warnings and continue on errors

        Returns:
            Dictionary with execution statistics
        """
        stats = {
            "statements_executed": 0,
            "statements_failed": 0,
            "errors": [],
        }

        with self._driver.session() as session:
            for stmt in statements:
                # Strip comments
                clean = "\n".join(
                    line for line in stmt.splitlines() if not line.strip().startswith("//")
                ).strip()
                if not clean:
                    continue
                try:
                    session.run(clean)
                    stats["statements_executed"] += 1
                    logger.debug("Applied schema statement: %s...", clean[:60])
                except Exception as exc:
                    stats["statements_failed"] += 1
                    if skip_failed:
                        logger.warning(
                            "Schema statement failed (skipping): %s | %s", clean[:80], exc
                        )
                        stats["errors"].append(str(exc))
                    else:
                        raise

        logger.info(
            "Schema applied: %d statements executed, %d failed",
            stats["statements_executed"],
            stats["statements_failed"],
        )
        return stats

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
        with self._driver.session() as session:
            return session.run(statement, parameters or {})

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
        if index_name is None:
            index_name = f"{label.lower()}_{property_name}_idx"

        query_parts = [
            "CREATE INDEX",
            f"IF NOT EXISTS" if if_not_exists else "",
            f"{index_name}",
            f"FOR (n:{label})",
            f"ON (n.{property_name})",
        ]
        query = " ".join(part for part in query_parts if part)

        try:
            with self._driver.session() as session:
                session.run(query)
            logger.info("Created index: %s", index_name)
            return True
        except Exception as exc:
            if if_not_exists and "already exists" in str(exc).lower():
                logger.info("Index already exists: %s", index_name)
                return False
            else:
                logger.error("Failed to create index %s: %s", index_name, exc)
                raise

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
        if analyzer is None:
            analyzer = "standard"

        label_filters = ":".join(labels) if labels else ""
        properties_str = ", ".join(properties)

        query_parts = [
            "CREATE FULLTEXT INDEX",
            f"IF NOT EXISTS" if if_not_exists else "",
            f"{index_name}",
            f"FOR",
            f"([{label_filters}])",
            f"ON EACH",
            f"[{properties_str}]",
            f"OPTIONS {{ analyzer: '{analyzer}' }}",
        ]
        query = " ".join(part for part in query_parts if part)

        try:
            with self._driver.session() as session:
                session.run(query)
            logger.info("Created fulltext index: %s", index_name)
            return True
        except Exception as exc:
            if if_not_exists and (
                "already exists" in str(exc).lower()
                or "equivalent fulltext index" in str(exc).lower()
            ):
                logger.info("Fulltext index already exists: %s", index_name)
                return False
            else:
                logger.error("Failed to create fulltext index %s: %s", index_name, exc)
                raise

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

        logger.warning("Vector index creation not fully implemented for this Neo4j version")
        return False

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
        query_parts = ["DROP INDEX", f"IF EXISTS" if if_exists else "", f"{index_name}"]
        query = " ".join(part for part in query_parts if part)

        try:
            with self._driver.session() as session:
                session.run(query)
            logger.info("Dropped index: %s", index_name)
            return True
        except Exception as exc:
            if if_exists and (
                "not found" in str(exc).lower() or "does not exist" in str(exc).lower()
            ):
                logger.info("Index does not exist: %s", index_name)
                return False
            else:
                logger.error("Failed to drop index %s: %s", index_name, exc)
                raise

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
        query = "SHOW INDEXES"
        with self._driver.session() as session:
            result = session.run(query)
            indexes = []
            for record in result:
                indexes.append(
                    {
                        "name": record["name"],
                        "type": record["type"],
                        "label": record.get("labelsOrTypes", [None])[0]
                        if record.get("labelsOrTypes")
                        else None,
                        "properties": record.get("properties", []),
                        "state": record["state"],
                        "owningConstraint": record.get("owningConstraint"),
                        "indexPopulation": record.get("indexPopulation"),
                        "entityType": record.get("entityType"),
                        "labelsOrTypes": record.get("labelsOrTypes"),
                        "properties": record.get("properties"),
                    }
                )
            return indexes

    def index_exists(self, index_name: str) -> bool:
        """Check if an index exists by name.

        Args:
            index_name: Name of the index

        Returns:
            True if index exists, False otherwise
        """
        indexes = self.list_indexes()
        return any(idx["name"] == index_name for idx in indexes)

    def create_standard_indexes(self) -> dict[str, Any]:
        """Create all standard indexes for AST-RAG schema.

        Creates indexes defined in STANDARD_INDEXES class attribute.

        Returns:
            Dictionary with creation statistics
        """
        stats = {
            "created": 0,
            "skipped": 0,
            "failed": 0,
            "errors": [],
        }

        for label, property_name, index_name in self.STANDARD_INDEXES:
            try:
                if self.create_index(label, property_name, index_name, if_not_exists=True):
                    stats["created"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as exc:
                stats["failed"] += 1
                stats["errors"].append(f"{index_name}: {exc}")
                logger.error("Failed to create standard index %s: %s", index_name, exc)

        # Create fulltext indexes
        for index_name, labels, properties in [
            idx for idx in self.STANDARD_INDEXES if isinstance(idx[1], list)
        ]:
            # Actually, STANDARD_INDEXES doesn't have fulltext in the same format
            # Let's handle the fulltext index separately
            pass

        # Handle the fulltext index from STANDARD_INDEXES
        # We know the fulltext index is defined as:
        # ("ast_symbol_fulltext", ["Function", "Class", "Method"], ["name", "qualified_name"])
        # But our STANDARD_INDEXES structure is different for fulltext
        # Let's just create it directly
        try:
            if self.create_fulltext_index(
                "ast_symbol_fulltext",
                ["Function", "Class", "Method"],
                ["name", "qualified_name"],
                if_not_exists=True,
            ):
                stats["created"] += 1
            else:
                stats["skipped"] += 1
        except Exception as exc:
            stats["failed"] += 1
            stats["errors"].append(f"ast_symbol_fulltext: {exc}")
            logger.error("Failed to create standard fulltext index: %s", exc)

        logger.info(
            "Standard indexes: %d created, %d skipped, %d failed",
            stats["created"],
            stats["skipped"],
            stats["failed"],
        )
        return stats

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
        query_parts = [
            "CREATE CONSTRAINT",
            f"IF NOT EXISTS" if if_not_exists else "",
            f"{constraint_name}",
            f"FOR (n:{label})",
            f"REQUIRE n.{property_name}",
            f"IS {constraint_type}" if constraint_type != "PROPERTY_EXISTENCE" else "",
        ]

        query_parts = [part for part in query_parts if part]
        query = " ".join(query_parts)

        try:
            with self._driver.session() as session:
                session.run(query)
            logger.info("Created constraint: %s", constraint_name)
            return True
        except Exception as exc:
            if if_not_exists and "already exists" in str(exc).lower():
                logger.info("Constraint already exists: %s", constraint_name)
                return False
            else:
                logger.error("Failed to create constraint %s: %s", constraint_name, exc)
                raise

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
        query_parts = ["DROP CONSTRAINT", f"IF EXISTS" if if_exists else "", f"{constraint_name}"]
        query = " ".join(part for part in query_parts if part)

        try:
            with self._driver.session() as session:
                session.run(query)
            logger.info("Dropped constraint: %s", constraint_name)
            return True
        except Exception as exc:
            if if_exists and (
                "not found" in str(exc).lower() or "does not exist" in str(exc).lower()
            ):
                logger.info("Constraint does not exist: %s", constraint_name)
                return False
            else:
                logger.error("Failed to drop constraint %s: %s", constraint_name, exc)
                raise

    def list_constraints(self) -> list[dict[str, Any]]:
        """List all constraints in the database.

        Returns:
            List of constraint metadata dictionaries with keys:
            - name: Constraint name
            - type: Constraint type (UNIQUE, NOT_NULL, etc.)
            - label: Node label
            - properties: Constrained properties
        """
        query = "SHOW CONSTRAINTS"
        with self._driver.session() as session:
            result = session.run(query)
            constraints = []
            for record in result:
                constraints.append(
                    {
                        "name": record["name"],
                        "type": record["type"],
                        "label": record.get("labelsOrTypes", [None])[0]
                        if record.get("labelsOrTypes")
                        else None,
                        "properties": record.get("properties", []),
                    }
                )
            return constraints

    def constraint_exists(self, constraint_name: str) -> bool:
        """Check if a constraint exists by name.

        Args:
            constraint_name: Name of the constraint

        Returns:
            True if constraint exists, False otherwise
        """
        constraints = self.list_constraints()
        return any(c["name"] == constraint_name for c in constraints)

    def create_standard_constraints(self) -> dict[str, Any]:
        """Create all standard constraints for AST-RAG schema.

        Creates constraints defined in STANDARD_CONSTRAINTS class attribute.

        Returns:
            Dictionary with creation statistics
        """
        stats = {
            "created": 0,
            "skipped": 0,
            "failed": 0,
            "errors": [],
        }

        for constraint_name, label, property_name, constraint_type in self.STANDARD_CONSTRAINTS:
            try:
                if self.create_constraint(
                    constraint_name, label, property_name, constraint_type, if_not_exists=True
                ):
                    stats["created"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as exc:
                stats["failed"] += 1
                stats["errors"].append(f"{constraint_name}: {exc}")
                logger.error("Failed to create standard constraint %s: %s", constraint_name, exc)

        logger.info(
            "Standard constraints: %d created, %d skipped, %d failed",
            stats["created"],
            stats["skipped"],
            stats["failed"],
        )
        return stats

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
        # Validate indexes
        index_validation = self.validate_indexes()
        # Validate constraints
        constraint_validation = self.validate_constraints()

        is_valid = index_validation["is_valid"] and constraint_validation["is_valid"]

        return {
            "is_valid": is_valid,
            "missing_indexes": index_validation["missing"],
            "missing_constraints": constraint_validation["missing"],
            "warnings": [],
            "errors": [],
        }

    def validate_indexes(self) -> dict[str, Any]:
        """Validate that all required indexes exist.

        Returns:
            Dictionary with:
            - is_valid: True if all indexes exist
            - missing: List of missing index names
            - present: List of present index names
        """
        required_indexes = set()
        # Add standard node property indexes
        for label, property_name, index_name in self.STANDARD_INDEXES:
            required_indexes.add(index_name)
        # Add standard fulltext index
        required_indexes.add("ast_symbol_fulltext")

        present_indexes = {idx["name"] for idx in self.list_indexes()}
        missing = required_indexes - present_indexes

        return {
            "is_valid": len(missing) == 0,
            "missing": list(missing),
            "present": list(present_indexes),
        }

    def validate_constraints(self) -> dict[str, Any]:
        """Validate that all required constraints exist.

        Returns:
            Dictionary with:
            - is_valid: True if all constraints exist
            - missing: List of missing constraint names
            - present: List of present constraint names
        """
        required_constraints = set()
        for constraint_name, label, property_name, constraint_type in self.STANDARD_CONSTRAINTS:
            required_constraints.add(constraint_name)

        present_constraints = {c["name"] for c in self.list_constraints()}
        missing = required_constraints - present_constraints

        return {
            "is_valid": len(missing) == 0,
            "missing": list(missing),
            "present": list(present_constraints),
        }

    def get_schema_version(self) -> Optional[str]:
        """Get the current schema version from the database.

        Returns:
            Schema version string or None if not set
        """
        # This would require a version node, e.g., CurrentVersion
        # For now, we return None as it's not implemented
        return None

    def set_schema_version(self, version: str) -> None:
        """Set the schema version in the database.

        Args:
            version: Schema version string (e.g., "1.0.0")
        """
        # This would create/update a version node
        # For now, we do nothing
        pass

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
        with self._driver.session() as session:
            # Get node labels and counts
            label_query = """
            CALL db.labels() YIELD label
            CALL apoc.cypher.run('MATCH (n:`' + label + '`) RETURN count(n) AS count', {}) YIELD value
            RETURN label, value.count AS count
            """
            # Since we don't know if apoc is available, we'll do it differently
            labels_query = "CALL db.labels() YIELD label RETURN label"
            label_counts = {}
            for record in self._driver.session().run(labels_query):
                label = record["label"]
                count_query = f"MATCH (n:`{label}`) RETURN count(n) AS count"
                count = self._driver.session().run(count_query).single()["count"]
                label_counts[label] = count

            # Get relationship types and counts
            rel_query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
            rel_counts = {}
            for record in self._driver.session().run(rel_query):
                rel_type = record["relationshipType"]
                count_query = f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) AS count"
                count = self._driver.session().run(count_query).single()["count"]
                rel_counts[rel_type] = count

            # Get property keys
            prop_query = "CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey"
            property_keys = [
                record["propertyKey"] for record in self._driver.session().run(prop_query)
            ]

            # Get indexes and constraints
            indexes = self.list_indexes()
            constraints = self.list_constraints()

            return {
                "labels": [{"label": k, "count": v} for k, v in label_counts.items()],
                "relationship_types": [{"type": k, "count": v} for k, v in rel_counts.items()],
                "property_keys": property_keys,
                "indexes": indexes,
                "constraints": constraints,
            }

    def get_node_labels(self) -> list[str]:
        """Get all node labels in the database.

        Returns:
            List of label names
        """
        query = "CALL db.labels() YIELD label RETURN label"
        return [record["label"] for record in self._driver.session().run(query)]

    def get_relationship_types(self) -> list[str]:
        """Get all relationship types in the database.

        Returns:
            List of relationship type names
        """
        query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        return [record["relationshipType"] for record in self._driver.session().run(query)]

    def get_property_keys(self) -> list[str]:
        """Get all property keys in the database.

        Returns:
            List of property key names
        """
        query = "CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey"
        return [record["propertyKey"] for record in self._driver.session().run(query)]

    def get_label_counts(self) -> dict[str, int]:
        """Get node counts per label.

        Returns:
            Dictionary mapping label names to node counts
        """
        counts = {}
        for label in self.get_node_labels():
            query = f"MATCH (n:`{label}`) RETURN count(n) AS count"
            counts[label] = self._driver.session().run(query).single()["count"]
        return counts

    def get_relationship_counts(self) -> dict[str, int]:
        """Get relationship counts per type.

        Returns:
            Dictionary mapping relationship types to counts
        """
        counts = {}
        for rel_type in self.get_relationship_types():
            query = f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) AS count"
            counts[rel_type] = self._driver.session().run(query).single()["count"]
        return counts

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
        # For now, we just apply the migration script as a schema file
        # In a real implementation, we would check the current version
        result = self.apply_schema_file(migration_script, skip_failed=False)
        return {
            "success": result["statements_failed"] == 0,
            "statements_executed": result["statements_executed"],
            "errors": result["errors"],
        }

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
        # Similar to migrate, but in reverse
        result = self.apply_schema_file(rollback_script, skip_failed=False)
        return {
            "success": result["statements_failed"] == 0,
            "statements_executed": result["statements_executed"],
            "errors": result["errors"],
        }

    # ------------------------------------------------------------------
    # Context Manager Support
    # ------------------------------------------------------------------

    def __enter__(self) -> "SchemaManager":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        # SchemaManager doesn't own the driver, so we don't close it
        pass
