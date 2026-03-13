"""
qdrant_repository.py - Qdrant vector database repository layer.

Provides QdrantRepository class for all vector database operations:
- Vector search (similarity, range, filtered)
- Point operations (upsert, delete, retrieve)
- Collection management (create, delete, list)
- Payload filtering and indexing

This repository abstracts Qdrant client operations and provides
a clean API for semantic search and vector indexing.
"""

from __future__ import annotations

from typing import Any, Optional
from collections.abc import Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PointIdsList,
    Filter,
    FieldCondition,
    MatchValue,
    MatchText,
    Range,
    SearchParams,
    ScoredPoint,
)

from ast_rag.dto import QdrantConfig, SearchResult, ASTNode


class QdrantRepository:
    """Repository for Qdrant vector database operations.

    This class provides a high-level API for interacting with Qdrant,
    abstracting away the low-level client operations and providing
    convenient methods for vector search and point management.

    Usage::

        config = QdrantConfig(url="http://localhost:6333", collection_name="ast_rag")
        repo = QdrantRepository(config)
        
        # Upsert points
        points = [PointStruct(id=1, vector=[...], payload={...})]
        repo.upsert_points(points)
        
        # Search
        results = repo.search_vectors(query_vector=[...], limit=10)

    Attributes:
        config: Qdrant configuration
        client: Qdrant client instance (lazy-initialized)
    """

    def __init__(self, config: QdrantConfig) -> None:
        """Initialize Qdrant repository with configuration.

        Args:
            config: Qdrant connection configuration
        """
        self._config = config
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        """Get or create Qdrant client instance (lazy initialization)."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> QdrantClient:
        """Create Qdrant client from configuration."""
        raise NotImplementedError("Subclasses must implement _create_client")

    def close(self) -> None:
        """Close the Qdrant client connection."""
        if self._client is not None:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # Collection Management
    # ------------------------------------------------------------------

    def create_collection(
        self,
        collection_name: Optional[str] = None,
        vector_size: int = 1024,
        distance: Distance = Distance.COSINE,
        vectors_config: Optional[VectorParams] = None,
        recreate: bool = False,
    ) -> None:
        """Create a new collection for storing vectors.

        Args:
            collection_name: Name of the collection (defaults to config value)
            vector_size: Dimension of vectors (e.g., 1024 for bge-m3)
            distance: Distance metric for similarity search
            vectors_config: Optional VectorParams object (overrides other params)
            recreate: If True, delete existing collection first

        Example::

            repo.create_collection(
                vector_size=1024,
                distance=Distance.COSINE
            )
        """
        raise NotImplementedError("Subclasses must implement create_collection")

    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """Delete a collection.

        Args:
            collection_name: Name of the collection (defaults to config value)

        Returns:
            True if collection was deleted, False if it didn't exist
        """
        raise NotImplementedError("Subclasses must implement delete_collection")

    def collection_exists(self, collection_name: Optional[str] = None) -> bool:
        """Check if a collection exists.

        Args:
            collection_name: Name of the collection (defaults to config value)

        Returns:
            True if collection exists, False otherwise
        """
        raise NotImplementedError("Subclasses must implement collection_exists")

    def get_collection_info(
        self,
        collection_name: Optional[str] = None,
    ) -> Optional[CollectionInfo]:
        """Get information about a collection.

        Args:
            collection_name: Name of the collection (defaults to config value)

        Returns:
            CollectionInfo object if collection exists, None otherwise
        """
        raise NotImplementedError("Subclasses must implement get_collection_info")

    def list_collections(self) -> list[str]:
        """List all collections in the Qdrant instance.

        Returns:
            List of collection names
        """
        raise NotImplementedError("Subclasses must implement list_collections")

    def get_collection_size(self, collection_name: Optional[str] = None) -> int:
        """Get the number of points in a collection.

        Args:
            collection_name: Name of the collection (defaults to config value)

        Returns:
            Number of points in the collection
        """
        raise NotImplementedError("Subclasses must implement get_collection_size")

    # ------------------------------------------------------------------
    # Point Operations (Upsert/Delete/Retrieve)
    # ------------------------------------------------------------------

    def upsert_point(
        self,
        point_id: str | int,
        vector: list[float],
        payload: Optional[dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> bool:
        """Upsert a single point (create or update).

        Args:
            point_id: Unique identifier for the point
            vector: Vector embedding as list of floats
            payload: Optional metadata dictionary
            collection_name: Collection name (defaults to config value)

        Returns:
            True if operation succeeded
        """
        raise NotImplementedError("Subclasses must implement upsert_point")

    def upsert_points(
        self,
        points: Iterable[PointStruct],
        collection_name: Optional[str] = None,
        batch_size: int = 64,
        wait: bool = True,
    ) -> bool:
        """Batch upsert multiple points.

        Args:
            points: Iterable of PointStruct objects
            collection_name: Collection name (defaults to config value)
            batch_size: Number of points per batch
            wait: If True, wait for operation to complete

        Returns:
            True if operation succeeded
        """
        raise NotImplementedError("Subclasses must implement upsert_points")

    def retrieve_points(
        self,
        point_ids: list[str | int],
        collection_name: Optional[str] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> list[ScoredPoint]:
        """Retrieve points by their IDs.

        Args:
            point_ids: List of point identifiers
            collection_name: Collection name (defaults to config value)
            with_payload: Include payload in response
            with_vectors: Include vectors in response

        Returns:
            List of ScoredPoint objects
        """
        raise NotImplementedError("Subclasses must implement retrieve_points")

    def retrieve_point(
        self,
        point_id: str | int,
        collection_name: Optional[str] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> Optional[ScoredPoint]:
        """Retrieve a single point by its ID.

        Args:
            point_id: Point identifier
            collection_name: Collection name (defaults to config value)
            with_payload: Include payload in response
            with_vectors: Include vectors in response

        Returns:
            ScoredPoint if found, None otherwise
        """
        raise NotImplementedError("Subclasses must implement retrieve_point")

    def delete_points(
        self,
        point_ids: list[str | int],
        collection_name: Optional[str] = None,
        wait: bool = True,
    ) -> bool:
        """Delete points by their IDs.

        Args:
            point_ids: List of point identifiers to delete
            collection_name: Collection name (defaults to config value)
            wait: If True, wait for operation to complete

        Returns:
            True if operation succeeded
        """
        raise NotImplementedError("Subclasses must implement delete_points")

    def delete_point(
        self,
        point_id: str | int,
        collection_name: Optional[str] = None,
        wait: bool = True,
    ) -> bool:
        """Delete a single point by its ID.

        Args:
            point_id: Point identifier to delete
            collection_name: Collection name (defaults to config value)
            wait: If True, wait for operation to complete

        Returns:
            True if operation succeeded
        """
        raise NotImplementedError("Subclasses must implement delete_point")

    def delete_by_filter(
        self,
        filter_condition: Filter,
        collection_name: Optional[str] = None,
        wait: bool = True,
    ) -> bool:
        """Delete points matching a filter condition.

        Args:
            filter_condition: Qdrant Filter object
            collection_name: Collection name (defaults to config value)
            wait: If True, wait for operation to complete

        Returns:
            True if operation succeeded
        """
        raise NotImplementedError("Subclasses must implement delete_by_filter")

    # ------------------------------------------------------------------
    # Vector Search Operations
    # ------------------------------------------------------------------

    def search_vectors(
        self,
        query_vector: list[float],
        limit: int = 10,
        filter_condition: Optional[Filter] = None,
        collection_name: Optional[str] = None,
        score_threshold: Optional[float] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> list[ScoredPoint]:
        """Search for similar vectors.

        Args:
            query_vector: Query vector embedding
            limit: Maximum number of results
            filter_condition: Optional payload filter
            collection_name: Collection name (defaults to config value)
            score_threshold: Minimum similarity score threshold
            with_payload: Include payload in response
            with_vectors: Include vectors in response

        Returns:
            List of ScoredPoint objects sorted by similarity
        """
        raise NotImplementedError("Subclasses must implement search_vectors")

    def search_with_payload(
        self,
        query_vector: list[float],
        limit: int = 10,
        filter_condition: Optional[Filter] = None,
        collection_name: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> list[SearchResult]:
        """Search vectors and return as SearchResult objects.

        Convenience method that converts ScoredPoint results to
        SearchResult objects with ASTNode payload.

        Args:
            query_vector: Query vector embedding
            limit: Maximum number of results
            filter_condition: Optional payload filter
            collection_name: Collection name (defaults to config value)
            score_threshold: Minimum similarity score threshold

        Returns:
            List of SearchResult objects
        """
        raise NotImplementedError("Subclasses must implement search_with_payload")

    def search_batch(
        self,
        query_vectors: list[list[float]],
        limit: int = 10,
        filter_condition: Optional[Filter] = None,
        collection_name: Optional[str] = None,
    ) -> list[list[ScoredPoint]]:
        """Search with multiple query vectors in a single request.

        Args:
            query_vectors: List of query vector embeddings
            limit: Maximum number of results per query
            filter_condition: Optional payload filter
            collection_name: Collection name (defaults to config value)

        Returns:
            List of result lists (one per query vector)
        """
        raise NotImplementedError("Subclasses must implement search_batch")

    def range_search(
        self,
        query_vector: list[float],
        min_score: float = 0.0,
        max_score: float = 1.0,
        limit: int = 100,
        filter_condition: Optional[Filter] = None,
        collection_name: Optional[str] = None,
    ) -> list[ScoredPoint]:
        """Search vectors within a similarity score range.

        Args:
            query_vector: Query vector embedding
            min_score: Minimum similarity score
            max_score: Maximum similarity score
            limit: Maximum number of results
            filter_condition: Optional payload filter
            collection_name: Collection name (defaults to config value)

        Returns:
            List of ScoredPoint objects within score range
        """
        raise NotImplementedError("Subclasses must implement range_search")

    # ------------------------------------------------------------------
    # Filtered Search
    # ------------------------------------------------------------------

    def search_by_field(
        self,
        query_vector: list[float],
        field_name: str,
        field_value: Any,
        limit: int = 10,
        collection_name: Optional[str] = None,
    ) -> list[ScoredPoint]:
        """Search vectors filtered by a payload field match.

        Args:
            query_vector: Query vector embedding
            field_name: Payload field name to filter on
            field_value: Field value to match
            limit: Maximum number of results
            collection_name: Collection name (defaults to config value)

        Returns:
            List of matching ScoredPoint objects
        """
        raise NotImplementedError("Subclasses must implement search_by_field")

    def search_by_fields(
        self,
        query_vector: list[float],
        field_filters: dict[str, Any],
        limit: int = 10,
        collection_name: Optional[str] = None,
        match_all: bool = True,
    ) -> list[ScoredPoint]:
        """Search vectors filtered by multiple payload fields.

        Args:
            query_vector: Query vector embedding
            field_filters: Dictionary of field_name: value pairs
            limit: Maximum number of results
            collection_name: Collection name (defaults to config value)
            match_all: If True, all filters must match (AND); otherwise any (OR)

        Returns:
            List of matching ScoredPoint objects
        """
        raise NotImplementedError("Subclasses must implement search_by_fields")

    def search_with_range_filter(
        self,
        query_vector: list[float],
        field_name: str,
        range_min: Optional[float] = None,
        range_max: Optional[float] = None,
        limit: int = 10,
        collection_name: Optional[str] = None,
    ) -> list[ScoredPoint]:
        """Search vectors filtered by a numeric range.

        Args:
            query_vector: Query vector embedding
            field_name: Payload field name for range filter
            range_min: Minimum value (inclusive)
            range_max: Maximum value (inclusive)
            limit: Maximum number of results
            collection_name: Collection name (defaults to config value)

        Returns:
            List of matching ScoredPoint objects
        """
        raise NotImplementedError("Subclasses must implement search_with_range_filter")

    def search_with_text_match(
        self,
        query_vector: list[float],
        field_name: str,
        query_text: str,
        limit: int = 10,
        collection_name: Optional[str] = None,
    ) -> list[ScoredPoint]:
        """Search vectors with full-text match on a payload field.

        Requires a text index on the specified field.

        Args:
            query_vector: Query vector embedding
            field_name: Payload field name for text match
            query_text: Text to match
            limit: Maximum number of results
            collection_name: Collection name (defaults to config value)

        Returns:
            List of matching ScoredPoint objects
        """
        raise NotImplementedError("Subclasses must implement search_with_text_match")

    # ------------------------------------------------------------------
    # Payload Indexing
    # ------------------------------------------------------------------

    def create_payload_index(
        self,
        field_name: str,
        index_type: str = "keyword",
        collection_name: Optional[str] = None,
    ) -> None:
        """Create an index on a payload field for faster filtering.

        Args:
            field_name: Payload field to index
            index_type: Index type ("keyword", "integer", "float", "text", "bool")
            collection_name: Collection name (defaults to config value)
        """
        raise NotImplementedError("Subclasses must implement create_payload_index")

    def drop_payload_index(
        self,
        field_name: str,
        collection_name: Optional[str] = None,
    ) -> None:
        """Drop a payload index.

        Args:
            field_name: Payload field index to drop
            collection_name: Collection name (defaults to config value)
        """
        raise NotImplementedError("Subclasses must implement drop_payload_index")

    # ------------------------------------------------------------------
    # Filter Helpers
    # ------------------------------------------------------------------

    def build_match_filter(
        self,
        field_name: str,
        value: Any,
    ) -> FieldCondition:
        """Build a field match filter condition.

        Args:
            field_name: Payload field name
            value: Value to match

        Returns:
            FieldCondition for use in Filter
        """
        raise NotImplementedError("Subclasses must implement build_match_filter")

    def build_range_filter(
        self,
        field_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> FieldCondition:
        """Build a range filter condition.

        Args:
            field_name: Payload field name
            min_value: Minimum value (inclusive)
            max_value: Maximum value (inclusive)

        Returns:
            FieldCondition for use in Filter
        """
        raise NotImplementedError("Subclasses must implement build_range_filter")

    def build_text_filter(
        self,
        field_name: str,
        query: str,
    ) -> FieldCondition:
        """Build a full-text match filter condition.

        Args:
            field_name: Payload field name
            query: Text query

        Returns:
            FieldCondition for use in Filter
        """
        raise NotImplementedError("Subclasses must implement build_text_filter")

    def build_geo_filter(
        self,
        field_name: str,
        center_lat: float,
        center_lon: float,
        radius_meters: float,
    ) -> FieldCondition:
        """Build a geo radius filter condition.

        Args:
            field_name: Payload field name (must be geo coordinate type)
            center_lat: Center latitude
            center_lon: Center longitude
            radius_meters: Search radius in meters

        Returns:
            FieldCondition for use in Filter
        """
        raise NotImplementedError("Subclasses must implement build_geo_filter")

    # ------------------------------------------------------------------
    # Context Manager Support
    # ------------------------------------------------------------------

    def __enter__(self) -> QdrantRepository:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and close connection."""
        self.close()
