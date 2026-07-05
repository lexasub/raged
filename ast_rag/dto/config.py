"""DTO - Configuration models.

Defines configuration data structures for Neo4j, Qdrant, embeddings, and project settings.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Neo4jConfig(BaseModel):
    """Neo4j graph database configuration.

    Attributes:
        uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
        user: Database user name
        password: Database password
        database: Database name (Community Edition: always "neo4j")
        project_id: Project identifier for data isolation (prefix-based)
    """

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    project_id: str = "default"


class QdrantConfig(BaseModel):
    """Qdrant vector database configuration.

    Attributes:
        url: Qdrant server URL (e.g., "http://localhost:6333")
        collection_name: Collection name for embeddings
        local_path: If set, use local file mode instead of server
    """

    url: str = "http://localhost:6333"
    collection_name: str = "ast_rag_nodes"
    local_path: Optional[str] = None


class EmbeddingConfig(BaseModel):
    """Embedding model configuration.

    Attributes:
        model_name: Sentence transformer model name
        device: Device for model inference ("cpu" or "cuda")
        remote_url: Remote embedding server URL (optional)
        dimension: Embedding dimension for remote encoding
        remote_batch_size: Batch size for remote encoding
        hybrid_search: Enable hybrid search (vector + keyword)
        vector_weight: Weight for vector similarity scores
        keyword_weight: Weight for keyword search scores
    """

    model_name: str = "BAAI/bge-m3"
    device: str = "cpu"
    remote_url: Optional[str] = None
    dimension: Optional[int] = None
    remote_batch_size: int = 32
    hybrid_search: bool = True
    vector_weight: float = 0.7
    keyword_weight: float = 0.3


class ParseCacheConfig(BaseModel):
    """Parse cache configuration.

    Attributes:
        max_entries: Maximum number of cached parse trees (default: 10000)
        max_size_mb: Maximum memory usage in MB (default: 500)
        persistence_enabled: Use SQLite persistent cache instead of in-memory
        db_path: Path for the SQLite cache file
    """

    max_entries: int = 10_000
    max_size_mb: int = 500
    persistence_enabled: bool = False
    db_path: str = ".ast_rag_parse_cache.sqlite"


class ProjectConfig(BaseModel):
    """Top-level configuration for the AST-RAG system.

    Attributes:
        neo4j: Neo4j configuration
        qdrant: Qdrant configuration
        embedding: Embedding configuration
        parse_cache: Parse cache configuration
        language_extensions: File extensions per language
        exclude_patterns: Patterns to exclude during indexing
        ignore_file: Path to a .cgrignore-style ignore file
            (default: .cgrignore in the indexed root)
    """

    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    parse_cache: ParseCacheConfig = Field(default_factory=ParseCacheConfig)
    language_extensions: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "cpp": [".cpp", ".cxx", ".cc", ".c", ".hpp", ".hxx", ".hh", ".h"],
            "java": [".java"],
            "rust": [".rs"],
            "python": [".py"],
            "typescript": [".ts", ".tsx"],
        }
    )
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            ".git",
            "__pycache__",
            "node_modules",
            "target",
            "build",
            "dist",
            ".gradle",
            ".idea",
            ".vscode",
            "venv",
            ".venv",
        ]
    )
    ignore_file: Optional[str] = None
