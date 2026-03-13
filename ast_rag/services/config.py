"""AST-RAG Services Configuration.

Unified configuration for all AST-RAG services.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ast_rag.dto import Neo4jConfig, QdrantConfig, EmbeddingConfig, ProjectConfig


@dataclass
class LLMConfig:
    """LLM configuration for summarization and analysis.

    Attributes:
        url: LLM API base URL (e.g., "http://localhost:11434/v1")
        model: Model name to use
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds
    """
    url: Optional[str] = None
    model: str = "qwen2.5-coder:14b"
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: int = 120


@dataclass
class ServiceConfig:
    """Unified configuration for AST-RAG services.

    This configuration class provides a simple interface for initializing
    all AST-RAG services through a single configuration object.

    Attributes:
        neo4j: Neo4j graph database configuration
        qdrant: Qdrant vector database configuration
        embedding: Embedding model configuration
        llm: LLM configuration for summarization
        exclude_patterns: Patterns to exclude during indexing

    Example:
        >>> config = ServiceConfig(
        ...     neo4j_uri="bolt://localhost:7687",
        ...     neo4j_user="neo4j",
        ...     neo4j_password="password",
        ...     qdrant_url="http://localhost:6333",
        ...     llm_url="http://localhost:11434/v1"
        ... )
    """
    # Neo4j configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    neo4j_timeout: int = 30
    neo4j_max_pool_size: int = 50

    # Qdrant configuration
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "ast_rag_nodes"
    qdrant_local_path: Optional[str] = None
    qdrant_timeout: int = 60

    # Embedding configuration
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_remote_url: Optional[str] = None
    embedding_remote_batch_size: int = 32
    embedding_dimension: int = 1024
    embedding_timeout: int = 120

    # LLM configuration
    llm_url: Optional[str] = None
    llm_model: str = "qwen2.5-coder:14b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    llm_timeout: int = 120

    # Indexing configuration
    exclude_patterns: list[str] = field(default_factory=lambda: [
        ".git", "__pycache__", "node_modules", "target", "build", "dist",
        ".gradle", ".idea", ".vscode", "venv", ".venv",
    ])

    def to_neo4j_config(self) -> Neo4jConfig:
        """Convert to Neo4jConfig."""
        return Neo4jConfig(
            uri=self.neo4j_uri,
            user=self.neo4j_user,
            password=self.neo4j_password,
            database=self.neo4j_database,
        )

    def to_qdrant_config(self) -> QdrantConfig:
        """Convert to QdrantConfig."""
        return QdrantConfig(
            url=self.qdrant_url,
            collection_name=self.qdrant_collection,
            local_path=self.qdrant_local_path,
        )

    def to_embedding_config(self) -> EmbeddingConfig:
        """Convert to EmbeddingConfig."""
        return EmbeddingConfig(
            model_name=self.embedding_model,
            device=self.embedding_device,
            remote_url=self.embedding_remote_url,
            remote_batch_size=self.embedding_remote_batch_size,
            dimension=self.embedding_dimension,
        )

    def to_llm_config(self) -> LLMConfig:
        """Convert to LLMConfig."""
        return LLMConfig(
            url=self.llm_url,
            model=self.llm_model,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
            timeout=self.llm_timeout,
        )

    def to_project_config(self) -> ProjectConfig:
        """Convert to ProjectConfig."""
        return ProjectConfig(
            neo4j=self.to_neo4j_config(),
            qdrant=self.to_qdrant_config(),
            embedding=self.to_embedding_config(),
            exclude_patterns=self.exclude_patterns,
        )

    @classmethod
    def from_json(cls, config_path: str) -> "ServiceConfig":
        """Load configuration from JSON file.

        Args:
            config_path: Path to JSON configuration file

        Returns:
            ServiceConfig instance loaded from file

        Example:
            >>> config = ServiceConfig.from_json("ast_rag_config.json")
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        data = json.loads(path.read_text(encoding="utf-8"))

        # Map from ProjectConfig format to ServiceConfig format
        neo4j = data.get("neo4j", {})
        qdrant = data.get("qdrant", {})
        embedding = data.get("embedding", {})

        return cls(
            neo4j_uri=neo4j.get("uri", "bolt://localhost:7687"),
            neo4j_user=neo4j.get("user", "neo4j"),
            neo4j_password=neo4j.get("password", "password"),
            neo4j_timeout=neo4j.get("connection_timeout", 30),
            neo4j_max_pool_size=neo4j.get("max_connection_pool_size", 50),

            qdrant_url=qdrant.get("url", "http://localhost:6333"),
            qdrant_collection=qdrant.get("collection_name", "ast_rag_nodes"),
            qdrant_timeout=qdrant.get("timeout", 60),

            embedding_model=embedding.get("model_name", "BAAI/bge-m3"),
            embedding_remote_url=embedding.get("remote_url"),
            embedding_remote_batch_size=embedding.get("remote_batch_size", 32),
            embedding_dimension=embedding.get("dimension", 1024),
            embedding_timeout=embedding.get("timeout", 120),
        )
