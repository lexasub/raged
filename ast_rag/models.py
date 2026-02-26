"""
models.py - Pydantic v2 data models for AST-RAG.

Defines data structures for:
- AST nodes (classes, functions, methods, etc.)
- AST edges (relationships between nodes)
- Diff results for MVCC tracking
- Query result wrappers
- System configuration
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
import hashlib

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class NodeKind(str, Enum):
    """All recognised AST node kinds persisted in the graph."""
    PROJECT = "Project"
    PACKAGE = "Package"        # Java package / Python module dir
    NAMESPACE = "Namespace"    # C++ namespace
    MODULE = "Module"          # Python module file / Rust module
    FILE = "File"
    CLASS = "Class"
    INTERFACE = "Interface"    # Java interface
    STRUCT = "Struct"          # C++ struct / Rust struct
    ENUM = "Enum"
    TRAIT = "Trait"            # Rust trait
    FUNCTION = "Function"      # top-level / free function
    METHOD = "Method"          # class member function
    CONSTRUCTOR = "Constructor"
    DESTRUCTOR = "Destructor"  # C++
    FIELD = "Field"            # class member field
    VARIABLE = "Variable"
    PARAMETER = "Parameter"
    CURRENT_VERSION = "CurrentVersion"


class EdgeKind(str, Enum):
    """All edge types in the graph."""
    CONTAINS_PACKAGE = "CONTAINS_PACKAGE"
    CONTAINS_FILE = "CONTAINS_FILE"
    CONTAINS_CLASS = "CONTAINS_CLASS"
    CONTAINS_METHOD = "CONTAINS_METHOD"
    CONTAINS_FUNCTION = "CONTAINS_FUNCTION"
    CONTAINS_FIELD = "CONTAINS_FIELD"
    HAS_PARAMETER = "HAS_PARAMETER"
    IMPORTS = "IMPORTS"      # Java / Python / TS import
    INCLUDES = "INCLUDES"    # C++ #include
    CALLS = "CALLS"
    INHERITS = "INHERITS"    # C++ inheritance
    EXTENDS = "EXTENDS"      # Java extends
    IMPLEMENTS = "IMPLEMENTS"
    INJECTS = "INJECTS"      # DI heuristic: field of another class type
    OVERRIDES = "OVERRIDES"
    DEPENDS_ON = "DEPENDS_ON"
    TYPES = "TYPES"
    VIRTUAL_CALL = "VIRTUAL_CALL"
    LAMBDA_CALL = "LAMBDA_CALL"
    CROSS_FILE_CALL = "CROSS_FILE_CALL"


class Language(str, Enum):
    """Supported source languages."""
    CPP = "cpp"
    JAVA = "java"
    RUST = "rust"
    PYTHON = "python"
    TYPESCRIPT = "typescript"


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------

class ASTNode(BaseModel):
    """Represents a single extracted AST entity.

    The `id` field is a stable, content-addressed identifier derived from
    file_path, kind, and qualified_name. It survives re-parses as long as
    the entity has not moved.
    """
    id: str = Field(default="", description="Stable SHA-256 based identifier")
    kind: NodeKind
    name: str
    qualified_name: str
    lang: Language
    file_path: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    code_hash: str = Field(default="", description="SHA-256 of the raw source text")
    signature: Optional[str] = None  # for functions/methods
    valid_from: str = "INIT"
    valid_to: Optional[str] = None   # None means 'current'
    # Extra storage for raw source text (not persisted to graph, used for embeddings)
    source_text: Optional[str] = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def compute_derived_fields(self) -> "ASTNode":
        """Auto-compute id and code_hash if not provided."""
        if not self.id:
            raw = f"{self.file_path}:{self.kind.value}:{self.qualified_name}"
            self.id = hashlib.sha256(raw.encode()).hexdigest()[:24]
        if not self.code_hash and self.source_text:
            self.code_hash = hashlib.sha256(self.source_text.encode()).hexdigest()[:24]
        return self

    def to_neo4j_props(self) -> dict[str, Any]:
        """Serialize node to a flat dict suitable for Neo4j property map."""
        return {
            "id": self.id,
            "kind": self.kind.value,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "lang": self.lang.value,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "code_hash": self.code_hash,
            "signature": self.signature or "",
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
        }

    def to_standard_result(self, score: Optional[float] = None, edge_type: Optional[str] = None) -> StandardResult:
        """Convert ASTNode to StandardResult."""
        return StandardResult(
            id=self.id,
            name=self.name,
            qualified_name=self.qualified_name,
            kind=self.kind.value,
            lang=self.lang.value,
            file_path=self.file_path,
            start_line=self.start_line,
            end_line=self.end_line,
            score=score,
            edge_type=edge_type,
        )


class ASTEdge(BaseModel):
    """Represents a directed relationship between two AST nodes.

    The `id` is derived from from_id + edge_kind + to_id.
    """
    id: str = Field(default="", description="Stable edge identifier")
    kind: EdgeKind
    from_id: str
    to_id: str
    # Optional label for extra context (e.g. the import path string)
    label: Optional[str] = None
    valid_from: str = "INIT"
    valid_to: Optional[str] = None
    dep_kind: Optional[str] = None  # For DEPENDS_ON: "system", "local", "import"
    raw_type_string: Optional[str] = None  # For TYPES: original type annotation
    confidence: Optional[float] = None  # For OVERRIDES: certainty of override detection

    @model_validator(mode="after")
    def compute_id(self) -> "ASTEdge":
        if not self.id:
            raw = f"{self.from_id}:{self.kind.value}:{self.to_id}:{self.dep_kind or ''}:{self.raw_type_string or ''}:{self.confidence or 0.0}"
            self.id = hashlib.sha256(raw.encode()).hexdigest()[:24]
        return self

    def to_neo4j_props(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "from_id": self.from_id,
            "to_id": self.to_id,
            "label": self.label or "",
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "dep_kind": self.dep_kind or "",
            "raw_type_string": self.raw_type_string or "",
            "confidence": self.confidence or 0.0,
        }


# ---------------------------------------------------------------------------
# Diff result model (used by graph_updater)
# ---------------------------------------------------------------------------

class DiffResult(BaseModel):
    """Result of diffing old vs. new AST extraction for a set of files."""
    added_nodes: list[ASTNode] = Field(default_factory=list)
    deleted_node_ids: list[str] = Field(default_factory=list)
    updated_nodes: list[ASTNode] = Field(default_factory=list)     # new versions
    old_updated_node_ids: list[str] = Field(default_factory=list)  # old ids to expire

    added_edges: list[ASTEdge] = Field(default_factory=list)
    deleted_edge_ids: list[str] = Field(default_factory=list)
    updated_edges: list[ASTEdge] = Field(default_factory=list)
    old_updated_edge_ids: list[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return (
            not self.added_nodes
            and not self.deleted_node_ids
            and not self.updated_nodes
            and not self.added_edges
            and not self.deleted_edge_ids
            and not self.updated_edges
        )


# ---------------------------------------------------------------------------
# Query result wrappers
# ---------------------------------------------------------------------------

class SubGraph(BaseModel):
    """Partial subgraph returned by neighbourhood expansion queries."""
    nodes: list[ASTNode] = Field(default_factory=list)
    edges: list[ASTEdge] = Field(default_factory=list)


class SearchResult(BaseModel):
    """Single semantic search result with distance score."""
    node: ASTNode
    score: float

    def to_standard_result(self, edge_type: Optional[str] = None) -> StandardResult:
        """Convert SearchResult to StandardResult."""
        return self.node.to_standard_result(score=self.score, edge_type=edge_type)


class StandardResult(BaseModel):
    """Unified output format for all MCP tools and API methods.
    
    Any tool returning code references should use this format
    so agents can process results uniformly.
    """
    id: str
    name: str
    qualified_name: str
    kind: str  # Class, Method, Function, Field, etc.
    lang: str  # java, cpp, rust, python, typescript
    file_path: str
    start_line: int
    end_line: int
    score: Optional[float] = None  # For search results (0.0-1.0)
    edge_type: Optional[str] = None  # CALLS, TYPES, OVERRIDES, etc.
    metadata: Optional[dict] = None  # Extra fields (confidence, raw_type_string, etc.)
    
    def to_markdown(self) -> str:
        """Render as Markdown for chat display."""
        return (
            f"**{self.kind} `{self.name}`**\n"
            f"- **Qualified:** `{self.qualified_name}`\n"
            f"- **Location:** `{self.file_path}:{self.start_line}-{self.end_line}`\n"
            f"- **Language:** `{self.lang}`\n"
            + (f"- **Score:** `{self.score:.2f}`\n" if self.score else "")
            + (f"- **Edge:** `{self.edge_type}`\n" if self.edge_type else "")
        )


# ---------------------------------------------------------------------------
# Configuration model
# ---------------------------------------------------------------------------

class Neo4jConfig(BaseModel):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"


class QdrantConfig(BaseModel):
    url: str = "http://localhost:6333"          # Qdrant server URL (Docker/remote)
    collection_name: str = "ast_rag_nodes"
    local_path: Optional[str] = None            # if set, use local file mode instead of URL


class EmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-m3"
    device: str = "cpu"  # set to "cuda" for GPU; ignored when remote_url is set
    # When set, AST-RAG calls this HTTP API for encoding instead of loading the model locally.
    # Set this to the URL of a running embedding_server.py instance on a GPU machine.
    # Example: "http://gpu-server:8765"
    remote_url: Optional[str] = None
    # Maximum number of texts to send per request to the remote embedding server.
    # Many OpenAI-compatible servers have limits on batch size.
    remote_batch_size: int = 32
    # Hybrid search configuration
    hybrid_search: bool = True  # Enable/disable hybrid search (requires Neo4j fulltext index)
    vector_weight: float = 0.7  # Weight for vector similarity scores (0.0-1.0)
    keyword_weight: float = 0.3  # Weight for keyword search scores (0.0-1.0)
    # Note: vector_weight + keyword_weight should ideally sum to 1.0 for normalized fusion


class ProjectConfig(BaseModel):
    """Top-level configuration for the AST-RAG system."""
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    # Extensions to scan; key = Language enum value, value = list of file extensions
    language_extensions: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "cpp": [".cpp", ".cxx", ".cc", ".c", ".hpp", ".hxx", ".hh", ".h"],
            "java": [".java"],
            "rust": [".rs"],
            "python": [".py"],
            "typescript": [".ts", ".tsx"],
        }
    )
    # Directories / patterns to exclude when walking
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            ".git", "__pycache__", "node_modules", "target", "build", "dist",
            ".gradle", ".idea", ".vscode", "venv", ".venv",
        ]
    )
