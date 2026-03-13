"""DTO - AST Node and Edge models.

Core data models representing AST entities and relationships.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from ast_rag.dto.enums import NodeKind, EdgeKind, Language


class ASTNode(BaseModel):
    """Represents a single extracted AST entity.

    The `id` field is a stable, content-addressed identifier derived from
    file_path, kind, qualified_name, and project_id. It survives re-parses as long as
    the entity has not moved.

    Attributes:
        id: Stable SHA-256 based identifier
        kind: Type of AST node (Class, Function, Method, etc.)
        name: Simple name of the entity
        qualified_name: Fully qualified name including package/namespace
        lang: Source language
        file_path: Path to source file
        start_line: Start line number (1-indexed)
        end_line: End line number
        start_byte: Start byte offset in file
        end_byte: End byte offset in file
        code_hash: SHA-256 of the raw source text
        signature: Function/method signature if applicable
        project_id: Project identifier for data isolation
        valid_from: Version when this node was added
        valid_to: Version when this node was deleted (None = current)
        source_text: Raw source text (not persisted to graph)
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
    signature: Optional[str] = None
    project_id: str = "default"
    valid_from: str = "INIT"
    valid_to: Optional[str] = None
    source_text: Optional[str] = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def compute_derived_fields(self) -> "ASTNode":
        """Auto-compute id and code_hash if not provided."""
        if not self.id:
            # Include project_id in the hash for proper isolation
            raw = f"{self.project_id}:{self.file_path}:{self.kind.value}:{self.qualified_name}"
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
            "project_id": self.project_id,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
        }

    def to_standard_result(self, score: Optional[float] = None, edge_type: Optional[str] = None) -> "StandardResult":
        """Convert ASTNode to StandardResult."""
        from ast_rag.dto.result import StandardResult
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

    Attributes:
        id: Stable edge identifier
        kind: Type of relationship
        from_id: Source node ID
        to_id: Target node ID
        label: Optional label for extra context
        valid_from: Version when this edge was added
        valid_to: Version when this edge was deleted
        dep_kind: Dependency kind for DEPENDS_ON edges
        raw_type_string: Original type annotation for TYPES edges
        confidence: Certainty score for OVERRIDES edges
    """
    id: str = Field(default="", description="Stable edge identifier")
    kind: EdgeKind
    from_id: str
    to_id: str
    label: Optional[str] = None
    valid_from: str = "INIT"
    valid_to: Optional[str] = None
    dep_kind: Optional[str] = None
    raw_type_string: Optional[str] = None
    confidence: Optional[float] = None

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
