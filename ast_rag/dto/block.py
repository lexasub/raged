"""DTO - AST Block model.

Represents code blocks extracted from within functions.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from ast_rag.dto.enums import BlockType, Language


class ASTBlock(BaseModel):
    """Represents a code block extracted from within a function or lambda.

    Blocks are nested code structures that have their own scope and control flow.
    Examples: if/else branches, loop bodies, try/catch blocks, lambda expressions,
    with statements, match arms, etc.

    Attributes:
        id: Stable SHA-256 based identifier
        block_type: Type of block (if, for, while, try, lambda, etc.)
        name: Optional name (e.g., lambda argument names)
        parent_function_id: ID of the containing function/method
        lang: Source language
        file_path: Path to source file
        start_line: Start line number
        end_line: End line number
        start_byte: Start byte offset
        end_byte: End byte offset
        nesting_depth: Nesting depth within the parent function
        source_text: Raw source code of the block
        code_hash: SHA-256 of the raw source text
        captured_variables: Variables captured from outer scope (for lambdas)
        valid_from: Version when this block was added
        valid_to: Version when this block was deleted
    """
    id: str = Field(default="", description="Stable SHA-256 based identifier")
    block_type: BlockType
    name: str = Field(default="", description="Optional name")
    parent_function_id: str = Field(..., description="ID of the containing function/method")
    lang: Language
    file_path: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    nesting_depth: int = Field(default=1, description="Nesting depth within the parent function")
    source_text: Optional[str] = Field(default=None, exclude=True)
    code_hash: str = Field(default="", description="SHA-256 of the raw source text")
    captured_variables: list[str] = Field(default_factory=list)
    valid_from: str = "INIT"
    valid_to: Optional[str] = None

    @model_validator(mode="after")
    def compute_derived_fields(self) -> "ASTBlock":
        """Auto-compute id and code_hash if not provided."""
        if not self.id:
            raw = f"{self.file_path}:{self.block_type.value}:{self.parent_function_id}:{self.start_line}"
            self.id = hashlib.sha256(raw.encode()).hexdigest()[:24]
        if not self.code_hash and self.source_text:
            self.code_hash = hashlib.sha256(self.source_text.encode()).hexdigest()[:24]
        return self

    def to_neo4j_props(self) -> dict[str, dict]:
        """Serialize block to a flat dict suitable for Neo4j property map."""
        return {
            "id": self.id,
            "block_type": self.block_type.value,
            "name": self.name,
            "parent_function_id": self.parent_function_id,
            "lang": self.lang.value,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "nesting_depth": self.nesting_depth,
            "code_hash": self.code_hash,
            "captured_variables": self.captured_variables,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
        }

    def to_standard_result(self, score: Optional[float] = None) -> "StandardResult":
        """Convert ASTBlock to StandardResult."""
        from ast_rag.dto.result import StandardResult
        return StandardResult(
            id=self.id,
            name=self.name or f"{self.block_type.value}_block",
            qualified_name=f"{self.parent_function_id}.{self.block_type.value}@{self.start_line}",
            kind="Block",
            lang=self.lang.value,
            file_path=self.file_path,
            start_line=self.start_line,
            end_line=self.end_line,
            score=score,
            metadata={
                "block_type": self.block_type.value,
                "nesting_depth": self.nesting_depth,
                "captured_variables": self.captured_variables,
            },
        )
