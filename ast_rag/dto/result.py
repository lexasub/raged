"""DTO - Query results and diff models.

Defines result wrappers for search operations and diff tracking.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ast_rag.dto.node import ASTNode, ASTEdge
from ast_rag.dto.block import ASTBlock


class DiffResult(BaseModel):
    """Result of diffing old vs. new AST extraction for a set of files.

    Attributes:
        added_nodes: New nodes that were added
        deleted_node_ids: IDs of nodes that were removed
        updated_nodes: New versions of updated nodes
        old_updated_node_ids: Old IDs of updated nodes to expire
        added_edges: New edges that were added
        deleted_edge_ids: IDs of edges that were removed
        updated_edges: New versions of updated edges
        old_updated_edge_ids: Old IDs of updated edges to expire
    """
    added_nodes: list[ASTNode] = Field(default_factory=list)
    deleted_node_ids: list[str] = Field(default_factory=list)
    updated_nodes: list[ASTNode] = Field(default_factory=list)
    old_updated_node_ids: list[str] = Field(default_factory=list)

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


class SubGraph(BaseModel):
    """Partial subgraph returned by neighbourhood expansion queries.

    Attributes:
        nodes: List of AST nodes in the subgraph
        edges: List of edges connecting the nodes
    """
    nodes: list[ASTNode] = Field(default_factory=list)
    edges: list[ASTEdge] = Field(default_factory=list)


class SearchResult(BaseModel):
    """Single semantic search result with distance score.

    Attributes:
        node: The matched AST node
        score: Relevance score (0.0-1.0)
    """
    node: ASTNode
    score: float

    def to_standard_result(self, edge_type: Optional[str] = None) -> StandardResult:
        """Convert SearchResult to StandardResult."""
        return self.node.to_standard_result(score=self.score, edge_type=edge_type)


class StandardResult(BaseModel):
    """Unified output format for all MCP tools and API methods.

    Any tool returning code references should use this format
    so agents can process results uniformly.

    Attributes:
        id: Node/edge ID
        name: Simple name
        qualified_name: Fully qualified name
        kind: Node kind (Class, Method, Function, etc.)
        lang: Source language
        file_path: Path to source file
        start_line: Start line number
        end_line: End line number
        score: Relevance score for search results
        edge_type: Edge type for reference results
        metadata: Extra fields (confidence, raw_type_string, etc.)
    """
    id: str
    name: str
    qualified_name: str
    kind: str
    lang: str
    file_path: str
    start_line: int
    end_line: int
    score: Optional[float] = None
    edge_type: Optional[str] = None
    metadata: Optional[dict] = None

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
