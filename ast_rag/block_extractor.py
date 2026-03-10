"""
block_extractor.py - Extract code blocks (if/for/while/try/lambda/with) from AST.

Responsibilities:
- Extract nested blocks from within functions/methods
- Track nesting depth and parent function relationships
- Capture variables for lambda/closure expressions
- Support multiple languages (Python, Rust, with extensibility for others)

Usage::

    extractor = BlockExtractor()
    blocks = extractor.extract_blocks(tree, source, function_nodes, lang="python")
"""

from __future__ import annotations

import logging
from typing import Optional
from collections import defaultdict

from tree_sitter import Tree, QueryCursor, Node

from ast_rag.models import ASTBlock, BlockType, Language, ASTNode
from ast_rag.language_queries import LANGUAGE_QUERIES

logger = logging.getLogger(__name__)

# Mapping from query name to BlockType for each language
PYTHON_BLOCK_QUERIES: dict[str, BlockType] = {
    "if_blocks": BlockType.IF,
    "for_blocks": BlockType.FOR,
    "while_blocks": BlockType.WHILE,
    "try_blocks": BlockType.TRY,
    "with_blocks": BlockType.WITH,
    "lambda_expr": BlockType.LAMBDA,
    "match_blocks": BlockType.MATCH,
}

RUST_BLOCK_QUERIES: dict[str, BlockType] = {
    "if_blocks": BlockType.IF,
    "for_blocks": BlockType.FOR,
    "while_blocks": BlockType.WHILE,
    "loop_blocks": BlockType.LOOP,
    "match_blocks": BlockType.MATCH,
    "try_blocks": BlockType.TRY,
    "closure_expr": BlockType.LAMBDA,  # Rust closures mapped to LAMBDA
}

# Language-specific block query mappings
LANGUAGE_BLOCK_QUERIES: dict[str, dict[str, BlockType]] = {
    "python": PYTHON_BLOCK_QUERIES,
    "rust": RUST_BLOCK_QUERIES,
}


class BlockExtractor:
    """Extracts code blocks from parsed AST trees.

    Blocks are nested control flow structures within functions:
    - Control flow: if, for, while, match/switch
    - Exception handling: try/catch/finally
    - Scope blocks: with (Python), using (C#)
    - Anonymous functions: lambda, closure

    Each block is associated with a parent function and tracks:
    - Source location (lines, bytes)
    - Nesting depth within the function
    - Captured variables (for lambdas/closures)
    """

    def __init__(self) -> None:
        pass

    def extract_blocks(
        self,
        tree: Tree,
        source: bytes,
        function_nodes: list[ASTNode],
        lang: str,
        commit_hash: str = "INIT",
    ) -> list[ASTBlock]:
        """Extract all blocks from within the given function nodes.

        Args:
            tree: Parsed tree-sitter Tree
            source: Source code bytes
            function_nodes: List of function/method ASTNodes to extract blocks from
            lang: Language key (python, rust, etc.)
            commit_hash: Version hash for MVCC tracking

        Returns:
            List of ASTBlock objects with fully populated fields
        """
        block_query_map = LANGUAGE_BLOCK_QUERIES.get(lang, {})
        if not block_query_map:
            logger.debug("No block queries configured for language: %s", lang)
            return []

        # Get compiled queries for this language
        lang_queries = LANGUAGE_QUERIES.get(lang, {})
        compiled_queries = {}
        for qname in block_query_map.keys():
            qstr = lang_queries.get(qname)
            if qstr:
                try:
                    from tree_sitter import Language, Query
                    # We need the language object - extract from context
                    # For now, we'll compile queries on-demand
                    compiled_queries[qname] = qstr
                except Exception as exc:
                    logger.warning("Failed to compile block query '%s': %s", qname, exc)

        all_blocks: list[ASTBlock] = []

        for func_node in function_nodes:
            # Extract blocks for this function
            func_blocks = self._extract_blocks_for_function(
                tree,
                source,
                func_node,
                lang,
                block_query_map,
                compiled_queries,
                commit_hash,
            )
            all_blocks.extend(func_blocks)

        return all_blocks

    def _extract_blocks_for_function(
        self,
        tree: Tree,
        source: bytes,
        func_node: ASTNode,
        lang: str,
        block_query_map: dict[str, BlockType],
        compiled_queries: dict[str, str],
        commit_hash: str,
    ) -> list[ASTBlock]:
        """Extract blocks from a single function."""
        blocks: list[ASTBlock] = []

        # For each block type query, find matches within the function's byte range
        for qname, block_type in block_query_map.items():
            qstr = compiled_queries.get(qname)
            if not qstr:
                continue

            try:
                from tree_sitter import Language, Query, Parser
                # Get the language
                lang_map = {
                    "python": "python",
                    "rust": "rust",
                }
                # We need to get the actual Language object
                # This is a bit tricky - we'll use a different approach
                # Just iterate through the tree manually for now
                pass
            except Exception:
                continue

        # Alternative approach: manually traverse the tree within function bounds
        blocks = self._manual_extract_blocks(
            tree.root_node,
            source,
            func_node,
            lang,
            block_query_map,
            commit_hash,
        )

        return blocks

    def _manual_extract_blocks(
        self,
        root_node: Node,
        source: bytes,
        func_node: ASTNode,
        lang: str,
        block_query_map: dict[str, BlockType],
        commit_hash: str,
    ) -> list[ASTBlock]:
        """Manually extract blocks by traversing the tree within function bounds."""
        blocks: list[ASTBlock] = []

        # Define node type mappings for each language
        if lang == "python":
            type_map = {
                "if_statement": BlockType.IF,
                "for_statement": BlockType.FOR,
                "while_statement": BlockType.WHILE,
                "try_statement": BlockType.TRY,
                "with_statement": BlockType.WITH,
                "lambda": BlockType.LAMBDA,
                "match_statement": BlockType.MATCH,
            }
        elif lang == "rust":
            type_map = {
                "if_expression": BlockType.IF,
                "for_expression": BlockType.FOR,
                "while_expression": BlockType.WHILE,
                "loop_expression": BlockType.LOOP,
                "match_expression": BlockType.MATCH,
                "try_expression": BlockType.TRY,
                "closure_expression": BlockType.LAMBDA,
            }
        else:
            return blocks

        # Recursively traverse the tree
        def traverse(node: Node, depth: int = 1) -> None:
            # Check if this node is within the function's range
            if (node.start_byte < func_node.start_byte or
                node.end_byte > func_node.end_byte):
                return

            # Check if this node type matches a block type
            block_type = type_map.get(node.type)
            if block_type:
                block = self._create_block(
                    node=node,
                    source=source,
                    func_node=func_node,
                    lang=lang,
                    block_type=block_type,
                    nesting_depth=depth,
                    commit_hash=commit_hash,
                )
                if block:
                    blocks.append(block)

            # Recurse into children
            for child in node.children:
                traverse(child, depth + 1)

        traverse(root_node)
        return blocks

    def _create_block(
        self,
        node: Node,
        source: bytes,
        func_node: ASTNode,
        lang: str,
        block_type: BlockType,
        nesting_depth: int,
        commit_hash: str,
    ) -> Optional[ASTBlock]:
        """Create an ASTBlock from a tree-sitter node."""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        start_byte = node.start_byte
        end_byte = node.end_byte

        # Extract source text
        source_text = source[start_byte:end_byte].decode("utf-8", errors="replace")

        # Extract captured variables for lambdas/closures
        captured_vars: list[str] = []
        if block_type == BlockType.LAMBDA:
            captured_vars = self._extract_captured_variables(node, source, lang)

        # Extract name for lambdas (parameter names)
        name = ""
        if block_type == BlockType.LAMBDA:
            name = self._extract_lambda_name(node, source, lang)

        block = ASTBlock(
            block_type=block_type,
            name=name,
            parent_function_id=func_node.id,
            lang=Language(lang),
            file_path=func_node.file_path,
            start_line=start_line,
            end_line=end_line,
            start_byte=start_byte,
            end_byte=end_byte,
            nesting_depth=nesting_depth,
            source_text=source_text,
            captured_variables=captured_vars,
            valid_from=commit_hash,
        )

        return block

    def _extract_captured_variables(
        self,
        node: Node,
        source: bytes,
        lang: str,
    ) -> list[str]:
        """Extract variables captured from outer scope by a lambda/closure.

        For Python lambdas, this includes variables used in the body that
        are not parameters.

        For Rust closures, this includes variables from the environment.
        """
        captured: set[str] = set()

        if lang == "python":
            # Get lambda parameters
            params = set()
            params_node = node.child_by_field_name("parameters")
            if params_node:
                for child in params_node.children:
                    if child.type == "identifier":
                        params.add(_node_text(child, source))

            # Find identifiers in body that are not params
            body_node = node.child_by_field_name("body")
            if body_node:
                captured = self._find_free_variables(body_node, source, params)

        elif lang == "rust":
            # For Rust, get closure parameters
            params = set()
            params_node = node.child_by_field_name("parameters")
            if params_node:
                for child in params_node.children:
                    if child.type == "identifier":
                        params.add(_node_text(child, source))

            # Find identifiers in body
            body_node = node.child_by_field_name("body")
            if body_node:
                captured = self._find_free_variables(body_node, source, params)

        return list(captured)

    def _find_free_variables(
        self,
        node: Node,
        source: bytes,
        excluded: set[str],
    ) -> set[str]:
        """Find identifier references that are not in the excluded set."""
        free_vars: set[str] = set()

        def traverse(n: Node) -> None:
            if n.type == "identifier":
                name = _node_text(n, source)
                if name and name not in excluded:
                    # Check if this is a definition (assignment target)
                    if n.parent and n.parent.type in ("assignment", "typed_parameter"):
                        return
                    free_vars.add(name)
            for child in n.children:
                traverse(child)

        traverse(node)
        return free_vars

    def _extract_lambda_name(
        self,
        node: Node,
        source: bytes,
        lang: str,
    ) -> str:
        """Extract a descriptive name for a lambda expression."""
        if lang == "python":
            params_node = node.child_by_field_name("parameters")
            if params_node:
                param_texts = []
                for child in params_node.children:
                    if child.type == "identifier":
                        param_texts.append(_node_text(child, source))
                return f"lambda({', '.join(param_texts)})"
            return "lambda()"

        elif lang == "rust":
            params_node = node.child_by_field_name("parameters")
            if params_node:
                param_texts = []
                for child in params_node.children:
                    if child.type == "identifier":
                        param_texts.append(_node_text(child, source))
                return f"|{', '.join(param_texts)}|"
            return "||"

        return ""


def _node_text(node: Optional[Node], source: bytes) -> str:
    """Extract text from a tree-sitter node."""
    if node is None:
        return ""
    try:
        return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
    except Exception:
        return ""


def extract_blocks_from_file(
    tree: Tree,
    source: bytes,
    function_nodes: list[ASTNode],
    lang: str,
    commit_hash: str = "INIT",
) -> list[ASTBlock]:
    """Convenience function to extract blocks from a file.

    Args:
        tree: Parsed tree-sitter Tree
        source: Source code bytes
        function_nodes: List of function/method nodes
        lang: Language key
        commit_hash: Version hash

    Returns:
        List of extracted ASTBlock objects
    """
    extractor = BlockExtractor()
    return extractor.extract_blocks(tree, source, function_nodes, lang, commit_hash)
