"""
test_tsx_jsx_support.py - Tests for TSX/JSX language detection and parsing.

Verifies:
- .tsx and .jsx files are mapped to a dedicated "tsx" language (not plain
  "typescript"), since tree-sitter's plain TypeScript grammar cannot parse
  JSX syntax.
- Files containing JSX parse without syntax errors and extract expected
  nodes (e.g. function/class declarations).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ast_rag.services.parsing.parser_manager import ParserManager, EXT_TO_LANG


def _tmp(suffix: str, content: bytes) -> str:
    fh = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="wb")
    fh.write(content)
    fh.close()
    return fh.name


TSX_SOURCE = b"""
import React from "react";

interface GreetingProps {
  name: string;
}

function Greeting({ name }: GreetingProps) {
  return <div className="greeting">Hello, {name}!</div>;
}

export default Greeting;
"""

JSX_SOURCE = b"""
function Greeting({ name }) {
  return <div className="greeting">Hello, {name}!</div>;
}

export default Greeting;
"""


class TestExtensionMapping:
    def test_tsx_maps_to_dedicated_language(self):
        assert EXT_TO_LANG[".tsx"] == "tsx"

    def test_jsx_maps_to_dedicated_language(self):
        assert EXT_TO_LANG[".jsx"] == "tsx"

    def test_ts_still_maps_to_plain_typescript(self):
        assert EXT_TO_LANG[".ts"] == "typescript"


class TestTsxJsxParsing:
    def test_tsx_file_parses_without_syntax_errors(self):
        path = _tmp(".tsx", TSX_SOURCE)
        try:
            pm = ParserManager()
            assert pm.detect_language(path) == "tsx"
            tree = pm.parse_file(path)
            assert not tree.root_node.has_error
        finally:
            os.unlink(path)

    def test_jsx_file_parses_without_syntax_errors(self):
        path = _tmp(".jsx", JSX_SOURCE)
        try:
            pm = ParserManager()
            assert pm.detect_language(path) == "tsx"
            tree = pm.parse_file(path)
            assert not tree.root_node.has_error
        finally:
            os.unlink(path)

    def test_tsx_extract_nodes_finds_function_and_interface(self):
        path = _tmp(".tsx", TSX_SOURCE)
        try:
            pm = ParserManager()
            tree = pm.parse_file(path)
            nodes = pm.extract_nodes(tree, path, "tsx", source=TSX_SOURCE)
            names = {n.name for n in nodes}
            assert "Greeting" in names
            assert "GreetingProps" in names
        finally:
            os.unlink(path)

    def test_plain_typescript_grammar_cannot_parse_jsx(self):
        """Regression guard: confirms *why* a dedicated tsx grammar is needed.

        If this ever starts passing, the plain "typescript" tree-sitter
        grammar has gained JSX support and the special-casing above may be
        revisited.
        """
        import tree_sitter as ts
        import tree_sitter_typescript as tsts

        parser = ts.Parser(ts.Language(tsts.language_typescript()))
        tree = parser.parse(JSX_SOURCE)
        assert tree.root_node.has_error
