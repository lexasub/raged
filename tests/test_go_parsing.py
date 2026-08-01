"""Basic Go extraction (issue #17).

Go models types as `type_declaration -> type_spec`, with the concrete shape on
the `type` field, so structs and interfaces are matched on the type_spec rather
than on a dedicated node. Methods carry a `receiver` and are a distinct node
type from plain functions.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tree_sitter import Query, QueryCursor

from ast_rag.models import NodeKind
from ast_rag.services.parsing import LANGUAGE_QUERIES
from ast_rag.services.parsing.go import GO_QUERIES
from ast_rag.services.parsing.parser_manager import EXT_TO_LANG, ParserManager

GO_SRC = b"""
package main

import "fmt"

type Shape interface {
	Area() float64
}

type Rect struct {
	W float64
	H float64
}

func (r Rect) Area() float64 {
	return r.W * r.H
}

func describe(s Shape) string {
	return fmt.Sprintf("area=%v", s.Area())
}

func main() {
	fmt.Println(describe(Rect{W: 2, H: 3}))
}
"""


@pytest.fixture(scope="module")
def pm() -> ParserManager:
    return ParserManager()


@pytest.fixture()
def parsed(pm: ParserManager, tmp_path: Path):
    path = tmp_path / "main.go"
    path.write_bytes(GO_SRC)
    tree = pm.parse_file(str(path), source=GO_SRC)
    assert tree is not None, "Go source failed to parse"
    nodes = pm.extract_nodes(tree, str(path), "go")
    edges = pm.extract_edges(tree, nodes, str(path), "go", source=GO_SRC)
    return nodes, edges


def _named(nodes, kind: NodeKind):
    return {n.name for n in nodes if n.kind == kind}


def test_go_is_registered():
    assert EXT_TO_LANG[".go"] == "go"
    assert "go" in LANGUAGE_QUERIES


def test_language_detected_from_extension(pm: ParserManager, tmp_path: Path):
    path = tmp_path / "x.go"
    path.write_bytes(b"package main\n")
    assert pm.detect_language(str(path)) == "go"


def test_structs_and_interfaces_extracted(parsed):
    nodes, _ = parsed
    assert "Rect" in _named(nodes, NodeKind.STRUCT)
    assert "Shape" in _named(nodes, NodeKind.INTERFACE)


def test_functions_and_methods_distinguished(parsed):
    nodes, _ = parsed
    functions = _named(nodes, NodeKind.FUNCTION)
    methods = _named(nodes, NodeKind.METHOD)
    assert {"describe", "main"} <= functions
    # Area has a receiver, so it is a method rather than a function
    assert "Area" in methods
    assert "Area" not in functions


def test_struct_fields_extracted(parsed):
    nodes, _ = parsed
    assert {"W", "H"} <= _named(nodes, NodeKind.FIELD)


def test_imports_extracted(parsed):
    _, edges = parsed
    kinds = {str(e.kind) for e in edges}
    assert any("IMPORTS" in k for k in kinds)


def test_calls_query_captures_bare_and_selector_calls():
    """Both `helper()` and `pkg.Helper()` must yield a callee_name.

    Asserted at the query level: turning these matches into CALLS edges also
    requires the _extract_call_edges fix, which is a separate change.
    """
    import tree_sitter as ts
    import tree_sitter_go as tsgo

    lang = ts.Language(tsgo.language())
    tree = ts.Parser(lang).parse(GO_SRC)
    matches = list(QueryCursor(Query(lang, GO_QUERIES["calls"])).matches(tree.root_node))

    names = set()
    for _, md in matches:
        cap = md.get("callee_name")
        if cap is None:
            continue
        node = cap[0] if isinstance(cap, list) else cap
        names.add(node.text.decode())

    assert "describe" in names, "bare identifier call not captured"
    assert "Println" in names, "selector call not captured"
    assert "Area" in names


@pytest.mark.parametrize("query_name", sorted(GO_QUERIES))
def test_every_go_query_compiles(query_name: str):
    import tree_sitter as ts
    import tree_sitter_go as tsgo

    Query(ts.Language(tsgo.language()), GO_QUERIES[query_name])
