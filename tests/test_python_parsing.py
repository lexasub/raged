"""
test_python_parsing.py - Tests for Python language parsing (issue #6).

Covers the four areas from the issue: class definitions, function
definitions, import statements, and method calls. Python extraction is
currently skeletal (see module docstring of parser_manager.py); tests
pin the behavior that exists today, and known gaps are marked xfail so
they turn into loud signals once implemented.

Structure
---------
TestPythonClassParsing    - classes, inheritance, containment edges
TestPythonFunctionParsing - functions, signatures, async/decorated
TestPythonImportParsing   - import statement variants
TestPythonCallEdges       - method calls (known gap, xfail)
TestPythonNodeIdentity    - stable IDs across re-parses
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ast_rag.models import ASTEdge, ASTNode, EdgeKind, NodeKind
from ast_rag.services.parsing.parser_manager import ParserManager


@pytest.fixture(scope="module")
def pm() -> ParserManager:
    return ParserManager()


def _parse(pm: ParserManager, tmp_path: Path, src: str, name: str = "sample.py"):
    """Parse Python source and return (nodes, edges)."""
    path = tmp_path / name
    path.write_text(src, encoding="utf-8")
    tree = pm.parse_file(str(path), resolve=True)
    assert tree is not None
    nodes = pm.extract_nodes(tree, str(path), "python", src.encode())
    edges = pm.extract_edges(tree, nodes, str(path), "python", src.encode())
    return nodes, edges


def _by_name(nodes: list[ASTNode], name: str, kind: NodeKind) -> ASTNode:
    matches = [n for n in nodes if n.name == name and n.kind == kind]
    assert matches, f"no {kind.value} node named {name!r}"
    return matches[0]


def _edges_of(edges: list[ASTEdge], kind: EdgeKind) -> list[ASTEdge]:
    return [e for e in edges if e.kind == kind]


CLASS_SRC = """
class Animal:
    species: str = "generic"

    def __init__(self, name: str) -> None:
        self.name = name

    def describe(self) -> str:
        return self.name

class Dog(Animal):
    def bark(self) -> str:
        return "woof"
"""


# ===========================================================================
# TestPythonClassParsing
# ===========================================================================


class TestPythonClassParsing:
    def test_class_definitions_extracted(self, pm: ParserManager, tmp_path: Path) -> None:
        nodes, _ = _parse(pm, tmp_path, CLASS_SRC)
        class_names = {n.name for n in nodes if n.kind == NodeKind.CLASS}
        assert class_names == {"Animal", "Dog"}

    def test_qualified_name_is_module_prefixed(self, pm: ParserManager, tmp_path: Path) -> None:
        nodes, _ = _parse(pm, tmp_path, CLASS_SRC, name="zoo.py")
        animal = _by_name(nodes, "Animal", NodeKind.CLASS)
        assert animal.qualified_name == "zoo.Animal"
        assert animal.lang == "python"

    def test_class_line_span_covers_body(self, pm: ParserManager, tmp_path: Path) -> None:
        nodes, _ = _parse(pm, tmp_path, CLASS_SRC)
        animal = _by_name(nodes, "Animal", NodeKind.CLASS)
        lines = CLASS_SRC.splitlines()
        assert lines[animal.start_line - 1].startswith("class Animal")
        assert animal.end_line > animal.start_line

    def test_inheritance_edge(self, pm: ParserManager, tmp_path: Path) -> None:
        nodes, edges = _parse(pm, tmp_path, CLASS_SRC)
        animal = _by_name(nodes, "Animal", NodeKind.CLASS)
        dog = _by_name(nodes, "Dog", NodeKind.CLASS)
        inherits = _edges_of(edges, EdgeKind.INHERITS)
        assert len(inherits) == 1
        assert inherits[0].from_id == dog.id
        assert inherits[0].to_id == animal.id
        assert inherits[0].label == "Animal"

    def test_methods_linked_to_class(self, pm: ParserManager, tmp_path: Path) -> None:
        nodes, edges = _parse(pm, tmp_path, CLASS_SRC)
        animal = _by_name(nodes, "Animal", NodeKind.CLASS)
        init = _by_name(nodes, "__init__", NodeKind.FUNCTION)
        contains = _edges_of(edges, EdgeKind.CONTAINS_METHOD)
        assert any(e.from_id == animal.id and e.to_id == init.id for e in contains)

    def test_nested_class_extracted(self, pm: ParserManager, tmp_path: Path) -> None:
        src = "class Outer:\n    class Inner:\n        pass\n"
        nodes, _ = _parse(pm, tmp_path, src)
        class_names = {n.name for n in nodes if n.kind == NodeKind.CLASS}
        assert class_names == {"Outer", "Inner"}

    def test_same_name_methods_collapse_to_one_node(
        self, pm: ParserManager, tmp_path: Path
    ) -> None:
        # Skeletal quirk, pinned on purpose: Python qualified names do not
        # include the enclosing class, so Animal.speak and Dog.speak hash to
        # the same node id and both classes contain the same method node.
        src = (
            "class Animal:\n"
            "    def speak(self):\n"
            "        return 'generic'\n"
            "class Dog(Animal):\n"
            "    def speak(self):\n"
            "        return 'woof'\n"
        )
        nodes, edges = _parse(pm, tmp_path, src)
        speak_ids = {n.id for n in nodes if n.name == "speak"}
        assert len(speak_ids) == 1
        containing = {
            e.from_id for e in _edges_of(edges, EdgeKind.CONTAINS_METHOD) if e.to_id in speak_ids
        }
        assert len(containing) == 2


# ===========================================================================
# TestPythonFunctionParsing
# ===========================================================================


class TestPythonFunctionParsing:
    def test_function_definition_extracted(self, pm: ParserManager, tmp_path: Path) -> None:
        src = "def top_level(x: int, y: int = 2) -> int:\n    return x + y\n"
        nodes, _ = _parse(pm, tmp_path, src)
        fn = _by_name(nodes, "top_level", NodeKind.FUNCTION)
        assert fn.qualified_name.endswith(".top_level")

    def test_signature_keeps_hints_and_defaults(self, pm: ParserManager, tmp_path: Path) -> None:
        src = "def top_level(x: int, y: int = 2) -> int:\n    return x + y\n"
        nodes, _ = _parse(pm, tmp_path, src)
        fn = _by_name(nodes, "top_level", NodeKind.FUNCTION)
        assert fn.signature == "top_level(x: int, y: int = 2)"

    def test_star_args_signature(self, pm: ParserManager, tmp_path: Path) -> None:
        src = "def multi(a, *args, **kwargs):\n    return len(args)\n"
        nodes, _ = _parse(pm, tmp_path, src)
        fn = _by_name(nodes, "multi", NodeKind.FUNCTION)
        assert fn.signature == "multi(a, *args, **kwargs)"

    def test_async_function_extracted(self, pm: ParserManager, tmp_path: Path) -> None:
        src = "async def fetch(url: str) -> bytes:\n    return b''\n"
        nodes, _ = _parse(pm, tmp_path, src)
        fn = _by_name(nodes, "fetch", NodeKind.FUNCTION)
        assert fn.signature == "fetch(url: str)"

    def test_decorated_function_extracted(self, pm: ParserManager, tmp_path: Path) -> None:
        src = "@staticmethod\ndef decorated() -> None:\n    pass\n"
        nodes, _ = _parse(pm, tmp_path, src)
        assert _by_name(nodes, "decorated", NodeKind.FUNCTION)

    def test_top_level_functions_contained_by_file(self, pm: ParserManager, tmp_path: Path) -> None:
        src = "def a():\n    pass\ndef b():\n    pass\n"
        nodes, edges = _parse(pm, tmp_path, src)
        fn_ids = {n.id for n in nodes if n.kind == NodeKind.FUNCTION}
        contained = {e.to_id for e in _edges_of(edges, EdgeKind.CONTAINS_FUNCTION)}
        assert fn_ids <= contained


# ===========================================================================
# TestPythonImportParsing
# ===========================================================================


class TestPythonImportParsing:
    def test_plain_import(self, pm: ParserManager, tmp_path: Path) -> None:
        nodes, edges = _parse(pm, tmp_path, "import os\n")
        imports = _edges_of(edges, EdgeKind.IMPORTS)
        assert [e.label for e in imports] == ["os"]

    def test_from_import_labels_module(self, pm: ParserManager, tmp_path: Path) -> None:
        _, edges = _parse(pm, tmp_path, "from pathlib import Path\n")
        imports = _edges_of(edges, EdgeKind.IMPORTS)
        assert [e.label for e in imports] == ["pathlib"]

    def test_from_import_multiple_names(self, pm: ParserManager, tmp_path: Path) -> None:
        _, edges = _parse(pm, tmp_path, "from collections import OrderedDict, defaultdict\n")
        imports = _edges_of(edges, EdgeKind.IMPORTS)
        assert [e.label for e in imports] == ["collections", "collections"]

    def test_relative_import(self, pm: ParserManager, tmp_path: Path) -> None:
        _, edges = _parse(pm, tmp_path, "from . import sibling\n")
        imports = _edges_of(edges, EdgeKind.IMPORTS)
        assert [e.label for e in imports] == ["."]

    @pytest.mark.xfail(
        reason="known gap: aliased imports (import json as j) produce no IMPORTS edge",
        strict=True,
    )
    def test_aliased_import(self, pm: ParserManager, tmp_path: Path) -> None:
        _, edges = _parse(pm, tmp_path, "import json as j\n")
        assert _edges_of(edges, EdgeKind.IMPORTS)


# ===========================================================================
# TestPythonCallEdges
# ===========================================================================


CALL_SRC = """
class Dog:
    def bark(self) -> str:
        return "woof"

    def speak(self) -> str:
        return self.bark()

def make_noise() -> str:
    d = Dog()
    return d.bark()
"""


class TestPythonCallEdges:
    @pytest.mark.xfail(
        reason="known gap: Python extraction is skeletal — method/function call "
        "edges (CALLS) are not extracted yet",
        strict=True,
    )
    def test_method_call_edges_extracted(self, pm: ParserManager, tmp_path: Path) -> None:
        _, edges = _parse(pm, tmp_path, CALL_SRC)
        call_kinds = {
            EdgeKind.CALLS,
            EdgeKind.VIRTUAL_CALL,
            EdgeKind.LAMBDA_CALL,
            EdgeKind.CROSS_FILE_CALL,
        }
        assert any(e.kind in call_kinds for e in edges)

    def test_structure_around_calls_still_extracted(
        self, pm: ParserManager, tmp_path: Path
    ) -> None:
        # Even without CALLS edges, the surrounding structure must be intact.
        nodes, edges = _parse(pm, tmp_path, CALL_SRC)
        assert _by_name(nodes, "Dog", NodeKind.CLASS)
        assert _by_name(nodes, "bark", NodeKind.FUNCTION)
        assert _by_name(nodes, "make_noise", NodeKind.FUNCTION)
        assert _edges_of(edges, EdgeKind.CONTAINS_METHOD)


# ===========================================================================
# TestPythonNodeIdentity
# ===========================================================================


class TestPythonNodeIdentity:
    def test_node_ids_stable_across_reparse(self, pm: ParserManager, tmp_path: Path) -> None:
        # Stable IDs are what make incremental git-diff updates possible.
        nodes_a, _ = _parse(pm, tmp_path, CLASS_SRC)
        pm.clear_tree_cache()
        nodes_b, _ = _parse(pm, tmp_path, CLASS_SRC)
        assert {n.id for n in nodes_a} == {n.id for n in nodes_b}
