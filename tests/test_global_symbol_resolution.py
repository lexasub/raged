"""Cross-file reference resolution via a project-wide symbol table.

Edge extraction resolves references against a name -> node id map. Built from a
single file's nodes, any reference to a symbol defined elsewhere is dropped, so
only same-file calls ever link. `extract_edges(global_symbols=...)` supplies a
project-wide map so cross-file references resolve too.

Local definitions must still win, otherwise a file-local helper would be
shadowed by an unrelated same-named symbol from another file.
"""

from __future__ import annotations

import pytest

from ast_rag.dto.enums import EdgeKind
from ast_rag.services.parsing.parser_manager import ParserManager

CALLEE_SRC = b"""
def shared_helper(x):
    return x + 1
"""

CALLER_SRC = b"""
def caller(y):
    return shared_helper(y)
"""

SHADOWING_SRC = b"""
def shared_helper(x):
    return x * 2


def local_caller(y):
    return shared_helper(y)
"""


@pytest.fixture(scope="module")
def pm() -> ParserManager:
    return ParserManager()


def _parse(pm: ParserManager, path, src: bytes):
    path.write_bytes(src)
    tree = pm.parse_file(str(path), source=src)
    assert tree is not None
    return tree, pm.extract_nodes(tree, str(path), "python")


def _calls(edges):
    return [e for e in edges if e.kind == EdgeKind.CALLS]


def test_cross_file_call_is_unresolved_without_a_global_table(pm, tmp_path):
    """Baseline: this is the behaviour the two-phase index exists to fix."""
    _, callee_nodes = _parse(pm, tmp_path / "callee.py", CALLEE_SRC)
    tree, caller_nodes = _parse(pm, tmp_path / "caller.py", CALLER_SRC)

    edges = pm.extract_edges(
        tree, caller_nodes, str(tmp_path / "caller.py"), "python", source=CALLER_SRC
    )
    assert _calls(edges) == [], "expected no cross-file CALLS without a project-wide map"
    assert callee_nodes, "fixture lost"


def test_cross_file_call_resolves_with_a_global_table(pm, tmp_path):
    _, callee_nodes = _parse(pm, tmp_path / "callee.py", CALLEE_SRC)
    tree, caller_nodes = _parse(pm, tmp_path / "caller.py", CALLER_SRC)

    global_symbols = {n.name: n.id for n in callee_nodes}
    edges = pm.extract_edges(
        tree,
        caller_nodes,
        str(tmp_path / "caller.py"),
        "python",
        source=CALLER_SRC,
        global_symbols=global_symbols,
    )

    calls = _calls(edges)
    assert calls, "cross-file call did not resolve"
    helper_id = next(n.id for n in callee_nodes if n.name == "shared_helper")
    assert any(e.to_id == helper_id for e in calls), "edge did not point at the other file's symbol"


def test_local_definition_shadows_the_global_one(pm, tmp_path):
    """A file-local symbol must win over a same-named symbol from another file."""
    _, other_nodes = _parse(pm, tmp_path / "other.py", CALLEE_SRC)
    tree, local_nodes = _parse(pm, tmp_path / "local.py", SHADOWING_SRC)

    global_symbols = {n.name: n.id for n in other_nodes}
    edges = pm.extract_edges(
        tree,
        local_nodes,
        str(tmp_path / "local.py"),
        "python",
        source=SHADOWING_SRC,
        global_symbols=global_symbols,
    )

    calls = _calls(edges)
    assert calls, "local call did not resolve"
    local_id = next(n.id for n in local_nodes if n.name == "shared_helper")
    foreign_id = next(n.id for n in other_nodes if n.name == "shared_helper")
    assert any(e.to_id == local_id for e in calls), "local definition was not preferred"
    assert all(e.to_id != foreign_id for e in calls), "resolved to the other file's symbol"


def test_global_table_does_not_disturb_same_file_resolution(pm, tmp_path):
    """Passing an unrelated global table must not change same-file behaviour."""
    tree, nodes = _parse(pm, tmp_path / "solo.py", SHADOWING_SRC)

    without = _calls(
        pm.extract_edges(tree, nodes, str(tmp_path / "solo.py"), "python", source=SHADOWING_SRC)
    )
    with_unrelated = _calls(
        pm.extract_edges(
            tree,
            nodes,
            str(tmp_path / "solo.py"),
            "python",
            source=SHADOWING_SRC,
            global_symbols={"unrelated": "0" * 24},
        )
    )
    assert {(e.from_id, e.to_id) for e in without} == {(e.from_id, e.to_id) for e in with_unrelated}
