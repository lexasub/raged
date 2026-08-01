"""Regression tests for call-edge and block extraction.

`EdgeExtractor._extract_call_edges` built its candidate-caller list from an
empty list literal (``for n in []``) instead of the ``nodes`` argument its
caller already had, so ``_find_enclosing_callable`` never resolved a caller and
no ``CALLS`` edge was ever emitted -- in any language.

``ParserManager.extract_blocks`` had the mirror-image defect, so no code blocks
were extracted either.

Both are silent failures: ``ast-rag callers`` reported "No callers found." and
indexing reported "0 blocks" as successful output.
"""

from __future__ import annotations

import pytest

from ast_rag.dto.enums import EdgeKind
from ast_rag.services.parsing.parser_manager import ParserManager

PYTHON_SRC = b"""
def helper(x):
    return x + 1


def caller(y):
    return helper(y)


class Thing:
    def method(self, z):
        return helper(z)
"""

JAVA_SRC = b"""
public class Sample {
    int helper(int x) {
        return x + 1;
    }

    int caller(int y) {
        return helper(y);
    }
}
"""


def _edges(tmp_path, name: str, src: bytes, lang: str):
    path = tmp_path / name
    path.write_bytes(src)
    pm = ParserManager()
    tree = pm.parse_file(str(path), source=src)
    assert tree is not None, f"parser returned no tree for {name}"
    nodes = pm.extract_nodes(tree, str(path), lang)
    edges = pm.extract_edges(tree, nodes, str(path), lang, source=src)
    return nodes, edges


def _calls(edges):
    return [e for e in edges if e.kind == EdgeKind.CALLS]


@pytest.mark.parametrize(
    "name,src,lang",
    [
        ("sample.py", PYTHON_SRC, "python"),
        ("Sample.java", JAVA_SRC, "java"),
    ],
)
def test_call_edges_are_extracted(tmp_path, name, src, lang):
    """A function calling another in the same file must yield a CALLS edge."""
    nodes, edges = _edges(tmp_path, name, src, lang)
    assert nodes, "no nodes extracted"

    calls = _calls(edges)
    assert calls, (
        f"no CALLS edges extracted from {name}; got kinds {sorted({str(e.kind) for e in edges})}"
    )

    # The edge must point at `helper`, and originate from a real callable.
    helper_ids = {n.id for n in nodes if n.name == "helper"}
    assert helper_ids, "test fixture lost: no node named 'helper'"
    assert any(e.to_id in helper_ids for e in calls), "no CALLS edge resolved to 'helper'"

    node_ids = {n.id for n in nodes}
    assert all(e.from_id in node_ids for e in calls), "CALLS edge originates from an unknown node"


def test_python_blocks_are_extracted(tmp_path):
    """if/for/try/with bodies must be surfaced by extract_blocks."""
    src = b"""
def busy(items):
    total = 0
    for item in items:
        if item:
            total += item
    try:
        total += 1
    except ValueError:
        pass
    with open("f") as fh:
        fh.read()
    return total
"""
    path = tmp_path / "blocks.py"
    path.write_bytes(src)
    pm = ParserManager()
    tree = pm.parse_file(str(path), source=src)
    nodes = pm.extract_nodes(tree, str(path), "python")
    blocks, _ = pm.extract_blocks(tree, nodes, str(path), "python", source=src)
    assert blocks, "no blocks extracted from a function with for/if/try/with"
