"""Every Cypher parameter a query references must actually be bound.

Neo4j only reports an unbound parameter at execution time, as
``Neo.ClientError.Statement.ParameterMissing``. A query rewritten to use a new
parameter therefore keeps compiling, keeps passing any test that does not reach
that exact code path against a live database, and fails only in front of a user.

That happened: the call traversals were changed to filter on ``$call_kinds``,
two ``session.run`` sites were not given the parameter, and ``ast-rag refs`` and
``ast-rag symbol-impact`` broke.

Runtime tests are a poor fit -- reaching every branch needs a populated graph.
This checks it statically instead: for each ``session.run(<var>, **kwargs)`` in
the API layer, resolve ``<var>`` back to its query string and assert every
``$parameter`` in that string appears in the keyword arguments.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

PARAM = re.compile(r"\$([a-zA-Z_][a-zA-Z0-9_]*)")

SOURCES = [
    Path(__file__).parent.parent / "ast_rag" / "api" / "ast_rag_api.py",
    Path(__file__).parent.parent / "ast_rag" / "repositories" / "queries.py",
]


def _string_constants(tree: ast.AST) -> dict[str, str]:
    """Map local variable names to the string literals assigned to them."""
    literals: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            literals[target.id] = value.value
        elif isinstance(value, ast.JoinedStr):
            # f-string: keep the literal parts, which is where $params live
            literals[target.id] = "".join(
                p.value for p in value.values if isinstance(p, ast.Constant)
            )
    return literals


def _run_calls(tree: ast.AST):
    """Yield (query_text, bound_param_names, lineno) for each session.run(...)."""
    literals = _string_constants(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "run"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            query = first.value
        elif isinstance(first, ast.Name) and first.id in literals:
            query = literals[first.id]
        elif isinstance(first, ast.JoinedStr):
            query = "".join(p.value for p in first.values if isinstance(p, ast.Constant))
        else:
            continue  # cannot resolve statically; skipped rather than guessed at
        bound = {kw.arg for kw in node.keywords if kw.arg}
        yield query, bound, node.lineno


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.name)
def test_every_cypher_parameter_is_bound(source: Path):
    tree = ast.parse(source.read_text())
    problems = []
    for query, bound, lineno in _run_calls(tree):
        for name in sorted(set(PARAM.findall(query))):
            if name not in bound:
                problems.append(f"{source.name}:{lineno} uses ${name} but does not bind it")
    assert not problems, "unbound Cypher parameters:\n  " + "\n  ".join(problems)


def test_checker_resolves_real_queries():
    """Guard the checker itself: it must actually be finding queries."""
    tree = ast.parse(SOURCES[0].read_text())
    calls = list(_run_calls(tree))
    assert len(calls) >= 5, f"static checker only resolved {len(calls)} queries; it may be broken"
    assert any("$node_id" in q for q, _, _ in calls), "expected to see $node_id in some query"
