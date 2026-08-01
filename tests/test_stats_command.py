"""Tests for the `ast-rag stats` command (issue #13)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from typer.testing import CliRunner

from ast_rag.cli import _collect_index_stats, app

runner = CliRunner()


class _Rec(dict):
    """Minimal stand-in for a neo4j Record."""

    def __getitem__(self, key):
        return super().__getitem__(key)


def _driver_returning(node_rows, edge_rows, lang_rows, files):
    """Build a mock driver whose session.run dispatches on the query text."""
    session = MagicMock()

    def run(query, **_kwargs):
        result = MagicMock()
        if "labels(n)[0]" in query:
            return iter([_Rec(kind=k, n=n) for k, n in node_rows])
        if "coalesce(r.kind" in query:
            return iter([_Rec(kind=k, n=n) for k, n in edge_rows])
        if "n.lang" in query:
            return iter([_Rec(lang=lang, n=n) for lang, n in lang_rows])
        result.single.return_value = _Rec(files=files)
        return result

    session.run.side_effect = run
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    return driver


def test_collect_index_stats_shapes_and_totals():
    driver = _driver_returning(
        node_rows=[("Method", 368), ("Class", 100)],
        edge_rows=[("CONTAINS_METHOD", 392), ("CALLS", 150)],
        lang_rows=[("java", 468)],
        files=103,
    )
    stats = _collect_index_stats(driver)

    assert stats["files"] == 103
    assert stats["nodes"]["total"] == 468
    assert stats["nodes"]["by_kind"] == {"Method": 368, "Class": 100}
    assert stats["edges"]["total"] == 542
    assert stats["edges"]["by_kind"]["CALLS"] == 150
    assert stats["languages"] == {"java": 468}


def test_edges_grouped_by_kind_property_not_relationship_type():
    """Edges are stored as :EDGE with the semantic type in `kind`."""
    driver = _driver_returning([], [], [], 0)
    _collect_index_stats(driver)
    queries = [
        c.args[0] for c in driver.session.return_value.__enter__.return_value.run.call_args_list
    ]
    edge_query = next(q for q in queries if "]->()" in q and "count(*)" in q)
    assert "r.kind" in edge_query, "edge stats must group by the kind property"


def test_bookkeeping_node_excluded():
    driver = _driver_returning([], [], [], 0)
    _collect_index_stats(driver)
    queries = [
        c.args[0] for c in driver.session.return_value.__enter__.return_value.run.call_args_list
    ]
    node_query = next(q for q in queries if "labels(n)[0]" in q)
    assert "CurrentVersion" in node_query


def test_empty_index_reports_hint(monkeypatch):
    monkeypatch.setattr("ast_rag.cli.create_driver", lambda _cfg: _driver_returning([], [], [], 0))
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "No indexed nodes found" in result.stdout


def test_json_output_is_valid_json(monkeypatch):
    monkeypatch.setattr(
        "ast_rag.cli.create_driver",
        lambda _cfg: _driver_returning([("Method", 5)], [("CALLS", 2)], [("python", 5)], 3),
    )
    result = runner.invoke(app, ["stats", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"]["by_kind"] == {"Method": 5}
    assert payload["edges"]["by_kind"] == {"CALLS": 2}
    assert payload["files"] == 3
