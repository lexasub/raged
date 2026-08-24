"""JSON is the default CLI output, so stdout must always be one JSON document.

``get_formatter(humanize=False)`` returns ``JSONFormatter``, so ``query``,
``goto`` and ``callers`` are machine-readable by default -- the point of a
tool whose README opens with "for AI agents".

All three abandon that contract on their failure paths: they ``console.print``
a Rich-coloured line to stdout and exit before the formatter is reached. So

    json.loads(subprocess.check_output(["ast-rag", "goto", name]))

works while the symbol exists and raises ``JSONDecodeError`` the moment it
does not. ``_warn_if_ambiguous`` breaks it from the other side -- its
docstring says the note is returned "for JSON consumers", but it prints the
note as well, so an ambiguous name emits a bare line *before* the document.

Exit codes are not the problem and are left alone: 0 and 1 still separate the
cases. What has to hold is that stdout parses, and that anything the user
needs to know survives into the JSON rather than being lost with the
Rich line.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from ast_rag import cli
from ast_rag.cli import app
from ast_rag.dto import ASTNode, SearchResult
from ast_rag.dto.enums import Language, NodeKind

runner = CliRunner()


def _node(qualified_name: str, *, name: str | None = None, line: int = 1) -> ASTNode:
    return ASTNode(
        kind=NodeKind.FUNCTION,
        name=name or qualified_name.rsplit(".", 1)[-1],
        qualified_name=qualified_name,
        lang=Language.PYTHON,
        file_path=f"src/{qualified_name.split('.')[0]}.py",
        start_line=line,
        end_line=line + 5,
        start_byte=0,
        end_byte=100,
    )


@pytest.fixture
def api(monkeypatch):
    """Replace config loading and the API so no service is needed."""
    fake = MagicMock()
    monkeypatch.setattr(cli, "_load_config", lambda *_a, **_k: MagicMock())
    monkeypatch.setattr(cli, "_build_api", lambda *_a, **_k: fake)
    return fake


def _stdout_json(result, argv: list[str]) -> dict:
    """Parse stdout, failing with the offending text rather than a traceback."""
    assert result.stdout.strip(), (
        f"`ast-rag {' '.join(argv)}` wrote nothing to stdout; "
        f"an agent parsing it gets an empty string (exit {result.exit_code})"
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"`ast-rag {' '.join(argv)}` is in JSON mode but stdout does not parse "
            f"({exc}):\n{result.stdout!r}"
        )


def _run(argv: list[str]):
    return runner.invoke(app, argv)


# --------------------------------------------------------------------------
# Empty results must still be a JSON document, not a coloured line
# --------------------------------------------------------------------------


def test_goto_not_found_still_emits_json(api):
    api.find_definition.return_value = []
    argv = ["goto", "NoSuchSymbol"]

    payload = _stdout_json(_run(argv), argv)

    assert payload["count"] == 0
    assert payload["definitions"] == []


def test_query_no_results_still_emits_json(api):
    api.search_semantic.return_value = []
    argv = ["query", "nothing matches this"]

    payload = _stdout_json(_run(argv), argv)

    assert payload["count"] == 0
    assert payload["results"] == []


def test_callers_symbol_not_found_still_emits_json(api):
    api.find_definition.return_value = []
    argv = ["callers", "NoSuchSymbol"]

    payload = _stdout_json(_run(argv), argv)

    assert payload["count"] == 0
    assert payload["callers"] == []


def test_callers_with_no_callers_still_emits_json(api):
    api.find_definition.return_value = [_node("pkg.orphan")]
    api.find_callers.return_value = []
    argv = ["callers", "pkg.orphan"]

    payload = _stdout_json(_run(argv), argv)

    assert payload["count"] == 0
    assert payload["callers"] == []


# --------------------------------------------------------------------------
# The ambiguity note must travel inside the document, not before it
# --------------------------------------------------------------------------


def test_ambiguous_name_keeps_stdout_parseable(api):
    api.find_definition.return_value = [_node("a.main"), _node("b.main")]
    api.find_callers.return_value = [_node("c.caller")]
    argv = ["callers", "main"]

    payload = _stdout_json(_run(argv), argv)

    assert payload["count"] == 1


def test_ambiguity_note_survives_into_the_json(api):
    """#64's warning is useless to an agent if it only exists as a Rich line."""
    api.find_definition.return_value = [_node("a.main"), _node("b.main")]
    api.find_callers.return_value = [_node("c.caller")]
    argv = ["callers", "main"]

    payload = _stdout_json(_run(argv), argv)

    note = payload.get("ambiguous")
    assert note, f"ambiguity was not reported in the JSON payload: {payload}"
    assert "a.main" in note, note


# --------------------------------------------------------------------------
# Guards: the working paths and human mode must not regress
# --------------------------------------------------------------------------


def test_goto_found_emits_json(api):
    api.find_definition.return_value = [_node("pkg.handler", line=42)]
    argv = ["goto", "pkg.handler"]

    payload = _stdout_json(_run(argv), argv)

    assert payload["count"] == 1
    assert payload["definitions"][0]["qualified_name"] == "pkg.handler"
    assert payload["definitions"][0]["start_line"] == 42


def test_query_with_results_emits_json(api):
    api.search_semantic.return_value = [SearchResult(node=_node("pkg.cache"), score=0.91)]
    argv = ["query", "cache with expiration"]

    payload = _stdout_json(_run(argv), argv)

    assert payload["count"] == 1
    assert payload["results"][0]["qualified_name"] == "pkg.cache"


def test_humanize_still_produces_text_not_json(api):
    """--humanize is the opt-out; it must keep producing Rich output."""
    api.find_definition.return_value = [_node("pkg.handler")]
    result = _run(["goto", "--humanize", "pkg.handler"])

    with pytest.raises(json.JSONDecodeError):
        json.loads(result.stdout)
    assert "pkg.handler" in result.stdout


def test_humanize_empty_result_is_reported_to_the_user(api):
    """The human path must not go silent when the JSON path stops printing."""
    api.find_definition.return_value = []
    result = _run(["goto", "--humanize", "NoSuchSymbol"])

    combined = result.stdout + result.stderr
    assert "NoSuchSymbol" in combined, f"nothing told the user the lookup failed:\n{combined!r}"
