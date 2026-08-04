"""Regression test for issue #63.

``ast-rag sig`` builds a Neo4j regex to match against stored function
signatures. Signatures with several parameters are persisted wrapped across
multiple lines (e.g. ``"parse_file(\n    self,\n    file_path: str,\n)"``).
Neo4j's ``=~`` operator uses Java regex semantics, where ``.`` does not match
``\n`` unless the ``DOTALL`` flag is active. Without that flag every ``.*`` in
the generated pattern silently fails to bridge a line break, so multi-line
signatures never match -- 26% of functions in the reporter's repo were
unreachable via signature search.

This test does not require a live Neo4j/Qdrant connection: it exercises the
pure string-building method directly, then confirms the resulting pattern
behaves correctly under Python's ``re`` module with the equivalent
``re.DOTALL`` semantics that Neo4j's inline ``(?s)`` flag requests.
"""

from __future__ import annotations

import re

from ast_rag.api.ast_rag_api import ASTRagAPI


def _build_regex(pattern: str) -> str:
    # driver/embedding_manager are unused by _build_signature_regex, so None
    # stand-ins are enough to exercise this pure string-building method
    # without touching Neo4j or Qdrant.
    api = ASTRagAPI(driver=None, embedding_manager=None)  # type: ignore[arg-type]
    parsed = api._parse_signature_pattern(pattern)
    return api._build_signature_regex(parsed)


def test_signature_regex_enables_dotall():
    """The generated pattern must opt in to DOTALL so '.' can cross lines."""
    regex = _build_regex("parse_file")
    assert regex.startswith("(?s)"), (
        f"expected inline DOTALL flag '(?s)' at the start of the generated regex, got: {regex!r}"
    )


def test_signature_regex_matches_multiline_signature():
    """A signature wrapped across multiple lines must still match.

    This mirrors the exact repro from issue #63: a multi-parameter function
    whose signature is stored with embedded newlines.
    """
    wrapped_signature = "parse_file(\n        self,\n        file_path: str,\n    )"
    regex = _build_regex("parse_file")

    # Neo4j evaluates the pattern with Java's Pattern class, applying any
    # inline flags found in the pattern text itself (e.g. "(?s)"). Python's
    # `re` module honors the same inline-flag syntax, so this exercises the
    # identical semantics without requiring a running Neo4j instance.
    assert re.match(regex, wrapped_signature) is not None, (
        f"pattern {regex!r} failed to match a multi-line signature; "
        "the DOTALL flag is likely missing"
    )


def test_signature_regex_still_matches_single_line_signature():
    """Guard against regressing the common (single-line) case."""
    single_line_signature = "parse_file(self, file_path: str)"
    regex = _build_regex("parse_file")
    assert re.match(regex, single_line_signature) is not None


def test_wildcard_pattern_matches_multiline_signature():
    """Wildcard name patterns should also cross line breaks."""
    wrapped_signature = "def process_batch(\n    self,\n    items: list,\n) -> None"
    regex = _build_regex("process*")
    assert re.match(regex, wrapped_signature) is not None
