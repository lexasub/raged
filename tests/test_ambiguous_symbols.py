"""An ambiguous name must say so instead of silently answering about one match.

``find_definition`` returns every symbol matching a name; six CLI sites took
``defs[0]`` and reported on that one alone. For names that are ambiguous by
nature -- ``main``, ``run``, ``parse``, ``__init__`` -- the answer is usually
about a symbol the user did not mean, and an empty result reads as "this
function has no blocks" rather than "I looked at the wrong function".

Reported in #64 against ``blocks``, where ``ast-rag blocks main`` printed "No
blocks found." while another ``main`` in the same repo had 45.

The wording follows the issue discussion: lead with "ambiguous" so an agent
reading the output hits it first, and no exclamation mark.
"""

from __future__ import annotations

from dataclasses import dataclass

from ast_rag.cli import _describe_ambiguity


@dataclass
class _Def:
    """Minimal stand-in for ASTNode: only the fields the note reads."""

    qualified_name: str
    lang: str
    kind: str = "Function"


def test_single_match_is_not_ambiguous():
    assert _describe_ambiguity("main", [_Def("test_phase2.main", "python")]) is None


def test_no_match_is_not_ambiguous():
    assert _describe_ambiguity("main", []) is None


def test_note_leads_with_ambiguous_and_counts_matches():
    defs = [
        _Def("TestCallResolution.main", "java"),
        _Def("test_phase2.main", "python"),
        _Def("benchmark_hybrid.main", "python"),
    ]
    note = _describe_ambiguity("main", defs)

    assert note is not None
    assert note.lower().startswith("ambiguous"), note
    assert "!" not in note, f"issue asked for no exclamation mark: {note}"
    assert "3" in note, note


def test_note_breaks_down_by_language():
    defs = [
        _Def("a.main", "python"),
        _Def("b.main", "python"),
        _Def("c.main", "cpp"),
    ]
    note = _describe_ambiguity("main", defs)
    assert "2 python" in note, note
    assert "1 cpp" in note, note


def test_note_names_the_symbol_actually_used_and_an_alternative():
    defs = [
        _Def("TestCallResolution.main", "java"),
        _Def("test_phase2.main", "python"),
    ]
    note = _describe_ambiguity("main", defs)

    # The whole point of #64: the user could not tell which symbol was chosen.
    assert "TestCallResolution.main" in note, note
    # ...and needs to know how to ask for the other one.
    assert "test_phase2.main" in note, note
