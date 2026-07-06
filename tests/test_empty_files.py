"""
test_empty_files.py - Tests for graceful handling of empty source files
(issue #1).

Structure
---------
TestParserManagerEmptyFiles  - parse_file skips empty files read from disk
TestParsingServiceEmptyFiles - ParsingService returns no results, no raise
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ast_rag.services.parsing.parser_manager import ParserManager
from ast_rag.services.parsing_service import ParsingService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pm() -> ParserManager:
    return ParserManager()


def _file(tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


# ===========================================================================
# TestParserManagerEmptyFiles
# ===========================================================================


class TestParserManagerEmptyFiles:
    def test_empty_file_skipped(self, pm: ParserManager, tmp_path: Path, caplog) -> None:
        path = _file(tmp_path, "empty.py", "")
        with caplog.at_level("WARNING"):
            assert pm.parse_file(path) is None
        assert any("empty" in rec.message.lower() for rec in caplog.records)

    def test_whitespace_only_file_skipped(self, pm: ParserManager, tmp_path: Path) -> None:
        path = _file(tmp_path, "blank.java", "   \n\n\t  \n")
        assert pm.parse_file(path) is None

    def test_empty_files_skipped_for_all_languages(self, pm: ParserManager, tmp_path: Path) -> None:
        for name in ("a.py", "b.java", "c.cpp", "d.rs", "e.ts"):
            assert pm.parse_file(_file(tmp_path, name, "")) is None

    def test_explicit_empty_source_still_parses(self, pm: ParserManager, tmp_path: Path) -> None:
        # The git updater passes source= explicitly when diffing an emptied
        # file against its old version; it must still get a valid (empty)
        # tree so stale nodes are expired. The skip only covers disk reads.
        path = _file(tmp_path, "emptied.py", "")
        tree = pm.parse_file(path, source=b"")
        assert tree is not None

    def test_non_empty_file_parses_normally(self, pm: ParserManager, tmp_path: Path) -> None:
        path = _file(tmp_path, "code.py", "def foo():\n    return 1\n")
        tree = pm.parse_file(path, resolve=True)
        assert tree is not None
        assert tree.root_node is not None

    def test_comment_only_file_parses_normally(self, pm: ParserManager, tmp_path: Path) -> None:
        # Not whitespace-only: still parsed (yields no nodes, but no skip)
        path = _file(tmp_path, "comments.py", "# just a comment\n")
        assert pm.parse_file(path) is not None


# ===========================================================================
# TestParsingServiceEmptyFiles
# ===========================================================================


class TestParsingServiceEmptyFiles:
    def test_parse_file_returns_empty_results(self, tmp_path: Path, caplog) -> None:
        service = ParsingService()
        path = _file(tmp_path, "empty.py", "")
        with caplog.at_level("WARNING"):
            nodes, edges = service.parse_file(path)
        assert nodes == []
        assert edges == []
        assert any("empty" in rec.message.lower() for rec in caplog.records)

    def test_parse_file_does_not_raise_for_whitespace(self, tmp_path: Path) -> None:
        service = ParsingService()
        path = _file(tmp_path, "blank.rs", "  \n\n")
        nodes, edges = service.parse_file(path)
        assert (nodes, edges) == ([], [])

    def test_parse_directory_skips_empty_keeps_rest(self, tmp_path: Path) -> None:
        service = ParsingService()
        _file(tmp_path, "empty.py", "")
        _file(tmp_path, "real.py", "class Foo:\n    def bar(self):\n        pass\n")

        nodes, _edges = service.parse_directory(tmp_path)
        names = {n.name for n in nodes}
        assert "Foo" in names
        # The empty file contributed nothing and did not abort the walk
        assert all(not n.file_path.endswith("empty.py") for n in nodes)

    def test_non_empty_file_unaffected(self, tmp_path: Path) -> None:
        service = ParsingService()
        path = _file(tmp_path, "code.py", "def foo():\n    return 1\n")
        nodes, _edges = service.parse_file(path)
        assert any(n.name == "foo" for n in nodes)
