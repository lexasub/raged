"""
test_unsupported_language.py - Tests for informative error messages when
an unsupported file extension is encountered (issue #2).

Structure
---------
TestSupportedExtensionsHelpers - supported_extensions / format helpers
TestParserManagerUnsupported   - parse_file warns and returns None
TestParsingServiceUnsupported  - ParsingService raises a helpful ValueError
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ast_rag.services.parsing.parser_manager import (
    EXT_TO_LANG,
    ParserManager,
    format_supported_languages,
    supported_extensions,
)
from ast_rag.services.parsing_service import ParsingService


@pytest.fixture(scope="module")
def pm() -> ParserManager:
    return ParserManager()


# ===========================================================================
# TestSupportedExtensionsHelpers
# ===========================================================================


class TestSupportedExtensionsHelpers:
    def test_supported_extensions_covers_all_mapped(self) -> None:
        mapping = supported_extensions()
        flattened = {ext for exts in mapping.values() for ext in exts}
        assert flattened == set(EXT_TO_LANG)

    def test_supported_extensions_grouped_by_language(self) -> None:
        mapping = supported_extensions()
        assert mapping["java"] == [".java"]
        assert mapping["typescript"] == [".ts", ".tsx"]
        assert ".cpp" in mapping["cpp"]
        assert ".h" in mapping["cpp"]

    def test_format_lists_every_language(self) -> None:
        text = format_supported_languages()
        for lang in ("cpp", "java", "python", "rust", "typescript"):
            assert lang in text
        assert ".java" in text
        assert ".tsx" in text


# ===========================================================================
# TestParserManagerUnsupported
# ===========================================================================


class TestParserManagerUnsupported:
    def test_returns_none_and_warns(self, pm: ParserManager, tmp_path: Path, caplog) -> None:
        path = tmp_path / "main.go"
        path.write_text("package main\n", encoding="utf-8")
        with caplog.at_level("WARNING"):
            assert pm.parse_file(str(path)) is None
        messages = [rec.message for rec in caplog.records]
        assert any(".go" in m and "Supported languages" in m for m in messages)

    def test_extension_less_file_warns(self, pm: ParserManager, tmp_path: Path, caplog) -> None:
        path = tmp_path / "Makefile"
        path.write_text("all:\n", encoding="utf-8")
        with caplog.at_level("WARNING"):
            assert pm.parse_file(str(path)) is None
        assert any("<none>" in rec.message for rec in caplog.records)

    def test_supported_file_does_not_warn(self, pm: ParserManager, tmp_path: Path, caplog) -> None:
        path = tmp_path / "ok.py"
        path.write_text("def f():\n    pass\n", encoding="utf-8")
        with caplog.at_level("WARNING"):
            assert pm.parse_file(str(path)) is not None
        assert not any("unsupported" in rec.message.lower() for rec in caplog.records)


# ===========================================================================
# TestParsingServiceUnsupported
# ===========================================================================


class TestParsingServiceUnsupported:
    def test_value_error_lists_supported_languages(self, tmp_path: Path) -> None:
        service = ParsingService()
        path = tmp_path / "main.go"
        path.write_text("package main\n", encoding="utf-8")
        with pytest.raises(ValueError) as exc_info:
            service.parse_file(str(path))
        message = str(exc_info.value)
        assert "'.go'" in message
        assert "Supported languages" in message
        assert "java" in message

    def test_explicit_lang_hint_bypasses_detection(self, tmp_path: Path) -> None:
        service = ParsingService()
        path = tmp_path / "script.py"
        path.write_text("def f():\n    pass\n", encoding="utf-8")
        nodes, _edges = service.parse_file(str(path), lang="python")
        assert any(n.name == "f" for n in nodes)
