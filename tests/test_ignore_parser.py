"""
test_ignore_parser.py - Tests for CgrIgnoreParser and its integration
with walk_source_files (.cgrignore support, issue #38).

Structure
---------
TestCgrIgnoreParser          - parser unit tests (load, defaults, matching)
TestWalkSourceFilesIgnore    - walk_source_files honoring .cgrignore rules
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ast_rag.utils.ignore_parser import CgrIgnoreParser, DEFAULT_IGNORE_PATTERNS
from ast_rag.services.parsing.parser_manager import walk_source_files


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(root: Path, rel_path: str, content: str = "pass\n") -> Path:
    """Create a file (and parent dirs) under root; return its path."""
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _walked_names(root: Path, **kwargs) -> set[str]:
    """Run walk_source_files and return the set of file basenames found."""
    return {os.path.basename(fp) for fp, _lang in walk_source_files(str(root), **kwargs)}


# ===========================================================================
# TestCgrIgnoreParser
# ===========================================================================


class TestCgrIgnoreParser:
    def test_load_reads_patterns_and_skips_comments(self, tmp_path: Path) -> None:
        _write(tmp_path, ".cgrignore", "# comment\n\n*.pyc\nbuild/\n  \n!keep.pyc\n")
        parser = CgrIgnoreParser(str(tmp_path))
        parser.load()
        assert parser.get_patterns() == ["*.pyc", "build/", "!keep.pyc"]

    def test_defaults_used_when_file_missing(self, tmp_path: Path) -> None:
        parser = CgrIgnoreParser(str(tmp_path))
        parser.load()
        assert parser.get_patterns() == DEFAULT_IGNORE_PATTERNS
        assert parser.should_ignore(str(tmp_path / "a.pyc"))
        assert not parser.should_ignore(str(tmp_path / "a.py"))

    def test_empty_file_disables_defaults(self, tmp_path: Path) -> None:
        _write(tmp_path, ".cgrignore", "# only comments\n")
        parser = CgrIgnoreParser(str(tmp_path))
        parser.load()
        assert parser.get_patterns() == []
        assert not parser.should_ignore(str(tmp_path / "a.pyc"))

    def test_explicit_ignore_file_path(self, tmp_path: Path) -> None:
        custom = _write(tmp_path, "custom.rules", "*.rs\n")
        parser = CgrIgnoreParser(str(tmp_path))
        parser.load(str(custom))
        assert parser.should_ignore(str(tmp_path / "lib.rs"))
        assert not parser.should_ignore(str(tmp_path / "lib.py"))

    def test_negation_pattern(self, tmp_path: Path) -> None:
        _write(tmp_path, ".cgrignore", "*.py\n!main.py\n")
        parser = CgrIgnoreParser(str(tmp_path))
        parser.load()
        assert parser.should_ignore(str(tmp_path / "util.py"))
        assert not parser.should_ignore(str(tmp_path / "main.py"))

    def test_directory_pattern_requires_is_dir(self, tmp_path: Path) -> None:
        _write(tmp_path, ".cgrignore", "build/\n")
        parser = CgrIgnoreParser(str(tmp_path))
        parser.load()
        # gitwildmatch only applies `dir/` patterns to paths marked as dirs
        assert parser.should_ignore(str(tmp_path / "build"), is_dir=True)
        assert not parser.should_ignore(str(tmp_path / "build"), is_dir=False)
        assert parser.should_ignore(str(tmp_path / "build" / "gen.py"))

    def test_glob_across_directories(self, tmp_path: Path) -> None:
        _write(tmp_path, ".cgrignore", "**/generated/**\n")
        parser = CgrIgnoreParser(str(tmp_path))
        parser.load()
        assert parser.should_ignore(str(tmp_path / "src" / "generated" / "x.py"))
        assert not parser.should_ignore(str(tmp_path / "src" / "handwritten" / "x.py"))

    def test_relative_paths_are_root_relative(self, tmp_path: Path) -> None:
        _write(tmp_path, ".cgrignore", "build/\n")
        parser = CgrIgnoreParser(str(tmp_path))
        parser.load()
        assert parser.should_ignore("build/gen.py")

    def test_path_outside_root_never_ignored(self, tmp_path: Path) -> None:
        _write(tmp_path, ".cgrignore", "*.py\n")
        parser = CgrIgnoreParser(str(tmp_path / "sub"))
        parser.load(str(tmp_path / ".cgrignore"))
        assert not parser.should_ignore(str(tmp_path.parent / "elsewhere.py"))

    def test_get_patterns_returns_copy(self, tmp_path: Path) -> None:
        parser = CgrIgnoreParser(str(tmp_path))
        parser.load()
        parser.get_patterns().append("mutated")
        assert "mutated" not in parser.get_patterns()


# ===========================================================================
# TestWalkSourceFilesIgnore
# ===========================================================================


class TestWalkSourceFilesIgnore:
    def test_cgrignore_excludes_files_and_dirs(self, tmp_path: Path) -> None:
        _write(tmp_path, ".cgrignore", "gen/\nlegacy_*.py\n")
        _write(tmp_path, "src/main.py")
        _write(tmp_path, "src/legacy_api.py")
        _write(tmp_path, "gen/stubs.py")

        names = _walked_names(tmp_path)
        assert names == {"main.py"}

    def test_negation_keeps_file(self, tmp_path: Path) -> None:
        _write(tmp_path, ".cgrignore", "*.py\n!main.py\n")
        _write(tmp_path, "main.py")
        _write(tmp_path, "util.py")

        assert _walked_names(tmp_path) == {"main.py"}

    def test_defaults_prune_venv_when_no_cgrignore(self, tmp_path: Path) -> None:
        _write(tmp_path, "src/main.py")
        _write(tmp_path, "venv/lib/site.py")

        assert _walked_names(tmp_path) == {"main.py"}

    def test_explicit_ignore_file_option(self, tmp_path: Path) -> None:
        rules = _write(tmp_path, "ci.rules", "src/\n")
        _write(tmp_path, "src/main.py")
        _write(tmp_path, "app/app.py")

        names = _walked_names(tmp_path, ignore_file=str(rules))
        assert names == {"app.py"}

    def test_exclude_dirs_still_respected(self, tmp_path: Path) -> None:
        _write(tmp_path, ".cgrignore", "gen/\n")
        _write(tmp_path, "src/main.py")
        _write(tmp_path, "gen/stubs.py")
        _write(tmp_path, "skipme/other.py")

        names = _walked_names(tmp_path, exclude_dirs=["skipme"])
        assert names == {"main.py"}

    def test_relative_root(self, tmp_path: Path) -> None:
        _write(tmp_path, ".cgrignore", "gen/\n")
        _write(tmp_path, "src/main.py")
        _write(tmp_path, "gen/stubs.py")

        old_cwd = os.getcwd()
        os.chdir(tmp_path.parent)
        try:
            names = _walked_names(Path(tmp_path.name))
        finally:
            os.chdir(old_cwd)
        assert names == {"main.py"}

    def test_languages_detected_for_kept_files(self, tmp_path: Path) -> None:
        _write(tmp_path, ".cgrignore", "vendor/\n")
        _write(tmp_path, "a.py")
        _write(tmp_path, "B.java", "class B {}\n")
        _write(tmp_path, "vendor/c.rs", "fn main() {}\n")

        langs = {os.path.basename(fp): lang for fp, lang in walk_source_files(str(tmp_path))}
        assert langs == {"a.py": "python", "B.java": "java"}
