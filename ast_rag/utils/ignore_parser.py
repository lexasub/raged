""".cgrignore file parser for AST-RAG.

Format: same as .gitignore
- One pattern per line
- ``#`` for comments
- ``!`` for negation
- ``**`` for matching across directories
- ``*`` for wildcard matching

If no ``.cgrignore`` file exists, a set of sensible default patterns is used
(VCS metadata, build artifacts, dependency directories, IDE files). An
existing but empty ``.cgrignore`` disables the defaults and ignores nothing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pathspec

logger = logging.getLogger(__name__)

DEFAULT_IGNORE_PATTERNS: list[str] = [
    ".git/",
    ".svn/",
    ".hg/",
    "__pycache__/",
    "node_modules/",
    "target/",
    "build/",
    "dist/",
    ".gradle/",
    ".idea/",
    ".vscode/",
    "venv/",
    ".venv/",
    "*.pyc",
    "*.pyo",
    "*.class",
    "*.o",
    "*.so",
    "*.dll",
]


class CgrIgnoreParser:
    """Parser for ``.cgrignore`` files (gitignore-style exclusion rules).

    Usage::

        parser = CgrIgnoreParser("/path/to/project")
        parser.load()  # reads /path/to/project/.cgrignore or falls back to defaults
        parser.should_ignore("/path/to/project/build/out.o")  # True
    """

    def __init__(self, root_path: str) -> None:
        self.root_path = Path(root_path).resolve()
        self.spec: Optional[pathspec.PathSpec] = None
        self.patterns: list[str] = []

    def load(self, ignore_file: Optional[str] = None) -> None:
        """Load ignore patterns from a ``.cgrignore`` file.

        Args:
            ignore_file: Path to the ignore file. If None, looks for
                ``.cgrignore`` in the root path. If the file does not exist,
                default patterns are used.
        """
        if ignore_file is None:
            ignore_file = os.path.join(self.root_path, ".cgrignore")

        if not os.path.exists(ignore_file):
            self._load_defaults()
            return

        try:
            with open(ignore_file, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
        except OSError as exc:
            logger.warning("Cannot read ignore file '%s': %s — using defaults", ignore_file, exc)
            self._load_defaults()
            return

        patterns = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)

        self.patterns = patterns
        self.spec = pathspec.GitIgnoreSpec.from_lines(patterns)
        logger.debug("Loaded %d ignore patterns from %s", len(patterns), ignore_file)

    def _load_defaults(self) -> None:
        """Load default ignore patterns (used when no ignore file exists)."""
        self.patterns = list(DEFAULT_IGNORE_PATTERNS)
        self.spec = pathspec.GitIgnoreSpec.from_lines(self.patterns)

    def should_ignore(self, file_path: str, is_dir: bool = False) -> bool:
        """Check if a path should be ignored.

        Args:
            file_path: Absolute or root-relative path to check.
            is_dir: True when the path is a directory. Required for
                directory-only patterns like ``build/`` to match, since
                gitwildmatch only applies them to paths marked as directories.

        Returns:
            True if the path matches the loaded ignore patterns.
        """
        if self.spec is None:
            self._load_defaults()
        assert self.spec is not None  # for type checkers

        path = Path(file_path)
        if path.is_absolute():
            try:
                rel_path = path.resolve().relative_to(self.root_path)
            except ValueError:
                # Path is not under root — never ignore.
                return False
        else:
            rel_path = path

        candidate = rel_path.as_posix()
        if is_dir:
            candidate += "/"
        return self.spec.match_file(candidate)

    def get_patterns(self) -> list[str]:
        """Return a copy of the loaded patterns."""
        return self.patterns.copy()
