"""Guards for entry points whose dependencies live in an optional extra.

``[project.scripts]`` installs every console script unconditionally, so
``ast-rag-mcp`` and ``ast-rag-watch`` land on the PATH of a user who ran a
plain ``pip install ast-rag``. Without a guard they die at import with a
``ModuleNotFoundError`` that never mentions the extra that fixes it.
"""

from __future__ import annotations

import importlib.util
from typing import Callable, NoReturn


def module_available(name: str) -> bool:
    """Whether *name* can be imported, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def missing_dependency_entry_point(
    module: str, *, feature: str, extra: str
) -> Callable[..., NoReturn]:
    """Build a ``main`` that names the extra to install and exits non-zero."""

    def main(*_args, **_kwargs) -> NoReturn:
        raise SystemExit(
            f"{feature} needs the optional '{module}' package, which is not installed.\n"
            f"Install it with:  pip install 'ast-rag[{extra}]'"
        )

    return main
