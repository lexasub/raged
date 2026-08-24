"""Every console script must survive a plain ``pip install ast-rag``.

``[project.scripts]`` puts ``ast-rag-mcp`` and ``ast-rag-watch`` on the user's
PATH unconditionally, but ``mcp`` and ``watchdog`` ship only in the ``[mcp]``
extra. After the documented plain install both commands died at import with a
bare ``ModuleNotFoundError`` traceback that never names the extra that fixes
it.

``test_optional_watchdog`` pins the same rule for ``ast_rag.cli`` and
``ast_rag.services``; this covers the two entry points it does not reach.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"

# Top-level distributions that live only in an optional extra.
OPTIONAL_MODULES = ("mcp", "watchdog")

_BLOCKER = f"""
import sys
_BLOCKED = {OPTIONAL_MODULES!r}


class _Missing:
    def find_spec(self, name, path=None, target=None):
        top = name.split(".")[0]
        if top in _BLOCKED:
            raise ModuleNotFoundError("No module named %r" % top, name=top)
        return None


sys.meta_path.insert(0, _Missing())
"""


def _console_scripts() -> dict[str, str]:
    """``{script name: "module:attr"}`` from ``[project.scripts]``."""
    text = PYPROJECT.read_text()
    block = re.search(r"^\[project\.scripts\]\n(.*?)(?=^\[|\Z)", text, re.M | re.S)
    assert block, "pyproject.toml has no [project.scripts] table"
    return dict(re.findall(r'^(\S+)\s*=\s*"([^"]+)"', block.group(1), re.M))


def _run_without_extras(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCKER + code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize("script,target", sorted(_console_scripts().items()))
def test_console_script_imports_without_optional_extras(script: str, target: str):
    """Resolving the entry point must not need an optional dependency."""
    module, _, attr = target.partition(":")
    result = _run_without_extras(
        f"import importlib;"
        f"m = importlib.import_module({module!r});"
        f"getattr(m, {attr!r});"
        f"print('resolved')"
    )
    assert result.returncode == 0, (
        f"`{script}` cannot even be resolved after a plain install:\n{result.stderr}"
    )
    assert "resolved" in result.stdout


@pytest.mark.parametrize(
    "script,target,extra",
    [
        ("ast-rag-mcp", "ast_rag.ast_rag_mcp:main", "mcp"),
        ("ast-rag-watch", "ast_rag.watcher:main", "mcp"),
    ],
)
def test_optional_script_names_the_extra_instead_of_crashing(script: str, target: str, extra: str):
    """Running it without the extra must say how to install it."""
    module, _, attr = target.partition(":")
    result = _run_without_extras(
        f"import importlib;getattr(importlib.import_module({module!r}), {attr!r})()"
    )
    output = result.stdout + result.stderr

    assert result.returncode != 0, f"`{script}` reported success without {extra!r} installed"
    assert "Traceback" not in output, (
        f"`{script}` failed with a raw traceback instead of install guidance:\n{output}"
    )
    assert "pip install" in output, f"`{script}` did not tell the user how to fix it:\n{output}"
    assert f"ast-rag[{extra}]" in output, f"`{script}` did not name the `{extra}` extra:\n{output}"
