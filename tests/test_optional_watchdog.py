"""The CLI must import without the optional ``watchdog`` dependency.

``watchdog`` is declared only in the ``mcp`` / ``dev`` extras, but
``ast_rag.services.__init__`` imported ``watcher_service`` eagerly, which
imports ``watchdog`` at module scope. The documented install path

    pip install -e .

therefore produced a CLI that died on ``ast-rag --help`` with
``ModuleNotFoundError: No module named 'watchdog'``.
"""

from __future__ import annotations

import subprocess
import sys


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_cli_import_does_not_pull_in_watchdog():
    """Importing the CLI must not drag the optional dependency in."""
    result = _run("import sys; import ast_rag.cli; print('watchdog' in sys.modules)")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", (
        f"ast_rag.cli imported watchdog eagerly: {result.stdout!r}"
    )


def test_services_package_imports_without_watchdog():
    """``ast_rag.services`` must import even when watchdog cannot be found."""
    result = _run(
        "import sys;"
        "sys.modules['watchdog'] = None;"
        "sys.modules['watchdog.observers'] = None;"
        "import ast_rag.services;"
        "print('ok')"
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_workspace_watcher_still_exported():
    """The lazy attribute must still resolve when watchdog is installed."""
    from ast_rag import services

    assert "WorkspaceWatcher" in services.__all__
    try:
        import watchdog  # noqa: F401
    except ModuleNotFoundError:
        return  # optional extra absent; nothing further to assert
    assert services.WorkspaceWatcher is not None
