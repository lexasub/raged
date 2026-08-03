"""Regression tests for CLI short-flag collisions (issue #65).

`ast-rag lambdas` used to declare ``-c`` twice: once for ``--captured`` and
once for ``--config``. Click/Typer silently lets the later declaration win, so
``--config`` shadowed ``--captured`` and the documented ``-c`` shorthand for
``--captured`` was unusable. These tests introspect every registered command's
short flags to guard against that class of bug returning.
"""

from __future__ import annotations

from collections import Counter

import typer

from ast_rag.cli import app


def _short_flags(params) -> list[str]:
    """Return the single-dash short flags (e.g. ``-c``) declared on ``params``."""
    shorts: list[str] = []
    for param in params:
        for opt in getattr(param, "opts", []):
            if opt.startswith("-") and not opt.startswith("--") and len(opt) == 2:
                shorts.append(opt)
    return shorts


def _commands() -> dict:
    """Map command name -> click.Command for every registered CLI command."""
    return typer.main.get_command(app).commands


def test_no_duplicate_short_flags_in_any_command() -> None:
    """No command may bind the same short flag to two different options."""
    offenders: dict[str, list[str]] = {}
    for name, command in _commands().items():
        counts = Counter(_short_flags(command.params))
        dupes = sorted(flag for flag, n in counts.items() if n > 1)
        if dupes:
            offenders[name] = dupes
    assert not offenders, f"commands with duplicate short flags: {offenders}"


def test_lambdas_captured_uses_capital_c_and_config_keeps_lowercase_c() -> None:
    """``--captured`` moved to ``-C`` so ``--config`` keeps the shared ``-c``."""
    opts = {param.name: list(param.opts) for param in _commands()["lambdas"].params}
    assert "-C" in opts["captured"], opts["captured"]
    assert "-c" not in opts["captured"], opts["captured"]
    assert "-c" in opts["config"], opts["config"]


def test_config_short_flag_convention_preserved() -> None:
    """``-c`` is the project-wide shorthand for ``--config`` and nothing else."""
    for name, command in _commands().items():
        for param in command.params:
            opts = getattr(param, "opts", [])
            if "--config" in opts:
                assert "-c" in opts, f"{name}: --config lost its -c shorthand: {opts}"
            if "-c" in opts:
                assert "--config" in opts, f"{name}: -c bound to non-config option: {opts}"
