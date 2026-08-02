"""Regression test for the `ast-rag lambdas` short-flag collision (issue #65)."""

from __future__ import annotations

from typer.main import get_command

from ast_rag.cli import app


def _short_flag_map(command):
    """Map each short flag (e.g. ``-c``) to the parameter names that declare it."""
    shorts: dict[str, list[str]] = {}
    for param in command.params:
        for opt in list(param.opts) + list(param.secondary_opts):
            if opt.startswith("-") and not opt.startswith("--"):
                shorts.setdefault(opt, []).append(param.name)
    return shorts


def test_lambdas_short_flags_do_not_collide():
    """No short flag may be bound to more than one option.

    Before the fix, both ``--captured`` and ``--config`` declared ``-c``; the
    second registration won, so ``--captured``'s documented ``-c`` shorthand
    silently became ``--config``.
    """
    lambdas = get_command(app).commands["lambdas"]
    shorts = _short_flag_map(lambdas)

    collisions = {flag: names for flag, names in shorts.items() if len(names) > 1}
    assert not collisions, f"colliding short flags: {collisions}"

    # -c stays the project-wide shorthand for --config; --captured gets -C.
    assert shorts.get("-c") == ["config"]
    assert shorts.get("-C") == ["captured"]
