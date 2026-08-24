"""``A..B`` must describe B, not whatever is checked out right now.

Both git-diff paths read the *old* side from the object store with
``_read_blob`` and the *new* side with ``open(file_path)`` -- the working
tree. ``compute_diff_for_commits`` promises "added_nodes: New AST nodes in
to_commit", and ``ast-rag update`` takes ``--from-commit`` and ``--to-commit``
as required options, so a user naming any pair other than
``<something>..HEAD`` on a clean tree gets the current source parsed and
stamped with ``to_commit``.

``update_from_git`` has the same line, and that one writes to Neo4j: the
graph ends up holding code that never existed at the commit it is labelled
with. Uncommitted work already has its own path in ``get_workspace_diff``, so
the two are not meant to be the same thing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ast_rag.services.graph_updater_service import compute_diff_for_commits

pytest.importorskip("git", reason="GitPython is needed to build the fixture repo")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> dict:
    """A repo where `alpha` sits on a different line in A, B and the tree."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")

    module = root / "mod.py"

    # A: alpha on line 1
    module.write_text("def alpha():\n    return 1\n")
    commit_a = _commit(root, "a")

    # B: one comment pushes alpha to line 2, and the body changes so the
    # node is genuinely part of the diff rather than untouched.
    module.write_text("# header\ndef alpha():\n    return 2\n")
    commit_b = _commit(root, "b")

    # Working tree only: five more comments push alpha to line 7. Never committed.
    module.write_text("# header\n# 1\n# 2\n# 3\n# 4\n# 5\ndef alpha():\n    return 2\n")

    return {"path": str(root), "a": commit_a, "b": commit_b, "file": module}


def _alpha(diff) -> object:
    nodes = [
        n
        for n in (*diff.added_nodes, *diff.updated_nodes)
        if n.name == "alpha" and n.kind.value in {"Function", "Method"}
    ]
    assert nodes, (
        "diff reported no `alpha` node at all; "
        f"added={[n.name for n in diff.added_nodes]} "
        f"updated={[n.name for n in diff.updated_nodes]}"
    )
    return nodes[0]


def test_working_tree_is_not_mistaken_for_the_target_commit(repo):
    """A..B must parse B's blob, not the file on disk."""
    diff = compute_diff_for_commits(repo["path"], repo["a"], repo["b"])
    alpha = _alpha(diff)

    assert alpha.start_line == 2, (
        f"`alpha` is on line 2 at to_commit and line 7 in the working tree; "
        f"the diff reported line {alpha.start_line}, so it parsed the working tree"
    )


def test_diff_is_stable_while_the_working_tree_moves(repo):
    """The same A..B must not change answer when an unrelated edit lands."""
    before = _alpha(compute_diff_for_commits(repo["path"], repo["a"], repo["b"])).start_line

    repo["file"].write_text("# a\n# b\n# c\n" + repo["file"].read_text())
    after = _alpha(compute_diff_for_commits(repo["path"], repo["a"], repo["b"])).start_line

    assert before == after, (
        f"editing an uncommitted file changed the reported diff of two fixed "
        f"commits ({before} -> {after})"
    )


def test_source_text_comes_from_the_target_commit(repo):
    """The body parsed must be B's body, not the tree's."""
    diff = compute_diff_for_commits(repo["path"], repo["a"], repo["b"])
    alpha = _alpha(diff)

    assert alpha.end_line == 3, (
        f"`alpha` spans lines 2-3 at to_commit; got {alpha.start_line}-{alpha.end_line}"
    )
