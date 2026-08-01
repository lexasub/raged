"""Progress reporting for long indexing phases (issue #5)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from ast_rag.cli import _index_progress
from ast_rag.models import ASTNode, NodeKind, Language
from ast_rag.services.embedding_manager import EmbeddingManager


def _node(i: int) -> ASTNode:
    return ASTNode(
        id=f"{i:024x}",
        kind=NodeKind.FUNCTION,
        name=f"fn_{i}",
        qualified_name=f"mod.fn_{i}",
        file_path="mod.py",
        start_line=i,
        end_line=i + 1,
        start_byte=i * 10,
        end_byte=i * 10 + 9,
        lang=Language.PYTHON,
    )


def _manager() -> EmbeddingManager:
    qcfg, ecfg = MagicMock(), MagicMock()
    qcfg.collection_name = "test"
    ecfg.hybrid_search = False  # otherwise __init__ compares MagicMock weights
    em = EmbeddingManager(qcfg, ecfg)
    em._get_client = MagicMock(return_value=MagicMock())
    em._encode = lambda texts: np.zeros((len(texts), 4), dtype=np.float32)
    return em


def test_progress_callback_reports_monotonic_completion():
    nodes = [_node(i) for i in range(10)]
    seen: list[tuple[int, int]] = []

    with patch("ast_rag.services.embedding_manager._node_to_payload", return_value={}):
        count = _manager().build_embeddings(
            nodes, batch_size=4, progress_callback=lambda d, t: seen.append((d, t))
        )

    assert count == 10
    assert seen, "progress_callback was never invoked"
    assert [d for d, _ in seen] == sorted(d for d, _ in seen), "completion went backwards"
    assert seen[-1] == (10, 10), f"final callback should report completion, got {seen[-1]}"
    assert all(t == 10 for _, t in seen), "total changed mid-run"


def test_progress_callback_is_optional():
    """Omitting the callback must preserve the original behaviour."""
    nodes = [_node(i) for i in range(3)]
    with patch("ast_rag.services.embedding_manager._node_to_payload", return_value={}):
        assert _manager().build_embeddings(nodes, batch_size=2) == 3


def test_progress_callback_fires_on_empty_input():
    seen: list[tuple[int, int]] = []
    assert _manager().build_embeddings([], progress_callback=lambda d, t: seen.append((d, t))) == 0
    assert seen == [(0, 0)], "callers need a terminal update even when there is nothing to embed"


def test_index_progress_reports_counts_and_eta():
    progress = _index_progress()
    columns = {type(c).__name__ for c in progress.columns}
    assert "BarColumn" in columns
    assert "MofNCompleteColumn" in columns, "issue #5 asks for a file count"
    assert "TimeRemainingColumn" in columns
    assert progress.live.transient
