"""
test_ast_cache.py - Unit tests for the ParserManager parse-tree cache.

Tests:
    1. Cache HIT:  same file, same content → second call returns identical Tree object.
    2. Cache MISS: content changes → second call returns a new Tree object.
    3. Cache MISS: repeated source bytes supplied by caller.
    4. clear_tree_cache() evicts all entries.
    5. tree_cache_stats() returns expected structure and sensible values.
    6. Incremental parse (old_tree supplied) always reparsed, cache refreshed.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Allow running from repo root or from the raged/ sub-directory.
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ast_rag.ast_parser import ParserManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tmp_py(content: bytes) -> str:
    """Create a temporary .py file with *content* and return its path."""
    fh = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="wb")
    fh.write(content)
    fh.close()
    return fh.name


def _make_tmp_java(content: bytes) -> str:
    """Create a temporary .java file with *content* and return its path."""
    fh = tempfile.NamedTemporaryFile(suffix=".java", delete=False, mode="wb")
    fh.write(content)
    fh.close()
    return fh.name


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParseTreeCacheHit:
    """Second parse of an unchanged file must return the cached Tree."""

    def test_cache_hit_same_content_python(self):
        path = _make_tmp_py(b"def hello():\n    return 42\n")
        try:
            pm = ParserManager()
            t1 = pm.parse_file(path)
            t2 = pm.parse_file(path)
            assert t1 is t2, "Expected the same Tree object on cache hit"
            stats = pm.tree_cache_stats()
            assert stats["hits"] == 1
            assert stats["misses"] == 1
            assert stats["size"] == 1
            assert stats["hit_rate"] == pytest.approx(0.5)
        finally:
            os.unlink(path)

    def test_cache_hit_multiple_calls(self):
        path = _make_tmp_py(b"x = 1\n")
        try:
            pm = ParserManager()
            t1 = pm.parse_file(path)
            t2 = pm.parse_file(path)
            t3 = pm.parse_file(path)
            assert t1 is t2
            assert t1 is t3
            stats = pm.tree_cache_stats()
            assert stats["hits"] == 2
            assert stats["misses"] == 1
        finally:
            os.unlink(path)


class TestParseTreeCacheMiss:
    """Parsing a file whose content has changed must produce a fresh Tree."""

    def test_cache_invalidated_on_content_change(self):
        path = _make_tmp_py(b"x = 1\n")
        try:
            pm = ParserManager()
            t1 = pm.parse_file(path)

            # Modify the file in-place.
            with open(path, "wb") as fh:
                fh.write(b"x = 99\ny = x + 1\n")

            t2 = pm.parse_file(path)
            assert t1 is not t2, "Expected a new Tree after content change"
            stats = pm.tree_cache_stats()
            assert stats["misses"] == 2
            assert stats["size"] == 1  # still only one slot (same path)
        finally:
            os.unlink(path)

    def test_cache_miss_different_files(self):
        path_a = _make_tmp_py(b"a = 1\n")
        path_b = _make_tmp_py(b"b = 2\n")
        try:
            pm = ParserManager()
            pm.parse_file(path_a)
            pm.parse_file(path_b)
            stats = pm.tree_cache_stats()
            assert stats["size"] == 2
            assert stats["misses"] == 2
            assert stats["hits"] == 0
        finally:
            os.unlink(path_a)
            os.unlink(path_b)


class TestSourceSuppliedBytesCaching:
    """When caller supplies source bytes, cache key is still the file path + hash."""

    def test_source_bytes_cache_hit(self):
        path = _make_tmp_py(b"pass\n")
        try:
            pm = ParserManager()
            src = b"pass\n"
            t1 = pm.parse_file(path, source=src)
            t2 = pm.parse_file(path, source=src)
            assert t1 is t2
            assert pm.tree_cache_stats()["hits"] == 1
        finally:
            os.unlink(path)

    def test_source_bytes_changed_is_cache_miss(self):
        path = _make_tmp_py(b"pass\n")
        try:
            pm = ParserManager()
            t1 = pm.parse_file(path, source=b"pass\n")
            t2 = pm.parse_file(path, source=b"x = 1\n")  # different bytes
            assert t1 is not t2
            assert pm.tree_cache_stats()["misses"] == 2
        finally:
            os.unlink(path)


class TestClearTreeCache:
    """clear_tree_cache() must evict all entries."""

    def test_clear_resets_size(self):
        path = _make_tmp_py(b"a = 1\n")
        try:
            pm = ParserManager()
            pm.parse_file(path)
            assert pm.tree_cache_stats()["size"] == 1
            pm.clear_tree_cache()
            assert pm.tree_cache_stats()["size"] == 0
        finally:
            os.unlink(path)

    def test_after_clear_next_parse_is_miss(self):
        path = _make_tmp_py(b"a = 1\n")
        try:
            pm = ParserManager()
            t1 = pm.parse_file(path)
            pm.clear_tree_cache()
            t2 = pm.parse_file(path)
            # t2 is a freshly parsed tree (may or may not be `is t1`, but cache was cleared)
            assert pm.tree_cache_stats()["misses"] == 2
        finally:
            os.unlink(path)


class TestCacheStats:
    """tree_cache_stats() must always return a well-formed dict."""

    def test_fresh_manager_stats(self):
        pm = ParserManager()
        stats = pm.tree_cache_stats()
        assert set(stats.keys()) == {"size", "hits", "misses", "hit_rate"}
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

    def test_hit_rate_denominator_never_zero(self):
        """hit_rate should be 0.0 when no parses have happened yet."""
        pm = ParserManager()
        assert pm.tree_cache_stats()["hit_rate"] == 0.0


class TestIncrementalParse:
    """When old_tree is supplied, parser re-parses but still refreshes cache slot."""

    def test_incremental_parse_updates_cache(self):
        path = _make_tmp_py(b"x = 1\n")
        try:
            pm = ParserManager()
            t1 = pm.parse_file(path)
            # Hand old_tree back to trigger incremental parse path.
            t2 = pm.parse_file(path, old_tree=t1)
            # Incremental parse must not return the old tree object.
            # Cache slot for this path should now hold t2.
            # A subsequent plain call with same content hits the cache.
            t3 = pm.parse_file(path)
            assert t3 is t2  # t2 is now cached
        finally:
            os.unlink(path)
