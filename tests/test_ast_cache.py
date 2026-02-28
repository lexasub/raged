"""
test_ast_cache.py - Tests for ParseCache (unit) and ParserManager cache integration.

Structure:
    TestParseCacheUnit          - ParseCache in isolation (pure unit tests)
    TestParserManagerIntegration - ParserManager delegating to ParseCache
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ast_rag.parse_cache import ParseCache
from ast_rag.ast_parser import ParserManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmp(suffix: str, content: bytes) -> str:
    fh = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="wb")
    fh.write(content)
    fh.close()
    return fh.name


# ===========================================================================
# Unit tests: ParseCache in complete isolation (no ParserManager, no tree-sitter)
# ===========================================================================

class TestParseCacheUnit:
    """Test ParseCache directly without involving any parsing."""

    def test_get_returns_none_on_empty_cache(self):
        cache = ParseCache()
        assert cache.get("/some/file.py", b"x=1") is None

    def test_put_then_get_same_source_returns_value(self):
        cache = ParseCache()
        sentinel = object()          # stand-in for a Tree
        cache._store["/f.py"] = (ParseCache.hash_source(b"x=1"), sentinel)
        result = cache.get("/f.py", b"x=1")
        assert result is sentinel

    def test_get_returns_none_when_hash_mismatch(self):
        cache = ParseCache()
        sentinel = object()
        cache._store["/f.py"] = (ParseCache.hash_source(b"x=1"), sentinel)
        # Different source bytes → stale hash
        assert cache.get("/f.py", b"x=999") is None

    def test_put_overwrites_stale_entry(self):
        cache = ParseCache()
        old = object()
        new = object()
        cache._store["/f.py"] = (ParseCache.hash_source(b"old"), old)
        cache.put("/f.py", b"new", new)
        result = cache.get("/f.py", b"new")
        assert result is new

    def test_evict_removes_entry(self):
        cache = ParseCache()
        sentinel = object()
        cache._store["/f.py"] = (ParseCache.hash_source(b"x"), sentinel)
        cache.evict("/f.py")
        assert cache.get("/f.py", b"x") is None

    def test_evict_nonexistent_key_is_safe(self):
        cache = ParseCache()
        cache.evict("/doesnt/exist.py")   # should not raise

    def test_clear_empties_store(self):
        cache = ParseCache()
        cache._store["/a.py"] = (ParseCache.hash_source(b"a"), object())
        cache._store["/b.py"] = (ParseCache.hash_source(b"b"), object())
        cache.clear()
        assert len(cache._store) == 0

    def test_stats_structure_on_fresh_cache(self):
        cache = ParseCache()
        s = cache.stats()
        assert set(s.keys()) == {"size", "hits", "misses", "hit_rate"}
        assert s["size"] == 0
        assert s["hits"] == 0
        assert s["misses"] == 0
        assert s["hit_rate"] == 0.0

    def test_stats_hit_rate_calculation(self):
        cache = ParseCache()
        sentinel = object()
        cache._store["/f.py"] = (ParseCache.hash_source(b"src"), sentinel)
        cache.get("/f.py", b"src")   # HIT
        cache.get("/f.py", b"other") # MISS (hash mismatch)
        s = cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["hit_rate"] == pytest.approx(0.5)

    def test_hash_source_is_deterministic(self):
        h1 = ParseCache.hash_source(b"hello")
        h2 = ParseCache.hash_source(b"hello")
        assert h1 == h2

    def test_hash_source_differs_on_different_input(self):
        assert ParseCache.hash_source(b"a") != ParseCache.hash_source(b"b")


# ===========================================================================
# Integration tests: ParserManager delegating correctly to ParseCache
# ===========================================================================

class TestParserManagerIntegration:
    """Verify ParserManager delegates to ParseCache (no inline cache logic)."""

    def test_parse_file_hit_returns_same_tree(self):
        path = _tmp(".py", b"def hello(): pass\n")
        try:
            pm = ParserManager()
            t1 = pm.parse_file(path)
            t2 = pm.parse_file(path)
            assert t1 is t2
            assert pm.tree_cache_stats()["hits"] == 1
        finally:
            os.unlink(path)

    def test_parse_file_miss_on_content_change(self):
        path = _tmp(".py", b"x = 1\n")
        try:
            pm = ParserManager()
            t1 = pm.parse_file(path)
            with open(path, "wb") as f:
                f.write(b"x = 99\n")
            t2 = pm.parse_file(path)
            assert t1 is not t2
            assert pm.tree_cache_stats()["misses"] == 2
        finally:
            os.unlink(path)

    def test_custom_cache_injected_via_constructor(self):
        """ParserManager should use a caller-supplied ParseCache instance."""
        custom_cache = ParseCache()
        pm = ParserManager(cache=custom_cache)
        assert pm._cache is custom_cache

    def test_clear_tree_cache_delegates(self):
        path = _tmp(".py", b"pass\n")
        try:
            pm = ParserManager()
            pm.parse_file(path)
            assert pm.tree_cache_stats()["size"] == 1
            pm.clear_tree_cache()
            assert pm.tree_cache_stats()["size"] == 0
        finally:
            os.unlink(path)

    def test_tree_cache_stats_delegates(self):
        pm = ParserManager()
        stats = pm.tree_cache_stats()
        assert "size" in stats and "hits" in stats

    def test_incremental_parse_refreshes_cache(self):
        path = _tmp(".py", b"x = 1\n")
        try:
            pm = ParserManager()
            t1 = pm.parse_file(path)
            t2 = pm.parse_file(path, old_tree=t1)  # incremental
            # plain call now should return the refreshed entry (t2)
            t3 = pm.parse_file(path)
            assert t3 is t2
        finally:
            os.unlink(path)

    def test_source_supplied_by_caller_is_cached(self):
        path = _tmp(".py", b"pass\n")
        try:
            pm = ParserManager()
            src = b"pass\n"
            t1 = pm.parse_file(path, source=src)
            t2 = pm.parse_file(path, source=src)
            assert t1 is t2
        finally:
            os.unlink(path)
