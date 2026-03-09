"""
test_ast_cache.py - Tests for ParseCache, LazyTree, SQLiteParseCache, and
ParserManager cache integration.

Structure
---------
TestLazyTree                 - LazyTree unit tests (no tree-sitter, no DB)
TestParseCacheUnit           - ParseCache in complete isolation
TestSQLiteParseCache         - SQLiteParseCache persistence and interface parity
TestParserManagerIntegration - ParserManager delegating to ParseCache (in-memory)
TestParserManagerSQLite      - ParserManager delegating to SQLiteParseCache
TestWorkerResolve            - parse_file(resolve=True) returns a plain Tree
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ast_rag.parse_cache import LazyTree, ParseCache, SQLiteParseCache
from ast_rag.ast_parser import ParserManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmp(suffix: str, content: bytes) -> str:
    fh = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="wb")
    fh.write(content)
    fh.close()
    return fh.name


def _tmp_db() -> str:
    fh = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    fh.close()
    os.unlink(fh.name)  # SQLiteParseCache will create it
    return fh.name


def _make_sentinel_tree():
    """Return a mock object that quacks like a tree-sitter Tree."""
    mock = MagicMock()
    mock.root_node = MagicMock()
    return mock


# ===========================================================================
# Unit tests: LazyTree
# ===========================================================================

class TestLazyTree:
    """Test LazyTree without involving any real parsing or storage."""

    def test_loader_not_called_until_access(self):
        called = []
        sentinel = _make_sentinel_tree()

        def loader():
            called.append(1)
            return sentinel

        lazy = LazyTree(loader)
        assert called == [], "loader must not be called at construction"

    def test_first_attribute_access_triggers_load(self):
        sentinel = _make_sentinel_tree()
        lazy = LazyTree(loader=lambda: sentinel)
        _ = lazy.root_node  # triggers load
        assert object.__getattribute__(lazy, "_tree") is sentinel

    def test_loader_called_exactly_once(self):
        call_count = [0]
        sentinel = _make_sentinel_tree()

        def loader():
            call_count[0] += 1
            return sentinel

        lazy = LazyTree(loader)
        _ = lazy.root_node
        _ = lazy.root_node
        _ = lazy.root_node
        assert call_count[0] == 1, "loader must be invoked exactly once"

    def test_attribute_delegation(self):
        sentinel = _make_sentinel_tree()
        sentinel.special_attr = 42
        lazy = LazyTree(loader=lambda: sentinel)
        assert lazy.special_attr == 42

    def test_resolve_returns_plain_tree(self):
        sentinel = _make_sentinel_tree()
        lazy = LazyTree(loader=lambda: sentinel)
        result = lazy.resolve()
        assert result is sentinel

    def test_pre_populated_tree_not_re_loaded(self):
        call_count = [0]
        sentinel = _make_sentinel_tree()

        def loader():
            call_count[0] += 1
            return sentinel

        lazy = LazyTree(loader)
        # Simulate what ParseCache.put() does: pre-populate _tree.
        object.__setattr__(lazy, "_tree", sentinel)
        _ = lazy.root_node  # should NOT call loader
        assert call_count[0] == 0, "loader must not be called when _tree is pre-set"

    def test_repr_unresolved(self):
        lazy = LazyTree(loader=lambda: _make_sentinel_tree())
        assert "unresolved" in repr(lazy)

    def test_repr_resolved(self):
        sentinel = _make_sentinel_tree()
        lazy = LazyTree(loader=lambda: sentinel)
        lazy.resolve()
        assert "unresolved" not in repr(lazy)


# ===========================================================================
# Unit tests: ParseCache in complete isolation (no ParserManager, no tree-sitter)
# ===========================================================================

class TestParseCacheUnit:
    """Test ParseCache directly without involving any parsing."""

    def test_get_returns_none_on_empty_cache(self):
        cache = ParseCache()
        assert cache.get("/some/file.py", b"x=1") is None

    def test_put_then_get_same_source_returns_lazy(self):
        cache = ParseCache()
        sentinel = _make_sentinel_tree()
        cache.put("/f.py", b"x=1", sentinel)
        lazy = cache.get("/f.py", b"x=1")
        assert lazy is not None
        # Pre-loaded — resolve() must return the sentinel without re-parsing
        assert lazy.resolve() is sentinel

    def test_get_returns_same_lazytre_instance(self):
        """In-memory backend must return the same LazyTree instance on repeated hits."""
        cache = ParseCache()
        sentinel = _make_sentinel_tree()
        cache.put("/f.py", b"x=1", sentinel)
        lazy1 = cache.get("/f.py", b"x=1")
        lazy2 = cache.get("/f.py", b"x=1")
        assert lazy1 is lazy2

    def test_get_returns_none_when_hash_mismatch(self):
        cache = ParseCache()
        sentinel = _make_sentinel_tree()
        cache.put("/f.py", b"x=1", sentinel)
        assert cache.get("/f.py", b"x=999") is None  # different source

    def test_put_overwrites_stale_entry(self):
        cache = ParseCache()
        old = _make_sentinel_tree()
        new = _make_sentinel_tree()
        cache.put("/f.py", b"old", old)
        cache.put("/f.py", b"new", new)
        lazy = cache.get("/f.py", b"new")
        assert lazy is not None
        assert lazy.resolve() is new

    def test_loader_kwarg_is_ignored_by_in_memory_backend(self):
        """In-memory backend must ignore loader= and return the stored tree."""
        cache = ParseCache()
        sentinel = _make_sentinel_tree()
        cache.put("/f.py", b"src", sentinel)
        called = []
        loader = lambda: called.append(1) or _make_sentinel_tree()
        lazy = cache.get("/f.py", b"src", loader=loader)
        assert lazy is not None
        _ = lazy.resolve()
        assert called == [], "in-memory backend must not call the provided loader"

    def test_evict_removes_entry(self):
        cache = ParseCache()
        cache.put("/f.py", b"x", _make_sentinel_tree())
        cache.evict("/f.py")
        assert cache.get("/f.py", b"x") is None

    def test_evict_nonexistent_key_is_safe(self):
        cache = ParseCache()
        cache.evict("/doesnt/exist.py")  # should not raise

    def test_clear_empties_store(self):
        cache = ParseCache()
        cache.put("/a.py", b"a", _make_sentinel_tree())
        cache.put("/b.py", b"b", _make_sentinel_tree())
        cache.clear()
        assert cache.stats()["size"] == 0

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
        cache.put("/f.py", b"src", _make_sentinel_tree())
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
# Unit tests: SQLiteParseCache
# ===========================================================================

class TestSQLiteParseCache:
    """Test SQLiteParseCache — persistence, interface parity, and observability."""

    def test_miss_on_empty_db(self):
        db = _tmp_db()
        try:
            cache = SQLiteParseCache(db)
            sentinel = _make_sentinel_tree()
            loader = lambda: sentinel
            assert cache.get("/f.py", b"src", loader=loader) is None
        finally:
            if os.path.exists(db):
                os.unlink(db)

    def test_put_then_get_returns_lazytree(self):
        db = _tmp_db()
        try:
            cache = SQLiteParseCache(db)
            sentinel = _make_sentinel_tree()
            cache.put("/f.py", b"src", sentinel)
            lazy = cache.get("/f.py", b"src", loader=lambda: sentinel)
            assert lazy is not None
        finally:
            if os.path.exists(db):
                os.unlink(db)

    def test_lazytree_from_sqlite_resolves_via_loader(self):
        db = _tmp_db()
        try:
            cache = SQLiteParseCache(db)
            sentinel = _make_sentinel_tree()
            cache.put("/f.py", b"src", sentinel)

            resolved = []
            def loader():
                resolved.append(1)
                return sentinel

            lazy = cache.get("/f.py", b"src", loader=loader)
            assert lazy is not None
            result = lazy.resolve()
            assert result is sentinel
            assert resolved == [1], "SQLite backend must use the provided loader"
        finally:
            if os.path.exists(db):
                os.unlink(db)

    def test_miss_on_hash_mismatch(self):
        db = _tmp_db()
        try:
            cache = SQLiteParseCache(db)
            cache.put("/f.py", b"original", _make_sentinel_tree())
            assert cache.get("/f.py", b"different", loader=lambda: None) is None
        finally:
            if os.path.exists(db):
                os.unlink(db)

    def test_persistence_across_instances(self):
        """A second SQLiteParseCache on the same DB must find entries from the first."""
        db = _tmp_db()
        sentinel = _make_sentinel_tree()
        try:
            cache1 = SQLiteParseCache(db)
            cache1.put("/f.py", b"src", sentinel)
            cache1.close()

            cache2 = SQLiteParseCache(db)
            lazy = cache2.get("/f.py", b"src", loader=lambda: sentinel)
            assert lazy is not None, "entry must persist across process restarts"
            cache2.close()
        finally:
            if os.path.exists(db):
                os.unlink(db)

    def test_evict_removes_entry(self):
        db = _tmp_db()
        try:
            cache = SQLiteParseCache(db)
            cache.put("/f.py", b"src", _make_sentinel_tree())
            cache.evict("/f.py")
            assert cache.get("/f.py", b"src", loader=lambda: None) is None
        finally:
            if os.path.exists(db):
                os.unlink(db)

    def test_clear_removes_all(self):
        db = _tmp_db()
        try:
            cache = SQLiteParseCache(db)
            cache.put("/a.py", b"a", _make_sentinel_tree())
            cache.put("/b.py", b"b", _make_sentinel_tree())
            cache.clear()
            assert cache.stats()["size"] == 0
        finally:
            if os.path.exists(db):
                os.unlink(db)

    def test_stats_structure(self):
        db = _tmp_db()
        try:
            cache = SQLiteParseCache(db)
            s = cache.stats()
            assert {"size", "hits", "misses", "hit_rate", "db_path"}.issubset(s.keys())
            assert s["db_path"] == db
        finally:
            if os.path.exists(db):
                os.unlink(db)

    def test_stats_hit_miss_counts(self):
        db = _tmp_db()
        try:
            sentinel = _make_sentinel_tree()
            cache = SQLiteParseCache(db)
            cache.put("/f.py", b"src", sentinel)
            cache.get("/f.py", b"src", loader=lambda: sentinel)        # HIT
            cache.get("/f.py", b"different", loader=lambda: sentinel)  # MISS
            s = cache.stats()
            assert s["hits"] == 1
            assert s["misses"] == 1
        finally:
            if os.path.exists(db):
                os.unlink(db)


# ===========================================================================
# Integration tests: ParserManager with in-memory ParseCache
# ===========================================================================

class TestParserManagerIntegration:
    """Verify ParserManager delegates to ParseCache (no inline cache logic)."""

    def test_parse_file_hit_returns_same_tree(self):
        path = _tmp(".py", b"def hello(): pass\n")
        try:
            pm = ParserManager()
            t1 = pm.parse_file(path)
            t2 = pm.parse_file(path)
            # t1 is a plain Tree (cold miss); t2 is a LazyTree proxy wrapping
            # the same underlying Tree.  Compare via resolve() for identity.
            assert t1 is (t2.resolve() if isinstance(t2, LazyTree) else t2)
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
            # Plain call now should return the refreshed entry (t2).
            t3 = pm.parse_file(path)
            # t2 is a plain Tree (incremental parse); t3 is a LazyTree wrapping it.
            t2_raw = t2.resolve() if isinstance(t2, LazyTree) else t2
            t3_raw = t3.resolve() if isinstance(t3, LazyTree) else t3
            assert t3_raw is t2_raw
        finally:
            os.unlink(path)

    def test_source_supplied_by_caller_is_cached(self):
        path = _tmp(".py", b"pass\n")
        try:
            pm = ParserManager()
            src = b"pass\n"
            t1 = pm.parse_file(path, source=src)
            t2 = pm.parse_file(path, source=src)
            # t1 is plain Tree (cold miss); t2 is LazyTree wrapping the same Tree.
            t1_raw = t1.resolve() if isinstance(t1, LazyTree) else t1
            t2_raw = t2.resolve() if isinstance(t2, LazyTree) else t2
            assert t1_raw is t2_raw
        finally:
            os.unlink(path)


# ===========================================================================
# Integration tests: ParserManager with SQLiteParseCache
# ===========================================================================

class TestParserManagerSQLite:
    """ParserManager with SQLite backend — persistence across two instances."""

    def test_factory_creates_sqlite_cache_from_config(self):
        db = _tmp_db()
        try:
            pm = ParserManager(config={"parse_cache": {
                "persistence_enabled": True,
                "db_path": db,
            }})
            assert isinstance(pm._cache, SQLiteParseCache)
        finally:
            if os.path.exists(db):
                os.unlink(db)

    def test_factory_creates_memory_cache_by_default(self):
        pm = ParserManager()
        assert isinstance(pm._cache, ParseCache)

    def test_sqlite_hit_after_restart(self):
        """Second ParserManager on the same DB must benefit from the first run."""
        db = _tmp_db()
        path = _tmp(".py", b"x = 42\n")
        config = {"parse_cache": {"persistence_enabled": True, "db_path": db}}
        try:
            pm1 = ParserManager(config=config)
            t1 = pm1.parse_file(path)
            assert t1 is not None
            # "Restart" — create a fresh ParserManager pointing at the same DB.
            pm2 = ParserManager(config=config)
            stats_before = pm2.tree_cache_stats()
            t2 = pm2.parse_file(path)
            stats_after = pm2.tree_cache_stats()
            assert t2 is not None
            assert stats_after["hits"] == stats_before["hits"] + 1, (
                "second ParserManager must hit the SQLite cache"
            )
        finally:
            os.unlink(path)
            if os.path.exists(db):
                os.unlink(db)


# ===========================================================================
# Worker resolve tests: parse_file(resolve=True)
# ===========================================================================

class TestWorkerResolve:
    """parse_file(resolve=True) must return a plain Tree, not a LazyTree.

    Worker processes in ProcessPoolExecutor cannot pickle lambdas, so any
    LazyTree with a deferred loader would fail. resolve=True forces eager
    resolution before the result crosses process boundaries.
    """

    def test_resolve_true_returns_non_lazytree(self):
        path = _tmp(".py", b"def foo(): pass\n")
        try:
            pm = ParserManager()
            # First call — cold miss, populates cache.
            pm.parse_file(path)
            # Second call — should hit cache; resolve=True forces plain Tree.
            result = pm.parse_file(path, resolve=True)
            assert not isinstance(result, LazyTree), (
                "resolve=True must return a plain Tree, not a LazyTree"
            )
        finally:
            os.unlink(path)

    def test_resolve_true_result_is_not_none(self):
        path = _tmp(".py", b"pass\n")
        try:
            pm = ParserManager()
            tree = pm.parse_file(path, resolve=True)
            assert tree is not None
        finally:
            os.unlink(path)
