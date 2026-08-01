"""Thread-safety tests for ParserManager and the parse cache backends.

Motivation
----------
``WorkspaceWatcher`` builds its ``ParserManager`` on the main thread but parses
from ``threading.Timer`` callbacks — a *new* thread per debounce cycle (see
``ast_rag/services/watcher_service.py``). So a single manager and its cache are
already reached from several threads during a watch session.

Before this change that meant one shared ``tree_sitter.Parser`` (which holds
mutable state across a parse) and unsynchronised cache bookkeeping. These tests
pin down the fixes.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from ast_rag.services.parsing.parser_manager import ParserManager
from ast_rag.utils.bounded_ast_cache import BoundedParseCache
from ast_rag.utils.parse_cache import LazyTree, ParseCache, SQLiteParseCache

PY_TEMPLATE = '''
class Widget{n}:
    """Docstring for Widget{n}."""

    def __init__(self, value):
        self.value = value

    def compute(self, other):
        if other > 0:
            return self.value + other
        return helper_{n}(self.value)


def helper_{n}(value):
    return value * 2
'''


@pytest.fixture
def source_files(tmp_path):
    """Enough files that thread interleaving actually varies between runs."""
    files = []
    for i in range(40):
        p = tmp_path / f"module_{i:03d}.py"
        p.write_text(PY_TEMPLATE.format(n=i))
        files.append(str(p))
    return files


def _parse_all(pm: ParserManager, files: list[str], workers: int) -> list:
    def job(path: str):
        tree = pm.parse_file(path)
        if tree is None:
            return []
        with open(path, "rb") as fh:
            source = fh.read()
        return pm.extract_nodes(tree, path, "python", source, "TEST")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(job, files))


# ---------------------------------------------------------------------------
# Parser isolation
# ---------------------------------------------------------------------------


def test_each_thread_gets_its_own_parser():
    """A tree_sitter.Parser must never be shared between threads."""
    pm = ParserManager(cache=ParseCache())
    seen: dict[int, int] = {}
    lock = threading.Lock()

    def grab() -> None:
        parser = pm._get_parser("python")
        with lock:
            seen[threading.get_ident()] = id(parser)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda _: grab(), range(24)))

    assert len(set(seen.values())) == len(seen)


def test_same_thread_reuses_its_parser():
    """Thread-local caching must not allocate a fresh parser per call."""
    pm = ParserManager(cache=ParseCache())
    assert pm._get_parser("python") is pm._get_parser("python")


def test_constructing_thread_reuses_prebuilt_parsers():
    """Single-threaded callers keep the parsers built in _init_languages."""
    pm = ParserManager(cache=ParseCache())
    assert pm._get_parser("python") is pm._parsers["python"]


# ---------------------------------------------------------------------------
# Correctness under concurrency
# ---------------------------------------------------------------------------


def test_concurrent_parsing_matches_sequential(source_files):
    """Threading must not change what gets extracted."""
    sequential = _parse_all(ParserManager(cache=ParseCache()), source_files, workers=1)
    concurrent = _parse_all(ParserManager(cache=ParseCache()), source_files, workers=8)

    assert [[n.id for n in f] for f in sequential] == [[n.id for n in f] for f in concurrent]
    assert all(nodes for nodes in concurrent)


@pytest.mark.parametrize(
    "cache_factory", [ParseCache, BoundedParseCache], ids=["in-memory", "bounded-lru"]
)
def test_cache_counters_stay_coherent(source_files, cache_factory):
    """Hits + misses must equal the number of lookups, with no lost updates."""
    pm = ParserManager(cache=cache_factory())
    _parse_all(pm, source_files * 2, workers=8)

    stats = pm.tree_cache_stats()
    assert stats["hits"] > 0
    assert stats["hits"] + stats["misses"] == len(source_files) * 2


def test_sqlite_backend_tolerates_concurrent_access(source_files, tmp_path):
    """One sqlite connection serves every thread; the lock must serialise it."""
    cache = SQLiteParseCache(str(tmp_path / "cache.sqlite"))
    pm = ParserManager(cache=cache)
    try:
        results = _parse_all(pm, source_files * 2, workers=8)
        assert all(nodes for nodes in results)
        assert pm.tree_cache_stats()["hits"] > 0
    finally:
        cache.close()


# ---------------------------------------------------------------------------
# Specific defects fixed
# ---------------------------------------------------------------------------


def test_lazy_tree_loader_runs_exactly_once_under_contention():
    """LazyTree._ensure claims 'exactly once' — hold it to that under a race."""
    calls: list[int] = []
    calls_lock = threading.Lock()
    # Line the threads up *before* resolve() so they enter _ensure together.
    # The barrier must not live inside the loader: only one thread reaches it,
    # so waiting there would deadlock the rest.
    start = threading.Barrier(8)

    def loader():
        with calls_lock:
            calls.append(1)
        return object()

    lazy = LazyTree(loader=loader)

    def worker(_):
        start.wait()
        return lazy.resolve()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(8)))

    assert len(calls) == 1


def test_bounded_cache_hashes_are_dropped_on_eviction(source_files):
    """_hashes previously grew forever: the LRU evicted trees but not hashes."""
    cache = BoundedParseCache(max_entries=5)
    pm = ParserManager(cache=cache)
    _parse_all(pm, source_files, workers=4)

    assert len(cache._hashes) <= 5
    assert set(cache._hashes) <= set(cache._inner._cache)
