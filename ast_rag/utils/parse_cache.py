"""
parse_cache.py - In-memory and SQLite parse-tree cache for ParserManager.

Architecture
------------
                ParserManager  (owns parsers, provides loader=)
                      │
                      │  loader: Callable[[], Tree]
                      ▼
          ParseCache / SQLiteParseCache  (storage only, tree-sitter agnostic)
                      │
                      │  returns LazyTree | None
                      ▼
                  LazyTree  (defers tree loading until first attribute access)

Classes
-------
LazyTree          - Thin proxy; loads on first attribute access via a caller-supplied
                    loader callable.  In-memory backend pre-populates _tree so no
                    re-parse ever happens.  SQLite backend defers via loader.
                    Call .resolve() before crossing process/pickle boundaries.

ParseCache        - Default in-memory backend.  Content-addressed (SHA-256) dict.
                    Returns the *same* pre-loaded LazyTree instance on cache hit
                    (zero re-parses, zero redundant tree-sitter calls).

SQLiteParseCache  - Persistent backend.  Stores (file_path, content_hash,
                    source_bytes) in a local SQLite DB.  On hit, hands back a
                    LazyTree(loader) so the tree is reconstructed lazily from the
                    stored bytes — no parsing until first attribute access.

Swapping backends
-----------------
Both backends expose the same interface:
    get(abs_path, source, loader=None) -> Optional[LazyTree]
    put(abs_path, source, tree)        -> None
    evict(abs_path)                    -> None
    clear()                            -> None
    stats()                            -> dict

Pass a different backend to ParserManager(cache=...) or set
parse_cache.persistence_enabled = true in ast_rag_config.json to have
ParserManager's factory pick SQLiteParseCache automatically.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import time
from typing import Any, Callable, Optional

from tree_sitter import Tree

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LazyTree
# ---------------------------------------------------------------------------

class LazyTree:
    """Thin proxy that defers tree loading until first attribute access.

    Both in-memory and SQLite backends return ``LazyTree | None``.

    In-memory backend
        ``put()`` pre-populates ``_tree`` immediately — no re-parse on access,
        ``get()`` returns the **same resolved instance** so callers share it.

    SQLite backend
        ``_tree`` starts as ``None``; the caller-supplied ``loader`` is invoked
        once on first ``__getattr__`` access and the result is cached in
        ``_tree`` for all subsequent accesses.

    Multiprocessing / pickling
        Python lambdas are not picklable.  Worker processes launched via
        ``ProcessPoolExecutor`` must call ``parse_file(resolve=True)`` so they
        receive a plain ``Tree`` object rather than a ``LazyTree``.

    Usage::

        lazy = cache.get(path, source, loader=lambda: parser.parse(source))
        if lazy is not None:
            root = lazy.root_node    # triggers load on first access (SQLite)
            children = root.children # subsequent accesses free (cached _tree)

        # Force resolution before pickling (worker processes):
        tree: Tree = lazy.resolve()
    """

    def __init__(self, loader: Callable[[], Tree]) -> None:
        # Use object.__setattr__ to avoid triggering __getattr__ during init.
        object.__setattr__(self, "_loader", loader)
        object.__setattr__(self, "_tree", None)
        # Content hash — set by ParseCache.put() so get() can do a fast
        # identity check without re-hashing on every access.
        object.__setattr__(self, "_hash", "")

    # ------------------------------------------------------------------
    # Core load / resolve
    # ------------------------------------------------------------------

    def _ensure(self) -> None:
        """Invoke the loader exactly once and cache the result."""
        if object.__getattribute__(self, "_tree") is None:
            tree = object.__getattribute__(self, "_loader")()
            object.__setattr__(self, "_tree", tree)

    def resolve(self) -> Tree:
        """Force eager resolution and return the underlying Tree.

        Call this before crossing process/pickle boundaries::

            tree = parse_file(path, resolve=True)   # returns plain Tree
        """
        self._ensure()
        return object.__getattribute__(self, "_tree")  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Transparent proxy
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Delegate all attribute access to the underlying Tree."""
        self._ensure()
        tree = object.__getattribute__(self, "_tree")
        return getattr(tree, name)

    def __repr__(self) -> str:
        tree = object.__getattribute__(self, "_tree")
        if tree is None:
            return "<LazyTree: unresolved>"
        return f"<LazyTree: {tree!r}>"


# ---------------------------------------------------------------------------
# ParseCache  (in-memory, default backend)
# ---------------------------------------------------------------------------

class ParseCache:
    """Content-addressed in-memory cache for tree-sitter parse trees.

    On a cache hit, the *same* pre-loaded ``LazyTree`` instance is returned so
    all callers share one ``Tree`` object — no re-parsing ever occurs within the
    same process lifetime.

    The ``loader`` parameter accepted by ``get()`` is included for interface
    parity with ``SQLiteParseCache`` but is intentionally **ignored** by this
    backend (the stored tree is already available).

    Usage::

        cache = ParseCache()
        lazy = cache.get(abs_path, source)
        if lazy is None:
            tree = parser.parse(source)
            cache.put(abs_path, source, tree)
            lazy = cache.get(abs_path, source)
        root = lazy.root_node
    """

    def __init__(self) -> None:
        # key: absolute file path → pre-loaded LazyTree
        self._store: dict[str, LazyTree] = {}
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @staticmethod
    def hash_source(source: bytes) -> str:
        """Return the SHA-256 hex digest of *source* bytes.

        Extracted here so all backends share one hashing implementation.
        """
        return hashlib.sha256(source).hexdigest()

    def get(
        self,
        abs_path: str,
        source: bytes,
        loader: Optional[Callable[[], Tree]] = None,  # ignored by in-memory backend
    ) -> Optional[LazyTree]:
        """Return the cached ``LazyTree`` if the stored hash matches *source*.

        Args:
            abs_path: Absolute path to the file (cache key).
            source:   Current source bytes of the file.
            loader:   Ignored — included for interface parity with SQLite backend.

        Returns:
            The **same** pre-loaded ``LazyTree`` instance on a hit, or ``None``
            on a miss / stale entry.
        """
        lazy = self._store.get(abs_path)
        if lazy is not None:
            stored_hash = object.__getattribute__(lazy, "_hash")
            if stored_hash == self.hash_source(source):
                self._hits += 1
                logger.debug("ParseCache HIT : %s", abs_path)
                return lazy  # same instance — _tree already populated

        self._misses += 1
        logger.debug("ParseCache MISS: %s", abs_path)
        return None

    def put(self, abs_path: str, source: bytes, tree: Tree) -> None:
        """Store (or refresh) a pre-loaded cache entry.

        The ``LazyTree`` is **eagerly resolved** here (``_tree`` set immediately)
        so that subsequent ``get()`` calls never trigger a re-parse.

        Args:
            abs_path: Absolute path to the file (cache key).
            source:   Source bytes the tree was parsed from.
            tree:     The freshly parsed Tree to cache.
        """
        content_hash = self.hash_source(source)
        # Create a no-op loader — tree is already available.
        lazy = LazyTree(loader=lambda t=tree: t)
        object.__setattr__(lazy, "_tree", tree)   # pre-populate → eager
        object.__setattr__(lazy, "_hash", content_hash)
        self._store[abs_path] = lazy
        logger.debug("ParseCache PUT : %s", abs_path)

    def evict(self, abs_path: str) -> None:
        """Remove a single entry (e.g. on file deletion).

        Args:
            abs_path: Absolute path of the file to evict.
        """
        removed = self._store.pop(abs_path, None)
        if removed is not None:
            logger.debug("ParseCache EVICT: %s", abs_path)

    def clear(self) -> None:
        """Evict *all* cached trees (e.g. after a full re-index)."""
        self._store.clear()
        logger.debug("ParseCache cleared")

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Snapshot of cache performance counters.

        Returns::

            {
                'size':     int,    # number of cached trees currently stored
                'hits':     int,    # cumulative hits since instantiation
                'misses':   int,    # cumulative misses since instantiation
                'hit_rate': float,  # hits / (hits + misses), or 0.0 if no ops
            }
        """
        total = self._hits + self._misses
        return {
            "size": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total else 0.0,
        }


# ---------------------------------------------------------------------------
# SQLiteParseCache  (persistent backend)
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS parse_cache (
    file_path      TEXT PRIMARY KEY,
    content_hash   TEXT NOT NULL,
    source_bytes   BLOB NOT NULL,
    last_accessed  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_last_accessed ON parse_cache(last_accessed);
"""


class SQLiteParseCache:
    """Persistent parse-cache backend backed by a local SQLite database.

    Stores ``(file_path, content_hash, source_bytes)`` so the cache survives
    process restarts.  On a hit, a ``LazyTree`` is returned that re-parses
    from the stored bytes only when first accessed — so unchanged files are
    never read from disk a second time, even across restarts.

    This backend is **tree-sitter agnostic**: it never calls a parser directly.
    The ``loader`` callable passed by ``ParserManager`` is responsible for the
    actual parse call.

    Args:
        db_path: Path for the SQLite file.  Created if it does not exist.
                 Default: ``".ast_rag_parse_cache.sqlite"`` next to the CWD.

    Usage::

        cache = SQLiteParseCache(".ast_rag_parse_cache.sqlite")
        pm = ParserManager(cache=cache)
    """

    def __init__(self, db_path: str = ".ast_rag_parse_cache.sqlite") -> None:
        self._db_path = db_path
        self._hits: int = 0
        self._misses: int = 0
        self._conn: sqlite3.Connection = sqlite3.connect(
            db_path, check_same_thread=False
        )
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.debug("SQLiteParseCache opened: %s", db_path)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @staticmethod
    def hash_source(source: bytes) -> str:
        """Return the SHA-256 hex digest of *source* bytes."""
        return hashlib.sha256(source).hexdigest()

    def get(
        self,
        abs_path: str,
        source: bytes,
        loader: Optional[Callable[[], Tree]] = None,
    ) -> Optional[LazyTree]:
        """Return a ``LazyTree`` if the stored hash matches *source*, else None.

        The ``LazyTree`` wraps *loader* so the tree is re-parsed from the stored
        bytes only when first accessed (lazy).  ``loader`` is provided by
        ``ParserManager`` and keeps ``SQLiteParseCache`` agnostic of tree-sitter.

        Args:
            abs_path: Absolute path to the file (cache key).
            source:   Current source bytes — used only for hash comparison.
            loader:   ``Callable[[], Tree]`` supplied by the caller.  Called at
                      most once when the returned ``LazyTree`` is first accessed.

        Returns:
            A ``LazyTree(loader)`` on a hit, or ``None`` on a miss / stale entry.
        """
        cur = self._conn.execute(
            "SELECT content_hash FROM parse_cache WHERE file_path = ?",
            (abs_path,),
        )
        row = cur.fetchone()
        if row is not None and row[0] == self.hash_source(source):
            # Update last_accessed timestamp for future LRU eviction.
            self._conn.execute(
                "UPDATE parse_cache SET last_accessed = ? WHERE file_path = ?",
                (time.time(), abs_path),
            )
            self._conn.commit()
            self._hits += 1
            logger.debug("SQLiteParseCache HIT : %s", abs_path)
            if loader is None:
                # No loader supplied — caller must handle resolution themselves.
                return None
            lazy = LazyTree(loader=loader)
            return lazy

        self._misses += 1
        logger.debug("SQLiteParseCache MISS: %s", abs_path)
        return None

    def put(self, abs_path: str, source: bytes, tree: Tree) -> None:  # noqa: ARG002
        """Persist ``(abs_path, content_hash, source_bytes)`` to SQLite.

        The *tree* parameter is accepted for interface parity but is **not**
        stored — trees are not directly serializable.  Instead, ``source_bytes``
        are stored so the tree can be re-parsed via the loader on the next load.

        Args:
            abs_path: Absolute path to the file (cache key).
            source:   Source bytes the tree was parsed from.
            tree:     The freshly parsed Tree (not stored; accepted for parity).
        """
        self._conn.execute(
            """
            INSERT INTO parse_cache (file_path, content_hash, source_bytes, last_accessed)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                content_hash  = excluded.content_hash,
                source_bytes  = excluded.source_bytes,
                last_accessed = excluded.last_accessed
            """,
            (abs_path, self.hash_source(source), source, time.time()),
        )
        self._conn.commit()
        logger.debug("SQLiteParseCache PUT : %s", abs_path)

    def evict(self, abs_path: str) -> None:
        """Remove a single entry from the database.

        Args:
            abs_path: Absolute path of the file to evict.
        """
        cur = self._conn.execute(
            "DELETE FROM parse_cache WHERE file_path = ?", (abs_path,)
        )
        self._conn.commit()
        if cur.rowcount:
            logger.debug("SQLiteParseCache EVICT: %s", abs_path)

    def clear(self) -> None:
        """Delete *all* rows from the cache table."""
        self._conn.execute("DELETE FROM parse_cache")
        self._conn.commit()
        logger.debug("SQLiteParseCache cleared")

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Snapshot of cache performance counters.

        Returns::

            {
                'size':     int,    # number of rows currently in the DB
                'hits':     int,    # cumulative hits since instantiation
                'misses':   int,    # cumulative misses since instantiation
                'hit_rate': float,  # hits / (hits + misses), or 0.0 if no ops
                'db_path':  str,    # path to the SQLite file
            }
        """
        total = self._hits + self._misses
        cur = self._conn.execute("SELECT COUNT(*) FROM parse_cache")
        size = cur.fetchone()[0]
        return {
            "size": size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total else 0.0,
            "db_path": self._db_path,
        }
