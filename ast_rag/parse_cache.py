"""
parse_cache.py - In-memory parse-tree cache for ParserManager.

Responsibilities (single):
    Cache parsed tree-sitter Tree objects by file path + content hash so
    unchanged files are never re-parsed in the same process lifetime.

Design notes
------------
- The cache is keyed by *absolute* file path so relative/absolute callers
  always hit the same slot.
- Invalidation is content-driven (SHA-256 of source bytes), not time-based.
  A file is re-parsed only when its bytes actually change.
- Swappable backend: the public interface (get / put / evict / clear / stats)
  is intentionally narrow so a future SQLite or Redis backend can be dropped
  in without touching ParserManager at all.

Future backends (drop-in replacements):
    SQLiteParseCache  - survives process restarts
    RedisParseCache   - shared across workers
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Optional

from tree_sitter import Tree

logger = logging.getLogger(__name__)


class ParseCache:
    """Content-addressed in-memory cache for tree-sitter parse trees.

    Usage::

        cache = ParseCache()

        # Before parsing:
        tree = cache.get(abs_path, source_bytes)

        # After parsing:
        cache.put(abs_path, source_bytes, tree)

        # Stats:
        print(cache.stats())
    """

    def __init__(self) -> None:
        # key: absolute file path
        # value: (sha256_hex, Tree)
        self._store: dict[str, tuple[str, Tree]] = {}
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @staticmethod
    def hash_source(source: bytes) -> str:
        """Return the SHA-256 hex digest of source bytes.

        Extracted here so callers (and future backends) share one
        hashing implementation.
        """
        return hashlib.sha256(source).hexdigest()

    def get(self, abs_path: str, source: bytes) -> Optional[Tree]:
        """Return a cached Tree if the stored hash matches *source*, else None.

        Args:
            abs_path: Absolute path to the file (used as cache key).
            source:   Current source bytes of the file.

        Returns:
            The cached Tree on a hit, or None on a miss / stale entry.
        """
        entry = self._store.get(abs_path)
        if entry is not None:
            stored_hash, cached_tree = entry
            if stored_hash == self.hash_source(source):
                self._hits += 1
                logger.debug("ParseCache HIT : %s", abs_path)
                return cached_tree

        self._misses += 1
        logger.debug("ParseCache MISS: %s", abs_path)
        return None

    def put(self, abs_path: str, source: bytes, tree: Tree) -> None:
        """Store (or refresh) a cache entry.

        Args:
            abs_path: Absolute path to the file (cache key).
            source:   Source bytes the tree was parsed from.
            tree:     The freshly parsed Tree to cache.
        """
        self._store[abs_path] = (self.hash_source(source), tree)
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
