"""
Bounded LRU AST Cache with memory limits.

Provides O(1) cache operations with automatic eviction based on:
- Maximum entry count
- Maximum memory usage (in MB)

Two classes:

BoundedASTCache
    Dict-style LRU cache storing (Tree, lang) tuples keyed by Path.
    Enforces dual limits (max entries + max memory).

BoundedParseCache
    Adapter that wraps BoundedASTCache with the get/put/evict/clear/stats
    interface expected by ParserManager (compatible with ParseCache and
    SQLiteParseCache).
"""

from __future__ import annotations

import hashlib
import logging
import sys
from collections import OrderedDict
from typing import Any, Callable, ItemsView, Optional, Tuple

from tree_sitter import Tree

from ast_rag.utils.parse_cache import LazyTree

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BoundedASTCache  (dict-style LRU with dual limits)
# ---------------------------------------------------------------------------


class BoundedASTCache:
    """LRU cache for AST trees with bounded size and memory usage.

    Attributes:
        max_entries: Maximum number of entries (default: 10000)
        max_memory_mb: Maximum memory usage in MB (default: 500)
    """

    def __init__(
        self,
        max_entries: Optional[int] = None,
        max_memory_mb: Optional[int] = None,
    ) -> None:
        """Initialize bounded cache.

        Args:
            max_entries: Max entries before eviction. Default: 10000
            max_memory_mb: Max memory in MB before eviction. Default: 500
        """
        # Cache: file_path -> (tree, language_string)
        self._cache: OrderedDict[str, Tuple[Tree, str]] = OrderedDict()

        # Limits
        self.max_entries = max_entries if max_entries is not None else 10_000
        self.max_memory_bytes = (max_memory_mb if max_memory_mb is not None else 500) * 1024 * 1024

        # Track approximate memory usage
        self._memory_usage: int = 0
        # Per-key memory tracking for accurate subtraction on eviction
        self._entry_sizes: dict[str, int] = {}

    def __getitem__(self, key: str) -> Tuple[Tree, str]:
        """Get item from cache, moving it to end (most recently used)."""
        value = self._cache[key]  # Raises KeyError if not found
        self._cache.move_to_end(key)
        return value

    def __setitem__(self, key: str, value: Tuple[Tree, str]) -> None:
        """Add item to cache, evicting if necessary."""
        if key in self._cache:
            # Remove old entry's memory contribution
            self._memory_usage -= self._entry_sizes.pop(key, 0)
            del self._cache[key]

        # Add new entry
        self._cache[key] = value
        # Enforce limits
        self._enforce_limits()

    def set_with_source(self, key: str, value: Tuple[Tree, str], source: bytes) -> None:
        """Add item to cache with explicit source bytes for memory estimation.

        This is the preferred method when source bytes are available, as it
        gives a more accurate memory estimate than the fallback.

        Args:
            key: Absolute file path (cache key).
            value: (tree, language_string) tuple.
            source: Source bytes the tree was parsed from.
        """
        if key in self._cache:
            self._memory_usage -= self._entry_sizes.pop(key, 0)
            del self._cache[key]

        size = sys.getsizeof(source)
        self._cache[key] = value
        self._entry_sizes[key] = size
        self._memory_usage += size

        self._enforce_limits()

    def __delitem__(self, key: str) -> None:
        """Remove item from cache."""
        if key in self._cache:
            self._memory_usage -= self._entry_sizes.pop(key, 0)
            del self._cache[key]

    def __contains__(self, key: object) -> bool:
        """Check if key is in cache."""
        return key in self._cache

    def __len__(self) -> int:
        """Return number of entries in cache."""
        return len(self._cache)

    def __iter__(self):
        """Iterate over cache keys."""
        return iter(self._cache)

    def items(self) -> ItemsView[str, Tuple[Tree, str]]:
        """Return all items in cache."""
        return self._cache.items()

    def clear(self) -> None:
        """Clear all entries from cache."""
        self._cache.clear()
        self._entry_sizes.clear()
        self._memory_usage = 0

    def get_memory_usage_mb(self) -> float:
        """Return current memory usage in MB."""
        return self._memory_usage / (1024 * 1024)

    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        return {
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "memory_mb": round(self.get_memory_usage_mb(), 4),
            "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
            "utilization_entries": (
                len(self._cache) / self.max_entries if self.max_entries else 0.0
            ),
            "utilization_memory": (
                self._memory_usage / self.max_memory_bytes if self.max_memory_bytes else 0.0
            ),
        }

    def resize(
        self,
        max_entries: Optional[int] = None,
        max_memory_mb: Optional[int] = None,
    ) -> None:
        """Resize cache limits and evict if necessary.

        Args:
            max_entries: New max entries limit
            max_memory_mb: New max memory limit in MB
        """
        if max_entries is not None:
            self.max_entries = max_entries
        if max_memory_mb is not None:
            self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._enforce_limits()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enforce_limits(self) -> None:
        """Evict oldest entries to stay within limits."""
        while len(self._cache) > self.max_entries:
            self._evict_oldest()
        while self._memory_usage > self.max_memory_bytes and len(self._cache) > 0:
            self._evict_oldest()

    def _evict_oldest(self) -> None:
        """Evict the oldest (least recently used) entry."""
        if not self._cache:
            return
        key, _value = self._cache.popitem(last=False)
        evicted_size = self._entry_sizes.pop(key, 0)
        self._memory_usage -= evicted_size
        logger.debug("BoundedASTCache EVICT: %s (freed %d bytes)", key, evicted_size)


# ---------------------------------------------------------------------------
# BoundedParseCache  (adapter with ParseCache-compatible interface)
# ---------------------------------------------------------------------------


class BoundedParseCache:
    """Bounded in-memory parse cache compatible with ParseCache / SQLiteParseCache.

    Wraps a ``BoundedASTCache`` and exposes the ``get / put / evict / clear /
    stats`` interface that ``ParserManager`` expects.

    Usage::

        cache = BoundedParseCache(max_entries=5000, max_memory_mb=512)
        pm = ParserManager(cache=cache)
    """

    def __init__(
        self,
        max_entries: Optional[int] = None,
        max_memory_mb: Optional[int] = None,
    ) -> None:
        self._inner = BoundedASTCache(
            max_entries=max_entries,
            max_memory_mb=max_memory_mb,
        )
        self._hits: int = 0
        self._misses: int = 0
        # Store content hashes alongside entries for staleness checks
        self._hashes: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Core interface  (same as ParseCache / SQLiteParseCache)
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
        """Return a cached ``LazyTree`` if the stored hash matches *source*.

        Args:
            abs_path: Absolute path to the file (cache key).
            source:   Current source bytes — used for hash comparison.
            loader:   Ignored by in-memory backend (tree already available).

        Returns:
            A pre-loaded ``LazyTree`` on a hit, or ``None`` on a miss.
        """
        if abs_path not in self._inner:
            self._misses += 1
            logger.debug("BoundedParseCache MISS: %s", abs_path)
            return None

        stored_hash = self._hashes.get(abs_path)
        if stored_hash != self.hash_source(source):
            self._misses += 1
            logger.debug("BoundedParseCache MISS (stale): %s", abs_path)
            return None

        self._hits += 1
        # Touch for LRU ordering
        tree, _lang = self._inner[abs_path]

        # Return a pre-loaded LazyTree (same pattern as ParseCache)
        lazy = LazyTree(loader=lambda t=tree: t)
        object.__setattr__(lazy, "_tree", tree)
        object.__setattr__(lazy, "_hash", stored_hash)
        logger.debug("BoundedParseCache HIT : %s", abs_path)
        return lazy

    def put(self, abs_path: str, source: bytes, tree: Tree) -> None:
        """Store (or refresh) a cache entry with bounded eviction.

        Args:
            abs_path: Absolute path to the file (cache key).
            source:   Source bytes the tree was parsed from.
            tree:     The freshly parsed Tree to cache.
        """
        content_hash = self.hash_source(source)
        lang = ""  # language is not critical for cache storage
        self._hashes[abs_path] = content_hash
        self._inner.set_with_source(abs_path, (tree, lang), source)
        logger.debug("BoundedParseCache PUT : %s", abs_path)

    def evict(self, abs_path: str) -> None:
        """Remove a single entry.

        Args:
            abs_path: Absolute path of the file to evict.
        """
        if abs_path in self._inner:
            del self._inner[abs_path]
            self._hashes.pop(abs_path, None)
            logger.debug("BoundedParseCache EVICT: %s", abs_path)

    def clear(self) -> None:
        """Evict all cached trees."""
        self._inner.clear()
        self._hashes.clear()
        logger.debug("BoundedParseCache cleared")

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Snapshot of cache performance counters.

        Returns::

            {
                'size':            int,
                'hits':            int,
                'misses':          int,
                'hit_rate':        float,
                'max_entries':     int,
                'memory_mb':       float,
                'max_memory_mb':   float,
            }
        """
        total = self._hits + self._misses
        inner_stats = self._inner.get_stats()
        return {
            "size": inner_stats["entries"],
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total else 0.0,
            "max_entries": inner_stats["max_entries"],
            "memory_mb": inner_stats["memory_mb"],
            "max_memory_mb": inner_stats["max_memory_mb"],
        }
