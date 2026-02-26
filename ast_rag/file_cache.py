"""
file_cache.py - File change detection for skipping unchanged files.

Provides:
- FileCache: Stores file hashes to detect changes
- file_changed_since_last_index(): Check if a file has changed
- save_file_cache(): Persist cache to disk

This optimization avoids re-parsing files that haven't changed since
the last indexing run, significantly speeding up incremental updates.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

# Cache file location
CACHE_FILE = ".ast_rag_file_cache.json"


def _compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file's contents."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def _get_git_hash(file_path: str, repo_path: str = ".") -> Optional[str]:
    """Get Git object hash for a file (fast, no file reading needed)."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "ls-files", "-s", file_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            # Format: "mode hash stage path"
            parts = result.stdout.strip().split()
            if len(parts) >= 2:
                return parts[1]
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return None


def _load_cache(cache_path: str) -> dict:
    """Load file cache from disk."""
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load file cache: {e}")
    return {}


def _save_cache(cache_path: str, cache: dict) -> None:
    """Save file cache to disk."""
    try:
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
        logger.debug(f"Saved file cache with {len(cache)} entries")
    except OSError as e:
        logger.warning(f"Failed to save file cache: {e}")


class FileCache:
    """Manages file change detection cache."""

    def __init__(self, root_path: str):
        self.root_path = os.path.abspath(root_path)
        self.cache_path = os.path.join(self.root_path, CACHE_FILE)
        self.cache: dict[str, str] = _load_cache(self.cache_path)
        self._git_repo: Optional[str] = None

        # Check if root is a git repo
        try:
            result = subprocess.run(
                ["git", "-C", self.root_path, "rev-parse", "--git-dir"],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                self._git_repo = self.root_path
                logger.debug(f"Git repository detected: {self.root_path}")
        except Exception:
            pass

    def has_changed(self, file_path: str) -> bool:
        """Check if a file has changed since last index.

        Uses Git hash if available (fast), otherwise computes file hash.

        Args:
            file_path: Absolute path to the file

        Returns:
            True if file has changed or is new, False if unchanged
        """
        file_path = os.path.abspath(file_path)

        if not os.path.exists(file_path):
            return True  # Deleted files are "changed"

        # Try Git hash first (fast)
        if self._git_repo:
            rel_path = os.path.relpath(file_path, self._git_repo)
            git_hash = _get_git_hash(rel_path, self._git_repo)
            if git_hash:
                old_hash = self.cache.get(file_path)
                return old_hash != git_hash

        # Fallback to file hash (slower but works without Git)
        current_hash = _compute_file_hash(file_path)
        old_hash = self.cache.get(file_path)
        return old_hash != current_hash

    def update(self, file_path: str) -> None:
        """Update cache entry for a file."""
        file_path = os.path.abspath(file_path)

        if self._git_repo:
            rel_path = os.path.relpath(file_path, self._git_repo)
            git_hash = _get_git_hash(rel_path, self._git_repo)
            if git_hash:
                self.cache[file_path] = git_hash
                return

        # Fallback to file hash
        self.cache[file_path] = _compute_file_hash(file_path)

    def remove(self, file_path: str) -> None:
        """Remove cache entry for a deleted file."""
        file_path = os.path.abspath(file_path)
        self.cache.pop(file_path, None)

    def save(self) -> None:
        """Persist cache to disk."""
        _save_cache(self.cache_path, self.cache)

    def stats(self) -> dict:
        """Get cache statistics."""
        return {
            "entries": len(self.cache),
            "has_git": self._git_repo is not None,
            "cache_path": self.cache_path,
        }


# Module-level convenience functions
_default_cache: Optional[FileCache] = None


def init_file_cache(root_path: str) -> FileCache:
    """Initialize the global file cache."""
    global _default_cache
    _default_cache = FileCache(root_path)
    return _default_cache


def file_changed_since_last_index(file_path: str) -> bool:
    """Check if a file has changed since last index.

    Args:
        file_path: Absolute path to the file

    Returns:
        True if file has changed or cache not initialized
    """
    if _default_cache is None:
        return True  # No cache, assume changed
    return _default_cache.has_changed(file_path)


def update_file_cache(file_path: str) -> None:
    """Update cache for a file."""
    if _default_cache is not None:
        _default_cache.update(file_path)


def save_file_cache() -> None:
    """Save cache to disk."""
    if _default_cache is not None:
        _default_cache.save()


def get_cache_stats() -> dict:
    """Get cache statistics."""
    if _default_cache is None:
        return {"initialized": False}
    return _default_cache.stats()
