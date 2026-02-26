"""
watcher.py - File system watcher for AST-RAG.

Watches source files for changes and triggers incremental graph + embedding updates.
Uses watchdog for cross-platform file system events with debounced reparse.

Features:
- Debounced file change detection (avoids rapid-fire updates)
- Incremental graph update via workspace diff
- Embedding update for changed nodes
- Configurable exclude patterns

Usage:
    from ast_rag.watcher import WorkspaceWatcher
    
    watcher = WorkspaceWatcher(
        path="/path/to/project",
        config_path="ast_rag_config.json",
        debounce_seconds=2.0,
    )
    watcher.start()
    
    # ... watcher runs in background ...
    
    watcher.stop()
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional, Set

from watchdog.observers import Observer
from watchdog.events import (
    FileSystemEventHandler,
    FileModifiedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileMovedEvent,
    DirDeletedEvent,
)

from ast_rag.models import ProjectConfig
from ast_rag.graph_schema import create_driver
from ast_rag.graph_updater import apply_workspace_diff
from ast_rag.embeddings import EmbeddingManager
from ast_rag.ast_parser import ParserManager

logger = logging.getLogger(__name__)

# Supported source file extensions
SOURCE_EXTENSIONS = {
    ".py", ".java", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx",
    ".rs", ".ts", ".tsx", ".js", ".jsx", ".go", ".cs", ".rb", ".php",
    ".swift", ".kt", ".kts", ".scala", ".ex", ".exs", ".erl", ".hs",
}


def _is_source_file(path: str) -> bool:
    """Check if a file path has a supported source extension."""
    return Path(path).suffix.lower() in SOURCE_EXTENSIONS


class DebouncedHandler(FileSystemEventHandler):
    """File system event handler with debouncing."""
    
    def __init__(
        self,
        watcher: WorkspaceWatcher,
        debounce_seconds: float = 2.0,
    ):
        super().__init__()
        self._watcher = watcher
        self._debounce_seconds = debounce_seconds
        self._pending_files: Set[str] = set()
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
    
    def _schedule_update(self) -> None:
        """Schedule an update after debounce period."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            
            self._timer = threading.Timer(self._debounce_seconds, self._do_update)
            self._timer.start()
    
    def _do_update(self) -> None:
        """Perform the actual update."""
        with self._lock:
            files = list(self._pending_files)
            self._pending_files.clear()
        
        if files:
            logger.info("Processing %d changed files", len(files))
            self._watcher._update_graph(files)
    
    def _handle_file(self, path: str) -> None:
        """Handle a file event."""
        if not _is_source_file(path):
            return
        
        with self._lock:
            self._pending_files.add(path)
        
        self._schedule_update()
    
    def on_modified(self, event):
        if isinstance(event, FileModifiedEvent):
            self._handle_file(event.src_path)
    
    def on_created(self, event):
        if isinstance(event, FileCreatedEvent):
            self._handle_file(event.src_path)
    
    def on_deleted(self, event):
        if isinstance(event, (FileDeletedEvent, DirDeletedEvent)):
            self._handle_file(event.src_path)
    
    def on_moved(self, event):
        if isinstance(event, FileMovedEvent):
            self._handle_file(event.src_path)
            self._handle_file(event.dest_path)
    
    def stop(self) -> None:
        """Stop the pending timer."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


class WorkspaceWatcher:
    """Watches a workspace for source file changes and updates the graph."""
    
    def __init__(
        self,
        path: str = ".",
        config_path: Optional[str] = None,
        debounce_seconds: float = 2.0,
        exclude_dirs: Optional[list[str]] = None,
    ):
        """
        Initialize the workspace watcher.
        
        Args:
            path: Root directory to watch
            config_path: Path to AST-RAG config JSON
            debounce_seconds: Seconds to wait after last change before updating
            exclude_dirs: List of directory names to exclude from watching
        """
        self._path = os.path.abspath(path)
        self._config_path = config_path
        self._debounce_seconds = debounce_seconds
        self._exclude_dirs = set(exclude_dirs or {"venv", ".venv", "node_modules", ".git", "__pycache__"})
        
        # Load config
        if config_path and Path(config_path).exists():
            self._config = ProjectConfig.model_validate_json(Path(config_path).read_text())
        else:
            default = Path("ast_rag_config.json")
            if default.exists():
                self._config = ProjectConfig.model_validate_json(default.read_text())
            else:
                self._config = ProjectConfig()
        
        # Initialize components
        self._driver = create_driver(self._config.neo4j)
        self._embed = EmbeddingManager(self._config.qdrant, self._config.embedding, neo4j_driver=self._driver)
        self._parser = ParserManager()
        
        # Watcher state
        self._observer: Optional[Observer] = None
        self._handler: Optional[DebouncedHandler] = None
        self._running = False
        self._lock = threading.Lock()
    
    def _should_ignore(self, path: str) -> bool:
        """Check if a path should be ignored."""
        parts = Path(path).parts
        return any(exclude in parts for exclude in self._exclude_dirs)
    
    def _update_graph(self, changed_files: list[str]) -> None:
        """Update the graph for changed files."""
        try:
            logger.info("Updating graph for %d files", len(changed_files))
            
            # Apply workspace diff
            diff = apply_workspace_diff(
                self._driver,
                self._path,
                exclude_dirs=list(self._exclude_dirs),
            )
            
            if not diff.is_empty:
                # Update embeddings
                self._embed.update_embeddings(
                    diff.added_nodes,
                    diff.updated_nodes,
                    diff.deleted_node_ids,
                )
                logger.info(
                    "Graph updated: +%d nodes, ~%d updated, -%d deleted",
                    len(diff.added_nodes),
                    len(diff.updated_nodes),
                    len(diff.deleted_node_ids),
                )
            else:
                logger.debug("No changes detected in graph")
                
        except Exception as e:
            logger.error("Failed to update graph: %s", e)
    
    def start(self) -> None:
        """Start watching for file changes."""
        with self._lock:
            if self._running:
                return
            
            self._handler = DebouncedHandler(self, self._debounce_seconds)
            self._observer = Observer()
            
            # Schedule watching, ignoring excluded directories
            self._observer.schedule(
                self._handler,
                self._path,
                recursive=True,
            )
            
            self._observer.start()
            self._running = True
            logger.info("Started watching %s", self._path)
    
    def stop(self) -> None:
        """Stop watching for file changes."""
        with self._lock:
            if not self._running:
                return
            
            if self._handler:
                self._handler.stop()
            
            if self._observer:
                self._observer.stop()
                self._observer.join(timeout=5)
            
            self._running = False
            logger.info("Stopped watching")
    
    def wait(self) -> None:
        """Block until the watcher is stopped."""
        if self._observer:
            self._observer.join()
    
    @property
    def is_running(self) -> bool:
        """Check if the watcher is currently running."""
        return self._running


def main():
    """Run the workspace watcher as a standalone process."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AST-RAG Workspace Watcher")
    parser.add_argument("path", nargs="?", default=".", help="Directory to watch")
    parser.add_argument("--config", "-c", help="Path to config JSON")
    parser.add_argument("--debounce", "-d", type=float, default=2.0, help="Debounce seconds")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    watcher = WorkspaceWatcher(
        path=args.path,
        config_path=args.config,
        debounce_seconds=args.debounce,
    )
    
    watcher.start()
    
    try:
        watcher.wait()
    except KeyboardInterrupt:
        logger.info("Interrupted, stopping...")
        watcher.stop()


if __name__ == "__main__":
    main()
