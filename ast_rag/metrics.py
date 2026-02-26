"""
Prometheus metrics for AST-RAG monitoring.

Usage:
    from ast_rag.metrics import track_latency, SEARCH_LATENCY

    @track_latency(SEARCH_LATENCY)
    def search_semantic(...):
        ...
"""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import TYPE_CHECKING, Callable, TypeVar

from prometheus_client import Counter, Gauge, Histogram, start_http_server

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

T = TypeVar("T")

SEARCH_LATENCY = Histogram(
    "ast_rag_search_latency_seconds",
    "Semantic search latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

FIND_DEFINITION_LATENCY = Histogram(
    "ast_rag_find_definition_latency_seconds",
    "Find definition latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

FIND_REFERENCES_LATENCY = Histogram(
    "ast_rag_find_references_latency_seconds",
    "Find references latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

UPDATE_LATENCY = Histogram(
    "ast_rag_update_latency_seconds",
    "Git update latency",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0],
)

SEARCH_TOTAL = Counter(
    "ast_rag_search_total",
    "Total search requests",
    ["lang", "kind"],
)

UPDATE_TOTAL = Counter(
    "ast_rag_update_total",
    "Total update operations",
    ["status"],
)

GRAPH_NODES_TOTAL = Gauge(
    "ast_rag_graph_nodes_total",
    "Total nodes in Neo4j graph",
)

GRAPH_EDGES_TOTAL = Gauge(
    "ast_rag_graph_edges_total",
    "Total edges in Neo4j graph",
)

SKIP_RATIO = Gauge(
    "ast_rag_skip_ratio",
    "Ratio of skipped files during update",
)


def track_latency(metric: Histogram) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to track function latency."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: object, **kwargs: object) -> T:
            start = time.time()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{func.__name__} failed: {e}")
                raise
            finally:
                metric.observe(time.time() - start)

        return wrapper  # type: ignore[return-value]

    return decorator


def start_metrics_server(port: int = 8000) -> None:
    """Start Prometheus metrics HTTP server."""
    start_http_server(port)
    logger.info(f"Metrics server started on port {port}")
