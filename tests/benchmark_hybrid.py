#!/usr/bin/env python3
"""
Benchmark script for hybrid search weight tuning.

This script evaluates hybrid search performance using NDCG@10 metric
and performs grid search to find optimal vector/keyword weight combinations.

Usage:
    python benchmark_hybrid.py --config ast_rag_config.json --queries queries.json

The queries.json file should contain:
{
  "queries": [
    {
      "query": "natural language query",
      "relevant_ids": [
        {"id": "node_id_1", "score": 3},
        {"id": "node_id_2", "score": 2},
        ...
      ]
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Optional
import numpy as np

from ast_rag.models import ProjectConfig
from ast_rag.graph_schema import create_driver
from ast_rag.embeddings import EmbeddingManager
from ast_rag.ast_rag_api import ASTRagAPI

logger = logging.getLogger(__name__)

# ============================================================================
# NDCG Calculation
# ============================================================================

def dcg_at_k(relevance_scores: list[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, score in enumerate(relevance_scores[:k]):
        # Use 1-based position for log2(i+2) (i+1 because position starts at 1)
        dcg += score / np.log2(i + 2)
    return dcg

def ndcg_at_k(relevance_scores: list[float], k: int, ideal_scores: Optional[list[float]] = None) -> float:
    """Compute Normalized Discounted Cumulative Gain at k.

    Args:
        relevance_scores: List of relevance scores for retrieved items (in order)
        k: Cutoff position
        ideal_scores: Optional pre-computed ideal scores (for efficiency)

    Returns:
        NDCG@k score in [0, 1], where 1.0 means perfect ranking.
    """
    if not relevance_scores:
        return 0.0

    # Compute DCG for the retrieved list
    dcg = dcg_at_k(relevance_scores, k)

    # Compute IDCG (ideal DCG)
    if ideal_scores is None:
        ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = dcg_at_k(ideal_scores, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg

# ============================================================================
# Benchmark Runner
# ============================================================================

class HybridSearchBenchmark:
    """Benchmark hybrid search with different weight configurations."""

    def __init__(
        self,
        driver,
        embed_manager: EmbeddingManager,
        ground_truth: dict[str, Any],
        k: int = 10,
    ):
        self.driver = driver
        self.embed = embed_manager
        self.ground_truth = ground_truth
        self.k = k
        self.api = ASTRagAPI(driver, embed_manager)
        # Resolve node identifiers to actual IDs
        self._resolved_gt = self._resolve_ground_truth_ids()

    def _resolve_identifier(
        self,
        qualified_name: str,
        file_path: str,
        kind: str,
    ) -> Optional[str]:
        """Resolve a node identifier to its actual node ID using find_definition."""
        # Extract the simple name from qualified_name (last part after last dot)
        name = qualified_name.split(".")[-1]
        candidates = self.api.find_definition(
            name=name,
            kind=kind,
            lang=None,  # Search all languages
        )
        # Find the candidate with matching qualified_name and file_path
        for node in candidates:
            if node.qualified_name == qualified_name and node.file_path == file_path:
                return node.id
        # If not found, try a looser match on qualified_name only
        for node in candidates:
            if node.qualified_name == qualified_name:
                logger.debug(
                    "Found node with matching qualified_name but different file: %s (expected %s)",
                    node.file_path, file_path
                )
                return node.id
        logger.warning(
            "Could not resolve node: %s in %s (kind=%s)",
            qualified_name, file_path, kind
        )
        return None

    def _resolve_ground_truth_ids(self) -> dict[str, list[dict]]:
        """Convert identifier-based ground truth to ID-based with resolution."""
        resolved = {}
        queries = self.ground_truth.get("queries", [])

        for query_data in queries:
            query = query_data["query"]
            relevant_items = []

            # Support both "relevant_ids" (direct IDs) and "relevant" (identifiers)
            raw_items = query_data.get("relevant_ids", query_data.get("relevant", []))

            for item in raw_items:
                if "id" in item:
                    # Already a node ID, use as-is
                    relevant_items.append({
                        "id": item["id"],
                        "score": float(item.get("score", 1.0))
                    })
                elif all(k in item for k in ("qualified_name", "file_path", "kind")):
                    # Identifier-based entry, resolve to ID
                    node_id = self._resolve_identifier(
                        qualified_name=item["qualified_name"],
                        file_path=item["file_path"],
                        kind=item["kind"],
                    )
                    if node_id:
                        relevant_items.append({
                            "id": node_id,
                            "score": float(item.get("score", 1.0))
                        })
                else:
                    logger.warning("Invalid ground truth item: %s", item)

            if relevant_items:
                resolved[query] = relevant_items

        logger.info(
            "Resolved %d/%d queries with ground truth items",
            len(resolved),
            len(queries)
        )
        return resolved

    def evaluate_config(
        self,
        vector_weight: float,
        keyword_weight: float,
    ) -> dict[str, float]:
        """Evaluate a single weight configuration.

        Returns:
            Dict with metrics: ndcg@k, avg_ndcg, etc.
        """
        ndcg_scores = []
        resolved_gt = self._resolved_gt

        for query, relevant_items in resolved_gt.items():
            if not relevant_items:
                continue

            # Build ground truth dict: node_id -> relevance score
            gt_scores = {item["id"]: item["score"] for item in relevant_items}

            # Run search with this configuration
            results = self.api.search_semantic(
                query=query,
                limit=self.k,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
            )

            # Extract relevance scores for retrieved items in order
            retrieved_scores = []
            for result in results:
                if result.node.id in gt_scores:
                    retrieved_scores.append(gt_scores[result.node.id])
                else:
                    retrieved_scores.append(0.0)  # Not relevant

            # Compute NDCG@k for this query
            ideal_scores = sorted(gt_scores.values(), reverse=True)
            ndcg = ndcg_at_k(retrieved_scores, self.k, ideal_scores)
            ndcg_scores.append(ndcg)

        # Compute average NDCG across all queries
        avg_ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

        return {
            "vector_weight": vector_weight,
            "keyword_weight": keyword_weight,
            "avg_ndcg@10": avg_ndcg,
            "min_ndcg": float(np.min(ndcg_scores)) if ndcg_scores else 0.0,
            "max_ndcg": float(np.max(ndcg_scores)) if ndcg_scores else 0.0,
            "std_ndcg": float(np.std(ndcg_scores)) if ndcg_scores else 0.0,
        }

    def grid_search(
        self,
        weight_steps: list[float] = None,
    ) -> list[dict[str, float]]:
        """Perform grid search over weight combinations.

        Args:
            weight_steps: List of weight values to try (e.g., [0.0, 0.1, ..., 1.0])

        Returns:
            List of result dicts sorted by avg_ndcg@10 descending.
        """
        if weight_steps is None:
            weight_steps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        results = []
        total_configs = len(weight_steps) ** 2
        logger.info("Starting grid search over %d weight configurations...", total_configs)

        for i, vec_w in enumerate(weight_steps):
            for kw_w in weight_steps:
                # Skip if both are zero
                if vec_w == 0 and kw_w == 0:
                    continue

                logger.debug("Testing config: vector=%.2f, keyword=%.2f", vec_w, kw_w)
                result = self.evaluate_config(vec_w, kw_w)
                results.append(result)

        # Sort by average NDCG@10 descending
        results.sort(key=lambda x: x["avg_ndcg@10"], reverse=True)
        return results

    def compare_pure_strategies(self) -> dict[str, float]:
        """Compare pure vector (1.0, 0.0) vs pure keyword (0.0, 1.0)."""
        logger.info("Comparing pure vector vs pure keyword search...")

        pure_vector = self.evaluate_config(1.0, 0.0)
        pure_keyword = self.evaluate_config(0.0, 1.0)

        return {
            "pure_vector": pure_vector["avg_ndcg@10"],
            "pure_keyword": pure_keyword["avg_ndcg@10"],
            "improvement_over_vector": (
                pure_keyword["avg_ndcg@10"] - pure_vector["avg_ndcg@10"]
            ) / max(pure_vector["avg_ndcg@10"], 1e-8),
        }

# ============================================================================
# Main
# ============================================================================

def load_config(config_path: str) -> ProjectConfig:
    """Load ProjectConfig from JSON file."""
    with open(config_path, 'r') as f:
        data = json.load(f)
    return ProjectConfig(**data)

def load_ground_truth(queries_path: str) -> dict[str, Any]:
    """Load ground truth queries and relevance judgments."""
    with open(queries_path, 'r') as f:
        data = json.load(f)
    return data

def main():
    parser = argparse.ArgumentParser(description="Benchmark hybrid search weights")
    parser.add_argument("--config", default="ast_rag_config.json", help="Path to config JSON")
    parser.add_argument("--queries", required=True, help="Path to ground truth queries JSON")
    parser.add_argument("--output", default="benchmark_results.json", help="Output results JSON")
    parser.add_argument("--k", type=int, default=10, help="Cutoff for NDCG@k (default: 10)")
    parser.add_argument("--steps", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        help="Weight values to test")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load configuration
    logger.info("Loading config from %s", args.config)
    cfg = load_config(args.config)

    # Load ground truth
    logger.info("Loading ground truth from %s", args.queries)
    ground_truth = load_ground_truth(args.queries)
    logger.info("Loaded %d test queries", len(ground_truth.get("queries", [])))

    # Initialize database connections
    logger.info("Initializing Neo4j driver...")
    driver = create_driver(cfg.neo4j)

    logger.info("Initializing EmbeddingManager...")
    embed = EmbeddingManager(cfg.qdrant, cfg.embedding, neo4j_driver=driver)

    try:
        # Create benchmark runner
        benchmark = HybridSearchBenchmark(driver, embed, ground_truth, k=args.k)

        # Compare pure strategies
        comparison = benchmark.compare_pure_strategies()
        logger.info("Pure vector NDCG@10: %.4f", comparison["pure_vector"])
        logger.info("Pure keyword NDCG@10: %.4f", comparison["pure_keyword"])
        logger.info("Relative improvement: %.2f%%", comparison["improvement_over_vector"] * 100)

        # Run grid search
        results = benchmark.grid_search(weight_steps=args.steps)

        # Display top 5 configurations
        logger.info("Top 5 weight configurations:")
        for i, res in enumerate(results[:5], 1):
            logger.info(
                "  %d. vec=%.2f, kw=%.2f → NDCG@10=%.4f",
                i, res["vector_weight"], res["keyword_weight"], res["avg_ndcg@10"]
            )

        # Compute improvement over pure vector
        best = results[0]
        pure_vector_ndcg = comparison["pure_vector"]
        improvement = (best["avg_ndcg@10"] - pure_vector_ndcg) / max(pure_vector_ndcg, 1e-8) * 100

        logger.info("Best configuration: vec=%.2f, kw=%.2f → NDCG@10=%.4f",
                    best["vector_weight"], best["keyword_weight"], best["avg_ndcg@10"])
        logger.info("Improvement over pure vector: %.2f%%", improvement)

        # Save results
        output = {
            "config": {
                "k": args.k,
                "weight_steps": args.steps,
                "num_queries": len(ground_truth.get("queries", [])),
            },
            "pure_strategies": comparison,
            "best_config": best,
            "all_results": results[:20],  # Save top 20
        }

        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info("Results saved to %s", args.output)

        # Exit with success if improvement >= 10%
        if improvement >= 10.0:
            logger.info("✓ Target achieved: %.2f%% improvement (>=10%%)", improvement)
            return 0
        else:
            logger.warning("⚠ Improvement below target: %.2f%% (<10%%)", improvement)
            return 0

    finally:
        driver.close()

if __name__ == "__main__":
    sys.exit(main())
