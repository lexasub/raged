#!/usr/bin/env python3
"""
Script to generate valid ground truth queries from actual indexed data in Neo4j.

This script:
1. Connects to Neo4j and retrieves 20 random Class/Function/Method nodes
2. For each node, creates a natural language query that should match it
3. Outputs a JSON file with the format: {"query": "...", "relevant_ids": [{"id": "...", "score": 3}]}

The script generates queries like:
- For classes: "class that handles {name}"
- For functions: "function named {name}"
- For methods: "method {name} in {class}"
"""

import json
import random
import logging
from typing import List, Dict, Any

from ast_rag.models import ProjectConfig
from ast_rag.graph_schema import create_driver

logger = logging.getLogger(__name__)


class GroundTruthGenerator:
    """Generates ground truth queries from actual Neo4j data."""

    def __init__(self, driver):
        self.driver = driver

    def get_random_nodes(self, node_label: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve random nodes of a specific type from Neo4j."""
        query = f"""
        MATCH (n:{node_label})
        WHERE n.id IS NOT NULL AND n.name IS NOT NULL
        RETURN n.id as id, n.name as name, n.qualified_name as qualified_name, n.file_path as file_path
        ORDER BY rand()
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, limit=limit)
            return [dict(record) for record in result]

    def generate_query_for_node(self, node: Dict[str, Any], node_type: str) -> Dict[str, Any]:
        """Generate a natural language query for a specific node."""
        name = node.get('name', '')
        qualified_name = node.get('qualified_name', '')

        if node_type == 'Class':
            query = f"class that handles {name}"
        elif node_type == 'Function':
            query = f"function named {name}"
        elif node_type == 'Method':
            # Extract class name from qualified_name (e.g., "EmbeddingManager.hybrid_search")
            parts = qualified_name.split('.')
            if len(parts) >= 2:
                class_name = parts[0]
                method_name = parts[-1]
                query = f"method {method_name} in {class_name}"
            else:
                query = f"method {name}"
        else:
            query = f"{node_type.lower()} named {name}"

        return {
            "query": query,
            "relevant_ids": [
                {"id": node['id'], "score": 3}
            ]
        }

    def generate_ground_truth(self, output_path: str = "ground_truth_queries.json", num_nodes_per_type: int = 7) -> None:
        """Generate ground truth queries for random nodes."""
        # Get random nodes of each type
        classes = self.get_random_nodes('Class', num_nodes_per_type)
        functions = self.get_random_nodes('Function', num_nodes_per_type)
        methods = self.get_random_nodes('Method', num_nodes_per_type)

        # Combine all nodes
        all_entries = []
        for node in classes:
            all_entries.append((node, 'Class'))
        for node in functions:
            all_entries.append((node, 'Function'))
        for node in methods:
            all_entries.append((node, 'Method'))

        random.shuffle(all_entries)
        all_entries = all_entries[:20]  # Take up to 20 total

        # Generate queries for each node
        queries = []
        for node, node_type in all_entries:
            query_data = self.generate_query_for_node(node, node_type)
            queries.append(query_data)

        # Save to JSON file
        output = {
            "description": "Ground truth queries generated from actual Neo4j data",
            "queries": queries
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Generated {len(queries)} ground truth queries in {output_path}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load configuration
    logger.info("Loading config from ast_rag_config.json")
    with open('ast_rag_config.json', 'r') as f:
        config_data = json.load(f)
    cfg = ProjectConfig(**config_data)

    # Initialize Neo4j driver
    logger.info("Initializing Neo4j driver...")
    driver = create_driver(cfg.neo4j)

    try:
        # Create ground truth generator
        generator = GroundTruthGenerator(driver)

        # Generate ground truth queries
        generator.generate_ground_truth()

    finally:
        driver.close()


if __name__ == "__main__":
    main()
