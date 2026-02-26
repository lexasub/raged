#!/usr/bin/env python3
"""
AST-RAG Quality Benchmark: Ground Truth Creation

This script helps create ground truth data for evaluating AST-RAG quality.
"""

import json
from pathlib import Path
from ast_rag.ast_rag_api import ASTRagAPI
from ast_rag.graph_schema import create_driver
from ast_rag.embeddings import EmbeddingManager
from ast_rag.models import ProjectConfig

print("=" * 80)
print(" " * 20 + "AST-RAG GROUND TRUTH CREATOR")
print("=" * 80)

# Load config
config_path = Path("ast_rag_config.json")
cfg = ProjectConfig.model_validate_json(config_path.read_text())

# Initialize
driver = create_driver(cfg.neo4j)
embed = EmbeddingManager(cfg.qdrant, cfg.embedding, neo4j_driver=driver)
api = ASTRagAPI(driver, embed)

print("\n📋 Available benchmark scenarios:\n")

scenarios = [
    {
        "id": "java_refs_001",
        "name": "Java Method References",
        "description": "Find all usages of a Java method",
        "query": {"name": "find_references", "params": {}},
    },
    {
        "id": "java_impact_001",
        "name": "Java Class Impact",
        "description": "Analyze impact of changing a Java class",
        "query": {"name": "symbol_impact", "params": {}},
    },
    {
        "id": "cpp_calls_001",
        "name": "C++ Call Chain",
        "description": "Trace C++ function call chain",
        "query": {"name": "find_callers", "params": {}},
    },
    {
        "id": "rust_impl_001",
        "name": "Rust Trait Implementations",
        "description": "Find all implementations of a Rust trait",
        "query": {"name": "find_references", "params": {}},
    },
]

for i, scenario in enumerate(scenarios, 1):
    print(f"{i}. {scenario['name']}")
    print(f"   {scenario['description']}")
    print(f"   Tool: {scenario['query']['name']}")
    print()

# Example: Create ground truth for Java references
print("\n" + "=" * 80)
print("EXAMPLE: Creating ground truth for Java method references")
print("=" * 80)

# Query the graph
print("\n🔍 Querying: find_references('ASTRagAPI', kind='Class', lang='java')...")

try:
    refs = api.find_references("ASTRagAPI", kind="Class", lang="java", limit=100)
    
    print(f"   Found {refs['total']} references")
    
    # Create ground truth structure
    ground_truth = {
        "id": "java_refs_001",
        "scenario": "Find all usages of a Java class",
        "tool": "find_references",
        "query": {
            "name": "ASTRagAPI",
            "kind": "Class",
            "lang": "java",
        },
        "ground_truth": {
            "total_expected": refs['total'],
            "references": [
                {
                    "file": ref['node']['file_path'],
                    "line": ref['node']['start_line'],
                    "type": ref['reference_type'],
                    "context": ref['node'].get('qualified_name', ''),
                }
                for ref in refs['references'][:20]  # First 20 for ground truth
            ]
        },
        "metrics": {
            "precision": None,  # To be filled by evaluation
            "recall": None,
            "f1_score": None,
        },
        "created_at": str(Path.cwd()),
    }
    
    # Save to file
    output_path = Path("benchmarks/ground_truth/java_refs_001.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"\n✅ Ground truth saved to: {output_path}")
    
    # Show sample
    print("\n📄 Sample ground truth structure:")
    print(json.dumps(ground_truth, indent=2)[:1000] + "...")
    
except Exception as e:
    print(f"\n❌ Error: {e}")

# Example: Create benchmark query
print("\n" + "=" * 80)
print("Creating benchmark query file")
print("=" * 80)

benchmark_query = {
    "id": "refactor_001",
    "category": "refactoring",
    "task": "Find all usages for renaming",
    "question": "Where is ASTRagAPI used in the codebase? I want to rename it.",
    "expected_tool": "find_references",
    "expected_params": {
        "name": "ASTRagAPI",
        "kind": "Class",
        "lang": "java",
        "limit": 100,
    },
    "ground_truth_file": "ground_truth/java_refs_001.json",
    "evaluation": {
        "min_precision": 0.9,
        "min_recall": 0.85,
        "max_time_seconds": 5.0,
    },
}

query_path = Path("benchmarks/queries/refactor_001.json")
with open(query_path, "w") as f:
    json.dump(benchmark_query, f, indent=2)

print(f"\n✅ Benchmark query saved to: {query_path}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
Created files:
  - {output_path} (ground truth)
  - {query_path} (benchmark query)

Next steps:
  1. Review ground truth for accuracy
  2. Add more scenarios (cpp_calls, rust_impls, etc.)
  3. Run evaluation: python benchmarks/run_evaluation.py
  4. Compare with baseline (text search without AST-RAG)

To evaluate:
  python benchmarks/run_evaluation.py --query refactor_001
""")

driver.close()
