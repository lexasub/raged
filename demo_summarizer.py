#!/usr/bin/env python3
"""
demo_summarizer.py - Demonstration of the code summarization feature.

This script shows how to use the SummarizerService to generate
structured summaries of code entities.

Usage:
    python demo_summarizer.py

Requirements:
    - Running Neo4j instance with indexed code
    - Running Ollama (or other OpenAI-compatible LLM)
    - ast_rag package installed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ast_rag.models import ProjectConfig
from ast_rag.graph_schema import create_driver
from ast_rag.embeddings import EmbeddingManager
from ast_rag.ast_rag_api import ASTRagAPI
from ast_rag.summarizer import SummarizerService, NodeSummary


def load_config(config_path: str = "ast_rag_config.json") -> ProjectConfig:
    """Load project configuration."""
    cfg_file = Path(config_path)
    if cfg_file.exists():
        return ProjectConfig.model_validate_json(cfg_file.read_text())
    return ProjectConfig()


def initialize_api(cfg: ProjectConfig) -> ASTRagAPI:
    """Initialize the AST-RAG API."""
    print("Initializing Neo4j connection...")
    driver = create_driver(cfg.neo4j)
    
    print("Initializing EmbeddingManager...")
    embed = EmbeddingManager(cfg.qdrant, cfg.embedding, neo4j_driver=driver)
    
    print("Creating ASTRagAPI...")
    return ASTRagAPI(driver, embed)


def demo_basic_summarization(api: ASTRagAPI, summarizer: SummarizerService):
    """Demonstrate basic summarization."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Summarization")
    print("="*60)
    
    # Find a function to summarize
    print("\nSearching for a function to summarize...")
    
    # Try to find any function
    nodes = api.find_definition("", kind="Function")
    
    if not nodes:
        print("No functions found. Make sure the codebase is indexed.")
        return
    
    node = nodes[0]
    print(f"Found: {node.qualified_name} ({node.kind.value})")
    print(f"  File: {node.file_path}:{node.start_line}-{node.end_line}")
    
    # Generate summary
    print("\nGenerating summary with LLM...")
    try:
        summary = summarizer.summarize_node(
            node_id=node.id,
            api=api,
            max_callers=3,
            max_callees=3,
        )
        
        print("\n" + "-"*60)
        print("SUMMARY (Markdown):")
        print("-"*60)
        print(summary.to_markdown())
        
    except Exception as e:
        print(f"Error generating summary: {e}")


def demo_json_output(api: ASTRagAPI, summarizer: SummarizerService):
    """Demonstrate JSON output format."""
    print("\n" + "="*60)
    print("DEMO 2: JSON Output Format")
    print("="*60)
    
    # Find a function
    nodes = api.find_definition("", kind="Function")
    if not nodes:
        print("No functions found.")
        return
    
    node = nodes[0]
    
    try:
        summary = summarizer.summarize_node(
            node_id=node.id,
            api=api,
            force_regenerate=False,
        )
        
        print("\nJSON Output:")
        print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"Error: {e}")


def demo_batch_summarization(api: ASTRagAPI, summarizer: SummarizerService):
    """Demonstrate batch summarization."""
    print("\n" + "="*60)
    print("DEMO 3: Batch Summarization")
    print("="*60)
    
    # Find multiple functions
    nodes = api.find_definition("", kind="Function")
    
    if len(nodes) < 2:
        print("Need at least 2 functions for batch demo.")
        return
    
    # Take first 3 functions
    target_nodes = nodes[:3]
    node_ids = [n.id for n in target_nodes]
    
    print(f"\nSummarizing {len(node_ids)} functions...")
    
    try:
        summaries = summarizer.summarize_nodes(
            node_ids=node_ids,
            api=api,
            max_callers=2,
            max_callees=2,
        )
        
        print("\nResults:")
        for i, (node, summary) in enumerate(zip(target_nodes, summaries), 1):
            if summary:
                print(f"\n{i}. {node.qualified_name}")
                print(f"   Summary: {summary.summary[:100]}...")
                print(f"   Complexity: {summary.complexity.value}")
                print(f"   Tags: {', '.join(summary.tags) if summary.tags else 'none'}")
            else:
                print(f"\n{i}. {node.qualified_name} - Failed to summarize")
        
    except Exception as e:
        print(f"Error: {e}")


def demo_cache_stats(summarizer: SummarizerService):
    """Demonstrate cache statistics."""
    print("\n" + "="*60)
    print("DEMO 4: Cache Statistics")
    print("="*60)
    
    stats = summarizer.get_cache_stats()
    
    print("\nCache Statistics:")
    print(f"  Entries: {stats['entries']}")
    print(f"  Cache Path: {stats['cache_path']}")
    print(f"  Enabled: {stats['cache_enabled']}")


def main():
    """Run all demonstrations."""
    print("="*60)
    print("AST-RAG Code Summarization Demo")
    print("="*60)
    
    # Load configuration
    print("\nLoading configuration...")
    cfg = load_config()
    
    # Initialize API
    api = initialize_api(cfg)
    
    # Initialize summarizer
    print("\nInitializing SummarizerService...")
    summarizer = SummarizerService(
        base_url="http://localhost:11434/v1",
        model="qwen2.5-coder:14b",
        cache_enabled=True,
    )
    
    # Run demos
    try:
        demo_basic_summarization(api, summarizer)
        demo_json_output(api, summarizer)
        demo_batch_summarization(api, summarizer)
        demo_cache_stats(summarizer)
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "="*60)
        print("Demo completed!")
        print("="*60)


if __name__ == "__main__":
    main()
