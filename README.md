# AST-RAG — Code Analysis & Navigation System

**AST-RAG** parses code into AST (via Tree-sitter), builds a graph in **Neo4j**, and indexes semantic embeddings in **Qdrant** (bge-m3).

## 🚀 Features

| Feature | Description | Example |
|---------|----------|---------|
| 🔍 **Semantic Search** | Search by natural language | `ast-rag query "batch upsert nodes"` |
| 📍 **Definition Lookup** | Jump to class/function by name | `ast-rag goto EmbeddingManager` |
| 📞 **Call Graph** | Find callers/callees | `ast-rag callers build_embeddings --depth 2` |
| 📋 **Find References** | Find all symbol usages | `ast-rag refs UserService` |
| 🎯 **Signature Search** | Search by function pattern | `ast-rag sig "process(int, String)"` |

## 🌐 Supported Languages

| Language | Depth | Features |
|----------|-------|----------|
| **Java** | ⭐⭐⭐ Full | Classes, interfaces, methods, DI, inheritance, overrides |
| **C++** | ⭐⭐⭐ Full | Classes, templates, virtual calls, lambdas |
| **Rust** | ⭐⭐⭐ Full | Structs, traits, impls, generics, macros |
| **Python** | ⭐⭐ Good | Classes, functions, imports, type hints |
| **TypeScript** | ⭐⭐ Good | Classes, interfaces, functions, imports |

## 📦 Installation

```bash
# 1. Clone repository
git clone <repo> && cd raged

# 2. Create virtual environment
python -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -e .

# 4. Start Neo4j and Qdrant (Docker)
docker run -d --name neo4j -p 7687:7687 -p 7474:7474 neo4j:latest
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest
```

## ⚙️ Configuration

Create `ast_rag_config.json` in project root:

```json
{
  "neo4j": {
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "your_password"
  },
  "qdrant": {
    "url": "http://localhost:6333",
    "collection_name": "ast_rag_nodes"
  },
  "embedding": {
    "model_name": "bge-m3",
    "remote_url": "http://localhost:1113/v1/embeddings"
  }
}
```

## 🎯 Quick Start

```bash
# 1. Index project
ast-rag init /path/to/codebase

# 2. Find definition
ast-rag goto MyClass

# 3. Find callers
ast-rag callers my_function --depth 2

# 4. Semantic search
ast-rag query "API request handling"

# 5. Check quality
ast-rag evaluate --all
```

## 📚 Documentation

| Document | Description |
|----------|----------|
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | ⭐ **Start here** — detailed quick start |
| [docs/configuration.md](docs/configuration.md) | Configuration and troubleshooting |
| [docs/python-api.md](docs/python-api.md) | Python API for scripts |
| [docs/graph-schema.md](docs/graph-schema.md) | Neo4j graph schema |
| [AGENTS.md](AGENTS.md) | Guide for AI agents |
| [scripts/README.md](scripts/README.md) | Indexing utilities |
| [tests/README.md](tests/README.md) | Tests and benchmarks |

## 🛠️ CLI Commands

```
ast-rag init <path>              # Full indexing
ast-rag update <path>            # Update from git diff
ast-rag query "<text>"           # Semantic search
ast-rag goto <name>              # Find definition
ast-rag callers <name>           # Find callers
ast-rag refs <name>              # Find references
ast-rag sig <pattern>            # Signature search
ast-rag evaluate                 # Quality evaluation
ast-rag index-folder <path>      # Index a folder
ast-rag workspace <path>         # Show workspace changes
ast-rag sandbox <lang> <cmd>     # Run in Docker sandbox
```

## 📊 Quality

Current metrics (Phase 2):

| Metric | Target | Actual |
|--------|--------|--------|
| **Pass Rate** | >80% | **100%** ✅ |
| **F1 Score** | >0.85 | **0.98** ✅ |
| **Precision** | >0.85 | **0.98** ✅ |
| **Recall** | >0.85 | **0.97** ✅ |

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│           Input (codebase)              │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  ParserManager (Tree-sitter)            │
│    └─ language_queries.py               │
└────────────────┬────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌──────────────┐  ┌──────────────┐
│   Neo4j      │  │   Qdrant     │
│  (graph)     │  │  (vectors)   │
└──────────────┘  └──────────────┘
        │                 │
        └────────┬────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│         ASTRagAPI (query layer)         │
└────────────────┬────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌──────────────┐  ┌──────────────┐
│  CLI (Typer) │  │  MCP Server  │
└──────────────┘  └──────────────┘
```

## 🔧 For Developers

```bash
# Run tests
pytest tests/ -v

# Check quality
ast-rag evaluate --all

# Index a folder
ast-rag index-folder ./ast_rag --no-schema

# Update after changes
ast-rag workspace . --apply
```

## Additionaly docs
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/lexasub/raged)

## 📝 License

LGPL v3
