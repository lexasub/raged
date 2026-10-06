# AST-RAG — Code Analysis & Navigation System

[![gitcgr](https://gitcgr.com/badge/lexasub/raged.svg)](https://gitcgr.com/lexasub/raged)

Context-aware code intelligence system for AI agents and developers. Provides semantic code search, definition lookup, call graph analysis, and codebase navigation powered by AST parsing (Tree-sitter), graph database (Neo4j), and vector embeddings (Qdrant).

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

| Language | Extensions | Depth | Features |
|----------|------------|-------|----------|
| **Java** | `.java` | ⭐⭐⭐ Full | Classes, interfaces, methods, DI, inheritance, overrides |
| **C++** | `.cpp` `.cxx` `.cc` `.c` `.hpp` `.hxx` `.hh` `.h` | ⭐⭐⭐ Full | Classes, templates, virtual calls, lambdas |
| **Rust** | `.rs` | ⭐⭐⭐ Full | Structs, traits, impls, generics, macros |
| **Python** | `.py` | ⭐⭐ Good | Classes, functions, imports, type hints |
| **TypeScript** | `.ts` | ⭐⭐ Good | Classes, interfaces, functions, imports |
| **TSX / JSX** | `.tsx` `.jsx` | ⭐⭐ Good | Classes, interfaces, functions, imports, JSX elements |
| **Go** | `.go` | ⭐⭐ Good | Functions, methods, structs, interfaces, imports |

Files with other extensions are skipped during indexing with a warning listing the supported languages.

## 📦 Installation

```bash
# 1. Clone and install
git clone https://github.com/lexasub/raged && cd raged
python -m venv venv && source venv/bin/activate
pip install -e .

# 2. Start Neo4j and Qdrant
docker compose up -d

# 3. Index and query — no config file needed
ast-rag index-folder ./ast_rag
ast-rag query "batch upsert nodes"
```

The defaults point at the services `docker compose up -d` starts, so this works
on a fresh clone with no configuration. To point elsewhere, set environment
variables rather than editing a tracked file:

```bash
export AST_RAG_NEO4J_URI=bolt://myhost:7687
export AST_RAG_NEO4J_PASSWORD=secret
export AST_RAG_QDRANT_URL=http://myhost:6333
```

Or copy `ast_rag_config.example.json` to `ast_rag_config.json` and edit it
(it is gitignored).

### Optional extras

`pip install -e .` gives you the `ast-rag` CLI. The other two console scripts
need extras:

```bash
pip install -e ".[mcp]"     # ast-rag-mcp (MCP server), ast-rag-watch (file watcher)
pip install -e ".[server]"  # embedding_server.py, for the GPU host
pip install -e ".[dev]"     # pytest, mypy, ruff
```

Without `[mcp]`, `ast-rag-mcp` and `ast-rag-watch` are on `PATH` but fail with
`ModuleNotFoundError: No module named 'mcp'` / `'watchdog'`.

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
    "model_name": "BAAI/bge-m3"
  }
}
```

Embeddings are computed locally by default. To offload them to a separate
embedding server, add `remote_url` — and `dimension` with it, which is required
whenever `remote_url` is set (`docker compose up -d` does not start such a
server):

```json
"embedding": {
  "model_name": "bge-m3",
  "remote_url": "http://localhost:1113/v1/embeddings",
  "dimension": 1024
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
| [docs/IGNORE_FILES.md](docs/IGNORE_FILES.md) | Excluding files from indexing (`.cgrignore`) |
| [docs/python-api.md](docs/python-api.md) | Python API for scripts |
| [docs/graph-schema.md](docs/graph-schema.md) | Neo4j graph schema |
| [AGENTS.md](AGENTS.md) | Guide for AI agents |
| [scripts/README.md](scripts/README.md) | Indexing utilities |
| [tests/README.md](tests/README.md) | Tests and benchmarks |

## 🛠️ CLI Commands

```
ast-rag init <path>                    # Full indexing
ast-rag index-folder <path>            # Index a folder
ast-rag create-database <name>         # Create a Neo4j database
ast-rag update <path> --from-commit <a> --to-commit <b>
                                       # Update from git diff a..b
ast-rag workspace <path>               # Show workspace changes
ast-rag query "<text>"                 # Semantic search
ast-rag goto <name>                    # Find definition
ast-rag callers <name>                 # Find callers
ast-rag refs <name>                    # Find references
ast-rag call-graph <name>              # Callers/callees graph around a function
ast-rag symbol-impact <name>           # Definition + refs + callers + callees
ast-rag sig <pattern>                  # Signature search
ast-rag blocks <function>              # if/for/while/try/lambda/with blocks
ast-rag lambdas                        # List lambda/closure blocks
ast-rag summarize <name>               # LLM summary of a function/method/class
ast-rag analyze-stacktrace [file]      # Map a stack trace onto the graph
ast-rag stats                          # Statistics about the indexed codebase
ast-rag cache-stats                    # Parse cache statistics
ast-rag evaluate                       # Quality evaluation
ast-rag sandbox <lang> [workdir] --cmd "<command>"
                                       # Run in Docker sandbox
```

`ast-rag <command> --help` lists the flags for each.

## 📊 Quality

Phase 2 benchmark run, recorded 2026-02-26 and not re-measured since:

| Metric | Target | Measured |
|--------|--------|--------|
| **Pass Rate** | >80% | **100%** ✅ |
| **F1 Score** | >0.85 | **0.98** ✅ |
| **Precision** | >0.85 | **0.98** ✅ |
| **Recall** | >0.85 | **0.97** ✅ |

The benchmarks in `benchmarks/queries/` are ground-truthed against AST-RAG's own
source, so they measure this repo rather than yours. To reproduce, index this
repo with Neo4j and Qdrant running, then from the repo root:

```bash
ast-rag evaluate --all
```

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

## 🚧 Roadmap

Planned features and improvements:

| Feature | Status | Description |
|---------|--------|-------------|
| **Code Summaries** | ✅ Done | `ast-rag summarize` — LLM summaries for functions/classes ([docs](docs/SUMMARIZATION.md)) |
| **Refactoring Hints** | 🔜 Planned | Detect code smells and suggest improvements |
| **More Languages** | 🔜 Planned | C#, Kotlin with full AST support |
| **IDE Integration** | 🔄 In Progress | MCP, skills, CLI for OpenCode, Kilocode, Claude Code, Cursor |
| **Incremental Indexing** | ✅ Done | Git-based and filesystem watcher updates (improving for large codebases) |
| **Rust Rewrite** | 🔜 Planned | Full rewrite in Rust for performance, type safety, and easier integration |
| **AST Patching** | 🔜 Future Project | Separate project for generating code patches from AST |
| **Multi-Project Support** | 🔜 Planned | Work across multiple related projects (microservices, monorepos) |
| **Vector DB Flexibility** | 🔜 Planned | Support for alternative vector stores (Chroma, etc.) and distance metrics |

## 📝 License

LGPL v3
