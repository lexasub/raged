# AGENTS.md — Guide for AI Agents

**For AI agents:** Use AST-RAG to analyze this codebase. Don't guess — look up via tools.

---

## 🎯 Quick Start

**If unsure where to start:**

```bash
# 1. Find code by description
ast-rag query "batch upsert neo4j nodes"

# 2. Go to definition
ast-rag goto <found_name> --snippet

# 3. Find callers
ast-rag callers <name> --depth 2
```

---

## 🛠️ Available Tools

### CLI (recommended)

| Command | Example | When to use |
|---------|---------|-------------|
| `query` | `ast-rag query "batch database operations"` | Search by description |
| `goto` | `ast-rag goto EmbeddingManager --snippet` | Go to definition |
| `callers` | `ast-rag callers build_embeddings --depth 2` | Find callers |
| `refs` | `ast-rag refs UserService` | Find all usages |
| `sig` | `ast-rag sig "process(int, String)"` | Signature search |
| `evaluate` | `ast-rag evaluate --all` | Check quality |
| `index-folder` | `ast-rag index-folder ./ast_rag` | Index a folder |
| `workspace` | `ast-rag workspace . --apply` | Update from git diff |

### Python API

```python
from ast_rag.ast_rag_api import ASTRagAPI
from ast_rag.models import ProjectConfig
from ast_rag.graph_schema import create_driver
from ast_rag.embeddings import EmbeddingManager

# Initialize
cfg = ProjectConfig()
driver = create_driver(cfg.neo4j)
embed = EmbeddingManager(cfg.qdrant, cfg.embedding, neo4j_driver=driver)
api = ASTRagAPI(driver, embed)

# Semantic search
results = api.search_semantic("batch upsert nodes", limit=5)

# Find definition
nodes = api.find_definition("EmbeddingManager", kind="Class")

# Find callers
callers = api.find_callers(nodes[0].id, max_depth=2)

# Get code
code = api.get_code_snippet(nodes[0].file_path, 
                            nodes[0].start_line, 
                            nodes[0].end_line)
```

---

## 📋 Typical Scenarios

### Scenario 1: Find code to modify

```bash
# 1. Find by description
ast-rag query "batch insert nodes transaction"

# 2. Go to definition
ast-rag goto batch_upsert_nodes --snippet

# 3. Check callers
ast-rag callers batch_upsert_nodes --depth 2
```

### Scenario 2: Refactoring

```bash
# 1. Find all usages
ast-rag refs old_method_name --kind Method

# 2. Check impact
ast-rag callers old_method_name --depth 3

# 3. Get context
ast-rag goto old_method_name --snippet
```

### Scenario 3: Understand architecture

```bash
# 1. Find main modules
ast-rag query "main entry point initialization"

# 2. Build call graph
ast-rag callers main_function --depth 3

# 3. Find dependencies
ast-rag refs ModuleName --kind Class
```

### Scenario 4: Add tests

```bash
# 1. Find module without tests
ast-rag query "graph schema cypher"
ast-rag goto apply_schema --snippet

# 2. Check if test exists
ls tests/test_graph_schema.py 2>/dev/null || echo "NO TEST"

# 3. Generate test (agent writes)
```

---

## 🔧 Configuration

**Check config:**

```bash
cat ast_rag_config.json
```

**Check connection:**

```bash
# Neo4j
cypher-shell "MATCH (n) RETURN count(n)"

# Qdrant
curl http://localhost:6333/collections
```

**Check index:**

```bash
# How many nodes in graph
cypher-shell "MATCH (n) RETURN count(n)"

# How many files indexed
grep "COMPLETE" /tmp/index_*.log | wc -l
```

---

## ✅ Best Practices

### ✅ DO:

1. **Always search via AST-RAG** before making changes
2. **Check callers** before refactoring
3. **Use `--snippet`** to get code
4. **Run `evaluate`** after indexing
5. **Index by folders** for large changes

### ❌ DON'T:

1. ❌ Don't guess code location
2. ❌ Don't use grep for search (inaccurate)
3. ❌ Don't change code without checking callers
4. ❌ Don't ignore `evaluate` (risk of degradation)

---

## 📊 Quality Check

**Before commit:**

```bash
# Run evaluation
ast-rag evaluate --all

# Expected:
# Pass Rate: >80%
# F1 Score: >0.85
```

**If quality dropped:**

```bash
# Check what's indexed
grep "COMPLETE" /tmp/index_*.log | wc -l

# Index remaining
./scripts/index-remaining.sh

# Run again
ast-rag evaluate --all
```

---

## 📚 Documentation

| Document | Description |
|----------|----------|
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | Quick start guide |
| [docs/agent-scenarios.md](docs/agent-scenarios.md) | Detailed scenarios |
| [docs/python-api.md](docs/python-api.md) | Python API |
| [scripts/README.md](scripts/README.md) | Indexing utilities |

---

## 🆘 Help

```bash
# All commands
ast-rag --help

# Help for command
ast-rag query --help
ast-rag callers --help

# Examples
ast-rag evaluate --help
```
