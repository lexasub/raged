# Configuration & Troubleshooting

## ⚙️ Configuration

### Config File

`ast_rag_config.json` in project root:

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
    "remote_url": "http://localhost:1113/v1/embeddings",
    "dimension": 1024,
    "remote_batch_size": 32
  }
}
```

### Environment Variables (optional)

These are applied last, on top of anything loaded from `ast_rag_config.json`:

```bash
export AST_RAG_NEO4J_URI="bolt://localhost:7687"
export AST_RAG_NEO4J_USER="neo4j"
export AST_RAG_NEO4J_PASSWORD="password"
export AST_RAG_NEO4J_DATABASE="neo4j"
export AST_RAG_QDRANT_URL="http://localhost:6333"
export AST_RAG_QDRANT_COLLECTION="ast_rag_nodes"
```

That is the complete set. There is no environment override for the embedding
URL — set `embedding.remote_url` in the config file.

---

## 🔧 Troubleshooting

### Installation & Launch

| Issue | Cause | Fix |
|-------|-------|-----|
| `ast-rag: command not found` | Package not installed | `pip install -e .` |
| `ModuleNotFoundError` | Dependencies missing | `pip install -r requirements.txt` |
| `venv not activated` | Virtual env not active | `source venv/bin/activate` |

### Neo4j

| Issue | Cause | Fix |
|-------|-------|-----|
| Connection refused | Neo4j not running | `docker run -d --name neo4j -p 7687:7687 neo4j:latest` |
| Auth failed | Wrong password | Check `NEO4J_PASSWORD` in config |
| Empty results | Graph not indexed | `ast-rag init /path/to/codebase` |
| Deadlock error | Parallel indexing | Index serially: `ast-rag index-folder <path> --workers 1` |

### Qdrant

| Issue | Cause | Fix |
|-------|-------|-----|
| Connection refused | Qdrant not running | `docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest` |
| Version mismatch | Client ≠ server version | Update Qdrant or ignore warning |
| Collection not found | Collection not created | Run `ast-rag init` |

### Embeddings

| Issue | Cause | Fix |
|-------|-------|-----|
| 500 Error | Server overloaded | Reduce `remote_batch_size` |
| Timeout | Slow response | Increase timeout in config |
| Wrong dimension | Model mismatch | Check `model_name` and `dimension` |

### Quality

| Issue | Cause | Fix |
|-------|-------|-----|
| Empty results | Graph not indexed | `ast-rag init .` |
| Low recall (<70%) | Not all files indexed | Check with `ast-rag stats`, then `ast-rag index-folder <path> --no-schema` |
| Low precision | Noise in embeddings | Tune thresholds in `embeddings.py` |
| Stale results | Index outdated | `ast-rag workspace . --apply` |

---

## 🔍 Diagnostics

### Check Connection

```bash
# Neo4j
cypher-shell -u neo4j -p password "RETURN 1"

# Qdrant
curl http://localhost:6333/collections

# Embeddings
curl -X POST http://localhost:1113/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3", "prompt": "test"}'
```

### Check Graph

```bash
# Node count
cypher-shell "MATCH (n) RETURN count(n)"

# File count
cypher-shell "MATCH (f:File) RETURN count(f)"

# Node types
cypher-shell "MATCH (n) RETURN labels(n)[0] as label, count(n) ORDER BY count(n) DESC"

# Edge types
cypher-shell "MATCH ()-[r]->() RETURN type(r) as type, count(r) ORDER BY count(r) DESC"
```

### Check Indexing

```bash
# Node/edge counts, language distribution, indexed file count
ast-rag stats

# Same, as JSON
ast-rag stats --json

# Parse cache hit rate and configuration
ast-rag cache-stats
```

### Check Quality

```bash
# Run evaluation
ast-rag evaluate --all

# View results
cat benchmarks/results/evaluation.json | python -m json.tool
```

---

## 🚀 Performance Optimization

### Neo4j Settings (neo4j.conf)

```properties
# Memory
dbms.memory.heap.initial_size=2g
dbms.memory.heap.max_size=4g

# Page cache
dbms.memory.pagecache.size=2g

# Parallelism
dbms.threads.worker_count=8
```

### Qdrant Settings

```yaml
# config.yaml
performance:
  max_search_threads: 4
  indexing:
    hnsw:
      m: 16
      ef_construct: 100
```

### AST-RAG Settings

```json
{
  "embedding": {
    "remote_batch_size": 32,
    "timeout": 60
  },
  "neo4j": {
    "max_connection_pool_size": 50
  }
}
```

---

## 📊 Logging

### Enable Debug Logs

```bash
# For CLI (init, update, sig, evaluate, index-folder, workspace, summarize, ...)
ast-rag index-folder ./src --verbose

# For Python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Indexing Logs

Indexing logs to stderr; redirect it if you want a file to tail:

```bash
ast-rag index-folder ./src --verbose 2>index.log
tail -f index.log
grep "ERROR" index.log | tail -20
```

---

## 🆘 Help

```bash
# All commands
ast-rag --help

# Help for command
ast-rag query --help
ast-rag index-folder --help

# Documentation
cat docs/QUICKSTART.md
cat AGENTS.md
```
