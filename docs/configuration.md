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

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
export QDRANT_URL="http://localhost:6333"
export EMBEDDING_URL="http://localhost:1113/v1/embeddings"
```

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
| Deadlock error | Parallel indexing | Use `./scripts/index-sequential.sh` |

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
| Low recall (<70%) | Not all files indexed | `./scripts/index-remaining.sh` |
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
# How many folders indexed
grep "COMPLETE" /tmp/index_*.log | wc -l

# Which folders completed
grep "COMPLETE" /tmp/index_*.log | cut -d: -f2

# Errors
grep "ERROR" /tmp/index_*.log
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
# For CLI
ast-rag query "test" --verbose

# For Python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Indexing Logs

```bash
# Main log
tail -f index_sequential.log

# Per-folder logs
tail -f /tmp/index_*.log

# Errors
grep "ERROR" /tmp/index_*.log | tail -20
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
