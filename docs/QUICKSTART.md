# AST-RAG Quickstart

**5 minutes to first result**

---

## ⚡ Installation in 2 minutes

### 1. Install dependencies

```bash
# Clone repository
cd /path/to/raged

# Activate venv
source venv/bin/activate

# Install package
pip install -e .
```

### 2. Start services (Docker)

```bash
# Neo4j
docker run -d --name neo4j \
  -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# Qdrant
docker run -d --name qdrant \
  -p 6333:6333 \
  qdrant/qdrant:latest

# Embedding server (optional, for bge-m3)
docker run -d --name embeddings \
  -p 1113:1113 \
  your-embedding-image
```

### 3. Check connection

```bash
# Neo4j
cypher-shell -u neo4j -p password "RETURN 1"

# Qdrant
curl http://localhost:6333/collections
```

---

## 🎯 First Run

### 1. Create config

```bash
cat > ast_rag_config.json <<EOF
{
  "neo4j": {
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "password"
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
EOF
```

### 2. Index project

```bash
# Full indexing
ast-rag init /path/to/your/codebase

# Or specific folder
ast-rag index-folder ./src
```

**Indexing time:**
- Small project (<100 files): ~1 min
- Medium project (100-500 files): ~5-10 min
- Large project (>500 files): ~30+ min

### 3. Check quality

```bash
ast-rag evaluate --all
```

**Expected result:**
```
📊 Benchmarks: 10
✅ Passed: 10
❌ Failed: 0
📈 Pass Rate: 100.0%
📈 F1 Score: 0.98
```

---

## 🔍 First Usage

### Find definition

```bash
ast-rag goto MyClass
ast-rag goto my_function --lang python
ast-rag goto UserService --snippet
```

### Find callers

```bash
ast-rag callers my_function
ast-rag callers MyClass.my_method --depth 2
```

### Semantic search

```bash
ast-rag query "handle HTTP requests"
ast-rag query "batch database operations"
ast-rag query "input validation" --lang java
```

### Find all usages

```bash
ast-rag refs UserService
ast-rag refs processRequest --kind Method
```

### Signature search

```bash
ast-rag sig "process(int, String)"
ast-rag sig "get*" --lang java
```

---

## 🔄 Update Index

### After code changes

```bash
# Show changes
ast-rag workspace .

# Apply changes
ast-rag workspace . --apply
```

### Update from git

```bash
# Update from git diff
ast-rag update . --from HEAD~1 --to HEAD

# Update current branch
ast-rag update . --current-branch
```

---

## 📊 Typical Scenarios

### Scenario 1: Refactoring

**Task:** Rename a method, find all usages

```bash
# 1. Find all usages
ast-rag refs my_method --kind Method

# 2. Check callers
ast-rag callers my_method --depth 2

# 3. Get snippet for editing
ast-rag goto my_method --snippet
```

### Scenario 2: Impact Analysis

**Task:** Understand what breaks when changing a class

```bash
# 1. Find definition
ast-rag goto MyClass

# 2. Find all callers
ast-rag callers MyClass --depth 3

# 3. Find all references
ast-rag refs MyClass
```

### Scenario 3: Search code by description

**Task:** Find where API requests are handled

```bash
ast-rag query "handle API request response"
ast-rag query "HTTP client GET POST" --lang python
```

---

## 🐛 Troubleshooting

### `ast-rag: command not found`

```bash
# Reinstall package
pip install -e .

# Check PATH
echo $PATH | grep venv
```

### Neo4j connection refused

```bash
# Check status
docker ps | grep neo4j

# Restart
docker restart neo4j

# Check logs
docker logs neo4j
```

### Empty search results

```bash
# Check if graph is indexed
cypher-shell -u neo4j -p password "MATCH (n) RETURN count(n)"

# Re-index
ast-rag init /path/to/codebase
```

### Low quality (<70%)

```bash
# Check how many indexed
grep "COMPLETE" /tmp/index_*.log | wc -l

# Index remaining
./scripts/index-remaining.sh

# Run evaluation again
ast-rag evaluate --all
```

---

## 📚 Next Steps

1. **Learn CLI commands** — `ast-rag --help`
2. **Check scenarios** — [docs/agent-scenarios.md](agent-scenarios.md)
3. **Configure for project** — [docs/configuration.md](configuration.md)
4. **Use Python API** — [docs/python-api.md](python-api.md)
5. **For AI agents** — [AGENTS.md](../AGENTS.md)

---

## 🎯 Next Commands to Explore

```bash
# Show all commands
ast-rag --help

# Help for specific command
ast-rag query --help
ast-rag callers --help

# Examples
ast-rag evaluate --help
```
