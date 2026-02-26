# Agent Scenarios — Practical Guide

**Ready-to-use patterns for working with AST-RAG**

---

## 🔍 Scenario 1: Find Code to Modify

**Task:** Need to change logic — where is it?

```bash
# 1. Semantic search
ast-rag query "batch upsert neo4j nodes"
ast-rag query "handle HTTP response parsing"

# 2. Go to definition
ast-rag goto <found_name> --snippet

# 3. Check context (callers)
ast-rag callers <name> --depth 2
```

**Python API:**
```python
results = api.search_semantic("batch upsert nodes", limit=5)
code = api.get_code_snippet(results[0].node.file_path,
                            results[0].node.start_line,
                            results[0].node.end_line)
```

---

## 🔧 Scenario 2: Refactor Method

**Task:** Rename/delete method, find all usages

```bash
# 1. Find all references
ast-rag refs processRequest --kind Method

# 2. Find callers (depth 2)
ast-rag callers processRequest --depth 2

# 3. Get code for changes
ast-rag goto processRequest --snippet

# 4. Check impact after changes
ast-rag evaluate --all
```

**Python API:**
```python
# Find all references
refs = api.find_references("processRequest", kind="Method")

# Find callers
node = api.find_definition("processRequest")[0]
callers = api.find_callers(node.id, max_depth=2)
```

---

## 📊 Scenario 3: Impact Analysis

**Task:** What breaks when changing a class?

```bash
# 1. Find definition
ast-rag goto UserService

# 2. Find all callers (depth 3)
ast-rag callers UserService --depth 3

# 3. Find all references
ast-rag refs UserService --kind Class

# 4. Assess scope
# Check result count
```

**Python API:**
```python
# Find node
node = api.find_definition("UserService")[0]

# Find callers
callers = api.find_callers(node.id, max_depth=3)

# Find neighbors (full context)
subgraph = api.expand_neighbourhood(node.id, depth=2)
print(f"Affected nodes: {len(subgraph.nodes)}")
```

---

## 🧪 Scenario 4: Find Code Without Tests

**Task:** Create tests for uncovered modules

```bash
# 1. Find module
ast-rag query "graph schema cypher application"
ast-rag goto apply_schema --snippet

# 2. Check if test exists
ls tests/test_apply_schema.py 2>/dev/null || echo "NO TEST"

# 3. Find callers (understand usage)
ast-rag callers apply_schema --depth 2

# 4. Generate test (agent writes)
```

---

## 🗂️ Scenario 5: Understand Module Architecture

**Task:** Understand unfamiliar code

```bash
# 1. Find entry point
ast-rag query "main initialization entry point"

# 2. Build call graph from main
ast-rag callers main --depth 3

# 3. Find main dependencies
ast-rag refs ModuleName --kind Class

# 4. Export graph (optional)
# Via Python API:
# subgraph = api.expand_neighbourhood(node_id, depth=3)
```

---

## 🔄 Scenario 6: Update Index After Changes

**Task:** Code changed — update graph

```bash
# 1. Show changes
ast-rag workspace .

# 2. Apply changes
ast-rag workspace . --apply

# 3. Check quality
ast-rag evaluate --all
```

**For large changes:**
```bash
# Index specific folder
ast-rag index-folder ./src/modified_module --no-schema

# Or all remaining
./scripts/index-remaining.sh
```

---

## 🎯 Scenario 7: Signature Search

**Task:** Find function by parameter pattern

```bash
# Exact pattern
ast-rag sig "process(int, String)"

# With wildcard
ast-rag sig "get*" --lang java
ast-rag sig "*Handler" --lang python

# With filter
ast-rag sig "build*" --lang rust --kind Function
```

**Python API:**
```python
results = api.search_by_signature("process(int, String)", limit=10)
for r in results:
    print(f"{r.file_path}:{r.start_line} {r.name}")
```

---

## 📈 Scenario 8: Check Search Quality

**Task:** Ensure AST-RAG finds correctly

```bash
# Run all benchmarks
ast-rag evaluate --all

# Expected:
# ✅ Pass Rate: >80%
# ✅ F1 Score: >0.85

# Run specific benchmark
ast-rag evaluate --query benchmarks/queries/def_001.json
```

**If quality is low:**
```bash
# Check how many indexed
grep "COMPLETE" /tmp/index_*.log | wc -l

# Index remaining
./scripts/index-remaining.sh

# Check again
ast-rag evaluate --all
```

---

## 🐛 Scenario 9: Debug Empty Results

**Task:** Search returns nothing

```bash
# 1. Check graph
cypher-shell "MATCH (n) RETURN count(n)"

# 2. Check specific type
cypher-shell "MATCH (n:Method) RETURN count(n)"

# 3. Check indexing
grep "COMPLETE" /tmp/index_*.log | wc -l

# 4. Re-index
ast-rag init /path/to/codebase

# 5. Check again
ast-rag query "test query"
```

---

## 📚 Scenario 10: Self-Optimization of AST-RAG

**Task:** Improve AST-RAG performance

```bash
# 1. Find bottleneck
ast-rag query "batch neo4j transaction performance"
ast-rag goto batch_upsert_nodes --snippet

# 2. Analyze callers
ast-rag callers batch_upsert_nodes --depth 2

# 3. Get full context
# Python API:
# subgraph = api.expand_neighbourhood(node_id, depth=2)

# 4. Optimize code
# ... changes ...

# 5. Run tests
ast-rag sandbox python . --cmd "pytest tests/ -v"

# 6. Check quality
ast-rag evaluate --all
```

---

## 🎯 Command Cheat Sheet

| Task | Command |
|------|---------|
| Find code | `ast-rag query "description"` |
| Go to definition | `ast-rag goto <name> --snippet` |
| Find callers | `ast-rag callers <name> --depth N` |
| Find usages | `ast-rag refs <name>` |
| Signature search | `ast-rag sig "pattern(params)"` |
| Update index | `ast-rag workspace . --apply` |
| Check quality | `ast-rag evaluate --all` |
| Index folder | `ast-rag index-folder ./path` |

---

## 📚 See Also

- [AGENTS.md](../AGENTS.md) — Guide for AI agents
- [docs/QUICKSTART.md](QUICKSTART.md) — Quick start
- [docs/python-api.md](python-api.md) — Python API
