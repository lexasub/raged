# Python API

Programmatic access to AST-RAG from scripts and applications.

---

## 🚀 Quick Start

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

# Don't forget to close after use
# driver.close()
```

---

## 🔍 Search

### Semantic Search

```python
results = api.search_semantic("batch upsert nodes", limit=10)

for r in results:
    print(f"{r.score:.3f}  {r.node.name}  {r.node.file_path}:{r.node.start_line}")
```

### Signature Search

```python
# Exact pattern
results = api.search_by_signature("process(int, String)", lang="java")

# With wildcard
results = api.search_by_signature("get*", lang="python")

for r in results:
    print(f"{r.file_path}:{r.start_line} {r.name}")
```

### Find Definition

```python
# By name
nodes = api.find_definition("EmbeddingManager")

# With filter
nodes = api.find_definition("UserService", kind="Class", lang="java")

for n in nodes:
    print(f"{n.qualified_name}  {n.file_path}:{n.start_line}")
```

### Find References

```python
# All usages
refs = api.find_references("processRequest", kind="Method")

for ref in refs:
    print(f"{ref.node.file_path}:{ref.node.start_line}  {ref.reference_type}")
```

---

## 📞 Call Graph

### Find Callers

```python
# Find node
node = api.find_definition("build_embeddings")[0]

# Find callers (depth 2)
callers = api.find_callers(node.id, max_depth=2)

for caller in callers:
    print(f"{caller.file_path}:{caller.start_line} {caller.name}")
```

### Find Callees

```python
callees = api.find_callees(node.id, max_depth=1)

for callee in callees:
    print(f"{callee.file_path}:{callee.start_line} {callee.name}")
```

### Expand Subgraph

```python
subgraph = api.expand_neighbourhood(
    node.id,
    depth=2,
    edge_types=["CALLS", "CONTAINS_METHOD"]
)

print(f"Nodes: {len(subgraph.nodes)}, Edges: {len(subgraph.edges)}")
```

---

## 📄 Working with Code

### Get Snippet

```python
node = api.find_definition("MyClass")[0]

code = api.get_code_snippet(
    node.file_path,
    node.start_line,
    node.end_line
)

print(code)
```

### Get Diff Between Versions

```python
diff = api.get_diff(
    from_commit="abc123",
    to_commit="def456",
    file_path="src/main.py"
)

print(f"Added: {diff.added}, Removed: {diff.removed}")
```

---

## 📊 Quality Evaluation

```python
from ast_rag.benchmarks.evaluator import BenchmarkEvaluator

evaluator = BenchmarkEvaluator()

# Run all benchmarks
results = evaluator.run_all()

print(f"Pass Rate: {results['pass_rate']*100:.1f}%")
print(f"F1 Score: {results['average_metrics']['f1_score']:.2f}")
```

---

## 🔄 Indexing

### Full Indexing

```python
from ast_rag.graph_updater import full_index
from ast_rag.graph_schema import apply_schema

# Apply schema
apply_schema(driver)

# Index
stats = full_index(driver, "/path/to/codebase", commit="v1.0")

print(f"Nodes: {stats.nodes}, Edges: {stats.edges}")
```

### Index Folder

```python
from ast_rag.graph_updater import index_directory

stats = index_directory(
    driver,
    "/path/to/folder",
    exclude_patterns=[".git", "venv", "__pycache__"]
)
```

### Update from Git Diff

```python
from ast_rag.graph_updater import update_from_git

diff_stats = update_from_git(
    driver,
    root="/path/to/codebase",
    from_commit="HEAD~1",
    to_commit="HEAD"
)

print(f"Changed: +{diff_stats.added}, -{diff_stats.deleted}")
```

### Update Workspace

```python
from ast_rag.graph_updater import get_workspace_diff, apply_workspace_diff

# Get changes
diff = get_workspace_diff(driver, root=".")

if not diff.is_empty:
    print(f"+{len(diff.added_nodes)} nodes, +{len(diff.added_edges)} edges")
    
    # Apply
    apply_workspace_diff(driver, root=".")
```

---

## 🔧 Utilities

### Check Connection

```python
from ast_rag.graph_schema import create_driver
from ast_rag.models import ProjectConfig

cfg = ProjectConfig()
driver = create_driver(cfg.neo4j)

# Check
with driver.session() as session:
    result = session.run("RETURN 1")
    print("Neo4j connected:", result.single()[0])

driver.close()
```

### Graph Statistics

```python
with driver.session() as session:
    # Total nodes
    count = session.run("MATCH (n) RETURN count(n)").single()[0]
    print(f"Total nodes: {count}")
    
    # By type
    result = session.run("""
        MATCH (n) 
        RETURN labels(n)[0] as label, count(n) as count
        ORDER BY count DESC
    """)
    for row in result:
        print(f"{row['label']}: {row['count']}")
```

---

## 📚 Examples

### Example 1: Impact Analysis

```python
# Find class
node = api.find_definition("UserService")[0]

# Find all callers
callers = api.find_callers(node.id, max_depth=3)

# Assess scope
print(f"Affected functions: {len(callers)}")

# Get code for analysis
for caller in callers[:5]:  # First 5
    code = api.get_code_snippet(caller.file_path, 
                                caller.start_line, 
                                caller.end_line)
    print(f"\n{caller.file_path}:{caller.start_line}")
    print(code[:200] + "...")
```

### Example 2: Find Duplicates

```python
# Find all functions with similar name
results = api.search_semantic("validate user input", limit=20)

# Group by name
from collections import defaultdict
by_name = defaultdict(list)

for r in results:
    by_name[r.node.name].append(r.node)

# Find duplicates
for name, nodes in by_name.items():
    if len(nodes) > 1:
        print(f"\n{name}: {len(nodes)} variants")
        for n in nodes:
            print(f"  - {n.file_path}:{n.start_line}")
```

### Example 3: Generate Documentation

```python
# Find all public classes
classes = api.find_definition("", kind="Class")

for cls in classes:
    if not cls.name.startswith("_"):
        # Find methods
        subgraph = api.expand_neighbourhood(
            cls.id, 
            depth=1, 
            edge_types=["CONTAINS_METHOD"]
        )
        
        print(f"\n## {cls.name}")
        print(f"File: {cls.file_path}\n")
        
        for method in subgraph.nodes:
            if method.node_type == "Method":
                print(f"- `{method.name}`")
```

---

## 🆘 Troubleshooting

### Connection Error

```python
try:
    driver = create_driver(cfg.neo4j)
except Exception as e:
    print(f"Neo4j connection failed: {e}")
    # Check config and if Neo4j is running
```

### Empty Results

```python
results = api.search_semantic("test")
if not results:
    # Check if graph is indexed
    with driver.session() as session:
        count = session.run("MATCH (n) RETURN count(n)").single()[0]
        print(f"Nodes in graph: {count}")
```

---

## 📚 See Also

- [docs/QUICKSTART.md](QUICKSTART.md) — Quick start
- [AGENTS.md](../AGENTS.md) — Guide for AI agents
- [docs/configuration.md](configuration.md) — Configuration
