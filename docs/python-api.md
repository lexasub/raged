# Python API

Programmatic access to AST-RAG from scripts and applications.

---

## 🚀 Quick Start

```python
from ast_rag.ast_rag_api import ASTRagAPI
from ast_rag.models import ProjectConfig
from ast_rag.repositories import create_driver
from ast_rag.services import EmbeddingManager

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

`find_references` returns a paginated dict, and each reference is a plain dict:

```python
# All usages
result = api.find_references("processRequest", kind="Method", limit=50, offset=0)
# {"references": [...], "total": int, "limit": int, "offset": int, "has_more": bool}

for ref in result["references"]:
    node = ref["node"]
    print(f"{node['file_path']}:{node['start_line']}  {ref['reference_type']}")
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
    repo_path="/path/to/codebase",
    from_commit="abc123",
    to_commit="def456",   # defaults to HEAD
)

print(f"Added: {diff['added_count']}, Deleted: {diff['deleted_count']}, "
      f"Updated: {diff['updated_count']}")
```

`get_diff` covers the whole repository — there is no per-file filter — and
returns a dict with `added` / `deleted` / `updated` node lists alongside those
counts, plus `limit`, `offset` and `has_more`.

---

## 📊 Quality Evaluation

There is no Python entry point for the benchmarks — they run through the CLI,
from the repo root:

```bash
ast-rag evaluate --all --output results.json
```

The JSON it writes (`benchmarks/results/evaluation.json` by default) can be read
back directly:

```python
import json

results = json.load(open("benchmarks/results/evaluation.json"))

print(f"Pass Rate: {results['pass_rate']*100:.1f}%")
print(f"F1 Score: {results['average_metrics']['f1_score']:.2f}")
```

---

## 🔄 Indexing

There is no single "index this directory" function — walking files, parsing and
writing to Neo4j are separate steps. For a whole codebase, use the CLI
(`ast-rag init <path>` / `ast-rag index-folder <path>`); the pieces below are for
building on top of it.

### Parse a File

```python
from ast_rag.services.parsing import ParserManager

pm = ParserManager()

path = "/path/to/file.py"
lang = pm.detect_language(path)             # "python", "go", "java", ... or None
source = open(path, "rb").read()

tree = pm.parse_file(path, source=source)
nodes = pm.extract_nodes(tree, path, lang, source, commit_hash="v1.0")
edges = pm.extract_edges(tree, nodes, path, lang, source, commit_hash="v1.0")
```

### Write to the Graph

```python
from ast_rag.repositories import apply_schema
from ast_rag.graph_updater import full_index

apply_schema(driver)
full_index(driver, nodes, edges, commit_hash="v1.0")   # returns None
```

### Update from Git Diff

```python
from ast_rag.graph_updater import update_from_git

diff = update_from_git(
    driver,
    repo_path="/path/to/codebase",
    old_commit="HEAD~1",
    new_commit="HEAD",
)

print(f"+{len(diff.added_nodes)} nodes, -{len(diff.deleted_node_ids)} nodes")
```

### Update Workspace

```python
from ast_rag.graph_updater import get_workspace_diff, apply_workspace_diff

# Get changes (read-only)
diff = get_workspace_diff(driver, repo_path=".")

if not diff.is_empty:
    print(f"+{len(diff.added_nodes)} nodes, +{len(diff.added_edges)} edges")

    # Apply
    apply_workspace_diff(driver, repo_path=".")
```

All three return a `DiffResult` with `added_nodes`, `deleted_node_ids`,
`updated_nodes`, `added_edges`, `deleted_edge_ids`, `updated_edges` and the
`is_empty` property.

---

## 🔧 Utilities

### Check Connection

```python
from ast_rag.repositories import create_driver
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
            if method.kind == "Method":
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
