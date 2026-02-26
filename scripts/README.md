# AST-RAG Scripts

Utility scripts for AST-RAG indexing and maintenance.

---

## 📁 Available Scripts

### `index-sequential.sh`

Index all folders sequentially (one at a time).

**Use case:** Stable indexing without parallelism issues.

```bash
./scripts/index-sequential.sh
```

**Features:**
- Applies Neo4j schema once at the beginning
- Indexes folders one by one (no parallel conflicts)
- Skips already completed folders (checks `/tmp/index_*.log`)
- Runs quality evaluation at the end

**Time:** ~3-4 hours for full codebase

---

### `index-remaining.sh`

Index only folders that haven't been completed yet.

**Use case:** Resume interrupted indexing.

```bash
./scripts/index-remaining.sh
```

**Features:**
- Automatically detects completed folders from logs
- Only indexes remaining folders
- Applies schema once
- Runs quality evaluation at the end

**Time:** Depends on how many folders remain

---

## 🔧 CLI Commands

Most indexing tasks are now available via CLI:

### Index a single folder
```bash
ast-rag index-folder /path/to/folder --workers 4 --batch-size 50
```

### Skip schema application (if already applied)
```bash
ast-rag index-folder /path/to/folder --no-schema
```

### Run quality evaluation
```bash
ast-rag evaluate --all
```

### Run single benchmark
```bash
ast-rag evaluate --query benchmarks/queries/def_001.json
```

---

## 📊 Typical Workflow

### Fresh installation
```bash
# 1. Full index
ast-rag init /path/to/codebase

# 2. Verify quality
ast-rag evaluate --all
```

### Update after code changes
```bash
# Index specific changed folder
ast-rag index-folder ./src/modified_module --no-schema
```

### Resume interrupted indexing
```bash
# Index remaining folders
./scripts/index-remaining.sh
```

---

## 🎯 Best Practices

1. **Use CLI for single folders** - `ast-rag index-folder`
2. **Use scripts for bulk operations** - `./scripts/index-remaining.sh`
3. **Always run evaluation after indexing** - `ast-rag evaluate --all`
4. **Target >80% pass rate** for production use

---

## 📝 Log Files

- **Folder logs:** `/tmp/index_*.log` (one per folder)
- **Script logs:** `index_sequential.log`, `index_remaining.log`
- **Evaluation results:** `benchmarks/results/evaluation.json`

---

## 🐛 Troubleshooting

### Process dies during indexing
- Check Neo4j connection: `cypher-shell "MATCH (n) RETURN count(n)"`
- Check memory: `free -h`
- Try sequential indexing: `./scripts/index-sequential.sh`

### Quality is low (<70%)
- Re-run evaluation: `ast-rag evaluate --all`
- Check if all folders indexed: `grep "COMPLETE" /tmp/index_*.log | wc -l`
- Index remaining: `./scripts/index-remaining.sh`

### Neo4j schema errors
- Apply schema manually: `ast-rag init /tmp/empty --commit test`
- Or use `--no-schema` flag for subsequent indexing
