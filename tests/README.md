# AST-RAG Tests

Tests and benchmarks for AST-RAG.

---

## 📁 Structure

```
tests/
├── test_phase2.py                    # Phase 2 tests (76 tests)
├── test_standard_result.py           # StandardResult tests
├── test_update_project_dry_run.py    # Update dry-run tests
├── test_call_resolution.*            # Call resolution tests (cpp, h, rs)
├── test_rust_queries.rs              # Rust query tests
├── benchmark_hybrid.py               # Hybrid benchmarks
├── generate_ground_truth.py          # Ground truth generation
├── ground_truth_queries.json         # Ground truth data
└── queries_sample.json               # Sample queries
```

---

## 🧪 Running Tests

### Phase 2 Tests

```bash
python tests/test_phase2.py
```

### Ground Truth Generation

```bash
python tests/generate_ground_truth.py
```

### Benchmarks

```bash
python tests/benchmark_hybrid.py
```

---

## 📊 Quality Evaluation

Use CLI command for quality evaluation:

```bash
ast-rag evaluate --all
```

**Target:** >80% pass rate, F1 > 0.85

---

## 📝 Ground Truth

Ground truth files stored in:
- `tests/ground_truth_queries.json` — old data
- `benchmarks/ground_truth/*.json` — new data (11 files)

---

## 🔧 Updating Ground Truth

```bash
# Generate new ground truth
python tests/generate_ground_truth.py

# Move to benchmarks
mv tests/ground_truth_queries.json benchmarks/ground_truth/
```

---

## 📈 Metrics

| Metric | Target | Current |
|--------|--------|---------|
| **Pass Rate** | >80% | 100% ✅ |
| **F1 Score** | >0.85 | 0.98 ✅ |
| **Precision** | >0.85 | 0.98 ✅ |
| **Recall** | >0.85 | 0.97 ✅ |

---

## 🐛 Troubleshooting

### Tests Failing

1. Check if graph is indexed: `ast-rag evaluate --all`
2. Check Neo4j: `cypher-shell "MATCH (n) RETURN count(n)"`
3. Re-index: `ast-rag index-folder ./ast_rag --no-schema`

### Low Pass Rate

1. Check indexed folders: `grep "COMPLETE" /tmp/index_*.log | wc -l`
2. Index remaining: `./scripts/index-remaining.sh`
3. Update ground truth: `python tests/generate_ground_truth.py`
