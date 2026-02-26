# AST-RAG Benchmarks

**Goal:** Evaluate AST-RAG search and navigation quality.

---

## 🚀 Quick Start

```bash
# Run all benchmarks
ast-rag evaluate --all

# Run specific benchmark
ast-rag evaluate --query benchmarks/queries/def_001.json

# View results
cat benchmarks/results/evaluation.json
```

---

## 📊 Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Precision** | TP / (TP + FP) | >90% |
| **Recall** | TP / (TP + FN) | >85% |
| **F1 Score** | 2 × (P × R) / (P + R) | >0.85 |
| **Pass Rate** | % passed tests | >80% |

**Current results:**
- ✅ Pass Rate: **100%**
- ✅ F1 Score: **0.98**
- ✅ Precision: **0.98**
- ✅ Recall: **0.97**

---

## 📁 Structure

```
benchmarks/
├── queries/            # Test queries
│   ├── def_001.json
│   ├── callers_001.json
│   └── semantic_001.json
├── ground_truth/       # Expected results
│   ├── java_refs_001.json
│   ├── cpp_calls_001.json
│   └── rust_impls_001.json
├── results/            # Evaluation results
│   └── evaluation.json
├── create_ground_truth.py  # Generate ground truth
└── README.md
```

---

## 🧪 Creating Ground Truth

```bash
# Generate ground truth for existing queries
python benchmarks/create_ground_truth.py

# This creates:
# - benchmarks/ground_truth/*.json
# - benchmarks/queries/*.json
```

---

## 📈 Running Evaluation

### All Benchmarks

```bash
ast-rag evaluate --all
```

**Output:**
```
🔍 Running: def_001.json
   ✅ PASS F1=1.00 P=1.00 R=1.00 t=0.36s

🔍 Running: callers_001.json
   ✅ PASS F1=1.00 P=1.00 R=1.00 t=0.82s

...

SUMMARY
📊 Benchmarks: 10
✅ Passed: 10
❌ Failed: 0
📈 Pass Rate: 100.0%
📈 F1 Score: 0.98
```

### Single Benchmark

```bash
ast-rag evaluate --query benchmarks/queries/def_001.json
```

### Output to File

```bash
ast-rag evaluate --all --output results/my_results.json
```

---

## 📊 Benchmark Types

### 1. Definition Lookup

**Goal:** Find class/function definition

```json
{
  "id": "def_001",
  "expected_tool": "find_definition",
  "expected_params": {
    "name": "EmbeddingManager",
    "kind": "Class"
  },
  "ground_truth_file": "ground_truth/java_defs_001.json",
  "evaluation": {
    "min_precision": 0.9,
    "min_recall": 0.9,
    "max_time_seconds": 5
  }
}
```

### 2. Call Graph

**Goal:** Find all function callers

```json
{
  "id": "callers_001",
  "expected_tool": "find_callers",
  "expected_params": {
    "name": "build_embeddings"
  },
  "ground_truth_file": "ground_truth/cpp_callers_001.json"
}
```

### 3. Semantic Search

**Goal:** Find code by natural language

```json
{
  "id": "semantic_001",
  "expected_tool": "search_semantic",
  "expected_params": {
    "query": "batch upsert nodes to neo4j"
  },
  "ground_truth_file": "ground_truth/semantic_batch_001.json"
}
```

### 4. Signature Search

**Goal:** Find function by signature pattern

```json
{
  "id": "sig_001",
  "expected_tool": "search_by_signature",
  "expected_params": {
    "signature": "process(int, String)"
  },
  "ground_truth_file": "ground_truth/java_sigs_001.json"
}
```

---

## 🔧 Troubleshooting

### Low Pass Rate (<70%)

```bash
# 1. Check how many indexed
grep "COMPLETE" /tmp/index_*.log | wc -l

# 2. Index remaining
./scripts/index-remaining.sh

# 3. Run again
ast-rag evaluate --all
```

### Low Recall

**Cause:** Not all files indexed

```bash
# Check graph
cypher-shell "MATCH (f:File) RETURN count(f)"

# Index remaining
./scripts/index-remaining.sh
```

### Low Precision

**Cause:** Noise in embeddings or imprecise search

```python
# Tune thresholds in embeddings.py
# Increase similarity_threshold
```

### Timeout

**Cause:** Slow search

```bash
# Increase max_time_seconds in benchmarks/queries/*.json
# Or optimize Neo4j indexes
```

---

## 📚 Interpreting Results

### Excellent Result
- **Pass Rate:** >90%
- **F1 Score:** >0.95
- **Precision:** >0.95
- **Recall:** >0.90

### Good Result
- **Pass Rate:** >80%
- **F1 Score:** >0.85
- **Precision:** >0.85
- **Recall:** >0.80

### Needs Improvement
- **Pass Rate:** <70%
- **F1 Score:** <0.75
- **Precision/Recall:** <0.75

---

## 📊 Example Results

```json
{
  "total_benchmarks": 10,
  "passed": 10,
  "pass_rate": 1.0,
  "average_metrics": {
    "f1_score": 0.98,
    "precision": 0.98,
    "recall": 0.97
  },
  "results": [
    {
      "benchmark_id": "def_001",
      "tool": "find_definition",
      "metrics": {
        "f1_score": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "time_seconds": 0.36
      },
      "overall_pass": true
    }
  ]
}
```

---

## 📚 See Also

- [tests/README.md](../tests/README.md) — Tests
- [docs/QUICKSTART.md](../docs/QUICKSTART.md) — Quick start
- [AGENTS.md](../AGENTS.md) — Guide for AI agents
