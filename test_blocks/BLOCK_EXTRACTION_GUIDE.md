# AST Block Extraction - Usage Guide

## Overview

This extension adds support for extracting and analyzing code blocks (if/for/while/try/lambda/with) from functions in the AST-RAG parser.

## Supported Block Types

| Block Type | Python | Rust | Description |
|-----------|--------|------|----------|
| `if`      | ✓      | ✓    | Conditional expressions |
| `for`     | ✓      | ✓    | For loops |
| `while`   | ✓      | ✓    | While loops |
| `try`     | ✓      | ✓    | Exception handling |
| `lambda`  | ✓      | ✓    | Lambdas/closures |
| `with`    | ✓      | ✗    | Context managers |
| `match`   | ✓      | ✓    | Pattern matching |
| `loop`    | ✗      | ✓    | Infinite loops |

## Data Model: ASTBlock

```python
class ASTBlock(BaseModel):
    id: str                    # Unique identifier
    block_type: BlockType      # Block type (if/for/while/try/lambda/with)
    name: str                  # Name (for lambdas - parameters)
    parent_function_id: str    # ID of containing function
    lang: Language             # Language (python/rust)
    file_path: str             # File path
    start_line: int            # Start line
    end_line: int              # End line
    start_byte: int            # Start byte
    end_byte: int              # End byte
    nesting_depth: int         # Nesting depth
    source_text: str           # Block source code
    code_hash: str             # Code hash
    captured_variables: List[str]  # Captured variables (for lambdas)
    valid_from: str            # MVCC version
    valid_to: Optional[str]    # MVCC version
```

## CLI Commands

### 1. Get Function Blocks

```bash
# All function blocks
ast-rag blocks my_function

# Only lambda blocks
ast-rag blocks my_function --type lambda

# With source code
ast-rag blocks my_function --type lambda --source

# Filter by language
ast-rag blocks my_function --lang python
```

### 2. Block Statistics

```bash
# Global statistics
ast-rag blocks --stats

# Output:
# Block Statistics
#   Total blocks:      150
#   If blocks:         45
#   For blocks:        30
#   While blocks:      15
#   Try blocks:        10
#   Lambda blocks:     25
#   With blocks:       15
#   Match blocks:      10
#   Avg nesting:       2.3
#   Max nesting:       5
```

### 3. Lambda List

```bash
# All lambdas
ast-rag lambdas

# Lambdas with captured variables
ast-rag lambdas --captured

# Python lambdas
ast-rag lambdas --lang python

# With source code
ast-rag lambdas --source
```

## API Methods

### Python API

```python
from ast_rag.ast_rag_api import ASTRagAPI

api = ASTRagAPI(driver, embedding_manager)

# Get function blocks
blocks = api.get_blocks_for_function(
    function_id="abc123",
    block_type="lambda",  # optional
    limit=100
)

# Get block source code
source = api.get_block_source(block_id="xyz789")

# Search blocks by type
if_blocks = api.search_blocks(
    block_type="if",
    lang="python",
    min_nesting=2,
    limit=50
)

# Get lambdas
lambdas = api.get_lambda_blocks(
    lang="python",
    with_captured_vars=True,  # only with captured variables
    limit=50
)

# Statistics
stats = api.get_block_statistics(function_id="abc123")
# or for entire codebase
stats = api.get_block_statistics()
```

## Cypher Queries

### Get all function blocks

```cypher
MATCH (f:Function {id: $function_id})-[r:CONTAINS_BLOCK]->(b:Block)
WHERE b.valid_to IS NULL
RETURN b
ORDER BY b.start_line
```

### Find lambdas with captured variables

```cypher
MATCH (b:Block)
WHERE b.block_type = 'lambda'
  AND b.valid_to IS NULL
  AND size(b.captured_variables) > 0
RETURN b
ORDER BY size(b.captured_variables) DESC
```

### Block statistics

```cypher
MATCH (b:Block)
WHERE b.valid_to IS NULL
RETURN
  count(b) AS total_blocks,
  b.block_type AS type,
  avg(b.nesting_depth) AS avg_depth,
  max(b.nesting_depth) AS max_depth
GROUP BY type
```

### Find deeply nested blocks

```cypher
MATCH (b:Block)
WHERE b.valid_to IS NULL
  AND b.nesting_depth >= 4
RETURN b.file_path, b.start_line, b.block_type, b.nesting_depth
ORDER BY b.nesting_depth DESC
```

## Usage Examples

### 1. Function complexity analysis

```python
# Find functions with many blocks
stats = api.get_block_statistics()
if stats['total_blocks'] > 20:
    print("Function has high cyclomatic complexity")
```

### 2. Search closures with captured variables

```python
# Find all closures with captured variables
closures = api.get_lambda_blocks(
    lang="rust",
    with_captured_vars=True
)
for closure in closures:
    print(f"Closure captures: {closure['captured_variables']}")
```

### 3. Nesting analysis

```python
# Find blocks with nesting depth > 3
deep_blocks = api.search_blocks(
    block_type="if",
    min_nesting=4
)
for block in deep_blocks:
    print(f"Deep if block: {block['file_path']}:{block['start_line']}")
```

## Integration in indexing process

Blocks are extracted automatically when indexing Python and Rust files:

```bash
# Full indexing with block extraction
ast-rag init /path/to/codebase

# Blocks will be saved to Neo4j with CONTAINS_BLOCK relationships
```

## Architecture

```
ast_parser.py
    └── extract_blocks()
            └── BlockExtractor.extract_blocks()
                    ├── Python: if/for/while/try/with/lambda/match
                    └── Rust: if/for/while/loop/try/match/closure

graph_updater.py
    ├── batch_upsert_blocks()
    └── batch_upsert_block_edges()

ast_rag_api.py
    ├── get_blocks_for_function()
    ├── get_block_source()
    ├── search_blocks()
    ├── get_lambda_blocks()
    └── get_block_statistics()

cli.py
    ├── ast-rag blocks
    └── ast-rag lambdas
```

## Extension to other languages

To add support for a new language:

1. Add queries to `language_queries.py`:
```python
NEW_LANG_QUERIES = {
    "if_blocks": "...",
    "for_blocks": "...",
    # ...
}
```

2. Add mapping to `block_extractor.py`:
```python
NEW_LANG_BLOCK_QUERIES = {
    "if_blocks": BlockType.IF,
    # ...
}
LANGUAGE_BLOCK_QUERIES["new_lang"] = NEW_LANG_BLOCK_QUERIES
```

3. Update `ast_parser.py.extract_blocks()`:
```python
if lang in ("python", "rust", "new_lang"):
    # extract blocks
```

## Testing

Code examples for testing are in `test_blocks/`:
- `example_python.py` - Python examples
- `example_rust.rs` - Rust examples

```bash
# Index test files
ast-rag init test_blocks/

# Check block extraction
ast-rag blocks --stats
ast-rag lambdas --lang python
```
