# .cgrignore File Format

AST-RAG supports `.cgrignore` files to exclude files and directories from indexing, using the same syntax as `.gitignore`.

## Location

Place a `.cgrignore` file in the root of the codebase you index. `ast-rag init <path>` loads `<path>/.cgrignore` automatically. A different location can be given explicitly:

```bash
ast-rag init /path/to/codebase --ignore-file /path/to/custom.cgrignore
```

or via `ast_rag_config.json`:

```json
{
  "ignore_file": "/path/to/custom.cgrignore"
}
```

## Format

The format is identical to `.gitignore`:

```
# Comment
*.pyc          # Ignore all .pyc files
build/         # Ignore build directory
!important.py  # But don't ignore important.py
**/test/**     # Ignore test directories anywhere
```

## Patterns

| Pattern | Meaning |
|---------|---------|
| `*.ext` | Ignore all files with extension `.ext` |
| `dir/` | Ignore directory `dir` (and everything under it) |
| `!file` | Negation: don't ignore `file` |
| `**/dir` | Match `dir` in any directory |
| `dir/**` | Match everything under `dir` |

Note: as in git, negation cannot re-include a file if its parent directory is excluded — the walker prunes excluded directories without descending into them.

## Example

```
# Ignore generated code
**/generated/**
*_pb2.py

# Ignore vendored dependencies
vendor/
third_party/

# But keep the one vendored header we patched
!third_party/patched.h
```

A ready-to-copy template ships as [`.cgrignore.example`](../.cgrignore.example) in the repo root.

## Interaction with `exclude_patterns`

The `exclude_patterns` list in `ast_rag_config.json` (exact directory names) still applies and is combined with `.cgrignore` rules. `.cgrignore` is the more expressive mechanism — prefer it for new setups.

## Fallback

If no `.cgrignore` file exists, AST-RAG uses sensible defaults:

- VCS metadata: `.git/`, `.svn/`, `.hg/`
- Caches and dependencies: `__pycache__/`, `node_modules/`, `venv/`, `.venv/`
- Build artifacts: `build/`, `dist/`, `target/`, `.gradle/`
- IDE files: `.idea/`, `.vscode/`
- Compiled objects: `*.pyc`, `*.pyo`, `*.class`, `*.o`, `*.so`, `*.dll`

An existing but **empty** `.cgrignore` disables these defaults (nothing is ignored beyond `exclude_patterns` and hidden directories).
