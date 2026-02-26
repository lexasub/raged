# Contributing to AST-RAG

Thank you for considering contributing to AST-RAG! This document provides guidelines and instructions for contributing.

## 🎯 How to Contribute

### Reporting Bugs

Before creating a bug report, please check existing issues. When creating a bug report, include:

- **Clear title** describing the issue
- **Steps to reproduce** the problem
- **Expected behavior** vs actual behavior
- **Environment details** (Python version, OS, Neo4j/Qdrant versions)
- **Logs or error messages**

**Example:**
```markdown
### Bug Report

**Description:** Empty results when searching for Java methods

**Steps to Reproduce:**
1. Index Java project
2. Run `ast-rag refs MyClass`
3. No results returned

**Expected:** List of references to MyClass

**Environment:**
- Python: 3.11
- Neo4j: 5.15
- Qdrant: 1.14

**Logs:** [Attach relevant logs]
```

### Suggesting Features

Feature suggestions are welcome! Please include:

- **Use case** — why this feature is needed
- **Proposed solution** — how it should work
- **Alternatives considered** — other approaches

### Pull Requests

1. **Fork** the repository
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```
3. **Make changes** following code style
4. **Add tests** for new functionality
5. **Run tests** to ensure everything passes:
   ```bash
   pytest tests/ -v
   ast-rag evaluate --all
   ```
6. **Commit** with clear messages (see below)
7. **Push** and create a Pull Request

## 📝 Commit Message Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `chore:` Build/config changes

**Examples:**
```
feat(rust): add macro definition extraction
fix(neo4j): resolve deadlock in batch upsert
docs: update QUICKSTART with new CLI commands
test: add benchmarks for signature search
```

## 🧪 Testing

### Running Tests

```bash
# Unit tests
pytest tests/ -v

# Quality evaluation
ast-rag evaluate --all

# Specific test file
pytest tests/test_phase2.py -v
```

### Adding Tests

New features should include:

1. **Unit tests** in `tests/`
2. **Benchmark queries** in `benchmarks/queries/`
3. **Ground truth** in `benchmarks/ground_truth/`

## 📚 Documentation

When contributing:

1. **Update README.md** if changing user-facing functionality
2. **Update relevant docs/** files for detailed changes
3. **Add docstrings** to new functions/classes
4. **Update AGENTS.md** if changing agent workflows

## 🔧 Development Setup

```bash
# Clone fork
git clone https://github.com/lexasub/raged.git
cd raged

# Create virtual environment
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## 📋 Code Style

- **Python:** Follow PEP 8
- **Type hints:** Use type annotations
- **Docstrings:** Google style
- **Line length:** 100 characters

## 🎯 Areas Needing Contribution

- [ ] More language support (Go, Kotlin, Swift)
- [ ] Performance optimizations
- [ ] Additional MCP tools
- [ ] Better error handling
- [ ] Extended test coverage
- [ ] Documentation improvements

## 📞 Questions?

- **General questions:** Open a [Discussion](https://github.com/lexasub/raged/discussions)
- **Bug reports:** Open an [Issue](https://github.com/lexasub/raged/issues)
- **Security issues:** Open a [Security Advisory](https://github.com/lexasub/raged/security/advisories/new)

## 🙏 Thank You!

All contributions are appreciated, no matter how small. AST-RAG is built by the community, for the community.
