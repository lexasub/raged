# Smart Stack Trace Mapping

A service that analyzes stack traces: it maps each frame to an AST node, pulls the code, and suggests fixes.

## Features

- **Stack trace parsing** for 4 languages: Python, C++, Java, Rust
- **AST mapping**: each frame is matched to a node in the code graph
- **Code extraction**: a code snippet for each frame
- **Root cause analysis**: error category, severity and likely causes
- **Fix suggestions**: concrete steps to fix the error
- **Similar issue search**: semantic search for similar errors in the codebase

## Supported stack trace formats

### Python
```
Traceback (most recent call last):
  File "main.py", line 42, in <module>
    result = process_data(data)
  File "processor.py", line 15, in process_data
    return transform(item)
ValueError: Invalid input
```

### C++ (GDB/LLDB style)
```
terminate called after throwing an instance of 'std::out_of_range'
Stack trace:
#0  0x00007fff5fbff6c0 in std::vector<int>::at(unsigned long) at vector.h:1134
#1  0x00007fff5fbff700 in processData(std::vector<int>&) at processor.cpp:25
#2  0x00007fff5fbff740 in main at main.cpp:15
```

### Java
```
java.lang.NullPointerException: Cannot invoke method on null object
    at com.example.UserService.getUser(UserService.java:42)
    at com.example.Main.main(Main.java:15)
Caused by: java.lang.IllegalArgumentException: Invalid argument
    at com.example.UserService.validateId(UserService.java:55)
```

### Rust
```
thread 'main' panicked at 'index out of bounds: len is 3 but index is 5', src/main.rs:42:5
stack backtrace:
   0: rust_begin_unwind
   1: my_crate::process_array
              at src/main.rs:42:5
   2: my_crate::main
              at src/main.rs:10:1
```

## Usage

### CLI

```bash
# Analyze a file
ast-rag analyze-stacktrace error.log

# Analyze stdin
echo "$STACKTRACE" | ast-rag analyze-stacktrace

# JSON output
ast-rag analyze-stacktrace error.log -o json

# Plain text output
ast-rag analyze-stacktrace error.log -o text

# Skip AST mapping for speed
ast-rag analyze-stacktrace error.log --no-ast-mapping

# Verbose output
ast-rag analyze-stacktrace error.log -v
```

### Python API

```python
from ast_rag.stack_trace import StackTraceService
from ast_rag.repositories import create_driver
from ast_rag.services import EmbeddingManager
from ast_rag.models import ProjectConfig

# Initialize
config = ProjectConfig.model_validate_json(open("ast_rag_config.json").read())
driver = create_driver(config.neo4j)
embed = EmbeddingManager(config.qdrant, config.embedding, neo4j_driver=driver)

service = StackTraceService(driver, embed)

# Analyze a stack trace
trace = """
Traceback (most recent call last):
  File "main.py", line 42, in <module>
    result = process_data(data)
ValueError: Invalid input
"""

report = service.analyze(trace)

# Print the results
print(report.to_markdown())  # Markdown for people
print(report.to_json())      # JSON for machines

# Details
print(f"Error: {report.error_type}")
print(f"Root cause: {report.root_cause.likely_cause}")
print(f"Suggested fix: {report.root_cause.suggested_fix}")
print(f"Mapped frames: {report.mapped_frames}/{report.total_frames}")

# Analyze a file
report = service.analyze_from_file("error.log")
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Stack Trace Input                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              StackTraceParserFactory                     │
│  ┌──────────┬──────────┬──────────┬──────────┐          │
│  │  Python  │   C++    │   Java   │   Rust   │          │
│  │  Parser  │  Parser  │  Parser  │  Parser  │          │
│  └──────────┴──────────┴──────────┴──────────┘          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   StackFrames[]                          │
│  - frame_index, function_name, class_name               │
│  - file_path, line_number, language                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              AST Mapping (StackTraceService)             │
│  1. Find by file_path + line_number                     │
│  2. Find by function/class name                         │
│  3. Semantic search                                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Code Snippet Retrieval                      │
│  - get_code_snippet(file, start_line, end_line)         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Root Cause Analysis                         │
│  - Categorize error (null_pointer, out_of_bounds, ...)  │
│  - Determine severity (critical, high, medium, low)     │
│  - Generate likely cause explanation                    │
│  - Suggest fixes                                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Similar Issues Search                       │
│  - Semantic search by error type + message              │
│  - Find related code patterns                           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  StackTraceReport                        │
│  - error_type, message, language                        │
│  - root_cause: {category, severity, fix, confidence}    │
│  - call_chain: [StackFrame with code snippets]          │
│  - similar_issues: [SimilarIssue]                       │
│  - summary                                              │
└─────────────────────────────────────────────────────────┘
```

## Data model

### StackFrame
```json
{
  "frame_index": 0,
  "function_name": "process_data",
  "class_name": "DataProcessor",
  "file_path": "/path/to/processor.py",
  "line_number": 42,
  "language": "python",
  "frame_type": "method_call",
  "code_snippet": "...",
  "ast_node_id": "abc123",
  "ast_node_qualified_name": "DataProcessor.process_data"
}
```

### RootCause
```json
{
  "error_type": "NullPointerException",
  "error_message": "Cannot invoke method on null",
  "likely_cause": "A null reference was accessed...",
  "severity": "high",
  "category": "null_pointer",
  "suggested_fix": "1. Add null checks...\n2. Use Optional...",
  "confidence": 0.85,
  "related_frames": [0, 1]
}
```

### StackTraceReport
```json
{
  "error_type": "NullPointerException",
  "message": "Cannot invoke method on null",
  "language": "java",
  "root_cause": {...},
  "call_chain": [...],
  "similar_issues": [...],
  "summary": "...",
  "total_frames": 5,
  "mapped_frames": 3
}
```

## Error categories

| Category | Examples | Severity |
|----------|----------|----------|
| `null_pointer` | NullPointerException, NoneType | high |
| `out_of_bounds` | IndexError, out_of_range | high |
| `type_error` | TypeError, ClassCastException | medium |
| `value_error` | ValueError, IllegalArgumentException | medium |
| `key_error` | KeyError, NoSuchElement | low |
| `attribute_error` | AttributeError, MissingProperty | low |
| `file_error` | FileNotFoundError, IOException | medium |
| `memory_error` | MemoryError, OutOfMemory | critical |
| `concurrency` | ConcurrentModification, Deadlock | critical |
| `panic` | panic, assertion failed | critical |

## Integration with analyze_text

The service uses the existing `analyze_text` API for extra context:

```python
# In StackTraceService.analyze()
text_results = self._analyze_with_text_api(stacktrace)
if text_results and not report.similar_issues:
    report.similar_issues = self._convert_text_results_to_issues(text_results)
```

This finds relevant code even when exact AST mapping fails.

## Tests

```bash
# Run the tests
pytest tests/test_stack_trace.py -v

# Parser tests
pytest tests/test_stack_trace.py::TestPythonParser -v
pytest tests/test_stack_trace.py::TestJavaParser -v
pytest tests/test_stack_trace.py::TestCppParser -v
pytest tests/test_stack_trace.py::TestRustParser -v

# Model tests
pytest tests/test_stack_trace.py::TestStackFrame -v
pytest tests/test_stack_trace.py::TestStackTraceReport -v
```

## Examples

See `ast_rag/stack_trace/examples.py` for sample stack traces and usage.

## Extending

### Adding a new parser

```python
from .models import StackFrame, Language
from .parsers import StackTraceParser

class GoParser(StackTraceParser):
    def detect_language(self, stacktrace: str) -> Language:
        # Detection logic
        return Language.UNKNOWN  # or a new enum value
    
    def extract_error_info(self, stacktrace: str) -> tuple[str, str]:
        # Extract the error type and message
        return "Error", ""
    
    def parse(self, stacktrace: str) -> list[StackFrame]:
        # Parse the frames
        return []

# Register it in the factory
StackTraceParserFactory._parsers[Language.GO] = GoParser
```

## Limitations

- AST mapping needs a running Neo4j
- Semantic search needs a running Qdrant
- Mapping accuracy depends on how completely the codebase is indexed
- Some stack trace formats may need parser work

## Future improvements

- [ ] Go, TypeScript and C# support
- [ ] Parse JSON/XML error logs
- [ ] GitHub Issues integration to find similar problems
- [ ] ML model for error classification
- [ ] Open a PR with the fix automatically
- [ ] Per-project stats on frequent errors
