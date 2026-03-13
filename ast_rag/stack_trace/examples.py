"""
examples.py - Example stack traces and usage demonstrations.

This module provides example stack traces for each supported language
and demonstrates how to use the StackTraceService.
"""

# ============================================================================
# EXAMPLE STACK TRACES
# ============================================================================

PYTHON_STACKTRACE_EXAMPLE = """
Traceback (most recent call last):
  File "/home/user/project/main.py", line 42, in <module>
    result = process_data(data)
  File "/home/user/project/processor.py", line 15, in process_data
    return transform(item)
  File "/home/user/project/transform.py", line 8, in transform
    raise ValueError("Invalid input: expected non-empty list")
ValueError: Invalid input: expected non-empty list
"""

PYTHON_ASYNC_STACKTRACE_EXAMPLE = """
Task exception was never retrieved
future: <Task finished name='Task-1' coro=<fetch_data() done, defined at async_example.py:10>>
Traceback (most recent call last):
  File "async_example.py", line 12, in fetch_data
    response = await api.get(url)
  File "api.py", line 25, in get
    return await self._request("GET", url)
  File "api.py", line 40, in _request
    raise ConnectionError(f"Failed to connect to {url}")
ConnectionError: Failed to connect to http://api.example.com
"""

CPP_STACKTRACE_EXAMPLE = """
terminate called after throwing an instance of 'std::out_of_range'
  what():  vector::_M_range_check: __n (which is 5) >= this->size() (which is 3)

Stack trace:
#0  0x00007fff5fbff6c0 in std::vector<int>::at(unsigned long) at /usr/include/c++/11/bits/stl_vector.h:1134
#1  0x00007fff5fbff700 in processData(std::vector<int>&) at processor.cpp:25
#2  0x00007fff5fbff740 in main at main.cpp:15
"""

JAVA_STACKTRACE_EXAMPLE = """
java.lang.NullPointerException: Cannot invoke method on null object
    at com.example.service.UserService.getUser(UserService.java:42)
    at com.example.controller.UserController.handleRequest(UserController.java:25)
    at com.example.Main.main(Main.java:15)
Caused by: java.lang.IllegalArgumentException: User ID cannot be null
    at com.example.service.UserService.validateId(UserService.java:55)
    ... 3 more
"""

RUST_STACKTRACE_EXAMPLE = """
thread 'main' panicked at 'index out of bounds: the len is 3 but the index is 5', src/main.rs:42:5
stack backtrace:
   0: rust_begin_unwind
              at /rustc/a55dd71d5fb0ec5a6a3a9e8c27b2127ba491ce52/library/std/src/panicking.rs:593:5
   1: core::panicking::panic_fmt
              at /rustc/a55dd71d5fb0ec5a6a3a9e8c27b2127ba491ce52/library/core/src/panicking.rs:67:14
   2: my_crate::process_array
              at src/main.rs:42:5
   3: my_crate::main
              at src/main.rs:10:1
"""

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

USAGE_EXAMPLE_CLI = """
# ============================================================================
# CLI Usage Examples
# ============================================================================

# 1. Analyze stack trace from file
ast-rag analyze-stacktrace error.log

# 2. Analyze stack trace from stdin
echo "$STACKTRACE" | ast-rag analyze-stacktrace

# 3. Output as JSON for programmatic processing
ast-rag analyze-stacktrace error.log -o json

# 4. Output as plain text
ast-rag analyze-stacktrace error.log -o text

# 5. Verbose mode with detailed logging
ast-rag analyze-stacktrace error.log -v

# 6. Skip AST mapping for faster analysis
ast-rag analyze-stacktrace error.log --no-ast-mapping

# 7. Pipe from Python test
python test.py 2>&1 | ast-rag analyze-stacktrace

# 8. Pipe from Java application
java -jar app.jar 2>&1 | ast-rag analyze-stacktrace

# 9. Pipe from C++ application with backtrace
./myapp 2>&1 | ast-rag analyze-stacktrace

# 10. Use custom config file
ast-rag analyze-stacktrace error.log -c custom_config.json
"""

USAGE_EXAMPLE_PYTHON_API = """
# ============================================================================
# Python API Usage Examples
# ============================================================================

from ast_rag.stack_trace import StackTraceService, StackTraceParserFactory
from ast_rag.repositories import create_driver
from ast_rag.services.embedding_manager import EmbeddingManager
from ast_rag.models import ProjectConfig

# Load configuration
config = ProjectConfig.model_validate_json(open("ast_rag_config.json").read())

# Initialize components
driver = create_driver(config.neo4j)
embed = EmbeddingManager(config.qdrant, config.embedding, neo4j_driver=driver)

# Create service
service = StackTraceService(driver, embed)

# Example 1: Analyze Python stack trace
python_trace = '''
Traceback (most recent call last):
  File "main.py", line 42, in <module>
    result = process_data(data)
  File "processor.py", line 15, in process_data
    return transform(item)
ValueError: Invalid input
'''

report = service.analyze(python_trace)
print(report.to_markdown())

# Example 2: Analyze Java stack trace
java_trace = '''
java.lang.NullPointerException: Cannot invoke method on null object
    at com.example.UserService.getUser(UserService.java:42)
    at com.example.Main.main(Main.java:15)
'''

report = service.analyze(java_trace)
print(report.to_json())

# Example 3: Analyze from file
report = service.analyze_from_file("error.log")
print(f"Error: {report.error_type}")
print(f"Root cause: {report.root_cause.likely_cause}")
print(f"Suggested fix: {report.root_cause.suggested_fix}")

# Example 4: Parse without full analysis
parser, frames, language = StackTraceParserFactory.detect_and_parse(python_trace)
print(f"Detected language: {language}")
print(f"Parsed {len(frames)} frames:")
for frame in frames:
    print(f"  {frame.frame_index}. {frame.function_name}() at {frame.file_path}:{frame.line_number}")

# Example 5: Access individual components
from ast_rag.stack_trace import PythonParser

parser = PythonParser()
error_type, error_message = parser.extract_error_info(python_trace)
print(f"Error: {error_type} - {error_message}")

frames = parser.parse(python_trace)
for frame in frames:
    print(f"Frame {frame.frame_index}: {frame.function_name}")
"""

# ============================================================================
# DEMONSTRATION OUTPUT
# ============================================================================

DEMO_OUTPUT_MARKDOWN = """
# Stack Trace Analysis Report

## Error: `ValueError`

**Message:** Invalid input: expected non-empty list

**Language:** python

**Total Frames:** 3 (2 mapped to AST)

## Root Cause Analysis

- **Type:** ValueError
- **Category:** value_error
- **Severity:** medium
- **Confidence:** 80%
- **Likely Cause:** A function received an argument with the correct type but invalid value. Validate input values before processing.
- **Suggested Fix:** 1. Add input validation
2. Use try-catch for expected invalid inputs
3. Document valid value ranges
4. Provide clear error messages

## Call Chain

1. `<module>()` at /home/user/project/main.py:42
2. `process_data()` at /home/user/project/processor.py:15
3. `transform()` at /home/user/project/transform.py:8

## Similar Issues

- [85%] Similar code pattern in transform_data
- [72%] Similar code pattern in validate_input
- [65%] Similar code pattern in process_items

## Summary

Error: ValueError | Message: Invalid input: expected non-empty list | Language: python | Stack frames: 3 total, 2 mapped to AST | Root cause: A function received an argument with the correct type but invalid value. Validate input values before processing. | Found 3 similar issues in codebase
"""

DEMO_OUTPUT_JSON = """
{
  "error_type": "ValueError",
  "message": "Invalid input: expected non-empty list",
  "language": "python",
  "root_cause": {
    "error_type": "ValueError",
    "error_message": "Invalid input: expected non-empty list",
    "likely_cause": "A function received an argument with the correct type but invalid value.",
    "severity": "medium",
    "category": "value_error",
    "suggested_fix": "1. Add input validation\\n2. Use try-catch...",
    "confidence": 0.8,
    "related_frames": [0, 1, 2]
  },
  "call_chain": [
    {
      "frame_index": 0,
      "function_name": "<module>",
      "file_path": "/home/user/project/main.py",
      "line_number": 42,
      "language": "python",
      "code_snippet": "..."
    },
    {
      "frame_index": 1,
      "function_name": "process_data",
      "file_path": "/home/user/project/processor.py",
      "line_number": 15,
      "language": "python"
    },
    {
      "frame_index": 2,
      "function_name": "transform",
      "file_path": "/home/user/project/transform.py",
      "line_number": 8,
      "language": "python"
    }
  ],
  "similar_issues": [
    {
      "issue_id": "similar_0",
      "title": "Similar code pattern in transform_data",
      "similarity_score": 0.85
    }
  ],
  "total_frames": 3,
  "mapped_frames": 2
}
"""

if __name__ == "__main__":
    print("Stack Trace Analysis Examples")
    print("=" * 60)
    print("\nAvailable examples:")
    print("1. Python stack trace")
    print("2. Python async stack trace")
    print("3. C++ stack trace")
    print("4. Java stack trace")
    print("5. Rust stack trace")
    print("\nSee USAGE_EXAMPLE_CLI and USAGE_EXAMPLE_PYTHON_API for usage instructions.")
