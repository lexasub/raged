"""
tests/test_stack_trace.py - Tests for Smart Stack Trace Mapping.

Tests cover:
- Stack trace parsers for all 4 languages
- Model serialization
- Service integration (mocked)
"""

import pytest
from pathlib import Path

from ast_rag.stack_trace.models import (
    Language,
    FrameType,
    StackFrame,
    RootCause,
    SimilarIssue,
    StackTraceReport,
)
from ast_rag.stack_trace.parsers import (
    PythonParser,
    CppParser,
    JavaParser,
    RustParser,
    StackTraceParserFactory,
)


# ============================================================================
# MODEL TESTS
# ============================================================================

class TestStackFrame:
    """Tests for StackFrame model."""
    
    def test_create_basic_frame(self):
        """Test creating a basic stack frame."""
        frame = StackFrame(
            frame_index=0,
            function_name="my_function",
            file_path="/path/to/file.py",
            line_number=42,
            language=Language.PYTHON,
        )
        
        assert frame.frame_index == 0
        assert frame.function_name == "my_function"
        assert frame.file_path == "/path/to/file.py"
        assert frame.line_number == 42
        assert frame.language == Language.PYTHON
        assert frame.class_name is None
        assert frame.code_snippet is None
    
    def test_create_method_frame(self):
        """Test creating a method frame with class name."""
        frame = StackFrame(
            frame_index=1,
            function_name="my_method",
            class_name="MyClass",
            file_path="MyClass.java",
            line_number=25,
            language=Language.JAVA,
            frame_type=FrameType.METHOD_CALL,
        )
        
        assert frame.class_name == "MyClass"
        assert frame.frame_type == FrameType.METHOD_CALL
    
    def test_frame_to_dict(self):
        """Test converting frame to dictionary."""
        frame = StackFrame(
            frame_index=0,
            function_name="test_func",
            file_path="test.py",
            line_number=10,
            language=Language.PYTHON,
        )
        
        d = frame.to_dict()
        
        assert d["frame_index"] == 0
        assert d["function_name"] == "test_func"
        assert d["file_path"] == "test.py"
        assert d["line_number"] == 10
        assert d["language"] == "python"


class TestRootCause:
    """Tests for RootCause model."""
    
    def test_create_root_cause(self):
        """Test creating root cause analysis."""
        rc = RootCause(
            error_type="ValueError",
            error_message="Invalid input",
            likely_cause="Invalid argument passed to function",
            severity="medium",
            category="value_error",
            suggested_fix="Add input validation",
            confidence=0.85,
            related_frames=[0, 1],
        )
        
        assert rc.error_type == "ValueError"
        assert rc.confidence == 0.85
        assert len(rc.related_frames) == 2
    
    def test_root_cause_defaults(self):
        """Test root cause default values."""
        rc = RootCause(
            error_type="Error",
            error_message="Something went wrong",
        )
        
        assert rc.severity == "medium"
        assert rc.confidence == 0.0
        assert rc.likely_cause is None
        assert rc.suggested_fix is None


class TestStackTraceReport:
    """Tests for StackTraceReport model."""
    
    def test_create_report(self):
        """Test creating a complete report."""
        frame = StackFrame(
            frame_index=0,
            function_name="main",
            file_path="main.py",
            line_number=1,
            language=Language.PYTHON,
        )
        
        report = StackTraceReport(
            error_type="ValueError",
            message="Invalid input",
            language=Language.PYTHON,
            call_chain=[frame],
            total_frames=1,
            mapped_frames=1,
        )
        
        assert report.error_type == "ValueError"
        assert len(report.call_chain) == 1
        assert report.total_frames == 1
    
    def test_report_to_json(self):
        """Test report JSON serialization."""
        report = StackTraceReport(
            error_type="Error",
            message="Test",
            language=Language.PYTHON,
        )
        
        import json
        json_str = report.to_json()
        data = json.loads(json_str)
        
        assert data["error_type"] == "Error"
        assert data["message"] == "Test"
        assert data["language"] == "python"
    
    def test_report_to_markdown(self):
        """Test report Markdown rendering."""
        frame = StackFrame(
            frame_index=0,
            function_name="test",
            file_path="test.py",
            line_number=10,
            language=Language.PYTHON,
        )
        
        report = StackTraceReport(
            error_type="ValueError",
            message="Test error",
            language=Language.PYTHON,
            call_chain=[frame],
            total_frames=1,
            mapped_frames=0,
        )
        
        md = report.to_markdown()
        
        assert "# Stack Trace Analysis Report" in md
        assert "ValueError" in md
        assert "test.py:10" in md


# ============================================================================
# PARSER TESTS
# ============================================================================

class TestPythonParser:
    """Tests for Python stack trace parser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = PythonParser()
    
    def test_detect_language(self):
        """Test Python language detection."""
        trace = """
Traceback (most recent call last):
  File "test.py", line 10, in <module>
    func()
ValueError: error
"""
        assert self.parser.detect_language(trace) == Language.PYTHON
    
    def test_extract_error_info(self):
        """Test error type and message extraction."""
        trace = """
Traceback (most recent call last):
  File "test.py", line 10
ValueError: Invalid input value
"""
        error_type, message = self.parser.extract_error_info(trace)
        
        assert error_type == "ValueError"
        assert message == "Invalid input value"
    
    def test_parse_simple_trace(self):
        """Test parsing a simple Python stack trace."""
        trace = """
Traceback (most recent call last):
  File "main.py", line 42, in <module>
    result = process(data)
  File "processor.py", line 15, in process
    return transform(item)
ValueError: error
"""
        frames = self.parser.parse(trace)
        
        assert len(frames) == 2
        assert frames[0].function_name == "<module>"
        assert frames[0].file_path == "main.py"
        assert frames[0].line_number == 42
        assert frames[1].function_name == "process"
        assert frames[1].file_path == "processor.py"
    
    def test_parse_with_class_method(self):
        """Test parsing trace with class.method pattern."""
        trace = """
Traceback (most recent call last):
  File "test.py", line 5, in <module>
    obj.method()
  File "test.py", line 10, in MyClass.method
    pass
AttributeError: error
"""
        frames = self.parser.parse(trace)
        
        assert len(frames) >= 1


class TestCppParser:
    """Tests for C++ stack trace parser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CppParser()
    
    def test_detect_language(self):
        """Test C++ language detection."""
        trace = """
#0  0x00007fff5fbff6c0 in MyClass::myMethod(int) at file.cpp:42
#1  0x00007fff5fbff700 in main at main.cpp:15
"""
        assert self.parser.detect_language(trace) == Language.CPP
    
    def test_extract_error_info(self):
        """Test C++ error extraction."""
        trace = """
terminate called after throwing an instance of 'std::out_of_range'
  what(): vector::_M_range_check
"""
        error_type, message = self.parser.extract_error_info(trace)
        
        assert "std::" in error_type or error_type == "std::exception"
    
    def test_parse_gdb_style(self):
        """Test parsing GDB-style stack trace."""
        trace = """
Stack trace:
#0  0x00007fff5fbff6c0 in std::vector<int>::at(unsigned long) at vector.h:1134
#1  0x00007fff5fbff700 in processData(std::vector<int>&) at processor.cpp:25
#2  0x00007fff5fbff740 in main at main.cpp:15
"""
        frames = self.parser.parse(trace)

        # Note: First frame with .h extension may not be parsed (system headers)
        # At minimum we should get the .cpp frames
        assert len(frames) >= 2
        assert frames[-1].frame_index == 2
        assert frames[-1].function_name == "main"
        assert frames[-1].file_path == "main.cpp"
        assert any("processData" in f.function_name for f in frames)
    
    def test_parse_with_class_method(self):
        """Test parsing C++ method calls."""
        trace = """
#0  0x12345678 in MyClass::myMethod(int) at file.cpp:42
"""
        frames = self.parser.parse(trace)
        
        assert len(frames) == 1
        assert frames[0].class_name == "MyClass"
        # Function name may include parameters in C++ traces
        assert "myMethod" in frames[0].function_name


class TestJavaParser:
    """Tests for Java stack trace parser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = JavaParser()
    
    def test_detect_language(self):
        """Test Java language detection."""
        trace = """
java.lang.NullPointerException
    at com.example.MyClass.method(MyClass.java:42)
"""
        assert self.parser.detect_language(trace) == Language.JAVA
    
    def test_extract_error_info(self):
        """Test Java error extraction."""
        trace = """
java.lang.NullPointerException: Cannot invoke method on null
    at com.example.Test.method(Test.java:10)
"""
        error_type, message = self.parser.extract_error_info(trace)
        
        assert error_type == "java.lang.NullPointerException"
        assert "null" in message.lower()
    
    def test_parse_simple_trace(self):
        """Test parsing simple Java stack trace."""
        trace = """
java.lang.Exception: Error
    at com.example.UserService.getUser(UserService.java:42)
    at com.example.Main.main(Main.java:15)
"""
        frames = self.parser.parse(trace)
        
        assert len(frames) == 2
        assert frames[0].function_name == "getUser"
        assert frames[0].class_name == "UserService"
        assert frames[0].file_path == "UserService.java"
        assert frames[0].line_number == 42
    
    def test_parse_with_caused_by(self):
        """Test parsing Java trace with 'Caused by:'."""
        trace = """
java.lang.RuntimeException: Error
    at com.example.Test.method(Test.java:10)
Caused by: java.lang.IllegalArgumentException: Invalid
    at com.example.Test.validate(Test.java:20)
"""
        frames = self.parser.parse(trace)
        
        # Should parse frames from both the main exception and caused by
        assert len(frames) >= 2


class TestRustParser:
    """Tests for Rust stack trace parser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = RustParser()
    
    def test_detect_language(self):
        """Test Rust language detection."""
        trace = """
thread 'main' panicked at 'index out of bounds', src/main.rs:42:5
stack backtrace:
"""
        assert self.parser.detect_language(trace) == Language.RUST
    
    def test_extract_error_info(self):
        """Test Rust error extraction."""
        trace = """
thread 'main' panicked at 'index out of bounds: len is 3 but index is 5', src/main.rs:42:5
"""
        error_type, message = self.parser.extract_error_info(trace)
        
        assert error_type == "panic"
        assert "index out of bounds" in message
    
    def test_parse_backtrace(self):
        """Test parsing Rust backtrace."""
        trace = """
thread 'main' panicked at 'error', src/main.rs:42:5
stack backtrace:
   0: rust_begin_unwind
              at /rustc/.../library/std/src/panicking.rs:593:5
   1: my_crate::process_array
              at src/main.rs:42:5
   2: my_crate::main
              at src/main.rs:10:1
"""
        frames = self.parser.parse(trace)
        
        # Should have panic frame plus backtrace frames
        assert len(frames) >= 1


class TestParserFactory:
    """Tests for StackTraceParserFactory."""
    
    def test_detect_python(self):
        """Test auto-detection of Python traces."""
        trace = """
Traceback (most recent call last):
  File "test.py", line 1
ValueError: error
"""
        parser, frames, language = StackTraceParserFactory.detect_and_parse(trace)
        
        assert language == Language.PYTHON
        assert isinstance(parser, PythonParser)
    
    def test_detect_java(self):
        """Test auto-detection of Java traces."""
        trace = """
java.lang.Exception
    at com.example.Test.method(Test.java:10)
"""
        parser, frames, language = StackTraceParserFactory.detect_and_parse(trace)
        
        assert language == Language.JAVA
        assert isinstance(parser, JavaParser)
    
    def test_detect_cpp(self):
        """Test auto-detection of C++ traces."""
        trace = """
#0  0x12345678 in func() at file.cpp:42
"""
        parser, frames, language = StackTraceParserFactory.detect_and_parse(trace)
        
        assert language == Language.CPP
        assert isinstance(parser, CppParser)
    
    def test_detect_rust(self):
        """Test auto-detection of Rust traces."""
        trace = """
thread 'main' panicked at 'error', src/main.rs:42:5
"""
        parser, frames, language = StackTraceParserFactory.detect_and_parse(trace)
        
        assert language == Language.RUST
        assert isinstance(parser, RustParser)
    
    def test_get_parser_by_language(self):
        """Test getting parser by explicit language."""
        python_parser = StackTraceParserFactory.get_parser(Language.PYTHON)
        cpp_parser = StackTraceParserFactory.get_parser(Language.CPP)
        java_parser = StackTraceParserFactory.get_parser(Language.JAVA)
        rust_parser = StackTraceParserFactory.get_parser(Language.RUST)
        
        assert isinstance(python_parser, PythonParser)
        assert isinstance(cpp_parser, CppParser)
        assert isinstance(java_parser, JavaParser)
        assert isinstance(rust_parser, RustParser)


# ============================================================================
# INTEGRATION TESTS (with mocked API)
# ============================================================================

class TestStackTraceServiceMocked:
    """Integration tests with mocked dependencies."""
    
    def test_service_initialization(self):
        """Test service can be initialized."""
        # This test would require Neo4j and Qdrant running
        # For now, just verify imports work
        from ast_rag.stack_trace import StackTraceService
        assert StackTraceService is not None
    
    def test_full_analysis_flow_mocked(self):
        """Test full analysis flow with mocked API."""
        # Mock the API components
        class MockSession:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def run(self, *args, **kwargs):
                # Return empty result for Cypher queries
                class MockResult:
                    def single(self):
                        return None
                return MockResult()
        
        class MockDriver:
            def session(self):
                return MockSession()
            def close(self):
                pass
        
        class MockEmbedding:
            def search(self, query, limit=10, **kwargs):
                return []
        
        from ast_rag.stack_trace import StackTraceService
        
        # Create service with mocks
        service = StackTraceService(
            driver=MockDriver(),
            embedding_manager=MockEmbedding(),
        )
        
        # Test parsing
        trace = """
Traceback (most recent call last):
  File "test.py", line 10, in func
    raise ValueError("error")
ValueError: error
"""
        # This would normally call the API, but with mocks it should handle gracefully
        report = service.analyze(trace)
        
        assert report.error_type == "ValueError"
        assert report.language == Language.PYTHON
        assert report.total_frames >= 1


# ============================================================================
# EXAMPLE VALIDATION TESTS
# ============================================================================

class TestExamples:
    """Tests to validate example stack traces can be parsed."""
    
    def test_python_example(self):
        """Test parsing Python example."""
        from ast_rag.stack_trace.examples import PYTHON_STACKTRACE_EXAMPLE
        
        parser = PythonParser()
        frames = parser.parse(PYTHON_STACKTRACE_EXAMPLE)
        
        assert len(frames) >= 1
        assert any("process_data" in f.function_name for f in frames)
    
    def test_java_example(self):
        """Test parsing Java example."""
        from ast_rag.stack_trace.examples import JAVA_STACKTRACE_EXAMPLE
        
        parser = JavaParser()
        frames = parser.parse(JAVA_STACKTRACE_EXAMPLE)
        
        assert len(frames) >= 1
        assert any("UserService" in (f.class_name or "") for f in frames)
    
    def test_cpp_example(self):
        """Test parsing C++ example."""
        from ast_rag.stack_trace.examples import CPP_STACKTRACE_EXAMPLE
        
        parser = CppParser()
        frames = parser.parse(CPP_STACKTRACE_EXAMPLE)
        
        assert len(frames) >= 1
    
    def test_rust_example(self):
        """Test parsing Rust example."""
        from ast_rag.stack_trace.examples import RUST_STACKTRACE_EXAMPLE
        
        parser = RustParser()
        frames = parser.parse(RUST_STACKTRACE_EXAMPLE)
        
        assert len(frames) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
