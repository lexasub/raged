"""
parsers.py - Stack trace parsers for multiple languages.

Supports:
- Python: File "x.py", line 42, in func
- C++: #0 0x... in func() at file.cpp:42
- Java: at com.example.Class.method(Class.java:42)
- Rust: at src/file.rs:42

Each parser extracts structured StackFrame objects from raw stack trace text.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Optional

from .models import StackFrame, FrameType, Language


class StackTraceParser(ABC):
    """Abstract base class for stack trace parsers."""
    
    @abstractmethod
    def parse(self, stacktrace: str) -> list[StackFrame]:
        """Parse a stack trace string into a list of StackFrame objects.
        
        Args:
            stacktrace: Raw stack trace text
            
        Returns:
            List of StackFrame objects in order (top to bottom)
        """
        pass
    
    @abstractmethod
    def detect_language(self, stacktrace: str) -> Language:
        """Detect the language of a stack trace.
        
        Args:
            stacktrace: Raw stack trace text
            
        Returns:
            Detected Language enum value
        """
        pass
    
    @abstractmethod
    def extract_error_info(self, stacktrace: str) -> tuple[str, str]:
        """Extract error type and message from stack trace.
        
        Args:
            stacktrace: Raw stack trace text
            
        Returns:
            Tuple of (error_type, error_message)
        """
        pass


class PythonParser(StackTraceParser):
    """Parser for Python stack traces.
    
    Format examples:
        Traceback (most recent call last):
          File "main.py", line 42, in <module>
            result = process_data(data)
          File "processor.py", line 15, in process_data
            return transform(item)
          File "transform.py", line 8, in transform
            raise ValueError("Invalid input")
        ValueError: Invalid input
    
    Also handles asyncio traces:
        Task exception was never retrieved
        future: <Task finished name='Task-1' coro=<async_func() done, defined at test.py:10>>
        Traceback (most recent call last):
          File "test.py", line 12, in async_func
    """
    
    # Pattern for standard Python stack frames
    FRAME_PATTERN = re.compile(
        r'^\s+File\s+"(?P<file>[^"]+)"'  # File "path.py"
        r',\s+line\s+(?P<line>\d+)'       # , line 42
        r'(?:,\s+in\s+(?P<func>[^\s]+))?'  # , in function_name
        r'(?:\s*->\s*(?P<async_marker>async))?',  # -> async (for async frames)
        re.MULTILINE
    )
    
    # Pattern for error type and message (last line)
    ERROR_PATTERN = re.compile(
        r'^(?P<error>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)'  # Error type
        r'(?::\s*(?P<message>.*))?$',  # : message
        re.MULTILINE
    )
    
    # Pattern for asyncio task traces
    ASYNC_PATTERN = re.compile(
        r'coro=<(?P<func>[^\s()]+)\(\)\s+done,\s+defined\s+at\s+(?P<file>[^:]+):(?P<line>\d+)'
    )
    
    def detect_language(self, stacktrace: str) -> Language:
        """Detect Python stack trace by characteristic patterns."""
        indicators = [
            'Traceback (most recent call last)',
            'File "',
            ', line ',
            'in <module>',
            'Task exception was never retrieved',
        ]
        return Language.PYTHON if any(ind in stacktrace for ind in indicators) else Language.UNKNOWN
    
    def extract_error_info(self, stacktrace: str) -> tuple[str, str]:
        """Extract error type and message from Python stack trace."""
        lines = stacktrace.strip().split('\n')
        
        # Look for the last non-empty line that matches error pattern
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            
            # Skip common non-error lines
            if line.startswith('File "') or line.startswith('Traceback') or line.startswith('Task'):
                continue
            
            match = self.ERROR_PATTERN.match(line)
            if match:
                error_type = match.group('error')
                message = match.group('message') or ''
                return error_type, message
        
        # Fallback: use last line as error message
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('File "') and not line.startswith('Traceback'):
                if ':' in line:
                    parts = line.split(':', 1)
                    return parts[0].strip(), parts[1].strip()
                return 'Error', line
        
        return 'UnknownError', ''
    
    def parse(self, stacktrace: str) -> list[StackFrame]:
        """Parse Python stack trace into frames."""
        frames = []
        lines = stacktrace.split('\n')
        
        frame_index = 0
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Try to match standard frame pattern
            match = self.FRAME_PATTERN.match(line)
            if match:
                file_path = match.group('file')
                line_number = int(match.group('line'))
                func_name = match.group('func') or '<unknown>'
                is_async = bool(match.group('async_marker'))
                
                # Determine frame type
                frame_type = FrameType.FUNCTION_CALL
                class_name = None
                module = None
                
                if func_name == '<module>':
                    func_name = '<module>'
                    frame_type = FrameType.FUNCTION_CALL
                elif func_name.startswith('<'):
                    frame_type = FrameType.LAMBDA if 'lambda' in func_name else FrameType.FUNCTION_CALL
                elif '.' in func_name:
                    # Might be a method call
                    parts = func_name.rsplit('.', 1)
                    if len(parts) == 2:
                        class_name = parts[0]
                        func_name = parts[1]
                        frame_type = FrameType.METHOD_CALL
                
                # Extract module from file path
                if file_path:
                    module = file_path.replace('/', '.').replace('\\', '.').replace('.py', '')
                
                frames.append(StackFrame(
                    frame_index=frame_index,
                    function_name=func_name,
                    class_name=class_name,
                    file_path=file_path,
                    line_number=line_number,
                    language=Language.PYTHON,
                    frame_type=frame_type,
                    raw_line=line.strip(),
                    module=module,
                    is_async=is_async,
                ))
                frame_index += 1
                i += 1
                continue
            
            # Try async pattern
            async_match = self.ASYNC_PATTERN.search(line)
            if async_match:
                frames.append(StackFrame(
                    frame_index=frame_index,
                    function_name=async_match.group('func'),
                    file_path=async_match.group('file'),
                    line_number=int(async_match.group('line')),
                    language=Language.PYTHON,
                    frame_type=FrameType.ASYNC_CALLBACK,
                    raw_line=line.strip(),
                    is_async=True,
                ))
                frame_index += 1
            
            i += 1
        
        return frames


class CppParser(StackTraceParser):
    """Parser for C++ stack traces.
    
    Format examples (GDB/LLDB style):
        #0  0x00007fff5fbff6c0 in MyClass::myMethod(int) at file.cpp:42
        #1  0x00007fff5fbff700 in process() at main.cpp:15
        #2  0x00007fff5fbff740 in main at main.cpp:8
    
    Also handles:
        #0 0x4005f4 in foo() /path/to/file.cpp:10
        #1 0x400623 in bar() /path/to/file.cpp:20
    """
    
    # Pattern for GDB/LLDB style frames - improved to handle function signatures
    # Matches: #0  0x... in func() at file.cpp:42
    FRAME_PATTERN = re.compile(
        r'^\s*#(?P<frame_num>\d+)'                    # #0
        r'\s+(?:0x[0-9a-fA-F]+)?'                      # Optional address
        r'\s+in\s+(?P<func>[^\s]+)'                    # in function_name
        r'\s+at\s+'                                    # at
        r'(?P<file>(?:[A-Za-z]:)?[^:\n]+?'             # file path (handle Windows drive letters)
        r'\.(?:cpp|cxx|cc|c|hpp|hxx|hh|h|cpp|ipp))'    # extension
        r'(?::(?P<line>\d+))?',                        # :line (optional)
        re.MULTILINE
    )
    
    # Pattern for error info (often from exception)
    ERROR_PATTERN = re.compile(
        r'(?:terminate|exception|error)[:\s]+(?:what\(\):\s*)?(?P<message>.+?)(?:\n|$)',
        re.IGNORECASE
    )
    
    def detect_language(self, stacktrace: str) -> Language:
        """Detect C++ stack trace by characteristic patterns."""
        indicators = [
            re.compile(r'#\d+\s+0x[0-9a-fA-F]+\s+in\s+'),
            re.compile(r'in\s+[A-Za-z_][A-Za-z0-9_:]*\(\)'),
            re.compile(r'\.cpp:\d+'),
            re.compile(r'terminate\s+called'),
            re.compile(r'std::'),
        ]
        return Language.CPP if any(p.search(stacktrace) for p in indicators) else Language.UNKNOWN
    
    def extract_error_info(self, stacktrace: str) -> tuple[str, str]:
        """Extract error type and message from C++ stack trace."""
        # Look for exception info
        match = self.ERROR_PATTERN.search(stacktrace)
        if match:
            message = match.group('message').strip()
            # Try to extract exception type
            if 'std::' in message:
                type_match = re.search(r'std::([A-Za-z]+)', message)
                if type_match:
                    return f'std::{type_match.group(1)}', message
            return 'std::exception', message
        
        # Look for terminate messages
        term_match = re.search(r'terminate\s+called\s+(?:after\s+throwing\s+)?(?:an\s+instance of\s+)?[\'"]?([^\'"\n]+)[\'"]?', stacktrace, re.IGNORECASE)
        if term_match:
            error_text = term_match.group(1).strip()
            if 'std::' in error_text:
                type_match = re.search(r'std::([A-Za-z]+)', error_text)
                if type_match:
                    return f'std::{type_match.group(1)}', error_text
            return 'std::exception', error_text
        
        return 'UnknownError', ''
    
    def parse(self, stacktrace: str) -> list[StackFrame]:
        """Parse C++ stack trace into frames."""
        frames = []
        
        for match in self.FRAME_PATTERN.finditer(stacktrace):
            frame_num = int(match.group('frame_num'))
            func_name = match.group('func')
            file_path = match.group('file')
            line_number = int(match.group('line')) if match.group('line') else None
            
            # Clean function name - remove parameters like (int, char*)
            # e.g., "myMethod(int)" -> "myMethod"
            # e.g., "std::vector<int>::at(unsigned long)" -> "at"
            clean_func = func_name
            paren_idx = func_name.find('(')
            if paren_idx != -1:
                clean_func = func_name[:paren_idx]
            
            # Parse function name for class::method pattern
            class_name = None
            frame_type = FrameType.FUNCTION_CALL
            
            if '::' in clean_func:
                parts = clean_func.rsplit('::', 1)
                if len(parts) == 2:
                    class_name = parts[0]
                    func_name = parts[1]
                    if func_name == (class_name.split('::')[-1]):
                        frame_type = FrameType.CONSTRUCTOR
                    elif func_name.startswith('~'):
                        frame_type = FrameType.DESTRUCTOR
                        func_name = func_name[1:]
                    else:
                        frame_type = FrameType.METHOD_CALL
            else:
                func_name = clean_func
            
            # Extract module from file path
            module = None
            if file_path:
                module = file_path.replace('/', '.').replace('\\', '.')
                if module.endswith('.cpp'):
                    module = module[:-4]
            
            frames.append(StackFrame(
                frame_index=frame_num,
                function_name=func_name,
                class_name=class_name,
                file_path=file_path,
                line_number=line_number,
                language=Language.CPP,
                frame_type=frame_type,
                raw_line=match.group(0).strip(),
                module=module,
            ))
        
        return frames


class JavaParser(StackTraceParser):
    """Parser for Java stack traces.
    
    Format examples:
        java.lang.NullPointerException: Cannot invoke method on null object
            at com.example.MyClass.myMethod(MyClass.java:42)
            at com.example.Processor.process(Processor.java:15)
            at com.example.Main.main(Main.java:8)
        Caused by: java.lang.IllegalArgumentException: Invalid argument
            at com.example.Validator.validate(Validator.java:25)
            ... 5 more
    
    Also handles:
        - "Caused by:" chains
        - "... N more" suppressed frames
        - Native methods: at java.lang.Object.wait(Native Method)
    """
    
    # Pattern for stack frames
    FRAME_PATTERN = re.compile(
        r'^\s+at\s+'  # Leading whitespace + "at "
        r'(?P<func>[^\(]+)'  # function name
        r'\((?P<location>[^\)]+)\)',  # (File.java:42)
        re.MULTILINE
    )
    
    # Pattern for exception type and message (first line)
    ERROR_PATTERN = re.compile(
        r'^(?P<error>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)'  # Full class name
        r'(?:\s*:\s*(?P<message>.*))?$',  # : message
        re.MULTILINE
    )
    
    # Pattern for "Caused by:" lines
    CAUSED_BY_PATTERN = re.compile(
        r'^Caused by:\s+(?P<error>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)'
        r'(?:\s*:\s*(?P<message>.*))?$',
        re.MULTILINE
    )
    
    def detect_language(self, stacktrace: str) -> Language:
        """Detect Java stack trace by characteristic patterns."""
        indicators = [
            re.compile(r'^\s+at\s+[a-z]+\.[a-z]+'),
            re.compile(r'\.java:\d+\)'),
            re.compile(r'Caused by:'),
            re.compile(r'\.\.\.\s+\d+\s+more'),
            re.compile(r'java\.lang\.'),
        ]
        return Language.JAVA if any(p.search(stacktrace) for p in indicators) else Language.UNKNOWN
    
    def extract_error_info(self, stacktrace: str) -> tuple[str, str]:
        """Extract error type and message from Java stack trace."""
        lines = stacktrace.strip().split('\n')
        
        # First line usually has the error
        if lines:
            first_line = lines[0].strip()
            match = self.ERROR_PATTERN.match(first_line)
            if match:
                return match.group('error'), match.group('message') or ''
        
        # Look for "Caused by:" which might have the root cause
        for line in lines:
            match = self.CAUSED_BY_PATTERN.match(line.strip())
            if match:
                return match.group('error'), match.group('message') or ''
        
        return 'Exception', ''
    
    def parse(self, stacktrace: str) -> list[StackFrame]:
        """Parse Java stack trace into frames."""
        frames = []
        frame_index = 0
        
        for match in self.FRAME_PATTERN.finditer(stacktrace):
            func_full = match.group('func')
            location = match.group('location')
            
            # Parse function name: package.Class.method
            class_name = None
            func_name = func_full
            module = None
            
            if '.' in func_full:
                parts = func_full.rsplit('.', 1)
                if len(parts) == 2:
                    module_class = parts[0]
                    func_name = parts[1]
                    
                    # Split module and class
                    if '.' in module_class:
                        module_parts = module_class.rsplit('.', 1)
                        module = module_parts[0]
                        class_name = module_parts[1]
                    else:
                        class_name = module_class
                        module = module_class
            
            # Parse location: File.java:42 or Native Method
            file_path = None
            line_number = None
            is_native = False
            
            if location == 'Native Method':
                is_native = True
            elif ':' in location:
                loc_parts = location.rsplit(':', 1)
                file_path = loc_parts[0]
                if len(loc_parts) > 1 and loc_parts[1].isdigit():
                    line_number = int(loc_parts[1])
            
            # Determine frame type
            frame_type = FrameType.FUNCTION_CALL
            if class_name:
                frame_type = FrameType.METHOD_CALL
                if func_name == class_name or func_name == '__init__':
                    frame_type = FrameType.CONSTRUCTOR
            
            frames.append(StackFrame(
                frame_index=frame_index,
                function_name=func_name,
                class_name=class_name,
                file_path=file_path,
                line_number=line_number,
                language=Language.JAVA,
                frame_type=frame_type,
                raw_line=match.group(0).strip(),
                module=module,
                is_native=is_native,
            ))
            frame_index += 1
        
        return frames


class RustParser(StackTraceParser):
    """Parser for Rust stack traces.
    
    Format examples (backtrace style):
        thread 'main' panicked at 'index out of bounds', src/main.rs:42:5
        stack backtrace:
           0: rust_begin_unwind
                      at /rustc/.../library/std/src/panicking.rs:593:5
           1: core::panicking::panic_fmt
                      at /rustc/.../library/core/src/panicking.rs:67:14
           2: my_crate::my_function
                      at src/main.rs:42:5
           3: my_crate::main
                      at src/main.rs:10:1
    
    Also handles panic messages and backtrace formats.
    """
    
    # Pattern for panic line
    PANIC_PATTERN = re.compile(
        r"thread\s+'(?P<thread>[^']+)'?\s+panicked\s+(?:at\s+)?(?P<message>[^,]+)?,\s*(?P<file>[^:]+):(?P<line>\d+)(?::(?P<col>\d+))?",
        re.IGNORECASE
    )
    
    # Pattern for stack frames
    FRAME_PATTERN = re.compile(
        r'^\s*(?:(?P<frame_num>\d+):)?\s*(?P<func>[^\n]+?)'
        r'(?:\n\s+at\s+(?P<file>[^:]+):(?P<line>\d+)(?::(?P<col>\d+))?)?',
        re.MULTILINE
    )
    
    def detect_language(self, stacktrace: str) -> Language:
        """Detect Rust stack trace by characteristic patterns."""
        indicators = [
            re.compile(r"thread\s+'[^']+'\s+panicked"),
            re.compile(r'\.rs:\d+'),
            re.compile(r'rust_begin_unwind'),
            re.compile(r'core::panicking'),
            re.compile(r'stack backtrace:'),
        ]
        return Language.RUST if any(p.search(stacktrace) for p in indicators) else Language.UNKNOWN
    
    def extract_error_info(self, stacktrace: str) -> tuple[str, str]:
        """Extract error type and message from Rust stack trace."""
        match = self.PANIC_PATTERN.search(stacktrace)
        if match:
            message = match.group('message').strip().strip("'")
            return 'panic', message
        
        # Look for panic message without location
        panic_match = re.search(r"panicked\s+(?:at\s+)?'([^']+)'", stacktrace)
        if panic_match:
            return 'panic', panic_match.group(1)
        
        return 'panic', ''
    
    def parse(self, stacktrace: str) -> list[StackFrame]:
        """Parse Rust stack trace into frames."""
        frames = []
        frame_index = 0
        
        # First, try to extract panic info as a special frame
        panic_match = self.PANIC_PATTERN.search(stacktrace)
        if panic_match:
            file_path = panic_match.group('file')
            line_number = int(panic_match.group('line')) if panic_match.group('line') else None
            column = int(panic_match.group('col')) if panic_match.group('col') else None
            
            frames.append(StackFrame(
                frame_index=frame_index,
                function_name='<panic>',
                file_path=file_path,
                line_number=line_number,
                column=column,
                language=Language.RUST,
                frame_type=FrameType.EXCEPTION_HANDLER,
                raw_line=panic_match.group(0).strip(),
            ))
            frame_index += 1
        
        # Parse stack frames - Rust backtrace format
        # Format: "   N: function_name\n              at file.rs:line:col"
        frame_regex = re.compile(
            r'^\s*(?:(?P<frame_num>\d+):)\s*(?P<func>[^\n]+?)'
            r'(?:\n\s+at\s+(?P<file>[^:\n]+):(?P<line>\d+)(?::(?P<col>\d+))?)?',
            re.MULTILINE
        )
        
        for match in frame_regex.finditer(stacktrace):
            func_name = match.group('func').strip()
            file_path = match.group('file')
            line_number = int(match.group('line')) if match.group('line') else None
            column = int(match.group('col')) if match.group('col') else None
            
            # Skip if no file path (runtime frames often don't have useful info)
            if not file_path:
                continue
            
            # Skip standard library frames from /rustc/ path for cleaner output
            if file_path and '/rustc/' in file_path:
                # Still include but mark as library
                pass
            
            # Parse function name for module::function pattern
            class_name = None
            module = None
            
            if '::' in func_name:
                parts = func_name.rsplit('::', 1)
                if len(parts) == 2:
                    module = parts[0]
                    func_name = parts[1]
            
            # Determine frame type
            frame_type = FrameType.FUNCTION_CALL
            if func_name == '<panic>' or 'panic' in func_name.lower():
                frame_type = FrameType.EXCEPTION_HANDLER
            
            frames.append(StackFrame(
                frame_index=frame_index,
                function_name=func_name,
                class_name=class_name,
                file_path=file_path,
                line_number=line_number,
                column=column,
                language=Language.RUST,
                frame_type=frame_type,
                raw_line=match.group(0).strip(),
                module=module,
            ))
            frame_index += 1
        
        return frames


class StackTraceParserFactory:
    """Factory for creating appropriate stack trace parsers."""
    
    _parsers: dict[Language, type[StackTraceParser]] = {
        Language.PYTHON: PythonParser,
        Language.CPP: CppParser,
        Language.JAVA: JavaParser,
        Language.RUST: RustParser,
    }
    
    @classmethod
    def get_parser(cls, language: Language) -> StackTraceParser:
        """Get parser for a specific language."""
        parser_class = cls._parsers.get(language)
        if parser_class:
            return parser_class()
        raise ValueError(f"No parser available for language: {language}")
    
    @classmethod
    def detect_and_parse(cls, stacktrace: str) -> tuple[StackTraceParser, list[StackFrame], Language]:
        """Auto-detect language and parse stack trace.
        
        Returns:
            Tuple of (parser, frames, detected_language)
        """
        # Try each parser's detection logic
        for language, parser_class in cls._parsers.items():
            parser = parser_class()
            if parser.detect_language(stacktrace) != Language.UNKNOWN:
                frames = parser.parse(stacktrace)
                return parser, frames, language
        
        # Fallback: return unknown
        return PythonParser(), [], Language.UNKNOWN
    
    @classmethod
    def get_all_parsers(cls) -> dict[Language, StackTraceParser]:
        """Get all available parsers."""
        return {lang: parser_class() for lang, parser_class in cls._parsers.items()}
