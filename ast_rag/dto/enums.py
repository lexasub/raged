"""DTO - Enumerations for AST-RAG.

Defines all enumerations used across the system:
- NodeKind: AST node types
- EdgeKind: Relationship types
- Language: Supported programming languages
- BlockType: Code block types
"""

from __future__ import annotations

from enum import Enum


class NodeKind(str, Enum):
    """All recognised AST node kinds persisted in the graph."""
    PROJECT = "Project"
    PACKAGE = "Package"        # Java package / Python module dir
    NAMESPACE = "Namespace"    # C++ namespace
    MODULE = "Module"          # Python module file / Rust module
    FILE = "File"
    CLASS = "Class"
    INTERFACE = "Interface"    # Java interface
    STRUCT = "Struct"          # C++ struct / Rust struct
    ENUM = "Enum"
    TRAIT = "Trait"            # Rust trait
    FUNCTION = "Function"      # top-level / free function
    METHOD = "Method"          # class member function
    CONSTRUCTOR = "Constructor"
    DESTRUCTOR = "Destructor"  # C++
    FIELD = "Field"            # class member field
    VARIABLE = "Variable"
    PARAMETER = "Parameter"
    BLOCK = "Block"            # Code block (if/for/while/try/lambda/with)
    CURRENT_VERSION = "CurrentVersion"


class EdgeKind(str, Enum):
    """All edge types in the graph."""
    CONTAINS_PACKAGE = "CONTAINS_PACKAGE"
    CONTAINS_FILE = "CONTAINS_FILE"
    CONTAINS_CLASS = "CONTAINS_CLASS"
    CONTAINS_METHOD = "CONTAINS_METHOD"
    CONTAINS_FUNCTION = "CONTAINS_FUNCTION"
    CONTAINS_FIELD = "CONTAINS_FIELD"
    CONTAINS_BLOCK = "CONTAINS_BLOCK"  # Function contains block
    HAS_PARAMETER = "HAS_PARAMETER"
    IMPORTS = "IMPORTS"      # Java / Python / TS import
    INCLUDES = "INCLUDES"    # C++ #include
    CALLS = "CALLS"
    INHERITS = "INHERITS"    # C++ inheritance
    EXTENDS = "EXTENDS"      # Java extends
    IMPLEMENTS = "IMPLEMENTS"
    INJECTS = "INJECTS"      # DI heuristic: field of another class type
    OVERRIDES = "OVERRIDES"
    DEPENDS_ON = "DEPENDS_ON"
    TYPES = "TYPES"
    VIRTUAL_CALL = "VIRTUAL_CALL"
    LAMBDA_CALL = "LAMBDA_CALL"
    CROSS_FILE_CALL = "CROSS_FILE_CALL"


class Language(str, Enum):
    """Supported source languages."""
    CPP = "cpp"
    JAVA = "java"
    RUST = "rust"
    PYTHON = "python"
    TYPESCRIPT = "typescript"


class BlockType(str, Enum):
    """Types of code blocks that can be extracted."""
    IF = "if"
    FOR = "for"
    WHILE = "while"
    TRY = "try"
    LAMBDA = "lambda"
    WITH = "with"
    MATCH = "match"        # Rust match, Python match (3.10+)
    SWITCH = "switch"      # C++/Java/TS switch
    LOOP = "loop"          # Rust loop
