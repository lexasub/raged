"""
typescript.py - Tree-sitter S-expression queries for TypeScript.

SKELETAL extraction: Classes, interfaces, functions, imports, basic calls.
"""

from __future__ import annotations

TYPESCRIPT_QUERIES: dict[str, str] = {
    "class_defs": """
(class_declaration
  name: (type_identifier) @name
  (class_heritage
    (extends_clause
      value: (identifier) @base_class
    )?
    (implements_clause
      (type_identifier) @iface
    )?
  )?
) @node
""",
    "interface_defs": """
(interface_declaration
  name: (type_identifier) @name
) @node
""",
    "function_defs": """
(function_declaration
  name: (identifier) @name
  parameters: (formal_parameters) @params
) @node
""",
    # `const add = (a, b) => a + b` is how most TypeScript code declares
    # functions, and none of it was indexed. function_expression is included so
    # `const f = function () {}` is treated the same way.
    "arrow_function_defs": """
(lexical_declaration
  (variable_declarator
    name: (identifier) @name
    value: [
      (arrow_function)
      (function_expression)
    ]
  )
) @node
""",
    # A type alias names a type just as an interface does, so it is indexed
    # with the same kind; there is no dedicated NodeKind for it.
    "type_alias_defs": """
(type_alias_declaration
  name: (type_identifier) @name
) @node
""",
    "method_defs": """
(method_definition
  name: (property_identifier) @name
  parameters: (formal_parameters) @params
) @node
""",
    "imports": """
(import_statement
  source: (string) @import_path
) @node
""",
    "calls": """
(call_expression
  function: [
    (identifier) @callee_name
    (member_expression property: (property_identifier) @callee_name)
  ]
) @node
""",
}
