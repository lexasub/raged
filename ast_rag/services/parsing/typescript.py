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
