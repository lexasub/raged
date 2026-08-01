"""
go.py - Tree-sitter S-expression queries for Go.

BASIC extraction: structs, interfaces, functions, methods, imports, calls.

Go models types through `type_declaration -> type_spec`, with the concrete
shape (`struct_type` / `interface_type`) hanging off the `type` field, so
structs and interfaces are matched on the type_spec rather than on a
dedicated node. Methods are `method_declaration` (they carry a `receiver`);
plain functions are `function_declaration`.
"""

from __future__ import annotations

GO_QUERIES: dict[str, str] = {
    "struct_defs": """
(type_spec
  name: (type_identifier) @name
  type: (struct_type) @body
) @node
""",
    "interface_defs": """
(type_spec
  name: (type_identifier) @name
  type: (interface_type) @body
) @node
""",
    "function_defs": """
(function_declaration
  name: (identifier) @name
  parameters: (parameter_list) @params
) @node
""",
    "method_defs": """
(method_declaration
  receiver: (parameter_list) @receiver
  name: (field_identifier) @name
  parameters: (parameter_list) @params
) @node
""",
    "field_defs": """
(field_declaration
  name: (field_identifier) @name
  type: (_) @field_type
) @node
""",
    "imports": """
(import_spec
  path: (interpreted_string_literal) @path
) @node
""",
    # `callee_name` is what EdgeExtractor._extract_call_edges reads. Go calls
    # are either a bare identifier (`helper()`) or a selector
    # (`fmt.Println()`); for the latter the method name is the useful half.
    "calls": """
[
  (call_expression
    function: (identifier) @callee_name)
  (call_expression
    function: (selector_expression
      field: (field_identifier) @callee_name))
] @node
""",
}
