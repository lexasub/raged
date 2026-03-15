"""
python.py - Tree-sitter S-expression queries for Python.

SKELETAL extraction: Classes, functions, imports, basic calls.
"""

from __future__ import annotations

PYTHON_QUERIES: dict[str, str] = {
    "class_defs": """
(class_definition
  name: (identifier) @name
  (argument_list
    (identifier) @base_class
  )?
) @node
""",
    "function_defs": """
(function_definition
  name: (identifier) @name
  parameters: (parameters) @params
) @node
""",
    "imports": """
[
  (import_statement
    name: (dotted_name) @import_path
  ) @node
  (import_from_statement
    module_name: (_) @module_path
    name: (dotted_name) @imported_name
  ) @node
]
""",
    "calls": """
(call
  function: [
    (identifier) @callee_name
    (attribute attribute: (identifier) @callee_name)
  ]
) @node
""",
    "if_blocks": """
(if_statement
  condition: (_) @condition
  consequence: (block)? @consequence
  alternative: (else_clause (block)? @alternative)?
) @node
""",
    "for_blocks": """
    (for_statement
      left: (_) @target
      right: (_) @iterable
      body: (block)? @body
    ) @node
    """,
    "while_blocks": """
(while_statement
  condition: (_) @condition
  body: (block)? @body
) @node
""",
    "try_blocks": """
    (try_statement
      body: (block)? @body
      (except_clause
        value: (_)? @exception_type
        body: (block)? @handler_body
      )*
      (finally_clause
        (block)? @finally_body
      )?
    ) @node
    """,
    "with_blocks": """
    (with_statement
      (with_clause
        (with_item
          value: (_) @subject
        )*
      )?
      body: (block)? @body
    ) @node
    """,
    "lambda_expr": """
(lambda
  parameters: (lambda_parameters)? @params
  body: (_) @body
) @node
""",
    "match_blocks": """
    (match_statement
      subject: (_) @subject
      body: (block) @body
      (case_clause
        (case_pattern) @pattern
        consequence: (block)? @consequence
      )*
    ) @node
    """,
}
