"""
rust.py - Tree-sitter S-expression queries for Rust.

FULL extraction: Structs, traits, impls, generics, macros.
"""

from __future__ import annotations

RUST_QUERIES: dict[str, str] = {
    "struct_defs": """
(struct_item
  name: (type_identifier) @name
  type_parameters: (type_parameters)? @type_params
) @node
""",
    "enum_defs": """
(enum_item
  name: (type_identifier) @name
  type_parameters: (type_parameters)? @type_params
) @node
""",
    "trait_defs": """
(trait_item
  name: (type_identifier) @name
  type_parameters: (type_parameters)? @type_params
  body: (_) @body
) @node
""",
    "function_defs": """
(function_item
  name: (identifier) @name
  type_parameters: (type_parameters)? @type_params
  parameters: (parameters) @params
  return_type: (_) @return_type
) @node
""",
    "impl_defs": """
(impl_item
  trait: (type_identifier) @trait_name
  type: (_) @impl_type
) @node
""",
    "generic_defs": """
(type_parameter
  name: (type_identifier) @name
) @node
""",
    "generic_param_list": """
(type_parameters) @node
""",
    "where_clauses": """
(where_clause) @node
""",
    "where_predicate": """
(where_predicate) @node
""",
    "macro_defs": """
(macro_definition
  name: (identifier) @name
) @node
""",
    "macro_calls": """
(macro_invocation) @node
""",
    "declarative_macros": """
(macro_definition) @node
""",
    "macro_rules": """
(macro_definition) @node
""",
    "macro_rule": """
(macro_rule) @node
""",
    "trait_impls": """
(impl_item
  (type_identifier) @trait_name
) @node
""",
    "trait_method_impls": """
(impl_item) @node
""",
    "associated_functions": """
(associated_type) @node
""",
    "associated_consts": """
(static_item) @node
""",
    "match_expr": """
(match_expression) @node
""",
    "match_arm": """
(match_arm) @node
""",
    "patterns": """
(_pattern) @node
""",
    "pattern_fields": """
(field_pattern) @node
""",
    "mod_defs": """
(mod_item) @node
""",
    "use_decls": """
(use_declaration) @node
""",
    "use_trees": """
(scoped_use_list) @node
""",
    "calls": """
(call_expression
  function: [
    (identifier) @callee_name
    (field_expression
      field: (field_identifier) @callee_name)
  ]
) @node
""",
    "method_calls": """
(call_expression
  function: (field_expression
    value: (_) @receiver
    field: (field_identifier) @method_name
  )
) @node
""",
    "closure_calls": """
(call_expression
  function: (closure_expression) @closure
) @node
""",
    "field_defs": """
(field_declaration) @node
""",
    "tuple_struct_fields": """
(tuple_struct_pattern) @node
""",
    "associated_types": """
(associated_type) @node
""",
    "trait_bounds": """
(trait_bounds) @node
""",
    "if_blocks": """
(if_expression
  condition: (_) @condition
  consequence: (block)? @consequence
  alternative: (else_clause (block)? @alternative)?
) @node
""",
    "for_blocks": """
    (for_expression
      pattern: (_) @pattern
      value: (_) @iterable
      body: (block)? @body
    ) @node
    """,
    "while_blocks": """
(while_expression
  condition: (_) @condition
  body: (block)? @body
) @node
""",
    "loop_blocks": """
    (loop_expression
      body: (block)? @body
    ) @node
    """,
    "match_blocks": """
    (match_expression
      value: (_) @subject
      (match_arm
        pattern: (_) @pattern
        value: (_) @consequence
      )*
    ) @node
    """,
    "try_blocks": """
(try_expression
  (block)? @body
) @node
""",
    "closure_expr": """
(closure_expression
  parameters: (closure_parameters)? @params
  body: (_) @body
) @node
""",
}
