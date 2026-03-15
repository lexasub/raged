"""
cpp.py - Tree-sitter S-expression queries for C++.

FULL extraction: Classes, templates, virtual calls, lambdas.
"""

from __future__ import annotations

CPP_QUERIES: dict[str, str] = {
    # --- Type definitions ---
    "class_defs": """
(class_specifier
  name: (type_identifier) @name
) @node
""",
    "struct_defs": """
(struct_specifier
  name: (type_identifier) @name
) @node
""",
    "enum_defs": """
(enum_specifier
  name: (type_identifier) @name
) @node
""",
    # --- Function / method definitions (top-level and within class bodies) ---
    "function_defs": """
(function_definition
  declarator: [
    ; Simple function/method with identifier
    (function_declarator
      declarator: (identifier) @name
      parameters: (parameter_list) @params
    )
    ; Simple function/method with field_identifier (C++ methods)
    (function_declarator
      declarator: (field_identifier) @name
      parameters: (parameter_list) @params
    )
    ; Qualified function (e.g., Namespace::func)
    (function_declarator
      declarator: (qualified_identifier) @name
      parameters: (parameter_list) @params
    )
    ; Nested function_declarator (for methods with qualifiers like virtual, const, override)
    (function_declarator
      declarator: (function_declarator
        declarator: (identifier) @name
        parameters: (parameter_list) @params
      )
    )
    (function_declarator
      declarator: (function_declarator
        declarator: (field_identifier) @name
        parameters: (parameter_list) @params
      )
    )
    (function_declarator
      declarator: (function_declarator
        declarator: (qualified_identifier) @name
        parameters: (parameter_list) @params
      )
    )
    ; Reference declarator variants
    (reference_declarator
      (function_declarator
        declarator: (identifier) @name
        parameters: (parameter_list) @params
      )
    )
    (reference_declarator
      (function_declarator
        declarator: (field_identifier) @name
        parameters: (parameter_list) @params
      )
    )
    (reference_declarator
      (function_declarator
        declarator: (function_declarator
          declarator: (identifier) @name
          parameters: (parameter_list) @params
        )
      )
    )
    (reference_declarator
      (function_declarator
        declarator: (function_declarator
          declarator: (field_identifier) @name
          parameters: (parameter_list) @params
        )
      )
    )
    ; Pointer declarator variants
    (pointer_declarator
      (function_declarator
        declarator: (identifier) @name
        parameters: (parameter_list) @params
      )
    )
    (pointer_declarator
      (function_declarator
        declarator: (field_identifier) @name
        parameters: (parameter_list) @params
      )
    )
    (pointer_declarator
      (function_declarator
        declarator: (function_declarator
          declarator: (identifier) @name
          parameters: (parameter_list) @params
        )
      )
    )
    (pointer_declarator
      (function_declarator
        declarator: (function_declarator
          declarator: (field_identifier) @name
          parameters: (parameter_list) @params
        )
      )
    )
  ]
) @node
""",
    # --- Method definitions within class bodies (separate query to ensure capture) ---
    "method_defs": """
(class_specifier
  body: (field_declaration_list
    (function_definition
      declarator: [
        ; Simple method with identifier
        (function_declarator
          declarator: (identifier) @name
          parameters: (parameter_list) @params
        )
        ; Simple method with field_identifier
        (function_declarator
          declarator: (field_identifier) @name
          parameters: (parameter_list) @params
        )
        ; Qualified method
        (function_declarator
          declarator: (qualified_identifier) @name
          parameters: (parameter_list) @params
        )
        ; Nested function_declarator (for methods with qualifiers)
        (function_declarator
          declarator: (function_declarator
            declarator: (identifier) @name
            parameters: (parameter_list) @params
          )
        )
        (function_declarator
          declarator: (function_declarator
            declarator: (field_identifier) @name
            parameters: (parameter_list) @params
          )
        )
        (function_declarator
          declarator: (function_declarator
            declarator: (qualified_identifier) @name
            parameters: (parameter_list) @params
          )
        )
        ; Reference declarator variants
        (reference_declarator
          (function_declarator
            declarator: (identifier) @name
            parameters: (parameter_list) @params
          )
        )
        (reference_declarator
          (function_declarator
            declarator: (field_identifier) @name
            parameters: (parameter_list) @params
          )
        )
        (reference_declarator
          (function_declarator
            declarator: (function_declarator
              declarator: (identifier) @name
              parameters: (parameter_list) @params
            )
          )
        )
        (reference_declarator
          (function_declarator
            declarator: (function_declarator
              declarator: (field_identifier) @name
              parameters: (parameter_list) @params
            )
          )
        )
        ; Pointer declarator variants
        (pointer_declarator
          (function_declarator
            declarator: (identifier) @name
            parameters: (parameter_list) @params
          )
        )
        (pointer_declarator
          (function_declarator
            declarator: (field_identifier) @name
            parameters: (parameter_list) @params
          )
        )
        (pointer_declarator
          (function_declarator
            declarator: (function_declarator
              declarator: (identifier) @name
              parameters: (parameter_list) @params
            )
          )
        )
        (pointer_declarator
          (function_declarator
            declarator: (function_declarator
              declarator: (field_identifier) @name
              parameters: (parameter_list) @params
            )
          )
        )
      ]
    ) @node
  )
)
""",
    # --- Destructor definitions ---
    "destructor_defs": """
(function_definition
  declarator: (function_declarator
    declarator: (destructor_name) @name
    parameters: (parameter_list) @params
  )
) @node
""",
    # --- #include directives ---
    "includes": """
(preproc_include
  path: [
    (string_literal) @path
    (system_lib_string) @path
  ]
) @node
""",
    # --- Namespace definitions ---
    "namespace_defs": """
(namespace_definition
  name: (namespace_identifier) @name
) @node
""",
    # --- Function/method calls ---
    "calls": """
(call_expression
  function: [
    (identifier) @callee_name
    (field_expression
      field: (field_identifier) @callee_name)
    (qualified_identifier
      name: (identifier) @callee_name)
  ]
) @node
""",
    # --- Field declarations inside classes ---
    "field_defs": """
(field_declaration
  declarator: (field_identifier) @name
) @node
""",
}
