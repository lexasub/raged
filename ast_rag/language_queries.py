"""
language_queries.py - Tree-sitter S-expression query definitions per language.

FULL queries for Java and C++.
SKELETAL queries for Rust, Python, TypeScript.

Each language maps to a dict of query_name -> query_string.
Query results use captured node names (prefixed with @) to identify
the relevant text / position.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# JAVA (full)
# ---------------------------------------------------------------------------

JAVA_QUERIES: dict[str, str] = {
    # --- Type definitions ---
    "class_defs": """
(class_declaration
  name: (identifier) @name
  (superclass (type_identifier) @superclass)?
  (super_interfaces (type_list (type_identifier) @iface))*
) @node
""",
    "interface_defs": """
(interface_declaration
  name: (identifier) @name
  (extends_interfaces (type_list (type_identifier) @parent_iface))*
) @node
""",
    "enum_defs": """
(enum_declaration
  name: (identifier) @name
) @node
""",
    "annotation_type_defs": """
(annotation_type_declaration
  name: (identifier) @name
) @node
""",
    # --- Method / Constructor definitions ---
    "method_defs": """
(method_declaration
  (modifiers)? @modifiers
  type: (_) @return_type
  name: (identifier) @name
  parameters: (formal_parameters) @params
) @node
""",
    "constructor_defs": """
(constructor_declaration
  (modifiers)? @modifiers
  name: (identifier) @name
  parameters: (formal_parameters) @params
) @node
""",
    # --- Field definitions ---
    "field_defs": """
(field_declaration
  type: (_) @field_type
  declarator: (variable_declarator
    name: (identifier) @name
  )
) @node
""",
    # --- Import statements ---
    "imports": """
(import_declaration
  (_) @import_path
) @node
""",
    # --- Method / Function calls ---
    "calls": """
(method_invocation
  object: (_)? @receiver
  name: (identifier) @callee_name
) @node
""",
    # --- DI annotations (heuristic): @Autowired, @Inject on fields ---
    "di_annotations": """
(field_declaration
  (modifiers
    (annotation
      name: (identifier) @annotation_name
      (#match? @annotation_name "^(Autowired|Inject|Resource)$")
    )
  )
  type: (type_identifier) @injected_type
  declarator: (variable_declarator name: (identifier) @field_name)
) @node
""",
    # --- DI fields: @Autowired, @Inject, @Resource on fields ---
    "di_fields": """
(field_declaration
  (modifiers
    (annotation
      name: (identifier) @annotation_name
      (#match? @annotation_name "^(Autowired|Inject|Resource)$")
    )
  )
  type: (type_identifier) @injected_type
  declarator: (variable_declarator
    name: (identifier) @field_name
  )
) @node
""",
    # --- DI constructors: @Autowired, @Inject on constructor parameters ---
    "di_constructors": """
(constructor_declaration
  (modifiers
    (annotation
      name: (identifier) @annotation_name
      (#match? @annotation_name "^(Autowired|Inject)$")
    )
  )
  parameters: (formal_parameters
    (formal_parameter
      type: (type_identifier) @injected_type
      name: (identifier) @param_name
    )
  )*
) @node
""",
    # --- Override annotation on methods ---
    "overrides": """
(method_declaration
  (modifiers
    (marker_annotation name: (identifier) @ann (#eq? @ann "Override"))
  )
  name: (identifier) @name
) @node
""",
}

# ---------------------------------------------------------------------------
# C++ (primary, but simplified — no template meta-programming)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# RUST (full — Phase 1: Generics + Basic Traits/Impl + Simple Macros)
# ---------------------------------------------------------------------------

RUST_QUERIES: dict[str, str] = {
    # --- Type definitions ---
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
    # Impl block — used to associate methods with their parent type
    "impl_defs": """
(impl_item
  trait: (type_identifier) @trait_name
  type: (_) @impl_type
) @node
""",
    # --- Generic parameter extraction ---
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
    # --- Macro definitions and invocations ---
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
    # --- Trait implementations and resolution ---
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
    # --- Pattern matching node extraction ---
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
    # --- Module system ---
    "mod_defs": """
(mod_item) @node
""",
    "use_decls": """
(use_declaration) @node
""",
    "use_trees": """
(scoped_use_list) @node
""",
    # --- Function/method calls ---
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
    # --- Field definitions ---
    "field_defs": """
(field_declaration) @node
""",
    "tuple_struct_fields": """
(tuple_struct_pattern) @node
""",
    # --- Associated types in traits ---
    "associated_types": """
(associated_type) @node
""",
    "trait_bounds": """
(trait_bounds) @node
""",
}

# ---------------------------------------------------------------------------
# PYTHON (skeletal — classes, functions, imports, basic calls)
# ---------------------------------------------------------------------------

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
}

# ---------------------------------------------------------------------------
# TYPESCRIPT (skeletal — classes, interfaces, functions, imports, basic calls)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Registry (used by ast_parser.py)
# ---------------------------------------------------------------------------

LANGUAGE_QUERIES: dict[str, dict[str, str]] = {
    "java": JAVA_QUERIES,
    "cpp": CPP_QUERIES,
    "rust": RUST_QUERIES,
    "python": PYTHON_QUERIES,
    "typescript": TYPESCRIPT_QUERIES,
}
