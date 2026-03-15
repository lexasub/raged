"""
java.py - Tree-sitter S-expression queries for Java.

FULL extraction: Classes, interfaces, methods, DI, inheritance, overrides.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# JAVA QUERIES
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
