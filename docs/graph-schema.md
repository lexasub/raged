# Graph Schema — Node and Edge Types

See also: [schema/graph_schema.cql](../schema/graph_schema.cql) for the raw Cypher.

## Node Labels

`Project`, `Package`, `Module`, `Namespace`, `File`, `Class`, `Interface`, `Struct`, `Enum`, `Trait`, `Function`, `Method`, `Constructor`, `Destructor`, `Field`, `Variable`, `Parameter`, `Block`

## Edge Types

| Edge | Meaning |
|---|---|
| `CONTAINS_*` | Structural nesting: Project → Package → File → Class → Method → Block |
| `IMPORTS` / `INCLUDES` | File-level dependencies |
| `CALLS` | Function/method invocation |
| `CROSS_FILE_CALL` | Call resolved to a definition in another file |
| `INHERITS` / `EXTENDS` | Class inheritance (`EXTENDS` for Java, `INHERITS` elsewhere) |
| `IMPLEMENTS` | Interface/trait implementation |
| `INJECTS` | Dependency injection — heuristic |
| `OVERRIDES` | Method override |
| `TYPES` | Symbol is used as a type annotation |
| `DEPENDS_ON` | Field/variable declaration depends on a type |

`EdgeKind` also declares `HAS_PARAMETER`, `VIRTUAL_CALL` and `LAMBDA_CALL`.
Queries treat `VIRTUAL_CALL` and `LAMBDA_CALL` as call edges and the schema
indexes them, but nothing emits any of the three yet.

## MVCC Properties

Every node and edge carries:

- `valid_from` — commit hash when this version was created
- `valid_to` — commit hash when superseded, or `NULL` for current

Only nodes/edges with `valid_to IS NULL` are live in the current graph version.

Special singleton: `CurrentVersion { hash }` — tracks the active graph version.

## Stable IDs

- **Node ID**: `SHA256(project_id : file_path : kind : qualified_name)[:24]`
- **Edge ID**: `SHA256(from_id : kind : to_id : dep_kind : raw_type_string : confidence)[:24]`
- **Block ID**: `SHA256(file_path : block_type : parent_function_id : start_line)[:24]`

`project_id` is part of the node hash so several projects can share one graph.
The trailing edge fields are empty strings / `0.0` when unset.
