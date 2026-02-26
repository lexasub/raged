# Graph Schema — Node and Edge Types

See also: [schema/graph_schema.cql](../schema/graph_schema.cql) for the raw Cypher.

## Node Labels

`Project`, `Package`, `Module`, `Namespace`, `File`, `Class`, `Interface`, `Struct`, `Enum`, `Trait`, `Function`, `Method`, `Constructor`, `Destructor`, `Field`, `Variable`, `Parameter`

## Edge Types

| Edge | Meaning |
|---|---|
| `CONTAINS_*` | Structural nesting: Project → Package → File → Class → Method |
| `IMPORTS` / `INCLUDES` | File-level dependencies |
| `CALLS` | Function/method invocation |
| `INHERITS` / `EXTENDS` | Class inheritance |
| `IMPLEMENTS` | Interface/trait implementation |
| `INJECTS` | Dependency injection — heuristic |
| `OVERRIDES` | Method override |

## MVCC Properties

Every node and edge carries:

- `valid_from` — commit hash when this version was created
- `valid_to` — commit hash when superseded, or `NULL` for current

Only nodes/edges with `valid_to IS NULL` are live in the current graph version.

Special singleton: `CurrentVersion { hash }` — tracks the active graph version.

## Stable IDs

- **Node ID**: `SHA256(file_path : node_type : qualified_name)[:24]`
- **Edge ID**: `SHA256(from_id : edge_type : to_id)[:24]`
