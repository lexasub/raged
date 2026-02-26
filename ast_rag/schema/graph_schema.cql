// =============================================================================
// graph_schema.cql - Neo4j schema for AST-RAG
//
// Run this against a Neo4j database to create all constraints and indexes.
// Execute statements individually or via APOC / neo4j-admin.
// =============================================================================

// ---------------------------------------------------------------------------
// Unique constraints (also create an implicit index on the constrained property)
// ---------------------------------------------------------------------------

// Every node type uses a stable 'id' field.
CREATE CONSTRAINT ast_project_id        IF NOT EXISTS FOR (n:Project)        REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT ast_package_id        IF NOT EXISTS FOR (n:Package)        REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT ast_namespace_id      IF NOT EXISTS FOR (n:Namespace)      REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT ast_module_id         IF NOT EXISTS FOR (n:Module)         REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT ast_file_id           IF NOT EXISTS FOR (n:File)           REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT ast_class_id          IF NOT EXISTS FOR (n:Class)          REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT ast_interface_id      IF NOT EXISTS FOR (n:Interface)      REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT ast_struct_id         IF NOT EXISTS FOR (n:Struct)         REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT ast_enum_id           IF NOT EXISTS FOR (n:Enum)           REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT ast_trait_id          IF NOT EXISTS FOR (n:Trait)          REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT ast_function_id       IF NOT EXISTS FOR (n:Function)       REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT ast_method_id         IF NOT EXISTS FOR (n:Method)         REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT ast_constructor_id    IF NOT EXISTS FOR (n:Constructor)    REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT ast_destructor_id     IF NOT EXISTS FOR (n:Destructor)     REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT ast_field_id          IF NOT EXISTS FOR (n:Field)          REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT ast_parameter_id      IF NOT EXISTS FOR (n:Parameter)      REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT ast_currentversion_id IF NOT EXISTS FOR (n:CurrentVersion) REQUIRE n.id IS UNIQUE;

// ---------------------------------------------------------------------------
// Edge constraints for new edge types
// ---------------------------------------------------------------------------

CREATE CONSTRAINT ast_depends_on_edge IF NOT EXISTS FOR (e:DEPENDS_ON) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT ast_overrides_edge IF NOT EXISTS FOR (e:OVERRIDES) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT ast_types_edge IF NOT EXISTS FOR (e:TYPES) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT ast_virtual_call_edge IF NOT EXISTS FOR (e:VIRTUAL_CALL) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT ast_lambda_call_edge IF NOT EXISTS FOR (e:LAMBDA_CALL) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT ast_cross_file_call_edge IF NOT EXISTS FOR (e:CROSS_FILE_CALL) REQUIRE e.id IS UNIQUE;

// ---------------------------------------------------------------------------
// Lookup indexes: qualified_name + lang (frequent search path)
// ---------------------------------------------------------------------------

CREATE INDEX ast_class_qname_lang     IF NOT EXISTS FOR (n:Class)       ON (n.qualified_name, n.lang);
CREATE INDEX ast_interface_qname_lang IF NOT EXISTS FOR (n:Interface)    ON (n.qualified_name, n.lang);
CREATE INDEX ast_struct_qname_lang    IF NOT EXISTS FOR (n:Struct)       ON (n.qualified_name, n.lang);
CREATE INDEX ast_function_qname_lang  IF NOT EXISTS FOR (n:Function)     ON (n.qualified_name, n.lang);
CREATE INDEX ast_method_qname_lang    IF NOT EXISTS FOR (n:Method)       ON (n.qualified_name, n.lang);

// Shortcut index: name only (for partial lookups)
CREATE INDEX ast_class_name     IF NOT EXISTS FOR (n:Class)    ON (n.name);
CREATE INDEX ast_function_name  IF NOT EXISTS FOR (n:Function) ON (n.name);
CREATE INDEX ast_method_name    IF NOT EXISTS FOR (n:Method)   ON (n.name);

// MVCC: filter by valid_to IS NULL for active records
CREATE INDEX ast_class_valid_to    IF NOT EXISTS FOR (n:Class)    ON (n.valid_to);
CREATE INDEX ast_function_valid_to IF NOT EXISTS FOR (n:Function) ON (n.valid_to);
CREATE INDEX ast_method_valid_to   IF NOT EXISTS FOR (n:Method)   ON (n.valid_to);

// ---------------------------------------------------------------------------
// Full-text index (useful for fuzzy name search via Cypher)
// Note: requires Neo4j >= 5.x (uses modern CREATE FULLTEXT INDEX syntax).
// ---------------------------------------------------------------------------

CREATE FULLTEXT INDEX ast_symbol_fulltext IF NOT EXISTS
FOR (n:Class|Interface|Struct|Enum|Trait|Function|Method|Constructor|Destructor)
ON EACH [n.name, n.qualified_name, n.signature];

// ---------------------------------------------------------------------------
// Indexes for edge types
// ---------------------------------------------------------------------------

// Index for DEPENDS_ON edges
CREATE INDEX ast_depends_on_from_id IF NOT EXISTS FOR ()-[e:DEPENDS_ON]-() ON (e.from_id);
CREATE INDEX ast_depends_on_to_id IF NOT EXISTS FOR ()-[e:DEPENDS_ON]-() ON (e.to_id);

// Index for OVERRIDES edges
CREATE INDEX ast_overrides_from_id IF NOT EXISTS FOR ()-[e:OVERRIDES]-() ON (e.from_id);
CREATE INDEX ast_overrides_to_id IF NOT EXISTS FOR ()-[e:OVERRIDES]-() ON (e.to_id);

// Index for TYPES edges
CREATE INDEX ast_types_from_id IF NOT EXISTS FOR ()-[e:TYPES]-() ON (e.from_id);
CREATE INDEX ast_types_to_id IF NOT EXISTS FOR ()-[e:TYPES]-() ON (e.to_id);

// Index for VIRTUAL_CALL edges
CREATE INDEX ast_virtual_call_from_id IF NOT EXISTS FOR ()-[e:VIRTUAL_CALL]-() ON (e.from_id);
CREATE INDEX ast_virtual_call_to_id IF NOT EXISTS FOR ()-[e:VIRTUAL_CALL]-() ON (e.to_id);

// Index for LAMBDA_CALL edges
CREATE INDEX ast_lambda_call_from_id IF NOT EXISTS FOR ()-[e:LAMBDA_CALL]-() ON (e.from_id);
CREATE INDEX ast_lambda_call_to_id IF NOT EXISTS FOR ()-[e:LAMBDA_CALL]-() ON (e.to_id);

// Index for CROSS_FILE_CALL edges
CREATE INDEX ast_cross_file_call_from_id IF NOT EXISTS FOR ()-[e:CROSS_FILE_CALL]-() ON (e.from_id);
CREATE INDEX ast_cross_file_call_to_id IF NOT EXISTS FOR ()-[e:CROSS_FILE_CALL]-() ON (e.to_id);

// ---------------------------------------------------------------------------
// Initial singleton: CurrentVersion
// Merged (not created) so re-running this script is safe.
// ---------------------------------------------------------------------------

MERGE (v:CurrentVersion {id: "singleton"})
ON CREATE SET v.hash = "INIT", v.updated_at = datetime();