"""Schema DDL must be valid Cypher for Neo4j 5.

``CREATE CONSTRAINT`` / ``CREATE INDEX`` take the name *before* ``IF NOT
EXISTS``. The builder emitted them the other way round::

    CREATE CONSTRAINT IF NOT EXISTS ast_node_id_unique FOR (n:ASTNode) ...

which Neo4j 5 rejects with::

    Invalid input 'ast_node_id_unique': expected 'FOR' or 'ON'

Every constraint and index therefore failed to be created, while indexing
carried on and reported success.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock

from ast_rag.repositories.schema_manager import SchemaManager


def _captured_query(fn_name: str, **kwargs) -> str:
    """Invoke a SchemaManager DDL helper against a mock driver, return the Cypher."""
    session = MagicMock()
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session

    manager = SchemaManager(driver)
    getattr(manager, fn_name)(**kwargs)

    assert session.run.called, f"{fn_name} issued no query"
    return session.run.call_args[0][0]


def test_create_constraint_places_name_before_if_not_exists():
    query = _captured_query(
        "create_constraint",
        constraint_name="ast_node_id_unique",
        label="ASTNode",
        property_name="id",
    )
    assert re.match(r"^CREATE CONSTRAINT\s+ast_node_id_unique\s+IF NOT EXISTS\s+FOR ", query), (
        f"invalid Neo4j 5 constraint syntax: {query!r}"
    )


def test_create_index_places_name_before_if_not_exists():
    query = _captured_query(
        "create_index",
        label="ASTNode",
        property_name="name",
    )
    assert re.match(r"^CREATE INDEX\s+\S+\s+IF NOT EXISTS\s+FOR ", query), (
        f"invalid Neo4j 5 index syntax: {query!r}"
    )


def test_constraint_name_not_immediately_after_if_not_exists():
    """Guard against the specific regression, in either helper."""
    for fn, kwargs in (
        (
            "create_constraint",
            {"constraint_name": "c_name", "label": "L", "property_name": "p"},
        ),
        (
            "create_index",
            {"label": "L", "property_name": "p", "index_name": "i_name"},
        ),
    ):
        query = _captured_query(fn, **kwargs)
        assert "IF NOT EXISTS c_name" not in query
        assert "IF NOT EXISTS i_name" not in query


def test_fulltext_index_name_precedes_if_not_exists():
    """Same ordering rule as CREATE CONSTRAINT, at the sibling call site.

    The fulltext builder emitted ``CREATE FULLTEXT INDEX IF NOT EXISTS <name>``,
    which Neo4j 5 rejects, so the symbol fulltext index was never created while
    indexing reported success.
    """
    query = _captured_query(
        "create_fulltext_index",
        index_name="ast_symbol_fulltext",
        labels=["Function", "Class", "Method"],
        properties=["name", "qualified_name"],
    )
    assert re.search(r"CREATE FULLTEXT INDEX\s+ast_symbol_fulltext\s+IF NOT EXISTS", query), query


def test_fulltext_index_uses_label_alternation_and_qualified_properties():
    """``FOR ([A:B])`` and bare ``ON EACH [p]`` are both invalid Cypher."""
    query = _captured_query(
        "create_fulltext_index",
        index_name="ast_symbol_fulltext",
        labels=["Function", "Class", "Method"],
        properties=["name", "qualified_name"],
    )
    assert "FOR (n:Function|Class|Method)" in query, query
    assert "ON EACH [n.name, n.qualified_name]" in query, query
    assert "([" not in query, f"label filter still bracketed: {query}"


def test_standard_indexes_are_all_btree_shaped():
    """STANDARD_INDEXES is unpacked as (label, property, name) and fed to
    create_index. A fulltext entry (name, [labels], [properties]) in that list
    produced CREATE INDEX ['name','qualified_name'] FOR (n:ast_symbol_fulltext).
    """
    from ast_rag.repositories.schema_manager import SchemaManager

    for entry in SchemaManager.STANDARD_INDEXES:
        label, property_name, index_name = entry
        assert isinstance(label, str), entry
        assert isinstance(property_name, str), entry
        assert isinstance(index_name, str), entry

    for index_name, labels, properties in SchemaManager.STANDARD_FULLTEXT_INDEXES:
        assert isinstance(index_name, str)
        assert isinstance(labels, list) and labels
        assert isinstance(properties, list) and properties
