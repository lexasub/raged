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
