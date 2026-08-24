"""Every relationship type a query matches must be one some writer emits.

``batch_upsert_edges`` stores *all* edges as a single untyped ``:EDGE``
relationship and keeps the semantic kind as a property::

    MERGE (a)-[r:EDGE {id: e.id}]->(b)
    SET r += e

A query that instead matches ``[:INHERITS]`` or ``[r:TYPES]`` is therefore
valid Cypher over an empty set: it compiles, it runs, and it returns nothing,
for ever, with no error. Eight read sites did exactly that, so
``get_inheritance_tree``, ``find_overrides`` and the type-usage and inheritance
counts in the node-detail and impact paths silently reported nothing.

A runtime test is a poor fit for the same reason it was for
``test_query_parameters_bound``: proving a traversal returns rows needs a
populated graph with real inheritance and override edges. This checks the
invariant statically instead -- collect the relationship types the writers
create, collect the ones the readers match, and assert the second set is a
subset of the first.

The check is deliberately derived from the source rather than hard-coded: if a
writer is later changed to emit typed relationships (option B in #61), the
matching readers stop failing this test on their own.
"""

from __future__ import annotations

import re
from pathlib import Path

PACKAGE = Path(__file__).parent.parent / "ast_rag"

# -[var:TYPE]-, -[:A|B|C*1..3]-, -[rels:EDGE*1..{max_depth}]- ...
RELATIONSHIP = re.compile(r"-\[\s*\w*\s*:\s*([A-Z_][A-Z0-9_]*(?:\s*\|\s*[A-Z_][A-Z0-9_]*)*)")

# A Cypher clause that *creates* a relationship rather than matching one.
WRITES = re.compile(r"\b(?:MERGE|CREATE)\b")
READS = re.compile(r"\bMATCH\b")


def _types(fragment: str) -> set[str]:
    """Split an alternation such as ``INHERITS|EXTENDS`` into its types."""
    return {t.strip() for t in fragment.split("|") if t.strip()}


def _scan() -> tuple[set[str], dict[str, list[str]]]:
    """Return (types written, {type: [where it is matched]}).

    Cypher is embedded as string literals, including f-strings, so this reads
    the source as text. A relationship pattern is attributed to the nearest
    preceding clause keyword on the same line, which is how these queries are
    written throughout the package.
    """
    written: set[str] = set()
    matched: dict[str, list[str]] = {}

    for path in sorted(PACKAGE.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            found = RELATIONSHIP.findall(line)
            if not found:
                continue
            where = f"{path.relative_to(PACKAGE.parent)}:{lineno}"
            for fragment in found:
                types = _types(fragment)
                if WRITES.search(line):
                    written |= types
                elif READS.search(line):
                    for t in types:
                        matched.setdefault(t, []).append(where)

    return written, matched


def test_writers_emit_at_least_one_relationship_type():
    """Guard the guard: if this finds nothing, the scan itself is broken."""
    written, _ = _scan()
    assert "EDGE" in written, (
        "no writer emitting :EDGE was found -- the source scan is not working, "
        f"so the subset check below proves nothing. Found: {sorted(written)}"
    )


def test_every_matched_relationship_type_is_emitted_somewhere():
    written, matched = _scan()

    unmatchable = {t: sites for t, sites in matched.items() if t not in written}

    assert not unmatchable, (
        "These queries match relationship types no writer creates, so they can "
        "only ever return empty:\n"
        + "\n".join(
            f"  :{t} matched at {', '.join(sites)}" for t, sites in sorted(unmatchable.items())
        )
        + f"\n\nRelationship types actually written: {sorted(written)}."
        "\nEdges are stored as :EDGE with the semantic kind on the `kind` "
        "property, so match [r:EDGE] and filter with `WHERE r.kind IN [...]`."
    )
