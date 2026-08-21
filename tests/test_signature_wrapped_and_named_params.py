"""Regression tests for wrapped and named-parameter signatures in ``ast-rag sig``.

Found during production E2E verification on a foreign codebase (outlines):
``SteerableGenerator.__init__`` is stored exactly as written in source --

    "__init__(\\n        self,\\n        model: SteerableModel,\\n ...)"

Two shapes broke ``search_by_signature`` even after #63's DOTALL fix:

1. **Wrapped params** -- the generated params regex was ``\\(param1,\\s*param2``,
   demanding the first parameter immediately after the opening paren. Real
   wrapped signatures put a newline + indentation there, so no multi-line
   signature whose first line ends with ``(`` could ever match.

2. **Named parameters** -- Java/TS store ``process(int count, String name)``.
   The documented example ``*(int, String)`` built ``(int,\\s*String``, which
   requires a comma directly after the type; a following parameter name broke
   the match. The README/help example therefore returned nothing against real
   indexed code.

Like test_signature_regex_dotall.py this exercises the pure string-building
path under Python's re module with DOTALL semantics equivalent to Neo4j's
inline (?s).
"""

from __future__ import annotations

import re

from ast_rag.api.ast_rag_api import ASTRagAPI


def _build_regex(pattern: str) -> str:
    api = ASTRagAPI(driver=None, embedding_manager=None)  # type: ignore[arg-type]
    parsed = api._parse_signature_pattern(pattern)
    return api._build_signature_regex(parsed)


WRAPPED = 'parse_file(\n    self,\n    file_path: str,\n    mode: str = "r",\n)'
NAMED_JAVA = "process(int count, String name)"


def test_wrapped_first_param_matches_after_paren():
    """`(\\n    self,` must match: whitespace may follow the open paren."""
    regex = _build_regex("parse_file(self, file_path*)")
    assert re.match(regex, WRAPPED, re.DOTALL), (
        f"pattern {regex!r} did not match wrapped signature {WRAPPED!r}"
    )


def test_named_java_params_match_documented_example():
    """The documented `*(int, String)` form must match `process(int count, String name)`."""
    regex = _build_regex("*(int, String)")
    assert re.search(regex, NAMED_JAVA, re.DOTALL), (
        f"pattern {regex!r} did not match named-param signature {NAMED_JAVA!r}"
    )


def test_trailing_params_after_matched_ones_are_tolerated():
    """Matching the first two params must not require the signature to end there."""
    regex = _build_regex("__init__(self, model*)")
    sig = "__init__(\n        self,\n        model: SteerableModel,\n        output_type: Optional[Any],\n    )"
    assert re.search(regex, sig, re.DOTALL), (
        f"pattern {regex!r} did not match real outlines wrapped signature"
    )


def test_single_line_still_matches():
    """Guard against regression: the simple single-line shape keeps matching."""
    regex = _build_regex("__init__(self, model: SteerableModel)")
    assert re.search(regex, "__init__(self, model: SteerableModel)", re.DOTALL)
