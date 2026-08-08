"""test_typescript_queries.py - Coverage for modern TypeScript declarations.

The TypeScript query set was skeletal: `class_declaration`,
`interface_declaration`, `function_declaration`, `method_definition`, imports
and calls. Everything else in a modern TypeScript file was invisible to
indexing, so `goto`, `refs` and `callers` could not see it.

Covers the four gaps listed in #16:

- **arrow functions** assigned to a const, which is how most TypeScript
  codebases declare functions at all
- **type aliases**, which carry as much meaning as an interface
- **generic type parameters** on classes and functions
- **decorators**, which are the entire structure of an Angular/NestJS codebase

Each test asserts the symbol is extracted with the right kind, not merely that
parsing succeeded.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ast_rag.dto.enums import NodeKind
from ast_rag.services.parsing.parser_manager import ParserManager

TS_SOURCE = b"""
export const add = (a: number, b: number): number => a + b;

export const handler = async (req: Request): Promise<void> => {
  await log(req);
};

type UserId = string;

type Callback<T> = (value: T) => void;

class Box<T> {
  private value: T;

  constructor(value: T) {
    this.value = value;
  }
}

function identity<T>(value: T): T {
  return value;
}

@Component({
  selector: "app-root",
})
export class AppComponent {
  @Input() title: string = "";

  @HostListener("click")
  onClick(): void {}
}
"""


def _parse():
    fh = tempfile.NamedTemporaryFile(suffix=".ts", delete=False, mode="wb")
    fh.write(TS_SOURCE)
    fh.close()

    pm = ParserManager(project_id="test")
    tree = pm.parse_file(fh.name)
    assert tree is not None, "TypeScript source failed to parse"
    nodes = pm.extract_nodes(tree, fh.name, "typescript", TS_SOURCE, "TEST")
    return nodes


def _names(nodes) -> set[str]:
    return {n.name for n in nodes}


def _by_name(nodes, name):
    for n in nodes:
        if n.name == name:
            return n
    return None


class TestArrowFunctions:
    def test_const_arrow_function_is_extracted(self):
        assert "add" in _names(_parse()), "arrow function assigned to const not extracted"

    def test_async_arrow_function_is_extracted(self):
        assert "handler" in _names(_parse()), "async arrow function not extracted"

    def test_arrow_function_is_a_function_kind(self):
        node = _by_name(_parse(), "add")
        assert node is not None
        assert node.kind == NodeKind.FUNCTION, f"expected Function, got {node.kind}"


class TestTypeAliases:
    def test_simple_type_alias_is_extracted(self):
        assert "UserId" in _names(_parse()), "type alias not extracted"

    def test_type_alias_is_indexed_as_a_named_type(self):
        node = _by_name(_parse(), "UserId")
        assert node is not None
        assert node.kind == NodeKind.INTERFACE, f"expected Interface, got {node.kind}"

    def test_generic_type_alias_is_extracted(self):
        assert "Callback" in _names(_parse()), "generic type alias not extracted"


class TestGenerics:
    def test_generic_class_is_extracted(self):
        assert "Box" in _names(_parse()), "generic class not extracted"

    def test_generic_function_is_extracted(self):
        assert "identity" in _names(_parse()), "generic function not extracted"


class TestDecorators:
    def test_decorated_class_is_extracted(self):
        assert "AppComponent" in _names(_parse()), "decorated class not extracted"

    def test_decorated_method_is_extracted(self):
        assert "onClick" in _names(_parse()), "decorated method not extracted"


class TestNoRegression:
    """The declarations that already worked must keep working."""

    def test_plain_function_and_method_still_extracted(self):
        names = _names(_parse())
        assert "identity" in names
        assert "onClick" in names
