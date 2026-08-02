"""Measure whether parallel file parsing is worth doing (issue #12).

Run::

    python benchmarks/parallel_parsing_benchmark.py

Summary of the result on an Apple M-series box (10 cores, CPython 3.12,
tree-sitter 0.24): **neither threads nor processes speed this up.**

    parse only (tree-sitter C) 8 threads    0.99x
    extract only (Python)      8 threads    1.17x
    end-to-end                 8 threads    0.96x
    end-to-end                 8 processes  0.22x

Why threads don't help
    py-tree-sitter does not release the GIL around ``Parser.parse``, so the C
    parsing that ought to be the win is fully serialised. Extraction is Python
    (walking query matches, building ASTNode/ASTEdge objects) and is GIL-bound
    by construction. What's left is lock and scheduling overhead, which is why
    the end-to-end number lands slightly *below* 1.0.

Why processes don't help either
    Per-file work is only a couple of milliseconds, while each task has to ship
    its extracted nodes and edges back over a pipe. ``Tree`` objects aren't
    picklable at all, so workers cannot share the parse cache and must re-parse.
    Pickling cost dominates and the pool ends up several times slower.

Conclusion
    Parallelising at this layer is not the lever. The issue's own suggestion —
    moving parsing to Rust, or otherwise getting the work out from under the
    GIL — is the direction that would actually pay. This file exists so the
    numbers can be re-checked rather than re-argued.
"""

from __future__ import annotations

import pathlib
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from ast_rag.services.parsing.parser_manager import ParserManager
from ast_rag.utils.parse_cache import ParseCache

FILE_COUNT = 800


def _make_corpus(directory: str) -> list[str]:
    """Write FILE_COUNT copies of a real source file — realistic size and shape."""
    template = pathlib.Path("ast_rag/utils/parse_cache.py").read_text()
    paths = []
    for i in range(FILE_COUNT):
        p = pathlib.Path(directory) / f"module_{i:04d}.py"
        p.write_text(template)
        paths.append(str(p))
    return paths


def _time(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def main() -> None:
    directory = tempfile.mkdtemp(prefix="ast_rag_bench_")
    try:
        paths = _make_corpus(directory)
        sources = [(p, pathlib.Path(p).read_bytes()) for p in paths]

        print(f"corpus: {len(paths)} files\n")

        # --- parse only: pure tree-sitter C ---------------------------------
        print("parse only (tree-sitter C):")
        baseline = None
        for workers in (1, 4, 8):
            pm = ParserManager(cache=ParseCache())

            def run() -> None:
                def job(item):
                    _path, src = item
                    return pm._get_parser("python").parse(src)

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    list(pool.map(job, sources))

            elapsed = _time(run)
            baseline = baseline or elapsed
            print(f"  threads={workers}: {elapsed:5.2f}s  speedup={baseline / elapsed:.2f}x")

        # --- end to end: parse + extract ------------------------------------
        print("\nend to end (parse + extract):")
        baseline = None
        for workers in (1, 4, 8):
            pm = ParserManager(cache=ParseCache())

            def run() -> None:
                def job(path: str):
                    tree = pm.parse_file(path)
                    if tree is None:
                        return None
                    src = pathlib.Path(path).read_bytes()
                    nodes = pm.extract_nodes(tree, path, "python", src, "BENCH")
                    return pm.extract_edges(tree, nodes, path, "python", src, "BENCH")

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    list(pool.map(job, paths))

            elapsed = _time(run)
            baseline = baseline or elapsed
            print(f"  threads={workers}: {elapsed:5.2f}s  speedup={baseline / elapsed:.2f}x")
    finally:
        shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    main()
