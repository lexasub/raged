"""
cli.py - Typer-based CLI for AST-RAG.

Commands:
  init   <path>                     Full indexing of a codebase
  update <path> --from OLD --to NEW Incremental update from git diff
  query  "<text>"                   Semantic search
  goto   <qualified_name>           Find definition
  callers <qualified_name>          Find callers
  refs   <symbol_name>              Find references
  sig    <pattern>                  Signature search
  call-graph <name>                 Visualize call graph
  symbol-impact <name>              Analyze symbol impact
  sandbox <lang> <command>          Run a command in a sandbox
  workspace <path>                  Show workspace diff
  evaluate                          Evaluate quality against benchmarks
  index-folder <path>               Index a single folder
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from ast_rag.models import ProjectConfig
from ast_rag.ast_parser import ParserManager, walk_source_files
from ast_rag.graph_schema import apply_schema, create_driver
from ast_rag.graph_updater import full_index, update_from_git, get_workspace_diff, apply_workspace_diff
from ast_rag.embeddings import EmbeddingManager
from ast_rag.ast_rag_api import ASTRagAPI
from ast_rag.output import get_formatter

app = typer.Typer(
    name="ast-rag",
    help="AST-based Retrieval-Augmented Generation for code analysis.",
    add_completion=False,
)
console = Console()


def _get_humanize_callback() -> callable:
    def callback(value: bool) -> bool:
        return value

    return callback


humanize_option = typer.Option(
    False,
    "--humanize",
    "-H",
    help="Use human-readable output (tables) instead of JSON",
    callback=_get_humanize_callback(),
)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _load_config(config_path: Optional[str] = None) -> ProjectConfig:
    """Load project config from JSON file or return defaults."""
    if config_path and Path(config_path).exists():
        return ProjectConfig.model_validate_json(Path(config_path).read_text())
    # Check for ast_rag_config.json in CWD
    default = Path("ast_rag_config.json")
    if default.exists():
        return ProjectConfig.model_validate_json(default.read_text())
    return ProjectConfig()


def _build_api(cfg: ProjectConfig) -> ASTRagAPI:
    driver = create_driver(cfg.neo4j)
    embed = EmbeddingManager(cfg.qdrant, cfg.embedding, neo4j_driver=driver)
    return ASTRagAPI(driver, embed)


# ---------------------------------------------------------------------------
# init command
# ---------------------------------------------------------------------------


@app.command()
def init(
    path: str = typer.Argument(..., help="Root directory of the codebase to index"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config JSON"),
    commit: str = typer.Option("INIT", "--commit", help="Commit hash label for this index"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Perform a full initial indexing of the codebase at PATH."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    cfg = _load_config(config)
    root = os.path.abspath(path)

    console.rule(f"[bold blue]AST-RAG init[/bold blue]: {root}")

    # 1. Apply schema
    driver = create_driver(cfg.neo4j)
    with console.status("Applying Neo4j schema..."):
        apply_schema(driver)

    # 2. Parse all source files
    pm = ParserManager()
    files = walk_source_files(root, exclude_dirs=cfg.exclude_patterns)
    console.print(f"Found [bold]{len(files)}[/bold] source files.")

    all_nodes = []
    all_edges = []

    with console.status(f"Parsing {len(files)} files...") as status:
        for i, (fp, lang) in enumerate(files):
            status.update(f"Parsing [{i + 1}/{len(files)}] {os.path.relpath(fp, root)}")
            tree = pm.parse_file(fp)
            if tree is None:
                continue
            with open(fp, "rb") as fh:
                source = fh.read()
            nodes = pm.extract_nodes(tree, fp, lang, source, commit)
            edges = pm.extract_edges(tree, nodes, fp, lang, source, commit)
            all_nodes.extend(nodes)
            all_edges.extend(edges)

    console.print(
        f"Extracted [bold]{len(all_nodes)}[/bold] nodes and [bold]{len(all_edges)}[/bold] edges."
    )

    # 3. Write to Neo4j
    with console.status("Writing to Neo4j..."):
        full_index(driver, all_nodes, all_edges, commit_hash=commit)
    console.print("[green]Graph database updated.[/green]")

    # 4. Build embeddings
    embed = EmbeddingManager(cfg.qdrant, cfg.embedding, neo4j_driver=driver)
    with console.status("Building embeddings..."):
        count = embed.build_embeddings(all_nodes)
    console.print(f"[green]Indexed {count} node embeddings.[/green]")

    console.rule("[bold green]Done[/bold green]")


# ---------------------------------------------------------------------------
# update command
# ---------------------------------------------------------------------------


@app.command()
def update(
    path: str = typer.Argument(..., help="Root directory / git repository path"),
    from_commit: str = typer.Option(..., "--from-commit", help="Old commit hash"),
    to_commit: str = typer.Option(..., "--to-commit", help="New commit hash"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Incrementally update the index from git diff OLD..NEW."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    cfg = _load_config(config)
    driver = create_driver(cfg.neo4j)
    embed = EmbeddingManager(cfg.qdrant, cfg.embedding, neo4j_driver=driver)

    console.rule(f"[bold blue]AST-RAG update[/bold blue] {from_commit[:8]}..{to_commit[:8]}")

    with console.status("Computing diff and updating graph..."):
        diff = update_from_git(driver, path, from_commit, to_commit)

    console.print(
        f"[green]+{len(diff.added_nodes)} added[/green]  "
        f"[yellow]~{len(diff.updated_nodes)} updated[/yellow]  "
        f"[red]-{len(diff.deleted_node_ids)} deleted[/red]  (nodes)"
    )

    with console.status("Updating embeddings..."):
        embed.update_embeddings(
            diff.added_nodes,
            diff.updated_nodes,
            diff.deleted_node_ids,
        )
    console.print("[green]Embeddings updated.[/green]")
    console.rule("[bold green]Done[/bold green]")


# ---------------------------------------------------------------------------
# query command
# ---------------------------------------------------------------------------


@app.command()
def query(
    text: str = typer.Argument(..., help="Semantic search query"),
    limit: int = typer.Option(10, "--limit", "-n"),
    lang: Optional[str] = typer.Option(None, "--lang", "-l"),
    kind: Optional[str] = typer.Option(None, "--kind", "-k"),
    vector_weight: Optional[float] = typer.Option(
        None, "--vector-weight", "-vw",
        help="Weight for vector similarity (0.0-1.0). Overrides config if specified."
    ),
    keyword_weight: Optional[float] = typer.Option(
        None, "--keyword-weight", "-kw",
        help="Weight for keyword search (0.0-1.0). Overrides config if specified."
    ),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    humanize: bool = humanize_option,
) -> None:
    """Perform a semantic search over the indexed codebase."""
    cfg = _load_config(config)
    api = _build_api(cfg)
    formatter = get_formatter(humanize)

    results = api.search_semantic(
        text,
        limit=limit,
        lang=lang,
        kind=kind,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
    )

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        raise typer.Exit(0)

    formatter.format_search_results(results, text)


# ---------------------------------------------------------------------------
# goto command
# ---------------------------------------------------------------------------


@app.command()
def goto(
    qualified_name: str = typer.Argument(..., help="Qualified name to look up"),
    kind: Optional[str] = typer.Option(None, "--kind", "-k"),
    lang: Optional[str] = typer.Option(None, "--lang", "-l"),
    snippet: bool = typer.Option(False, "--snippet", "-s", help="Print source snippet"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    humanize: bool = humanize_option,
) -> None:
    """Find the definition of a symbol by qualified name."""
    cfg = _load_config(config)
    api = _build_api(cfg)
    formatter = get_formatter(humanize)

    nodes = api.find_definition(qualified_name, kind=kind, lang=lang)

    if not nodes:
        console.print(f"[yellow]Definition not found for: {qualified_name}[/yellow]")
        raise typer.Exit(1)

    formatter.format_definitions(nodes, api=api, snippet=snippet)


# ---------------------------------------------------------------------------
# callers command
# ---------------------------------------------------------------------------


@app.command()
def callers(
    qualified_name: str = typer.Argument(..., help="Qualified name of the function/method"),
    depth: int = typer.Option(1, "--depth", "-d", help="Call depth to traverse (1-3)"),
    lang: Optional[str] = typer.Option(None, "--lang", "-l"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    humanize: bool = humanize_option,
) -> None:
    """Find all callers of a given function or method."""
    cfg = _load_config(config)
    api = _build_api(cfg)
    formatter = get_formatter(humanize)

    defs = api.find_definition(qualified_name, lang=lang)
    if not defs:
        console.print(f"[yellow]Symbol not found: {qualified_name}[/yellow]")
        raise typer.Exit(1)

    target = defs[0]
    if humanize:
        console.print(f"Finding callers of [bold]{target.qualified_name}[/bold]...")

    caller_nodes = api.find_callers(target.id, max_depth=depth)

    if not caller_nodes:
        console.print("[yellow]No callers found.[/yellow]")
        raise typer.Exit(0)

    formatter.format_callers(target.qualified_name, caller_nodes)


# ---------------------------------------------------------------------------
# refs command
# ---------------------------------------------------------------------------


@app.command()
def refs(
    name: str = typer.Argument(..., help="Symbol name"),
    kind: Optional[str] = typer.Option(None, "--kind", "-k", help="Node kind"),
    lang: Optional[str] = typer.Option(None, "--lang", "-l", help="Language"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max results"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    format: str = typer.Option("table", "--format", "-f", help="Output format"),
) -> None:
    """Find all references/usages of a symbol."""
    cfg = _load_config(config)
    driver = create_driver(cfg.neo4j)
    embed = EmbeddingManager(cfg.qdrant, cfg.embedding, neo4j_driver=driver)
    api = ASTRagAPI(driver, embed)
    
    with console.status(f"Finding references to '{name}'..."):
        results = api.find_references(name, kind=kind, lang=lang, limit=limit)
    
    if not results["references"]:
        console.print("[yellow]No references found[/yellow]")
        return
    
    if format == "json":
        print(json.dumps(results, indent=2))
    else:
        console.print(f"\n[bold]References to '{name}':[/bold] {results['total']} total\n")
        
        # Group by file
        by_file = {}
        for ref in results["references"]:
            fp = ref["node"]["file_path"]
            by_file.setdefault(fp, []).append(ref)
        
        for file_path, refs_in_file in by_file.items():
            console.print(f"[cyan]{file_path}[/cyan]")
            for ref in refs_in_file[:10]:  # Show first 10 per file
                node = ref["node"]
                console.print(
                    f"  {node['start_line']:4d}: {ref['reference_type']:10s} "
                    f"{node['name']}"
                )
            if len(refs_in_file) > 10:
                console.print(f"  ... and {len(refs_in_file) - 10} more")
            console.print()
    
    driver.close()


# ---------------------------------------------------------------------------
# call-graph command
# ---------------------------------------------------------------------------


@app.command("call-graph")
def call_graph(
    name: str = typer.Argument(..., help="Function/method name"),
    direction: str = typer.Option("both", "--direction", "-d",
                                  help="Direction: callers, callees, or both"),
    depth: int = typer.Option(2, "--depth", help="Graph depth"),
    lang: Optional[str] = typer.Option(None, "--lang", "-l", help="Language filter"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Visualize call graph for a function."""
    cfg = _load_config(config)
    api = _build_api(cfg)

    defs = api.find_definition(name, lang=lang)
    if not defs:
        console.print(f"[red]Function '{name}' not found[/red]")
        raise typer.Exit(1)

    node = defs[0]

    if direction in ("callers", "both"):
        with console.status("Finding callers..."):
            callers = api.find_callers(node.id, max_depth=depth)
        console.print(f"\n[bold green]Callers of {name}:[/bold green] ({len(callers)})")
        for c in callers[:20]:
            console.print(f"  ← {c.qualified_name} ({c.file_path}:{c.start_line})")

    if direction in ("callees", "both"):
        with console.status("Finding callees..."):
            callees = api.find_callees(node.id, max_depth=depth)
        console.print(f"\n[bold blue]Callees of {name}:[/bold blue] ({len(callees)})")
        for c in callees[:20]:
            console.print(f"  → {c.qualified_name} ({c.file_path}:{c.start_line})")


# ---------------------------------------------------------------------------
# symbol-impact command
# ---------------------------------------------------------------------------


@app.command("symbol-impact")
def symbol_impact(
    name: str = typer.Argument(..., help="Symbol name to analyze"),
    kind: Optional[str] = typer.Option(None, "--kind", "-k", help="Node kind filter"),
    lang: Optional[str] = typer.Option(None, "--lang", "-l", help="Language filter"),
    depth: int = typer.Option(2, "--depth", "-d", help="Call graph depth"),
    format: str = typer.Option("table", "--format", "-f", help="Output format (table/json)"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Analyze impact of a symbol (definition + references + callers + callees)."""
    cfg = _load_config(config)
    api = _build_api(cfg)

    # Find definition
    with console.status(f"Finding '{name}'..."):
        defs = api.find_definition(name, kind=kind, lang=lang)

    if not defs:
        console.print(f"[red]Symbol '{name}' not found[/red]")
        raise typer.Exit(1)

    node = defs[0]

    # Gather all info
    with console.status("Gathering references..."):
        refs = api.find_references(name, kind=kind, lang=lang, limit=100)
    with console.status("Finding callers..."):
        callers = api.find_callers(node.id, max_depth=depth)
    with console.status("Finding callees..."):
        callees = api.find_callees(node.id, max_depth=depth)

    # Output
    if format == "json":
        output = {
            "definition": node.dict(),
            "references_count": refs["total"],
            "callers_count": len(callers),
            "callees_count": len(callees),
        }
        print(json.dumps(output, indent=2))
    else:
        console.print(f"\n[bold blue]📍 Definition:[/bold blue] {node.qualified_name}")
        console.print(f"   [dim]File:[/dim] {node.file_path}:{node.start_line}")
        console.print(f"   [dim]Kind:[/dim] {node.kind.value}")
        console.print(f"   [dim]Language:[/dim] {node.lang.value}")

        console.print("\n[bold green]📊 Impact:[/bold green]")
        console.print(f"   [green]✓[/green] References: {refs['total']}")
        console.print(f"   [green]✓[/green] Callers: {len(callers)}")
        console.print(f"   [green]✓[/green] Callees: {len(callees)}")


# ---------------------------------------------------------------------------
# sandbox command (thin wrapper over sandbox.py)
# ---------------------------------------------------------------------------


@app.command()
def sandbox(
    lang: str = typer.Argument(..., help="Language: java|cpp|rust|python|typescript"),
    workdir: str = typer.Argument(".", help="Working directory inside the container"),
    command: Optional[str] = typer.Option(None, "--cmd", help="Override default test command"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Run tests in a Docker sandbox for the given language."""
    from ast_rag.sandbox import run_in_sandbox, DEFAULT_COMMANDS

    cmd = command or DEFAULT_COMMANDS.get(lang)
    if not cmd:
        console.print(f"[red]Unknown language: {lang}[/red]")
        raise typer.Exit(1)

    console.rule(f"[bold]Sandbox[/bold]: {lang} — {cmd}")
    stdout, stderr, exit_code = run_in_sandbox(cmd, workdir=workdir, lang=lang)

    console.print(f"[bold]Exit code:[/bold] {exit_code}")
    if stdout:
        console.print("[bold]stdout:[/bold]")
        console.print(stdout)
    if stderr:
        console.print("[bold red]stderr:[/bold red]")
        console.print(stderr)

    if exit_code != 0:
        raise typer.Exit(exit_code)


# ---------------------------------------------------------------------------
# workspace command
# ---------------------------------------------------------------------------


@app.command()
def workspace(
    path: str = typer.Argument(".", help="Root directory of the codebase"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    apply: bool = typer.Option(False, "--apply", "-a", help="Apply changes to the graph"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """
    Show uncommitted changes in the working tree (git diff HEAD).
    
    This command compares the current working directory against HEAD commit
    and shows what would change in the graph if applied.
    
    Use --apply to update the graph with uncommitted changes.
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    cfg = _load_config(config)
    driver = create_driver(cfg.neo4j)
    root = os.path.abspath(path)

    console.rule(f"[bold blue]AST-RAG workspace[/bold blue]: {root}")

    with console.status("Computing workspace diff..."):
        diff = get_workspace_diff(driver, root, exclude_dirs=cfg.exclude_patterns)

    if diff.is_empty:
        console.print("[green]No uncommitted changes.[/green]")
        console.rule("[bold green]Done[/bold green]")
        return

    # Show summary
    console.print(
        f"[green]+{len(diff.added_nodes)} added[/green]  "
        f"[yellow]~{len(diff.updated_nodes)} updated[/yellow]  "
        f"[red]-{len(diff.deleted_node_ids)} deleted[/red]  (nodes)"
    )
    console.print(
        f"[green]+{len(diff.added_edges)} added[/green]  "
        f"[red]-{len(diff.deleted_edge_ids)} deleted[/red]  (edges)"
    )

    if apply:
        with console.status("Applying workspace diff to graph..."):
            apply_workspace_diff(driver, root, exclude_dirs=cfg.exclude_patterns)

        with console.status("Updating embeddings..."):
            embed = EmbeddingManager(cfg.qdrant, cfg.embedding, neo4j_driver=driver)
            embed.update_embeddings(
                diff.added_nodes,
                diff.updated_nodes,
                diff.deleted_node_ids,
            )
        
        console.print("[green]Workspace changes applied to graph and embeddings.[/green]")
    else:
        console.print("\n[yellow]Hint: Use --apply to apply these changes to the graph.[/yellow]")

    console.rule("[bold green]Done[/bold green]")


# ---------------------------------------------------------------------------
# evaluate command
# ---------------------------------------------------------------------------


@app.command()
def evaluate(
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Run single query file"),
    all_queries: bool = typer.Option(False, "--all", "-a", help="Run all benchmarks"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config JSON"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file for results"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """
    Evaluate AST-RAG quality against ground truth benchmarks.

    By default, runs all benchmarks in benchmarks/queries/.

    Examples:

      ast-rag evaluate --all
      ast-rag evaluate --query benchmarks/queries/def_001.json
      ast-rag evaluate --all --output results.json
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    import json
    import time
    from pathlib import Path

    from ast_rag.ast_rag_api import ASTRagAPI
    from ast_rag.graph_schema import create_driver
    from ast_rag.embeddings import EmbeddingManager

    # Load configuration
    cfg = _load_config(config)

    # Initialize components
    console.rule("[bold blue]AST-RAG QUALITY EVALUATION[/bold blue]")
    console.print("[yellow]Initializing Neo4j and EmbeddingManager...[/yellow]")

    driver = create_driver(cfg.neo4j)
    embed = EmbeddingManager(cfg.qdrant, cfg.embedding, neo4j_driver=driver)
    api = ASTRagAPI(driver, embed)

    def run_query(query: dict) -> dict:
        """Execute a benchmark query."""
        tool_name = query["expected_tool"]
        params = query["expected_params"]

        start_time = time.time()

        if tool_name == "find_references":
            results = api.find_references(
                name=params["name"],
                kind=params.get("kind"),
                lang=params.get("lang"),
                limit=params.get("limit", 50),
            )
            returned_items = results.get("references", [])

        elif tool_name == "find_definition":
            results = api.find_definition(
                name=params["name"],
                kind=params.get("kind"),
                lang=params.get("lang"),
            )
            returned_items = results

        elif tool_name == "find_callers":
            defs = api.find_definition(params["name"], lang=params.get("lang"))
            if defs:
                results = api.find_callers(defs[0].id, max_depth=params.get("depth", 1))
                returned_items = results
            else:
                returned_items = []

        elif tool_name == "search_semantic":
            results = api.search_semantic(
                query=params["query"],
                limit=params.get("limit", 20),
                lang=params.get("lang"),
                kind=params.get("kind"),
            )
            returned_items = list(results)

        elif tool_name == "search_by_signature":
            results = api.search_by_signature(
                pattern=params["signature"],
                lang=params.get("lang"),
                limit=params.get("limit", 20),
            )
            returned_items = list(results)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

        elapsed = time.time() - start_time

        return {
            "items": returned_items,
            "count": len(returned_items),
            "elapsed": elapsed,
        }

    def evaluate_query(query_file: str) -> dict:
        """Evaluate a single benchmark query."""
        query_path = Path(query_file)
        with open(query_path, "r") as f:
            query = json.load(f)

        # Load ground truth
        gt_file = Path("benchmarks") / query["ground_truth_file"]
        with open(gt_file, "r") as f:
            ground_truth = json.load(f)

        # Run query
        tool_name = query["expected_tool"]
        result = run_query(query)

        # Get expected data
        gt_data = ground_truth["ground_truth"]
        returned = result["items"]

        if tool_name == "search_semantic":
            expected = gt_data.get("results", [])
        elif tool_name == "search_by_signature":
            expected = gt_data.get("functions", [])
        elif tool_name == "find_definition":
            expected = gt_data.get("definitions", [])
        elif tool_name == "find_callers":
            expected = gt_data.get("callers", [])
        else:
            expected = gt_data.get("references", [])

        # Normalize for comparison
        if tool_name == "search_semantic":
            expected_set = {(e["file"], e["line"], e.get("name", "")) for e in expected}
            returned_set = {(r.node.file_path, r.node.start_line, r.node.name) for r in returned}
        elif tool_name == "search_by_signature":
            expected_set = {(e["file"], e["line"], e.get("name", "")) for e in expected}
            returned_set = {(r.file_path, r.start_line, r.name) for r in returned}
        elif tool_name == "find_definition":
            expected_set = {(e["file"], e["line"]) for e in ground_truth["ground_truth"]["definitions"]}
            returned_set = {(r.file_path, r.start_line) for r in returned}
        elif tool_name == "find_callers":
            expected_set = {(e["file"], e["line"]) for e in ground_truth["ground_truth"]["callers"]}
            returned_set = {(r.file_path, r.start_line) for r in returned}
        else:
            expected_set = {(e["file"], e["line"]) for e in expected}
            returned_set = {(r["node"]["file_path"], r["node"]["start_line"]) for r in returned}

        # Calculate metrics
        tp = len(expected_set & returned_set)
        fp = len(returned_set - expected_set)
        fn = len(expected_set - returned_set)

        if len(expected_set) == 0 and len(returned_set) == 0:
            precision = 1.0
            recall = 1.0
        elif len(expected_set) == 0 and len(returned_set) > 0:
            precision = 0.0
            recall = 0.0
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        max_time = query["evaluation"]["max_time_seconds"]
        time_score = 1.0 if result["elapsed"] < max_time else 0.5

        return {
            "benchmark_id": query["id"],
            "tool": tool_name,
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "time_seconds": result["elapsed"],
                "time_score": time_score,
            },
            "counts": {
                "expected": len(expected),
                "returned": result["count"],
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
            },
            "passed": {
                "precision": precision >= query["evaluation"]["min_precision"],
                "recall": recall >= query["evaluation"]["min_recall"],
                "time": result["elapsed"] < max_time,
            },
            "overall_pass": (
                precision >= query["evaluation"]["min_precision"]
                and recall >= query["evaluation"]["min_recall"]
                and result["elapsed"] < max_time
            ),
        }

    # Run evaluation
    if query:
        # Single query
        console.print(f"\n[bold]Evaluating:[/bold] {query}")
        result = evaluate_query(query)
        status = "[green]✅ PASS[/green]" if result["overall_pass"] else "[red]❌ FAIL[/red]"
        console.print(f"{status} F1={result['metrics']['f1_score']:.2f} P={result['metrics']['precision']:.2f} R={result['metrics']['recall']:.2f}")
        console.print(json.dumps(result, indent=2))
    else:
        # All queries
        queries_dir = Path("benchmarks/queries")
        if not queries_dir.exists():
            console.print("[red]Error: benchmarks/queries/ directory not found[/red]")
            console.print("[yellow]Run from project root or create benchmark queries.[/yellow]")
            return

        results = []
        for query_file in queries_dir.glob("*.json"):
            console.print(f"\n[bold]🔍 Running:[/bold] {query_file.name}")
            result = evaluate_query(str(query_file))
            results.append(result)

            status = "[green]✅ PASS[/green]" if result["overall_pass"] else "[red]❌ FAIL[/red]"
            console.print(
                f"   {status} F1={result['metrics']['f1_score']:.2f} "
                f"P={result['metrics']['precision']:.2f} "
                f"R={result['metrics']['recall']:.2f} "
                f"t={result['metrics']['time_seconds']:.2f}s"
            )

        # Summary
        total = len(results)
        passed = sum(1 for r in results if r["overall_pass"])
        avg_f1 = sum(r["metrics"]["f1_score"] for r in results) / total if total > 0 else 0
        avg_precision = sum(r["metrics"]["precision"] for r in results) / total if total > 0 else 0
        avg_recall = sum(r["metrics"]["recall"] for r in results) / total if total > 0 else 0

        console.rule("[bold]SUMMARY[/bold]")
        console.print(f"\n[bold]📊 Benchmarks:[/bold] {total}")
        console.print(f"   [green]✅ Passed:[/green] {passed}")
        console.print(f"   [red]❌ Failed:[/red] {total - passed}")
        console.print(f"   [bold]📈 Pass Rate:[/bold] {passed / total * 100:.1f}%")
        console.print("\n[bold]📈 Average Metrics:[/bold]")
        console.print(f"   F1 Score: [green]{avg_f1:.2f}[/green]")
        console.print(f"   Precision: [green]{avg_precision:.2f}[/green]")
        console.print(f"   Recall: [green]{avg_recall:.2f}[/green]")

        # Save results
        output_path = Path(output) if output else Path("benchmarks/results/evaluation.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "total_benchmarks": total,
                "passed": passed,
                "pass_rate": passed / total if total > 0 else 0,
                "average_metrics": {
                    "f1_score": avg_f1,
                    "precision": avg_precision,
                    "recall": avg_recall,
                },
                "results": results,
            }, f, indent=2)
        console.print(f"\n[green]💾 Results saved to:[/green] {output_path}")

    driver.close()


# ---------------------------------------------------------------------------
# sig command (signature search)
# ---------------------------------------------------------------------------


@app.command(name="sig")
def signature_search(
    pattern: str = typer.Argument(..., help="Signature pattern (e.g., 'process(int, String)')"),
    lang: Optional[str] = typer.Option(None, "--lang", "-l", help="Filter by language"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """
    Search by signature pattern.

    Examples:

      ast-rag sig "process(int, String)"
      ast-rag sig "get*" --lang java
      ast-rag sig "*Handler" --lang python
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    cfg = _load_config(config)
    api = _build_api(cfg)

    console.rule(f"[bold blue]SIGNATURE SEARCH[/bold blue]: {pattern}")

    results = list(api.search_by_signature(pattern, lang=lang, limit=limit))

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print(f"[green]Found {len(results)} results:[/green]\n")

    for r in results:
        console.print(
            f"[cyan]{r.file_path}[/cyan]:[yellow]{r.start_line}[/yellow]  "
            f"[bold]{r.name}[/bold]"
        )

    console.rule("[bold green]Done[/bold green]")


# ---------------------------------------------------------------------------
# index-folder command
# ---------------------------------------------------------------------------


@app.command()
def index_folder(
    path: str = typer.Argument(..., help="Folder to index"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config JSON"),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of parallel workers"),
    batch_size: int = typer.Option(50, "--batch-size", "-b", help="Batch size for Neo4j uploads"),
    no_schema: bool = typer.Option(False, "--no-schema", help="Skip schema application"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """
    Index a single folder into the graph.

    Useful for indexing specific directories without re-indexing everything.

    Examples:

      ast-rag index-folder /path/to/folder
      ast-rag index-folder ./src --workers 8
      ast-rag index-folder ./module --no-schema  # if schema already applied
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from pathlib import Path

    from ast_rag.ast_parser import ParserManager, EXT_TO_LANG
    from ast_rag.graph_schema import create_driver, apply_schema
    from ast_rag.graph_updater import _nodes_to_batch_by_label, batch_upsert_nodes, batch_upsert_edges

    cfg = _load_config(config)
    folder_path = Path(path).resolve()

    if not folder_path.exists():
        console.print(f"[red]Error: Folder not found: {folder_path}[/red]")
        return

    console.rule(f"[bold blue]INDEXING FOLDER[/bold blue]: {folder_path}")

    # Connect to Neo4j
    console.print("[yellow]Connecting to Neo4j...[/yellow]")
    driver = create_driver(cfg.neo4j)
    if not no_schema:
        console.print("[yellow]Applying schema...[/yellow]")
        apply_schema(driver)
    else:
        console.print("[green]Schema skipped (--no-schema)[/green]")

    # Scan files
    console.print(f"[yellow]Scanning {folder_path}...[/yellow]")
    all_files = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        # Skip hidden and build directories
        dirnames[:] = [
            d for d in dirnames if d not in [
                ".git", ".venv", "venv", "__pycache__", "node_modules",
                "target", "build", "dist", ".idea", ".vscode", ".pytest_cache"
            ]
        ]
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            lang = EXT_TO_LANG.get(ext)
            if lang:
                fp = os.path.join(dirpath, fname)
                all_files.append((fp, lang))

    console.print(f"[green]Found {len(all_files)} files to index[/green]")

    if not all_files:
        console.print("[yellow]No indexable files, exiting.[/yellow]")
        driver.close()
        return

    # Parser function for multiprocessing
    def parse_file(args):
        file_path, lang, source = args
        commit = os.environ.get("AST_RAG_COMMIT", "INIT")
        try:
            pm = ParserManager()
            tree = pm.parse_file(file_path, source=source)
            if tree is None:
                return (file_path, [], [])
            nodes = pm.extract_nodes(tree, file_path, lang, source, commit)
            edges = pm.extract_edges(tree, nodes, file_path, lang, source, commit)
            return (file_path, nodes, edges)
        except Exception as e:
            return (file_path, [], str(e))

    # Index in batches
    total_nodes = 0
    total_edges = 0
    errors = 0
    start_time = time.time()

    for i in range(0, len(all_files), batch_size):
        batch = all_files[i:i+batch_size]
        batch_start = time.time()

        # Read files
        files_with_source = []
        for fp, lang in batch:
            try:
                with open(fp, "rb") as f:
                    source = f.read()
                files_with_source.append((fp, lang, source))
            except Exception:
                errors += 1

        # Parse in parallel
        parsed = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(parse_file, args) for args in files_with_source]
            for future in as_completed(futures):
                parsed.append(future.result())

        # Collect nodes and edges
        batch_nodes = []
        batch_edges = []
        for fp, nodes, edges_or_error in parsed:
            if isinstance(edges_or_error, str):
                errors += 1
            else:
                batch_nodes.extend(nodes)
                batch_edges.extend(edges_or_error)

        # Insert to Neo4j
        if batch_nodes:
            try:
                with driver.session() as session:
                    by_label = _nodes_to_batch_by_label(batch_nodes)
                    for label, props_list in by_label.items():
                        batch_upsert_nodes(session, {label: props_list})

                    all_edge_dicts = [e.to_neo4j_props() for e in batch_edges]
                    batch_upsert_edges(session, all_edge_dicts)

                total_nodes += len(batch_nodes)
                total_edges += len(batch_edges)
            except Exception as e:
                console.print(f"[red]Neo4j error: {e}[/red]")
                errors += 1

        # Progress
        elapsed = time.time() - start_time
        files_done = min(i + batch_size, len(all_files))
        files_per_sec = files_done / elapsed if elapsed > 0 else 0

        console.print(
            f"[cyan][{files_done:>6}/{len(all_files)}][/cyan] "
            f"+{len(batch_nodes):>4} nodes, +{len(batch_edges):>4} edges | "
            f"{files_per_sec:>5.1f} files/s | "
            f"Batch: {time.time()-batch_start:.2f}s"
        )

    total_time = time.time() - start_time

    console.rule("[bold green]FOLDER COMPLETE[/bold green]")
    console.print(f"[bold]Time:[/bold]      {total_time/60:.1f} minutes")
    console.print(f"[bold]Files:[/bold]     {files_done}")
    console.print(f"[bold]Speed:[/bold]     {files_done/total_time:.1f} files/s")
    console.print(f"[bold]Nodes:[/bold]     {total_nodes:,}")
    console.print(f"[bold]Edges:[/bold]     {total_edges:,}")
    console.print(f"[bold]Errors:[/bold]    {errors}")

    driver.close()


if __name__ == "__main__":
    app()
