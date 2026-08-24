"""
cli.py - Typer-based CLI for AST-RAG.

Commands:
  init   <path>                     Full indexing of a codebase
  create-database <name>           Create a new Neo4j database
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
  summarize <qualified_name>        Generate LLM summary for a function/class
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import typer
from neo4j.exceptions import AuthError
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ast_rag.models import ProjectConfig
from ast_rag.services.parsing.parser_manager import ParserManager, walk_source_files
from ast_rag.repositories import apply_schema, create_driver
from ast_rag.services.graph_updater_service import (
    full_index,
    update_from_git,
    get_workspace_diff,
    apply_workspace_diff,
)
from ast_rag.services.embedding_manager import EmbeddingManager
from ast_rag.api import ASTRagAPI
from ast_rag.utils.output import err_console, get_formatter
from ast_rag.services.summarizer_service import SummarizerService

app = typer.Typer(
    name="ast-rag",
    help="AST-based Retrieval-Augmented Generation for code analysis.",
    add_completion=False,
)
console = Console()


def _index_progress() -> Progress:
    """Progress bar used by the indexing phases.

    Indexing a large repository can run for many minutes -- embedding in
    particular -- so the phases report count, elapsed and ETA rather than an
    indeterminate spinner. `transient` keeps the finished bar from cluttering
    the summary output.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )


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
    """Load project config from JSON file or return defaults.

    Resolution order, lowest priority first: built-in defaults (which point at
    localhost, so a fresh clone works against the documented Docker setup with
    no config file at all), then ``ast_rag_config.json`` in the CWD, then an
    explicit ``--config``, then ``AST_RAG_*`` environment variables.

    The env layer is last so that a container or CI job can point the same
    checkout at different services without editing a tracked file.
    """
    if config_path and Path(config_path).exists():
        cfg = ProjectConfig.model_validate_json(Path(config_path).read_text())
    else:
        default = Path("ast_rag_config.json")
        if default.exists():
            cfg = ProjectConfig.model_validate_json(default.read_text())
        else:
            cfg = ProjectConfig()

    return _apply_env_overrides(cfg)


def _apply_env_overrides(cfg: ProjectConfig) -> ProjectConfig:
    """Overlay ``AST_RAG_*`` environment variables onto a loaded config."""
    env_map = {
        "AST_RAG_NEO4J_URI": (cfg.neo4j, "uri"),
        "AST_RAG_NEO4J_USER": (cfg.neo4j, "user"),
        "AST_RAG_NEO4J_PASSWORD": (cfg.neo4j, "password"),
        "AST_RAG_NEO4J_DATABASE": (cfg.neo4j, "database"),
        "AST_RAG_QDRANT_URL": (cfg.qdrant, "url"),
        "AST_RAG_QDRANT_COLLECTION": (cfg.qdrant, "collection_name"),
    }
    for var, (section, field) in env_map.items():
        value = os.environ.get(var)
        if value:
            setattr(section, field, value)
    return cfg


def _connect(cfg: ProjectConfig):
    """Create a Neo4j driver and fail fast with a usable message if it is down.

    Every command needs a driver, so the reachability check lives here rather
    than being repeated at each call site.
    """
    driver = create_driver(cfg.neo4j)
    _verify_neo4j(driver, cfg)
    return driver


def _build_api(cfg: ProjectConfig) -> ASTRagAPI:
    driver = _connect(cfg)
    embed = EmbeddingManager(cfg.qdrant, cfg.embedding, neo4j_driver=driver)
    return ASTRagAPI(driver, embed)


def _lang_of(node) -> str:
    """Language as a plain string, whether it is an enum or already a str."""
    lang = getattr(node, "lang", None)
    return getattr(lang, "value", None) or str(lang or "unknown")


def _describe_ambiguity(name: str, defs: list) -> Optional[str]:
    """Describe a name that resolved to more than one symbol, else None.

    ``find_definition`` returns every match and the commands act on the first,
    so without this the caller cannot tell that a choice was made at all -- an
    empty result reads as "this symbol has nothing" rather than "I looked at
    the wrong symbol". See #64.

    Leads with "ambiguous" so an agent scanning the output hits it first, and
    avoids an exclamation mark, per the discussion on the issue.
    """
    if len(defs) < 2:
        return None

    counts: dict[str, int] = {}
    for d in defs:
        lang = _lang_of(d)
        counts[lang] = counts.get(lang, 0) + 1
    breakdown = ", ".join(
        f"{n} {lang}" for lang, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )

    chosen = defs[0].qualified_name
    alternatives = [d.qualified_name for d in defs[1:4] if d.qualified_name != chosen]
    hint = ""
    if alternatives:
        hint = " Re-run with a qualified name to pick another, for example " + ", ".join(
            alternatives
        )

    return (
        f"ambiguous: '{name}' matched {len(defs)} symbols ({breakdown}). "
        f"Reporting on {chosen}.{hint}"
    )


def _warn_if_ambiguous(name: str, defs: list) -> Optional[str]:
    """Print the ambiguity note (if any) and return it for JSON consumers.

    The note goes to stderr: stdout is a JSON document by default, and a
    Rich-wrapped line printed into the middle of it is not parseable.
    Callers that use a formatter pass the return value on so the note
    survives inside the document too.
    """
    note = _describe_ambiguity(name, defs)
    if note:
        err_console.print(f"[yellow]{note}[/yellow]")
    return note


def _verify_neo4j(driver, cfg: ProjectConfig) -> None:
    """Fail fast with a usable message when Neo4j is not reachable.

    Without this the first query blocks for the driver's default connection
    timeout and then raises ServiceUnavailable, which Typer renders as a full
    traceback. For anyone who has not started the services yet -- the common
    case on a fresh clone -- that is a wall of Bolt internals rather than
    "start Neo4j".
    """
    try:
        driver.verify_connectivity()
    except AuthError:
        console.print(
            f"[red]Neo4j rejected the credentials for[/red] {cfg.neo4j.uri} "
            f"(user: {cfg.neo4j.user}).\n"
            "Set AST_RAG_NEO4J_USER / AST_RAG_NEO4J_PASSWORD, or edit "
            "ast_rag_config.json."
        )
        raise typer.Exit(code=1)
    except Exception as exc:  # ServiceUnavailable and friends
        console.print(
            f"[red]Cannot reach Neo4j at[/red] {cfg.neo4j.uri}\n\n"
            "Start the services:\n"
            "  [cyan]docker compose up -d[/cyan]\n\n"
            "Or point AST-RAG somewhere else:\n"
            "  [cyan]export AST_RAG_NEO4J_URI=bolt://host:7687[/cyan]\n\n"
            f"[dim]{type(exc).__name__}: {str(exc).splitlines()[0]}[/dim]"
        )
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# init command
# ---------------------------------------------------------------------------


@app.command()
def init(
    path: str = typer.Argument(..., help="Root directory of the codebase to index"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config JSON"),
    commit: str = typer.Option("INIT", "--commit", help="Commit hash label for this index"),
    cache_entries: Optional[int] = typer.Option(
        None, "--cache-entries", help="Max parse cache entries (overrides config)"
    ),
    cache_memory_mb: Optional[int] = typer.Option(
        None, "--cache-memory-mb", help="Max parse cache memory in MB (overrides config)"
    ),
    ignore_file: Optional[str] = typer.Option(
        None,
        "--ignore-file",
        "-i",
        help="Path to a .cgrignore-style ignore file (default: .cgrignore in PATH)",
    ),
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
    driver = _connect(cfg)
    with console.status("Applying Neo4j schema..."):
        apply_schema(driver)

    # 2. Parse all source files
    pc_cfg = cfg.parse_cache.model_dump()
    if cache_entries is not None:
        pc_cfg["max_entries"] = cache_entries
    if cache_memory_mb is not None:
        pc_cfg["max_size_mb"] = cache_memory_mb
    pm = ParserManager(
        project_id=cfg.neo4j.project_id,
        config={"parse_cache": pc_cfg},
    )
    files = walk_source_files(
        root,
        exclude_dirs=cfg.exclude_patterns,
        ignore_file=ignore_file or cfg.ignore_file,
    )
    console.print(f"Found [bold]{len(files)}[/bold] source files.")

    all_nodes = []
    all_edges = []
    all_blocks = []
    all_block_edges = []

    # Two-phase index. Edge resolution matches references against a name -> id
    # map; when that map only holds the current file's nodes, any reference to a
    # symbol defined elsewhere is silently dropped. Collecting every symbol first
    # and resolving afterwards is what lets cross-file references link at all.
    parsed: list[tuple[str, str, bytes, object, list]] = []
    global_symbols: dict[str, str] = {}

    with _index_progress() as progress:
        task = progress.add_task("Parsing", total=len(files))
        for fp, lang in files:
            progress.update(task, description=f"Parsing {os.path.relpath(fp, root)}")
            progress.advance(task)
            tree = pm.parse_file(fp)
            if tree is None:
                continue
            with open(fp, "rb") as fh:
                source = fh.read()
            nodes = pm.extract_nodes(tree, fp, lang, source, commit)
            parsed.append((fp, lang, source, tree, nodes))
            all_nodes.extend(nodes)
            # First definition of a name wins; files are walked in a stable order
            # so the choice is deterministic across runs.
            for node in nodes:
                global_symbols.setdefault(node.name, node.id)

    with _index_progress() as progress:
        task = progress.add_task("Resolving", total=len(parsed))
        for fp, lang, source, tree, nodes in parsed:
            progress.update(task, description=f"Resolving {os.path.relpath(fp, root)}")
            progress.advance(task)
            all_edges.extend(
                pm.extract_edges(
                    tree, nodes, fp, lang, source, commit, global_symbols=global_symbols
                )
            )

            # Extract blocks for Python and Rust files
            if lang in ("python", "rust"):
                blocks, block_edges = pm.extract_blocks(tree, nodes, fp, lang, source, commit)
                all_blocks.extend(blocks)
                all_block_edges.extend(block_edges)

    console.print(
        f"Extracted [bold]{len(all_nodes)}[/bold] nodes, [bold]{len(all_edges)}[/bold] edges, "
        f"and [bold]{len(all_blocks)}[/bold] blocks."
    )

    # 3. Write to Neo4j
    with console.status("Writing to Neo4j..."):
        full_index(driver, all_nodes, all_edges, commit_hash=commit)

        # Store blocks and CONTAINS_BLOCK edges
        if all_blocks:
            from ast_rag.services.graph_updater_service import (
                batch_upsert_blocks,
                batch_upsert_block_edges,
            )

            with driver.session() as session:
                batch_upsert_blocks(session, all_blocks, commit)
                batch_upsert_block_edges(session, all_block_edges)
            console.print(
                f"[green]Stored {len(all_blocks)} blocks and {len(all_block_edges)} CONTAINS_BLOCK edges.[/green]"
            )

    console.print("[green]Graph database updated.[/green]")

    # 4. Build embeddings
    embed = EmbeddingManager(cfg.qdrant, cfg.embedding, neo4j_driver=driver)
    with _index_progress() as progress:
        task = progress.add_task("Building embeddings", total=None)

        def _on_batch(done: int, total: int) -> None:
            progress.update(task, completed=done, total=total or None)

        count = embed.build_embeddings(all_nodes, progress_callback=_on_batch)
    console.print(f"[green]Indexed {count} node embeddings.[/green]")

    console.rule("[bold green]Done[/bold green]")


# ---------------------------------------------------------------------------
# create-database command
# ---------------------------------------------------------------------------


@app.command()
def create_database(
    database_name: str = typer.Argument(..., help="Name of the Neo4j database to create"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """
    Create a new Neo4j database.

    If the database already exists, it will be skipped.

    Examples:

      ast-rag create-database my_project
      ast-rag create-database lrp --config ast_rag_config.json
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    cfg = _load_config(config)

    console.rule(f"[bold blue]Create Neo4j Database[/bold blue] {database_name}")
    console.print(f"Neo4j URI: {cfg.neo4j.uri}")
    console.print(f"Database: {database_name}")

    from neo4j import GraphDatabase
    from neo4j.exceptions import ClientError

    # Connect to default database
    driver = GraphDatabase.driver(
        cfg.neo4j.uri,
        auth=(cfg.neo4j.user, cfg.neo4j.password),
        database="neo4j",
    )

    try:
        with driver.session(database="neo4j") as session:
            # Check if database exists
            result = session.run(
                "SHOW DATABASES WHERE name = $name",
                name=database_name,
            )
            db_info = result.single()

            if db_info is None:
                console.print("[yellow]Database does not exist. Creating...[/yellow]")
                session.run(f"CREATE DATABASE `{database_name}`")
                console.print(f"[green]Database '{database_name}' created successfully.[/green]")
            else:
                current_status = db_info.get("currentStatus", "unknown")
                console.print(
                    f"[green]Database '{database_name}' already exists (status: {current_status}).[/green]"
                )
    except ClientError as exc:
        console.print(f"[red]Neo4j error: {exc.code} - {exc.message}[/red]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        driver.close()


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
    driver = _connect(cfg)
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
        None,
        "--vector-weight",
        "-vw",
        help="Weight for vector similarity (0.0-1.0). Overrides config if specified.",
    ),
    keyword_weight: Optional[float] = typer.Option(
        None,
        "--keyword-weight",
        "-kw",
        help="Weight for keyword search (0.0-1.0). Overrides config if specified.",
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
        err_console.print("[yellow]No results found.[/yellow]")
        formatter.format_search_results([], text)
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
        err_console.print(f"[yellow]Definition not found for: {qualified_name}[/yellow]")
        formatter.format_definitions([], api=api, snippet=snippet)
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
        err_console.print(f"[yellow]Symbol not found: {qualified_name}[/yellow]")
        formatter.format_callers(qualified_name, [])
        raise typer.Exit(1)

    note = _warn_if_ambiguous(qualified_name, defs)
    target = defs[0]
    if humanize:
        err_console.print(f"Finding callers of [bold]{target.qualified_name}[/bold]...")

    caller_nodes = api.find_callers(target.id, max_depth=depth)

    if not caller_nodes:
        err_console.print("[yellow]No callers found.[/yellow]")
        formatter.format_callers(target.qualified_name, [], note=note)
        raise typer.Exit(0)

    formatter.format_callers(target.qualified_name, caller_nodes, note=note)


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
    driver = _connect(cfg)
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
                    f"  {node['start_line']:4d}: {ref['reference_type']:10s} {node['name']}"
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
    direction: str = typer.Option(
        "both", "--direction", "-d", help="Direction: callers, callees, or both"
    ),
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

    _warn_if_ambiguous(name, defs)
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

    _warn_if_ambiguous(name, defs)
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
    driver = _connect(cfg)
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

    from ast_rag.api import ASTRagAPI
    from ast_rag.services.embedding_manager import EmbeddingManager

    # Load configuration
    cfg = _load_config(config)

    # Initialize components
    console.rule("[bold blue]AST-RAG QUALITY EVALUATION[/bold blue]")
    console.print("[yellow]Initializing Neo4j and EmbeddingManager...[/yellow]")

    driver = _connect(cfg)
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
                _warn_if_ambiguous(params["name"], defs)
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
            expected_set = {
                (e["file"], e["line"]) for e in ground_truth["ground_truth"]["definitions"]
            }
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
        console.print(
            f"{status} F1={result['metrics']['f1_score']:.2f} P={result['metrics']['precision']:.2f} R={result['metrics']['recall']:.2f}"
        )
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
            json.dump(
                {
                    "total_benchmarks": total,
                    "passed": passed,
                    "pass_rate": passed / total if total > 0 else 0,
                    "average_metrics": {
                        "f1_score": avg_f1,
                        "precision": avg_precision,
                        "recall": avg_recall,
                    },
                    "results": results,
                },
                f,
                indent=2,
            )
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
            f"[cyan]{r.file_path}[/cyan]:[yellow]{r.start_line}[/yellow]  [bold]{r.name}[/bold]"
        )

    console.rule("[bold green]Done[/bold green]")


# ---------------------------------------------------------------------------
# index-folder command
# ---------------------------------------------------------------------------


def _print_batch_progress(done, total, n_nodes, n_edges, batch_start, start_time):
    """One progress line per batch, shared by both indexing phases."""
    elapsed = time.time() - start_time
    rate = done / elapsed if elapsed > 0 else 0
    console.print(
        f"[cyan][{done:>6}/{total}][/cyan] "
        f"+{n_nodes:>4} nodes, +{n_edges:>4} edges | "
        f"{rate:>5.1f} files/s | "
        f"Batch: {time.time() - batch_start:.2f}s"
    )


# Set once per worker process by _init_symbol_worker so the project-wide symbol
# map is pickled per worker rather than per file.
_WORKER_SYMBOLS: dict[str, str] = {}


def _init_symbol_worker(symbols: dict[str, str]) -> None:
    """ProcessPoolExecutor initializer: publish the symbol map to this worker."""
    global _WORKER_SYMBOLS
    _WORKER_SYMBOLS = symbols


def _worker_sys_path() -> None:
    """Make ast_rag importable in a subprocess started with spawn."""
    import os
    import sys

    project_root = os.environ.get("AST_RAG_PROJECT_ROOT")
    if project_root and project_root not in sys.path:
        sys.path.insert(0, project_root)


def _parse_nodes_for_multiprocessing(args):
    """Phase 1: extract nodes only. Must be module level for pickle."""
    import os

    _worker_sys_path()
    file_path, lang, source = args
    commit = os.environ.get("AST_RAG_COMMIT", "INIT")
    project_id = os.environ.get("AST_RAG_PROJECT_ID", "default")
    try:
        from ast_rag.services.parsing.parser_manager import ParserManager

        pm = ParserManager(project_id=project_id)
        tree = pm.parse_file(file_path, source=source)
        if tree is None:
            return (file_path, [], None)
        nodes = pm.extract_nodes(tree, file_path, lang, source, commit)
        return (file_path, nodes, None)
    except Exception as e:
        return (file_path, [], str(e))


def _parse_edges_for_multiprocessing(args):
    """Phase 2: extract edges and blocks, resolving against the symbol map.

    The tree is re-parsed rather than carried over from phase 1 because
    tree-sitter trees are not picklable across processes. Blocks are extracted
    here too rather than in a third pass, since this phase already holds both
    the tree and the file's nodes, which is everything extract_blocks needs.
    """
    import os

    _worker_sys_path()
    file_path, lang, source = args
    commit = os.environ.get("AST_RAG_COMMIT", "INIT")
    project_id = os.environ.get("AST_RAG_PROJECT_ID", "default")
    try:
        from ast_rag.services.parsing.parser_manager import ParserManager

        pm = ParserManager(project_id=project_id)
        tree = pm.parse_file(file_path, source=source)
        if tree is None:
            return (file_path, [], [], [], None)
        nodes = pm.extract_nodes(tree, file_path, lang, source, commit)
        edges = pm.extract_edges(
            tree, nodes, file_path, lang, source, commit, global_symbols=_WORKER_SYMBOLS
        )
        blocks: list = []
        block_edges: list = []
        if lang in ("python", "rust"):
            blocks, block_edges = pm.extract_blocks(tree, nodes, file_path, lang, source, commit)
        return (file_path, edges, blocks, block_edges, None)
    except Exception as e:
        return (file_path, [], [], [], str(e))


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
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from pathlib import Path

    # Set project root for subprocess
    os.environ["AST_RAG_PROJECT_ROOT"] = str(Path(__file__).parent.parent.parent)

    from ast_rag.services.parsing.parser_manager import EXT_TO_LANG
    from ast_rag.repositories import apply_schema
    from ast_rag.services.graph_updater_service import (
        _nodes_to_batch_by_label,
        batch_upsert_nodes,
        batch_upsert_edges,
        batch_upsert_blocks,
        batch_upsert_block_edges,
    )

    cfg = _load_config(config)
    folder_path = Path(path).resolve()
    # Workers read this from the environment; keep the parent in step so
    # blocks are stamped with the same commit as the nodes and edges.
    commit = os.environ.get("AST_RAG_COMMIT", "INIT")

    if not folder_path.exists():
        console.print(f"[red]Error: Folder not found: {folder_path}[/red]")
        return

    console.rule(f"[bold blue]INDEXING FOLDER[/bold blue]: {folder_path}")
    console.print(f"Project ID: {cfg.neo4j.project_id}")

    # Set project ID for worker processes
    import os

    os.environ["AST_RAG_PROJECT_ID"] = cfg.neo4j.project_id

    # Connect to Neo4j
    console.print("[yellow]Connecting to Neo4j...[/yellow]")
    driver = _connect(cfg)
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
            d
            for d in dirnames
            if d
            not in [
                ".git",
                ".venv",
                "venv",
                "__pycache__",
                "node_modules",
                "target",
                "build",
                "dist",
                ".idea",
                ".vscode",
                ".pytest_cache",
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

    # Index in batches
    total_nodes = 0
    total_edges = 0
    total_blocks = 0
    errors = 0
    start_time = time.time()

    # Two-phase index, matching `init`. Edge resolution matches references
    # against a name -> id map; when that map holds only the current file's
    # nodes, every reference to a symbol defined elsewhere is silently dropped.
    # Indexing this repo that way produced 330 CALLS edges and *zero* of them
    # crossing a file boundary. Collecting all symbols first is what lets
    # cross-file references link at all. See #56.
    sources: dict[str, bytes] = {}
    for fp, lang in all_files:
        try:
            with open(fp, "rb") as f:
                sources[fp] = f.read()
        except Exception:
            errors += 1

    indexable = [(fp, lang) for fp, lang in all_files if fp in sources]
    global_symbols: dict[str, str] = {}

    # ---- Phase 1: nodes, and the project-wide symbol map -------------------
    console.print("[yellow]Phase 1/2: extracting symbols...[/yellow]")
    for i in range(0, len(indexable), batch_size):
        batch = indexable[i : i + batch_size]
        batch_start = time.time()
        args_list = [(fp, lang, sources[fp]) for fp, lang in batch]

        batch_nodes = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for future in as_completed(
                executor.submit(_parse_nodes_for_multiprocessing, a) for a in args_list
            ):
                fp, nodes, error = future.result()
                if error:
                    errors += 1
                else:
                    batch_nodes.extend(nodes)

        # First definition of a name wins; files are walked in a stable order
        # so the choice is deterministic across runs.
        for node in batch_nodes:
            global_symbols.setdefault(node.name, node.id)

        if batch_nodes:
            try:
                with driver.session() as session:
                    by_label = _nodes_to_batch_by_label(batch_nodes)
                    for label, props_list in by_label.items():
                        batch_upsert_nodes(session, {label: props_list})
                total_nodes += len(batch_nodes)
            except Exception as e:
                console.print(f"[red]Neo4j error: {e}[/red]")
                errors += 1

        _print_batch_progress(
            min(i + batch_size, len(indexable)),
            len(indexable),
            len(batch_nodes),
            0,
            batch_start,
            start_time,
        )

    # ---- Phase 2: edges, resolved against every symbol ---------------------
    console.print(
        f"[yellow]Phase 2/2: resolving edges against {len(global_symbols)} symbols...[/yellow]"
    )
    for i in range(0, len(indexable), batch_size):
        batch = indexable[i : i + batch_size]
        batch_start = time.time()
        args_list = [(fp, lang, sources[fp]) for fp, lang in batch]

        batch_edges = []
        batch_blocks = []
        batch_block_edges = []
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_symbol_worker, initargs=(global_symbols,)
        ) as executor:
            for future in as_completed(
                executor.submit(_parse_edges_for_multiprocessing, a) for a in args_list
            ):
                fp, edges, blocks, block_edges, error = future.result()
                if error:
                    errors += 1
                else:
                    batch_edges.extend(edges)
                    batch_blocks.extend(blocks)
                    batch_block_edges.extend(block_edges)

        if batch_edges or batch_blocks:
            try:
                with driver.session() as session:
                    if batch_edges:
                        batch_upsert_edges(session, [e.to_neo4j_props() for e in batch_edges])
                        total_edges += len(batch_edges)
                    # `init` stores blocks; index-folder never did, so
                    # `ast-rag blocks` and `ast-rag lambdas` returned nothing
                    # for any index built this way.
                    if batch_blocks:
                        batch_upsert_blocks(session, batch_blocks, commit)
                        batch_upsert_block_edges(session, batch_block_edges)
                        total_blocks += len(batch_blocks)
            except Exception as e:
                console.print(f"[red]Neo4j error: {e}[/red]")
                errors += 1

        _print_batch_progress(
            min(i + batch_size, len(indexable)),
            len(indexable),
            0,
            len(batch_edges),
            batch_start,
            start_time,
        )

    total_time = time.time() - start_time
    files_done = len(indexable)

    console.rule("[bold green]FOLDER COMPLETE[/bold green]")
    console.print(f"[bold]Time:[/bold]      {total_time / 60:.1f} minutes")
    console.print(f"[bold]Files:[/bold]     {files_done}")
    console.print(f"[bold]Speed:[/bold]     {files_done / total_time:.1f} files/s")
    console.print(f"[bold]Nodes:[/bold]     {total_nodes:,}")
    console.print(f"[bold]Edges:[/bold]     {total_edges:,}")
    console.print(f"[bold]Blocks:[/bold]    {total_blocks:,}")
    console.print(f"[bold]Errors:[/bold]    {errors}")

    driver.close()


# ---------------------------------------------------------------------------
# blocks command
# ---------------------------------------------------------------------------


@app.command("blocks")
def blocks(
    function: str = typer.Argument(..., help="Function name or ID to get blocks for"),
    block_type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Filter by block type: if/for/while/try/lambda/with/match/loop"
    ),
    lang: Optional[str] = typer.Option(None, "--lang", "-l", help="Language filter"),
    source: bool = typer.Option(False, "--source", "-s", help="Show source code for blocks"),
    stats: bool = typer.Option(False, "--stats", help="Show block statistics"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    humanize: bool = humanize_option,
) -> None:
    """
    Extract and display code blocks (if/for/while/try/lambda/with) from functions.

    Examples:

      # Get all blocks from a function
      ast-rag blocks my_function

      # Get only lambda blocks
      ast-rag blocks my_function --type lambda

      # Get blocks with source code
      ast-rag blocks my_function --type lambda --source

      # Get global block statistics
      ast-rag blocks --stats

      # Get all lambdas with captured variables
      ast-rag blocks --type lambda --lang python
    """
    cfg = _load_config(config)
    api = _build_api(cfg)

    if stats:
        # Show global statistics
        stats_data = api.get_block_statistics()

        if humanize:
            console.print("\n[bold]Block Statistics[/bold]\n")
            console.print(
                f"  Total blocks:      [green]{stats_data.get('total_blocks', 0)}[/green]"
            )
            console.print(f"  If blocks:         {stats_data.get('if_count', 0)}")
            console.print(f"  For blocks:        {stats_data.get('for_count', 0)}")
            console.print(f"  While blocks:      {stats_data.get('while_count', 0)}")
            console.print(f"  Try blocks:        {stats_data.get('try_count', 0)}")
            console.print(f"  Lambda blocks:     {stats_data.get('lambda_count', 0)}")
            console.print(f"  With blocks:       {stats_data.get('with_count', 0)}")
            console.print(f"  Match blocks:      {stats_data.get('match_count', 0)}")
            console.print(f"  Avg nesting:       {stats_data.get('avg_nesting', 0):.2f}")
            console.print(f"  Max nesting:       {stats_data.get('max_nesting', 0)}")
        else:
            print(json.dumps(stats_data, indent=2))
        return

    # Find the function by name
    defs = api.find_definition(function, lang=lang)
    if not defs:
        # Try as function_id directly
        function_id = function
        function_name = function
    else:
        _warn_if_ambiguous(function, defs)
        function_id = defs[0].id
        function_name = defs[0].qualified_name

    if humanize:
        console.print(f"\n[bold]Blocks in[/bold] {function_name} (ID: {function_id[:12]}...)\n")

    # Get blocks
    blocks = api.get_blocks_for_function(function_id, block_type=block_type)

    if not blocks:
        console.print("[yellow]No blocks found.[/yellow]")
        return

    if humanize:
        for i, block in enumerate(blocks, 1):
            block_type_display = block.get("block_type", "unknown").upper()
            console.print(
                f"[cyan]{i}. {block_type_display}[/cyan] block at line {block.get('start_line')}-{block.get('end_line')}"
            )
            console.print(f"   Nesting depth: {block.get('nesting_depth')}")

            if block.get("captured_variables"):
                console.print(
                    f"   Captured vars: [yellow]{', '.join(block['captured_variables'])}[/yellow]"
                )

            if block.get("name"):
                console.print(f"   Name: {block['name']}")

            if source:
                source_code = api.get_block_source(block["id"])
                if source_code:
                    console.print("   [dim]Source:[/dim]")
                    for line in source_code.split("\n")[:5]:  # Show first 5 lines
                        console.print(f"     [dim]{line}[/dim]")
                    if len(source_code.split("\n")) > 5:
                        console.print("     [dim]...[/dim]")
            console.print()
    else:
        print(json.dumps(blocks, indent=2))


# ---------------------------------------------------------------------------
# lambdas command (shortcut for blocks --type lambda)
# ---------------------------------------------------------------------------


@app.command("lambdas")
def list_lambdas(
    lang: Optional[str] = typer.Option(None, "--lang", "-l", help="Language filter"),
    captured: bool = typer.Option(
        # "-c" is the project-wide shorthand for "--config" (used by 19 other
        # commands), so "--captured" takes "-C" to avoid shadowing it. See #65.
        False,
        "--captured",
        "-C",
        help="Only show lambdas with captured variables",
    ),
    source: bool = typer.Option(False, "--source", "-s", help="Show source code"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    humanize: bool = humanize_option,
) -> None:
    """
    List all lambda/closure blocks in the codebase.

    Shortcut for: ast-rag blocks --type lambda

    Examples:

      # Get all lambdas
      ast-rag lambdas

      # Get lambdas with captured variables (closures)
      ast-rag lambdas --captured
      ast-rag lambdas -C

      # Get Python lambdas only
      ast-rag lambdas --lang python
    """
    cfg = _load_config(config)
    api = _build_api(cfg)

    if humanize:
        caption = "Lambda blocks"
        if captured:
            caption += " with captured variables"
        if lang:
            caption += f" in {lang}"
        console.print(f"\n[bold]{caption}[/bold]\n")

    lambdas = api.get_lambda_blocks(lang=lang, with_captured_vars=captured)

    if not lambdas:
        console.print("[yellow]No lambdas found.[/yellow]")
        return

    if humanize:
        for i, lam in enumerate(lambdas, 1):
            console.print(
                f"[cyan]{i}.[/cyan] {lam.get('name', 'lambda')} at line {lam.get('start_line')}"
            )
            console.print(f"   File: {lam.get('file_path')}")
            console.print(f"   Parent: {lam.get('parent_function_id', 'unknown')[:24]}...")

            if lam.get("captured_variables"):
                console.print(
                    f"   Captured: [yellow]{', '.join(lam['captured_variables'])}[/yellow]"
                )

            if source:
                source_code = api.get_block_source(lam["id"])
                if source_code:
                    console.print(f"   [dim]Source: {source_code[:100]}...[/dim]")
            console.print()
    else:
        print(json.dumps(lambdas, indent=2))


# ---------------------------------------------------------------------------
# summarize command
# ---------------------------------------------------------------------------


@app.command("summarize")
def summarize(
    qualified_name: str = typer.Argument(
        ..., help="Qualified name of the function/class to summarize"
    ),
    lang: Optional[str] = typer.Option(None, "--lang", "-l", help="Language filter"),
    kind: Optional[str] = typer.Option(
        None, "--kind", "-k", help="Node kind filter (Function, Method, Class, etc.)"
    ),
    max_callers: int = typer.Option(
        5, "--max-callers", help="Maximum number of callers to include in context"
    ),
    max_callees: int = typer.Option(
        5, "--max-callees", help="Maximum number of callees to include in context"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force regeneration, ignore cache"),
    output: str = typer.Option(
        "markdown", "--output", "-o", help="Output format: markdown, json, text"
    ),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    # LLM options
    llm_base_url: str = typer.Option(
        "http://localhost:11434/v1", "--llm-url", help="Base URL of OpenAI-compatible LLM API"
    ),
    llm_model: str = typer.Option("qwen2.5-coder:14b", "--llm-model", help="LLM model name"),
    llm_api_key: Optional[str] = typer.Option(
        None, "--llm-api-key", help="API key for LLM (not needed for Ollama)"
    ),
) -> None:
    """
    Generate an LLM-based summary for a function, method, or class.

    This command uses a local OpenAI-compatible LLM (Ollama, vLLM, etc.)
    to analyze code and generate a structured summary including:
    - Natural language description
    - Input parameters
    - Return values
    - Side effects
    - Call graph context (callers and callees)
    - Complexity estimate
    - Relevant tags

    Examples:

      # Summarize a function (uses cached result if available)
      ast-rag summarize com.example.MyService.processRequest

      # Summarize with JSON output
      ast-rag summarize MyFunction --output json

      # Force regeneration ignoring cache
      ast-rag summarize MyFunction --force

      # Use a different LLM model
      ast-rag summarize MyClass --llm-model codellama:34b

      # Include more context
      ast-rag summarize MyFunction --max-callers 10 --max-callees 10

    LLM Setup:

      By default, uses Ollama at http://localhost:11434/v1 with model qwen2.5-coder:14b.

      To use with Ollama:
        1. Install Ollama: https://ollama.ai
        2. Pull a model: ollama pull qwen2.5-coder:14b
        3. Ollama runs automatically on localhost:11434

      To use with vLLM or other OpenAI-compatible APIs:
        ast-rag summarize MyFunction --llm-url http://localhost:8000/v1 --llm-model my-model

    Output Formats:

      - markdown: Human-readable Markdown (default)
      - json: Structured JSON with all fields
      - text: Plain text summary
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    cfg = _load_config(config)
    api = _build_api(cfg)

    # Find the node
    with console.status(f"Finding '{qualified_name}'..."):
        defs = api.find_definition(qualified_name, kind=kind, lang=lang)

    if not defs:
        console.print(f"[red]Symbol not found: {qualified_name}[/red]")
        raise typer.Exit(1)

    _warn_if_ambiguous(qualified_name, defs)
    node = defs[0]

    # Check if node kind is summarizable
    summarizable_kinds = {
        "Function",
        "Method",
        "Constructor",
        "Destructor",
        "Class",
        "Interface",
        "Struct",
        "Trait",
    }
    if node.kind.value not in summarizable_kinds:
        console.print(f"[yellow]Warning: {node.kind.value} may not have detailed summary[/yellow]")

    console.print(f"Summarizing [bold]{node.qualified_name}[/bold] ({node.kind.value})")
    console.print(f"  File: {node.file_path}:{node.start_line}-{node.end_line}")

    # Initialize summarizer
    summarizer = SummarizerService(
        base_url=llm_base_url,
        model=llm_model,
        api_key=llm_api_key,
    )

    # Generate summary
    with console.status("Generating summary with LLM..."):
        try:
            summary = summarizer.summarize_node(
                node_id=node.id,
                api=api,
                max_callers=max_callers,
                max_callees=max_callees,
                force_regenerate=force,
            )
        except Exception as exc:
            console.print(f"[red]Error generating summary: {exc}[/red]")
            raise typer.Exit(1)

    # Output
    console.print()

    if output == "json":
        print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))
    elif output == "text":
        console.print(summary.summary)
        console.print()
        if summary.inputs:
            console.print("[bold]Inputs:[/bold]")
            for inp in summary.inputs:
                console.print(f"  - {inp.get('name', 'unknown')}: {inp.get('description', '')}")
        if summary.outputs:
            console.print("[bold]Outputs:[/bold]")
            for out in summary.outputs:
                console.print(f"  - {out.get('name', 'return')}: {out.get('description', '')}")
        if summary.side_effects:
            console.print("[bold]Side Effects:[/bold]")
            for effect in summary.side_effects:
                console.print(f"  - {effect}")
        console.print(f"[bold]Complexity:[/bold] {summary.complexity.value}")
        console.print(f"[bold]Tags:[/bold] {', '.join(summary.tags) if summary.tags else 'none'}")
    else:  # markdown
        console.print(summary.to_markdown())

    # Show cache stats if verbose
    if verbose:
        stats = summarizer.get_cache_stats()
        console.print()
        console.print(f"[dim]Cache: {stats['entries']} entries[/dim]")


# ---------------------------------------------------------------------------
# analyze-stacktrace command
# ---------------------------------------------------------------------------


@app.command("analyze-stacktrace")
def analyze_stacktrace(
    input_path: Optional[str] = typer.Argument(
        None,
        help="Path to file containing stack trace. If not provided, reads from stdin.",
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config JSON file",
    ),
    output: str = typer.Option(
        "markdown",
        "--output",
        "-o",
        help="Output format: markdown, json, text",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
    no_ast_mapping: bool = typer.Option(
        False,
        "--no-ast-mapping",
        help="Skip AST node mapping (faster, less context)",
    ),
    limit_similar: int = typer.Option(
        5,
        "--limit-similar",
        "-n",
        help="Maximum number of similar issues to find",
    ),
) -> None:
    """
    Analyze a stack trace and map it to code with context.

    This command parses stack traces from Python, C++, Java, or Rust,
    maps each frame to AST nodes, retrieves code snippets, and provides
    root cause analysis with suggested fixes.

    **Input:**

    Reads stack trace from:
    - File path (if provided as argument)
    - Standard input (if no argument)

    **Supported Formats:**

    - **Python:** File "x.py", line 42, in func
    - **C++:** #0 0x... in func() at file.cpp:42
    - **Java:** at com.example.Class.method(Class.java:42)
    - **Rust:** at src/file.rs:42

    **Output:**

    Generates a StackTraceReport with:
    - Parsed call chain with frame details
    - Root cause analysis and severity
    - Code snippets for each frame
    - Suggested fixes
    - Similar issues from codebase

    **Examples:**

      # Read from stdin (paste stack trace, then Ctrl+D)
      ast-rag analyze-stacktrace

      # Read from file
      ast-rag analyze-stacktrace error.log

      # Output as JSON
      ast-rag analyze-stacktrace error.log -o json

      # Skip AST mapping for faster analysis
      ast-rag analyze-stacktrace error.log --no-ast-mapping

      # Pipe from another command
      python test.py 2>&1 | ast-rag analyze-stacktrace
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Read stack trace
    if input_path:
        try:
            stacktrace = Path(input_path).read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            console.print(f"[red]File not found: {input_path}[/red]")
            raise typer.Exit(1)
        except OSError as e:
            console.print(f"[red]Error reading file: {e}[/red]")
            raise typer.Exit(1)
    else:
        # Read from stdin
        console.print("[yellow]Reading stack trace from stdin (paste then Ctrl+D)...[/yellow]")
        import sys

        stacktrace = sys.stdin.read()

    if not stacktrace.strip():
        console.print("[red]Empty stack trace provided[/red]")
        raise typer.Exit(1)

    # Load config and initialize service
    cfg = _load_config(config)

    try:
        driver = _connect(cfg)
        embed = EmbeddingManager(cfg.qdrant, cfg.embedding, neo4j_driver=driver)

        from ast_rag.stack_trace import StackTraceService

        service = StackTraceService(
            driver=driver,
            embedding_manager=embed,
            codebase_root=os.getcwd(),
        )

        # Analyze stack trace
        with console.status("[bold blue]Analyzing stack trace...[/bold blue]"):
            report = service.analyze(stacktrace)

        # Output results
        console.print()

        if output == "json":
            print(report.to_json(indent=2))
        elif output == "text":
            console.print(report.summary or "No summary available")
            console.print()
            if report.root_cause:
                rc = report.root_cause
                console.print(f"[bold]Error Type:[/bold] {rc.error_type}")
                console.print(f"[bold]Category:[/bold] {rc.category or 'unknown'}")
                console.print(f"[bold]Severity:[/bold] {rc.severity}")
                console.print(f"[bold]Confidence:[/bold] {rc.confidence:.0%}")
                if rc.likely_cause:
                    console.print(f"[bold]Likely Cause:[/bold] {rc.likely_cause}")
                if rc.suggested_fix:
                    console.print("[bold]Suggested Fix:[/bold]")
                    console.print(rc.suggested_fix)
        else:  # markdown
            console.print(report.to_markdown())

        # Show stats
        console.print()
        console.print(
            f"[dim]Parsed {report.total_frames} frames, mapped {report.mapped_frames} to AST[/dim]"
        )
        if report.similar_issues:
            console.print(f"[dim]Found {len(report.similar_issues)} similar issues[/dim]")

        driver.close()

    except Exception as e:
        console.print(f"[red]Error analyzing stack trace: {e}[/red]")
        if verbose:
            import traceback

            console.print(traceback.format_exc())
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# cache-stats command
# ---------------------------------------------------------------------------


@app.command("cache-stats")
def cache_stats(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """
    Show parse cache statistics and configuration.

    Useful for debugging memory usage and cache efficiency.

    Examples:

      ast-rag cache-stats
      ast-rag cache-stats --config ast_rag_config.json
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    cfg = _load_config(config)

    console.rule("[bold blue]AST-RAG Cache Configuration[/bold blue]")
    console.print(
        f"  Backend: {'SQLite (persistent)' if cfg.parse_cache.persistence_enabled else 'In-memory (bounded)'}"
    )
    console.print(f"  Max entries: {cfg.parse_cache.max_entries:,}")
    console.print(f"  Max memory: {cfg.parse_cache.max_size_mb} MB")
    if cfg.parse_cache.persistence_enabled:
        console.print(f"  DB path: {cfg.parse_cache.db_path}")

    # Create a parser manager to show live stats (only useful in long-running scenarios)
    pm = ParserManager(
        project_id=cfg.neo4j.project_id,
        config={"parse_cache": cfg.parse_cache.model_dump()},
    )
    stats = pm.tree_cache_stats()

    console.print()
    console.print("[bold]Live Cache Stats:[/bold]")
    console.print(f"  Entries: {stats.get('size', 0)}")
    console.print(f"  Hits:    {stats.get('hits', 0)}")
    console.print(f"  Misses:  {stats.get('misses', 0)}")
    console.print(f"  Hit rate: {stats.get('hit_rate', 0.0):.1%}")
    if "memory_mb" in stats:
        console.print(
            f"  Memory:  {stats['memory_mb']:.2f} MB / {stats.get('max_memory_mb', '?')} MB"
        )
    console.rule("[bold green]Done[/bold green]")


# ---------------------------------------------------------------------------


def _collect_index_stats(driver) -> dict:
    """Gather node/edge/language/file counts from the graph.

    Edges are stored as generic ``:EDGE`` relationships carrying the semantic
    type in a ``kind`` property (see ``repositories.queries.batch_upsert_edges``),
    so edge types are grouped by that property rather than by relationship type.
    """
    stats: dict = {
        "nodes": {"total": 0, "by_kind": {}},
        "edges": {"total": 0, "by_kind": {}},
        "languages": {},
        "files": 0,
    }

    with driver.session() as session:
        # CurrentVersion is the graph's bookkeeping pointer, not a code node.
        for record in session.run(
            "MATCH (n) WHERE n.id IS NOT NULL AND NOT n:CurrentVersion "
            "RETURN labels(n)[0] AS kind, count(*) AS n ORDER BY n DESC"
        ):
            stats["nodes"]["by_kind"][record["kind"] or "unlabelled"] = record["n"]
        stats["nodes"]["total"] = sum(stats["nodes"]["by_kind"].values())

        for record in session.run(
            "MATCH ()-[r]->() RETURN coalesce(r.kind, type(r)) AS kind, "
            "count(*) AS n ORDER BY n DESC"
        ):
            stats["edges"]["by_kind"][record["kind"]] = record["n"]
        stats["edges"]["total"] = sum(stats["edges"]["by_kind"].values())

        for record in session.run(
            "MATCH (n) WHERE n.lang IS NOT NULL "
            "RETURN n.lang AS lang, count(*) AS n ORDER BY n DESC"
        ):
            stats["languages"][record["lang"]] = record["n"]

        record = session.run(
            "MATCH (n) WHERE n.file_path IS NOT NULL RETURN count(DISTINCT n.file_path) AS files"
        ).single()
        stats["files"] = record["files"] if record else 0

    return stats


@app.command("stats")
def stats(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config JSON"),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a table"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """
    Show statistics about the indexed codebase.

    Reports node counts by kind, edge counts by type, the language
    distribution and the number of indexed files.

    Examples:

      ast-rag stats
      ast-rag stats --json
      ast-rag stats --config ast_rag_config.json
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    cfg = _load_config(config)
    driver = _connect(cfg)
    try:
        data = _collect_index_stats(driver)
    finally:
        driver.close()

    if as_json:
        print(json.dumps(data, indent=2))
        return

    if data["nodes"]["total"] == 0:
        console.print("[yellow]No indexed nodes found. Run `ast-rag init <path>` first.[/yellow]")
        return

    console.rule("[bold blue]AST-RAG Index Statistics[/bold blue]")
    console.print(f"  Files indexed: {data['files']:,}")

    console.print()
    console.print(f"[bold]Nodes[/bold] ({data['nodes']['total']:,} total)")
    for kind, count in data["nodes"]["by_kind"].items():
        console.print(f"  {kind:<20} {count:>8,}")

    console.print()
    console.print(f"[bold]Edges[/bold] ({data['edges']['total']:,} total)")
    if data["edges"]["by_kind"]:
        for kind, count in data["edges"]["by_kind"].items():
            console.print(f"  {kind:<20} {count:>8,}")
    else:
        console.print("  [dim]none[/dim]")

    console.print()
    console.print("[bold]Languages[/bold]")
    if data["languages"]:
        total = sum(data["languages"].values()) or 1
        for lang, count in data["languages"].items():
            console.print(f"  {lang:<20} {count:>8,}  ({count / total:.0%})")
    else:
        console.print("  [dim]none[/dim]")


if __name__ == "__main__":
    app()
