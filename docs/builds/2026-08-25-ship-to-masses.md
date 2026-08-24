# Ship raged to masses: close the gap between what it claims and what it does

- **Type:** Build
- **Status:** VERIFIED
- **Oracle:** A person who has never seen this repo runs `pip install ast-rag`, follows only the README, and gets: every advertised entry point starting, every command emitting parseable output, and no README claim that the binary cannot back up.
- **Misfire:** Green suite while the installed product is broken — the 2026-08-22 run caught two such defects with 252 tests passing. Every item below is verified through the installed CLI, not only through pytest.
- **Protocol:** One item at a time. RED test first, watch it fail with the reported symptom, fix, full suite matches baseline, then stash the source fix and watch the test fail again.

## Baseline (main @93f45c7, 2026-08-25)
`255 passed, 1 skipped, 1 xfailed in 12.51s` — clean tree, both clones reconciled.

## Ship blockers, ranked

| # | Blocker | Evidence | Status |
|---|---|---|---|
| S1 | Typed-relationship queries can never return a row (#61) | 6 relationship types matched in Cypher, 3 ever written (`CONTAINS_BLOCK`, `EDGE`, `RELATES`) | **MERGED** — PR #70 squashed as `83d8add`, closed #61 |
| S2 | ~~17 of 20 commands emit no machine-readable output~~ **wrong as stated** — `JSONFormatter` is already the default for `query`/`goto`/`callers` via `--humanize`. The real defect: JSON mode emits non-JSON on every failure path | Empty results and the #64 ambiguity note print Rich text to stdout ahead of / instead of the document | **FIXED** — PR #76 |
| S3 | Two shipped entry points crash on a plain install | `[project.scripts]` declares `ast-rag-mcp` and `ast-rag-watch` unconditionally; `mcp` and `watchdog` ship only in the `[mcp]` extra, so both died with a bare `ModuleNotFoundError` traceback that never names the extra | **FIXED** — `f1e5887`, branch `fix/console-scripts-without-extras` |
| S4 | Positioning claims in #59 that main cannot back | Temporal is **FALSE**, not partial — `MERGE` on a stable id under `REQUIRE n.id IS UNIQUE` allows one row per symbol, and `valid_from` appears in zero `WHERE` clauses. `index-lib` absent; `index-folder` never builds embeddings | INVESTIGATED — needs an owner decision, see below |
| S5 | README drift | **Go claim was stale** — already fixed upstream, #66 closed; carried from the 2026-08-22 record instead of re-read. Nine CLI commands undocumented, `python-api.md` documented a call that raises `TypeError`, config example wrong twice | **FIXED** — PR #78 |
| S6 | Relevance scores unnormalised | The 0.7 **is `vector_weight`**, not a similarity. `_normalize_scores` min-max rescales so the top hit is always 1.0, then `0.7*1.0 + 0.3*0` = 0.7 whenever nothing matches by keyword. Raw cosines of 0.99, 0.05 and −0.50 all print 0.7 | ROOT-CAUSED — fix not written |

## Evidence ledger
| ID | What | Result |
|---|---|---|
| B1 | Baseline suite, main @93f45c7 | 255 passed / 1 skipped / 1 xfailed |
| B2 | PR #70 rebased onto main | clean rebase; suite 257 passed / 1 skipped / 1 xfailed |
| B3 | PR #70 stash-the-fix check | `test_relationship_types_emitted` FAILS with source reverted, naming all 6 unmatchable types and their sites |
| B4 | JSON-output audit of `cli.py` | 3/20 commands have any format flag, across 3 conventions (cli.py:625, :718, :2168) |
| B5 | Plain-install entry-point probe | `ast-rag-mcp` → ModuleNotFoundError `mcp`; `ast-rag-watch` → ModuleNotFoundError `watchdog`; `ast-rag` imports OK |
| B6 | `index-lib` grep across repo | zero implementation hits; only prior-run notes recording the claim as false |
| B7 | S3 RED | `test_console_scripts_without_extras` — 4 failed / 1 passed; failures show the raw `ModuleNotFoundError` traceback |
| B8 | S3 GREEN | 5 passed; full suite 260 passed / 1 skipped / 1 xfailed = baseline 255 + 5 new |
| B9 | S3 stash-the-fix | source removed, test kept → 4 fail again |
| B10 | S3 oracle: real clean `uv pip install -e .`, no extras | before: `ast-rag-mcp` → traceback `No module named 'mcp'` (exit 1); after: `The AST-RAG MCP server needs the optional 'mcp' package… pip install 'ast-rag[mcp]'` (exit 1). Same for `ast-rag-watch`. `ast-rag --help` unaffected |
| B11 | S3 lint | `ruff check ast_rag/ tests/…` clean; `ruff format --check` clean on all four touched files |
| B12 | S2 RED / GREEN / stash | 6 failed of 10 → 10 passed → 6 fail again with source reverted; suite 257→267 |
| B13 | S2 stderr routing in a real process | `err_console` writes to fd 2, `console` to fd 1, verified outside `CliRunner` |
| B14 | Diff bug reproduced independently | throwaway repo, `alpha` on line 1 at A, 2 at B, 7 uncommitted; `A..B` reported **7-8**, stamped `valid_from=<B>` |
| B15 | Diff fix | 3 RED → 3 GREEN; suite 257→260; both `compute_diff_for_commits` and `update_from_git` corrected |
| B16 | S5 claims re-verified before landing | `model_name` default `BAAI/bge-m3` (`dto/config.py:63`); `dimension` guard raises at `embedding_manager.py:266-270`; `get_diff` signature at `ast_rag_api.py:1085`; #66 already CLOSED |

## Outcome

| PR | What | State |
|---|---|---|
| #70 | typed-relationship queries (#61) | merged `83d8add` |
| #75 | optional entry points name their extra | open |
| #76 | stdout stays parseable in JSON mode | open |
| #77 | `A..B` reads the target commit, not the working tree | open |
| #78 | docs corrected against the source | open |

## Open, needs the owner

- **#59 claim 3 (temporal) should be retracted or rescoped.** The storage model
  forecloses history rather than being neutral about it. A Tier-2 append-only
  change log (`(:Change)` / `(:Commit)` written in `apply_diff`) is ~2-4 days and
  is the honest version of the claim.
- **#59 claim 4 (`index-lib`)** is ~250-350 LOC over `index-folder` plus a `lib`
  tag on nodes and the Qdrant payload — but `index-folder` builds no embeddings
  today, so the semantic query the claim leads with returns nothing.
- **Score fusion.** RRF, or normalise the vector leg against cosine's own fixed
  range instead of the batch, plus a real `score_threshold`. A test pinning the
  current behaviour with `xfail(strict=True)` is written and uncommitted at
  `tests/test_search_score_semantics.py`.
- **`update_from_git` filters through `file_changed_since_last_index`**, which
  hashes the file on disk — during a historical replay it can skip files whose
  current contents match the cache. Pre-existing, left alone in #77.
