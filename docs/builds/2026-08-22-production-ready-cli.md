# Make AST-RAG a working production CLI: audit, prioritise, fix, verify end-to-end

- **Type:** Build
- **Status:** VERIFIED
- **Oracle:** From the canonical copy (`~/Desktop/PROJECTS/raged`) + fresh venv, following only the repo's own README/docs, with real Neo4j + Qdrant + embedding services running: `ast-rag init <foreign real codebase>` completes, and `goto` / `callers` / `refs` / `query` / `sig` each return results verifiably true of that foreign codebase (ground-truthed by reading it). Adversarial cases included.
- **Misfire:** Suite-green-in-isolation while the actual product path is broken; or verifying only on AST-RAG's own codebase where behaviour is effectively tuned to pass. Both excluded by verifying on a foreign repo via the documented path only.
- **Budget:** 3 rounds (+1 conditional if the final round closes ≥1 criterion)
- **Reference:** none external — the repo's own documented claims are themselves test subjects (C4).

## User context (stated)
- Target: "working production" = CLI end-to-end (chosen)
- Canonical copy: `~/Desktop/PROJECTS/raged` (chosen)
- A "why it is different" text exists somewhere in the repo — find it, audit its claims
- Verify heavily like a proper tester; full-stack with real services (chosen)
- See everything done/cloned; catalogue issues; prioritise

## Acceptance criteria
| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| C1 | Documented audit: both clones reconciled, all branches mapped (merged/unmerged/diverged), docs inventoried, GitHub issues listed | — | — |
| C2 | Issue catalogue: every discovered issue recorded w/ evidence and severity, ranked by production impact | — | — |
| C3 | "Why different" text located in-repo; each positioning claim tested against observed behaviour with PASS/FAIL + evidence | — | — |
| C4 | Baseline quality recorded on canonical copy: full pytest suite + ruff + build, pre-existing failures catalogued before any fix | — | — |
| C5 `[oracle]` | E2E production path on a FOREIGN real codebase over real services: init + goto + callers + refs + query + sig all return ground-truthed-correct results, incl. ≥3 adversarial probes | — | — |
| C6 | Top-priority issues from C2 either fixed + re-verified in this run, or explicitly deferred with reason in the report | — | — |

## Tasks
- [x] T1 — Audit: clone divergence map, branch inventory, docs sweep incl. locating the "why different" text, open GitHub issues
- [x] T2 — Bring-up: fresh venv, deps install, docker Neo4j + Qdrant + embedding service up and healthy
- [x] T3 — Baseline: full pytest + ruff + package build on canonical copy; record pre-existing state (C4)
- [x] T4 — E2E production run on a foreign codebase; log every defect surfaced (feeds C2)
- [x] T5 — Fix the prioritised top blockers; re-run affected verifications
- [x] T6 — Final report: audit map, issue ranking, what was fixed vs deferred, verdicts; hand back

## Evidence ledger
| ID | What | Result |
|---|---|---|
| E1 | `git fetch --all --prune` + branch analysis (canonical copy) | origin/main advanced cd75313→14327eb; checked-out branch `perf/lazy-sentence-transformers` remote DELETED ([gone]); `git cherry`: 2 of 5 local commits patch-identical to main, 3 superseded by merged PRs #56/#60; main now +1502/−728 lines vs local branch |
| E2 | new remote branches after fetch | `fix/first-run-experience`: 4 commits NOT in main — "fresh clone runs without editing anything. Fixes #62", index-folder code blocks, cross-file refs in index-folder, schema DDL repair; `fix/typed-relationship-queries`: 1 commit (fixes #61) |
| E3 | GitHub issues inventory (api.github.com, 20 fetched) | OPEN production-relevant: #62 (committed config hangs fresh clone ~6min), #61 (typed-relationship queries match never-emitted edges), #66 (README languages table missing Go), #30 (MCP ignores config); drafts: #59 uniqueness, #58 quality layer, #57 SQLite storage; closed recently: #63/#64/#65 bugs fixed on main |
| E4 | GitHub issue #59 body fetched | **"why different" text located** (issue tracker, not files): 4 uniqueness claims — (1) semantic search enriched by graph, (2) stacktrace→symbols mapping "nobody has this", (3) temporal graph via valid_from/valid_to "foundation already ours", (4) library indexing `index-lib` — each now testable under C3 |
| E5 | oss-contributions clone pulled | fast-forwarded to 14327eb, same tip as canonical main; both copies now reconciled |
| E6 | services bring-up | raged-neo4j + raged-qdrant containers up 2wks; qdrant /healthz OK; bolt 7687 reachable; effective config = untracked `local_config.json` (bge-m3, dim 1024, no remote_url) |
| E7 | fresh venv `.venv-prod` (py3.12) + `uv pip install -e .` | clean install; `ast-rag --help` shows all README-documented commands |
| E8 | baseline pytest full suite (main @14327eb) | **246 passed / 3 failed / 1 xfailed in 11.9s** — all 3 failures `ModuleNotFoundError: No module named 'mcp'` (tests import MCP server; `mcp` absent from deps/extras) |
| E9 | baseline ruff check + format --check | 4 errors (auto-fixes hidden behind --unsafe-fixes); 7 files would be reformatted |
| E10 | `pip wheel` in venv | failed — uv venv ships no pip; use `uv build` (deferred) |
| E11 | root-cause of E8's 3 failures | NOT product bugs — `mcp` lives in `[dev]` extra (pyproject:50, CONTRIBUTING.md:133 documents `pip install -e ".[dev]"`); my venv had omitted it. **Correction to E8/E9 record** |
| E12 | baseline re-run with documented dev install | **252 passed / 0 failed / 1 xfailed in 15.0s** — suite fully green on main+merge |
| E13 | merge of `origin/fix/first-run-experience` into local main | 1 conflict in cli.py (main's #64 ambiguity helpers vs branch's `_verify_neo4j`) — resolved keeping BOTH; suite after: 249→252 green incl. branch's new schema tests; commit on local main |
| E14 | C3 claim probes (no services needed) | temporal graph: `valid_from/valid_to` present across node/block DTOs + queries + repository ✓; `analyze-stacktrace` command exists + docs + tests ✓; **`index-lib` claim FALSE on main** — no such command in CLI (issue #24 unimplemented) |
| E15 | E2E prep | foreign repo chosen: `oss-contributions/outlines` (Python, src-layout); ground truth captured: `SteerableGenerator` @generator.py:197, multi-line `__init__` @:216 (sig-DOTALL target), `get_cache` @caching.py:43, BlackBoxGenerator 8 grep hits, SteerableGenerator refs in applications.py; compose stack `ast_rag_neo4j`(healthy)/`ast_rag_qdrant` up on documented defaults (password "password"); old `raged-*` containers STOPPED for port freedom (restart at hand-back); bge-m3 downloading (~1.2G/2.3G at check) |
| E16 | bge-m3 loaded locally ("LOADED OK, dim: 1024") | embedding layer operational on documented default path |
| E17 | **ORACLE: `ast-rag init <outlines>` — zero config files** | SUCCESS: 1979 node embeddings, 898 blocks + CONTAINS_BLOCK edges, 7m02s wall; #62 fresh-clone fix confirmed working end-to-end |
| E18 | goto SteerableGenerator | EXACT ground truth: generator.py:197, kind Class ✓ |
| E19 | refs BlackBoxGenerator | "8 total" == grep truth (8 lines) ✓ |
| E20 | callers get_cache | ambiguity warning fired live (#64 behaviour): "matched 2 symbols… Reporting on caching.get_cache"; 4/4 callers verified true vs source (decorator@108≈call@109, clear_cache@180→182, test_cache:41/:55) ✓ |
| E21 | query "cache with expiration" | top hit = caching.cache @line 86 score 1.0 (true function); clear_cache second ✓ |
| E22 | analyze-stacktrace via stdin (real-shaped traceback) | 2/2 frames mapped to AST; call chain __call__@generator.py:280 → decorator@caching.py:109 matches true call site ✓ (#59 claim 2 VERIFIED) |
| E23 | sig adversarial investigation | wrapped signatures stored with newlines (145 fns) but unmatchable → root causes: (a) params regex demanded literal adjacency after `(`, (b) bare `*` name double-escaped into mandatory literal `\.` — documented example `*(int, String)` could never match anything. FIXED commit f46c262 + 4 regression tests (RED first: 3 failed → GREEN) |
| E24 | sig post-fix live re-verify | `*(int, String)` finds Hello.process(int count, String name) ✓; `__init__(self, model*)` ranks generator.py:216 FIRST ✓; suite 256 passed / 0 failed incl. 4 new tests |
| E25 | remaining adversarials | goto non-existent → clean error, no traceback ✓; gibberish query returns top-score 0.7 (score semantics unclear — logged as issue, not blocking) |
| E26 | C3 claim scoreboard | (1) semantic-over-graph: behavioural PASS on query; enrichment mechanism present in code, not isolated — PARTIAL; (2) stacktrace mapping: PASS live (E22); (3) temporal foundation: valid_from='INIT'/valid_to written, CurrentVersion node present — foundation TRUE, diff-feature itself absent; (4) index-lib: **FALSE** — no CLI command exists (issue #24 unimplemented) |
| E27 | env restored + build | compose stack stopped, original raged-* containers restarted ✓; `uv build` wheel produced cleanly (C4 complete) |

## Round log

### Round 1 (2026-08-22)
- Steering folded in at round top: "pull first" → both clones synced; audit re-based on origin/main @14327eb; canonical switched off deleted-remote branch onto main.
- T1 done: divergence map (E1), issue inventory (E3), "why different" located (E4), unmerged fix branches identified (E2).
- T2 done: services healthy (E6), fresh venv + editable install clean (E7).
- T3 done: baseline recorded BEFORE any fixes (E8–E10); corrected same round when failures traced to my env, not the repo (E11–E12).
- Key finding: `fix/first-run-experience` holds the exact fresh-clone production fixes our oracle path needs.
- Judged: C1 PASS, C4 PASS (with correction noted). C2/C3/C5/C6 pending.

### Round 2 (2026-08-22)
- Merged `fix/first-run-experience` into local main after conflict resolution keeping BOTH main's #64 helpers and branch's fail-fast Neo4j check (E13).
- T4 executed to completion: oracle battery init/goto/callers/refs/query/stacktrace all verified against ground-truthed foreign codebase (E16–E22); adversarials run (E25).
- T4 defects surfaced and root-caused: sig broken for wrapped + named params (E23), index-lib claim false (E14/E26), UX inconsistencies catalogued (mixed console+JSON, path line-wrap, qdrant version pin warning, gibberish-query scores).
- Judged: C2 PASS (catalogue below), C3 PASS (all four claims tested, E26), C5 PASS pending sig-fix re-verify, C6 partial (fix designed, RED tests written).

### Round 3 (2026-08-22) — final
- T5 done: sig fix implemented test-first (RED 3/4 → GREEN), committed f46c262; suite 256 green; live re-verification on real indexes positive including exact ground-truth ranking (E23–E24).
- Environment restored to pre-run state (E27). Build completed (E27).
- Judged: C5 PASS `[oracle]` (E17–E25), C6 PASS (fixed: #62-class fresh-clone path via merge, sig wrapped/named params via f46c262; deferred items listed with reasons below). **All criteria pass.**

## Issue catalogue (ranked by production impact)

| Rank | Issue | Status this run |
|---|---|---|
| 1 | Fresh clone unusable: committed config pointed at private LAN, ~6min hang (#62); index-folder produced no blocks/cross-file refs; schema DDL silently failed | FIXED — merged fix/first-run-experience (E13, E17) |
| 2 | `sig` blind to wrapped multi-line + named-param signatures (incl. its own documented example) | FIXED — commit f46c262 + 4 regression tests (E23–E24) |
| 3 | Tests import `mcp` but it is only in `[dev]` extra; plain `pip install -e .` + pytest = confusing failures | Deferred w/ reason: behaviour correct per CONTRIBUTING.md; suggest moving mcp to main deps or skip-if-missing guards |
| 4 | #61 typed-edge queries match never-emitted relationship types | Deferred — branch `fix/typed-relationship-queries` exists upstream; needs own verification cycle |
| 5 | README claims drift: languages table missing Go (#66); quality metrics table cites Phase-2 numbers unverifiable without benchmark rerun; `index-lib` referenced in #59 but doesn't exist | Partially deferred — metrics re-run recommended next; index-lib = unimplemented feature |
| 6 | Output-format inconsistency: some commands print console tables, others raw JSON; rich wraps paths mid-token breaking copy-paste for agents | Deferred — UX refactor across commands |
| 7 | Qdrant client/server minor-version warning (client 1.19 vs server latest 1.17) | Deferred — pin compatible pair in compose |
| 8 | Relevance scores unclear: gibberish query returns 0.7-confidence hits | Deferred — define score normalisation/cutoff |

## Verification pass (step 6)
| # | Layer | Result |
|---|---|---|
| 1 | Test suite whole project | evidenced — 256 passed / 0 failed / 1 xfailed post-fix (E24); baseline 252 green pre-fix (E12) |
| 2 | Types / static analysis | not run — repo has no mypy gate configured in CI path exercised; ruff IS the project's linter |
| 3 | Lint / format | evidenced partially — ruff check ran at baseline (4 errors, pre-existing, unfixed by choice: out of scope, would pollute diff); format --check: 7 files pre-existing |
| 4 | Build / bundle | evidenced — wheel builds clean (E27) |
| 5 | Artefact actually runs | evidenced — full CLI battery executed against live services (E16–E22, E24) |
| 6 | End-to-end real path | evidenced — the oracle itself IS the E2E: fresh-config init on foreign repo over real Neo4j/Qdrant/embeddings (E17) |
| 7 | Independent read of full diff | done — merge conflict resolution reviewed line-by-line before commit; sig fix diff read as reviewer; no debug prints/leftovers committed |
| 8 | Docs still true | improved + flagged — README's sig examples now actually work (were false); remaining drift catalogued rank-5 |
| 9 | Regression adjacent | evidenced — full suite green post-merge and post-fix (E13, E24); unrelated commands re-verified (goto/refs/query unchanged behaviour) |

**Misfire check:** the named failure was suite-green-in-isolation or self-tuned verification. Avoided: verification ran against a FOREIGN codebase (outlines) through the documented user path only, with every result compared to independently captured ground truth — which is exactly how the two real defects (sig shapes) were caught despite a fully green suite.

**Status: VERIFIED** — all six criteria pass; verification layers evidenced or explicitly not-run with reason.
