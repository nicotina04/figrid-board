# Changes

## 0.4.0 (2026-04-21)
* **NNUE engine integrated** (powered by the [noru](https://crates.io/crates/noru) core). New binary `pbrain-figrid-noru` targets Gomocup 2026 Freestyle 15×15. Binary name is tentative and may switch to `pbrain-figrid` once the original author's 0.3.1 submission withdrawal is confirmed.
* **Breaking:** the pre-0.4 symbolic evaluator, record keeper, rule checker, and `Eval<SZ>` / `Rec<SZ>` / `Tree` stack are now under `figrid_board::legacy::*`. Previously top-level imports like `use figrid_board::{Eval, FreeEvaluator15, Rec, Rows, ...}` must become `use figrid_board::legacy::{...}`.
* New top-level API: `Board`, `Stone`, `Move`, `BitBoard`, `to_idx`, `to_rc`, `BOARD_SIZE`, `NUM_CELLS`, `GameResult`, `Searcher`, `SearchResult`, `evaluate`, `IncrementalEval`, `search_vct`, `VctConfig`, `GOMOKU_NNUE_CONFIG`, `scan_line`, `LineInfo`, `DIR`. `Coord`, `Coord15`, `Coord20`, `CoordState`, `Rotation`, and `Error` keep their existing paths.
* Tactical search: root VCT (AND/OR threat tree with Zobrist TT), iterative-deepening α-β with killer/history heuristics, threat-aware move ordering, Piskvork-compatible `timeout_turn` budgeting. Addresses the engine-strength roadmap items in `docs/INHERITED_TODO.md`.
* NNUE weights (v6-A, 4096-feature PS + LP-Rich + Compound + Density → 512 accumulator → 64 hidden → 1 output, ~4.1 MB) are embedded into the `pbrain-figrid-noru` binary via `include_bytes!`. No external files or dependencies required at runtime.
* `Rows` is no longer at the crate root — it is now `legacy::Rows` and the crate-root `Board` type replaces its role (addresses `INHERITED_TODO.md` §6).
* Crate layout normalized under `src/`, with the legacy engine under `src/legacy/`. Remaining work on items §1 (depth assertion), §2 (RAM/cache), §3 (Tree safety), and §4/§5 (dynamic size/rule) is scoped to the legacy module and deferred.
* Only `rule = 0` (freestyle) is supported by `pbrain-figrid-noru`. Any other rule is rejected at `START` with `ERROR - unsupported rule`. The original `pbrain-figrid` binary continues to support all four rules via the legacy engine.

## 0.3.2 (2026-04-20)
* Maintainership transferred from wuwbobo2021 to nicotina04. Repository moved to <https://github.com/nicotina04/figrid-board>.
* No functional changes to the engine yet. Future 0.4+ releases will introduce an NNUE-based evaluator (via the [noru](https://crates.io/crates/noru) core) and stronger tactical search (VCF/VCT, iterative deepening, transposition table) for Gomocup 2026.
* Known technical debt inherited from 0.3.x is tracked in `docs/INHERITED_TODO.md`.

## 0.3.1 (2025-06-17)
* Fixes a bug that prevents the evaluator from making connection of 5 or blocking the opponent's 4/live3 under some situations.
* Tries to fix the timeout issue which affects the first 6 moves.
* Fixes the Caro rule.
* Fixes the `BOARD` command; allows rule setting after `START` command.
* Makes a few optimizations.
* Adds documentation.

## 0.3.0 (2025-05-31)
* Initial Rust version. The author has made this bold attempt in a hurry after catching a cold. Suffered in critical bugs, got the lowest ranking in Gomocup 2025.
