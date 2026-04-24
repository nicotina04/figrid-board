# Changes

## 0.5.2 (2026-04-25)
* **Revert the VCT budget expansion from 0.5.1**. Live Pela matches with
  0.5.1 were *faster* losses than 0.5.0, not slower — games ended in 28-50
  plies (average 40) vs 48-66 plies (average 55) in 0.5.0. `analyze_psq` on
  the 0.5.1 losses showed a paradox: the d4 vs d8 disagreement rate dropped
  (30-54% → 14-23%), but static_eval stayed positive (Black-favor +150 to
  +200) right up to the move before Pela's mate completed. Classic
  "overconfident loss" pattern — d4 and d8 agreed on aggressive attacking
  moves while both missed Pela's 10+ ply mate preparation. On top of that
  the VCT's enlarged 1.25 s / 20-ply budget (from `ROOT_VCT_BUDGET_FRACTION
  = 4`, `ROOT_VCT_DEPTH = 20`) ate into α-β's time for defensive reading.
  Rolled back to `FRACTION = 8` (5 s turn → 625 ms VCT), `DEPTH = 14`, and
  `BUDGET_MS = 150` (no-time-limit default).
* **Keep the stone-driven feature extraction from 0.5.1**. That change was
  orthogonal — iter_ones only touches how the feature set is enumerated,
  not what gets emitted, and test outputs are byte-identical to 0.5.0. The
  speedup frees a few cycles for α-β, so keeping it is a pure gain.
* Weights unchanged — v13_broken_rapfi.bin still shipped. A v14 candidate
  trained on 12 999-game Rapfi data (9 999 + 3 000 with wider openings) was
  measured at 46.7% in arena vs v13's 53.3% — within the 30-game noise band
  (σ ≈ 9 pp) but directionally worse, so not adopted.

## 0.5.1 (2026-04-24)
* **VCT budget expanded**. Three Pela losses analysed via `analyze_psq` showed
  mate sequences were detected too late at α-β depth 4 (d4 vs d8 mismatch
  rate 30-54%), while VCT's 625 ms / 14-ply budget was not catching 10+ ply
  chains that Pela routinely builds. Adjusted to `ROOT_VCT_DEPTH = 20`
  (10-ply mutual search) and `ROOT_VCT_BUDGET_FRACTION = 4` (5 s turn →
  1.25 s VCT). `α-β` still gets the remaining 3.75 s.
* **Stone-driven feature extraction**. `compute_active_features` previously
  scanned all 225 cells × 6 feature sections with an `if empty` branch on
  every cell. Replaced with `BitBoard::iter_ones()` (u128×2 bit-walking
  iterator) so feature loops only visit actual stones. Order-preserving
  (lowest index first), so feature push order and downstream weights are
  identical. Net effect: fewer cycles per leaf evaluation, enabling the
  existing α-β to reach slightly deeper within the same turn budget.
* No weight changes — v13_broken_rapfi.bin still shipped.

## 0.5.0 (2026-04-24)
* **NNUE weights updated → v13_broken_rapfi**. Training backstory in short:
  - The 0.4.x "60% arena" baseline was found to be inflated by two bugs: (a) the
    `compound_combo_id` extractor was filtering threats to `is_line_start`
    stones only, so every cross-shaped double-three/four-three hotspot in a
    line's interior was silently invisible, and (b) the internal arena
    generated openings as "center + ±2 random", which is a narrow manifold
    that overestimates the NNUE's real-game generalization. After fixing both
    the clean `v10` baseline measured at **33.3%** on the new
    Gomocup-balanced opening arena — the "60%" figure from 0.4.x was not
    directly comparable.
  - On top of the fixed baseline we added Rapfi self-play distillation (9 999
    games, 222 k labeled positions) with a PSQ anchor to prevent catastrophic
    forgetting: `v11 = v10 + Rapfi` → **50.0%** (+17 pp).
  - Separately we added a **Broken / Jump** feature section covering gap-1
    patterns (`_●●_●_`, `_●●●_●_`, `_●_●_●_`) that the previous `scan_line`
    encoder could not see: `v12 = v10 + broken/jump` → **43.3%** (+10 pp).
  - Combined: `v13 = v12 + Rapfi distillation` → **53.3%** (+20 pp from v10).
* **Feature layout expanded** (same 4 096 slot budget): a new `[3416..3848)`
  section encodes three broken/jump shapes × open(2) × dir(4) × zone(9) ×
  perspective(2) = 432 slots. Reserved shrinks from 680 to 248. No change to
  A/B/C/D/E sections.
* **Engine-side bugfix** (inherited from noru-tactic's 2026-04-23 fix):
  `compute_compound_threats` no longer gates on `is_line_start`, so all four
  directions are scanned from every stone; single-direction threats are now
  excluded from compound (covered by LP-Rich), preventing double-counting.
* **noru bumped 1.2 → 2.0**. `NnueConfig.hidden_sizes` switched to
  `Cow<'static, [usize]>`, eliminating the `Box::leak` memory-leak pattern in
  FFI paths while keeping the const-friendly borrowed form at zero cost.
* **PSQ training filter** narrowed to Freestyle only. Standard (exact-5)
  games have a different winning-row semantics than our engine's `check_win`
  (≥ 5) and were label-polluting the training set. Standard support, if
  pursued, will be a separate branch.
* Real-game validation against Pela is the next step (see
  `docs/INHERITED_TODO.md`). The v13 weights are now the default model shipped
  in the `pbrain-figrid` binary.

## 0.4.4 (2026-04-22)
* **Principal Variation Search (PVS)** in `alpha_beta`: the first move from the ordered list is still searched with a full `[-β, -α]` window, but every subsequent move is first probed with a zero-width `[-α-1, -α]` null window. If the null-window result fails high (lies inside `(α, β)`), it is re-searched with the full window. When move ordering is accurate, this roughly halves the effective branching factor at interior nodes and lets the same per-turn time budget reach one to two plies deeper on average. No change when ordering is wrong (PVS degenerates to plain α-β plus one extra re-search).
* **PV-move priority at the root**: the best move from iteration `depth-1` is pulled to the front of the move list at iteration `depth`. Combined with the root PVS this makes the null-window pass on the remaining root moves almost always fail low, which is where PVS's speedup comes from in practice.
* **Motivation — `analyze_psq` PVS study (2026-04-22)**: running the new `analyze_psq` binary over ten Pela-loss games showed that in about 30% of positions the engine's depth-4 best move differed from the depth-8 teacher's pick — i.e. the engine was being out-searched rather than out-evaluated. These two changes target that gap directly.
* No weight changes; no API changes. Arena `psq-arena` at fixed depth 4 stays within noise of the 60% baseline (63.3% on 30 games, +3.3 pp). Real gain shows up on time-based play where the extra plies reached by PVS actually matter.

## 0.4.3 (2026-04-22)
* **Binary rename**: the NNUE engine is now shipped as `pbrain-figrid` (previously `pbrain-figrid-noru`). The original author's 0.3.1 Gomocup 2026 submission was withdrawn by the Gomocup administrators on 2026-04-21, so the canonical binary name is free again. The previous symbolic engine remains available as `pbrain-figrid-legacy` — this preservation is a maintainer choice, not an API-stability promise, so downstream users who want the 0.3.x engine can still build it. Based on follow-up feedback from the original author, the `legacy::` module is intentionally kept under that name (an earlier draft had renamed it to `fpga::`, which was factually wrong — the serial calculators were the CPU-oriented variant of the design).
* **ABOUT response updated**: the `ABOUT` pbrain command now reports `name="figrid", version="0.4.3", author="nicotina04 (successor to wuwbobo2021)", country="KR"`, so matchmaking and tournament records cleanly identify this engine as the successor to the 0.3.x entry.
* **README build instructions split** into two clearly separated sections — "local / development" (may use `-C target-cpu=native` for host performance) and "Gomocup submission" (must target the tournament machine: `-C target-feature=+crt-static -C target-cpu=x86-64-v3`, matching the 2026 announcement's SSE4.1 / SSE4.2 / POPCNT / AVX / AVX2 guarantee). Also notes that an additional `x86-64-v4` (AVX-512) build may be submitted as a second zip entry.
* **`INHERITED_TODO.md` §1 resolved** — added a `debug_assert_eq!` on `cur_depth()` immediately after `$operation` in `traversal_in_depth!` (see `src/legacy/evaluator.rs`), matching the invariant the original author flagged in his handover notes. Release builds are unaffected (`debug_assert`). Existing test suite still passes, so the invariant was already respected by every `$operation` site — the assertion simply codifies it.
* No engine logic changes.

## 0.4.2 (2026-04-21)
* **Composite threat priority in move ordering**: `alpha_beta`'s `move_score` now adds per-move scores derived from the multi-direction `classify_move` (double-three, four-three, double-four, etc.) on both attack and defense sides. The previous `scan_line`-only evaluation could miss composite threats like 3-3 that require seeing all four directions at once, which in practice caused losses against mid-tier Gomocup engines (e.g. Pela) that make such threats. Attack-side weight is slightly higher than defense (10% bias) so the engine still prefers its own winning moves over blocking. Arena (baseline 60% vs heuristic) unchanged — heuristic opponent doesn't exercise these patterns; gains are expected only against engines that actually exploit composite threats.
* No API changes.

## 0.4.1 (2026-04-21)
* **Dynamic root VCT budget**: the root tactical search now scales its time budget with the turn time limit (1/8 of the per-turn budget, clamped to 100 ms–2 s). Under Piskvork's 5 s fastgame timing this gives the VCT ~625 ms instead of the previous fixed 150 ms, allowing deeper mate-sequence discovery; under the 30 s timing it scales to the 2 s cap. Arena (fixed depth, no time limit) behavior is unchanged — baseline 60% vs the heuristic is preserved.
* **Larger VCT transposition table**: root VCT's Zobrist TT initial capacity increased from 4 096 to 65 536 entries, reducing re-expansion of duplicate nodes on deeper searches. Memory impact negligible (< 2 MB per call).
* **Tighter time-check cadence in α-β**: deadline check frequency in `alpha_beta` increased from every 1 024 nodes to every 128. Combined with the 150 ms adapter safety margin already in `pbrain-figrid-noru`, overshoots of Piskvork's `timeout_turn` are now consistently kept under the limit (observed ~4.9 s finish under a 5 s budget, previously ~5.1 s).
* No API changes; `Searcher::search`'s signature is unchanged and the dynamic budget is derived from the existing `time_limit` parameter.

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
