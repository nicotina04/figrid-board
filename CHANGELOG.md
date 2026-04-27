# Changes

## 0.6.5 (2026-04-27)
* **Center-distance bonus removed from quiet-move ordering.** A 4-game
  Pela analysis on 0.6.4 surfaced a clustering tendency: when given a
  non-standard (corner-seeded) opening, our pieces drifted back toward
  the centre and bunched up — a structurally losing pattern in Gomoku.
  Tracing it inside `move_score_and_forcing`, the culprit was the
  trailing `score += 14 - center_dist` line. Forcing moves are tier-
  separated by ≥10⁵ so the bonus never affected them, but for tier-0
  quiet moves the bonus dominated the killer/history/scan-line signals
  the rest of the function relies on, dragging ordering toward the
  centre on every empty board area. Deleting the line lets the quiet
  ordering fall back to genuine signals.
* Internal NNUE arena win rate at 5 s/move stays inside noise (80.0 % →
  **83.3 %**, +3.3 percentage points, σ ≈ 7 % over 30 games), but the
  more interesting effect is on game length: average game time drops
  60.62 s → 41.50 s (-32 %), TT hit rate rises 39.7 % → 42.5 %, mean
  completed depth holds at 6.06. The engine reaches decisive lines
  faster because the ordering no longer wastes its first-move slot on
  a cosmetic centre move.
* This change is targeted at the corner-opening (swap2-style) regime
  observed in live Pela matches, where the heuristic-arena ceiling does
  not measure the regression. Pela re-runs are needed to confirm the
  clustering itself is gone.
* No protocol or API regression. LMP, IIR, TT 256 K + push-down,
  Aspiration, qsearch, threat-gated LMR, and the 128-node deadline
  check all carry over from 0.6.4 unchanged.

## 0.6.4 (2026-04-27)
* **Late Move Pruning (LMP).** At non-PV nodes with depth in 1..=3,
  quiet (non-forcing, non-killer) moves whose `move_idx ≥ 8 + 4 × depth`
  are skipped outright. Count-based instead of eval-based, so it sidesteps
  the BCE-narrow eval-distribution problem that made 0.6.3's razoring /
  futility prototypes inert.
* Internal NNUE arena win rate at 5 s time budget stays at **80.0 %**
  (24 W − 6 L over 30 games) — the same as 0.6.3. The fixed-depth-4
  heuristic is the ceiling at this rate; the lift shows up elsewhere:
  * Mean completed depth 5.63 → **6.00**, p50 7 → 8, p95 9 → 10, max
    10 → 11. One extra ply at the high end and at the median.
  * Average game time 60.62 s → 42.37 s (-30 %), the engine reaches
    decisive lines faster.
* The lose-game analysis on 0.6.1 placed Pela's killing mate sequences
  7–9 ply ahead of our typical search horizon. Adding a ply at the high
  end is exactly where this matters; the arena ceiling does not capture it.
* No protocol or API regression. IIR, TT 256 K + push-down, Aspiration,
  qsearch, threat-gated LMR, and the 128-node deadline check all carry
  over from 0.6.3 unchanged.

## 0.6.3 (2026-04-27)
* **Internal Iterative Reduction (IIR).** TT-miss non-PV nodes at
  `depth ≥ 4` now search at `depth - 1`. The standard chess-engine
  trick: a node with no transposition entry has no good first-move
  guess, so its full-depth search wastes branching factor on weak
  ordering. Reducing one ply lets the search finish faster, and the
  resulting TT entry guides ordering at the same node when iterative
  deepening reaches it again the following iteration. One-line change
  inside `alpha_beta`.
* Internal NNUE arena win rate at 5 s time budget: 73.3 % (0.6.2) →
  **80.0 %** (24 W − 6 L over 30 games, σ ≈ 7 %, +6.7 percentage points).
  Mean completed depth 5.53 → 5.63, p95 8 → 9 — same time budget,
  one extra ply at the high end.
* Razoring and futility pruning were prototyped on the 0.6.2 base and
  did **not** make this release. Two configurations (margins 200 / 400
  and 80 / 150 for razor; 50 + 80 d and 20 + 30 d for futility) both
  measured at 73.3 % — identical to the 0.6.2 baseline down to the
  TT-cutoff rate. Our NNUE was trained against sigmoid (BCE) targets,
  so its static-eval distribution stays inside ±100 cp on quiet
  positions; razor/futility margins built for cp-regression engines
  (Stockfish-style) almost never trigger here. The code was reverted
  rather than left as dead-code.
* No protocol or API regression. TT, Aspiration, qsearch, threat-gated
  LMR, and the 128-node deadline check all carry over from 0.6.2
  unchanged.

## 0.6.2 (2026-04-27)
* **Transposition table grown from 64 K to 256 K buckets.** A diagnostic
  pass on 0.6.1 (5 s budget, 5-game arena) recorded 28.5 % of stores
  evicting an existing depth-preferred entry while the always-replace
  slot stayed at 2.1 % usage — a textbook bucket-shortage signature.
  `TT_BUCKET_BITS` rises from 16 to 18 (2 MB → 8 MB), still well inside
  Piskvork's ≥350 MB budget. Same diagnostic on the larger table drops
  displaced rate to 15.3 % and lifts hit rate from 36.6 % to 39.2 %.
* **Push-down replacement on the depth-preferred slot.** When a new
  store evicts a non-empty depth-preferred entry whose key differs from
  the incoming one, the evicted entry is now copied into the
  always-replace slot before being overwritten — instead of being
  silently discarded as in 0.6.1. The always-replace slot now actually
  acts as the second-best entry per bucket, the role its name implies.
  Same-key updates (the common case during iterative deepening) skip the
  push-down so a stale shallow entry doesn't shadow the fresh deep one.
* **TT diagnostic counters**: `probes`, `hits`, `stores`,
  `displaced_depth_pref`, `stored_to_always`, and a 16-bucket store
  depth histogram, exposed via `TranspositionTable::stats()` and
  `Searcher::tt_cutoffs`. Counters live in `Cell<u64>` so the hot
  `probe()` path stays `&self`. Cost is ~1 ns per probe and the
  Piskvork binary does not print them — the data is for the trainer's
  arena harness.
* No protocol or API regression. Engine algorithm is unchanged in 0.6.2;
  the ply budget materializes purely as fewer collisions plus a small
  cutoff-rate lift.

## 0.6.1 (2026-04-26)
* **Search overhaul** — internal NNUE arena win rate at 5 s time budget
  jumps from 40 % to 63.3 % (30-game), powered by four orthogonal changes
  layered on top of 0.6.0's PVS + TT + root VCT engine:
  * **Move ordering tier separation.** Threat priorities (win, must-block,
    open-four, double-four / four-three, double-three, closed-four,
    open-three) now live in absolute non-overlapping buckets ≥10⁵ apart so
    no killer / history / center bonus can ever lift a quiet move past a
    tactical one. The PVS first-move quality this guarantees is the gate
    every other search trick depends on.
  * **Aspiration windows.** From depth 4 onward the root iteration starts
    with `[score-50, score+50]` and re-searches with exponentially growing
    deltas only on fail-low / fail-high. Most iterations land first try.
  * **Quiescence lite.** When `alpha_beta` reaches depth 0 the leaf is
    handed to a forcing-only quiescence search (immediate win, must-block,
    open-four, double-four, four-three, open-four block) capped at 4 ply.
    The NNUE static eval still scores both the stand-pat and the leaf at
    the end of every quiet line — qsearch only changes *where* the eval
    is sampled, not what does the sampling.
  * **Threat-gated late-move reductions.** Non-PV / non-killer / non-
    forcing moves at depth ≥ 3 and `move_idx ≥ 3` are reduced by 1–2 ply
    in the null window; reduced fail-highs trigger a full-depth re-search
    before the PVS full-window verification. Forcing moves (open-three or
    higher, either side) are *never* reduced — that gating is what makes
    LMR safe here, where the naive 0.4.x experiment lost 43 percentage
    points.
* Combined effect on a 5 s/move arena vs the depth-4 heuristic: mean
  completed depth 3.73 → 5.67, p50 4 → 7, max 6 → 10. The new ply budget
  is what materializes as win rate; fixed-depth-4 arena stays at 50 %
  because there is nothing to reduce.
* **Weights revert to v14_broken_rapfi_wide.** Pattern4 mini (G section)
  emits are removed from `eval.rs`; `TOTAL_FEATURE_SIZE` returns to 4 096.
  v17–v20 retraining did not move the win rate against Rapfi-equivalent
  opponents (feature redundancy), and the larger 36 864-feature variant
  cost depth in the 5 s search budget. The `pattern_table` infrastructure
  (canonicalisation, top-K dense table, swap table, board-side
  `line_pattern_ids` maintained incrementally on make/undo) is preserved
  for future move-ordering / TT-augmentation experiments. The bundled
  top-K table shrinks to 4 096 entries (`data/topk.bin`, 16 KB);
  `top16k.bin` is removed.
* `pbrain_figrid_noru` now embeds `models/gomoku_v14_broken_rapfi_wide.bin`
  (4.3 MB, the same baseline used through 0.5.x).
* No protocol or API change. The 128-node deadline check, dynamic VCT
  budget, and 150 ms safety margin all carry over from 0.4.1 unchanged.

## 0.6.0 (2026-04-25)
* **Pattern4 mini integrated.** Eleven-cell line patterns are now part of
  the NNUE feature space. The new `pattern_table` module enumerates the
  ~200k canonical 11-cell windows, dumps the top 16 384 by frequency
  (measured on PSQ + Rapfi data, 99.24% coverage) into a 64 KB
  `data/top16k.bin`, and at runtime serves an O(1) `lookup_mapped_id` from
  a 4 M-entry lazy dense table. Each cell carries 4 line pattern IDs (one
  per direction) maintained incrementally on `make_move`/`undo_move` —
  black-relative storage, with a small swap table for stm-perspective
  lookups. A new G section in the feature layout adds 16 385 IDs ×
  perspective(2) = 32 770 features, growing `TOTAL_FEATURE_SIZE` from
  4 096 to 36 864.
* **Weights swap to v18_small_p4_rapfi.** Trained from scratch on the
  expanded feature space: small network (512 acc / 1×64 hidden) + PSQ 1 M
  base (15 epoch, lr 1e-3) + Rapfi 33 k distillation with PSQ 500 k
  anchor (3 epoch, lr 2e-6). Real-weights consistency harness passes 100
  trials × 160 plies with no saturation divergence.
* Internal arena puts v18 at 53.3 %, identical to v13. The Pattern4
  mini's contribution does not show up against the heuristic opponent —
  loss did drop further (0.564 → 0.472), but the win rate stays flat,
  consistent with feature redundancy against existing PS / LP-Rich /
  Compound / Cross-line / Broken sections. The arena alone cannot tell
  whether the new line patterns help in real play; this release ships
  the weights specifically so live Pela matches can be inspected.
* No engine algorithm change. TT, iterative deepening, VCT budget, and
  the Searcher plumbing all carry over from 0.5.5 unchanged. The 128-node
  deadline check remains figrid-specific.

## 0.5.5 (2026-04-25)
* **Transposition table on the α-β path.** A 2 MB table (16-bit bucket index,
  two slots per bucket: depth-preferred + always-replace) caches each node's
  result keyed by Zobrist hash. Iterative deepening now reuses the previous
  iteration's PV and cutoffs without re-searching the same positions, and
  the TT-best move is pulled to the front of the move list so the PVS
  null-window search hits the right move first. Empty bound, exact / lower
  / upper kinds are all handled; only entries whose stored depth meets or
  exceeds the current need can short-circuit.
* **Zobrist hashing on `Board`.** A new `zobrist: u64` field is updated
  incrementally in `make_move` / `undo_move` (XOR of the placed stone's
  `(color, square)` key plus the side-to-move toggle). Stone keys are
  generated at compile time from a deterministic splitmix64 seed schedule,
  so no runtime initialization or randomness is involved. Two regression
  tests cover make/undo round-trip back to zero and same-position-same-key
  invariance under move-order permutations.
* **Reverted the 1024-acc / 2×128 hidden network experiment.** It was
  shipped briefly as a 0.5.x prototype on noru-tactic v15/v16 weights, but
  the 30-game arena landed at 30 % (vs. v13's 53 %): a parameter-to-sample
  ratio of roughly 4-to-1 that was clearly over-fitting on our distillation
  set. `GOMOKU_NNUE_CONFIG` is back to 512-acc / 1×64 hidden, the same
  shape `v13_broken_rapfi.bin` was trained against, so existing v13/v14
  weights load cleanly again.
* The shipped binary still embeds `v14_broken_rapfi_wide.bin` from 0.5.4
  unchanged. The expected gain in this release is search-side: live
  Piskvork play should reach one or two extra plies inside the same time
  budget, and the TT/PV interaction should reduce the kind of "self
  re-discovery" patterns the 0.5.4 analyses showed in the middlegame.

## 0.5.4 (2026-04-25)
* **Weights swap: v13 → v14_broken_rapfi_wide**. v14 was trained on a
  12 999-game Rapfi corpus (9 999 + 3 000 with wider random openings,
  `--opening-moves 6`) versus v13's 9 999. Internal arena measured v14 at
  46.7% vs v13's 53.3%, but the arena uses a narrow balanced-opening set
  and that ~7 pp gap is inside the 30-game noise band (σ ≈ 9 pp). Live
  Pela matches against 0.5.3 (v13) kept losing with a "slow-grind"
  trajectory rather than sudden blunders, which points at eval-space
  blindness to Pela's more off-centre play. The wider-opening v14 saw a
  lot more of that distribution during distillation, so it's the better
  candidate to probe whether Pela's wins survive a broader training
  prior, despite the lower arena number.
* No engine code change: incremental NNUE, VCT budget, and deadline
  plumbing are all carried over unmodified from 0.5.3. This release is
  purely a weights replacement so the comparison stays clean.
* Default path for the real-weights consistency test is bumped to
  `models/gomoku_v14_broken_rapfi_wide.bin` to match the shipped binary.
* Real-game validation once again goes through Piskvork. The arena
  number on its own cannot decide between v13 and v14; Pela score is
  the tie-breaker.

## 0.5.3 (2026-04-25)
* **True incremental NNUE evaluation**. The 0.5.x pipeline had been paying
  for a full `compute_active_features` pass (225-cell × 6 feature sections)
  at every α-β leaf — almost all of the leaf cost. `analyze_psq` on 0.5.2
  losses showed a "slow grind" pattern: static_eval drifts between +200 and
  −100 while Pela assembles a winning row figrid's d4 α-β never sees. The
  only remaining lever is deeper search within the same turn budget, and the
  only way to get that without a faster CPU is to stop redoing the feature
  extraction at every leaf.
* Refactor: (1) `compute_active_features` is now a loop of a new
  `features_from_cell(my, opp, sq, …)` emitter that returns the A/B/C/E/F
  features produced *by that one cell*; density (D) stays global. Output is
  byte-identical to 0.5.2 (all existing tests pass). (2) `IncrementalEval`
  is rewritten from a snapshot wrapper into a real incremental state: it
  caches per-cell features (and density), and on `push_move(board_after,
  mv)` it recomputes only the 11×11 `affected_cells(mv)` window, diffs the
  emitted features against the cache, and applies chunked `FeatureDelta`s
  through `Accumulator::update_incremental`. Perspective swap (stm ↔ nstm)
  is applied to the accumulator, every cached cell, and density since the
  side-to-move just flipped. `pop_move` restores an accumulator snapshot
  plus per-cell rollback and unwinds the swap.
* `Searcher` now plumbs `&mut IncrementalEval` through `alpha_beta`: the
  root calls `refresh` once, recursive `make_move/push_move` and
  `undo_move/pop_move` stay paired. Leaf evaluation is pure forward pass.
* Correctness is guarded by two tests: the near-centre harness from 0.5.1
  plus a new far-apart harness that places moves at opposite corners so
  later moves fall outside each other's `affected_cells` window — this is
  what actually catches the perspective-swap bug in the partial-update path
  (the near-centre harness passes even with the bug present because every
  cell is re-emitted every ply). Both harnesses assert `inc.eval(weights)`
  equals `evaluate(board, weights)` over a 20-move sequence in both push
  and pop directions.
* Weights unchanged — v13_broken_rapfi.bin still shipped. A v14 trained on
  12 999-game data measured 46.7% arena vs v13's 53.3%, not adopted yet.
* Practical effect for Pela: same 5 s turn, more plies in α-β. Whether
  that translates to wins depends on how quickly Pela's plan reaches
  within reach of the extra depth — validation is live Piskvork play.

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
