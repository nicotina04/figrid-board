# figrid 0.7.1 — Gomocup 2026 Ship Postmortem

> Engine: figrid (Five-in-a-Row, NNUE-based, Rust).
> Released: 2026-05-08 (`v0.7.1`).
> Tournament target: Gomocup 2026 (deadline 2026-05-29, tournament June 5–7).
> This document covers the work between `v0.6.10` and `v0.7.1`, the ship
> decision, and what we learned that we'll carry into the next round.

## TL;DR

| Metric                                      | Value          |
| ------------------------------------------- | -------------- |
| `v0.7.0` vs Pela, 100 g, 2 s/move           | **19 / 100**   |
| `v0.7.1` vs Pela, 100 g, 2 s/move           | **29 / 100**   |
| `v0.7.1` vs Pela, Gomocup TC (180 s + 30 s) | **4 / 24**     |
| `v0.7.1` vs Pela, Standard rule, 50 g       | **17 / 50**    |
| `v0.7.1` vs Pela, fastgame TC (5 s + 120 s) | **9 / 30**     |
| Single biggest lift in this cycle           | **TM bug fix** |

Three things actually moved the needle, in order of impact:

1. **Time management was broken.** Fixing it alone gave +10 pp vs. Pela.
2. **Search-side improvements (Pattern4 fast classify, continuation history)
   landed**, but their per-change sibling lifts were inside the noise floor
   for our measurement budget — much of what we thought was an Elo gain
   was sample variance.
3. **Two NNUE fine-tune attempts (v521, v90) failed.** Both confirmed an
   architectural ceiling of the 1D-global-accumulator NNUE class. Neither
   landed in 0.7.1; the v52 weights ship unchanged.

## 1. Timeline

Compressed `git log` for the cycle (most recent at top):

```
68ca75e  fix(0.7.1): phase-based time budget for tournament play
1dc29c8  feat(0.7.0): Pattern4 fast classify + continuation history
d9c8a33  feat(eval): wire noru 2.2 dense-input branch into evaluation pipeline
e9297e1  fix(0.6.10): make avx512 opt-in for crates.io publish
922a575  perf(search): branchless qsearch + AVX-512 feature flag
39327c9  perf(search): TT prefetch + packed sort for move ordering
6785653  feat(0.6.9): add Standard rule (Gomocup rule=1) support
bb87854  feat: 0.6.8 — port v52 weights + 5-stone + conv kernel features
6b10430  release: 0.6.7 — revert v23 hidden expansion, back to v14 weights
f44c8c6  release: 0.6.6 — NNUE hidden layer expanded to [128, 64]
8bfbcee  release: 0.6.5 — drop centre-distance bonus from quiet-move ordering
```

The 0.6.x line is mostly *evaluation* and *search* refinements on top of
the v14 / v23 / v52 weights. The 0.7.0 → 0.7.1 jump is the part of the
cycle this postmortem focuses on.

## 2. The time management bug — search improvements were masked

The single largest lift in this cycle came from a one-commit fix to time
allocation. We discovered it almost by accident.

### Symptom

Under Gomocup-realistic time control (Freestyle-15-2: 180 s match cap +
30 s per-move cap), a calibration sparring of 24 games against Pela
returned **0 wins for figrid 0.7.0**. The engine was losing on time
before the midgame even began.

Same engine, same opponent, with a 2 s/move cap and no match budget,
sat at ~19 % winrate. So the engine was not weaker in the abstract — it
was weaker the way the tournament would actually run it.

### Cause

The pre-fix budget burned 30 s on every move regardless of how much
match time was left. Under a 180 s match budget, that meant the random
opening alone consumed the whole budget, leaving the engine with only a
fixed minimum (~50 ms) for every move thereafter.

### Fix (`pbrain_figrid_noru.rs:turn_budget`)

A five-phase share-based budget that mirrors standard chess-engine TM:

```text
expected_per_side  = 35
remaining          = max(35 - played_this_side, 5)
equal_share        = time_left / remaining

phase multiplier (move_count, ×100):
  0..=5    →  30   (random opening — minimal investment)
  6..=11   →  80   (early midgame ramp-up)
  12..=24  → 150   (tactical peak boost)
  25..=34  → 100   (late midgame)
  35..     →  60   (endgame, often forced)

budget = clamp(equal_share × mul / 100,
               lo = 100 ms,
               hi = min(timeout_turn, time_left / 3))
```

Arena environments that don't announce a match budget keep the original
per-move-cap behaviour via a `timeout_match >= DEFAULT/2` short-circuit.

### Measured impact

| Setting                                    | 0.7.0    | 0.7.1   | Δ            |
| ------------------------------------------ | -------- | ------- | ------------ |
| vs Pela, 100 g, 2 s/move                   | 19 / 100 | 29 / 100 | **+10 pp**   |
| vs Pela, Gomocup TC 180 s + 30 s, 24 g     | 0 / 24   | 4 / 24  | **+17 pp**   |
| Self-play sibling vs 0.7.0, 30 g, 2 s/move | —        | 20 / 30 | **+16.7 pp** |

Side-split of the 100 g cohort:

| Side  | 0.7.0   | 0.7.1   |
| ----- | ------- | ------- |
| Black | 18 / 50 | 27 / 50 |
| White |  1 / 50 |  2 / 50 |

The white-side asymmetry is real and pre-existing; we revisit it in §3.

### The lesson — TM gates everything else

A previously-rejected leaf-VCF gate change had been benched as **−7 pp**
under the buggy TM. We re-tested it under the new TM and it produced an
**exact 50 % sibling**: the −7 pp wasn't a search defect, it was the
broken TM failing to absorb the cost of the extra search.

Generalising: at this Elo level, every search/eval experiment we ran
prior to the TM fix was being measured *through* the broken TM. Most of
the per-change deltas we treated as signal were noise riding on a
hidden constant cost.

> **If your TM is broken, you cannot trust any other measurement on the
> engine.** Isolate and verify TM first.

## 3. Fine-tune attempts — distribution dominates the prior

We tried twice, on the same v52 base, to fix the white-side asymmetry
(2 / 50 white wins in 0.7.1) by Rapfi-distilled fine-tuning on
white-loss positions. Both attempts failed in instructive ways.

### Attempt 1 — v521 (catastrophic)

- **Data**: 55 white-loss games (2 source pools), Rapfi-labelled at
  every figrid-to-move ply. 336 raw positions.
- **Distribution**: low (cp/200 < −1) **224 (67 %)**, mid 103 (31 %),
  high 9 (3 %).
- **Training**: lr 5 × 10⁻⁷, 2 epochs, batch 32, no anchor mix, fine-
  tune from `gomoku_v52_5stone_conv_93k.bin`.
- **Result**: catastrophic regression on the immediate sibling
  arena — 3 / 30 wins (≈ 10 %).

What happened: 67 % of training samples were "white losing". The
network learned a "this position is losing" prior over essentially
every reasonable position. v521 became uniformly pessimistic and lost
to the unmodified v52 in head-to-head.

### Attempt 2 — v90 (no catastrophic, no lift)

We addressed v521's failure modes directly:

1. **Distribution control**: filter the targeted corpus to (a) drop
   "structurally lost" games (first Rapfi mate detection ≥ 18 ply)
   and (b) include short-mate positions as cp targets via a graded
   mapping `cp_logit = sign × max(1, 10 − |mate|)`. Output: 246
   positions, distribution 58 % low / 41 % mid / 0.4 % high — still
   skewed but much better than v521.
2. **Anchor mix**: 95 % v52's original training corpus + 5 % targeted
   (oversampled ×5 to preserve per-epoch signal density). Final mixed
   buckets seen by the trainer: low 9653 (39 %), mid 5633 (23 %),
   high 9214 (38 %).
3. **Same lr, same epochs** as v521 to keep the comparison clean.

Stop-loss criteria fixed in advance:

- Pre-train: mid+high < 30 % → don't train. (v90 hit 41 %, passed.)
- Sibling 30 g: regression > 10 pp → revert. (v90 hit +3.3 pp,
  passed.)
- vs Pela 100 g: < +3 pp → ship the prior version, not v90.

| Phase                          | Result               | Decision           |
| ------------------------------ | -------------------- | ------------------ |
| v90 sibling vs 0.7.1, 30 g     | 16-14 (53.3 %, +3.3) | continue           |
| v90 vs Pela, 100 g             | 23 / 100             | **−6 pp regression** |
| v90 by side                    | W 0/50, B 23/50      | white untouched    |

**v90 was scrapped.** The sibling 30 g had said "+3.3 pp"; the 100 g vs
Pela said "−6 pp on the variable that matters". The sibling
measurement was inside its noise floor.

### Pattern lessons

| Lesson                                                 | v521         | v90          |
| ------------------------------------------------------ | ------------ | ------------ |
| Distribution skew > anchor mix capacity to absorb      | yes (67 %)   | mostly (58 %)|
| Catastrophic regression                                | yes          | no           |
| Meaningful lift on ship-decision metric (100 g vs Pela)| no (untested) | no (−6 pp)  |
| Anchor-mix prevented catastrophic prior shift?         | n/a          | **yes**      |

So the anchor mix worked at exactly what it was designed for —
preventing a prior collapse — but did not enable any measurable
improvement on the underlying problem.

The mate-to-cp graded mapping and the β-region game filter
(`docs/feedback_mate_to_cp_mapping.md`) survive as design pieces worth
reusing whenever we touch fine-tune again. The data extraction
pipeline (`extract_pela_v5.py` style) is reusable across base models.

## 4. Measurement protocol — what we trust for what

A side effect of running v521 and v90 under tight stop-loss criteria
was that we generated enough sibling-vs-Pela cross-data to estimate
our own measurement variance.

### Two protocols, two error bars

| Protocol           | N    | σ (binomial, p=0.5) | Use case                          |
| ------------------ | ---- | ------------------- | --------------------------------- |
| Sibling head-to-head | 30 | ≈ 2.7 wins ≈ **9 pp** | regression sanity, "did we break it?" |
| vs Pela            | 100  | ≈ 5.0 wins ≈ **5 pp** | ship decisions, "is this a real lift?" |

### The cumulative-measurement reality check

If every per-change sibling result we'd reported in this cycle had
been a real Elo gain, they would have stacked:

```
Pattern4 fast classify     +13 pp sibling vs 0.6.10
Continuation history       +10 pp sibling vs Pattern4
TM fix                     +16.7 pp sibling vs 0.7.0
─────────────────────────  ─────────
Naïve sum                  ≈ +40 pp
```

We then ran the cumulative test directly: `0.7.1 vs 0.6.x baseline`,
30 g sibling. Result: 13-17 (≈ 43 %, **−7 pp from 50 %**).

In other words, most of the per-change sibling lifts were within
σ ≈ 9 pp of zero. The TM fix is the one that survived a 100 g vs Pela
re-measurement (29 % vs 19 %, 10 pp clear of σ ≈ 5 pp).

### Operating rule (now codified in memory)

- **30 g sibling**: regression sanity only — "did the build stop working?"
  Pass criterion: not worse by more than σ. Never used as a ship signal.
- **100 g vs the strongest external opponent we can run cheaply (Pela)**:
  the only signal allowed to gate `--features embed-weights` ship.
- **Pre-train data audit**: distribution buckets reported before any
  fine-tune. Refuse to start training if distribution makes the prior
  outcome predictable.

## 5. Architectural ceiling — NNUE-class is saturated

Two distribution-controlled fine-tunes (v521 67 % low, v90 58 % low /
post-mix 39 % low) on the same v52 base, both targeted at the same
white-side gap, both produced no measurable lift on the ship-decision
metric. v90 with anchor mix didn't even regress.

Read together with the `docs/figrid_nnue_v1.2_report.md` cross-LLM
architecture review, the diagnosis converges:

- v52 is a 1D-global-accumulator NNUE: `feature_size=N → accumulator∈ℝ⁵¹² → hidden [128, 64] → 3-way WDL head`.
- Per-cell positional information is collapsed into the global
  accumulator by the first linear layer. Refining downstream layers
  with more / better cp targets cannot recover what is geometrically
  lost in that collapse.
- The `figrid_nnue_v1.2_report.md` "Phase B" recommendation
  (MixNet-lite — per-cell feature map with a line-pattern codebook) is
  an architectural change of class, not a hyperparameter sweep. It
  doesn't fit inside the Gomocup 2026 deadline.

We are not training-data-limited. We are encoding-capacity-limited.

## 6. What 0.7.1 ships

| Artefact                          | Path / URL                                            | Size    |
| --------------------------------- | ----------------------------------------------------- | ------- |
| `pbrain-figrid.exe`               | `dist/gomocup-2026/`                                  | 2.30 MB |
| `pbrain-figrid-avx512.exe`        | `dist/gomocup-2026/`                                  | 2.30 MB |
| `figrid-0.7.1.zip` (dual)         | `dist/`                                               | 4.0 MB  |
| GitHub release (`v0.7.1`)         | github.com/nicotina04/figrid-board/releases/v0.7.1    | 3 assets |
| crates.io publish                 | crates.io/crates/figrid-board/0.7.1                   | 1.95 MB |

All binaries are self-contained: `embed-weights` bakes the v52
NNUE weights (`gomoku_v52_5stone_conv_93k.bin.gz`, 1.7 MB) into the
executable. The C runtime is statically linked (`+crt-static`), so
no Visual C++ runtime install is needed on the target machine.

### Gomocup 2026 category fit

| Category                | Decision     | Reason                                           |
| ----------------------- | ------------ | ------------------------------------------------ |
| Freestyle 15×15         | **register** | 100 g vs Pela = 29 %, baseline cohort            |
| Freestyle 20×20         | skip         | `BOARD_SIZE = 15` hardcoded; v52 is 15×15-only   |
| Standard 15×15          | **register** | 50 g vs Pela = 34 % (within Freestyle noise)     |
| Fastgame (5 s + 120 s)  | **register** | 30 g vs Pela = 30 % (within Freestyle noise)     |
| Renju                   | skip         | black-restriction rules (3-3, 4-4, overline) not implemented |
| Caro                    | skip         | Caro-specific win conditions not implemented     |

Three categories of seven. Acceptable for a first NNUE submission;
expanding the category mix is a deliberate next-cycle goal, not a
deadline-pressed scramble.

## 7. Forward — paradigm shift, not parameter sweep

Two distribution-controlled fine-tunes have now confirmed that the
1D-global-accumulator NNUE class cannot be pushed further on this
problem with the data and training infrastructure we have. The MixNet-
lite recommendation in `figrid_nnue_v1.2_report.md` is *also* a
NNUE-class refinement (per-cell features + Star Block), which will
likely add Elo but does not change the underlying paradigm.

The next-cycle direction is left intentionally open in this document.
The salient constraints are:

- **Incremental update is non-negotiable.** Tournament TC
  (5 s/move down to 120 s/match) means whatever evaluator we ship next
  must support delta-update on stone placement, the way NNUE does today.
- **Anything that wins must be quantizable to int16 / int8.** The
  `+crt-static` self-contained binary contract holds; we are not
  shipping a Python runtime.
- **Search-side scaffolding (Pattern4 mini, continuation history,
  leaf VCF gating) is reusable.** Whatever new evaluator goes in plugs
  into the same α-β + qsearch + TT + history surface.

Gomocup 2026 ships on 0.7.1. The next round is a different evaluator
class, not a re-train.

## Appendix — reproducibility pointers

- Sparring runners: `python/gomoku-sparring/sparring_vs.py` (random-
  opening, swap-sides, `--rule` for Gomocup rule 0/1).
- Game records used in this writeup:
  - `games/v070_conthist_vs_pela_100g.jsonl` — 0.7.0 vs Pela 100 g
  - `games/v070tm_vs_pela_100g.jsonl` — 0.7.1 vs Pela 100 g
  - `games/v90_vs_pela_100g.jsonl` — v90 vs Pela 100 g
  - `games/v071_vs_pela_standard_50g.jsonl` — 0.7.1 vs Pela Standard 50 g
  - `games/v071_vs_pela_fastgame_30g.jsonl` — 0.7.1 vs Pela fastgame 30 g
- Training logs (separate `noru-tactic` repo):
  - `experiments/2026-05-08/v521_pela_black_finetune.log`
  - `experiments/2026-05-08/v90_pela_b_finetune.log`
- TM fix: commit `68ca75e`, file `bin/pbrain_figrid_noru.rs` function
  `turn_budget`.
- Search-side ship: commit `1dc29c8`, files `src/pattern_table.rs`,
  `src/vct.rs`, `src/search.rs`.
