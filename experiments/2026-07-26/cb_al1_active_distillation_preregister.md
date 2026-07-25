# CB-AL1 label-blind active-distillation precondition

Date: 2026-07-26

Status: preregistered before any CB-AL1 row-bearing run

Product baseline: `figrid-board 0.8.2`

This document freezes the cheap CB-AL1 precondition before either row-bearing
input is opened for this card. It does not authorize a teacher query, training,
validation, artifact replacement, search change, arena, or release-weight
change.

## Pre-run erratum

The initial preregistration commit is `c585617`. A static implementation audit,
performed before any row-bearing input was opened, found that its exact
critical-source list named `build.rs`, although neither that commit nor the
working tree has such a file. The exact stream below removes only that
nonexistent entry. No source file is substituted, no metric/gate changes, and
the commit containing this erratum is the implementation's registered
preregistration ancestor.

## Question and claim boundary

The accepted product remains the flat swap-closed Pattern4 codebook, shipped as
the exact class-base plus `i8` residual CBF. CB-GH1 found no tractable graph
state space, so CB-AL1 does not reopen a graph coreset or add a representation.

At a fixed budget of 125 paired units / 250 color slots, this P0 asks:

> Does a White-prioritized, label-blind selector based on archived
> product-search versus current deployed-static disagreement produce at least
> `+3 pp` more **K=6-local usable-error discoveries per attempted slot** than
> a deterministic hash-random control, while also producing strictly positive
> usable-regret enrichment, at least `+3 pp` measurable-only conditional error
> enrichment combined/White with no Black reversal, and measurable-only
> conditional regret enrichment combined/White with no Black reversal?

The point comparison is accompanied by a finite-support random-control
distribution and a dependence stress test. Regret has a direction gate, not a
`3 pp` magnitude gate.

This is an active-distillation precondition, not a strength or causal claim.
It tests retrospective query-attempt utility on consumed labels. It cannot show
that a trainer can repair the errors, that search will expose a repaired
evaluator, that the selector generalizes beyond this finite support, or that
games will improve.

### Exact label boundary

The Phase-2 rows are:

- pre-RQ615C-teacher, RQ615C-label-free, and outcome-free;
- answer-opaque at selection time;
- **not wholly answer-free in lineage**.

Their historical membership was filtered by an opaque firewall containing
answer-derived RQ614 identities. P0A can observe only the firewall contract and
seal recorded in the Phase-2 manifest, never its members or answers. Therefore
the narrow valid claim is only that the AL1 ranking and arm selection read no
teacher answer, teacher score, candidate role, or outcome. A GO can authorize
only a separately preregistered fresh acquisition test with this limitation
removed or explicitly controlled.

## Historical facts available at registration

- RQ515 found 86,337 model mistakes in 102,317 usable groups. Pair-only training
  repaired its mined-pair metric to 96.7% while catastrophically damaging
  global ordering.
- Later mixed/listwise attempts improved offline metrics without a reliable
  runtime bump. Archived product-search moves are therefore a second required
  choice family.
- RQ615C prepared 1,000 paired-color units before its Rapfi queries. The rows
  contain histories, archived product moves, and full legal inventories, but
  no RQ615C answer or outcome.
- RQ615C train labels are consumed. They may close or open a future card but
  cannot promote a model.
- QAT1 FP32/PTQ disagreement failed the required direction and White gates. It
  is forbidden as an AL1 acquisition signal.
- The old `mine-codebook-hard-negatives` executable is forbidden: it reads
  positives before mining, calls `candidate_moves()` despite an all-legal
  claim, uses an old FP32 path, lacks the D4/component firewall, and overwrites.

No CB-AL1 selected UID, label statistic, acquisition/control comparison, or
row-level census was inspected before this registration.

## Frozen artifacts

### P0A selection-time inputs

| input | bytes | SHA-256 |
|---|---:|---|
| `rq615c_prepared_units_1000.jsonl` | 78,707,493 | `2B5391DD9BB78969F119AD70162CDCA185E62B25FAB720CA7AB852030DDFC74B` |
| `rq615c_phase2_prepared_manifest.json` | 3,500 | `92D6BF8E6F42181F0A25BDDF41D839B59A9B758D25637081DC57A375C50F4C4D` |
| product FP32 JSON, `gomoku_codebook_v1_swapclosed.json` | 1,410,562 | `42968FDAB01BA8CCD1DE3DED05C532E4B237DD47EEFFD7AE1C2F264D77BA7DA2` |
| product factored CBF, `gomoku_codebook_v1_swapclosed_factored.cbf` | 353,582 | `141014529417A73E58B210832AFD189AD970E045A8907F7D2879693C5B171A8D` |
| swap-closed vocabulary, `data/topk.bin` | 17,060 | `103891DCD1DCD978C654593ABE78EF32C56E2E350B500EE665BC45AC051AA16D` |

The Phase-2 manifest must reproduce:

- format `rq615c-phase2-prepared-manifest-v1`, status `complete`;
- contract flags `answer_opaque_firewall_only=true`, `outcome_free=true`,
  `full_legal_inventory_present=true`, and
  `exact_1000_units_2000_parents=true`;
- historical-firewall seal: 292,573,334 bytes,
  `3886D4645881531CEC0698B9BC9DCA8E27E12BB517F039A95C8625297B48D4E6`;
- exactly 1,000 units / 2,000 parents and split units `700/150/150`;
- exactly 1,000 Black and 1,000 White parents;
- 428,320 legal-inventory entries and 2,000 unique parent D4+side hashes;
- maximum two selected units per opening;
- source representation `E32/H2048/F2048`,
  `base_logit_perspective="p(Black)"`, `guard_k=6`;
- full legal inventory in ascending `cell = y * 15 + x`.

P0A may read all rows to validate the single artifact but may select only its
700 `train` units. Its CLI has no train-label, Phase-3, dev, safety, RQ508,
RQ607/RQ612, trace, outcome, or arena input.

### P0B reveal-time inputs

P0B may open labels only after checking a valid create-new P0A output against
literal expected bytes and SHA-256 supplied on its command line.

| input | bytes | SHA-256 |
|---|---:|---|
| P0A selector manifest | create-new; literal bytes/SHA supplied to P0B |
| `rq615c_k6_train.jsonl` | 54,991,200 | `E00A2DA513B05D7631A01003C7E6274E9A3D7575E2C2BD92D5199F1B5385CEB6` |
| `rq615c_final_corpus_manifest.json` | 5,463 | `579D1387D7E4DE8F5CB34DB168B6D15655DB229D992751B1DC17BB6CF4260AA7` |
| RQ569 lineage JSON | 1,413,542 | `69BB7C599ADA3A1151577CE3315356BC33C40EDB49A003C9BC4EB90A98F82E18` |

The train projection must independently reproduce 1,336 train slates, 668
paired units, 388 train components/assignments, 668/668 Black/White rows, K=6,
and 8,016 candidate children. Because dev/safety rows are forbidden inputs,
P0B requires—but does not claim to independently recompute—the exact sealed
final manifest's zero-valued cross-split opening/parent/child/state overlap
audit. P0B has no dev, safety, Phase-3, RQ508, RQ607/RQ612, trace, outcome,
new-teacher, timing, or arena input.

## Hashing, replay, and numeric conventions

Every SHA-256 preimage named below is the exact ASCII/UTF-8 byte sequence shown,
with no NUL or newline. Digests are rendered as 64 uppercase hexadecimal
characters. Digest ordering compares the raw 32 bytes ascending.

The eight D4 coordinate transforms, in order with `n=14`, are:

```text
(x,y), (n-y,x), (n-x,n-y), (y,n-x),
(n-x,y), (x,n-y), (y,x), (n-y,n-x)
```

For a parent state, transform every stone, encode it as
`{B|W}{cell:03}`, sort the stone tokens, and form
`rule=0|side={B|W}|` plus their comma join. The lexicographically smallest of
the eight strings is `canonical`; the state hash is:

```text
SHA256("RQ608-state-v1|" || canonical)
```

For the exactly four-ply ordered opening, preserve ply order and encode each
transformed stone as `{ply}:{B|W}{cell:03}`. Prefix the comma join with
`rule=0|`, choose the lexicographically smallest D4 form, then hash:

```text
SHA256("RQ608-ordered-opening-v1|" || canonical)
```

The split is the base-16 integer value of
`SHA256("RQ615C|opening-group|" || opening_hash)` modulo 100:
`0..69=train`, `70..84=dev`, `85..99=safety`.

The exact identity preimages are:

```text
unit_uid =
  SHA256("RQ615C|structural-unit|" || opening_hash || "|" || ordinal ||
         "|" || black_parent_hash || "|" || white_parent_hash)

parent_uid =
  SHA256("RQ615C|structural-parent|" || unit_uid || "|" || B_or_W ||
         "|" || parent_hash)

projected_row_uid =
  SHA256("RQ615C|projected-slate-v1|" || opening_hash || "|" || ordinal ||
         "|" || parent_hash)
```

All selector scores, utilities, margins, gaps, and means are binary32.
All teacher probabilities, regrets, sums, means, and reported deltas are
binary64. Finite score maxima use numeric comparison and lower cell as the
exact tie-break. Selector score ordering uses `f32::total_cmp` only where
specified. Each binary32 value and its `to_bits()` are recorded.

Regret sums use Neumaier compensated binary64 summation. The fixed point arm
order is `(ordinal ascending, uppercase unit_uid ascending, Black then White)`.
Random controls are sorted into that order before summation. Cluster replicate
order is sampled-cluster draw order, then the same within-cluster order.

## Build and provenance boundary

`Cargo.lock` is 11,841 bytes with SHA-256
`3F90AA762C0D7B1F0172C22397588835C79B9C924BB5A931D162B2A5714A202C`.
The registered toolchain is:

- `rustc 1.88.0 (6b00bc388 2025-06-23)`;
- rustc commit `6b00bc3880198600130e1cf62b8f8a93494488cc`;
- host `x86_64-pc-windows-msvc`, LLVM `20.1.5`;
- `cargo 1.88.0 (873a06493 2025-05-10)`.

The working directory for build and both runs is exactly:

```text
C:\Users\concreate\.codex\worktrees\06f2\noru-tactic\target\figrid-release-0.8.2
```

In PowerShell, set the exact environment entry first:

```powershell
$env:RUSTFLAGS = '-C target-cpu=x86-64-v3'
```

Then the clean committed release build argv is exactly:

```powershell
cargo build --release --locked --features cb-al1-audit --bin cb-al1-selector
```

`cb-al1-audit` is default-off and implies `codebook-eval`. The executable must
record and require:

- `target_profile="release"` and `debug_assertions=false`;
- enabled features exactly `["cb-al1-audit","codebook-eval"]`;
- runtime `RUSTFLAGS="-C target-cpu=x86-64-v3"`;
- compile-time and runtime AVX2, BMI2, and FMA all true;
- clean git worktree, a 40-hex HEAD descending from this preregistration
  commit, release-directory executable, and unchanged executable SHA;
- exact working directory, subcommand, argument vector, rustc/cargo/CPU
  identities, input pre/post seals, and start/end times.

Environment names are compared case-insensitively. `NORU_*`, `FIGRID_*`,
`RAYON_*`, `CARGO_PROFILE_*`, `LLVM_PROFILE_FILE`, `GCOV_PREFIX`,
`GCOV_PREFIX_STRIP`, `RUSTC_WRAPPER`, `RUSTC_WORKSPACE_WRAPPER`,
`RUSTDOCFLAGS`, `CARGO_ENCODED_RUSTFLAGS`, `RUSTC_BOOTSTRAP`, and
`CARGO_INCREMENTAL` must all be absent. Unknown or duplicate CLI arguments are
rejected. At build and both runs, case-insensitive environment enumeration must
contain exactly one `RUSTFLAGS` name with the exact registered value; a
case-variant duplicate is invalid.

The exact critical-source stream, in this order, is:

```text
Cargo.toml
Cargo.lock
src/lib.rs
src/board.rs
src/codebook_eval.rs
src/factored_codebook.rs
src/pattern_table.rs
src/d4_hash.rs
src/search.rs
src/token_delta.rs
bin/cb_gh1_graph_census/graph.rs
bin/cb_gh1_graph_census/provenance.rs
bin/cb_gh1_graph_census/corpus.rs
bin/cb_al1_selector.rs
bin/cb_al1_selector/hash.rs
bin/cb_al1_selector/prepared.rs
bin/cb_al1_selector/reveal.rs
bin/cb_al1_selector/stats.rs
bin/cb_al1_selector/tests.rs
experiments/2026-07-26/cb_al1_active_distillation_preregister.md
data/topk.bin
```

For each path, append `u32_le(path_byte_len)`, path UTF-8 bytes,
`u64_le(file_byte_len)`, then file bytes, and SHA-256 the concatenation. Every
file must be tracked, its disk bytes must match the bytes compiled into the
executable, and the stream identity must be unchanged pre/post stage.

## P0A: label-blind selector

### Exact prepared schema and fail-closed replay

A prepared row has exactly:

```text
format, unit_uid, split, opening_group_hash, figrid_ordinal, parents
```

`format` is `rq615c-preteacher-paired-unit-v1`; ordinal is one of
`[1,2,4,6,8]`. `parents` has exactly `figrid_black` and `figrid_white`.
Each parent has exactly:

```text
parent_uid, parent_d4_side_hash, side_to_move, history,
figrid_actual_move, legal_inventory
```

History stones have exactly `x,y,color`; moves have exactly `x,y`. Inventory
entries have exactly:

```text
move, child_d4_side_hash, base_logit_f32, base_logit_f32_bits
```

P0A validates every one of the 1,000 rows, the split/UID/hash preimages above,
global unit/parent/hash uniqueness, literal Black/White four-ply opening
identity, alternating turns, Freestyle ongoing roots, legal archived moves,
all empty cells exactly once in strict cell order, and every child hash.

The stored legacy inventory is independently replayed by quantizing the product
FP32 payload as `E32/H2048/F2048`, making each child, temporarily forcing the
child evaluator perspective to Black, and requiring exact
`base_logit_f32_bits`. These stored p(Black) bits are lineage audit fields and
never selector scores.

For both the legacy `E32/H2048/F2048` lattice and current `E32/H64/F64`
lattice, each finite FP32 weight is quantized independently as:

```text
scaled  = value * (scale as f32)             // binary32 operation
rounded = scaled.round()                     // Rust f32::round, ties away from zero
require i16::MIN <= rounded <= i16::MAX
quantized = rounded as i16                   // no saturation
```

Embedding, head, and factor scales are applied to their respective full
vectors; the FP32 bias is never quantized and must preserve its exact bits.
Any non-finite input, non-finite intermediate, or overflow is invalid.

### Product payload and current scoring

The CBF must parse as the factored product kind. Its entire source payload
(`dim`, `fm_rank`, embeddings, head, factors, FP32 bias) must be bit-identical
to the frozen product JSON. Its public factored payload must validate;
reconstructing the flat `E32/H64/F64` payload must equal a fresh quantization
of that JSON in every embedding, head, factor, scale, and bias bit.

The default-off `cb-al1-audit` feature explicitly authorizes exactly one
doc-hidden audit-only public wrapper in `src/codebook_eval.rs`:

```text
pub fn evaluate_full_factored_quantized_for_audit(
    board: &Board,
    weights: &FactoredQuantizedCodebookWeights,
) -> f32
```

Its body constructs `IncrementalQuantizedCodebookEval` through
`new_with_access(weights, false)`, performs one
`refresh_with_access(board, weights)`, then returns
`value_profiled_with_access(board, weights, false).0`. It performs no search,
incremental move push, directional delta, profiling, cache reuse, or mutation
of `board`. This feature-only wrapper is the sole authorized library API
addition for AL1 and is absent unless `cb-al1-audit` is enabled.

For every P0A root and legal child, the audit-only factored runtime path and
reconstructed-flat full evaluator must be bit-identical. Product scoring uses
that factored runtime result:

1. make the move;
2. obtain the natural child-side-to-move binary32 value `ell`;
3. set binary32 root-mover utility `u = -ell`;
4. undo and require exact board/root restoration.

All scores are finite. No legacy stored logit participates in selection.

### Quiet paired-unit eligibility

For a specified stone and an empty cell, the immediate-five audit virtually
places that stone without consulting `side_to_move`, then counts the placed
stone plus contiguous equal stones in both directions of the four Freestyle
axes. A count at least five is an immediate win. Empty cells are visited in
ascending cell order.

A parent is quiet-eligible only if it has at least six legal moves, neither the
mover nor opponent has an immediate win, and the archived actual move is legal.
A unit is eligible only if both colors qualify. Exclusions are reported by
ordinal, split, color, and reason, without inspecting later labels or roles.

### Diagnostics, support, and arms

For each parent:

```text
static_top         = highest all-legal deployed utility
static_second      = next move under the same ordering
margin             = static_top_u - static_second_u       (binary32)
actual_gap         = static_top_u - archived_actual_u      (binary32)
search_disagreement = static_top != archived_actual
mean_margin        = (margin_white + margin_black) / 2.0f32
```

Every intermediate is finite and nonnegative and is stored with its bits.

For each ordinal, sort eligible `train` units by the raw digest:

```text
SHA256("CB-AL1|support-v1|" || uppercase(unit_uid))
```

and retain the first 100. The active arm takes the first 25 per ordinal under:

1. White disagreement, `true` first;
2. White `actual_gap`, descending `f32::total_cmp`;
3. White `margin`, ascending `f32::total_cmp`;
4. `mean_margin`, ascending `f32::total_cmp`;
5. raw `SHA256("CB-AL1|active-v1|" || uppercase(unit_uid))` ascending.

The deterministic control takes the first 25 per ordinal by raw:

```text
SHA256("CB-AL1|control-v1|" || uppercase(unit_uid))
```

Arms are independent. Overlap is retained and never refilled. P0A support
requires:

- 100 support units and 25 arm units in every ordinal;
- exactly 125 units / 250 color slots per arm;
- no duplicate within an arm;
- active/control overlap at most 50 units;
- at most two units per opening, at least 63 distinct openings per arm;
- byte-identical selection and serialization from a second in-process pass.

Fewer than 100 eligible units, excess overlap, or opening support failure is
the valid status `NO_GO_SELECTOR_SUPPORT`, not corruption.

Each support/active/control stream is hashed as the ordered sequence of
uppercase UID ASCII followed by one LF per UID. The create-new P0A JSON uses
`serde_json::to_vec_pretty` plus one terminal LF and records complete
diagnostics, ordered UID streams, hashes, provenance, and either
`P0A_READY_FOR_REVEAL` or `NO_GO_SELECTOR_SUPPORT`.

## P0B: sealed consumed-label reveal

P0B must, before opening or hashing the train/manifest/lineage files:

1. require the P0A path, literal `--expected-p0a-bytes`, and literal
   `--expected-p0a-sha256`;
2. seal the P0A file and match both literals;
3. re-run all P0A input, source, executable, and selector checks;
4. require byte-identical UID streams and the same implementation
   commit/executable seal as P0A.

Only then may it open the consumed labels.

### Exact projected schema and lineage replay

Each train row has format `rq615c-k6-projected-slate-v1`,
`final_split="train"`, and exactly:

```text
format, row_uid, final_split, component_uid, opening_group_hash,
parent_d4_side_hash, side_to_move, figrid_ordinal, history, candidates,
legal_inventory, repeat_scores_mover, repeat_bands_mover, q_teacher
```

Candidate keys are exactly:

```text
candidate_index, move, roles, child_d4_side_hash,
base_logit_f32, base_logit_f32_bits
```

Inventory keys are the same except index/roles. There are six unique candidates
at indices `0..5`. Roles are nonempty lists drawn only from `teacher_top`,
`deployed_actual`, and `base_rank_1` through `base_rank_6`; teacher-top and
deployed-actual each occur on exactly one candidate.

Rows pair uniquely by `(opening_group_hash, figrid_ordinal)` into one Black and
one White row with the same component. P0B reconstructs the unit UID from the
paired parent hashes. For every joined P0A parent it requires exact history,
side, parent/opening hashes, full cell-ordered legal inventory, child hashes,
and legacy FP32 bits. Every candidate is bit-identical to its inventory entry,
and the unique `deployed_actual` coordinate equals P0A's archived move.

The product JSON and RQ569 lineage JSON must have bit-identical FP32 E/H/F/bias
payloads. All full-inventory legacy bits are separately replayed on the
`E32/H2048/F2048`, forced-Black lattice. All current selector scores are
separately replayed on the CBF/reconstructed-flat `E32/H64/F64`,
natural-child lattice. History, hashes, raw-to-mapped Pattern4 IDs, both
lattices, and make/undo must have zero mismatch.

For corpus components, union stable units sharing any identity token among:

```text
O:{opening_hash}
S:{Black parent hash}
S:{White parent hash}
S:{each Black/White K6 child hash}
```

For each component, sort/unique member UIDs and identity tokens, then require:

```text
component_uid =
  SHA256("RQ615C|component-v1|units=" || comma_join(unit_uids) ||
         "|identities=" || comma_join(identity_tokens))
```

The stored 388 component UIDs and final train assignments must reproduce.

Each repeat score is an integer in `[-3000,3000]`. Its band is `forced_win` for
score `>=2500`, `forced_loss` for score `<=-2500`, otherwise `cp`; the two
repeat bands agree candidate-wise. For repeat `r`, in candidate-index order:

```text
z_i = score_i / 400.0
w_i = exp(z_i - max(z))
q_r_i = w_i / sum(w)
q_teacher_i = (q_0_i + q_1_i) / 2.0
```

Stored values must be finite, sum within `1e-12` of one, and reproduce within
absolute `1e-15`.

### Join and K=6-local measurement

All 500 support units are joined so that the random-control distribution can
be computed retrospectively. A unit absent from final train consumes its fixed
slots but has no measurable choice. The file is never searched outside train.

The archived actual move must occur exactly once as `deployed_actual`;
absence or drift violates the RQ615C contract and is invalid. The current H64
`static_top` was not guaranteed by historical K=6 construction. Its normal
absence is therefore a valid unmeasurable slot, never invalid and never
prefiltered in P0A.

For a measurable choice, let `q_max = max_i q_teacher[i]`, with exact binary64
ties accepted:

```text
error  = 1 iff q_teacher[choice] < q_max
regret = q_max - q_teacher[choice]
```

Claims are strictly K=6-local. Missing/unmeasurable slots contribute zero error
and zero regret to fixed-slot acquisition yield and are excluded from
measurable-only conditional denominators.

For each arm, choice family, combined/color/ordinal:

- fixed-slot usable-error discovery = errors / attempted slots;
- fixed-slot usable mean regret = regret sum / attempted slots;
- conditional error rate = errors / measurable slots;
- conditional mean regret = regret sum / measurable slots;
- matched units, measurable slots, ties, and coverage are reported.

Attempted denominators are 250 combined, 125 by color, 50 by ordinal combined,
and 25 by ordinal/color. Component reports use the actual fixed slots attached
to that component.

Measurement support requires, per arm:

- at least 115 of 125 units have complete paired train rows;
- at least 115 measurable White and 115 measurable Black slots for each choice
  family;
- at least 30 dependence clusters and no cluster containing more than 12 units
  from either arm.

Failure is valid `NO_GO_MEASUREMENT_SUPPORT`.

For fixed-slot error gates, use exact count differences:

- combined active minus control at least `8` (`8/250 = 3.2 pp`);
- White at least `4` (`4/125 = 3.2 pp`);
- Black at least `0`.

For conditional error gates, let `err_A, den_A, err_C, den_C` be the active and
control error/measurable counts. Combined and White use this exact checked
`u128` predicate:

```text
100 * err_A * den_C
  >= 100 * err_C * den_A + 3 * den_A * den_C
```

Black uses `err_A * den_C >= err_C * den_A`. Combined and color gates are
independent; White `+3 pp` plus Black zero does not by itself satisfy combined
`+3 pp`.

For both fixed and conditional mean regret, active must be strictly greater
combined and White and no smaller for Black. These Black checks are observed
no-harm guards, not statistical noninferiority.

Identical overlapping UIDs must cancel exactly in the fixed error numerator and
the paired fixed regret numerator. The report records active-only,
control-only, overlap, and neither over the 500-unit support, by choice family
and ordinal, and verifies the paired-difference invariant.

### Finite-support random controls

This distribution measures how the fixed active arm ranks against uniform
budget-matched pseudorandom controls in the frozen finite support. It is not a
population CI, p-value, or causal estimate.

Use 100,000 replicates and SplitMix64 seed `2026727102`. For every replicate
and ordinal in `[1,2,4,6,8]`, start with indices `0..99` in support-hash order.
For `i=0..24`, draw `j = i + bounded(100-i)`, swap indices `i,j`, and take the
first 25. The same 125-unit control is reused for every metric.

The exact SplitMix64 transition is:

```text
state = state + 0x9E3779B97F4A7C15 (wrapping u64)
z = state
z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 (wrapping u64)
z = (z ^ (z >> 27)) * 0x94D049BB133111EB (wrapping u64)
return z ^ (z >> 31)
```

For nonzero `n`, the exact unbiased bounded sampler is:

```text
threshold = (0u64 - n) % n                 // wrapping subtraction
loop:
    x = next_u64()
    if x >= threshold: return x % n
```

For both choice families and combined/White, calculate active-minus-random
fixed-slot usable-error discovery and fixed-slot usable mean regret. Sort each
100,000-element finite binary64 vector with `f64::total_cmp`; `p10` is
zero-based index `9,999`. RNG consumption, selected-control stream hashes, and
quantile-input hashes are recorded. Error signs are computed from the signed
integer error-count difference. Regret uses signed per-UID paired
active-minus-random contributions in sorted union order, so overlapping UIDs
cancel before summation.

The randomization RNG hash appends every raw `next_u64()` result consumed,
including bounded-sampler rejections, as `u64_le` in call order. Its selected
UID hash appends uppercase UID ASCII plus LF in
`replicate -> ordinal [1,2,4,6,8] -> selected index 0..24` order. For each
metric, the pre-sort quantile-input hash appends `f64::to_bits()` as `u64_le`
in replicate order; the sorted-input hash appends the same encoding after
`f64::total_cmp` sorting.

### Dependence-cluster stress test

This secondary calculation is conditional stability for the selected
active/deterministic-control sets, not a selector-vs-random CI.

Build a DSU over the selected-arm union. For every selected unit, union units
sharing any answer-opaque prepared identity token:

```text
O:{opening_group_hash}
S:{either parent_d4_side_hash}
S:{any full-legal child_d4_side_hash from either color}
```

Additionally union matched units sharing reconstructed RQ615C `component_uid`.
Thus missing units retain every pre-label dependence visible in Phase 2 and
join any matched unit sharing one of those identities; no missing UID is
silently promoted to its own cluster. A cluster key is:

```text
SHA256("CB-AL1|cluster-v1|" || comma_join(sorted_uppercase_member_uids))
```

Sort keys ascending. With 100,000 accepted replicates and SplitMix64 seed
`2026727101`, draw `G` keys with replacement using the same `bounded(G)`
sampler defined above, where `G` is cluster count. One draw stream is reused
for all metrics. Each arm's replicate rate
uses the attempted slots attached to that arm among the sampled cluster
occurrences as its denominator. A draw with zero sampled attempted slots in
either arm is discarded deterministically while retaining RNG state; failure
to obtain 100,000 accepted replicates within 1,000,000 attempts is
`NO_GO_MEASUREMENT_SUPPORT`.

For both choice families and combined/White, compute active-minus-control
fixed-slot usable-error discovery and fixed-slot usable mean regret. Sort by
`f64::total_cmp`; `p10` is index `9,999`. Record accepted/discarded counts,
cluster membership, stream hashes, and quantile-input hashes.

The cluster RNG hash appends every raw `next_u64()` output, including bounded
rejections and discarded-replicate draws, as `u64_le`. A cluster-index hash
appends each returned bounded index as `u64_le` in
`attempt -> draw-within-attempt` order. For each metric, pre-sort and sorted
quantile-input hashes use the same `f64::to_bits()` little-endian encoding as
the finite-support calculation and include accepted replicates only.

## Registered decision

A schema, seal, hash, replay, legality, payload, source, executable, finite
value, actual-role, join-identity, or independent-replay mismatch is
`INVALID_CB_AL1_P0`. Invalid runs write no stage output and exit 1.

Valid P0A statuses are `P0A_READY_FOR_REVEAL` and
`NO_GO_SELECTOR_SUPPORT`; both write one create-new P0A output and exit 0.
P0B is forbidden after the latter.

Valid P0B returns `GO_FRESH_AL1_PREREG_ONLY` only if all of these pass for both
`static_top` and archived actual:

1. complete-pair, per-family/color coverage, and cluster support;
2. fixed-slot error-count gates combined, White, and Black;
3. measurable-only conditional error gates combined, White, and Black;
4. fixed-slot and conditional regret direction gates;
5. finite-support random-control `p10 > 0` for error discovery and mean regret,
   combined and White;
6. dependence-cluster `p10 > 0` for error discovery and mean regret, combined
   and White;
7. all cancellation, correctness, provenance, and finality checks.

A valid coverage/bootstrap-support failure is
`NO_GO_MEASUREMENT_SUPPORT`. Any other valid gate failure is
`NO_GO_SELECTOR_UPPER_BOUND`. Valid P0B statuses write one create-new output
and exit 0.

`GO_FRESH_AL1_PREREG_ONLY` authorizes no collection by itself. It permits a new
independently audited preregistration that freezes, before any new row:

- a new opening- and D4+side-disjoint paired-color pool whose selector cannot
  access teacher values, roles, outcomes, or answer-derived ranking signals;
- a query-budget-matched active/random comparison and fixed stronger teacher;
- equal initialization, broad/listwise anchor, optimizer, updates, PTQ lattice,
  and CBF packaging;
- fresh opening-group-disjoint decision validation;
- 100k make/undo/full-rebuild and evaluator identity;
- equal-artifact evaluator-cost A/B;
- a causal active-versus-random arena before product-versus-incumbent arena.

No RQ515 pair-only loss, selfplay pseudo-label, QAT retry, graph vocabulary,
scale/architecture change, threshold sweep, or result-based rescue is allowed.
Every P0 outcome leaves product assets, defaults, search, White-root ordering,
ordinary library API, and pbrain behavior unchanged. The audit-only feature is
default-off and is not used by product builds.

## Exact stage commands and output finality

The exact P0A command, from the registered working directory with registered
`RUSTFLAGS`, is:

```text
.\target\release\cb-al1-selector.exe p0a --prepared-units C:\Users\concreate\Documents\workspace\noru-tactic\experiments\2026-07-11\rq615c_prepared_units_1000.jsonl --phase2-manifest C:\Users\concreate\Documents\workspace\noru-tactic\experiments\2026-07-11\rq615c_phase2_prepared_manifest.json --product-model C:\Users\concreate\.codex\worktrees\06f2\noru-tactic\target\figrid-release-0.8.2\models\gomoku_codebook_v1_swapclosed.json --product-cbf C:\Users\concreate\.codex\worktrees\06f2\noru-tactic\target\figrid-release-0.8.2\models\gomoku_codebook_v1_swapclosed_factored.cbf --topk C:\Users\concreate\.codex\worktrees\06f2\noru-tactic\target\figrid-release-0.8.2\data\topk.bin --out-selector C:\Users\concreate\.codex\worktrees\06f2\noru-tactic\target\figrid-release-0.8.2-artifacts\2026-07-26\cb-al1-p0a\cb_al1_p0a_selector.json
```

If and only if its status is `P0A_READY_FOR_REVEAL`, let `P0A_BYTES` and
`P0A_SHA256` be the exact decimal length and uppercase digest printed by P0A
after its flush/sync/re-read. Copy those literals without transformation into:

```text
.\target\release\cb-al1-selector.exe p0b --selector C:\Users\concreate\.codex\worktrees\06f2\noru-tactic\target\figrid-release-0.8.2-artifacts\2026-07-26\cb-al1-p0a\cb_al1_p0a_selector.json --expected-p0a-bytes <P0A_BYTES> --expected-p0a-sha256 <P0A_SHA256> --prepared-units C:\Users\concreate\Documents\workspace\noru-tactic\experiments\2026-07-11\rq615c_prepared_units_1000.jsonl --phase2-manifest C:\Users\concreate\Documents\workspace\noru-tactic\experiments\2026-07-11\rq615c_phase2_prepared_manifest.json --product-model C:\Users\concreate\.codex\worktrees\06f2\noru-tactic\target\figrid-release-0.8.2\models\gomoku_codebook_v1_swapclosed.json --product-cbf C:\Users\concreate\.codex\worktrees\06f2\noru-tactic\target\figrid-release-0.8.2\models\gomoku_codebook_v1_swapclosed_factored.cbf --topk C:\Users\concreate\.codex\worktrees\06f2\noru-tactic\target\figrid-release-0.8.2\data\topk.bin --train C:\Users\concreate\Documents\workspace\noru-tactic\experiments\2026-07-11\rq615c_k6_train.jsonl --final-manifest C:\Users\concreate\Documents\workspace\noru-tactic\experiments\2026-07-11\rq615c_final_corpus_manifest.json --lineage-model C:\Users\concreate\Documents\workspace\noru-tactic\experiments\2026-07-08\rq569_codebook_full_matefirst_ep3_model_swapclosed.json --out-reveal C:\Users\concreate\.codex\worktrees\06f2\noru-tactic\target\figrid-release-0.8.2-artifacts\2026-07-26\cb-al1-p0b\cb_al1_p0b_reveal.json
```

Each parent directory must exist and the target must not exist before its sole
authoritative run. The program performs all analysis before opening the output,
then uses create-new, writes pretty JSON plus LF, flushes, `sync_all`, seals,
and re-reads. A partial output from an I/O failure is removed before exit 1.

No rebuild is allowed between P0A and P0B. No overwrite, retry, alternate seed,
expanded support, relaxed quiet filter, changed selector, changed threshold, or
same-card rescue is allowed after a stage output becomes visible.
