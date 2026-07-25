# CB-VOC1: decision-weighted color-orbit vocabulary preregistration

Date: 2026-07-25 KST

Status: **preregistered before the production census implementation and before
any vocabulary, model, training, arena, or product-default change**.

## Question and claim boundary

The deployed Pattern4 table spends 4,265 non-RARE rows on the original
frequency-ordered 4,096 patterns plus 169 color-swap closure partners. CB-VOC1
asks whether the same row budget can address materially more of the deployed
codebook's K=6 decision-loss gradient by selecting complete mine/opp color
orbits instead of selecting by occurrence frequency alone.

This card is an **upper-bound and robustness census**, not a model candidate.
It changes no runtime default, does not tie color-partner embeddings, and does
not claim that first-order gradient coverage is game strength. Only a
precondition GO may open a separate paired-retraining card. A STOP ends
CB-VOC1 without consuming dev, safety, the 64-game search holdout, or an arena.

The RQ615C K=6 candidate slates were constructed with an older high-precision
`E32/H2048/F2048` base. They remain valid outcome-free strong-teacher labels,
but they are a local six-move mechanism corpus rather than a fresh strength
set or a full-legal product-policy sample. CB-VOC1 therefore recomputes every
candidate with the 0.8.2 product lattice `E32/H64/F64`; stored RQ615C base
logits are lineage diagnostics only and are never the selector score.

## Frozen baseline and inputs

Repository:

- working branch: `codex/cb-token-delta`;
- preregistration parent: `fc974da15b1eabedc8f52d73588675ec0f0e7826`;
- product lineage: figrid-board 0.8.2 plus promoted CB-D1 and CB-TD1;
- CB-F1 direct factored runtime remains default OFF and is not an arm;
- rules: 15x15 Freestyle;
- canonical build flags: `RUSTFLAGS=-C target-cpu=x86-64-v3`.

Exact computational inputs:

| role | bytes | SHA-256 |
|---|---:|---|
| current product f32 codebook JSON, `models/gomoku_codebook_v1_swapclosed.json` | 1,410,562 | `42968FDAB01BA8CCD1DE3DED05C532E4B237DD47EEFFD7AE1C2F264D77BA7DA2` |
| current vocabulary, `data/topk.bin` | 17,060 | `103891DCD1DCD978C654593ABE78EF32C56E2E350B500EE665BC45AC051AA16D` |
| RQ615C train, original workspace `experiments/2026-07-11/rq615c_k6_train.jsonl` | 54,991,200 | `E00A2DA513B05D7631A01003C7E6274E9A3D7575E2C2BD92D5199F1B5385CEB6` |
| RQ615C corpus manifest, original workspace `experiments/2026-07-11/rq615c_final_corpus_manifest.json` | 5,463 | `579D1387D7E4DE8F5CB34DB168B6D15655DB229D992751B1DC17BB6CF4260AA7` |
| RQ569 high-precision lineage JSON, original workspace `experiments/2026-07-08/rq569_codebook_full_matefirst_ep3_model_swapclosed.json` | 1,413,542 | `69BB7C599ADA3A1151577CE3315356BC33C40EDB49A003C9BC4EB90A98F82E18` |

The manifest must close as `READY_FOR_RQ615D`. Train must contain exactly
1,336 slates, 668 paired-color units, 388 unsplit components, 668 Black and
668 White rows, six ordered candidates per row, normalized positive
`q_teacher`, and the registered ordinal set `{1,2,4,6,8}`. The current product
JSON and RQ569 JSON must have bit-identical f32 embedding/head/factor/bias
payloads even though their file identities differ.

Only RQ615C **train** row content is authorized for scoring and selection.
The dev file, pre-accessed `safety_internal`, RQ508, the frozen 64-game search
holdout, and all game outcomes are forbidden inputs. Reading their already
published identities or aggregate manifest counts is not row consumption.

## Vocabulary geometry and fixed capacity

A token is a realizable 11-cell packed window after left/right reflection
canonicalization. Let `sigma(t)` swap mine and opponent cells and
canonicalize again. The implementation must prove `sigma(sigma(t))=t`.

The universe is all 199,827 canonical windows accepted by the released
`pattern_table::is_realizable` / `enumerate_patterns` semantics, including its
537 zero-support anchor-boundary forms, not only patterns already in the table
or observed in RQ615C. Unobserved rows have zero score. The universe is
partitioned into:

- fixed color orbits `{t}` with cost 1;
- paired color orbits `{t,sigma(t)}` with cost 2.

The selected non-RARE capacity is exactly

`sum_orbit selected(o) * |o| = 4,265 rows`.

Let `mu0(t)` be the incumbent lookup from raw canonical packed token to its
current ID `0..4264`, or to RARE ID 4265 when absent. Raw token identity `t`
and mapped model row `mu0(t)` are distinct objects throughout this card.

The number of selected orbits is not fixed. The incumbent happens to contain
29 fixed orbits and 2,118 pairs, or 2,147 orbits and 4,265 rows. RARE is a
separate shared fallback row and is never selectable.

The 169 closure-tail rows at incumbent IDs `4096..4264` remain real,
independently addressable rows even where their current embedding equals the
RARE embedding. They are not deduplicated, made free, or counted as RARE.
Color partners remain independent parameters; CB-VOC1 must never score a pair
as a tied weight.

## Authoritative product evaluator

The current JSON is quantized with the released Rust rules and constants:

- embedding `Qe = round_f32(e * 32)`, stored i16;
- head `Qh = round_f32(h * 64)`, stored i16;
- factor `Qv = round_f32(v * 64)`, stored i16;
- bias retains its original f32 bits.

For every RQ615C candidate, reconstruct the child board from parent history
plus candidate move. Let `ell_pi` be the released evaluator's child
side-to-move value. Root-mover utility is

`u_pi = -ell_pi`.

Equivalently, for the stored Black-value coordinate `z_pi`,
`u_pi=s_p*z_pi`, where `s_p=+1` for a Black root and `-1` for a White root.
The selector uses the natural child-to-move coordinate because that is the
deployed evaluator's native perspective.

The census implementation must independently rebuild the product forward
pass and match `evaluate_full_quantized(...).to_bits()` for all
`1,336 * 6 = 8,016` children. The independent pre-cast f64 result and final
f32 bits are both recorded. Separately, the old `E32/H2048/F2048` forward pass
must replay stored RQ615C base bits as a lineage audit. Differences between
the two lattices are reported but cannot affect the selector.

Raw identity extraction must not use `line_pattern_ids`, because an unseen
pattern has already collapsed to RARE there. For every cell/direction, read
the black-relative raw 22-bit window from board occupancy. If child
side-to-move is White, swap raw mine/opp symbols first; then apply left/right
reflection canonicalization and pack the result as the natural-perspective
raw token `tau`. Only after preserving that identity may the evaluator apply
`mu0(tau)` to obtain the current embedding row.

## Exact first-order decision score

For slate `p`, candidate `i`, stored teacher target `q_pi`, and product
distribution

`pi_pi = softmax(u_p)_i`,

the K=6 cross-entropy coefficient in the natural child score is

`alpha_pi = dCE_p/dell_pi = q_pi - pi_pi`.

For child `x`, cell `c`, dimension `k`, and natural-perspective raw token
`tau(x,c,d)`, define the integer cell preactivation

`a_xck = sum_(d=0..3) Qe[mu0(tau(x,c,d)),k]`.

Each of the nine 5x5 regions has 25 cells. The normalized feature is

`X_xrk = sum_(c in region r) max(a_xck,0) / (32*25)`.

With `h_j=Qh_j/64`, `v_jl=Qv_jl/64`, and
`S_l=sum_j v_jl*X_j`, the exact derivative of the product pre-cast score is

`beta_j = dell/dX_j = h_j + sum_l v_jl*(S_l-v_jl*X_j)`.

If raw token `t` occurs `m_xct` times among the four directions at cell `c`,
the one-integer-bin embedding sensitivity is

`A_xtk = dell_x/dQe_new(t),k`

`      = sum_c m_xct * 1[a_xck>0] * beta_(region(c),k)/(32*25)`.

Here `t` remains the raw identity even when `mu0(t)=RARE`; `A_xtk` is the
gradient available if that raw pattern is split into its own row at the
current base point. The base activation itself always uses the incumbent
mapped row `Qe[mu0(t)]`.

The frozen ReLU subgradient is the existing trainer convention
`ReLU'(0)=0`. A slate row-gradient is

`d_ptk = sum_(i=0..5) alpha_pi*A_pitk`.

Accumulation order is physical train row, candidate index `0..5`, cell
`0..224`, direction `0..3`, dimension `0..15`. All analytic accumulation uses
f64 Neumaier sums. Softmax uses the released f32 child value promoted to f64;
the final f32 cast is treated with a straight-through derivative and this
claim boundary is explicit.

For a slate set `A`,

`G_A,t = mean_(p in A) d_pt` and `v_A(t)=||G_A,t||_2^2`.

A pair orbit's value is `v(t)+v(sigma(t))`, never
`||G_t+G_sigma(t)||^2`, because its two rows are not tied. Shared RARE-row
updates are common to both arms and excluded from selector value.

Raw occurrence residual

`delta_n_pt = sum_i alpha_pi*n_pit`

is emitted only as a diagnostic. It may not replace the full
ReLU/head/FM derivative above.

## Exact 4,265-row optimization

Sort fixed and paired orbits independently by:

1. value descending using total f64 order;
2. incumbent rows retained descending;
3. canonical packed tuple ascending.

Build fixed and pair prefix sums. Enumerate every feasible fixed count `f`
such that `0<=f<=N_fixed`, `f<=4265`, and
`p=(4265-f)/2` is an integer with `p<=N_pair`. Choose the maximum
`F[f]+P[p]`.

Every row value and orbit value must be finite. Prefix sums use Neumaier
accumulation in the frozen sorted orbit order. `Phi(V)` uses Neumaier
accumulation over selected raw packed rows in ascending numeric order, and
`Phi(V0)` must be finite and strictly positive.

An exact objective tie is resolved by:

1. maximum incumbent rows retained;
2. minimum row symmetric difference from incumbent;
3. lexicographically smallest sorted packed-row list.

This exhaustive fixed-count enumeration is the exact cost-{1,2} knapsack,
not a greedy approximation. Incumbent inclusion in the universe guarantees a
feasible solution.

Define

`Phi(V)=sum_(t in V) v_train(t)`.

The full-train selector upper-bound gain is

`R_phi = (Phi(V*)-Phi(V0))/Phi(V0)`.

## Sequential census

### Stage A0: integrity and geometry

Before a score is available:

- rehash every input before and after the run;
- validate every row, candidate, q vector, move, side, unit, component, and
  manifest count;
- validate all 199,827 realizable patterns, orbit partition, involution,
  incumbent closure, 29/2,118 incumbent orbit census, 169-row tail, and exact
  capacity;
- require every gradient, row value, orbit value, prefix, `Phi`, CE,
  addressability statistic, and bootstrap statistic to be finite, with
  `Phi(V0)>0`;
- require zero released-vs-independent product score mismatches over 8,016
  children;
- require zero old-lattice lineage replay mismatches;
- prove forbidden row inputs were not opened.

Any failure is `INVALID_CB_VOC1`.

### Stage A1: point upper bound

Compute the exact full-train gradient vocabulary and these point metrics:

- `R_phi`;
- raw-slot addressability `M_s(V)`, the fraction of the
  `6*225*4` natural token slots per slate whose raw pattern is in `V`;
- combined, Black, White, and ordinal addressability deltas;
- gross lost incumbent-addressed mass;
- gained/lost rows and orbits;
- support buckets by full-train raw occurrence:
  `[1]`, `[2..7]`, `[8..31]`, `[32..127]`, `[128+]`.

An orbit is protected by the support-128 gate when
`max(n_t,n_sigma(t))>=128`; partner counts are not pooled to manufacture
protection. In every combined/color/ordinal stratum, gross loss is

`100 * lost incumbent-addressed slots / all slots in that stratum`

in percentage points.

Stage A1 opens robustness work only if all are true:

- `R_phi >= 0.03`;
- combined addressability gain `>= +1.00 percentage point`;
- Black and White point gains each `>= +0.75 percentage point`;
- every ordinal point gain `>= 0`;
- combined gross lost incumbent-addressed mass `<=0.25 percentage point`;
- color and ordinal gross loss each `<=0.50 percentage point`;
- zero removed incumbent orbit with support `>=128`.

Failure stops immediately as `NO_GO_PRECONDITION`; later stages, retraining,
artifact building, search benchmarks, and strength games are skipped.

### Stage A2: component-held-out robustness

If and only if A1 passes, assign all 388 components to five folds without
splitting a component. Sort components by descending slate count, then
`SHA256("CB-VOC1|fold-v1|"+uppercase_component_uid)`, then UID. Assign each
component to the fold with the smallest current slate count, breaking ties by
the smaller fold index.

For fold `f`, select `V_-f` and descent direction `-G_-f` using only the other
four folds. For held-out component `C` define

`D_C,t = sum_(p in C) d_pt`

and candidate-vs-incumbent held-out first-order gain

`Y_C = sum_t (1[t in V_-f]-1[t in V0]) * dot(D_C,t,G_-f,t)`.

Positive `Y` means the candidate addresses more of a training-excluded
descent direction. Color and ordinal diagnostics restrict `D_C` to those
held-out rows but retain the combined training direction; no post-hoc
color-specific selector is allowed.

Run 100,000 component-cluster bootstrap replicates with SplitMix64 seed
`0xCB01202607260001`. Advance state by wrapping addition of
`0x9E3779B97F4A7C15`; mix with xor-shift 30 and multiplication by
`0xBF58476D1CE4E5B9`, xor-shift 27 and multiplication by
`0x94D049BB133111EB`, then xor-shift 31, all wrapping u64. Iterate folds
`0..4`; within a fold sort uppercase component UIDs by raw ASCII bytes. Draw
index is `next_u64 % component_count` with no rejection sampling. Within each
fold draw exactly that fold's original component count with replacement.
Preserve whole components and use one continuous stream across folds and
replicates. Statistics are row-weighted ratios with Neumaier sums.
Nearest-rank p05/p95 uses zero-based index `ceil(q*N)-1`. Selection is frozen
outside the bootstrap; bootstrap re-selection cannot rescue a gate.

Robustness requires:

- combined OOF gain point `>0` and p05 `>0`;
- White OOF gain point `>0` and p05 `>0`;
- Black OOF gain point `>=0` and p05 `>=0`;
- combined and White fold point gain positive in at least four of five folds;
- combined OOF addressability gain point `>=+1.00pp`, p05 `>=+0.75pp`;
- Black and White OOF addressability point gains each `>=+0.75pp`;
- every ordinal OOF addressability point gain `>=0`;
- mean pairwise selected-row Jaccard across the five folds `>=0.98`;
- at least 80% of newly selected full-train rows appear in at least four fold
  vocabularies.

### Stage A3: zero-fit remap and lattice-boundary audit

If and only if A2 passes, evaluate a zero-fit semantic remap. For each
held-out fold `f`, the gating remap uses only `V_-f`:

- rows in `V0 intersect V_-f` keep their exact current quantized embedding;
- rows in `V_-f \ V0` copy the current RARE embedding;
- rows in `V0 \ V_-f` map to RARE;
- head, factors, bias, product scales, and all search code stay fixed.

The full-train `V*` remap is report-only and cannot gate.

Recompute full FM/ReLU K=6 CE on held-out folds. Cluster-bootstrap
`CE_remap-CE_base` with the A2 stream contract and a fresh seed
`0xCB01202607260002`. Require p95:

- combined `<=+0.0002`;
- Black and White each `<=+0.0003`;
- every ordinal point delta `<=+0.0005`.

Audit every exact-zero and one-bin-crossable ReLU preactivation touched by a
row in `V0 union V*`. Recompute the selector with the exact symmetric
one-integer-bin cell activation slope

`[ReLU(a+m)-ReLU(a-m)]/2`

for multiplicity `m`.

The finite-difference witness population is every distinct
`(uppercase row_uid, raw_packed_t, dimension)` for which `t` occurs at least
once in any of that slate's six children, `t` is in `V0 union V*`, and
`dimension` is `0..15`. Both `Qe[mu0(t),k]-1` and `+1` must fit i16 or the
card is invalid. Rank each tuple by the raw SHA-256 digest of

`"CB-VOC1|fd-witness-v1|" + UID + "|" + uppercase_hex8(t) + "|" + decimal(k)`

then UID ASCII bytes, numeric packed value, and dimension; take exactly the
first 10,000 distinct tuples. Fewer than 10,000 is invalid. For each witness,
perturb only occurrences of raw identity `t` throughout that one K=6 slate.
The cached perturbation path must match a from-scratch product forward for all
12 plus/minus child f32 logit bits and the two f64 CEs within absolute
`1e-12`. These exact loss differences are an audit; they do not replace the
frozen first-order selector. The symmetric-slope audit must build the
corresponding training-excluded `V_sym,-f` for every fold and rerun the same
A2/A3 calculations; it may not substitute a full-train symmetric vocabulary.
Require:

- zero parity or arithmetic witness failures;
- selected-row Jaccard versus the frozen zero-subgradient selector `>=0.99`;
- identical pass/fail direction for every A1-A3 gate.

Failure in A2 or A3 is `NO_GO_PRECONDITION`.

## Decision and next-card boundary

Only passage of A0, A1, A2, and A3 is `GO_PROTOTYPE`. It authorizes one later
vocabulary-only prototype card; it does not authorize promotion.

That later card must, before training:

- materialize a deterministic candidate table and bind its SHA in the model
  artifact;
- keep both arms on the flat quantized runtime because CB-F1 direct factoring
  was not promoted;
- use the same 6,163,315-label mate-first source lineage derived from
  `rapfi_avx512vnni_300k_1s.jsonl` (1,187,813,619 bytes,
  SHA-256
  `F2BECA8104E902EE700A22ED13B722EF09AFBF83CFB2E0CF3E801680646CC481`),
  initialization, seed, row order, optimizer, fixed epoch, and product
  quantization for both arms;
- forbid validation-selected epochs;
- freeze new component-separated K=6 development and safety evidence before
  seeing either trained arm;
- require quantized artifact reload, color-swap closure, full rebuild and
  100k make/undo equality, same-binary ABBA performance, and only then paired
  strength gates.

RQ615C dev/internal replication and the current 64-game holdout remain
ineligible for promotion. RQ508 may be used only as its already-consumed
generic non-inferiority guard.

Final labels for this card are exactly:

- `INVALID_CB_VOC1`;
- `NO_GO_PRECONDITION`;
- `GO_PROTOTYPE`.
