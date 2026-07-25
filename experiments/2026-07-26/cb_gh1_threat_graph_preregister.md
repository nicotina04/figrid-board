# CB-GH1 P0: rooted threat-interaction graph signal preregistration

Date: 2026-07-26 KST

Status: **preregistered before the graph implementation and before any graph
identity, collision, support, residual, cross-entropy, or best-move result is
observed**.

## Question and claim boundary

CB-GH1 asks whether a small relational code can explain decision residual that
the deployed flat Pattern4 codebook misses. The fixed P0 representation is a
role-relative, rooted one-ply transition factor graph of:

- tactical empty cells;
- mover and opponent stones incident to those tactical factors;
- board-boundary cells incident to those factors;
- directional `WindowThreat` factors; and
- a rooted marker for the candidate move.

P0 is a **train-only, static-full-rebuild, optimistic signal census**. It does
not implement message passing, graph training, graph pruning, an incremental
runtime, a search consumer, a model artifact, an arena, or a product-default
change. RQ615C train was already consumed by CB-VOC1, so even a passing result
is exploratory precondition evidence, not fresh validation or strength
evidence.

The sequential claim is deliberately narrow:

1. the fixed graph must be exact under its stated abstraction and equivariant
   under D4 and color-role swaps;
2. exact rooted transition identities must repeat across independent opening
   components often enough to be codebook-like;
3. the fixed graph partition must explain at least 3% of the ideal logit
   residual and reduce excess decision loss by at least 3% in-sample; and
4. that correction must survive leave-one-component-out prediction.

Only all four conditions may open a separate incremental-correctness and cost
card. Passage does not open graph training or runtime promotion directly.

## Frozen repository and product baseline

- repository branch: `codex/cb-token-delta`;
- preregistration parent: `9e5592f`;
- product lineage: figrid-board 0.8.2 with CB-D1 and CB-TD1 promoted;
- CB-F1, CB-GH0, and every new graph path remain default OFF;
- board/rule: 15x15 Freestyle;
- product evaluator: embedded quantized Pattern4 codebook with White-root
  ordering `auto`;
- canonical build flags: `RUSTFLAGS=-C target-cpu=x86-64-v3`.

CB-GH0's 64-bit D4 sidecar was exact but failed its registered transition-cost
gate. P0 may reuse its public offline D4 tables and exact-state serializer, but
must not enable the sidecar in product search or claim that GH1 repays GH0's
cost.

## Frozen computational inputs

| role | bytes | SHA-256 |
|---|---:|---|
| current product f32 codebook JSON, `models/gomoku_codebook_v1_swapclosed.json` | 1,410,562 | `42968FDAB01BA8CCD1DE3DED05C532E4B237DD47EEFFD7AE1C2F264D77BA7DA2` |
| current Pattern4 vocabulary, `data/topk.bin` | 17,060 | `103891DCD1DCD978C654593ABE78EF32C56E2E350B500EE665BC45AC051AA16D` |
| RQ615C train, original workspace `experiments/2026-07-11/rq615c_k6_train.jsonl` | 54,991,200 | `E00A2DA513B05D7631A01003C7E6274E9A3D7575E2C2BD92D5199F1B5385CEB6` |
| RQ615C final corpus manifest, original workspace `experiments/2026-07-11/rq615c_final_corpus_manifest.json` | 5,463 | `579D1387D7E4DE8F5CB34DB168B6D15655DB229D992751B1DC17BB6CF4260AA7` |
| RQ569 high-precision lineage JSON, original workspace `experiments/2026-07-08/rq569_codebook_full_matefirst_ep3_model_swapclosed.json` | 1,413,542 | `69BB7C599ADA3A1151577CE3315356BC33C40EDB49A003C9BC4EB90A98F82E18` |

The manifest must close as `READY_FOR_RQ615D`. Train must contain exactly:

- 1,336 K=6 slates;
- 668 paired-color units;
- 388 unsplit `component_uid` groups;
- 668 Black-root and 668 White-root rows;
- ordinals `{1,2,4,6,8}`;
- one unique `teacher_top` and one unique `deployed_actual` role per slate;
- two valid teacher-score repeats; and
- strictly positive, normalized `q_teacher` replaying those two repeats.

Every history, legal inventory entry, K=6 move, stored child identity, and
stored high-precision f32 lineage value is revalidated. The released evaluator
and an independent integer forward pass must agree bit-for-bit on all 8,016
children. The old high-precision lattice must replay all stored lineage bits.

Only the RQ615C **train** rows are authorized. RQ615C dev,
`safety_internal`, RQ508, game outcomes, the frozen 64-game search holdout,
and the frozen 1,022-root timing trace are forbidden row inputs. The analyzer
must expose no CLI option for them. Published aggregate identities and counts
are not row consumption.

## Fixed semantic flags and viewpoint

The graph uses released product threat semantics, not the rejected RQ598
full-experimental vocabulary:

```text
QuietThreatConfig {
    min_gain: 1,
    enable_jump_three: false,
    enable_gap_four: false,
}
```

`min_gain` does not affect single-cell classification. `JumpThree` and
experimental gap-four promotion are fixed OFF because product VCT and search
default them OFF.

For each parent, `mover` is the parent's side to move and `opponent` is the
other side. That role assignment remains fixed after a K=6 move is made, even
though the child board's side to move is then `opponent`. Absolute Black/White
identity is excluded from graph bytes.

For rooted move `m`, enumerate the exact affected `(source,direction)` sites:
for each of the four frozen axes and offset `-5..=+5`, include the in-bounds
source `m + offset*axis`. This is the same at-most-44 cell-direction frontier
whose 11-cell Pattern4 window contains `m`; it is a stricter site-preserving
form of `Board::line_pattern_dirty_cells(m)`.

For each affected site and each role, call the released
`classify_move_with_directions` semantic reference separately on the parent
whenever the source is empty there and on the child whenever it is empty
there, retaining only that site's enumerated direction. An occupied source has
directional kind `None` for that state. A directional transition factor exists
exactly when its parent or child `WindowThreat` is not `None`. Eligible
non-`None` kinds are:

```text
OpenTwo, ClosedThree, OpenThree, ClosedFour, OpenFour, Five
```

No threshold, top-N, radius, learned selector, teacher score, product score, or
candidate role may affect graph construction.

Unchanged factors elsewhere on the board are deliberately excluded. This is a
rooted local-delta signature, not a whole-board tactical fingerprint. It keeps
P0 aligned with the bounded CB-TD1 update frontier and prevents an unrelated
distant threat from destroying otherwise reusable local interaction identity.

## Fixed graph

### Cell and boundary nodes

Each factor uses the RQ598 structural footprint: offsets `-4..=+4` from its
source along its axis.

The rooted move is the serialization origin. Create one board-cell node for
every in-bounds coordinate incident to at least one parent/child-union factor.
Its label is:

```text
(root-relative coordinate,
 parent_occupancy={empty,mover,opponent},
 child_occupancy={empty,mover,opponent},
 rooted_marker={false,true})
```

All factor-incident empty cells are retained; this is not a stones-only graph.
The rooted candidate cell is retained even when it is incident to no factor.
An occupied stone outside every factor footprint is deliberately omitted
unless it is the rooted marker. The parent/child occupancy pair makes the
one-ply change explicit rather than assuming that a downstream consumer will
reconstruct the parent from the marker. This omission of unrelated quiet
stones is the fixed lossy abstraction.

For an out-of-bounds footprint coordinate, create a virtual boundary node at
that root-relative signed coordinate. Boundary nodes with the same relative
coordinate are shared. Because affected sources are at most five cells from
the root and footprints extend four more cells, both coordinate components
are bounded by `-9..9`.

### Directional factor nodes and typed incidence

Each directional transition factor is labelled by:

```text
(owner_role,
 parent_WindowThreat_or_None,
 child_WindowThreat_or_None,
 root-relative_source_coordinate,
 undirected_axis)
```

The axis is normalized by negating it when its first non-zero
`(delta_row,delta_col)` component is negative.

For every offset `-4..=+4`, add one incidence edge from the factor to the board
or boundary node at that coordinate. Its fixed label is:

```text
(absolute_distance, parent_relation, child_relation)
```

where each state relation is exactly one of:

- `anchor` for offset zero when the source is empty in that state;
- `support` for an in-bounds stone owned by the factor role;
- `blocker` for an in-bounds stone owned by the other role;
- `footprint_empty` for an in-bounds empty; or
- `boundary` for an out-of-bounds coordinate.

The factor's parent/child kind pair records `before_only`, `after_only`, and
`both` status without a redundant tuned tag. Shared cell nodes encode
footprint overlap and support reuse. No explicit factor-factor edge,
message-passing layer, causal-response edge, or RQ603 defense reduction is
added. This avoids rescuing the rejected RQ598/RQ603 families after seeing
their labels.

### Rooted-transition population

Build one jointly rooted parent-to-child transition graph for each of the
8,016 K=6 candidates. The 1,336 parents remain independently reconstructed,
sealed, and state-hash audited, but there is no unrooted graph-code
population: without a candidate origin it would answer a different question.

The transition graph is built from the parent, the legal child, and a marker
on the new mover stone. Its parent, child, and marker must share one D4
transform. Independently canonicalized parent and child byte strings must
never be concatenated. The decision code used by the residual projection is
the exact rooted transition identity. It is never derived by applying a
single canonical parent transform to the move.

The joint before/after lanes and rooted marker make this a
**candidate-conditioned ranking code**, not a position-value code. Neither a
parent value cache nor two equal child occupancies reached through different
rooted actions may silently erase that conditioning.

## Exact serialization and D4 identity

The byte format is versioned as `CB-GH1-GRAPH-V1`. It contains:

1. a fixed `rooted_transition` tag;
2. sorted cell/boundary records;
3. sorted factor records; and
4. sorted incidence records referring to the sorted node/factor ordinals.

Counts use fixed-width little-endian integers. Root-relative signed
coordinates are encoded as their explicitly specified two's-complement i8
byte (`value mod 256`). Enum tags and record widths are fixed in code and
covered by golden tests. Serialization must be prefix-unambiguous; debug text
and platform layout are forbidden.

For each of the eight frozen `D4_MAP` transforms:

1. apply that same transform to parent, child, rooted marker, in-bounds
   coordinates, and extended coordinates, then subtract the transformed
   marker from every serialized coordinate;
2. transform and renormalize factor axes;
3. rebuild sorted ordinals; and
4. serialize.

After transforming both the absolute coordinate and rooted origin, the frozen
formulas on relative displacement `(r,c)` are:

```text
T0(r,c) = (r,    c)
T1(r,c) = (c,   -r)
T2(r,c) = (-r,  -c)
T3(r,c) = (-c,   r)
T4(r,c) = (r,   -c)
T5(r,c) = (-r,   c)
T6(r,c) = (c,    r)
T7(r,c) = (-c,  -r)
```

For every in-board cell/root pair, decode `D4_MAP[t][cell]` and
`D4_MAP[t][root]` separately into `(row,column)`, subtract the two components,
and require agreement with these formulas. Raw row-major u8 indices are never
subtracted. The same formulas are golden-tested directly over every virtual
relative coordinate in `[-9,9]^2`.

The exact graph identity is the lexicographically least byte string of those
eight serializations. Preserve an 8-bit mask of **all** transforms attaining
the minimum. The implementation must not choose the transform with the
minimum 64-bit Zobrist fingerprint, and it must not use only the lowest
minimum-transform index to map candidate moves through an automorphism.

The implementation must also reconstruct both geometrically transformed
boards and the transformed rooted marker, rerun parent/child directional
classification there, and match the coordinate-transformed serialization.
This catches direction, reflection, endpoint-parity, and independent-lane
canonicalization errors rather than assuming them away.

The graph digest is:

```text
SHA256("CB-GH1-GRAPH-V1\0" || exact_graph_bytes)
```

The prospective lightweight key is the little-endian u64 formed from digest
bytes `0..8`. Exact graph bytes remain the authority.

## Exact role-relative board identity and collision classes

For collision accounting, separately serialize the complete parent/child
transition under the same eight **joint** geometric transforms using:

- parent mover/opponent bitboards;
- child mover/opponent bitboards;
- effective rule;
- rooted-transition tag; and
- rooted marker coordinate.

The lexicographically least joint serialization is the exact role-relative
transition identity. It includes all quiet stones omitted by the graph and
retains absolute board placement modulo D4; translation equivalence belongs
only to the rooted graph abstraction.

Collision categories are mutually exclusive:

1. for either parent or child, same production D4 u64 key but different
   production exact 66-byte board: true state-hash collision, invalid;
2. same graph SHA-256 or prospective graph u64 key but different exact graph
   bytes: graph-hash collision, invalid;
3. same exact graph bytes but different exact role-relative transition bytes:
   intentional abstraction collision, reported;
4. same exact role-relative transition but different absolute colors:
   color-role isomorphism, not an abstraction collision; and
5. same exact role-relative transition and rooted marker: duplicate.

Abstraction collisions are not themselves an error. In particular, the
fixed `-4..+4` incidence graph omits the outer `read_window` offsets at
`-5/+5`; the already computed `WindowThreat` kind is retained, but P0 makes no
proof, defense-relevance, or response-completeness claim from that footprint.
Collision label consistency is tested only by the fixed residual and decision
gates below.

## A0: integrity and equivariance

Before any support or label metric is emitted:

- seal every input before and after the run;
- reject an existing output path;
- record source HEAD, dirty status, executable bytes/SHA-256, canonical flags,
  CPU identity, environment controls, and wall-clock provenance;
- require all corpus, model, lineage, legality, and evaluator checks above;
- require deterministic byte-identical rebuilds;
- require zero production state-hash collision;
- require zero graph SHA-256 and graph-u64 collision;
- rebuild every joint parent/child/rooted-marker transition through every D4
  transform and require the exact canonical graph bytes/digest to agree;
- color-swap every parent/child pair and its role viewpoint and require exact
  graph bytes/digest to agree;
- require transformed minimum-mask composition to agree with the frozen
  `D4_COMPOSE` table;
- require every A0 structural count, byte length, parsed numeric input, model
  value, and evaluator output to be finite and in its registered domain.

Any failure is `INVALID_CB_GH1_P0`. No later metric may rescue it.

## A1: label-blind reuse and state-size census

A1 may use graph identities, corpus component/color/ordinal identities, and
structure counts, but not `q_teacher`, teacher scores/roles, or product logits.

A rooted candidate observation is **recurrent** when its exact transition
graph bytes occur in at least three distinct `component_uid` groups. This
guarantees at least two donor components remain when one observed component
is held out. A code seen in only one or two components is unsupported for
every gating estimator. Report:

- rooted-transition distinct-code and multiplicity distributions;
- exact duplicates, color-role isomorphisms, and abstraction collisions;
- node, factor, edge, byte-size, and minimum-transform-mask distributions;
- recurrent rooted-transition coverage by combined/color/ordinal strata;
- recurrent-code teacher mass later, as a label-bearing A2 diagnostic only.

The fixed A1 gates are:

- combined recurrent rooted-candidate fraction `>=25%`;
- Black-root recurrent fraction `>=15%`;
- White-root recurrent fraction `>=15%`; and
- every ordinal recurrent fraction `>=10%`.

Failure is `NO_GO_STATE_EXPLOSION`. A2/A3 are skipped. These gates prevent a
near-unique graph fingerprint from masquerading as a lightweight codebook.

## Frozen product and target coordinates

For slate `p` and candidate `i`, let `ell_pi` be the released evaluator's
natural child-side-to-move logit. Root-mover utility is:

```text
u_pi = -ell_pi
```

Let:

```text
t_pi = log(q_teacher_pi) - mean_j log(q_teacher_pj)
b_pi = u_pi             - mean_j u_pj
r_pi = t_pi - b_pi
```

The target logit vector `t_p` exactly reproduces `q_teacher` under softmax.
Both target and product logits are centered within the K=6 slate, so
unidentifiable slate-wide constants are removed. All logs, centering,
log-sum-exp, means, sums of squares, and aggregate metrics use f64 with
Neumaier accumulation in physical train-row/candidate order.

No game result, search holdout, candidate ranker, or post-hoc score transform
is used.

For each slate define target entropy and excess decision loss:

```text
H_p       = -sum_i q_pi * log(q_pi)
CE_p(v)   = -sum_i q_pi * log(softmax(v)_i)
KL_p(v)   = CE_p(v) - H_p
```

`KL_p` is evaluated directly as `sum_i q_pi*log(q_pi/softmax(v)_i)` with a
stable log-softmax path. A per-row value below `-1e-12` is invalid. Values in
`[-1e-12,0)` are retained without clamping in row and aggregate arithmetic.
Relative gain is measured against aggregate excess loss, not raw
cross-entropy with irreducible target entropy in its denominator. Equality,
tie sets, and gate comparisons use the computed f64 values without epsilon.

## A2: optimistic full-fit graph residual projection

For each recurrent exact rooted transition code `c`, fit the
observation-weighted residual mean:

```text
a_c = mean_(p,i : code(p,i)=c) r_pi
u'_pi = u_pi + a_code(p,i)
```

For a code observed in fewer than three components, `a_c=0` in every gating
calculation. An all-code full-fit projection may be emitted as an explicitly
non-gating memorization diagnostic, but it cannot supply any gate or alter
the recurrent projection.

This is the exact least-squares projection of the ideal centered-logit
residual onto the recurrent part of the fixed graph-code partition. It is an
optimistic in-sample fit, not a trained model and not the CE-optimal graph
head. P0 may call it an upper fit for this **fixed least-squares residual head
only**; it must not call it an upper bound on every possible graph model.

Compute:

- base and corrected K=6 cross-entropy and excess loss/KL;
- residual SSE before and after the bucket projection;
- q-top credit before and after correction;
- the same diagnostics by color and ordinal;
- teacher-probability mass on recurrent-code observations; and
- within-code residual dispersion split by duplicate versus abstraction
  collision groups.

Every A2 correction, entropy, CE, KL, SSE, credit, mass, and aggregate must be
finite; a violation is `INVALID_CB_GH1_P0`, not a signal failure.

The acceptable target set `T_p` is every index exactly equal to
`max(q_teacher)`. Target ties are valid and their cardinality is reported.
The stored unique `teacher_top` role is separately validated and its
membership or non-membership in `T_p` is reported, but it cannot replace or
invalidate the fixed probability-target truth. For the exact maximum-logit
prediction set `P_p`, q-top credit is:

```text
credit_p = |P_p intersect T_p| / |P_p|
```

Thus any single choice from a tied acceptable target set receives full credit,
while widening the predicted tie set with unacceptable moves is penalized.

Define:

```text
R_KL_full  = (KL_base - KL_full) / KL_base
R_SSE_full = (SSE_base - SSE_full) / SSE_base
DeltaQTop_full_pp = 100 * (Credit_full - Credit_base)
```

For any slate stratum `S`, recurrent teacher mass is:

```text
M(S) =
    sum_(p in S, i=0..5) q_pi * 1[code(p,i) is recurrent]
    / |S|
```

Each slate contributes total teacher mass one, so Black and White percentages
use their own stratum denominators rather than the full-corpus row count.

A2 opens A3 only if:

- aggregate `KL_base` and `SSE_base` are finite and strictly positive;
- `R_KL_full >= 0.03`;
- `R_SSE_full >= 0.03`;
- `DeltaQTop_full_pp >= +1.00`;
- combined recurrent-code teacher mass `>=25%`; and
- Black and White recurrent-code teacher mass each `>=15%`.

Failure is `STOP_NO_GRAPH_SIGNAL`; A3 is skipped. This is the registered 3%
signal gate for the frozen least-squares estimator. It is not claimed as the
theoretical ceiling of all graph models; failure closes this fixed GH1
representation/estimator pair, not graph learning in general.

## A3: leave-one-component-out residual projection

For every exact graph code and component `C`:

1. compute each other component's mean residual for that code;
2. average those component means with equal component weight; and
3. use zero correction unless at least two distinct donor components remain
   outside `C`.

Thus:

```text
a_(c,-C) = mean_(D != C, c observed in D) mean_(p,i in D, code=c) r_pi
u^loo_pi = u_pi + a_(code(p,i),-component(p))
```

This is equivalent to allowing corrections only for A1 recurrent codes when
evaluating a component in which the code occurs. There is no global mean,
smoothing, shrinkage, nearest-code backoff, or learned support threshold.

Both colors of a paired unit and every row from the same opening component
are excluded together. Candidate observations are never split into
independent training/test samples.

Compute combined, Black, White, and ordinal CE/KL/SSE/q-top-credit deltas. The
primary point metrics are:

```text
R_KL_loo
R_SSE_loo
DeltaQTop_loo_pp
```

For any combined, color, or ordinal stratum `S`, its reported and gated KL
gain is the stratum-specific relative ratio:

```text
R_KL_loo(S) =
    (sum_(p in S) KL_base_p - sum_(p in S) KL_loo_p)
    / sum_(p in S) KL_base_p
```

Every gating stratum denominator must be finite and strictly positive or the
card is invalid. Q-top credit is the arithmetic mean of per-slate fractional
credit within the stratum; its delta is reported in percentage points.
Every A3 correction, point statistic, bootstrap replicate statistic, and
quantile must be finite or the card is `INVALID_CB_GH1_P0`.

### Component-cluster bootstrap

Sort the 388 uppercase component UIDs by raw ASCII bytes. Run 100,000
replicates with SplitMix64 seed `0xCB01202607260011`:

- advance by wrapping addition of `0x9E3779B97F4A7C15`;
- xor-shift 30, multiply by `0xBF58476D1CE4E5B9`;
- xor-shift 27, multiply by `0x94D049BB133111EB`;
- xor-shift 31;
- draw `next_u64 % 388`, with no rejection sampling;
- draw exactly 388 whole components with replacement per replicate; and
- preserve one continuous stream across all replicates.

LOO corrections are frozen before bootstrap and are not refit or reselected
inside replicates. KL relative ratios are row-weighted ratios of resampled
component sums; every combined/color gating relative ratio uses that
stratum's resampled excess-loss/KL denominator exactly as above.
Q-top-credit deltas are row-weighted percentage-point differences.
Black/White use the same sampled component multiplicities.
Nearest-rank p05/p95 uses zero-based index `ceil(q*N)-1`.

A3 passes only if all are true:

- combined `R_KL_loo >=0.03`;
- combined `R_KL_loo` bootstrap p05 `>0`;
- combined `R_SSE_loo >=0`;
- `DeltaQTop_loo_pp >=+1.00`;
- q-top-credit delta bootstrap p05 `>=0`;
- Black and White KL gain points each `>=0`;
- Black and White KL-gain bootstrap p05 values each `>=0`;
- Black and White q-top-credit delta points each `>=0`; and
- every ordinal KL-gain and q-top-credit delta point is `>=0`.

These are fixed whole-corpus gates. No recurrent-only denominator may replace
them.

## Sequential decision

Decision precedence is:

1. any A0 failure: `INVALID_CB_GH1_P0`;
2. A1 support failure: `NO_GO_STATE_EXPLOSION`;
3. A2 or A3 signal/robustness failure: `STOP_NO_GRAPH_SIGNAL`;
4. all gates pass: `OPEN_GH1_INCREMENTAL_GATE`.

`OPEN_GH1_INCREMENTAL_GATE` authorizes only a new preregistered card that:

- freezes a deterministic incremental graph state before measuring it;
- uses CB-TD1-style reversible deltas rather than rescanning the whole board;
- prices the required Black/White role lanes and all D4 lanes explicitly,
  because the actor-relative viewpoint flips globally after a ply;
- requires full-rebuild equality through make/undo and D4/color audits;
- uses a fresh non-forbidden cost workload;
- keeps graph state and every consumer default OFF; and
- decides correctness and cost before any graph-residual model is trained.

Only a later incremental pass may open a separately preregistered GR1 model
card with a frozen dictionary/hash and fresh component-separated evidence.

## Prohibitions

- no graph definition, relation, threat flag, radius, support threshold, or
  serialization change after A1/A2/A3 is read;
- no Weisfeiler-Lehman iteration count or message-passing experiment in P0;
- no D1.5 rescue of RQ598 and no footprint/radius rescue of RQ603;
- no graph dictionary selection by teacher/product labels;
- no dev, safety, 64-game, 1,022-root, outcome, arena, or Pela access;
- no runtime/API/model/dependency/default-feature change;
- no 30-game or 400-game screening; and
- no strength claim from this train-only residual census.
