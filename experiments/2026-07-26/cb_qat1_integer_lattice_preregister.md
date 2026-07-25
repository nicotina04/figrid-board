# CB-QAT1 integer-lattice quantization-aware fine-tuning preregistration

Date: 2026-07-26 KST

Status: **preregistered before CB-QAT1 implementation, headroom census, paired
fine-tuning, validation, artifact construction, timing, or games.**

Pre-implementation factual erratum: the initial preregistration commit
`55aee08` correctly recorded the `Cargo.lock` SHA-256 but incorrectly copied
its byte length as 20,498. The committed git blob and current byte-identical
file are 11,841 bytes. This revision corrects only that provenance field
before any production-corpus P0 run, fitting, artifact construction, timing,
or games; no question, input, statistic, threshold, seed, or decision rule
changes.

## Question and claim boundary

The accepted 0.8.2 product representation remains the flat swap-closed
Pattern4 codebook:

- 4,266 pattern rows;
- 16 embedding dimensions;
- nine region rows;
- FM rank eight;
- runtime lattice `E32/H64/F64`;
- FP32 bias;
- quantized incremental evaluator and White-root ordering.

CB-GH1 stopped before label analysis with `NO_GO_STATE_EXPLOSION`, so CB-QAT1
does not train or quantize a graph representation. It asks one causal question:

> When the current FP32 product is fine-tuned with the same samples, order,
> optimizer, update budget, initialization, and output lattice, does using the
> deployed integer lattice in the forward pass improve the deployed artifact
> over ordinary FP32 fine-tuning followed by PTQ?

The causal comparison is `PTQ-control` versus `QAT`. The current 0.8.2 product
is a separate incumbent guardrail. A better training loss, a smaller
shadow-to-deployment drift, or a better reused validation score is not by
itself a release-strength claim.

All new code is experimental and default OFF. The embedded product weights,
runtime scale, search policy, White-root ordering, VCT policy, evaluator shape,
Pattern4 vocabulary, and pbrain defaults must not change before every gate
passes.

## Historical information available before registration

RQ613 already established the following facts, which are not CB-QAT1 results:

- `E32/H64/F64` is range-safe for the incumbent;
- on 61,782 reused professional-validation rows, the incumbent's scale-64
  representation drift versus its FP32 form had probability-error p99
  `0.0126031637` and max `0.0293332338`;
- scale 2,048 was selected for earlier one-bin lattice searches because very
  small FP32 updates often disappeared at scale 64;
- RQ615, RQ615B, and RQ615D changed only tiny head/factor coordinate surfaces
  and did not clear their registered decision gates.
- the earlier RQ554 PTQ diagnostic moved reused overall BCE by about
  `-0.000127` but live-band BCE by about `+0.000491`, so broad value fidelity
  and live decision fidelity did not point in the same direction;
- in RQ554's one registered 100-game runtime comparison, the FP32 codebook
  scored `52/100` and its otherwise matched quantized form scored `48/100`.
  That noisy four-point estimate is the already-visible `>=3%` prototype
  rationale for this card, not evidence that QAT will recover four points.

Those results make a full-surface, deployment-lattice QAT comparison
mechanistically distinct. They do not establish that scale-64 PTQ currently
hurts K6 decisions, that STE will repair it, or that a repaired validation
metric improves games.

## Sequential access boundary

The card has five strictly ordered stages:

1. `P0`: RQ615C-train-only quantization-headroom census.
2. `P1`: one paired PTQ/QAT fine-tuning invocation, only if P0 passes.
3. `P2`: one fixed reused noninferiority/mechanism guard and deployment-
   artifact gate, only after both terminal artifacts are sealed.
4. `P3`: correctness and evaluator-cost checks on the frozen 1,022-root trace,
   only after the QAT artifact and its paired PTQ control both pass P2.
5. `P4`: 30-game protocol sanity, one 300-game QAT-versus-PTQ causal
   gate, and, only after that gate passes, one disjoint 300-game
   QAT-versus-incumbent product gate.

No later-stage input may be parsed by an earlier-stage executable. A stage
failure leaves every later output absent. No result-dependent retry, alternate
seed, extra epoch, learning-rate change, checkpoint selection, scale change,
threshold change, or second strength run is allowed.

The frozen 1,022 roots are authorized only for P3 correctness and cost. They
are forbidden for P0 headroom selection, fitting, epoch or hyperparameter
selection, P2 model selection, vocabulary construction, coreset construction,
or strength claims.

RQ615C train is already-consumed local diagnostic material. Its only decision
in this card is the binary P0 open/stop decision under the thresholds below.
It cannot choose a recipe, scale, checkpoint, artifact, or promotion claim.
RQ615C dev and `safety_internal` were consumed or pre-accessed by earlier
cards and are forbidden in every CB-QAT1 stage. No result in this card may be
described as a fresh K6 validation result.

## Frozen product and P0 inputs

| input | bytes | SHA-256 |
|---|---:|---|
| `models/gomoku_codebook_v1_swapclosed.json` | 1,410,562 | `42968FDAB01BA8CCD1DE3DED05C532E4B237DD47EEFFD7AE1C2F264D77BA7DA2` |
| `data/topk.bin` | 17,060 | `103891DCD1DCD978C654593ABE78EF32C56E2E350B500EE665BC45AC051AA16D` |
| RQ615C train `rq615c_k6_train.jsonl` | 54,991,200 | `E00A2DA513B05D7631A01003C7E6274E9A3D7575E2C2BD92D5199F1B5385CEB6` |
| RQ615C final corpus manifest | 5,463 | `579D1387D7E4DE8F5CB34DB168B6D15655DB229D992751B1DC17BB6CF4260AA7` |
| RQ569 high-precision lineage model | 1,413,542 | `69BB7C599ADA3A1151577CE3315356BC33C40EDB49A003C9BC4EB90A98F82E18` |

P0 must reproduce the sealed corpus inventory before reading a gate:

- manifest status `READY_FOR_RQ615D`;
- 1,336 slates, 668 paired-color units, and 388 components;
- 668 Black and 668 White slates;
- ordinals `1,2,4,6,8`;
- exactly six candidates per slate and 8,016 candidate children;
- 285,900 complete legal-inventory children;
- product/RQ569 FP32 payload identity;
- stored lineage replay, child hashes, mapped Pattern4 ids, public quantized
  forward, and independent quantized forward with zero mismatch.

RQ615C dev, `safety_internal`, the reused professional validation, frozen
1,022 roots, 64-game logs, game outcomes, Pela artifacts, and arena artifacts
must not be accepted by the P0 CLI.

## P0: fixed quantization-headroom census

### Scores

For every candidate child, evaluate the same incumbent FP32 payload twice:

- `FP32`: the public float codebook evaluator;
- `PTQ0`: the public product evaluator after exact `E32/H64/F64`
  quantization.

The bias remains its incumbent FP32 value in both arms. For a root candidate,
utility is the negative of the natural child-side-to-move logit. Candidate
order is the stored RQ615C order.

For each six-candidate slate, use the stored binary64 `q_teacher` without
renormalization. Compute

`CE = sum_i q_i * (logsumexp(u) - u_i)`

with max subtraction, Rust binary64 `exp`/`ln`, fixed candidate order, and
Neumaier summation. The registered point delta is:

`delta_ce = CE(PTQ0) - CE(FP32)`.

Top-1 uses maximum utility with lowest candidate index as the exact tie-break.
On an FP32/PTQ0 top-1 disagreement, compare the two selected candidates'
stored `q_teacher` values:

- `fp32_q_superior`;
- `ptq_q_superior`;
- `q_equal`.

For each unordered pair `i<j`, obtain the finite binary64 `partial_cmp`
ordering in each arm; any difference among Less/Equal/Greater is one pair-
order disagreement. Logit drift is candidate-level `abs(u_ptq-u_fp32)`.
Probability drift is candidate-level absolute difference between the two
six-way softmax vectors. Sort each 8,016-value stream with `f64::total_cmp`;
p50/p90/p95/p99 use nearest-rank index `ceil(p*N)-1`, and max is the final
element.

Report combined, Black, White, ordinal, and component aggregates. Also report
absolute logit/probability drift p50/p90/p95/p99/max, all 15 within-slate
pair-order disagreements, q-argmax accuracy, teacher-top accuracy, and
top-1-disagreement counts. Combined/color/ordinal point CE values are
equal-slate arithmetic means. These are descriptive unless named in the gate.

### Component bootstrap

Run exactly 100,000 component-cluster bootstrap replicates with SplitMix64 seed
`2026726001`. Each replicate samples exactly 388 components with replacement
using `next_u64() % 388` into the uppercase-UID lexicographically sorted
component list, and includes every slate in each sampled component. Compute
the equal-slate mean `delta_ce`; sort with `f64::total_cmp`. The one-sided 90%
lower endpoint is zero-based element `ceil(0.10*N)-1 = 9,999`.

### P0 gate

P0 returns `GO_PAIRED_QAT_TRAIN` only if all conditions hold:

1. every provenance, corpus, replay, finite-value, mapping, and independent
   forward check has zero mismatch;
2. combined point `delta_ce > 0`;
3. component-bootstrap p10 `delta_ce > 0`;
4. Black and White point `delta_ce >= 0`;
5. FP32 and PTQ0 disagree at top-1 on at least seven of 1,336 slates;
6. combined `fp32_q_superior - ptq_q_superior >= 2`;
7. neither color has a negative `fp32_q_superior - ptq_q_superior` net.

Any correctness or provenance failure is `INVALID_QAT1_P0`. A valid gate
failure is `NO_GO_PRECONDITION`, ends CB-QAT1, and leaves all training,
validation, artifact, timing, and game outputs absent.

## P1: frozen paired fine-tuning

P1 is authorized only by a sealed P0 `GO_PAIRED_QAT_TRAIN` result.

### Fit inputs

| input | bytes | SHA-256 |
|---|---:|---|
| current product initialization | 1,410,562 | `42968FDAB01BA8CCD1DE3DED05C532E4B237DD47EEFFD7AE1C2F264D77BA7DA2` |
| Rapfi 300k/1s raw games `rapfi_avx512vnni_300k_1s.jsonl` | 1,187,813,619 | `F2BECA8104E902EE700A22ED13B722EF09AFBF83CFB2E0CF3E801680646CC481` |
| current `data/topk.bin` | 17,060 | `103891DCD1DCD978C654593ABE78EF32C56E2E350B500EE665BC45AC051AA16D` |

The raw extractor is frozen to the RQ552 semantics:

- minimum ply five;
- `mate-first` label filter;
- `rapfi-value` target;
- mate clamp 40 and Rapfi evaluation scale 400;
- exactly 6,163,315 accepted samples;
- 2,976,437 Black-to-move and 3,186,878 White-to-move samples;
- no game result or winner target;
- current swap-closed Pattern4 table.

Positions are captured immediately before their engine move. Only moves with
literal `source=="engine"` and pre-move ply at least five are eligible.
`mate_in` takes precedence over `eval_cp`. At each game's start the remembered
mate pair is empty. It changes only on an eligible mate row, to
`(sign(mate_in), raw move color)`; CP, non-engine, and pre-min-ply rows do not
clear it. A mate row is accepted only if that pair differs from the remembered
pair before the update. This is the complete RQ552 mate-run state machine.

All target arithmetic is binary32 in the written order. For a CP label, the
target is the branch-stable binary32 sigmoid of `cp as f32 / 400.0f32`. For
mate distance `m`, let `a=min(abs(m),40) as f32`,
`edge=1.0f32-a/(40.0f32*2.0f32)`, and
`magnitude=0.5f32+edge*0.5f32`; the target is `magnitude` for positive mate
and `1.0f32-magnitude` for negative mate. Each sample has equal weight.

For both arms, `p` is the historical branch-stable binary32 sigmoid: for
`z>=0`, `1/(1+exp(-z))`; otherwise `exp(z)/(1+exp(z))`. Reported loss is:

`L = -y*ln(max(sigmoid(z),1e-7)) - (1-y)*ln(max(1-sigmoid(z),1e-7))`.

Optimization does not differentiate the reporting clamp:
`dlogit=p-target`. Targets, sigmoid, surrogate gradients, shadow weights, Adam
states, and updates are Rust binary32. Each epoch's online loss is the
binary64 Neumaier mean of the per-row pre-update binary32 losses. A separate
post-epoch full-source BCE is not computed. Neither arm receives the other
arm's prediction as a target.

The paired fit CLI accepts no validation, RQ615C dev/safety, 1,022-root,
game-result, arena, or Pela path.

### Arms and the single changed factor

Both arms:

- initialize every shadow FP32 weight and optimizer state identically from the
  current product;
- train all 68,256 embedding weights, 144 head weights, 1,152 FM-factor
  weights, and the FP32 bias;
- consume the same in-memory samples in the same shuffled order;
- use the existing RQ552 codebook recipe: three epochs and per-sample online
  Adam updates;
- use Adam with `lr=0.001`, beta1 `0.9`, beta2 `0.999`, epsilon `1e-8`;
- use registered base seed `552`;
- publish only terminal epoch three; no validation is loaded during fitting
  and there is no best checkpoint.

The sample order uses `noru::trainer::SimpleRng`, initialized once as
`552 ^ 0xC0DE_B00C_F00D_0001 ^ 0x5150_5150`, and the existing in-place
descending Fisher-Yates shuffle. The order vector starts as `0..6,163,315`.
It is shuffled once before each epoch, with the same RNG stream continuing
across epochs; epoch two shuffles epoch one's permutation in place and epoch
three shuffles epoch two's. `SimpleRng` is the locked noru 2.2.0 xorshift64:
`x^=x<<13; x^=x>>7; x^=x<<17`, with wrapping u64 shifts, and
`next_usize(n)=next_u64()%n`. Each resulting order is shared by both arms; the
paired loop updates PTQ control and then QAT for one row before advancing.
Each epoch-order SHA-256 is over every index encoded as little-endian u64 in
order with no header. Each arm therefore performs exactly
`3 * 6,163,315 = 18,489,945` Adam steps.

The only arm difference is the forward/backward weight view:

- `PTQ-control`: FP32 shadow weights in forward and backward; quantize the
  terminal shadow once to `E32/H64/F64`.
- `QAT`: fake-quantized `E32/H64/F64` weights in every forward; backward uses
  the straight-through derivative one when the rounded pre-clamp integer is
  representable as i16 and zero when clamping occurs. The FP32 bias is not
  fake-quantized. Requantize the QAT forward view after every sample update.

Quantization is exactly the product Rust operation
`(w * scale).round().clamp(i16::MIN, i16::MAX)`: binary32 multiplication,
Rust `f32::round` halfway-away-from-zero semantics, then clamp and i16 cast.
An independent validator must implement those semantics explicitly; Python's
ties-to-even `round()` is not conformant.

PTQ-control uses the historical float kernel exactly: cells ascending; four
stored Pattern4 direction slots in array order; binary32 embedding additions;
per-cell binary32 ReLU; `activated/25.0f32` added to its regional feature per
cell; then head features ascending and FM ranks ascending/features ascending,
all with naive binary32 accumulation.

PTQ-control backward is also fixed in binary32. Let `X_i` be the exact
regional feature produced by that arm's forward, `H_i` and `V_ik` its
pre-update shadow head and factor values, and
`S_k=sum_j X_j*V_jk` accumulated over features ascending. Then, with ranks
ascending:

- `beta_i = H_i + sum_k V_ik*(S_k-X_i*V_ik)`;
- `dH_i = dlogit*X_i`;
- `dV_ik = dlogit*X_i*(S_k-X_i*V_ik)`;
- each active embedding occurrence receives `dlogit*beta_i/25.0f32`.

The PTQ ReLU derivative is one only when its exact pre-update binary32 cell
preactivation is strictly positive. PTQ gradients receive no quantizer clamp
mask. Products, subtraction, multiplication by `dlogit`, and additions are
evaluated in the written order with no fused or compensated reduction. This
is the historical trainer derivative made explicit rather than an invitation
to choose another autodiff ordering.

The QAT forward is the exact deployment kernel. Four mapped i16 embedding rows
are summed into i32 cell accumulators in stored direction order, ReLU is
applied in i32, and the 25-cell region sum remains integer. Starting with bias
cast to f64, head terms are accumulated naively in ascending feature order;
FM is accumulated naively in ascending rank then feature order with the
public binary64 denominators. There is no compensated summation. The sole
terminal cast is `logit as f32`. Its backward pass differentiates the
corresponding dequantized expression through the STE. ReLU is evaluated on
each arm's own forward embedding sum; head and FM gradients use that same
arm's forward values. QAT must not silently use FP32 activations or FP32
head/factor values in its forward or parameter-gradient formulas.

For QAT backward, all following values and operations are binary32 in
ascending index order:

- `X_i = feature_integer_i as f32 / (32.0f32*25.0f32)`;
- `H_i = qH_i as f32 / 64.0f32`;
- `V_ik = qV_ik as f32 / 64.0f32`;
- `S_k = sum_j X_j*V_jk`.

The registered QAT surrogate derivatives are:

- `beta_i = H_i + sum_k V_ik*(S_k-X_i*V_ik)`;
- `dH_i = dlogit*X_i`;
- `dV_ik = dlogit*X_i*(S_k-X_i*V_ik)`;
- each active embedding occurrence receives `dlogit*beta_i/25`.

The ReLU STE is one only when the exact integer preactivation is strictly
positive. For both arms, White-sample embedding identity is the post-
`swap_mapped_id` row. Occurrences are traversed cell, dimension, then stored
direction slot ascending; duplicates are summed in that stable occurrence
order and each touched embedding index is updated once in ascending index
order. Shadow-weight STE gradients do not receive an extra inverse-scale
factor.

Every QAT E/H/F gradient is multiplied by
`mask(w,s)=1` iff binary32 `round(w*s)` lies inclusively in
`[-32768,32767]`, otherwise zero. Bias has `dbias=dlogit` and no mask. A value
is saturated only when its rounded pre-clamp value is outside that inclusive
range; an exact boundary bin is not saturation. Each updated quantized
parameter is checked before the next row. Any transient or terminal
saturation aborts P1 without publishing an artifact.

All shadow tensors are cloned from the product; every Adam `m`, `v`, and the
global step start at zero. The step increments once before each sample's bias
correction. Binary32 corrections are
`1.0-0.9.powi(step as i32)` and `1.0-0.999.powi(step as i32)`. All gradients
come from one pre-update forward snapshot. Head calls Adam for every feature,
including zero gradients; factors skip all ranks only when that arm's
`X_i==0.0`; bias updates last. Untouched sparse embedding moments are not
decayed. The Adam scalar update is the historical binary32 sequence
`m=.9*m+.1*g`, `v=.999*v+.001*g*g`, `m_hat=m/bc1`,
`v_hat=v/bc2`, `w-=lr*m_hat/(sqrt(v_hat)+1e-8)`.

Full E/H/F training is required in both arms: freezing embeddings
would change the trainable surface as well as quantization awareness and is
therefore not an allowed same-card ablation.

No stochastic rounding, learned scale, per-row scale, clipping calibration,
weight decay, label smoothing, D4 augmentation, auxiliary loss, distillation
term, graph feature, vocabulary change, int8/PQ, or extra optimizer warm-up is
permitted.

### Terminal artifacts

The one fit invocation creates, with create-new semantics:

- PTQ shadow FP32 checkpoint, audit-only;
- QAT shadow FP32 checkpoint, audit-only;
- PTQ deployed-lattice JSON;
- QAT deployed-lattice JSON;
- aggregate paired-fit report.

Deployed JSON stores each i16 bin as the exactly representable power-of-two
fraction `embedding/32`, `head/64`, or `factor/64`; reloading through the public
quantizer must recover every integer bin exactly. Bias remains the terminal
arm's FP32 bias. Every output is `deployment_eligible=false`.

Both arms are freshly quantized after the final Adam update. Their stored bins
must equal fresh quantization of the sealed terminal shadows; stored bias must
equal the shadow bias by `f32::to_bits`. Reloading JSON must reproduce every
bin, scale, shape, and bias bit. The training implementation's QAT forward
kernel is rerun using the sealed terminal lattice and must match the public
JSON evaluator by `f32::to_bits`; P2 later extends the same check to CBF.

The report must bind input, executable, source, output, sample-order, and
artifact hashes; per-epoch and terminal losses; update counts; touched Pattern4 rows;
shadow-to-deployed drift; integer-bin changes versus incumbent; and
saturation/overflow counts.

Any non-finite value, i16 overflow, saturation, sample-count mismatch,
sample-order mismatch, unequal update budget, output overwrite, or artifact
round-trip mismatch is `INVALID_QAT1_P1`.

## P2: reused noninferiority/mechanism guard and artifact gates

P2 may start only after an independent file-only validator reconstructs both
terminal lattice payloads from the fit report and shadow checkpoints without
training-row access.

The first validation input is the repeatedly used RQ508 professional
validation:

| input | bytes | SHA-256 |
|---|---:|---|
| `rq508_rapfi30k_common_distill_val.jsonl` | 90,028,835 | `BBB10699AC16E322C553BAD576E2055B8BFE48A48AD89A51B460DC31A871C8A7` |

It contains 61,782 fixed mate-first rows: 15,492 CP, 23,207 mate-loss, and
23,083 mate-win. It is repeatedly reused internal evidence, not fresh positive
evidence, and may not choose a checkpoint or hyperparameter. Rows are grouped
by exact `(source.path, source.game_id, source.seed)` for paired game-cluster
telemetry; rows are never treated as independent experimental units.

Evaluate the deployed incumbent, deployed PTQ control, deployed QAT, and both
shadow checkpoints. Report overall BCE/accuracy, live `|cp|<=150`, remaining
CP bands, mate-win, mate-loss, Black/White, probability drift, and exact
classification transitions.

The fixed reused guard passes only if:

1. deployed QAT overall BCE is strictly below deployed PTQ-control BCE;
2. deployed QAT live-band BCE is no worse than deployed PTQ-control;
3. the training implementation's terminal-lattice QAT forward and reloaded
   deployed artifact are bit-identical on every validation logit; PTQ
   shadow-to-deployed and QAT shadow-to-deployed drift remain descriptive
   diagnostics rather than a requirement that shadow weights sit at bin
   centers;
4. versus the deployed incumbent, QAT overall BCE delta is `<= +0.001`,
   live-band BCE delta is `<= +0.002`, and accuracy delta is `>= -0.0005`;
5. all overall, band, color, and label-kind values are finite, and no arm
   changes row inventory or target interpretation.

The validator reports game-cluster paired bootstrap intervals with exactly
100,000 SplitMix64 replicates and seed `2026726201`, but these intervals are
descriptive because this corpus has already been consumed. The load-bearing
P2 comparisons are the fixed point guards above. Failure is
`REJECT_REUSED_GUARD` and leaves P3/P4 absent. Passing permits correctness,
cost, and then a fresh arena; it is not an offline promotion claim.

RQ508 train/test, RQ615C dev/`safety_internal`, the frozen 1,022-root trace,
game outcomes, and arena inputs must not be accepted by the P2 validator.

### Product-format deployment artifact

The actual product embeds the exact factored CBF container and reconstructs
the established flat i16 runtime table when `NORU_CODEBOOK_FACTORED` remains
OFF. The frozen incumbent artifact is:

| artifact | bytes | SHA-256 |
|---|---:|---|
| `models/gomoku_codebook_v1_swapclosed_factored.cbf` | 353,582 | `141014529417A73E58B210832AFD189AD970E045A8907F7D2879693C5B171A8D` |

After the reused guard passes, pack both PTQ-control and QAT lattice payloads
with the existing exact class-base plus i8-residual factored packer. Each CBF
must have the incumbent's format, kind, shape, source-f32 rollback section,
integer scales, and exact 353,582-byte length. Reconstructing its flat i16
table must reproduce every terminal bin and bias bit. Public deployed JSON,
factored CBF reconstructed-flat, and the independent trainer evaluator must
produce bit-identical f32 logits on every P2 row.

The packer input for each arm is that arm's sealed terminal shadow-FP32
checkpoint and its digest. The CBF source section must reproduce that shadow
checkpoint by f32 bits, while the CBF quantized section must reproduce the
arm's deployed-lattice JSON bins and bias bits. Packing the already
dequantized lattice JSON as the source section is forbidden because it would
silently change the product's quantization-OFF rollback semantics.

The existing factor representation is a packaging constraint, not a new model
factor. No residual clipping, alternate class assignment, approximate pack,
or compact-flat fallback is allowed. If either terminal lattice is not exactly
representable, the outcome is `SHADOW_PACKAGING_BLOCKED`; P3/P4 remain absent.
The packer's source-JSON digest and payload bytes are expected to differ from
the incumbent; artifact compatibility means exact integer reconstruction,
schema and size identity, not container byte identity.

No embedded model or product default changes in P2.

## P3: correctness and cost

P3 is the only stage allowed to access the frozen campaign trace of 64 games /
1,022 roots. It may not read game outcomes and may not use any metric to alter
the model.

The trace is
`../figrid-dp-campaign/experiments/2026-07-25/dp_a1_fresh_holdout_64g.jsonl`,
317,511 bytes, SHA-256
`1FD40D8948F113AD236FA44F5EEADCA1907C0C3103987CB4C704B67A9B47531A`.
Its path and hash are not accepted by P0-P2. P3 opens it only after the two
terminal checkpoints, deployed lattice payloads, both packed factored
artifacts, P2 report, evaluator binary, and every remaining threshold are
sealed.

Correctness is run independently for terminal PTQ-control and QAT and
requires:

- 100,000 deterministic mixed make/undo transitions;
- each arm's incremental value equals its own full rebuild bit-for-bit after
  every materialized step and undo;
- directional-delta and legacy incremental paths agree bit-for-bit;
- deployed JSON, factored-CBF reconstructed-flat, and independent evaluator
  agree bit-for-bit;
- on every D4-transformed and color-swapped test board, those three
  implementations agree with one another bit-for-bit; transformed boards are
  not required to equal the untransformed board;
- stale undo, overflow, illegal move, abort, and stack-bound tests pass.

The mixed test uses the existing xorshift `TestRng` seed
`0xCB01_2026_0726_0001`, 100,000 operations, undo when history is nonempty and
either move count is at least 180 or `next_usize(4)==0`, otherwise one
uniformly indexed legal move. Materialize when `next_usize(8)==0`, every 97th
operation, and the final operation; full rebuild occurs every 97th operation
and at the end. Complete undo must restore the empty board and evaluator bits.

Evaluator cost uses one release binary, pinned process affinity, warm-up, and
alternating `ABBA/BAAB` order on the same 1,022 boards. Time only evaluator
refresh plus value, not file I/O or search. Both factored artifacts are parsed
before timing and reconstructed once into their flat i16 runtime tables.
Compare incumbent to QAT on Windows logical processor 2 using
`QueryPerformanceCounter`:

- four unrecorded full-corpus warm-up blocks;
- 20 measured blocks, odd blocks `A-B-B-A`, even blocks `B-A-A-B`;
- each arm occurrence is one leg; within a leg, visit the 64 source-game
  groups in first-appearance order, and within each group perform 24 complete
  passes over that group's boards in trace-file order;
- for each group in each leg, read QPC immediately before its first refresh
  and immediately after its final value call, so there is exactly one timed
  start/stop pair per group per leg and no per-board timer call;
- retain each interval as its raw nonnegative u64 tick delta and sum ticks for
  each group across that arm's 40 legs, producing exactly 64 group totals per
  arm; all load-bearing ratios and bootstrap replicates use these raw tick
  sums, while nanoseconds are descriptive binary64 values computed with the
  single run-start QPC frequency;
- point ratio is the sum of the 64 QAT group totals divided by the sum of the
  64 incumbent group totals;
- exactly 100,000 64-game cluster-bootstrap replicates use SplitMix64 seed
  `2026726301`, sampling `next_u64()%64`; each replicate is a ratio of sampled
  sums, sorted by `f64::total_cmp`, with upper endpoint element 94,999.

The cost gate is:

- identical executed sample counts;
- point wall ratio `QAT/incumbent <= 1.005`;
- one-sided 95% upper bound `< 1.01`;
- exact factored payload byte-length equality and exact reconstructed runtime
  tensor lengths.

Because the shape and kernel are fixed, a cost failure is treated as an
implementation/provenance defect, not as a weight-quality trade. Fixed-depth
search wall time and node count are diagnostic only because different weights
may change the search tree. CB-QAT1 makes no speedup claim; int8, activation
quantization, and the campaign's `<=0.95` speed gate require a separate card.

## P4: causal and product strength

Only a P3 PASS may build one release pbrain executable. Before games, add one
default-inert loader capability: when `FIGRID_CODEBOOK_WEIGHTS` names a valid
product-format CBF, parse it with `PackedCodebookArtifact`, reconstruct its
source weights, quantize them into the established flat i16 runtime table, and
mark White-root `auto` support exactly as for the embedded product. External
JSON behavior remains unchanged. No external path still loads the embedded
product exactly as before.

Loader tests must prove malformed/wrong-kind CBF rejection, external incumbent
CBF versus embedded incumbent i16 identity, identical White-root auto state,
JSON backward compatibility, and `NORU_CODEBOOK_FACTORED=off` flat-runtime
identity. On all 1,022 P3 roots, external incumbent CBF must equal embedded
incumbent; external PTQ/QAT CBF must equal their public JSON and independent
evaluators. The arena wrappers all set the same product-format CBF path
variable, so their only difference is the sealed weight artifact.

Runtime evaluation scale remains `15.72016213285046`. All search, VCT,
White-root ordering, time management, opening, and protocol settings remain
identical.

### Frozen paired harness and openings

A committed `scripts/cb_qat1_paired_arena.py` may reuse only the behavior of
these frozen upstream files:

| file | bytes | SHA-256 |
|---|---:|---|
| `sparring_vs.py` | 14,669 | `06C6779AD82C89EEE2C914A168A920926B482D6C72706141E505A832CB7105F5` |
| `sparring.py` | 16,784 | `3D26DFD63CE96ED34B0F9F2F407BC2275E8CBD40849F9BF4B902D8D75DC39C8D` |
| `pbrain.py` | 6,438 | `7D042DD64B947BD89038BDFBF26FD7D03873385146963ABC9DA4591E01A2F288` |
| `board.py` | 2,151 | `15920070A0ADAF5825BBD64ADEF65C837C489DABCAD99F423F7C934AB27C48F5` |

The new harness fixes a defect in ordinary `--swap-sides`: each pair, not each
game, owns one seed and one literal opening. It launches fresh engine processes
for both games, swaps colors on the second, and alternates which color order
runs first by pair index. Four workers each execute whole pairs sequentially;
there is no cross-pair engine reuse.

Openings use the frozen Python 3.12.13 `random_opening(seed,4)` algorithm:
choose one of the fixed balanced three-stone templates with
`random.Random(seed)`, apply its selected D4 transform, then append one unique
cell sampled in the center's `[-4,4]^2`. For each stage, iterate consecutive
seeds from its base and retain the first required literal ordered openings not
already retained by that or an earlier opened P4 stage:

- sanity: base `2099726300`, retain 15;
- causal QAT/PTQ: base `2099730000`, retain 150;
- product QAT/incumbent: base `2099740000`, retain 150.

Manifests are opened sequentially, never speculatively. Create and seal the
sanity manifest before the sanity launch; after sanity passes, create and seal
the causal manifest before the causal launch; only after `GO_QAT_CAUSAL`
create and seal the product manifest before P4B. Each create-new manifest
binds every attempted seed, rejection, literal four moves, and
ordered-opening SHA-256, and checks zero literal overlap against every
previously opened P4 set. An unopened later-stage manifest must remain absent.
Fewer than the quota within the first 10,000 seeds of a namespace is invalid,
with no alternate namespace.

Every game is 15x15 Freestyle (`INFO rule 0`), 2,000 ms/turn, 180,000 ms
match timeout, four opening moves, and four parallel pair workers. At 150
total plies a draw is adjudicated only after the latest 30 engine moves carry
no mate metadata; otherwise play continues until a later quiet window, win,
or full board. Outputs use create-new semantics, include pair/opening/color
identity and full telemetry, and have an eight-hour process-tree wall cap.

### P4A: QAT training effect

First run 30 QAT-versus-PTQ games from the 15 sanity pairs. Its WDL is
non-load-bearing and cannot change a later setting; any error, crash, timeout
corruption, illegal move, or missing telemetry is `UNSOUND`.

Then run exactly 300 QAT-versus-PTQ games from the 150 causal pairs. For pair
`i`, define `d_i` as QAT's score across its two color-swapped games minus one.
Set `t_i=2*d_i`, discard exactly-zero `t_i`, let `m` be the remaining count,
and let `T_obs=sum_i t_i` in the observed QAT orientation. Using
arbitrary-precision integer dynamic programming, count all `2^m` sign vectors
`s` for which `sum_i s_i*abs(t_i) >= T_obs`. The exact one-sided p-value is
that count divided by `2^m`; zero deltas contribute neither a sign bit nor a
denominator factor. The observed assignment is included. If `m=0`, `p=1`.
The load-bearing comparison `p<0.05` is performed exactly as
`20*tail_count < 2^m`, not through rounded floating point.

`GO_QAT_CAUSAL` requires QAT score at least 165/300, paired p-value `<0.05`,
each color score at least 72/150, and zero errors. Only this conjunctive result
opens P4B. A score strictly above 150 but below 165 is `SHADOW_CANDIDATE`; a
score at least 165 missing p/color is `INCONCLUSIVE`; a score at most 150 is
`SAFE_NO_NET_GAIN`. All three stop the card default OFF without rescue.

### P4B: product promotion, only after P4A GO

Run exactly one further 300-game QAT-versus-incumbent comparison on the 150
product pairs. Apply the same score, paired-p, per-color, and zero-error
conjuncts. Passing is `GO_PRODUCT_QAT`; otherwise use the same
`SHADOW_CANDIDATE`, `INCONCLUSIVE`, `SAFE_NO_NET_GAIN`, or `UNSOUND` labels
and leave the product unchanged.

Neither 300-game run is a top-up or rescue: they answer different registered
comparisons. Do not pool either run with the 30-game sanity, the other
300-game comparison, historical 64 games, frozen 1,022-root trace, RQ554
games, or any other arena. There is no Pela follow-up, alternate seed, extra
games, scale retuning, or threshold rescue in CB-QAT1.

## Promotion and rollback

Only `GO_PRODUCT_QAT` permits a separate integration commit that replaces the
embedded product model and exact factored CBF payload. That commit must retain
the incumbent artifact and an explicit default-OFF rollback selector until the
final 0.8.2 release regression closes.

Every other valid terminal state leaves the current embedded product and
defaults unchanged. Regardless of outcome, CB-AL1 may begin only after the
CB-QAT1 result and all actually executed artifacts are committed.

## Build and provenance contract

Every production-stage executable is built from a clean committed tree with:

```text
RUSTFLAGS=-C target-cpu=x86-64-v3
cargo build --release --locked --features codebook-eval --bin <stage>
```

The executable must record release/debug-assertion status, target profile,
compile-time source and embedded-input hashes, executable hash, CPU identity,
and compiled/runtime AVX2, BMI2, and FMA availability. AVX2 and BMI2 are
required. The registered toolchain is rustc 1.88.0 commit
`6b00bc3880198600130e1cf62b8f8a93494488cc`, LLVM 20.1.5, and cargo 1.88.0
commit `873a06493`. `Cargo.lock` is 11,841 bytes, SHA-256
`3F90AA762C0D7B1F0172C22397588835C79B9C924BB5A931D162B2A5714A202C`;
it locks noru 2.2.0 with registry checksum
`83654c4f008197f515f315b6760a4781c6f734a6f0205e522b90119a8e1d29f2`.
The runtime records `rustc -Vv` and `cargo -V` again. Every registered source,
Cargo input, Pattern4 table, model, corpus, manifest, and output path is
byte-checked both before and after use.

All `NORU_*`, `FIGRID_*`, `RAYON_*`, and Rust profiling/instrumentation
variables are forbidden during P0-P3 unless the exact variable is named by a
later sealed command. P4 wrappers clear that environment first, rely on the
sealed compiled product evaluation-scale default, and set only the one
registered external-weight path. Training is single-process and single-thread
deterministic; timing uses one pinned process. Source drift, dirty
tracked/untracked files, debug builds, assertion-enabled release artifacts,
unexpected target features, output aliases, or overwrite attempts invalidate
the affected stage.

## Planned implementation and outputs

Names are fixed before results:

- `bin/cb_qat1_headroom.rs`;
- `bin/cb_qat1_train.rs`;
- `bin/cb_qat1_validate.rs`;
- `scripts/cb_qat1_paired_arena.py`;
- `experiments/2026-07-26/cb_qat1_p0_headroom.json`;
- `experiments/2026-07-26/cb_qat1_paired_fit_report.json`;
- `experiments/2026-07-26/cb_qat1_ptq_shadow.json`;
- `experiments/2026-07-26/cb_qat1_qat_shadow.json`;
- `experiments/2026-07-26/cb_qat1_ptq_lattice.json`;
- `experiments/2026-07-26/cb_qat1_qat_lattice.json`;
- `experiments/2026-07-26/cb_qat1_professional_report.json`;
- `models/gomoku_codebook_cb_qat1_ptq_factored.cbf`;
- `models/gomoku_codebook_cb_qat1_qat_factored.cbf`;
- `experiments/2026-07-26/cb_qat1_correctness.json`;
- `experiments/2026-07-26/cb_qat1_cost.json`;
- `experiments/2026-07-26/cb_qat1_sanity_openings.json`;
- `experiments/2026-07-26/cb_qat1_causal_openings.json`;
- `experiments/2026-07-26/cb_qat1_product_openings.json`;
- `experiments/2026-07-26/cb_qat1_sanity_qat_vs_ptq_30g_games.jsonl`;
- `experiments/2026-07-26/cb_qat1_causal_qat_vs_ptq_300g_games.jsonl`;
- `experiments/2026-07-26/cb_qat1_product_qat_vs_incumbent_300g_games.jsonl`;
- `experiments/2026-07-26/cb_qat1_results.md`.

Outputs from unopened stages must remain absent. Large generated reports,
checkpoints, packed artifacts, and game logs belong in the external campaign
artifact directory when excluded by the crate package; committed result
documents bind their byte length and SHA-256.
