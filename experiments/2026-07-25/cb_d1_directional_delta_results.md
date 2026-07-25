# CB-D1 exact directional-delta results

Date: 2026-07-25 (Asia/Seoul)

Decision: **GO — promote CB-D1 in the shipped quantized-codebook pbrain.**

Ordinary library `Searcher` callers remain default-OFF. The pbrain defaults
ON and can immediately roll back with
`NORU_CODEBOOK_DIRECTIONAL_DELTA=off`.

## Changed factor

The 0.8.2 evaluator backed up every dirty post-ReLU cell vector and rebuilt
all four direction embeddings for both color perspectives. CB-D1 keeps exact
pre-ReLU Black/White cell sums plus logical Pattern4 IDs and applies only:

```text
raw += embedding(new_id) - embedding(old_id)
region += ReLU(raw_after) - ReLU(raw_before)
```

Undo applies the inverse token deltas. Float evaluation, model weights,
feature mapping, 3x3 region pooling, FM head, move order, alpha-beta, and VCT
proof rules are unchanged.

The two raw caches cost 28,800 bytes. Removing the 225-frame Black/White cell
backup arrays, after accounting for old IDs and raw caches, reduces an
evaluator's optional state by about 1.08 MiB.

## Correctness

| Gate | Result |
|---|---|
| Deterministic mixed make/undo | 100,000 operations; logical IDs, raw Black/White sums, activated cells, region features, value bits, and final empty-root undo all exact |
| Deployed model + frozen trace | 100,000 transitions; incremental mismatch 0, undo mismatch 0, full-quantized max difference 0 |
| Normal tests | 109 passed, 0 failed, 4 ignored |
| `codebook-eval` tests | 123 passed, 0 failed, 7 ignored |
| Public codebook asset | loader semantics and deployed quantization frozen; pass |
| 30-game protocol sanity | D1 OFF 13, D1 ON 17, draws 0, errors 0 |

Integer operations are bounded independently of observed data. Four i16
embeddings give a per-component raw absolute bound of 131,072. One
replacement intermediate is at most 196,607. A 25-cell region is bounded by
3,276,800 and its activation-update intermediate by 3,407,872. All are far
below `i32::MAX`; compile-time assertions freeze these bounds.

The 30-game score is only a safety sanity, not an Elo or strength claim.

## Same-binary performance

Every comparison used one release executable and the registered
`A1 -> B1 -> B2 -> A2` order.

| Workload | Wall B/A | 95% upper | Nodes B/A | NPS B/A | Depth / identity |
|---|---:|---:|---:|---:|---|
| VCT OFF, depth 4, 64 games / 1,022 roots | 0.803242 | 0.807166 | 1.000000 | 1.244955 | all semantic and node fields exact |
| Product VCT ON, depth 4, sealed rerun | 0.907485 | 0.917260 | 1.000000 | 1.101947 | all semantic and node fields exact |
| 2-second class, 16 positions | 0.993505 | descriptive | 1.215790 | 1.223750 | p50 9 -> 9 |
| 30-second class, 4 positions | 0.998393 | descriptive | 1.187710 | 1.189620 | p50 12 -> 12 |

The deterministic primary gate therefore saves 19.68% wall time. The product
VCT-ON integration rerun saves 9.25%. Timed workloads fill their deadline, so
their useful result is 21.58% and 18.77% more visited work rather than lower
wall time.

The first VCT-ON run is retained as negative measurement evidence. Two of
1,022 roots crossed the pre-evaluator 150 ms VCT deadline differently,
including disagreement between the two control arms, and that summary
correctly reports `gate.pass=false`. CB-D1 is not constructed until root VCT
returns. A single complete rerun with the v2 seal recorded baseline commit,
model hash, input, policy, depth, time mode, root-VCT state, seed, actual move,
and root Zobrist; it had zero decision, result, or node mismatches. The
deadline-contaminated 9.18% number is not used for promotion.

## Attribution

The instrumented 16-position profile reported:

- evaluator evaluation time ratio: 0.4820;
- dirty-cell backup removed entirely;
- recompute ratio: 0.4673;
- aggregate ratio: 0.5100;
- total evaluator push/pop ratio: 0.7423;
- forward-head ratio: 1.0186, effectively unchanged.

Inverse-delta restore is slower than the old memcpy restore in isolation
(1.305x), but removing backups and four-direction rebuild dominates the total.
This supports the intended causal mechanism rather than a search-policy
change.

## Evidence

SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| `cb_d1_quant_100k.json` | `1920DD6DF80118712549955DEBC68700D2DDF746C51C15F5F2A715CACF9D1DD2` |
| `cb_d1_product_vctoff_summary.json` | `29AB8FF6C1347B67FFD59B58E95847F1FB487B1EF1BCC686258E093427E442ED` |
| `cb_d1_product_vcton_rerun_summary.json` | `D186BB42709427375E25D21BD247524BB5A6E40D190AE876003C0C680D2A4EDB` |
| `cb_d1_2s_summary.json` | `566605941A2D1903C5FCB112489075BC85BFEE6E8CB0F95CFAFBBB7D6ABBD795` |
| `cb_d1_30s_summary.json` | `4A342A6ADC521DB06BD66940FE11C51FDF996F5FC348F5748CCF2F7814F4F051` |
| `cb_d1_profile_summary.json` | `B3F6234C1403C5B2C2C6BF90742693790F5778BBD0B32370C9E6F786179E31D2` |
| `cb_d1_h2h_30g.jsonl` | `761AE33E81A2964AFF32C4725318934530991846044754B11889B3D0E438BCF9` |

The larger raw ABBA arms are retained beside the release worktree under
`target/figrid-release-0.8.2-artifacts/2026-07-25/cb-d1`; canonical summaries,
the harness, analyzer, preregistration, and 30-game record are checked in.

CB-D1 closes as a promoted product optimization. The next card may now open:
extract the exact update into a generic
`TokenDelta -> reversible accumulator` interface without changing runtime
behavior.
