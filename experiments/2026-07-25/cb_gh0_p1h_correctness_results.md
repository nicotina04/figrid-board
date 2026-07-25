# CB-GH0 P1-H exact D4 hash correctness result

Date: 2026-07-25 KST

Final P1-H label: **OPEN_GH0_HASH_COST_GATE**

The default-OFF exact D4 state-hash sidecar passed its registered correctness,
collision, rule-domain, transform-relation, composition, and search-
transparency gates. This result opens only P2-H hash-maintenance cost
measurement.

It does not open canonical transposition-table score, bound, or move sharing.
The released evaluator remains orientation-specific under the P0 result
`OPEN_GH0_HASH_ONLY_TT_BLOCKED`, so that branch stays blocked. This result
also does not open a benchmark or arena claim, proof-cache identity by u64
alone, a pbrain environment switch, product promotion, or a default change.

## Sealed implementation

The implementation and authoritative harness were committed before the
registered workload:

```text
c2d38e3dee70766308bca984503276c9c5abe7bd
```

The report required all critical sources to be tracked in that HEAD and the
tracked worktree to be clean at execution. It sealed `Cargo.toml`,
`Cargo.lock`, `src/lib.rs`, `src/board.rs`, `src/search.rs`,
`src/d4_hash.rs`, the correctness harness, the parent preregistration, and
the P1-H amendment by byte length and SHA-256.

The release executable was built once with:

```text
RUSTFLAGS=-C target-cpu=x86-64-v3
```

Both create-new runs reused that exact executable:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `cb-gh0-hash-correctness.exe` | 1,581,568 | `43F3870356605CA015E8E059840F1605F630EB9CFA03F112EAE794347E96718A` |
| each authoritative report | 19,147 | `75C96E3FC3B284CF803F58382A36135DE1462C6EE501BDE68D8AD491544BD620` |

The two reports were byte-identical. The process environment contained no
`NORU_*` overrides. The harness also disabled root VCT structurally in both
fixed-depth smoke-search arms, after verifying that the production default
remained enabled.

## Frozen transition tape

The SplitMix64 tape used seed `0xCB60_2026_0725_0001` and completed exactly
100,000 registered transitions.

| Count | Observed |
|---|---:|
| make transitions | 50,090 |
| undo transitions | 49,910 |
| PRNG draws | 150,090 |
| rule switches before every 251st transition | 398 |
| registered state audits | 100,399 |
| final ply before unwind | 180 |
| unwind undo operations | 180 |
| unwind state audits | 181 |

The final PRNG state was `0x840BED2552B4F013`. The maximum move count was
180, and the full unwind returned to a fresh Freestyle empty state. An empty
undo was also verified as a no-op.

## Incremental, prediction, and exact-state checks

All registered comparisons passed:

| Check | Count | Failures |
|---|---:|---:|
| independent full rebuilds | 100,580 | 0 |
| incremental hash-lane comparisons | 804,640 | 0 |
| predicted make contexts | 50,090 | 0 |
| predicted hash-lane comparisons | 801,440 | 0 |
| canonical-context checks | 100,580 | 0 |
| independent 66-byte exact-state checks | 100,580 | 0 |
| retained-versus-rebuilt lane comparisons | 804,640 | 0 |

The exact state independently serialized both colors' bitboards, side to
move, and the effective rule. Canonical selection was the lexicographic
minimum of all eight transformed 66-byte states with the registered
lower-transform-index tie rule.

## D4, rule, and collision gates

The registered D4 convention passed every structural and sampled relation:

| Check | Count | Failures |
|---|---:|---:|
| map bijection checks | 1,800 | 0 |
| composition checks | 14,400 | 0 |
| registered relation states | 1,032 | 0 |
| named fixture relation states | 6 | 0 |
| transformed boards | 8,304 | 0 |
| transform-pair relations | 66,432 | 0 |
| mapped-move round trips | 1,868,400 | 0 |

All six named symmetry fixtures had the registered stabilizers: empty,
center-only, full D4, vertical-reflection-only, 180-degree-only, and
asymmetric. The synthetic equal-minimum fixture selected transform 1 for
hashes `[9,3,3,7,8,5,6,4]` in both the production selector and the independent
selector.

Freestyle, Standard, Caro, and Renju empty-state rule keys were pairwise
distinct. Formal Standard and legacy `exact5` Standard agreed on a nonempty
board, and returning to Freestyle cleared `exact5` and restored the
Freestyle-domain hashes.

The collision audit observed:

| Classification | Count |
|---|---:|
| D4-equivalent repeats | 51,533 |
| true hash collisions | 0 |
| intra-orbit minimum-hash collisions | 0 |

A canonical u64 therefore passed this workload as a fingerprint, but it is
not authorized as sole proof-cache identity. Any future proof-cache consumer
must also verify the exact canonical state or original state.

## Default-OFF and search transparency

All default and composition gates passed:

- a fresh board-side search state leaves D4 hashing OFF;
- a fresh `Searcher` requests D4 hashing OFF;
- ordinary board make/undo does not activate the sidecar;
- packed Pattern4 windows, candidate frontier, and D4 hashing remained
  synchronized through the registered 16-transition composition sequence;
- the public exhaustive `Board` literal from 0.8.1 still compiled.

Three deterministic depth-2, all-zero-NNUE smoke roots (`empty`, `sparse`,
and `edge`) returned identical best move, score, completed depth, node count,
and restored final board with the sidecar OFF versus maintained-but-
unconsumed ON. Node counts were 51, 515, and 451 respectively.

Focused verification before the authoritative run:

```text
correctness harness tests: 9 passed
D4 sidecar/hash tests:     12 passed
library tests:             127 passed, 4 registered long audits ignored
public Board API test:     1 passed
```

## Decision

Every registered failure class was zero, both authoritative reports were
byte-identical, and the final decision is:

```text
OPEN_GH0_HASH_COST_GATE
```

P2-H may now measure only the maintenance cost of keeping the exact D4 hash
sidecar ON but unconsumed, using the separately frozen timing protocol.
Canonical TT score/bound/move reuse remains forbidden, and no playing-
strength or product-speed claim follows from P1-H.

Raw reports:

- `target/figrid-release-0.8.2-artifacts/2026-07-25/cb-gh0-p1h/cb_gh0_p1h_authoritative.json`
- `target/figrid-release-0.8.2-artifacts/2026-07-25/cb-gh0-p1h/cb_gh0_p1h_authoritative_rerun.json`
