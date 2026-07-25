# CB-GH0 exact D4 canonical-state precondition result

Date: 2026-07-25 KST

Final P0 label: **OPEN_GH0_HASH_ONLY_TT_BLOCKED**

CB-GH0 separates two claims that a canonical D4 key can otherwise conflate:

1. exact geometric identity of the Gomoku game state; and
2. exact reuse of orientation-independent evaluator scores and search bounds.

The first claim passed. The second failed decisively for the released
0.8.2 codebook evaluator. Incremental exact-state hash implementation may
therefore proceed as P1-H, still default OFF, but canonical score/bound TT
probe or store is forbidden in this card.

This is only the registered P0 semantic precondition. No incremental hash,
canonical TT, benchmark arm, arena, product environment switch, or default
change was implemented or opened by this result.

## Frozen inputs

| Input | Bytes | SHA-256 |
|---|---:|---|
| `models/gomoku_codebook_v1_swapclosed.json` | 1,410,562 | `42968FDAB01BA8CCD1DE3DED05C532E4B237DD47EEFFD7AE1C2F264D77BA7DA2` |
| `data/topk.bin` | 17,060 | `103891DCD1DCD978C654593ABE78EF32C56E2E350B500EE665BC45AC051AA16D` |
| frozen 64-game trace | 317,511 | `1FD40D8948F113AD236FA44F5EEADCA1907C0C3103987CB4C704B67A9B47531A` |

The analyzer accepts only these three inputs and a create-new report path.
It seals every input by byte length and SHA-256 before analysis and rechecks
all three after analysis. The trace was used only for correctness witnesses;
it selected no feature, vocabulary, threshold, graph label, or training data.

## P0-H: game-state geometry

P0-H passed with zero mismatches.

- All 8 transforms were full 225-cell bijections with the registered inverse
  and composition conventions.
- Every cell in every 5x5 region agreed with the induced 3x3 region map.
- 88 maximal row, column, and diagonal lines were checked under all
  transforms: 704 line checks, 48,640 contiguous-segment checks, and 82,880
  open-end adjacency checks.
- The four rule domains and both colors produced 2,880 D4 applications of the
  registered `(side, length, open_ends)` terminal predicate with zero
  mismatches.
- The sealed trace contained 64 valid games and supplied the registered first
  1,022 product roots.
- Rebuilding all 8 full transformed histories produced 8,176 boards with
  zero mismatch in occupancy/color, side to move, effective rule, move count,
  last move, game result, legal-move set, or candidate set.
- Candidate vectors were mapped, sorted, and compared as sets; any duplicate
  candidate was an invalid run. All transformed moves were also checked for
  legality and inverse-map legality.

Selected audit counts included 3,679,200 occupancy/color checks, 1,839,600
per-cell legality checks, 180,456 history map/inverse checks, 8,176 full
legal-set checks, 8,176 candidate-set checks, and 656,800 candidate
legality/inverse checks.

The Pattern4 coordinate and boundary lemma also passed independently of model
weights: 7,200 anchor/direction/transform sequences were direct or reversed
matches, and all 4,194,304 raw 22-bit windows had identical canonical token
and released mapped ID under reversal.

## P0-TT: deployed evaluator exactness

P0-TT failed all three evaluator proof obligations that matter here.

### Region tensors are not D4 tied

The registered direct-row equality census reported 13,680 per-transform
tensor element mismatches and 576 mismatched corner/edge orbit groups.

| Tensor | t1 | t2 | t3 | t4 | t5 | t6 | t7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw f32 head | 128 | 128 | 128 | 96 | 96 | 96 | 96 |
| product i16 head | 125 | 120 | 125 | 94 | 94 | 90 | 92 |
| raw f32 factors | 1,024 | 1,024 | 1,024 | 768 | 768 | 768 | 768 |
| product i16 factors | 1,004 | 1,006 | 1,004 | 752 | 752 | 756 | 754 |

Every registered corner/edge group failed: 16/16 head groups and 128/128
factor groups, in both raw f32 and deployed i16 form.

### Released final accumulation has no structural bit-exact proof

Pattern feature construction and pooling are bounded integer operations, but
the released final forward pass accumulates linear and FM terms as f64 in
physical feature-index order. A D4 transform permutes those terms, and f64
addition is not associative. The registered structural gate is therefore
false independently of any finite sample.

### Product roots provide constructive mismatches

The released `evaluate_full_quantized` entry point was evaluated from natural
side-to-move perspective on every root and transform. Of 7,154 nonidentity
comparisons, 7,153 had a different final f32 bit pattern.

| Transform | Mismatches / 1,022 | abs diff p50 | p99 | max |
|---:|---:|---:|---:|---:|
| t1 | 1,022 | 0.0631690 | 0.495522 | 0.903526 |
| t2 | 1,022 | 0.0630763 | 0.562109 | 0.801296 |
| t3 | 1,022 | 0.0614986 | 0.581804 | 1.037494 |
| t4 | 1,021 | 0.0491407 | 0.433483 | 0.607933 |
| t5 | 1,022 | 0.0593394 | 0.546381 | 1.167387 |
| t6 | 1,022 | 0.0511603 | 0.411930 | 0.917587 |
| t7 | 1,022 | 0.0566741 | 0.445291 | 0.755915 |

The first deterministic witness was root 0, game 5, ply 5, White to move.
Its t0 value was `-1.3570354` (`0xBFADB356`); t1 was `-1.3558294`
(`0xBFAD8BD1`). The report retains a full first witness, including transformed
values and move history, for every nonidentity transform.

These are evaluator differences, not hash collisions. They demonstrate that
two exactly D4-equivalent positions cannot safely share the released
orientation-specific score or bound.

## Reproducibility and decision

Canonical build flags:

```text
RUSTFLAGS=-C target-cpu=x86-64-v3
```

Focused analyzer tests: 6 passed, 0 failed. The release analyzer executable
was 454,144 bytes with SHA-256:

```text
FD9CFD0D0F0F363AA7BD20CD8C81D036316D879C0E25126581CAFB735811AE44
```

Two independent create-new production runs produced byte-identical
36,263-byte reports:

```text
BCF021820ECF7A841B2A65A594EC99FF200E5D8EEC0A1C5C0FF287BF6098F50A
```

Raw reports:

- `target/figrid-release-0.8.2-artifacts/2026-07-25/cb-gh0/cb_gh0_p0_authoritative.json`
- `target/figrid-release-0.8.2-artifacts/2026-07-25/cb-gh0/cb_gh0_p0_authoritative_rerun.json`

The P0 decision is therefore **OPEN_GH0_HASH_ONLY_TT_BLOCKED**:

- open P1-H incremental exact D4 state hashing and its correctness/cost gates;
- do not implement canonical score/bound TT probe or store;
- do not run the registered TT hit/time A/B;
- do not infer any product speedup or playing-strength gain from this stage.

GH1 remains independent: its deliberately lossy threat-graph signature may be
used for duplicate census, tactical classification, coreset construction, and
later codebook candidates, but never as an exact proof/TT identity without
verifying the GH0 exact canonical state or the original state.
