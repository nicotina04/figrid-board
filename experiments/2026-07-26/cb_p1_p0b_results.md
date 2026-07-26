# CB-P1 P0b exact-deduplicated bounded-DFPN result

Date: 2026-07-26 KST

Final label: **NO_GO_STATE_EXPLOSION**

CB-P1 P0b asked whether an exact bounded DFPN backend, using the current
product forcing vocabulary for attacker moves and every legal defender reply,
has enough proofability and budget sensitivity to justify a later
proofability/uncertainty budget head.

It does not. At the registered 262,144-expansion ceiling:

- 0/232 roots were `ProvenWin`;
- 159/232 were soundly certified `ExhaustedBounded`;
- 73/232 ended `UnknownMemory`;
- the memory gate allowed at most 11 roots; and
- the perfect cap-allocation oracle added 0 proofs.

The result is valid and sound. It is not a claim that the 159 bounded
disproofs are full-game losses, nor that the 73 memory-capped roots are
disproved. It says only that the exact horizon-14 query with the registered
forcing vocabulary cannot support this budget-head proposal.

## P0 lineage and fresh P0b estimand

The original P0 was preregistered at:

```text
0f0c1e483582a3586a8530342bad8a6019c775ad
```

Its implementation was committed at:

```text
cc9421612037dc362701516835839a1f2dd274d2
```

P0 made zero DFPN root calls. Its first attempt stopped in provenance before
opening the input; its second stopped in `load_roots()` because raw line 59
duplicated an exact state. It is terminally recorded as
`INVALID_CB_P1_P0`, with no solver or gate conclusion.

P0b was then preregistered as a fresh 232-unique-root estimand at:

```text
5deaae1c7dd72db1e2f38ceed69fd3722441baa8
```

It retained the byte-frozen solver core, horizon, move vocabularies, all-legal
defense rule, terminal rule, checkpoint ladder, 64 MiB per-session accounting,
certificate replay, actual-cost oracle, and no-rescue rules. Only exact-root
deduplication, its population metadata, and the population-adjusted memory
gate changed.

The independently reviewed P0b adapter was committed before the row-bearing
run at:

```text
6a89f83389b3bc1c97bf3924a10401fa31c55fd7
```

The loader-only preflight reproduced raw 307 to unique 232, checked every
dropped materialization, and asserted DFPN root calls equal to zero.

## Sole authoritative run

The clean release executable was built with:

```text
RUSTFLAGS=-C target-cpu=x86-64-v3
cargo build --release --locked --features cb-p1-audit --bin cb-p1-census
```

| artifact | bytes | SHA-256 |
|---|---:|---|
| `target/release/cb-p1-census.exe` | 1,128,448 | `0D559A9CEE6A49E21F4DFF39779A9B4AD21D2B8F3BCB115A48EE4C1FFC44CC8F` |
| authoritative P0b JSON | 2,715,747 | `60A71AE114FD3744FFEC8267EDC27D077A9BB68099131B15E60E037FE834E1EC` |
| compact scientific member | 1,388,964 | `0220FC4231FE8D719261BF66790CE49FAD633FB8B6F17700A4F7FA43FF87981C` |

The JSON was created once with exit code zero and no `.invalid` artifact.
The run took 123,324 ms.

Before and after the solver loop it reproduced:

- clean git HEAD
  `6a89f83389b3bc1c97bf3924a10401fa31c55fd7`;
- the P0b preregistration ancestor;
- the same input and manifest seals;
- 13/13 compiled critical sources equal to disk, with aggregate SHA-256
  `01EA77D50EDED0292D8921A3F4AF1EA55A91FD2E2236BDCB9A3C372125E6D50D`;
- the same executable length and SHA-256;
- rustc 1.88.0, LLVM 20.1.5, and cargo 1.88.0;
- release mode with the canonical `RUSTFLAGS`;
- compiled and runtime AVX2, BMI2, and FMA; and
- only the `cb-p1-audit` experiment feature enabled among the registered
  selectors.

The maximum process peak-working-set diagnostic recorded during checkpoints
was 149,028,864 bytes. This is diagnostic only; the scientific memory gate
uses the preregistered per-session ledger formula.

## Exact-root population

| item | observed |
|---|---:|
| raw rows | 307 |
| counted unique roots | 232 |
| Black / White roots | 91 / 141 |
| singleton exact roots | 159 |
| duplicate groups | 73 |
| size-two / size-four groups | 72 / 1 |
| excess duplicate instances | 75 |
| dropped histories given solver sessions or gate weight | 0 |

The sealed manifest is 56,742 bytes with SHA-256
`AECADEEB31F7BE5ED1DA481586D4F3A4B348A23C54393A46B9719C7FA1176086`.
The independently reconstructed retained-order digest is
`D877F832C596EFF72AD7012E238AFCD576D68D9DAE72739A5941E9ABE756055A`.

All 73 duplicate groups and 75 dropped histories matched their retained root
on Black and White bitboards, side to move, effective Freestyle rule,
line-pattern state, and Zobrist value. The 232 output UIDs, sides, ordinals,
and retained raw lines match the manifest one for one. Oracle assignments
also contain exactly those 232 roots, so duplicate multiplicity contributes
nothing to a proof or gate.

## Checkpoint census

`ExhaustedBounded` means only that the registered horizon-14 bounded query
contains no force. `UnknownNodeBudget` and `UnknownMemory` remain unknown.

| expansion cap | `ExhaustedBounded` | `UnknownNodeBudget` | `UnknownMemory` | `ProvenWin` | cumulative expansions |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 126 | 106 | 0 | 0 | 144,421 |
| 4,096 | 137 | 95 | 0 | 0 | 449,694 |
| 16,384 | 142 | 90 | 0 | 0 | 1,583,782 |
| 65,536 | 159 | 3 | 70 | 0 | 3,883,779 |
| 262,144 | 159 | 0 | 73 | 0 | 3,908,148 |

At the ceiling, the 159 bounded disproofs comprise 40 Black and 119 White
roots. The 73 memory-capped unknowns comprise 51 Black and 22 White roots.
Seventy roots had already reached the memory cap by the 65,536 checkpoint;
the remaining three did so before 77,667 per-root cumulative expansions. Their final
accounted ledgers range from 67,072,904 to 67,108,864 bytes, immediately below
or at the registered 64 MiB limit after atomic rejection of the next entry.

The cumulative-expansion distribution at the ceiling was:

| statistic | expansions |
|---|---:|
| minimum | 1 |
| p50 | 725 |
| p95 | 53,579 |
| maximum | 77,666 |

Raising the node ceiling from 65,536 to 262,144 therefore added only 24,369
expansions across the entire population: unresolved large graphs reached the
memory boundary first.

## Registered gates

| gate | observed | required | pass |
|---|---:|---:|:---:|
| correctness/certificate errors | 0 | `== 0` | yes |
| exact-state alias errors | 0 | `== 0` | yes |
| fingerprint collisions | 0 | all separated by exact equality | yes |
| memory-capped roots | **73** | `<= 11` | **no** |
| ceiling proofs | **0** | `>= 30` | **no** |
| budget-sensitive roots | **0** | `>= 10` | **no** |
| budget-sensitive Black roots | **0** | `>= 3` | **no** |
| budget-sensitive White roots | **0** | `>= 3` | **no** |
| oracle within 65,536 actual-cost budget | 0 / 3,883,779 | `<=` budget | yes |
| oracle preserves reference proofs | 0 / 0 | all | yes |
| oracle added proofs | **0** | `>= 10` | **no** |

The memory count is 73/232 = 31.466%, versus an allowed 11/232 = 4.741%;
it exceeds the fixed limit by 62 roots. Per the registered terminal
precedence, a valid run above that limit is
`NO_GO_STATE_EXPLOSION`. The other failed positive gates make the rejection
stronger but do not change its label.

The fixed 65,536 reference had zero proofs at an actual cost of 3,883,779.
No root was proved at any registered cap, so the perfect cap-allocation oracle
correctly selected no solver allocation, spent zero, preserved the empty
reference proof set, and added zero proofs. There is no observed cap-routing
opportunity for a learned uncertainty budget head.

## Correctness, certificates, and determinism

- all 159 `ExhaustedBounded` roots have independently replayed bounded-disproof
  certificates;
- missing certificates: 0;
- `UnknownMemory` roots are not serialized or counted as disproofs;
- `UnknownAbort` at the ceiling: 0;
- exact aliases, observed fingerprint collisions, and collision entries: 0;
- board and certificate restoration mismatches: 0; and
- 32 deterministic reruns: 0 scientific mismatches.

The pre-run suite passed:

| suite | result |
|---|---:|
| P0b adapter | 11 passed, 1 explicit loader preflight ignored |
| explicit loader-only preflight | 1 passed, DFPN calls 0 |
| focused DFPN core | 10/10 |
| default workspace | 127 passed, 4 ignored, plus all binary/integration/doc tests |

The byte-frozen DFPN core remained 59,306 bytes with SHA-256
`2CA097B35F666F7790955DA92F6B9C8BD068974E9C763913029CB97FE13BA4AD`.

## Interpretation and campaign boundary

This result does not show that DFPN or PNS is generally unsuitable for
Gomoku. It rejects the registered combination of:

- current product attacker forcing vocabulary;
- every-legal-move defender expansion;
- horizon 14;
- exact state identity;
- the registered session representation and 64 MiB ledger; and
- pure cap routing over the frozen checkpoint ladder.

The card forbids rescue by changing the horizon, vocabulary, subset, memory
cap, budget ladder, or threshold after seeing this result. A materially
different proof backend would need a new question and fresh preregistration;
it is not a continuation of CB-P1.

The direct product-VCT router was already frozen as
`NO_GO_VCT_PROXY: VCT_CURVE_FLAT`. P0b now also closes the exact bounded-DFPN
budget-head route before backend prototyping or training. Therefore CB-P1
does not open:

- a fixed-budget DFPN product comparison;
- a learned proofability or uncertainty head;
- proof-cache promotion or defense pruning;
- fresh grouped head-training data;
- model or artifact changes;
- search integration, arena, or release changes.

Post-sequence review also finds no inherited permission to open compiled VQ,
int8/PQ, or a bounded local mixer:

- CB-F1's exact factored i8-residual representation was safe and compact but
  slower in end-to-end scalar search, so only its exact packed-source
  packaging remains;
- CB-QAT1 found broad CE drift but failed the teacher-aligned decision
  direction needed to justify a larger quantization campaign; and
- CB-GH1 found no tractable graph state space while CB-P1 found neither a
  proof target nor a memory-viable proof substrate for a bounded mixer.

This is a conservative campaign triage, not a universal impossibility claim.
Those ideas require a genuinely new signal or backend before a fresh card can
be justified. The accepted flat Pattern4 `figrid-board 0.8.3` product,
quantized artifact, evaluator, search policy, and rollback behavior remain
unchanged.

## Independent read-only audits

Independent audits rehashed the whole JSON and compact scientific member,
reconstructed the 307-to-232 partition, matched every retained UID/side/order
entry to the manifest, confirmed that all 75 dropped instances are
non-counting, recomputed every checkpoint distribution and cumulative
expansion total, replayed the gate inequalities, and verified the
before/after executable, source-stream, input, manifest, and git identities.

No artifact, population, solver-status, certificate, oracle, gate,
serialization, or provenance blocker was found.

Raw report:

- `experiments/2026-07-26/cb_p1_bounded_dfpn_census_p0b.json`
