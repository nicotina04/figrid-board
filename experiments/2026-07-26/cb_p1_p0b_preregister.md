# CB-P1 P0b exact-deduplicated bounded-DFPN precondition

Date: 2026-07-26 KST

Status: preregistered before any CB-P1 row-bearing DFPN root invocation

Product baseline: `figrid-board 0.8.3`, commit
`a3efbbe26d507e2d0843948897cfead230d0a70e`

Implementation baseline: commit
`cc9421612037dc362701516835839a1f2dd274d2`

This is a fresh audit-only prerequisite. It does not amend or continue the
invalid P0 run, and it does not authorize a learned head, product integration,
branch pruning, proof-cache promotion, model change, arena, or release change.

## P0 lineage and absence of outcome access

The original preregistration is:

| artifact | bytes | SHA-256 |
|---|---:|---|
| `experiments/2026-07-26/cb_p1_bounded_dfpn_preregister.md` | 16,275 | `655E71928F41FF469D095AB1E30F08A3C1FBD5AA49D283C4FF2A809604802DD0` |

It was committed as
`0f0c1e483582a3586a8530342bad8a6019c775ad`.

P0 is terminally recorded as `INVALID_CB_P1_P0` in:

| artifact | bytes | SHA-256 |
|---|---:|---|
| `experiments/2026-07-26/cb_p1_p0_invalid.md` | 3,435 | `C10282D2B5C214CE3C34C1DF8E91D7DD7527927052D71ED9D526F586EA536E6B` |

The first attempt stopped in provenance before opening the input. The second
attempt sealed the input but stopped in `load_roots()` at raw line 59 because
the P0 exact-uniqueness assertion was false. `load_roots()` had to finish
before the first `run_root()` call. Therefore both attempts together made
exactly zero DFPN root calls and observed no proof status, checkpoint cost,
certificate, oracle assignment, or gate value.

The schema audit and deduplication manifest used only the six P0-permitted
top-level fields and structurally skipped every forbidden value. It did not
invoke DFPN. This P0b is therefore fixed before any solver outcome.

## Normative inheritance and replacements

Except for the explicit replacements below, P0 is incorporated by reference
in full. In particular, P0b inherits unchanged:

- the prior-result boundary and `NO_GO_VCT_PROXY: VCT_CURVE_FLAT`;
- the bounded `ProvenWin`/`ExhaustedBounded` claim boundary and all forbidden
  label, result, score, VCT-verdict, model, and corpus inputs;
- the exact OR forcing vocabulary and order;
- every-legal-move AND expansion and ascending cell order;
- complete-bitboard Freestyle terminal scans and horizon 14;
- exact state identity and policy digest;
- proof-number, disproof-number, threshold, saturation, and tie invariants;
- cumulative checkpoints
  `1,024 / 4,096 / 16,384 / 65,536 / 262,144`;
- per-root 64 MiB accounted-memory formula and atomic ledger commitment;
- watchdog, checkpoint fields, exact-collision separation, and restoration;
- independent proof and bounded-disproof certificate replay;
- exhaustive-reference, collision, interruption/resume, and restoration tests;
- perfect-oracle caps, actual-expansion costs, preservation constraint,
  integer optimization, and retained-order lexicographic tie-break;
- the ban on alternate horizon, vocabulary, budget ladder, memory cap,
  outcome-selected subset, or threshold rescue; and
- `GO_PROTOTYPE` authorizing only another preregistered audit-only backend
  comparison, never a learned head directly.

Only these P0 clauses are replaced:

1. P0's exact-unique 307-root assertion and its 307 counted sessions are
   replaced by the sealed first-occurrence manifest and 232 counted roots.
2. Every scientific denominator and oracle population is the 232 retained
   exact roots; 307 remains only the raw-row diagnostic.
3. The original positive absolute gates remain 30, 10, 3+3, and oracle +10.
4. The memory-root gate becomes
   `floor(15 * 232 / 307) = 11`.
5. The registered JSON output and result document use the P0b names below.
6. P0 terminal label `INVALID_CB_P1_P0` becomes
   `INVALID_CB_P1_P0B` for this card.
7. P0's implementation critical-source list is replaced by the P0b list and
   tightly scoped adapter/package allowlist below.

If this document and P0 appear to conflict outside those seven replacements,
P0 controls and the run is invalid.

## Frozen raw input and dedup manifest

The sole raw row-bearing input remains:

| input | bytes | SHA-256 |
|---|---:|---|
| `rq547a_tactical_positions.jsonl` | 309,683 | `F02663E51716A13F54E0AB22829F7B6FBC7D237F843FAA79BCF62CE3A8EA171F` |

Registered path:

```text
C:\Users\concreate\Documents\workspace\noru-tactic\experiments\2026-07-05\rq547a_tactical_positions.jsonl
```

The frozen selection manifest is:

| artifact | bytes | SHA-256 |
|---|---:|---|
| `experiments/2026-07-26/cb_p1_p0b_dedup_manifest.json` | 56,742 | `AECADEEB31F7BE5ED1DA481586D4F3A4B348A23C54393A46B9719C7FA1176086` |

Its registered census is:

```text
raw rows                    307
unique exact roots          232
retained Black roots         91
retained White roots        141
singleton roots             159
duplicate groups             73
excess duplicate instances  75
```

There are 72 duplicate groups of size two and one group of size four. The
manifest's retained-order SHA-256 is:

```text
D877F832C596EFF72AD7012E238AFCD576D68D9DAE72739A5941E9ABE756055A
```

The exact key, UID construction, and retained-order encoding are exactly those
declared in the sealed manifest. For each exact
`(black, white, side-to-move, Freestyle)` key, retain only the lowest 1-based
raw line. Retained roots stay in first-occurrence file order. No replacement
root is added. No row may be selected, removed, reordered, or weighted using
color balance, difficulty, solver cost, history length, label, result, score,
actual move, historical VCT verdict, or any DFPN value.

The adapter must independently reconstruct all 307 raw rows, recompute the
232-root partition and retained-order digest, and compare every manifest
field before any DFPN call. A mismatch is `INVALID_CB_P1_P0B`.

## Non-counted duplicate-history materialization audit

Before any DFPN call, every raw history in all 73 duplicate groups is replayed
under the unchanged P0 legality and ongoing-root rules. Within each group,
every materialization must match the retained first occurrence in:

- Black and White bitboards;
- side to move and effective `RuleSet::Freestyle`;
- complete line-pattern state/cache; and
- Zobrist identity.

`position_history` and `last_move` are deliberately excluded from equality:
different legal histories may reach the same exact solver state. Dropped
histories receive no DFPN session, checkpoint, certificate, oracle cost,
proof count, gate weight, or output row. Any materialization mismatch is
`INVALID_CB_P1_P0B`, and the solver loop must remain unentered.

## Counted sessions, determinism, and oracle

Exactly 232 sessions are run, in sealed retained order. Root ordinal and
lexicographic oracle ties use that order. The first 32 retained roots and all
ceiling positives are rerun in fresh sessions under the unchanged P0
determinism rules.

For each retained root, `minimum_proof_cap`, `actual_cost(root, cap)`, and
status are defined exactly as in P0. The fixed-reference total budget is now:

```text
sum(actual_cost(root, 65,536)) over the 232 retained unique roots
```

The perfect oracle assigns one of
`0 / 1,024 / 4,096 / 16,384 / 65,536 / 262,144` to each of those 232 roots,
must preserve every unique-root 65,536-reference proof, and uses the same
actual-cost constraint and retained-order tie-break. Duplicate multiplicity
never contributes budget or score.

## Fixed gates

`GO_PROTOTYPE` requires all of:

1. raw-input, manifest, dedup, materialization, provenance, implementation,
   source, serialization, correctness, determinism, and certificate errors
   are zero;
2. fingerprint aliases are zero and every observed collision is separated by
   exact equality;
3. at most **11/232** roots end in `UnknownMemory`, and no positive
   certificate touches an incomplete ledger;
4. at least **30/232** roots are proved at the 262,144 ceiling;
5. at least **10/232** roots are unproved at 4,096 and proved at a later
   checkpoint;
6. those budget-sensitive roots contain at least **3 Black** and at least
   **3 White** roots; and
7. the perfect oracle preserves every 65,536-reference proof and adds at
   least **10/232** proofs.

The positive counts are intentionally not scaled down from P0. Oracle +10 is
4.310 percentage points on 232 unique roots. The memory limit is scaled
downward without relaxation:

```text
floor(15 * 232 / 307) = 11
```

No raw-row multiplicity may satisfy a gate.

## Solver-core and implementation freeze

The DFPN solver core from implementation commit `cc94216` is byte-frozen:

| path | bytes | SHA-256 |
|---|---:|---|
| `src/vct/dfpn.rs` | 59,306 | `2CA097B35F666F7790955DA92F6B9C8BD068974E9C763913029CB97FE13BA4AD` |

The following supporting scientific sources must also remain byte-identical
to the P0 implementation:

| path | bytes | SHA-256 |
|---|---:|---|
| `Cargo.lock` | 11,841 | `6A6B62449A235ABA53C777484C5D34E18EDB556155B1964A4B2BA6DA7DE2059C` |
| `src/lib.rs` | 2,750 | `78186D8290AB9EA8A83DFEEF31942C55AD323A9DDC9ABFE573E21CB5CD6F0F6A` |
| `src/board.rs` | 62,634 | `E233F3550710768D9861043C181DDE890F1559F276DCAD1A1E01B6E86C009038` |
| `src/vct.rs` | 114,527 | `50F30D02874E94165516B4E32810612D4C185F86BB43B406A7A65455633B5853` |
| `src/pattern_table.rs` | 23,045 | `E16B06744A02AE6DDDDBECE2CC3C15DCB7D65A7FACBB67FA1A70A72E9992DE93` |
| `bin/cb_al1_selector/hash.rs` | 12,601 | `C78991EB3FB4405BA34716AF9CC87F274246453DC3DE6ABEDE09A630AF62D773` |

At registration, `Cargo.toml` is 6,115 bytes with SHA-256
`3292481D808215CD5AD220B5E48FDB35A8AC350837B32EC77E0933277DA42FD1`.
It may change only by adding package-exclusion negations for the P0 invalid
record, P0b manifest, and P0b preregistration so the feature-gated audit
binary remains buildable from the packaged crate. No dependency, feature,
binary, profile, version, or other package setting may change.

At registration, `bin/cb_p1_census.rs` is 71,264 bytes with SHA-256
`1C76CBF7674A2474FC2B08CA88EF913236DA56706F66D8DD5BE4BC13F839CC09`.
Only after this P0b preregistration is committed may `Cargo.toml` and that
adapter change, and only to:

- add the three package-exclusion negations just described;
- verify and apply the sealed manifest;
- perform the non-counted materialization audit;
- change 307 counted rows to 232 retained rows;
- change the fixed gates and oracle population exactly as above;
- bind the P0b preregistration, lineage, manifest, output name, and terminal
  labels into provenance;
- add only the required P0b top-level scientific population, manifest,
  retained-order, and duplicate-materialization metadata; and
- repair adapter-only path/provenance handling.

`run_root`, DFPN configuration, checkpoint semantics, certificate replay,
the existing per-checkpoint and per-root solver-field serialization,
actual-cost definition, and oracle optimization may not change. The new
top-level P0b metadata required under **Registered outputs** is the sole
scientific-serialization exception. The final adapter diff from `cc94216`
must be reviewed against this allowlist.

After implementation, a label-blind loader-only preflight test is expressly
allowed. It may open only the sealed raw input and manifest, execute schema,
legality, dedup, retained-order, and materialization checks, assert
`raw=307`, `counted=232`, and then terminate before the solver loop. It must
record and assert **DFPN calls = 0**. It may not emit or inspect any proof,
cost, certificate, oracle, or gate outcome. A row-bearing census run is
forbidden until the adapter is committed and every required test passes.

## Critical-source stream and tests

The P0b critical-source stream contains exactly, in this order:

```text
Cargo.toml
Cargo.lock
src/lib.rs
src/board.rs
src/vct.rs
src/vct/dfpn.rs
src/pattern_table.rs
bin/cb_p1_census.rs
bin/cb_al1_selector/hash.rs
experiments/2026-07-26/cb_p1_bounded_dfpn_preregister.md
experiments/2026-07-26/cb_p1_p0_invalid.md
experiments/2026-07-26/cb_p1_p0b_dedup_manifest.json
experiments/2026-07-26/cb_p1_p0b_preregister.md
```

Each path and file body is length-delimited before hashing. The executable
records the stream digest and every individual file's bytes and SHA-256.
Source, raw input, manifest, solver-core, and preregistration seals are checked
both before the first DFPN call and after the last.

Required tests before the census are:

- all P0 DFPN focused tests and exhaustive-reference comparisons;
- default workspace tests and feature-enabled library tests;
- manifest seal and 307-to-232 first-occurrence selection;
- retained-order digest and 91/141 side counts;
- duplicate-history materialization equivalence;
- loader-only preflight proving zero DFPN calls;
- fixed 30/10/3+3/+10 gates and memory threshold 11;
- unique-root oracle budget and retained-order tie-break; and
- create-new output and every terminal-label/exit-code path.

## Build and runtime provenance

The P0 toolchain is inherited unchanged:

- `rustc 1.88.0 (6b00bc388 2025-06-23)`;
- host `x86_64-pc-windows-msvc`, LLVM `20.1.5`;
- `cargo 1.88.0 (873a06493 2025-05-10)`.

The default-off feature and binary remain:

```text
feature: cb-p1-audit
binary:  cb-p1-census
```

The clean committed release build remains:

```powershell
$env:RUSTFLAGS = '-C target-cpu=x86-64-v3'
cargo build --release --locked --features cb-p1-audit --bin cb-p1-census
```

The executable must require release mode, `debug_assertions=false`, runtime
and compile-time AVX2/BMI2/FMA, exact raw-input and manifest seals, all frozen
source seals, a clean worktree before output creation, and a 40-hex HEAD
descending from the commit that contains this P0b preregistration. `NORU_*`,
`FIGRID_*`, `RAYON_*`, profiling, coverage, wrapper, bootstrap, and
incremental-compilation environment variables remain forbidden.

The executable records compiled and disk executable bytes, input and manifest
seals, command line, canonical working directory, git HEAD, ancestry,
toolchain, CPU features, environment audit, start/end time, peak working-set
diagnostic, critical-source stream, and output seal.

## Registered outputs

The sole scientific JSON is:

```text
experiments/2026-07-26/cb_p1_bounded_dfpn_census_p0b.json
```

The human decision record is:

```text
experiments/2026-07-26/cb_p1_p0b_results.md
```

The JSON must report raw rows 307, counted roots 232, manifest identity,
retained-order identity, non-counted duplicate audit totals, 232 per-root
records, summaries, oracle assignment, gates, provenance, and a deterministic
scientific-member seal. Forbidden raw fields may not appear.

The registered path uses create-new semantics and must not exist before the
run. Any partial output from an invalid or unsound execution is retained under
a non-registered `.invalid` suffix; the registered path remains absent.
Source, input, or manifest mutation after the run invalidates the output.

## Terminal decisions and exit taxonomy

- schema, raw-input, manifest, dedup, retained-order, materialization,
  provenance, implementation, frozen-source, build, environment,
  serialization, output, or incomplete-run failure:
  `INVALID_CB_P1_P0B`, exit code 1, no scientific conclusion;
- false proof, missing legal defense, wrong terminal, exact-state alias,
  unknown-as-solved reuse, incomplete certificate ledger, or restoration
  mismatch: `UNSOUND`, exit code 2, and the card stops;
- a valid run with more than 11 memory-capped counted roots:
  `NO_GO_STATE_EXPLOSION`, exit code 0;
- a valid run within the memory gate but missing any other fixed gate:
  `NO_GO_PRECONDITION`, exit code 0;
- a valid run passing every fixed gate:
  `GO_PROTOTYPE`, exit code 0.

`UnknownAbort` at the ceiling is an incomplete run and therefore invalid.
`UnknownNodeBudget` at a registered finite checkpoint is the normal unresolved
status inherited from P0 and is not a disproof. No failed label may be changed
by rerunning a different subset, threshold, horizon, policy, or budget.
