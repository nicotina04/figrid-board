# CB-GH0 P2-H exact D4 hash-maintenance cost result

Date: 2026-07-26 KST

Final P2-H label: **HASH_CORRECT_BUT_TOO_COSTLY**

The default-OFF exact D4 state-hash sidecar remained semantically transparent
through every registered transition and search comparison. It passed the
whole-search cost gate, but failed both registered transition hot-path cost
bounds. The sidecar therefore remains default OFF and is not promoted to a
product search dependency.

This result does not open canonical transposition-table score, bound, or move
sharing, proof-cache identity by a u64 alone, an arena or playing-strength
claim, a pbrain switch, or a product-default change. It also does not block
CB-GH1's separately registered, deliberately lossy train-only graph
abstraction.

## Sealed implementation and execution

The cost harness was first committed at:

```text
5ae7489f7444ebdfb77edc8ccb7607729878c8be
```

Its first invocation stopped during source preflight because Windows returned
the canonical manifest in `\\?\C:\...` form and the process-local Git
`safe.directory` value retained that verbatim prefix. No report was created,
and scheduling setup, clock calibration, transition timing, and search timing
had not begun. The failure and correction are recorded in
`cb_gh0_p2h_preflight_incident.md`.

The corrected source preflight and incident record were committed before the
successful run:

```text
0783f9eb956fea2c10a266cac691e91e7c373113
```

The release executable was built with the registered command and environment:

```text
RUSTFLAGS=-C target-cpu=x86-64-v3
cargo build --release --locked --features codebook-eval --bin cb-gh0-hash-cost
```

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `cb-gh0-hash-cost.exe` | 1,767,936 | `23E6B2C2A33D485987173CB6A567246CA4C0BED09DCB52AB573B6BE1381753F5` |
| authoritative JSON | 125,471,100 | `B0E16983472604AF1C8207D336878FEBAB2BB1ECCC837CC0F16950A58CB9FBD2` |

The first successfully created JSON is the sole authoritative report. It was
not rerun.

## Runtime and protocol integrity

The run observed:

- AMD Family 25 Model 97 Stepping 2, 16 logical processors;
- `rustc 1.88.0`, LLVM 20.1.5, MSVC x86-64 host;
- no `NORU_*` environment variables;
- QPC frequency 10,000,000 before and after timing;
- inherited affinity mask `FFFF`, pinned timing thread on CPU 15 with mask
  `8000`;
- `HIGH_PRIORITY_CLASS` during measurement and successful restoration to the
  prior affinity and priority before postflight;
- identical preflight and postflight source, executable, input, environment,
  and clock identities.

The immediate QPC calibration contained 8,302 zero deltas among 10,000 pairs,
with p50 0 tick and p95/p99 1 tick. Clock overhead was not subtracted. Every
registered primary transition cluster and search-root timing sample was
strictly positive, so this was not an invalidation condition.

The measured arm order was exactly `A1 -> B1 -> B2 -> A2`; both transition
and search warmups used outermost `A -> B -> B -> A`. A kept the exact D4
sidecar OFF. B maintained it but no TT, proof, evaluation, ordering, VCT, or
decision path consumed it.

## Transition hot-path result

The frozen tape contained 100,000 transitions:

| Count | Observed |
|---|---:|
| makes | 50,090 |
| undos | 49,910 |
| PRNG draws | 150,090 |
| rule switches | 398 |
| consecutive paired clusters | 64 |
| repetitions per measured arm | 8 |

All 2,048 primary arm/repetition/cluster values were retained. Every
repetition reproduced all 64 registered state digests, rebuilt B's eight D4
hash lanes exactly, and unwound to the same fresh Freestyle root. Initial
enable and rule-domain rebuild timing remained in the report as diagnostics
but was excluded from the registered transition-only gate.

| Metric | Exact value | Decimal | Gate |
|---|---:|---:|---|
| point B/A | `18,946,082 / 18,614,078` | `1.017836177543` | FAIL (`<= 1.005`) |
| one-sided 95% upper | `18,946,212 / 18,537,452` | `1.022050495397` | FAIL (`< 1.01`) |

The observed transition-only overhead was **+1.783617754%**, with a
one-sided upper of **+2.205049540%**. At 10 MHz, the paired A and B totals
were 1.8614078 s and 1.8946082 s across 1.6 million transitions per side.
The diagnostic mean delta was 20.75025 ns per transition.

The exact registered products were:

```text
point: 3,789,216,400 <= 3,741,429,678  false
upper: 1,894,621,200 <  1,872,282,652  false
```

## Whole-search result

The frozen workload contained 64 games and 1,022 roots. Every arm retained
all 1,022 root observations, for 4,088 raw search results. Across all arms,
root identity, best move, score, completed depth, returned/main/qsearch node
counts and their exact sum, TT counters and occupancy, and the complete
restored board signature were identical.

| Metric | Exact value | Decimal | Gate |
|---|---:|---:|---|
| point B/A | `336,350,832 / 336,029,400` | `1.000956559158` | PASS (`<= 1.005`) |
| one-sided 95% upper | `339,009,348 / 337,656,471` | `1.004006666882` | PASS (`< 1.01`) |

The whole-search overhead was **+0.095655916%**, with a one-sided upper of
**+0.400666688%**. Paired A and B totals were 33.6029400 s and 33.6350832 s
across 2,044 roots per side, a diagnostic mean delta of about 15.726 us per
root.

The exact registered products were:

```text
point: 67,270,166,400 <= 67,541,909,400  true
upper: 33,900,934,800 <  34,103,303,571  true
```

## Independent artifact audit

Two independent read-only audits parsed the authoritative JSON without
running the harness or any timing workload.

They:

- reconstructed all 64 transition and 64 search ABBA pairs from raw samples;
- matched all 128 reported pairs element by element;
- replayed both SplitMix64 bootstraps for 100,000 replicates × 64 draws;
- reproduced exact rational sorting at index 94,999;
- reproduced final RNG states `528B16E7EC87CA01` and
  `528B16E7EC87CA02`;
- directly rehashed the 16 critical sources, five frozen inputs, and
  executable against both report seals;
- checked every root's four-arm identity, search result, node accounting, TT
  state, and full restored board;
- reproduced the decision precedence and final label.

The broader integrity audit executed 31,787 checks with zero errors. The
report contained zero invalid failures and zero exactness failures.

## Decision and next boundary

P2-H requires both registered metrics to pass both bounds. Search passed, but
transition maintenance did not. The final decision is therefore:

```text
HASH_CORRECT_BUT_TOO_COSTLY
```

The exact D4 implementation may remain as a default-OFF observer and research
primitive, but it is not enabled in the product search path. Canonical TT
score/bound/move sharing remains blocked.

CB-GH1 may proceed independently for duplicate census, tactical structure
classification, coreset construction, and later codebook candidate
generation. Its lossy graph signature is never an exact TT or proof key, and
the frozen 1,022-root correctness/performance trace is not eligible for graph
vocabulary selection.

Raw report:

- `target/figrid-release-0.8.2-artifacts/2026-07-25/cb-gh0-p2h-cost/cb_gh0_p2h_cost_authoritative.json`
