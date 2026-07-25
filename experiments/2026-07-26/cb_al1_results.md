# CB-AL1 label-blind active-distillation precondition result

Date: 2026-07-26 KST

Final label: **INVALID_CB_AL1_P0**

CB-AL1 was designed as a two-stage retrospective precondition:

1. P0A would select fixed active and deterministic-control arms without
   opening RQ615C train labels.
2. P0B would first reproduce and authenticate P0A, then reveal the already
   consumed train labels and evaluate whether the label-blind active arm
   discovers more K=6-local usable errors than the control.

P0A completed successfully and returned `P0A_READY_FOR_REVEAL`. The sole P0B
attempt then failed in a pre-label byte-canonicality guard. No train label,
final manifest, or lineage model was opened or hashed by P0B, and no P0B
output was created.

This is an implementation-validity failure, not a scientific GO/NO_GO result.
It provides no evidence for or against active distillation.

## Frozen registration and implementation

The preregistration and pre-run source-list erratum were committed before any
row-bearing P0A or P0B execution:

```text
c585617  Preregister CB-AL1 active selection precondition
5c63f04  Correct CB-AL1 preregistered source stream
```

The independently audited implementation was committed at:

```text
675dbd596559cdf2b2a9725299d8f07cdb4540df
```

The final static audit found no remaining schema, ordering, label-firewall,
join, statistical, RNG, hash, gate, provenance, path, or output-finality
violation. Its synthetic suite passed 37/37 tests. Those tests did not open
the prepared or train row corpora.

## Canonical build and P0A

The clean registered release executable was built once with:

```text
RUSTFLAGS=-C target-cpu=x86-64-v3
cargo build --release --locked --features cb-al1-audit --bin cb-al1-selector
```

| artifact | bytes | SHA-256 |
|---|---:|---|
| `target/release/cb-al1-selector.exe` | 2,039,296 | `E77149E3ED52013C78A195324F233DE4C1B6BFBBA784ABFB86C5A725ECF82FC8` |
| sole P0A selector JSON | 294,480,214 | `91AD5ADD0DF0C2312A9C487068CFE461C68BD5753D8ED6662968ADA4DCDAA265` |

P0A ran for 76,371 ms and recorded:

- 1,000 paired units, 2,000 parents, and 428,320 full-legal children;
- 2,000 root factored/flat parity checks;
- 428,320 child factored/flat parity checks;
- 428,320 legacy child replays;
- bit-identical product JSON/CBF source payloads and reconstructed-flat/current
  quantized payloads;
- 100 support units and 25 selected units in each ordinal stratum;
- 125 active and 125 deterministic-control units;
- active/control overlap of 33 units;
- 115 distinct active openings and 114 distinct control openings; and
- status `P0A_READY_FOR_REVEAL`.

The ordered selector stream digests were:

| stream | SHA-256 |
|---|---|
| support | `4CDEC67430BBC4D3726271A239E982159B72B989BEFF6BE196E02D6BA949A7D0` |
| active | `C291AA97A57012F67191FF9BC25A4577638E1BB7BF13F1040A24036164DE5C62` |
| deterministic control | `86570D2F794AEDB267853E0B879DD5D7D28C56C5ECFEF30A6BF2C630CD80E0FA` |

P0A was answer-opaque. It did not accept a train-label path and did not open
RQ615C train, the final manifest, or the lineage model.

## Sole P0B attempt and failure

Without rebuilding or changing the worktree, the exact P0A byte count and
digest printed above were copied into the registered P0B command.

P0B exited 1 before its first label-bearing file operation:

```text
CB-AL1 INVALID_CB_AL1_P0: P0A selector bytes are not canonical pretty JSON plus one terminal LF
```

The failure occurred after P0A seal, JSON parse, and attempted canonical
re-serialization, but before `p0b_label_seals`, corpus loading, joining,
statistics, or output creation. The registered P0B output path remains absent.

## Root cause

The guard incorrectly assumed that this transformation is byte-idempotent:

```text
f32 passed into json! and promoted to an exact f64 JSON number
  -> to_vec_pretty
  -> parse through serde_json's default f64 path
  -> to_vec_pretty
```

It is not under the frozen serde_json build. `json!` immediately promoted
each binary32 value to its exact binary64 value. The first serialization
therefore emitted the promoted value's decimal representation. The default
serde_json parser, built without `float_roundtrip`, reconstructed one such
decimal as the adjacent binary64 value, whose next serialization was shorter.

A read-only diagnostic reproduced the exact failure:

| field | observed |
|---|---:|
| original bytes | 294,480,214 |
| parse/re-serialized bytes | 294,398,402 |
| first differing byte offset | 169,316 |
| original numeric text | `1.0229682922363281` |
| re-serialized numeric text | `1.022968292236328` |
| recorded binary32 bits | `3F82F0A0` |

Thus the P0A artifact was produced by the registered `to_vec_pretty + LF`
writer, but P0B's parse/re-serialize test was not a sound way to authenticate
that fact. P0A had already performed a second byte-identical serialization
from the original typed value before its create-new write.

## Decision and boundary

Per the preregistered invalidity and no-retry rules:

- the same P0A/P0B card was not retried;
- the analyzer was not rebuilt between P0A and the sole P0B attempt;
- no alternate seed, selector, threshold, support, or byte rule was used;
- no train label or result statistic was inspected;
- no AL1 teacher query, training, model, artifact, product default, or
  promotion step was opened; and
- neither `GO_FRESH_AL1_PREREG_ONLY` nor any scientific no-go label may be
  claimed.

A future AL1 attempt requires a fresh preregistration. Its P0A authentication
must not parse and re-serialize binary32 JSON numbers as generic binary64
numbers. Suitable fresh designs include authenticating the literal P0A seal
plus its in-process duplicate serialization, or encoding load-bearing
binary32 values as exact bit strings.

CB-AL1 is closed as `INVALID_CB_AL1_P0`. Because the card was observer-only,
default-off, and made no product mutation, the accepted pre-AL1 product
candidate remains unchanged and may proceed to the 0.8.3 release checks.

Raw artifact:

- `target/figrid-release-0.8.2-artifacts/2026-07-26/cb-al1-p0a/cb_al1_p0a_selector.json`
