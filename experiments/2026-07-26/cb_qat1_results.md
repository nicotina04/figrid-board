# CB-QAT1 integer-lattice headroom result

Date: 2026-07-26 KST

Final label: **NO_GO_PRECONDITION**

CB-QAT1 asked whether the accepted flat Pattern4 product loses enough
teacher-aligned K=6 decision quality when mapped from its FP32 weights to the
deployed `E32/H64/F64` integer lattice to justify one expensive paired
PTQ-control/QAT fit.

RQ615C train is already-consumed local diagnostic material, not fresh K=6
validation evidence. Its only role here is the preregistered P0 open/stop
decision.

The broad cross-entropy headroom exists and is component-robust:

- `CE(PTQ)-CE(FP32) = +0.00036551997282722174`;
- component-bootstrap p10 is `+0.00015044667369783807`;
- Black and White CE deltas are both positive; and
- FP32 and PTQ choose different top moves on 53 of 1,336 slates.

The registered decision-direction gates fail, however. On those 53
disagreements, FP32 chooses the higher-`q_teacher` candidate 21 times, PTQ
does so 24 times, and eight are equal. The combined FP32 net is therefore
`-3`, below the required `+2`. White roots are the decisive failure:
`7-13 = -6`, below the required nonnegative net.

Accordingly, the scale-64 lattice is measurably different and slightly worse
in aggregate CE, but the observed top-choice changes do not point in the
teacher-preferred direction. Per preregistration, this closes CB-QAT1 before
training. No 6,163,315-sample extraction, paired fit, validation, artifact,
correctness/timing trace, game, or promotion stage was opened.

## Preregistration and implementation

The experiment was preregistered before implementation or results:

```text
55aee08  Preregister CB-QAT1 integer-lattice experiment
```

A pre-implementation audit found that the registered `Cargo.lock` SHA-256 was
correct but its copied byte length was not. Before any production-corpus run,
the factual field alone was corrected from 20,498 to 11,841 bytes:

```text
c08aa68  Correct CB-QAT1 Cargo lock provenance
```

No input, statistic, seed, threshold, or decision rule changed. The
authoritative analyzer was then implemented and independently audited at:

```text
acad6e9f5097daacb5603d40809d271017568b8c
```

The implementation fail-closes on the registered lock length/SHA, rustc,
LLVM, Cargo, release profile, debug assertions, compiled/runtime AVX2/BMI2,
canonical `RUSTFLAGS`, forbidden environment variables, dirty source,
compiled-source drift, executable drift, input drift, and output overwrite.

Synthetic and regression tests before the run:

| suite | result |
|---|---:|
| `cb-qat1-headroom` | 35/35 |
| existing `cb-gh1-graph-census` | 30/30 |

No production corpus was opened by these tests.

## Sole authoritative run

The clean release executable was built with:

```text
RUSTFLAGS=-C target-cpu=x86-64-v3
cargo build --release --locked --features codebook-eval --bin cb-qat1-headroom
```

The first successfully created report is the sole authoritative P0 result.
The workload was not rerun.

| artifact | bytes | SHA-256 |
|---|---:|---|
| `target/release/cb-qat1-headroom.exe` | 3,567,104 | `A4BDFE5F0D0668B6E94DF2DBCC802FD1546B6B46C6E3A8E1B12CDF9A815A2378` |
| authoritative P0 JSON | 385,020 | `B8014472B3A5B64ADFCE2074362F7D5F0483CF71021A9A8E16E4BF1256D87E26` |

The run completed in 15,548 ms. It recorded:

- git HEAD `acad6e9f5097daacb5603d40809d271017568b8c`;
- preregistration/erratum ancestor `c08aa68`;
- 43/43 compiled critical-source seals matching a clean worktree;
- `RUSTFLAGS=-C target-cpu=x86-64-v3`;
- compiled and runtime AVX2, BMI2, and FMA;
- rustc 1.88.0, LLVM 20.1.5, and cargo 1.88.0;
- no `NORU_*`, `FIGRID_*`, `RAYON_*`, or registered instrumentation
  overrides; and
- all five input seals unchanged before and after analysis.

The PowerShell launch wrapper emitted a non-terminating environment-provider
duplicate-key warning before starting the executable. This did not supply or
hide a load-bearing setting: the executable independently enumerated and
rejected forbidden variables, recorded an empty forbidden-variable list, and
verified the exact canonical `RUSTFLAGS`. The report was created normally
with exit code zero.

### Telemetry-schema audit note

One read-only audit found a reporting-only omission: the JSON does not repeat
`debug_assertions=false` and `target_profile=release` as dedicated scalar
fields. Those facts are nevertheless cryptographically reconstructible from
the same report because it seals:

- the critical source implementing a hard failure when
  `cfg!(debug_assertions)` is true;
- the hard requirement that the executable parent directory is `release`;
- the full release executable path, byte length, and SHA-256; and
- the canonical release build command and compiled target features.

The executable could not have produced this report had either condition
failed. Because the result was already visible, the analyzer was not patched
and rerun merely to duplicate these facts into two convenience fields. This
omission is non-load-bearing, is disclosed here, and cannot rescue or weaken
the conservative no-go decision.

## Frozen inputs and replay

| input | bytes | SHA-256 |
|---|---:|---|
| product FP32 codebook | 1,410,562 | `42968FDAB01BA8CCD1DE3DED05C532E4B237DD47EEFFD7AE1C2F264D77BA7DA2` |
| Pattern4 `topk.bin` | 17,060 | `103891DCD1DCD978C654593ABE78EF32C56E2E350B500EE665BC45AC051AA16D` |
| RQ615C train | 54,991,200 | `E00A2DA513B05D7631A01003C7E6274E9A3D7575E2C2BD92D5199F1B5385CEB6` |
| RQ615C manifest | 5,463 | `579D1387D7E4DE8F5CB34DB168B6D15655DB229D992751B1DC17BB6CF4260AA7` |
| RQ569 lineage model | 1,413,542 | `69BB7C599ADA3A1151577CE3315356BC33C40EDB49A003C9BC4EB90A98F82E18` |

The shared A0 loader reproduced:

- 1,336 slates, 8,016 K=6 candidates, 388 components;
- 668 Black and 668 White roots;
- ordinals `{1,2,4,6,8}`;
- 285,900 complete legal-inventory children;
- product/RQ569 FP32 payload identity;
- zero public-versus-independent quantized forward mismatch;
- zero stored lineage, child-hash, or Pattern4 mapping mismatch; and
- finite FP32 and PTQ utilities for every candidate.

RQ615C dev/safety, RQ508, the frozen 1,022-root trace, game outcomes, and
arena/Pela artifacts were not opened.

## Registered gate result

| gate | observed | required | pass |
|---|---:|---:|:---:|
| prerequisite mismatches | 0 | `== 0` | yes |
| combined `delta_ce` | +0.00036551997282722174 | `> 0` | yes |
| component-bootstrap p10 `delta_ce` | +0.00015044667369783807 | `> 0` | yes |
| Black `delta_ce` | +0.00030842406640208084 | `>= 0` | yes |
| White `delta_ce` | +0.00042261587925236264 | `>= 0` | yes |
| FP32/PTQ top-1 disagreements | 53 | `>= 7` | yes |
| combined FP32 q-superiority net | -3 | `>= 2` | **no** |
| Black FP32 q-superiority net | +3 | `>= 0` | yes |
| White FP32 q-superiority net | -6 | `>= 0` | **no** |

Seven of nine conjunctive conditions pass. The two failures make
`all_gates_pass=false` and deterministically select `NO_GO_PRECONDITION`.

### Decision diagnostics

| stratum | slates | `delta_ce` | top-1 disagreements | FP32/PTQ/equal q-superiority | FP32 net |
|---|---:|---:|---:|---:|---:|
| combined | 1,336 | +0.00036551997282722174 | 53 | 21 / 24 / 8 | -3 |
| Black | 668 | +0.00030842406640208084 | 30 | 14 / 11 / 5 | +3 |
| White | 668 | +0.00042261587925236264 | 23 | 7 / 13 / 3 | -6 |

The lowest-index-tie-break `q_teacher` argmax accuracy is 309/1,336 for FP32
and 313/1,336 for PTQ. Stored `teacher_top` accuracy is tied at 297/1,336.
Thus the failed q-direction gate is not a count artifact hidden by a broad CE
average.

There are 457 changed pair orderings among the 20,040 within-slate candidate
pairs. Candidate-level absolute drift is:

| metric | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| logit | 0.02420032 | 0.05492616 | 0.06532866 | 0.08886409 | 0.16576278 |
| six-way probability | 0.00134677 | 0.00357806 | 0.00446935 | 0.00642852 | 0.01107781 |

The 100,000-replicate component bootstrap used seed `2026726001`, sampled
388 uppercase-UID-lexicographic components with `next_u64()%388`, and used
the registered p10 element 9,999. Its range was
`[-0.00041619191204290984, +0.001020600327103515]`.

## Independent read-only audits

Two audits inspected the sole JSON without rerunning the analyzer:

1. The statistical audit re-evaluated all nine inequalities, reconciled every
   color/ordinal/component count, reproduced the combined metrics from
   strata, checked q-category conservation, and confirmed the final label.
2. The provenance audit rehashed 43/43 critical sources, the executable, all
   five inputs, `Cargo.lock`, and the JSON; verified clean ancestry,
   toolchain, features, environment, and create-new output semantics; and
   confirmed every P1-P4 output remains absent.

Apart from the explicitly disclosed duplicate telemetry fields, no issue was
found.

## Interpretation and next boundary

This is not evidence that QAT can never help a codebook. It is evidence that
the current product, current `E32/H64/F64` lattice, and current sealed K=6
teacher distribution do not provide the preregistered decision-aligned
headroom needed to justify a full-surface paired QAT campaign.

The result specifically forbids:

- paired PTQ/QAT fitting;
- using RQ508 or any fresh validation to rescue QAT;
- packing PTQ/QAT CBF artifacts;
- opening the frozen correctness/timing trace;
- QAT/PTQ or QAT/incumbent games; and
- any product or default change.

The accepted flat Pattern4 product, factored CBF packaging, integer runtime,
White-root ordering, evaluator scale, search policy, and rollback behavior
remain unchanged.

CB-QAT1 is closed. Per the registered sequence, the next permitted card is
CB-AL1.

Raw report:

- `target/figrid-release-0.8.2-artifacts/2026-07-26/cb-qat1-p0/cb_qat1_p0_headroom.json`
