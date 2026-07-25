# CB-TD1 reversible TokenDelta journal results

Date: 2026-07-25 (Asia/Seoul)

Decision: **GO_INTERNAL_EXTRACT — promote the private reversible journal as
the sole internal CB-D1 implementation.**

This is an architecture promotion. It makes no speed or playing-strength
claim.

## Changed factor

Control A was the promoted CB-D1 evaluator at commit
`74c93bca23c031b1401d63acdf1831dbd782e805`: logical Pattern4 IDs plus an
evaluator-specific per-ply undo ledger. Candidate B performed the same integer
updates through a private generic journal:

```text
TokenDelta { site: u16, lane: u8, old: T, new: T }
```

The final implementation preallocates fixed-capacity frames, owns logical and
materialized depths, batches replay by site, and delegates numeric work to a
monomorphized sink. `TokenDelta<u16>` is eight bytes. The journal imports no
board, Pattern4, codebook, pooling, or search type.

After promotion there is no direct/journal product selector. CB-D1 ON uses the
journal exclusively; `NORU_CODEBOOK_DIRECTIONAL_DELTA=off` still restores the
legacy full-cell evaluator. Library `Searcher` callers remain default-OFF and
the pbrain remains default-ON. No released public API, model asset, serialized
format, dependency, feature default, evaluator arithmetic, move order, VCT
rule, or time policy changed.

The restored `cb-d1-ab-v2` harness retains commit `74c93bc` in its seal as the
start of this extraction card, not as the pre-CB-D1 legacy control.

## Correctness

| Gate | Result |
|---|---|
| Generic journal tests | 6 passed: pending-prefix replay, materialized/unmaterialized pop, reverse order, 225-ply reuse, explicit overflow/duplicate rejection, and successful retry after validation panic |
| Mixed make/undo full rebuild | 100,000 operations; 50,090 makes, 49,910 undos, 13,213 materializations, 143,991 direction deltas; logical IDs, raw Black/White sums, activated cells, regions, orbit48, quantized value, and final empty root exact |
| Default-feature suite | library 115 passed / 4 ignored; public API integration 1 passed; doctests 7 passed |
| `codebook-eval` suite | library 130 passed / 7 ignored; pbrain tests 2 passed; integrations 2 passed; doctests 7 passed |
| Fixed-depth search identity | best move, score, depth, total/main/qsearch nodes all exact over 1,022 VCT-OFF roots |
| Product VCT-ON identity | depth 4 with the 30-second product time policy; best move, score, depth, total/main/qsearch nodes all exact over 1,022 roots |

The hot path has no capacity-changing operation: logical tokens, 225 frames,
dirty-site generations, and every 44-delta frame are allocated at
construction. `push_after`, materialization, and pop only index preallocated
vectors and fixed arrays; frame overflow is explicit. The concrete sink is
infallible and performs only bounded integer arithmetic. Existing CB-D1
compile-time `i32` bounds remain unchanged.

## Performance

All decision measurements used one release executable and sealed
`A1 -> B1 -> B2 -> A2` arms.

| Workload | Wall B/A | One-sided 95% upper | Identity | Gate |
|---|---:|---:|---|---|
| VCT OFF, depth 4, 64 games / 1,022 roots | 1.003137 | 1.007070 | 4,488,455 nodes in both arms; 0 decision/result/node mismatches | pass |
| Product VCT ON, depth 4, 30-second time policy, 1,022 roots | 1.002479 | 1.006352 | 3,502,671 nodes in both arms; 0 decision/result/node mismatches | pass |

The deterministic primary result is therefore a measured 0.31% cost, not an
improvement. It is inside the registered near-zero-cost extraction bounds
(`point <= 1.005`, upper `< 1.01`).

The first 16-position exploratory profile correctly failed: wall `1.02213`,
evaluation `1.02126`, and push/pop `1.03325`. It triggered a batched
single-pass replay optimization. The sealed 256-position / 1,024-measurement
profile then reported:

- wall `0.99808`;
- evaluation `0.99791`;
- push/pop `1.00349`;
- result mismatches `0`;
- profile-call mismatches `0`.

Frame writing remained `1.12762x`, but occupied only 0.94% of candidate wall
time. Aggregate replay was `1.01412x`; that bucket was diagnostic rather than
a registered gate. Both registered internal ratios passed.

## Product VCT deadline audit

Five diagnostic VCT-ON ABBA sets with `time_ms=None` exposed occasional
transitions between an exact VCT proof and the ordinary depth-4 search near
the fallback 150 ms root VCT deadline. Those raw summaries correctly failed
strict all-row identity. The same behavior occurred within control
repetitions as well as candidate repetitions, and root VCT runs before the
codebook evaluator is constructed.

A seal-validating stability analyzer compared 20 observations per root. Only
six of 1,022 roots were unstable. After excluding exactly that same union
from both arms, all remaining 1,016 roots were identical down to TT
probes/hits/cutoffs. The censored aggregate passed the registered integration
bounds. This is evidence of pre-evaluator deadline nondeterminism, not a
candidate semantic difference.

The registered exact product-integration gate was then rerun at depth 4 with
the product 30-second time policy. That policy gives root VCT its capped
2-second budget; alpha-beta still stops at the requested depth 4. All 1,022
roots matched exactly, including 3,502,671 total visited nodes in each arm.
Its wall ratio was `1.002479` with one-sided 95% upper `1.006352`, so both the
correctness and near-zero-cost bounds passed without censoring. VCT-OFF
remains the primary performance workload.

## Binary gate

The frozen 2,840,064-byte artifact was an MSVC `codebook-eval` audit build,
not the separate self-contained `embed-weights` tournament bundle. Its
matching recipe is:

```powershell
$env:RUSTFLAGS='-C target-cpu=x86-64-v3'
cargo build --release --locked --bin pbrain-figrid --features codebook-eval
```

The promoted candidate is 2,832,896 bytes, SHA-256
`15030571829A44FBCE34A831740A92102F9FE109142A8A31A1610229DC992FC0`:
7,168 bytes (0.252%) smaller than the frozen baseline. It passes both the 1%
and 32 KiB bounds.

An initially audited 4.78 MiB build was also MSVC, but mixed in
`embed-weights` and `+crt-static`. The embedded compressed flat NNUE alone is
1,778,683 bytes, so that output is a different artifact contract and is not
used for this gate. The README's tournament recipe additionally uses its own
GNU target contract.

## Strength and decision

A separate arena was intentionally not run. This card changes only the
reversible bookkeeping beneath bit-exact evaluator and search outputs; exact
full-rebuild and 1,022-root search identity are stronger attribution evidence
than a noisy match. No Elo or strength gain is claimed.

All correctness gates passed, the registered profile and near-zero-cost wall
gates passed, the same-feature binary shrank, and the temporary selector was
removed. The common journal therefore closes as `GO_INTERNAL_EXTRACT`.
CB-F1 may now open.

## Evidence

- baseline commit: `74c93bca23c031b1401d63acdf1831dbd782e805`
- preregistration: `8fece225dda376b8522372e61305f35e60cb45a2`
- prototype: `bcccfc4ba037b8408ba4620cff7851fbe03a3427`
- batched hot path: `34305f41750565550b9e2795d58185ae22a7a59b`
- sealed optimized gate: `58800344f1e23de228d44704c26e0b108ec8eadd`
- model SHA-256:
  `42968FDAB01BA8CCD1DE3DED05C532E4B237DD47EEFFD7AE1C2F264D77BA7DA2`
- holdout SHA-256:
  `1FD40D8948F113AD236FA44F5EEADCA1907C0C3103987CB4C704B67A9B47531A`
- raw optimized profile summary SHA-256:
  `08BFF1FEB7D66A5A3B0F6CE70BD2E2C3E3BD89A9070DC87E7CE501EDD936F702`
- raw fixed-depth summary SHA-256:
  `B3D4E3C1D1AB83A9036DBA76F8CC450353C1B44A0393729BED5446426F6C0124`
- raw `cb_td1_vct_stability_summary_all5.json` SHA-256:
  `0A40150E796DEFA660A2B5F0FE1E0F70CE901F959459B1768C08821FF9954E6D`
- raw `cb_td1_vcton_product30s_summary.json` SHA-256:
  `11AEA4314DB99A9EEEAF7519068A29A8D3291AA2F36BE9AAE62FD1263E32F9CE`
- checked-in compact profile summary SHA-256:
  `F1ED163D29BD61E6B7D034EC63EB06FB34538E88CCD375D2386A1058BC6CD57C`
- checked-in compact fixed-depth summary SHA-256:
  `C238778B425E3675CE188E139D55FD9DE99AB9E46638B3ADBD122BD8DA73DEE6`
- checked-in compact VCT stability summary SHA-256:
  `CD4F5A3481C2909FF6AF1F5CB913F784CE2F9D95AAA96C9C793F75909D547462`
- checked-in compact exact product VCT-ON summary SHA-256:
  `1F27A843AB83410642215D20BB00AA2169BF281667334AF2FB46C1C81C627A98`
- checked-in binary audit summary SHA-256:
  `D0980EC5832FBA76E35794D02C81E46DD4336C245B66ABF3AF1C82207BB990BA`

Large raw ABBA arms remain beside the worktree under
`target/figrid-release-0.8.2-artifacts/2026-07-25/cb-token-delta`.
