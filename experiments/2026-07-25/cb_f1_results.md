# CB-F1 exact factored residual codebook results

Date: 2026-07-25 (Asia/Seoul)

Decision: **SAFE_BUT_NO_PROMOTION.**

The five-class base plus i8 residual representation is exact, compact, and
safe to retain behind an opt-in selector. It is not the product runtime
default because the registered scalar deployment path missed every
performance bound. `NORU_CODEBOOK_FACTORED` therefore remains default OFF.

The packed source-f32 container is retained as an exact serialization and
packaging improvement. With the selector OFF, the pbrain reconstructs the
established flat quantized runtime from those bit-exact source floats. The
larger JSON is no longer embedded, but evaluator and search semantics do not
change.

## Changed factor

The frozen control stores every quantized embedding row directly as i16.
Candidate B reads the same integers as:

```text
E_q(token, d) = Base_q[Class[token], d] + Residual_q[token, d]
```

- 4,266 token IDs and dimension 16;
- five classes: four semantic anchor classes (empty, mine, opponent, RARE)
  plus fixed singleton token 585;
- class map u8, bases i16, residuals i8;
- unchanged head, factors, bias, scales, Pattern4 IDs, color swap, pooling,
  directional TokenDelta journal, move ordering, search, and VCT policy.

All existing flat APIs remain available. The additive factored type and search
entry point are feature-gated. Product B does not flatten at startup and
performs no representation tag dispatch inside the component loop.

## Correctness

| Gate | Result |
|---|---|
| Artifact parser | Valid flat/factored artifacts pass; malformed magic, version, dimensions, counts, scales, payload length, class IDs, and trailing bytes fail closed |
| Source f32 identity | Embeddings, head, factors, and bias equal the source JSON bit-for-bit |
| Quantized identity | All 68,256 embedding elements, head, factors, bias, scales, all Black rows, all color-swapped White rows, tokens 188/585, and the 4096..4265 tail match |
| Mixed make/undo | 100,000 operations: 50,090 makes, 49,910 undos, 13,213 materializations, 1,031 full rebuilds; logical IDs, raw Black/White cells, activated cells, 9x16 features, orbit48, f32 value bits, and final empty root exact |
| Focused search | Factored and reconstructed-flat depth-2 search have identical move, score, depth, and nodes |
| Fixed VCT-OFF | 1,022/1,022 roots exact; 4,488,455 nodes in both arms |
| Product VCT-ON | 1,022/1,022 roots exact under depth 4 and `time_ms=30000`; 3,502,671 nodes in both arms |
| Allocation/runtime | No allocation in token lookup, update, or value evaluation after evaluator construction; product B drops source floats and retains no decoded flat table |

The permanent ignored release test
`cb_f1_reusable_full_refresh_microbenchmark` also makes the separately
registered refresh workload reproducible on the frozen roots.

## Size

| Item | Flat | Factored | Delta |
|---|---:|---:|---:|
| Embedding representation | 136,512 B | 72,682 B | -63,830 B |
| Total quantized payload | 139,108 B | 75,278 B | -63,830 B (-45.885%) |
| Dual-purpose CBF artifact | 417,412 B | 353,582 B | -63,830 B |
| Matching x86-64-v3 product binary | 1,899,008 B | 1,839,104 B | -59,904 B |

The factored-asset binary is 993,792 B (35.080%, 970.5 KiB) smaller than the
frozen 2,832,896 B JSON-embedding baseline. Of that binary difference,
933,888 B is exact JSON-to-packed serialization and 59,904 B is the compiled
flat-versus-factored artifact difference. Only the exact 63,830 B artifact
delta is attributed to the factored architecture.

All preregistered size bounds passed.

## Canonical scalar performance

Every canonical timing binary used:

```powershell
$env:RUSTFLAGS='-C target-cpu=x86-64-v3'
cargo build --release --locked --features codebook-eval `
  --bin cb-d1-ab --bin rq582-search-profile --bin pbrain-figrid
```

The primary ABBA order was `A1 -> B1 -> B2 -> A2`.

| Workload | B/A | One-sided 95% upper | Exactness | Gate |
|---|---:|---:|---|---|
| VCT OFF, depth 4, 64 games / 1,022 roots | 1.038437 | 1.045790 | 0 decision/result/node mismatches | fail |
| Product VCT ON, depth 4, `time_ms=30000`, 1,022 roots | 1.012149 | 1.019256 | 0 decision/result/node mismatches | exactness pass; timing diagnostic fails |
| Reusable full refresh, 1,022 roots x 24 repeats | 1.157427 | n/a | exact rolling checksum | fail |

An immediate corrected-root refresh confirmation was `1.133956`; both runs
are far beyond the `<= 1.02` bound, so no favorable rerun was selected.

The sealed 256-position / 1,024-measurement profile reported:

- wall: `1.022494`;
- evaluator: `1.022093` (gate `<= 1.01`);
- aggregate push/pop: `1.020977` (gate `<= 1.01`);
- recompute: `1.076557`;
- restore: `1.075955`;
- result mismatches: `0`;
- profile-call mismatches: `0`.

Thus the exact factorization saves memory but its scalar i8 residual decode
cost is observable in search. This is a performance rejection, not a semantic
or strength regression.

## Bounded optimization audit

Three preregistration-compatible implementations were tested independently on
256 canonical x86-64-v3 roots. Every arm remained decision/node exact.

| Candidate | Search B/A | 95% upper | Full refresh | Decision |
|---|---:|---:|---:|---|
| Clean scalar | 1.025172 | 1.039309 | about 1.14x in the temporary audit | reject |
| Paired Black/White delta loop | 1.034569 | 1.051866 | not advanced | revert |
| Same-class four-token sum | 1.027762 | 1.037233 | 1.128288 | revert |
| Explicit fixed-16 AVX2 | 1.023462 | 1.033647 | 0.709645 (0.719854 repeat) | revert |

The AVX2 implementation passed an independent unsafe-code audit, all 25 class
transition pairs, boundary tokens, mixed/same-class sums, and the 100,000
operation correctness gate. It accelerated isolated refresh by roughly 29%,
but did not move end-to-end search inside the registered bounds. All three
optimization diffs and temporary tests were removed. Only the reproducible
scalar refresh benchmark remains.

No decoded i16 cache, precomputed delta table, post-construction allocation,
search change, retraining, or rescue threshold was admitted.

## Strength and decision

A direct 30-game H2H was not run because the preregistration allows it only
after every correctness and performance gate passes. The integer evaluator
and all 1,022 fixed/product search outputs are exact, so no playing-strength
change is claimed.

The card closes as `SAFE_BUT_NO_PROMOTION`:

- keep `NORU_CODEBOOK_FACTORED=off` as the product default;
- retain the exact packed artifact, parser, direct runtime, and opt-in
  selector for future memory-constrained experiments;
- retain the exact packed-source packaging improvement;
- do not ship direct factored evaluation as the normal runtime;
- open CB-VOC1 with flat quantized runtime in both arms.

## Evidence

- frozen baseline: `dc0d9afae658113747e5666c3864b381cc971582`
- preregistration: `e04e089bd7e62b77ab1386c1cffa34fc6e38c786`
- implementation: `fabc8794a753beedea2f76faffab69644c31d88c`
- source model SHA-256:
  `42968FDAB01BA8CCD1DE3DED05C532E4B237DD47EEFFD7AE1C2F264D77BA7DA2`
- frozen holdout SHA-256:
  `1FD40D8948F113AD236FA44F5EEADCA1907C0C3103987CB4C704B67A9B47531A`
- factored artifact SHA-256:
  `141014529417A73E58B210832AFD189AD970E045A8907F7D2879693C5B171A8D`
- compact-flat artifact SHA-256:
  `9A5E3D3FC47EEF79468F021F78E9130F5842764F579EE68A2FD270E8289B3250`
- raw fixed summary SHA-256:
  `7A2F3E1C5B36E008692C35CE5481833F371B0A779263033AD9A0792F92161C7E`
- raw profile summary SHA-256:
  `07F0A73A34C7D1CAA259F083AAA09EBDD3442216DA82CC049653F7DE61CE49E3`
- raw VCT summary SHA-256:
  `5A74CD81F5D209D5280F901B4234C2D0BAEE2C77EDE1915068AA05C45AF8A49D`
- raw scalar refresh stdout SHA-256:
  `10F0358CAB89713332B6165EB092575C1B0F0E9B50A0A616B64718D8D2CDCD3B`
- x86-64-v3 factored-asset pbrain SHA-256:
  `ADE3776DB8C5A97AEB87994008D460FED9253325A6172EDD9823324CFFC97FE3`
- x86-64-v3 compact-flat pbrain SHA-256:
  `71B9BF71F34C3E5023566EC5D5171073434318E1D3EDAB017B8454E125E1080F`

Large raw arms and rejected-kernel diagnostics remain beside the worktree
under
`target/figrid-release-0.8.2-artifacts/2026-07-25/cb-token-delta/cb-f1`.
