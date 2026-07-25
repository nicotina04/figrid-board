# CB-GH1 P0 rooted threat-graph signal result

Date: 2026-07-26 KST

Final label: **NO_GO_STATE_EXPLOSION**

CB-GH1 P0 asked whether the fixed role-relative, rooted one-ply threat graph
repeats across independent opening components often enough to act like a
small relational code before any graph model or incremental runtime is built.
It does not.

The graph passed every A0 integrity, D4, color-role, state-hash, and graph-hash
check. Across 8,016 rooted K=6 candidates it produced 6,778 exact graph codes,
a gross identity reduction of 15.4441%. However, only 1,058 candidates
(13.1986%) used a code present in at least three independent components,
against the registered 25% combined gate. White-root coverage was 7.3852%,
and ordinals 4, 6, and 8 were also below their gates.

Per preregistration, the A1 support decision did not consume `q_teacher` or
product utilities. A0 still validated those sealed numeric fields and replayed
the evaluator as required, but the statistical layer returned before using
them. A2 residual projection, A3 component LOO, incremental graph state, GR1
training, model artifacts, benchmarks, arenas, and product changes were not
run.

## Sealed implementation and execution

The graph and decision protocol were preregistered before implementation at:

```text
3e52280
```

The analyzer was implemented and frozen at:

```text
23cbd3a  Implement CB-GH1 rooted graph census
f2d4f79  Fix GH1 compiled target preflight
```

The sole authoritative run used full HEAD:

```text
f2d4f79616de2ac19e74f97aa769c0c93d259003
```

The release executable was built with:

```text
RUSTFLAGS=-C target-cpu=x86-64-v3
cargo build --release --locked --features codebook-eval --bin cb-gh1-graph-census
```

The executable required a release profile, compiled AVX2/BMI2 target
features, the canonical runtime `RUSTFLAGS`, no `NORU_*` overrides, a clean
worktree including untracked files, and byte equality between 44 source/build
inputs embedded at compile time and the current worktree.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `cb-gh1-graph-census.exe` | 4,252,160 | `6055B056A4EB536E6D569FE03676A42A17B8A2596851D4F2010F33FA01D0EBEA` |
| authoritative JSON | 36,564 | `07CB55D3B59940DF7DCF3F7244063F92BD0A568B3C1A681FDBEDA2A411B7D57C` |

The run completed in 23,175 ms. The first successfully created JSON is the
sole authoritative report; the workload was not rerun.

## Frozen inputs and corpus replay

| Input | Bytes | SHA-256 |
|---|---:|---|
| product f32 codebook | 1,410,562 | `42968FDAB01BA8CCD1DE3DED05C532E4B237DD47EEFFD7AE1C2F264D77BA7DA2` |
| Pattern4 `topk.bin` | 17,060 | `103891DCD1DCD978C654593ABE78EF32C56E2E350B500EE665BC45AC051AA16D` |
| RQ615C train | 54,991,200 | `E00A2DA513B05D7631A01003C7E6274E9A3D7575E2C2BD92D5199F1B5385CEB6` |
| RQ615C manifest | 5,463 | `579D1387D7E4DE8F5CB34DB168B6D15655DB229D992751B1DC17BB6CF4260AA7` |
| RQ569 high-precision lineage model | 1,413,542 | `69BB7C599ADA3A1151577CE3315356BC33C40EDB49A003C9BC4EB90A98F82E18` |

A0 reconstructed and validated:

- 1,336 train slates, 8,016 K=6 children;
- 668 paired-color units and 388 unsplit components;
- 668 Black-root and 668 White-root rows;
- ordinals `{1,2,4,6,8}`;
- 285,900 complete legal-inventory children;
- zero product public-versus-independent bit mismatches over K=6;
- zero natural raw-token versus released mapped-ID mismatches;
- zero stored child-hash replay mismatches; and
- zero high-precision lineage-bit mismatches over all 285,900 legal children.

No dev, safety, RQ508, game outcome, frozen 64-game holdout, or frozen
1,022-root timing row was exposed to the analyzer.

## A0 graph integrity

The analyzer built exactly 8,016 jointly rooted parent-to-child graphs.

| Check | Observed |
|---|---:|
| production exact-state checks | 9,352 |
| distinct production D4 u64 keys | 9,286 |
| true production state-hash collisions | 0 |
| distinct exact graph codes | 6,778 |
| distinct graph SHA-256 values | 6,778 |
| distinct prospective graph u64 values | 6,778 |
| graph SHA-256 collisions | 0 |
| graph u64 collisions | 0 |
| D4 checks per audit family | 64,128 |
| total D4/equivariance mismatches | 0 |
| color-role checks | 40,080 |
| color-role mismatches | 0 |

The frozen coordinate formulas passed 405,000 in-board and 2,888 virtual
coordinate checks with zero mismatches.

Intentional abstraction accounting found:

- 528 exact graph groups containing more than one exact role-relative
  transition;
- 1,234 excess exact-role identities inside those graph groups;
- 4 exact duplicate occurrence excesses after color accounting; and
- zero observed exact color-role-isomorphism groups.

These abstraction groups are not graph-hash collisions. Exact graph bytes,
not SHA-256 or the prospective u64, remained authoritative.

Graph size stayed bounded:

| Per-transition metric | p05 | p50 | p95 | max |
|---|---:|---:|---:|---:|
| affected `(source,direction)` sites | 33 | 44 | 44 | 44 |
| total nodes | 36 | 44 | 50 | 58 |
| factors | 7 | 10 | 12 | 16 |
| incidences | 63 | 90 | 108 | 144 |
| serialized bytes | 999 | 1,347 | 1,613 | 2,049 |

## A1 label-blind reuse result

A code was recurrent only when its exact graph bytes occurred in at least
three distinct `component_uid` groups.

- 6,250 of 6,778 codes occurred only once.
- 6,313 codes occurred in only one component.
- 282 more occurred in exactly two components.
- Only 183 codes met the three-component recurrence definition.
- Those recurrent codes covered 1,058 of 8,016 candidates.

| Stratum | Recurrent / total | Observed | Gate | Pass |
|---|---:|---:|---:|:---:|
| combined | 1,058 / 8,016 | 13.1986% | >=25% | No |
| Black root | 762 / 4,008 | 19.0120% | >=15% | Yes |
| White root | 296 / 4,008 | 7.3852% | >=15% | No |
| ordinal 1 | 532 / 1,392 | 38.2184% | >=10% | Yes |
| ordinal 2 | 368 / 1,752 | 21.0046% | >=10% | Yes |
| ordinal 4 | 106 / 1,524 | 6.9554% | >=10% | No |
| ordinal 6 | 37 / 1,848 | 2.0022% | >=10% | No |
| ordinal 8 | 15 / 1,500 | 1.0000% | >=10% | No |

The combined, White, and every-ordinal gates therefore failed. The registered
precedence makes the final result:

```text
NO_GO_STATE_EXPLOSION
```

The statistical report confirms `q_or_product_fields_read=false`, evaluates
only `A1_LABEL_BLIND_REUSE`, and stores A2 and A3 as null.

## Independent artifact audit

Three read-only audits parsed and rehashed the authoritative report without
rerunning the workload.

They independently:

- reproduced all combined, color, and ordinal fractions from raw numerators
  and denominators;
- closed both multiplicity histograms to 6,778 codes and 8,016 observations,
  including exactly 183 codes with support in at least three components;
- verified all D4, color-role, coordinate, state-hash, graph-hash, shape, and
  fixed-width serialization counts;
- closed abstraction accounting as
  `6,778 graph identities + 1,234 role-identity excess + 4 duplicate excess
  = 8,016 observations`;
- rehashed all five sealed inputs, the executable, and all 44 compiled source
  inputs with zero mismatch;
- independently parsed the train file as 1,336 K=6 rows, 668 exact
  paired-color units, 388 components, and 285,900 legal children;
- verified that the input list contains only the five authorized artifacts
  and no forbidden row source; and
- reproduced the A1 early-stop precedence, null A2/A3, final label, and full
  downstream closure.

All three audits reported zero issues. The report bytes and SHA-256 remained
unchanged before and after inspection.

## Interpretation and next boundary

The graph was not literally unique for every candidate: it removed 15.44% of
gross identities and intentionally merged 528 groups of exact board
transitions. The failure is narrower and more useful: the fixed exact rooted
graph remains too specific to provide the preregistered component-level reuse,
especially for White roots and later ordinals. It therefore cannot justify
the cost of an incremental graph state or a relational codebook under this
card.

Because A1 stopped before labels, this result does **not** establish that
within-code residual variance is high or that every coarser relational
representation lacks predictive signal. It does establish that post-hoc
coarsening, radius changes, threshold changes, or label-aware graph rescue
cannot be claimed from CB-GH1 P0. Any such representation would require a new
preregistered card.

The current GH1 representation is closed:

- incremental GH1 correctness/cost: not opened;
- graph dictionary or artifact construction: not opened;
- conditional CB-GR1 training: not opened;
- graph runtime/search consumer: not opened;
- benchmark or arena: not opened; and
- product promotion: not opened.

The sequential codebook campaign therefore returns to the accepted flat
Pattern4 product representation and advances to CB-QAT1. CB-AL1 remains after
QAT1.

Raw report:

- `target/figrid-release-0.8.2-artifacts/2026-07-26/cb-gh1-p0/cb_gh1_p0_authoritative.json`
