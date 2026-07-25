# CB-VOC1 decision-weighted color-orbit vocabulary result

Date: 2026-07-25 KST

Final label: **NO_GO_PRECONDITION**

CB-VOC1 asked whether the existing 4,265 non-RARE Pattern4 rows could cover
materially more of the deployed K=6 decision-loss gradient by selecting whole
mine/opponent color orbits instead of the incumbent frequency-first rows. The
answer on the preregistered RQ615C train corpus is no. The exact selector found
only a 0.1302% relative gradient-objective gain and a +0.0694 percentage-point
raw-slot addressability gain, far below the registered 3% and +1.00pp gates.

No vocabulary, model, runtime default, product artifact, benchmark arm, or
strength arena was changed. Per preregistration, A2, A3, retraining, artifact
building, search benchmarks, and games were not run.

## Frozen inputs

| Input | SHA-256 |
|---|---|
| `models/gomoku_codebook_v1_swapclosed.json` | `42968FDAB01BA8CCD1DE3DED05C532E4B237DD47EEFFD7AE1C2F264D77BA7DA2` |
| `data/topk.bin` | `103891DCD1DCD978C654593ABE78EF32C56E2E350B500EE665BC45AC051AA16D` |
| RQ615C train | `E00A2DA513B05D7631A01003C7E6274E9A3D7575E2C2BD92D5199F1B5385CEB6` |
| RQ615C manifest | `579D1387D7E4DE8F5CB34DB168B6D15655DB229D992751B1DC17BB6CF4260AA7` |
| RQ569 high-precision lineage model | `69BB7C599ADA3A1151577CE3315356BC33C40EDB49A003C9BC4EB90A98F82E18` |

The analyzer accepts only those five sealed inputs and a create-new output
path. It has no CLI option for RQ615C dev, safety, the frozen 64-game holdout,
or game outcomes. Input byte length and SHA-256 were checked both before and
after each production run.

## A0 integrity

- Corpus: 1,336 train slates, 668 paired-color units, 388 unsplit components,
  668 Black rows, 668 White rows, K=6.
- Product lattice: E32/H64/F64.
- Product public evaluator versus independent forward: 0 mismatches over
  8,016 children.
- Product-gradient forward versus A0 independent pre-cast f64 and final f32:
  0 mismatches.
- Old E32/H2048/F2048 forced-Black lineage replay versus stored logits:
  0 mismatches.
- Natural raw token to released mapped-ID audit: 0 mismatches.
- Released universe: 199,827 rows, 215 fixed and 99,806 paired color orbits,
  including 537 anchor-boundary rows.
- Incumbent: 4,265 selectable rows, 29 fixed and 2,118 paired color orbits,
  including the 169-row color-closure tail; RARE was not selectable.

The exact cost-{1,2} capacity solver was also checked against exhaustive
enumeration on a synthetic small problem.

## A1 point result

The selector used the natural child-to-move score, `u=-ell`, exact
ReLU/head/FM embedding sensitivities, f64 Neumaier accumulation in the frozen
train/candidate/cell/direction/dimension order, and independent values
`v(t)+v(sigma(t))` for paired rows.

| Metric | Observed | Gate | Pass |
|---|---:|---:|:---:|
| Relative gradient objective `R_phi` | 0.0013016783 (0.1302%) | >= 0.03 | No |
| Combined addressability gain | +0.0693890pp | >= +1.00pp | No |
| Black addressability gain | +0.0642881pp | >= +0.75pp | No |
| White addressability gain | +0.0744899pp | >= +0.75pp | No |
| Every ordinal addressability gain | +0.0406pp to +0.0839pp | >= 0pp | Yes |
| Combined gross lost incumbent mass | 0.1030716pp | <= 0.25pp | Yes |
| Every color gross loss | max 0.1384731pp | <= 0.50pp | Yes |
| Every ordinal gross loss | max 0.2848148pp | <= 0.50pp | Yes |
| Removed incumbent orbit with member support >=128 | 5 | 0 | No |

Additional point facts:

- `Phi(V*) = 3.79760740407121e-6`.
- `Phi(V0) = 3.79267056722200e-6`.
- Incumbent combined addressability: 99.5939787%.
- Candidate combined addressability: 99.6633677%.
- Candidate geometry: 27 fixed plus 2,119 paired orbits = 4,265 rows.
- 3,071 incumbent rows were retained; 1,194 rows were gained and 1,194 lost.
- The train corpus exposed 8,499 raw rows; 8,023 had nonzero mean decision
  gradient.
- Selected-table SHA-256 over ascending little-endian u32 rows:
  `ADC6491F0F53BAFC15EB7604A0BAD8751E009E8AE5705FD20255F4D5CF0FCB80`.
- Row-value/support stream SHA-256:
  `2865C424E852785FB3336347D6D2ECBAE6F6F843355F6545B449E617214F7140`.

The five protected removed orbits had maximum member supports 170, 134, 211,
180, and 169. Member counts were compared with `max(n_t,n_sigma(t))`; they
were not pooled.

## Reproducibility and decision

Canonical build flags:

```text
RUSTFLAGS=-C target-cpu=x86-64-v3
```

Focused analyzer tests: 13 passed, 0 failed. Two independent production runs
produced byte-identical 485,130-byte reports:

```text
F0C32CED68092A34832E4F7C3EA6B91191A5FE25C2C96BEF689CF053F8D163FC
```

Raw reports:

- `target/figrid-release-0.8.2-artifacts/2026-07-25/cb-voc1/cb_voc1_a0_a1_corrected.json`
- `target/figrid-release-0.8.2-artifacts/2026-07-25/cb-voc1/cb_voc1_a0_a1_corrected_rerun.json`

CB-VOC1 therefore closes as **NO_GO_PRECONDITION**. The incumbent
frequency-first, color-closed vocabulary already addresses 99.59% of the
registered train slots, and the decision-weighted exact reallocation changes
many rows for too little objective or coverage gain while dropping five
support-protected incumbent orbits. This is evidence against spending the
next experiment budget on a vocabulary-only paired retraining card under this
signal definition.
