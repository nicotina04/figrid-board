# CB-F1 exact factored residual codebook preregistration

Date: 2026-07-25 (Asia/Seoul)

## Frozen baseline

- baseline commit:
  `dc0d9afae658113747e5666c3864b381cc971582`
  (`CB-TD1-promote-reversible-token-journal`)
- development branch: `codex/cb-token-delta`
- rules/product policy: 15x15 Freestyle, quantized codebook, White-root
  ordering auto, packed Pattern4 windows ON, exact-order candidate frontier
  ON, CB-D1/CB-TD1 ON, root VCT product default ON
- source model:
  `models/gomoku_codebook_v1_swapclosed.json`
- source model SHA-256:
  `42968FDAB01BA8CCD1DE3DED05C532E4B237DD47EEFFD7AE1C2F264D77BA7DA2`
- frozen 64-game / 1,022-root holdout SHA-256:
  `1FD40D8948F113AD236FA44F5EEADCA1907C0C3103987CB4C704B67A9B47531A`
- Windows x86_64-v3 product binary:
  `target/release/pbrain-figrid.exe`,
  2,832,896 B,
  SHA-256
  `15030571829A44FBCE34A831740A92102F9FE109142A8A31A1610229DC992FC0`
- toolchain: rustc 1.88.0, LLVM 20.1.5, Cargo 1.88.0,
  `x86_64-pc-windows-msvc`
- CPU identifier:
  `AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD`

The worktree and index were clean before the precondition census and this
preregistration.

## Precondition result

The read-only census is frozen in
`cb_f1_precondition_census.md`. The selected candidate exactly represents the
deployed 4,266 x 16 S32/i16 embedding table as:

```text
E_q(token, d) = Base_q[Class[token], d] + Residual_q[token, d]
```

- `Class`: u8, five classes
- `Base`: i16, 5 x 16
- `Residual`: i8, 4,266 x 16
- semantic classes: anchor empty, mine, opponent, RARE
- fixed singleton class: token 585
- base rule: componentwise integer ceil-midpoint of class min/max
- residual range: `[-128, 127]`

This is a lossless integer reparameterization, not a learned or approximate
model.

## Decision boundary and single changed factor

Control A uses the existing flat quantized embedding rows. Candidate B reads
the exact five-class base/residual representation directly. Both use the same
source floats, quantization scales, head, factors, bias, Pattern4 IDs,
color-swap mapping, ReLU, region pooling, FM accumulation, search, move
ordering, VCT, timing, and product policy.

The experiment selector is `NORU_CODEBOOK_FACTORED`, default OFF until every
gate passes. External JSON model loading and the explicit float evaluator keep
their existing behavior.

One compact embedded artifact may contain:

1. the source f32 arrays, bit-exact, for the flat rollback and explicit float
   path;
2. the factored quantized embeddings;
3. the unchanged quantized head, factors, and bias;
4. dimensions, scales, source SHA-256, and fail-closed length/version fields.

This avoids embedding the 1.41 MB decimal JSON while preserving immediate
same-binary rollback. The compact serialization is reported separately from
the factored quantized payload so serialization savings cannot be mistaken for
architecture savings.

Forbidden in CB-F1:

- retraining, distillation, residual regularization, SVD, VQ/PQ, QAT, STE, or
  new labels;
- vocabulary edits, frequency reordering, symmetry-orbit tying, or RARE
  remapping;
- lane-specific or axis/diagonal embeddings;
- flattening candidate B at startup;
- changes to CB-TD1 replay or search policy;
- breaking changes to any existing released public API.

An additive, feature-gated `FactoredQuantizedCodebookWeights` view and matching
factored search entry point are explicitly permitted because the pbrain is a
separate crate. Every existing flat type, field, constructor, search entry
point, and public test must remain source- and behavior-compatible.

## Correctness gates

Every gate requires zero mismatches or failures.

1. Artifact generation and parsing:
   - the f32 rollback section equals the source JSON arrays bit-for-bit;
   - malformed magic, version, dimensions, counts, scales, lengths, class
     IDs, and trailing bytes fail closed;
   - every reconstructed i16 component stays in range.
2. Weight identity:
   - all 68,256 reconstructed embedding values equal the existing Rust
     quantizer output exactly;
   - all 4,266 raw Black token rows and all 4,266 color-swapped White rows
     match;
   - head, factors, bias, scales, token 188, token 585, and the
     `4096..4265` duplicate/RARE tail match exactly.
3. At least 100,000 deterministic mixed make/undo transitions:
   - logical token states, raw Black/White cells, post-ReLU cells, 9 x 16
     regions, orbit48, and final f32 value bits match flat A and full rebuild;
   - complete undo restores the empty root exactly.
4. Frozen-root fixed-depth VCT-OFF:
   - best move, score, completed depth, completed nodes, actual main nodes,
     and qsearch nodes match exactly on all 1,022 roots.
5. Product VCT-ON:
   - run the frozen CB-TD1 product protocol on all 1,022 roots at depth 4 with
     `--time-ms 30000`; this leaves the production root-VCT budget capped at
     2 seconds while removing the alpha-beta fallback deadline;
   - best move, result class, completed depth, and completed nodes must match
     on every root; no root may be removed after observing B;
   - the ordinary fallback-budget stability audit may be reported only as a
     labelled secondary result.
6. No heap allocation occurs in evaluator token lookup, update, or value
   evaluation after evaluator construction. Existing i32 accumulator bounds
   remain unchanged.

## Size gates

- factored embedding payload: at most 72,682 B;
- total factored quantized weight payload: at most 75,300 B;
- reduction versus the frozen 139,108 B quantized payload: at least 45%;
- compact dual-purpose artifact: at most 354 KiB;
- promoted one-artifact product binary: at least 900 KiB smaller than the
  frozen 2,832,896 B binary under the identical build recipe;
- the promoted binary must not retain the source JSON or a decoded flat i16
  table when candidate B is active.

Runtime-resident payload and final executable size are both required. A
startup-expanded flat table cannot pass this card.

Size attribution must include a compact-flat counterfactual containing the
same source-f32 rollback section and header but the original 139,108 B flat
quantized payload. Both compact-flat and compact-factored artifacts and product
binaries are built with the same recipe. The expected 63,830 B artifact delta
is the only size reduction attributed to CB-F1; the larger JSON-to-binary
reduction is reported separately as packaging-only. Timing A/B still uses one
compact-factored artifact and one executable.

## Performance gates

- one release binary and explicit flat/factored selector;
- sealed order `A1 -> B1 -> B2 -> A2`;
- same frozen roots and policy in every arm;
- primary fixed-depth paired wall ratio B/A point estimate `<= 1.005`;
- one-sided 95% game-block bootstrap upper bound `< 1.01`;
- evaluator and aggregate push/pop ratios each `<= 1.01`;
- full-refresh ratio is recorded separately and must be `<= 1.02`;
- exact node/depth/search-output equality is mandatory.

The direct implementation may cancel the base term when old and new token
classes are equal. Site batching already supplied by CB-TD1 may be used, but
no other evaluator or search optimization is admitted.

## Quantized artifact and strength gate

The packed factored section is the deployment artifact; a float-only or
startup-flattened prototype cannot advance.

Because A and B reconstruct the same integer evaluator, fixed-depth
search-output identity is the primary strength-preservation proof. After all
correctness and performance gates pass, run a paired 30-game direct H2H sanity
with side swaps and a frozen seed:

- protocol/search errors: 0;
- telemetry complete;
- no result is used to claim a playing-strength gain;
- any material time-driven regression reopens the performance gate.

No 100/300-game rescue run is allowed for a correctness or performance
failure. Statistical strength testing belongs only to a later card that
changes model semantics.

## Outcomes

- `GO_EXACT_FACTOR`: every gate passes; make the factored artifact/runtime
  product default and retain `NORU_CODEBOOK_FACTORED=off` as rollback.
- `SAFE_BUT_NO_PROMOTION`: exactness passes but size or performance does not.
- `REJECT`: any semantic, integer, rebuild, undo, or search-output mismatch.

CB-VOC1 and all later cards remain closed until CB-F1 has a recorded decision
and a clean promotion or rejection commit.
