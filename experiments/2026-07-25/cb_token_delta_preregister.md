# CB-TD1 reversible TokenDelta journal preregistration

Date: 2026-07-25 (Asia/Seoul)

## Frozen baseline

- promoted evaluator baseline:
  `74c93bca23c031b1401d63acdf1831dbd782e805`
  (`CB-D1-exact-directional-deltas`)
- development branch: `codex/cb-token-delta`
- rules/product policy: 15x15 Freestyle, embedded quantized codebook,
  White-root ordering auto, packed Pattern4 windows ON, exact-order candidate
  frontier ON, CB-D1 ON, root VCT product default ON
- model:
  `models/gomoku_codebook_v1_swapclosed.json`
  SHA-256
  `42968FDAB01BA8CCD1DE3DED05C532E4B237DD47EEFFD7AE1C2F264D77BA7DA2`
- frozen 64-game / 1,022-root holdout SHA-256:
  `1FD40D8948F113AD236FA44F5EEADCA1907C0C3103987CB4C704B67A9B47531A`
- promoted Windows x86_64-v3 product binary:
  `target/release/pbrain-figrid.exe`
  SHA-256
  `493C94578834A23B35C58DC1697ABF1D170FFB24C4171177C381535356F33A2A`,
  2,840,064 bytes
- promoted same-binary harness:
  `target/release/cb-d1-ab.exe`
  SHA-256
  `FB7F70F58F12634D429A96FA9310E63B9D5996DBB1A1F592F3FBDC393418E14F`,
  2,772,992 bytes
- toolchain: rustc 1.88.0, LLVM 20.1.5, Cargo 1.88.0,
  `x86_64-pc-windows-msvc`
- CPU identifier:
  `AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD`

The working tree and index were clean before this preregistration.

## Decision boundary

This card extracts only a reusable reversible change journal. It does not
generalize evaluator arithmetic, activation, pooling, model layout, or search.

The private, allocation-free hot-path interface records individual changes:

```text
TokenDelta { site, lane, old, new }
```

and owns:

1. a logical token mirror;
2. fixed-capacity per-ply frames;
3. logical and materialized stack depths;
4. forward materialization and exact reverse replay through a monomorphized
   sink.

The prefix invariant is:

```text
[0, materialized_depth)       numerically applied
[materialized_depth, depth)   logical-only pending
```

The journal must not import Board, Stone, Pattern4, codebook weights, region
pooling, or search types. Production integration in this card is limited to
the quantized CB-D1 Pattern4-token consumer. Other consumers remain census
items, not implementation scope.

## Single changed factor

Control A is the promoted CB-D1 direct implementation. It stores logical
four-direction IDs and a bespoke per-ply directional undo frame, then applies
exact embedding deltas.

Candidate B performs the identical CB-D1 integer arithmetic through the common
TokenDelta journal and sink. Both arms keep CB-D1 ON. No model, feature mapping,
embedding, activation, region pooling, FM head, search policy, move ordering,
VCT rule, or time policy may change.

The candidate selector is OFF by default until every registered gate passes.

## Correctness gates

All must pass with zero failures.

1. Generic journal tests:
   - one and multiple pending frames;
   - materialization of the pending suffix exactly once;
   - materialized and unmaterialized pop;
   - reverse site/lane replay order;
   - reset/reuse and maximum game depth;
   - fixed-capacity overflow is explicit rather than allocating.
2. At least 100,000 deterministic mixed make/undo transitions:
   - A and B emit equivalent logical token states;
   - raw Black/White cell accumulators equal a full four-direction rebuild;
   - post-ReLU cell vectors and 9x16 region features equal full rebuild;
   - orbit48 features and final quantized value are bit-exact;
   - complete undo restores the empty root exactly.
3. Fixed-depth VCT-OFF primary and product VCT-ON integration:
   - best move, score, completed depth, completed nodes, actual main nodes,
     and qsearch nodes match exactly between A and B.
4. The journal hot path performs zero heap allocations after construction.
5. Scalar integer arithmetic remains inside the existing explicit CB-D1 bounds.

## Performance gates

- one release binary, explicit direct/journal selector, sealed arms in order
  `A1 -> B1 -> B2 -> A2`;
- same frozen roots and product policy in every arm;
- primary fixed-depth metric: paired wall ratio B/A;
- internal extraction GO requires point ratio `<= 1.005` and one-sided 95%
  game-block bootstrap upper `< 1.01`;
- evaluator and aggregate push/pop ratios must each be `<= 1.01`;
- exact node/depth/search-output equality is mandatory;
- final shipping product binary growth versus the frozen promoted binary must
  be no more than both 1% and 32 KiB;
- no public API, model asset, serialization, dependency, default feature, or
  product-policy change.

This card may receive `GO_INTERNAL_EXTRACT` without claiming playing-strength
gain. Failure of correctness is `REJECT`; correctness with performance or
complexity regression is `SAFE_BUT_NO_EXTRACT`.

## Promotion and follow-up lock

On GO, the journal becomes the sole internal implementation of CB-D1 and the
temporary direct/journal experiment selector is removed from the shipping
binary. Hex, float-codebook, NNUE, Zobrist, VCT, and candidate-frontier
consumers are not migrated by this card.

CB-F1 and all later codebook cards remain closed until this card has a recorded
decision and a clean commit.
