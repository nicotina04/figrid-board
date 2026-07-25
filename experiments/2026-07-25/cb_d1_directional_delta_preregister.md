# CB-D1 exact directional-delta preregistration

Date: 2026-07-25 (Asia/Seoul)

## Frozen baseline

- source: figrid-board 0.8.2
  `8d7d47d60ec37313a0c459ffdd18616bb9df4be0`
- rules/product policy: 15x15 Freestyle, embedded quantized codebook,
  White-root ordering auto, packed Pattern4 windows ON, exact-order candidate
  frontier ON, root VCT product default ON
- model:
  `models/gomoku_codebook_v1_swapclosed.json`
  SHA-256
  `42968FDAB01BA8CCD1DE3DED05C532E4B237DD47EEFFD7AE1C2F264D77BA7DA2`
- released Windows x86_64-v3 binary SHA-256:
  `5E3241428EF7FC591B9F10AF0F5C4AEAEDF6739B677A4663316DEF7F9F60B89`
- release zip SHA-256:
  `93899FF840A6F5BB08323B16F50ECE6C75AE9E1CB59B9D0D0FEF829FD5D736FC`
- toolchain: rustc 1.88.0, LLVM 20.1.5, Cargo 1.88.0,
  `x86_64-pc-windows-msvc`
- CPU identifier:
  `AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD`
- frozen 64-game / 1,022-root holdout SHA-256:
  `1FD40D8948F113AD236FA44F5EEADCA1907C0C3103987CB4C704B67A9B47531A`

The working tree was clean before this preregistration. CB-D1 is developed
on `codex/cb-d1-directional-delta`.

## Single changed factor

Control A keeps the 0.8.2 quantized evaluator:

1. collect every unique dirty cell;
2. back up both post-ReLU cell vectors;
3. remove both vectors from region features;
4. recompute four direction embeddings for both color perspectives;
5. add the recomputed vectors back.

Candidate B keeps a pre-ReLU raw accumulator and the logical mapped Pattern4
IDs. For each changed `(cell, direction)`:

```text
raw += embedding(new_id) - embedding(old_id)
region += ReLU(raw_after) - ReLU(raw_before)
```

Undo applies the exact inverse deltas. No evaluator weight, feature mapping,
region pooling, FM head, search policy, move order, or proof rule may change.
The candidate is OFF by default until every registered gate passes.

## Prototype-open precondition

The existing quantized evaluator profile attributes materially more than 3%
of product wall time to dirty-cell backup/recompute/aggregate/restore. The
candidate removes four-direction recomputation and full cell-vector backup,
so its perfect-case wall ceiling clears the campaign's prototype threshold.
A fresh 0.8.2 product-policy profile will be recorded before promotion.

## Correctness gates

All must pass with zero failures.

1. 100,000 deterministic mixed make/undo transitions:
   - logical mapped IDs equal the board after every operation;
   - raw Black/White cell accumulators equal full four-direction rebuild;
   - ReLU cell values and 9x16 region features equal full rebuild;
   - final quantized value equals the legacy incremental and full evaluator;
   - complete undo restores the empty root exactly.
2. Lazy-stack coverage:
   - materialized and unmaterialized push/pop sequences;
   - multiple pending frames materialized at once;
   - root refresh and repeated evaluator reuse.
3. Fixed-depth VCT-OFF and product VCT-ON:
   - best move, score, completed depth, completed nodes, actual main nodes,
     and qsearch nodes match exactly between A and B.
4. Scalar integer arithmetic must not overflow under explicit bounds.

## Performance gates

- one release binary, explicit A/B switch, order `A1 -> B1 -> B2 -> A2`;
- same frozen roots and product policy in every arm;
- primary fixed-depth metric: paired wall ratio B/A;
- promotion requires wall ratio `<= 0.99` and one-sided 95% game-block
  bootstrap upper `< 1.0`;
- 2-second and 30-second checks require actual-visited NPS `> 1.0`, visited
  nodes `>= 1.0`, and no median-depth regression;
- zero protocol/search errors in a same-binary 30-game sanity.

A reproducible exact gain below 1% is retained only as a shadow candidate;
it does not justify product complexity by itself. If correctness passes but
the performance gate fails, the result is `SAFE_NO_NET_GAIN`.

## Follow-up lock

The generic `TokenDelta -> reversible accumulator` extraction and every
later codebook card remain closed until CB-D1 receives a final decision.
