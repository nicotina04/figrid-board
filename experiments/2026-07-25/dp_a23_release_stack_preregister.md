# 0.8.2 A2+A3 release-stack preregistration

Date: 2026-07-25 (Asia/Seoul)

## Frozen comparison

- source base: figrid-board 0.8.1
  `cdfe13af6842b0def8a09da57d4c3e7f26c535f1`
- rules/eval/search policy: 15x15 Freestyle, embedded quantized codebook,
  White-root ordering ON, root VCT product default ON
- common optimization in every arm: DP-A2 packed Pattern4 windows ON
- only changed factor:
  - A: candidate frontier OFF
  - B: candidate frontier ON after root VCT fails and only around the main
    quantized-codebook alpha-beta search
- trace: campaign fresh holdout, 64 games / 1,022 product roots,
  SHA-256
  `1FD40D8948F113AD236FA44F5EEADCA1907C0C3103987CB4C704B67A9B47531A`
- order: A1 -> B1 -> B2 -> A2, same release binary

## A3 inclusion gates

All gates must pass. Failure removes A3 but does not block the A2-only 0.8.2
release.

1. Combined 100,000 mixed make/undo operations:
   - board, mapped Pattern4 ids, and ordered candidates match the legacy
     control after every operation;
   - packed windows and candidate frontier match full rebuild every 97
     operations;
   - mismatches: 0.
2. VCT-OFF fixed-depth ABBA:
   - best move, score/depth, completed nodes, actual main/qsearch nodes:
     mismatch 0;
   - wall ratio B/A `<= 0.995`;
   - one-sided 95% game-block bootstrap upper `< 1.0`.
3. Product VCT-ON fixed-depth ABBA:
   - wall ratio B/A `<= 0.995`;
   - one-sided 95% upper `< 1.0`;
   - any decision/node differences must be attributable only to deadline
     proof completion.
4. Fixed time:
   - 2-second class: 16 positions x ABBA;
   - 30-second class: 4 positions x ABBA;
   - actual-visited NPS B/A `> 1.0`;
   - visited nodes B/A `>= 1.0`;
   - median depth does not regress.
5. If gates 1-4 pass, same-binary 30-game sanity:
   - alternating sides, 2 seconds/move, four-ply openings;
   - errors 0;
   - composite score at least 12/30.

The original campaign's 5% standalone gate is not reused here. This
shipping-stack test intentionally values a reproducible `>=0.5%` incremental
gain above the already promoted A2 baseline.
