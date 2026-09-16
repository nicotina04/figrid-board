# 0.8.8 candidate-generation audit

Base: `v0.8.7` / `4cc5fb290644f0bb87580e481209ee8f8db8240d`.

`Board::candidate_moves` now uses 225 precomputed radius-two bitmasks to
remove occupied and already emitted cells. Stones and fresh neighbors are
enumerated in ascending order, preserving the former candidate vector exactly.
No Board fields, evaluator weights, search rules, or frontier state change.

Validation on 2026-09-16:

- 180,000 ordered-vector checks across random full-board make/undo, including
  boundaries, terminal synthetic boards, empty and full boards.
- 30 archived openings, both frontier settings, both root-VCT settings,
  10 alternating AB/BA rounds: 1,200 matching move/score/depth/node records.
- Root VCT off, 8,192 nodes per turn: 12 complete games and 206 moves identical.
- Root VCT on uses a 150ms wall-clock tactical budget even when alpha-beta has
  a node cap. One complete-game replay diverged at opening 4, ply 22: the
  baseline finished a VCT proof just before the deadline, while the other run
  continued into alpha-beta. Timed full-game equivalence is not claimed.
- 155 product/library all-feature tests and 120 default-feature tests passed;
  existing ignored tests remain ignored. All targets compile with all features.
  A broader historical research-tool test requires an unavailable absolute-path
  trace (`cb-gh0-hash-cost::frozen_trace_parser_selects_registered_roots`).
  No historical test or fixture was modified to conceal that limitation.

Ten paired rounds were collected with no concurrent training/build/teacher
processes; the first two were warmup. These are small engineering measurements
on this machine and corpus, not strength results. Without the incremental
frontier, total search time fell about 5.2% (VCT off) and 3.8% (VCT on).
The shipped pbrain already uses the frontier: its measured change was near zero,
with no demonstrated improvement. The optimization primarily benefits public
Board users and search paths that call the ordinary candidate generator.

Reproduction example (copy this example into the baseline checkout too):

```text
cargo +1.89.0 build --release --locked --example candidate_mask_audit --features codebook-eval,embed-weights
# Clear unrelated NORU_/FIGRID_ overrides; select NORU_ROOT_VCT=0 or 1.
target/release/examples/candidate_mask_audit OPENINGS.jsonl OUT.json [games]
```

Opening rows contain `history: [{x, y}, ...]`. Run both revisions with the same
weights, inputs, compiler options and machine; compare deterministic fields
separately from wall time. Keep root VCT off for full-game deterministic parity.
