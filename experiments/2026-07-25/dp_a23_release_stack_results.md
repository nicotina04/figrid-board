# 0.8.2 A2+A3 release-stack results

Date: 2026-07-25 (Asia/Seoul)

Decision: **include both packed Pattern4 windows and the exact-order
candidate frontier in figrid-board 0.8.2**.

The exact public `Board` layout remains unchanged from 0.8.1. Optional state
lives in a `Searcher` sidecar; `pbrain-figrid` enables both paths by default,
with independent runtime rollback.

## Registered gates

| Gate | Result | Decision |
|---|---|---|
| Combined 100k make/undo/full rebuild | 0 mismatches | PASS |
| VCT-OFF fixed depth, 64 games / 1,022 roots | wall 0.968153, upper 0.974997, all semantic/node mismatches 0 | PASS |
| Product VCT-ON fixed depth, 64 games / 1,022 roots | wall 0.986855, upper 0.991573, all semantic/node mismatches 0 | PASS |
| 2-second class, 16 positions | NPS 1.00389, nodes 1.00216, p50 depth 9 -> 9 | PASS |
| 30-second class, 4 positions | NPS 1.03329, nodes 1.03283, p50 depth 12 -> 12 | PASS |
| Same-binary 30-game sanity | combined 17, packed-only 13, errors 0 | PASS |

The fixed-depth release-stack threshold was wall ratio `<=0.995` with a
one-sided 95% upper below 1.0. Timed gates required NPS above 1.0, visited
nodes at least 1.0, and no median-depth regression. All four machine-readable
summaries report `release_stack_gate.pass = true`.

## Product interpretation

Packed windows had already measured a `0.78885` product wall ratio versus the
sealed 0.8.1 baseline. The frontier then measured `0.9868546` on top of
packed windows. Their independently registered ratios compound to `0.77848`,
or an indicated 22.15% fixed-depth wall reduction versus 0.8.1.

No evaluator, model, feature mapping, move-order policy, or VCT proof rule
changed. The 30-game result is a safety sanity, not an Elo estimate.

The first product B1 arm had one root-VCT proof finish on the 150 ms deadline
boundary before A3 could be enabled. That raw arm was retained outside the
crate and rerun once; the rerun restored zero fixed-depth mismatches and is
the checked-in product summary.

## Evidence

SHA-256 values below are for the repository's canonical LF bytes.

| Artifact | SHA-256 |
|---|---|
| `dp_a23_vctoff_summary.json` | `9B532FA1A9FE99D46FD48D43954EC3659D5ABC667A067C52CE7C4CC0E8B08AC8` |
| `dp_a23_product_summary.json` | `1DBD735B9FF90A94ADBD807791C10456159E5ED37E6E2925355673A86E6096D3` |
| `dp_a23_2s_summary.json` | `4DFB6C006218EDE67663435124F127A2DB87F259501B7BA9F82376AF8EF28878` |
| `dp_a23_30s_summary.json` | `F795D2E5659CA0A748C5C5537A101F4C217E0826BA87C4CBD85BA4FF50F30BDE` |
| `dp_a23_h2h_30g.jsonl` | `B0BC3382FA0B82486F9CCC5183C638404439A7E8975B8F6A7E5225AA560AC1FC` |
| `dp_release_stack_ab.rs` | `4D4D39E7A0E90CF61FDCC7D46DF71B2456EA76F21B4B59B72F44FD7BC353B101` |
| `dp_release_stack_abba.py` | `577745C44A59DCF8481ADCFA86A2186F91F9C5A524316A1DEBC060BD1C070F1A` |

The larger raw ABBA arms are retained outside the crate package; the checked
in summaries, harness, analyzer, preregistration, and 30-game game records are
sufficient to audit the release decision.
