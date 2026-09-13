# Compulsory replies in qsearch

The old qsearch returned a static stand-pat score before checking whether the
opponent could win next turn. Even after searching a mandatory block, it kept
stand-pat as a score floor. A position with no winning counter and two opponent
winning cells could therefore return a false fail-high bound.

The correction checks these threats before static evaluation and the optional
qply cap. An immediate winning counter takes precedence; otherwise compulsory
replies cannot use stand-pat. Necessary replies continue beyond the optional
cap, subject to the existing node and deadline limits. Quiet behavior and
own-win-only positions retain the prior search policy.

Freestyle/Standard double winning cells certify loss on the next opponent turn
when there is no winning counter. Caro needs additional care: closing the far
end can invalidate an otherwise winning exact five. Its compulsory replies
are selected from every legal move by checking all current winning cells after
the defense. Placing a defender stone cannot create a new opponent winning cell.

No public API, model, configuration flag, or search pruning parameter changes.

## Permanent regressions

`src/search_qsearch_tests.rs` covers false fail-highs, forced blocks before and
after the qply cap, winning counters, both Caro defense cases, budget abort and
state restoration, and unchanged quiet cutoffs. The fixture file preserves all
12 false fail-high witnesses from the TRI-D2 investigation.

On the original source, eight new tests fail and the quiet control passes.
All nine pass with the correction, including the test that replays all 12
recorded witnesses. These use exact game rules and deliberately chosen static
scores, rather than treating teacher estimates as proofs.

## Validation on 2026-09-13

- Codebook/embedded-weight build: 141 library tests passed, 9 existing optional
  audits ignored; 6 pbrain tests passed; 5 public API/asset tests passed.
- Library without optional features: 121 passed, 4 existing audits ignored.
- A combined Cargo integration-test command also tries to build historical
  audit binaries. The pre-existing `cb-gh1-graph-census` target references eight
  removed `src/legacy` files and cannot build. The five original public tests
  were run unchanged through a separate test manifest pointing to these files.
  This unrelated historical-tool defect remains open.
- 48 previously selected positions, 65,536 nodes, root VCT disabled: 5 teacher
  utility gains, 4 losses, 39 neutral; 13 move changes. The +/-0.05 repeated
  teacher-utility threshold is descriptive and does not measure win rate.
- Deployed pbrain path, root VCT enabled, 16 paired openings and 250 ms requested
  per turn: corrected engine 18 wins / 14 losses, no draws. All 748 generated
  moves were independently replayed for legality and terminal results. No
  response exceeded the requested 250 ms; maximum observed response 101.65 ms.
  This is a small regression smoke test, not proof of increased strength.

The baseline and corrected pbrain builds use the same 0.8.6 source base, Rust
1.89.0, `--release --offline --locked --features codebook-eval,embed-weights`,
and embedded deployment assets. Openings use fixed seeds 913000 through 913015,
four initial moves, both engine colors, one game at a time, and a 180-ply cap.
No outcome-dependent retuning or additional games were used.

This change repairs a demonstrated invalid bound. The remaining teacher-rated
regressions require separate investigation; the correctness fix is not a claim
that the earlier TRI-D2 performance-selection gate was passed.

## Prospective strength measurement, 2026-09-13/14

A separate fixed-size comparison used 200 new four-stone openings, removing
D4-equivalent colored boards and the previous smoke-test openings. The primary
5-second condition used the first 100 openings and both engine colors (200 games);
the secondary 1-second condition used all 200 openings (400 games). Both used
Freestyle, default root VCT, identical embedded assets, four concurrent game
workers, fresh processes per game, and a 180-ply cap. The existing 150 ms margin
gave effective search budgets of 4,850 and 850 ms. No outcome-based stopping,
retuning, or pooling with the previous experiments was used.

| Requested time | Corrected W/D/L | Score rate | Paired bootstrap 95% interval | Exact paired two-sided p |
|---|---:|---:|---:|---:|
| 5 seconds (primary) | 100 / 2 / 98 | 50.50% | 50.00–51.50% | 1.000 |
| 1 second (secondary) | 201 / 0 / 199 | 50.25% | 49.00–51.50% | 1.000 |

Each opening's color-swapped games form one statistical cluster. The intervals
use 50,000 whole-pair bootstrap replicates. The exact test randomizes the signs
of pair differences, conditional on their magnitudes. Neither condition met
the registered improvement criterion (interval lower bound above 50% and p<0.05).

Only 1 of 100 primary pairs was nonneutral: corrected won both games at seed
913113. Of the other 99 pairs, 98 split wins and one had two capped draws
(seed 913125). At 1 second, corrected swept four pairs and baseline swept three;
193 pairs were neutral. Entire move histories matched across engine color swaps
in 97/100 primary pairs and 176/200 secondary pairs.

With so few nonneutral pairs, the narrow bootstrap interval must not be read as
proof that regressions are impossible. A separately labeled post-hoc sensitivity
bound used the exact 95% upper bound U on nonneutral-pair probability and the
conservative expected-score range 0.5 +/- U/2: 47.67–52.33% at 5 seconds and
46.76–53.24% at 1 second. It does not replace the registered analysis.

All 600 games and 18,654 engine moves passed independent legality, result,
pairing, hash and statistics verification. There were no protocol failures,
illegal moves, or responses exceeding the requested move time. Both draws were
the registered 180-ply cap. The opening population was strongly favorable to
Black; these measurements do not establish strength against other opponents or
opening distributions.

The result supports classifying this patch as a demonstrated correctness repair,
without claiming a measured playing-strength increase. The measured code commit
is `1a7397280fede045b3e58af83546476a37044284`; subsequent validation documentation
does not change the measured engine code.

Machine-readable conditions, per-opening scores, summaries, audit hashes and
post-hoc sensitivity are preserved in
[the validation record](validation/qsearch_strength_2026-09-14.json).
