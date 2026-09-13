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
