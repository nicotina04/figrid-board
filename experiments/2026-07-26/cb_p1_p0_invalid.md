# CB-P1 P0 invalid-run record

Date: 2026-07-26 KST
Terminal label: `INVALID_CB_P1_P0`
Scientific conclusion: none
DFPN root calls: 0
Gate evaluation: not performed

## Registered artifacts

- release baseline: `figrid-board 0.8.3`, commit
  `a3efbbe26d507e2d0843948897cfead230d0a70e`;
- preregistration:
  `experiments/2026-07-26/cb_p1_bounded_dfpn_preregister.md`,
  commit `0f0c1e483582a3586a8530342bad8a6019c775ad`,
  16,275 bytes, SHA-256
  `655E71928F41FF469D095AB1E30F08A3C1FBD5AA49D283C4FF2A809604802DD0`;
- input:
  `C:\Users\concreate\Documents\workspace\noru-tactic\experiments\2026-07-05\rq547a_tactical_positions.jsonl`,
  309,683 bytes, SHA-256
  `F02663E51716A13F54E0AB22829F7B6FBC7D237F843FAA79BCF62CE3A8EA171F`.

## Attempt 1: provenance adapter failure

Implementation commit before amendment:
`2a0fdcb215abf0ff289933015479b78392220c33`.

Registered release executable:

- 975,872 bytes;
- SHA-256
  `BCADDA1BA2B384C6D3A547E06C8DC6AC8A796E99AC903B7AB88E257E9DAA761C`.

The executable stopped in `git_identity()` before opening the input. Windows
`Path::canonicalize()` returned a verbatim `\\?\C:\...` path, but that prefix
was not removed before constructing Git's command-local `safe.directory`.
Git therefore rejected the repository ownership. No input row, DFPN node, or
output was produced.

The adapter was fixed by stripping only the Windows verbatim prefix, covered
by a focused unit test, and the implementation commit was amended before any
row-bearing execution.

## Attempt 2: registered input contradiction

Amended implementation commit:
`cc9421612037dc362701516835839a1f2dd274d2`.

Registered release executable:

- 976,896 bytes;
- SHA-256
  `7D5DE3F1D84B1DFC3D8F1CF835718AEBC13A89CCE0A2B297FC97674E1B734228`.

Build and runtime both used the exact registered
`RUSTFLAGS=-C target-cpu=x86-64-v3`. Provenance and the input seal passed.
`load_roots()` then stopped at raw line 59 because it duplicated an earlier
exact `(black, white, side-to-move, Freestyle)` root:

```text
CB-P1 INVALID_CB_P1_P0:
input line 59 duplicates exact (black, white, side, rule) root
```

The implementation constructs the complete root vector before entering the
`run_root()` loop. Consequently this failure occurred with zero DFPN calls
and before any proof status, checkpoint cost, certificate, oracle assignment,
or gate value was observed. The registered JSON output and its `.invalid`
path both remained absent because no partial output had been created.

## Label-blind schema audit

A follow-up audit structurally skipped every forbidden top-level value and
materialized only the six fields permitted by P0. It found:

- 307 raw rows and 307 unique `(source_path, game_id)` pairs;
- 232 unique exact roots;
- 73 duplicate groups: 72 groups of size 2 and one group of size 4;
- 75 excess duplicate instances and 159 singleton roots;
- zero schema, history, post-terminal, or terminal-root errors.

The P0 assertion that all 307 rows were exact-unique is therefore false for
the sealed input. P0 ends here as `INVALID_CB_P1_P0`; it must not be reported
as `NO_GO`, `UNSOUND`, or a solver result.

No replacement root may be selected by difficulty, color, solver cost,
historical label, result, score, or VCT verdict. A follow-up is permitted only
under a fresh preregistration that freezes first-occurrence exact deduplication
before the first DFPN invocation and keeps the original positive gates at
least as strict.
