# CB-P1 exact bounded-DFPN precondition

Date: 2026-07-26

Status: preregistered before any CB-P1 DFPN row-bearing run

Product baseline: `figrid-board 0.8.3`, commit
`a3efbbe26d507e2d0843948897cfead230d0a70e`

This card is an audit-only prerequisite for a possible proofability or
uncertainty budget head. It does not authorize a learned head, product search
integration, branch pruning, proof-cache promotion, model change, arena, or
release change.

## Prior result and question

The current product VCT is a Boolean depth-first search over a restricted
forcing vocabulary. A direct VCT-versus-alpha-beta budget router already lacks
the required upper bound:

- RQ547a found only one additional solve beyond the approximate product budget
  among 307 consumed tactical positions;
- the later strict RQ550 readjudication classified only 3/307 positions as
  budget-limited;
- RQ566's root-K curve was exactly 12/26 at 1, 2, 5, and 15 seconds;
- RQ602 could not complete a single quiet-move/all-reply query in 45 design
  positions under one second;
- RQ603's two static response reductions covered only 80.0% and 75.6% of
  stable best defenses.

Therefore this card freezes the direct VCT router verdict as
`NO_GO_VCT_PROXY: VCT_CURVE_FLAT`. It will not rerun or rescue that proposal.

The remaining question is narrower:

> On a soundly bounded graph with the current product forcing vocabulary for
> attacker moves, every legal defender reply, and only an actual Freestyle
> five as a winning terminal, can exact DFPN create at least 10/307 additional
> proofs under the same actual expansion total consumed by the fixed
> 65,536-cap reference?

This is deliberately a perfect-information upper bound. Failure closes the
CB-P1 head before training. Passing opens only a separately preregistered
fixed-budget DFPN backend prototype; it does not open a router directly.

## Claim boundary

`ProvenWin` means only:

> From this root, the root side can force an actual Freestyle five within 14
> further plies while its own moves remain in the registered forcing
> vocabulary, against every legal defender move.

It does not prove that a root is a full-game win outside that bounded query.
Conversely, `ExhaustedBounded` means only that this exact bounded query has no
proof. It is never exported as a full-game loss.

`UnknownNodeBudget`, `UnknownMemory`, `UnknownAbort`, and any internal DFPN
threshold return are unknown. They are never a disproof, never a terminal
alpha-beta value, and never stored as an exact proof-cache result.

The consumed tactical corpus is a development/mechanism suite. Its Rapfi
scores, game results, actual moves, class labels, and historical VCT verdicts
are forbidden inputs to the solver and to every gate in this card. No
generalization, calibration, or strength claim is permitted.

## Frozen input

The sole row-bearing input is:

| input | bytes | SHA-256 |
|---|---:|---|
| `rq547a_tactical_positions.jsonl` | 309,683 | `F02663E51716A13F54E0AB22829F7B6FBC7D237F843FAA79BCF62CE3A8EA171F` |

Registered path:

```text
C:\Users\concreate\Documents\workspace\noru-tactic\experiments\2026-07-05\rq547a_tactical_positions.jsonl
```

The file must contain exactly 307 `rq547-tactical-position-v1` rows and one
first-threshold root per `(source_path, game_id)`. The audit executable may
deserialize only:

```text
format, source_path, game_id, ply, side_to_move, position_history
```

Every other key is ignored and is forbidden from the scientific output.
History entries contain exactly `x`, `y`, and `color`. Replay must establish:

- Freestyle, alternating Black/White play from an empty board;
- stored side-to-move equals the replayed side;
- `ply == position_history.len()`;
- every move is in range, empty, and legal;
- before every replayed move, the preceding position is ongoing; a move after
  either color has already completed a five is invalid;
- the root is ongoing;
- all 307 exact `(black, white, side, rule)` roots are unique;
- complete board, history, line-pattern, side, move-count, last-move, and
  Zobrist restoration after every query and replay.

No RQ615C row, teacher label, game outcome, the frozen 1,022-root campaign
trace, arena log, or model-training corpus may be opened by this card.
RQ615C remains reserved for a future head only after a solver prototype passes;
its opening/parent/legal-child connected components must be split before any
future DFPN labels.

## Exact bounded graph

The root attacker is the root `side_to_move`. The root is an OR node.

### OR nodes

Attacker children are exactly the current product forcing generator with:

```text
fast classify                   ON
reach mask                      ON
jump-three attack/defense       OFF
jump-three counter              OFF
gap-four                        OFF
threat index                    OFF
fast-immediate-five shortcut    OFF
scratch-buffer reuse            OFF
```

The vocabulary is therefore the product-default forcing set recognized by the
current classifier. Existing move order is retained, including its exact
threat-priority and cell tie-break.

### AND nodes

Defender children are every legal move in ascending
`cell = y * 15 + x` order. The current `find_defenses_with_counters`, RQ603
footprints, graph signatures, relevance zones, radius filters, top-N filters,
and learned filters are forbidden.

### Terminals and horizon

- At every node, scan the complete Black and White bitboards for an actual
  rule-valid five before consulting draw, horizon, or child generation.
- Exactly one winning color is permitted. Both colors winning is invalid.
- The only positive terminal is an actual Freestyle five found on the
  materialized board.
- A defender five, full-board draw, exhausted registered attacker vocabulary,
  or horizon exhaustion is a disproof only of this bounded query.
- `OpenFour`, `DoubleFour`, `FourThree`, and `DoubleThree` are never terminal
  shortcuts.
- Horizon is exactly 14 further plies. It decreases after every move.
- The graph is a DAG because every edge adds exactly one stone.
- `Board::game_result()`, `last_move`, and history-dependent terminal shortcuts
  are forbidden inside DFPN and replay. Terminal identity is therefore fully
  determined by the registered exact key's bitboards and rule.

## Exact state identity

Every node identity contains:

```text
black.lo, black.hi,
white.lo, white.hi,
side_to_move,
root attacker,
OR/AND role,
remaining horizon,
effective RuleSet,
policy digest
```

The policy digest binds every setting in the previous section. A deterministic
64-bit fingerprint may index the table, but lookup succeeds only after full
field equality. Fingerprint collisions are counted and retained as separate
states. GH1 graph signatures, D4 canonical signatures, product Zobrist alone,
and any model output are forbidden proof keys.

## Proof-number and threshold invariants

Use the classic values:

```text
exact proved leaf:     pn=0,   dn=INF
exact disproved leaf:  pn=INF, dn=0
unexpanded frontier:   pn=1,   dn=1

OR:  pn=min(child pn),  dn=sum(child dn)
AND: pn=sum(child pn),  dn=min(child dn)
```

`INF = 2^60`; additions saturate at `INF`.

For the most-proving child `c1`, ties retain registered child order. Let
`second` be the second-smallest relevant proof number, or `INF` when absent.
Thresholds are:

```text
OR:
  child_pt = min(parent_pt, second_pn + 1)
  child_dt = parent_dt - parent_dn + c1.dn

AND:
  child_pt = parent_pt - parent_pn + c1.pn
  child_dt = min(parent_dt, second_dn + 1)
```

`second + 1` saturates at `INF`. For each expression of the form
`threshold - total + selected`, an `INF` threshold produces `INF`; otherwise
compute `threshold + selected - total` in an exact wider integer and clamp the
result to `[1, INF]`. The DFPN loop invariant requires finite
`total < threshold` before this calculation. Reaching a threshold returns to
the parent; it is not a proof or disproof. Unknown frontier initialization
remains exactly `(1,1)`; no learned or heuristic initialization is allowed.

## Cumulative checkpoints and accounting

Each root owns one resumable deterministic DFPN session. The same table is
advanced cumulatively through:

```text
1,024 / 4,096 / 16,384 / 65,536 / 262,144
```

One expansion is the first complete expansion or terminal evaluation of one
previously unexpanded exact node. Generating its full child ledger is atomic.
Generated edges, recursive calls, threshold returns, exact states,
transposition hits, fingerprint collisions, OR/AND widths, and wall time are
reported but do not change the expansion count. There is no wall deadline.

The 4,096 checkpoint is the possible future observer probe; 65,536 is the
fixed reference; 262,144 is the offline ceiling.

Each root has a deterministic 64 MiB accounted-memory cap:

```text
accounted_bytes =
    144 * exact_state_count
  + 24  * stored_edge_count
  + 32  * distinct_fingerprint_count
  + 24  * collision_entry_count
```

The formula is a conservative registered accounting unit, not an OS allocator
claim. Before a node is committed, its entire child ledger and new exact
states must fit. Otherwise nothing from that expansion is committed and the
root becomes `UnknownMemory`. Actual peak working set is diagnostic.

A recursive-call watchdog of `64 * current_expansion_checkpoint + 1,000,000`
is permitted only to turn a non-progressing implementation into
`UnknownAbort`; it cannot create a solved result.

At every checkpoint record:

- `pn`, `dn`, status, expansions, calls, exact states, and stored edges;
- OR/AND expansion counts and width histogram;
- exact transposition hits and collision count;
- accounted bytes and elapsed time;
- a deterministic digest of the root state and scientific checkpoint fields.

Once `pn=0` or `dn=0`, later checkpoints repeat the exact solved status and
counts without further work.

## Independent proof replay and correctness

Every `ProvenWin` and every `ExhaustedBounded` reached by 262,144 expansions
must pass a separate certificate-DAG replay:

- an OR proof node contains at least one registered proved child;
- an AND proof node contains every legal defender move exactly once and every
  child is proved;
- an OR disproof node contains every registered attacker child exactly once
  and every child is disproved;
- an AND disproof node contains at least one exact disproved legal child;
- regenerated child order and exact child keys match the stored ledger;
- every edge performs a real make/undo;
- every positive terminal is an actual Freestyle five;
- every negative terminal, draw, horizon, and no-attacker-child condition is
  independently recomputed;
- no unknown node or incomplete ledger is reachable from the certificate;
- shared nodes are accepted only after exact-key equality;
- replay restores the root bit-for-bit.

The replayer independently regenerates the graph from the board. It may read
the final exact-node table but not trust stored terminal flags or child
coverage without recomputation.

Library tests must also compare DFPN with a simple exhaustive bounded
OR/all-legal-AND reference on small synthetic horizons, including:

- immediate attacker five;
- defender immediate-five counter;
- irrelevant distant defense;
- no forcing attacker move;
- budget interruption and resume;
- deliberate 64-bit fingerprint alias;
- make/undo restoration.

False proof, missing legal defense, wrong terminal, exact-key alias,
unknown-as-solved reuse, or restoration mismatch is `UNSOUND` and stops the
card.

For determinism, all ceiling positives and the first 32 roots in file order
are rerun in fresh sessions. Every scientific field and certificate digest
must match exactly; elapsed time and process-memory diagnostics are excluded.

## Perfect-oracle upper bound

For each root, define `minimum_proof_cap` as the first registered checkpoint
whose status is `ProvenWin`, or absent.

For each root and cap, `actual_cost(root, cap)` is the expansion count recorded
after advancing that cumulative session to the cap. A session that solved
earlier keeps its smaller terminal expansion count; an unresolved session
normally consumes the cap; a memory/abort exit keeps the exact work consumed
before that exit.

The fixed-reference total budget is the sum of
`actual_cost(root, 65,536)` across all 307 roots. The perfect oracle may assign
each root exactly one cap from:

```text
0 / 1,024 / 4,096 / 16,384 / 65,536 / 262,144
```

A root scores one proof only when the status recorded at its assigned cap is
`ProvenWin`. A cap of zero skips the solver and costs zero. Integer dynamic
programming maximizes proof count subject to:

```text
sum(actual_cost(root, assigned cap)) <=
sum(actual_cost(root, 65,536))
```

Ties choose the lexicographically smallest vector of assigned caps in input
order. The oracle must preserve every fixed-reference proof. This optimistic
oracle is an upper bound; it is not a realizable head result.

`GO_PROTOTYPE` requires all of:

1. input, provenance, correctness, determinism, and certificate errors are 0;
2. fingerprint aliases are 0 and every observed collision is separated by
   exact equality;
3. at most 15/307 roots end in `UnknownMemory`, and no positive certificate
   touches an incomplete ledger;
4. at least 30/307 roots are proved at the 262,144 ceiling;
5. at least 10/307 roots are unproved at 4,096 and proved at a later
   checkpoint;
6. those budget-sensitive roots include at least 3 Black roots and at least
   3 White roots;
7. the perfect oracle preserves every 65,536-reference proof and adds at least
   10/307 proofs, i.e. at least 3.257 percentage points.

The thresholds are fixed before seeing any DFPN outcome. There is no alternate
horizon, vocabulary, budget ladder, memory cap, subset, or threshold rescue.

## Terminal decisions

- provenance, schema, implementation, source, serialization, or incomplete-run
  failure: `INVALID_CB_P1_P0` — no scientific conclusion;
- any false proof, missing defense, exact-state alias, unknown reuse, or
  restoration failure: `UNSOUND`;
- more than 15 memory-capped roots: `NO_GO_STATE_EXPLOSION`;
- valid run missing any remaining upper-bound gate:
  `NO_GO_PRECONDITION`;
- every gate passes: `GO_PROTOTYPE`.

`GO_PROTOTYPE` authorizes only a new preregistration for an audit-only,
fixed-budget backend comparison. A learned codebook head requires another
fresh grouped corpus and another preregistration after that backend passes.

## Build and provenance

The registered toolchain is:

- `rustc 1.88.0 (6b00bc388 2025-06-23)`;
- host `x86_64-pc-windows-msvc`, LLVM `20.1.5`;
- `cargo 1.88.0 (873a06493 2025-05-10)`.

At registration, `Cargo.lock` is 11,841 bytes with SHA-256
`6A6B62449A235ABA53C777484C5D34E18EDB556155B1964A4B2BA6DA7DE2059C`.

The planned default-off feature and executable are:

```text
feature: cb-p1-audit
binary:  cb-p1-census
```

The clean committed release build is:

```powershell
$env:RUSTFLAGS = '-C target-cpu=x86-64-v3'
cargo build --release --locked --features cb-p1-audit --bin cb-p1-census
```

The executable must require release mode, `debug_assertions=false`, runtime
and compile-time AVX2/BMI2/FMA, the exact input seal, a clean worktree before
output creation, a 40-hex HEAD descending from this preregistration, and
create-new output semantics. `NORU_*`, `FIGRID_*`, `RAYON_*`, profiling,
coverage, wrapper, bootstrap, and incremental-compilation environment
variables are forbidden.

The implementation critical-source stream will contain exactly:

```text
Cargo.toml
Cargo.lock
src/lib.rs
src/board.rs
src/vct.rs
src/vct/dfpn.rs
src/pattern_table.rs
bin/cb_p1_census.rs
bin/cb_al1_selector/hash.rs
experiments/2026-07-26/cb_p1_bounded_dfpn_preregister.md
```

Each path is length-delimited before hashing. The compiled bytes, disk bytes,
input file, executable, command line, working directory, git HEAD, toolchain,
CPU features, start/end time, and output seal are recorded. Source and input
seals are checked before and after the run.

Planned outputs are:

```text
experiments/2026-07-26/cb_p1_bounded_dfpn_census.json
experiments/2026-07-26/cb_p1_results.md
```

The JSON is created once. If the run is invalid, its partial file is retained
under a `.invalid` suffix and the registered output path remains absent.
