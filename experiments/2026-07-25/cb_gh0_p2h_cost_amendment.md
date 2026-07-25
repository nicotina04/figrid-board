# CB-GH0 P2-H exact hash-maintenance cost amendment

Date frozen: 2026-07-25 KST

Parent protocol:
`experiments/2026-07-25/cb_gh0_exact_d4_hash_preregister.md`

P1-H result:
`experiments/2026-07-25/cb_gh0_p1h_correctness_results.md`

Opening result commit:
`a5b7972fb08e9e2f0bb7a3fd3356a34407ec79d7`

Opening result:
`OPEN_GH0_HASH_COST_GATE`, authoritative-report SHA-256
`75C96E3FC3B284CF803F58382A36135DE1462C6EE501BDE68D8AD491544BD620`.

Status: frozen before the P2-H timing harness is implemented or compiled and
before any P2-H timing workload is run.

This amendment removes the remaining degrees of freedom in the parent
protocol's maintenance-cost gate. It does not reopen canonical TT score,
bound, or move sharing. Candidate B maintains the exact D4 hashes but no TT,
proof cache, evaluator, move ordering, VCT, or search decision consumes them.

## Outcomes and claim boundary

The authoritative report has exactly four possible labels:

```text
INVALID_CB_GH0_P2
REJECT_GH0_HASH_EXACTNESS
HASH_CORRECT_BUT_TOO_COSTLY
GO_GH0_HASH_SIDECAR
```

`INVALID_CB_GH0_P2` covers source, executable, input, environment, clock,
affinity, arm-order, count, pairing, root-set, or bootstrap-protocol failure.

`REJECT_GH0_HASH_EXACTNESS` covers any incremental/rebuild/unwind mismatch or
any OFF/ON search-result, node, TT, or restored-board mismatch.

When the run is otherwise valid and exact,
`GO_GH0_HASH_SIDECAR` requires both primary metrics to satisfy:

```text
point B/A <= 1.005
one-sided 95% paired-bootstrap upper < 1.01
```

If either primary metric misses either bound, the result is
`HASH_CORRECT_BUT_TOO_COSTLY`.

Even `GO_GH0_HASH_SIDECAR` opens only an affordable default-OFF exact-state
observer for later explicitly registered consumers. It does not authorize:

- canonical score, bound, or move reuse in the orientation-specific TT;
- u64-only proof-cache identity;
- an arena or playing-strength claim;
- a pbrain environment switch;
- a product-default change.

No root, block, repetition, arm, or timing sample may be removed. There is no
outlier rejection, adaptive repetition, cooldown chosen after observation,
alternate seed, alternate estimator, favorable rerun, or rescue run.

Decision precedence is total and fixed:

1. any protocol, identity, environment, clock, count, pairing, or bootstrap
   failure is `INVALID_CB_GH0_P2`;
2. otherwise, any semantic, hash, search, TT, or board mismatch is
   `REJECT_GH0_HASH_EXACTNESS`;
3. otherwise, apply the two cost gates and choose
   `GO_GH0_HASH_SIDECAR` or `HASH_CORRECT_BUT_TOO_COSTLY`.

The first successfully created report is authoritative regardless of label.
An invalid report is terminal and may not be replaced. A preflight failure
before any timing returns without creating a report. A process crash or
external interruption that creates no report may be restarted only after its
reason is recorded; the binary prints no ratio before the final report is
successfully created.

## Frozen inputs and product path

The harness reads these paths relative to `CARGO_MANIFEST_DIR`; they are not
CLI-selectable.

| Input | Bytes | SHA-256 |
|---|---:|---|
| `models/gomoku_codebook_v1_swapclosed.json` | 1,410,562 | `42968FDAB01BA8CCD1DE3DED05C532E4B237DD47EEFFD7AE1C2F264D77BA7DA2` |
| `models/gomoku_codebook_v1_swapclosed_compact_flat.cbf` | 417,412 | `9A5E3D3FC47EEF79468F021F78E9130F5842764F579EE68A2FD270E8289B3250` |
| `data/topk.bin` | 17,060 | `103891DCD1DCD978C654593ABE78EF32C56E2E350B500EE665BC45AC051AA16D` |
| `models/gomoku_v52_5stone_conv_93k.bin` | 14,960,159 | `A961F378A3E73B3CF66C3D15B9A9AB857FA1B81123D98855EE04180A71EAFEFD` |
| `../figrid-dp-campaign/experiments/2026-07-25/dp_a1_fresh_holdout_64g.jsonl` | 317,511 | `1FD40D8948F113AD236FA44F5EEADCA1907C0C3103987CB4C704B67A9B47531A` |

The compact-flat CBF supplies the incumbent flat quantized codebook runtime;
the factored CB-F1 runtime remains OFF. The raw JSON is sealed as the source
identity, not parsed in a timed region. The flat NNUE weights are used only
by the unchanged ordering path required by the current quantized-codebook
search API.

All five files are byte-length/SHA-256 checked before warmup and after all
timed arms. Any change is invalid. The harness also seals its executable,
critical source files, git HEAD, and toolchain. Every critical source must be
tracked in HEAD, and the tracked worktree must be clean both before and after
measurement.

The critical-source list is frozen as:

```text
Cargo.toml
Cargo.lock
src/lib.rs
src/board.rs
src/d4_hash.rs
src/search.rs
src/transposition.rs
src/codebook_eval.rs
src/token_delta.rs
src/pattern_table.rs
src/factored_codebook.rs
bin/cb_gh0_hash_cost.rs
experiments/2026-07-25/cb_gh0_exact_d4_hash_preregister.md
experiments/2026-07-25/cb_gh0_p1h_correctness_amendment.md
experiments/2026-07-25/cb_gh0_p1h_correctness_results.md
experiments/2026-07-25/cb_gh0_p2h_cost_amendment.md
```

Canonical build:

```text
RUSTFLAGS=-C target-cpu=x86-64-v3
cargo build --release --locked --features codebook-eval \
  --bin cb-gh0-hash-cost
```

Runtime `RUSTFLAGS` must equal the canonical string. Any process environment
variable whose name starts with case-insensitive `NORU_` invalidates the run.

The authoritative machine is the existing campaign host:

```text
AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD
16 logical processors
x86_64-pc-windows-msvc
rustc 1.88.0, LLVM 20.1.5
```

## One binary, one process, fixed order

The release binary accepts only:

```text
cb-gh0-hash-cost --out-report NEW.json
```

Unknown, duplicate, or missing options are invalid. Output is create-new and
is written only after all arms, postflight seals, exactness checks, and
bootstrap calculations complete.

One process performs, in this exact order:

1. source/input/executable/environment preflight;
2. Windows clock, priority, and affinity setup;
3. clock calibration;
4. transition warmup;
5. transition `A1 -> B1 -> B2 -> A2`;
6. search warmup;
7. search `A1 -> B1 -> B2 -> A2`;
8. paired bootstraps and decision;
9. input/source/executable postflight;
10. create-new report.

A1 and A2 keep D4 hashing OFF. B1 and B2 keep it ON but unconsumed. No
selector other than this one factor differs.

There are no explicit sleeps, yields, manual cooldowns, or adaptive waits in
the binary.

## Windows clock and scheduling

Primary timing uses raw Windows `QueryPerformanceCounter` ticks. The positive
`QueryPerformanceFrequency` must be identical before and after timing. Gate
calculations use integer ticks; rounded nanoseconds are reporting-only.

The harness records 10,000 back-to-back clock deltas after scheduling setup,
including zero, p50, p95, p99, and maximum ticks. Clock overhead is never
subtracted. Every primary transition block and search root must take at least
one tick.

Before calibration, the harness must:

- read the inherited process and system affinity masks;
- bind its timing thread to the highest-numbered available processor in the
  inherited process mask;
- set its own process to Windows `HIGH_PRIORITY_CLASS`;
- record the old/new masks and priority classes.

Failure to query, set, or verify either setting is invalid. The harness
restores the previous affinity and priority before writing the report; process
exit is the final fallback. Failure to restore either setting is also invalid.
No worker threads are created.

## Precomputed transition tape

The P1-H SplitMix64 policy is regenerated once outside all timed regions:

```text
seed                         0xCB60_2026_0725_0001
transitions                  100,000
makes                         50,090
undos                         49,910
PRNG draws                   150,090
rule switches                   398
maximum move count               180
final move count                 180
final PRNG state             0x840BED2552B4F013
```

Rules cycle before every 251st transition exactly as in P1-H:

```text
Standard -> Caro -> Renju -> legacy Standard -> Freestyle
```

The generator records the exact make move and placed color, undo move and
removed color, and any rule change. PRNG and `legal_moves()` never enter a
timed region. Replaying the tape must reproduce every registered count and
the same 64 state-block digests in every warmup, repetition, and arm.

The tape SHA-256 serialization is frozen in transition order as:

```text
transition index       u32 big-endian, zero-based
rule-before tag        u8, 0xFF for none,
                       0 Freestyle, 1 Standard, 2 Caro,
                       3 Renju, 4 legacy Standard
action tag             u8, 0 make, 1 undo
move                   u16 big-endian
placed/removed color   u8, 0 Black, 1 White
```

The block-state SHA-256 serialization is:

```text
black lo/hi            two u128 big-endian
white lo/hi            two u128 big-endian
side                   u8, 0 Black, 1 White
formal rule            u8, 0 Freestyle, 1 Standard, 2 Caro, 3 Renju
exact5                 u8, 0 false, 1 true
move count             u16 big-endian
last move              u16 big-endian, 0xFFFF for none
history length         u16 big-endian
history moves          repeated u16 big-endian
zobrist                u64 big-endian
Pattern4 IDs           225x4 u16 big-endian in cell-major/direction order
```

## Primary isolated transition workload

The 100,000 transitions are partitioned into exactly 64 consecutive paired
clusters. Let `q=1562`, `r=32`; block `k` starts at:

```text
k*q + min(k,r)
```

Thus blocks 0 through 31 contain 1,563 transitions and blocks 32 through 63
contain 1,562. No block crosses the tape end or overlaps another block.

Warmup replays one complete untimed tape per arm in order `A -> B -> B -> A`.
Each warmup is correctness-checked and fully unwound; none contributes to a
timing or bootstrap.

Each measured arm performs exactly eight repetitions. Every repetition starts
from a fresh Freestyle board and a fresh `BoardSearchState`, replays the full
tape, verifies every block outside the timer, and fully unwinds outside the
timer.

Primary block timing includes only the routed make/undo operations. When a
registered rule switch occurs, the current transition segment is stopped,
the rule change and sidecar synchronization are measured separately, and the
next transition segment starts afterward. Initial sidecar enable/rebuild is
also measured separately.

This separation makes the primary transition metric the incremental
make/undo hot path. Initial enable and 398 rule-domain rebuilds are retained
as mandatory absolute-cost diagnostics. Actual per-search root rebuild cost
is included in the primary whole-search metric below.

For every block and repetition, outside the timer:

- the board fields and Pattern4 cache equal the precomputed reference state;
- the sidecar enabled state equals the arm selector;
- B's eight hashes equal a fresh full rebuild;
- synchronization is exact.

After the eight repetitions, all arms must have identical block-state
digests, final states, and unwind states.

Every warmup and measured repetition ends, outside all timers, by unwinding
all stones, calling `set_rule_set(Freestyle)`, synchronizing the sidecar, and
requiring the complete board, eight B hashes, canonical context, and exact
canonical state to equal a fresh Freestyle empty root. This rule reset is
mandatory even when the tape's last effective rule is not Freestyle.

The report includes:

- 64 primary transition cluster totals for each ABBA arm;
- initial enable/rebuild ticks for every repetition;
- all rule synchronization/rebuild ticks;
- transition-only, rebuild-only, and combined lifecycle totals;
- rounded nanoseconds and per-transition delta;
- p50, p95, and maximum absolute rebuild diagnostics.

Only the transition-only paired ratio is a P2-H primary gate. Combined
lifecycle and component times are diagnostics and cannot rescue or reject the
registered primary result.

## Frozen 1,022-root whole-search workload

The trace parser follows the parent P0 contract exactly. It validates all 64
nonblank games, declared move counts/results, colors, sources, legality, and
terminal placement. Exactly one engine name must contain case-insensitive
`figrid`. In file order, it selects the first 1,022 states for which:

```text
source == "engine" && side_to_move == product_side
```

Sampling stride is one. The 64 games contribute 8 through 36 roots each; no
game block is empty.

Every Searcher uses the incumbent product policy:

- Freestyle;
- compact-flat quantized codebook;
- CB-D1/CB-TD1 directional delta ON;
- packed Pattern4 windows ON;
- exact-order candidate frontier ON;
- White-root ordering ON;
- CB-F1 factored runtime OFF;
- eager/lazy threat field OFF;
- move picker and tail materialization OFF;
- node limit `None`;
- fixed depth 4;
- time limit `None`;
- root VCT and defensive root-VCT veto structurally OFF through the
  per-Searcher audit selector;
- unchanged orientation-specific Zobrist TT, cleared by the existing reset
  before every search.

The harness must first observe the production audit-selector default as ON,
then set it OFF in both arms. The frozen implementation gates both initial
root VCT and the defensive veto as
`use_root_vct_for_audit && process_level_switch`, so ambient timers cannot
enter either arm.

Each game receives a fresh configured Searcher, reused for that game's
selected roots. Searcher construction, input replay, board clone, validation,
and report assembly are outside the timer. Each root timer begins immediately
before `search_codebook_eval_quantized` and ends immediately after return, so
B includes root D4 rebuild/allocation and all descendant make/undo
maintenance.

Before measured arms, the first selected root from each of the 64 games is
searched untimed at depth 4 in order `A -> B -> B -> A`. All warmup outputs
and restored boards must be exact; warmup contributes no timing sample.

Measured search arms then run all 1,022 roots in order
`A1 -> B1 -> B2 -> A2`.

For every root and all four arms, the following must be exactly identical:

- composite root identity `(game_id, seed, ply, actual_move, root_zobrist)`;
- best move, score, completed depth, and returned nodes;
- main nodes, qsearch nodes, and their sum;
- TT probes, hits, cutoffs, stores, displaced-depth-preferred count,
  always-replace count, depth histogram, and final occupancy;
- both bitboards, side to move, effective/formal rule, `exact5`, move count,
  last move, full history, Zobrist, and all 225x4 Pattern4 IDs after return;
- equality of every final board with its pre-search root.

Any mismatch is `REJECT_GH0_HASH_EXACTNESS`; it is never treated as timing
noise.

## ABBA estimators

No individual ratio, mean of ratios, geometric mean, median, trimmed mean, or
outlier filter is used.

For transition block `k`, sum all eight repetitions before division:

```text
A_k = sum_r (A1[r,k] + A2[r,k])
B_k = sum_r (B1[r,k] + B2[r,k])

R_transition = sum_k B_k / sum_k A_k
```

For search game `g`, sum every selected root in that game:

```text
A_g = sum_roots (A1[root] + A2[root])
B_g = sum_roots (B1[root] + B2[root])

R_search = sum_g B_g / sum_g A_g
```

Every A and B cluster total and both global denominators must be positive.
The point gate is evaluated exactly without floating-point rounding:

```text
B/A <= 1.005  iff  B*200 <= A*201
```

Every tick sum and cross product uses checked u128 arithmetic. Overflow is
`INVALID_CB_GH0_P2`; wrapping, saturation, or floating-point fallback is
forbidden.

## Paired cluster bootstrap

Transition uses its 64 consecutive tape blocks. Search uses the actual 64
game blocks. The two bootstraps are independent and frozen as:

```text
replicates       100,000
transition seed  0xCB60_2026_0725_0201
search seed      0xCB60_2026_0725_0202
draws/replicate  64
```

Both use the P1-H SplitMix64 algorithm. Every replicate draws 64 cluster
indices with replacement using `next() & 63`; 64 is a power of two, so there
is no modulo bias. A and B always use the same sampled indices. Repetitions
and roots inside a selected cluster are not resampled.

Each replicate is retained as an exact rational `sum(B)/sum(A)`. The 100,000
ratios are sorted by exact cross multiplication with deterministic
numerator/denominator tie-breaking. The one-sided 95% upper is the ratio at
zero-based sorted index 94,999. There is no interpolation, BCa correction,
studentization, reseeding, or secondary interval.

The upper gate is also exact:

```text
upper < 1.01  iff  upper_num*100 < upper_den*101
```

The report records seeds, draw counts, selected quantile index, exact point
and upper numerators/denominators, decimal renderings, and each Boolean gate.

## Authoritative report

The create-new JSON report contains:

- claim boundary and final decision;
- git/source, toolchain, executable, input, CPU, priority, affinity, build,
  and clock seals;
- all frozen constants and observed protocol counts;
- warmup and measured arm order;
- raw per-arm transition cluster totals and rebuild diagnostics;
- raw per-root search ticks and exact search/TT/board signatures;
- paired transition and game cluster totals;
- exact point/upper rationals and gate decisions;
- first deterministic witness for every failure class;
- preflight and postflight identities.

An independent artifact audit must reproduce all counts, ABBA sums, exact
cross-product decisions, and labels from the raw report before the result is
committed.
