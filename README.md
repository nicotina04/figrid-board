<p align="center">
  <img src="https://raw.githubusercontent.com/nicotina04/figrid-board/main/docs/logo.png" alt="figrid-board logo" width="200">
</p>

<h1 align="center">figrid-board</h1>

<p align="center">
  A Rust library and Piskvork-compatible engine for Five-in-a-Row.
</p>

<p align="center">
  <a href="https://crates.io/crates/figrid-board"><img src="https://img.shields.io/crates/v/figrid-board.svg" alt="crates.io"></a>
  <a href="https://crates.io/crates/figrid-board"><img src="https://img.shields.io/crates/l/figrid-board.svg" alt="license"></a>
  <a href="https://docs.rs/figrid-board"><img src="https://docs.rs/figrid-board/badge.svg" alt="docs.rs"></a>
</p>

## What's inside

`figrid-board` provides two public roles:

- **Library** (`figrid_board`) — board representation, rule logic, move generation, threat detection, transposition table, and learned-evaluator surfaces. Reusable from any Rust project that wants Gomoku game state and search primitives without an engine attached.
- **Engine binaries**:
  - `pbrain-figrid` — the current engine. It combines a
    [NORU](https://crates.io/crates/noru) NNUE ordering model with an optional
    [CB2Vec](https://crates.io/crates/cb2vec) codebook leaf evaluator, speaks
    the Piskvork pbrain protocol, and is the binary intended for tournament
    play.

The reusable categorical-codebook layer now lives in the standalone
[CB2Vec (`cb2vec`)](https://github.com/nicotina04/cb2vec) package, published
on [crates.io](https://crates.io/crates/cb2vec). It owns the game-independent
model, quantized artifact, scoring, and reversible token-journal primitives.
`figrid-board` keeps Pattern4 mapping, board updates, Gomoku policy, and
search integration. The dependency is optional and is activated only by the
`codebook-eval` feature; the default board/rules build does not pull it in.

## Download

The [latest release](https://github.com/nicotina04/figrid-board/releases/latest) ships a Windows zip with two
self-contained engines (embedded models, static C runtime, x86_64-v3 CPU required):

| File | Board | Rules |
|---|---|---|
| `pbrain-figrid.exe` | 15×15 | Freestyle, Standard, Renju, Caro (dedicated Caro and Renju models) |
| `pbrain-figrid_20.exe` | 20×20 | Freestyle, Fastgame |

Both pass GomocupJudge `test_zip_rules.py` in all six formats, and the judge picks the `_20` executable for 20×20 games.
Load either one in Piskvork or any pbrain-compatible manager. The release binaries embed trained models that are not
part of the crate package; a build from source with `embed-weights,codebook-eval` embeds the older default models
instead, and `FIGRID_CODEBOOK_WEIGHTS` loads any other model file.

## Features

- Pure Rust, no C dependencies. With embedded weights and a statically linked
  C runtime, the engine can be packaged as one self-contained binary.
- NNUE-based evaluation through [noru](https://crates.io/crates/noru), with incremental accumulator updates.
- Optional `codebook-eval` through the standalone `cb2vec` crate. The embedded
  swap-closed model supplies the deployed quantized leaf evaluator while
  `figrid-board` supplies all game-specific token and search semantics.
- α-β search with transposition table, threat-aware move ordering, killer/history heuristics, late-move pruning, and a quiescence layer for forcing sequences.
- Optional VCF / VCT tactical search at the search root.
- Rule support: Freestyle (`rule 0`), Standard exact-five (`rule 1`), Renju (`rule 4`: black exact five with
  forbidden double-four / double-three / overline, white five or more), and Caro as Gomocup plays it (`rule 9`:
  exactly five, not blocked at both ends by stones; the board edge does not block). Continuous games and bare
  `rule 8` are answered with `ERROR - unsupported rule`.
- Board sizes: 15×15 by default, 20×20 with the `board20` cargo feature (Gomocup Freestyle / Fastgame). The side is
  a compile-time constant, so the two engines are separate builds of the same source.
- Exact-length VCT analysis (`vct::search_vct_exact`): only an actual five ends a line, and the first attack can be
  pinned. Iterating the depth gives the minimal number of attacker moves to a five against every defence the prover
  considers. This is useful for puzzles and move review. The engine's own prover is unchanged.
- Per-rule models: `FIGRID_CODEBOOK_WEIGHTS_{STANDARD,CARO,RENJU}` load a codebook used only under that rule, each
  with its own `NORU_CODEBOOK_EVAL_SCALE_{STANDARD,CARO,RENJU}`; other rules keep `FIGRID_CODEBOOK_WEIGHTS`.
- Optional `avx512` cargo feature: opportunistic ~2× evaluation speedup on AVX-512 hardware, with automatic AVX-2 runtime fallback. Requires Rust ≥ 1.89; off by default so library users on older toolchains and crates.io itself can build.
- Optional `embed-weights` feature: bake the v52-lineage NNUE ordering weights
  into the binary at build time. Enable `codebook-eval` separately to embed
  the packed swap-closed codebook and use the quantized codebook evaluator.
- Compact storage without a runtime representation change. The embedded CBF
  stores exact source weights plus a five-class base-and-i8-residual
  quantized payload. The normal product path reconstructs the established
  flat i16 table from the exact source weights. Direct factored evaluation
  remains an explicit experiment because it was exact but slower in
  end-to-end search.
- Built-in Freestyle White root quiet-move ordering for the embedded quantized codebook.
  It refines only eligible quiet runs and leaves tactical/PV/killer boundaries
  intact. Set `FIGRID_WHITE_ROOT_ORDER=off` for the 0.8.0 ordering path;
  custom and floating-point codebooks, plus non-Freestyle rules, disable it
  automatically.
- Incremental packed Pattern4 windows and exact-order candidate-frontier
  maintenance in `pbrain-figrid`. They reduce repeated board scanning without
  changing evaluation or move order. Search acceleration lives in an optional
  `Searcher` sidecar, leaving the public `Board` layout unchanged from 0.8.1;
  the shipped pbrain enables both paths by default.
- Selective search (1.1). Pre-empting the square of an opponent three is no longer treated as forcing, and a small
  per-move policy (a 213 → 16 → 1 MLP over the codebook evaluator's own state, embedded, 13.8 KB) orders quiet moves
  and prunes late ones at nodes with two or more remaining plies. About 1.3 plies deeper at 2 s per move; positive
  against every engine and rule set it was measured on (see the [changelog](CHANGELOG.md)). On by default in
  `pbrain-figrid`; the policy applies to full-vocabulary dim-32 codebooks such as the release models.
- Exact directional deltas for the quantized codebook evaluator. Make/undo
  applies only changed `(cell, direction)` embeddings, then one activation
  and region delta per affected cell. The shipped pbrain enables this 0.8.3
  path by default; ordinary library `Searcher` instances remain opt-in.

## How the engine developed

figrid was inherited as a small board library in April 2026 and rebuilt as an engine over one season:

1. **Search backbone (0.4–0.6).** A NORU NNUE port with a root VCT/VCF prover, threat-tiered move ordering, a larger
   transposition table, internal iterative reduction and late-move pruning. Reductions and pruning never touch
   forcing moves; an early version that did lost most of its games.
2. **Codebook evaluator (0.7–0.8).** The leaf evaluator moved from the flat NNUE to a categorical codebook over
   11-cell line patterns. An early codebook matched the flat network at about 1/112 of its parameters; the current one
   uses the complete 199,827-pattern vocabulary, because truncating rare patterns turned out to hide exactly the
   tactical ones.
3. **Distilled labels (0.9–0.10).** Value labels come from stronger engines, with a fail-closed rule that keeps a
   position only when two search budgets agree. A clean relabelling gave the largest single gain of the season, and
   rule-specific teachers carried it to Caro, Renju and 20×20.
4. **Speed (1.0).** The whole state-update path stopped repeating work, with search results held identical; this
   converted into strength where an evaluation-only speedup earlier had not.
5. **Selectivity (1.1).** Learned move policies had repeatedly failed to help. Counting where the search spends its
   nodes showed why: two thirds of the tree lies below forcing moves, which the policies never touched, and one kind
   of "forcing" move — pre-empting an opponent's three — was less useful than a quiet move. Narrowing the forcing set
   and letting a cheap policy prune quiet moves made the search about 1.3 plies deeper and stronger against every
   opponent measured.

Every step was gated the same way: refactors must reproduce evaluation dumps and fixed-depth searches exactly, and
strength claims come from matched-seed matches against external engines, replicated on fresh seeds.

## Measured state-update path

Same-binary engineering measurements of the incremental-update work. The last row also has a playing-strength check:
against Pela at 2000 ms per move, the 1.0 search scored 335.5/600 vs 313/600 for 0.10.0 on the same seeds.

| Release | Change | Measured result | Correctness |
|---|---|---|---|
| 0.8.2 | Packed 11-cell Pattern4 windows | 21.11% less fixed-depth time than 0.8.1 | zero mismatches in a 100,000-operation rebuild audit |
| 0.8.2 | Exact-order candidate frontier on top of the packed windows | 1.31% further saving; 22.15% combined | identical decisions and node counts over 1,022 roots |
| 0.8.3 | Exact codebook directional deltas | 19.68% less time without root VCT, 9.25% with it | exact 100,000-operation and 100,000-transition audits; identical decisions and nodes over 1,022 roots |
| 1.0 | Full-vocabulary journal recomputes only the window direction that contains the move; qsearch scores an immediate five without making it | fixed-depth search 1.25–1.55× faster across rules 0/1/4/9 and 20×20 | bestmove, depth and eval identical to 0.10.0 at fixed depth; eval dumps bit-identical |

The generic reversible journal was subsequently extracted into `cb2vec`.
That boundary is an architecture and reuse change, not a separate speed or
strength claim. Details are in the [changelog](CHANGELOG.md) and the
[packed-window and frontier](https://github.com/nicotina04/figrid-board/blob/main/experiments/2026-07-25/dp_a23_release_stack_results.md),
[directional-delta](https://github.com/nicotina04/figrid-board/blob/main/experiments/2026-07-25/cb_d1_directional_delta_results.md),
and [journal extraction](https://github.com/nicotina04/figrid-board/blob/main/experiments/2026-07-25/cb_token_delta_results.md)
reports.

## Quick start

### Use as a Piskvork engine

Build the engine binary:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin pbrain-figrid --features embed-weights,codebook-eval
```

Add `target/release/pbrain-figrid` (or `.exe` on Windows) to Piskvork as an AI player. With `embed-weights,codebook-eval`, the NNUE ordering weights and the packed codebook artifact are both available without external model files.

If you build without `embed-weights`, set `FIGRID_WEIGHTS=path/to/weights.bin`
or place the file at `./models/` so the binary can locate the ordering weights
at startup. In a `codebook-eval` build, `FIGRID_CODEBOOK_WEIGHTS` is the
single codebook knob: unset loads the embedded codebook, a path loads that
model (JSON or the NGCB1 binary format, detected by content, with either the
legacy 4,266-id or the full 199,827-id pattern vocabulary), and
`off`/`0`/`false`/`no`/empty disables the codebook leaf evaluator and returns
to the v52-lineage NNUE leaf evaluator. This fallback is different from the
flat i16 representation used by the normal codebook runtime. The codebook
always runs through the quantized kernel; `NORU_CODEBOOK_EVAL_SCALE=<float>`
overrides the eval scale.

The embedded artifact uses compact factored storage, but direct factored
evaluation is not the default. Leave `NORU_CODEBOOK_FACTORED` unset or set it
to `off` for the established flat i16 runtime. Setting
`NORU_CODEBOOK_FACTORED=on` opts into the exact, memory-smaller direct path;
the 0.8.3 audit measured wall ratios `1.038437` with VCT off and `1.012149`
with product VCT on, so it was not promoted.

`FIGRID_WHITE_ROOT_ORDER` accepts `auto` (default), `on`, or `off`. Explicit
`on` fails closed unless the embedded quantized codebook is active.

The state-update optimizations have independent rollback switches:

- `NORU_PACKED_LINE_WINDOWS=off` restores the 0.8.1 Pattern4 updater.
- `NORU_CANDIDATE_FRONTIER=off` keeps packed windows but restores legacy
  candidate generation.
- `NORU_CODEBOOK_DIRECTIONAL_DELTA=off` restores full accumulator refreshes
  for the quantized codebook evaluator instead of the 0.8.3 directional
  delta journal.

Selective-search switches (on by default since 1.1):

- `NORU_POLICY_MOVE` — unset or empty uses the embedded per-move policy, `off` disables it, any other value is a
  `PGM1` policy file. The policy applies only with a full-vocabulary codebook of matching dimension.
- `NORU_DEMOTE_PREEMPT_THREE=off` treats pre-empting an opponent three as forcing again.

With both off the search is identical to 1.0.

Opt-in search switches (all off by default):

- `NORU_POLICY_ORDER=path/to/table.bin` ranks quiet moves with a learned
  pattern-conditioned policy table (`PCB1v1` format, legacy or full
  vocabulary). Threat tiers and killers keep precedence. Tables whose scores
  could saturate the ordering band are rescaled once at load.
- `NORU_POLICY_REDUCE=on` (requires `NORU_POLICY_ORDER`) adds policy-rank late
  move reductions and an earlier late-move-pruning cutoff.
- `NORU_FORCED_REPLY_RESTRICTION=on` searches only the rules-determined reply
  when the opponent threatens an immediate five (Freestyle and Standard).

An empty value means "default" for every boolean switch. `pbrain-figrid`
fails closed on stale or misspelled switches: any non-empty `NORU_*` /
`FIGRID_*` variable outside its known list (the variables above plus
`NORU_PBRAIN_FIXED_DEPTH`, `NORU_PBRAIN_MAX_DEPTH`, `NORU_SEARCH_PROFILE`,
`NORU_TEST_WEIGHTS`, `FIGRID_BENCH_WEIGHTS`, and `FIGRID_VCT_*`) makes it print
`ERROR unknown engine variable <NAME>` and exit. The effective configuration
is announced once as a `MESSAGE config: ...` line before the first `START`
reply.

Model tools (with `codebook-eval`): `ngcb-convert IN.json OUT.ngcb` converts a
JSON codebook to NGCB1 with a bit-exact round-trip check, and
`FIGRID_CODEBOOK_WEIGHTS=<model> t1-eval-dump --input games.jsonl --output
dump.csv [--stride N]` writes a bit-exact static-eval dump
(`game_id,ply,value_bits`) for comparing builds.

### Use as a library

```toml
[dependencies]
figrid-board = "1"
```

```rust
use figrid_board::{to_idx, Board};

let mut board = Board::new();
board.make_move(to_idx(7, 7)); // black H8 (row 7, col 7, 0-indexed)
board.make_move(to_idx(7, 8)); // white I8 (row 7, col 8)
println!("{:?}", board.side_to_move); // Black (the side about to move)
```

NNUE weights and the search struct are exposed for users who want to drive the engine programmatically rather than through the Piskvork protocol.
The packed-window and candidate-frontier accelerators are off in a newly
constructed `Searcher`; library callers opt in through
`set_use_packed_line_windows` and `set_use_candidate_frontier`. The
directional-delta path has been on by
default since 0.8.6 (`set_use_codebook_directional_delta(false)` restores full
refreshes). The 1.1 selective search is opt-in for library callers through
`set_policy_move(Some(Arc::new(PolicyMove::embedded())))` and `set_demote_preempt_three(true)`. Enabling
`codebook-eval` also activates
the optional `cb2vec` dependency. Consumers that only need the generic
codebook and reversible-journal primitives can use the standalone package
directly.

## Build

**Local / development** — target the host CPU for maximum performance:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

On PowerShell, set `RUSTFLAGS` first:

```powershell
$env:RUSTFLAGS='-C target-cpu=native'
cargo build --release
```

**Reproduce the GitHub Windows x86_64-v3 release asset** with the native MSVC
target, static C runtime, and deterministic linker mode:

```powershell
$env:RUSTFLAGS='-C target-cpu=x86-64-v3 -C target-feature=+crt-static -C link-arg=/Brepro'
cargo build --release --locked --target x86_64-pc-windows-msvc `
    --bin pbrain-figrid `
    --features embed-weights,codebook-eval
```

Release preparation builds this command in two clean target directories and
requires byte-identical executables before packaging.

**20×20 engine** (Gomocup Freestyle / Fastgame): the same recipe with the
`board20` feature, shipped as `pbrain-figrid_20.exe` beside the 15×15 binary:

```powershell
cargo build --release --locked --target x86_64-pc-windows-msvc `
    --bin pbrain-figrid `
    --features embed-weights,codebook-eval,board20
```

**Portable builds with the GNU toolchain** — `-C target-cpu=native` is wrong
for a portable binary because it targets the build host. The Gomocup 2026
tournament machines guaranteed SSE4.1, SSE4.2, POPCNT, AVX, and AVX2, which
matches `x86_64-v3`. With the GNU target instead of MSVC:

```bash
RUSTFLAGS="-C target-feature=+crt-static -C target-cpu=x86-64-v3" \
    cargo build --release --target x86_64-pc-windows-gnu \
    --bin pbrain-figrid --features embed-weights,codebook-eval
```

For an AVX-512-targeted variant, additionally enable the `avx512` cargo
feature. It requires Rust ≥ 1.89 on the build host:

```bash
RUSTFLAGS="-C target-feature=+crt-static -C target-cpu=x86-64-v4" \
    cargo build --release --target x86_64-pc-windows-gnu \
    --bin pbrain-figrid --features embed-weights,codebook-eval,avx512
```

The `x86_64-v4` binary itself requires a compatible machine, so retain the
`x86_64-v3` build as the portable fallback. NORU's `avx512` feature performs
runtime AVX2 fallback only when the surrounding binary is compiled for a
compatible baseline.

## Current direction

1.0 covers every Gomocup board: Freestyle and Fastgame on 20×20, plus Freestyle, Standard, Renju and Caro on 15×15.
Development favours exact, independently checked changes. Speedups must leave fixed-depth search identical, and
strength claims come from matched-seed matches against external engines. Reusable codebook mechanics are developed in
[CB2Vec](https://github.com/nicotina04/cb2vec); Gomoku-specific evaluation, search, and protocol policy remain in
`figrid-board`.

## Maintainership

As of 2026-04-20, primary maintainership has been transferred from the original author [wuwbobo](https://github.com/wuwbobo) to [nicotina04](https://github.com/nicotina04). Future development targets a stronger NNUE-based engine; some of the board / rule / tree library features in the 0.3.x series versions might be refactored and introduced again in the future (if needed).

## Legacy users

Users who need the pre-Rust `figrid-board` as a Linux alternative to Renlib can download [tag v0.20](https://github.com/nicotina04/figrid-board/releases/tag/v0.20).

## Acknowledgments

- [Rapfi](https://github.com/dhbloo/rapfi) for advancing public NNUE work in Gomoku and for serving as a reference point during evaluation development.
- [noru](https://crates.io/crates/noru) for the underlying Rust NNUE training and inference stack.
- [wuwbobo](https://github.com/wuwbobo) for the original engine and for entrusting `figrid-board` to its current maintainer.
- [CB2Vec](https://crates.io/crates/cb2vec) for the reusable categorical
  training, quantization, artifact, scoring, and reversible token-update
  primitives used by the codebook evaluator.

## License

Dual-licensed under either of [MIT](https://opensource.org/licenses/MIT) or [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) at your option, matching the SPDX identifier `MIT OR Apache-2.0` declared in `Cargo.toml`.
