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

## Features

- Pure Rust, no C dependencies. With embedded weights and a statically linked
  C runtime, the engine can be packaged as one self-contained binary.
- NNUE-based evaluation through [noru](https://crates.io/crates/noru), with incremental accumulator updates.
- Optional `codebook-eval` through the standalone `cb2vec` crate. The embedded
  swap-closed model supplies the deployed quantized leaf evaluator while
  `figrid-board` supplies all game-specific token and search semantics.
- α-β search with transposition table, threat-aware move ordering, killer/history heuristics, late-move pruning, and a quiescence layer for forcing sequences.
- Optional VCF / VCT tactical search at the search root.
- Rule support: Freestyle and Standard (exact-five). Renju and Caro currently rejected at the protocol layer.
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
- Exact directional deltas for the quantized codebook evaluator. Make/undo
  applies only changed `(cell, direction)` embeddings, then one activation
  and region delta per affected cell. The shipped pbrain enables this 0.8.3
  path by default; ordinary library `Searcher` instances remain opt-in.

## Measured state-update path

The following are same-binary, preregistered engineering measurements, not
playing-strength claims:

| Card | Change | Frozen result | Correctness |
|---|---|---|---|
| A2 | Packed 11-cell Pattern4 windows | wall ratio `0.78885` versus 0.8.1, or 21.11% less fixed-depth time | zero mismatches in the 100,000-operation rebuild audit |
| A3 | Exact-order candidate frontier on top of A2 | product VCT-ON wall ratio `0.986855`, or 1.31% additional saving; A2 and A3 compound to an indicated 22.15% | identical decisions and node fields over 1,022 roots |
| D1 | Exact codebook directional deltas | VCT-OFF wall ratio `0.803242` and sealed product VCT-ON ratio `0.907485`, or 19.68% and 9.25% less time | exact 100,000-operation and 100,000-transition audits; identical decisions and nodes over 1,022 roots |

The generic reversible journal was subsequently extracted into `cb2vec`.
That boundary is an architecture and reuse change, not a separate speed or
strength claim. Detailed evidence is recorded in the
[0.8.3 changelog](CHANGELOG.md) and the
[A2+A3](https://github.com/nicotina04/figrid-board/blob/main/experiments/2026-07-25/dp_a23_release_stack_results.md),
[D1](https://github.com/nicotina04/figrid-board/blob/main/experiments/2026-07-25/cb_d1_directional_delta_results.md),
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
at startup. In a `codebook-eval` build,
`FIGRID_CODEBOOK_EVAL=off` or `FIGRID_CODEBOOK_WEIGHTS=off` disables the
codebook leaf evaluator and returns to the v52-lineage NNUE leaf evaluator.
This fallback is different from the flat i16 representation used by the
normal codebook runtime.

The embedded artifact uses compact factored storage, but direct factored
evaluation is not the default. Leave `NORU_CODEBOOK_FACTORED` unset or set it
to `off` for the established flat i16 runtime. Setting
`NORU_CODEBOOK_FACTORED=on` opts into the exact, memory-smaller direct path;
the 0.8.3 audit measured wall ratios `1.038437` with VCT off and `1.012149`
with product VCT on, so it was not promoted.

`FIGRID_WHITE_ROOT_ORDER` accepts `auto` (default), `on`, or `off`. Explicit
`on` fails closed unless the embedded quantized codebook is active and no
other root rank/replace/veto hook is configured.

The state-update optimizations have independent rollback switches:

- `NORU_PACKED_LINE_WINDOWS=off` restores the 0.8.1 Pattern4 updater.
- `NORU_CANDIDATE_FRONTIER=off` keeps packed windows but restores legacy
  candidate generation.
- `NORU_CODEBOOK_DIRECTIONAL_DELTA=off` restores full accumulator refreshes
  for the quantized codebook evaluator instead of the 0.8.3 directional
  delta journal.

### Use as a library

```toml
[dependencies]
figrid-board = "0.8"
```

```rust
use figrid_board::{to_idx, Board};

let mut board = Board::new();
board.make_move(to_idx(7, 7)); // black H8 (row 7, col 7, 0-indexed)
board.make_move(to_idx(7, 8)); // white I8 (row 7, col 8)
println!("{:?}", board.side_to_move); // Black (the side about to move)
```

NNUE weights and the search struct are exposed for users who want to drive the engine programmatically rather than through the Piskvork protocol.
The A2, A3, and D1 accelerators are off in a newly constructed `Searcher`;
library callers opt in through `set_use_packed_line_windows`,
`set_use_candidate_frontier`, and
`set_use_codebook_directional_delta`. Enabling `codebook-eval` also activates
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

**Reproduce the portable Gomocup 2026 build** — `-C target-cpu=native` is
wrong for a portable binary because it targets the build host. The 2026
tournament machines guaranteed SSE4.1, SSE4.2, POPCNT, AVX, and AVX2, which
matches `x86_64-v3`. The retained release recipe statically links the C
runtime and embeds both weight assets:

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

The [Gomocup 2026](https://gomocup.org/) submission deadline and June 5–7
tournament have passed. The compatible build recipes remain above for
reproducibility. Current 0.8.x maintenance favors exact, independently
reversible changes with full-rebuild audits and same-binary measurements.
Reusable codebook mechanics are developed in
[CB2Vec](https://github.com/nicotina04/cb2vec); Gomoku-specific
evaluation, search, and protocol policy remain in `figrid-board`.

## Maintainership

As of 2026-04-20, primary maintainership has been transferred from the original author [wuwbobo2021](https://github.com/wuwbobo2021) to [nicotina04](https://github.com/nicotina04). Future development targets a stronger NNUE-based engine; some of the board / rule / tree library features in the 0.3.x series versions might be refactored and introduced again in the future (if needed).

## Legacy users

Users who need the pre-Rust `figrid-board` as a Linux alternative to Renlib can download [tag v0.20](https://github.com/nicotina04/figrid-board/releases/tag/v0.20).

## Acknowledgments

- [Rapfi](https://github.com/dhbloo/rapfi) for advancing public NNUE work in Gomoku and for serving as a reference point during evaluation development.
- [noru](https://crates.io/crates/noru) for the underlying Rust NNUE training and inference stack.
- [wuwbobo2021](https://github.com/wuwbobo2021) for the original engine and for entrusting `figrid-board` to its current maintainer.
- [CB2Vec](https://crates.io/crates/cb2vec) for the reusable categorical
  training, quantization, artifact, scoring, and reversible token-update
  primitives used by the codebook evaluator.

## License

Dual-licensed under either of [MIT](https://opensource.org/licenses/MIT) or [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) at your option, matching the SPDX identifier `MIT OR Apache-2.0` declared in `Cargo.toml`.
