<p align="center">
  <img src="https://raw.githubusercontent.com/nicotina04/figrid-board/main/docs/logo.png" alt="figrid-board logo" width="200">
</p>

<h1 align="center">figrid-board</h1>

<p align="center">
  A library for the Five-in-a-Row (Gomoku) game, and the first Gomoku engine written in Rust.
</p>

<p align="center">
  <a href="https://crates.io/crates/figrid-board"><img src="https://img.shields.io/crates/v/figrid-board.svg" alt="crates.io"></a>
  <a href="https://crates.io/crates/figrid-board"><img src="https://img.shields.io/crates/l/figrid-board.svg" alt="license"></a>
  <a href="https://docs.rs/figrid-board"><img src="https://docs.rs/figrid-board/badge.svg" alt="docs.rs"></a>
</p>

## What's inside

`figrid-board` ships in two roles from a single crate:

- **Library** (`figrid_board`) — board representation, rule logic, move generation, threat detection, transposition table, and an NNUE evaluation surface. Reusable from any Rust project that wants Gomoku game state and search primitives without an engine attached.
- **Engine binaries**:
  - `pbrain-figrid` — the NNUE engine, powered by [noru](https://crates.io/crates/noru). Speaks the Piskvork pbrain protocol and is the binary intended for tournament play.
  - `pbrain-figrid-legacy` — preserves the original pre-NNUE engine by [wuwbobo2021](https://github.com/wuwbobo2021), kept as a reference baseline.

## Features

- Pure Rust, no C dependencies. Single static binary after build.
- NNUE-based evaluation through [noru](https://crates.io/crates/noru), with incremental accumulator updates.
- α-β search with transposition table, threat-aware move ordering, killer/history heuristics, late-move pruning, and a quiescence layer for forcing sequences.
- Optional VCF / VCT tactical search at the search root.
- Rule support: Freestyle and Standard (exact-five). Renju and Caro currently rejected at the protocol layer.
- Optional `avx512` feature on the noru dependency: opportunistic ~2× evaluation speedup on AVX-512 hardware, with automatic AVX-2 fallback.
- Optional `embed-weights` feature: bake the NNUE weights into the binary at build time, producing a single self-contained executable suitable for tournament submission.

## Quick start

### Use as a Piskvork engine

Build the engine binary:

```
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin pbrain-figrid --features embed-weights
```

Add `target/release/pbrain-figrid` (or `.exe` on Windows) to Piskvork as an AI player. The `embed-weights` feature bundles the NNUE weights into the executable so you do not have to ship them separately.

If you build without `embed-weights`, set `FIGRID_WEIGHTS=path/to/weights.bin` or place the file at `./models/` so the binary can locate it at startup.

### Use as a library

```toml
[dependencies]
figrid-board = "0.6"
```

```rust
use figrid_board::{to_idx, Board};

let mut board = Board::new();
board.make_move(to_idx(7, 7)); // black H8 (row 7, col 7, 0-indexed)
board.make_move(to_idx(7, 8)); // white I8 (row 7, col 8)
println!("{:?}", board.side_to_move); // Black (the side about to move)
```

NNUE weights and the search struct are exposed for users who want to drive the engine programmatically rather than through the Piskvork protocol.

## Build

**Local / development** — target the host CPU for maximum performance:

```
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

**Gomocup submission** — `-C target-cpu=native` is **wrong** here, because the binary must run on the tournament's machines, not yours. The Gomocup 2026 announcement guarantees SSE4.1, SSE4.2, POPCNT, AVX, and AVX2, which is exactly `x86_64-v3`. Link the C runtime statically and embed the NNUE weights so the submission is one self-contained file:

```
RUSTFLAGS="-C target-feature=+crt-static -C target-cpu=x86-64-v3" \
    cargo build --release --target x86_64-pc-windows-gnu \
    --bin pbrain-figrid --features embed-weights
```

An additional build with `-C target-cpu=x86-64-v4` for AVX-512 hardware may be submitted as a second zip entry; GomocupJudge picks the fastest compatible binary.

## Roadmap

Pushing the engine harder for [Gomocup 2026](https://gomocup.org/) (submission deadline 2026-05-29 UTC, tournament June 5–7). Strength work continues on every layer — search, evaluation, and training data — because there is always more performance to squeeze out before the deadline.

Pre-NNUE technical debt inherited from the 0.3.x series is tracked in [`docs/INHERITED_TODO.md`](docs/INHERITED_TODO.md).

## Maintainership

As of 2026-04-20, primary maintainership has been transferred from the original author [wuwbobo2021](https://github.com/wuwbobo2021) to [nicotina04](https://github.com/nicotina04). Future development targets a stronger NNUE-based engine while preserving the existing board / rule / tree library surface.

## Legacy users

Users who need the pre-Rust `figrid-board` as a Linux alternative to Renlib can download [tag v0.20](https://github.com/nicotina04/figrid-board/releases/tag/v0.20).

## Acknowledgments

- [wuwbobo2021](https://github.com/wuwbobo2021) for the original engine and for entrusting `figrid-board` to its current maintainer.
- [Rapfi](https://github.com/dhbloo/rapfi) for advancing public NNUE work in Gomoku and for serving as a reference point during evaluation development.
- [noru](https://crates.io/crates/noru) for the underlying Rust NNUE training and inference stack.

## License

Dual-licensed under either of [MIT](https://opensource.org/licenses/MIT) or [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) at your option, matching the SPDX identifier `MIT OR Apache-2.0` declared in `Cargo.toml`.
