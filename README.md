# figrid-board

A library for the Five-in-a-Row (Gomoku) game, and the first Gomoku engine written in Rust.

## Maintainership

As of 2026-04-20, primary maintainership has been transferred from the original author [wuwbobo2021](https://github.com/wuwbobo2021) to [nicotina04](https://github.com/nicotina04). Future development targets a stronger NNUE-based engine while preserving the existing board/rule/tree library surface.

## Roadmap

- **Gomocup 2026** (submission deadline 2026-05-29 UTC, tournament June 5–7). Goal: submit a substantially stronger successor engine powered by the [noru](https://crates.io/crates/noru) NNUE core.
- **Search strengthening** before submission: VCF / VCT tactical search, iterative deepening, transposition table.
- **Known technical debt** from the 0.3.x series is tracked in [`docs/INHERITED_TODO.md`](docs/INHERITED_TODO.md).

## Build

**Local / development** — target the host CPU for maximum performance:

```
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

**Gomocup submission** — `-C target-cpu=native` is **wrong** here, because
the binary must run on the tournament's machines, not yours. The Gomocup
2026 announcement guarantees SSE4.1, SSE4.2, POPCNT, AVX, and AVX2, which
is exactly `x86_64-v3`. Link the C runtime statically so the submission is
self-contained:

```
RUSTFLAGS="-C target-feature=+crt-static -C target-cpu=x86-64-v3" \
    cargo build --release --target x86_64-pc-windows-gnu \
    --bin pbrain-figrid
```

(An additional build with `-C target-cpu=x86-64-v4` for AVX-512 hardware
may be submitted as a second zip entry; GomocupJudge picks the fastest
compatible binary.)

## Legacy users

Users who need the pre-Rust `figrid-board` as a Linux alternative to Renlib can download [tag v0.20](https://github.com/nicotina04/figrid-board/releases/tag/v0.20).
