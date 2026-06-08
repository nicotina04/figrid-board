//! Speed / regression micro-benchmarks for the figrid engine hot paths.
//!
//! These measure **throughput and latency of code**, not playing strength.
//! Strength is measured separately by the game-level sparring harness
//! (`python/gomoku-sparring`), which carries a much larger noise floor
//! (σ ≈ 5–9 pp). The two are complementary: a change can be faster here and
//! still weaker there, so never read a bench win as an Elo win.
//!
//! Why these exist (issue #3): a cheap, low-noise, deterministic signal makes
//! bold algorithm changes — and removing the `legacy` module — safe to attempt,
//! because a speed regression or a correctness break shows up immediately.
//!
//! Weights: by default these run on `NnueWeights::zeros(..)`, which keeps the
//! bench self-contained (no model file, runs in CI). This is representative:
//! the NNUE forward pass is dense matrix math over fixed dimensions, so its
//! cost is independent of the weight *values*; and every weight file currently
//! in `models/` loads as **sparse** under `GOMOKU_NNUE_CONFIG` (the dense
//! Pattern4 projection branch is dormant), so real weights time the same path.
//! To bench a specific file anyway, set `FIGRID_BENCH_WEIGHTS=/path/to.bin`.
//!
//! The eval group benches two board densities (14-ply and ~60-ply) on purpose:
//! the incremental delta-update only beats a from-scratch refresh once the
//! board is dense enough that full feature extraction exceeds the fixed
//! per-ply incremental overhead (perspective swap + accumulator clones). The
//! crossover is the number that tells you whether incremental is paying off.
//!
//! Run: `cargo bench`            (all groups)
//!      `cargo bench eval`       (filter by name)

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use noru::network::NnueWeights;

use figrid_board::vct::classify_move_fast;
use figrid_board::{evaluate, to_idx, Board, IncrementalEval, Searcher, GOMOKU_NNUE_CONFIG};

/// Bench weights. Defaults to zeros (self-contained); override with
/// `FIGRID_BENCH_WEIGHTS=<path>` to time a real model file.
fn bench_weights() -> NnueWeights {
    match std::env::var("FIGRID_BENCH_WEIGHTS") {
        Ok(path) => {
            let data = std::fs::read(&path)
                .unwrap_or_else(|e| panic!("FIGRID_BENCH_WEIGHTS={path}: {e}"));
            NnueWeights::load_from_bytes(&data, Some(GOMOKU_NNUE_CONFIG.clone()))
                .unwrap_or_else(|e| panic!("load_from_bytes({path}): {e}"))
        }
        Err(_) => NnueWeights::zeros(GOMOKU_NNUE_CONFIG),
    }
}

/// Deterministic ~14-ply midgame position around the centre. No line reaches
/// five, so the position stays `Ongoing` and every hot path has real work to do.
fn midgame_board() -> Board {
    const SEQ: [(usize, usize); 14] = [
        (7, 7), (7, 8), (8, 8), (6, 7), (8, 6), (8, 7), (6, 8),
        (9, 5), (5, 9), (6, 6), (9, 9), (5, 5), (10, 7), (7, 10),
    ];
    let mut b = Board::new();
    for &(r, c) in SEQ.iter() {
        b.make_move(to_idx(r, c));
    }
    b
}

/// Deterministic dense position of `plies` stones. Picks pseudo-random legal
/// moves (fixed xorshift seed → reproducible) and never plays a winning move,
/// so the board stays `Ongoing` and densely populated — the regime where the
/// incremental accumulator is supposed to win.
fn dense_board(plies: usize) -> Board {
    let mut b = Board::new();
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    while b.move_count < plies {
        let moves = b.candidate_moves();
        if moves.is_empty() {
            break;
        }
        // Try a handful of candidates; take the first non-winning one.
        let mut chosen = None;
        for _ in 0..8 {
            let mv = moves[(next() as usize) % moves.len()];
            if !b.check_win(mv) {
                chosen = Some(mv);
                break;
            }
        }
        match chosen {
            Some(mv) => b.make_move(mv),
            None => break, // every sampled move wins — stop before finishing
        }
    }
    b
}

fn bench_eval(c: &mut Criterion) {
    let mut g = c.benchmark_group("eval");
    let weights = bench_weights();

    for (tag, board) in [("p14", midgame_board()), ("p60", dense_board(60))] {
        // Full evaluation from scratch — the baseline the incremental path
        // must beat. Cost grows with stone count (more active features).
        g.bench_function(format!("full/{tag}"), |b| {
            b.iter(|| evaluate(black_box(&board), black_box(&weights)));
        });

        // One incremental ply: make → delta-update → eval → undo. This is the
        // per-node cost inside α-β. Fixed overhead (perspective swap over all
        // cells + accumulator clones) is roughly constant in stone count, so
        // comparing full/{tag} vs incremental_step/{tag} across p14 and p60
        // shows the crossover where delta-update starts paying off.
        g.bench_function(format!("incremental_step/{tag}"), |b| {
            let mut board = board.clone();
            let mut inc = IncrementalEval::new(&weights);
            inc.refresh(&board, &weights);
            let mv = board.candidate_moves()[0];
            b.iter(|| {
                board.make_move(mv);
                inc.push_move(&board, mv, &weights);
                let v = inc.eval(&weights, &board);
                inc.pop_move();
                board.undo_move();
                black_box(v)
            });
        });
    }

    // Full accumulator rebuild (refresh) on the midgame board — what a search
    // does when it cannot reuse incremental state (TT jump / fresh root).
    g.bench_function("incremental_refresh/p14", |b| {
        let board = midgame_board();
        let mut inc = IncrementalEval::new(&weights);
        b.iter(|| inc.refresh(black_box(&board), black_box(&weights)));
    });

    g.finish();
}

fn bench_movegen(c: &mut Criterion) {
    let board = midgame_board();
    c.bench_function("movegen/candidate_moves", |b| {
        b.iter(|| black_box(black_box(&board).candidate_moves()));
    });
}

fn bench_pattern(c: &mut Criterion) {
    let board = midgame_board();
    let moves = board.candidate_moves();
    let side = board.side_to_move;
    c.bench_function("pattern/classify_move_fast", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for &mv in &moves {
                acc = acc.wrapping_add(classify_move_fast(black_box(&board), mv, side) as u32);
            }
            black_box(acc)
        });
    });
}

fn bench_search(c: &mut Criterion) {
    // Fixed-depth, no time limit → deterministic work unit (a coarse
    // nodes/sec proxy). Zero weights flatten move ordering, so the absolute
    // number is pessimistic vs. real weights; track it for *relative* change.
    let weights = bench_weights();
    c.bench_function("search/depth4", |b| {
        b.iter_batched(
            midgame_board,
            |mut board| {
                let mut s = Searcher::new();
                black_box(s.search(&mut board, &weights, 4, None))
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_eval, bench_movegen, bench_pattern, bench_search);
criterion_main!(benches);
