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

use criterion::{Criterion, criterion_group, criterion_main};
use noru::network::NnueWeights;

use figrid_board::board::BoardSearchState;
use figrid_board::pattern_table::{PATTERN_NUM_IDS, swap_mapped_id};
use figrid_board::vct::classify_move_fast;
use figrid_board::{
    BOARD_SIZE, Board, GOMOKU_NNUE_CONFIG, IncrementalEval, NUM_CELLS, Searcher, Stone, evaluate,
    to_idx,
};

const CODEBOOK_DIM: usize = 16;
const CODEBOOK_REGIONS: usize = 9;
const CODEBOOK_FEATURES: usize = CODEBOOK_DIM * CODEBOOK_REGIONS;
const CODEBOOK_FM_RANK: usize = 8;

/// Bench weights. Defaults to zeros (self-contained); override with
/// `FIGRID_BENCH_WEIGHTS=<path>` to time a real model file.
fn bench_weights() -> NnueWeights {
    match std::env::var("FIGRID_BENCH_WEIGHTS") {
        Ok(path) => {
            let data =
                std::fs::read(&path).unwrap_or_else(|e| panic!("FIGRID_BENCH_WEIGHTS={path}: {e}"));
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
        (7, 7),
        (7, 8),
        (8, 8),
        (6, 7),
        (8, 6),
        (8, 7),
        (6, 8),
        (9, 5),
        (5, 9),
        (6, 6),
        (9, 9),
        (5, 5),
        (10, 7),
        (7, 10),
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

struct CodebookBenchModel {
    embeddings: Vec<f32>,
    star_self: [f32; CODEBOOK_DIM],
    star_global: [f32; CODEBOOK_DIM],
    star_bias: [f32; CODEBOOK_DIM],
    head: Vec<f32>,
    factors: Vec<f32>,
}

impl CodebookBenchModel {
    fn new() -> Self {
        Self {
            embeddings: deterministic_vec(PATTERN_NUM_IDS * CODEBOOK_DIM, 0.02),
            star_self: deterministic_array(0xA11C_E001, 0.75),
            star_global: deterministic_array(0xA11C_E002, 0.25),
            star_bias: deterministic_array(0xA11C_E003, 0.01),
            head: deterministic_vec(CODEBOOK_FEATURES, 0.02),
            factors: deterministic_vec(CODEBOOK_FEATURES * CODEBOOK_FM_RANK, 0.02),
        }
    }

    fn per_cell_map(&self, board: &Board, cell_pre: &mut [f32]) {
        debug_assert_eq!(cell_pre.len(), NUM_CELLS * CODEBOOK_DIM);
        cell_pre.fill(0.0);
        let swap = board.side_to_move == Stone::White;
        for (cell, dirs) in board.line_pattern_ids.iter().enumerate() {
            let cell_base = cell * CODEBOOK_DIM;
            for &pid in dirs {
                let pid = if swap { swap_mapped_id(pid) } else { pid };
                let emb_base = pid as usize * CODEBOOK_DIM;
                for d in 0..CODEBOOK_DIM {
                    cell_pre[cell_base + d] += self.embeddings[emb_base + d];
                }
            }
            for d in 0..CODEBOOK_DIM {
                cell_pre[cell_base + d] = cell_pre[cell_base + d].max(0.0);
            }
        }
    }

    fn region_pool(&self, cell_pre: &[f32], features: &mut [f32]) {
        debug_assert_eq!(features.len(), CODEBOOK_FEATURES);
        features.fill(0.0);
        for cell in 0..NUM_CELLS {
            let region = region_of_cell(cell);
            let feature_base = region * CODEBOOK_DIM;
            let cell_base = cell * CODEBOOK_DIM;
            for d in 0..CODEBOOK_DIM {
                features[feature_base + d] += cell_pre[cell_base + d] / 25.0;
            }
        }
    }

    fn star_block(&self, features: &[f32], out: &mut [f32]) {
        debug_assert_eq!(features.len(), CODEBOOK_FEATURES);
        debug_assert_eq!(out.len(), CODEBOOK_FEATURES);
        let mut global = [0.0f32; CODEBOOK_DIM];
        for region in 0..CODEBOOK_REGIONS {
            let base = region * CODEBOOK_DIM;
            for d in 0..CODEBOOK_DIM {
                global[d] += features[base + d] / CODEBOOK_REGIONS as f32;
            }
        }
        for region in 0..CODEBOOK_REGIONS {
            let base = region * CODEBOOK_DIM;
            for d in 0..CODEBOOK_DIM {
                let x = features[base + d] * self.star_self[d]
                    + global[d] * self.star_global[d]
                    + self.star_bias[d];
                out[base + d] = x.max(0.0);
            }
        }
    }

    fn head(&self, features: &[f32]) -> f32 {
        debug_assert_eq!(features.len(), CODEBOOK_FEATURES);
        let mut logit = 0.0f32;
        for (x, w) in features.iter().zip(&self.head) {
            logit += x * w;
        }
        for rank in 0..CODEBOOK_FM_RANK {
            let mut sum = 0.0f32;
            let mut square_sum = 0.0f32;
            for (idx, &x) in features.iter().enumerate() {
                let vx = self.factors[idx * CODEBOOK_FM_RANK + rank] * x;
                sum += vx;
                square_sum += vx * vx;
            }
            logit += 0.5 * (sum * sum - square_sum);
        }
        logit
    }

    fn full_forward(
        &self,
        board: &Board,
        cell_pre: &mut [f32],
        pooled: &mut [f32],
        starred: &mut [f32],
    ) -> f32 {
        self.per_cell_map(board, cell_pre);
        self.region_pool(cell_pre, pooled);
        self.star_block(pooled, starred);
        self.head(starred)
    }
}

fn bench_codebook_eval(c: &mut Criterion) {
    let mut g = c.benchmark_group("codebook_eval");
    let model = CodebookBenchModel::new();

    for (tag, board) in [("p14", midgame_board()), ("p60", dense_board(60))] {
        let mut cell_pre = vec![0.0f32; NUM_CELLS * CODEBOOK_DIM];
        let mut pooled = vec![0.0f32; CODEBOOK_FEATURES];
        let mut starred = vec![0.0f32; CODEBOOK_FEATURES];

        model.per_cell_map(&board, &mut cell_pre);
        model.region_pool(&cell_pre, &mut pooled);
        model.star_block(&pooled, &mut starred);

        g.bench_function(format!("per_cell_map/{tag}"), |b| {
            b.iter(|| model.per_cell_map(black_box(&board), black_box(&mut cell_pre)));
        });
        g.bench_function(format!("region_pool/{tag}"), |b| {
            b.iter(|| model.region_pool(black_box(&cell_pre), black_box(&mut pooled)));
        });
        g.bench_function(format!("star_block/{tag}"), |b| {
            b.iter(|| model.star_block(black_box(&pooled), black_box(&mut starred)));
        });
        g.bench_function(format!("head/{tag}"), |b| {
            b.iter(|| black_box(model.head(black_box(&starred))));
        });
        g.bench_function(format!("full_forward/{tag}"), |b| {
            b.iter(|| {
                black_box(model.full_forward(
                    black_box(&board),
                    black_box(&mut cell_pre),
                    black_box(&mut pooled),
                    black_box(&mut starred),
                ))
            });
        });
    }

    g.finish();
}

fn deterministic_vec(n: usize, scale: f32) -> Vec<f32> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = ((state >> 40) as u32) as f32 / (1u32 << 24) as f32;
            (unit * 2.0 - 1.0) * scale
        })
        .collect()
}

fn deterministic_array(mut state: u64, scale: f32) -> [f32; CODEBOOK_DIM] {
    let mut out = [0.0f32; CODEBOOK_DIM];
    for item in &mut out {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = ((state >> 40) as u32) as f32 / (1u32 << 24) as f32;
        *item = (unit * 2.0 - 1.0) * scale;
    }
    out
}

fn region_of_cell(cell: usize) -> usize {
    let row = cell / BOARD_SIZE;
    let col = cell % BOARD_SIZE;
    let rr = (row / 5).min(2);
    let cc = (col / 5).min(2);
    rr * 3 + cc
}

fn bench_movegen(c: &mut Criterion) {
    let mut group = c.benchmark_group("candidate_frontier");

    for (tag, board) in [("p14", midgame_board()), ("p60", dense_board(60))] {
        let mv = (0..NUM_CELLS)
            .filter(|&cell| board.is_empty(cell))
            .min_by_key(|&cell| {
                let row = cell / BOARD_SIZE;
                let col = cell % BOARD_SIZE;
                row.abs_diff(7) + col.abs_diff(7)
            })
            .expect("benchmark board has an empty cell");

        for (mode, enabled) in [("legacy", false), ("incremental", true)] {
            let mut subject = board.clone();
            let mut search_state = BoardSearchState::new();
            search_state.set_candidate_frontier_enabled(&subject, enabled);
            group.bench_function(format!("query/{mode}/{tag}"), |b| {
                b.iter(|| black_box(black_box(&search_state).candidate_moves(black_box(&subject))));
            });
            group.bench_function(format!("make_undo/{mode}/{tag}"), |b| {
                b.iter(|| {
                    search_state.make_move(&mut subject, black_box(mv));
                    search_state.undo_move(&mut subject);
                    black_box(search_state.candidate_moves(&subject).len());
                });
            });
        }

        group.bench_function(format!("enable_root/{tag}"), |b| {
            b.iter_batched(
                || (board.clone(), BoardSearchState::new()),
                |(subject, mut search_state)| {
                    search_state.set_candidate_frontier_enabled(&subject, true);
                    black_box(search_state.candidate_moves(&subject).len());
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
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

fn bench_packed_window(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_window_update");

    for (tag, board) in [("p14", midgame_board()), ("p60", dense_board(60))] {
        let mv = (0..NUM_CELLS)
            .filter(|&cell| board.is_empty(cell))
            .min_by_key(|&cell| {
                let row = cell / BOARD_SIZE;
                let col = cell % BOARD_SIZE;
                row.abs_diff(7) + col.abs_diff(7)
            })
            .expect("benchmark board has an empty cell");

        for (mode, enabled) in [("legacy", false), ("packed", true)] {
            let mut subject = board.clone();
            let mut search_state = BoardSearchState::new();
            search_state.set_packed_line_windows_enabled(&subject, enabled);
            group.bench_function(format!("{mode}/{tag}"), |b| {
                b.iter(|| {
                    search_state.make_move(&mut subject, black_box(mv));
                    black_box(subject.line_pattern_ids[mv]);
                    search_state.undo_move(&mut subject);
                    black_box(subject.line_pattern_ids[mv]);
                });
            });
        }

        group.bench_function(format!("packed_enable_root/{tag}"), |b| {
            b.iter_batched(
                || (board.clone(), BoardSearchState::new()),
                |(subject, mut search_state)| {
                    search_state.set_packed_line_windows_enabled(&subject, true);
                    black_box(search_state.packed_line_window(mv, 0));
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
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

criterion_group!(
    benches,
    bench_eval,
    bench_codebook_eval,
    bench_movegen,
    bench_pattern,
    bench_packed_window,
    bench_search
);
criterion_main!(benches);
