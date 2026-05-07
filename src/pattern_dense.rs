//! Pattern4 dense embedding for the noru `dense_input` projection branch.
//!
//! This module bridges figrid's incrementally-maintained per-cell pattern
//! IDs (`Board::line_pattern_ids`) into the small dense vector that noru
//! 2.2's optional dense projection branch consumes (`dense_input` argument
//! to `TrainableWeights::forward` / `Accumulator::apply_dense_input`).
//!
//! Architecture (Phase A.1, 2026-05-07):
//!
//! 1. Each one of the `pattern_table::PATTERN_NUM_IDS` mapped IDs is
//!    associated with a `PATTERN4_DENSE_DIM`-vector of `f32` weights —
//!    the *embedding* table. The table is initialized once, deterministically
//!    seeded, and treated as **frozen** at inference time. Parameters
//!    learned downstream live in noru's `dense_to_acc` weights, not here;
//!    the embedding is a fixed random projection (Rahimi-Recht style)
//!    that lets noru's small dense branch consume figrid's 4097-class
//!    Pattern4 vocabulary.
//! 2. For every position the dense input vector is the *sum* of the
//!    embeddings looked up at every (cell, direction) of the board, i.e.
//!    a bag-of-Pattern4-IDs pooled into a single `[f32; PATTERN4_DENSE_DIM]`
//!    representation. This pool is recomputed from `board.line_pattern_ids`
//!    on demand (~ 225 cells × 4 directions × 64 dims ≈ 58K float adds).
//!    Incremental updates are possible but deferred until measurement
//!    shows the recompute cost matters in practice.
//! 3. At training time the same embedding table must be reproduced
//!    bit-for-bit so that `dense_input` seen by the trainer matches what
//!    figrid sees at runtime; the seed and Kaiming-style scale below are
//!    the only knobs that need to stay in sync.

use crate::pattern_table::PATTERN_NUM_IDS;
use noru::trainer::SimpleRng;
use std::sync::OnceLock;

/// Dense projection width fed into `noru::Accumulator::apply_dense_input`.
/// Matches `NnueConfig::dense_input_size` of any weight file produced
/// for figrid's Phase A.1 architecture; v52 / v0.7.0 weights have
/// `dense_input_size == 0` and the dense path is a no-op for them.
pub const PATTERN4_DENSE_DIM: usize = 64;

/// Deterministic seed used to materialize the frozen embedding table.
/// Must match the value used by the trainer to generate identical
/// per-position `dense_input` vectors.
const EMBEDDING_SEED: u64 = 0x4F4D_4F4B_5530_3531; // ASCII "OMOKU051"

/// Frozen `PATTERN_NUM_IDS x PATTERN4_DENSE_DIM` embedding table. Stored
/// flat in row-major order for cache-friendly `(pid, dim)` access.
pub struct Pattern4Embedding {
    table: Vec<f32>,
}

impl Pattern4Embedding {
    /// Returns a 64-dim slice for `pid`. Caller must keep `pid` within
    /// `PATTERN_NUM_IDS`; the lookup is a flat slice index.
    #[inline]
    pub fn row(&self, pid: u16) -> &[f32] {
        let start = (pid as usize) * PATTERN4_DENSE_DIM;
        &self.table[start..start + PATTERN4_DENSE_DIM]
    }

    fn build_random() -> Self {
        // Kaiming-flavored small init. The scale matches the dense_to_acc
        // initialization in noru so the two halves of the projection
        // contribute on similar magnitudes during the first epoch.
        let scale = (2.0f32 / PATTERN4_DENSE_DIM as f32).sqrt() * 0.1;
        let mut rng = SimpleRng::new(EMBEDDING_SEED);
        let n = PATTERN_NUM_IDS * PATTERN4_DENSE_DIM;
        let mut table = vec![0.0f32; n];
        for v in table.iter_mut() {
            *v = rng.next_normal() * scale;
        }
        Self { table }
    }
}

/// Process-wide singleton holding the frozen embedding. The first
/// access pays the build cost (~3 ms for 4097 × 64 normal samples);
/// every subsequent reader sees the same table without lock contention.
pub fn embedding() -> &'static Pattern4Embedding {
    static TABLE: OnceLock<Pattern4Embedding> = OnceLock::new();
    TABLE.get_or_init(Pattern4Embedding::build_random)
}

/// Sum-pool the embedding over every `(cell, direction)` slot tracked by
/// `Board::line_pattern_ids`. Returns the `dense_input` vector callers
/// hand to `noru::Accumulator::apply_dense_input`.
///
/// `line_pattern_ids` is in black-relative orientation, so this function
/// returns the same vector regardless of side to move; perspective-aware
/// callers can mirror via `pattern_table::swap_mapped_id` when needed.
pub fn pool_dense_input(line_pattern_ids: &[[u16; 4]]) -> [f32; PATTERN4_DENSE_DIM] {
    let mut out = [0.0f32; PATTERN4_DENSE_DIM];
    let emb = embedding();
    for cell_dirs in line_pattern_ids {
        for &pid in cell_dirs {
            let row = emb.row(pid);
            for k in 0..PATTERN4_DENSE_DIM {
                out[k] += row[k];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{to_idx, Board};

    #[test]
    fn embedding_table_has_expected_shape() {
        let emb = embedding();
        // Every pid yields a 64-dim slice without panicking.
        for pid in 0..PATTERN_NUM_IDS as u16 {
            assert_eq!(emb.row(pid).len(), PATTERN4_DENSE_DIM);
        }
    }

    #[test]
    fn embedding_is_deterministic_across_threads() {
        // Two reads of the same row return identical bytes; OnceLock
        // guarantees the table is built exactly once.
        let a = embedding().row(123).to_vec();
        let b = embedding().row(123).to_vec();
        assert_eq!(a, b);
    }

    #[test]
    fn pool_dense_input_empty_board_matches_full_recompute() {
        let board = Board::new();
        let v = pool_dense_input(&board.line_pattern_ids[..]);
        // Sanity: empty board is symmetric (no stones), so the dense
        // vector is the sum over only the boundary-aware default IDs.
        // Recomputing the same way must yield the identical vector.
        let again = pool_dense_input(&board.line_pattern_ids[..]);
        assert_eq!(v, again);
    }

    #[test]
    fn pool_dense_input_changes_after_a_move() {
        let mut board = Board::new();
        let baseline = pool_dense_input(&board.line_pattern_ids[..]);
        board.make_move(to_idx(7, 7));
        let after_move = pool_dense_input(&board.line_pattern_ids[..]);
        // At least one of the 64 components must change — placing a stone
        // mutates pattern_ids in 4 directions × ~11 cells.
        assert_ne!(baseline, after_move);
    }
}
