//! Per-move policy: a small shared-weight MLP that scores quiet candidate moves from the codebook evaluator's own
//! incremental state. The search uses the scores to order quiet moves and to prune and reduce late ones (see
//! [`crate::search::Searcher::set_policy_move`]).
//!
//! Inputs per empty cell, side-to-move perspective (`hin = 6·dim + 21`):
//! the cell's codebook activation `a_c` (dim), board planes (empty, own, opponent), ReLU of the four direction
//! embeddings (4·dim), their pairwise interaction `½[(Σe)² − Σe²]` (dim), and one-hot own and opponent threat kinds
//! (9 each). Hidden layer: `width` ReLU units; output: one logit.
//!
//! File format `PGM1` (little-endian): b"PGM1", u32 hin, u32 width, f32 w1[width][hin], f32 b1[width],
//! f32 w2[width], f32 b2.

/// The policy shipped with figrid 1.1 (width 16, trained on strong-engine moves over a dim-32 full-vocabulary
/// codebook). It applies to any full-vocabulary dim-32 codebook.
#[cfg(feature = "codebook-eval")]
pub const EMBEDDED_POLICY_MOVE: &[u8] = include_bytes!("../models/policy_move_v1.pgm1");

pub struct PolicyMove {
    hin: usize,
    width: usize,
    /// `[hin][width]`
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: f32,
}

fn read_f32s(bytes: &[u8], at: &mut usize, n: usize) -> Result<Vec<f32>, String> {
    let end = *at + 4 * n;
    let slice = bytes.get(*at..end).ok_or("PGM1 file truncated")?;
    *at = end;
    Ok(slice.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn read_u32(bytes: &[u8], at: &mut usize) -> Result<u32, String> {
    let s = bytes.get(*at..*at + 4).ok_or("PGM1 file truncated")?;
    *at += 4;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

impl PolicyMove {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.get(..4) != Some(b"PGM1") {
            return Err("not a PGM1 policy file".into());
        }
        let mut at = 4;
        let hin = read_u32(bytes, &mut at)? as usize;
        let width = read_u32(bytes, &mut at)? as usize;
        if hin < 21 || (hin - 21) % 6 != 0 || width == 0 {
            return Err(format!("PGM1 file has unsupported shape hin={hin} width={width}"));
        }
        let raw = read_f32s(bytes, &mut at, width * hin)?;
        let mut w1 = vec![0f32; hin * width];
        for o in 0..width {
            for i in 0..hin {
                w1[i * width + o] = raw[o * hin + i];
            }
        }
        let b1 = read_f32s(bytes, &mut at, width)?;
        let w2 = read_f32s(bytes, &mut at, width)?;
        let b2 = read_f32s(bytes, &mut at, 1)?[0];
        if at != bytes.len() {
            return Err(format!("PGM1 file has {} trailing bytes", bytes.len() - at));
        }
        Ok(Self { hin, width, w1, b1, w2, b2 })
    }

    /// The bundled model ([`EMBEDDED_POLICY_MOVE`]).
    #[cfg(feature = "codebook-eval")]
    pub fn embedded() -> Self {
        Self::from_bytes(EMBEDDED_POLICY_MOVE).expect("embedded policy model is valid")
    }

    /// Number of inputs per move (`6·dim + 21`).
    pub fn input_len(&self) -> usize {
        self.hin
    }

    /// The codebook embedding dimension this model expects.
    pub fn codebook_dim(&self) -> usize {
        (self.hin - 21) / 6
    }

    /// Logit of one candidate from its `hin` inputs; `hidden` is a reusable buffer.
    pub fn score(&self, x: &[f32], hidden: &mut Vec<f32>) -> f32 {
        debug_assert_eq!(x.len(), self.hin);
        hidden.clear();
        hidden.extend_from_slice(&self.b1);
        for (i, &v) in x.iter().enumerate() {
            if v == 0.0 {
                continue;
            }
            let w = &self.w1[i * self.width..(i + 1) * self.width];
            for (h, &wv) in hidden.iter_mut().zip(w) {
                *h += wv * v;
            }
        }
        let mut logit = self.b2;
        for (h, &wv) in hidden.iter().zip(&self.w2) {
            logit += h.max(0.0) * wv;
        }
        logit
    }
}

#[cfg(all(test, feature = "codebook-eval"))]
mod tests {
    use super::*;
    use crate::board::{Board, NUM_CELLS, Stone};
    use crate::codebook_eval::{CodebookWeights, IncrementalQuantizedCodebookEval};

    #[test]
    fn embedded_model_loads() {
        let m = PolicyMove::embedded();
        assert_eq!(m.input_len(), 213);
        assert_eq!(m.codebook_dim(), 32);
    }

    #[test]
    fn rejects_bad_files() {
        assert!(PolicyMove::from_bytes(b"PGC1").is_err());
        let mut bytes = EMBEDDED_POLICY_MOVE.to_vec();
        bytes.push(0);
        assert!(PolicyMove::from_bytes(&bytes).is_err());
        assert!(PolicyMove::from_bytes(&EMBEDDED_POLICY_MOVE[..100]).is_err());
    }

    /// End to end: engine inputs (codebook state + tokens) and MLP vs PyTorch logits on quiet empty cells.
    /// `PGM_MODEL=x.pgm1 PGM_FIXTURE=x.pgm1.fixture.bin PGM_CODEBOOK=ng3.ngcb cargo test --release --features
    /// codebook-eval engine_inputs_match_pytorch -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn engine_inputs_match_pytorch() {
        let (Ok(model), Ok(fixture), Ok(cb)) =
            (std::env::var("PGM_MODEL"), std::env::var("PGM_FIXTURE"), std::env::var("PGM_CODEBOOK"))
        else {
            return;
        };
        let pgm = PolicyMove::from_bytes(&std::fs::read(model).unwrap()).unwrap();
        let weights = CodebookWeights::from_bytes_auto(&std::fs::read(cb).unwrap()).unwrap();
        crate::pattern_table::warm_full_vocab();
        let q = weights.quantize_i16_s32_s64();
        let fx = std::fs::read(fixture).unwrap();
        let n = u32::from_le_bytes(fx[0..4].try_into().unwrap()) as usize;
        let rec = 2 + NUM_CELLS * 19;
        let (mut max_err, mut top_agree, mut scored) = (0f32, 0usize, 0usize);
        let mut x = vec![0f32; pgm.input_len()];
        let mut hidden = Vec::new();
        for k in 0..n {
            let base = 4 + k * (rec + NUM_CELLS * 4);
            let r = &fx[base..base + rec];
            let want: Vec<f32> = fx[base + rec..base + rec + NUM_CELLS * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let cells = &r[2..2 + NUM_CELLS];
            let my = &r[2 + NUM_CELLS + NUM_CELLS * 16..2 + NUM_CELLS * 18];
            let op = &r[2 + NUM_CELLS * 18..2 + NUM_CELLS * 19];
            let own: Vec<usize> = (0..NUM_CELLS).filter(|&c| cells[c] == 1).collect();
            let opp: Vec<usize> = (0..NUM_CELLS).filter(|&c| cells[c] == 2).collect();
            let black_to_move = own.len() == opp.len();
            let (black, white) = if black_to_move { (&own, &opp) } else { (&opp, &own) };
            let mut board = Board::new();
            for i in 0..black.len().max(white.len()) {
                if i < black.len() {
                    board.make_move(black[i]);
                }
                if i < white.len() {
                    board.make_move(white[i]);
                }
            }
            assert_eq!(board.side_to_move == Stone::Black, black_to_move);
            let mut inc = IncrementalQuantizedCodebookEval::new(&q);
            inc.refresh(&board, &q);
            inc.materialize_for_policy(&q);
            let (mut best_e, mut best_t) = ((f32::MIN, 0), (f32::MIN, 0));
            for c in 0..NUM_CELLS {
                if cells[c] != 0 || my[c] != 0 || op[c] != 0 {
                    continue;
                }
                inc.policy_move_inputs_with_access(&board, &q, c, 0, 0, &mut x);
                let got = pgm.score(&x, &mut hidden);
                max_err = max_err.max((got - want[c]).abs());
                scored += 1;
                if got > best_e.0 {
                    best_e = (got, c);
                }
                if want[c] > best_t.0 {
                    best_t = (want[c], c);
                }
            }
            top_agree += (best_e.1 == best_t.1) as usize;
        }
        eprintln!("pgm parity: {n} positions, {scored} quiet cells, max |logit diff| {max_err:.3e}, quiet argmax agree {top_agree}/{n}");
        assert!(top_agree * 10 >= n * 9);
    }
}
