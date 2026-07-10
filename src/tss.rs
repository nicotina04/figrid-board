//! Threat-space search candidate vocabulary.
//!
//! RQ597 is intentionally a static reference implementation. It identifies
//! non-forcing moves that improve the attacker's next-move threat sources.
//! Proof search and main-search integration remain separate, preregistered
//! work.

use crate::board::{Board, Move, Stone, NUM_CELLS};
use crate::vct::{classify_move_fast_with_flags, ThreatKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuietThreatConfig {
    pub min_gain: u8,
    pub enable_jump_three: bool,
    pub enable_gap_four: bool,
}

impl Default for QuietThreatConfig {
    fn default() -> Self {
        Self {
            min_gain: 1,
            enable_jump_three: true,
            enable_gap_four: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuietThreatCandidate {
    pub mv: Move,
    pub forcing_gains: u8,
    pub winning_gains: u8,
}

/// Generate quiet preparation moves using a full-board semantic reference.
///
/// A candidate is not forcing in the current position, does not allow an
/// immediate opponent Five after it is played, and upgrades at least
/// `config.min_gain` next-move threat sources for the original attacker.
pub fn generate_quiet_threat_candidates(
    board: &mut Board,
    config: QuietThreatConfig,
) -> Vec<QuietThreatCandidate> {
    let attacker = board.side_to_move;
    let history_len = board.history.len();
    let zobrist = board.zobrist;
    let last_move = board.last_move;
    let base = classify_all(board, attacker, config);
    let mut out = Vec::new();

    for mv in 0..NUM_CELLS {
        if !board.is_empty(mv) || base[mv].is_forcing() {
            continue;
        }

        board.make_move(mv);
        if has_immediate_five(board, attacker.opponent(), config) {
            board.undo_move();
            continue;
        }

        let after = classify_all(board, attacker, config);
        let mut forcing_gains = 0u8;
        let mut winning_gains = 0u8;
        for cell in 0..NUM_CELLS {
            let before_kind = base[cell];
            let after_kind = after[cell];
            if after_kind.is_forcing() && threat_strength(after_kind) > threat_strength(before_kind)
            {
                forcing_gains = forcing_gains.saturating_add(1);
                if after_kind.is_winning() && !before_kind.is_winning() {
                    winning_gains = winning_gains.saturating_add(1);
                }
            }
        }
        board.undo_move();

        if forcing_gains >= config.min_gain {
            out.push(QuietThreatCandidate {
                mv,
                forcing_gains,
                winning_gains,
            });
        }
    }

    debug_assert_eq!(board.history.len(), history_len);
    debug_assert_eq!(board.zobrist, zobrist);
    debug_assert_eq!(board.last_move, last_move);
    debug_assert_eq!(board.side_to_move, attacker);

    out.sort_unstable_by(|a, b| {
        b.winning_gains
            .cmp(&a.winning_gains)
            .then_with(|| b.forcing_gains.cmp(&a.forcing_gains))
            .then_with(|| a.mv.cmp(&b.mv))
    });
    out
}

fn classify_all(board: &Board, side: Stone, config: QuietThreatConfig) -> [ThreatKind; NUM_CELLS] {
    let mut kinds = [ThreatKind::None; NUM_CELLS];
    for (cell, kind) in kinds.iter_mut().enumerate() {
        if board.is_empty(cell) {
            *kind = classify_move_fast_with_flags(
                board,
                cell,
                side,
                config.enable_jump_three,
                config.enable_gap_four,
            );
        }
    }
    kinds
}

fn has_immediate_five(board: &Board, side: Stone, config: QuietThreatConfig) -> bool {
    (0..NUM_CELLS).any(|cell| {
        board.is_empty(cell)
            && classify_move_fast_with_flags(
                board,
                cell,
                side,
                config.enable_jump_three,
                config.enable_gap_four,
            ) == ThreatKind::Five
    })
}

fn threat_strength(kind: ThreatKind) -> u8 {
    match kind {
        ThreatKind::None => 0,
        ThreatKind::OpenThree | ThreatKind::JumpThree => 1,
        ThreatKind::ClosedFour => 2,
        ThreatKind::DoubleThree => 3,
        ThreatKind::FourThree => 4,
        ThreatKind::OpenFour | ThreatKind::DoubleFour => 5,
        ThreatKind::Five => 6,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::to_idx;

    #[test]
    fn quiet_pair_builder_is_generated_and_board_is_restored() {
        let mut board = Board::new();
        board.make_move(to_idx(7, 7));
        board.make_move(to_idx(0, 0));
        let before_history = board.history.clone();
        let before_black = board.black;
        let before_white = board.white;
        let before_zobrist = board.zobrist;

        let candidates = generate_quiet_threat_candidates(
            &mut board,
            QuietThreatConfig {
                min_gain: 1,
                ..QuietThreatConfig::default()
            },
        );

        assert!(candidates.iter().any(|c| c.mv == to_idx(7, 8)));
        assert_eq!(board.history, before_history);
        assert!(board.black == before_black);
        assert!(board.white == before_white);
        assert_eq!(board.zobrist, before_zobrist);
        assert_eq!(board.side_to_move, Stone::Black);
    }

    #[test]
    fn direct_forcing_move_is_not_a_quiet_candidate() {
        let mut board = Board::new();
        board.make_move(to_idx(7, 7));
        board.make_move(to_idx(0, 0));
        board.make_move(to_idx(7, 8));
        board.make_move(to_idx(0, 1));

        let forcing = to_idx(7, 9);
        assert!(
            classify_move_fast_with_flags(&board, forcing, Stone::Black, true, true).is_forcing()
        );
        let candidates = generate_quiet_threat_candidates(&mut board, QuietThreatConfig::default());
        assert!(!candidates.iter().any(|c| c.mv == forcing));
    }

    #[test]
    fn candidate_does_not_leave_an_opponent_immediate_five() {
        let mut board = Board::new();
        board.make_move(to_idx(7, 7));
        board.make_move(to_idx(0, 0));
        board.make_move(to_idx(7, 8));
        board.make_move(to_idx(0, 1));
        board.make_move(to_idx(8, 7));
        board.make_move(to_idx(0, 2));
        board.make_move(to_idx(8, 8));
        board.make_move(to_idx(0, 3));

        let block = to_idx(0, 4);
        let candidates = generate_quiet_threat_candidates(&mut board, QuietThreatConfig::default());

        assert!(candidates.iter().all(|candidate| candidate.mv == block));
        assert_eq!(board.side_to_move, Stone::Black);
        assert_eq!(board.history.len(), 8);
    }

    #[test]
    fn generation_order_is_deterministic() {
        let mut board = Board::new();
        board.make_move(to_idx(7, 7));
        board.make_move(to_idx(6, 7));
        let a = generate_quiet_threat_candidates(&mut board, QuietThreatConfig::default());
        let b = generate_quiet_threat_candidates(&mut board, QuietThreatConfig::default());
        assert_eq!(a, b);
    }
}
