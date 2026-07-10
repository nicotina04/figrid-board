//! Threat-space search candidate vocabulary.
//!
//! RQ597 is intentionally a static reference implementation. It identifies
//! non-forcing moves that improve the attacker's next-move threat sources.
//! Proof search and main-search integration remain separate, preregistered
//! work.

use crate::board::{BitBoard, Board, Move, Stone, BOARD_SIZE, NUM_CELLS};
use crate::heuristic::DIR;
use crate::pattern_table::{classify_window_after_play_with_flags, read_window, WindowThreat};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DependencyQuietCandidate {
    pub mv: Move,
    pub forcing_gains: u8,
    pub winning_gains: u8,
    pub dependency_links: u16,
    pub max_reused_support: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DependencyCandidateArms {
    pub d1: Vec<DependencyQuietCandidate>,
    pub d2: Vec<DependencyQuietCandidate>,
}

#[derive(Clone, Copy)]
struct ThreatLine {
    footprint: BitBoard,
    support: BitBoard,
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

/// Generate the two preregistered RQ598 dependency arms in one semantic pass.
///
/// D1 requires an upgraded forcing line to intersect an existing forcing-line
/// footprint. D2 additionally requires at least two pre-existing attacker
/// stones to be shared by one potential/existing line pair.
pub fn generate_dependency_quiet_candidates(
    board: &mut Board,
    config: QuietThreatConfig,
) -> DependencyCandidateArms {
    let attacker = board.side_to_move;
    let history_len = board.history.len();
    let zobrist = board.zobrist;
    let last_move = board.last_move;
    let original_black = board.black;
    let original_white = board.white;
    let original_mine = if attacker == Stone::Black {
        original_black
    } else {
        original_white
    };
    let base = classify_all(board, attacker, config);
    let base_directions = classify_all_directions(board, attacker, config);
    let existing_lines = collect_existing_lines(board, &base, &base_directions, &original_mine);
    let mut d1 = Vec::new();
    let mut d2 = Vec::new();

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
        let after_directions = classify_all_directions(board, attacker, config);
        let mut forcing_gains = 0u8;
        let mut winning_gains = 0u8;
        let mut dependency_links = 0u16;
        let mut max_reused_support = 0u8;

        for source in 0..NUM_CELLS {
            let before_kind = base[source];
            let after_kind = after[source];
            if !after_kind.is_forcing()
                || threat_strength(after_kind) <= threat_strength(before_kind)
            {
                continue;
            }
            forcing_gains = forcing_gains.saturating_add(1);
            if after_kind.is_winning() && !before_kind.is_winning() {
                winning_gains = winning_gains.saturating_add(1);
            }

            for dir_idx in 0..DIR.len() {
                let before_line = base_directions[source][dir_idx];
                let after_line = after_directions[source][dir_idx];
                if !direction_is_forcing(after_line)
                    || direction_strength(after_line) <= direction_strength(before_line)
                {
                    continue;
                }
                let potential = make_threat_line(source, dir_idx, &original_mine);
                for existing in &existing_lines {
                    if bitboards_intersect(&potential.footprint, &existing.footprint) {
                        dependency_links = dependency_links.saturating_add(1);
                    }
                    let reused = intersection_count(&potential.support, &existing.support) as u8;
                    max_reused_support = max_reused_support.max(reused);
                }
            }
        }
        board.undo_move();

        let candidate = DependencyQuietCandidate {
            mv,
            forcing_gains,
            winning_gains,
            dependency_links,
            max_reused_support,
        };
        if forcing_gains >= 1 && dependency_links > 0 {
            d1.push(candidate);
        }
        if forcing_gains >= 1 && max_reused_support >= 2 {
            d2.push(candidate);
        }
    }

    debug_assert_eq!(board.history.len(), history_len);
    debug_assert_eq!(board.zobrist, zobrist);
    debug_assert_eq!(board.last_move, last_move);
    debug_assert_eq!(board.side_to_move, attacker);
    debug_assert!(board.black == original_black);
    debug_assert!(board.white == original_white);

    sort_dependency_candidates(&mut d1);
    sort_dependency_candidates(&mut d2);
    debug_assert!(d2.iter().all(|candidate| {
        d1.iter()
            .any(|d1_candidate| d1_candidate.mv == candidate.mv)
    }));
    DependencyCandidateArms { d1, d2 }
}

/// Count disagreements between the directional semantic reference and the
/// production aggregate classifier for every empty cell and both sides.
pub fn directional_aggregation_mismatches(board: &Board, config: QuietThreatConfig) -> usize {
    [Stone::Black, Stone::White]
        .into_iter()
        .map(|side| {
            (0..NUM_CELLS)
                .filter(|&cell| board.is_empty(cell))
                .filter(|&cell| {
                    aggregate_directional(classify_directions(board, cell, side, config), config)
                        != classify_move_fast_with_flags(
                            board,
                            cell,
                            side,
                            config.enable_jump_three,
                            config.enable_gap_four,
                        )
                })
                .count()
        })
        .sum()
}

/// Return the production-equivalent aggregate kind and its four directional
/// components for one legal empty move. Intended for offline label audits.
pub fn classify_move_with_directions(
    board: &Board,
    mv: Move,
    side: Stone,
    config: QuietThreatConfig,
) -> (ThreatKind, [WindowThreat; 4]) {
    let directions = classify_directions(board, mv, side, config);
    let aggregate = aggregate_directional(directions, config);
    debug_assert_eq!(
        aggregate,
        classify_move_fast_with_flags(
            board,
            mv,
            side,
            config.enable_jump_three,
            config.enable_gap_four,
        )
    );
    (aggregate, directions)
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

fn classify_all_directions(
    board: &Board,
    side: Stone,
    config: QuietThreatConfig,
) -> [[WindowThreat; 4]; NUM_CELLS] {
    let mut directions = [[WindowThreat::None; 4]; NUM_CELLS];
    for (cell, line_kinds) in directions.iter_mut().enumerate() {
        if board.is_empty(cell) {
            *line_kinds = classify_directions(board, cell, side, config);
        }
    }
    directions
}

fn classify_directions(
    board: &Board,
    source: Move,
    side: Stone,
    config: QuietThreatConfig,
) -> [WindowThreat; 4] {
    debug_assert!(board.is_empty(source));
    let (mine, opp) = if side == Stone::Black {
        (&board.black, &board.white)
    } else {
        (&board.white, &board.black)
    };
    let row = (source / BOARD_SIZE) as i32;
    let col = (source % BOARD_SIZE) as i32;
    std::array::from_fn(|dir_idx| {
        let (dr, dc) = DIR[dir_idx];
        let window = read_window(mine, opp, row, col, dr, dc);
        classify_window_after_play_with_flags(
            &window,
            board.effective_rule_set(),
            side,
            config.enable_jump_three,
            config.enable_gap_four,
        )
    })
}

fn aggregate_directional(lines: [WindowThreat; 4], config: QuietThreatConfig) -> ThreatKind {
    let fives = lines
        .iter()
        .filter(|&&kind| kind == WindowThreat::Five)
        .count();
    let open_fours = lines
        .iter()
        .filter(|&&kind| kind == WindowThreat::OpenFour)
        .count();
    let closed_fours = lines
        .iter()
        .filter(|&&kind| kind == WindowThreat::ClosedFour)
        .count();
    let open_threes = lines
        .iter()
        .filter(|&&kind| kind == WindowThreat::OpenThree)
        .count();
    let jump_threes = lines
        .iter()
        .filter(|&&kind| kind == WindowThreat::JumpThree)
        .count();
    let fours = open_fours + closed_fours;

    if fives >= 1 {
        ThreatKind::Five
    } else if open_fours >= 1 {
        ThreatKind::OpenFour
    } else if fours >= 2 {
        ThreatKind::DoubleFour
    } else if closed_fours >= 1 && open_threes >= 1 {
        ThreatKind::FourThree
    } else if open_threes >= 2 {
        ThreatKind::DoubleThree
    } else if closed_fours >= 1 {
        ThreatKind::ClosedFour
    } else if open_threes >= 1 {
        ThreatKind::OpenThree
    } else if config.enable_jump_three && jump_threes >= 1 {
        ThreatKind::JumpThree
    } else {
        ThreatKind::None
    }
}

fn collect_existing_lines(
    board: &Board,
    aggregate: &[ThreatKind; NUM_CELLS],
    directions: &[[WindowThreat; 4]; NUM_CELLS],
    original_mine: &BitBoard,
) -> Vec<ThreatLine> {
    let mut lines = Vec::new();
    for source in 0..NUM_CELLS {
        if !board.is_empty(source) || !aggregate[source].is_forcing() {
            continue;
        }
        for dir_idx in 0..DIR.len() {
            if direction_is_forcing(directions[source][dir_idx]) {
                lines.push(make_threat_line(source, dir_idx, original_mine));
            }
        }
    }
    lines
}

fn make_threat_line(source: Move, dir_idx: usize, original_mine: &BitBoard) -> ThreatLine {
    let row = (source / BOARD_SIZE) as i32;
    let col = (source % BOARD_SIZE) as i32;
    let (dr, dc) = DIR[dir_idx];
    let mut footprint = BitBoard::EMPTY;
    let mut support = BitBoard::EMPTY;
    for offset in -4i32..=4 {
        let r = row + dr * offset;
        let c = col + dc * offset;
        if r < 0 || c < 0 || r >= BOARD_SIZE as i32 || c >= BOARD_SIZE as i32 {
            continue;
        }
        let cell = r as usize * BOARD_SIZE + c as usize;
        footprint.set(cell);
        if original_mine.get(cell) {
            support.set(cell);
        }
    }
    ThreatLine { footprint, support }
}

fn direction_is_forcing(kind: WindowThreat) -> bool {
    matches!(
        kind,
        WindowThreat::OpenThree
            | WindowThreat::JumpThree
            | WindowThreat::ClosedFour
            | WindowThreat::OpenFour
            | WindowThreat::Five
    )
}

fn direction_strength(kind: WindowThreat) -> u8 {
    match kind {
        WindowThreat::OpenThree | WindowThreat::JumpThree => 1,
        WindowThreat::ClosedFour => 2,
        WindowThreat::OpenFour => 3,
        WindowThreat::Five => 4,
        _ => 0,
    }
}

fn bitboards_intersect(a: &BitBoard, b: &BitBoard) -> bool {
    (a.lo & b.lo) != 0 || (a.hi & b.hi) != 0
}

fn intersection_count(a: &BitBoard, b: &BitBoard) -> u32 {
    (a.lo & b.lo).count_ones() + (a.hi & b.hi).count_ones()
}

fn sort_dependency_candidates(candidates: &mut [DependencyQuietCandidate]) {
    candidates.sort_unstable_by(|a, b| {
        b.winning_gains
            .cmp(&a.winning_gains)
            .then_with(|| b.max_reused_support.cmp(&a.max_reused_support))
            .then_with(|| b.dependency_links.cmp(&a.dependency_links))
            .then_with(|| b.forcing_gains.cmp(&a.forcing_gains))
            .then_with(|| a.mv.cmp(&b.mv))
    });
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

    #[test]
    fn directional_aggregation_matches_production_classifier() {
        let mut board = Board::new();
        for mv in [
            to_idx(7, 7),
            to_idx(6, 7),
            to_idx(7, 8),
            to_idx(6, 8),
            to_idx(8, 8),
            to_idx(5, 9),
        ] {
            board.make_move(mv);
        }
        assert_eq!(
            directional_aggregation_mismatches(&board, QuietThreatConfig::default()),
            0
        );
    }

    #[test]
    fn dependency_d2_is_a_subset_of_d1_and_board_is_restored() {
        let mut board = Board::new();
        for mv in [
            to_idx(7, 7),
            to_idx(0, 0),
            to_idx(7, 8),
            to_idx(0, 1),
            to_idx(6, 7),
            to_idx(1, 0),
        ] {
            board.make_move(mv);
        }
        let before_history = board.history.clone();
        let before_black = board.black;
        let before_white = board.white;
        let before_zobrist = board.zobrist;
        let arms = generate_dependency_quiet_candidates(&mut board, QuietThreatConfig::default());

        assert!(arms.d2.iter().all(|candidate| {
            arms.d1
                .iter()
                .any(|d1_candidate| d1_candidate.mv == candidate.mv)
        }));
        assert_eq!(board.history, before_history);
        assert!(board.black == before_black);
        assert!(board.white == before_white);
        assert_eq!(board.zobrist, before_zobrist);
    }
}
