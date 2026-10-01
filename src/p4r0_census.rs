//! P4-R0 census: how much of the search tree is *forced*, and how much of that
//! the main search currently pays for anyway.
//!
//! This module only counts. It never changes a move list, a bound or a score,
//! so an instrumented build must produce byte-identical search results.
//!
//! The question it answers: at interior nodes where the opponent already
//! threatens an immediate five, the set of non-losing replies is fixed by the
//! rules — play our own five if we have one, otherwise block the single
//! completing cell, and if there are two distinct completing cells the node is
//! already lost. Restricting to that set is a theorem, not a heuristic. The
//! census measures how often such nodes occur and how many moves the search
//! actually visits there, which bounds what a sound restriction could win.

use crate::board::{Board, Move, NUM_CELLS, Stone, to_rc};

/// Cells where `side` playing immediately completes a winning line.
///
/// Counted directly from the bitboards rather than through the mini pattern
/// table: that table buckets anything outside its top-K vocabulary into `RARE`
/// and reports `WindowThreat::None` for it, which is exactly the dense,
/// many-stone shape a five-completion has. Undercounting here would bias the
/// census toward "no forced nodes", so the census pays the scan instead.
pub fn five_cells(board: &Board, side: Stone, out: &mut Vec<Move>) {
    out.clear();
    let stones = match side {
        Stone::Black => &board.black,
        Stone::White => &board.white,
    };
    let rules = board.effective_rule_set();
    const DIRECTIONS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
    for cell in 0..NUM_CELLS {
        if !board.is_empty(cell) {
            continue;
        }
        let (row, col) = to_rc(cell);
        for &(dr, dc) in &DIRECTIONS {
            // `line_run_with_edges` starts the count at the anchor itself and walks
            // outward, so an empty anchor yields the count as if `side` had
            // just played there.
            let (count, open_ends, edge_ends) =
                board.line_run_with_edges(stones, row as i32, col as i32, dr, dc);
            if rules.line_wins_with_edges(side, count, open_ends, edge_ends) {
                out.push(cell);
                break;
            }
        }
    }
}

/// How a node is bounded, mirroring the search's own TT classification.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NodeBound {
    Cut,
    All,
    Exact,
}

/// Number of moves a sound forced-reply restriction would be allowed to search.
/// `None` means the node is not forced (no restriction is sound).
pub fn sound_restricted_len(my_fives: usize, their_fives: usize) -> Option<usize> {
    if their_fives == 0 {
        return None;
    }
    if my_fives > 0 {
        return Some(1); // we complete five first and win outright
    }
    if their_fives == 1 {
        return Some(1); // exactly one completing cell: block it or lose
    }
    Some(0) // two or more distinct completing cells: unavoidable loss
}

/// The rules-determined reply at a node where the opponent threatens five.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SoundReply {
    /// No immediate-five threat: no restriction is sound here.
    NotForced,
    /// We complete five first and win outright.
    WinNow(Move),
    /// Exactly one completing cell: occupy it or lose.
    OnlyBlock(Move),
    /// Two or more distinct completing cells and no five of our own: lost.
    Lost,
}

/// Whether occupying the completing cell is the *only* way to defuse a five.
///
/// Freestyle (`count >= 5`) and Standard (`count == 5`) qualify: our stone
/// never joins the opponent's line, so nothing we play except the completing
/// cell changes the outcome. Caro does not — it requires an open end, so
/// filling that end defuses the threat without touching the completing cell.
/// Renju adds forbidden-move interactions. Both are excluded rather than
/// reasoned about.
pub fn rule_admits_restriction(rules: crate::board::RuleSet) -> bool {
    matches!(
        rules,
        crate::board::RuleSet::Freestyle | crate::board::RuleSet::Standard
    )
}

/// Classify the node. `scratch` is reused to avoid allocating per node.
pub fn sound_reply(board: &Board, scratch: &mut Vec<Move>) -> SoundReply {
    if !rule_admits_restriction(board.effective_rule_set()) {
        return SoundReply::NotForced;
    }
    let us = board.side_to_move;
    five_cells(board, us.opponent(), scratch);
    match scratch.len() {
        0 => SoundReply::NotForced,
        1 => {
            let block = scratch[0];
            five_cells(board, us, scratch);
            match scratch.first() {
                Some(&win) => SoundReply::WinNow(win),
                None => SoundReply::OnlyBlock(block),
            }
        }
        _ => {
            five_cells(board, us, scratch);
            match scratch.first() {
                Some(&win) => SoundReply::WinNow(win),
                None => SoundReply::Lost,
            }
        }
    }
}

#[derive(Default, Clone)]
pub struct ForcedCensus {
    pub enabled: bool,
    pub interior_nodes: u64,

    pub forced_nodes: u64,
    pub forced_moves_searched: u64,
    pub forced_sound_moves: u64,
    pub forced_own_five_nodes: u64,
    pub forced_lost_nodes: u64,
    pub forced_cut: u64,
    pub forced_all: u64,
    pub forced_exact: u64,
    /// Moves searched at forced nodes, bucketed 1,2,3,4,5-8,9-16,17+.
    pub forced_searched_hist: [u64; 7],

    pub quiet_nodes: u64,
    pub quiet_moves_searched: u64,
    pub quiet_cut: u64,
    pub quiet_all: u64,
    pub quiet_exact: u64,
}

fn bucket(n: usize) -> usize {
    match n {
        0 | 1 => 0,
        2 => 1,
        3 => 2,
        4 => 3,
        5..=8 => 4,
        9..=16 => 5,
        _ => 6,
    }
}

impl ForcedCensus {
    /// Classify a node before its move loop. Returns the sound restricted size
    /// when the node is forced, for the caller to pass back to `record`.
    pub fn observe_node(&mut self, board: &Board, scratch: &mut Vec<Move>) -> Option<usize> {
        self.interior_nodes += 1;
        let us = board.side_to_move;
        let them = us.opponent();

        five_cells(board, them, scratch);
        let their_fives = scratch.len();
        if their_fives == 0 {
            return None;
        }
        five_cells(board, us, scratch);
        let my_fives = scratch.len();

        let sound = sound_restricted_len(my_fives, their_fives).unwrap_or(0);
        self.forced_nodes += 1;
        self.forced_sound_moves += sound as u64;
        if my_fives > 0 {
            self.forced_own_five_nodes += 1;
        } else if their_fives >= 2 {
            self.forced_lost_nodes += 1;
        }
        Some(sound)
    }

    /// Record the node's outcome after its move loop.
    pub fn record(&mut self, forced: Option<usize>, searched: usize, bound: NodeBound) {
        if forced.is_some() {
            self.forced_moves_searched += searched as u64;
            self.forced_searched_hist[bucket(searched)] += 1;
            match bound {
                NodeBound::Cut => self.forced_cut += 1,
                NodeBound::All => self.forced_all += 1,
                NodeBound::Exact => self.forced_exact += 1,
            }
        } else {
            self.quiet_nodes += 1;
            self.quiet_moves_searched += searched as u64;
            match bound {
                NodeBound::Cut => self.quiet_cut += 1,
                NodeBound::All => self.quiet_all += 1,
                NodeBound::Exact => self.quiet_exact += 1,
            }
        }
    }

    /// Accumulate another node's worth of counters (one Searcher per position).
    pub fn merge(&mut self, o: &ForcedCensus) {
        self.interior_nodes += o.interior_nodes;
        self.forced_nodes += o.forced_nodes;
        self.forced_moves_searched += o.forced_moves_searched;
        self.forced_sound_moves += o.forced_sound_moves;
        self.forced_own_five_nodes += o.forced_own_five_nodes;
        self.forced_lost_nodes += o.forced_lost_nodes;
        self.forced_cut += o.forced_cut;
        self.forced_all += o.forced_all;
        self.forced_exact += o.forced_exact;
        for i in 0..self.forced_searched_hist.len() {
            self.forced_searched_hist[i] += o.forced_searched_hist[i];
        }
        self.quiet_nodes += o.quiet_nodes;
        self.quiet_moves_searched += o.quiet_moves_searched;
        self.quiet_cut += o.quiet_cut;
        self.quiet_all += o.quiet_all;
        self.quiet_exact += o.quiet_exact;
    }

    pub fn report(&self) -> String {
        let pct = |a: u64, b: u64| if b == 0 { 0.0 } else { 100.0 * a as f64 / b as f64 };
        let avg = |a: u64, b: u64| if b == 0 { 0.0 } else { a as f64 / b as f64 };
        let mut s = String::new();
        s.push_str(&format!("interior_nodes        {}\n", self.interior_nodes));
        s.push_str(&format!(
            "forced_nodes          {} ({:.2}% of interior)\n",
            self.forced_nodes,
            pct(self.forced_nodes, self.interior_nodes)
        ));
        s.push_str(&format!(
            "  own_five (win now)  {}\n  provably_lost       {}\n",
            self.forced_own_five_nodes, self.forced_lost_nodes
        ));
        s.push_str(&format!(
            "  moves searched      {} (avg {:.2}/node)\n",
            self.forced_moves_searched,
            avg(self.forced_moves_searched, self.forced_nodes)
        ));
        s.push_str(&format!(
            "  sound-restricted    {} (avg {:.2}/node)\n",
            self.forced_sound_moves,
            avg(self.forced_sound_moves, self.forced_nodes)
        ));
        s.push_str(&format!(
            "  wasted moves        {}\n",
            self.forced_moves_searched
                .saturating_sub(self.forced_sound_moves)
        ));
        s.push_str(&format!(
            "  bound cut/all/exact {}/{}/{}\n",
            self.forced_cut, self.forced_all, self.forced_exact
        ));
        s.push_str(&format!(
            "  searched hist 1,2,3,4,5-8,9-16,17+  {:?}\n",
            self.forced_searched_hist
        ));
        s.push_str(&format!(
            "quiet_nodes           {} (avg {:.2} moves/node)\n",
            self.quiet_nodes,
            avg(self.quiet_moves_searched, self.quiet_nodes)
        ));
        s.push_str(&format!(
            "  bound cut/all/exact {}/{}/{}\n",
            self.quiet_cut, self.quiet_all, self.quiet_exact
        ));
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    #[test]
    fn sound_set_semantics() {
        assert_eq!(sound_restricted_len(0, 0), None);
        assert_eq!(sound_restricted_len(0, 1), Some(1));
        assert_eq!(sound_restricted_len(0, 2), Some(0));
        assert_eq!(sound_restricted_len(3, 2), Some(1));
    }

    /// Four black stones in a row leave exactly two completing cells for Black.
    #[test]
    fn open_four_has_two_five_cells() {
        let mut board = Board::new();
        // Black 7,7 / 7,8 / 7,9 / 7,10 with White answering far away.
        let blacks = [7 * 15 + 7, 7 * 15 + 8, 7 * 15 + 9, 7 * 15 + 10];
        let whites = [0, 1, 2];
        for i in 0..4 {
            board.make_move(blacks[i]);
            if i < 3 {
                board.make_move(whites[i]);
            }
        }
        let mut cells = Vec::new();
        five_cells(&board, Stone::Black, &mut cells);
        assert_eq!(cells.len(), 2, "open four completes at both ends: {cells:?}");
        assert!(cells.contains(&(7 * 15 + 6)));
        assert!(cells.contains(&(7 * 15 + 11)));

        five_cells(&board, Stone::White, &mut cells);
        assert!(cells.is_empty(), "White has no five here: {cells:?}");
    }

    #[test]
    fn empty_board_has_no_five_cells() {
        let board = Board::new();
        let mut cells = Vec::new();
        for side in [Stone::Black, Stone::White] {
            five_cells(&board, side, &mut cells);
            assert!(cells.is_empty());
        }
    }

    /// Cross-check against the engine's own `check_win` by actually playing
    /// each empty cell. If these ever disagree the whole census is worthless,
    /// so this covers many random positions rather than a hand-picked few.
    #[test]
    fn five_cells_agrees_with_check_win() {
        let mut state = 0x2026_09_19_u64;
        let mut rng = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut checked_positions = 0usize;
        let mut positions_with_a_five = 0usize;

        for _game in 0..60 {
            let mut board = Board::new();
            // Cluster the stones so real five-threats actually appear.
            for _ply in 0..40 {
                let side = board.side_to_move;
                let mut expected: Vec<Move> = Vec::new();
                for cell in 0..NUM_CELLS {
                    if !board.is_empty(cell) {
                        continue;
                    }
                    board.make_move(cell);
                    let won = board.check_win(cell);
                    board.undo_move();
                    if won {
                        expected.push(cell);
                    }
                }
                let mut actual = Vec::new();
                five_cells(&board, side, &mut actual);
                assert_eq!(
                    actual, expected,
                    "five_cells disagreed with check_win for {side:?} at ply {}",
                    board.move_count
                );
                checked_positions += 1;
                if !expected.is_empty() {
                    positions_with_a_five += 1;
                }

                // Play near the last stone to keep the position tactical.
                let anchor = board.history.last().copied().unwrap_or(7 * 15 + 7);
                let (ar, ac) = to_rc(anchor);
                let mut played = None;
                for _try in 0..40 {
                    let dr = (rng() % 5) as i32 - 2;
                    let dc = (rng() % 5) as i32 - 2;
                    let r = ar as i32 + dr;
                    let c = ac as i32 + dc;
                    if !(0..15).contains(&r) || !(0..15).contains(&c) {
                        continue;
                    }
                    let cell = (r as usize) * 15 + c as usize;
                    if board.is_empty(cell) {
                        played = Some(cell);
                        break;
                    }
                }
                let Some(cell) = played else { break };
                board.make_move(cell);
                if board.check_win(cell) {
                    break;
                }
            }
        }

        assert!(checked_positions > 500, "too few positions: {checked_positions}");
        assert!(
            positions_with_a_five > 50,
            "sample never produced five-threats ({positions_with_a_five}); the test would be vacuous"
        );
    }

    /// The restriction is only worth having if it is a theorem. This checks the
    /// theorem itself by brute force rather than checking the search that uses
    /// it: whatever `sound_reply` discards must genuinely lose.
    ///
    /// * `OnlyBlock(b)` — every legal move other than `b` must leave the
    ///   opponent a five to play.
    /// * `Lost` — *every* legal move must leave the opponent a five to play.
    /// * `WinNow(w)` — playing `w` must actually win on the spot.
    #[test]
    fn discarded_moves_really_lose() {
        let mut state = 0xC0FFEE_2026_u64;
        let mut rng = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut seen = [0usize; 4]; // NotForced, WinNow, OnlyBlock, Lost
        let mut scratch = Vec::new();
        let mut probe = Vec::new();

        for _game in 0..200 {
            let mut board = Board::new();
            for _ply in 0..45 {
                let verdict = sound_reply(&board, &mut scratch);
                let them = board.side_to_move.opponent();
                match verdict {
                    SoundReply::NotForced => seen[0] += 1,
                    SoundReply::WinNow(w) => {
                        seen[1] += 1;
                        board.make_move(w);
                        let won = board.check_win(w);
                        board.undo_move();
                        assert!(won, "WinNow({w}) did not actually win");
                    }
                    SoundReply::OnlyBlock(b) => {
                        seen[2] += 1;
                        for cell in 0..NUM_CELLS {
                            if cell == b || !board.is_empty(cell) {
                                continue;
                            }
                            board.make_move(cell);
                            five_cells(&board, them, &mut probe);
                            let opponent_still_wins = !probe.is_empty();
                            board.undo_move();
                            assert!(
                                opponent_still_wins,
                                "discarding {cell} was unsound: it defuses the five that \
                                 sound_reply said only {b} could block"
                            );
                        }
                    }
                    SoundReply::Lost => {
                        seen[3] += 1;
                        for cell in 0..NUM_CELLS {
                            if !board.is_empty(cell) {
                                continue;
                            }
                            board.make_move(cell);
                            let saved = board.check_win(cell);
                            five_cells(&board, them, &mut probe);
                            let opponent_still_wins = !probe.is_empty();
                            board.undo_move();
                            assert!(
                                opponent_still_wins && !saved,
                                "node reported Lost but {cell} survives or wins"
                            );
                        }
                    }
                }

                // Keep play local so five-threats actually arise.
                let anchor = board.history.last().copied().unwrap_or(7 * 15 + 7);
                let (ar, ac) = to_rc(anchor);
                let mut played = None;
                for _try in 0..40 {
                    let r = ar as i32 + (rng() % 5) as i32 - 2;
                    let c = ac as i32 + (rng() % 5) as i32 - 2;
                    if !(0..15).contains(&r) || !(0..15).contains(&c) {
                        continue;
                    }
                    let cell = (r as usize) * 15 + c as usize;
                    if board.is_empty(cell) {
                        played = Some(cell);
                        break;
                    }
                }
                let Some(cell) = played else { break };
                board.make_move(cell);
                if board.check_win(cell) {
                    break;
                }
            }
        }

        assert!(
            seen[2] > 30 && seen[3] > 10,
            "sample must actually exercise OnlyBlock and Lost, saw {seen:?}"
        );
    }
}
