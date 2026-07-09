//! Incremental main-search threat field.
//!
//! RQ587 L1 deliberately keeps this as an unconsumed substrate: the search
//! tree may maintain it, but move ordering, pruning, eval, VCT, and qsearch
//! must not read it until a later preregistered consumer card.

use crate::board::{BitBoard, Board, LINE_PATTERN_FRONTIER_MAX, Move, NUM_CELLS, Stone};
use crate::vct::{THREAT_KIND_COUNT, ThreatKind, classify_move_fast_with_flags};

const SIDES: usize = 2;

#[derive(Clone)]
struct ThreatFieldFrame {
    len: u8,
    cells: [u16; LINE_PATTERN_FRONTIER_MAX],
}

impl ThreatFieldFrame {
    fn from_move(mv: Move) -> Self {
        let mut dirty = [0usize; LINE_PATTERN_FRONTIER_MAX];
        let len = Board::line_pattern_dirty_cells(mv, &mut dirty);
        let mut cells = [0u16; LINE_PATTERN_FRONTIER_MAX];
        for i in 0..len {
            cells[i] = dirty[i] as u16;
        }
        Self {
            len: len as u8,
            cells,
        }
    }

    #[inline]
    fn cells(&self) -> impl Iterator<Item = usize> + '_ {
        self.cells[..self.len as usize]
            .iter()
            .copied()
            .map(usize::from)
    }
}

#[derive(Clone)]
pub struct IncrementalThreatField {
    cell_kinds: Box<[[u8; NUM_CELLS]; SIDES]>,
    tier_sources: [[BitBoard; THREAT_KIND_COUNT]; SIDES],
    frames: Vec<ThreatFieldFrame>,
}

pub fn threat_field_transition_check_for_audit(moves: &[Move]) -> Result<(usize, usize), String> {
    let mut board = Board::new();
    let mut field = IncrementalThreatField::new(&board);
    let mut transitions = 0usize;
    for &mv in moves {
        if !board.is_empty(mv) {
            return Err(format!("occupied move at ply {transitions}: {mv}"));
        }
        board.make_move(mv);
        field.push_move(&board, mv);
        transitions += 1;
        if let Some(mismatch) = field.first_mismatch(&board) {
            return Err(format!(
                "threat-field mismatch after make ply {transitions} move {mv}: {mismatch}"
            ));
        }
    }

    let mut undos = 0usize;
    while board.move_count > 0 {
        let ply_before = board.move_count;
        board.undo_move();
        field.pop_move(&board);
        undos += 1;
        if let Some(mismatch) = field.first_mismatch(&board) {
            return Err(format!(
                "threat-field mismatch after undo from ply {ply_before}: {mismatch}"
            ));
        }
    }
    if field.stack_len() != 0 {
        return Err(format!("threat-field stack leak: {}", field.stack_len()));
    }
    Ok((transitions, undos))
}

impl IncrementalThreatField {
    pub fn new(board: &Board) -> Self {
        let mut field = Self {
            cell_kinds: Box::new([[ThreatKind::None as u8; NUM_CELLS]; SIDES]),
            tier_sources: [[BitBoard::EMPTY; THREAT_KIND_COUNT]; SIDES],
            frames: Vec::with_capacity(NUM_CELLS),
        };
        field.refresh(board);
        field
    }

    pub fn refresh(&mut self, board: &Board) {
        self.frames.clear();
        self.cell_kinds.fill([ThreatKind::None as u8; NUM_CELLS]);
        self.tier_sources = [[BitBoard::EMPTY; THREAT_KIND_COUNT]; SIDES];
        for cell in 0..NUM_CELLS {
            self.recompute_cell(board, cell);
        }
    }

    pub fn push_move(&mut self, board: &Board, mv: Move) {
        let frame = ThreatFieldFrame::from_move(mv);
        for cell in frame.cells() {
            self.recompute_cell(board, cell);
        }
        self.frames.push(frame);
    }

    pub fn pop_move(&mut self, board: &Board) {
        let Some(frame) = self.frames.pop() else {
            debug_assert!(false, "threat-field pop without matching push");
            return;
        };
        for cell in frame.cells() {
            self.recompute_cell(board, cell);
        }
    }

    #[inline]
    pub fn immediate_five(&self, side: Stone) -> BitBoard {
        self.tier_sources[side_idx(side)][ThreatKind::Five as usize]
    }

    #[inline]
    pub fn tier_sources(&self, side: Stone, kind: ThreatKind) -> BitBoard {
        self.tier_sources[side_idx(side)][kind as usize]
    }

    #[inline]
    pub fn cell_kind(&self, side: Stone, cell: Move) -> ThreatKind {
        threat_kind_from_u8(self.cell_kinds[side_idx(side)][cell])
    }

    #[inline]
    pub fn stack_len(&self) -> usize {
        self.frames.len()
    }

    pub fn matches_rebuild(&self, board: &Board) -> bool {
        self.first_mismatch(board).is_none()
    }

    pub fn first_mismatch(&self, board: &Board) -> Option<String> {
        let rebuilt = IncrementalThreatField::new(board);
        for side in [Stone::Black, Stone::White] {
            let s = side_idx(side);
            for cell in 0..NUM_CELLS {
                if self.cell_kinds[s][cell] != rebuilt.cell_kinds[s][cell] {
                    return Some(format!(
                        "cell_kind side={side:?} cell={cell} inc={:?} rebuild={:?}",
                        threat_kind_from_u8(self.cell_kinds[s][cell]),
                        threat_kind_from_u8(rebuilt.cell_kinds[s][cell])
                    ));
                }
            }
            for kind_idx in 0..THREAT_KIND_COUNT {
                if self.tier_sources[s][kind_idx] != rebuilt.tier_sources[s][kind_idx] {
                    return Some(format!(
                        "tier_sources side={side:?} kind={:?}",
                        threat_kind_from_u8(kind_idx as u8)
                    ));
                }
            }
        }
        None
    }

    fn recompute_cell(&mut self, board: &Board, cell: Move) {
        self.clear_cell(cell);
        if !board.is_empty(cell) {
            return;
        }
        for side in [Stone::Black, Stone::White] {
            let s = side_idx(side);
            let kind = classify_move_fast_with_flags(board, cell, side, false, false);
            self.cell_kinds[s][cell] = kind as u8;
            if kind != ThreatKind::None {
                self.tier_sources[s][kind as usize].set(cell);
            }
        }
    }

    fn clear_cell(&mut self, cell: Move) {
        for s in 0..SIDES {
            let kind = self.cell_kinds[s][cell] as usize;
            if kind != ThreatKind::None as usize {
                self.tier_sources[s][kind].clear(cell);
                self.cell_kinds[s][cell] = ThreatKind::None as u8;
            }
        }
    }
}

#[inline]
fn side_idx(side: Stone) -> usize {
    match side {
        Stone::Black => 0,
        Stone::White => 1,
    }
}

fn threat_kind_from_u8(kind: u8) -> ThreatKind {
    match kind as usize {
        x if x == ThreatKind::Five as usize => ThreatKind::Five,
        x if x == ThreatKind::OpenFour as usize => ThreatKind::OpenFour,
        x if x == ThreatKind::ClosedFour as usize => ThreatKind::ClosedFour,
        x if x == ThreatKind::OpenThree as usize => ThreatKind::OpenThree,
        x if x == ThreatKind::DoubleThree as usize => ThreatKind::DoubleThree,
        x if x == ThreatKind::FourThree as usize => ThreatKind::FourThree,
        x if x == ThreatKind::DoubleFour as usize => ThreatKind::DoubleFour,
        x if x == ThreatKind::JumpThree as usize => ThreatKind::JumpThree,
        _ => ThreatKind::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GameResult;
    use crate::board::to_idx;

    #[test]
    fn threat_field_query_api_smoke() {
        let mut board = Board::new();
        board.make_move(to_idx(7, 3));
        board.make_move(to_idx(0, 0));
        board.make_move(to_idx(7, 4));
        board.make_move(to_idx(0, 1));
        board.make_move(to_idx(7, 5));
        board.make_move(to_idx(0, 2));
        board.make_move(to_idx(7, 6));

        let field = IncrementalThreatField::new(&board);
        let win = to_idx(7, 7);
        assert_eq!(field.cell_kind(Stone::Black, win), ThreatKind::Five);
        assert!(field.immediate_five(Stone::Black).get(win));
        assert!(field.tier_sources(Stone::Black, ThreatKind::Five).get(win));
        assert_eq!(field.cell_kind(Stone::White, win), ThreatKind::None);
    }

    #[test]
    fn threat_field_incremental_matches_rebuild_smoke() {
        let moves = [
            112, 113, 97, 98, 127, 128, 111, 114, 96, 99, 126, 129, 82, 83, 84, 85,
        ];
        let mut board = Board::new();
        let mut field = IncrementalThreatField::new(&board);
        assert!(field.matches_rebuild(&board));

        for &mv in &moves {
            board.make_move(mv);
            field.push_move(&board, mv);
            assert!(
                field.matches_rebuild(&board),
                "mismatch after push move {mv}: {:?}",
                field.first_mismatch(&board)
            );
        }

        while let Some(&mv) = board.history.last() {
            board.undo_move();
            field.pop_move(&board);
            assert!(
                field.matches_rebuild(&board),
                "mismatch after pop move {mv}: {:?}",
                field.first_mismatch(&board)
            );
        }
        assert_eq!(field.stack_len(), 0);
    }

    #[test]
    fn threat_field_audit_transition_helper_smoke() {
        let moves = [
            112, 113, 97, 98, 127, 128, 111, 114, 96, 99, 126, 129, 82, 83, 84, 85,
        ];
        let (transitions, undos) = threat_field_transition_check_for_audit(&moves).unwrap();
        assert_eq!(transitions, moves.len());
        assert_eq!(undos, moves.len());
    }

    #[test]
    #[ignore = "RQ587 G1 gate: run explicitly in release mode"]
    fn threat_field_100k_transition_gate() {
        let mut rng = TestRng::new(0x5870_0001);
        let mut board = Board::new();
        let mut field = IncrementalThreatField::new(&board);
        let mut transitions = 0usize;
        let mut mismatch = 0usize;
        let mut undo_fail = 0usize;

        while transitions < 100_000 {
            if board.move_count >= 160
                || board.game_result() != GameResult::Ongoing
                || board.candidate_moves().is_empty()
            {
                while board.move_count > 0 {
                    board.undo_move();
                    field.pop_move(&board);
                    if !field.matches_rebuild(&board) {
                        undo_fail += 1;
                    }
                }
                board = Board::new();
                field.refresh(&board);
            }

            let moves = board.candidate_moves();
            let mv = moves[rng.usize(moves.len())];
            board.make_move(mv);
            field.push_move(&board, mv);
            transitions += 1;

            if !field.matches_rebuild(&board) {
                mismatch += 1;
            }
        }

        while board.move_count > 0 {
            board.undo_move();
            field.pop_move(&board);
            if !field.matches_rebuild(&board) {
                undo_fail += 1;
            }
        }

        eprintln!("RQ587 transitions={transitions} mismatch={mismatch} undo_fail={undo_fail}");
        assert_eq!(mismatch, 0, "incremental-vs-rebuild mismatch count");
        assert_eq!(undo_fail, 0, "undo roundtrip mismatch count");
        assert_eq!(field.stack_len(), 0, "threat-field stack leak");
    }

    struct TestRng(u64);

    impl TestRng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn usize(&mut self, n: usize) -> usize {
            (self.next_u64() as usize) % n
        }
    }
}
