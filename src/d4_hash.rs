//! Incremental hashes for all eight symmetries of a 15x15 Gomoku board.
//!
//! The 64-bit canonical key is a fingerprint, not an exact proof identity.
//! Callers that require exact identity must also compare the canonical
//! serialization returned by [`exact_canonical_state`].

use crate::board::{
    BOARD_SIZE, BitBoard, Board, Move, NUM_CELLS, RuleSet, Stone, ZOBRIST_SIDE, d4_rule_key,
    zobrist_stone_key,
};

const LAST_COORD: usize = BOARD_SIZE - 1;

const fn transform_cell(transform: usize, cell: usize) -> u8 {
    let row = cell / BOARD_SIZE;
    let col = cell % BOARD_SIZE;
    let (mapped_row, mapped_col) = match transform {
        0 => (row, col),
        1 => (col, LAST_COORD - row),
        2 => (LAST_COORD - row, LAST_COORD - col),
        3 => (LAST_COORD - col, row),
        4 => (row, LAST_COORD - col),
        5 => (LAST_COORD - row, col),
        6 => (col, row),
        7 => (LAST_COORD - col, LAST_COORD - row),
        _ => panic!("D4 transform index must be in 0..8"),
    };
    (mapped_row * BOARD_SIZE + mapped_col) as u8
}

const fn build_d4_map() -> [[u8; NUM_CELLS]; 8] {
    let mut map = [[0u8; NUM_CELLS]; 8];
    let mut transform = 0;
    while transform < 8 {
        let mut cell = 0;
        while cell < NUM_CELLS {
            map[transform][cell] = transform_cell(transform, cell);
            cell += 1;
        }
        transform += 1;
    }
    map
}

/// `D4_MAP[t][cell]` is the row-major cell reached by frozen transform `t`.
pub const D4_MAP: [[u8; 225]; 8] = build_d4_map();

/// Inverse transform for each entry in [`D4_MAP`].
pub const D4_INVERSE: [u8; 8] = [0, 3, 2, 1, 4, 5, 6, 7];

/// D4 composition table, with `D4_COMPOSE[a][b] = T_a(T_b(cell))`.
pub const D4_COMPOSE: [[u8; 8]; 8] = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 0, 7, 6, 4, 5],
    [2, 3, 0, 1, 5, 4, 7, 6],
    [3, 0, 1, 2, 6, 7, 5, 4],
    [4, 6, 5, 7, 0, 2, 1, 3],
    [5, 7, 4, 6, 2, 0, 3, 1],
    [6, 5, 7, 4, 3, 1, 0, 2],
    [7, 4, 6, 5, 1, 3, 2, 0],
];

/// Minimum D4 fingerprint and the transform that maps into that orientation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CanonicalContext {
    pub key: u64,
    pub to_canonical: u8,
}

/// Exact, lexicographically minimal D4 serialization of a board state.
///
/// Bytes `0..16`, `16..32`, `32..48`, and `48..64` are respectively the
/// big-endian Black-low, Black-high, White-low, and White-high bitboard
/// limbs. Byte 64 is side to move (`0=Black`, `1=White`) and byte 65 is the
/// effective rule (`0=Freestyle`, `1=Standard`, `2=Caro`, `3=Renju`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactCanonicalState {
    pub bytes: [u8; 66],
    pub to_canonical: u8,
}

impl ExactCanonicalState {
    /// Borrow the frozen 66-byte exact-identity payload.
    #[inline]
    pub const fn serialization(&self) -> &[u8; 66] {
        &self.bytes
    }
}

/// Incrementally maintained fingerprints for all eight D4 orientations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct D4HashState {
    hashes: [u64; 8],
}

impl D4HashState {
    /// Rebuild all eight fingerprints from the board's semantic state.
    pub fn rebuild(board: &Board) -> Self {
        let domain = d4_rule_key(board.effective_rule_set())
            ^ if board.side_to_move == Stone::White {
                ZOBRIST_SIDE
            } else {
                0
            };
        let mut hashes = [domain; 8];

        for cell in board.black.iter_ones() {
            for transform in 0..8 {
                hashes[transform] ^=
                    zobrist_stone_key(Stone::Black, D4_MAP[transform][cell] as usize);
            }
        }
        for cell in board.white.iter_ones() {
            for transform in 0..8 {
                hashes[transform] ^=
                    zobrist_stone_key(Stone::White, D4_MAP[transform][cell] as usize);
            }
        }

        Self { hashes }
    }

    /// Borrow the eight orientation fingerprints in frozen transform order.
    #[inline]
    pub const fn hashes(&self) -> &[u64; 8] {
        &self.hashes
    }

    /// Return the minimum fingerprint, choosing the lower transform on ties.
    #[inline]
    pub fn canonical_context(&self) -> CanonicalContext {
        canonical_context_from_hashes(&self.hashes)
    }

    /// XOR one placed/removed stone and toggle side to move.
    ///
    /// XOR is involutive, so the same operation maintains both make and undo.
    #[inline]
    pub fn apply_move(&mut self, placed: Stone, mv: Move) {
        debug_assert!(mv < NUM_CELLS);
        for transform in 0..8 {
            self.hashes[transform] ^=
                zobrist_stone_key(placed, D4_MAP[transform][mv] as usize) ^ ZOBRIST_SIDE;
        }
    }

    /// Alias emphasizing that [`Self::apply_move`] is an XOR toggle.
    #[inline]
    pub fn toggle_move(&mut self, placed: Stone, mv: Move) {
        self.apply_move(placed, mv);
    }

    /// Predict all child hashes without mutating this state or the board.
    pub fn predicted_child_hashes(&self, board: &Board, mv: Move) -> Option<[u64; 8]> {
        if mv >= NUM_CELLS || !board.is_empty(mv) {
            return None;
        }
        let mut child = *self;
        child.apply_move(board.side_to_move, mv);
        Some(child.hashes)
    }

    /// Predict the child's canonical fingerprint without making the move.
    pub fn predicted_child_context(&self, board: &Board, mv: Move) -> Option<CanonicalContext> {
        self.predicted_child_hashes(board, mv)
            .map(|hashes| canonical_context_from_hashes(&hashes))
    }

    /// Build an exact serialization in one requested D4 orientation.
    #[inline]
    pub fn exact_transformed_serialization(board: &Board, transform: u8) -> Option<[u8; 66]> {
        exact_transformed_serialization(board, transform)
    }

    /// Build the exact lexicographically minimal D4 serialization.
    #[inline]
    pub fn exact_canonical_state(board: &Board) -> ExactCanonicalState {
        exact_canonical_state(board)
    }
}

/// Apply the production canonical tie rule to an explicit audit hash array.
#[doc(hidden)]
#[inline]
pub fn canonical_context_from_hashes(hashes: &[u64; 8]) -> CanonicalContext {
    let mut key = hashes[0];
    let mut to_canonical = 0u8;
    for transform in 1..8 {
        if hashes[transform] < key {
            key = hashes[transform];
            to_canonical = transform as u8;
        }
    }
    CanonicalContext { key, to_canonical }
}

#[inline]
fn transformed_bitboard(source: &BitBoard, transform: usize) -> BitBoard {
    let mut transformed = BitBoard::EMPTY;
    for cell in source.iter_ones() {
        transformed.set(D4_MAP[transform][cell] as usize);
    }
    transformed
}

#[inline]
const fn side_tag(side: Stone) -> u8 {
    match side {
        Stone::Black => 0,
        Stone::White => 1,
    }
}

#[inline]
const fn rule_tag(rule: RuleSet) -> u8 {
    match rule {
        RuleSet::Freestyle => 0,
        RuleSet::Standard => 1,
        RuleSet::Caro => 2,
        RuleSet::Renju => 3,
    }
}

/// Return the exact 66-byte semantic-state serialization under one transform.
///
/// Returns `None` when `transform` is outside the frozen `0..8` domain.
pub fn exact_transformed_serialization(board: &Board, transform: u8) -> Option<[u8; 66]> {
    let transform = transform as usize;
    if transform >= 8 {
        return None;
    }

    let black = transformed_bitboard(&board.black, transform);
    let white = transformed_bitboard(&board.white, transform);
    let mut bytes = [0u8; 66];
    bytes[0..16].copy_from_slice(&black.lo.to_be_bytes());
    bytes[16..32].copy_from_slice(&black.hi.to_be_bytes());
    bytes[32..48].copy_from_slice(&white.lo.to_be_bytes());
    bytes[48..64].copy_from_slice(&white.hi.to_be_bytes());
    bytes[64] = side_tag(board.side_to_move);
    bytes[65] = rule_tag(board.effective_rule_set());
    Some(bytes)
}

/// Return the exact lexicographic D4 representative, independently of hash
/// minima. Ties retain the lower transform index.
pub fn exact_canonical_state(board: &Board) -> ExactCanonicalState {
    let mut bytes = exact_transformed_serialization(board, 0).expect("identity transform is valid");
    let mut to_canonical = 0u8;
    for transform in 1..8 {
        let candidate = exact_transformed_serialization(board, transform)
            .expect("loop contains only valid transforms");
        if candidate < bytes {
            bytes = candidate;
            to_canonical = transform;
        }
    }
    ExactCanonicalState {
        bytes,
        to_canonical,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn transformed_board(board: &Board, transform: usize) -> Board {
        let mut transformed = Board::new();
        transformed.set_rule_set(board.effective_rule_set());
        for &mv in &board.history {
            transformed.make_move(D4_MAP[transform][mv] as usize);
        }
        transformed
    }

    fn sample_board() -> Board {
        let mut board = Board::new();
        board.set_rule_set(RuleSet::Caro);
        for mv in [0, 17, 33, 71, 112, 149, 207] {
            board.make_move(mv);
        }
        board
    }

    #[test]
    fn maps_are_bijections_with_frozen_inverses() {
        for transform in 0..8 {
            let mut seen = [false; NUM_CELLS];
            for cell in 0..NUM_CELLS {
                let mapped = D4_MAP[transform][cell] as usize;
                assert!(mapped < NUM_CELLS);
                assert!(!seen[mapped], "duplicate t={transform}, cell={cell}");
                seen[mapped] = true;
                assert_eq!(
                    D4_MAP[D4_INVERSE[transform] as usize][mapped] as usize,
                    cell
                );
            }
            assert!(seen.into_iter().all(|present| present));
        }
    }

    #[test]
    fn maps_follow_the_frozen_coordinate_convention() {
        let cell = 2 * BOARD_SIZE + 5;
        let expected = [
            2 * BOARD_SIZE + 5,
            5 * BOARD_SIZE + 12,
            12 * BOARD_SIZE + 9,
            9 * BOARD_SIZE + 2,
            2 * BOARD_SIZE + 9,
            12 * BOARD_SIZE + 5,
            5 * BOARD_SIZE + 2,
            9 * BOARD_SIZE + 12,
        ];
        for transform in 0..8 {
            assert_eq!(D4_MAP[transform][cell] as usize, expected[transform]);
        }
    }

    #[test]
    fn composition_table_matches_cell_maps() {
        for a in 0..8 {
            for b in 0..8 {
                let composed = D4_COMPOSE[a][b] as usize;
                for cell in 0..NUM_CELLS {
                    let via_b = D4_MAP[b][cell] as usize;
                    assert_eq!(
                        D4_MAP[a][via_b], D4_MAP[composed][cell],
                        "a={a}, b={b}, cell={cell}"
                    );
                }
            }
        }
    }

    #[test]
    fn empty_hash_tie_chooses_identity_and_rule_domains_are_distinct() {
        let mut keys = HashSet::new();
        for rule in [
            RuleSet::Freestyle,
            RuleSet::Standard,
            RuleSet::Caro,
            RuleSet::Renju,
        ] {
            let mut board = Board::new();
            board.set_rule_set(rule);
            let state = D4HashState::rebuild(&board);
            assert!(state.hashes().iter().all(|&hash| hash == d4_rule_key(rule)));
            assert_eq!(
                state.canonical_context(),
                CanonicalContext {
                    key: d4_rule_key(rule),
                    to_canonical: 0
                }
            );
            assert!(keys.insert(d4_rule_key(rule)));
        }
        assert_eq!(keys.len(), 4);
    }

    #[test]
    fn synthetic_equal_minimum_hash_chooses_lower_transform() {
        assert_eq!(
            canonical_context_from_hashes(&[9, 3, 3, 7, 8, 5, 6, 4]),
            CanonicalContext {
                key: 3,
                to_canonical: 1,
            }
        );
    }

    #[test]
    fn rebuild_and_transform_relationship_are_exact() {
        let board = sample_board();
        let original = D4HashState::rebuild(&board);
        let canonical = original.canonical_context();
        let exact = exact_canonical_state(&board);

        for outer in 0..8 {
            let transformed = transformed_board(&board, outer);
            let rebuilt = D4HashState::rebuild(&transformed);
            for inner in 0..8 {
                assert_eq!(
                    rebuilt.hashes()[inner],
                    original.hashes()[D4_COMPOSE[inner][outer] as usize]
                );
            }
            assert_eq!(rebuilt.canonical_context().key, canonical.key);
            assert_eq!(exact_canonical_state(&transformed).bytes, exact.bytes);
        }
    }

    #[test]
    fn prediction_and_xor_updates_match_full_rebuild() {
        let mut board = sample_board();
        let mut state = D4HashState::rebuild(&board);
        let mv = 93;
        let placed = board.side_to_move;
        let predicted_hashes = state.predicted_child_hashes(&board, mv).unwrap();
        let predicted_context = state.predicted_child_context(&board, mv).unwrap();

        state.apply_move(placed, mv);
        board.make_move(mv);
        let rebuilt = D4HashState::rebuild(&board);
        assert_eq!(state, rebuilt);
        assert_eq!(predicted_hashes, *rebuilt.hashes());
        assert_eq!(predicted_context, rebuilt.canonical_context());

        board.undo_move();
        state.toggle_move(placed, mv);
        assert_eq!(state, D4HashState::rebuild(&board));
        assert!(state.predicted_child_hashes(&board, NUM_CELLS).is_none());
        assert!(
            state
                .predicted_child_context(&board, board.history[0])
                .is_none()
        );
    }

    #[test]
    fn exact_serialization_layout_and_ties_are_frozen() {
        let mut empty = Board::new();
        empty.exact5 = true;
        let exact_empty = exact_canonical_state(&empty);
        assert_eq!(exact_empty.to_canonical, 0);
        assert_eq!(&exact_empty.bytes[..64], &[0u8; 64]);
        assert_eq!(exact_empty.bytes[64], 0);
        assert_eq!(exact_empty.bytes[65], 1);

        let board = sample_board();
        let exact = exact_canonical_state(&board);
        let all = (0..8)
            .map(|transform| exact_transformed_serialization(&board, transform).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(exact.bytes, *all.iter().min().unwrap());
        assert_eq!(
            exact.to_canonical as usize,
            all.iter().position(|bytes| *bytes == exact.bytes).unwrap()
        );
        assert!(exact_transformed_serialization(&board, 8).is_none());
    }
}
