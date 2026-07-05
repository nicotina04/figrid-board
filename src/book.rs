//! Opening book for Gomocup 2026 official opening sets.
//!
//! Each entry pairs a Gomocup-published opening sequence with the best
//! reply chosen by the strongest engine we could query offline (Rapfi at
//! 30 s/move). At runtime we hash each entry's position via `Board::zobrist`
//! once on first lookup, then serve subsequent hits as a constant-time
//! shortcut from `Searcher::choose_move`.
//!
//! Coverage is limited to the post-opening move where figrid happens to be
//! the side to move; when figrid plays the other colour the opening hash
//! will not match and the search runs as usual.

use std::sync::OnceLock;

use crate::board::{Board, to_idx};

/// Raw book entries. Coordinates are `(x, y)` = `(column, row)` matching
/// pbrain protocol and `idx_to_xy`. The replay logic in `build_book` converts
/// them to internal `to_idx(row, col)` indices.
type Opening = &'static [(u8, u8)];
const BOOK_RAW: &[(&'static str, Opening, (u8, u8))] = &[
    (
        "freestyle15#0",
        &[(13, 6), (6, 5), (11, 2), (12, 3), (8, 5)],
        (9, 4),
    ),
    (
        "freestyle15#1",
        &[(4, 11), (8, 12), (1, 13), (4, 10), (5, 8), (5, 9), (6, 10)],
        (6, 8),
    ),
    (
        "freestyle15#2",
        &[
            (10, 11),
            (8, 12),
            (5, 12),
            (4, 11),
            (3, 9),
            (7, 10),
            (6, 10),
            (5, 9),
            (6, 9),
            (6, 8),
            (7, 7),
        ],
        (5, 7),
    ),
    (
        "freestyle15#3",
        &[(3, 11), (12, 6), (8, 9), (3, 6), (12, 5), (3, 8)],
        (6, 11),
    ),
    (
        "freestyle15#4",
        &[
            (12, 0),
            (14, 2),
            (11, 0),
            (14, 3),
            (10, 0),
            (14, 4),
            (14, 5),
            (9, 0),
            (12, 2),
            (10, 4),
            (12, 4),
            (10, 5),
        ],
        (14, 0),
    ),
    (
        "freestyle15#5",
        &[
            (14, 2),
            (14, 3),
            (13, 2),
            (13, 3),
            (12, 2),
            (12, 3),
            (14, 8),
            (14, 7),
            (13, 8),
            (13, 7),
            (12, 8),
            (12, 7),
        ],
        (10, 3),
    ),
    (
        "freestyle15#6",
        &[
            (10, 7),
            (8, 5),
            (7, 4),
            (6, 3),
            (6, 4),
            (6, 5),
            (6, 6),
            (6, 7),
            (4, 7),
            (6, 9),
            (8, 11),
            (7, 10),
            (8, 9),
            (8, 8),
            (5, 8),
            (9, 6),
            (9, 7),
            (9, 8),
            (8, 10),
            (11, 7),
            (7, 8),
            (3, 7),
            (6, 2),
            (8, 12),
            (11, 8),
        ],
        (8, 4),
    ),
    (
        "freestyle15#7",
        &[(5, 5), (5, 4), (7, 4), (6, 3), (8, 1)],
        (4, 4),
    ),
    (
        "freestyle15#8",
        &[(5, 5), (5, 4), (5, 7), (6, 3), (7, 5)],
        (4, 5),
    ),
    (
        "freestyle15#9",
        &[(0, 5), (0, 7), (0, 9), (1, 7), (3, 8)],
        (4, 9),
    ),
    (
        "freestyle15#10",
        &[(0, 5), (0, 7), (1, 0), (1, 4), (3, 4)],
        (4, 7),
    ),
    (
        "freestyle15#11",
        &[(2, 4), (2, 5), (5, 5), (4, 4), (3, 2)],
        (4, 7),
    ),
    (
        "standard#0",
        &[(4, 11), (8, 12), (1, 13), (4, 10), (5, 8), (5, 9), (6, 10)],
        (6, 8),
    ),
    (
        "standard#1",
        &[(1, 12), (3, 13), (2, 11), (4, 9), (2, 10)],
        (5, 11),
    ),
    (
        "standard#2",
        &[
            (10, 11),
            (8, 12),
            (5, 12),
            (4, 11),
            (3, 9),
            (7, 10),
            (6, 10),
            (5, 9),
            (6, 9),
            (6, 8),
            (7, 7),
        ],
        (5, 7),
    ),
    (
        "standard#3",
        &[(7, 7), (7, 5), (8, 9), (6, 5), (5, 5)],
        (6, 6),
    ),
    (
        "standard#4",
        &[(8, 2), (10, 4), (11, 6), (10, 10), (13, 5)],
        (8, 8),
    ),
    (
        "standard#5",
        &[(7, 7), (13, 7), (7, 13), (4, 10), (1, 7), (1, 13), (1, 10)],
        (3, 11),
    ),
    (
        "standard#6",
        &[(3, 10), (13, 7), (9, 3), (5, 2), (3, 5), (12, 13), (12, 9)],
        (10, 10),
    ),
    (
        "standard#7",
        &[(7, 5), (7, 4), (9, 3), (8, 4), (5, 4)],
        (8, 6),
    ),
    (
        "standard#8",
        &[(7, 5), (8, 4), (9, 6), (8, 3), (8, 1)],
        (7, 3),
    ),
    (
        "standard#9",
        &[(7, 3), (7, 4), (5, 4), (6, 3), (9, 3)],
        (6, 5),
    ),
    (
        "standard#10",
        &[(7, 3), (7, 2), (7, 0), (5, 2), (6, 5)],
        (5, 4),
    ),
    (
        "standard#11",
        &[(3, 3), (4, 4), (6, 6), (4, 5), (4, 6)],
        (5, 6),
    ),
];

static BOOK: OnceLock<Vec<(u64, (u8, u8))>> = OnceLock::new();

fn build_book() -> Vec<(u64, (u8, u8))> {
    BOOK_RAW
        .iter()
        .filter_map(|(_label, moves, response)| {
            let mut board = Board::new();
            for &(x, y) in *moves {
                let idx = to_idx(y as usize, x as usize);
                if !board.is_empty(idx) {
                    return None;
                }
                board.make_move(idx);
            }
            Some((board.zobrist, *response))
        })
        .collect()
}

/// Look up a prepared response by current `Board::zobrist`. Returns the
/// `(x, y)` move if the position is one we have a book entry for, otherwise
/// `None` (in which case the caller should fall back to search).
pub fn lookup(zobrist: u64) -> Option<(u8, u8)> {
    BOOK.get_or_init(build_book)
        .iter()
        .find_map(|&(z, mv)| (z == zobrist).then_some(mv))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn book_entries_replay_cleanly() {
        let book = build_book();
        assert_eq!(
            book.len(),
            BOOK_RAW.len(),
            "some book entries failed replay (duplicate stones in opening?)"
        );
    }

    #[test]
    fn duplicate_hashes_agree_on_response() {
        // Freestyle and Standard share several published openings. A hash
        // collision is fine iff the prepared response is the same — if two
        // book entries map the same position to different responses, the
        // linear `lookup` would silently return whichever sits first in
        // BOOK_RAW, which is a footgun. Fail loudly in that case.
        let mut seen: std::collections::HashMap<u64, (u8, u8)> = Default::default();
        for &(zobrist, response) in &build_book() {
            if let Some(prev) = seen.insert(zobrist, response) {
                assert_eq!(
                    prev, response,
                    "openings hash to {zobrist:#x} with conflicting responses {prev:?} vs {response:?}"
                );
            }
        }
    }

    #[test]
    fn book_responses_are_on_empty_cell() {
        for (_label, moves, response) in BOOK_RAW {
            let mut board = Board::new();
            for &(x, y) in *moves {
                board.make_move(to_idx(y as usize, x as usize));
            }
            let idx = to_idx(response.1 as usize, response.0 as usize);
            assert!(
                board.is_empty(idx),
                "book response for {:?} lands on occupied cell",
                moves
            );
        }
    }
}
