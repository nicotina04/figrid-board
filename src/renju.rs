//! Renju forbidden moves for black.
//!
//! A black move is forbidden when it does not make an exact five and makes a double-four, a double-three or an
//! overline. A four is a line shape that one more black stone turns into an exact five; a straight-four line
//! (`.XXXX.`) counts once, while same-line double fours (`X.XXX.X`, `XX.XX.XX`, `XXX.X.XXX`) count twice. A three is a
//! line shape that one more black stone turns into a straight four, and it only counts if that completing move is
//! itself not forbidden (checked recursively). Precedence follows the Gomocup judge: five, double-four,
//! double-three, overline. The board edge blocks like a white stone.

use crate::board::{BOARD_SIZE, BitBoard, Move};
use std::collections::HashMap;

const DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
/// Cells on each side of the probed move that the line shapes can reach.
const R: i32 = 6;
const LEN: usize = (2 * R + 1) as usize;
const EMPTY: u8 = 0;
const BLACK: u8 = 1;
const BLOCK: u8 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Foul {
    None,
    Five,
    DoubleFour,
    DoubleThree,
    Overline,
}

/// Whether black may not play `mv` (an empty cell) under Renju.
pub fn is_forbidden(black: &BitBoard, white: &BitBoard, mv: Move) -> bool {
    matches!(foul(black, white, mv), Foul::DoubleFour | Foul::DoubleThree | Foul::Overline)
}

/// Classification of black playing the empty cell `mv`.
pub fn foul(black: &BitBoard, white: &BitBoard, mv: Move) -> Foul {
    // `HashMap::new` does not allocate; the memo only fills (and allocates) once a double-three needs recursion.
    Ctx { white, memo: HashMap::new() }.classify(black, mv)
}

/// One top-level query: white is fixed, recursive sub-queries (is this three's completion forbidden?) are memoised
/// on (black stones, cell) so shared sub-positions are solved once.
struct Ctx<'a> {
    white: &'a BitBoard,
    memo: HashMap<(u128, u128, Move), Foul>,
}

impl Ctx<'_> {
    fn foul(&mut self, black: &BitBoard, mv: Move) -> Foul {
        let key = (black.lo, black.hi, mv);
        if let Some(&f) = self.memo.get(&key) {
            return f;
        }
        let f = self.classify(black, mv);
        self.memo.insert(key, f);
        f
    }

    fn classify(&mut self, black: &BitBoard, mv: Move) -> Foul {
        let mut b = *black;
        b.set(mv);
        let row = (mv / BOARD_SIZE) as i32;
        let col = (mv % BOARD_SIZE) as i32;
        let lines = DIRS.map(|(dr, dc)| read_line(&b, self.white, row, col, dr, dc));
        if lines.iter().any(|l| run_len(l, R) == 5) {
            return Foul::Five;
        }
        if lines.iter().map(fours_through_center).sum::<u32>() >= 2 {
            return Foul::DoubleFour;
        }
        // Candidate completions per line, without recursion; a double-three needs two lines with candidates.
        let cands = lines.map(|l| straight_four_completions(&l));
        let mut remaining = cands.iter().filter(|c| c.n > 0).count();
        if remaining >= 2 {
            let mut threes = 0;
            for (d, c) in cands.iter().enumerate() {
                if c.n == 0 {
                    continue;
                }
                remaining -= 1;
                let (dr, dc) = DIRS[d];
                let real = c.offs[..c.n].iter().any(|&q| {
                    let qmv = ((row + dr * (q - R)) as usize) * BOARD_SIZE + (col + dc * (q - R)) as usize;
                    self.foul(&b, qmv) == Foul::None
                });
                if real {
                    threes += 1;
                    if threes >= 2 {
                        return Foul::DoubleThree;
                    }
                }
                if threes + remaining < 2 {
                    break;
                }
            }
        }
        if lines.iter().any(|l| run_len(l, R) >= 6) {
            return Foul::Overline;
        }
        Foul::None
    }
}

/// Up to 6 line offsets (absolute indices into the line) whose black stone makes a straight four through the centre.
struct Completions {
    offs: [i32; 6],
    n: usize,
}

fn straight_four_completions(l: &[u8; LEN]) -> Completions {
    let mut c = Completions { offs: [0; 6], n: 0 };
    for q in (R - 3)..=(R + 3) {
        if q == R || at(l, q) != EMPTY {
            continue;
        }
        let mut t = *l;
        t[q as usize] = BLACK;
        if is_straight_four_through(&t, R, q) {
            c.offs[c.n] = q;
            c.n += 1;
        }
    }
    c
}

fn read_line(black: &BitBoard, white: &BitBoard, row: i32, col: i32, dr: i32, dc: i32) -> [u8; LEN] {
    let mut l = [BLOCK; LEN];
    for (i, cell) in l.iter_mut().enumerate() {
        let off = i as i32 - R;
        let (r, c) = (row + dr * off, col + dc * off);
        if r < 0 || r >= BOARD_SIZE as i32 || c < 0 || c >= BOARD_SIZE as i32 {
            continue;
        }
        let idx = r as usize * BOARD_SIZE + c as usize;
        *cell = if black.get(idx) {
            BLACK
        } else if white.get(idx) {
            BLOCK
        } else {
            EMPTY
        };
    }
    l
}

#[inline]
fn at(l: &[u8; LEN], i: i32) -> u8 {
    if (0..LEN as i32).contains(&i) { l[i as usize] } else { BLOCK }
}

/// Length of the contiguous black run through index `i` (0 when `i` is not black).
fn run_len(l: &[u8; LEN], i: i32) -> u32 {
    if at(l, i) != BLACK {
        return 0;
    }
    let mut n = 1;
    let mut j = i - 1;
    while at(l, j) == BLACK {
        n += 1;
        j -= 1;
    }
    let mut j = i + 1;
    while at(l, j) == BLACK {
        n += 1;
        j += 1;
    }
    n
}

/// Number of distinct fours through the centre: five-cell windows containing the centre with four black stones and
/// one empty cell whose completion is an exact five. Windows sharing the same four stones (a straight four) count once.
fn fours_through_center(l: &[u8; LEN]) -> u32 {
    let mut seen: [u16; 5] = [0; 5];
    let mut n = 0;
    for s in (R - 4)..=R {
        let mut blacks = 0;
        let mut empties = 0;
        let mut mask = 0u16;
        for k in 0..5 {
            match at(l, s + k) {
                BLACK => {
                    blacks += 1;
                    mask |= 1 << (s + k);
                }
                EMPTY => empties += 1,
                _ => {}
            }
        }
        if blacks == 4 && empties == 1 && at(l, s - 1) != BLACK && at(l, s + 5) != BLACK && !seen[..n].contains(&mask) {
            seen[n] = mask;
            n += 1;
        }
    }
    n as u32
}

/// `.XXXX.` containing both `a` and `b`, with no black just beyond either end (both completions are exact fives).
fn is_straight_four_through(t: &[u8; LEN], a: i32, b: i32) -> bool {
    if run_len(t, a) != 4 {
        return false;
    }
    let mut lo = a;
    while at(t, lo - 1) == BLACK {
        lo -= 1;
    }
    let hi = lo + 3;
    b >= lo && b <= hi && at(t, lo - 1) == EMPTY && at(t, hi + 1) == EMPTY && at(t, lo - 2) != BLACK && at(t, hi + 2) != BLACK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::to_idx;

    fn board(stones: &[(usize, usize, bool)]) -> (BitBoard, BitBoard) {
        let (mut b, mut w) = (BitBoard::EMPTY, BitBoard::EMPTY);
        for &(r, c, is_black) in stones {
            if is_black { b.set(to_idx(r, c)) } else { w.set(to_idx(r, c)) }
        }
        (b, w)
    }

    #[test]
    fn double_three_is_forbidden() {
        // . X X . horizontally and vertically through (7,7)
        let (b, w) = board(&[(7, 5, true), (7, 6, true), (5, 7, true), (6, 7, true)]);
        assert_eq!(foul(&b, &w, to_idx(7, 7)), Foul::DoubleThree);
    }

    #[test]
    fn four_three_is_allowed() {
        let (b, w) = board(&[(7, 4, true), (7, 5, true), (7, 6, true), (5, 7, true), (6, 7, true)]);
        assert_eq!(foul(&b, &w, to_idx(7, 7)), Foul::None);
    }

    #[test]
    fn double_four_and_overline_are_forbidden() {
        let (b, w) = board(&[(7, 3, true), (7, 4, true), (7, 5, true), (3, 7, true), (4, 7, true), (5, 7, true)]);
        assert_eq!(foul(&b, &w, to_idx(7, 7)), Foul::DoubleFour); // two broken fours XXX.X
        let (b, w) = board(&[(7, 4, true), (7, 5, true), (7, 6, true), (4, 7, true), (5, 7, true), (6, 7, true)]);
        assert_eq!(foul(&b, &w, to_idx(7, 7)), Foul::DoubleFour);
        let (b, w) = board(&[(7, 2, true), (7, 3, true), (7, 4, true), (7, 5, true), (7, 6, true)]);
        assert_eq!(foul(&b, &w, to_idx(7, 7)), Foul::Overline);
    }

    #[test]
    fn same_line_double_four_is_forbidden() {
        // X.XXX.X with the move in the middle block
        let (b, w) = board(&[(7, 3, true), (7, 5, true), (7, 6, true), (7, 9, true)]);
        assert_eq!(foul(&b, &w, to_idx(7, 7)), Foul::DoubleFour);
    }

    #[test]
    fn exact_five_beats_fouls() {
        let (b, w) = board(&[(7, 3, true), (7, 4, true), (7, 5, true), (7, 6, true), (4, 7, true), (5, 7, true), (6, 7, true)]);
        assert_eq!(foul(&b, &w, to_idx(7, 7)), Foul::Five);
    }
}
