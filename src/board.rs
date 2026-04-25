/// 오목 보드 엔진
///
/// 15×15 보드. Bitboard 표현 (u128 × 2로 225비트 커버).
/// 흑(선공)과 백(후공) 각각 bitboard 보유.

use std::fmt;

pub const BOARD_SIZE: usize = 15;
pub const NUM_CELLS: usize = BOARD_SIZE * BOARD_SIZE; // 225

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stone {
    Black,
    White,
}

impl Stone {
    pub fn opponent(self) -> Stone {
        match self {
            Stone::Black => Stone::White,
            Stone::White => Stone::Black,
        }
    }
}

/// 225비트를 u128 × 2로 표현
/// lo: 비트 0~127, hi: 비트 128~224
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BitBoard {
    pub lo: u128,
    pub hi: u128,
}

impl BitBoard {
    pub const EMPTY: Self = Self { lo: 0, hi: 0 };

    #[inline]
    pub fn get(&self, idx: usize) -> bool {
        if idx < 128 {
            (self.lo >> idx) & 1 != 0
        } else {
            (self.hi >> (idx - 128)) & 1 != 0
        }
    }

    #[inline]
    pub fn set(&mut self, idx: usize) {
        if idx < 128 {
            self.lo |= 1u128 << idx;
        } else {
            self.hi |= 1u128 << (idx - 128);
        }
    }

    #[inline]
    pub fn clear(&mut self, idx: usize) {
        if idx < 128 {
            self.lo &= !(1u128 << idx);
        } else {
            self.hi &= !(1u128 << (idx - 128));
        }
    }

    #[inline]
    pub fn or(&self, other: &BitBoard) -> BitBoard {
        BitBoard {
            lo: self.lo | other.lo,
            hi: self.hi | other.hi,
        }
    }

    #[inline]
    pub fn count_ones(&self) -> u32 {
        self.lo.count_ones() + self.hi.count_ones()
    }

    /// Iterate over the indices of set bits, lowest first.
    /// Enables feature extraction loops to skip empty cells entirely —
    /// critical when the board is sparse (early/midgame), since a
    /// stone-driven pass is ~6× cheaper than scanning all 225 cells.
    #[inline]
    pub fn iter_ones(&self) -> BitBoardIter {
        BitBoardIter {
            lo: self.lo,
            hi: self.hi,
        }
    }
}

pub struct BitBoardIter {
    lo: u128,
    hi: u128,
}

impl Iterator for BitBoardIter {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.lo != 0 {
            let idx = self.lo.trailing_zeros() as usize;
            self.lo &= self.lo - 1;
            Some(idx)
        } else if self.hi != 0 {
            let idx = 128 + self.hi.trailing_zeros() as usize;
            self.hi &= self.hi - 1;
            Some(idx)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    BlackWin,
    WhiteWin,
    Draw,
    Ongoing,
}

/// 착수 = 보드 인덱스 (0~224)
pub type Move = usize;

#[inline]
pub fn to_rc(idx: usize) -> (usize, usize) {
    (idx / BOARD_SIZE, idx % BOARD_SIZE)
}

#[inline]
pub fn to_idx(row: usize, col: usize) -> usize {
    row * BOARD_SIZE + col
}

/// Zobrist 키 — 보드 상태의 고유 해시.
/// `(cell, color)` 별로 고정 random u64를 XOR 해서 만든다.
/// `side_to_move` 도 별도 키로 toggle. make/undo 시 incremental XOR 갱신.
mod zobrist {
    use super::{NUM_CELLS, Stone};

    /// 결정적이지만 잘 분산된 splitmix64 변형으로 컴파일 타임 키 생성.
    const fn splitmix64(seed: u64) -> u64 {
        let mut x = seed;
        x = x.wrapping_add(0x9E3779B97F4A7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
        x ^ (x >> 31)
    }

    const fn build_keys() -> [[u64; NUM_CELLS]; 2] {
        let mut out = [[0u64; NUM_CELLS]; 2];
        let mut color = 0;
        while color < 2 {
            let mut cell = 0;
            while cell < NUM_CELLS {
                let seed = (color as u64) * 0x9E3779B97F4A7C15 ^ (cell as u64);
                out[color][cell] = splitmix64(seed);
                cell += 1;
            }
            color += 1;
        }
        out
    }

    pub const STONE_KEYS: [[u64; NUM_CELLS]; 2] = build_keys();
    pub const SIDE_TO_MOVE_KEY: u64 = splitmix64(0xCAFE_BABE_DEAD_BEEF);

    #[inline]
    pub const fn key_for(stone: Stone, cell: usize) -> u64 {
        let color = match stone {
            Stone::Black => 0,
            Stone::White => 1,
        };
        STONE_KEYS[color][cell]
    }
}

pub use zobrist::SIDE_TO_MOVE_KEY as ZOBRIST_SIDE;

#[inline]
pub const fn zobrist_stone_key(stone: Stone, cell: usize) -> u64 {
    zobrist::key_for(stone, cell)
}

#[derive(Clone)]
pub struct Board {
    pub black: BitBoard,
    pub white: BitBoard,
    pub side_to_move: Stone,
    pub move_count: usize,
    pub last_move: Option<Move>,
    /// 착수 이력 (undo를 위해)
    pub history: Vec<Move>,
    /// Zobrist 해시. make_move/undo_move에서 XOR로 incremental 갱신.
    /// 보드 상태(돌 배치 + side_to_move)의 64-bit fingerprint —
    /// transposition table 키로 사용.
    pub zobrist: u64,
}

impl Board {
    pub fn new() -> Self {
        Self {
            black: BitBoard::EMPTY,
            white: BitBoard::EMPTY,
            side_to_move: Stone::Black,
            move_count: 0,
            last_move: None,
            history: Vec::with_capacity(NUM_CELLS),
            // 빈 보드 + Black to move = 0 ^ SIDE = SIDE.
            // 관례: side_to_move == Black일 때 SIDE 키 없음.
            // 더 단순하게 빈 키 0으로 시작하고 side toggle은 make_move 마다 XOR.
            zobrist: 0,
        }
    }

    /// 해당 칸이 비어있는지
    #[inline]
    pub fn is_empty(&self, idx: usize) -> bool {
        let occupied = self.black.or(&self.white);
        !occupied.get(idx)
    }

    /// 현재 턴의 돌 bitboard
    #[inline]
    pub fn current_stones(&self) -> &BitBoard {
        match self.side_to_move {
            Stone::Black => &self.black,
            Stone::White => &self.white,
        }
    }

    /// 상대 턴의 돌 bitboard
    #[inline]
    pub fn opponent_stones(&self) -> &BitBoard {
        match self.side_to_move {
            Stone::Black => &self.white,
            Stone::White => &self.black,
        }
    }

    /// 합법 수 목록 생성
    pub fn legal_moves(&self) -> Vec<Move> {
        let occupied = self.black.or(&self.white);
        let mut moves = Vec::with_capacity(NUM_CELLS - self.move_count);
        for idx in 0..NUM_CELLS {
            if !occupied.get(idx) {
                moves.push(idx);
            }
        }
        moves
    }

    /// 빈 칸 주변(2칸 이내)만 후보로 생성 — 탐색 효율화
    pub fn candidate_moves(&self) -> Vec<Move> {
        if self.move_count == 0 {
            // 첫 수: 천원
            return vec![to_idx(7, 7)];
        }

        let occupied = self.black.or(&self.white);
        let mut seen = [false; NUM_CELLS];
        let mut moves = Vec::with_capacity(64);

        for idx in 0..NUM_CELLS {
            if !occupied.get(idx) {
                continue;
            }
            let (r, c) = to_rc(idx);
            for dr in -2i32..=2 {
                for dc in -2i32..=2 {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr < 0 || nr >= BOARD_SIZE as i32 || nc < 0 || nc >= BOARD_SIZE as i32 {
                        continue;
                    }
                    let nidx = to_idx(nr as usize, nc as usize);
                    if !seen[nidx] && !occupied.get(nidx) {
                        seen[nidx] = true;
                        moves.push(nidx);
                    }
                }
            }
        }

        moves
    }

    /// 착수
    pub fn make_move(&mut self, mv: Move) {
        debug_assert!(mv < NUM_CELLS);
        debug_assert!(self.is_empty(mv));

        let placed = self.side_to_move;
        match placed {
            Stone::Black => self.black.set(mv),
            Stone::White => self.white.set(mv),
        }
        // Zobrist incremental: 새 돌의 (color, cell) 키 XOR + side toggle.
        self.zobrist ^= zobrist_stone_key(placed, mv);
        self.zobrist ^= ZOBRIST_SIDE;

        self.history.push(mv);
        self.last_move = Some(mv);
        self.move_count += 1;
        self.side_to_move = placed.opponent();
    }

    /// 착수 취소
    pub fn undo_move(&mut self) {
        if let Some(mv) = self.history.pop() {
            self.side_to_move = self.side_to_move.opponent();
            let placed = self.side_to_move;
            self.move_count -= 1;
            match placed {
                Stone::Black => self.black.clear(mv),
                Stone::White => self.white.clear(mv),
            }
            // Zobrist는 XOR이라 같은 키 한 번 더 적용 = 원복.
            self.zobrist ^= zobrist_stone_key(placed, mv);
            self.zobrist ^= ZOBRIST_SIDE;

            self.last_move = self.history.last().copied();
        }
    }

    /// 5목 승리 판정 (마지막 착수 기준)
    pub fn check_win(&self, mv: Move) -> bool {
        let (row, col) = to_rc(mv);
        let stone = if self.black.get(mv) {
            &self.black
        } else if self.white.get(mv) {
            &self.white
        } else {
            return false;
        };

        // 4방향: 가로, 세로, 대각선(\), 역대각선(/)
        let directions: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];

        for &(dr, dc) in &directions {
            let mut count = 1;

            // 정방향
            for step in 1..5 {
                let nr = row as i32 + dr * step;
                let nc = col as i32 + dc * step;
                if nr < 0 || nr >= BOARD_SIZE as i32 || nc < 0 || nc >= BOARD_SIZE as i32 {
                    break;
                }
                if stone.get(to_idx(nr as usize, nc as usize)) {
                    count += 1;
                } else {
                    break;
                }
            }

            // 역방향
            for step in 1..5 {
                let nr = row as i32 - dr * step;
                let nc = col as i32 - dc * step;
                if nr < 0 || nr >= BOARD_SIZE as i32 || nc < 0 || nc >= BOARD_SIZE as i32 {
                    break;
                }
                if stone.get(to_idx(nr as usize, nc as usize)) {
                    count += 1;
                } else {
                    break;
                }
            }

            if count >= 5 {
                return true;
            }
        }

        false
    }

    /// 게임 결과 확인
    pub fn game_result(&self) -> GameResult {
        if let Some(mv) = self.last_move {
            if self.check_win(mv) {
                // 마지막에 둔 사람이 이김 (side_to_move는 이미 넘어간 상태)
                return match self.side_to_move {
                    Stone::Black => GameResult::WhiteWin,
                    Stone::White => GameResult::BlackWin,
                };
            }
        }
        if self.move_count >= NUM_CELLS {
            GameResult::Draw
        } else {
            GameResult::Ongoing
        }
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "   ")?;
        for c in 0..BOARD_SIZE {
            write!(f, "{:2}", (b'A' + c as u8) as char)?;
        }
        writeln!(f)?;

        for r in 0..BOARD_SIZE {
            write!(f, "{:2} ", r + 1)?;
            for c in 0..BOARD_SIZE {
                let idx = to_idx(r, c);
                if self.black.get(idx) {
                    write!(f, " X")?;
                } else if self.white.get(idx) {
                    write!(f, " O")?;
                } else {
                    write!(f, " .")?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_undo_move() {
        let mut board = Board::new();
        let mv = to_idx(7, 7);
        board.make_move(mv);
        assert!(board.black.get(mv));
        assert_eq!(board.side_to_move, Stone::White);

        board.undo_move();
        assert!(!board.black.get(mv));
        assert_eq!(board.side_to_move, Stone::Black);
        assert_eq!(board.move_count, 0);
    }

    /// Zobrist 정합성: make/undo가 incremental XOR로 정확히 원복되는지.
    #[test]
    fn zobrist_make_undo_roundtrip() {
        let mut board = Board::new();
        let initial = board.zobrist;
        assert_eq!(initial, 0, "empty board zobrist should be 0");

        let moves = [112, 113, 97, 98, 127, 128, 200, 14];
        let mut keys = vec![initial];
        for &m in &moves {
            board.make_move(m);
            keys.push(board.zobrist);
        }
        // 모든 중간 키가 unique해야 함 (충돌 없는 상태에서)
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), keys.len(), "zobrist sequence collided");

        // undo 시 역순으로 정확히 같은 키 복귀
        for i in (1..keys.len()).rev() {
            board.undo_move();
            assert_eq!(
                board.zobrist, keys[i - 1],
                "zobrist mismatch after undo step {i}"
            );
        }
        assert_eq!(board.zobrist, 0, "zobrist did not return to 0 after full undo");
    }

    /// 수 순서 무관 same position → same zobrist.
    /// 두 시퀀스가 같은 final position을 만들면 zobrist도 같아야 함.
    #[test]
    fn zobrist_path_independence() {
        let seq1 = [112, 113, 97, 98]; // B(7,7) W(7,8) B(6,7) W(6,8)
        let seq2 = [112, 98, 97, 113]; // 같은 4 돌, 다른 순서 — 단 흑/백 같은 셀에 두는 순서 보존되어야 함

        // seq2 invalid (흑이 (7,7)→(6,8)→(6,7)→(7,8) 순서로 두면 백도 다른 셀)
        // 정확한 path-equivalent 짝: 두 흑 수 순서 바꾸기
        // seq1: B(112), W(113), B(97), W(98)  → black={112,97}, white={113,98}
        // seq2: B(97), W(113), B(112), W(98)  → black={97,112}, white={113,98}  같은 final
        let seq2 = [97, 113, 112, 98];

        let mut b1 = Board::new();
        for &m in &seq1 { b1.make_move(m); }
        let mut b2 = Board::new();
        for &m in &seq2 { b2.make_move(m); }

        assert_eq!(b1.black.lo, b2.black.lo);
        assert_eq!(b1.black.hi, b2.black.hi);
        assert_eq!(b1.white.lo, b2.white.lo);
        assert_eq!(b1.white.hi, b2.white.hi);
        assert_eq!(b1.side_to_move, b2.side_to_move);

        assert_eq!(b1.zobrist, b2.zobrist, "same position should have same zobrist");
    }

    #[test]
    fn test_horizontal_win() {
        let mut board = Board::new();
        // 흑: (7,3) (7,4) (7,5) (7,6) (7,7)
        // 백: (8,3) (8,4) (8,5) (8,6)
        for i in 0..5 {
            board.make_move(to_idx(7, 3 + i)); // 흑
            if i < 4 {
                board.make_move(to_idx(8, 3 + i)); // 백
            }
        }
        assert_eq!(board.game_result(), GameResult::BlackWin);
    }

    #[test]
    fn test_diagonal_win() {
        let mut board = Board::new();
        // 흑: (0,0) (1,1) (2,2) (3,3) (4,4) — 대각선
        // 백: (0,1) (1,2) (2,3) (3,4)
        for i in 0..5 {
            board.make_move(to_idx(i, i)); // 흑
            if i < 4 {
                board.make_move(to_idx(i, i + 1)); // 백
            }
        }
        assert_eq!(board.game_result(), GameResult::BlackWin);
    }

    #[test]
    fn test_no_win_with_four() {
        let mut board = Board::new();
        for i in 0..4 {
            board.make_move(to_idx(7, 3 + i)); // 흑
            board.make_move(to_idx(8, 3 + i)); // 백
        }
        assert_eq!(board.game_result(), GameResult::Ongoing);
    }

    #[test]
    fn test_candidate_moves_first() {
        let board = Board::new();
        let moves = board.candidate_moves();
        assert_eq!(moves, vec![to_idx(7, 7)]);
    }
}
