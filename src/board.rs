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

#[derive(Clone)]
pub struct Board {
    pub black: BitBoard,
    pub white: BitBoard,
    pub side_to_move: Stone,
    pub move_count: usize,
    pub last_move: Option<Move>,
    /// 착수 이력 (undo를 위해)
    pub history: Vec<Move>,
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

        match self.side_to_move {
            Stone::Black => self.black.set(mv),
            Stone::White => self.white.set(mv),
        }
        self.history.push(mv);
        self.last_move = Some(mv);
        self.move_count += 1;
        self.side_to_move = self.side_to_move.opponent();
    }

    /// 착수 취소
    pub fn undo_move(&mut self) {
        if let Some(mv) = self.history.pop() {
            self.side_to_move = self.side_to_move.opponent();
            self.move_count -= 1;
            match self.side_to_move {
                Stone::Black => self.black.clear(mv),
                Stone::White => self.white.clear(mv),
            }
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
