/// 오목 보드 엔진
///
/// 15×15 보드. Bitboard 표현 (u128 × 2로 225비트 커버).
/// 흑(선공)과 백(후공) 각각 bitboard 보유.
use std::fmt;

pub const BOARD_SIZE: usize = 15;
pub const NUM_CELLS: usize = BOARD_SIZE * BOARD_SIZE; // 225
pub const LINE_PATTERN_FRONTIER_MAX: usize = 41;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleSet {
    Freestyle,
    Standard,
    Caro,
    /// Terminal-line semantics only. Renju forbidden-move legality is not
    /// implemented yet, so pbrain keeps rejecting Renju games for now.
    Renju,
}

impl RuleSet {
    #[inline]
    pub const fn uses_exact_five(self) -> bool {
        matches!(self, RuleSet::Standard | RuleSet::Renju)
    }

    #[inline]
    pub const fn line_wins(self, side: Stone, count: u32, open_ends: u32) -> bool {
        match self {
            RuleSet::Freestyle => count >= 5,
            RuleSet::Standard => count == 5,
            RuleSet::Caro => count >= 6 || (count == 5 && open_ends > 0),
            RuleSet::Renju => match side {
                Stone::Black => count == 5,
                Stone::White => count >= 5,
            },
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

#[inline]
fn in_board(row: i32, col: i32) -> bool {
    row >= 0 && row < BOARD_SIZE as i32 && col >= 0 && col < BOARD_SIZE as i32
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

/// 4 directional 11-cell line pattern mapped IDs per cell.
/// Pattern4 mini의 incremental state cache. 값 ∈ [0, PATTERN_NUM_IDS)
/// (= 0..PATTERN_NUM_IDS): swap-closed mapped ids plus rare bucket. u16에 들어감.
///
/// Black-relative storage: 1=black, 2=white로 read_window. side_to_move
/// 변경에 따라 ID 재계산 안 함 (perspective 변환은 NNUE feature 매핑
/// 단계에서 처리).
pub type LinePatternState = Box<[[u16; 4]; NUM_CELLS]>;

const NO_CANDIDATE_SOURCE: u8 = u8::MAX;

/// Optional radius-2 candidate frontier.
///
/// `by_min_source[s]` contains exactly the empty candidates whose lowest
/// occupied radius-2 neighbor is `s`. Iterating non-empty buckets and then
/// their bits in ascending order reproduces the legacy discovery order.
#[derive(Clone)]
struct CandidateFrontierState {
    radius2_count: [u8; NUM_CELLS],
    candidates: BitBoard,
    min_source: [u8; NUM_CELLS],
    by_min_source: [BitBoard; NUM_CELLS],
    nonempty_sources: BitBoard,
}

impl CandidateFrontierState {
    fn empty() -> Self {
        Self {
            radius2_count: [0; NUM_CELLS],
            candidates: BitBoard::EMPTY,
            min_source: [NO_CANDIDATE_SOURCE; NUM_CELLS],
            by_min_source: [BitBoard::EMPTY; NUM_CELLS],
            nonempty_sources: BitBoard::EMPTY,
        }
    }

    #[inline]
    fn bucket_insert(&mut self, source: usize, cell: usize) {
        let bucket = &mut self.by_min_source[source];
        let was_empty = bucket.lo == 0 && bucket.hi == 0;
        bucket.set(cell);
        if was_empty {
            self.nonempty_sources.set(source);
        }
    }

    #[inline]
    fn bucket_remove(&mut self, source: usize, cell: usize) {
        let bucket = &mut self.by_min_source[source];
        debug_assert!(bucket.get(cell));
        bucket.clear(cell);
        if bucket.lo == 0 && bucket.hi == 0 {
            self.nonempty_sources.clear(source);
        }
    }

    #[inline]
    fn candidate_insert(&mut self, source: usize, cell: usize) {
        debug_assert!(!self.candidates.get(cell));
        debug_assert_eq!(self.min_source[cell], NO_CANDIDATE_SOURCE);
        self.candidates.set(cell);
        self.min_source[cell] = source as u8;
        self.bucket_insert(source, cell);
    }

    #[inline]
    fn candidate_remove(&mut self, cell: usize) {
        debug_assert!(self.candidates.get(cell));
        let source = self.min_source[cell] as usize;
        debug_assert!(source < NUM_CELLS);
        self.bucket_remove(source, cell);
        self.candidates.clear(cell);
        self.min_source[cell] = NO_CANDIDATE_SOURCE;
    }

    #[inline]
    fn candidate_rekey(&mut self, new_source: usize, cell: usize) {
        debug_assert!(self.candidates.get(cell));
        let old_source = self.min_source[cell] as usize;
        debug_assert!(old_source < NUM_CELLS);
        self.bucket_remove(old_source, cell);
        self.min_source[cell] = new_source as u8;
        self.bucket_insert(new_source, cell);
    }
}

/// Incrementally maintained raw 22-bit Pattern4 windows.
///
/// Entries use the same black-relative layout as `pack_window`: window
/// index 0 occupies bits 21..20 and index 10 occupies bits 1..0.
pub type LinePackedWindowState = Box<[[u32; 4]; NUM_CELLS]>;

/// Search-only acceleration state kept outside [`Board`].
///
/// This sidecar deliberately preserves the exact public `Board` field shape
/// from figrid-board 0.8.1, so downstream exhaustive struct literals remain
/// source-compatible. Callers that opt in must route every searched
/// make/undo through this state while it is enabled.
#[doc(hidden)]
#[derive(Clone, Default)]
pub struct BoardSearchState {
    line_packed_windows: Option<LinePackedWindowState>,
    candidate_frontier: Option<Box<CandidateFrontierState>>,
    synchronized_position: Option<(u64, usize)>,
}

impl BoardSearchState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether every enabled cache describes the supplied board position.
    #[inline]
    pub fn is_synchronized(&self, board: &Board) -> bool {
        if self.line_packed_windows.is_none() && self.candidate_frontier.is_none() {
            true
        } else {
            self.synchronized_position == Some((board.zobrist, board.move_count))
        }
    }

    /// Rebuild all currently enabled caches if the board changed outside the
    /// sidecar (for example, between two searches).
    pub fn synchronize(&mut self, board: &Board) {
        if self.is_synchronized(board) {
            return;
        }
        let packed_enabled = self.line_packed_windows.is_some();
        let frontier_enabled = self.candidate_frontier.is_some();
        self.line_packed_windows = None;
        self.candidate_frontier = None;
        self.synchronized_position = None;
        if packed_enabled {
            self.set_packed_line_windows_enabled(board, true);
        }
        if frontier_enabled {
            self.set_candidate_frontier_enabled(board, true);
        }
    }

    #[inline]
    fn record_position(&mut self, board: &Board) {
        self.synchronized_position = (self.line_packed_windows.is_some()
            || self.candidate_frontier.is_some())
        .then_some((board.zobrist, board.move_count));
    }

    /// Enable or disable incremental packed Pattern4 windows.
    ///
    /// Enabling performs one full rebuild at the supplied root. Descendant
    /// make/undo operations routed through this sidecar then update only one
    /// 2-bit slot per affected window.
    pub fn set_packed_line_windows_enabled(&mut self, board: &Board, enabled: bool) {
        self.synchronize(board);
        if enabled {
            if self.line_packed_windows.is_none() {
                let mut windows = Box::new([[0u32; 4]; NUM_CELLS]);
                const DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
                for cell in 0..NUM_CELLS {
                    let row = (cell / BOARD_SIZE) as i32;
                    let col = (cell % BOARD_SIZE) as i32;
                    for (dir_idx, &(dr, dc)) in DIRS.iter().enumerate() {
                        let window = crate::pattern_table::read_window(
                            &board.black,
                            &board.white,
                            row,
                            col,
                            dr,
                            dc,
                        );
                        let packed = crate::pattern_table::pack_window(&window);
                        debug_assert_eq!(
                            board.line_pattern_ids[cell][dir_idx],
                            crate::pattern_table::lookup_mapped_id(packed),
                            "line pattern ID stale before packed-window enable"
                        );
                        windows[cell][dir_idx] = packed;
                    }
                }
                self.line_packed_windows = Some(windows);
            }
        } else {
            self.line_packed_windows = None;
        }
        self.record_position(board);
    }

    #[inline]
    pub fn packed_line_windows_enabled(&self) -> bool {
        self.line_packed_windows.is_some()
    }

    /// Expose one packed value for correctness and audit harnesses.
    #[inline]
    pub fn packed_line_window(&self, cell: usize, dir_idx: usize) -> Option<u32> {
        self.line_packed_windows
            .as_ref()
            .map(|windows| windows[cell][dir_idx])
    }

    /// Enable or disable the exact-order incremental candidate frontier.
    ///
    /// Enabling performs one full rebuild at the supplied root.
    pub fn set_candidate_frontier_enabled(&mut self, board: &Board, enabled: bool) {
        self.synchronize(board);
        if enabled {
            if self.candidate_frontier.is_none() {
                self.candidate_frontier = Some(board.rebuild_candidate_frontier());
            }
        } else {
            self.candidate_frontier = None;
        }
        self.record_position(board);
    }

    #[inline]
    pub fn candidate_frontier_enabled(&self) -> bool {
        self.candidate_frontier.is_some()
    }

    /// Generate candidates in exactly the legacy discovery order.
    pub fn candidate_moves(&self, board: &Board) -> Vec<Move> {
        if !self.is_synchronized(board) {
            return board.candidate_moves();
        }
        self.candidate_moves_synchronized(board)
    }

    #[inline]
    pub(crate) fn candidate_moves_synchronized(&self, board: &Board) -> Vec<Move> {
        debug_assert!(
            self.is_synchronized(board),
            "BoardSearchState is stale for candidate generation"
        );
        if board.move_count == 0 {
            return vec![to_idx(7, 7)];
        }
        let Some(frontier) = self.candidate_frontier.as_ref() else {
            return board.candidate_moves();
        };
        board.candidate_moves_from_frontier(frontier)
    }

    /// Apply a move while maintaining every enabled sidecar cache.
    #[inline]
    pub fn make_move(&mut self, board: &mut Board, mv: Move) {
        self.synchronize(board);
        self.make_move_synchronized(board, mv);
    }

    #[inline]
    pub(crate) fn make_move_synchronized(&mut self, board: &mut Board, mv: Move) {
        debug_assert!(
            self.is_synchronized(board),
            "BoardSearchState is stale before make_move"
        );
        board.make_move_with_search_state(
            mv,
            self.line_packed_windows.as_deref_mut(),
            self.candidate_frontier.as_deref_mut(),
        );
        self.record_position(board);
    }

    /// Undo a move while maintaining every enabled sidecar cache.
    #[inline]
    pub fn undo_move(&mut self, board: &mut Board) {
        self.synchronize(board);
        self.undo_move_synchronized(board);
    }

    #[inline]
    pub(crate) fn undo_move_synchronized(&mut self, board: &mut Board) {
        debug_assert!(
            self.is_synchronized(board),
            "BoardSearchState is stale before undo_move"
        );
        board.undo_move_with_search_state(
            self.line_packed_windows.as_deref_mut(),
            self.candidate_frontier.as_deref_mut(),
        );
        self.record_position(board);
    }
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
    /// Pattern4 mini state cache. 각 (cell, dir) 11-cell 윈도우의
    /// canonical pattern ID (black-relative). 빈 보드는 모두 ID 0
    /// (empty_pattern_id). make/undo가 영향받는 cell의 ID만 lookup으로
    /// 재계산해 region recompute를 피함. NNUE feature 매핑은 Phase 3에서.
    pub line_pattern_ids: LinePatternState,
    /// Active Gomoku rule set for terminal line checks.
    pub rule_set: RuleSet,
    /// Backward-compatible Standard-rule flag used by older tools.
    /// Prefer `set_rule_set` for new code.
    pub exact5: bool,
}

impl Board {
    pub fn new() -> Self {
        let mut b = Self {
            black: BitBoard::EMPTY,
            white: BitBoard::EMPTY,
            side_to_move: Stone::Black,
            move_count: 0,
            last_move: None,
            history: Vec::with_capacity(NUM_CELLS),
            // 빈 보드 + Black to move 의 zobrist 는 0.
            zobrist: 0,
            // 정확한 초기값은 fill_initial_pattern_ids 에서 채움 (가장자리는
            // boundary 포함이라 ID ≠ 0).
            rule_set: RuleSet::Freestyle,
            line_pattern_ids: Box::new([[0u16; 4]; NUM_CELLS]),
            exact5: false,
        };
        b.fill_initial_pattern_ids();
        b
    }

    #[inline]
    pub fn set_rule_set(&mut self, rule_set: RuleSet) {
        self.rule_set = rule_set;
        self.exact5 = rule_set.uses_exact_five();
    }

    #[inline]
    pub fn effective_rule_set(&self) -> RuleSet {
        // Backward compatibility for older tools that still do
        // `board.exact5 = true` after Board::new().
        if self.exact5 && matches!(self.rule_set, RuleSet::Freestyle) {
            RuleSet::Standard
        } else {
            self.rule_set
        }
    }

    /// 빈 보드 기준 모든 (cell, dir) line pattern mapped ID를 lookup해 채움.
    /// new() 에서만 호출. 가장자리 cell은 boundary 포함 패턴이라 빈 cell의
    /// 안쪽 ID(보통 0)과 다른 mapped ID로 채워짐.
    fn fill_initial_pattern_ids(&mut self) {
        const DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
        for cell in 0..NUM_CELLS {
            let row = (cell / BOARD_SIZE) as i32;
            let col = (cell % BOARD_SIZE) as i32;
            for (dir_idx, &(dr, dc)) in DIRS.iter().enumerate() {
                let w =
                    crate::pattern_table::read_window(&self.black, &self.white, row, col, dr, dc);
                let packed = crate::pattern_table::pack_window(&w);
                self.line_pattern_ids[cell][dir_idx] =
                    crate::pattern_table::lookup_mapped_id(packed);
            }
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

    #[inline]
    fn for_radius2_neighbors(cell: usize, mut f: impl FnMut(usize)) {
        let (row, col) = to_rc(cell);
        let row_start = row.saturating_sub(2);
        let row_end = (row + 2).min(BOARD_SIZE - 1);
        let col_start = col.saturating_sub(2);
        let col_end = (col + 2).min(BOARD_SIZE - 1);
        for neighbor_row in row_start..=row_end {
            for neighbor_col in col_start..=col_end {
                let neighbor = to_idx(neighbor_row, neighbor_col);
                if neighbor != cell {
                    f(neighbor);
                }
            }
        }
    }

    #[inline]
    fn first_radius2_source(occupied: &BitBoard, cell: usize) -> Option<usize> {
        let mut first = None;
        Self::for_radius2_neighbors(cell, |neighbor| {
            if first.is_none() && occupied.get(neighbor) {
                first = Some(neighbor);
            }
        });
        first
    }

    fn rebuild_candidate_frontier(&self) -> Box<CandidateFrontierState> {
        let occupied = self.black.or(&self.white);
        let mut state = Box::new(CandidateFrontierState::empty());
        for source in occupied.iter_ones() {
            Self::for_radius2_neighbors(source, |cell| {
                state.radius2_count[cell] += 1;
            });
        }
        for cell in 0..NUM_CELLS {
            if !occupied.get(cell) && state.radius2_count[cell] > 0 {
                let source = Self::first_radius2_source(&occupied, cell)
                    .expect("positive radius-2 count must have a source");
                state.candidate_insert(source, cell);
            }
        }
        state
    }

    fn candidate_moves_legacy(&self) -> Vec<Move> {
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

    /// 빈 칸 주변(2칸 이내)만 후보로 생성 — 탐색 효율화
    pub fn candidate_moves(&self) -> Vec<Move> {
        if self.move_count == 0 {
            // 첫 수: 천원
            return vec![to_idx(7, 7)];
        }
        return self.candidate_moves_legacy();
    }

    fn candidate_moves_from_frontier(&self, frontier: &CandidateFrontierState) -> Vec<Move> {
        let mut moves = Vec::with_capacity(frontier.candidates.count_ones() as usize);
        for source in frontier.nonempty_sources.iter_ones() {
            moves.extend(frontier.by_min_source[source].iter_ones());
        }
        debug_assert_eq!(moves.len(), frontier.candidates.count_ones() as usize);
        moves
    }

    #[inline]
    fn update_candidate_frontier_after_make(
        &self,
        frontier: &mut CandidateFrontierState,
        mv: Move,
    ) {
        let occupied = self.black.or(&self.white);

        if frontier.candidates.get(mv) {
            frontier.candidate_remove(mv);
        }
        Self::for_radius2_neighbors(mv, |cell| {
            let old_count = frontier.radius2_count[cell];
            debug_assert!(old_count < 24);
            frontier.radius2_count[cell] = old_count + 1;
            if occupied.get(cell) {
                return;
            }
            if old_count == 0 {
                frontier.candidate_insert(mv, cell);
            } else {
                let old_source = frontier.min_source[cell] as usize;
                debug_assert!(old_source < NUM_CELLS);
                if mv < old_source {
                    frontier.candidate_rekey(mv, cell);
                }
            }
        });
    }

    #[inline]
    fn update_candidate_frontier_after_undo(
        &self,
        frontier: &mut CandidateFrontierState,
        mv: Move,
    ) {
        let occupied = self.black.or(&self.white);

        Self::for_radius2_neighbors(mv, |cell| {
            let old_count = frontier.radius2_count[cell];
            debug_assert!(old_count > 0);
            frontier.radius2_count[cell] = old_count - 1;
            if occupied.get(cell) {
                return;
            }
            if old_count == 1 {
                frontier.candidate_remove(cell);
            } else if frontier.min_source[cell] as usize == mv {
                let new_source = Self::first_radius2_source(&occupied, cell)
                    .expect("remaining radius-2 count must have a source");
                frontier.candidate_rekey(new_source, cell);
            }
        });

        if frontier.radius2_count[mv] > 0 {
            let source = Self::first_radius2_source(&occupied, mv)
                .expect("newly empty candidate must have a radius-2 source");
            frontier.candidate_insert(source, mv);
        }
    }

    /// 착수
    pub fn make_move(&mut self, mv: Move) {
        self.make_move_with_search_state(mv, None, None);
    }

    #[inline]
    fn make_move_with_search_state(
        &mut self,
        mv: Move,
        line_packed_windows: Option<&mut [[u32; 4]; NUM_CELLS]>,
        candidate_frontier: Option<&mut CandidateFrontierState>,
    ) {
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

        // Maintain the optional radius-2 frontier after the stone is visible.
        if let Some(frontier) = candidate_frontier {
            self.update_candidate_frontier_after_make(frontier, mv);
        }

        // Pattern4 mini state cache: mv 주변 4방향 ±5 cell의 pattern_id 갱신.
        // black-relative: read_window의 첫 인자 = black. side_to_move 무관.
        self.update_line_patterns_around(
            mv,
            match placed {
                Stone::Black => 1,
                Stone::White => 2,
            },
            line_packed_windows,
        );
    }

    /// 착수 취소
    pub fn undo_move(&mut self) {
        self.undo_move_with_search_state(None, None);
    }

    #[inline]
    fn undo_move_with_search_state(
        &mut self,
        line_packed_windows: Option<&mut [[u32; 4]; NUM_CELLS]>,
        candidate_frontier: Option<&mut CandidateFrontierState>,
    ) {
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

            // The stone is already cleared, so restore the inverse radius-2
            // delta and reinsert `mv` when another stone reaches it.
            if let Some(frontier) = candidate_frontier {
                self.update_candidate_frontier_after_undo(frontier, mv);
            }

            // Pattern4 state cache: mv 주변 4방향 ±5 cell 다시 read+lookup.
            // mv는 이미 cleared된 상태라 새 윈도우에서 mv = empty.
            self.update_line_patterns_around(mv, 0, line_packed_windows);
        }
    }

    /// Unique cells touched by the same 4-direction +/-5 frontier used for
    /// incremental `line_pattern_ids` maintenance. The maximum is 41
    /// (center + 10 cells in each of four directions).
    pub fn line_pattern_dirty_cells(
        mv: Move,
        out: &mut [usize; LINE_PATTERN_FRONTIER_MAX],
    ) -> usize {
        let mut seen = [false; NUM_CELLS];
        let mut len = 0usize;
        Self::for_line_pattern_frontier(mv, |cell, _dir_idx, _offset| {
            if !seen[cell] {
                seen[cell] = true;
                debug_assert!(len < LINE_PATTERN_FRONTIER_MAX);
                out[len] = cell;
                len += 1;
            }
        });
        len
    }

    #[inline]
    fn for_line_pattern_frontier(mut mv: Move, mut f: impl FnMut(usize, usize, i32)) {
        const DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
        debug_assert!(mv < NUM_CELLS);
        let row = (mv / BOARD_SIZE) as i32;
        let col = (mv % BOARD_SIZE) as i32;
        for (dir_idx, &(dr, dc)) in DIRS.iter().enumerate() {
            for offset in -5i32..=5 {
                let r = row + dr * offset;
                let c = col + dc * offset;
                if r < 0 || r >= BOARD_SIZE as i32 || c < 0 || c >= BOARD_SIZE as i32 {
                    continue;
                }
                mv = (r as usize) * BOARD_SIZE + c as usize;
                f(mv, dir_idx, offset);
            }
        }
    }

    /// `mv` 주변 4방향 각 ±5 cell (총 ~30~44 cell-dir 쌍)의 11-cell window
    /// pattern ID를 다시 lookup해 cache 갱신. 보드 경계로 일부 잘림.
    /// black-relative — read_window의 첫 인자 = black, 둘째 = white.
    #[inline]
    fn update_line_patterns_around(
        &mut self,
        mv: Move,
        new_cell: u32,
        line_packed_windows: Option<&mut [[u32; 4]; NUM_CELLS]>,
    ) {
        const DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
        debug_assert!(new_cell <= 2);
        if let Some(windows) = line_packed_windows {
            let ids = &mut self.line_pattern_ids;
            Self::for_line_pattern_frontier(mv, |cell, dir_idx, offset| {
                // The changed board cell is at window index (5 - offset).
                // Index 10 is in the low bits, hence:
                // shift = (10 - (5 - offset)) * 2 = (5 + offset) * 2.
                let shift = ((5 + offset) * 2) as u32;
                let mask = 0b11u32 << shift;
                let packed = (windows[cell][dir_idx] & !mask) | (new_cell << shift);
                windows[cell][dir_idx] = packed;
                ids[cell][dir_idx] = crate::pattern_table::lookup_mapped_id(packed);
            });
        } else {
            Self::for_line_pattern_frontier(mv, |cell, dir_idx, _offset| {
                let (dr, dc) = DIRS[dir_idx];
                let r = (cell / BOARD_SIZE) as i32;
                let c = (cell % BOARD_SIZE) as i32;
                let window =
                    crate::pattern_table::read_window(&self.black, &self.white, r, c, dr, dc);
                let packed = crate::pattern_table::pack_window(&window);
                self.line_pattern_ids[cell][dir_idx] =
                    crate::pattern_table::lookup_mapped_id(packed);
            });
        }
    }

    /// Return whether the stone at `mv` completes a winning line under the active rule.
    pub fn check_win(&self, mv: Move) -> bool {
        let (row, col) = to_rc(mv);
        let (side, stone) = if self.black.get(mv) {
            (Stone::Black, &self.black)
        } else if self.white.get(mv) {
            (Stone::White, &self.white)
        } else {
            return false;
        };
        let rules = self.effective_rule_set();

        let directions: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
        for &(dr, dc) in &directions {
            let (count, open_ends) = self.line_run(stone, row as i32, col as i32, dr, dc);
            if rules.line_wins(side, count, open_ends) {
                return true;
            }
        }
        false
    }

    #[inline]
    fn line_run(&self, stone: &BitBoard, row: i32, col: i32, dr: i32, dc: i32) -> (u32, u32) {
        let mut count = 1u32;
        let mut open_ends = 0u32;

        let mut r = row + dr;
        let mut c = col + dc;
        while in_board(r, c) && stone.get(to_idx(r as usize, c as usize)) {
            count += 1;
            r += dr;
            c += dc;
        }
        if in_board(r, c) && self.is_empty(to_idx(r as usize, c as usize)) {
            open_ends += 1;
        }

        let mut r = row - dr;
        let mut c = col - dc;
        while in_board(r, c) && stone.get(to_idx(r as usize, c as usize)) {
            count += 1;
            r -= dr;
            c -= dc;
        }
        if in_board(r, c) && self.is_empty(to_idx(r as usize, c as usize)) {
            open_ends += 1;
        }

        (count, open_ends)
    }

    #[inline]
    pub fn is_legal_move(&self, mv: Move) -> bool {
        mv < NUM_CELLS && self.is_empty(mv)
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

    fn put_stone(board: &mut Board, side: Stone, row: usize, col: usize) -> Move {
        let mv = to_idx(row, col);
        assert!(board.is_empty(mv));
        match side {
            Stone::Black => board.black.set(mv),
            Stone::White => board.white.set(mv),
        }
        board.move_count += 1;
        mv
    }

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
                board.zobrist,
                keys[i - 1],
                "zobrist mismatch after undo step {i}"
            );
        }
        assert_eq!(
            board.zobrist, 0,
            "zobrist did not return to 0 after full undo"
        );
    }

    /// Pattern4 state cache 정합성: incremental update 결과가 같은 보드를
    /// 처음부터 fill_initial_pattern_ids 로 채운 결과와 모든 (cell, dir)에서
    /// 동일해야 한다. region recompute가 아닌 진짜 incremental의 핵심 invariant.
    #[test]
    fn line_pattern_state_make_undo_consistency() {
        const DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
        let moves = [112, 113, 97, 98, 127, 128, 200, 14, 0, 224, 7, 217, 50, 100];

        let mut board = Board::new();
        let initial_ids = board.line_pattern_ids.clone();

        // make_move 각 단계마다 incremental ids == 처음부터 재계산한 ids
        for (i, &mv) in moves.iter().enumerate() {
            if !board.is_empty(mv) {
                continue;
            }
            board.make_move(mv);

            // incremental 후 fresh 보드 재구성 (history replay) + fill_initial 비교
            let mut fresh = Board::new();
            for &m in &moves[..=i] {
                if fresh.is_empty(m) {
                    fresh.make_move(m);
                }
            }
            // 또는 더 강하게: 직접 처음부터 재구성한 board의 line_pattern_ids
            // == 우리 incremental board의 line_pattern_ids
            // fresh 도 incremental 사용하므로 다른 검증: 직접 read_window 계산
            for cell in 0..NUM_CELLS {
                let row = (cell / BOARD_SIZE) as i32;
                let col = (cell % BOARD_SIZE) as i32;
                for (dir_idx, &(dr, dc)) in DIRS.iter().enumerate() {
                    let w = crate::pattern_table::read_window(
                        &board.black,
                        &board.white,
                        row,
                        col,
                        dr,
                        dc,
                    );
                    let packed = crate::pattern_table::pack_window(&w);
                    let expected = crate::pattern_table::lookup_mapped_id(packed);
                    let actual = board.line_pattern_ids[cell][dir_idx];
                    assert_eq!(
                        actual,
                        expected,
                        "mismatch at cell {} dir {} after move {} (ply {})",
                        cell,
                        dir_idx,
                        mv,
                        i + 1
                    );
                }
            }
        }

        // undo 모두 → initial ids 복원
        for _ in 0..moves.len() {
            if !board.history.is_empty() {
                board.undo_move();
            }
        }
        assert_eq!(board.move_count, 0);
        // initial board 와 같은 ids
        for cell in 0..NUM_CELLS {
            for d in 0..4 {
                assert_eq!(
                    board.line_pattern_ids[cell][d], initial_ids[cell][d],
                    "after full undo: cell {} dir {} not restored",
                    cell, d
                );
            }
        }
    }

    fn assert_packed_windows_match_full_rebuild(
        board: &Board,
        state: &BoardSearchState,
        operation: usize,
    ) {
        const DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
        assert!(state.packed_line_windows_enabled());
        assert!(state.is_synchronized(board));
        for cell in 0..NUM_CELLS {
            let row = (cell / BOARD_SIZE) as i32;
            let col = (cell % BOARD_SIZE) as i32;
            for (dir_idx, &(dr, dc)) in DIRS.iter().enumerate() {
                let window =
                    crate::pattern_table::read_window(&board.black, &board.white, row, col, dr, dc);
                let expected_packed = crate::pattern_table::pack_window(&window);
                let actual_packed = state.packed_line_window(cell, dir_idx).unwrap();
                assert_eq!(
                    actual_packed, expected_packed,
                    "packed mismatch after operation {operation}, cell {cell}, dir {dir_idx}"
                );
                assert_eq!(
                    board.line_pattern_ids[cell][dir_idx],
                    crate::pattern_table::lookup_mapped_id(expected_packed),
                    "mapped-id mismatch after operation {operation}, cell {cell}, dir {dir_idx}"
                );
            }
        }
    }

    #[test]
    fn packed_line_windows_are_opt_in_for_library_callers_and_toggle_cleanly() {
        let mut board = Board::new();
        let mut state = BoardSearchState::new();
        assert!(!state.packed_line_windows_enabled());
        board.make_move(to_idx(7, 7));
        state.set_packed_line_windows_enabled(&board, true);
        assert_packed_windows_match_full_rebuild(&board, &state, 1);
        state.set_packed_line_windows_enabled(&board, false);
        assert!(!state.packed_line_windows_enabled());
        board.undo_move();
        let empty = Board::new();
        assert!(board.black == empty.black);
        assert!(board.white == empty.white);
        assert_eq!(board.side_to_move, empty.side_to_move);
        assert_eq!(board.move_count, 0);
        assert_eq!(board.line_pattern_ids, empty.line_pattern_ids);
    }

    /// Release correctness gate: 100k deterministic mixed make/undo
    /// operations, lockstep equality with the legacy updater at every step,
    /// plus periodic full raw-window rebuild equality.
    #[test]
    #[ignore = "100k release audit; run explicitly with --release --ignored"]
    fn packed_line_windows_100k_make_undo_full_rebuild_equality() {
        const OPERATIONS: usize = 100_000;
        const FULL_REBUILD_PERIOD: usize = 97;

        let mut legacy = Board::new();
        let mut packed = Board::new();
        let mut state = BoardSearchState::new();
        state.set_packed_line_windows_enabled(&packed, true);
        let mut rng = 0xDFA2_2026_0725_0001u64;

        for operation in 1..=OPERATIONS {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let should_undo =
                !legacy.history.is_empty() && (legacy.move_count >= 180 || (rng & 0b11) == 0);

            if should_undo {
                legacy.undo_move();
                state.undo_move(&mut packed);
            } else {
                let start = (rng as usize) % NUM_CELLS;
                let mv = (0..NUM_CELLS)
                    .map(|delta| (start + delta) % NUM_CELLS)
                    .find(|&cell| legacy.is_empty(cell))
                    .expect("at least one empty cell before make");
                legacy.make_move(mv);
                state.make_move(&mut packed, mv);
            }

            assert!(
                packed.black == legacy.black,
                "black at operation {operation}"
            );
            assert!(
                packed.white == legacy.white,
                "white at operation {operation}"
            );
            assert_eq!(
                packed.side_to_move, legacy.side_to_move,
                "side at operation {operation}"
            );
            assert_eq!(
                packed.line_pattern_ids, legacy.line_pattern_ids,
                "mapped IDs at operation {operation}"
            );

            if operation % FULL_REBUILD_PERIOD == 0 || operation == OPERATIONS {
                assert_packed_windows_match_full_rebuild(&packed, &state, operation);
            }
        }

        while !legacy.history.is_empty() {
            legacy.undo_move();
            state.undo_move(&mut packed);
        }
        assert_eq!(packed.line_pattern_ids, legacy.line_pattern_ids);
        assert_packed_windows_match_full_rebuild(&packed, &state, OPERATIONS + 1);
    }

    /// 수 순서 무관 same position → same zobrist.
    /// 두 시퀀스가 같은 final position을 만들면 zobrist도 같아야 함.
    #[test]
    fn zobrist_path_independence() {
        let seq1 = [112, 113, 97, 98]; // B(7,7) W(7,8) B(6,7) W(6,8)
        let _seq2 = [112, 98, 97, 113]; // 같은 4 돌, 다른 순서 — 단 흑/백 같은 셀에 두는 순서 보존되어야 함

        // seq2 invalid (흑이 (7,7)→(6,8)→(6,7)→(7,8) 순서로 두면 백도 다른 셀)
        // 정확한 path-equivalent 짝: 두 흑 수 순서 바꾸기
        // seq1: B(112), W(113), B(97), W(98)  → black={112,97}, white={113,98}
        // seq2: B(97), W(113), B(112), W(98)  → black={97,112}, white={113,98}  같은 final
        let seq2 = [97, 113, 112, 98];

        let mut b1 = Board::new();
        for &m in &seq1 {
            b1.make_move(m);
        }
        let mut b2 = Board::new();
        for &m in &seq2 {
            b2.make_move(m);
        }

        assert_eq!(b1.black.lo, b2.black.lo);
        assert_eq!(b1.black.hi, b2.black.hi);
        assert_eq!(b1.white.lo, b2.white.lo);
        assert_eq!(b1.white.hi, b2.white.hi);
        assert_eq!(b1.side_to_move, b2.side_to_move);

        assert_eq!(
            b1.zobrist, b2.zobrist,
            "same position should have same zobrist"
        );
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
    fn freestyle_overline_wins() {
        let mut board = Board::new();
        let mut last = 0;
        for col in 3..=8 {
            last = put_stone(&mut board, Stone::Black, 7, col);
        }
        assert!(board.check_win(last));
    }

    #[test]
    fn standard_overline_does_not_win() {
        let mut board = Board::new();
        board.set_rule_set(RuleSet::Standard);
        let mut last = 0;
        for col in 3..=8 {
            last = put_stone(&mut board, Stone::Black, 7, col);
        }
        assert!(!board.check_win(last));
    }

    #[test]
    fn standard_exact_five_wins() {
        let mut board = Board::new();
        board.set_rule_set(RuleSet::Standard);
        let mut last = 0;
        for col in 3..=7 {
            last = put_stone(&mut board, Stone::Black, 7, col);
        }
        assert!(board.check_win(last));
    }

    #[test]
    fn caro_blocked_exact_five_does_not_win() {
        let mut board = Board::new();
        board.set_rule_set(RuleSet::Caro);
        put_stone(&mut board, Stone::White, 7, 3);
        put_stone(&mut board, Stone::White, 7, 9);
        let mut last = 0;
        for col in 4..=8 {
            last = put_stone(&mut board, Stone::Black, 7, col);
        }
        assert!(!board.check_win(last));
    }

    #[test]
    fn caro_one_open_exact_five_wins() {
        let mut board = Board::new();
        board.set_rule_set(RuleSet::Caro);
        put_stone(&mut board, Stone::White, 7, 3);
        let mut last = 0;
        for col in 4..=8 {
            last = put_stone(&mut board, Stone::Black, 7, col);
        }
        assert!(board.check_win(last));
    }

    #[test]
    fn caro_overline_wins_even_when_blocked() {
        let mut board = Board::new();
        board.set_rule_set(RuleSet::Caro);
        put_stone(&mut board, Stone::White, 7, 3);
        put_stone(&mut board, Stone::White, 7, 10);
        let mut last = 0;
        for col in 4..=9 {
            last = put_stone(&mut board, Stone::Black, 7, col);
        }
        assert!(board.check_win(last));
    }

    #[test]
    fn test_candidate_moves_first() {
        let board = Board::new();
        let moves = board.candidate_moves();
        assert_eq!(moves, vec![to_idx(7, 7)]);
    }

    fn assert_candidate_frontier_matches_full_rebuild(
        board: &Board,
        state: &BoardSearchState,
        operation: usize,
    ) {
        assert!(state.is_synchronized(board));
        let actual = state
            .candidate_frontier
            .as_ref()
            .expect("candidate frontier must be enabled");
        let rebuilt = board.rebuild_candidate_frontier();
        assert_eq!(
            actual.radius2_count, rebuilt.radius2_count,
            "radius2 counts at operation {operation}"
        );
        assert_eq!(
            actual.min_source, rebuilt.min_source,
            "minimum sources at operation {operation}"
        );
        assert!(
            actual.candidates == rebuilt.candidates,
            "candidate bitboard at operation {operation}"
        );
        assert!(
            actual.nonempty_sources == rebuilt.nonempty_sources,
            "non-empty source buckets at operation {operation}"
        );
        for source in 0..NUM_CELLS {
            assert!(
                actual.by_min_source[source] == rebuilt.by_min_source[source],
                "source bucket {source} at operation {operation}"
            );
        }
        assert_eq!(
            state.candidate_moves(board),
            if board.move_count == 0 {
                vec![to_idx(7, 7)]
            } else {
                board.candidate_moves_legacy()
            },
            "exact candidate order at operation {operation}"
        );
    }

    #[test]
    fn candidate_frontier_is_opt_in_and_preserves_exact_order() {
        let sequence = [
            to_idx(7, 7),
            to_idx(0, 0),
            to_idx(14, 14),
            to_idx(7, 8),
            to_idx(2, 13),
            to_idx(12, 1),
        ];
        let mut board = Board::new();
        let mut state = BoardSearchState::new();
        assert!(!state.candidate_frontier_enabled());
        for &mv in &sequence {
            board.make_move(mv);
        }
        let expected = board.candidate_moves();
        state.set_candidate_frontier_enabled(&board, true);
        assert!(state.candidate_frontier_enabled());
        assert_eq!(state.candidate_moves(&board), expected);
        assert_candidate_frontier_matches_full_rebuild(&board, &state, sequence.len());
        state.set_candidate_frontier_enabled(&board, false);
        assert!(!state.candidate_frontier_enabled());
        assert_eq!(state.candidate_moves(&board), expected);
    }

    /// Release composition correctness gate: 100k deterministic mixed
    /// make/undo operations, exact ordered-vector equality against the legacy
    /// generator, and periodic complete frontier rebuild equality.
    #[test]
    #[ignore = "100k release audit; run explicitly with --release --ignored"]
    fn candidate_frontier_100k_make_undo_full_rebuild_equality() {
        const OPERATIONS: usize = 100_000;
        const FULL_REBUILD_PERIOD: usize = 97;

        let mut legacy = Board::new();
        let mut incremental = Board::new();
        let mut state = BoardSearchState::new();
        state.set_packed_line_windows_enabled(&incremental, true);
        state.set_candidate_frontier_enabled(&incremental, true);
        let mut rng = 0xDFA3_2026_0725_0001u64;

        for operation in 1..=OPERATIONS {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let should_undo =
                !legacy.history.is_empty() && (legacy.move_count >= 180 || (rng & 0b11) == 0);

            if should_undo {
                legacy.undo_move();
                state.undo_move(&mut incremental);
            } else {
                let start = (rng as usize) % NUM_CELLS;
                let mv = (0..NUM_CELLS)
                    .map(|delta| (start + delta) % NUM_CELLS)
                    .find(|&cell| legacy.is_empty(cell))
                    .expect("at least one empty cell before make");
                legacy.make_move(mv);
                state.make_move(&mut incremental, mv);
            }

            assert!(incremental.black == legacy.black);
            assert!(incremental.white == legacy.white);
            assert_eq!(incremental.side_to_move, legacy.side_to_move);
            assert_eq!(
                incremental.line_pattern_ids, legacy.line_pattern_ids,
                "mapped pattern IDs at operation {operation}"
            );
            assert_eq!(
                state.candidate_moves(&incremental),
                legacy.candidate_moves(),
                "ordered candidates at operation {operation}"
            );

            if operation % FULL_REBUILD_PERIOD == 0 || operation == OPERATIONS {
                assert_packed_windows_match_full_rebuild(&incremental, &state, operation);
                assert_candidate_frontier_matches_full_rebuild(&incremental, &state, operation);
            }
        }

        while !legacy.history.is_empty() {
            legacy.undo_move();
            state.undo_move(&mut incremental);
            assert_eq!(
                state.candidate_moves(&incremental),
                legacy.candidate_moves()
            );
        }
        assert_packed_windows_match_full_rebuild(&incremental, &state, OPERATIONS + 1);
        assert_candidate_frontier_matches_full_rebuild(&incremental, &state, OPERATIONS + 1);
    }
}
