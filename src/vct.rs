//! VCT (Victory by Continuous Threats) 탐색.
//!
//! 공격 측이 강제수(4목/열린3 계열)만으로 상대를 몰아가 5목을 완성하는
//! 수열을 찾는다. 일반 알파-베타와 독립적으로 돌아가는 AND-OR 트리 탐색.
//!
//! - OR 노드(공격 턴): 공격 수 중 하나라도 승리로 이어지면 승리
//! - AND 노드(수비 턴): 모든 방어 수에 대해 공격 측이 여전히 이길 수 있어야 승리
//!
//! 강제수 분류 (ThreatKind):
//!   - Five            : 즉시 승리 (5목 완성)
//!   - OpenFour        : 다음 수에 Five 확정, 방어 불가
//!   - DoubleFour      : 두 방향 동시 4목, 방어 불가
//!   - FourThree       : 4목 + 열린3 공존, 4목 방어 시 3 → 열린4로 승리
//!   - DoubleThree     : 두 열린3 공존, 한쪽 방어 시 다른 쪽 열린4로 승리
//!   - ClosedFour      : 방어 가능하지만 강제수 (안 막으면 즉시 5목)
//!   - OpenThree       : 방어 가능하지만 강제수 (안 막으면 열린4)
//!
//! 승리 Threat(Five/OpenFour/DoubleFour/FourThree/DoubleThree)을 만들면 해당
//! 수를 반환하고 즉시 성공. 그 외 Forcing Threat(ClosedFour/OpenThree)은 재귀.

use crate::board::{BOARD_SIZE, BitBoard, Board, Move, NUM_CELLS, RuleSet, Stone};
use crate::heuristic::{DIR, scan_line};
use crate::pattern_table::{
    PATTERN_RARE_ID, WindowThreat, pattern_threat_after_my_play, pattern_threat_after_my_play_caro,
    pattern_threat_after_my_play_exact5, read_window, swap_mapped_id, visit_gap_four_gaps,
    window_has_gap_four, window_has_jump_three,
};
use noru::trainer::SimpleRng;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

// === Zobrist hashing ===

static ZOBRIST_KEYS: OnceLock<[[u64; 2]; NUM_CELLS]> = OnceLock::new();
const ZOBRIST_SIDE_WHITE: u64 = 0x5A5A_5A5A_A5A5_A5A5;

fn zobrist_keys() -> &'static [[u64; 2]; NUM_CELLS] {
    ZOBRIST_KEYS.get_or_init(|| {
        let mut rng = SimpleRng::new(0xDEAD_BEEF_CAFE_BABE);
        let mut arr = [[0u64; 2]; NUM_CELLS];
        for slot in arr.iter_mut() {
            slot[0] = rng.next_u64();
            slot[1] = rng.next_u64();
        }
        arr
    })
}

/// 보드 상태를 Zobrist hash로 인코딩.
fn zobrist_hash(board: &Board) -> u64 {
    let keys = zobrist_keys();
    let mut h = 0u64;
    for idx in 0..NUM_CELLS {
        if board.black.get(idx) {
            h ^= keys[idx][0];
        }
        if board.white.get(idx) {
            h ^= keys[idx][1];
        }
    }
    if board.side_to_move == Stone::White {
        h ^= ZOBRIST_SIDE_WHITE;
    }
    h
}

/// TT 엔트리 — depth가 실제 탐색한 깊이보다 크거나 같을 때 재사용 가능.
#[derive(Clone, Copy)]
struct TtEntry {
    /// 탐색된 최대 깊이 (현재 노드 기준 공격-수비 쌍 수).
    depth: u32,
    /// 해당 깊이 이하에서 AND/OR 결과가 확정됐는지.
    result: TtResult,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TtResult {
    /// 공격자가 이김 (OR 노드에서 true 확정).
    AttackerWins,
    /// 공격자가 진다 (OR 노드에서 false 확정, 모든 공격 수 실패).
    Fails,
}

type TransTable = HashMap<u64, TtEntry>;

/// 단일 방향에서 특정 돌이 만드는 패턴 등급.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum LineThreat {
    None,
    OpenTwo,     // (2, open 2)
    ClosedThree, // (3, open 1)
    OpenThree,   // (3, open 2)
    ClosedFour,  // (4, open 1)
    OpenFour,    // (4, open 2)
    Five,        // (>=5)
}

fn classify_line(count: u32, open_ends: u32, rule_set: RuleSet, side: Stone) -> LineThreat {
    if rule_set.line_wins(side, count, open_ends) {
        return LineThreat::Five;
    }
    match (count, open_ends) {
        (4, 2) => LineThreat::OpenFour,
        (4, 1) => LineThreat::ClosedFour,
        (3, 2) => LineThreat::OpenThree,
        (3, 1) => LineThreat::ClosedThree,
        (2, 2) => LineThreat::OpenTwo,
        _ => LineThreat::None,
    }
}

/// 한 수가 종합적으로 만드는 위협 종합 평가.
///
/// `#[repr(u8)]` + 명시 discriminant: search.rs에서 packed score table 인덱스
/// (`kind as usize`)로 쓰기 위해. 값이 바뀌면 search.rs의 `THREAT_*_TABLE`도
/// 같이 수정해야 함 (THREAT_KIND_COUNT도).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ThreatKind {
    None = 0,
    ClosedFour = 1,
    OpenThree = 2,
    Five = 3,
    OpenFour = 4,
    DoubleFour = 5,
    FourThree = 6,
    DoubleThree = 7,
    JumpThree = 8,
}

/// `ThreatKind` discriminant의 수 — 테이블 크기 상수.
pub const THREAT_KIND_COUNT: usize = 9;

impl ThreatKind {
    /// 이 Threat이 형성되면 상대가 1수로 막을 수 없는지.
    pub fn is_winning(self) -> bool {
        matches!(
            self,
            ThreatKind::Five
                | ThreatKind::OpenFour
                | ThreatKind::DoubleFour
                | ThreatKind::FourThree
                | ThreatKind::DoubleThree
        )
    }

    /// 재귀 탐색해볼 가치가 있는 Forcing move인가 (방어 가능하지만 강제).
    pub fn is_forcing(self) -> bool {
        matches!(
            self,
            ThreatKind::ClosedFour | ThreatKind::OpenThree | ThreatKind::JumpThree
        ) || self.is_winning()
    }
}

fn is_vct_terminal_win(kind: ThreatKind) -> bool {
    // RQ547b-1/RQ549: DoubleThree and FourThree are forcing threats, but
    // accepting them as terminal proof shortcuts produced false VCT proofs in
    // selected losses.
    matches!(
        kind,
        ThreatKind::Five | ThreatKind::OpenFour | ThreatKind::DoubleFour
    )
}

/// side 쪽이 mv 좌표에 돌을 두면 어떤 Threat이 생기는지 분석.
///
/// my_bb는 side의 돌 bitboard, opp_bb는 상대 bitboard. **mv 위치에는 아직 돌이
/// 없는 상태**로 가정. 내부적으로 mv가 놓였을 때의 4방향 라인을 시뮬레이션.
pub fn classify_move(my_bb: &BitBoard, opp_bb: &BitBoard, mv: Move, exact5: bool) -> ThreatKind {
    let rule_set = if exact5 {
        RuleSet::Standard
    } else {
        RuleSet::Freestyle
    };
    classify_move_rules(my_bb, opp_bb, mv, Stone::Black, rule_set)
}

pub fn classify_move_rules(
    my_bb: &BitBoard,
    opp_bb: &BitBoard,
    mv: Move,
    side: Stone,
    rule_set: RuleSet,
) -> ThreatKind {
    classify_move_rules_with_flags(my_bb, opp_bb, mv, side, rule_set, false, false)
}

#[cfg(test)]
fn classify_move_rules_with_jump_three(
    my_bb: &BitBoard,
    opp_bb: &BitBoard,
    mv: Move,
    side: Stone,
    rule_set: RuleSet,
    enable_jump_three: bool,
) -> ThreatKind {
    classify_move_rules_with_flags(my_bb, opp_bb, mv, side, rule_set, enable_jump_three, false)
}

fn classify_move_rules_with_flags(
    my_bb: &BitBoard,
    opp_bb: &BitBoard,
    mv: Move,
    side: Stone,
    rule_set: RuleSet,
    enable_jump_three: bool,
    enable_gap_four: bool,
) -> ThreatKind {
    let row = (mv / BOARD_SIZE) as i32;
    let col = (mv % BOARD_SIZE) as i32;

    let mut my_tmp = *my_bb;
    my_tmp.set(mv);

    let mut fours = 0u32;
    let mut open_fours = 0u32;
    let mut open_threes = 0u32;
    let mut closed_fours = 0u32;
    let mut fives = 0u32;
    let mut jump_threes = 0u32;

    for &(dr, dc) in &DIR {
        let info = scan_line(&my_tmp, opp_bb, row, col, dr, dc);
        let open_ends = info.open_front as u32 + info.open_back as u32;
        let mut line_threat = classify_line(info.count, open_ends, rule_set, side);
        let mut window = None;
        if enable_gap_four
            && !matches!(
                line_threat,
                LineThreat::Five | LineThreat::OpenFour | LineThreat::ClosedFour
            )
        {
            let w = read_window(&my_tmp, opp_bb, row, col, dr, dc);
            if window_has_gap_four(&w) {
                line_threat = LineThreat::ClosedFour;
            }
            window = Some(w);
        }
        match line_threat {
            LineThreat::Five => fives += 1,
            LineThreat::OpenFour => {
                open_fours += 1;
                fours += 1;
            }
            LineThreat::ClosedFour => {
                closed_fours += 1;
                fours += 1;
            }
            LineThreat::OpenThree => open_threes += 1,
            _ => {}
        }
        if enable_jump_three
            && !matches!(
                line_threat,
                LineThreat::Five | LineThreat::OpenFour | LineThreat::ClosedFour
            )
        {
            let w = window.unwrap_or_else(|| read_window(&my_tmp, opp_bb, row, col, dr, dc));
            if window_has_jump_three(&w) {
                jump_threes += 1;
            }
        }
    }

    if fives >= 1 {
        return ThreatKind::Five;
    }
    if open_fours >= 1 {
        return ThreatKind::OpenFour;
    }
    if fours >= 2 {
        return ThreatKind::DoubleFour;
    }
    if closed_fours >= 1 && open_threes >= 1 {
        return ThreatKind::FourThree;
    }
    if open_threes >= 2 {
        return ThreatKind::DoubleThree;
    }
    if closed_fours >= 1 {
        return ThreatKind::ClosedFour;
    }
    if open_threes >= 1 {
        return ThreatKind::OpenThree;
    }
    if enable_jump_three && jump_threes >= 1 {
        return ThreatKind::JumpThree;
    }
    ThreatKind::None
}

/// Pattern4 fast path. Uses `board.line_pattern_ids` (incrementally
/// maintained on every `make_move` / `undo_move`) to look up each
/// direction's line threat in O(1) and aggregate into a `ThreatKind`.
///
/// Semantically equivalent to `classify_move(my_bb, opp_bb, mv)` — assumes
/// `mv` is an empty cell that `side` is about to play on. Top-K patterns
/// (~97.5% of positions) hit the precomputed lookup; the RARE bucket falls
/// back to a direct window read + scan so correctness is preserved.
///
/// In the search hot path this replaces `~64` `BitBoard::get` calls per
/// candidate move (four directions × `scan_line` of eight cells × two
/// bitboards) with four array lookups, so the classification cost drops
/// roughly an order of magnitude. NNUE forward evaluation still dominates
/// per-node cost, so the actual nodes-per-second uplift is measured rather
/// than assumed.
pub fn classify_move_fast(board: &Board, mv: Move, side: Stone) -> ThreatKind {
    classify_move_fast_with_flags(board, mv, side, false, false)
}

/// RQ572 audit hook: expose the slow rule-scan classifier with the same
/// flags as the fast Pattern4 classifier. This is not used by search.
pub fn classify_move_rules_with_flags_for_audit(
    board: &Board,
    mv: Move,
    side: Stone,
    enable_jump_three: bool,
    enable_gap_four: bool,
) -> ThreatKind {
    debug_assert!(board.is_empty(mv), "candidate move cell must be empty");
    let (my, opp) = bb_pair(board, side);
    classify_move_rules_with_flags(
        my,
        opp,
        mv,
        side,
        board.effective_rule_set(),
        enable_jump_three,
        enable_gap_four,
    )
}

/// RQ572 audit hook: expose the fast Pattern4 classifier with feature flags.
/// This is paired with `classify_move_rules_with_flags_for_audit`.
pub fn classify_move_fast_with_flags_for_audit(
    board: &Board,
    mv: Move,
    side: Stone,
    enable_jump_three: bool,
    enable_gap_four: bool,
) -> ThreatKind {
    debug_assert!(board.is_empty(mv), "candidate move cell must be empty");
    classify_move_fast_with_flags(board, mv, side, enable_jump_three, enable_gap_four)
}

#[cfg(test)]
fn classify_move_fast_with_jump_three(
    board: &Board,
    mv: Move,
    side: Stone,
    enable_jump_three: bool,
) -> ThreatKind {
    classify_move_fast_with_flags(board, mv, side, enable_jump_three, false)
}

fn classify_move_fast_with_flags(
    board: &Board,
    mv: Move,
    side: Stone,
    enable_jump_three: bool,
    enable_gap_four: bool,
) -> ThreatKind {
    let row = (mv / BOARD_SIZE) as i32;
    let col = (mv % BOARD_SIZE) as i32;
    let side_is_black = matches!(side, Stone::Black);
    let (mine, opp) = if side_is_black {
        (&board.black, &board.white)
    } else {
        (&board.white, &board.black)
    };
    let rule_set = board.effective_rule_set();

    let mut fours = 0u32;
    let mut open_fours = 0u32;
    let mut closed_fours = 0u32;
    let mut open_threes = 0u32;
    let mut fives = 0u32;
    let mut jump_threes = 0u32;

    for (dir_idx, &(dr, dc)) in DIR.iter().enumerate() {
        // line_pattern_ids stores patterns from the black-relative frame.
        // For the white-to-move query we look up the swapped pattern ID.
        let pid_black = board.line_pattern_ids[mv][dir_idx];
        let pid_my = if side_is_black {
            pid_black
        } else {
            swap_mapped_id(pid_black)
        };

        let mut threat = if pid_my == PATTERN_RARE_ID {
            // Slow path: read the actual window and classify directly.
            let mut w = read_window(mine, opp, row, col, dr, dc);
            // Anchor (index 5) must be empty for the use case. Set anchor=mine.
            debug_assert_eq!(w[5], 0, "candidate move cell must be empty");
            w[5] = 1;
            // We only consume `pattern_table::WindowThreat` so we can inline
            // a classify here (no `scan_line` / `LineThreat` round trip): walk
            // outward from the anchor counting consecutive mines and check
            // both endpoints for openness.
            let mut count = 1u32;
            let mut open_front = false;
            for off in 1usize..=5 {
                match w[5 + off] {
                    1 => count += 1,
                    0 => {
                        open_front = true;
                        break;
                    }
                    _ => break,
                }
            }
            let mut open_back = false;
            for off in 1usize..=5 {
                match w[5 - off] {
                    1 => count += 1,
                    0 => {
                        open_back = true;
                        break;
                    }
                    _ => break,
                }
            }
            let open_ends = open_front as u32 + open_back as u32;
            if rule_set.line_wins(side, count, open_ends) {
                WindowThreat::Five
            } else {
                match (count, open_ends) {
                    (4, 2) => WindowThreat::OpenFour,
                    (4, 1) => WindowThreat::ClosedFour,
                    (3, 2) => WindowThreat::OpenThree,
                    (3, 1) => WindowThreat::ClosedThree,
                    _ if enable_gap_four && window_has_gap_four(&w) => WindowThreat::ClosedFour,
                    _ if enable_jump_three && window_has_jump_three(&w) => WindowThreat::JumpThree,
                    (2, 2) => WindowThreat::OpenTwo,
                    _ => WindowThreat::None,
                }
            }
        } else {
            match rule_set {
                RuleSet::Caro => pattern_threat_after_my_play_caro(pid_my),
                RuleSet::Standard => pattern_threat_after_my_play_exact5(pid_my),
                RuleSet::Renju if matches!(side, Stone::Black) => {
                    pattern_threat_after_my_play_exact5(pid_my)
                }
                _ => pattern_threat_after_my_play(pid_my),
            }
        };
        if enable_gap_four
            && !matches!(
                threat,
                WindowThreat::Five | WindowThreat::OpenFour | WindowThreat::ClosedFour
            )
        {
            let mut w = read_window(mine, opp, row, col, dr, dc);
            debug_assert_eq!(w[5], 0, "candidate move cell must be empty");
            w[5] = 1;
            if window_has_gap_four(&w) {
                threat = WindowThreat::ClosedFour;
            }
        }

        match threat {
            WindowThreat::Five => fives += 1,
            WindowThreat::OpenFour => {
                open_fours += 1;
                fours += 1;
            }
            WindowThreat::ClosedFour => {
                closed_fours += 1;
                fours += 1;
            }
            WindowThreat::OpenThree => open_threes += 1,
            WindowThreat::JumpThree if enable_jump_three => jump_threes += 1,
            _ => {}
        }
        // dr / dc are only consumed inside the RARE fallback's `read_window`;
        // silence the unused-binding lint on the fast lookup branch.
        let _ = (dr, dc);
    }

    if fives >= 1 {
        return ThreatKind::Five;
    }
    if open_fours >= 1 {
        return ThreatKind::OpenFour;
    }
    if fours >= 2 {
        return ThreatKind::DoubleFour;
    }
    if closed_fours >= 1 && open_threes >= 1 {
        return ThreatKind::FourThree;
    }
    if open_threes >= 2 {
        return ThreatKind::DoubleThree;
    }
    if closed_fours >= 1 {
        return ThreatKind::ClosedFour;
    }
    if open_threes >= 1 {
        return ThreatKind::OpenThree;
    }
    if enable_jump_three && jump_threes >= 1 {
        return ThreatKind::JumpThree;
    }
    ThreatKind::None
}

/// VCT 설정.
pub struct VctConfig {
    /// 최대 재귀 깊이 (공격-수비 쌍 수). 너무 크면 폭발.
    pub max_depth: u32,
    /// 전체 시간 예산. 초과 시 None 반환.
    pub time_budget: Option<Duration>,
    /// RQ573 deterministic semantic gate: stop after this many entered nodes.
    pub node_budget: Option<u64>,
    /// RQ560 compatibility gate. When true, enables all JumpThree sub-flags.
    pub enable_jump_three: bool,
    /// RQ561 factor: attack JumpThree candidates plus matching gap/completion defenses.
    pub enable_jump_three_attack_defense: bool,
    /// RQ561 factor: defender counter-forcing scan may include JumpThree moves.
    pub enable_jump_three_counter: bool,
    /// RQ562 reach-control: only add direct JumpThree defenses for JumpThree OR moves.
    pub enable_jump_three_kind_scoped_defense: bool,
    /// RQ563 scheduling: generate JumpThree attack candidates only for OR levels < K.
    pub jump_attack_max_or_levels: u32,
    /// RQ567 gate: broken-four (`X.XXX` / `XX.XX` / `XXX.X`) vocabulary.
    pub enable_gap_four: bool,
    /// RQ572 gate: use Pattern4 fast classify in prover attack/counter scans.
    pub use_fast_classify: bool,
    /// RQ574 gate: use a VCT-local incremental threat index.
    pub use_threat_index: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct VctJumpThreeFlags {
    attack_defense: bool,
    counter: bool,
    kind_scoped_defense: bool,
    attack_max_or_levels: u32,
    gap_four: bool,
    use_fast_classify: bool,
    use_threat_index: bool,
}

impl VctConfig {
    fn jump_three_flags(&self) -> VctJumpThreeFlags {
        VctJumpThreeFlags {
            attack_defense: self.enable_jump_three || self.enable_jump_three_attack_defense,
            counter: self.enable_jump_three || self.enable_jump_three_counter,
            kind_scoped_defense: self.enable_jump_three_kind_scoped_defense,
            attack_max_or_levels: self.jump_attack_max_or_levels,
            gap_four: self.enable_gap_four,
            use_fast_classify: self.use_fast_classify,
            use_threat_index: self.use_threat_index,
        }
    }
}

const THREAT_INDEX_FLAG_COMBOS: usize = 4;
const THREAT_INDEX_SIDES: usize = 2;

#[derive(Clone)]
struct VctThreatIndex {
    cell_kinds: Box<[[[u8; NUM_CELLS]; THREAT_INDEX_FLAG_COMBOS]; THREAT_INDEX_SIDES]>,
    bits: [[[BitBoard; THREAT_KIND_COUNT]; THREAT_INDEX_FLAG_COMBOS]; THREAT_INDEX_SIDES],
}

impl VctThreatIndex {
    fn new(board: &Board) -> Self {
        let mut index = Self {
            cell_kinds: Box::new(
                [[[ThreatKind::None as u8; NUM_CELLS]; THREAT_INDEX_FLAG_COMBOS];
                    THREAT_INDEX_SIDES],
            ),
            bits: [[[BitBoard::EMPTY; THREAT_KIND_COUNT]; THREAT_INDEX_FLAG_COMBOS];
                THREAT_INDEX_SIDES],
        };
        for cell in 0..NUM_CELLS {
            index.recompute_cell(board, cell);
        }
        index
    }

    fn attack_moves(
        &self,
        side: Stone,
        enable_jump_three: bool,
        enable_gap_four: bool,
    ) -> Vec<(Move, ThreatKind)> {
        let side_idx = threat_index_side(side);
        let flags_idx = threat_index_flags(enable_jump_three, enable_gap_four);
        let mut out = Vec::new();
        for &kind in &THREAT_PRIORITY_ORDER {
            let bits = self.bits[side_idx][flags_idx][kind as usize];
            for mv in bits.iter_ones() {
                out.push((mv, kind));
            }
        }
        out
    }

    fn forcing_moves_in_cell_order(
        &self,
        side: Stone,
        enable_jump_three: bool,
        enable_gap_four: bool,
    ) -> Vec<Move> {
        let side_idx = threat_index_side(side);
        let flags_idx = threat_index_flags(enable_jump_three, enable_gap_four);
        let mut out = Vec::new();
        for cell in 0..NUM_CELLS {
            let kind = threat_kind_from_index(self.cell_kinds[side_idx][flags_idx][cell]);
            if kind.is_forcing() {
                out.push(cell);
            }
        }
        out
    }

    fn has_kind(
        &self,
        side: Stone,
        enable_jump_three: bool,
        enable_gap_four: bool,
        kind: ThreatKind,
    ) -> bool {
        self.bits[threat_index_side(side)][threat_index_flags(enable_jump_three, enable_gap_four)]
            [kind as usize]
            .count_ones()
            > 0
    }

    fn clear_cell(&mut self, cell: usize) {
        for side_idx in 0..THREAT_INDEX_SIDES {
            for flags_idx in 0..THREAT_INDEX_FLAG_COMBOS {
                let kind_idx = self.cell_kinds[side_idx][flags_idx][cell] as usize;
                if kind_idx != ThreatKind::None as usize {
                    self.bits[side_idx][flags_idx][kind_idx].clear(cell);
                    self.cell_kinds[side_idx][flags_idx][cell] = ThreatKind::None as u8;
                }
            }
        }
    }

    fn recompute_cell(&mut self, board: &Board, cell: usize) {
        self.clear_cell(cell);
        if !board.is_empty(cell) {
            return;
        }
        for &side in &[Stone::Black, Stone::White] {
            let side_idx = threat_index_side(side);
            for flags_idx in 0..THREAT_INDEX_FLAG_COMBOS {
                let (enable_jump_three, enable_gap_four) = threat_index_combo_flags(flags_idx);
                let kind = classify_move_fast_with_flags(
                    board,
                    cell,
                    side,
                    enable_jump_three,
                    enable_gap_four,
                );
                self.cell_kinds[side_idx][flags_idx][cell] = kind as u8;
                if kind != ThreatKind::None {
                    self.bits[side_idx][flags_idx][kind as usize].set(cell);
                }
            }
        }
    }

    fn clear_cells(&mut self, cells: &[usize]) {
        for &cell in cells {
            self.clear_cell(cell);
        }
    }

    fn recompute_cells(&mut self, board: &Board, cells: &[usize]) {
        for &cell in cells {
            self.recompute_cell(board, cell);
        }
    }

    fn matches_rebuild(&self, board: &Board) -> bool {
        let rebuilt = VctThreatIndex::new(board);
        self.cell_kinds == rebuilt.cell_kinds && self.bits == rebuilt.bits
    }

    #[cfg(test)]
    fn assert_matches_rebuild(&self, board: &Board) {
        assert!(
            self.matches_rebuild(board),
            "threat-index mismatch against rebuild"
        );
    }
}

const THREAT_PRIORITY_ORDER: [ThreatKind; THREAT_KIND_COUNT - 1] = [
    ThreatKind::Five,
    ThreatKind::OpenFour,
    ThreatKind::DoubleFour,
    ThreatKind::FourThree,
    ThreatKind::DoubleThree,
    ThreatKind::ClosedFour,
    ThreatKind::OpenThree,
    ThreatKind::JumpThree,
];

#[inline]
fn threat_index_side(side: Stone) -> usize {
    match side {
        Stone::Black => 0,
        Stone::White => 1,
    }
}

#[inline]
fn threat_index_flags(enable_jump_three: bool, enable_gap_four: bool) -> usize {
    (enable_jump_three as usize) | ((enable_gap_four as usize) << 1)
}

#[inline]
fn threat_index_combo_flags(flags_idx: usize) -> (bool, bool) {
    ((flags_idx & 1) != 0, (flags_idx & 2) != 0)
}

fn threat_kind_from_index(kind: u8) -> ThreatKind {
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

fn line_pattern_dirty_cells(mv: Move) -> Vec<usize> {
    let mut seen = [false; NUM_CELLS];
    let mut out = Vec::with_capacity(44);
    let row = (mv / BOARD_SIZE) as i32;
    let col = (mv % BOARD_SIZE) as i32;
    for &(dr, dc) in &DIR {
        for offset in -5i32..=5 {
            let r = row + dr * offset;
            let c = col + dc * offset;
            if r < 0 || r >= BOARD_SIZE as i32 || c < 0 || c >= BOARD_SIZE as i32 {
                continue;
            }
            let cell = (r as usize) * BOARD_SIZE + c as usize;
            if !seen[cell] {
                seen[cell] = true;
                out.push(cell);
            }
        }
    }
    out
}

fn make_vct_move(board: &mut Board, threat_index: &mut Option<VctThreatIndex>, mv: Move) {
    if let Some(index) = threat_index {
        let dirty = line_pattern_dirty_cells(mv);
        index.clear_cells(&dirty);
        board.make_move(mv);
        index.recompute_cells(board, &dirty);
    } else {
        board.make_move(mv);
    }
}

fn undo_vct_move(board: &mut Board, threat_index: &mut Option<VctThreatIndex>) {
    if let Some(index) = threat_index {
        if let Some(mv) = board.last_move {
            let dirty = line_pattern_dirty_cells(mv);
            index.clear_cells(&dirty);
            board.undo_move();
            index.recompute_cells(board, &dirty);
        } else {
            board.undo_move();
        }
    } else {
        board.undo_move();
    }
}

#[doc(hidden)]
pub fn vct_threat_index_transition_check_for_audit(
    moves: &[Move],
) -> Result<(usize, usize), String> {
    let mut board = Board::new();
    let mut threat_index = Some(VctThreatIndex::new(&board));
    let mut transitions = 0usize;

    for &mv in moves {
        if mv >= NUM_CELLS {
            return Err(format!("move out of board: {mv}"));
        }
        if !board.is_empty(mv) {
            return Err(format!("occupied move at ply {transitions}: {mv}"));
        }
        make_vct_move(&mut board, &mut threat_index, mv);
        transitions += 1;
        if !threat_index.as_ref().unwrap().matches_rebuild(&board) {
            return Err(format!(
                "index mismatch after make ply {transitions} move {mv}"
            ));
        }
    }

    let mut undos = 0usize;
    while board.last_move.is_some() {
        let ply_before = board.move_count;
        undo_vct_move(&mut board, &mut threat_index);
        undos += 1;
        if !threat_index.as_ref().unwrap().matches_rebuild(&board) {
            return Err(format!("index mismatch after undo from ply {ply_before}"));
        }
    }

    Ok((transitions, undos))
}

#[derive(Clone, Debug, Default)]
pub struct VctSearchStats {
    pub nodes: u64,
    pub deadline_hits: u64,
    pub node_budget_hits: u64,
}

impl VctSearchStats {
    fn enter_node(&mut self) {
        self.nodes += 1;
    }

    fn mark_deadline(&mut self) {
        self.deadline_hits += 1;
    }

    fn mark_node_budget(&mut self) {
        self.node_budget_hits += 1;
    }

    pub fn hit_deadline(&self) -> bool {
        self.deadline_hits > 0
    }

    pub fn hit_node_budget(&self) -> bool {
        self.node_budget_hits > 0
    }

    pub fn hit_stop(&self) -> bool {
        self.hit_deadline() || self.hit_node_budget()
    }
}

#[derive(Clone, Debug)]
pub struct VctSearchResult {
    pub sequence: Option<Vec<Move>>,
    pub stats: VctSearchStats,
}

impl VctSearchResult {
    pub fn termination_reason(&self) -> &'static str {
        if self.sequence.is_some() {
            "proved"
        } else if self.stats.hit_node_budget() {
            "node_budget"
        } else if self.stats.hit_deadline() {
            "deadline"
        } else {
            "exhausted"
        }
    }
}

impl Default for VctConfig {
    fn default() -> Self {
        Self {
            max_depth: 16,
            time_budget: Some(Duration::from_millis(500)),
            node_budget: None,
            enable_jump_three: false,
            enable_jump_three_attack_defense: false,
            enable_jump_three_counter: false,
            enable_jump_three_kind_scoped_defense: false,
            jump_attack_max_or_levels: u32::MAX,
            enable_gap_four: false,
            use_fast_classify: true,
            use_threat_index: false,
        }
    }
}

/// VCT 탐색 진입점.
///
/// 성공 시 최초 승리 수열(공격-수비-공격-... 순, 마지막 수는 공격 측 승리수)을
/// 반환. 실패 / 시간 초과 시 None.
pub fn search_vct(board: &mut Board, cfg: &VctConfig) -> Option<Vec<Move>> {
    search_vct_with_stats(board, cfg).sequence
}

pub fn search_vct_with_stats(board: &mut Board, cfg: &VctConfig) -> VctSearchResult {
    let deadline = cfg.time_budget.map(|d| Instant::now() + d);
    let attacker = board.side_to_move;
    let mut sequence = Vec::with_capacity(cfg.max_depth as usize * 2);
    let mut tt: TransTable = HashMap::with_capacity(65536);
    let mut stats = VctSearchStats::default();
    let flags = cfg.jump_three_flags();
    let mut threat_index = if flags.use_threat_index && flags.use_fast_classify {
        Some(VctThreatIndex::new(board))
    } else {
        None
    };
    let hit = vct_or(
        board,
        attacker,
        cfg.max_depth,
        0,
        deadline,
        cfg.node_budget,
        flags,
        &mut threat_index,
        &mut sequence,
        &mut tt,
        &mut stats,
    );
    VctSearchResult {
        sequence: if hit { Some(sequence) } else { None },
        stats,
    }
}

pub fn search_vct_audit_json(board: &mut Board, cfg: &VctConfig) -> Value {
    let deadline = cfg.time_budget.map(|d| Instant::now() + d);
    let attacker = board.side_to_move;
    let mut sequence = Vec::with_capacity(cfg.max_depth as usize * 2);
    let mut tt: TransTable = HashMap::with_capacity(65536);
    let mut audit = VctAuditLog::default();
    let mut stats = VctSearchStats::default();
    let flags = cfg.jump_three_flags();
    let mut threat_index = if flags.use_threat_index && flags.use_fast_classify {
        Some(VctThreatIndex::new(board))
    } else {
        None
    };
    let hit = vct_or_audit(
        board,
        attacker,
        cfg.max_depth,
        0,
        deadline,
        cfg.node_budget,
        flags,
        &mut threat_index,
        &mut sequence,
        &mut tt,
        &mut audit,
        &mut stats,
    );
    let result = VctSearchResult {
        sequence: if hit { Some(sequence.clone()) } else { None },
        stats,
    };
    json!({
        "format": "vct-proof-audit-v1",
        "hit": hit,
        "attacker": stone_json(attacker),
        "max_depth": cfg.max_depth,
        "time_budget_ms": cfg.time_budget.map(|d| d.as_millis() as u64),
        "node_budget": cfg.node_budget,
        "termination_reason": result.termination_reason(),
        "nodes": result.stats.nodes,
        "deadline_hits": result.stats.deadline_hits,
        "node_budget_hits": result.stats.node_budget_hits,
        "jump_three_attack_defense": flags.attack_defense,
        "jump_three_counter": flags.counter,
        "jump_three_kind_scoped_defense": flags.kind_scoped_defense,
        "jump_attack_max_or_levels": flags.attack_max_or_levels,
        "gap_four": flags.gap_four,
        "use_fast_classify": flags.use_fast_classify,
        "use_threat_index": flags.use_threat_index && flags.use_fast_classify,
        "sequence": if hit { Some(sequence.iter().map(|&mv| move_json(mv)).collect::<Vec<_>>()) } else { None },
        "and_nodes": audit.and_nodes,
        "terminal_event_count": audit.terminal_event_count,
        "terminal_event_counts": audit.terminal_event_counts,
        "terminal_event_samples": audit.terminal_event_samples,
        "tt_hit_count": audit.tt_hit_count,
        "tt_hit_events": audit.tt_hit_events,
    })
}

#[derive(Default)]
struct VctAuditLog {
    and_nodes: Vec<Value>,
    terminal_event_count: usize,
    terminal_event_counts: BTreeMap<String, usize>,
    terminal_event_samples: Vec<Value>,
    tt_hit_count: usize,
    tt_hit_events: Vec<Value>,
}

impl VctAuditLog {
    fn record_terminal(&mut self, kind: &str, event: Value) {
        self.terminal_event_count += 1;
        *self
            .terminal_event_counts
            .entry(kind.to_string())
            .or_default() += 1;
        if self.terminal_event_samples.len() < 64 {
            self.terminal_event_samples.push(event);
        }
    }
}

/// OR 노드 — 공격 측 턴. 공격 수 중 하나라도 승리로 이어지면 true.
fn vct_or(
    board: &mut Board,
    attacker: Stone,
    depth: u32,
    or_level: u32,
    deadline: Option<Instant>,
    node_budget: Option<u64>,
    jump_three: VctJumpThreeFlags,
    threat_index: &mut Option<VctThreatIndex>,
    sequence: &mut Vec<Move>,
    tt: &mut TransTable,
    stats: &mut VctSearchStats,
) -> bool {
    stats.enter_node();
    if node_budget_exceeded(node_budget, stats) {
        stats.mark_node_budget();
        return false;
    }
    if depth == 0 {
        return false;
    }
    if timed_out(deadline) {
        stats.mark_deadline();
        return false;
    }
    debug_assert_eq!(board.side_to_move, attacker);

    // TT 조회 — 같은 깊이 이상으로 탐색된 결과가 있으면 재사용.
    let hash = zobrist_hash(board);
    if let Some(entry) = tt.get(&hash) {
        if entry.depth >= depth {
            // RQ550: positive proof reuse can hide an unverified defender
            // branch. Keep only negative cutoffs and re-search wins.
            if matches!(entry.result, TtResult::Fails) {
                return false;
            }
        }
    }

    let (my, opp) = bb_pair(board, attacker);
    let rule_set = board.effective_rule_set();
    let opp_has_immediate_five = if let Some(index) = threat_index.as_ref() {
        index.has_kind(attacker.opponent(), false, false, ThreatKind::Five)
    } else {
        has_immediate_five(opp, my, attacker.opponent(), rule_set)
    };

    let enable_jump_three_attack =
        jump_three.attack_defense && or_level < jump_three.attack_max_or_levels;
    let attack_moves = if let Some(index) = threat_index.as_ref() {
        index.attack_moves(attacker, enable_jump_three_attack, jump_three.gap_four)
    } else {
        gather_attack_moves(
            board,
            my,
            opp,
            attacker,
            rule_set,
            enable_jump_three_attack,
            jump_three.gap_four,
            jump_three.use_fast_classify,
        )
    };
    if attack_moves.is_empty() {
        tt.insert(
            hash,
            TtEntry {
                depth,
                result: TtResult::Fails,
            },
        );
        return false;
    }

    for (mv, kind) in attack_moves {
        if is_vct_terminal_win(kind) {
            if opp_has_immediate_five && kind != ThreatKind::Five {
                continue;
            }
            sequence.push(mv);
            tt.insert(
                hash,
                TtEntry {
                    depth,
                    result: TtResult::AttackerWins,
                },
            );
            return true;
        }
        if opp_has_immediate_five {
            continue;
        }
        sequence.push(mv);
        make_vct_move(board, threat_index, mv);
        let won = vct_and(
            board,
            attacker,
            kind,
            depth - 1,
            or_level,
            deadline,
            node_budget,
            jump_three,
            threat_index,
            sequence,
            tt,
            stats,
        );
        undo_vct_move(board, threat_index);
        if stats.hit_stop() {
            sequence.pop();
            return false;
        }
        if won {
            tt.insert(
                hash,
                TtEntry {
                    depth,
                    result: TtResult::AttackerWins,
                },
            );
            return true;
        }
        sequence.pop();
    }
    tt.insert(
        hash,
        TtEntry {
            depth,
            result: TtResult::Fails,
        },
    );
    false
}

/// AND 노드 — 수비 측 턴. 모든 방어 수에 대해 공격이 여전히 이길 수 있어야
/// true. 하나라도 공격 실패를 만들면 false.
///
/// 성공 시 sequence에는 "마지막으로 검사한 방어 분기의 (수비 수 + 이후 공격
/// 수열)"이 남는다. 모든 분기가 성공해야 AND 성공이므로, 어느 분기를 대표로
/// 남겨도 재생 가능한 수열이 된다.
fn vct_and(
    board: &mut Board,
    attacker: Stone,
    attack_kind: ThreatKind,
    depth: u32,
    or_level: u32,
    deadline: Option<Instant>,
    node_budget: Option<u64>,
    jump_three: VctJumpThreeFlags,
    threat_index: &mut Option<VctThreatIndex>,
    sequence: &mut Vec<Move>,
    tt: &mut TransTable,
    stats: &mut VctSearchStats,
) -> bool {
    stats.enter_node();
    if node_budget_exceeded(node_budget, stats) {
        stats.mark_node_budget();
        return false;
    }
    if depth == 0 {
        return false;
    }
    if timed_out(deadline) {
        stats.mark_deadline();
        return false;
    }
    debug_assert_ne!(board.side_to_move, attacker);

    // 수비 측이 자기 턴에 즉시 5목을 완성할 수 있으면 공격 VCT는 실패.
    let (def_my, def_opp) = bb_pair(board, board.side_to_move);
    let rule_set = board.effective_rule_set();
    let defender_has_immediate_five = if let Some(index) = threat_index.as_ref() {
        index.has_kind(board.side_to_move, false, false, ThreatKind::Five)
    } else {
        has_immediate_five(def_my, def_opp, board.side_to_move, rule_set)
    };
    if defender_has_immediate_five {
        return false;
    }

    // 방어 후보: 공격자 직전 수 주변 좁힘 + 수비 측의 카운터 공격 수.
    // (좁힘만 쓰면 원거리 반격수가 누락되어 AND가 false positive를 냄 — VCT
    //  승리 오판. 수비 측이 **자기 winning threat**을 만들 수 있는 수는 반드시
    //  포함해야 함.)
    let defenses = match board.last_move {
        Some(attack_mv) => find_defenses_with_counters(
            board,
            attack_mv,
            attack_kind,
            jump_three,
            threat_index.as_ref(),
        ),
        None => board.candidate_moves(),
    };
    if defenses.is_empty() {
        return false;
    }

    let checkpoint = sequence.len();
    for mv in defenses {
        // 각 분기 시작 시 이전 분기의 흔적 제거.
        sequence.truncate(checkpoint);
        sequence.push(mv);

        make_vct_move(board, threat_index, mv);
        let attacker_still_wins = vct_or(
            board,
            attacker,
            depth - 1,
            or_level + 1,
            deadline,
            node_budget,
            jump_three,
            threat_index,
            sequence,
            tt,
            stats,
        );
        undo_vct_move(board, threat_index);
        if stats.hit_stop() {
            sequence.truncate(checkpoint);
            return false;
        }

        if !attacker_still_wins {
            // 이 방어로 공격 실패 → AND 실패. 수열 복원.
            sequence.truncate(checkpoint);
            return false;
        }
        // 성공 → 다음 분기로. 마지막 분기의 수열이 최종 sequence가 됨.
    }
    true
}

fn vct_or_audit(
    board: &mut Board,
    attacker: Stone,
    depth: u32,
    or_level: u32,
    deadline: Option<Instant>,
    node_budget: Option<u64>,
    jump_three: VctJumpThreeFlags,
    threat_index: &mut Option<VctThreatIndex>,
    sequence: &mut Vec<Move>,
    tt: &mut TransTable,
    audit: &mut VctAuditLog,
    stats: &mut VctSearchStats,
) -> bool {
    stats.enter_node();
    if node_budget_exceeded(node_budget, stats) {
        stats.mark_node_budget();
        return false;
    }
    if depth == 0 {
        return false;
    }
    if timed_out(deadline) {
        stats.mark_deadline();
        return false;
    }
    debug_assert_eq!(board.side_to_move, attacker);

    let hash = zobrist_hash(board);
    if let Some(entry) = tt.get(&hash) {
        if entry.depth >= depth {
            audit.tt_hit_count += 1;
            audit.tt_hit_events.push(json!({
                "node": "or",
                "hash": hash,
                "requested_depth": depth,
                "entry_depth": entry.depth,
                "result": tt_result_json(entry.result),
                "side_to_move": stone_json(board.side_to_move),
                "history": history_json(board),
            }));
            if matches!(entry.result, TtResult::Fails) {
                return false;
            }
        }
    }

    let (my, opp) = bb_pair(board, attacker);
    let rule_set = board.effective_rule_set();
    let opp_has_immediate_five = if let Some(index) = threat_index.as_ref() {
        index.has_kind(attacker.opponent(), false, false, ThreatKind::Five)
    } else {
        has_immediate_five(opp, my, attacker.opponent(), rule_set)
    };

    let enable_jump_three_attack =
        jump_three.attack_defense && or_level < jump_three.attack_max_or_levels;
    let attack_moves = if let Some(index) = threat_index.as_ref() {
        index.attack_moves(attacker, enable_jump_three_attack, jump_three.gap_four)
    } else {
        gather_attack_moves(
            board,
            my,
            opp,
            attacker,
            rule_set,
            enable_jump_three_attack,
            jump_three.gap_four,
            jump_three.use_fast_classify,
        )
    };
    if attack_moves.is_empty() {
        tt.insert(
            hash,
            TtEntry {
                depth,
                result: TtResult::Fails,
            },
        );
        return false;
    }

    for (mv, kind) in attack_moves {
        if is_vct_terminal_win(kind) {
            if opp_has_immediate_five && kind != ThreatKind::Five {
                audit.record_terminal(
                    "winning_attack_skipped_opp_immediate_five",
                    json!({
                        "kind": "winning_attack_skipped_opp_immediate_five",
                        "depth": depth,
                        "move": move_json(mv),
                        "threat": threat_json(kind),
                        "history": history_json(board),
                    }),
                );
                continue;
            }
            sequence.push(mv);
            audit.record_terminal(
                "winning_attack_accepted",
                json!({
                    "kind": "winning_attack_accepted",
                    "depth": depth,
                    "move": move_json(mv),
                    "threat": threat_json(kind),
                    "history": history_json(board),
                    "opp_has_immediate_five": opp_has_immediate_five,
                }),
            );
            tt.insert(
                hash,
                TtEntry {
                    depth,
                    result: TtResult::AttackerWins,
                },
            );
            return true;
        }
        if opp_has_immediate_five {
            audit.record_terminal(
                "forcing_attack_skipped_opp_immediate_five",
                json!({
                    "kind": "forcing_attack_skipped_opp_immediate_five",
                    "depth": depth,
                    "move": move_json(mv),
                    "threat": threat_json(kind),
                    "history": history_json(board),
                }),
            );
            continue;
        }
        sequence.push(mv);
        make_vct_move(board, threat_index, mv);
        let won = vct_and_audit(
            board,
            attacker,
            kind,
            depth - 1,
            or_level,
            deadline,
            node_budget,
            jump_three,
            threat_index,
            sequence,
            tt,
            audit,
            stats,
        );
        undo_vct_move(board, threat_index);
        if stats.hit_stop() {
            sequence.pop();
            return false;
        }
        if won {
            tt.insert(
                hash,
                TtEntry {
                    depth,
                    result: TtResult::AttackerWins,
                },
            );
            return true;
        }
        sequence.pop();
    }
    tt.insert(
        hash,
        TtEntry {
            depth,
            result: TtResult::Fails,
        },
    );
    false
}

fn vct_and_audit(
    board: &mut Board,
    attacker: Stone,
    attack_kind: ThreatKind,
    depth: u32,
    or_level: u32,
    deadline: Option<Instant>,
    node_budget: Option<u64>,
    jump_three: VctJumpThreeFlags,
    threat_index: &mut Option<VctThreatIndex>,
    sequence: &mut Vec<Move>,
    tt: &mut TransTable,
    audit: &mut VctAuditLog,
    stats: &mut VctSearchStats,
) -> bool {
    stats.enter_node();
    if node_budget_exceeded(node_budget, stats) {
        stats.mark_node_budget();
        return false;
    }
    if depth == 0 {
        return false;
    }
    if timed_out(deadline) {
        stats.mark_deadline();
        return false;
    }
    debug_assert_ne!(board.side_to_move, attacker);

    let node_history = history_json(board);
    let last_attack = board.last_move.map(move_json);
    let last_attack_threat = threat_json(attack_kind);
    let defender = board.side_to_move;
    let (def_my, def_opp) = bb_pair(board, defender);
    let rule_set = board.effective_rule_set();
    let defender_has_immediate_five = if let Some(index) = threat_index.as_ref() {
        index.has_kind(defender, false, false, ThreatKind::Five)
    } else {
        has_immediate_five(def_my, def_opp, defender, rule_set)
    };
    if defender_has_immediate_five {
        audit.and_nodes.push(json!({
            "node": "and",
            "depth": depth,
            "or_level": or_level,
            "attacker": stone_json(attacker),
            "defender": stone_json(defender),
            "history": node_history,
            "last_attack": last_attack,
            "last_attack_threat": last_attack_threat,
            "defender_has_immediate_five": true,
            "defenses": [],
            "result": false,
            "terminal_reason": "defender_immediate_five",
        }));
        return false;
    }

    let defenses = match board.last_move {
        Some(attack_mv) => find_defenses_with_counters(
            board,
            attack_mv,
            attack_kind,
            jump_three,
            threat_index.as_ref(),
        ),
        None => board.candidate_moves(),
    };
    if defenses.is_empty() {
        audit.and_nodes.push(json!({
            "node": "and",
            "depth": depth,
            "or_level": or_level,
            "attacker": stone_json(attacker),
            "defender": stone_json(defender),
            "history": node_history,
            "last_attack": last_attack,
            "last_attack_threat": last_attack_threat,
            "defender_has_immediate_five": false,
            "defenses": [],
            "result": false,
            "terminal_reason": "no_defenses",
        }));
        return false;
    }

    let checkpoint = sequence.len();
    let mut defense_results = Vec::with_capacity(defenses.len());
    for mv in defenses {
        sequence.truncate(checkpoint);
        sequence.push(mv);

        let tt_before = audit.tt_hit_count;
        make_vct_move(board, threat_index, mv);
        let attacker_still_wins = vct_or_audit(
            board,
            attacker,
            depth - 1,
            or_level + 1,
            deadline,
            node_budget,
            jump_three,
            threat_index,
            sequence,
            tt,
            audit,
            stats,
        );
        undo_vct_move(board, threat_index);
        if stats.hit_stop() {
            sequence.truncate(checkpoint);
            return false;
        }
        let tt_after = audit.tt_hit_count;
        let continuation = sequence[checkpoint..]
            .iter()
            .map(|&mv| move_json(mv))
            .collect::<Vec<_>>();

        defense_results.push(json!({
            "move": move_json(mv),
            "attacker_still_wins": attacker_still_wins,
            "tt_hits_delta": tt_after - tt_before,
            "sequence_after_len": sequence.len(),
            "continuation": continuation,
        }));

        if !attacker_still_wins {
            sequence.truncate(checkpoint);
            audit.and_nodes.push(json!({
                "node": "and",
                "depth": depth,
                "or_level": or_level,
                "attacker": stone_json(attacker),
                "defender": stone_json(defender),
                "history": node_history,
                "last_attack": last_attack,
                "last_attack_threat": last_attack_threat,
                "defender_has_immediate_five": false,
                "defenses": defense_results,
                "result": false,
                "terminal_reason": "defense_refutes",
            }));
            return false;
        }
    }
    audit.and_nodes.push(json!({
        "node": "and",
        "depth": depth,
        "or_level": or_level,
        "attacker": stone_json(attacker),
        "defender": stone_json(defender),
        "history": node_history,
        "last_attack": last_attack,
        "last_attack_threat": last_attack_threat,
        "defender_has_immediate_five": false,
        "defenses": defense_results,
        "result": true,
    }));
    true
}

fn gather_attack_moves(
    board: &Board,
    my: &BitBoard,
    opp: &BitBoard,
    side: Stone,
    rule_set: RuleSet,
    enable_jump_three: bool,
    enable_gap_four: bool,
    use_fast_classify: bool,
) -> Vec<(Move, ThreatKind)> {
    let mut out = Vec::new();
    let cells = my.count_ones() + opp.count_ones();
    // 첫 수면 패스 (vct 의미 없음).
    if cells == 0 {
        return out;
    }
    for idx in 0..(BOARD_SIZE * BOARD_SIZE) {
        if my.get(idx) || opp.get(idx) {
            continue;
        }
        let kind = if use_fast_classify {
            classify_move_fast_with_flags(board, idx, side, enable_jump_three, enable_gap_four)
        } else {
            classify_move_rules_with_flags(
                my,
                opp,
                idx,
                side,
                rule_set,
                enable_jump_three,
                enable_gap_four,
            )
        };
        if kind.is_forcing() {
            out.push((idx, kind));
        }
    }
    // 승리 위협을 먼저 시도.
    out.sort_by_key(|(_, k)| threat_priority(*k));
    out
}

fn threat_priority(k: ThreatKind) -> i32 {
    match k {
        ThreatKind::Five => 0,
        ThreatKind::OpenFour => 1,
        ThreatKind::DoubleFour => 2,
        ThreatKind::FourThree => 3,
        ThreatKind::DoubleThree => 4,
        ThreatKind::ClosedFour => 5,
        ThreatKind::OpenThree => 6,
        ThreatKind::JumpThree => 7,
        ThreatKind::None => 100,
    }
}

fn has_immediate_five(my: &BitBoard, opp: &BitBoard, side: Stone, rule_set: RuleSet) -> bool {
    for idx in 0..(BOARD_SIZE * BOARD_SIZE) {
        if my.get(idx) || opp.get(idx) {
            continue;
        }
        if classify_move_rules(my, opp, idx, side, rule_set) == ThreatKind::Five {
            return true;
        }
    }
    false
}

#[inline]
fn in_board(r: i32, c: i32) -> bool {
    r >= 0 && r < BOARD_SIZE as i32 && c >= 0 && c < BOARD_SIZE as i32
}

/// 좁은 방어 후보 + 수비 측의 winning counter moves (어디서든).
///
/// find_defenses만 쓰면 원거리 카운터 공격이 누락돼 AND가 false positive를
/// 내는 치명적 문제가 있어, 이 래퍼를 통해 "수비 측 관점에서 winning threat을
/// 만드는 모든 수"를 추가 포함한다. 비용 추가: 225 셀 classify_move 1회.
fn find_defenses_with_counters(
    board: &Board,
    attack_move: Move,
    attack_kind: ThreatKind,
    jump_three: VctJumpThreeFlags,
    threat_index: Option<&VctThreatIndex>,
) -> Vec<Move> {
    let direct_jump_three_defense = jump_three.attack_defense
        && (!jump_three.kind_scoped_defense || attack_kind == ThreatKind::JumpThree);
    let mut defenses = find_defenses(
        board,
        attack_move,
        direct_jump_three_defense,
        jump_three.gap_four,
    );
    let mut seen = BitBoard::EMPTY;
    for &d in &defenses {
        seen.set(d);
    }
    // 수비자(현재 side_to_move) 관점에서 자기 winning threat 만드는 수들.
    let (def_my, def_opp) = bb_pair(board, board.side_to_move);
    let rule_set = board.effective_rule_set();
    if let Some(index) = threat_index {
        for idx in index.forcing_moves_in_cell_order(
            board.side_to_move,
            jump_three.counter,
            jump_three.gap_four,
        ) {
            if !seen.get(idx) {
                seen.set(idx);
                defenses.push(idx);
            }
        }
        return defenses;
    }
    for idx in 0..NUM_CELLS {
        if def_my.get(idx) || def_opp.get(idx) || seen.get(idx) {
            continue;
        }
        let kind = if jump_three.use_fast_classify {
            classify_move_fast_with_flags(
                board,
                idx,
                board.side_to_move,
                jump_three.counter,
                jump_three.gap_four,
            )
        } else {
            classify_move_rules_with_flags(
                def_my,
                def_opp,
                idx,
                board.side_to_move,
                rule_set,
                jump_three.counter,
                jump_three.gap_four,
            )
        };
        // Winning 위협뿐 아니라 Forcing(ClosedFour/OpenThree) 반격도 포함해야
        // 원거리 카운터 공격을 AND가 놓치지 않음.
        if kind.is_forcing() {
            seen.set(idx);
            defenses.push(idx);
        }
    }
    defenses
}

/// 좁은 AND 노드 방어 후보 생성 — 공격자의 직전 수 주변만.
///
/// 반환: 주변 거리 ≤2 빈칸 + 4방향 라인 연장 ±3, ±4 빈칸. 실제로 공격
/// 위협을 막을 수 있는 모든 수를 포함하도록 의도된 conservative 범위.
/// 기존 candidate_moves(40~60개) 대비 보통 5~20개로 축소되어 AND 노드 브랜칭
/// 팩터 대폭 감소.
fn find_defenses(
    board: &Board,
    attack_move: Move,
    enable_jump_three: bool,
    enable_gap_four: bool,
) -> Vec<Move> {
    let row = (attack_move / BOARD_SIZE) as i32;
    let col = (attack_move % BOARD_SIZE) as i32;
    let mut seen = BitBoard::EMPTY;
    let mut out = Vec::with_capacity(24);

    // 1. 주변 체비셰프 거리 ≤2 빈칸.
    for dr in -2..=2 {
        for dc in -2..=2 {
            if dr == 0 && dc == 0 {
                continue;
            }
            let nr = row + dr;
            let nc = col + dc;
            if !in_board(nr, nc) {
                continue;
            }
            let idx = (nr as usize) * BOARD_SIZE + (nc as usize);
            if board.is_empty(idx) && !seen.get(idx) {
                seen.set(idx);
                out.push(idx);
            }
        }
    }

    // 2. 4방향 라인 원거리 차단 지점 (±3, ±4 step).
    // 열린3을 막는 데 필요한 양 끝 + 1칸 더(깊은 방어).
    for &(dr, dc) in &DIR {
        for step in [-4i32, -3, 3, 4] {
            let nr = row + dr * step;
            let nc = col + dc * step;
            if !in_board(nr, nc) {
                continue;
            }
            let idx = (nr as usize) * BOARD_SIZE + (nc as usize);
            if board.is_empty(idx) && !seen.get(idx) {
                seen.set(idx);
                out.push(idx);
            }
        }
    }

    if enable_jump_three {
        append_jump_three_defenses(board, attack_move, &mut seen, &mut out);
    }
    if enable_gap_four {
        append_gap_four_defenses(board, attack_move, &mut seen, &mut out);
    }

    out
}

fn append_jump_three_defenses(
    board: &Board,
    attack_move: Move,
    seen: &mut BitBoard,
    out: &mut Vec<Move>,
) {
    let row = (attack_move / BOARD_SIZE) as i32;
    let col = (attack_move % BOARD_SIZE) as i32;
    let attacker = board.side_to_move.opponent();
    let (my, opp) = bb_pair(board, attacker);

    for &(dr, dc) in &DIR {
        let w = read_window(my, opp, row, col, dr, dc);
        for start in 0..=5 {
            if !(start <= 5 && 5 < start + 6) {
                continue;
            }
            let s = &w[start..start + 6];
            if s != [0, 1, 0, 1, 1, 0] && s != [0, 1, 1, 0, 1, 0] {
                continue;
            }
            for i in 0..6 {
                if s[i] != 0 {
                    continue;
                }
                let off = (start + i) as i32 - 5;
                let nr = row + dr * off;
                let nc = col + dc * off;
                if !in_board(nr, nc) {
                    continue;
                }
                let idx = (nr as usize) * BOARD_SIZE + nc as usize;
                if board.is_empty(idx) && !seen.get(idx) {
                    seen.set(idx);
                    out.push(idx);
                }
            }
        }
    }
}

fn append_gap_four_defenses(
    board: &Board,
    attack_move: Move,
    seen: &mut BitBoard,
    out: &mut Vec<Move>,
) {
    let row = (attack_move / BOARD_SIZE) as i32;
    let col = (attack_move % BOARD_SIZE) as i32;
    let attacker = board.side_to_move.opponent();
    let (my, opp) = bb_pair(board, attacker);

    for &(dr, dc) in &DIR {
        let w = read_window(my, opp, row, col, dr, dc);
        visit_gap_four_gaps(&w, |off| {
            let nr = row + dr * off;
            let nc = col + dc * off;
            if !in_board(nr, nc) {
                return;
            }
            let idx = (nr as usize) * BOARD_SIZE + nc as usize;
            if board.is_empty(idx) && !seen.get(idx) {
                seen.set(idx);
                out.push(idx);
            }
        });
    }
}

fn bb_pair(board: &Board, side: Stone) -> (&BitBoard, &BitBoard) {
    match side {
        Stone::Black => (&board.black, &board.white),
        Stone::White => (&board.white, &board.black),
    }
}

fn node_budget_exceeded(node_budget: Option<u64>, stats: &VctSearchStats) -> bool {
    node_budget.is_some_and(|limit| stats.nodes > limit)
}

fn timed_out(deadline: Option<Instant>) -> bool {
    if let Some(d) = deadline {
        if Instant::now() >= d {
            return true;
        }
    }
    false
}

fn move_json(mv: Move) -> Value {
    json!({"x": mv % BOARD_SIZE, "y": mv / BOARD_SIZE})
}

fn history_json(board: &Board) -> Value {
    let mut side = Stone::Black;
    let moves = board
        .history
        .iter()
        .map(|&mv| {
            let out = json!({
                "x": mv % BOARD_SIZE,
                "y": mv / BOARD_SIZE,
                "color": stone_json(side),
            });
            side = side.opponent();
            out
        })
        .collect::<Vec<_>>();
    json!(moves)
}

fn stone_json(side: Stone) -> &'static str {
    match side {
        Stone::Black => "B",
        Stone::White => "W",
    }
}

fn threat_json(kind: ThreatKind) -> &'static str {
    match kind {
        ThreatKind::None => "None",
        ThreatKind::ClosedFour => "ClosedFour",
        ThreatKind::OpenThree => "OpenThree",
        ThreatKind::Five => "Five",
        ThreatKind::OpenFour => "OpenFour",
        ThreatKind::DoubleFour => "DoubleFour",
        ThreatKind::FourThree => "FourThree",
        ThreatKind::DoubleThree => "DoubleThree",
        ThreatKind::JumpThree => "JumpThree",
    }
}

fn tt_result_json(result: TtResult) -> &'static str {
    match result {
        TtResult::AttackerWins => "attacker_wins",
        TtResult::Fails => "fails",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::to_idx;
    use noru::trainer::SimpleRng;

    /// `classify_move_fast` (Pattern4-backed) and `classify_move` (scan_line
    /// based) must return the identical `ThreatKind` for every empty cell of
    /// every random position. 1.5K positions × ~200 candidates ≈ 300K
    /// comparisons; a single disagreement fails the test.
    #[test]
    fn pattern4_fast_classify_matches_baseline() {
        let mut rng = SimpleRng::new(0xCAFE_BABE);
        // generate 1500 positions spanning a range of game lengths
        for trial in 0..1500 {
            let mut board = Board::new();
            // play 6 to 50 random plies (random legal cell each side; stop on
            // any terminal result so we never query a finished game)
            let ply_target = 6 + rng.next_usize(45);
            for _ in 0..ply_target {
                if !matches!(board.game_result(), crate::board::GameResult::Ongoing) {
                    break;
                }
                let candidates = board.candidate_moves();
                if candidates.is_empty() {
                    break;
                }
                let idx = rng.next_usize(candidates.len());
                board.make_move(candidates[idx]);
            }
            if !matches!(board.game_result(), crate::board::GameResult::Ongoing) {
                continue;
            }

            let side = board.side_to_move;
            // Cover both the Freestyle (exact5=false) and Standard
            // (exact5=true) classifications — the fast path reads
            // `board.exact5`, so the slow baseline must agree under both.
            for &exact5 in &[false, true] {
                board.exact5 = exact5;
                let (my, opp) = match side {
                    Stone::Black => (&board.black, &board.white),
                    Stone::White => (&board.white, &board.black),
                };
                for cell in 0..NUM_CELLS {
                    if board.black.get(cell) || board.white.get(cell) {
                        continue;
                    }
                    let baseline = classify_move(my, opp, cell, exact5);
                    let fast = classify_move_fast(&board, cell, side);
                    assert_eq!(
                        baseline, fast,
                        "mismatch at trial {trial} cell {cell} side {side:?} exact5 {exact5}"
                    );
                }
            }
        }
    }

    /// mv 위치에 돌을 놓으면 Five가 완성되는가 — 열린4 상태에서 검증.
    #[test]
    fn test_classify_move_five() {
        let mut board = Board::new();
        // 흑: (7,3) (7,4) (7,5) (7,6) — 열린 4
        board.make_move(to_idx(7, 3));
        board.make_move(to_idx(0, 0));
        board.make_move(to_idx(7, 4));
        board.make_move(to_idx(0, 14));
        board.make_move(to_idx(7, 5));
        board.make_move(to_idx(14, 0));
        board.make_move(to_idx(7, 6));
        // 현재 백 턴이지만 흑 비트보드에 대해 (7,2)나 (7,7)이 Five인지 확인.
        let k1 = classify_move(&board.black, &board.white, to_idx(7, 2), false);
        let k2 = classify_move(&board.black, &board.white, to_idx(7, 7), false);
        assert_eq!(k1, ThreatKind::Five, "(7,2) should complete Five");
        assert_eq!(k2, ThreatKind::Five, "(7,7) should complete Five");
    }

    #[test]
    fn test_classify_move_open_four() {
        let mut board = Board::new();
        // 흑: (7,4) (7,5) (7,6) — 연속 열린 3 상태.
        board.make_move(to_idx(7, 4));
        board.make_move(to_idx(0, 0));
        board.make_move(to_idx(7, 5));
        board.make_move(to_idx(0, 14));
        board.make_move(to_idx(7, 6));
        // (7,3) 또는 (7,7)에 두면 열린 4.
        let k = classify_move(&board.black, &board.white, to_idx(7, 7), false);
        assert_eq!(k, ThreatKind::OpenFour);
        assert_ne!(
            k,
            ThreatKind::Five,
            "open four must not be classified as Five"
        );
    }

    #[test]
    fn test_classify_move_jump_three() {
        let mut board = Board::new();
        // Black has .X.XX. after playing (7,3):
        // empty (7,2), X (7,3), gap (7,4), X (7,5), X (7,6), empty (7,7).
        board.make_move(to_idx(7, 5));
        board.make_move(to_idx(0, 0));
        board.make_move(to_idx(7, 6));
        board.make_move(to_idx(0, 14));

        let mv = to_idx(7, 3);
        let slow = classify_move_rules_with_jump_three(
            &board.black,
            &board.white,
            mv,
            Stone::Black,
            board.effective_rule_set(),
            true,
        );
        let fast = classify_move_fast_with_jump_three(&board, mv, Stone::Black, true);
        assert_eq!(slow, ThreatKind::JumpThree);
        assert_eq!(fast, ThreatKind::JumpThree);
        assert_eq!(
            classify_move(&board.black, &board.white, mv, false),
            ThreatKind::None
        );
        assert_eq!(
            classify_move_fast(&board, mv, Stone::Black),
            ThreatKind::None
        );
        assert!(ThreatKind::JumpThree.is_forcing());
        assert!(!ThreatKind::JumpThree.is_winning());
        assert!(!is_vct_terminal_win(ThreatKind::JumpThree));
    }

    #[test]
    fn rq560_jump_three_flag_on_off_preserves_legacy_classification() {
        let mut left_gap = Board::new();
        left_gap.make_move(to_idx(7, 5));
        left_gap.make_move(to_idx(0, 0));
        left_gap.make_move(to_idx(7, 6));
        left_gap.make_move(to_idx(0, 14));
        let left_mv = to_idx(7, 3);
        assert_eq!(
            classify_move_rules_with_jump_three(
                &left_gap.black,
                &left_gap.white,
                left_mv,
                Stone::Black,
                left_gap.effective_rule_set(),
                false,
            ),
            ThreatKind::None
        );
        assert_eq!(
            classify_move_rules_with_jump_three(
                &left_gap.black,
                &left_gap.white,
                left_mv,
                Stone::Black,
                left_gap.effective_rule_set(),
                true,
            ),
            ThreatKind::JumpThree
        );
        assert_eq!(
            classify_move_fast_with_jump_three(&left_gap, left_mv, Stone::Black, false),
            ThreatKind::None
        );
        assert_eq!(
            classify_move_fast_with_jump_three(&left_gap, left_mv, Stone::Black, true),
            ThreatKind::JumpThree
        );

        let mut right_gap = Board::new();
        right_gap.make_move(to_idx(7, 3));
        right_gap.make_move(to_idx(0, 0));
        right_gap.make_move(to_idx(7, 4));
        right_gap.make_move(to_idx(0, 14));
        let right_mv = to_idx(7, 6);
        assert_eq!(
            classify_move_rules_with_jump_three(
                &right_gap.black,
                &right_gap.white,
                right_mv,
                Stone::Black,
                right_gap.effective_rule_set(),
                false,
            ),
            ThreatKind::None
        );
        assert_eq!(
            classify_move_rules_with_jump_three(
                &right_gap.black,
                &right_gap.white,
                right_mv,
                Stone::Black,
                right_gap.effective_rule_set(),
                true,
            ),
            ThreatKind::JumpThree
        );
    }

    fn board_with_black_stones(stones: &[(usize, usize)]) -> Board {
        let fillers = [(0, 0), (0, 14), (14, 0), (14, 14), (1, 0), (1, 14)];
        let mut board = Board::new();
        for (i, &(row, col)) in stones.iter().enumerate() {
            board.make_move(to_idx(row, col));
            board.make_move(to_idx(fillers[i].0, fillers[i].1));
        }
        board
    }

    #[test]
    fn rq567_gap_four_flag_on_off_preserves_legacy_classification() {
        let cases = [
            // X.XXX: move is the left stone, gap at +1.
            (&[(7, 5), (7, 6), (7, 7)][..], to_idx(7, 3)),
            // XX.XX: move is the second stone, gap at +1.
            (&[(7, 3), (7, 6), (7, 7)][..], to_idx(7, 4)),
            // XXX.X: move is the third stone, gap at +1.
            (&[(7, 3), (7, 4), (7, 7)][..], to_idx(7, 5)),
        ];
        for (stones, mv) in cases {
            let board = board_with_black_stones(stones);
            let legacy = classify_move(&board.black, &board.white, mv, false);
            assert_eq!(
                classify_move_rules_with_flags(
                    &board.black,
                    &board.white,
                    mv,
                    Stone::Black,
                    board.effective_rule_set(),
                    false,
                    false,
                ),
                legacy,
                "gap-four flag off must preserve legacy classification"
            );
            assert_ne!(legacy, ThreatKind::ClosedFour);
            assert_eq!(
                classify_move_rules_with_flags(
                    &board.black,
                    &board.white,
                    mv,
                    Stone::Black,
                    board.effective_rule_set(),
                    false,
                    true,
                ),
                ThreatKind::ClosedFour
            );
            assert_eq!(
                classify_move_fast_with_flags(&board, mv, Stone::Black, false, true),
                ThreatKind::ClosedFour
            );
        }
        assert!(!ThreatKind::ClosedFour.is_winning());
        assert!(!is_vct_terminal_win(ThreatKind::ClosedFour));
    }

    #[test]
    fn test_jump_three_is_attack_candidate() {
        let mut board = Board::new();
        board.make_move(to_idx(7, 5));
        board.make_move(to_idx(0, 0));
        board.make_move(to_idx(7, 6));
        board.make_move(to_idx(0, 14));

        let moves = gather_attack_moves(
            &board,
            &board.black,
            &board.white,
            Stone::Black,
            board.effective_rule_set(),
            true,
            false,
            false,
        );
        assert!(
            moves
                .iter()
                .any(|&(mv, kind)| mv == to_idx(7, 3) && kind == ThreatKind::JumpThree),
            "jump-three move must enter attack candidates: {:?}",
            moves
        );
    }

    #[test]
    fn rq560_g90_jump_three_attack_candidate() {
        let mut board = Board::new();
        for mv in [
            to_idx(7, 7),
            to_idx(8, 7),
            to_idx(5, 8),
            to_idx(10, 7),
            to_idx(5, 9),
            to_idx(6, 8),
            to_idx(5, 7),
            to_idx(5, 6),
            to_idx(4, 7),
            to_idx(6, 7),
            to_idx(6, 9),
            to_idx(7, 10),
        ] {
            board.make_move(mv);
        }

        let rapfi = to_idx(3, 9);
        assert_eq!(
            classify_move_fast_with_jump_three(&board, rapfi, Stone::Black, true),
            ThreatKind::JumpThree
        );
        assert_eq!(
            classify_move_fast(&board, rapfi, Stone::Black),
            ThreatKind::None
        );
        let moves = gather_attack_moves(
            &board,
            &board.black,
            &board.white,
            Stone::Black,
            board.effective_rule_set(),
            true,
            false,
            false,
        );
        assert!(
            moves
                .iter()
                .any(|&(mv, kind)| mv == rapfi && kind == ThreatKind::JumpThree),
            "g90 rapfi move (9,3) must be a JumpThree attack candidate: {:?}",
            moves
        );
    }

    #[test]
    fn rq567_g58_gap_four_attack_candidate() {
        let mut board = Board::new();
        for mv in [
            to_idx(7, 7),
            to_idx(6, 8),
            to_idx(8, 9),
            to_idx(11, 10),
            to_idx(8, 8),
            to_idx(8, 7),
            to_idx(9, 9),
            to_idx(10, 10),
            to_idx(7, 9),
            to_idx(10, 9),
            to_idx(9, 8),
            to_idx(9, 10),
            to_idx(8, 10),
            to_idx(10, 8),
        ] {
            board.make_move(mv);
        }

        let rapfi = to_idx(5, 9);
        assert_ne!(
            classify_move_fast(&board, rapfi, Stone::Black),
            ThreatKind::ClosedFour
        );
        assert_eq!(
            classify_move_fast_with_flags(&board, rapfi, Stone::Black, true, true),
            ThreatKind::ClosedFour
        );
        let moves = gather_attack_moves(
            &board,
            &board.black,
            &board.white,
            Stone::Black,
            board.effective_rule_set(),
            true,
            true,
            false,
        );
        assert!(
            moves
                .iter()
                .any(|&(mv, kind)| mv == rapfi && kind == ThreatKind::ClosedFour),
            "g58 rapfi move (9,5) must enter attack candidates as gap-four ClosedFour: {:?}",
            moves
        );

        board.make_move(rapfi);
        let defenses = find_defenses(&board, rapfi, false, true);
        assert!(
            defenses.contains(&to_idx(6, 9)),
            "gap-four defense set must include the gap cell (9,6); got {:?}",
            defenses
        );
    }

    #[test]
    fn test_jump_three_defenses_include_gap_and_completion_cells() {
        let mut board = Board::new();
        board.make_move(to_idx(7, 5));
        board.make_move(to_idx(0, 0));
        board.make_move(to_idx(7, 6));
        board.make_move(to_idx(0, 14));
        board.make_move(to_idx(7, 3));

        let defenses = find_defenses(&board, to_idx(7, 3), true, false);
        for expected in [to_idx(7, 2), to_idx(7, 4), to_idx(7, 7)] {
            assert!(
                defenses.contains(&expected),
                "jump-three defense set must include {:?}; got {:?}",
                expected,
                defenses
            );
        }
    }

    /// Standard 규칙(exact5)의 핵심: 6목(overline)을 만드는 수는 승리가 아니다.
    /// Gomocup 2026 Standard 리그 탈락의 직접 원인이었던 회귀를 가드한다.
    #[test]
    fn rq574_threat_index_make_undo_matches_rebuild() {
        let initial = Board::new();
        VctThreatIndex::new(&initial).assert_matches_rebuild(&initial);
        let moves = [
            to_idx(7, 7),
            to_idx(8, 7),
            to_idx(7, 6),
            to_idx(8, 6),
            to_idx(7, 5),
            to_idx(8, 5),
            to_idx(7, 4),
            to_idx(5, 5),
            to_idx(6, 6),
            to_idx(9, 5),
            to_idx(5, 7),
            to_idx(10, 4),
        ];
        let (makes, undos) = vct_threat_index_transition_check_for_audit(&moves).unwrap();
        assert_eq!(makes, moves.len());
        assert_eq!(undos, moves.len());
    }

    #[test]
    fn classify_move_exact5_overline_not_five() {
        let mut board = Board::new();
        // 흑 가로 (7,2)(7,3)(7,4)(7,5)(7,7). (7,6)에 두면 2~7 = 6목.
        for (b, w) in [
            ((7, 2), (0, 0)),
            ((7, 3), (0, 14)),
            ((7, 4), (14, 0)),
            ((7, 5), (14, 14)),
            ((7, 7), (3, 3)),
        ] {
            board.make_move(to_idx(b.0, b.1));
            board.make_move(to_idx(w.0, w.1));
        }
        let mv = to_idx(7, 6);
        // Freestyle: 6목도 5목 이상이므로 승리(Five).
        assert_eq!(
            classify_move(&board.black, &board.white, mv, false),
            ThreatKind::Five,
            "freestyle: overline still counts as a win"
        );
        // Standard: 6목(overline)은 승리도, 어떤 위협도 아님.
        assert_eq!(
            classify_move(&board.black, &board.white, mv, true),
            ThreatKind::None,
            "standard: overline is not a win"
        );
    }

    /// exact5에서도 '정확히 5목'은 여전히 승리여야 한다 (overline 처리가
    /// 정상적인 5목까지 죽이지 않는지 확인).
    #[test]
    fn classify_move_exact5_exact_five_still_wins() {
        let mut board = Board::new();
        // 흑 가로 (7,2)(7,3)(7,4)(7,5). (7,6)에 두면 2~6 = 정확히 5목.
        for (b, w) in [
            ((7, 2), (0, 0)),
            ((7, 3), (0, 14)),
            ((7, 4), (14, 0)),
            ((7, 5), (14, 14)),
        ] {
            board.make_move(to_idx(b.0, b.1));
            board.make_move(to_idx(w.0, w.1));
        }
        let mv = to_idx(7, 6);
        assert_eq!(
            classify_move(&board.black, &board.white, mv, true),
            ThreatKind::Five,
            "standard: exactly five is a win"
        );
        assert_eq!(
            classify_move(&board.black, &board.white, mv, false),
            ThreatKind::Five
        );
    }

    #[test]
    fn test_vct_open_four_mate_in_1() {
        let mut board = Board::new();
        // 흑이 이미 열린 4를 형성한 상태 → 흑 턴이면 5목 만들기만 하면 됨.
        board.make_move(to_idx(7, 3));
        board.make_move(to_idx(0, 0));
        board.make_move(to_idx(7, 4));
        board.make_move(to_idx(0, 14));
        board.make_move(to_idx(7, 5));
        board.make_move(to_idx(14, 0));
        board.make_move(to_idx(7, 6));
        // 지금 백 턴. 백 입장에서 VCT 돌리면 실패 (백은 공격 위협 없음).
        // 대신 한 수 진행해서 흑 턴 만들고 VCT.
        board.make_move(to_idx(14, 14));
        // 흑 턴: 열린 4 → (7,7) 또는 (7,2)로 Five → mate in 1.
        let cfg = VctConfig::default();
        let seq = search_vct(&mut board, &cfg);
        assert!(seq.is_some(), "should find mate");
        let seq = seq.unwrap();
        assert_eq!(seq.len(), 1, "mate in 1");
        assert!(
            [to_idx(7, 2), to_idx(7, 7)].contains(&seq[0]),
            "got {:?}",
            seq[0]
        );
    }

    #[test]
    fn test_classify_move_double_three() {
        let mut board = Board::new();
        // 흑: (7,4) (7,5)  세로: (5,6) (6,6)
        // (7,6)에 두면 가로 열린3 + 세로 열린3 = DoubleThree
        board.make_move(to_idx(7, 4));
        board.make_move(to_idx(0, 0));
        board.make_move(to_idx(7, 5));
        board.make_move(to_idx(0, 14));
        board.make_move(to_idx(5, 6));
        board.make_move(to_idx(14, 0));
        board.make_move(to_idx(6, 6));
        // 지금 백 턴. 흑 bitboard 기준으로 (7,6)이 DoubleThree인지.
        let k = classify_move(&board.black, &board.white, to_idx(7, 6), false);
        assert_eq!(
            k,
            ThreatKind::DoubleThree,
            "should be double three, got {:?}",
            k
        );
    }

    #[test]
    fn test_classify_move_four_three() {
        let mut board = Board::new();
        // 흑: 가로 (7,3) (7,4) (7,5) — 닫힌 3 or 열린 3
        //     세로 (5,6) (6,6) — 열린 2
        // (7,6) 두면: 가로 (7,3~6) 열린 4, 세로 (5~7,6) 열린 3 → FourThree
        board.make_move(to_idx(7, 3));
        board.make_move(to_idx(0, 0));
        board.make_move(to_idx(7, 4));
        board.make_move(to_idx(0, 14));
        board.make_move(to_idx(7, 5));
        board.make_move(to_idx(14, 0));
        board.make_move(to_idx(5, 6));
        board.make_move(to_idx(14, 14));
        board.make_move(to_idx(6, 6));
        // 현재 백 턴. 흑 bitboard에 (7,6) 놓으면 4-3.
        // 다만 가로 (7,3~6)은 열린 4가 돼서 OpenFour 판정이 먼저. FourThree가
        // 아니라 OpenFour가 나옴 — 이게 정상 (더 강한 Threat 우선).
        let k = classify_move(&board.black, &board.white, to_idx(7, 6), false);
        assert_eq!(k, ThreatKind::OpenFour, "open four dominates; got {:?}", k);
    }

    #[test]
    fn test_vct_double_three_mate_in_3() {
        let mut board = Board::new();
        // 흑 double-three를 한 수에 만들 수 있는 세팅:
        // 가로 (7,4)(7,5) + 세로 (5,6)(6,6). (7,6)에 두면 DoubleThree → 상대
        // 한쪽만 막을 수 있어 흑이 다음 턴 OpenFour → 그 다음 5목 = mate in 3.
        board.make_move(to_idx(7, 4));
        board.make_move(to_idx(0, 0));
        board.make_move(to_idx(7, 5));
        board.make_move(to_idx(0, 14));
        board.make_move(to_idx(5, 6));
        board.make_move(to_idx(14, 0));
        board.make_move(to_idx(6, 6));
        board.make_move(to_idx(14, 14));
        // 흑 턴. (7,6)이 DoubleThree — 승리 확정.
        let cfg = VctConfig::default();
        let seq = search_vct(&mut board, &cfg);
        assert!(seq.is_some(), "should find VCT mate");
        let seq = seq.unwrap();
        // 첫 수는 DoubleThree 만드는 수 중 하나(가장 강력한 (7,6)).
        assert!(seq.len() >= 1, "non-empty sequence");
        assert_eq!(seq[0], to_idx(7, 6), "first move must be (7,6) DoubleThree");
    }

    #[test]
    fn test_vct_no_winning_sequence() {
        let mut board = Board::new();
        // 흑 1수만 있고 위협 없는 상태.
        board.make_move(to_idx(7, 7));
        board.make_move(to_idx(6, 6));
        // 흑 턴. 강제 승리 수열 없음.
        let cfg = VctConfig {
            max_depth: 8,
            time_budget: Some(Duration::from_millis(100)),
            node_budget: None,
            enable_jump_three: false,
            enable_jump_three_attack_defense: false,
            enable_jump_three_counter: false,
            enable_jump_three_kind_scoped_defense: false,
            jump_attack_max_or_levels: u32::MAX,
            enable_gap_four: false,
            use_fast_classify: false,
            use_threat_index: false,
        };
        let seq = search_vct(&mut board, &cfg);
        assert!(seq.is_none(), "no VCT should exist, got {:?}", seq);
    }

    #[test]
    fn test_vct_loses_to_faster_counter_threat() {
        // 공격자가 OpenFour를 만들 수 있어도, 상대가 먼저 5목 완성 가능하면
        // VCT는 Five가 아닌 이상 실패해야 함 (상호위협 처리).
        let mut board = Board::new();
        // 백 4목 상태: (8,0)(8,1)(8,2)(8,3) — (8,4) 두면 5목.
        // 흑은 가로 (7,3)(7,4)(7,5)(7,6) — 열린 4 이미 형성.
        board.make_move(to_idx(7, 3));
        board.make_move(to_idx(8, 0));
        board.make_move(to_idx(7, 4));
        board.make_move(to_idx(8, 1));
        board.make_move(to_idx(7, 5));
        board.make_move(to_idx(8, 2));
        board.make_move(to_idx(7, 6));
        board.make_move(to_idx(8, 3));
        // 현재 흑 턴. 흑 (7,2) 또는 (7,7)로 Five 가능 — 이건 mate in 1로
        // 통과해야 함 (Five는 상대보다 빠름).
        let cfg = VctConfig::default();
        let seq = search_vct(&mut board, &cfg);
        assert!(seq.is_some(), "Five wins before opponent's 4");
        let seq = seq.unwrap();
        assert_eq!(seq.len(), 1);
        assert!([to_idx(7, 2), to_idx(7, 7)].contains(&seq[0]));
    }

    #[test]
    fn test_vct_mate_in_5_chain() {
        // 열린3 → 상대 방어 → OpenFour → 상대 방어 → Five 체인.
        // 흑 열린3 기준: (7,5)(7,6)(7,7) 가로 3목, 양 끝 빈.
        // 흑 턴에 먼저 열린3이 완성된 상태.
        let mut board = Board::new();
        board.make_move(to_idx(7, 5));
        board.make_move(to_idx(0, 0));
        board.make_move(to_idx(7, 6));
        board.make_move(to_idx(0, 14));
        board.make_move(to_idx(7, 7));
        board.make_move(to_idx(0, 7));
        // 흑 턴. 흑 열린3: (7,5~7). 흑이 (7,4) 또는 (7,8)로 열린4 → mate in 3.
        // 즉 이건 실제로는 mate in 3 체인 (열린3 이미 형성, 다음 수로 열린4).
        let cfg = VctConfig {
            max_depth: 8,
            time_budget: Some(Duration::from_millis(300)),
            node_budget: None,
            enable_jump_three: false,
            enable_jump_three_attack_defense: false,
            enable_jump_three_counter: false,
            enable_jump_three_kind_scoped_defense: false,
            jump_attack_max_or_levels: u32::MAX,
            enable_gap_four: false,
            use_fast_classify: false,
            use_threat_index: false,
        };
        let seq = search_vct(&mut board, &cfg);
        assert!(seq.is_some(), "should find mate via open-three chain");
    }

    #[test]
    fn test_vct_tt_consistency() {
        // 같은 포지션에 두 번 탐색 — 두 번 다 같은 결과 (TT hit 체크).
        let mut board = Board::new();
        board.make_move(to_idx(7, 5));
        board.make_move(to_idx(0, 0));
        board.make_move(to_idx(7, 6));
        board.make_move(to_idx(0, 14));
        board.make_move(to_idx(7, 7));
        board.make_move(to_idx(0, 7));
        let cfg = VctConfig {
            max_depth: 8,
            time_budget: Some(Duration::from_millis(500)),
            node_budget: None,
            enable_jump_three: false,
            enable_jump_three_attack_defense: false,
            enable_jump_three_counter: false,
            enable_jump_three_kind_scoped_defense: false,
            jump_attack_max_or_levels: u32::MAX,
            enable_gap_four: false,
            use_fast_classify: false,
            use_threat_index: false,
        };
        let s1 = search_vct(&mut board, &cfg);
        let s2 = search_vct(&mut board, &cfg);
        assert_eq!(s1.is_some(), s2.is_some(), "VCT should be deterministic");
    }

    // NOTE: 좁힘 regression을 유닛 테스트로 깔끔히 재현하는 건 포지션 구성이
    // 까다로워서 아레나 회귀 시험으로 대체. 실전에서 VCT가 대붕괴(1~10% 승률)
    // 나오면 find_defenses_with_counters 검토 재개.

    #[test]
    fn test_vct_cannot_ignore_opponent_five_threat_for_forcing() {
        // 공격자가 열린 3만 만들 수 있는데 상대가 즉시 5목 가능하면 VCT 실패.
        let mut board = Board::new();
        // 백 4목: (8,0..3). 흑은 가로 (7,4)(7,5) 열린 2만 있음.
        board.make_move(to_idx(7, 4));
        board.make_move(to_idx(8, 0));
        board.make_move(to_idx(7, 5));
        board.make_move(to_idx(8, 1));
        board.make_move(to_idx(0, 0));
        board.make_move(to_idx(8, 2));
        board.make_move(to_idx(0, 14));
        board.make_move(to_idx(8, 3));
        // 흑 턴. 흑은 (7,6) 또는 (7,3)으로 열린 3 가능 but 백 (8,4) 5목 먼저.
        // VCT는 None이어야 함.
        let cfg = VctConfig::default();
        let seq = search_vct(&mut board, &cfg);
        assert!(seq.is_none(), "no VCT when opponent has immediate Five");
    }
}
