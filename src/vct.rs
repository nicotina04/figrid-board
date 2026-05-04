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

use crate::board::{Board, BitBoard, Move, Stone, BOARD_SIZE, NUM_CELLS};
use crate::heuristic::{scan_line, DIR};
use noru::trainer::SimpleRng;
use std::collections::HashMap;
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
    OpenTwo,      // (2, open 2)
    ClosedThree,  // (3, open 1)
    OpenThree,    // (3, open 2)
    ClosedFour,   // (4, open 1)
    OpenFour,     // (4, open 2)
    Five,         // (>=5)
}

fn classify_line(count: u32, open_ends: u32) -> LineThreat {
    match (count, open_ends) {
        (5.., _) => LineThreat::Five,
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
    None        = 0,
    ClosedFour  = 1,
    OpenThree   = 2,
    Five        = 3,
    OpenFour    = 4,
    DoubleFour  = 5,
    FourThree   = 6,
    DoubleThree = 7,
}

/// `ThreatKind` discriminant의 수 — 테이블 크기 상수.
pub const THREAT_KIND_COUNT: usize = 8;

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
        matches!(self, ThreatKind::ClosedFour | ThreatKind::OpenThree)
            || self.is_winning()
    }
}

/// side 쪽이 mv 좌표에 돌을 두면 어떤 Threat이 생기는지 분석.
///
/// my_bb는 side의 돌 bitboard, opp_bb는 상대 bitboard. **mv 위치에는 아직 돌이
/// 없는 상태**로 가정. 내부적으로 mv가 놓였을 때의 4방향 라인을 시뮬레이션.
pub fn classify_move(my_bb: &BitBoard, opp_bb: &BitBoard, mv: Move) -> ThreatKind {
    let row = (mv / BOARD_SIZE) as i32;
    let col = (mv % BOARD_SIZE) as i32;

    // mv 위치에 돌이 있다고 가정하고 각 방향 라인 측정.
    // scan_line은 시작 셀이 돌을 가졌다고 가정하고 count=1부터 시작하므로
    // 임시로 my_bb에 mv를 set한 복사본을 만든다.
    let mut my_tmp = my_bb.clone();
    my_tmp.set(mv);

    let mut fours = 0u32; // OpenFour 또는 ClosedFour 방향 수
    let mut open_fours = 0u32;
    let mut open_threes = 0u32;
    let mut closed_fours = 0u32;
    let mut fives = 0u32;

    for &(dr, dc) in &DIR {
        let info = scan_line(&my_tmp, opp_bb, row, col, dr, dc);
        let open_ends = info.open_front as u32 + info.open_back as u32;
        match classify_line(info.count, open_ends) {
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
    ThreatKind::None
}

/// VCT 설정.
pub struct VctConfig {
    /// 최대 재귀 깊이 (공격-수비 쌍 수). 너무 크면 폭발.
    pub max_depth: u32,
    /// 전체 시간 예산. 초과 시 None 반환.
    pub time_budget: Option<Duration>,
}

impl Default for VctConfig {
    fn default() -> Self {
        Self {
            max_depth: 16,
            time_budget: Some(Duration::from_millis(500)),
        }
    }
}

/// VCT 탐색 진입점.
///
/// 성공 시 최초 승리 수열(공격-수비-공격-... 순, 마지막 수는 공격 측 승리수)을
/// 반환. 실패 / 시간 초과 시 None.
pub fn search_vct(board: &mut Board, cfg: &VctConfig) -> Option<Vec<Move>> {
    let deadline = cfg.time_budget.map(|d| Instant::now() + d);
    let attacker = board.side_to_move;
    let mut sequence = Vec::with_capacity(cfg.max_depth as usize * 2);
    let mut tt: TransTable = HashMap::with_capacity(65536);
    if vct_or(board, attacker, cfg.max_depth, deadline, &mut sequence, &mut tt) {
        Some(sequence)
    } else {
        None
    }
}

/// OR 노드 — 공격 측 턴. 공격 수 중 하나라도 승리로 이어지면 true.
fn vct_or(
    board: &mut Board,
    attacker: Stone,
    depth: u32,
    deadline: Option<Instant>,
    sequence: &mut Vec<Move>,
    tt: &mut TransTable,
) -> bool {
    if depth == 0 {
        return false;
    }
    if timed_out(deadline) {
        return false;
    }
    debug_assert_eq!(board.side_to_move, attacker);

    // TT 조회 — 같은 깊이 이상으로 탐색된 결과가 있으면 재사용.
    let hash = zobrist_hash(board);
    if let Some(entry) = tt.get(&hash) {
        if entry.depth >= depth {
            // 수열 복원은 포기 (TT hit 시 수열 비어있음). 승리 확정 정보만
            // 상위로 전달. Root 호출에서는 첫 승리 수열이 완전히 돌기 전까지
            // 중간 TT hit이 없으므로 수열 손실은 드묾.
            return matches!(entry.result, TtResult::AttackerWins);
        }
    }

    let (my, opp) = bb_pair(board, attacker);
    let opp_has_immediate_five = has_immediate_five(opp, my);

    let attack_moves = gather_attack_moves(my, opp);
    if attack_moves.is_empty() {
        tt.insert(hash, TtEntry { depth, result: TtResult::Fails });
        return false;
    }

    for (mv, kind) in attack_moves {
        if kind.is_winning() {
            if opp_has_immediate_five && kind != ThreatKind::Five {
                continue;
            }
            sequence.push(mv);
            tt.insert(hash, TtEntry { depth, result: TtResult::AttackerWins });
            return true;
        }
        if opp_has_immediate_five {
            continue;
        }
        sequence.push(mv);
        board.make_move(mv);
        let won = vct_and(board, attacker, depth - 1, deadline, sequence, tt);
        board.undo_move();
        if won {
            tt.insert(hash, TtEntry { depth, result: TtResult::AttackerWins });
            return true;
        }
        sequence.pop();
    }
    tt.insert(hash, TtEntry { depth, result: TtResult::Fails });
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
    depth: u32,
    deadline: Option<Instant>,
    sequence: &mut Vec<Move>,
    tt: &mut TransTable,
) -> bool {
    if depth == 0 {
        return false;
    }
    if timed_out(deadline) {
        return false;
    }
    debug_assert_ne!(board.side_to_move, attacker);

    // 수비 측이 자기 턴에 즉시 5목을 완성할 수 있으면 공격 VCT는 실패.
    let (def_my, def_opp) = bb_pair(board, board.side_to_move);
    if has_immediate_five(def_my, def_opp) {
        return false;
    }

    // 방어 후보: 공격자 직전 수 주변 좁힘 + 수비 측의 카운터 공격 수.
    // (좁힘만 쓰면 원거리 반격수가 누락되어 AND가 false positive를 냄 — VCT
    //  승리 오판. 수비 측이 **자기 winning threat**을 만들 수 있는 수는 반드시
    //  포함해야 함.)
    let defenses = match board.last_move {
        Some(attack_mv) => find_defenses_with_counters(board, attack_mv),
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

        board.make_move(mv);
        let attacker_still_wins = vct_or(board, attacker, depth - 1, deadline, sequence, tt);
        board.undo_move();

        if !attacker_still_wins {
            // 이 방어로 공격 실패 → AND 실패. 수열 복원.
            sequence.truncate(checkpoint);
            return false;
        }
        // 성공 → 다음 분기로. 마지막 분기의 수열이 최종 sequence가 됨.
    }
    true
}

fn gather_attack_moves(my: &BitBoard, opp: &BitBoard) -> Vec<(Move, ThreatKind)> {
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
        let kind = classify_move(my, opp, idx);
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
        ThreatKind::None => 100,
    }
}

fn has_immediate_five(my: &BitBoard, opp: &BitBoard) -> bool {
    for idx in 0..(BOARD_SIZE * BOARD_SIZE) {
        if my.get(idx) || opp.get(idx) {
            continue;
        }
        if classify_move(my, opp, idx) == ThreatKind::Five {
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
fn find_defenses_with_counters(board: &Board, attack_move: Move) -> Vec<Move> {
    let mut defenses = find_defenses(board, attack_move);
    let mut seen = BitBoard::EMPTY;
    for &d in &defenses {
        seen.set(d);
    }
    // 수비자(현재 side_to_move) 관점에서 자기 winning threat 만드는 수들.
    let (def_my, def_opp) = bb_pair(board, board.side_to_move);
    for idx in 0..NUM_CELLS {
        if def_my.get(idx) || def_opp.get(idx) || seen.get(idx) {
            continue;
        }
        let kind = classify_move(def_my, def_opp, idx);
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
fn find_defenses(board: &Board, attack_move: Move) -> Vec<Move> {
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

    out
}

fn bb_pair(board: &Board, side: Stone) -> (&BitBoard, &BitBoard) {
    match side {
        Stone::Black => (&board.black, &board.white),
        Stone::White => (&board.white, &board.black),
    }
}

fn timed_out(deadline: Option<Instant>) -> bool {
    if let Some(d) = deadline {
        if Instant::now() >= d {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::to_idx;

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
        let k1 = classify_move(&board.black, &board.white, to_idx(7, 2));
        let k2 = classify_move(&board.black, &board.white, to_idx(7, 7));
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
        let k = classify_move(&board.black, &board.white, to_idx(7, 7));
        assert_eq!(k, ThreatKind::OpenFour);
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
        let k = classify_move(&board.black, &board.white, to_idx(7, 6));
        assert_eq!(k, ThreatKind::DoubleThree, "should be double three, got {:?}", k);
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
        let k = classify_move(&board.black, &board.white, to_idx(7, 6));
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
