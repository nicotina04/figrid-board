//! NNUE 평가 — 4096 피처 (PS + LP-Rich + Compound + Density) 추출.

use crate::board::{Board, Stone, BOARD_SIZE, NUM_CELLS};
use crate::features::{
    broken_index, compound_index, count_bucket, cross_line_hash, cross_line_index, density_index,
    length_bucket, local_density_bucket, lp_rich_index, open_bucket, ps_index, zone_for,
    BROKEN_SHAPE_DOUBLE_THREE, BROKEN_SHAPE_JUMP_FOUR, BROKEN_SHAPE_THREE, DENSITY_CAT_LEGAL,
    DENSITY_CAT_MY_COUNT, DENSITY_CAT_MY_LOCAL, DENSITY_CAT_OPP_COUNT, DENSITY_CAT_OPP_LOCAL,
    MAX_ACTIVE_FEATURES,
};
use crate::heuristic::{scan_line, DIR};
use noru::network::{forward, Accumulator, NnueWeights};
use std::sync::OnceLock;

static COMPOUND_ENABLED: OnceLock<bool> = OnceLock::new();

fn compound_enabled() -> bool {
    *COMPOUND_ENABLED.get_or_init(|| std::env::var("NORU_NO_COMPOUND").is_err())
}

/// 보드 상태에서 활성 피처를 추출.
///
/// LP-Rich는 라인의 시작 셀에서 한 번만 카운트 (중복 방지). 같은 패턴이 보드의 다른 위치에
/// 또 있으면 zone이 같을 때만 같은 인덱스로 중복 합산됨 — 이는 의도된 동작.
pub fn compute_active_features(board: &Board) -> (Vec<usize>, Vec<usize>) {
    let (my_bb, opp_bb) = match board.side_to_move {
        Stone::Black => (&board.black, &board.white),
        Stone::White => (&board.white, &board.black),
    };

    let mut stm = Vec::with_capacity(MAX_ACTIVE_FEATURES);
    let mut nstm = Vec::with_capacity(MAX_ACTIVE_FEATURES);

    // === A. PS (stone-driven) ===
    for sq in my_bb.iter_ones() {
        stm.push(ps_index(0, sq));
        nstm.push(ps_index(1, sq));
    }
    for sq in opp_bb.iter_ones() {
        stm.push(ps_index(1, sq));
        nstm.push(ps_index(0, sq));
    }

    // === B. LP-Rich (stone-driven) ===
    // bitboard iter_ones로 돌만 스캔. `for 0..NUM_CELLS`의 빈 칸 분기 제거.
    // iter_ones는 lowest-idx-first라 기존 순서 완전 보존 (feature push 순서 유지).
    for idx in my_bb.iter_ones() {
        let row = (idx / BOARD_SIZE) as i32;
        let col = (idx % BOARD_SIZE) as i32;
        for (dir_idx, &(dr, dc)) in DIR.iter().enumerate() {
            if is_line_start(my_bb, row, col, dr, dc) {
                let info = scan_line(my_bb, opp_bb, row, col, dr, dc);
                let z = zone_for(row, col);
                let len = length_bucket(info.count);
                let op = open_bucket(info.open_front, info.open_back);
                stm.push(lp_rich_index(0, len, op, dir_idx, z));
                nstm.push(lp_rich_index(1, len, op, dir_idx, z));
            }
        }
    }
    for idx in opp_bb.iter_ones() {
        let row = (idx / BOARD_SIZE) as i32;
        let col = (idx % BOARD_SIZE) as i32;
        for (dir_idx, &(dr, dc)) in DIR.iter().enumerate() {
            if is_line_start(opp_bb, row, col, dr, dc) {
                let info = scan_line(opp_bb, my_bb, row, col, dr, dc);
                let z = zone_for(row, col);
                let len = length_bucket(info.count);
                let op = open_bucket(info.open_front, info.open_back);
                stm.push(lp_rich_index(1, len, op, dir_idx, z));
                nstm.push(lp_rich_index(0, len, op, dir_idx, z));
            }
        }
    }

    // === C. Compound Threats ===
    // 각 돌에서 4방향 패턴을 수집하여, 한 돌에 다중 위협이 동시에 걸리는 경우를 인코딩.
    // NORU_NO_COMPOUND=1 환경변수로 OFF 가능 (v3 재현/A-B 테스트용).
    if compound_enabled() {
        compute_compound_threats(my_bb, opp_bb, &mut stm, &mut nstm);
    }

    // === D. Density / Mobility ===
    let my_count = my_bb.count_ones();
    let opp_count = opp_bb.count_ones();
    push_density(&mut stm, &mut nstm, DENSITY_CAT_MY_COUNT, count_bucket(my_count));
    push_density(&mut stm, &mut nstm, DENSITY_CAT_OPP_COUNT, count_bucket(opp_count));

    // 마지막 수 주변 3×3 밀도 (last_move 없으면 0).
    let (my_local, opp_local) = local_density(board);
    push_density(
        &mut stm,
        &mut nstm,
        DENSITY_CAT_MY_LOCAL,
        local_density_bucket(my_local),
    );
    push_density(
        &mut stm,
        &mut nstm,
        DENSITY_CAT_OPP_LOCAL,
        local_density_bucket(opp_local),
    );

    let legal = (NUM_CELLS as u32).saturating_sub(my_count + opp_count);
    push_density(&mut stm, &mut nstm, DENSITY_CAT_LEGAL, count_bucket(legal));

    // === E. Cross-line 3×3 local window hash (stone-driven) ===
    // For every stone on the board, encode the 3×3 window around it as a
    // D4-canonicalized 256-bucket hash. Captures 2D interactions the
    // line-based LP-Rich encoder misses (corner squeeze, 十-shapes, etc.).
    // iter_ones로 두 색 돌 순회 — 빈 칸에 대한 early-continue가 아예 없어짐.
    for sq in my_bb.iter_ones().chain(opp_bb.iter_ones()) {
        let row = (sq / BOARD_SIZE) as i32;
        let col = (sq % BOARD_SIZE) as i32;

        let stm_cells = collect_3x3(my_bb, opp_bb, row, col);
        let stm_bucket = cross_line_hash(stm_cells);
        stm.push(cross_line_index(0, stm_bucket));
        nstm.push(cross_line_index(1, stm_bucket));

        let nstm_cells = swap_mine_opp(stm_cells);
        let nstm_bucket = cross_line_hash(nstm_cells);
        stm.push(cross_line_index(1, nstm_bucket));
        nstm.push(cross_line_index(0, nstm_bucket));
    }

    // === F. Broken / Jump patterns (stone-driven) ===
    // 각 자기 돌에서 4방향 11칸 창을 읽어 broken/jump/double 패턴을 감지.
    // iter_ones로 돌만 스캔 — 빈 칸 분기 제거.
    for idx in my_bb.iter_ones() {
        let row = (idx / BOARD_SIZE) as i32;
        let col = (idx % BOARD_SIZE) as i32;
        for (dir_idx, &(dr, dc)) in DIR.iter().enumerate() {
            detect_broken_and_push(
                my_bb, opp_bb, row, col, dr, dc, dir_idx, 0, 1, &mut stm, &mut nstm,
            );
        }
    }
    for idx in opp_bb.iter_ones() {
        let row = (idx / BOARD_SIZE) as i32;
        let col = (idx % BOARD_SIZE) as i32;
        for (dir_idx, &(dr, dc)) in DIR.iter().enumerate() {
            detect_broken_and_push(
                opp_bb, my_bb, row, col, dr, dc, dir_idx, 1, 0, &mut stm, &mut nstm,
            );
        }
    }

    (stm, nstm)
}

/// `(row, col)`이 `(dr, dc)` 방향 라인의 **왼쪽 앵커 돌**일 때 broken 패턴을
/// 찾아서 feature 벡터에 push한다. 중복 방지: `is_line_start` 역할 (뒤쪽이
/// 같은 색 돌이면 다른 돌이 앵커이므로 skip).
///
/// `perspective_mine` / `perspective_opp`: stm 관점에서 "내 돌"이 0, "상대 돌"이 1.
#[allow(clippy::too_many_arguments)]
fn detect_broken_and_push(
    stones: &crate::board::BitBoard,
    opp: &crate::board::BitBoard,
    row: i32,
    col: i32,
    dr: i32,
    dc: i32,
    dir_idx: usize,
    perspective_mine: usize,
    perspective_opp: usize,
    stm: &mut Vec<usize>,
    nstm: &mut Vec<usize>,
) {
    // Dedup: 같은 패턴의 모든 돌에서 push되지 않도록 "라인 시작" 돌에서만 처리.
    let pr = row - dr;
    let pc = col - dc;
    if pr >= 0 && pr < BOARD_SIZE as i32 && pc >= 0 && pc < BOARD_SIZE as i32 {
        if stones.get((pr as usize) * BOARD_SIZE + pc as usize) {
            return; // 뒤쪽에도 같은 색 돌이 있으면 이 돌은 앵커가 아님
        }
    }

    // 11칸 창 구성: -5..=5 (self = idx 5). 0=빈, 1=mine, 2=opp/boundary.
    let mut line = [2u8; 11]; // 기본값 boundary
    for off in -5i32..=5 {
        let nr = row + dr * off;
        let nc = col + dc * off;
        if nr < 0 || nr >= BOARD_SIZE as i32 || nc < 0 || nc >= BOARD_SIZE as i32 {
            continue; // boundary
        }
        let cell_idx = (nr as usize) * BOARD_SIZE + nc as usize;
        let slot = (off + 5) as usize;
        if stones.get(cell_idx) {
            line[slot] = 1;
        } else if opp.get(cell_idx) {
            line[slot] = 2;
        } else {
            line[slot] = 0;
        }
    }

    // 앵커 중심에서 오른쪽으로 "stones + 1 gap"까지 스캔해 분류.
    let zone = zone_for(row, col);
    if let Some((shape, is_open)) = classify_broken_shape(&line) {
        let open_bucket = if is_open { 1 } else { 0 };
        stm.push(broken_index(
            perspective_mine,
            shape,
            open_bucket,
            dir_idx,
            zone,
        ));
        nstm.push(broken_index(
            perspective_opp,
            shape,
            open_bucket,
            dir_idx,
            zone,
        ));
    }
}

/// `line[5]`가 앵커 돌 (mine), `line[4]`는 mine이 아님(앵커 조건 이미 체크됨).
/// 오른쪽 방향으로 최대 1개의 gap을 허용해 stones를 세고, open/closed와
/// shape(broken_three / jump_four / double_broken_three)을 판정.
fn classify_broken_shape(line: &[u8; 11]) -> Option<(usize, bool)> {
    debug_assert!(line[5] == 1);

    // 앵커 기준 오른쪽 6칸만 본다 (idx 5..=11). 앵커를 포함한 최대 6칸 창.
    //
    // 패턴 템플릿 (앵커=M, 다른 내 돌=m, 빈칸=_, 상대/경계=x):
    //   broken three  : M _ m m _   또는 M m _ m _  (양쪽 열림 + 5칸 창 안에 mine 3개, gap 1개)
    //   jump four     : M _ m m m _   또는 M m _ m m _   또는 M m m _ m _ (mine 4, gap 1, 오른쪽 열림 또는 양쪽)
    //   double broken : M _ m _ m _   (gap 2개 + mine 3 + 양쪽 열림; 2개의 broken three 시퀀스 공유)
    //
    // 모두 앵커 왼쪽은 empty인지(= open_left) 추가로 확인.

    let open_left = line[4] == 0;

    // 오른쪽 5칸(idx 6..=10) 시퀀스를 읽어 mine/empty/blocker 패턴화.
    // opp/boundary가 나오면 그 지점 이후 무효.
    let mut cells: [u8; 5] = [2; 5];
    for i in 0..5 {
        cells[i] = line[6 + i];
    }

    // 오른쪽으로 스캔해 "최대 1 gap 허용" 안에서 mine count와 구조 분석.
    // 유효 blocker 전까지의 (mine_count, gap_count, right_open) 을 계산.
    // 핵심: 연속 두 empty를 만나면 첫 empty는 패턴 내부 gap이 아닌 "trailing
    // open boundary"로 재해석해 gap_count에서 제외.
    let mut mine_right = 0u32;
    let mut gap_count = 0u32;
    let mut right_open = false;
    let mut prev_was_empty = false;
    let mut scan_ended_early = false;

    for &c in &cells {
        if c == 2 {
            scan_ended_early = true;
            break;
        }
        if c == 0 {
            if prev_was_empty {
                // 두 번째 연속 empty: 앞서 센 첫 empty는 실제론 trailing open.
                right_open = true;
                gap_count = gap_count.saturating_sub(1);
                scan_ended_early = true;
                break;
            }
            gap_count += 1;
            prev_was_empty = true;
        } else {
            mine_right += 1;
            prev_was_empty = false;
        }
    }

    // 스캔이 5칸 모두 소진한 경우: 마지막이 empty면 그게 trailing open.
    if !scan_ended_early && prev_was_empty {
        right_open = true;
        gap_count = gap_count.saturating_sub(1);
    }

    // gap이 0이면 solid line — LP-Rich가 이미 처리, skip.
    if gap_count == 0 {
        return None;
    }

    let total_mine = 1 + mine_right; // 앵커 포함
    let is_open = open_left && right_open;

    match (total_mine, gap_count) {
        (3, 1) => Some((BROKEN_SHAPE_THREE, is_open)),
        (4, 1) => Some((BROKEN_SHAPE_JUMP_FOUR, is_open)),
        (3, 2) => Some((BROKEN_SHAPE_DOUBLE_THREE, is_open)),
        _ => None,
    }
}

/// Collect a 3×3 window centered on (row, col). Out-of-board cells encode
/// as 3 (boundary), mine=1, opp=2, empty=0.
#[inline]
fn collect_3x3(
    my_bb: &crate::board::BitBoard,
    opp_bb: &crate::board::BitBoard,
    row: i32,
    col: i32,
) -> [u8; 9] {
    let mut cells = [0u8; 9];
    let mut i = 0;
    for dr in -1..=1 {
        for dc in -1..=1 {
            let nr = row + dr;
            let nc = col + dc;
            if nr < 0 || nr >= BOARD_SIZE as i32 || nc < 0 || nc >= BOARD_SIZE as i32 {
                cells[i] = 3;
            } else {
                let idx = (nr as usize) * BOARD_SIZE + (nc as usize);
                if my_bb.get(idx) {
                    cells[i] = 1;
                } else if opp_bb.get(idx) {
                    cells[i] = 2;
                } else {
                    cells[i] = 0;
                }
            }
            i += 1;
        }
    }
    cells
}

/// Swap mine↔opp markers (1↔2). Boundary (3) and empty (0) unchanged.
#[inline]
fn swap_mine_opp(c: [u8; 9]) -> [u8; 9] {
    let mut out = [0u8; 9];
    for i in 0..9 {
        out[i] = match c[i] {
            1 => 2,
            2 => 1,
            v => v,
        };
    }
    out
}

#[inline]
fn is_line_start(bb: &crate::board::BitBoard, row: i32, col: i32, dr: i32, dc: i32) -> bool {
    let pr = row - dr;
    let pc = col - dc;
    if pr < 0 || pr >= BOARD_SIZE as i32 || pc < 0 || pc >= BOARD_SIZE as i32 {
        return true;
    }
    !bb.get(pr as usize * BOARD_SIZE + pc as usize)
}

#[inline]
fn push_density(stm: &mut Vec<usize>, nstm: &mut Vec<usize>, cat: usize, bucket: usize) {
    let idx = density_index(cat, bucket);
    stm.push(idx);
    nstm.push(idx);
}

/// 패턴 등급 (높을수록 강한 위협).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Threat {
    None,
    OpenTwo,   // 열린 2
    ClosedThree, // 닫힌 3
    OpenThree, // 열린 3
    ClosedFour, // 닫힌 4
    OpenFour,  // 열린 4
    Five,      // 5목
}

fn classify_threat(count: u32, open_ends: u32) -> Threat {
    match (count, open_ends) {
        (5.., _) => Threat::Five,
        (4, 2) => Threat::OpenFour,
        (4, 1) => Threat::ClosedFour,
        (3, 2) => Threat::OpenThree,
        (3, 1) => Threat::ClosedThree,
        (2, 2) => Threat::OpenTwo,
        _ => Threat::None,
    }
}

/// Compound threat combo ID 매핑.
///
/// 한 돌의 4방향 패턴을 내림차순 정렬하여, 상위 2개 위협 조합으로 combo_id를 결정.
/// 50 슬롯 배분:
///   0..6   : reserved (과거 단일 위협 슬롯 — 현재 비활성; LP-Rich가 이미 커버)
///   6..12  : Five  + X  (Five+Five, Five+OF, Five+CF, Five+O3, Five+C3, Five+O2)
///  12..17  : OF    + X  (OF+OF, OF+CF, OF+O3, OF+C3, OF+O2)
///  17..21  : CF    + X  (CF+CF, CF+O3, CF+C3, CF+O2)
///  21..24  : O3    + X  (O3+O3, O3+C3, O3+O2)  ← 핵심: double-three
///  24..26  : C3    + X  (C3+C3, C3+O2)
///  26..27  : O2    + O2
///  27..33  : 3중 위협 (top1+top2+top3 보너스, 6가지)
///  33..49  : reserved (미래 확장)
///
/// compound는 "한 돌에 걸린 **다중** 위협 교차점"을 캡처하는 용도라, 단일
/// 위협은 LP-Rich가 이미 처리하므로 None을 반환해 중복 피처를 방지한다.
fn compound_combo_id(threats: &[Threat; 4]) -> Option<usize> {
    let mut sorted = *threats;
    sorted.sort_unstable_by(|a, b| b.cmp(a)); // 내림차순

    let t1 = sorted[0];
    let t2 = sorted[1];
    let t3 = sorted[2];

    if t1 == Threat::None {
        return None; // 위협 없음
    }

    let t1_rank = threat_rank(t1);
    let t2_rank = threat_rank(t2);

    // 단일 위협은 LP-Rich가 이미 커버 → compound는 다중 위협만.
    if t2 == Threat::None {
        return None;
    }

    // 이중 위협 combo
    let dual_id = match t1_rank {
        0 => 6 + t2_rank,               // Five + X: 6..12
        1 => 12 + (t2_rank - 1),        // OF + X: 12..17
        2 => 17 + (t2_rank - 2),        // CF + X: 17..21
        3 => 21 + (t2_rank - 3),        // O3 + X: 21..24
        4 => 24 + (t2_rank - 4),        // C3 + X: 24..26
        5 => 26,                         // O2 + O2: 26
        _ => return None,
    };

    // 3중 위협 보너스
    if t3 != Threat::None && dual_id < 33 {
        let triple_base = 27;
        let triple_id = triple_base + threat_rank(t1).min(5);
        return Some(triple_id);
    }

    Some(dual_id)
}

fn threat_rank(t: Threat) -> usize {
    match t {
        Threat::Five => 0,
        Threat::OpenFour => 1,
        Threat::ClosedFour => 2,
        Threat::OpenThree => 3,
        Threat::ClosedThree => 4,
        Threat::OpenTwo => 5,
        Threat::None => 6,
    }
}

/// 모든 돌에 대해 compound threats를 검출하여 피처 벡터에 추가.
fn compute_compound_threats(
    my_bb: &crate::board::BitBoard,
    opp_bb: &crate::board::BitBoard,
    stm: &mut Vec<usize>,
    nstm: &mut Vec<usize>,
) {
    // 자기 돌 순회 — iter_ones로 set bit만 스캔 (225-loop-per-empty 제거).
    // 교차점 hotspot 감지가 목적이므로 `is_line_start` 필터 없이 이 돌 기준
    // 모든 방향 라인을 스캔한다. 중복은 compound_combo_id에서 단일 위협
    // 컷오프로 방지.
    for idx in my_bb.iter_ones() {
        let row = (idx / BOARD_SIZE) as i32;
        let col = (idx % BOARD_SIZE) as i32;
        let mut threats = [Threat::None; 4];
        for (di, &(dr, dc)) in DIR.iter().enumerate() {
            let info = scan_line(my_bb, opp_bb, row, col, dr, dc);
            let open = info.open_front as u32 + info.open_back as u32;
            threats[di] = classify_threat(info.count, open);
        }
        if let Some(combo) = compound_combo_id(&threats) {
            stm.push(compound_index(0, combo));
            nstm.push(compound_index(1, combo));
        }
    }

    // 상대 돌 순회
    for idx in opp_bb.iter_ones() {
        let row = (idx / BOARD_SIZE) as i32;
        let col = (idx % BOARD_SIZE) as i32;
        let mut threats = [Threat::None; 4];
        for (di, &(dr, dc)) in DIR.iter().enumerate() {
            let info = scan_line(opp_bb, my_bb, row, col, dr, dc);
            let open = info.open_front as u32 + info.open_back as u32;
            threats[di] = classify_threat(info.count, open);
        }
        if let Some(combo) = compound_combo_id(&threats) {
            stm.push(compound_index(1, combo));
            nstm.push(compound_index(0, combo));
        }
    }
}

/// 마지막 수 주변 3×3 안의 (자기, 상대) 돌 수.
fn local_density(board: &Board) -> (u32, u32) {
    let Some(mv) = board.last_move else {
        return (0, 0);
    };
    let (my_bb, opp_bb) = match board.side_to_move {
        Stone::Black => (&board.black, &board.white),
        Stone::White => (&board.white, &board.black),
    };
    let r = (mv / BOARD_SIZE) as i32;
    let c = (mv % BOARD_SIZE) as i32;
    let mut my = 0u32;
    let mut op = 0u32;
    for dr in -1..=1 {
        for dc in -1..=1 {
            if dr == 0 && dc == 0 {
                continue;
            }
            let nr = r + dr;
            let nc = c + dc;
            if nr < 0 || nr >= BOARD_SIZE as i32 || nc < 0 || nc >= BOARD_SIZE as i32 {
                continue;
            }
            let i = (nr as usize) * BOARD_SIZE + nc as usize;
            if my_bb.get(i) {
                my += 1;
            }
            if opp_bb.get(i) {
                op += 1;
            }
        }
    }
    (my, op)
}

/// 보드를 평가 (전체 재계산)
pub fn evaluate(board: &Board, weights: &NnueWeights) -> i32 {
    let (stm_feats, nstm_feats) = compute_active_features(board);
    let mut acc = Accumulator::new(&weights.feature_bias);
    acc.refresh(weights, &stm_feats, &nstm_feats);
    forward(&acc, weights)
}

/// 증분 평가용 상태 (현재는 전체 재계산만).
pub struct IncrementalEval {
    pub accumulator: Accumulator,
    stack: Vec<Accumulator>,
}

impl IncrementalEval {
    pub fn new(weights: &NnueWeights) -> Self {
        Self {
            accumulator: Accumulator::new(&weights.feature_bias),
            stack: Vec::with_capacity(225),
        }
    }

    pub fn refresh(&mut self, board: &Board, weights: &NnueWeights) {
        let (stm_feats, nstm_feats) = compute_active_features(board);
        self.accumulator.refresh(weights, &stm_feats, &nstm_feats);
    }

    pub fn push_move(&mut self, board: &Board, _mv: usize, weights: &NnueWeights) {
        self.stack.push(self.accumulator.clone());
        self.refresh(board, weights);
    }

    pub fn pop_move(&mut self) {
        if let Some(prev) = self.stack.pop() {
            self.accumulator = prev;
        }
    }

    pub fn eval(&self, weights: &NnueWeights) -> i32 {
        forward(&self.accumulator, weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::features::{
        broken_index, compound_index, BROKEN_SHAPE_DOUBLE_THREE, BROKEN_SHAPE_JUMP_FOUR,
        BROKEN_SHAPE_THREE, GOMOKU_NNUE_CONFIG, HALF_FEATURE_SIZE, LP_BASE, MAX_ACTIVE_FEATURES,
        PS_BASE, TOTAL_FEATURE_SIZE,
    };

    #[test]
    fn empty_board_has_only_density_features() {
        let board = Board::new();
        let (stm, nstm) = compute_active_features(&board);
        // 빈 보드: PS 0개, LP 0개, Density 5개.
        assert_eq!(stm.len(), 5);
        assert_eq!(nstm.len(), 5);
    }

    #[test]
    fn evaluate_zero_weights() {
        let board = Board::new();
        let weights = NnueWeights::zeros(GOMOKU_NNUE_CONFIG);
        assert_eq!(evaluate(&board, &weights), 0);
    }

    #[test]
    fn features_include_lp_after_two_in_row() {
        let mut board = Board::new();
        board.make_move(7 * 15 + 7); // B
        board.make_move(0); // W
        board.make_move(7 * 15 + 8); // B (가로 2연)
        let (stm, _) = compute_active_features(&board);
        let has_lp = stm
            .iter()
            .any(|&f| f >= LP_BASE && f < LP_BASE + 2 * 1152);
        assert!(has_lp, "should have LP-Rich features after 2-in-row");
    }

    #[test]
    fn all_features_within_range() {
        let mut board = Board::new();
        for sq in [112, 0, 113, 1, 114, 15, 100, 50] {
            board.make_move(sq);
        }
        let (stm, nstm) = compute_active_features(&board);
        for &f in stm.iter().chain(nstm.iter()) {
            assert!(f < TOTAL_FEATURE_SIZE, "feature {f} >= {TOTAL_FEATURE_SIZE}");
        }
    }

    #[test]
    fn active_features_under_cap() {
        // 보드를 가득 채워도 활성 피처가 상한 안에 있어야 함.
        let mut board = Board::new();
        for sq in 0..NUM_CELLS {
            if board.is_empty(sq) {
                board.make_move(sq);
            }
        }
        let (stm, nstm) = compute_active_features(&board);
        assert!(stm.len() <= MAX_ACTIVE_FEATURES, "stm len={}", stm.len());
        assert!(nstm.len() <= MAX_ACTIVE_FEATURES, "nstm len={}", nstm.len());
    }

    #[test]
    fn push_pop_consistency() {
        let mut weights = NnueWeights::zeros(GOMOKU_NNUE_CONFIG);
        let acc_size = GOMOKU_NNUE_CONFIG.accumulator_size;
        for sq in 0..20 {
            for i in 0..acc_size {
                weights.feature_weights[sq][i] = ((sq * 7 + i) % 13) as i16 - 6;
                weights.feature_weights[sq + HALF_FEATURE_SIZE][i] =
                    ((sq * 3 + i) % 11) as i16 - 5;
            }
        }
        let mut board = Board::new();
        let mut inc = IncrementalEval::new(&weights);
        inc.refresh(&board, &weights);
        let before = inc.eval(&weights);
        board.make_move(112);
        inc.push_move(&board, 112, &weights);
        board.undo_move();
        inc.pop_move();
        assert_eq!(before, inc.eval(&weights));

        // PS_BASE silence
        let _ = PS_BASE;
    }

    /// 수정 전 compound 로직은 각 방향에서 `is_line_start` 돌에서만
    /// threats를 수집했기 때문에, 라인의 **중간 돌**이 여러 방향으로 open-three
    /// 교차점이 되어도 compound double-three 피처가 붙지 않았다. 이 테스트는
    /// 수정 후 line_start가 아닌 중심 돌에서도 hotspot이 잡히는지 검증한다.
    #[test]
    fn compound_catches_double_three_at_non_line_start_stone() {
        let mut board = Board::new();
        // Black: 가로 3연 (7,5)(7,6)(7,7), 세로 3연 (6,7)(7,7)(8,7)
        // (7,7) 은 가로 기준 왼쪽 인접이 흑(7,6), 세로 기준 위 인접이 흑(6,7)
        // 이라 두 방향 모두 line_start 아님. 수정 전엔 double-three 미검출.
        board.make_move(7 * 15 + 5); // B (7,5)
        board.make_move(0);          // W (0,0) — far away, no interference
        board.make_move(7 * 15 + 6); // B (7,6)
        board.make_move(1);          // W (0,1)
        board.make_move(7 * 15 + 7); // B (7,7) ← crossing stone
        board.make_move(2);          // W (0,2)
        board.make_move(6 * 15 + 7); // B (6,7)
        board.make_move(3);          // W (0,3)
        board.make_move(8 * 15 + 7); // B (8,7)
        // side_to_move = White (9수 후 흑이 마지막)

        let (stm, _) = compute_active_features(&board);

        // O3+O3 (combo_id = 21) 이 Black(상대) 관점으로 stm에 들어와야 함.
        // compute_compound_threats 내부: my_bb=White, opp_bb=Black.
        // `opp_bb.get(idx)` 분기 → stm.push(compound_index(1, combo)).
        let expected = compound_index(1, 21);
        assert!(
            stm.contains(&expected),
            "stm should contain opponent's O3+O3 compound at the non-line-start \
             crossing stone (7,7); expected feature index {expected} missing.\n\
             stm={stm:?}"
        );
    }

    /// Broken three 패턴 (`_●●_●_` 형태, gap 1개 포함 mine 3, 양쪽 열림).
    /// scan_line/LP-Rich의 연속-only 감지로는 놓치는 패턴.
    #[test]
    fn broken_three_detected_open() {
        let mut board = Board::new();
        // 가로: 빈(7,4) / 흑(7,5) / 흑(7,6) / 빈(7,7) / 흑(7,8) / 빈(7,9)
        // 양쪽 열림 (open_left at (7,4), open_right at (7,9)), gap 1개 at (7,7).
        board.make_move(7 * 15 + 5);  // B (7,5) 앵커
        board.make_move(0);           // W far
        board.make_move(7 * 15 + 6);  // B (7,6)
        board.make_move(1);           // W far
        board.make_move(7 * 15 + 8);  // B (7,8)
        // side_to_move = White (5수 후 흑이 마지막)

        let (stm, _) = compute_active_features(&board);

        // 앵커는 (7,5). dir=0 (가로), zone=zone_for(7,5)=4. Black은 nstm=1.
        let zone = zone_for(7, 5);
        let expected = broken_index(1, BROKEN_SHAPE_THREE, 1, 0, zone);
        assert!(
            stm.contains(&expected),
            "expected broken three (open) feature {expected} missing; stm={stm:?}"
        );
    }

    /// Jump four 패턴 (`_●●●_●_` 형태, gap 1개 포함 mine 4).
    #[test]
    fn jump_four_detected() {
        let mut board = Board::new();
        // 가로: 빈(7,4) 흑(7,5) 흑(7,6) 흑(7,7) 빈(7,8) 흑(7,9) 빈(7,10)
        board.make_move(7 * 15 + 5);  // B (7,5) 앵커
        board.make_move(0);
        board.make_move(7 * 15 + 6);  // B (7,6)
        board.make_move(1);
        board.make_move(7 * 15 + 7);  // B (7,7)
        board.make_move(2);
        board.make_move(7 * 15 + 9);  // B (7,9)
        // side_to_move = White

        let (stm, _) = compute_active_features(&board);

        let zone = zone_for(7, 5);
        let expected = broken_index(1, BROKEN_SHAPE_JUMP_FOUR, 1, 0, zone);
        assert!(
            stm.contains(&expected),
            "expected jump four (open) feature {expected} missing; stm={stm:?}"
        );
    }

    /// Double-broken three (`_●_●_●_` 형태, gap 2개 mine 3, 양쪽 열림).
    /// 한 라인 안에 두 번의 gap이 포함된 3개 돌 구조.
    #[test]
    fn double_broken_three_detected() {
        let mut board = Board::new();
        // 가로: 빈(7,4) 흑(7,5) 빈(7,6) 흑(7,7) 빈(7,8) 흑(7,9) 빈(7,10)
        board.make_move(7 * 15 + 5);  // B (7,5) 앵커
        board.make_move(0);
        board.make_move(7 * 15 + 7);  // B (7,7)
        board.make_move(1);
        board.make_move(7 * 15 + 9);  // B (7,9)
        // side_to_move = White

        let (stm, _) = compute_active_features(&board);

        let zone = zone_for(7, 5);
        let expected = broken_index(1, BROKEN_SHAPE_DOUBLE_THREE, 1, 0, zone);
        assert!(
            stm.contains(&expected),
            "expected double broken three (open) feature {expected} missing; stm={stm:?}"
        );
    }

    /// 단일 위협(한 방향에만 라인 있음)은 LP-Rich 피처가 이미 커버하므로
    /// compound에서 중복 push되어선 안 된다. 수정 전에는 combo_id 0..6 슬롯에
    /// single-direction threat가 들어갔지만, 수정 후엔 compound가 다중 위협만 잡는다.
    #[test]
    fn compound_excludes_single_threat() {
        let mut board = Board::new();
        // 가로 3연만 (세로/대각 threat 없음) → 각 흑 돌은 O3 단일 위협.
        board.make_move(7 * 15 + 6); // B (7,6)
        board.make_move(0);          // W
        board.make_move(7 * 15 + 7); // B (7,7)
        board.make_move(1);          // W
        board.make_move(7 * 15 + 8); // B (7,8)
        // side_to_move = White

        let (stm, _) = compute_active_features(&board);

        // 단일 O3 슬롯 (combo_id 3) 이 compound 영역에 들어가면 안 됨.
        let single_o3 = compound_index(1, 3);
        assert!(
            !stm.contains(&single_o3),
            "compound should skip single O3 (already handled by LP-Rich); \
             unexpected single-threat compound feature {single_o3} found in stm={stm:?}"
        );
    }
}
