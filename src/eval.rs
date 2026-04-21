//! NNUE 평가 — 4096 피처 (PS + LP-Rich + Compound + Density) 추출.

use crate::board::{Board, Stone, BOARD_SIZE, NUM_CELLS};
use crate::features::{
    compound_index, count_bucket, density_index, length_bucket, local_density_bucket,
    lp_rich_index, open_bucket, ps_index, zone_for, DENSITY_CAT_LEGAL, DENSITY_CAT_MY_COUNT,
    DENSITY_CAT_MY_LOCAL, DENSITY_CAT_OPP_COUNT, DENSITY_CAT_OPP_LOCAL, MAX_ACTIVE_FEATURES,
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

    // === A. PS ===
    for sq in 0..NUM_CELLS {
        if my_bb.get(sq) {
            stm.push(ps_index(0, sq));
            nstm.push(ps_index(1, sq));
        } else if opp_bb.get(sq) {
            stm.push(ps_index(1, sq));
            nstm.push(ps_index(0, sq));
        }
    }

    // === B. LP-Rich ===
    for idx in 0..NUM_CELLS {
        let row = (idx / BOARD_SIZE) as i32;
        let col = (idx % BOARD_SIZE) as i32;

        for (dir_idx, &(dr, dc)) in DIR.iter().enumerate() {
            // 자기 라인 (시작 셀에서만)
            if my_bb.get(idx) && is_line_start(my_bb, row, col, dr, dc) {
                let info = scan_line(my_bb, opp_bb, row, col, dr, dc);
                let z = zone_for(row, col);
                let len = length_bucket(info.count);
                let op = open_bucket(info.open_front, info.open_back);
                stm.push(lp_rich_index(0, len, op, dir_idx, z));
                nstm.push(lp_rich_index(1, len, op, dir_idx, z));
            }

            // 상대 라인
            if opp_bb.get(idx) && is_line_start(opp_bb, row, col, dr, dc) {
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

    (stm, nstm)
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
///   0..6   : 단일 위협 (하위 호환 — Five, OF, CF, O3, C3, O2, None)
///   6..12  : Five  + X  (Five+Five, Five+OF, Five+CF, Five+O3, Five+C3, Five+O2)
///  12..17  : OF    + X  (OF+OF, OF+CF, OF+O3, OF+C3, OF+O2)
///  17..21  : CF    + X  (CF+CF, CF+O3, CF+C3, CF+O2)
///  21..24  : O3    + X  (O3+O3, O3+C3, O3+O2)  ← 핵심: double-three
///  24..26  : C3    + X  (C3+C3, C3+O2)
///  26..27  : O2    + O2
///  27..33  : 3중 위협 (top1+top2+top3 보너스, 6가지)
///  33..49  : reserved (미래 확장)
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

    // 단일 위협 (두 번째가 None)
    if t2 == Threat::None {
        return Some(t1_rank);
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
    for idx in 0..NUM_CELLS {
        let row = (idx / BOARD_SIZE) as i32;
        let col = (idx % BOARD_SIZE) as i32;

        // 자기 돌에서 4방향 위협 수집
        if my_bb.get(idx) {
            let mut threats = [Threat::None; 4];
            for (di, &(dr, dc)) in DIR.iter().enumerate() {
                if is_line_start(my_bb, row, col, dr, dc) {
                    let info = scan_line(my_bb, opp_bb, row, col, dr, dc);
                    let open = info.open_front as u32 + info.open_back as u32;
                    threats[di] = classify_threat(info.count, open);
                }
            }
            if let Some(combo) = compound_combo_id(&threats) {
                stm.push(compound_index(0, combo));
                nstm.push(compound_index(1, combo));
            }
        }

        // 상대 돌
        if opp_bb.get(idx) {
            let mut threats = [Threat::None; 4];
            for (di, &(dr, dc)) in DIR.iter().enumerate() {
                if is_line_start(opp_bb, row, col, dr, dc) {
                    let info = scan_line(opp_bb, my_bb, row, col, dr, dc);
                    let open = info.open_front as u32 + info.open_back as u32;
                    threats[di] = classify_threat(info.count, open);
                }
            }
            if let Some(combo) = compound_combo_id(&threats) {
                stm.push(compound_index(1, combo));
                nstm.push(compound_index(0, combo));
            }
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
        GOMOKU_NNUE_CONFIG, HALF_FEATURE_SIZE, LP_BASE, MAX_ACTIVE_FEATURES, PS_BASE,
        TOTAL_FEATURE_SIZE,
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
}
