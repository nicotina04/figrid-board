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
use noru::network::{forward, Accumulator, FeatureDelta, NnueWeights};
use std::sync::OnceLock;

/// Incremental update시 한 수로 인해 feature가 바뀔 가능성이 있는 cell 집합 반환.
///
/// LP-Rich / Broken은 `scan_line`이 anchor 돌 ±4 범위를 보므로, mv 주변
/// ±5까지의 anchor 돌들의 feature가 영향을 받는다. Cross-line은 mv 주변
/// ±1 돌들의 3×3 window에 mv가 포함되므로 ±1 포함. Compound도 ±4 안.
///
/// 11×11 정사각형(중앙이 mv)을 반환 — 보드 밖은 스킵. 평균 ~100 cells.
pub(crate) fn affected_cells(mv: usize) -> Vec<usize> {
    let row = (mv / BOARD_SIZE) as i32;
    let col = (mv % BOARD_SIZE) as i32;
    let mut cells = Vec::with_capacity(121);
    for dr in -5..=5i32 {
        for dc in -5..=5i32 {
            let r = row + dr;
            let c = col + dc;
            if r >= 0 && r < BOARD_SIZE as i32 && c >= 0 && c < BOARD_SIZE as i32 {
                cells.push((r as usize) * BOARD_SIZE + c as usize);
            }
        }
    }
    cells
}

static COMPOUND_ENABLED: OnceLock<bool> = OnceLock::new();

fn compound_enabled() -> bool {
    *COMPOUND_ENABLED.get_or_init(|| std::env::var("NORU_NO_COMPOUND").is_err())
}

/// 보드 상태에서 활성 피처를 추출.
///
/// cell-centric 구조: 각 cell에서 `features_from_cell`로 A/B/C/E/F 섹션의
/// 그 cell에 해당하는 feature들을 emit. D (Density) 는 global이라 마지막에
/// 별도 추가. 이 구조는 incremental update의 기반 — 한 수로 바뀌는 cell
/// 영역만 재계산해 delta apply 가능.
pub fn compute_active_features(board: &Board) -> (Vec<usize>, Vec<usize>) {
    let (my_bb, opp_bb) = match board.side_to_move {
        Stone::Black => (&board.black, &board.white),
        Stone::White => (&board.white, &board.black),
    };

    let mut stm = Vec::with_capacity(MAX_ACTIVE_FEATURES);
    let mut nstm = Vec::with_capacity(MAX_ACTIVE_FEATURES);

    let compound_on = compound_enabled();

    // A/B/C/E/F/G: cell 단위 추출.
    for sq in my_bb.iter_ones().chain(opp_bb.iter_ones()) {
        features_from_cell(board, sq, compound_on, &mut stm, &mut nstm);
    }

    // D: Density — global + last_move 기반 local.
    push_density_features(board, my_bb, opp_bb, &mut stm, &mut nstm);

    (stm, nstm)
}

/// 한 cell에서 발생하는 non-density features (A/B/C/E/F 섹션) emit.
/// cell이 빈 칸이면 아무것도 push하지 않음 — caller가 돌 있는 cell만 호출해야
/// 효율적이지만, 빈 cell에 호출해도 결과는 empty (safe).
///
/// 이 함수가 **cell-local하게 self-contained**라는 것이 incremental update의
/// 핵심 invariant: mv 주변 영역의 cell들 각각에 대해 before/after features를
/// 독립 계산하고 delta를 Accumulator에 적용할 수 있다.
#[inline]
pub(crate) fn features_from_cell(
    board: &Board,
    sq: usize,
    compound_on: bool,
    stm: &mut Vec<usize>,
    nstm: &mut Vec<usize>,
) {
    let row = (sq / BOARD_SIZE) as i32;
    let col = (sq % BOARD_SIZE) as i32;

    // stm 관점에서 my/opp 결정
    let (my_bb, opp_bb) = match board.side_to_move {
        Stone::Black => (&board.black, &board.white),
        Stone::White => (&board.white, &board.black),
    };

    let (stones, opponent, persp_mine, persp_opp) = if my_bb.get(sq) {
        (my_bb, opp_bb, 0, 1)
    } else if opp_bb.get(sq) {
        (opp_bb, my_bb, 1, 0)
    } else {
        return; // 빈 cell — A/B/C/E/F/G 모두 emit 안 함
    };

    // A: PS
    stm.push(ps_index(persp_mine, sq));
    nstm.push(ps_index(persp_opp, sq));

    // B: LP-Rich — anchor(line start) 체크, 4방향
    for (dir_idx, &(dr, dc)) in DIR.iter().enumerate() {
        if is_line_start(stones, row, col, dr, dc) {
            let info = scan_line(stones, opponent, row, col, dr, dc);
            let z = zone_for(row, col);
            let len = length_bucket(info.count);
            let op = open_bucket(info.open_front, info.open_back);
            stm.push(lp_rich_index(persp_mine, len, op, dir_idx, z));
            nstm.push(lp_rich_index(persp_opp, len, op, dir_idx, z));
        }
    }

    // C: Compound — 4방향 threats 수집 후 combo (단일 위협은 None 반환됨)
    if compound_on {
        let mut threats = [Threat::None; 4];
        for (di, &(dr, dc)) in DIR.iter().enumerate() {
            let info = scan_line(stones, opponent, row, col, dr, dc);
            let open = info.open_front as u32 + info.open_back as u32;
            threats[di] = classify_threat(info.count, open);
        }
        if let Some(combo) = compound_combo_id(&threats) {
            stm.push(compound_index(persp_mine, combo));
            nstm.push(compound_index(persp_opp, combo));
        }
    }

    // E: Cross-line 3×3 window (stm-perspective + nstm-perspective)
    let stm_cells = collect_3x3(my_bb, opp_bb, row, col);
    let stm_bucket = cross_line_hash(stm_cells);
    stm.push(cross_line_index(0, stm_bucket));
    nstm.push(cross_line_index(1, stm_bucket));

    let nstm_cells = swap_mine_opp(stm_cells);
    let nstm_bucket = cross_line_hash(nstm_cells);
    stm.push(cross_line_index(1, nstm_bucket));
    nstm.push(cross_line_index(0, nstm_bucket));

    // F: Broken / Jump — 4방향, left-anchor dedup
    for (dir_idx, &(dr, dc)) in DIR.iter().enumerate() {
        detect_broken_and_push(
            stones, opponent, row, col, dr, dc, dir_idx, persp_mine, persp_opp, stm, nstm,
        );
    }

    // G section (Pattern4 mini NNUE feature) emit는 v17~v20 실험에서 효과
    // 입증 안 됨 (v18 = v13 동등 53.3%, v19 회귀 26.7%)이라 제거.
    // pattern_table 인프라 + Board::line_pattern_ids 는 보존됨 — 미래 다른
    // 활용 (예: TT key augmentation, move ordering hint)을 위해.
}

/// D 섹션 — 전역 카운트 + last_move 주변 3×3 local density.
fn push_density_features(
    board: &Board,
    my_bb: &crate::board::BitBoard,
    opp_bb: &crate::board::BitBoard,
    stm: &mut Vec<usize>,
    nstm: &mut Vec<usize>,
) {
    let my_count = my_bb.count_ones();
    let opp_count = opp_bb.count_ones();
    push_density(stm, nstm, DENSITY_CAT_MY_COUNT, count_bucket(my_count));
    push_density(stm, nstm, DENSITY_CAT_OPP_COUNT, count_bucket(opp_count));

    let (my_local, opp_local) = local_density(board);
    push_density(stm, nstm, DENSITY_CAT_MY_LOCAL, local_density_bucket(my_local));
    push_density(stm, nstm, DENSITY_CAT_OPP_LOCAL, local_density_bucket(opp_local));

    let legal = (NUM_CELLS as u32).saturating_sub(my_count + opp_count);
    push_density(stm, nstm, DENSITY_CAT_LEGAL, count_bucket(legal));
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

// compute_compound_threats는 features_from_cell에 흡수됨 (cell-centric 리팩토링).

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

/// 진짜 incremental NNUE 평가 상태.
///
/// 한 수 `mv`를 `push_move`하면 mv 주변 ±5 영역의 cell features만 재계산하고
/// 기존 accumulator에 delta를 적용한다. 225-cell 전체 재계산을 피해 leaf
/// 평가가 훨씬 빨라짐. Undo는 snapshot 복원 방식으로 단순·안전하게 처리.
///
/// Invariant:
/// - `cell_features[i]` 는 cell i가 현재 emit 중인 (stm, nstm) features.
///   빈 cell이면 `(vec![], vec![])`.
/// - `density_features` 는 D 섹션 (global) features.
/// - `accumulator` 는 `∪ cell_features + density_features` 를 반영.
pub struct IncrementalEval {
    pub accumulator: Accumulator,
    /// cell 인덱스별 현재 활성 features.
    cell_features: Vec<(Vec<usize>, Vec<usize>)>,
    /// D 섹션 features (global).
    density_features: (Vec<usize>, Vec<usize>),
    stack: Vec<UndoRecord>,
}

struct UndoRecord {
    accumulator: Accumulator,
    cell_changes: Vec<(usize, Vec<usize>, Vec<usize>)>,
    density: (Vec<usize>, Vec<usize>),
}

impl IncrementalEval {
    pub fn new(weights: &NnueWeights) -> Self {
        Self {
            accumulator: Accumulator::new(&weights.feature_bias),
            cell_features: vec![(Vec::new(), Vec::new()); NUM_CELLS],
            density_features: (Vec::new(), Vec::new()),
            stack: Vec::with_capacity(225),
        }
    }

    /// Full state rebuild — 탐색 시작 시 한 번 호출.
    pub fn refresh(&mut self, board: &Board, weights: &NnueWeights) {
        let (my_bb, opp_bb) = match board.side_to_move {
            Stone::Black => (&board.black, &board.white),
            Stone::White => (&board.white, &board.black),
        };
        let compound_on = compound_enabled();

        // cell_features 채우기
        for i in 0..NUM_CELLS {
            self.cell_features[i].0.clear();
            self.cell_features[i].1.clear();
        }
        for sq in my_bb.iter_ones().chain(opp_bb.iter_ones()) {
            let entry = &mut self.cell_features[sq];
            features_from_cell(board, sq, compound_on, &mut entry.0, &mut entry.1);
        }

        // density_features
        self.density_features.0.clear();
        self.density_features.1.clear();
        push_density_features(
            board,
            my_bb,
            opp_bb,
            &mut self.density_features.0,
            &mut self.density_features.1,
        );

        // accumulator 전체 재계산
        let (all_stm, all_nstm) = self.collect_all_features();
        self.accumulator.refresh(weights, &all_stm, &all_nstm);

        self.stack.clear();
    }

    /// 현재 cell_features + density_features를 flatten해서 반환.
    fn collect_all_features(&self) -> (Vec<usize>, Vec<usize>) {
        let mut stm = Vec::with_capacity(MAX_ACTIVE_FEATURES);
        let mut nstm = Vec::with_capacity(MAX_ACTIVE_FEATURES);
        for (s, n) in &self.cell_features {
            stm.extend(s.iter().copied());
            nstm.extend(n.iter().copied());
        }
        stm.extend(self.density_features.0.iter().copied());
        nstm.extend(self.density_features.1.iter().copied());
        (stm, nstm)
    }

    /// `mv`가 방금 board에 적용됐다고 가정. mv 주변 affected cells만 재계산하고
    /// accumulator에 delta 적용.
    ///
    /// **관점 전환**: make_move로 `side_to_move`가 바뀌었으므로 stm/nstm
    /// 라벨링이 반대가 됨. accumulator + 모든 cell_features + density를
    /// swap해서 "현재 side_to_move 관점"으로 재정렬한 뒤 delta 계산.
    pub fn push_move(&mut self, board: &Board, mv: usize, weights: &NnueWeights) {
        // 0. Undo 스냅샷 (swap 이전 상태 저장 — pop에서 복원)
        let mut undo = UndoRecord {
            accumulator: self.accumulator.clone(),
            cell_changes: Vec::new(),
            density: self.density_features.clone(),
        };

        // 1. 관점 swap
        self.accumulator.swap();
        for feats in self.cell_features.iter_mut() {
            std::mem::swap(&mut feats.0, &mut feats.1);
        }
        std::mem::swap(&mut self.density_features.0, &mut self.density_features.1);

        let (my_bb, opp_bb) = match board.side_to_move {
            Stone::Black => (&board.black, &board.white),
            Stone::White => (&board.white, &board.black),
        };
        let compound_on = compound_enabled();

        // 2. Affected cells 계산 + 각 cell의 new features 구해 delta 적용
        let cells = affected_cells(mv);
        let mut new_stm_buf: Vec<usize> = Vec::with_capacity(16);
        let mut new_nstm_buf: Vec<usize> = Vec::with_capacity(16);

        for &c in &cells {
            new_stm_buf.clear();
            new_nstm_buf.clear();
            features_from_cell(board, c, compound_on, &mut new_stm_buf, &mut new_nstm_buf);

            let (old_stm, old_nstm) = &self.cell_features[c];
            if old_stm.as_slice() == new_stm_buf.as_slice()
                && old_nstm.as_slice() == new_nstm_buf.as_slice()
            {
                continue; // 변화 없음
            }

            apply_delta_by_chunks(
                &mut self.accumulator,
                weights,
                &new_stm_buf,
                old_stm,
                &new_nstm_buf,
                old_nstm,
            );

            undo.cell_changes
                .push((c, std::mem::take(&mut self.cell_features[c].0), std::mem::take(&mut self.cell_features[c].1)));
            self.cell_features[c].0 = new_stm_buf.clone();
            self.cell_features[c].1 = new_nstm_buf.clone();
        }

        // 3. Density 재계산 (global)
        let mut new_dens_stm: Vec<usize> = Vec::with_capacity(8);
        let mut new_dens_nstm: Vec<usize> = Vec::with_capacity(8);
        push_density_features(board, my_bb, opp_bb, &mut new_dens_stm, &mut new_dens_nstm);

        if new_dens_stm != self.density_features.0 || new_dens_nstm != self.density_features.1 {
            apply_delta_by_chunks(
                &mut self.accumulator,
                weights,
                &new_dens_stm,
                &self.density_features.0,
                &new_dens_nstm,
                &self.density_features.1,
            );
            self.density_features.0 = new_dens_stm;
            self.density_features.1 = new_dens_nstm;
        }

        self.stack.push(undo);
    }

    /// Undo 마지막 push_move.
    ///
    /// `undo.cell_changes` 는 push 시점의 **post-swap** (새 side_to_move)
    /// 관점의 이전 값이므로, 복원 후 전체 cell_features를 한 번 더 swap해
    /// push **이전** 관점으로 되돌린다. accumulator와 density_features는
    /// push 이전 snapshot 그대로 복원됨.
    pub fn pop_move(&mut self) {
        if let Some(undo) = self.stack.pop() {
            self.accumulator = undo.accumulator;
            self.density_features = undo.density;
            for (c, old_stm, old_nstm) in undo.cell_changes {
                self.cell_features[c].0 = old_stm;
                self.cell_features[c].1 = old_nstm;
            }
            // Perspective 되돌림 (push_move에서 한 swap 상쇄)
            for feats in self.cell_features.iter_mut() {
                std::mem::swap(&mut feats.0, &mut feats.1);
            }
        }
    }

    pub fn eval(&self, weights: &NnueWeights) -> i32 {
        forward(&self.accumulator, weights)
    }
}

/// Multiset diff 기반 incremental accumulator update.
/// `old`에서 `new`로 변화한 features만 add/remove로 추출해 `FeatureDelta`
/// (32 슬롯)에 채워 넣고 `update_incremental`을 호출한다. 한 cell의 feature
/// 변경은 실전에선 10개 미만이라 한 번의 FeatureDelta 호출로 충분하지만,
/// 혹시 넘치면 `MAX_FEATURE_DELTA` 단위로 chunk 나눠서 여러 번 호출.
fn apply_delta_by_chunks(
    acc: &mut Accumulator,
    weights: &NnueWeights,
    new_stm: &[usize],
    old_stm: &[usize],
    new_nstm: &[usize],
    old_nstm: &[usize],
) {
    let (stm_add, stm_rem) = multiset_diff(new_stm, old_stm);
    let (nstm_add, nstm_rem) = multiset_diff(new_nstm, old_nstm);

    const MAX_FD: usize = noru::network::MAX_FEATURE_DELTA;

    // stm/nstm의 add·remove를 동일 chunk로 묶어 처리 (FeatureDelta는 한 쪽만
    // 넘쳐도 실패하므로 보수적으로 작은 쪽 맞춤).
    let max_chunk = MAX_FD;

    let stm_chunks = chunk_pairs(&stm_add, &stm_rem, max_chunk);
    let nstm_chunks = chunk_pairs(&nstm_add, &nstm_rem, max_chunk);
    let n = stm_chunks.len().max(nstm_chunks.len());

    for i in 0..n {
        let (sa, sr) = stm_chunks.get(i).cloned().unwrap_or((&[][..], &[][..]));
        let (na, nr) = nstm_chunks.get(i).cloned().unwrap_or((&[][..], &[][..]));

        let stm_delta = FeatureDelta::from_slices(sa, sr).expect("stm chunk overflow");
        let nstm_delta = FeatureDelta::from_slices(na, nr).expect("nstm chunk overflow");
        acc.update_incremental(weights, &stm_delta, &nstm_delta);
    }
}

/// add/rem slice 쌍을 `max_chunk` 단위로 쪼갠다. 각 chunk가 FeatureDelta의
/// 32 슬롯에 들어갈 수 있도록 add와 rem을 같은 i에서 잘라 나란히 반환.
fn chunk_pairs<'a>(
    add: &'a [usize],
    rem: &'a [usize],
    max_chunk: usize,
) -> Vec<(&'a [usize], &'a [usize])> {
    let n_add = add.len();
    let n_rem = rem.len();
    let chunks = n_add.div_ceil(max_chunk).max(n_rem.div_ceil(max_chunk)).max(1);
    let mut out = Vec::with_capacity(chunks);
    for i in 0..chunks {
        let a_start = (i * max_chunk).min(n_add);
        let a_end = ((i + 1) * max_chunk).min(n_add);
        let r_start = (i * max_chunk).min(n_rem);
        let r_end = ((i + 1) * max_chunk).min(n_rem);
        out.push((&add[a_start..a_end], &rem[r_start..r_end]));
    }
    out
}

/// `new`에 있고 `old`에 없는 항목 (add), `old`에 있고 `new`에 없는 항목 (remove).
/// multiset 개념: 같은 index가 여러 번 나올 수 있음 (compound / density 중복).
fn multiset_diff(new: &[usize], old: &[usize]) -> (Vec<usize>, Vec<usize>) {
    // 일반적인 cell delta는 작음 (<16개)이라 O(n²) 비교로 충분.
    let mut new_count: std::collections::HashMap<usize, i32> = std::collections::HashMap::new();
    for &x in new {
        *new_count.entry(x).or_insert(0) += 1;
    }
    for &x in old {
        *new_count.entry(x).or_insert(0) -= 1;
    }

    let mut add = Vec::new();
    let mut rem = Vec::new();
    for (&idx, &count) in new_count.iter() {
        if count > 0 {
            for _ in 0..count {
                add.push(idx);
            }
        } else if count < 0 {
            for _ in 0..(-count) {
                rem.push(idx);
            }
        }
    }
    (add, rem)
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

    /// Consistency harness — push_move incremental 이 full refresh와 동일
    /// 한 accumulator state를 만드는지 여러 수순에 걸쳐 검증.
    ///
    /// 규칙: 랜덤 가중치로 고정된 보드 수순에 대해 매 make_move 후
    /// incremental eval 값 = 새로 refresh한 eval 값이어야 함.
    #[test]
    fn incremental_matches_full_refresh() {
        // deterministic weight 생성 (zeros로는 모든 eval이 0이라 무의미)
        let mut weights = NnueWeights::zeros(GOMOKU_NNUE_CONFIG);
        let acc_size = GOMOKU_NNUE_CONFIG.accumulator_size;
        for f in 0..TOTAL_FEATURE_SIZE {
            for i in 0..acc_size {
                weights.feature_weights[f][i] =
                    ((f.wrapping_mul(13).wrapping_add(i) % 31) as i16) - 15;
            }
            weights.feature_bias[i_mod(f, acc_size)] =
                ((f.wrapping_mul(7) % 19) as i16) - 9;
        }

        let moves = [
            112, 113, 97, 98, 127, 128, 111, 114, 96, 99, // 대각·직선 혼합
            126, 129, 82, 83, 84, 85, 100, 101, 115, 116,
        ];

        let mut board = Board::new();
        let mut inc = IncrementalEval::new(&weights);
        inc.refresh(&board, &weights);

        let initial = inc.eval(&weights);
        assert_eq!(initial, evaluate(&board, &weights), "refresh mismatch at empty");

        for (i, &mv) in moves.iter().enumerate() {
            if !board.is_empty(mv) {
                continue;
            }
            board.make_move(mv);
            inc.push_move(&board, mv, &weights);

            let inc_val = inc.eval(&weights);
            let full_val = evaluate(&board, &weights);
            assert_eq!(
                inc_val, full_val,
                "mismatch after move {} (ply {}): inc={} full={}",
                mv, i + 1, inc_val, full_val
            );
        }

        // 모든 수 undo해서 다시 초기값으로 돌아가야 함
        for _ in 0..moves.len() {
            board.undo_move();
            inc.pop_move();
            let inc_val = inc.eval(&weights);
            let full_val = evaluate(&board, &weights);
            assert_eq!(inc_val, full_val, "mismatch during undo");
        }
    }

    fn i_mod(f: usize, acc: usize) -> usize {
        f % acc
    }

    /// Far-apart consistency test — 수들이 서로 ±5 cell 밖으로 떨어져 있어
    /// affected_cells에 서로 포함되지 않는 상황을 만들어 perspective swap
    /// 로직이 올바른지 검증. 근처 수만 놓는 기본 테스트로는 이 버그를 못 잡음.
    #[test]
    fn incremental_matches_full_refresh_far_apart() {
        let mut weights = NnueWeights::zeros(GOMOKU_NNUE_CONFIG);
        let acc_size = GOMOKU_NNUE_CONFIG.accumulator_size;
        for f in 0..TOTAL_FEATURE_SIZE {
            for i in 0..acc_size {
                weights.feature_weights[f][i] =
                    ((f.wrapping_mul(17).wrapping_add(i) % 37) as i16) - 18;
            }
        }
        for i in 0..acc_size {
            weights.feature_bias[i] = ((i % 23) as i16) - 11;
        }

        // 보드 극단 cell들 (서로 ±5 영역 밖)
        // 0=(0,0), 14=(0,14), 210=(14,0), 224=(14,14), 112=(7,7), 30=(2,0), 200=(13,5)
        let moves = [0, 224, 14, 210, 112, 30, 200, 58, 101, 150, 7, 217];

        let mut board = Board::new();
        let mut inc = IncrementalEval::new(&weights);
        inc.refresh(&board, &weights);

        assert_eq!(inc.eval(&weights), evaluate(&board, &weights));

        for (i, &mv) in moves.iter().enumerate() {
            if !board.is_empty(mv) {
                continue;
            }
            board.make_move(mv);
            inc.push_move(&board, mv, &weights);

            let inc_val = inc.eval(&weights);
            let full_val = evaluate(&board, &weights);
            assert_eq!(
                inc_val, full_val,
                "far-apart mismatch after move {} (ply {}): inc={} full={}",
                mv, i + 1, inc_val, full_val
            );
        }

        // Undo까지 검증
        for _ in 0..moves.len() {
            board.undo_move();
            inc.pop_move();
            assert_eq!(inc.eval(&weights), evaluate(&board, &weights), "undo mismatch");
        }
    }

    /// Real-weights consistency harness. noru의 i16 accumulator 연산은
    /// saturating이라, incremental(기존 값에 delta 적용)과 full refresh
    /// (bias에서 재합산)가 이론적으로 saturation 영역에서 분기될 수 있다.
    /// 재학습된 weights가 saturation 근접 영역을 건드리는지 자동 적발.
    ///
    /// 평시 `cargo test`에서는 `#[ignore]`로 빠짐 — weights 파일 경로를
    /// 환경변수로 지정해서 `cargo test -- --ignored --exact …` 로 실행.
    /// 기본 경로는 figrid 루트의 `models/gomoku_v14_broken_rapfi_wide.bin`.
    #[test]
    #[ignore = "requires a real NNUE weights file (env NORU_TEST_WEIGHTS or default models/gomoku_v14_broken_rapfi_wide.bin)"]
    fn incremental_matches_full_refresh_real_weights() {
        use crate::board::GameResult;
        use noru::trainer::SimpleRng;

        let path = std::env::var("NORU_TEST_WEIGHTS").unwrap_or_else(|_| {
            let manifest = env!("CARGO_MANIFEST_DIR");
            format!("{}/models/gomoku_v14_broken_rapfi_wide.bin", manifest)
        });
        let data = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("failed to read weights from {path}: {e}"));
        let weights =
            NnueWeights::load_from_bytes(&data, Some(GOMOKU_NNUE_CONFIG.clone()))
                .unwrap_or_else(|e| panic!("load_from_bytes failed for {path}: {e}"));

        // 100 random 160-ply trials — Codex가 수동 harness로 확인한 것과
        // 동일한 커버리지. 재학습 시 saturation divergence 자동 적발.
        let mut rng = SimpleRng::new(2026);
        for trial in 0..100 {
            let mut board = Board::new();
            let mut inc = IncrementalEval::new(&weights);
            inc.refresh(&board, &weights);

            for ply in 0..160 {
                if board.game_result() != GameResult::Ongoing {
                    break;
                }
                let moves = board.candidate_moves();
                if moves.is_empty() {
                    break;
                }
                let mv = moves[rng.next_usize(moves.len())];
                board.make_move(mv);
                inc.push_move(&board, mv, &weights);

                let inc_val = inc.eval(&weights);
                let full_val = evaluate(&board, &weights);
                assert_eq!(
                    inc_val, full_val,
                    "trial {trial} ply {ply} (move {mv}): inc={inc_val} full={full_val}"
                );
            }
        }
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
