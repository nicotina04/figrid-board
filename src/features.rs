//! 오목 NNUE 피처 인코딩 (4096 슬롯).
//!
//! ```text
//! [0..450)      A. PS (Piece-Square)              : 225 × persp(2) = 450
//! [450..2754)   B. LP-Rich                        : persp(2) × len(8) × open(4) × dir(4) × zone(9) = 2304
//! [2754..2854)  C. Compound Threats               : combo(50) × persp(2) = 100
//! [2854..2904)  D. Density / Mobility             : category(5) × bucket(10) = 50
//! [2904..3416)  E. Cross-line (3×3 local hash)    : bucket(256) × persp(2) = 512
//! [3416..3848)  F. Broken / Jump patterns         : shape(3) × open(2) × dir(4) × zone(9) × persp(2) = 432
//! [3848..4096)  R. Reserved (학습 후 확장용)       : 248
//! ```
//!
//! Pattern4 mini의 G section 통합 시도 (v17~v20)는 v13 대비 효과 입증 안 됨
//! — feature redundancy + tactical decision과 무관한 loss 개선 (Codex
//! 2026-04-26 진단). 인프라 코드 (`pattern_table`, `Board::line_pattern_ids`)
//! 는 보존하지만 NNUE feature space에는 emit하지 않음.

use noru::config::{Activation, NnueConfig};

pub const BOARD_SIZE: usize = 15;
pub const NUM_SQUARES: usize = BOARD_SIZE * BOARD_SIZE; // 225

// ===== 구간 베이스 인덱스 =====
pub const PS_BASE: usize = 0;
pub const LP_BASE: usize = 450;
pub const COMPOUND_BASE: usize = 2754;
pub const DENSITY_BASE: usize = 2854;
pub const CROSS_LINE_BASE: usize = 2904;
pub const BROKEN_BASE: usize = 3416;
pub const RESERVED_BASE: usize = 3848;
pub const TOTAL_FEATURE_SIZE: usize = 4096;

// ===== A. PS =====
pub const PS_PER_PERSP: usize = NUM_SQUARES; // 225
pub const HALF_FEATURE_SIZE: usize = PS_PER_PERSP; // 호환용 별칭

// ===== B. LP-Rich 차원 =====
pub const LP_NUM_LENGTH: usize = 8;
pub const LP_NUM_OPEN: usize = 4;
pub const LP_NUM_DIR: usize = 4;
pub const LP_NUM_ZONE: usize = 9;
pub const LP_PER_PERSP: usize = LP_NUM_LENGTH * LP_NUM_OPEN * LP_NUM_DIR * LP_NUM_ZONE; // 1152

// ===== C. Compound =====
pub const COMPOUND_PER_PERSP: usize = 50;

// ===== D. Density =====
pub const DENSITY_NUM_CATEGORIES: usize = 5;
pub const DENSITY_NUM_BUCKETS: usize = 10;
pub const DENSITY_CAT_MY_COUNT: usize = 0;
pub const DENSITY_CAT_OPP_COUNT: usize = 1;
pub const DENSITY_CAT_MY_LOCAL: usize = 2;
pub const DENSITY_CAT_OPP_LOCAL: usize = 3;
pub const DENSITY_CAT_LEGAL: usize = 4;

// ===== F. Broken / Jump patterns =====
// Gap-aware pattern detection (1칸 빈 칸 허용). scan_line의 연속-only 감지로는
// 놓쳤던 broken three (`_●●_●_`), jump four (`_●●●_●_`), double-broken three
// (한 라인에 broken 패턴이 두 개 포함) 같은 고수 전술 패턴을 잡는다.
// shape 3: broken_three, jump_four, double_broken_three
// open 2: 0 = 양쪽 일부 막힘 (closed form), 1 = 양쪽 모두 열림 (open form)
// dir 4, zone 9는 LP-Rich와 동일한 공간 인덱싱 재사용.
pub const BROKEN_NUM_SHAPES: usize = 3;
pub const BROKEN_NUM_OPEN: usize = 2;
pub const BROKEN_PER_PERSP: usize =
    BROKEN_NUM_SHAPES * BROKEN_NUM_OPEN * LP_NUM_DIR * LP_NUM_ZONE; // 216

pub const BROKEN_SHAPE_THREE: usize = 0;
pub const BROKEN_SHAPE_JUMP_FOUR: usize = 1;
pub const BROKEN_SHAPE_DOUBLE_THREE: usize = 2;

// ===== E. Cross-line 3×3 local window hash =====
//
// Each stone emits a feature for its own 3×3 window (D4-symmetry-
// canonicalized, then feature-hashed into 256 buckets). Captures
// cross-direction interactions (十-shapes, corner squeezes) that the
// line-based LP-Rich encoder misses.
//
// The reserved slot range [2904..3416) is freshly used — in v6-A these
// indices were never activated, so the learned weights there are still
// Kaiming-random. A short fine-tune (PSQ subset, ~5 epochs, lr ~5e-6)
// is required before this feature contributes anything useful.
pub const CROSS_LINE_BUCKETS: usize = 256;
pub const CROSS_LINE_PER_PERSP: usize = CROSS_LINE_BUCKETS;

// ===== 활성 피처 상한 =====
// Pattern4 mini 추가로 stone cell마다 4방향 × 2 perspective = 8 emit.
// 30 stones × 8 = 240 추가. 기존 cap 1536 + 600 여유 = 2400.
// 빈 보드에서도 가장자리 boundary patterns가 emit될 수 있어 보수적으로
// 4096까지 확장.
pub const MAX_ACTIVE_FEATURES: usize = 4096;

// 큰 네트워크(1024 acc + 2×128 hidden, v15/v16) 실험 결과 generalization
// 악화 (v16 아레나 30%, v13의 53%보다 -23pp). 4.5M params / 1.1M samples
// 비율(4:1)에서 overfit 의심. 작은 네트워크(v13 검증)로 회귀.
pub const GOMOKU_NNUE_CONFIG: NnueConfig = NnueConfig {
    feature_size: TOTAL_FEATURE_SIZE,
    accumulator_size: 512,
    // 0.6.7 (2026-04-27): hidden [128,64]→[64] 회귀. v23 (hidden [128,64])
    // Pela 0/5 패배 — heuristic-label PSQ-only 학습이 hidden capacity 늘어난
    // 만큼 "라인 잇기" 선호 over-fit. v14 weights + 0.6.5 search 조합이
    // Pela 1/5로 더 강함 → 안전한 baseline 복귀. 큰 모델 시도는 다른 학습
    // 데이터/loss로 가야 — heuristic label만으론 hidden 키워도 잇기 트랩.
    hidden_sizes: std::borrow::Cow::Borrowed(&[64]),
    activation: Activation::CReLU,
};

// === Compile-time layout sanity ===
const _: () = assert!(LP_BASE == PS_BASE + PS_PER_PERSP * 2);
const _: () = assert!(COMPOUND_BASE == LP_BASE + LP_PER_PERSP * 2);
const _: () = assert!(DENSITY_BASE == COMPOUND_BASE + COMPOUND_PER_PERSP * 2);
const _: () = assert!(CROSS_LINE_BASE == DENSITY_BASE + DENSITY_NUM_CATEGORIES * DENSITY_NUM_BUCKETS);
const _: () = assert!(BROKEN_BASE == CROSS_LINE_BASE + CROSS_LINE_PER_PERSP * 2);
const _: () = assert!(RESERVED_BASE == BROKEN_BASE + BROKEN_PER_PERSP * 2);
const _: () = assert!(RESERVED_BASE <= TOTAL_FEATURE_SIZE);

// ===================================================================
// 인덱스 계산 함수
// ===================================================================

/// PS 인덱스. `perspective` ∈ {0=자기, 1=상대}.
#[inline]
pub fn ps_index(perspective: usize, square: usize) -> usize {
    debug_assert!(perspective < 2);
    debug_assert!(square < NUM_SQUARES);
    PS_BASE + perspective * PS_PER_PERSP + square
}

// pattern_index 함수는 G section 통합 폐기로 더 이상 사용 안 함.
// 인프라 (pattern_table, Board::line_pattern_ids) 는 보존되지만
// NNUE feature 매핑 함수는 제거 — 미래 재도입 시 복원.

/// LP-Rich 인덱스.
#[inline]
pub fn lp_rich_index(
    perspective: usize,
    length: usize,
    open: usize,
    dir: usize,
    zone: usize,
) -> usize {
    debug_assert!(perspective < 2);
    debug_assert!(length < LP_NUM_LENGTH);
    debug_assert!(open < LP_NUM_OPEN);
    debug_assert!(dir < LP_NUM_DIR);
    debug_assert!(zone < LP_NUM_ZONE);
    LP_BASE
        + perspective * LP_PER_PERSP
        + length * (LP_NUM_OPEN * LP_NUM_DIR * LP_NUM_ZONE)
        + open * (LP_NUM_DIR * LP_NUM_ZONE)
        + dir * LP_NUM_ZONE
        + zone
}

/// Compound threat 인덱스. `combo_id` ∈ 0..50. (현재는 슬롯만 예약, 검출 미구현.)
#[inline]
pub fn compound_index(perspective: usize, combo_id: usize) -> usize {
    debug_assert!(perspective < 2);
    debug_assert!(combo_id < COMPOUND_PER_PERSP);
    COMPOUND_BASE + perspective * COMPOUND_PER_PERSP + combo_id
}

/// Density 인덱스. `category` ∈ 0..5, `bucket` ∈ 0..10.
#[inline]
pub fn density_index(category: usize, bucket: usize) -> usize {
    debug_assert!(category < DENSITY_NUM_CATEGORIES);
    debug_assert!(bucket < DENSITY_NUM_BUCKETS);
    DENSITY_BASE + category * DENSITY_NUM_BUCKETS + bucket
}

/// Cross-line feature 인덱스. `perspective` ∈ {0,1}, `bucket` ∈ 0..256.
#[inline]
pub fn cross_line_index(perspective: usize, bucket: usize) -> usize {
    debug_assert!(perspective < 2);
    debug_assert!(bucket < CROSS_LINE_BUCKETS);
    CROSS_LINE_BASE + perspective * CROSS_LINE_PER_PERSP + bucket
}

/// Broken / Jump 패턴 feature 인덱스.
/// - `shape` ∈ 0..3 (BROKEN_SHAPE_*)
/// - `open` ∈ 0..2  (0 = 일부 닫힘, 1 = 양쪽 열림)
/// - `dir`, `zone`: LP-Rich와 동일.
#[inline]
pub fn broken_index(
    perspective: usize,
    shape: usize,
    open: usize,
    dir: usize,
    zone: usize,
) -> usize {
    debug_assert!(perspective < 2);
    debug_assert!(shape < BROKEN_NUM_SHAPES);
    debug_assert!(open < BROKEN_NUM_OPEN);
    debug_assert!(dir < LP_NUM_DIR);
    debug_assert!(zone < LP_NUM_ZONE);
    BROKEN_BASE
        + perspective * BROKEN_PER_PERSP
        + shape * (BROKEN_NUM_OPEN * LP_NUM_DIR * LP_NUM_ZONE)
        + open * (LP_NUM_DIR * LP_NUM_ZONE)
        + dir * LP_NUM_ZONE
        + zone
}

/// 3×3 window 주변 상태 → D4-canonical → 256 bucket 해시.
///
/// Each cell is encoded 2 bits: 0=empty, 1=mine, 2=opp, 3=boundary (out of
/// board). The 9 cells are D4-canonicalized (min over the 8 rotations/
/// reflections) to reduce collisions, then multiplicatively hashed into
/// 256 buckets.
#[inline]
pub fn cross_line_hash(
    my_cells: [u8; 9], // 0=empty, 1=mine, 2=opp, 3=boundary
) -> usize {
    let canonical = d4_canonical_3x3(my_cells);
    let h = canonical.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (h >> (64 - 8)) as usize // top 8 bits → 0..255
}

/// Pack 9 cells into u64 (9 × 2 bits = 18 bits).
#[inline]
fn pack_3x3(c: &[u8; 9]) -> u64 {
    let mut v = 0u64;
    for &cell in c {
        v = (v << 2) | (cell as u64 & 0b11);
    }
    v
}

/// Rotate 3×3 90° clockwise. Indices 0..9 laid out row-major.
#[inline]
fn rotate_3x3(c: [u8; 9]) -> [u8; 9] {
    [c[6], c[3], c[0], c[7], c[4], c[1], c[8], c[5], c[2]]
}

/// Horizontal mirror of 3×3 (swap columns).
#[inline]
fn mirror_3x3(c: [u8; 9]) -> [u8; 9] {
    [c[2], c[1], c[0], c[5], c[4], c[3], c[8], c[7], c[6]]
}

/// D4 group (8 elements) → pick the smallest packed value as canonical.
#[inline]
fn d4_canonical_3x3(cells: [u8; 9]) -> u64 {
    let mut c = cells;
    let mut best = pack_3x3(&c);
    for _ in 0..3 {
        c = rotate_3x3(c);
        let v = pack_3x3(&c);
        if v < best {
            best = v;
        }
    }
    let mut c = mirror_3x3(cells);
    let v = pack_3x3(&c);
    if v < best {
        best = v;
    }
    for _ in 0..3 {
        c = rotate_3x3(c);
        let v = pack_3x3(&c);
        if v < best {
            best = v;
        }
    }
    best
}

// ===================================================================
// 분류 함수
// ===================================================================

/// 연속 돌 수 → length bucket.
/// 0: solo, 1: 2연, 2: 3연, 3: 4연, 4: 5연, 5: 6+(오버라인), 6,7: reserved.
#[inline]
pub fn length_bucket(count: u32) -> usize {
    match count {
        1 => 0,
        2 => 1,
        3 => 2,
        4 => 3,
        5 => 4,
        _ => 5,
    }
}

/// (open_front, open_back) → open bucket.
/// 0: 양쪽막힘, 1: 앞만열림, 2: 뒤만열림, 3: 양쪽열림.
#[inline]
pub fn open_bucket(open_front: bool, open_back: bool) -> usize {
    (open_front as usize) | ((open_back as usize) << 1)
}

/// (row, col) → zone (3×3 분할). 보드 15×15 → 각 zone은 5×5.
#[inline]
pub fn zone_for(row: i32, col: i32) -> usize {
    let r = (row.clamp(0, BOARD_SIZE as i32 - 1) / 5) as usize;
    let c = (col.clamp(0, BOARD_SIZE as i32 - 1) / 5) as usize;
    r * 3 + c
}

/// 자기/상대 돌 카운트 → bucket.
#[inline]
pub fn count_bucket(n: u32) -> usize {
    match n {
        0 => 0,
        1..=3 => 1,
        4..=7 => 2,
        8..=15 => 3,
        16..=30 => 4,
        31..=60 => 5,
        61..=100 => 6,
        101..=150 => 7,
        151..=200 => 8,
        _ => 9,
    }
}

/// 0..9 범위 카운트 → bucket (3×3 주변 밀도용; 그대로 매핑).
#[inline]
pub fn local_density_bucket(n: u32) -> usize {
    (n as usize).min(DENSITY_NUM_BUCKETS - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_sizes() {
        assert_eq!(PS_PER_PERSP * 2, LP_BASE);
        assert_eq!(LP_PER_PERSP, 1152);
        assert_eq!(LP_PER_PERSP * 2 + LP_BASE, COMPOUND_BASE);
        assert_eq!(CROSS_LINE_BASE, 2904);
        assert_eq!(CROSS_LINE_PER_PERSP * 2 + CROSS_LINE_BASE, BROKEN_BASE);
        assert_eq!(BROKEN_BASE, 3416);
        assert_eq!(BROKEN_PER_PERSP, 216);
        assert_eq!(BROKEN_PER_PERSP * 2 + BROKEN_BASE, RESERVED_BASE);
        assert_eq!(RESERVED_BASE, 3848);
        assert!(RESERVED_BASE < TOTAL_FEATURE_SIZE);
        assert_eq!(TOTAL_FEATURE_SIZE, 4096);
    }

    #[test]
    fn ps_indexing() {
        assert_eq!(ps_index(0, 0), 0);
        assert_eq!(ps_index(0, 224), 224);
        assert_eq!(ps_index(1, 0), 225);
        assert_eq!(ps_index(1, 224), 449);
    }

    #[test]
    fn lp_rich_in_range_and_unique() {
        let mut seen = std::collections::HashSet::new();
        for p in 0..2 {
            for l in 0..LP_NUM_LENGTH {
                for o in 0..LP_NUM_OPEN {
                    for d in 0..LP_NUM_DIR {
                        for z in 0..LP_NUM_ZONE {
                            let idx = lp_rich_index(p, l, o, d, z);
                            assert!(idx >= LP_BASE && idx < COMPOUND_BASE);
                            assert!(seen.insert(idx), "dup at {p},{l},{o},{d},{z}");
                        }
                    }
                }
            }
        }
        assert_eq!(seen.len(), LP_PER_PERSP * 2);
    }

    #[test]
    fn density_index_in_range() {
        for c in 0..DENSITY_NUM_CATEGORIES {
            for b in 0..DENSITY_NUM_BUCKETS {
                let idx = density_index(c, b);
                assert!(idx >= DENSITY_BASE && idx < RESERVED_BASE);
            }
        }
    }

    #[test]
    fn zone_grid() {
        assert_eq!(zone_for(0, 0), 0);
        assert_eq!(zone_for(7, 7), 4);
        assert_eq!(zone_for(14, 14), 8);
        assert_eq!(zone_for(0, 14), 2);
        assert_eq!(zone_for(14, 0), 6);
    }

    #[test]
    fn open_bucket_combinations() {
        assert_eq!(open_bucket(false, false), 0);
        assert_eq!(open_bucket(true, false), 1);
        assert_eq!(open_bucket(false, true), 2);
        assert_eq!(open_bucket(true, true), 3);
    }

    #[test]
    fn length_bucket_mapping() {
        assert_eq!(length_bucket(1), 0);
        assert_eq!(length_bucket(5), 4);
        assert_eq!(length_bucket(6), 5);
        assert_eq!(length_bucket(99), 5);
    }
}
