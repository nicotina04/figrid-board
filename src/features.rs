//! 오목 NNUE 피처 인코딩 (4096 슬롯).
//!
//! ```text
//! [0..450)      A. PS (Piece-Square)        : 225 × persp(2) = 450
//! [450..2754)   B. LP-Rich                  : persp(2) × len(8) × open(4) × dir(4) × zone(9) = 2304
//! [2754..2854)  C. Compound Threats         : combo(50) × persp(2) = 100
//! [2854..2904)  D. Density / Mobility       : category(5) × bucket(10) = 50
//! [2904..4096)  E. Reserved (학습 후 확장용) : 1192
//! ```

use noru::config::{Activation, NnueConfig};

pub const BOARD_SIZE: usize = 15;
pub const NUM_SQUARES: usize = BOARD_SIZE * BOARD_SIZE; // 225

// ===== 구간 베이스 인덱스 =====
pub const PS_BASE: usize = 0;
pub const LP_BASE: usize = 450;
pub const COMPOUND_BASE: usize = 2754;
pub const DENSITY_BASE: usize = 2854;
pub const RESERVED_BASE: usize = 2904;
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

// ===== 활성 피처 상한 =====
pub const MAX_ACTIVE_FEATURES: usize = 1024;

pub const GOMOKU_NNUE_CONFIG: NnueConfig = NnueConfig {
    feature_size: TOTAL_FEATURE_SIZE,
    accumulator_size: 512,
    hidden_sizes: &[64],
    activation: Activation::CReLU,
};

// === Compile-time layout sanity ===
const _: () = assert!(LP_BASE == PS_BASE + PS_PER_PERSP * 2);
const _: () = assert!(COMPOUND_BASE == LP_BASE + LP_PER_PERSP * 2);
const _: () = assert!(DENSITY_BASE == COMPOUND_BASE + COMPOUND_PER_PERSP * 2);
const _: () = assert!(RESERVED_BASE == DENSITY_BASE + DENSITY_NUM_CATEGORIES * DENSITY_NUM_BUCKETS);
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
        assert_eq!(RESERVED_BASE, 2904);
        assert!(RESERVED_BASE < TOTAL_FEATURE_SIZE);
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
