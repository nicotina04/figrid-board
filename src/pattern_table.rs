//! Pattern4 mini — 11-cell line window를 lossless ID로 압축하는 테이블.
//!
//! Rapfi식 Pattern4 인프라의 mini 버전. 한 cell의 한 방향에서 ±5 = 11칸의
//! 시퀀스를 정규화해 unique pattern ID를 부여하고, 보드 안에 (cell, dir)
//! 별 pattern_id 상태를 유지하면 매 수마다 영향받는 라인의 ID만 lookup으로
//! 갱신할 수 있어 region recompute가 사라진다.
//!
//! 이 모듈은 그 인프라의 **첫 단계**: 패턴 표현, canonicalize, packed
//! encoding, 그리고 모든 가능 패턴의 enumeration. NNUE 통합과 보드 상태
//! 유지는 후속 단계에서 추가된다.

use std::collections::HashMap;
use std::sync::OnceLock;

/// 11-cell line window. cell 값:
/// - 0 = empty
/// - 1 = mine (현재 stm 관점)
/// - 2 = opp
/// - 3 = boundary (board 밖)
///
/// 인덱스 0이 라인의 한 쪽 끝, 인덱스 5가 anchor cell, 10이 반대 끝.
pub type LineWindow = [u8; 11];

/// Cell 값 enum (가독성용). u8와 1:1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Cell {
    Empty = 0,
    Mine = 1,
    Opp = 2,
    Boundary = 3,
}

impl From<u8> for Cell {
    #[inline]
    fn from(v: u8) -> Self {
        match v {
            0 => Cell::Empty,
            1 => Cell::Mine,
            2 => Cell::Opp,
            3 => Cell::Boundary,
            _ => panic!("invalid cell value"),
        }
    }
}

/// 11-cell window를 22-bit u32로 packed. 각 cell 2 bit.
/// 인덱스 0이 high bits, 인덱스 10이 low bits.
#[inline]
pub fn pack_window(w: &LineWindow) -> u32 {
    let mut packed = 0u32;
    for &c in w {
        debug_assert!(c < 4);
        packed = (packed << 2) | (c as u32);
    }
    packed
}

/// packed u32에서 LineWindow 복원.
/// 인덱스 0이 high bits, 인덱스 10이 low bits — pack_window의 역.
#[inline]
pub fn unpack_window(packed: u32) -> LineWindow {
    let mut w = [0u8; 11];
    for i in 0..11 {
        let shift = (10 - i) * 2;
        w[i] = ((packed >> shift) & 0b11) as u8;
    }
    w
}

/// 좌우 reflection 정규화. canonical = min(w, reverse(w)) packed.
///
/// 색 swap (mine ↔ opp) 은 별도 perspective 차원으로 처리하므로 여기선
/// 적용하지 않는다. 즉 같은 패턴의 mine/opp 버전은 다른 ID를 갖는다.
#[inline]
pub fn canonicalize(w: &LineWindow) -> LineWindow {
    let reversed: LineWindow = std::array::from_fn(|i| w[10 - i]);
    let p1 = pack_window(w);
    let p2 = pack_window(&reversed);
    if p1 <= p2 {
        *w
    } else {
        reversed
    }
}

/// 패턴이 보드에서 실제 등장 가능한가?
/// 규칙: boundary는 양 끝에서만 연속으로 나타날 수 있다. 보드 안쪽에서는
/// boundary가 등장하지 않는다.
///
/// 즉 시퀀스가 [B*, valid*, B*] 형태여야 함 (B는 boundary, valid는 0/1/2).
/// 양 끝의 B 길이 합이 11을 넘지 않으면서 가운데가 모두 non-boundary.
pub fn is_realizable(w: &LineWindow) -> bool {
    // 왼쪽 boundary 길이
    let mut left_b = 0;
    while left_b < 11 && w[left_b] == 3 {
        left_b += 1;
    }
    // 오른쪽 boundary 길이
    let mut right_b = 0;
    while right_b < 11 && w[10 - right_b] == 3 {
        right_b += 1;
    }
    if left_b + right_b > 11 {
        return false;
    }
    // 가운데(left_b..11-right_b)에 boundary 없어야 함
    for i in left_b..(11 - right_b) {
        if w[i] == 3 {
            return false;
        }
    }
    true
}

/// Pattern ID type. canonical pattern 수가 65k 초과 (실측 ~수십만)이라
/// u16으로는 부족. u32 사용.
pub type PatternId = u32;

/// 모든 가능한 11-cell pattern을 enumerate해 canonical packed → ID로 매핑.
/// `is_realizable` 통과 + canonical form만 unique ID.
///
/// 결과 테이블 크기 실측: 수십만 pattern (4^11 = 4M raw 중 실현 가능 +
/// canonical 정규화 후 unique). 모델 weights table 크기 영향 큼 — NNUE
/// 통합 시 frequent pattern만 keep하는 추가 필터 필요할 수 있다.
pub fn enumerate_patterns() -> HashMap<u32, PatternId> {
    let mut table: HashMap<u32, PatternId> = HashMap::new();
    let mut next_id: PatternId = 0;

    // 4^11 = 4194304 가능. release 빌드에서 ~수 초.
    for raw in 0..(1u32 << 22) {
        let w = unpack_window(raw);
        if !is_realizable(&w) {
            continue;
        }
        let canonical = canonicalize(&w);
        let canonical_packed = pack_window(&canonical);
        if !table.contains_key(&canonical_packed) {
            table.insert(canonical_packed, next_id);
            next_id += 1;
        }
    }
    table
}

/// 우리 NNUE에서 실제로 추적하는 mapped pattern ID 수.
/// Top 16384 = 학습 데이터에서 99.24% 커버 + rare bucket 1개 = 16385.
// Top-K를 4096으로 축소 — 16384에서 17:1 (params:samples) ratio 였던 것을
// 4:1 (v13 수준)로 정상화. coverage는 99.24% → 약 97.5% (top 5K가 97.81%).
// 거의 lossless이면서 Pattern4 weights 대부분이 학습 가능 영역에 진입.
pub const PATTERN_TOP_K: usize = 4096;
pub const PATTERN_RARE_ID: u16 = PATTERN_TOP_K as u16; // = 4096, "그 외" bucket
pub const PATTERN_NUM_IDS: usize = PATTERN_TOP_K + 1;  // = 4097

/// `pattern_freq_stats --dump-top-k 16384` 가 만든 binary embed.
/// 16384 × u32 little-endian = 65 536 bytes. canonical packed value를
/// 빈도 내림차순으로 나열.
const TOPK_BYTES: &[u8] = include_bytes!("../data/topk.bin");

/// 4M 엔트리 dense lookup table. raw packed window → mapped pattern ID
/// (u16). Top 16K canonical packed에 들어가는 packed/reflection은 0..16383
/// 값, 그 외 realizable raw는 PATTERN_RARE_ID (16384).
///
/// 메모리 8 MB (u16 × 4M). lookup은 단순 array index.
/// OnceLock 으로 첫 호출 시 build (~수 초).
fn dense_mapped_table() -> &'static Vec<u16> {
    static TABLE: OnceLock<Vec<u16>> = OnceLock::new();
    TABLE.get_or_init(build_dense_mapped_table)
}

fn build_dense_mapped_table() -> Vec<u16> {
    let n = 1usize << 22;
    let mut t = vec![PATTERN_RARE_ID; n];

    // Top 16K canonical packed → mapped ID 0..16383
    let mut canonical_to_mapped: HashMap<u32, u16> = HashMap::with_capacity(PATTERN_TOP_K);
    debug_assert!(TOPK_BYTES.len() == PATTERN_TOP_K * 4);
    for (i, chunk) in TOPK_BYTES.chunks(4).enumerate() {
        let p = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        canonical_to_mapped.insert(p, i as u16);
    }

    // 모든 raw → canonicalize → topk 안이면 그 mapped id, 아니면 rare.
    for raw in 0..(n as u32) {
        let w = unpack_window(raw);
        if !is_realizable(&w) {
            continue; // unreachable raw → 기본값 RARE 유지 (사실 도달 안 함)
        }
        let canonical_packed = pack_window(&canonicalize(&w));
        let mapped = canonical_to_mapped
            .get(&canonical_packed)
            .copied()
            .unwrap_or(PATTERN_RARE_ID);
        t[raw as usize] = mapped;
    }

    t
}

/// raw packed 11-cell window → mapped pattern ID (0..16384, 16384=rare).
/// Caller가 read_window로 만든 packed를 그대로 넘기면 O(1) lookup.
#[inline]
pub fn lookup_mapped_id(packed: u32) -> u16 {
    debug_assert!((packed as usize) < (1usize << 22));
    dense_mapped_table()[packed as usize]
}

/// 모든 빈 셀의 LineWindow `[0; 11]` 의 mapped ID. 보드 안쪽 빈 cell의
/// 초기값 — 우리 freq 측정에서 11.94% 빈도 1위라 mapped id 0으로 매핑됨.
/// Board::new() 의 default fill 에 사용.
#[inline]
pub fn empty_pattern_mapped_id() -> u16 {
    let all_empty: LineWindow = [0u8; 11];
    lookup_mapped_id(pack_window(&all_empty))
}

/// mine ↔ opp swap 후의 mapped pattern ID 반환.
///
/// `line_pattern_ids` 는 black-relative storage이므로 stm == White일 때
/// stm-perspective feature를 emit하려면 ID를 swap해야 한다. 이 lookup이
/// O(1)로 가능하도록 16385-entry swap table을 미리 빌드.
///
/// rare bucket(16384)은 swap도 rare로 매핑.
fn swap_table() -> &'static [u16; PATTERN_NUM_IDS] {
    static TABLE: OnceLock<Box<[u16; PATTERN_NUM_IDS]>> = OnceLock::new();
    TABLE.get_or_init(|| Box::new(build_swap_table()))
}

fn build_swap_table() -> [u16; PATTERN_NUM_IDS] {
    let mut t = [PATTERN_RARE_ID; PATTERN_NUM_IDS];

    // Top 16K canonical packed의 swap → 그 canonical의 mapped ID.
    // 1) canonical_packed → mapped 사전 재구성.
    let mut canonical_to_mapped: HashMap<u32, u16> =
        HashMap::with_capacity(PATTERN_TOP_K);
    for (i, chunk) in TOPK_BYTES.chunks(4).enumerate() {
        let p = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        canonical_to_mapped.insert(p, i as u16);
    }

    // 2) 각 mapped id의 canonical → mine/opp swap → re-canonicalize → mapped.
    for (i, chunk) in TOPK_BYTES.chunks(4).enumerate() {
        let canonical_packed =
            u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let w = unpack_window(canonical_packed);
        let swapped: LineWindow = std::array::from_fn(|j| match w[j] {
            1 => 2,
            2 => 1,
            x => x,
        });
        let swapped_canonical = pack_window(&canonicalize(&swapped));
        let mapped = canonical_to_mapped
            .get(&swapped_canonical)
            .copied()
            .unwrap_or(PATTERN_RARE_ID);
        t[i] = mapped;
    }
    // rare ↔ rare
    t[PATTERN_RARE_ID as usize] = PATTERN_RARE_ID;
    t
}

/// mine/opp swap된 perspective의 mapped pattern ID.
#[inline]
pub fn swap_mapped_id(id: u16) -> u16 {
    debug_assert!((id as usize) < PATTERN_NUM_IDS);
    swap_table()[id as usize]
}

/// 보드 상태(stones bitboard 두 개)에서 (row, col, dir) 의 11-cell window를
/// 읽어내는 helper. mine/opp 관점에 따라 cell 값 결정.
#[inline]
pub fn read_window(
    mine: &crate::board::BitBoard,
    opp: &crate::board::BitBoard,
    row: i32,
    col: i32,
    dr: i32,
    dc: i32,
) -> LineWindow {
    use crate::board::BOARD_SIZE;
    let mut w = [3u8; 11]; // 기본 boundary
    for off in -5i32..=5 {
        let r = row + dr * off;
        let c = col + dc * off;
        if r < 0 || r >= BOARD_SIZE as i32 || c < 0 || c >= BOARD_SIZE as i32 {
            continue;
        }
        let idx = (r as usize) * BOARD_SIZE + c as usize;
        let slot = (off + 5) as usize;
        if mine.get(idx) {
            w[slot] = 1;
        } else if opp.get(idx) {
            w[slot] = 2;
        } else {
            w[slot] = 0;
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        let w: LineWindow = [3, 3, 0, 1, 2, 1, 0, 2, 0, 3, 3];
        let packed = pack_window(&w);
        let unpacked = unpack_window(packed);
        assert_eq!(w, unpacked);
    }

    #[test]
    fn canonicalize_idempotent() {
        let w: LineWindow = [3, 3, 0, 1, 2, 1, 0, 2, 0, 3, 3];
        let c1 = canonicalize(&w);
        let c2 = canonicalize(&c1);
        assert_eq!(c1, c2);
    }

    #[test]
    fn canonicalize_pairs_with_reverse() {
        let w: LineWindow = [3, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0];
        let reversed: LineWindow = std::array::from_fn(|i| w[10 - i]);
        assert_eq!(canonicalize(&w), canonicalize(&reversed));
    }

    #[test]
    fn realizability_rejects_internal_boundary() {
        // boundary가 가운데 끼어있으면 안 됨
        let bad: LineWindow = [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0];
        assert!(!is_realizable(&bad));

        // 양 끝의 boundary는 OK
        let good: LineWindow = [3, 3, 0, 1, 2, 1, 0, 0, 0, 3, 3];
        assert!(is_realizable(&good));

        // 모두 빈 칸도 OK
        let empty: LineWindow = [0; 11];
        assert!(is_realizable(&empty));

        // anchor (slot 5) 가 항상 보드 안이므로 모두 boundary는 불가능.
        // anchor 좌우 5칸씩 합 11 → boundary 합이 11이면 가운데도 boundary
        // 되어야 하는데 그건 가능 (left_b + right_b > 11 인 경우만 reject).
        let all_b: LineWindow = [3; 11];
        // left_b=11, right_b=11, 합 22 > 11 → reject
        assert!(!is_realizable(&all_b));

        // 한쪽 끝에서 anchor에 다다르기 직전까지 boundary
        // [3,3,3,3,3,0,0,0,0,0,0] — left_b=5, right_b=0, 합 5, OK
        let edge: LineWindow = [3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0];
        assert!(is_realizable(&edge));
    }

    #[test]
    fn enumerate_patterns_smoke() {
        // 너무 무거운 4M 순회는 release 빌드에서만 빠름.
        // 여기선 enumerate가 결정적임을 적은 sanity로 확인.
        let table = enumerate_patterns();
        // canonical은 reverse 적용해도 같은 ID여야 함
        let w1: LineWindow = [3, 3, 0, 1, 2, 0, 0, 0, 0, 3, 3];
        let w2: LineWindow = std::array::from_fn(|i| w1[10 - i]);
        let id1 = table[&pack_window(&canonicalize(&w1))];
        let id2 = table[&pack_window(&canonicalize(&w2))];
        assert_eq!(id1, id2, "left-right reflection should share canonical ID");

        // 패턴 수가 합리적 범위인지
        let n = table.len();
        eprintln!("[pattern_table] enumerated {n} canonical patterns");
        assert!(n > 1000, "too few patterns: {n}");
        assert!(n < 200_000, "too many patterns (u16 overflow risk): {n}");
    }

    #[test]
    fn read_window_centers_on_anchor() {
        use crate::board::BitBoard;
        let mut mine = BitBoard::EMPTY;
        let mut opp = BitBoard::EMPTY;
        mine.set(7 * 15 + 7); // (7,7) mine
        opp.set(7 * 15 + 8); // (7,8) opp

        let w = read_window(&mine, &opp, 7, 7, 0, 1);
        // anchor (7,7) at slot 5, (7,8) at slot 6
        assert_eq!(w[5], 1, "anchor should be mine");
        assert_eq!(w[6], 2, "right neighbor should be opp");
        assert_eq!(w[4], 0, "left neighbor empty");

        // 보드 가장자리: (0, 0) 에서 가로 방향 → 왼쪽 5칸 모두 boundary
        let mine2 = BitBoard::EMPTY;
        let opp2 = BitBoard::EMPTY;
        let w2 = read_window(&mine2, &opp2, 0, 0, 0, 1);
        for i in 0..5 {
            assert_eq!(w2[i], 3, "out-of-board should be boundary at slot {i}");
        }
        assert_eq!(w2[5], 0, "anchor (0,0) is empty");
    }
}
