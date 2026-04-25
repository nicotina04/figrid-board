//! α-β 탐색용 Transposition Table.
//!
//! Zobrist 키로 인덱싱된 sparse table. 기존 VCT 전용 TT와 별도로
//! α-β 노드의 결과를 캐시해 같은 포지션 재평가를 줄인다. Replacement는
//! depth-preferred + always-replace 혼합 (간단한 2-슬롯 bucket).
//!
//! 한 entry는 16 bytes — bucket 2개면 32 bytes. 64K bucket = 2 MB로
//! 시작 (figrid Piskvork submission 메모리 여유 안에).

use crate::board::Move;

/// TT entry value의 bound 종류.
/// - Exact: PV 노드, 정확한 score
/// - Lower: fail-high (score >= beta), 실제는 더 클 수 있음
/// - Upper: fail-low (score <= alpha), 실제는 더 작을 수 있음
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Bound {
    Exact = 0,
    Lower = 1,
    Upper = 2,
}

/// 한 entry. 16 바이트.
#[derive(Debug, Clone, Copy)]
pub struct TtEntry {
    pub key: u64,        // 8 bytes — 충돌 검증용 full key
    pub score: i32,      // 4 bytes
    pub depth: u8,       // 1 byte — 저장 시점의 depth
    pub bound: Bound,    // 1 byte
    pub best_move: u16,  // 2 bytes — Move (usize) 의 u16 표현. NUM_CELLS=225라 OK.
}

impl TtEntry {
    pub const EMPTY: Self = TtEntry {
        key: 0,
        score: 0,
        depth: 0,
        bound: Bound::Exact,
        best_move: u16::MAX, // sentinel — "no move"
    };

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.best_move == u16::MAX && self.key == 0
    }
}

/// 2-슬롯 bucket (depth-preferred + always-replace).
#[derive(Debug, Clone, Copy)]
struct Bucket {
    depth_pref: TtEntry,
    always_replace: TtEntry,
}

const EMPTY_BUCKET: Bucket = Bucket {
    depth_pref: TtEntry::EMPTY,
    always_replace: TtEntry::EMPTY,
};

pub struct TranspositionTable {
    buckets: Vec<Bucket>,
    mask: usize, // index = key as usize & mask (buckets.len()는 2^N)
}

impl TranspositionTable {
    /// `bucket_count_pow2`: 버킷 수 = 2^N. 예: 16 → 65 536 버킷 = 2 MB.
    pub fn new(bucket_count_pow2: u32) -> Self {
        let n = 1usize << bucket_count_pow2;
        Self {
            buckets: vec![EMPTY_BUCKET; n],
            mask: n - 1,
        }
    }

    /// 모든 슬롯 비우기 (탐색 시작 시).
    pub fn clear(&mut self) {
        for b in self.buckets.iter_mut() {
            b.depth_pref = TtEntry::EMPTY;
            b.always_replace = TtEntry::EMPTY;
        }
    }

    /// Lookup. 키 일치하는 entry 있으면 반환.
    #[inline]
    pub fn probe(&self, key: u64) -> Option<TtEntry> {
        let bucket = &self.buckets[(key as usize) & self.mask];
        if bucket.depth_pref.key == key && !bucket.depth_pref.is_empty() {
            return Some(bucket.depth_pref);
        }
        if bucket.always_replace.key == key && !bucket.always_replace.is_empty() {
            return Some(bucket.always_replace);
        }
        None
    }

    /// Store. depth-preferred slot에 더 깊거나 같으면 저장, 아니면
    /// always-replace slot으로.
    #[inline]
    pub fn store(&mut self, key: u64, score: i32, depth: u8, bound: Bound, best_move: Option<Move>) {
        let entry = TtEntry {
            key,
            score,
            depth,
            bound,
            best_move: best_move.map(|m| m as u16).unwrap_or(u16::MAX),
        };
        let bucket = &mut self.buckets[(key as usize) & self.mask];
        if bucket.depth_pref.is_empty() || depth >= bucket.depth_pref.depth {
            bucket.depth_pref = entry;
        } else {
            bucket.always_replace = entry;
        }
    }

    pub fn capacity(&self) -> usize {
        self.buckets.len() * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_and_probe_roundtrip() {
        let mut tt = TranspositionTable::new(4); // 16 buckets
        assert!(tt.probe(0xDEAD_BEEF).is_none());
        tt.store(0xDEAD_BEEF, 123, 5, Bound::Exact, Some(42));
        let e = tt.probe(0xDEAD_BEEF).expect("should hit");
        assert_eq!(e.score, 123);
        assert_eq!(e.depth, 5);
        assert_eq!(e.bound, Bound::Exact);
        assert_eq!(e.best_move, 42);
    }

    #[test]
    fn depth_preferred_slot_keeps_higher_depth() {
        let mut tt = TranspositionTable::new(4);
        // 같은 bucket index를 강제하기 위해 mask 안 들어가는 상위 비트만 다른 키들.
        let k1 = 0x0000_0000_0000_0000u64; // bucket 0
        let k2 = 0x0000_0001_0000_0000u64; // 같은 bucket 0 (mask=15)
        tt.store(k1, 100, 6, Bound::Exact, Some(1));
        tt.store(k2, 200, 4, Bound::Exact, Some(2));
        // k1 (depth 6)이 depth_pref에 남고, k2 (depth 4)는 always_replace로.
        assert_eq!(tt.probe(k1).unwrap().score, 100);
        assert_eq!(tt.probe(k2).unwrap().score, 200);
        // 더 얕은 깊이로 k1 update 시도 → always_replace로 밀려남
        tt.store(k1, 50, 2, Bound::Exact, Some(3));
        let probed = tt.probe(k1).unwrap();
        // depth_pref는 여전히 100/depth 6
        assert_eq!(probed.score, 100);
    }
}
