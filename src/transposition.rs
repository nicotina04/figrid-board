//! α-β 탐색용 Transposition Table.
//!
//! Zobrist 키로 인덱싱된 sparse table. 기존 VCT 전용 TT와 별도로
//! α-β 노드의 결과를 캐시해 같은 포지션 재평가를 줄인다. Replacement는
//! depth-preferred + always-replace 혼합 (간단한 2-슬롯 bucket).
//!
//! 한 entry는 16 bytes — bucket 2개면 32 bytes. 64K bucket = 2 MB로
//! 시작 (figrid Piskvork submission 메모리 여유 안에).

use crate::board::Move;
use std::cell::Cell;

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

/// 진단용 카운터 — TT 효과 측정. `Cell`이라 `&self probe()`에서도 갱신 가능.
/// 비용은 매 probe/store당 ~1ns. 정상 동작에 영향 없음.
#[derive(Debug, Default, Clone, Copy)]
pub struct TtStats {
    pub probes: u64,            // probe() 호출 총수
    pub hits: u64,               // 키 일치한 횟수 (cutoff 가능 여부 별개)
    pub stores: u64,             // store() 호출 총수
    pub displaced_depth_pref: u64, // depth_pref slot이 비어있지 않은데 덮어쓴 횟수
    pub stored_to_always: u64,   // always_replace slot에 저장한 횟수 (= depth_pref가 더 깊어서 보존)
    /// 저장 시점의 depth 분포 (0..15는 그 depth, 15는 15+ 통합).
    pub depth_hist: [u64; 16],
}

pub struct TranspositionTable {
    buckets: Vec<Bucket>,
    mask: usize, // index = key as usize & mask (buckets.len()는 2^N)
    probes: Cell<u64>,
    hits: Cell<u64>,
    stores: Cell<u64>,
    displaced_depth_pref: Cell<u64>,
    stored_to_always: Cell<u64>,
    depth_hist: [Cell<u64>; 16],
}

impl TranspositionTable {
    /// `bucket_count_pow2`: 버킷 수 = 2^N. 예: 16 → 65 536 버킷 = 2 MB.
    pub fn new(bucket_count_pow2: u32) -> Self {
        let n = 1usize << bucket_count_pow2;
        Self {
            buckets: vec![EMPTY_BUCKET; n],
            mask: n - 1,
            probes: Cell::new(0),
            hits: Cell::new(0),
            stores: Cell::new(0),
            displaced_depth_pref: Cell::new(0),
            stored_to_always: Cell::new(0),
            depth_hist: Default::default(),
        }
    }

    /// 모든 슬롯 비우기 (탐색 시작 시). 카운터는 보존 — search() 호출 누적.
    pub fn clear(&mut self) {
        for b in self.buckets.iter_mut() {
            b.depth_pref = TtEntry::EMPTY;
            b.always_replace = TtEntry::EMPTY;
        }
    }

    /// 진단 카운터 리셋 (한 search() 단위 측정 시작 시).
    pub fn reset_stats(&self) {
        self.probes.set(0);
        self.hits.set(0);
        self.stores.set(0);
        self.displaced_depth_pref.set(0);
        self.stored_to_always.set(0);
        for c in &self.depth_hist {
            c.set(0);
        }
    }

    pub fn stats(&self) -> TtStats {
        let mut hist = [0u64; 16];
        for (i, c) in self.depth_hist.iter().enumerate() {
            hist[i] = c.get();
        }
        TtStats {
            probes: self.probes.get(),
            hits: self.hits.get(),
            stores: self.stores.get(),
            displaced_depth_pref: self.displaced_depth_pref.get(),
            stored_to_always: self.stored_to_always.get(),
            depth_hist: hist,
        }
    }

    /// Bucket 점유율 — `(non-empty depth_pref 수, non-empty always_replace 수, 총 bucket 수)`.
    /// search 끝난 직후 호출하면 saturation 측정 가능.
    pub fn occupancy(&self) -> (usize, usize, usize) {
        let mut dp = 0usize;
        let mut ar = 0usize;
        for b in &self.buckets {
            if !b.depth_pref.is_empty() { dp += 1; }
            if !b.always_replace.is_empty() { ar += 1; }
        }
        (dp, ar, self.buckets.len())
    }

    /// Lookup. 키 일치하는 entry 있으면 반환.
    #[inline]
    pub fn probe(&self, key: u64) -> Option<TtEntry> {
        self.probes.set(self.probes.get() + 1);
        let bucket = &self.buckets[(key as usize) & self.mask];
        if bucket.depth_pref.key == key && !bucket.depth_pref.is_empty() {
            self.hits.set(self.hits.get() + 1);
            return Some(bucket.depth_pref);
        }
        if bucket.always_replace.key == key && !bucket.always_replace.is_empty() {
            self.hits.set(self.hits.get() + 1);
            return Some(bucket.always_replace);
        }
        None
    }

    /// Store. depth-preferred slot에 더 깊거나 같으면 저장, 아니면
    /// always-replace slot으로.
    ///
    /// Push-down 정책 (0.6.2~): depth_pref에 새 entry가 들어가야 할 때
    /// 기존 entry가 비어있지 않으면 그걸 always-replace로 밀어낸다. 이전
    /// 정책은 displaced된 entry를 그냥 버려서 always-replace slot이 거의
    /// 비어있는 채로 활용 안 됨 (진단 2026-04-27: always 사용률 2.1%).
    /// Push-down 후엔 always-replace가 "두 번째로 좋은 entry" 역할을 함.
    #[inline]
    pub fn store(&mut self, key: u64, score: i32, depth: u8, bound: Bound, best_move: Option<Move>) {
        self.stores.set(self.stores.get() + 1);
        let depth_idx = (depth as usize).min(15);
        let h = &self.depth_hist[depth_idx];
        h.set(h.get() + 1);
        let entry = TtEntry {
            key,
            score,
            depth,
            bound,
            best_move: best_move.map(|m| m as u16).unwrap_or(u16::MAX),
        };
        let bucket = &mut self.buckets[(key as usize) & self.mask];
        if bucket.depth_pref.is_empty() || depth >= bucket.depth_pref.depth {
            if !bucket.depth_pref.is_empty() {
                self.displaced_depth_pref.set(self.displaced_depth_pref.get() + 1);
                // Push-down: 같은 키 update가 아닐 때만 always-replace로 보존.
                // 같은 키면 새 entry가 갱신본이라 기존을 보존할 이유 없음.
                if bucket.depth_pref.key != key {
                    bucket.always_replace = bucket.depth_pref;
                }
            }
            bucket.depth_pref = entry;
        } else {
            self.stored_to_always.set(self.stored_to_always.get() + 1);
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
