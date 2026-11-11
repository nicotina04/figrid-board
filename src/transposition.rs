//! α-β 탐색용 Transposition Table.
//!
//! Zobrist 키로 인덱싱된 sparse table. 기존 VCT 전용 TT와 별도로
//! α-β 노드의 결과를 캐시해 같은 포지션 재평가를 줄인다. Replacement는
//! depth-preferred + always-replace 혼합 (간단한 2-슬롯 bucket).
//!
//! 한 entry는 16 bytes — bucket 2개면 32 bytes. 64K bucket = 2 MB로
//! 시작 (figrid Piskvork submission 메모리 여유 안에).

use super::engine;
use engine::board::Move;
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
    pub key: u64,       // 8 bytes — 충돌 검증용 full key
    pub score: i32,     // 4 bytes
    pub depth: u8,      // 1 byte — 저장 시점의 depth
    pub bound: Bound,   // 1 byte
    pub best_move: u16, // 2 bytes — Move (usize) 의 u16 표현. NUM_CELLS=225라 OK.
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

/// A stored entry: [`TtEntry`] plus the generation of the search that wrote it. Still 16 bytes — the bound and the
/// generation share one byte.
#[derive(Debug, Clone, Copy)]
struct Slot {
    key: u64,
    score: i32,
    depth: u8,
    /// Bits 0..2: [`Bound`]; bits 2..8: generation.
    bound_generation: u8,
    best_move: u16,
}

/// Generations wrap after this many searches. An entry that old then reads as current, which only delays its
/// replacement.
const GENERATION_COUNT: u8 = 64;

impl Slot {
    const EMPTY: Self = Slot {
        key: 0,
        score: 0,
        depth: 0,
        bound_generation: 0,
        best_move: u16::MAX,
    };

    #[inline]
    fn new(entry: TtEntry, generation: u8) -> Self {
        Slot {
            key: entry.key,
            score: entry.score,
            depth: entry.depth,
            bound_generation: entry.bound as u8 | (generation << 2),
            best_move: entry.best_move,
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.best_move == u16::MAX && self.key == 0
    }

    #[inline]
    fn generation(&self) -> u8 {
        self.bound_generation >> 2
    }

    #[inline]
    fn set_generation(&mut self, generation: u8) {
        self.bound_generation = (self.bound_generation & 3) | (generation << 2);
    }

    #[inline]
    fn entry(&self) -> TtEntry {
        TtEntry {
            key: self.key,
            score: self.score,
            depth: self.depth,
            bound: match self.bound_generation & 3 {
                0 => Bound::Exact,
                1 => Bound::Lower,
                _ => Bound::Upper,
            },
            best_move: self.best_move,
        }
    }
}

/// 2-슬롯 bucket (depth-preferred + always-replace).
#[derive(Debug, Clone, Copy)]
struct Bucket {
    depth_pref: Slot,
    always_replace: Slot,
}

const EMPTY_BUCKET: Bucket = Bucket {
    depth_pref: Slot::EMPTY,
    always_replace: Slot::EMPTY,
};

/// 진단용 카운터 — TT 효과 측정. `Cell`이라 `&self probe()`에서도 갱신 가능.
/// 비용은 매 probe/store당 ~1ns. 정상 동작에 영향 없음.
#[derive(Debug, Default, Clone, Copy)]
pub struct TtStats {
    pub probes: u64,               // probe() 호출 총수
    pub hits: u64,                 // 키 일치한 횟수 (cutoff 가능 여부 별개)
    pub stores: u64,               // store() 호출 총수
    pub displaced_depth_pref: u64, // depth_pref slot이 비어있지 않은데 덮어쓴 횟수
    pub stored_to_always: u64, // always_replace slot에 저장한 횟수 (= depth_pref가 더 깊어서 보존)
    /// 저장 시점의 depth 분포 (0..15는 그 depth, 15는 15+ 통합).
    pub depth_hist: [u64; 16],
}

pub struct TranspositionTable {
    buckets: Vec<Bucket>,
    mask: usize, // index = key as usize & mask (buckets.len()는 2^N)
    /// Generation given to new entries; advanced by [`Self::new_search`].
    generation: u8,
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
            generation: 0,
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
            b.depth_pref = Slot::EMPTY;
            b.always_replace = Slot::EMPTY;
        }
    }

    /// Start a new search on a table that keeps its entries: what was written before this call no longer outranks
    /// by depth what is written after it. A table that is cleared before every search never needs this.
    pub fn new_search(&mut self) {
        self.generation = (self.generation + 1) % GENERATION_COUNT;
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
            if !b.depth_pref.is_empty() {
                dp += 1;
            }
            if !b.always_replace.is_empty() {
                ar += 1;
            }
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
            return Some(bucket.depth_pref.entry());
        }
        if bucket.always_replace.key == key && !bucket.always_replace.is_empty() {
            self.hits.set(self.hits.get() + 1);
            return Some(bucket.always_replace.entry());
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
    pub fn store(
        &mut self,
        key: u64,
        score: i32,
        depth: u8,
        bound: Bound,
        best_move: Option<Move>,
    ) {
        self.stores.set(self.stores.get() + 1);
        let depth_idx = (depth as usize).min(15);
        let h = &self.depth_hist[depth_idx];
        h.set(h.get() + 1);
        let generation = self.generation;
        let entry = Slot::new(
            TtEntry {
                key,
                score,
                depth,
                bound,
                best_move: best_move.map(|m| m as u16).unwrap_or(u16::MAX),
            },
            generation,
        );
        let bucket = &mut self.buckets[(key as usize) & self.mask];
        let held = bucket.depth_pref;
        let held_empty = held.is_empty();
        // An entry left by an earlier search (possible only after `new_search`) gives way to another position
        // whatever its depth. Every generation is equal on a table cleared per search, which leaves the rule below
        // as plain depth preference.
        let held_stale = !held_empty && held.generation() != generation;
        if held_empty || depth >= held.depth || (held_stale && held.key != key) {
            if !held_empty {
                self.displaced_depth_pref
                    .set(self.displaced_depth_pref.get() + 1);
                // Push-down: 같은 키 update가 아닐 때만 always-replace로 보존.
                // 같은 키면 새 entry가 갱신본이라 기존을 보존할 이유 없음.
                if held.key != key {
                    bucket.always_replace = held;
                }
            }
            bucket.depth_pref = entry;
        } else {
            self.stored_to_always.set(self.stored_to_always.get() + 1);
            if held_stale {
                // The same position, searched deeper by an earlier search: keep it and count it as current.
                bucket.depth_pref.set_generation(generation);
            }
            bucket.always_replace = entry;
        }
    }

    pub fn capacity(&self) -> usize {
        self.buckets.len() * 2
    }

    /// Prefetch the bucket for `key` into L1 cache. Hides ~50-100 ns of
    /// memory-load latency before the actual `probe()` lands. Call this
    /// in the search loop as soon as you know the *next* zobrist key
    /// (e.g. via `current_zob ^ stone_key ^ ZOBRIST_SIDE`) but before
    /// committing the move + recursing.
    ///
    /// `_MM_HINT_T0` = "load into all cache levels" (most aggressive).
    /// Spurious prefetches are harmless on x86_64; on non-x86 the call
    /// is a no-op.
    #[inline]
    pub fn prefetch(&self, key: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            let idx = (key as usize) & self.mask;
            let bucket_ptr = self.buckets.as_ptr().add(idx);
            _mm_prefetch(bucket_ptr as *const i8, _MM_HINT_T0);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = key;
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

    #[test]
    fn slot_keeps_the_entry_and_stays_sixteen_bytes() {
        assert_eq!(std::mem::size_of::<Slot>(), 16);
        for (bound, generation) in [
            (Bound::Exact, 0),
            (Bound::Lower, 1),
            (Bound::Upper, GENERATION_COUNT - 1),
        ] {
            let entry = TtEntry {
                key: 0x1234_5678_9ABC_DEF0,
                score: -998_877,
                depth: 19,
                bound,
                best_move: 399,
            };
            let slot = Slot::new(entry, generation);
            assert_eq!(slot.generation(), generation);
            let back = slot.entry();
            assert_eq!(back.key, entry.key);
            assert_eq!(back.score, entry.score);
            assert_eq!(back.depth, entry.depth);
            assert_eq!(back.bound, entry.bound);
            assert_eq!(back.best_move, entry.best_move);
        }
    }

    #[test]
    fn an_earlier_search_gives_way_to_another_position() {
        let mut tt = TranspositionTable::new(4);
        let (k1, k2, k3) = (0u64, 0x0000_0001_0000_0000u64, 0x0000_0002_0000_0000u64);
        tt.store(k1, 100, 6, Bound::Exact, Some(1));
        tt.new_search();
        // Shallower, but k1 is from the earlier search: k2 takes the depth-preferred slot and k1 moves down.
        tt.store(k2, 200, 2, Bound::Exact, Some(2));
        assert_eq!(tt.probe(k2).unwrap().score, 200);
        assert_eq!(tt.probe(k1).unwrap().score, 100);
        // Within one search depth preference is unchanged: the shallower k3 lands in the second slot.
        tt.store(k3, 300, 1, Bound::Exact, Some(3));
        assert_eq!(tt.probe(k2).unwrap().score, 200);
        assert_eq!(tt.probe(k3).unwrap().score, 300);
        assert!(tt.probe(k1).is_none());
    }

    #[test]
    fn a_deeper_result_for_the_same_position_survives_a_new_search() {
        let mut tt = TranspositionTable::new(4);
        let (k1, k2) = (0u64, 0x0000_0001_0000_0000u64);
        tt.store(k1, 100, 6, Bound::Exact, Some(1));
        tt.new_search();
        tt.store(k1, 50, 2, Bound::Exact, Some(3));
        let probed = tt.probe(k1).unwrap();
        assert_eq!((probed.score, probed.depth), (100, 6));
        // Kept and counted as current: a shallower store of another position no longer displaces it.
        tt.store(k2, 200, 3, Bound::Exact, Some(2));
        assert_eq!(tt.probe(k1).unwrap().depth, 6);
        assert_eq!(tt.probe(k2).unwrap().score, 200);
    }

    #[test]
    fn clearing_removes_entries_of_every_generation() {
        let mut tt = TranspositionTable::new(4);
        tt.store(7, 1, 3, Bound::Lower, Some(4));
        tt.new_search();
        tt.store(9, 2, 3, Bound::Upper, None);
        tt.clear();
        assert!(tt.probe(7).is_none());
        assert!(tt.probe(9).is_none());
        assert_eq!(tt.occupancy(), (0, 0, 16));
    }
}
