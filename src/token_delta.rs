//! Private reversible journal for compact categorical token changes.
//!
//! The journal owns only logical tokens and per-ply deltas. Domain-specific
//! numeric state is updated by a statically dispatched sink.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TokenDelta<T> {
    pub(crate) site: u16,
    pub(crate) lane: u8,
    pub(crate) old: T,
    pub(crate) new: T,
}

impl<T: Copy> TokenDelta<T> {
    #[inline]
    pub(crate) fn reversed(self) -> Self {
        Self {
            site: self.site,
            lane: self.lane,
            old: self.new,
            new: self.old,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TokenDeltaReplay {
    Forward,
    Reverse,
}

pub(crate) trait TokenDeltaSink<T: Copy> {
    fn apply_site(&mut self, site: u16, deltas: &[TokenDelta<T>], replay: TokenDeltaReplay);
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TokenDeltaPop {
    pub(crate) deltas: usize,
    pub(crate) materialized: bool,
}

struct TokenDeltaFrame<T, const MAX_DELTAS: usize> {
    deltas: [TokenDelta<T>; MAX_DELTAS],
    len: usize,
}

impl<T: Copy + Default, const MAX_DELTAS: usize> TokenDeltaFrame<T, MAX_DELTAS> {
    fn new() -> Self {
        Self {
            deltas: [TokenDelta::default(); MAX_DELTAS],
            len: 0,
        }
    }
}

impl<T: Copy, const MAX_DELTAS: usize> TokenDeltaFrame<T, MAX_DELTAS> {
    #[inline]
    fn as_slice(&self) -> &[TokenDelta<T>] {
        &self.deltas[..self.len]
    }
}

/// Fixed-capacity, allocation-free-after-construction reversible token journal.
///
/// Frames below `materialized_depth` have been applied to the numeric sink.
/// Frames from `materialized_depth` through `depth` are logical-only.
pub(crate) struct ReversibleTokenJournal<T, const LANES: usize, const MAX_DELTAS: usize> {
    logical: Vec<[T; LANES]>,
    frames: Vec<TokenDeltaFrame<T, MAX_DELTAS>>,
    seen_sites: Vec<u32>,
    seen_generation: u32,
    depth: usize,
    materialized_depth: usize,
}

impl<T: Copy + Default + Eq, const LANES: usize, const MAX_DELTAS: usize>
    ReversibleTokenJournal<T, LANES, MAX_DELTAS>
{
    pub(crate) fn new(site_count: usize, max_depth: usize) -> Self {
        assert!(
            site_count <= u16::MAX as usize + 1,
            "TokenDelta site index exceeds u16"
        );
        assert!(
            LANES <= u8::MAX as usize + 1,
            "TokenDelta lane index exceeds u8"
        );
        Self {
            logical: vec![[T::default(); LANES]; site_count],
            frames: (0..max_depth).map(|_| TokenDeltaFrame::new()).collect(),
            seen_sites: vec![0; site_count],
            seen_generation: 0,
            depth: 0,
            materialized_depth: 0,
        }
    }

    pub(crate) fn reset(&mut self, tokens: &[[T; LANES]]) {
        assert_eq!(
            tokens.len(),
            self.logical.len(),
            "TokenDelta reset site count mismatch"
        );
        self.logical.copy_from_slice(tokens);
        self.seen_sites.fill(0);
        self.seen_generation = 0;
        self.depth = 0;
        self.materialized_depth = 0;
    }

    /// Record the changed lanes at the supplied sites and advance logical time.
    ///
    /// The capacity check happens before mutation, so an invalid producer
    /// cannot leave the logical mirror half-updated.
    pub(crate) fn push_after(&mut self, new_tokens: &[[T; LANES]], dirty_sites: &[usize]) -> usize {
        assert_eq!(
            new_tokens.len(),
            self.logical.len(),
            "TokenDelta push site count mismatch"
        );
        assert!(
            self.depth < self.frames.len(),
            "TokenDelta journal depth overflow"
        );

        self.seen_generation = self.seen_generation.wrapping_add(1);
        if self.seen_generation == 0 {
            self.seen_sites.fill(0);
            self.seen_generation = 1;
        }
        let generation = self.seen_generation;
        let frame = &mut self.frames[self.depth];
        frame.len = 0;
        for site in dirty_sites.iter().copied() {
            assert!(site < self.logical.len(), "TokenDelta dirty site overflow");
            assert_ne!(
                self.seen_sites[site], generation,
                "TokenDelta dirty sites must be unique"
            );
            self.seen_sites[site] = generation;
            for lane in 0..LANES {
                let old = self.logical[site][lane];
                let new = new_tokens[site][lane];
                if old == new {
                    continue;
                }
                assert!(
                    frame.len < MAX_DELTAS,
                    "TokenDelta frame overflow: more than {MAX_DELTAS}"
                );
                frame.deltas[frame.len] = TokenDelta {
                    site: site as u16,
                    lane: lane as u8,
                    old,
                    new,
                };
                frame.len += 1;
            }
        }
        let delta_count = frame.len;
        for delta in frame.as_slice() {
            self.logical[delta.site as usize][delta.lane as usize] = delta.new;
        }
        self.depth += 1;
        delta_count
    }

    pub(crate) fn materialize_pending<S: TokenDeltaSink<T>>(&mut self, sink: &mut S) {
        while self.materialized_depth < self.depth {
            replay_forward(&self.frames[self.materialized_depth], sink);
            self.materialized_depth += 1;
        }
    }

    pub(crate) fn pop<S: TokenDeltaSink<T>>(&mut self, sink: &mut S) -> Option<TokenDeltaPop> {
        if self.depth == 0 {
            return None;
        }
        self.depth -= 1;
        let frame = &self.frames[self.depth];
        let materialized = self.depth < self.materialized_depth;
        if materialized {
            debug_assert_eq!(
                self.materialized_depth,
                self.depth + 1,
                "materialized frames must form a prefix"
            );
            replay_reverse(frame, sink);
            self.materialized_depth -= 1;
        }
        for delta in frame.as_slice().iter().rev() {
            self.logical[delta.site as usize][delta.lane as usize] = delta.old;
        }
        Some(TokenDeltaPop {
            deltas: frame.len,
            materialized,
        })
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn logical_tokens(&self) -> &[[T; LANES]] {
        &self.logical
    }

    #[inline]
    pub(crate) fn depth(&self) -> usize {
        self.depth
    }

    #[inline]
    pub(crate) fn materialized_depth(&self) -> usize {
        self.materialized_depth
    }
}

#[inline]
fn replay_forward<T: Copy, S: TokenDeltaSink<T>, const MAX_DELTAS: usize>(
    frame: &TokenDeltaFrame<T, MAX_DELTAS>,
    sink: &mut S,
) {
    let deltas = frame.as_slice();
    let mut index = 0;
    while index < deltas.len() {
        let site = deltas[index].site;
        let begin = index;
        while index < deltas.len() && deltas[index].site == site {
            index += 1;
        }
        sink.apply_site(site, &deltas[begin..index], TokenDeltaReplay::Forward);
    }
}

#[inline]
fn replay_reverse<T: Copy, S: TokenDeltaSink<T>, const MAX_DELTAS: usize>(
    frame: &TokenDeltaFrame<T, MAX_DELTAS>,
    sink: &mut S,
) {
    let deltas = frame.as_slice();
    let mut index = deltas.len();
    while index > 0 {
        let site = deltas[index - 1].site;
        let end = index;
        while index > 0 && deltas[index - 1].site == site {
            index -= 1;
        }
        sink.apply_site(site, &deltas[index..end], TokenDeltaReplay::Reverse);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct RecordingSink {
        values: [i32; 3],
        events: Vec<String>,
    }

    impl RecordingSink {
        fn apply_one(&mut self, delta: TokenDelta<u16>) {
            self.values[delta.site as usize] += i32::from(delta.new) - i32::from(delta.old);
            self.events.push(format!(
                "delta:{}:{}:{}>{}",
                delta.site, delta.lane, delta.old, delta.new
            ));
        }
    }

    impl TokenDeltaSink<u16> for RecordingSink {
        fn apply_site(&mut self, site: u16, deltas: &[TokenDelta<u16>], replay: TokenDeltaReplay) {
            self.events.push(format!("begin:{site}"));
            match replay {
                TokenDeltaReplay::Forward => {
                    for &delta in deltas {
                        self.apply_one(delta);
                    }
                }
                TokenDeltaReplay::Reverse => {
                    for &delta in deltas.iter().rev() {
                        self.apply_one(delta.reversed());
                    }
                }
            }
            self.events.push(format!("end:{site}"));
        }
    }

    #[test]
    fn compact_delta_is_eight_bytes_for_u16_tokens() {
        assert_eq!(std::mem::size_of::<TokenDelta<u16>>(), 8);
    }

    #[test]
    fn pending_materialize_and_pop_preserve_prefix_invariant() {
        let initial = [[1u16, 2], [3, 4], [5, 6]];
        let mut journal = ReversibleTokenJournal::<u16, 2, 4>::new(3, 4);
        journal.reset(&initial);
        let mut sink = RecordingSink::default();

        let after_one = [[7, 2], [3, 4], [5, 8]];
        assert_eq!(journal.push_after(&after_one, &[0, 2]), 2);
        assert_eq!(journal.depth(), 1);
        assert_eq!(journal.materialized_depth(), 0);
        assert_eq!(journal.logical_tokens(), after_one);

        let after_two = [[7, 9], [3, 10], [5, 8]];
        assert_eq!(journal.push_after(&after_two, &[0, 1]), 2);
        assert_eq!(journal.depth(), 2);
        assert_eq!(journal.materialized_depth(), 0);

        let popped = journal.pop(&mut sink).unwrap();
        assert_eq!(
            popped,
            TokenDeltaPop {
                deltas: 2,
                materialized: false
            }
        );
        assert!(sink.events.is_empty());
        assert_eq!(journal.logical_tokens(), after_one);

        journal.materialize_pending(&mut sink);
        assert_eq!(journal.materialized_depth(), 1);
        assert_eq!(sink.values, [6, 0, 2]);
        let forward_events = sink.events.clone();
        assert_eq!(
            forward_events,
            [
                "begin:0",
                "delta:0:0:1>7",
                "end:0",
                "begin:2",
                "delta:2:1:6>8",
                "end:2"
            ]
        );

        let after_three = [[11, 12], [3, 4], [5, 8]];
        assert_eq!(journal.push_after(&after_three, &[0]), 2);
        journal.materialize_pending(&mut sink);
        assert_eq!(journal.depth(), 2);
        assert_eq!(journal.materialized_depth(), 2);
        assert_eq!(sink.values, [20, 0, 2]);

        sink.events.clear();
        let popped = journal.pop(&mut sink).unwrap();
        assert!(popped.materialized);
        assert_eq!(journal.logical_tokens(), after_one);
        assert_eq!(sink.values, [6, 0, 2]);
        assert_eq!(
            sink.events,
            ["begin:0", "delta:0:1:12>2", "delta:0:0:11>7", "end:0"]
        );

        journal.pop(&mut sink);
        assert_eq!(journal.logical_tokens(), initial);
        assert_eq!(journal.depth(), 0);
        assert_eq!(journal.materialized_depth(), 0);
        assert_eq!(sink.values, [0, 0, 0]);
    }

    #[test]
    fn reset_reuses_preallocated_depth() {
        let mut journal = ReversibleTokenJournal::<u16, 1, 1>::new(1, 225);
        let mut sink = RecordingSink::default();
        journal.reset(&[[0]]);
        for value in 1..=225u16 {
            assert_eq!(journal.push_after(&[[value]], &[0]), 1);
            journal.materialize_pending(&mut sink);
        }
        assert_eq!(journal.depth(), 225);
        for _ in 0..225 {
            assert!(journal.pop(&mut sink).is_some());
        }
        assert_eq!(journal.logical_tokens(), [[0]]);
        journal.reset(&[[17]]);
        assert_eq!(journal.depth(), 0);
        assert_eq!(journal.materialized_depth(), 0);
        assert_eq!(journal.logical_tokens(), [[17]]);
    }

    #[test]
    fn frame_overflow_is_explicit_before_logical_mutation() {
        let mut journal = ReversibleTokenJournal::<u16, 2, 1>::new(1, 1);
        journal.reset(&[[1, 2]]);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = journal.push_after(&[[3, 4]], &[0]);
        }));
        assert!(result.is_err());
        assert_eq!(journal.logical_tokens(), [[1, 2]]);
        assert_eq!(journal.depth(), 0);
    }

    #[test]
    fn duplicate_dirty_site_is_rejected_before_logical_mutation() {
        let mut journal = ReversibleTokenJournal::<u16, 2, 4>::new(1, 1);
        journal.reset(&[[1, 2]]);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = journal.push_after(&[[3, 4]], &[0, 0]);
        }));
        assert!(result.is_err());
        assert_eq!(journal.logical_tokens(), [[1, 2]]);
        assert_eq!(journal.depth(), 0);
    }
}
