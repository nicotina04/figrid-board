use std::error::Error;
use std::fmt;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TokenDelta<T> {
    site: u16,
    lane: u8,
    old: T,
    new: T,
}

impl<T: Copy> TokenDelta<T> {
    #[inline]
    pub fn site(self) -> u16 {
        self.site
    }

    #[inline]
    pub fn lane(self) -> u8 {
        self.lane
    }

    #[inline]
    pub fn old(self) -> T {
        self.old
    }

    #[inline]
    pub fn new_token(self) -> T {
        self.new
    }

    #[inline]
    pub fn reversed(self) -> Self {
        Self {
            site: self.site,
            lane: self.lane,
            old: self.new,
            new: self.old,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenDeltaReplay {
    Forward,
    Reverse,
}

/// Applies all changed lanes for one site to external numeric state.
///
/// Implementations must not unwind. Replay commits directly into the sink and
/// cannot roll back mutations made before a sink panic.
pub trait TokenDeltaSink<T: Copy> {
    fn apply_site(&mut self, site: u16, deltas: &[TokenDelta<T>], replay: TokenDeltaReplay);
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TokenDeltaPop {
    deltas: usize,
    materialized: bool,
}

impl TokenDeltaPop {
    #[inline]
    pub fn deltas(self) -> usize {
        self.deltas
    }

    #[inline]
    pub fn was_materialized(self) -> bool {
        self.materialized
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum JournalError {
    TooManySites {
        site_count: usize,
    },
    TooManyLanes {
        lane_count: usize,
    },
    CapacityOverflow {
        collection: &'static str,
        requested: usize,
    },
    AllocationFailed {
        collection: &'static str,
        requested: usize,
    },
    SiteCountMismatch {
        actual: usize,
        expected: usize,
    },
    DepthOverflow {
        depth: usize,
        capacity: usize,
    },
    DirtySiteOutOfRange {
        site: usize,
        site_count: usize,
    },
    DuplicateDirtySite {
        site: usize,
    },
    FrameOverflow {
        required: usize,
        capacity: usize,
    },
}

impl fmt::Display for JournalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManySites { site_count } => {
                write!(f, "site count {site_count} exceeds the u16 index space")
            }
            Self::TooManyLanes { lane_count } => {
                write!(f, "lane count {lane_count} exceeds the u8 index space")
            }
            Self::CapacityOverflow {
                collection,
                requested,
            } => write!(
                f,
                "{collection} capacity {requested} exceeds the addressable byte range"
            ),
            Self::AllocationFailed {
                collection,
                requested,
            } => write!(
                f,
                "failed to allocate {requested} elements for {collection}"
            ),
            Self::SiteCountMismatch { actual, expected } => {
                write!(f, "site count mismatch: got {actual}, expected {expected}")
            }
            Self::DepthOverflow { depth, capacity } => {
                write!(f, "journal depth {depth} exceeds capacity {capacity}")
            }
            Self::DirtySiteOutOfRange { site, site_count } => {
                write!(f, "dirty site {site} is outside site count {site_count}")
            }
            Self::DuplicateDirtySite { site } => {
                write!(f, "dirty site {site} appears more than once")
            }
            Self::FrameOverflow { required, capacity } => {
                write!(
                    f,
                    "frame requires {required} deltas but capacity is {capacity}"
                )
            }
        }
    }
}

impl Error for JournalError {}

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

/// Preallocated reversible token journal for make/undo search.
///
/// Frames below `materialized_depth` have been applied to the numeric sink.
/// Later frames update only the logical mirror until
/// [`materialize_pending`](Self::materialize_pending) is called.
pub struct ReversibleTokenJournal<T, const LANES: usize, const MAX_DELTAS: usize> {
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
    pub fn try_new(site_count: usize, max_depth: usize) -> Result<Self, JournalError> {
        if site_count > u16::MAX as usize + 1 {
            return Err(JournalError::TooManySites { site_count });
        }
        if LANES > u8::MAX as usize + 1 {
            return Err(JournalError::TooManyLanes { lane_count: LANES });
        }
        let mut logical = try_vec_capacity(site_count, "logical tokens")?;
        logical.resize(site_count, [T::default(); LANES]);
        let mut frames = try_vec_capacity(max_depth, "journal frames")?;
        frames.extend((0..max_depth).map(|_| TokenDeltaFrame::new()));
        let mut seen_sites = try_vec_capacity(site_count, "seen sites")?;
        seen_sites.resize(site_count, 0);
        Ok(Self {
            logical,
            frames,
            seen_sites,
            seen_generation: 0,
            depth: 0,
            materialized_depth: 0,
        })
    }

    pub fn new(site_count: usize, max_depth: usize) -> Self {
        Self::try_new(site_count, max_depth).expect("invalid reversible journal configuration")
    }

    pub fn try_reset(&mut self, tokens: &[[T; LANES]]) -> Result<(), JournalError> {
        if tokens.len() != self.logical.len() {
            return Err(JournalError::SiteCountMismatch {
                actual: tokens.len(),
                expected: self.logical.len(),
            });
        }
        self.logical.copy_from_slice(tokens);
        self.seen_sites.fill(0);
        self.seen_generation = 0;
        self.depth = 0;
        self.materialized_depth = 0;
        Ok(())
    }

    pub fn reset(&mut self, tokens: &[[T; LANES]]) {
        self.try_reset(tokens)
            .expect("TokenDelta reset site count mismatch");
    }

    /// Record changed lanes at the supplied sites and advance logical time.
    ///
    /// Validation completes before logical tokens or depth are mutated.
    pub fn try_push_after(
        &mut self,
        new_tokens: &[[T; LANES]],
        dirty_sites: &[usize],
    ) -> Result<usize, JournalError> {
        if new_tokens.len() != self.logical.len() {
            return Err(JournalError::SiteCountMismatch {
                actual: new_tokens.len(),
                expected: self.logical.len(),
            });
        }
        if self.depth >= self.frames.len() {
            return Err(JournalError::DepthOverflow {
                depth: self.depth,
                capacity: self.frames.len(),
            });
        }

        self.seen_generation = self.seen_generation.wrapping_add(1);
        if self.seen_generation == 0 {
            self.seen_sites.fill(0);
            self.seen_generation = 1;
        }
        let generation = self.seen_generation;
        let frame = &mut self.frames[self.depth];
        frame.len = 0;
        for &site in dirty_sites {
            if site >= self.logical.len() {
                return Err(JournalError::DirtySiteOutOfRange {
                    site,
                    site_count: self.logical.len(),
                });
            }
            if self.seen_sites[site] == generation {
                return Err(JournalError::DuplicateDirtySite { site });
            }
            self.seen_sites[site] = generation;
            for lane in 0..LANES {
                let old = self.logical[site][lane];
                let new = new_tokens[site][lane];
                if old == new {
                    continue;
                }
                if frame.len >= MAX_DELTAS {
                    return Err(JournalError::FrameOverflow {
                        required: frame.len + 1,
                        capacity: MAX_DELTAS,
                    });
                }
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
        Ok(delta_count)
    }

    pub fn push_after(&mut self, new_tokens: &[[T; LANES]], dirty_sites: &[usize]) -> usize {
        self.try_push_after(new_tokens, dirty_sites)
            .expect("invalid TokenDelta frame")
    }

    pub fn materialize_pending<S: TokenDeltaSink<T>>(&mut self, sink: &mut S) {
        while self.materialized_depth < self.depth {
            replay_forward(&self.frames[self.materialized_depth], sink);
            self.materialized_depth += 1;
        }
    }

    pub fn pop<S: TokenDeltaSink<T>>(&mut self, sink: &mut S) -> Option<TokenDeltaPop> {
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
    pub fn logical_tokens(&self) -> &[[T; LANES]] {
        &self.logical
    }

    #[inline]
    pub fn depth(&self) -> usize {
        self.depth
    }

    #[inline]
    pub fn materialized_depth(&self) -> usize {
        self.materialized_depth
    }

    #[inline]
    pub fn site_count(&self) -> usize {
        self.logical.len()
    }

    #[inline]
    pub fn max_depth(&self) -> usize {
        self.frames.len()
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

fn try_vec_capacity<T>(requested: usize, collection: &'static str) -> Result<Vec<T>, JournalError> {
    let bytes =
        std::mem::size_of::<T>()
            .checked_mul(requested)
            .ok_or(JournalError::CapacityOverflow {
                collection,
                requested,
            })?;
    if bytes > isize::MAX as usize {
        return Err(JournalError::CapacityOverflow {
            collection,
            requested,
        });
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(requested)
        .map_err(|_| JournalError::AllocationFailed {
            collection,
            requested,
        })?;
    Ok(values)
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
            self.values[delta.site() as usize] +=
                i32::from(delta.new_token()) - i32::from(delta.old());
            self.events.push(format!(
                "delta:{}:{}:{}>{}",
                delta.site(),
                delta.lane(),
                delta.old(),
                delta.new_token()
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
    fn pending_materialize_and_pop_preserve_prefix_invariant() {
        let initial = [[1u16, 2], [3, 4], [5, 6]];
        let mut journal = ReversibleTokenJournal::<u16, 2, 4>::new(3, 4);
        journal.reset(&initial);
        let mut sink = RecordingSink::default();

        let after_one = [[7, 2], [3, 4], [5, 8]];
        assert_eq!(journal.push_after(&after_one, &[0, 2]), 2);
        let after_two = [[7, 9], [3, 10], [5, 8]];
        assert_eq!(journal.push_after(&after_two, &[0, 1]), 2);
        assert!(!journal.pop(&mut sink).unwrap().was_materialized());
        assert_eq!(journal.logical_tokens(), after_one);

        journal.materialize_pending(&mut sink);
        assert_eq!(journal.materialized_depth(), 1);
        assert_eq!(sink.values, [6, 0, 2]);
        assert_eq!(journal.logical_tokens(), after_one);
        assert_eq!(journal.depth(), 1);
    }

    #[test]
    fn invalid_frame_does_not_mutate_logical_state() {
        let initial = [[1u16, 2]];
        let mut journal = ReversibleTokenJournal::<u16, 2, 1>::new(1, 2);
        journal.reset(&initial);
        assert!(matches!(
            journal.try_push_after(&[[3, 4]], &[0]),
            Err(JournalError::FrameOverflow { .. })
        ));
        assert_eq!(journal.logical_tokens(), initial);
        assert_eq!(journal.depth(), 0);
    }

    #[test]
    fn duplicate_dirty_site_fails_before_logical_mutation() {
        let initial = [[1u16, 2]];
        let mut journal = ReversibleTokenJournal::<u16, 2, 4>::new(1, 2);
        journal.reset(&initial);
        assert_eq!(
            journal.try_push_after(&[[3, 4]], &[0, 0]),
            Err(JournalError::DuplicateDirtySite { site: 0 })
        );
        assert_eq!(journal.logical_tokens(), initial);
        assert_eq!(journal.depth(), 0);
    }

    #[test]
    fn impossible_depth_capacity_returns_error() {
        assert!(matches!(
            ReversibleTokenJournal::<u8, 1, 1>::try_new(1, usize::MAX),
            Err(JournalError::CapacityOverflow {
                collection: "journal frames",
                requested: usize::MAX,
            })
        ));
    }
}
