//! Generic position storage; board-size specializations supply the algorithms.
//!
//! Both `Board::<BitBoard15>::new()` and `Board::<BitBoard20>::new()` are
//! available in one build. Each specialization retains its exact-sized caches.
use crate::bitboard::{BitBoard, BitBoard15, BitBoard20};
use crate::game::{RuleSet, Stone};

/// Associates fixed board storage with its fixed-size pattern cache.
pub trait BoardGeometry: BitBoard {
    type PatternState: Clone;
}

impl BoardGeometry for BitBoard15 {
    type PatternState = Box<[[u16; 4]; 225]>;
}

impl BoardGeometry for BitBoard20 {
    type PatternState = Box<[[u16; 4]; 400]>;
}

/// A position parameterized by its bitboard implementation.
#[derive(Clone)]
pub struct Board<B: BoardGeometry = BitBoard15> {
    pub black: B,
    pub white: B,
    pub side_to_move: Stone,
    pub move_count: usize,
    pub last_move: Option<usize>,
    pub history: Vec<usize>,
    pub zobrist: u64,
    pub line_pattern_ids: B::PatternState,
    pub rule_set: RuleSet,
    pub exact5: bool,
}
