//! A library for the Five-in-a-Row (Gomoku) game.
//!
//! Starting from 0.4.0 the primary engine is NNUE-based (powered by the
//! [`noru`](https://crates.io/crates/noru) core). The previous symbolic
//! evaluator, rule/rec/tree stack, and generic `Eval<SZ>` trait live under
//! the [`legacy`] module — they are preserved for binary compatibility of
//! the original `pbrain-figrid` executable and downstream users.

pub mod board;
pub mod coord;
pub mod eval;
pub mod features;
pub mod heuristic;
pub mod search;
pub mod vct;

pub mod legacy;

pub use board::{
    to_idx, to_rc, BitBoard, Board, GameResult, Move, Stone, BOARD_SIZE, NUM_CELLS,
};
pub use coord::{Coord, Coord15, Coord20, CoordState, Rotation};
pub use eval::{evaluate, IncrementalEval};
pub use features::GOMOKU_NNUE_CONFIG;
pub use heuristic::{scan_line, LineInfo, DIR};
pub use search::{SearchResult, Searcher};
pub use vct::{search_vct, VctConfig};

/// Possible errors returned from this crate.
#[derive(Clone, Debug, PartialEq)]
#[repr(u8)]
pub enum Error {
    ParseError,
    InvalidCoord,
    CoordNotEmpty,
    RecIsEmpty,
    RecIsFull,
    RecIsFinished,
    ItemNotExist,
    TransformFailed,
    CursorAtEnd,
}
