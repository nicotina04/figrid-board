//! A library for the Five-in-a-Row (Gomoku) game.
//!
//! Starting from 0.4.0 the primary engine is NNUE-based (powered by the
//! [`noru`](https://crates.io/crates/noru) core). The pre-0.4 symbolic
//! evaluator, rule/rec/tree stack, and generic `Eval<SZ>` trait live under
//! the [`legacy`] module — it is preserved alongside the new NNUE engine
//! so the original `pbrain-figrid-legacy` executable and any 0.3.x
//! downstream users still work. This preservation is a maintainer choice,
//! not a promise of ABI stability.

pub mod board;
pub mod book;
pub(crate) mod candidate_local_ensemble;
pub(crate) mod candidate_ranker;
#[cfg(feature = "codebook-eval")]
pub mod codebook_eval;
pub(crate) mod codebook_sidecar;
pub mod coord;
pub mod eval;
pub mod features;
pub mod heuristic;
pub mod pattern_dense;
pub mod pattern_table;
pub(crate) mod relation_fusion_gate;
pub(crate) mod relation_lite;
#[doc(hidden)]
pub mod rq423_root_accept;
pub mod search;
pub mod transposition;
pub mod vct;

pub mod legacy;

pub use board::{
    BOARD_SIZE, BitBoard, Board, GameResult, Move, NUM_CELLS, RuleSet, Stone, to_idx, to_rc,
};
pub use coord::{Coord, Coord15, Coord20, CoordState, Rotation};
pub use eval::{IncrementalEval, evaluate};
pub use features::GOMOKU_NNUE_CONFIG;
pub use heuristic::{DIR, LineInfo, scan_line};
pub use search::{SearchProfileSnapshot, SearchResult, Searcher};
pub use vct::{
    VctConfig, VctSearchResult, VctSearchStats, search_vct, search_vct_audit_json,
    search_vct_with_stats,
};

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
