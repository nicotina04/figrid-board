//! The 15x15 engine, available alongside the other size in every build.
pub(crate) use crate::size15 as engine;
pub const BOARD_SIZE: usize = 15;
pub type NativeBitBoard = crate::bitboard::BitBoard15;
pub type NativeBitBoardIter = crate::bitboard::BitBoardIter15;
pub type CellIdx = u8;
pub use crate::Error;
#[path = "../board.rs"]
pub mod board;
#[path = "../book.rs"]
pub mod book;
#[cfg(feature = "codebook-eval")]
#[path = "../codebook_eval.rs"]
pub mod codebook_eval;
pub use crate::coord;
#[path = "../d4_hash.rs"]
pub mod d4_hash;
#[path = "../eval.rs"]
pub mod eval;
#[cfg(feature = "codebook-eval")]
#[path = "../factored_codebook.rs"]
pub mod factored_codebook;
#[path = "../features.rs"]
pub mod features;
#[path = "../heuristic.rs"]
pub mod heuristic;
#[path = "../p4r0_census.rs"]
pub mod p4r0_census;
#[path = "../pattern_dense.rs"]
pub mod pattern_dense;
#[path = "../pattern_table.rs"]
pub mod pattern_table;
#[path = "../policy_move.rs"]
pub mod policy_move;
#[path = "../renju.rs"]
pub mod renju;
#[path = "../search.rs"]
pub mod search;
#[path = "../threat_field.rs"]
pub mod threat_field;
#[path = "../transposition.rs"]
pub mod transposition;
#[path = "../tss.rs"]
pub mod tss;
#[path = "../vct.rs"]
pub mod vct;
#[cfg(feature = "codebook-eval")]
#[path = "../white_root_order.rs"]
pub(crate) mod white_root_order;

pub use board::{BitBoard, Board, GameResult, Move, NUM_CELLS, RuleSet, Stone, to_idx, to_rc};
pub use coord::{Coord, Coord15, Coord20, CoordState, Rotation};
pub use eval::{IncrementalEval, evaluate};
pub use features::GOMOKU_NNUE_CONFIG;
pub use heuristic::{DIR, LineInfo, scan_line};
pub use search::{
    MovePickerStats, ProgressCallback, SearchProfileSnapshot, SearchProgress, SearchResult,
    SearchShapeStats, Searcher,
};
pub use threat_field::IncrementalThreatField;
pub use tss::{
    DependencyCandidateArms, DependencyQuietCandidate, Q1CandidateAttempt, Q1DefenseAttempt,
    Q1DefenseOutcome, Q1TssConfig, Q1TssResult, Q1TssStopReason, QuietThreatCandidate,
    QuietThreatConfig, ResponseRelevanceAudit, audit_quiet_response_relevance,
    classify_move_with_directions, directional_aggregation_mismatches,
    generate_dependency_quiet_candidates, generate_quiet_threat_candidates, search_q1_tss_root,
};
#[cfg(feature = "cb-p1-audit")]
pub use vct::dfpn::{
    BoundedDfpnConfig, BoundedDfpnSession, DfpnCertificateReplay, DfpnCheckpoint, DfpnError,
    DfpnStatus, DfpnWidthBin,
};
pub use vct::{
    VctConfig, VctSearchResult, VctSearchStats, search_vct, search_vct_audit_json,
    search_vct_exact, search_vct_with_stats,
};
