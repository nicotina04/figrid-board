//! Gomoku engines for 15x15 and 20x20 boards in the same build.
//!
//! Use [`size15`] and [`size20`] explicitly when using both sizes. The crate-root
//! exports retain the old default (`board20` selects the legacy default only).
pub mod bitboard;
pub mod coord;
pub mod game;
pub mod position;
pub mod size15;
pub mod size20;
pub use bitboard::{BitBoard15, BitBoard20};
#[cfg(not(feature = "board20"))]
pub use size15::*;
#[cfg(feature = "board20")]
pub use size20::*;
pub type Board15 = size15::Board;
pub type Board20 = size20::Board;
pub type Searcher15 = size15::Searcher;
pub type Searcher20 = size20::Searcher;
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
