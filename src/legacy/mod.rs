//! Original pre-0.4.0 engine: symbolic evaluator, record keeper, rule checker,
//! and a bump-storage game tree. Kept intact so the `pbrain-figrid` binary and
//! any 0.3.x downstream users keep working while the new NNUE engine lives at
//! the crate root.

mod evaluator;
mod rec;
mod rec_base;
mod rec_checker;
mod row;
mod rule;
mod tree;

pub use evaluator::*;
pub use rec::*;
pub use rec_base::*;
pub use rec_checker::*;
pub use row::*;
pub use rule::*;
pub use tree::*;
