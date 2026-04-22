//! Pre-0.4.0 engine — the symbolic evaluator, record keeper, rule checker,
//! and bump-storage game tree originally written by wuwbobo2021.
//!
//! The per-row serial calculators (`FreeSerCalc`, `StdSerCalc`,
//! `CaroSerCalc`) are CPU-oriented — they descend from the first program
//! in [CaroAI_BACKEND#1][issue] and were the version chosen for figrid
//! because they run faster on a CPU than the FPGA-oriented variant from
//! the same thread.
//!
//! This module is kept intact so the original `pbrain-figrid-legacy`
//! executable still works and so the 0.3.x library surface remains
//! available. It is not a fork, not a long-term API, and its presence
//! here is a maintainer choice rather than a stability guarantee.
//!
//! [issue]: https://github.com/nguyencongminh090/CaroAI_BACKEND/issues/1

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
