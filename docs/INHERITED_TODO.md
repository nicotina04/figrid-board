# Inherited Technical TODO (from 0.3.1)

Technical observations left by the original author during the 2026-04 maintainership handover. These are **not blockers for 0.3.2** and are expected to be addressed during the NNUE engine reconstruction leading up to 0.4.x.

Line references are against the 0.3.1 source published on docs.rs.

## 1. Missing depth assertion in `evaluator.rs` — **resolved in 0.4.3**

Location: `evaluator.rs` around lines 85–87.

After `$operation` in the macro expansion, an assertion should verify that the tree cursor's depth has not been changed by `$operation`. Without it, subtle corruption of cursor invariants can go undetected.

*Resolution (0.4.3, 2026-04-22, nicotina04):* added a `debug_assert_eq!` on `cur_depth()` immediately after `$operation` inside the `traversal_in_depth!` macro. The assertion is `debug_assert` rather than `assert` to keep release builds zero-cost.

## 2. RAM usage limit / CPU-cache friendliness

Location: `evaluator.rs` around lines 136–138.

The current design keeps a large structure in memory. Consider enforcing a smaller RAM usage limit so that the hot working set stays closer to CPU cache capacity. This is particularly relevant for the tournament build, where per-move time is short and cache locality dominates.

## 3. `Tree` safety encapsulation

Location: `tree.rs` (safety requirement noted around line 333).

The safety of `Tree` is currently maintained by the whole `tree.rs` module. It would be better to encapsulate the basic operations of `Tree` into a dedicated submodule that owns the unsafe boundary, so the rest of `tree.rs` — and the rest of the crate — only deals with safe wrappers. This would localize audits and simplify future refactors.

## 4. `DynEval` / dynamic `Coord` — **sidestepped in 0.4.0**

An alternative to `Eval<SZ>` (provisionally `DynEval`), and a matching alternative to `Coord<SZ>` with a dynamic board-size property, would significantly simplify the Gomocup protocol interface implementation. The protocol doesn't know the board size at compile time.

*Status (0.4.x, nicotina04):* the NNUE engine under `figrid_board::{board, search, vct}` is fixed at `BOARD_SIZE = 15`, which is the Gomocup Freestyle size we currently submit for. The Gomocup protocol adapter in `bin/pbrain_figrid_noru.rs` therefore does not need `DynEval` — it rejects `START` sizes other than 15 with `ERROR - unsupported board size`. Supporting the 20×20 and `standard` rule variants (wuwbobo2021's stated follow-up wish) will require either retrofitting `DynEval` onto the NNUE pipeline or building a second 20×20 NNUE; that is tracked as a roadmap item for a later 0.5+ release, not as a 0.4 task.

## 5. Dynamic board size / rule for `Evaluator` (open question)

Whether `Evaluator` itself needs a dynamic board-size / rule alternative is unclear. It would be useful, but more complex than (4) and may conflict with the static-dispatch performance assumptions. Revisit after (4) lands.

## 6. `Rows` rename / visibility — **resolved in 0.4.0**

Consider renaming `Rows` to `Board` with a custom `Display` impl, since it's currently the only board-state storage in the crate. Alternatively, `Rows` should not be exposed in the public API — it is presently orphaned there (no other public item uses it).

*Resolution (0.4.0, 2026-04-21, nicotina04):* the crate-root `Board` now lives in `src/board.rs` (noru-based bitboard with `Display` impl for inspection) and is the canonical board-state type for the 0.4 NNUE engine. The original `Rows` is no longer at the crate root — it is scoped inside `figrid_board::legacy::Rows`. Downstream users who need the original row representation can still reach it there, but it is no longer orphaned in the public surface.

---

## Engine-strength roadmap (beyond the above)

- VCF / VCT tactical search (currently missing; essential for competitive Gomocup play).
- Iterative deepening with time management (5 s/move Gomocup constraint).
- Transposition table with Zobrist hashing.
- NNUE-based static evaluation via the [noru](https://crates.io/crates/noru) core; training pipeline lives upstream in the `noru-tactic-engine` project.
