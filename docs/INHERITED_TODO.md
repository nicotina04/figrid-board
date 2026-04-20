# Inherited Technical TODO (from 0.3.1)

Technical observations left by the original author during the 2026-04 maintainership handover. These are **not blockers for 0.3.2** and are expected to be addressed during the NNUE engine reconstruction leading up to 0.4.x.

Line references are against the 0.3.1 source published on docs.rs.

## 1. Missing depth assertion in `evaluator.rs`

Location: `evaluator.rs` around lines 85–87.

After `$operation` in the macro expansion, an assertion should verify that the tree cursor's depth has not been changed by `$operation`. Without it, subtle corruption of cursor invariants can go undetected.

## 2. RAM usage limit / CPU-cache friendliness

Location: `evaluator.rs` around lines 136–138.

The current design keeps a large structure in memory. Consider enforcing a smaller RAM usage limit so that the hot working set stays closer to CPU cache capacity. This is particularly relevant for the tournament build, where per-move time is short and cache locality dominates.

## 3. `Tree` safety encapsulation

Location: `tree.rs` (safety requirement noted around line 333).

The safety of `Tree` is currently maintained by the whole `tree.rs` module. It would be better to encapsulate the basic operations of `Tree` into a dedicated submodule that owns the unsafe boundary, so the rest of `tree.rs` — and the rest of the crate — only deals with safe wrappers. This would localize audits and simplify future refactors.

## 4. `DynEval` / dynamic `Coord`

An alternative to `Eval<SZ>` (provisionally `DynEval`), and a matching alternative to `Coord<SZ>` with a dynamic board-size property, would significantly simplify the Gomocup protocol interface implementation. The protocol doesn't know the board size at compile time.

## 5. Dynamic board size / rule for `Evaluator` (open question)

Whether `Evaluator` itself needs a dynamic board-size / rule alternative is unclear. It would be useful, but more complex than (4) and may conflict with the static-dispatch performance assumptions. Revisit after (4) lands.

## 6. `Rows` rename / visibility

Consider renaming `Rows` to `Board` with a custom `Display` impl, since it's currently the only board-state storage in the crate. Alternatively, `Rows` should not be exposed in the public API — it is presently orphaned there (no other public item uses it).

---

## Engine-strength roadmap (beyond the above)

- VCF / VCT tactical search (currently missing; essential for competitive Gomocup play).
- Iterative deepening with time management (5 s/move Gomocup constraint).
- Transposition table with Zobrist hashing.
- NNUE-based static evaluation via the [noru](https://crates.io/crates/noru) core; training pipeline lives upstream in the `noru-tactic-engine` project.
