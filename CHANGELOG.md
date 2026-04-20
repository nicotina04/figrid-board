# Changes

## 0.3.2 (2026-04-20)
* Maintainership transferred from wuwbobo2021 to nicotina04. Repository moved to <https://github.com/nicotina04/figrid-board>.
* No functional changes to the engine yet. Future 0.4+ releases will introduce an NNUE-based evaluator (via the [noru](https://crates.io/crates/noru) core) and stronger tactical search (VCF/VCT, iterative deepening, transposition table) for Gomocup 2026.
* Known technical debt inherited from 0.3.x is tracked in `docs/INHERITED_TODO.md`.

## 0.3.1 (2025-06-17)
* Fixes a bug that prevents the evaluator from making connection of 5 or blocking the opponent's 4/live3 under some situations.
* Tries to fix the timeout issue which affects the first 6 moves.
* Fixes the Caro rule.
* Fixes the `BOARD` command; allows rule setting after `START` command.
* Makes a few optimizations.
* Adds documentation.

## 0.3.0 (2025-05-31)
* Initial Rust version. The author has made this bold attempt in a hurry after catching a cold. Suffered in critical bugs, got the lowest ranking in Gomocup 2025.
