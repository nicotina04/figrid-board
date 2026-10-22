# Both board sizes in one build

The library compiles both sizes from shared algorithm source. The protocol adapter
selects the engine at `START 15` or `START 20`; no board-size branch is added inside
the search loop. The 15x15 bitboard remains two `u128` words and the 20x20 bitboard
remains four. Cell indices, search arrays and pattern caches keep their size-specific
representations. There are no subprocesses or duplicate source crates.

`bitboard::BitBoard` is the common storage trait. `position::Board<B>` is generic
over that storage, with a `BoardGeometry` bound associating its exact-sized pattern
cache. Both specializations have the existing position operations:

```rust
use figrid_board::{BitBoard15, BitBoard20};
use figrid_board::position::Board;

let mut small = Board::<BitBoard15>::new();
let mut large = Board::<BitBoard20>::new();
small.make_move(224);
large.make_move(399);
assert!(small.black.get(224));
assert!(large.black.get(399));
```

Use `size15::Searcher` with the small board and `size20::Searcher` with the large
board. `Board15`, `Board20`, `Searcher15` and `Searcher20` are convenience aliases.
For compatibility the old crate-root `Board`, `BitBoard`, `Searcher` and module
paths still select 15x15 by default. The old `board20` feature changes only these
default aliases; it no longer removes the other implementation from the build.

## Models and protocol

Build `pbrain-figrid` with `codebook-eval` to play both board sizes. A
`release-models` build embeds the existing five models, retaining the 20x20
fine-tuned model and each 15x15 rule-specific model and scale. No retraining,
feature remapping or new model promotion is involved. Public crate builds use the
existing older public model; this refactor does not publish the private models.

`START` initializes the selected engine lazily, responds `OK` first, then prints
its configuration. `INFO` before `START` is retained. Repeated `START` can switch
15 -> 20 -> 15 within one process; all board-sized state is reconstructed on a
size change. `RESTART` clears the current board and retains size and rule.
Unsupported sizes return a flushed `ERROR` response. Flat NNUE remains 15x15-only;
a 20x20 game requires the codebook evaluation path.

The unified executable can be named `pbrain-figrid.exe`. A separate `_20`
executable is no longer necessary. Existing released ZIPs are unchanged until a
new release is made.
