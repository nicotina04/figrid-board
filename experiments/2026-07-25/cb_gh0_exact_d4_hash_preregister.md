# CB-GH0 exact D4 canonical state-hash protocol freeze

Date: 2026-07-25 KST

Status: protocol frozen before the authoritative precondition analyzer,
incremental hash implementation, TT integration, or performance A/B.

Disclosure: source inspection before this freeze showed that the deployed
codebook has nine separately trained region rows, so D4 evaluator invariance
is doubtful. A preliminary non-authoritative tensor inspection also suggested
region-weight mismatches. Those observations motivate this hard precondition;
all authoritative counts, witnesses, and the decision must come from the
sealed analyzer defined below. This is not represented as a blinded
preregistration.

## Question and exactness boundary

CB-GH0 has two deliberately separate questions:

1. **GH0-H:** can the eight rotations/reflections of the game-theoretic state
   be maintained as one canonical state identity?
2. **GH0-TT:** can the current product search also share depth-limited TT
   scores and bounds across those orientations without changing semantics?

The semantic state is:

```text
(black bitboard, white bitboard, side to move, effective RuleSet)
```

For placement-only Gomoku, `last_move` and move history are not part of the
game-theoretic state: reachable play terminates as soon as a winning line
appears, and there is no repetition, capture, or history-dependent legality
in the implemented rules. Current orientation, Pattern4 caches, candidate
frontiers, and evaluator journals are also not semantic state. However,
`Board::game_result()` is implemented through `last_move`, and selective
search consults recent history. P0 therefore rebuilds transformed full
histories, while P1 must prove that TT probe/store occurs only on valid
reachable nodes and must inventory every remaining history-dependent search
consumer. Freestyle, Standard, Caro, and Renju are distinct rule domains. The
canonical orientation is the minimum of eight D4-transformed 64-bit hashes;
ties choose the smaller transform index.

A geometrically canonical key is not by itself an exact TT key. A 64-bit
value is also only a fingerprint: exact proof-cache identity requires the
canonical hash plus equality verification of canonical Black/White
bitboards, side, and rule (or verification against the original state). The
current
TT stores depth-limited scores, bounds, depth, and a best move. Sharing those
entries between transformed positions is exact only if the whole relevant
search semantics is D4-equivariant:

- terminal and legal-move rules;
- static quantized evaluator;
- transformed best-move coordinates;
- selective search behavior that can affect returned bounds.

In particular, a D4-asymmetric static evaluator makes horizon scores and
therefore TT bounds orientation-dependent. That blocks GH0-TT, but it does
not refute GH0-H or pure game-rule proof semantics. A move-hint-only symmetry
table that does not share scores/bounds is a different future card, not a
fallback TT arm here.

## Frozen baseline and inputs

- repository parent:
  `5f116bd3b1744fced367382fffe4fc6ac7b6a80a`;
- branch: `codex/cb-token-delta`;
- product: figrid-board 0.8.2 with CB-D1 and CB-TD1 promoted, CB-F1 runtime
  default OFF, incumbent vocabulary retained after CB-VOC1 NO-GO;
- build: `RUSTFLAGS=-C target-cpu=x86-64-v3`;
- model, 1,410,562 bytes:
  `models/gomoku_codebook_v1_swapclosed.json`,
  SHA-256
  `42968FDAB01BA8CCD1DE3DED05C532E4B237DD47EEFFD7AE1C2F264D77BA7DA2`;
- vocabulary, 17,060 bytes:
  `data/topk.bin`,
  SHA-256
  `103891DCD1DCD978C654593ABE78EF32C56E2E350B500EE665BC45AC051AA16D`;
- frozen correctness/performance trace, 317,511 bytes and 64 games:
  `../figrid-dp-campaign/experiments/2026-07-25/`
  `dp_a1_fresh_holdout_64g.jsonl`,
  SHA-256
  `1FD40D8948F113AD236FA44F5EEADCA1907C0C3103987CB4C704B67A9B47531A`.

The trace may be used only for correctness and performance. It may not select
features, vocabulary, graph labels, thresholds, or training data. The
analyzer must read nonblank JSONL rows in file order. Exactly one of
`black_engine`/`white_engine` must contain case-insensitive `figrid`; that
color is `product_side`. Starting from a Freestyle empty board, immediately
before each recorded move, select the current board iff
`source=="engine"` and `side_to_move==product_side`. Sampling stride is one.
Then validate and play the recorded move. Stop after the first 1,022 selected
roots and require the sealed file to contain exactly 64 valid games.

All inputs are byte-length/SHA checked before and after the run. Output is
create-new. Any identity, parser, finite-value, root-count, transform, or
replay failure is `INVALID_CB_GH0`.

## D4 convention

For zero-based `(row,col)` and `n=14`, transform indices are frozen as:

| t | transformed coordinate |
|---:|---|
| 0 | `(row,col)` |
| 1 | `(col,n-row)` |
| 2 | `(n-row,n-col)` |
| 3 | `(n-col,row)` |
| 4 | `(row,n-col)` |
| 5 | `(n-row,col)` |
| 6 | `(col,row)` |
| 7 | `(n-col,n-row)` |

The analyzer must prove every cell map is a bijection, every inverse is
correct, and the induced 3x3 region map agrees with transforming any cell in
that region. Region index is `(row/5)*3+(col/5)`.

It must also prove the Pattern4 geometry lemma independently of model
weights. For every anchor cell, four undirected line directions, and D4
transform, the transformed 11-position coordinate/boundary sequence must
equal the target transformed direction's sequence either directly or
reversed. Because released Pattern4 canonicalization takes the minimum of a
window and its reverse, this proves that arbitrary Black/White occupancy maps
to the same canonical raw token and therefore the same mapped ID/RARE row.
Cell-map/inverse inconsistency is `INVALID_CB_GH0`; a validly measured
Pattern4 lemma failure closes the TT evaluator branch.

## Stage P0: split semantic precondition

### P0-H — canonical game-state geometry

This branch proves only game-state identity:

- all cell maps, inverses, and group composition are exact;
- D4 maps rows, columns, and diagonals to rows, columns, or diagonals while
  preserving stone color, contiguous length, and open-end count;
- `RuleSet::line_wins` depends only on side, length, and open-end count, so
  Freestyle, Standard, Caro, and the implemented Renju terminal semantics are
  D4-invariant;
- transformed full-history rebuilds preserve occupancy, side, effective
  rule, move count, legal-move set, and game result on every frozen root.

P0-H passage opens the incremental hash/canonical-state correctness work even
if the TT branch fails. P0-H failure is:

```text
STOP_GH0_STATE_GEOMETRY_PRECONDITION
```

### P0-T1 — deployed evaluator structure

Load the f32 model through the released parser and quantize with released
E32/H64/F64 rules. For every nonidentity transform, compare each region row
with its transformed region row:

- raw f32 head by `to_bits`, 9x16 entries;
- quantized i16 head, 9x16 entries;
- raw f32 FM factors by `to_bits`, 9x16x8 entries;
- quantized i16 FM factors, 9x16x8 entries.

Also report corner and edge orbit equality groups separately:

- head: 16 corner groups plus 16 edge groups;
- factors: `16*8` corner plus `16*8` edge groups.

Direct row equality is a deliberately conservative protocol gate
for this deployed implementation. FM latent reparameterizations are not
inferred post-hoc. Failure does not alone claim that every board differs; P0b
seeks and reports constructive product witnesses, which need not exist in a
finite sample or after quantization.

Even equal region rows would not by themselves prove bit-exact output. The
released forward pass accumulates linear and FM terms as f64 in physical
feature-index order. A D4 feature permutation can reorder an equal multiset
of floating-point terms, and floating-point addition is not associative.
Accordingly P0 also requires a structural forward-arithmetic proof:

- Pattern4 directional sums and region pooling are exact bounded integer
  operations, so their permutation is order-independent;
- the final forward must use exact/order-independent accumulation or a fixed
  D4-invariant canonical term order.

The current physical-index f64 loop does not satisfy that proof obligation.
Frozen-root equality remains useful evidence, but it cannot replace the
structural proof over all states.

### P0-T2 — frozen-root product witnesses

For every one of the 1,022 frozen product roots:

1. rebuild each of its eight D4-transformed boards from the transformed move
   history;
2. preserve side to move and effective rule;
3. require transformed occupancy, move count, game result, full legal-move
   set, and candidate **set** after D4 mapping and sorting to be consistent;
   candidate vector order is explicitly not compared because row-major
   discovery order is orientation-dependent and belongs to the P1-TT audit;
4. call the released `evaluate_full_quantized` from natural side-to-move
   perspective;
5. compare each nonidentity final f32 bit pattern with t0.

Report per-transform:

- roots and mismatch count;
- exact mismatch rate;
- finite absolute-difference p50, p95, p99, and maximum;
- first deterministic witness: root index, rule, side, history, t0/t value
  and bits.

The GH0-TT evaluator precondition passes only if:

- the Pattern4 geometry lemma has zero mismatches;
- every raw and quantized head/factor D4 mismatch count is zero;
- every corner/edge group mismatch count is zero;
- the released final forward has a documented exact or D4-invariant
  accumulation order rather than orientation-dependent physical f64 order;
- all 7,154 nonidentity frozen-root comparisons have identical f32 bits.

There is no tolerance. One constructive f32 mismatch is sufficient to stop
exact score/bound sharing. The TT-branch failure label is:

```text
STOP_GH0_TT_SEMANTIC_PRECONDITION
```

On TT-branch failure, the following are forbidden in this card:

- canonical score/bound TT probe or store;
- TT hit/time A/B;
- product environment switch that enables canonical TT;
- benchmark, arena, or default change.

The incremental canonical state-hash branch remains open if and only if P0-H
passes. The combined valid labels are:

- `OPEN_GH0_HASH_AND_TT` when P0-H and GH0-TT both pass;
- `OPEN_GH0_HASH_ONLY_TT_BLOCKED` when P0-H passes and GH0-TT fails.

## Stage P1-H: hash implementation, only if P0-H passes

The implementation remains default OFF and must preserve public `Board`
layout. The intended location is private `BoardSearchState`, not `Board`,
because downstream code uses exhaustive public struct literals.

The state is:

```text
D4HashState { hashes: [u64; 8] }
CanonicalContext { key: u64, to_canonical: u8 }
```

The hash formula and keys are frozen as follows. `splitmix64` is the existing
board implementation, including its initial addition by
`0x9E3779B97F4A7C15`. Define:

```text
RULE_KEY[Freestyle] = splitmix64(0xD4C0000000000000)
RULE_KEY[Standard]  = splitmix64(0xD4C0000000000001)
RULE_KEY[Caro]      = splitmix64(0xD4C0000000000002)
RULE_KEY[Renju]     = splitmix64(0xD4C0000000000003)

hash[t] = RULE_KEY[effective_rule]
        ^ (side_to_move == White ? SIDE_TO_MOVE_KEY : 0)
        ^ XOR_black_stones stone_key(Black,d4_map[t][cell])
        ^ XOR_white_stones stone_key(White,d4_map[t][cell])
```

The four rule keys must be pairwise distinct. Enabling performs a full root
rebuild. Each routed make/undo XORs, for all transforms:

```text
stone_key(placed_color, d4_map[t][move]) ^ SIDE_TO_MOVE_KEY
```

Rule changes require rebuild/synchronization. `BoardSearchState` synchronization
identity must become `(board.zobrist, board.move_count,
board.effective_rule_set())`; the current two-field tuple cannot detect
`set_rule_set` or legacy `exact5` changes. TokenDelta is not reused:
TokenDelta journals up to 41 delayed evaluator sites, whereas the hash is an
immediate one-move reversible board-state sidecar.

Hash correctness gates, all zero-failure:

1. D4 map bijection, inverse, and group-composition tests.
2. All four rules produce distinct empty-state domain keys.
3. At least 100,000 deterministic mixed make/undo operations, checking after
   every operation:
   - incremental `[u64;8]` equals full bitboard rebuild;
   - canonical context equals full minimum;
   - predicted child hashes equal post-make hashes;
   - complete unwind restores every root hash.
4. Every transformed state has the same canonical key and the expected
   transform relationship.
5. Collision audit stores
   `canonical u64 -> exact canonical (black,white,STM,rule)` and separately
   counts D4-equivalent repeats versus true unequal-state hash collisions.
   True collisions must be zero. The u64 alone is never called an exact proof
   identity.

Any failure is `REJECT_GH0_HASH_EXACTNESS`. Passing creates a default-OFF
hash sidecar only; it does not authorize TT score/bound sharing.

## Stage P2-H: hash-maintenance cost

After P1-H correctness, one release binary compares hash OFF versus hash
maintained-but-not-consumed, in order `A1 -> B1 -> B2 -> A2`:

- the registered 100,000 mixed make/undo workload;
- VCT-OFF depth 4 over the frozen 1,022 roots with identical TT behavior;
- exact best move, score, depth, node fields, and final boards;
- isolated hash update/rebuild nanoseconds and whole-search wall time;
- paired game-block bootstrap.

Because the sidecar is default OFF, this is a hot-path integration gate rather
than a product-strength claim. `GO_GH0_HASH_SIDECAR` requires both isolated
transition and whole-search B/A point ratios `<=1.005` and one-sided 95%
uppers `<1.01`. Otherwise the implementation is retained only as offline
reference and labeled `HASH_CORRECT_BUT_TOO_COSTLY`.

## Stage P1-TT: integration, only if the TT precondition passes

TT store maps the best move into canonical coordinates. TT probe maps it back
through the inverse transform and verifies legality. The TT entry binary
format must not change.

TT correctness gates, all zero-failure:

1. Synthetic TT store/probe preserves the current entry fields—score, depth,
   bound, and transformed best move—without changing the 16-byte entry
   format; inverse-mapped move illegality must be zero.
2. Before product score/bound sharing, perform a structural search-semantic
   audit and either:
   - make every orientation-sensitive consumer operate in one canonical
     search frame, including row-major candidate ties, TT moves,
     history/continuation tables, killers, recent-history features, LMR, and
     LMP; or
   - restrict the exact arm to a fully specified nonselective search profile
     whose returned value is structurally D4-equivariant.
   The selected canonical frame/profile must be frozen in a separate committed
   amendment before child implementation. It must be identical in A and B so
   canonical TT is the only A/B factor. If that frame differs from the current
   product policy, current-product versus frame semantics and strength require
   a separate gate before the TT A/B; the control may not silently cease to be
   the product baseline.
3. Cold, reset search over all 1,022 roots and transforms is then a regression
   check, not a proof by sampling. It must still have zero score/bound
   inconsistencies before performance evidence is admissible.

Any failure is `REJECT_GH0_TT_EXACTNESS`.

## Stage P2-TT: same-binary TT A/B, only after P1-TT

- one release binary;
- explicit default-OFF selector `NORU_D4_CANONICAL_TT`;
- control A uses the current orientation-specific Zobrist TT;
- candidate B uses canonical TT key plus move remapping;
- first run VCT-OFF fixed depth 4 over the frozen 1,022 roots;
- arm order `A1 -> B1 -> B2 -> A2`;
- reset TT and path-dependent search state per root/arm;
- record best move, score, completed depth, main/qsearch/total nodes, TT
  probes/hits/cutoffs/stores, wall time, and hash-maintenance time;
- exact semantic outputs and node-policy invariants are mandatory;
- paired game-block bootstrap for hit, node, and wall ratios;
- only a correctness-clean useful signal may open the existing product
  VCT-ON protocol and a 30-game side-swapped sanity arena.

`GO_GH0_TT_PROTOTYPE` requires:

- TT hit-rate gain at least +0.50 percentage point;
- and either total-node B/A point `<=0.99` with one-sided 95% upper `<1.00`,
  or wall B/A point `<=0.99` with one-sided 95% upper `<1.00`;
- the other of node/wall must have point `<=1.005` and upper `<1.01`;
- zero semantic, node-policy, move-remap, or legality discrepancy.

Correctness without that signal is `SAFE_HASH_ONLY_NO_TT_PROMOTION`.

P0 passage only opens the corresponding implementation branch; it is not
promotion. GH0 cannot change the default without its correctness/performance
gates and a separate recorded decision.

## GH1 boundary

GH0-TT failure blocks D4 canonicalization only as a product **score/bound TT
key**. It does not refute pure DFPN proof semantics; that requires its own
rule/move-generator equivariance check and exact collision verification, not
evaluator invariance. It also does not block CB-GH1's explicitly lossy graph
abstraction for duplicate census, tactical structure classification, coreset
construction, or later codebook candidate generation.

Any GH1/DFPN proof-cache experiment must verify equality using canonical
Black/White bitboards, side, and rule, or using the original uncanonicalized
state. Another unverified u64 Zobrist is not an “exact noncanonical
signature,” and a graph signature alone is never a proof key.
