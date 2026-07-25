# CB-GH0 P1-H exact hash correctness amendment

Date frozen: 2026-07-25 KST

Parent protocol:
`experiments/2026-07-25/cb_gh0_exact_d4_hash_preregister.md`

Opening result:
`OPEN_GH0_HASH_ONLY_TT_BLOCKED`, report SHA-256
`BCF021820ECF7A841B2A65A594EC99FF200E5D8EEC0A1C5C0FF287BF6098F50A`.

Status: frozen before the P1-H implementation is compiled or any registered
correctness workload is run.

This amendment removes the remaining degrees of freedom in the parent
protocol's “at least 100,000 deterministic mixed make/undo operations.” It
does not reopen the blocked score/bound TT branch and does not change the
registered hash formula, keys, D4 convention, or P2-H cost gates.

## Scope and pass label

P1-H may add only a default-OFF exact D4 hash sidecar. It may not:

- replace or canonicalize the product transposition-table key;
- probe or store a canonical score, bound, or move;
- add a pbrain environment switch;
- change the public `Board` field layout;
- change move generation, evaluation, VCT, or search policy.

Passing all gates produces:

```text
OPEN_GH0_HASH_COST_GATE
```

Any nonzero correctness, collision, rule-domain, D4-relation, or default-OFF
failure produces:

```text
REJECT_GH0_HASH_EXACTNESS
```

A pass opens only P2-H maintenance-cost measurement. It is not product
promotion.

## Exact canonical-state witness

The 64-bit canonical key remains a fingerprint. Collision auditing must build
an independent exact representation for every transform:

```text
black.lo  big-endian u128  16 bytes
black.hi  big-endian u128  16 bytes
white.lo  big-endian u128  16 bytes
white.hi  big-endian u128  16 bytes
side tag                    1 byte   Black=0, White=1
effective rule tag          1 byte   Freestyle=0, Standard=1,
                                     Caro=2, Renju=3
```

The exact canonical state is the lexicographic minimum of these eight
66-byte strings. Its chosen transform uses the same lower-index tie rule, but
it is computed independently of the minimum hash transform.

For each observed state:

- same canonical u64 plus same exact canonical bytes is a D4-equivalent
  repeat;
- same canonical u64 plus different exact canonical bytes is a true hash
  collision;
- if two transforms tie for the minimum u64 but their exact transformed
  strings differ, it is an intra-orbit hash collision.

True and intra-orbit collisions must both be zero. Any proof-cache consumer
must compare the exact state or original state as well as the u64.

## PRNG and transition tape

Use SplitMix64 with:

```text
seed = 0xCB60_2026_0725_0001

next():
    state += 0x9E3779B97F4A7C15
    z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)
```

All multiplication and addition are wrapping u64 operations. Generate exactly
100,000 board transitions from a fresh Freestyle empty board:

```text
decision = next()

undo iff history is nonempty and
          (move_count >= 180 or (decision & 3) == 0)

otherwise:
    legal = board.legal_moves()       # current ascending implementation
    pick = next()
    move = legal[pick % legal.len()]
    make(move)
```

No other PRNG draw is permitted. Because the tape never exceeds 180 stones,
the make branch always has at least one empty cell under the current
placement-only legality implementation.

Before transitions 251, 502, 753, and so on, change the effective rule by
cycling:

```text
Standard -> Caro -> Renju -> legacy Standard -> Freestyle -> repeat
```

Legacy Standard is represented exactly by:

```text
board.set_rule_set(Freestyle)
board.exact5 = true
```

The following Freestyle step must call `set_rule_set(Freestyle)` so
`exact5=false` is restored. A rule switch is not one of the 100,000
transitions and consumes no PRNG draw. Immediately after each switch,
synchronize the sidecar and compare it with a full rebuild.

## Checks on every state

Audit the initial state, every rule-switched state, and every post-transition
state.

For every make, compute predicted child hashes and predicted child canonical
context before mutation. After mutation they must equal the maintained state.
After every make or undo:

- all eight incremental hashes equal an independent full bitboard rebuild;
- the maintained canonical context equals a fresh minimum of
  `(hash, transform_index)`;
- exact canonical bytes are recomputed independently;
- the collision map is updated using the classification above.

After all 100,000 transitions, completely unwind the current history through
the sidecar, switch to Freestyle, synchronize, and require equality with a
fresh Freestyle empty-board sidecar in all eight hashes, canonical context,
and exact canonical bytes.

## D4 relation checks

At the initial state, every 97th post-transition state, and transition
100,000, build all eight transformed states and require for every `g,t`:

```text
hashes(T_g(S))[t] == hashes(S)[D4_COMPOSE[t][g]]
```

Also require:

- all eight transformed states have the same canonical key;
- all eight have the same exact canonical bytes;
- every mapped move round-trips through the registered inverse;
- all transformed effective rules and sides equal the source state.

The test suite must separately cover:

- empty board;
- one center stone;
- full D4 symmetry;
- vertical-reflection-only symmetry;
- 180-degree-only symmetry;
- an asymmetric state;
- synthetic equal minimum hashes choosing the lower transform index.

## Rule and composition gates

The registered D4 cell maps must be bijective, the inverse vector must be
`[0,3,2,1,4,5,6,7]`, and the full composition table must match the parent
convention.

All four empty-state rule keys must be pairwise distinct. On a nonempty board,
switch through all four formal rules and legacy Standard, synchronizing after
each change. Legacy Standard and formal Standard must have identical eight
hashes. Returning to Freestyle must restore `exact5=false` and the Freestyle
domain hashes.

## Default-OFF and composition checks

Before P2 timing, focused regression tests must establish:

- ordinary `Board` construction/make/undo does not allocate or maintain the
  D4 sidecar;
- a fresh `Searcher` has the sidecar selector OFF;
- hash OFF versus maintained-but-unconsumed ON returns identical best move,
  score, completed depth, node count, and final board in deterministic
  fixed-depth smoke searches;
- packed Pattern4 windows, candidate frontier, and D4 hashes can be enabled
  together and remain synchronized through make/undo;
- public exhaustive `Board` struct literals still compile.

## Authoritative report

The correctness binary accepts only:

```text
cb-gh0-hash-correctness --out-report NEW.json
```

Unknown, duplicate, or missing options are invalid. Output is create-new and
contains:

- source/commit and executable identity;
- all registered constants;
- transition, rule-switch, make, undo, prediction, rebuild, D4-relation,
  tie, default-OFF, and composition counts;
- D4-equivalent repeat, true collision, and intra-orbit collision counts;
- first deterministic failure witness for every nonzero failure class;
- final decision and exact claim boundary.

Two independent create-new release runs under:

```text
RUSTFLAGS=-C target-cpu=x86-64-v3
```

must produce byte-identical reports before P1-H is recorded. Timing from this
correctness binary is non-authoritative and may not be used for P2-H.
