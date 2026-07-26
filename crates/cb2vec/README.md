<h1 align="center">CB2Vec</h1>

<p align="center">
Exact categorical codebook embeddings and reversible token deltas for Rust search engines.
</p>

<p align="center">
  <a href="https://crates.io/crates/cb2vec"><img alt="crates.io" src="https://img.shields.io/crates/v/cb2vec.svg"></a>
  <a href="https://crates.io/crates/cb2vec"><img alt="license" src="https://img.shields.io/crates/l/cb2vec.svg"></a>
  <a href="https://docs.rs/cb2vec"><img alt="docs.rs" src="https://docs.rs/cb2vec/badge.svg"></a>
</p>

## What is CB2Vec?

CB2Vec is the game-independent runtime extracted from FIGRID's categorical
codebook evaluator. It provides:

- floating-point and `i16` codebook model representations;
- exact integer embedding lookup and replacement deltas;
- grouped linear and factorization-machine scoring;
- exact class-base plus `i8` residual storage;
- a versioned, fail-closed binary artifact;
- a preallocated reversible token journal for make/undo search.

CB2Vec is not Word2Vec, a tokenizer, a vocabulary learner, or a training
framework. A consuming application decides what its integer tokens mean,
which sites changed, how sites map to groups, and how legal actions and
search are implemented.

## Why this crate?

Codebook evaluators are often written directly inside one game engine. That
makes the useful runtime machinery difficult to reuse and easy to couple to a
specific board size, vocabulary, or color convention.

CB2Vec separates the reusable numerical core:

```text
domain state
  -> categorical tokens at (site, lane)
  -> shared embedding rows
  -> consumer-defined activation and grouped pooling
  -> linear or factorization-machine head
```

The domain adapter remains responsible for token production and perspective
mapping. The core never imports a board, move, ruleset, or search type.

## Quick start

Add CB2Vec to `Cargo.toml`:

```toml
[dependencies]
cb2vec = "0.1"
```

Create and quantize a small model, then replace one token exactly:

```rust
use cb2vec::{
    CodebookWeights, add_embedding_delta_to, add_embedding_to,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 16 tokens, 3 pooled groups, 8 embedding components, FM rank 2.
    let source = CodebookWeights::deterministic(16, 3, 8, 2);
    let model = source.quantize_i16_s32_s64();

    let mut site = vec![0i32; model.dim];
    add_embedding_to(&model, 2, &mut site)?;
    add_embedding_delta_to(&model, 2, 7, &mut site)?;

    let mut expected = vec![0i32; model.dim];
    add_embedding_to(&model, 7, &mut expected)?;
    assert_eq!(site, expected);
    Ok(())
}
```

The checked free functions reject an out-of-range token or incorrectly sized
output buffer. Validated search hot paths can call the statically dispatched
`QuantizedCodebookAccess` methods directly.

## Attaching a policy

CB2Vec can provide the state representation for a policy without making
actions part of the core crate. A domain adapter maps state into tokens and
maintains the embedding or pooled feature vector; a `PolicyHead` in the
consumer then maps that vector to action logits and applies the domain's
legal-action mask.

This boundary lets the same CB2Vec model support a value head, a policy head,
or both, while action encoding, legality, and search remain game-specific. A
reusable policy implementation can therefore live beside CB2Vec, or in a
future companion crate, rather than inside the token and journal primitives.

## Reversible token updates

A token state consists of:

- a `site`, such as a cell or entity;
- one or more categorical `lanes` at that site;
- a logical search depth;
- a materialized depth already applied to the numeric state.

`ReversibleTokenJournal<T, LANES, MAX_DELTAS>` owns the logical tokens and
preallocates every frame at construction. `push_after` records changed lanes,
`materialize_pending` applies grouped site deltas to a consumer-defined sink,
and `pop` reverses an applied frame when necessary.

See [`examples/reversible_journal.rs`](examples/reversible_journal.rs) for a
complete make/materialize/undo round trip.

## Model and head

`CodebookWeights` stores:

- `token_count * dim` floating-point embedding values;
- `group_count * dim` linear-head values;
- `(group_count * dim) * fm_rank` optional FM factors;
- one floating-point bias.

`QuantizedCodebookWeights` stores the same logical model with separate
positive scales for embeddings, the linear head, and FM factors. The initial
FIGRID deployment uses scales 32, 64, and 64; callers may choose other scales
with `quantize_i16`.

The checked `score_f32` function consumes already normalized floating-point
features. `score_quantized_uniform` consumes integer grouped sums when every
group has the same pooling divisor. Both return an error for an invalid feature
shape; the quantized scorer also rejects a zero divisor.

## Flat and factored storage

`FactoredQuantizedCodebookWeights` stores each embedding row as:

```text
class_base[token_class] + i8_residual[token]
```

Reconstruction is exact and checked for `i16` overflow. This representation
can reduce serialized size, but it is not assumed to be the fastest hot-loop
layout. Use `reconstruct_flat` at load time when flat row access is faster on
the target workload.

## Artifact format

The canonical v1 artifact uses:

- magic `CB2VEC01`;
- little-endian integers and floating-point bit patterns;
- explicit model shape and quantization scales;
- a flat or factored payload kind;
- exact payload lengths and zeroed reserved bytes;
- a caller-supplied 32-byte source provenance digest;
- rejection of unknown versions, malformed shapes, non-finite source values,
  and trailing bytes.

`PackedCodebookArtifact::to_bytes` writes the canonical format.
`PackedCodebookArtifact::parse` also reads the legacy `NORUCBF1` magic so
FIGRID artifacts can migrate without changing their numerical payload.
Legacy input is always rewritten with the canonical CB2Vec magic.

The source digest is provenance metadata supplied by the packer. CB2Vec does
not claim that it hashes or authenticates the original training file.

See [`examples/flat_roundtrip.rs`](examples/flat_roundtrip.rs).

## Core guarantees

- Integer embedding replacement is exact for a valid model.
- Journal validation errors do not mutate logical tokens or logical depth.
- Journal push, materialize, and pop allocate nothing after construction.
- Artifact parsing uses checked length arithmetic and rejects trailing data.
- The crate contains no game, board, action, vocabulary, or search policy.
- The crate contains no `unsafe` code.

A `TokenDeltaSink` must not panic. Sink mutations are deliberately
non-transactional because rollback would add overhead and cannot generally
undo arbitrary external side effects.

## Scope and non-goals

CB2Vec 0.1 focuses on inference, storage, and reversible state changes. It
does not yet provide:

- vocabulary construction or tokenization;
- gradient training, QAT, or an optimizer;
- legal-action masking or a policy head;
- board symmetry or color-perspective semantics;
- SIMD, a C ABI, or a `no_std` contract.

Policy or value heads built for one game should remain in that game's adapter
until a second domain demonstrates the same abstraction.

## Evidence and provenance

The code was extracted from the FIGRID 0.8.3 evaluator after the following
integration gates:

- exact mixed make/undo comparison against full rebuild over 100,000
  transitions;
- exact search decision and node-count comparison on a sealed 1,022-root
  corpus;
- bit-exact reconstruction of the deployed flat integer model from factored
  storage;
- same-binary measurement of the directional token-delta evaluator.

Those are FIGRID integration results, not universal speed claims for every
CB2Vec consumer. Workload-level performance still depends on token locality,
embedding dimension, group layout, and search behavior.

## Relationship to NORU

[NORU](https://crates.io/crates/noru) and CB2Vec are sibling Rust primitives:

- NORU maps sparse global features through an NNUE accumulator and dense MLP.
- CB2Vec maps local categorical tokens through shared embeddings and a small
  grouped head.

[figrid-board](https://crates.io/crates/figrid-board) uses NORU for its
legacy NNUE lineage and CB2Vec for the promoted codebook evaluator.

## Development

```sh
cargo fmt --all --check
cargo test -p cb2vec --all-features
cargo clippy -p cb2vec --all-targets --all-features -- -D warnings
cargo doc -p cb2vec --all-features --no-deps
cargo package -p cb2vec --locked
```

The repository keeps FIGRID integration tests beside the game adapter, while
generic model, journal, factored-storage, and artifact tests live in this
crate.

## License

Licensed under either of:

- Apache License, Version 2.0
- MIT License

at your option.
