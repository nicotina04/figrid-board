//! Incrementally updatable categorical codebook embeddings.
//!
//! CB2Vec provides the game-independent part of an evaluator built from
//! integer token IDs, shared embedding rows, grouped pooling, and a small
//! linear or factorization-machine head. It also provides a preallocated
//! reversible journal for search engines that use make/undo.
//!
//! Vocabulary construction, legal actions, board mutation, perspective
//! mapping, and search policy remain responsibilities of the consuming
//! application.

#![forbid(unsafe_code)]

mod artifact;
mod factored;
mod journal;
mod model;

pub use artifact::{
    ArtifactError, CB2VEC_ARTIFACT_HEADER_LEN, CB2VEC_ARTIFACT_MAGIC, CB2VEC_ARTIFACT_VERSION,
    LEGACY_NORU_CBF_MAGIC, PackedCodebookArtifact, PackedCodebookKind, PackedQuantizedPayload,
};
pub use factored::FactoredQuantizedCodebookWeights;
pub use journal::{
    JournalError, ReversibleTokenJournal, TokenDelta, TokenDeltaPop, TokenDeltaReplay,
    TokenDeltaSink,
};
pub use model::{
    CodebookWeights, FloatCodebookAccess, ModelError, ModelShape, QUANT_EMBED_SCALE,
    QUANT_FACTOR_SCALE, QUANT_HEAD_SCALE, QuantizedCodebookAccess, QuantizedCodebookWeights,
    add_embedding_delta_to, add_embedding_to, quantize_i16, score_f32, score_quantized_uniform,
};
