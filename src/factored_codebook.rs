//! FIGRID compatibility facade for packed CB2Vec artifacts.
//!
//! Parsing, shape arithmetic, factored storage, and reconstruction live in
//! the independent `cb2vec` crate. This module preserves the public 0.8.x
//! FIGRID types and applies the stricter Pattern4 deployment schema.

use cb2vec::QuantizedCodebookAccess;

use crate::codebook_eval::{
    CodebookWeights, QUANT_EMBED_SCALE, QUANT_FACTOR_SCALE, QUANT_HEAD_SCALE,
    QuantizedCodebookWeights,
};
use crate::pattern_table::PATTERN_NUM_IDS;

pub const PACKED_CODEBOOK_MAGIC: [u8; 8] = cb2vec::LEGACY_NORU_CBF_MAGIC;
pub const PACKED_CODEBOOK_VERSION: u16 = cb2vec::CB2VEC_ARTIFACT_VERSION;
pub const PACKED_CODEBOOK_HEADER_LEN: usize = cb2vec::CB2VEC_ARTIFACT_HEADER_LEN;

const FORMAT_REGIONS: usize = 9;
const FACTORED_CLASS_COUNT: usize = 5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackedCodebookKind {
    Flat,
    Factored,
}

#[derive(Clone, Debug)]
pub enum PackedQuantizedPayload {
    Flat(QuantizedCodebookWeights),
    Factored(FactoredQuantizedCodebookWeights),
}

#[derive(Clone, Debug)]
pub struct PackedCodebookArtifact {
    kind: PackedCodebookKind,
    source_weights: CodebookWeights,
    quantized: PackedQuantizedPayload,
    source_sha256: [u8; 32],
    artifact_payload_len: usize,
}

impl PackedCodebookArtifact {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let artifact =
            cb2vec::PackedCodebookArtifact::parse(bytes).map_err(|error| error.to_string())?;
        let artifact_payload_len = bytes
            .len()
            .checked_sub(PACKED_CODEBOOK_HEADER_LEN)
            .ok_or_else(|| "packed codebook is shorter than its header".to_string())?;
        let source_sha256 = *artifact.source_sha256();
        let (source, quantized) = artifact.into_parts();
        let source_shape = source.validate().map_err(|error| error.to_string())?;
        validate_figrid_shape(
            source_shape.token_count(),
            source_shape.group_count(),
            source_shape.dim(),
            source_shape.fm_rank(),
        )?;
        let source_weights = CodebookWeights {
            dim: source.dim,
            fm_rank: source.fm_rank,
            embeddings: source.embeddings,
            head: source.head,
            factors: source.factors,
            bias: source.bias,
        };

        let (kind, quantized) = match quantized {
            cb2vec::PackedQuantizedPayload::Flat(weights) => {
                validate_scales(
                    weights.embedding_scale,
                    weights.head_scale,
                    weights.factor_scale,
                )?;
                (
                    PackedCodebookKind::Flat,
                    PackedQuantizedPayload::Flat(QuantizedCodebookWeights {
                        dim: weights.dim,
                        fm_rank: weights.fm_rank,
                        embedding_scale: weights.embedding_scale,
                        head_scale: weights.head_scale,
                        factor_scale: weights.factor_scale,
                        embeddings: weights.embeddings,
                        head: weights.head,
                        factors: weights.factors,
                        bias: weights.bias,
                    }),
                )
            }
            cb2vec::PackedQuantizedPayload::Factored(weights) => {
                let wrapped = FactoredQuantizedCodebookWeights { inner: weights };
                wrapped.validate()?;
                (
                    PackedCodebookKind::Factored,
                    PackedQuantizedPayload::Factored(wrapped),
                )
            }
        };

        Ok(Self {
            kind,
            source_weights,
            quantized,
            source_sha256,
            artifact_payload_len,
        })
    }

    #[inline]
    pub fn kind(&self) -> PackedCodebookKind {
        self.kind
    }

    #[inline]
    pub fn source_weights(&self) -> &CodebookWeights {
        &self.source_weights
    }

    #[inline]
    pub fn source_sha256(&self) -> &[u8; 32] {
        &self.source_sha256
    }

    #[inline]
    pub fn artifact_payload_len(&self) -> usize {
        self.artifact_payload_len
    }

    #[inline]
    pub fn flat_quantized(&self) -> Option<&QuantizedCodebookWeights> {
        match &self.quantized {
            PackedQuantizedPayload::Flat(weights) => Some(weights),
            PackedQuantizedPayload::Factored(_) => None,
        }
    }

    #[inline]
    pub fn factored_quantized(&self) -> Option<&FactoredQuantizedCodebookWeights> {
        match &self.quantized {
            PackedQuantizedPayload::Flat(_) => None,
            PackedQuantizedPayload::Factored(weights) => Some(weights),
        }
    }

    pub fn into_source_weights(self) -> CodebookWeights {
        self.source_weights
    }

    pub fn into_quantized_payload(self) -> PackedQuantizedPayload {
        self.quantized
    }

    pub fn into_parts(self) -> (CodebookWeights, PackedQuantizedPayload) {
        (self.source_weights, self.quantized)
    }

    pub fn into_flat_quantized(self) -> Result<QuantizedCodebookWeights, String> {
        match self.quantized {
            PackedQuantizedPayload::Flat(weights) => Ok(weights),
            PackedQuantizedPayload::Factored(_) => {
                Err("packed codebook does not contain flat quantized weights".to_string())
            }
        }
    }

    pub fn into_factored_quantized(self) -> Result<FactoredQuantizedCodebookWeights, String> {
        match self.quantized {
            PackedQuantizedPayload::Factored(weights) => Ok(weights),
            PackedQuantizedPayload::Flat(_) => {
                Err("packed codebook does not contain factored quantized weights".to_string())
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct FactoredQuantizedCodebookWeights {
    inner: cb2vec::FactoredQuantizedCodebookWeights,
}

impl FactoredQuantizedCodebookWeights {
    pub fn validate(&self) -> Result<(), String> {
        let shape = self.inner.validate().map_err(|error| error.to_string())?;
        validate_figrid_shape(
            shape.token_count(),
            shape.group_count(),
            shape.dim(),
            shape.fm_rank(),
        )?;
        if self.class_count() != FACTORED_CLASS_COUNT {
            return Err(format!(
                "factored class count mismatch: got {}, expected {FACTORED_CLASS_COUNT}",
                self.class_count()
            ));
        }
        validate_scales(
            self.embedding_scale(),
            self.head_scale(),
            self.factor_scale(),
        )
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[inline]
    pub fn fm_rank(&self) -> usize {
        self.inner.fm_rank()
    }

    #[inline]
    pub fn feature_len(&self) -> usize {
        FORMAT_REGIONS * self.dim()
    }

    #[inline]
    pub fn scales(&self) -> (i32, i32, i32) {
        (
            self.embedding_scale(),
            self.head_scale(),
            self.factor_scale(),
        )
    }

    #[inline]
    pub fn embedding_scale(&self) -> i32 {
        self.inner.embedding_scale()
    }

    #[inline]
    pub fn head_scale(&self) -> i32 {
        self.inner.head_scale()
    }

    #[inline]
    pub fn factor_scale(&self) -> i32 {
        self.inner.factor_scale()
    }

    #[inline]
    pub fn token_count(&self) -> usize {
        self.inner.token_count()
    }

    #[inline]
    pub fn class_count(&self) -> usize {
        self.inner.class_count()
    }

    #[inline]
    pub fn classes(&self) -> &[u8] {
        self.inner.classes()
    }

    #[inline]
    pub fn bases(&self) -> &[i16] {
        self.inner.bases()
    }

    #[inline]
    pub fn residuals(&self) -> &[i8] {
        self.inner.residuals()
    }

    #[inline]
    pub fn head(&self) -> &[i16] {
        self.inner.head()
    }

    #[inline]
    pub fn factors(&self) -> &[i16] {
        self.inner.factors()
    }

    #[inline]
    pub fn bias(&self) -> f32 {
        self.inner.bias()
    }

    #[inline(always)]
    pub fn embedding(&self, pattern_id: u16, component: usize) -> i16 {
        assert!(
            usize::from(pattern_id) < self.token_count(),
            "pattern ID out of range"
        );
        assert!(component < self.dim(), "embedding component out of range");
        self.inner.embedding(pattern_id, component)
    }

    #[inline(always)]
    pub fn embedding_delta(&self, old: u16, new: u16, component: usize) -> i32 {
        assert!(
            usize::from(old) < self.token_count() && usize::from(new) < self.token_count(),
            "pattern ID out of range"
        );
        assert!(component < self.dim(), "embedding component out of range");
        self.inner.embedding_delta(old, new, component)
    }

    pub fn reconstruct_flat(&self) -> QuantizedCodebookWeights {
        self.validate()
            .expect("cannot reconstruct an invalid factored codebook");
        let flat = self.inner.reconstruct_flat();
        QuantizedCodebookWeights {
            dim: flat.dim,
            fm_rank: flat.fm_rank,
            embedding_scale: flat.embedding_scale,
            head_scale: flat.head_scale,
            factor_scale: flat.factor_scale,
            embeddings: flat.embeddings,
            head: flat.head,
            factors: flat.factors,
            bias: flat.bias,
        }
    }

    pub fn payload_bytes(&self) -> usize {
        self.classes().len()
            + std::mem::size_of_val(self.bases())
            + std::mem::size_of_val(self.residuals())
            + std::mem::size_of_val(self.head())
            + std::mem::size_of_val(self.factors())
            + size_of::<f32>()
    }
}

fn validate_figrid_shape(
    token_count: usize,
    group_count: usize,
    dim: usize,
    fm_rank: usize,
) -> Result<(), String> {
    if dim == 0 || fm_rank == 0 {
        return Err("packed codebook dimensions must be non-zero".to_string());
    }
    if token_count != PATTERN_NUM_IDS {
        return Err(format!(
            "packed codebook token count mismatch: got {token_count}, expected {PATTERN_NUM_IDS}"
        ));
    }
    if group_count != FORMAT_REGIONS {
        return Err(format!(
            "unsupported packed codebook region count: {group_count}"
        ));
    }
    Ok(())
}

fn validate_scales(embedding_scale: i32, head_scale: i32, factor_scale: i32) -> Result<(), String> {
    if embedding_scale != QUANT_EMBED_SCALE
        || head_scale != QUANT_HEAD_SCALE
        || factor_scale != QUANT_FACTOR_SCALE
    {
        return Err(format!(
            "unsupported packed codebook scales: embedding={embedding_scale}, head={head_scale}, factor={factor_scale}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern_table::swap_mapped_id;

    const FACTORED_ARTIFACT: &[u8] =
        include_bytes!("../models/gomoku_codebook_v1_swapclosed_factored.cbf");
    const SOURCE_JSON: &[u8] = include_bytes!("../models/gomoku_codebook_v1_swapclosed.json");

    fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
        bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn put_i32(bytes: &mut [u8], offset: usize, value: i32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn read_u32(bytes: &[u8], offset: usize) -> usize {
        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize
    }

    fn source_payload_bytes(bytes: &[u8]) -> usize {
        let embeddings = read_u32(bytes, 36);
        let head = read_u32(bytes, 40);
        let factors = read_u32(bytes, 44);
        (embeddings + head + factors + 1) * size_of::<f32>()
    }

    fn assert_parse_fails(bytes: Vec<u8>, label: &str) {
        assert!(
            PackedCodebookArtifact::parse(&bytes).is_err(),
            "{label} must fail closed"
        );
    }

    #[test]
    fn factored_artifact_source_and_quantized_weights_are_bit_exact() {
        let source = CodebookWeights::from_json_bytes(SOURCE_JSON).expect("source JSON");
        let artifact = PackedCodebookArtifact::parse(FACTORED_ARTIFACT).expect("factored artifact");
        assert_eq!(artifact.kind(), PackedCodebookKind::Factored);
        assert_eq!(artifact.source_weights().dim, source.dim);
        assert_eq!(artifact.source_weights().fm_rank, source.fm_rank);
        assert!(
            artifact
                .source_weights()
                .embeddings
                .iter()
                .zip(&source.embeddings)
                .all(|(left, right)| left.to_bits() == right.to_bits())
        );
        assert!(
            artifact
                .source_weights()
                .head
                .iter()
                .zip(&source.head)
                .all(|(left, right)| left.to_bits() == right.to_bits())
        );
        assert!(
            artifact
                .source_weights()
                .factors
                .iter()
                .zip(&source.factors)
                .all(|(left, right)| left.to_bits() == right.to_bits())
        );
        assert_eq!(
            artifact.source_weights().bias.to_bits(),
            source.bias.to_bits()
        );

        let expected = source.quantize_i16_s32_s64();
        let factored = artifact
            .into_factored_quantized()
            .expect("factored payload");
        let reconstructed = factored.reconstruct_flat();
        assert_eq!(reconstructed.embeddings, expected.embeddings);
        assert_eq!(reconstructed.head, expected.head);
        assert_eq!(reconstructed.factors, expected.factors);
        assert_eq!(reconstructed.bias.to_bits(), expected.bias.to_bits());
        assert_eq!(factored.scales(), (32, 64, 64));

        for pattern_id in 0..PATTERN_NUM_IDS {
            let black = pattern_id as u16;
            let white = swap_mapped_id(black);
            for component in 0..factored.dim() {
                assert_eq!(
                    factored.embedding(black, component),
                    expected.embeddings[pattern_id * factored.dim() + component]
                );
                assert_eq!(
                    factored.embedding(white, component),
                    expected.embeddings[usize::from(white) * factored.dim() + component]
                );
            }
        }
    }

    #[test]
    fn malformed_headers_and_lengths_fail_closed() {
        let mut bad_magic = FACTORED_ARTIFACT.to_vec();
        bad_magic[0] ^= 0xff;
        assert_parse_fails(bad_magic, "magic");

        let mut bad_version = FACTORED_ARTIFACT.to_vec();
        put_u16(&mut bad_version, 8, PACKED_CODEBOOK_VERSION + 1);
        assert_parse_fails(bad_version, "version");

        let mut bad_dimension = FACTORED_ARTIFACT.to_vec();
        put_u16(&mut bad_dimension, 12, 0);
        assert_parse_fails(bad_dimension, "dimension");

        let mut bad_count = FACTORED_ARTIFACT.to_vec();
        put_u32(&mut bad_count, 40, (FORMAT_REGIONS * 16 - 1) as u32);
        assert_parse_fails(bad_count, "count");

        let mut bad_scale = FACTORED_ARTIFACT.to_vec();
        put_i32(&mut bad_scale, 24, QUANT_EMBED_SCALE + 1);
        assert_parse_fails(bad_scale, "scale");

        let mut bad_payload_len = FACTORED_ARTIFACT.to_vec();
        let payload_len = (bad_payload_len.len() - PACKED_CODEBOOK_HEADER_LEN) as u32;
        put_u32(&mut bad_payload_len, 52, payload_len - 1);
        assert_parse_fails(bad_payload_len, "payload length");

        let mut trailing = FACTORED_ARTIFACT.to_vec();
        trailing.push(0);
        assert_parse_fails(trailing, "trailing byte");
    }

    #[test]
    fn out_of_range_factored_class_fails_closed() {
        let mut bytes = FACTORED_ARTIFACT.to_vec();
        let class_offset = PACKED_CODEBOOK_HEADER_LEN + source_payload_bytes(&bytes);
        bytes[class_offset] = u8::MAX;
        assert_parse_fails(bytes, "class ID");
    }
}
