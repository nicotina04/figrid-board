//! Fail-closed reader for the compact CB-F1 codebook artifact.
//!
//! The packed format carries the original floating-point weights for the
//! explicit float evaluator and one quantized deployment representation.  A
//! factored artifact keeps the embedding table as a class base plus an i8
//! residual; it is never expanded at load time.

use crate::codebook_eval::{
    CodebookWeights, QUANT_EMBED_SCALE, QUANT_FACTOR_SCALE, QUANT_HEAD_SCALE,
    QuantizedCodebookWeights,
};
use crate::pattern_table::PATTERN_NUM_IDS;

pub const PACKED_CODEBOOK_MAGIC: [u8; 8] = *b"NORUCBF1";
pub const PACKED_CODEBOOK_VERSION: u16 = 1;
pub const PACKED_CODEBOOK_HEADER_LEN: usize = 96;

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
        if bytes.len() < PACKED_CODEBOOK_HEADER_LEN {
            return Err(format!(
                "packed codebook is truncated: got {} bytes, need at least {PACKED_CODEBOOK_HEADER_LEN}",
                bytes.len()
            ));
        }
        if bytes[..8] != PACKED_CODEBOOK_MAGIC {
            return Err("invalid packed codebook magic".to_string());
        }

        let version = read_u16_at(bytes, 8)?;
        if version != PACKED_CODEBOOK_VERSION {
            return Err(format!("unsupported packed codebook version: {version}"));
        }
        let kind = match read_u16_at(bytes, 10)? {
            0 => PackedCodebookKind::Flat,
            1 => PackedCodebookKind::Factored,
            value => return Err(format!("unsupported packed codebook kind: {value}")),
        };
        let dim = usize::from(read_u16_at(bytes, 12)?);
        let fm_rank = usize::from(read_u16_at(bytes, 14)?);
        let regions = usize::from(read_u16_at(bytes, 16)?);
        let token_count = usize::from(read_u16_at(bytes, 18)?);
        let class_count = usize::from(read_u16_at(bytes, 20)?);
        if read_u16_at(bytes, 22)? != 0 || bytes[88..96].iter().any(|&byte| byte != 0) {
            return Err("non-zero reserved packed codebook header bytes".to_string());
        }
        if dim == 0 || fm_rank == 0 {
            return Err("packed codebook dimensions must be non-zero".to_string());
        }
        if regions != FORMAT_REGIONS {
            return Err(format!(
                "unsupported packed codebook region count: {regions}"
            ));
        }
        if token_count != PATTERN_NUM_IDS {
            return Err(format!(
                "packed codebook token count mismatch: got {token_count}, expected {PATTERN_NUM_IDS}"
            ));
        }
        match kind {
            PackedCodebookKind::Flat if class_count != 0 => {
                return Err("flat packed codebook must have zero classes".to_string());
            }
            PackedCodebookKind::Factored if class_count != FACTORED_CLASS_COUNT => {
                return Err(format!(
                    "factored packed codebook class count mismatch: got {class_count}, expected {FACTORED_CLASS_COUNT}"
                ));
            }
            _ => {}
        }

        let embedding_scale = read_i32_at(bytes, 24)?;
        let head_scale = read_i32_at(bytes, 28)?;
        let factor_scale = read_i32_at(bytes, 32)?;
        if embedding_scale != QUANT_EMBED_SCALE
            || head_scale != QUANT_HEAD_SCALE
            || factor_scale != QUANT_FACTOR_SCALE
        {
            return Err(format!(
                "unsupported packed codebook scales: embedding={embedding_scale}, head={head_scale}, factor={factor_scale}"
            ));
        }

        let f32_embedding_count = read_usize_u32_at(bytes, 36)?;
        let head_count = read_usize_u32_at(bytes, 40)?;
        let factor_count = read_usize_u32_at(bytes, 44)?;
        let quant_embedding_values = read_usize_u32_at(bytes, 48)?;
        let artifact_payload_len = read_usize_u32_at(bytes, 52)?;

        let expected_embeddings = checked_mul(token_count, dim, "embedding count")?;
        let expected_head = checked_mul(regions, dim, "head count")?;
        let expected_factors = checked_mul(expected_head, fm_rank, "factor count")?;
        if f32_embedding_count != expected_embeddings {
            return Err(format!(
                "source embedding count mismatch: got {f32_embedding_count}, expected {expected_embeddings}"
            ));
        }
        if head_count != expected_head {
            return Err(format!(
                "head count mismatch: got {head_count}, expected {expected_head}"
            ));
        }
        if factor_count != expected_factors {
            return Err(format!(
                "factor count mismatch: got {factor_count}, expected {expected_factors}"
            ));
        }
        if quant_embedding_values != expected_embeddings {
            return Err(format!(
                "quantized embedding count mismatch: got {quant_embedding_values}, expected {expected_embeddings}"
            ));
        }

        let source_scalar_count = checked_add(
            checked_add(
                checked_add(f32_embedding_count, head_count, "source scalar count")?,
                factor_count,
                "source scalar count",
            )?,
            1,
            "source scalar count",
        )?;
        let source_bytes = checked_mul(source_scalar_count, 4, "source byte count")?;
        let quantized_bytes = match kind {
            PackedCodebookKind::Flat => checked_add(
                checked_add(
                    checked_mul(quant_embedding_values, 2, "flat embedding bytes")?,
                    checked_mul(head_count, 2, "quantized head bytes")?,
                    "flat quantized bytes",
                )?,
                checked_mul(factor_count, 2, "quantized factor bytes")?,
                "flat quantized bytes",
            )?,
            PackedCodebookKind::Factored => {
                let base_values = checked_mul(class_count, dim, "factored base count")?;
                checked_add(
                    checked_add(
                        checked_add(
                            checked_add(
                                token_count,
                                checked_mul(base_values, 2, "factored base bytes")?,
                                "factored quantized bytes",
                            )?,
                            quant_embedding_values,
                            "factored quantized bytes",
                        )?,
                        checked_mul(head_count, 2, "quantized head bytes")?,
                        "factored quantized bytes",
                    )?,
                    checked_mul(factor_count, 2, "quantized factor bytes")?,
                    "factored quantized bytes",
                )?
            }
        };
        let expected_payload_len =
            checked_add(source_bytes, quantized_bytes, "artifact payload length")?;
        if artifact_payload_len != expected_payload_len {
            return Err(format!(
                "payload length mismatch: got {artifact_payload_len}, expected {expected_payload_len}"
            ));
        }
        let expected_total = checked_add(
            PACKED_CODEBOOK_HEADER_LEN,
            artifact_payload_len,
            "artifact total length",
        )?;
        if bytes.len() != expected_total {
            return Err(format!(
                "artifact length mismatch or trailing bytes: got {}, expected {expected_total}",
                bytes.len()
            ));
        }

        let mut source_sha256 = [0u8; 32];
        source_sha256.copy_from_slice(&bytes[56..88]);
        let mut cursor = PayloadCursor::new(&bytes[PACKED_CODEBOOK_HEADER_LEN..]);
        let embeddings = cursor.read_f32_vec(f32_embedding_count, "source embeddings")?;
        let head = cursor.read_f32_vec(head_count, "source head")?;
        let factors = cursor.read_f32_vec(factor_count, "source factors")?;
        let bias = cursor.read_f32("source bias")?;
        if embeddings
            .iter()
            .chain(&head)
            .chain(&factors)
            .any(|x| !x.is_finite())
            || !bias.is_finite()
        {
            return Err("packed codebook source weights contain a non-finite value".to_string());
        }
        let source_weights = CodebookWeights {
            dim,
            fm_rank,
            embeddings,
            head,
            factors,
            bias,
        };

        let quantized = match kind {
            PackedCodebookKind::Flat => {
                let embeddings = cursor.read_i16_vec(quant_embedding_values, "flat embeddings")?;
                let head = cursor.read_i16_vec(head_count, "quantized head")?;
                let factors = cursor.read_i16_vec(factor_count, "quantized factors")?;
                PackedQuantizedPayload::Flat(QuantizedCodebookWeights {
                    dim,
                    fm_rank,
                    embedding_scale,
                    head_scale,
                    factor_scale,
                    embeddings,
                    head,
                    factors,
                    bias,
                })
            }
            PackedCodebookKind::Factored => {
                let classes = cursor.read_u8_vec(token_count, "factored classes")?;
                let bases = cursor.read_i16_vec(
                    checked_mul(class_count, dim, "factored base count")?,
                    "factored bases",
                )?;
                let residuals = cursor.read_i8_vec(quant_embedding_values, "factored residuals")?;
                let head = cursor.read_i16_vec(head_count, "quantized head")?;
                let factors = cursor.read_i16_vec(factor_count, "quantized factors")?;
                let weights = FactoredQuantizedCodebookWeights {
                    dim,
                    fm_rank,
                    embedding_scale,
                    head_scale,
                    factor_scale,
                    classes,
                    bases,
                    residuals,
                    head,
                    factors,
                    bias,
                };
                weights.validate()?;
                PackedQuantizedPayload::Factored(weights)
            }
        };
        if !cursor.is_finished() {
            return Err("packed codebook payload has trailing bytes".to_string());
        }

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
    dim: usize,
    fm_rank: usize,
    embedding_scale: i32,
    head_scale: i32,
    factor_scale: i32,
    classes: Vec<u8>,
    bases: Vec<i16>,
    residuals: Vec<i8>,
    head: Vec<i16>,
    factors: Vec<i16>,
    bias: f32,
}

impl FactoredQuantizedCodebookWeights {
    pub fn validate(&self) -> Result<(), String> {
        if self.dim == 0 || self.fm_rank == 0 {
            return Err("factored codebook dimensions must be non-zero".to_string());
        }
        if self.embedding_scale != QUANT_EMBED_SCALE
            || self.head_scale != QUANT_HEAD_SCALE
            || self.factor_scale != QUANT_FACTOR_SCALE
        {
            return Err("factored codebook quantization scales are invalid".to_string());
        }
        if self.classes.len() != PATTERN_NUM_IDS {
            return Err(format!(
                "factored class length mismatch: got {}, expected {PATTERN_NUM_IDS}",
                self.classes.len()
            ));
        }
        let class_count = self.class_count();
        if class_count != FACTORED_CLASS_COUNT {
            return Err(format!(
                "factored class count mismatch: got {class_count}, expected {FACTORED_CLASS_COUNT}"
            ));
        }
        let expected_bases = checked_mul(class_count, self.dim, "factored base count")?;
        let expected_embeddings =
            checked_mul(PATTERN_NUM_IDS, self.dim, "factored residual count")?;
        let expected_head = checked_mul(FORMAT_REGIONS, self.dim, "factored head count")?;
        let expected_factors = checked_mul(expected_head, self.fm_rank, "factored factor count")?;
        if self.bases.len() != expected_bases
            || self.residuals.len() != expected_embeddings
            || self.head.len() != expected_head
            || self.factors.len() != expected_factors
        {
            return Err("factored codebook vector length mismatch".to_string());
        }
        if !self.bias.is_finite() {
            return Err("factored codebook bias is non-finite".to_string());
        }
        for (pattern_id, &class) in self.classes.iter().enumerate() {
            let class = usize::from(class);
            if class >= class_count {
                return Err(format!(
                    "factored class ID out of range at token {pattern_id}: {class}"
                ));
            }
            let base_offset = class * self.dim;
            let residual_offset = pattern_id * self.dim;
            for component in 0..self.dim {
                let reconstructed = i32::from(self.bases[base_offset + component])
                    + i32::from(self.residuals[residual_offset + component]);
                if i16::try_from(reconstructed).is_err() {
                    return Err(format!(
                        "factored embedding overflow at token {pattern_id}, component {component}: {reconstructed}"
                    ));
                }
            }
        }
        Ok(())
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    pub fn fm_rank(&self) -> usize {
        self.fm_rank
    }

    #[inline]
    pub fn feature_len(&self) -> usize {
        FORMAT_REGIONS * self.dim
    }

    #[inline]
    pub fn scales(&self) -> (i32, i32, i32) {
        (self.embedding_scale, self.head_scale, self.factor_scale)
    }

    #[inline]
    pub fn embedding_scale(&self) -> i32 {
        self.embedding_scale
    }

    #[inline]
    pub fn head_scale(&self) -> i32 {
        self.head_scale
    }

    #[inline]
    pub fn factor_scale(&self) -> i32 {
        self.factor_scale
    }

    #[inline]
    pub fn token_count(&self) -> usize {
        self.classes.len()
    }

    #[inline]
    pub fn class_count(&self) -> usize {
        self.bases.len() / self.dim
    }

    #[inline]
    pub fn classes(&self) -> &[u8] {
        &self.classes
    }

    #[inline]
    pub fn bases(&self) -> &[i16] {
        &self.bases
    }

    #[inline]
    pub fn residuals(&self) -> &[i8] {
        &self.residuals
    }

    #[inline]
    pub fn head(&self) -> &[i16] {
        &self.head
    }

    #[inline]
    pub fn factors(&self) -> &[i16] {
        &self.factors
    }

    #[inline]
    pub fn bias(&self) -> f32 {
        self.bias
    }

    #[inline(always)]
    pub fn embedding(&self, pattern_id: u16, component: usize) -> i16 {
        let pattern_id = usize::from(pattern_id);
        assert!(pattern_id < self.classes.len(), "pattern ID out of range");
        assert!(component < self.dim, "embedding component out of range");
        let class = usize::from(self.classes[pattern_id]);
        let base = self.bases[class * self.dim + component];
        let residual = self.residuals[pattern_id * self.dim + component];
        (i32::from(base) + i32::from(residual)) as i16
    }

    #[inline(always)]
    pub fn embedding_delta(&self, old: u16, new: u16, component: usize) -> i32 {
        let old = usize::from(old);
        let new = usize::from(new);
        assert!(
            old < self.classes.len() && new < self.classes.len(),
            "pattern ID out of range"
        );
        assert!(component < self.dim, "embedding component out of range");
        let old_class = usize::from(self.classes[old]);
        let new_class = usize::from(self.classes[new]);
        let mut delta = i32::from(self.residuals[new * self.dim + component])
            - i32::from(self.residuals[old * self.dim + component]);
        if old_class != new_class {
            delta += i32::from(self.bases[new_class * self.dim + component])
                - i32::from(self.bases[old_class * self.dim + component]);
        }
        delta
    }

    pub fn reconstruct_flat(&self) -> QuantizedCodebookWeights {
        self.validate()
            .expect("cannot reconstruct an invalid factored codebook");
        let mut embeddings = Vec::with_capacity(PATTERN_NUM_IDS * self.dim);
        for pattern_id in 0..PATTERN_NUM_IDS {
            for component in 0..self.dim {
                embeddings.push(self.embedding(pattern_id as u16, component));
            }
        }
        QuantizedCodebookWeights {
            dim: self.dim,
            fm_rank: self.fm_rank,
            embedding_scale: self.embedding_scale,
            head_scale: self.head_scale,
            factor_scale: self.factor_scale,
            embeddings,
            head: self.head.clone(),
            factors: self.factors.clone(),
            bias: self.bias,
        }
    }

    pub fn payload_bytes(&self) -> usize {
        self.classes.len()
            + self.bases.len() * size_of::<i16>()
            + self.residuals.len() * size_of::<i8>()
            + self.head.len() * size_of::<i16>()
            + self.factors.len() * size_of::<i16>()
            + size_of::<f32>()
    }
}

fn checked_mul(left: usize, right: usize, field: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("{field} overflows usize"))
}

fn checked_add(left: usize, right: usize, field: &str) -> Result<usize, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("{field} overflows usize"))
}

fn read_u16_at(bytes: &[u8], offset: usize) -> Result<u16, String> {
    let raw = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| "truncated packed codebook header".to_string())?;
    Ok(u16::from_le_bytes([raw[0], raw[1]]))
}

fn read_i32_at(bytes: &[u8], offset: usize) -> Result<i32, String> {
    let raw = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| "truncated packed codebook header".to_string())?;
    Ok(i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
}

fn read_usize_u32_at(bytes: &[u8], offset: usize) -> Result<usize, String> {
    let raw = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| "truncated packed codebook header".to_string())?;
    usize::try_from(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
        .map_err(|_| "packed codebook count does not fit usize".to_string())
}

struct PayloadCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> PayloadCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: usize, field: &str) -> Result<&'a [u8], String> {
        let end = checked_add(self.offset, len, field)?;
        let out = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| format!("truncated {field}"))?;
        self.offset = end;
        Ok(out)
    }

    fn read_f32(&mut self, field: &str) -> Result<f32, String> {
        let raw = self.take(4, field)?;
        Ok(f32::from_bits(u32::from_le_bytes([
            raw[0], raw[1], raw[2], raw[3],
        ])))
    }

    fn read_f32_vec(&mut self, count: usize, field: &str) -> Result<Vec<f32>, String> {
        let raw = self.take(checked_mul(count, 4, field)?, field)?;
        Ok(raw
            .chunks_exact(4)
            .map(|item| f32::from_bits(u32::from_le_bytes(item.try_into().unwrap())))
            .collect())
    }

    fn read_i16_vec(&mut self, count: usize, field: &str) -> Result<Vec<i16>, String> {
        let raw = self.take(checked_mul(count, 2, field)?, field)?;
        Ok(raw
            .chunks_exact(2)
            .map(|item| i16::from_le_bytes(item.try_into().unwrap()))
            .collect())
    }

    fn read_u8_vec(&mut self, count: usize, field: &str) -> Result<Vec<u8>, String> {
        Ok(self.take(count, field)?.to_vec())
    }

    fn read_i8_vec(&mut self, count: usize, field: &str) -> Result<Vec<i8>, String> {
        Ok(self
            .take(count, field)?
            .iter()
            .map(|&value| value as i8)
            .collect())
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
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

    fn source_payload_bytes(bytes: &[u8]) -> usize {
        let embeddings = read_usize_u32_at(bytes, 36).unwrap();
        let head = read_usize_u32_at(bytes, 40).unwrap();
        let factors = read_usize_u32_at(bytes, 44).unwrap();
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
        assert_eq!(
            (
                reconstructed.embedding_scale,
                reconstructed.head_scale,
                reconstructed.factor_scale,
            ),
            (
                expected.embedding_scale,
                expected.head_scale,
                expected.factor_scale,
            )
        );

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
        let head_count = u32::try_from(FORMAT_REGIONS * 16).unwrap();
        put_u32(&mut bad_count, 40, head_count - 1);
        assert_parse_fails(bad_count, "count");

        let mut bad_scale = FACTORED_ARTIFACT.to_vec();
        put_i32(&mut bad_scale, 24, QUANT_EMBED_SCALE + 1);
        assert_parse_fails(bad_scale, "scale");

        let mut bad_payload_len = FACTORED_ARTIFACT.to_vec();
        let payload_len =
            u32::try_from(bad_payload_len.len() - PACKED_CODEBOOK_HEADER_LEN).unwrap();
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
