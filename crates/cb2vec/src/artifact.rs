use std::error::Error;
use std::fmt;

use crate::{
    CodebookWeights, FactoredQuantizedCodebookWeights, ModelShape, QuantizedCodebookWeights,
};

pub const CB2VEC_ARTIFACT_MAGIC: [u8; 8] = *b"CB2VEC01";
pub const LEGACY_NORU_CBF_MAGIC: [u8; 8] = *b"NORUCBF1";
pub const CB2VEC_ARTIFACT_VERSION: u16 = 1;
pub const CB2VEC_ARTIFACT_HEADER_LEN: usize = 96;

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
    legacy_magic: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ArtifactError {
    Truncated { actual: usize, minimum: usize },
    InvalidMagic,
    UnsupportedVersion(u16),
    UnsupportedKind(u16),
    NonZeroReserved,
    InvalidShape(String),
    InvalidScale(String),
    LengthMismatch(String),
    BiasMismatch,
    NonFinite(String),
    ValueOutOfRange(String),
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { actual, minimum } => {
                write!(
                    f,
                    "artifact is truncated: got {actual} bytes, need {minimum}"
                )
            }
            Self::InvalidMagic => write!(f, "invalid CB2Vec artifact magic"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported CB2Vec artifact version {version}")
            }
            Self::UnsupportedKind(kind) => write!(f, "unsupported CB2Vec artifact kind {kind}"),
            Self::NonZeroReserved => write!(f, "reserved artifact bytes must be zero"),
            Self::InvalidShape(message) => write!(f, "invalid artifact shape: {message}"),
            Self::InvalidScale(message) => write!(f, "invalid artifact scale: {message}"),
            Self::LengthMismatch(message) => write!(f, "artifact length mismatch: {message}"),
            Self::BiasMismatch => {
                write!(
                    f,
                    "source and quantized artifact biases must be bit-identical"
                )
            }
            Self::NonFinite(message) => write!(f, "artifact contains non-finite {message}"),
            Self::ValueOutOfRange(message) => write!(f, "artifact value out of range: {message}"),
        }
    }
}

impl Error for ArtifactError {}

impl PackedCodebookArtifact {
    pub fn new_flat(
        source_weights: CodebookWeights,
        quantized: QuantizedCodebookWeights,
        source_sha256: [u8; 32],
    ) -> Result<Self, ArtifactError> {
        let source_shape = source_weights.validate().map_err(shape_error)?;
        same_shape(source_shape, quantized.validate().map_err(shape_error)?)?;
        same_bias(source_weights.bias, quantized.bias)?;
        ensure_serializable_shape(source_shape, 0, PackedCodebookKind::Flat)?;
        Ok(Self {
            kind: PackedCodebookKind::Flat,
            source_weights,
            quantized: PackedQuantizedPayload::Flat(quantized),
            source_sha256,
            legacy_magic: false,
        })
    }

    pub fn new_factored(
        source_weights: CodebookWeights,
        quantized: FactoredQuantizedCodebookWeights,
        source_sha256: [u8; 32],
    ) -> Result<Self, ArtifactError> {
        let source_shape = source_weights.validate().map_err(shape_error)?;
        same_shape(source_shape, quantized.validate().map_err(shape_error)?)?;
        same_bias(source_weights.bias, quantized.bias())?;
        ensure_serializable_shape(
            source_shape,
            quantized.class_count(),
            PackedCodebookKind::Factored,
        )?;
        Ok(Self {
            kind: PackedCodebookKind::Factored,
            source_weights,
            quantized: PackedQuantizedPayload::Factored(quantized),
            source_sha256,
            legacy_magic: false,
        })
    }

    pub fn parse(bytes: &[u8]) -> Result<Self, ArtifactError> {
        if bytes.len() < CB2VEC_ARTIFACT_HEADER_LEN {
            return Err(ArtifactError::Truncated {
                actual: bytes.len(),
                minimum: CB2VEC_ARTIFACT_HEADER_LEN,
            });
        }
        let legacy_magic = if bytes[..8] == CB2VEC_ARTIFACT_MAGIC {
            false
        } else if bytes[..8] == LEGACY_NORU_CBF_MAGIC {
            true
        } else {
            return Err(ArtifactError::InvalidMagic);
        };
        let version = read_u16(bytes, 8)?;
        if version != CB2VEC_ARTIFACT_VERSION {
            return Err(ArtifactError::UnsupportedVersion(version));
        }
        let kind = match read_u16(bytes, 10)? {
            0 => PackedCodebookKind::Flat,
            1 => PackedCodebookKind::Factored,
            value => return Err(ArtifactError::UnsupportedKind(value)),
        };
        let dim = usize::from(read_u16(bytes, 12)?);
        let fm_rank = usize::from(read_u16(bytes, 14)?);
        let group_count = usize::from(read_u16(bytes, 16)?);
        let token_count = usize::from(read_u16(bytes, 18)?);
        let class_count = usize::from(read_u16(bytes, 20)?);
        if read_u16(bytes, 22)? != 0 || bytes[88..96].iter().any(|&byte| byte != 0) {
            return Err(ArtifactError::NonZeroReserved);
        }
        let shape = ModelShape::new(token_count, group_count, dim, fm_rank).map_err(shape_error)?;
        match kind {
            PackedCodebookKind::Flat if class_count != 0 => {
                return Err(ArtifactError::InvalidShape(
                    "flat payload must have zero classes".to_string(),
                ));
            }
            PackedCodebookKind::Factored if class_count == 0 => {
                return Err(ArtifactError::InvalidShape(
                    "factored payload must have at least one class".to_string(),
                ));
            }
            _ => {}
        }

        let embedding_scale = read_i32(bytes, 24)?;
        let head_scale = read_i32(bytes, 28)?;
        let factor_scale = read_i32(bytes, 32)?;
        if embedding_scale <= 0 || head_scale <= 0 || factor_scale <= 0 {
            return Err(ArtifactError::InvalidScale(format!(
                "embedding={embedding_scale}, head={head_scale}, factor={factor_scale}"
            )));
        }

        let source_embedding_count = read_u32_usize(bytes, 36)?;
        let head_count = read_u32_usize(bytes, 40)?;
        let factor_count = read_u32_usize(bytes, 44)?;
        let quant_embedding_count = read_u32_usize(bytes, 48)?;
        let payload_len = read_u32_usize(bytes, 52)?;
        expect_count(
            "source embeddings",
            source_embedding_count,
            shape.embedding_len().map_err(shape_error)?,
        )?;
        expect_count(
            "head",
            head_count,
            shape.feature_len().map_err(shape_error)?,
        )?;
        expect_count(
            "factors",
            factor_count,
            shape.factor_len().map_err(shape_error)?,
        )?;
        expect_count(
            "quantized embeddings",
            quant_embedding_count,
            shape.embedding_len().map_err(shape_error)?,
        )?;

        let source_scalar_count = checked_add(
            checked_add(
                checked_add(source_embedding_count, head_count, "source scalars")?,
                factor_count,
                "source scalars",
            )?,
            1,
            "source scalars",
        )?;
        let source_bytes = checked_mul(source_scalar_count, 4, "source bytes")?;
        let quantized_bytes = match kind {
            PackedCodebookKind::Flat => checked_add(
                checked_add(
                    checked_mul(quant_embedding_count, 2, "flat embeddings")?,
                    checked_mul(head_count, 2, "quantized head")?,
                    "flat quantized payload",
                )?,
                checked_mul(factor_count, 2, "quantized factors")?,
                "flat quantized payload",
            )?,
            PackedCodebookKind::Factored => checked_add(
                checked_add(
                    checked_add(
                        checked_add(
                            token_count,
                            checked_mul(
                                checked_mul(class_count, dim, "class bases")?,
                                2,
                                "class base bytes",
                            )?,
                            "factored payload",
                        )?,
                        quant_embedding_count,
                        "factored payload",
                    )?,
                    checked_mul(head_count, 2, "quantized head")?,
                    "factored payload",
                )?,
                checked_mul(factor_count, 2, "quantized factors")?,
                "factored payload",
            )?,
        };
        let expected_payload = checked_add(source_bytes, quantized_bytes, "payload")?;
        expect_count("payload bytes", payload_len, expected_payload)?;
        let expected_total =
            checked_add(CB2VEC_ARTIFACT_HEADER_LEN, payload_len, "artifact bytes")?;
        expect_count("artifact bytes", bytes.len(), expected_total)?;

        let mut source_sha256 = [0u8; 32];
        source_sha256.copy_from_slice(&bytes[56..88]);
        let mut cursor = Cursor::new(&bytes[CB2VEC_ARTIFACT_HEADER_LEN..]);
        let source_weights = CodebookWeights {
            dim,
            fm_rank,
            embeddings: cursor.read_f32_vec(source_embedding_count, "source embeddings")?,
            head: cursor.read_f32_vec(head_count, "source head")?,
            factors: cursor.read_f32_vec(factor_count, "source factors")?,
            bias: cursor.read_f32("source bias")?,
        };
        source_weights.validate().map_err(shape_error)?;

        let quantized = match kind {
            PackedCodebookKind::Flat => {
                let weights = QuantizedCodebookWeights {
                    dim,
                    fm_rank,
                    embedding_scale,
                    head_scale,
                    factor_scale,
                    embeddings: cursor.read_i16_vec(quant_embedding_count, "flat embeddings")?,
                    head: cursor.read_i16_vec(head_count, "quantized head")?,
                    factors: cursor.read_i16_vec(factor_count, "quantized factors")?,
                    bias: source_weights.bias,
                };
                weights.validate().map_err(shape_error)?;
                PackedQuantizedPayload::Flat(weights)
            }
            PackedCodebookKind::Factored => {
                let weights = FactoredQuantizedCodebookWeights::new(
                    dim,
                    fm_rank,
                    embedding_scale,
                    head_scale,
                    factor_scale,
                    cursor.read_u8_vec(token_count, "classes")?,
                    cursor.read_i16_vec(
                        checked_mul(class_count, dim, "class bases")?,
                        "class bases",
                    )?,
                    cursor.read_i8_vec(quant_embedding_count, "residuals")?,
                    cursor.read_i16_vec(head_count, "quantized head")?,
                    cursor.read_i16_vec(factor_count, "quantized factors")?,
                    source_weights.bias,
                )
                .map_err(shape_error)?;
                PackedQuantizedPayload::Factored(weights)
            }
        };
        if !cursor.is_finished() {
            return Err(ArtifactError::LengthMismatch(
                "payload has trailing bytes".to_string(),
            ));
        }
        Ok(Self {
            kind,
            source_weights,
            quantized,
            source_sha256,
            legacy_magic,
        })
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, ArtifactError> {
        let source_shape = self.source_weights.validate().map_err(shape_error)?;
        let (
            kind_value,
            class_count,
            embedding_scale,
            head_scale,
            factor_scale,
            quant_embedding_count,
        ) = match &self.quantized {
            PackedQuantizedPayload::Flat(weights) => {
                same_shape(source_shape, weights.validate().map_err(shape_error)?)?;
                (
                    0u16,
                    0usize,
                    weights.embedding_scale,
                    weights.head_scale,
                    weights.factor_scale,
                    weights.embeddings.len(),
                )
            }
            PackedQuantizedPayload::Factored(weights) => {
                same_shape(source_shape, weights.validate().map_err(shape_error)?)?;
                (
                    1u16,
                    weights.class_count(),
                    weights.embedding_scale(),
                    weights.head_scale(),
                    weights.factor_scale(),
                    source_shape.embedding_len().map_err(shape_error)?,
                )
            }
        };

        let mut payload = Vec::new();
        append_f32(&mut payload, &self.source_weights.embeddings);
        append_f32(&mut payload, &self.source_weights.head);
        append_f32(&mut payload, &self.source_weights.factors);
        payload.extend_from_slice(&self.source_weights.bias.to_bits().to_le_bytes());
        match &self.quantized {
            PackedQuantizedPayload::Flat(weights) => {
                append_i16(&mut payload, &weights.embeddings);
                append_i16(&mut payload, &weights.head);
                append_i16(&mut payload, &weights.factors);
            }
            PackedQuantizedPayload::Factored(weights) => {
                payload.extend_from_slice(weights.classes());
                append_i16(&mut payload, weights.bases());
                payload.extend(weights.residuals().iter().map(|&value| value as u8));
                append_i16(&mut payload, weights.head());
                append_i16(&mut payload, weights.factors());
            }
        }
        let payload_len = u32_value(payload.len(), "payload length")?;
        let mut bytes = vec![0u8; CB2VEC_ARTIFACT_HEADER_LEN];
        bytes[..8].copy_from_slice(&CB2VEC_ARTIFACT_MAGIC);
        put_u16(&mut bytes, 8, CB2VEC_ARTIFACT_VERSION);
        put_u16(&mut bytes, 10, kind_value);
        put_u16(&mut bytes, 12, u16_value(source_shape.dim(), "dim")?);
        put_u16(
            &mut bytes,
            14,
            u16_value(source_shape.fm_rank(), "fm_rank")?,
        );
        put_u16(
            &mut bytes,
            16,
            u16_value(source_shape.group_count(), "group_count")?,
        );
        put_u16(
            &mut bytes,
            18,
            u16_value(source_shape.token_count(), "token_count")?,
        );
        put_u16(&mut bytes, 20, u16_value(class_count, "class_count")?);
        put_i32(&mut bytes, 24, embedding_scale);
        put_i32(&mut bytes, 28, head_scale);
        put_i32(&mut bytes, 32, factor_scale);
        put_u32(
            &mut bytes,
            36,
            u32_value(self.source_weights.embeddings.len(), "source embeddings")?,
        );
        put_u32(
            &mut bytes,
            40,
            u32_value(self.source_weights.head.len(), "head")?,
        );
        put_u32(
            &mut bytes,
            44,
            u32_value(self.source_weights.factors.len(), "factors")?,
        );
        put_u32(
            &mut bytes,
            48,
            u32_value(quant_embedding_count, "quantized embeddings")?,
        );
        put_u32(&mut bytes, 52, payload_len);
        bytes[56..88].copy_from_slice(&self.source_sha256);
        bytes.extend_from_slice(&payload);
        Ok(bytes)
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
    pub fn used_legacy_magic(&self) -> bool {
        self.legacy_magic
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
            PackedQuantizedPayload::Factored(weights) => Some(weights),
            PackedQuantizedPayload::Flat(_) => None,
        }
    }

    pub fn into_parts(self) -> (CodebookWeights, PackedQuantizedPayload) {
        (self.source_weights, self.quantized)
    }
}

fn same_shape(left: ModelShape, right: ModelShape) -> Result<(), ArtifactError> {
    if left != right {
        return Err(ArtifactError::InvalidShape(format!(
            "source {left:?} does not match quantized {right:?}"
        )));
    }
    Ok(())
}

fn same_bias(source: f32, quantized: f32) -> Result<(), ArtifactError> {
    if source.to_bits() != quantized.to_bits() {
        return Err(ArtifactError::BiasMismatch);
    }
    Ok(())
}

fn ensure_serializable_shape(
    shape: ModelShape,
    class_count: usize,
    kind: PackedCodebookKind,
) -> Result<(), ArtifactError> {
    u16_value(shape.dim(), "dim")?;
    u16_value(shape.fm_rank(), "fm_rank")?;
    u16_value(shape.group_count(), "group_count")?;
    u16_value(shape.token_count(), "token_count")?;
    u16_value(class_count, "class_count")?;

    let embedding_count = shape.embedding_len().map_err(shape_error)?;
    let head_count = shape.feature_len().map_err(shape_error)?;
    let factor_count = shape.factor_len().map_err(shape_error)?;
    u32_value(embedding_count, "source embeddings")?;
    u32_value(head_count, "head")?;
    u32_value(factor_count, "factors")?;

    let source_scalar_count = checked_add(
        checked_add(
            checked_add(embedding_count, head_count, "source scalars")?,
            factor_count,
            "source scalars",
        )?,
        1,
        "source scalars",
    )?;
    let source_bytes = checked_mul(source_scalar_count, 4, "source bytes")?;
    let quantized_bytes = match kind {
        PackedCodebookKind::Flat => checked_mul(
            checked_add(
                checked_add(embedding_count, head_count, "flat quantized scalars")?,
                factor_count,
                "flat quantized scalars",
            )?,
            2,
            "flat quantized bytes",
        )?,
        PackedCodebookKind::Factored => checked_add(
            checked_add(
                checked_add(
                    checked_add(
                        shape.token_count(),
                        checked_mul(
                            checked_mul(class_count, shape.dim(), "class bases")?,
                            2,
                            "class base bytes",
                        )?,
                        "factored payload",
                    )?,
                    embedding_count,
                    "factored payload",
                )?,
                checked_mul(head_count, 2, "quantized head")?,
                "factored payload",
            )?,
            checked_mul(factor_count, 2, "quantized factors")?,
            "factored payload",
        )?,
    };
    let payload_len = checked_add(source_bytes, quantized_bytes, "payload")?;
    u32_value(payload_len, "payload length")?;
    Ok(())
}

fn shape_error(error: crate::ModelError) -> ArtifactError {
    ArtifactError::InvalidShape(error.to_string())
}

fn expect_count(field: &str, actual: usize, expected: usize) -> Result<(), ArtifactError> {
    if actual != expected {
        return Err(ArtifactError::LengthMismatch(format!(
            "{field}: got {actual}, expected {expected}"
        )));
    }
    Ok(())
}

fn checked_mul(left: usize, right: usize, field: &str) -> Result<usize, ArtifactError> {
    left.checked_mul(right)
        .ok_or_else(|| ArtifactError::LengthMismatch(format!("{field} overflow")))
}

fn checked_add(left: usize, right: usize, field: &str) -> Result<usize, ArtifactError> {
    left.checked_add(right)
        .ok_or_else(|| ArtifactError::LengthMismatch(format!("{field} overflow")))
}

fn u16_value(value: usize, field: &str) -> Result<u16, ArtifactError> {
    u16::try_from(value).map_err(|_| ArtifactError::ValueOutOfRange(format!("{field}={value}")))
}

fn u32_value(value: usize, field: &str) -> Result<u32, ArtifactError> {
    u32::try_from(value).map_err(|_| ArtifactError::ValueOutOfRange(format!("{field}={value}")))
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, ArtifactError> {
    let slice = bytes
        .get(offset..offset + 2)
        .ok_or(ArtifactError::Truncated {
            actual: bytes.len(),
            minimum: offset + 2,
        })?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_i32(bytes: &[u8], offset: usize) -> Result<i32, ArtifactError> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or(ArtifactError::Truncated {
            actual: bytes.len(),
            minimum: offset + 4,
        })?;
    Ok(i32::from_le_bytes(
        slice.try_into().expect("four-byte slice"),
    ))
}

fn read_u32_usize(bytes: &[u8], offset: usize) -> Result<usize, ArtifactError> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or(ArtifactError::Truncated {
            actual: bytes.len(),
            minimum: offset + 4,
        })?;
    Ok(u32::from_le_bytes(slice.try_into().expect("four-byte slice")) as usize)
}

fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_i32(bytes: &mut [u8], offset: usize, value: i32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn append_f32(bytes: &mut Vec<u8>, values: &[f32]) {
    for &value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
}

fn append_i16(bytes: &mut Vec<u8>, values: &[i16]) {
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: usize, field: &str) -> Result<&'a [u8], ArtifactError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| ArtifactError::LengthMismatch(format!("{field} overflow")))?;
        let slice = self
            .bytes
            .get(self.offset..end)
            .ok_or(ArtifactError::Truncated {
                actual: self.bytes.len(),
                minimum: end,
            })?;
        self.offset = end;
        Ok(slice)
    }

    fn read_f32(&mut self, field: &str) -> Result<f32, ArtifactError> {
        let bytes = self.take(4, field)?;
        let value = f32::from_bits(u32::from_le_bytes(
            bytes.try_into().expect("four-byte slice"),
        ));
        if !value.is_finite() {
            return Err(ArtifactError::NonFinite(field.to_string()));
        }
        Ok(value)
    }

    fn read_f32_vec(&mut self, count: usize, field: &str) -> Result<Vec<f32>, ArtifactError> {
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.read_f32(field)?);
        }
        Ok(values)
    }

    fn read_i16_vec(&mut self, count: usize, field: &str) -> Result<Vec<i16>, ArtifactError> {
        let bytes = self.take(checked_mul(count, 2, field)?, field)?;
        Ok(bytes
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect())
    }

    fn read_u8_vec(&mut self, count: usize, field: &str) -> Result<Vec<u8>, ArtifactError> {
        Ok(self.take(count, field)?.to_vec())
    }

    fn read_i8_vec(&mut self, count: usize, field: &str) -> Result<Vec<i8>, ArtifactError> {
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

    fn exact_factored(source: &CodebookWeights, bias: f32) -> FactoredQuantizedCodebookWeights {
        let flat = source.quantize_i16_s32_s64();
        let token_count = flat.token_count();
        FactoredQuantizedCodebookWeights::new(
            flat.dim,
            flat.fm_rank,
            flat.embedding_scale,
            flat.head_scale,
            flat.factor_scale,
            (0..token_count).map(|token| token as u8).collect(),
            flat.embeddings,
            vec![0; token_count * flat.dim],
            flat.head,
            flat.factors,
            bias,
        )
        .unwrap()
    }

    #[test]
    fn flat_artifact_round_trip_is_canonical() {
        let source = CodebookWeights::deterministic(13, 3, 4, 2);
        let quantized = source.quantize_i16_s32_s64();
        let artifact = PackedCodebookArtifact::new_flat(source, quantized, [0x5a; 32]).unwrap();
        let bytes = artifact.to_bytes().unwrap();
        assert_eq!(&bytes[..8], &CB2VEC_ARTIFACT_MAGIC);
        let parsed = PackedCodebookArtifact::parse(&bytes).unwrap();
        assert!(!parsed.used_legacy_magic());
        assert_eq!(parsed.to_bytes().unwrap(), bytes);
        assert_eq!(parsed.source_sha256(), &[0x5a; 32]);
    }

    #[test]
    fn legacy_magic_is_read_but_rewritten_canonically() {
        let source = CodebookWeights::deterministic(7, 2, 3, 1);
        let quantized = source.quantize_i16_s32_s64();
        let mut bytes = PackedCodebookArtifact::new_flat(source, quantized, [0; 32])
            .unwrap()
            .to_bytes()
            .unwrap();
        bytes[..8].copy_from_slice(&LEGACY_NORU_CBF_MAGIC);
        let parsed = PackedCodebookArtifact::parse(&bytes).unwrap();
        assert!(parsed.used_legacy_magic());
        assert_eq!(&parsed.to_bytes().unwrap()[..8], &CB2VEC_ARTIFACT_MAGIC);
    }

    #[test]
    fn factored_artifact_round_trip_is_canonical() {
        let source = CodebookWeights::deterministic(11, 3, 4, 2);
        let quantized = exact_factored(&source, source.bias);
        let artifact = PackedCodebookArtifact::new_factored(source, quantized, [0xa5; 32]).unwrap();
        let bytes = artifact.to_bytes().unwrap();
        let parsed = PackedCodebookArtifact::parse(&bytes).unwrap();
        assert_eq!(parsed.kind(), PackedCodebookKind::Factored);
        assert_eq!(parsed.to_bytes().unwrap(), bytes);
    }

    #[test]
    fn constructors_reject_bias_mismatch() {
        let source = CodebookWeights::deterministic(7, 2, 3, 1);
        let mut flat = source.quantize_i16_s32_s64();
        flat.bias = source.bias + 1.0;
        assert!(matches!(
            PackedCodebookArtifact::new_flat(source.clone(), flat, [0; 32]),
            Err(ArtifactError::BiasMismatch)
        ));

        let factored = exact_factored(&source, source.bias + 1.0);
        assert!(matches!(
            PackedCodebookArtifact::new_factored(source, factored, [0; 32]),
            Err(ArtifactError::BiasMismatch)
        ));
    }

    #[test]
    fn artifact_token_count_respects_u16_header() {
        let source = CodebookWeights {
            dim: 1,
            fm_rank: 0,
            embeddings: vec![0.0; u16::MAX as usize],
            head: vec![0.0],
            factors: Vec::new(),
            bias: 0.0,
        };
        let flat = source.quantize_i16_s32_s64();
        assert!(PackedCodebookArtifact::new_flat(source, flat, [0; 32]).is_ok());

        let source = CodebookWeights {
            dim: 1,
            fm_rank: 0,
            embeddings: vec![0.0; u16::MAX as usize + 1],
            head: vec![0.0],
            factors: Vec::new(),
            bias: 0.0,
        };
        let flat = source.quantize_i16_s32_s64();
        assert!(matches!(
            PackedCodebookArtifact::new_flat(source, flat, [0; 32]),
            Err(ArtifactError::ValueOutOfRange(message))
                if message.starts_with("token_count=")
        ));
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let source = CodebookWeights::deterministic(5, 2, 3, 1);
        let quantized = source.quantize_i16_s32_s64();
        let mut bytes = PackedCodebookArtifact::new_flat(source, quantized, [0; 32])
            .unwrap()
            .to_bytes()
            .unwrap();
        bytes.push(0);
        assert!(matches!(
            PackedCodebookArtifact::parse(&bytes),
            Err(ArtifactError::LengthMismatch(_))
        ));
    }
}
