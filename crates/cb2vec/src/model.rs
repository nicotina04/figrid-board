use std::error::Error;
use std::fmt;

#[cfg(feature = "json")]
use serde_json::Value;

/// Quantization scale used by the first deployed CB2Vec evaluator.
pub const QUANT_EMBED_SCALE: i32 = 32;
/// Linear-head quantization scale used by the first deployed CB2Vec evaluator.
pub const QUANT_HEAD_SCALE: i32 = 64;
/// Factorization-machine quantization scale used by the first deployed CB2Vec evaluator.
pub const QUANT_FACTOR_SCALE: i32 = 64;

/// Validated logical dimensions of a codebook model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelShape {
    token_count: usize,
    group_count: usize,
    dim: usize,
    fm_rank: usize,
}

impl ModelShape {
    pub fn new(
        token_count: usize,
        group_count: usize,
        dim: usize,
        fm_rank: usize,
    ) -> Result<Self, ModelError> {
        if token_count == 0 {
            return Err(ModelError::ZeroDimension("token_count"));
        }
        if group_count == 0 {
            return Err(ModelError::ZeroDimension("group_count"));
        }
        if dim == 0 {
            return Err(ModelError::ZeroDimension("dim"));
        }
        if token_count > u16::MAX as usize + 1 {
            return Err(ModelError::DimensionTooLarge {
                field: "token_count",
                actual: token_count,
                maximum: u16::MAX as usize + 1,
            });
        }
        Ok(Self {
            token_count,
            group_count,
            dim,
            fm_rank,
        })
    }

    #[inline]
    pub fn token_count(self) -> usize {
        self.token_count
    }

    #[inline]
    pub fn group_count(self) -> usize {
        self.group_count
    }

    #[inline]
    pub fn dim(self) -> usize {
        self.dim
    }

    #[inline]
    pub fn fm_rank(self) -> usize {
        self.fm_rank
    }

    pub fn embedding_len(self) -> Result<usize, ModelError> {
        checked_mul(self.token_count, self.dim, "embedding length")
    }

    pub fn feature_len(self) -> Result<usize, ModelError> {
        checked_mul(self.group_count, self.dim, "feature length")
    }

    pub fn factor_len(self) -> Result<usize, ModelError> {
        checked_mul(self.feature_len()?, self.fm_rank, "factor length")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ModelError {
    ZeroDimension(&'static str),
    DimensionTooLarge {
        field: &'static str,
        actual: usize,
        maximum: usize,
    },
    ArithmeticOverflow(&'static str),
    LengthMismatch {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    NonPositiveScale(&'static str),
    NonFinite(&'static str),
    TokenOutOfRange {
        token: u16,
        token_count: usize,
    },
    FeatureLength {
        actual: usize,
        expected: usize,
    },
    InvalidJson(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension(field) => write!(f, "{field} must be non-zero"),
            Self::DimensionTooLarge {
                field,
                actual,
                maximum,
            } => write!(f, "{field} is {actual}, but the maximum is {maximum}"),
            Self::ArithmeticOverflow(field) => write!(f, "{field} overflow"),
            Self::LengthMismatch {
                field,
                actual,
                expected,
            } => write!(
                f,
                "{field} length mismatch: got {actual}, expected {expected}"
            ),
            Self::NonPositiveScale(field) => write!(f, "{field} must be positive"),
            Self::NonFinite(field) => write!(f, "{field} contains a non-finite value"),
            Self::TokenOutOfRange { token, token_count } => {
                write!(f, "token {token} is outside codebook size {token_count}")
            }
            Self::FeatureLength { actual, expected } => write!(
                f,
                "feature length mismatch: got {actual}, expected {expected}"
            ),
            Self::InvalidJson(message) => write!(f, "invalid codebook JSON: {message}"),
        }
    }
}

impl Error for ModelError {}

/// Floating-point codebook, grouped linear head, and optional FM factors.
///
/// Fields remain public to make training and conversion tools straightforward.
/// Call [`CodebookWeights::validate`] before using values from an untrusted
/// source.
#[derive(Clone, Debug)]
pub struct CodebookWeights {
    pub dim: usize,
    pub fm_rank: usize,
    pub embeddings: Vec<f32>,
    pub head: Vec<f32>,
    pub factors: Vec<f32>,
    pub bias: f32,
}

impl CodebookWeights {
    pub fn new(
        shape: ModelShape,
        embeddings: Vec<f32>,
        head: Vec<f32>,
        factors: Vec<f32>,
        bias: f32,
    ) -> Result<Self, ModelError> {
        let weights = Self {
            dim: shape.dim,
            fm_rank: shape.fm_rank,
            embeddings,
            head,
            factors,
            bias,
        };
        let actual = weights.validate()?;
        if actual != shape {
            return Err(ModelError::LengthMismatch {
                field: "model shape",
                actual: actual.feature_len()?,
                expected: shape.feature_len()?,
            });
        }
        Ok(weights)
    }

    pub fn deterministic(
        token_count: usize,
        group_count: usize,
        dim: usize,
        fm_rank: usize,
    ) -> Self {
        let shape = ModelShape::new(token_count, group_count, dim, fm_rank)
            .expect("deterministic model shape must be valid");
        let mut state = 0xC0DE_B00C_F00D_0542u64;
        Self {
            dim,
            fm_rank,
            embeddings: deterministic_vec(
                &mut state,
                shape.embedding_len().expect("embedding length"),
                0.02,
            ),
            head: deterministic_vec(
                &mut state,
                shape.feature_len().expect("feature length"),
                0.02,
            ),
            factors: deterministic_vec(
                &mut state,
                shape.factor_len().expect("factor length"),
                0.02,
            ),
            bias: 0.01,
        }
    }

    #[cfg(feature = "json")]
    pub fn from_json_bytes(data: &[u8]) -> Result<Self, ModelError> {
        let root: Value = serde_json::from_slice(data)
            .map_err(|error| ModelError::InvalidJson(error.to_string()))?;
        Self::from_json_value(&root)
    }

    #[cfg(feature = "json")]
    pub fn from_json_value(root: &Value) -> Result<Self, ModelError> {
        let format = json_str(root, "format")?;
        if format != "cb2vec-model-v1"
            && format != "noru-relation-fusion-eval-v1"
            && format != "noru-pattern4-codebook-eval-v1"
        {
            return Err(ModelError::InvalidJson(format!(
                "unsupported format {format:?}"
            )));
        }

        if let Some(model) = root.get("model").and_then(Value::as_str) {
            if model != "codebook-region-fm"
                && model != "region-codebook-fm"
                && model != "codebook-group-fm"
            {
                return Err(ModelError::InvalidJson(format!(
                    "unsupported model {model:?}"
                )));
            }
        }

        let metadata = root.get("metadata");
        let dim = metadata
            .and_then(|value| json_usize_opt(value, "embedding_dim"))
            .or_else(|| json_usize_opt(root, "embedding_dim"))
            .ok_or_else(|| ModelError::InvalidJson("missing embedding_dim".to_string()))?;
        let fm_rank = metadata
            .and_then(|value| json_usize_opt(value, "fm_rank"))
            .or_else(|| json_usize_opt(root, "fm_rank"))
            .unwrap_or(0);
        let weights = root
            .get("weights")
            .ok_or_else(|| ModelError::InvalidJson("missing weights object".to_string()))?;
        let result = Self {
            dim,
            fm_rank,
            embeddings: json_f32_array(weights, "embeddings")?,
            head: json_f32_array(weights, "head")?,
            factors: json_f32_array(weights, "factors")?,
            bias: json_f32_opt(weights, "bias")
                .ok_or_else(|| ModelError::InvalidJson("missing weights.bias".to_string()))?,
        };
        result.validate()?;
        Ok(result)
    }

    pub fn validate(&self) -> Result<ModelShape, ModelError> {
        if self.dim == 0 {
            return Err(ModelError::ZeroDimension("dim"));
        }
        if self.embeddings.len() % self.dim != 0 {
            return Err(ModelError::LengthMismatch {
                field: "embeddings",
                actual: self.embeddings.len(),
                expected: (self.embeddings.len() / self.dim) * self.dim,
            });
        }
        if self.head.len() % self.dim != 0 {
            return Err(ModelError::LengthMismatch {
                field: "head",
                actual: self.head.len(),
                expected: (self.head.len() / self.dim) * self.dim,
            });
        }
        let shape = ModelShape::new(
            self.embeddings.len() / self.dim,
            self.head.len() / self.dim,
            self.dim,
            self.fm_rank,
        )?;
        expect_len("embeddings", self.embeddings.len(), shape.embedding_len()?)?;
        expect_len("head", self.head.len(), shape.feature_len()?)?;
        expect_len("factors", self.factors.len(), shape.factor_len()?)?;
        if !self.bias.is_finite() {
            return Err(ModelError::NonFinite("bias"));
        }
        if self.embeddings.iter().any(|value| !value.is_finite()) {
            return Err(ModelError::NonFinite("embeddings"));
        }
        if self.head.iter().any(|value| !value.is_finite()) {
            return Err(ModelError::NonFinite("head"));
        }
        if self.factors.iter().any(|value| !value.is_finite()) {
            return Err(ModelError::NonFinite("factors"));
        }
        Ok(shape)
    }

    #[inline]
    pub fn token_count(&self) -> usize {
        self.embeddings.len() / self.dim.max(1)
    }

    #[inline]
    pub fn group_count(&self) -> usize {
        self.head.len() / self.dim.max(1)
    }

    #[inline]
    pub fn feature_len(&self) -> usize {
        self.head.len()
    }

    pub fn quantize_i16(
        &self,
        embedding_scale: i32,
        head_scale: i32,
        factor_scale: i32,
    ) -> Result<QuantizedCodebookWeights, ModelError> {
        quantize_i16(self, embedding_scale, head_scale, factor_scale)
    }

    pub fn quantize_i16_s32_s64(&self) -> QuantizedCodebookWeights {
        self.quantize_i16(QUANT_EMBED_SCALE, QUANT_HEAD_SCALE, QUANT_FACTOR_SCALE)
            .expect("codebook weights must be valid before quantization")
    }
}

/// Static-dispatch access to floating-point codebook weights.
pub trait FloatCodebookAccess {
    fn dim(&self) -> usize;
    fn fm_rank(&self) -> usize;
    fn embeddings(&self) -> &[f32];
    fn head(&self) -> &[f32];
    fn factors(&self) -> &[f32];
    fn bias(&self) -> f32;

    #[inline]
    fn token_count(&self) -> usize {
        self.embeddings().len() / self.dim()
    }

    #[inline]
    fn feature_len(&self) -> usize {
        self.head().len()
    }

    #[inline]
    fn group_count(&self) -> usize {
        self.head().len() / self.dim()
    }

    #[inline]
    fn validate_access(&self) {
        debug_assert!(self.dim() > 0);
        debug_assert_eq!(self.embeddings().len() % self.dim(), 0);
        debug_assert_eq!(self.head().len() % self.dim(), 0);
        debug_assert_eq!(
            self.head().len().checked_mul(self.fm_rank()),
            Some(self.factors().len())
        );
        debug_assert!(self.bias().is_finite());
    }
}

impl FloatCodebookAccess for CodebookWeights {
    #[inline(always)]
    fn dim(&self) -> usize {
        self.dim
    }

    #[inline(always)]
    fn fm_rank(&self) -> usize {
        self.fm_rank
    }

    #[inline(always)]
    fn embeddings(&self) -> &[f32] {
        &self.embeddings
    }

    #[inline(always)]
    fn head(&self) -> &[f32] {
        &self.head
    }

    #[inline(always)]
    fn factors(&self) -> &[f32] {
        &self.factors
    }

    #[inline(always)]
    fn bias(&self) -> f32 {
        self.bias
    }
}

pub fn quantize_i16<W: FloatCodebookAccess>(
    weights: &W,
    embedding_scale: i32,
    head_scale: i32,
    factor_scale: i32,
) -> Result<QuantizedCodebookWeights, ModelError> {
    validate_scale(embedding_scale, "embedding_scale")?;
    validate_scale(head_scale, "head_scale")?;
    validate_scale(factor_scale, "factor_scale")?;
    if weights.dim() == 0 {
        return Err(ModelError::ZeroDimension("dim"));
    }
    if weights.embeddings().len() % weights.dim() != 0 {
        return Err(ModelError::LengthMismatch {
            field: "embeddings",
            actual: weights.embeddings().len(),
            expected: (weights.embeddings().len() / weights.dim()) * weights.dim(),
        });
    }
    if weights.head().len() % weights.dim() != 0 {
        return Err(ModelError::LengthMismatch {
            field: "head",
            actual: weights.head().len(),
            expected: (weights.head().len() / weights.dim()) * weights.dim(),
        });
    }
    let shape = ModelShape::new(
        weights.embeddings().len() / weights.dim(),
        weights.head().len() / weights.dim(),
        weights.dim(),
        weights.fm_rank(),
    )?;
    expect_len("factors", weights.factors().len(), shape.factor_len()?)?;
    if !weights.bias().is_finite()
        || weights.embeddings().iter().any(|value| !value.is_finite())
        || weights.head().iter().any(|value| !value.is_finite())
        || weights.factors().iter().any(|value| !value.is_finite())
    {
        return Err(ModelError::NonFinite("float weights"));
    }
    weights.validate_access();
    Ok(QuantizedCodebookWeights {
        dim: weights.dim(),
        fm_rank: weights.fm_rank(),
        embedding_scale,
        head_scale,
        factor_scale,
        embeddings: quantize_vec_i16(weights.embeddings(), embedding_scale),
        head: quantize_vec_i16(weights.head(), head_scale),
        factors: quantize_vec_i16(weights.factors(), factor_scale),
        bias: weights.bias(),
    })
}

/// Integer deployment representation of a CB2Vec model.
#[derive(Clone, Debug)]
pub struct QuantizedCodebookWeights {
    pub dim: usize,
    pub fm_rank: usize,
    pub embedding_scale: i32,
    pub head_scale: i32,
    pub factor_scale: i32,
    pub embeddings: Vec<i16>,
    pub head: Vec<i16>,
    pub factors: Vec<i16>,
    pub bias: f32,
}

impl QuantizedCodebookWeights {
    pub fn validate(&self) -> Result<ModelShape, ModelError> {
        validate_scale(self.embedding_scale, "embedding_scale")?;
        validate_scale(self.head_scale, "head_scale")?;
        validate_scale(self.factor_scale, "factor_scale")?;
        if self.dim == 0 {
            return Err(ModelError::ZeroDimension("dim"));
        }
        if self.embeddings.len() % self.dim != 0 || self.head.len() % self.dim != 0 {
            return Err(ModelError::LengthMismatch {
                field: "quantized shape",
                actual: self.embeddings.len() + self.head.len(),
                expected: 0,
            });
        }
        let shape = ModelShape::new(
            self.embeddings.len() / self.dim,
            self.head.len() / self.dim,
            self.dim,
            self.fm_rank,
        )?;
        expect_len("embeddings", self.embeddings.len(), shape.embedding_len()?)?;
        expect_len("head", self.head.len(), shape.feature_len()?)?;
        expect_len("factors", self.factors.len(), shape.factor_len()?)?;
        if !self.bias.is_finite() {
            return Err(ModelError::NonFinite("bias"));
        }
        Ok(shape)
    }

    pub fn dequantized(&self) -> CodebookWeights {
        self.validate()
            .expect("quantized codebook weights must be valid before dequantization");
        CodebookWeights {
            dim: self.dim,
            fm_rank: self.fm_rank,
            embeddings: dequantize_vec_i16(&self.embeddings, self.embedding_scale),
            head: dequantize_vec_i16(&self.head, self.head_scale),
            factors: dequantize_vec_i16(&self.factors, self.factor_scale),
            bias: self.bias,
        }
    }

    #[inline]
    pub fn token_count(&self) -> usize {
        self.embeddings.len() / self.dim.max(1)
    }

    #[inline]
    pub fn group_count(&self) -> usize {
        self.head.len() / self.dim.max(1)
    }

    #[inline]
    pub fn feature_len(&self) -> usize {
        self.head.len()
    }
}

/// Static-dispatch access to either flat or computed integer embeddings.
///
/// Implementations must return the same logical row from `embedding` and
/// `add_embedding_to`. The latter methods may be overridden to exploit a
/// representation-specific fast path.
pub trait QuantizedCodebookAccess {
    fn dim(&self) -> usize;
    fn fm_rank(&self) -> usize;
    fn embedding_scale(&self) -> i32;
    fn head_scale(&self) -> i32;
    fn factor_scale(&self) -> i32;
    fn bias(&self) -> f32;
    fn token_count(&self) -> usize;
    fn head(&self) -> &[i16];
    fn factors(&self) -> &[i16];
    fn embedding(&self, token: u16, component: usize) -> i16;

    #[inline]
    fn pattern_count(&self) -> usize {
        self.token_count()
    }

    #[inline(always)]
    fn embedding_delta(&self, old_token: u16, new_token: u16, component: usize) -> i32 {
        i32::from(self.embedding(new_token, component))
            - i32::from(self.embedding(old_token, component))
    }

    #[inline(always)]
    fn add_embedding_to(&self, token: u16, out: &mut [i32]) {
        debug_assert_eq!(out.len(), self.dim());
        for (component, value) in out.iter_mut().enumerate() {
            *value += i32::from(self.embedding(token, component));
        }
    }

    #[inline(always)]
    fn add_embedding_delta_to(&self, old_token: u16, new_token: u16, out: &mut [i32]) {
        debug_assert_eq!(out.len(), self.dim());
        for (component, value) in out.iter_mut().enumerate() {
            *value += self.embedding_delta(old_token, new_token, component);
        }
    }

    #[inline]
    fn feature_len(&self) -> usize {
        self.head().len()
    }

    #[inline]
    fn group_count(&self) -> usize {
        self.feature_len() / self.dim()
    }

    #[inline]
    fn validate_access(&self) {
        debug_assert!(self.dim() > 0);
        debug_assert!(self.token_count() > 0);
        debug_assert!(self.embedding_scale() > 0);
        debug_assert!(self.head_scale() > 0);
        debug_assert!(self.factor_scale() > 0);
        debug_assert_eq!(self.head().len() % self.dim(), 0);
        debug_assert_eq!(
            self.feature_len().checked_mul(self.fm_rank()),
            Some(self.factors().len())
        );
    }
}

impl QuantizedCodebookAccess for QuantizedCodebookWeights {
    #[inline(always)]
    fn dim(&self) -> usize {
        self.dim
    }

    #[inline(always)]
    fn fm_rank(&self) -> usize {
        self.fm_rank
    }

    #[inline(always)]
    fn embedding_scale(&self) -> i32 {
        self.embedding_scale
    }

    #[inline(always)]
    fn head_scale(&self) -> i32 {
        self.head_scale
    }

    #[inline(always)]
    fn factor_scale(&self) -> i32 {
        self.factor_scale
    }

    #[inline(always)]
    fn bias(&self) -> f32 {
        self.bias
    }

    #[inline(always)]
    fn token_count(&self) -> usize {
        self.token_count()
    }

    #[inline(always)]
    fn head(&self) -> &[i16] {
        &self.head
    }

    #[inline(always)]
    fn factors(&self) -> &[i16] {
        &self.factors
    }

    #[inline(always)]
    fn embedding(&self, token: u16, component: usize) -> i16 {
        self.embeddings[token as usize * self.dim + component]
    }

    #[inline(always)]
    fn add_embedding_to(&self, token: u16, out: &mut [i32]) {
        let start = token as usize * self.dim;
        let embedding = &self.embeddings[start..start + self.dim];
        for (value, &component) in out.iter_mut().zip(embedding) {
            *value += i32::from(component);
        }
    }

    #[inline(always)]
    fn add_embedding_delta_to(&self, old_token: u16, new_token: u16, out: &mut [i32]) {
        let old_start = old_token as usize * self.dim;
        let new_start = new_token as usize * self.dim;
        let old = &self.embeddings[old_start..old_start + self.dim];
        let new = &self.embeddings[new_start..new_start + self.dim];
        for ((value, &old), &new) in out.iter_mut().zip(old).zip(new) {
            *value += i32::from(new) - i32::from(old);
        }
    }
}

#[inline]
pub fn add_embedding_to<W: QuantizedCodebookAccess>(
    weights: &W,
    token: u16,
    out: &mut [i32],
) -> Result<(), ModelError> {
    if token as usize >= weights.token_count() {
        return Err(ModelError::TokenOutOfRange {
            token,
            token_count: weights.token_count(),
        });
    }
    if out.len() != weights.dim() {
        return Err(ModelError::FeatureLength {
            actual: out.len(),
            expected: weights.dim(),
        });
    }
    for (component, &value) in out.iter().enumerate() {
        let delta = i32::from(weights.embedding(token, component));
        if value.checked_add(delta).is_none() {
            return Err(ModelError::ArithmeticOverflow("embedding accumulator"));
        }
    }
    weights.add_embedding_to(token, out);
    Ok(())
}

#[inline]
pub fn add_embedding_delta_to<W: QuantizedCodebookAccess>(
    weights: &W,
    old_token: u16,
    new_token: u16,
    out: &mut [i32],
) -> Result<(), ModelError> {
    if old_token as usize >= weights.token_count() {
        return Err(ModelError::TokenOutOfRange {
            token: old_token,
            token_count: weights.token_count(),
        });
    }
    if new_token as usize >= weights.token_count() {
        return Err(ModelError::TokenOutOfRange {
            token: new_token,
            token_count: weights.token_count(),
        });
    }
    if out.len() != weights.dim() {
        return Err(ModelError::FeatureLength {
            actual: out.len(),
            expected: weights.dim(),
        });
    }
    for (component, &value) in out.iter().enumerate() {
        let delta = weights.embedding_delta(old_token, new_token, component);
        if value.checked_add(delta).is_none() {
            return Err(ModelError::ArithmeticOverflow("embedding accumulator"));
        }
    }
    weights.add_embedding_delta_to(old_token, new_token, out);
    Ok(())
}

/// Score already-normalized floating-point grouped features.
#[inline]
pub fn score_f32<W: FloatCodebookAccess>(features: &[f32], weights: &W) -> Result<f32, ModelError> {
    validate_float_score_inputs(features, weights)?;
    let mut logit = weights.bias();
    for (x, w) in features.iter().zip(weights.head()) {
        logit += x * w;
    }
    for rank in 0..weights.fm_rank() {
        let mut sum = 0.0f32;
        let mut square_sum = 0.0f32;
        for (index, &x) in features.iter().enumerate() {
            let vx = weights.factors()[index * weights.fm_rank() + rank] * x;
            sum += vx;
            square_sum += vx * vx;
        }
        logit += 0.5 * (sum * sum - square_sum);
    }
    if !logit.is_finite() {
        return Err(ModelError::NonFinite("score"));
    }
    Ok(logit)
}

/// Score integer grouped sums when every group has the same pooling divisor.
///
/// The computation deliberately uses `f64` intermediates to preserve the
/// deployed scalar kernel's rounding behavior.
#[inline]
pub fn score_quantized_uniform<W: QuantizedCodebookAccess>(
    features: &[i32],
    weights: &W,
    group_divisor: usize,
) -> Result<f32, ModelError> {
    validate_quantized_score_inputs(features, weights, group_divisor)?;
    let feature_denom = weights.embedding_scale() as f64 * group_divisor as f64;
    let mut logit = weights.bias() as f64;

    let head_denom = feature_denom * weights.head_scale() as f64;
    for (&x, &w) in features.iter().zip(weights.head()) {
        logit += (x as f64 * w as f64) / head_denom;
    }

    let factor_denom = feature_denom * weights.factor_scale() as f64;
    for rank in 0..weights.fm_rank() {
        let mut sum = 0.0f64;
        let mut square_sum = 0.0f64;
        for (index, &x) in features.iter().enumerate() {
            let vx = (x as f64 * weights.factors()[index * weights.fm_rank() + rank] as f64)
                / factor_denom;
            sum += vx;
            square_sum += vx * vx;
        }
        logit += 0.5 * (sum * sum - square_sum);
    }
    let logit = logit as f32;
    if !logit.is_finite() {
        return Err(ModelError::NonFinite("score"));
    }
    Ok(logit)
}

fn validate_float_score_inputs<W: FloatCodebookAccess>(
    features: &[f32],
    weights: &W,
) -> Result<(), ModelError> {
    if features.len() != weights.head().len() {
        return Err(ModelError::FeatureLength {
            actual: features.len(),
            expected: weights.head().len(),
        });
    }
    if weights.dim() == 0 {
        return Err(ModelError::ZeroDimension("dim"));
    }
    if weights.embeddings().len() % weights.dim() != 0 {
        return Err(ModelError::LengthMismatch {
            field: "embeddings",
            actual: weights.embeddings().len(),
            expected: (weights.embeddings().len() / weights.dim()) * weights.dim(),
        });
    }
    if weights.head().len() % weights.dim() != 0 {
        return Err(ModelError::LengthMismatch {
            field: "head",
            actual: weights.head().len(),
            expected: (weights.head().len() / weights.dim()) * weights.dim(),
        });
    }
    let shape = ModelShape::new(
        weights.embeddings().len() / weights.dim(),
        weights.head().len() / weights.dim(),
        weights.dim(),
        weights.fm_rank(),
    )?;
    expect_len("factors", weights.factors().len(), shape.factor_len()?)?;
    if !weights.bias().is_finite()
        || features.iter().any(|value| !value.is_finite())
        || weights.head().iter().any(|value| !value.is_finite())
        || weights.factors().iter().any(|value| !value.is_finite())
    {
        return Err(ModelError::NonFinite("score inputs"));
    }
    Ok(())
}

fn validate_quantized_score_inputs<W: QuantizedCodebookAccess>(
    features: &[i32],
    weights: &W,
    group_divisor: usize,
) -> Result<(), ModelError> {
    if features.len() != weights.head().len() {
        return Err(ModelError::FeatureLength {
            actual: features.len(),
            expected: weights.head().len(),
        });
    }
    if group_divisor == 0 {
        return Err(ModelError::ZeroDimension("group_divisor"));
    }
    validate_scale(weights.embedding_scale(), "embedding_scale")?;
    validate_scale(weights.head_scale(), "head_scale")?;
    validate_scale(weights.factor_scale(), "factor_scale")?;
    if weights.dim() == 0 {
        return Err(ModelError::ZeroDimension("dim"));
    }
    if weights.head().len() % weights.dim() != 0 {
        return Err(ModelError::LengthMismatch {
            field: "head",
            actual: weights.head().len(),
            expected: (weights.head().len() / weights.dim()) * weights.dim(),
        });
    }
    let shape = ModelShape::new(
        weights.token_count(),
        weights.head().len() / weights.dim(),
        weights.dim(),
        weights.fm_rank(),
    )?;
    expect_len("factors", weights.factors().len(), shape.factor_len()?)?;
    if !weights.bias().is_finite() {
        return Err(ModelError::NonFinite("bias"));
    }
    Ok(())
}

fn validate_scale(scale: i32, field: &'static str) -> Result<(), ModelError> {
    if scale <= 0 {
        return Err(ModelError::NonPositiveScale(field));
    }
    Ok(())
}

fn expect_len(field: &'static str, actual: usize, expected: usize) -> Result<(), ModelError> {
    if actual != expected {
        return Err(ModelError::LengthMismatch {
            field,
            actual,
            expected,
        });
    }
    Ok(())
}

fn checked_mul(left: usize, right: usize, field: &'static str) -> Result<usize, ModelError> {
    left.checked_mul(right)
        .ok_or(ModelError::ArithmeticOverflow(field))
}

fn deterministic_vec(state: &mut u64, len: usize, scale: f32) -> Vec<f32> {
    (0..len).map(|_| deterministic_f32(state, scale)).collect()
}

fn deterministic_f32(state: &mut u64, scale: f32) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    let unit = ((*state >> 40) as u32) as f32 / ((1u32 << 24) as f32);
    (unit * 2.0 - 1.0) * scale
}

fn quantize_vec_i16(values: &[f32], scale: i32) -> Vec<i16> {
    values
        .iter()
        .map(|&value| {
            (value * scale as f32)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16
        })
        .collect()
}

fn dequantize_vec_i16(values: &[i16], scale: i32) -> Vec<f32> {
    values
        .iter()
        .map(|&value| value as f32 / scale as f32)
        .collect()
}

#[cfg(feature = "json")]
fn json_str<'a>(value: &'a Value, key: &str) -> Result<&'a str, ModelError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| ModelError::InvalidJson(format!("missing {key}")))
}

#[cfg(feature = "json")]
fn json_usize_opt(value: &Value, key: &str) -> Option<usize> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

#[cfg(feature = "json")]
fn json_f32_opt(value: &Value, key: &str) -> Option<f32> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .map(|value| value as f32)
}

#[cfg(feature = "json")]
fn json_f32_array(value: &Value, key: &str) -> Result<Vec<f32>, ModelError> {
    let values = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| ModelError::InvalidJson(format!("missing weights.{key}")))?;
    values
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|value| value as f32)
                .ok_or_else(|| ModelError::InvalidJson(format!("non-float value in weights.{key}")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_is_inferred_without_game_constants() {
        let weights = CodebookWeights::deterministic(17, 3, 8, 2);
        assert_eq!(
            weights.validate().unwrap(),
            ModelShape::new(17, 3, 8, 2).unwrap()
        );
        assert_eq!(weights.feature_len(), 24);
    }

    #[test]
    fn flat_embedding_delta_matches_two_rows() {
        let weights = CodebookWeights::deterministic(7, 2, 4, 1).quantize_i16_s32_s64();
        let mut direct = vec![0i32; 4];
        weights.add_embedding_to(2, &mut direct);
        weights.add_embedding_delta_to(2, 5, &mut direct);
        let mut expected = vec![0i32; 4];
        weights.add_embedding_to(5, &mut expected);
        assert_eq!(direct, expected);
    }

    #[test]
    fn checked_embedding_updates_reject_overflow_without_mutation() {
        let weights = QuantizedCodebookWeights {
            dim: 2,
            fm_rank: 0,
            embedding_scale: 32,
            head_scale: 64,
            factor_scale: 64,
            embeddings: vec![0, 0, 1, 1],
            head: vec![0, 0],
            factors: Vec::new(),
            bias: 0.0,
        };
        weights.validate().unwrap();

        let mut add_out = vec![0, i32::MAX];
        let add_before = add_out.clone();
        assert!(matches!(
            add_embedding_to(&weights, 1, &mut add_out),
            Err(ModelError::ArithmeticOverflow("embedding accumulator"))
        ));
        assert_eq!(add_out, add_before);

        let mut delta_out = vec![0, i32::MAX];
        let delta_before = delta_out.clone();
        assert!(matches!(
            add_embedding_delta_to(&weights, 0, 1, &mut delta_out),
            Err(ModelError::ArithmeticOverflow("embedding accumulator"))
        ));
        assert_eq!(delta_out, delta_before);
    }

    #[test]
    fn f32_and_dequantized_shapes_round_trip() {
        let source = CodebookWeights::deterministic(11, 4, 5, 3);
        let quantized = source.quantize_i16_s32_s64();
        assert_eq!(source.validate().unwrap(), quantized.validate().unwrap());
        assert_eq!(
            source.validate().unwrap(),
            quantized.dequantized().validate().unwrap()
        );
    }

    #[test]
    fn malformed_lengths_fail_closed() {
        let mut weights = CodebookWeights::deterministic(11, 4, 5, 3);
        weights.factors.pop();
        assert!(matches!(
            weights.validate(),
            Err(ModelError::LengthMismatch {
                field: "factors",
                ..
            })
        ));
    }

    #[test]
    fn token_domain_matches_u16_accessors() {
        assert!(ModelShape::new(u16::MAX as usize + 1, 1, 1, 0).is_ok());
        assert!(matches!(
            ModelShape::new(u16::MAX as usize + 2, 1, 1, 0),
            Err(ModelError::DimensionTooLarge {
                field: "token_count",
                ..
            })
        ));
    }

    #[test]
    fn quantization_rejects_factor_length_overflow() {
        let weights = CodebookWeights {
            dim: 1,
            fm_rank: 1usize << (usize::BITS - 1),
            embeddings: vec![0.0],
            head: vec![0.0; 2],
            factors: Vec::new(),
            bias: 0.0,
        };
        assert!(matches!(
            weights.quantize_i16(32, 64, 64),
            Err(ModelError::ArithmeticOverflow("factor length"))
        ));
    }

    #[test]
    fn uniform_quantized_head_is_deterministic() {
        let weights = CodebookWeights::deterministic(9, 3, 4, 2).quantize_i16_s32_s64();
        let features = vec![7; weights.feature_len()];
        let left = score_quantized_uniform(&features, &weights, 5).unwrap();
        let right = score_quantized_uniform(&features, &weights, 5).unwrap();
        assert_eq!(left.to_bits(), right.to_bits());
    }

    #[test]
    fn scoring_rejects_invalid_feature_lengths_and_divisor() {
        let source = CodebookWeights::deterministic(9, 3, 4, 2);
        let quantized = source.quantize_i16_s32_s64();
        let float_features = vec![0.0; source.feature_len()];
        let quantized_features = vec![0; quantized.feature_len()];

        assert!(matches!(
            score_f32(&float_features[..float_features.len() - 1], &source),
            Err(ModelError::FeatureLength { .. })
        ));
        let mut too_long = float_features.clone();
        too_long.push(0.0);
        assert!(matches!(
            score_f32(&too_long, &source),
            Err(ModelError::FeatureLength { .. })
        ));
        assert!(matches!(
            score_quantized_uniform(
                &quantized_features[..quantized_features.len() - 1],
                &quantized,
                5
            ),
            Err(ModelError::FeatureLength { .. })
        ));
        let mut too_long = quantized_features.clone();
        too_long.push(0);
        assert!(matches!(
            score_quantized_uniform(&too_long, &quantized, 5),
            Err(ModelError::FeatureLength { .. })
        ));
        assert!(matches!(
            score_quantized_uniform(&quantized_features, &quantized, 0),
            Err(ModelError::ZeroDimension("group_divisor"))
        ));
    }

    #[test]
    fn quantized_scoring_uses_head_length_as_the_authoritative_shape() {
        struct MisreportedFeatureLength;

        impl QuantizedCodebookAccess for MisreportedFeatureLength {
            fn dim(&self) -> usize {
                1
            }

            fn fm_rank(&self) -> usize {
                1
            }

            fn embedding_scale(&self) -> i32 {
                1
            }

            fn head_scale(&self) -> i32 {
                1
            }

            fn factor_scale(&self) -> i32 {
                1
            }

            fn bias(&self) -> f32 {
                0.0
            }

            fn token_count(&self) -> usize {
                1
            }

            fn head(&self) -> &[i16] {
                &[0]
            }

            fn factors(&self) -> &[i16] {
                &[0]
            }

            fn embedding(&self, _token: u16, _component: usize) -> i16 {
                0
            }

            fn feature_len(&self) -> usize {
                2
            }
        }

        assert!(matches!(
            score_quantized_uniform(&[0, 0], &MisreportedFeatureLength, 1),
            Err(ModelError::FeatureLength {
                actual: 2,
                expected: 1,
            })
        ));
    }

    #[test]
    fn float_quantization_uses_head_length_as_the_authoritative_shape() {
        struct MisreportedFeatureLength;

        impl FloatCodebookAccess for MisreportedFeatureLength {
            fn dim(&self) -> usize {
                1
            }

            fn fm_rank(&self) -> usize {
                1
            }

            fn embeddings(&self) -> &[f32] {
                &[0.0]
            }

            fn head(&self) -> &[f32] {
                &[0.0]
            }

            fn factors(&self) -> &[f32] {
                &[0.0]
            }

            fn bias(&self) -> f32 {
                0.0
            }

            fn feature_len(&self) -> usize {
                2
            }
        }

        let quantized = quantize_i16(&MisreportedFeatureLength, 32, 64, 64).unwrap();
        assert_eq!(quantized.feature_len(), 1);
        assert_eq!(quantized.factors.len(), 1);
    }
}
