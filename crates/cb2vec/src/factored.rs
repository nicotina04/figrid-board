use crate::{ModelError, ModelShape, QuantizedCodebookAccess, QuantizedCodebookWeights};

/// Exact class-base plus `i8` residual storage for an integer embedding table.
///
/// This representation is intended primarily for storage. Consumers may call
/// [`reconstruct_flat`](Self::reconstruct_flat) once at load time when flat
/// rows are faster in their hot loop.
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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
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
    ) -> Result<Self, ModelError> {
        let weights = Self {
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
        Ok(weights)
    }

    pub fn validate(&self) -> Result<ModelShape, ModelError> {
        if self.dim == 0 {
            return Err(ModelError::ZeroDimension("dim"));
        }
        if self.embedding_scale <= 0 {
            return Err(ModelError::NonPositiveScale("embedding_scale"));
        }
        if self.head_scale <= 0 {
            return Err(ModelError::NonPositiveScale("head_scale"));
        }
        if self.factor_scale <= 0 {
            return Err(ModelError::NonPositiveScale("factor_scale"));
        }
        if self.classes.is_empty() {
            return Err(ModelError::ZeroDimension("token_count"));
        }
        if self.bases.is_empty() || self.bases.len() % self.dim != 0 {
            return Err(ModelError::LengthMismatch {
                field: "bases",
                actual: self.bases.len(),
                expected: self.dim,
            });
        }
        if self.head.is_empty() || self.head.len() % self.dim != 0 {
            return Err(ModelError::LengthMismatch {
                field: "head",
                actual: self.head.len(),
                expected: self.dim,
            });
        }
        let shape = ModelShape::new(
            self.classes.len(),
            self.head.len() / self.dim,
            self.dim,
            self.fm_rank,
        )?;
        let class_count = self.class_count();
        if class_count > u8::MAX as usize + 1 {
            return Err(ModelError::DimensionTooLarge {
                field: "class_count",
                actual: class_count,
                maximum: u8::MAX as usize + 1,
            });
        }
        let expected_residuals = shape.embedding_len()?;
        if self.residuals.len() != expected_residuals {
            return Err(ModelError::LengthMismatch {
                field: "residuals",
                actual: self.residuals.len(),
                expected: expected_residuals,
            });
        }
        let expected_factors = shape.factor_len()?;
        if self.factors.len() != expected_factors {
            return Err(ModelError::LengthMismatch {
                field: "factors",
                actual: self.factors.len(),
                expected: expected_factors,
            });
        }
        if !self.bias.is_finite() {
            return Err(ModelError::NonFinite("bias"));
        }

        for (token, &class) in self.classes.iter().enumerate() {
            let class = usize::from(class);
            if class >= class_count {
                return Err(ModelError::LengthMismatch {
                    field: "class id",
                    actual: class,
                    expected: class_count,
                });
            }
            let base_offset = class * self.dim;
            let residual_offset = token * self.dim;
            for component in 0..self.dim {
                let reconstructed = i32::from(self.bases[base_offset + component])
                    + i32::from(self.residuals[residual_offset + component]);
                if i16::try_from(reconstructed).is_err() {
                    return Err(ModelError::LengthMismatch {
                        field: "reconstructed embedding",
                        actual: reconstructed.unsigned_abs() as usize,
                        expected: i16::MAX as usize,
                    });
                }
            }
        }
        Ok(shape)
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
    pub fn bias(&self) -> f32 {
        self.bias
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
    pub fn group_count(&self) -> usize {
        self.head.len() / self.dim
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

    pub fn reconstruct_flat(&self) -> QuantizedCodebookWeights {
        self.validate()
            .expect("factored codebook must be valid before reconstruction");
        let mut embeddings = Vec::with_capacity(self.token_count() * self.dim);
        for token in 0..self.token_count() {
            let class = usize::from(self.classes[token]);
            for component in 0..self.dim {
                let value = i32::from(self.bases[class * self.dim + component])
                    + i32::from(self.residuals[token * self.dim + component]);
                embeddings.push(value as i16);
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
}

impl QuantizedCodebookAccess for FactoredQuantizedCodebookWeights {
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
        self.classes.len()
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
        let token = usize::from(token);
        let class = usize::from(self.classes[token]);
        let base = self.bases[class * self.dim + component];
        let residual = self.residuals[token * self.dim + component];
        (i32::from(base) + i32::from(residual)) as i16
    }

    #[inline(always)]
    fn embedding_delta(&self, old_token: u16, new_token: u16, component: usize) -> i32 {
        let old_token = usize::from(old_token);
        let new_token = usize::from(new_token);
        let old_class = usize::from(self.classes[old_token]);
        let new_class = usize::from(self.classes[new_token]);
        let residual_delta = i32::from(self.residuals[new_token * self.dim + component])
            - i32::from(self.residuals[old_token * self.dim + component]);
        if old_class == new_class {
            residual_delta
        } else {
            residual_delta + i32::from(self.bases[new_class * self.dim + component])
                - i32::from(self.bases[old_class * self.dim + component])
        }
    }

    #[inline(always)]
    fn add_embedding_to(&self, token: u16, out: &mut [i32]) {
        let token = usize::from(token);
        let class = usize::from(self.classes[token]);
        let base = &self.bases[class * self.dim..(class + 1) * self.dim];
        let residual = &self.residuals[token * self.dim..(token + 1) * self.dim];
        for ((value, &base), &residual) in out.iter_mut().zip(base).zip(residual) {
            *value += i32::from(base) + i32::from(residual);
        }
    }

    #[inline(always)]
    fn add_embedding_delta_to(&self, old_token: u16, new_token: u16, out: &mut [i32]) {
        let old_token = usize::from(old_token);
        let new_token = usize::from(new_token);
        let old_class = usize::from(self.classes[old_token]);
        let new_class = usize::from(self.classes[new_token]);
        let old_residual = &self.residuals[old_token * self.dim..(old_token + 1) * self.dim];
        let new_residual = &self.residuals[new_token * self.dim..(new_token + 1) * self.dim];
        if old_class == new_class {
            for ((value, &old), &new) in out.iter_mut().zip(old_residual).zip(new_residual) {
                *value += i32::from(new) - i32::from(old);
            }
        } else {
            let old_base = &self.bases[old_class * self.dim..(old_class + 1) * self.dim];
            let new_base = &self.bases[new_class * self.dim..(new_class + 1) * self.dim];
            for ((((value, &old_residual), &new_residual), &old_base), &new_base) in out
                .iter_mut()
                .zip(old_residual)
                .zip(new_residual)
                .zip(old_base)
                .zip(new_base)
            {
                *value += i32::from(new_residual) - i32::from(old_residual) + i32::from(new_base)
                    - i32::from(old_base);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> FactoredQuantizedCodebookWeights {
        FactoredQuantizedCodebookWeights::new(
            2,
            1,
            32,
            64,
            64,
            vec![0, 1, 0],
            vec![10, -10, 20, -20],
            vec![1, 2, -1, -2, 3, 4],
            vec![5, 6, 7, 8],
            vec![1, 2, 3, 4],
            0.25,
        )
        .unwrap()
    }

    #[test]
    fn reconstruction_and_direct_access_are_identical() {
        let factored = fixture();
        let flat = factored.reconstruct_flat();
        for token in 0..factored.token_count() as u16 {
            for component in 0..factored.dim() {
                assert_eq!(
                    factored.embedding(token, component),
                    flat.embedding(token, component)
                );
            }
        }
    }

    #[test]
    fn same_class_delta_cancels_base() {
        let factored = fixture();
        let mut delta = vec![0; factored.dim()];
        factored.add_embedding_delta_to(0, 2, &mut delta);
        assert_eq!(delta, vec![2, 2]);
    }
}
