// Exact signed-INT8 embedding representation of the established INT16 evaluator.
// The head/factors and arithmetic are unchanged; channel offsets preserve every value.
/// Exact storage for the dim-32, rank-8, 199827-row S32/S64/S64 evaluator.
/// The channel offsets reconstruct the established INT16 embeddings losslessly.
/// Construct through `from_quantized` or `from_bytes` to validate this layout.
pub struct ExactI8Codebook {
    pub table: Vec<i8>,
    pub offsets: Vec<i16>,
    pub scorer: QuantizedCodebookWeights,
}

impl ExactI8Codebook {
    pub const MAGIC: &'static [u8; 8] = b"NGCB8E1\0";
    pub fn from_quantized(q: &QuantizedCodebookWeights) -> Result<Self, String> {
        if (
            q.dim,
            q.fm_rank,
            q.embeddings.len(),
            q.head.len(),
            q.factors.len(),
        ) != (32, 8, 199827 * 32, 9 * 32, 9 * 32 * 8)
        {
            return Err("unsupported exact-i8 shape".into());
        }
        if !q.bias.is_finite() || (q.embedding_scale, q.head_scale, q.factor_scale) != (32, 64, 64)
        {
            return Err("invalid exact-i8 scales/bias".into());
        }
        let mut lo = vec![i16::MAX; q.dim];
        let mut hi = vec![i16::MIN; q.dim];
        for (i, &v) in q.embeddings.iter().enumerate() {
            let d = i % q.dim;
            lo[d] = lo[d].min(v);
            hi[d] = hi[d].max(v);
        }
        let mut offsets = Vec::new();
        for d in 0..q.dim {
            if i32::from(hi[d]) - i32::from(lo[d]) > 255 {
                return Err(format!("channel {d} does not fit INT8 losslessly"));
            }
            offsets.push(i16::try_from(i32::from(lo[d]) + 128).map_err(|_| "offset overflow")?);
        }
        let mut table = Vec::with_capacity(q.embeddings.len());
        for (i, &v) in q.embeddings.iter().enumerate() {
            table.push(
                i8::try_from(i32::from(v) - i32::from(offsets[i % q.dim]))
                    .map_err(|_| "INT8 overflow")?,
            );
        }
        Ok(Self {
            table,
            offsets,
            scorer: QuantizedCodebookWeights {
                dim: q.dim,
                fm_rank: q.fm_rank,
                embedding_scale: q.embedding_scale,
                head_scale: q.head_scale,
                factor_scale: q.factor_scale,
                embeddings: vec![0; q.dim],
                head: q.head.clone(),
                factors: q.factors.clone(),
                bias: q.bias,
            },
        })
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(
            36 + self.offsets.len() * 2
                + self.table.len()
                + (self.scorer.head.len() + self.scorer.factors.len()) * 2,
        );
        b.extend_from_slice(Self::MAGIC);
        for v in [
            self.scorer.dim as u32,
            self.scorer.fm_rank as u32,
            (self.table.len() / self.scorer.dim) as u32,
        ] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(&self.scorer.bias.to_le_bytes());
        for v in [
            self.scorer.embedding_scale,
            self.scorer.head_scale,
            self.scorer.factor_scale,
        ] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.offsets {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend(self.table.iter().map(|&v| v as u8));
        for vs in [&self.scorer.head, &self.scorer.factors] {
            for &v in vs {
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        b
    }
    pub fn from_bytes(b: &[u8]) -> Result<Self, String> {
        if b.len() < 36 || &b[..8] != Self::MAGIC {
            return Err("invalid exact-i8 header".into());
        }
        let u = |o| u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize;
        let (dim, rank, n) = (u(8), u(12), u(16));
        if (dim, rank, n) != (32, 8, 199827) {
            return Err("unsupported exact-i8 shape".into());
        }
        let bias = f32::from_le_bytes(b[20..24].try_into().unwrap());
        let scale = |o| i32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        if !bias.is_finite() || (scale(24), scale(28), scale(32)) != (32, 64, 64) {
            return Err("invalid exact-i8 scales/bias".into());
        }
        let expected = 36 + 2 * dim + n * dim + 2 * (9 * dim + 9 * dim * rank);
        if b.len() != expected {
            return Err("invalid exact-i8 byte count".into());
        }
        let shorts = |off: usize, len: usize| {
            b[off..off + len * 2]
                .chunks_exact(2)
                .map(|v| i16::from_le_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let offsets = shorts(36, dim);
        let off = 36 + 2 * dim;
        let table = b[off..off + n * dim]
            .iter()
            .map(|&v| v as i8)
            .collect::<Vec<_>>();
        for (i, &v) in table.iter().enumerate() {
            if i16::try_from(i32::from(v) + i32::from(offsets[i % dim])).is_err() {
                return Err("reconstructed embedding overflow".into());
            }
        }
        let head = shorts(off + n * dim, 9 * dim);
        let factors = shorts(off + n * dim + 18 * dim, 9 * dim * rank);
        Ok(Self {
            table,
            offsets,
            scorer: QuantizedCodebookWeights {
                dim,
                fm_rank: rank,
                embedding_scale: 32,
                head_scale: 64,
                factor_scale: 64,
                embeddings: vec![0; dim],
                head,
                factors,
                bias,
            },
        })
    }
    pub fn exact_reconstruction(&self, q: &QuantizedCodebookWeights) -> bool {
        self.table.len() == q.embeddings.len()
            && self.scorer.dim == q.dim
            && self.scorer.fm_rank == q.fm_rank
            && self.scorer.head == q.head
            && self.scorer.factors == q.factors
            && self.scorer.bias.to_bits() == q.bias.to_bits()
            && (
                self.scorer.embedding_scale,
                self.scorer.head_scale,
                self.scorer.factor_scale,
            ) == (q.embedding_scale, q.head_scale, q.factor_scale)
            && self.table.iter().enumerate().all(|(i, &v)| {
                i32::from(v) + i32::from(self.offsets[i % q.dim]) == i32::from(q.embeddings[i])
            })
    }
}

#[cfg(test)]
mod exact_i8_tests {
    use super::*;
    fn fixture() -> QuantizedCodebookWeights {
        QuantizedCodebookWeights {
            dim: 32,
            fm_rank: 8,
            embedding_scale: 32,
            head_scale: 64,
            factor_scale: 64,
            embeddings: (0..199827 * 32)
                .map(|i| ((i / 32) % 256) as i16 - 128 + (i % 32) as i16)
                .collect(),
            head: vec![3; 9 * 32],
            factors: vec![-2; 9 * 32 * 8],
            bias: 0.25,
        }
    }
    #[test]
    fn exact_i8_roundtrip_gather_delta() {
        let q = fixture();
        let p = ExactI8Codebook::from_quantized(&q).unwrap();
        assert!(p.exact_reconstruction(&q));
        let bytes = p.to_bytes();
        let p = ExactI8Codebook::from_bytes(&bytes).unwrap();
        assert_eq!(bytes, p.to_bytes());
        assert!(p.exact_reconstruction(&q));
        for (old, new) in [(0, 255), (255, 256), (199826, 17)] {
            let mut out = vec![0; 32];
            p.add_embedding_to(old, &mut out);
            assert_eq!(
                out,
                q.embeddings[old as usize * 32..old as usize * 32 + 32]
                    .iter()
                    .map(|&x| i32::from(x))
                    .collect::<Vec<_>>()
            );
            p.add_embedding_delta_to(old, new, &mut out);
            assert_eq!(
                out,
                q.embeddings[new as usize * 32..new as usize * 32 + 32]
                    .iter()
                    .map(|&x| i32::from(x))
                    .collect::<Vec<_>>()
            );
        }
    }
    #[test]
    fn exact_i8_rejects_invalid_constructor_and_file() {
        let mut q = fixture();
        q.dim = 0;
        assert!(ExactI8Codebook::from_quantized(&q).is_err());
        q.dim = 32;
        q.head.pop();
        assert!(ExactI8Codebook::from_quantized(&q).is_err());
        q.head.push(3);
        q.embedding_scale = 31;
        assert!(ExactI8Codebook::from_quantized(&q).is_err());
        q.embedding_scale = 32;
        q.bias = f32::NAN;
        assert!(ExactI8Codebook::from_quantized(&q).is_err());
        q.bias = 0.25;
        let bytes = ExactI8Codebook::from_quantized(&q).unwrap().to_bytes();
        for len in [0, 7, 35, 100, bytes.len() - 1] {
            assert!(ExactI8Codebook::from_bytes(&bytes[..len]).is_err());
        }
        for (offset, value) in [(0, 0), (8, 0), (12, 0), (16, 0), (24, 0), (28, 0), (32, 0)] {
            let mut bad = bytes.clone();
            bad[offset] = value;
            assert!(ExactI8Codebook::from_bytes(&bad).is_err());
        }
        let mut bad = bytes.clone();
        bad.extend_from_slice(&[0]);
        assert!(ExactI8Codebook::from_bytes(&bad).is_err());
        let mut bad = bytes.clone();
        bad[20..24].copy_from_slice(&f32::INFINITY.to_le_bytes());
        assert!(ExactI8Codebook::from_bytes(&bad).is_err());
        let mut bad = bytes.clone();
        bad[36..38].copy_from_slice(&i16::MAX.to_le_bytes());
        bad[100] = 127;
        assert!(ExactI8Codebook::from_bytes(&bad).is_err());
        q.embeddings[0] = -2000;
        assert!(ExactI8Codebook::from_quantized(&q).is_err());
    }
}

impl QuantizedCodebookAccess for ExactI8Codebook {
    #[inline(always)]
    fn dim(&self) -> usize {
        self.scorer.dim
    }
    #[inline(always)]
    fn embedding_scale(&self) -> i32 {
        self.scorer.embedding_scale
    }
    #[inline(always)]
    fn feature_len(&self) -> usize {
        self.scorer.head.len()
    }
    #[inline(always)]
    fn pattern_count(&self) -> usize {
        self.table.len() / self.scorer.dim
    }
    #[inline]
    fn validate_access(&self) {
        debug_assert_eq!(self.offsets.len(), self.scorer.dim);
        debug_assert_eq!(self.pattern_count(), 199827);
    }
    #[inline(always)]
    fn add_embedding_to(&self, pid: u32, out: &mut [i32]) {
        let start = pid as usize * self.scorer.dim;
        for ((x, &v), &z) in out
            .iter_mut()
            .zip(&self.table[start..start + self.scorer.dim])
            .zip(&self.offsets)
        {
            *x += i32::from(v) + i32::from(z);
        }
    }
    #[inline(always)]
    fn add_embedding_delta_to(&self, old: u32, new: u32, out: &mut [i32]) {
        let a = old as usize * self.scorer.dim;
        let b = new as usize * self.scorer.dim;
        for ((x, &o), &n) in out
            .iter_mut()
            .zip(&self.table[a..a + self.scorer.dim])
            .zip(&self.table[b..b + self.scorer.dim])
        {
            *x += i32::from(n) - i32::from(o);
        }
    }
    #[inline]
    fn score_uniform(&self, f: &[i32], g: usize) -> f32 {
        cb2vec::score_quantized_uniform(f, &QuantizedHeadView(&self.scorer), g)
            .expect("validated exact-i8 head")
    }
}
