//! Experimental no-message-passing codebook evaluator.
//!
//! This module mirrors the normal `IncrementalEval` shape for the RQ542
//! correctness gate. It is deliberately feature-gated and is not wired into
//! search by default.

use crate::board::{BOARD_SIZE, Board, Move, NUM_CELLS, Stone};
use crate::factored_codebook::FactoredQuantizedCodebookWeights;
use crate::pattern_table::{PATTERN_NUM_IDS, swap_mapped_id};
pub use crate::search::EvalStateStepProfile;
pub(crate) use cb2vec::QuantizedCodebookAccess;
use cb2vec::{ReversibleTokenJournal, TokenDelta, TokenDeltaReplay, TokenDeltaSink};
use serde_json::Value;

const MAX_DIRTY_CELLS: usize = 41;
// One move changes at most eleven Pattern4 windows in each of four directions.
const MAX_DIRECTION_DELTAS: usize = 44;
// Per component, a cell is the sum of four i16 embeddings. A region contains
// 25 cells. The wider intermediate bounds also cover one replacement delta
// and one activation delta before the exact invariant is restored.
const _: () = {
    let raw_abs_bound = 4 * (i16::MAX as i64 + 1);
    let replacement_delta_abs_bound = i16::MAX as i64 - i16::MIN as i64;
    let raw_intermediate_abs_bound = raw_abs_bound + replacement_delta_abs_bound;
    let region_abs_bound = 25 * raw_abs_bound;
    let region_intermediate_abs_bound = region_abs_bound + raw_abs_bound;
    assert!(raw_intermediate_abs_bound <= i32::MAX as i64);
    assert!(region_intermediate_abs_bound <= i32::MAX as i64);
};
const REGIONS: usize = 9;
pub const QUANT_EMBED_SCALE: i32 = 32;
pub const QUANT_HEAD_SCALE: i32 = 64;
pub const QUANT_FACTOR_SCALE: i32 = 64;

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
    pub fn deterministic(dim: usize, fm_rank: usize) -> Self {
        let weights =
            cb2vec::CodebookWeights::deterministic(PATTERN_NUM_IDS, REGIONS, dim, fm_rank);
        Self {
            dim: weights.dim,
            fm_rank: weights.fm_rank,
            embeddings: weights.embeddings,
            head: weights.head,
            factors: weights.factors,
            bias: weights.bias,
        }
    }

    pub fn from_json_bytes(data: &[u8]) -> Result<Self, String> {
        let root: Value = serde_json::from_slice(data)
            .map_err(|error| format!("failed to parse codebook json: {error}"))?;
        Self::from_json_value(&root)
    }

    pub fn from_json_value(root: &Value) -> Result<Self, String> {
        validate_figrid_json_schema(root)?;
        let weights =
            cb2vec::CodebookWeights::from_json_value(root).map_err(|error| error.to_string())?;
        Self::from_cb2vec(weights)
    }

    fn from_cb2vec(weights: cb2vec::CodebookWeights) -> Result<Self, String> {
        let shape = weights.validate().map_err(|error| error.to_string())?;
        if shape.token_count() != PATTERN_NUM_IDS {
            return Err(format!(
                "embedding token count mismatch: got {}, expected {PATTERN_NUM_IDS}",
                shape.token_count()
            ));
        }
        if shape.group_count() != REGIONS {
            return Err(format!("unsupported region count: {}", shape.group_count()));
        }
        if shape.fm_rank() == 0 {
            return Err("fm_rank must be non-zero for the FIGRID evaluator".to_string());
        }
        Ok(Self {
            dim: weights.dim,
            fm_rank: weights.fm_rank,
            embeddings: weights.embeddings,
            head: weights.head,
            factors: weights.factors,
            bias: weights.bias,
        })
    }

    #[inline]
    pub fn feature_len(&self) -> usize {
        REGIONS * self.dim
    }

    pub fn quantize_i16_s32_s64(&self) -> QuantizedCodebookWeights {
        self.validate();
        let weights = cb2vec::quantize_i16(
            self,
            QUANT_EMBED_SCALE,
            QUANT_HEAD_SCALE,
            QUANT_FACTOR_SCALE,
        )
        .expect("FIGRID codebook weights must be valid before quantization");
        QuantizedCodebookWeights {
            dim: weights.dim,
            fm_rank: weights.fm_rank,
            embedding_scale: weights.embedding_scale,
            head_scale: weights.head_scale,
            factor_scale: weights.factor_scale,
            embeddings: weights.embeddings,
            head: weights.head,
            factors: weights.factors,
            bias: weights.bias,
        }
    }

    fn validate(&self) {
        debug_assert_eq!(self.embeddings.len(), PATTERN_NUM_IDS * self.dim);
        debug_assert_eq!(self.head.len(), self.feature_len());
        debug_assert_eq!(self.factors.len(), self.feature_len() * self.fm_rank);
    }
}

fn validate_figrid_json_schema(root: &Value) -> Result<(), String> {
    let format = root
        .get("format")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing format".to_string())?;
    if format != "noru-relation-fusion-eval-v1" && format != "noru-pattern4-codebook-eval-v1" {
        return Err(format!("unsupported codebook format: {format}"));
    }

    let model = root
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing model".to_string())?;
    if model != "codebook-region-fm" && model != "region-codebook-fm" {
        return Err(format!(
            "unsupported codebook model: {model}; expected codebook-region-fm"
        ));
    }

    let metadata = root.get("metadata");
    let regions = metadata
        .and_then(|value| value.get("regions"))
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .or_else(|| {
            root.get("regions")
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
        })
        .unwrap_or(REGIONS);
    if regions != REGIONS {
        return Err(format!("unsupported region count: {regions}"));
    }
    Ok(())
}

impl cb2vec::FloatCodebookAccess for CodebookWeights {
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
    #[inline]
    pub fn feature_len(&self) -> usize {
        REGIONS * self.dim
    }

    pub fn dequantized(&self) -> CodebookWeights {
        self.validate();
        CodebookWeights {
            dim: self.dim,
            fm_rank: self.fm_rank,
            embeddings: dequantize_vec_i16(&self.embeddings, self.embedding_scale),
            head: dequantize_vec_i16(&self.head, self.head_scale),
            factors: dequantize_vec_i16(&self.factors, self.factor_scale),
            bias: self.bias,
        }
    }

    fn validate(&self) {
        debug_assert!(self.embedding_scale > 0);
        debug_assert!(self.head_scale > 0);
        debug_assert!(self.factor_scale > 0);
        debug_assert_eq!(self.embeddings.len(), PATTERN_NUM_IDS * self.dim);
        debug_assert_eq!(self.head.len(), self.feature_len());
        debug_assert_eq!(self.factors.len(), self.feature_len() * self.fm_rank);
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
        self.embeddings.len() / self.dim
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
    fn embedding(&self, pattern_id: u16, component: usize) -> i16 {
        self.embeddings[pattern_id as usize * self.dim + component]
    }

    #[inline(always)]
    fn embedding_delta(&self, old_pattern_id: u16, new_pattern_id: u16, component: usize) -> i32 {
        let old = self.embeddings[old_pattern_id as usize * self.dim + component];
        let new = self.embeddings[new_pattern_id as usize * self.dim + component];
        i32::from(new) - i32::from(old)
    }

    #[inline(always)]
    fn add_embedding_to(&self, pattern_id: u16, out: &mut [i32]) {
        let start = pattern_id as usize * self.dim;
        let embedding = &self.embeddings[start..start + self.dim];
        for (value, &component) in out.iter_mut().zip(embedding) {
            *value += i32::from(component);
        }
    }

    #[inline(always)]
    fn add_embedding_delta_to(&self, old_pattern_id: u16, new_pattern_id: u16, out: &mut [i32]) {
        let old_start = old_pattern_id as usize * self.dim;
        let new_start = new_pattern_id as usize * self.dim;
        let old = &self.embeddings[old_start..old_start + self.dim];
        let new = &self.embeddings[new_start..new_start + self.dim];
        for ((value, &old), &new) in out.iter_mut().zip(old).zip(new) {
            *value += i32::from(new) - i32::from(old);
        }
    }
}

impl QuantizedCodebookAccess for FactoredQuantizedCodebookWeights {
    #[inline(always)]
    fn dim(&self) -> usize {
        self.dim()
    }

    #[inline(always)]
    fn fm_rank(&self) -> usize {
        self.fm_rank()
    }

    #[inline(always)]
    fn embedding_scale(&self) -> i32 {
        self.embedding_scale()
    }

    #[inline(always)]
    fn head_scale(&self) -> i32 {
        self.head_scale()
    }

    #[inline(always)]
    fn factor_scale(&self) -> i32 {
        self.factor_scale()
    }

    #[inline(always)]
    fn bias(&self) -> f32 {
        self.bias()
    }

    #[inline(always)]
    fn token_count(&self) -> usize {
        self.token_count()
    }

    #[inline(always)]
    fn head(&self) -> &[i16] {
        self.head()
    }

    #[inline(always)]
    fn factors(&self) -> &[i16] {
        self.factors()
    }

    #[inline(always)]
    fn embedding(&self, pattern_id: u16, component: usize) -> i16 {
        let pattern_id = pattern_id as usize;
        let dim = self.dim();
        let class = self.classes()[pattern_id] as usize;
        let base = self.bases()[class * dim + component];
        let residual = self.residuals()[pattern_id * dim + component];
        (i32::from(base) + i32::from(residual)) as i16
    }

    #[inline(always)]
    fn embedding_delta(&self, old_pattern_id: u16, new_pattern_id: u16, component: usize) -> i32 {
        let old_pattern_id = old_pattern_id as usize;
        let new_pattern_id = new_pattern_id as usize;
        let dim = self.dim();
        let old_class = self.classes()[old_pattern_id] as usize;
        let new_class = self.classes()[new_pattern_id] as usize;
        let old_residual = self.residuals()[old_pattern_id * dim + component];
        let new_residual = self.residuals()[new_pattern_id * dim + component];
        let residual_delta = i32::from(new_residual) - i32::from(old_residual);
        if old_class == new_class {
            // The shared base cancels exactly and is not loaded.
            residual_delta
        } else {
            let old_base = self.bases()[old_class * dim + component];
            let new_base = self.bases()[new_class * dim + component];
            residual_delta + i32::from(new_base) - i32::from(old_base)
        }
    }

    #[inline(always)]
    fn add_embedding_to(&self, pattern_id: u16, out: &mut [i32]) {
        let pattern_id = pattern_id as usize;
        let dim = self.dim();
        let class = self.classes()[pattern_id] as usize;
        let base_start = class * dim;
        let residual_start = pattern_id * dim;
        let base = &self.bases()[base_start..base_start + dim];
        let residual = &self.residuals()[residual_start..residual_start + dim];
        for ((value, &base), &residual) in out.iter_mut().zip(base).zip(residual) {
            *value += i32::from(base) + i32::from(residual);
        }
    }

    #[inline(always)]
    fn add_embedding_delta_to(&self, old_pattern_id: u16, new_pattern_id: u16, out: &mut [i32]) {
        let old_pattern_id = old_pattern_id as usize;
        let new_pattern_id = new_pattern_id as usize;
        let dim = self.dim();
        let old_class = self.classes()[old_pattern_id] as usize;
        let new_class = self.classes()[new_pattern_id] as usize;
        let old_start = old_pattern_id * dim;
        let new_start = new_pattern_id * dim;
        let old_residual = &self.residuals()[old_start..old_start + dim];
        let new_residual = &self.residuals()[new_start..new_start + dim];
        if old_class == new_class {
            // The shared class base cancels for the whole vector.
            for ((value, &old), &new) in out.iter_mut().zip(old_residual).zip(new_residual) {
                *value += i32::from(new) - i32::from(old);
            }
        } else {
            let old_base_start = old_class * dim;
            let new_base_start = new_class * dim;
            let old_base = &self.bases()[old_base_start..old_base_start + dim];
            let new_base = &self.bases()[new_base_start..new_base_start + dim];
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

pub struct IncrementalCodebookEval {
    cell_black: Vec<f32>,
    cell_white: Vec<f32>,
    features_black: Vec<f32>,
    features_white: Vec<f32>,
    stack: Vec<UndoRecord>,
    last_dirty_cells: usize,
}

struct UndoRecord {
    changes: Vec<CellUndo>,
}

struct CellUndo {
    cell: usize,
    black: Vec<f32>,
    white: Vec<f32>,
}

impl IncrementalCodebookEval {
    pub fn new(weights: &CodebookWeights) -> Self {
        weights.validate();
        Self {
            cell_black: vec![0.0; NUM_CELLS * weights.dim],
            cell_white: vec![0.0; NUM_CELLS * weights.dim],
            features_black: vec![0.0; weights.feature_len()],
            features_white: vec![0.0; weights.feature_len()],
            stack: Vec::with_capacity(NUM_CELLS),
            last_dirty_cells: 0,
        }
    }

    pub fn refresh(&mut self, board: &Board, weights: &CodebookWeights) {
        weights.validate();
        self.cell_black.fill(0.0);
        self.cell_white.fill(0.0);
        self.features_black.fill(0.0);
        self.features_white.fill(0.0);

        for cell in 0..NUM_CELLS {
            compute_cell(
                board,
                weights,
                cell,
                Stone::Black,
                cell_slice_mut(&mut self.cell_black, cell, weights.dim),
            );
            add_cell_to_features(
                &self.cell_black,
                &mut self.features_black,
                cell,
                weights.dim,
                1.0,
            );

            compute_cell(
                board,
                weights,
                cell,
                Stone::White,
                cell_slice_mut(&mut self.cell_white, cell, weights.dim),
            );
            add_cell_to_features(
                &self.cell_white,
                &mut self.features_white,
                cell,
                weights.dim,
                1.0,
            );
        }

        self.stack.clear();
        self.last_dirty_cells = 0;
    }

    pub fn push_move(&mut self, board: &Board, mv: Move, weights: &CodebookWeights) {
        weights.validate();
        let dirty = dirty_cells_for_move(mv);
        let mut undo = UndoRecord {
            changes: Vec::with_capacity(dirty.len()),
        };

        for cell in dirty.iter().copied() {
            let old_black = cell_slice(&self.cell_black, cell, weights.dim).to_vec();
            let old_white = cell_slice(&self.cell_white, cell, weights.dim).to_vec();

            add_cell_to_features(
                &self.cell_black,
                &mut self.features_black,
                cell,
                weights.dim,
                -1.0,
            );
            add_cell_to_features(
                &self.cell_white,
                &mut self.features_white,
                cell,
                weights.dim,
                -1.0,
            );

            compute_cell(
                board,
                weights,
                cell,
                Stone::Black,
                cell_slice_mut(&mut self.cell_black, cell, weights.dim),
            );
            compute_cell(
                board,
                weights,
                cell,
                Stone::White,
                cell_slice_mut(&mut self.cell_white, cell, weights.dim),
            );

            add_cell_to_features(
                &self.cell_black,
                &mut self.features_black,
                cell,
                weights.dim,
                1.0,
            );
            add_cell_to_features(
                &self.cell_white,
                &mut self.features_white,
                cell,
                weights.dim,
                1.0,
            );

            undo.changes.push(CellUndo {
                cell,
                black: old_black,
                white: old_white,
            });
        }

        self.last_dirty_cells = dirty.len();
        self.stack.push(undo);
    }

    pub fn pop_move(&mut self, weights: &CodebookWeights) {
        let Some(undo) = self.stack.pop() else {
            return;
        };
        for change in undo.changes.into_iter().rev() {
            add_cell_to_features(
                &self.cell_black,
                &mut self.features_black,
                change.cell,
                weights.dim,
                -1.0,
            );
            add_cell_to_features(
                &self.cell_white,
                &mut self.features_white,
                change.cell,
                weights.dim,
                -1.0,
            );

            cell_slice_mut(&mut self.cell_black, change.cell, weights.dim)
                .copy_from_slice(&change.black);
            cell_slice_mut(&mut self.cell_white, change.cell, weights.dim)
                .copy_from_slice(&change.white);

            add_cell_to_features(
                &self.cell_black,
                &mut self.features_black,
                change.cell,
                weights.dim,
                1.0,
            );
            add_cell_to_features(
                &self.cell_white,
                &mut self.features_white,
                change.cell,
                weights.dim,
                1.0,
            );
        }
        self.last_dirty_cells = 0;
    }

    pub fn value(&self, board: &Board, weights: &CodebookWeights) -> f32 {
        let features = match board.side_to_move {
            Stone::Black => &self.features_black,
            Stone::White => &self.features_white,
        };
        value_from_features(features, weights)
    }

    pub fn last_dirty_cells(&self) -> usize {
        self.last_dirty_cells
    }

    pub fn last_dirty_ratio(&self) -> f32 {
        self.last_dirty_cells as f32 / NUM_CELLS as f32
    }
}

pub fn evaluate_full(board: &Board, weights: &CodebookWeights) -> f32 {
    let mut inc = IncrementalCodebookEval::new(weights);
    inc.refresh(board, weights);
    inc.value(board, weights)
}

pub struct IncrementalQuantizedCodebookEval {
    cell_black: Vec<i32>,
    cell_white: Vec<i32>,
    features_black: Vec<i32>,
    features_white: Vec<i32>,
    stack: Vec<QuantUndoRecord>,
    stack_len: usize,
    last_dirty_cells: usize,
    last_direction_deltas: usize,
    directional_delta: Option<QuantDirectionalDeltaState>,
}

struct QuantUndoRecord {
    len: usize,
    materialized: bool,
    cells: [usize; MAX_DIRTY_CELLS],
    pattern_ids: [[u16; 4]; MAX_DIRTY_CELLS],
    black: Vec<i32>,
    white: Vec<i32>,
}

struct QuantDirectionalDeltaState {
    raw_black: Vec<i32>,
    raw_white: Vec<i32>,
    journal: ReversibleTokenJournal<u16, 4, MAX_DIRECTION_DELTAS>,
}

impl QuantDirectionalDeltaState {
    fn new(dim: usize) -> Self {
        Self {
            raw_black: vec![0; NUM_CELLS * dim],
            raw_white: vec![0; NUM_CELLS * dim],
            journal: ReversibleTokenJournal::new(NUM_CELLS, NUM_CELLS),
        }
    }

    #[cfg(test)]
    fn logical_pattern_ids(&self) -> &[[u16; 4]] {
        self.journal.logical_tokens()
    }
}

struct QuantizedCodebookTokenSink<'a, W: QuantizedCodebookAccess> {
    weights: &'a W,
    raw_black: &'a mut [i32],
    raw_white: &'a mut [i32],
    cell_black: &'a mut [i32],
    cell_white: &'a mut [i32],
    features_black: &'a mut [i32],
    features_white: &'a mut [i32],
    profile: &'a mut EvalStateStepProfile,
    profile_enabled: bool,
    restore: bool,
}

impl<'a, W: QuantizedCodebookAccess> QuantizedCodebookTokenSink<'a, W> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        weights: &'a W,
        raw_black: &'a mut [i32],
        raw_white: &'a mut [i32],
        cell_black: &'a mut [i32],
        cell_white: &'a mut [i32],
        features_black: &'a mut [i32],
        features_white: &'a mut [i32],
        profile: &'a mut EvalStateStepProfile,
        profile_enabled: bool,
        restore: bool,
    ) -> Self {
        Self {
            weights,
            raw_black,
            raw_white,
            cell_black,
            cell_white,
            features_black,
            features_white,
            profile,
            profile_enabled,
            restore,
        }
    }

    #[inline(always)]
    fn apply_delta(
        weights: &W,
        site: u16,
        delta: TokenDelta<u16>,
        raw_black: &mut [i32],
        raw_white: &mut [i32],
    ) {
        debug_assert_eq!(site, delta.site());
        debug_assert!((delta.lane() as usize) < 4);
        apply_quantized_token_delta_to_raw(
            delta.old(),
            delta.new_token(),
            weights,
            Stone::Black,
            raw_black,
        );
        apply_quantized_token_delta_to_raw(
            delta.old(),
            delta.new_token(),
            weights,
            Stone::White,
            raw_white,
        );
    }
}

impl<W: QuantizedCodebookAccess> TokenDeltaSink<u16> for QuantizedCodebookTokenSink<'_, W> {
    #[inline]
    fn apply_site(&mut self, site: u16, deltas: &[TokenDelta<u16>], replay: TokenDeltaReplay) {
        let cell = site as usize;
        let numeric_start = EvalStateStepProfile::start(self.profile_enabled);
        let weights = self.weights;
        let raw_black = quant_cell_slice_mut(self.raw_black, cell, weights.dim());
        let raw_white = quant_cell_slice_mut(self.raw_white, cell, weights.dim());
        match replay {
            TokenDeltaReplay::Forward => {
                for &delta in deltas {
                    Self::apply_delta(weights, site, delta, raw_black, raw_white);
                }
            }
            TokenDeltaReplay::Reverse => {
                for &delta in deltas.iter().rev() {
                    Self::apply_delta(weights, site, delta.reversed(), raw_black, raw_white);
                }
            }
        }
        if self.restore {
            self.profile.add_restore(numeric_start);
        } else {
            self.profile.add_recompute(numeric_start);
        }

        let aggregate_start = EvalStateStepProfile::start(self.profile_enabled);
        refresh_quantized_cell_activation(
            self.raw_black,
            self.cell_black,
            self.features_black,
            cell,
            self.weights.dim(),
        );
        refresh_quantized_cell_activation(
            self.raw_white,
            self.cell_white,
            self.features_white,
            cell,
            self.weights.dim(),
        );
        self.profile.add_aggregate(aggregate_start);
    }
}

impl QuantUndoRecord {
    fn new(dim: usize) -> Self {
        Self {
            len: 0,
            materialized: false,
            cells: [0; MAX_DIRTY_CELLS],
            pattern_ids: [[0u16; 4]; MAX_DIRTY_CELLS],
            black: vec![0; MAX_DIRTY_CELLS * dim],
            white: vec![0; MAX_DIRTY_CELLS * dim],
        }
    }

    fn clear(&mut self) {
        self.len = 0;
        self.materialized = false;
    }
}

impl IncrementalQuantizedCodebookEval {
    /// 0.8.6: the directional-delta materialization path (bit-identical to
    /// the legacy full-recompute path, roughly 2x faster on it) is now the
    /// default for every consumer. Use [`Self::new_with_directional_delta`]
    /// with `false` only to A/B the legacy path.
    pub fn new(weights: &QuantizedCodebookWeights) -> Self {
        Self::new_with_directional_delta(weights, true)
    }

    pub fn new_with_directional_delta(
        weights: &QuantizedCodebookWeights,
        directional_delta: bool,
    ) -> Self {
        Self::new_with_access(weights, directional_delta)
    }

    pub(crate) fn new_with_access<W: QuantizedCodebookAccess>(
        weights: &W,
        directional_delta: bool,
    ) -> Self {
        weights.validate_access();
        let dim = weights.dim();
        Self {
            cell_black: vec![0; NUM_CELLS * dim],
            cell_white: vec![0; NUM_CELLS * dim],
            features_black: vec![0; weights.feature_len()],
            features_white: vec![0; weights.feature_len()],
            stack: if directional_delta {
                Vec::new()
            } else {
                (0..NUM_CELLS).map(|_| QuantUndoRecord::new(dim)).collect()
            },
            stack_len: 0,
            last_dirty_cells: 0,
            last_direction_deltas: 0,
            directional_delta: directional_delta.then(|| QuantDirectionalDeltaState::new(dim)),
        }
    }

    pub fn directional_delta_enabled(&self) -> bool {
        self.directional_delta.is_some()
    }

    pub fn refresh(&mut self, board: &Board, weights: &QuantizedCodebookWeights) {
        self.refresh_with_access(board, weights);
    }

    pub(crate) fn refresh_with_access<W: QuantizedCodebookAccess>(
        &mut self,
        board: &Board,
        weights: &W,
    ) {
        weights.validate_access();
        let dim = weights.dim();
        self.cell_black.fill(0);
        self.cell_white.fill(0);
        self.features_black.fill(0);
        self.features_white.fill(0);

        for cell in 0..NUM_CELLS {
            compute_cell_quantized(
                board,
                weights,
                cell,
                Stone::Black,
                quant_cell_slice_mut(&mut self.cell_black, cell, dim),
            );
            add_quant_cell_to_features(&self.cell_black, &mut self.features_black, cell, dim, 1);

            compute_cell_quantized(
                board,
                weights,
                cell,
                Stone::White,
                quant_cell_slice_mut(&mut self.cell_white, cell, dim),
            );
            add_quant_cell_to_features(&self.cell_white, &mut self.features_white, cell, dim, 1);
        }

        if let Some(state) = self.directional_delta.as_mut() {
            state.journal.reset(board.line_pattern_ids.as_ref());
            for cell in 0..NUM_CELLS {
                compute_cell_quantized_raw_from_pattern_ids(
                    &board.line_pattern_ids[cell],
                    weights,
                    Stone::Black,
                    quant_cell_slice_mut(&mut state.raw_black, cell, dim),
                );
                compute_cell_quantized_raw_from_pattern_ids(
                    &board.line_pattern_ids[cell],
                    weights,
                    Stone::White,
                    quant_cell_slice_mut(&mut state.raw_white, cell, dim),
                );
            }
        }

        self.stack_len = 0;
        self.last_dirty_cells = 0;
        self.last_direction_deltas = 0;
    }

    pub fn push_move(&mut self, board: &Board, mv: Move, weights: &QuantizedCodebookWeights) {
        let _ = self.push_move_profiled_with_access(board, mv, weights, false);
    }

    pub fn push_move_profiled(
        &mut self,
        board: &Board,
        mv: Move,
        weights: &QuantizedCodebookWeights,
        profile_enabled: bool,
    ) -> EvalStateStepProfile {
        self.push_move_profiled_with_access(board, mv, weights, profile_enabled)
    }

    pub(crate) fn push_move_profiled_with_access<W: QuantizedCodebookAccess>(
        &mut self,
        board: &Board,
        mv: Move,
        weights: &W,
        profile_enabled: bool,
    ) -> EvalStateStepProfile {
        weights.validate_access();
        if self.directional_delta.is_some() {
            return self.push_move_directional_delta(board, mv, weights, profile_enabled);
        }
        let mut profile = EvalStateStepProfile {
            push_calls: 1,
            ..EvalStateStepProfile::default()
        };
        let start = EvalStateStepProfile::start(profile_enabled);
        let dirty = dirty_cells_for_move(mv);
        profile.add_dirty_list(start);
        debug_assert!(dirty.len() <= MAX_DIRTY_CELLS);
        debug_assert!(
            self.stack_len < self.stack.len(),
            "quantized codebook undo stack overflow"
        );
        let undo = &mut self.stack[self.stack_len];
        undo.clear();

        let start = EvalStateStepProfile::start(profile_enabled);
        for cell in dirty.iter().copied() {
            let undo_idx = undo.len;
            undo.cells[undo_idx] = cell;
            undo.pattern_ids[undo_idx] = board.line_pattern_ids[cell];
            undo.len += 1;
        }
        profile.add_frame_write(start);

        self.last_dirty_cells = dirty.len();
        self.stack_len += 1;
        profile
    }

    fn push_move_directional_delta<W: QuantizedCodebookAccess>(
        &mut self,
        board: &Board,
        mv: Move,
        weights: &W,
        profile_enabled: bool,
    ) -> EvalStateStepProfile {
        weights.validate_access();
        let mut profile = EvalStateStepProfile {
            push_calls: 1,
            ..EvalStateStepProfile::default()
        };
        let start = EvalStateStepProfile::start(profile_enabled);
        let dirty = dirty_cells_for_move(mv);
        profile.add_dirty_list(start);
        debug_assert!(dirty.len() <= MAX_DIRTY_CELLS);

        let start = EvalStateStepProfile::start(profile_enabled);
        let state = self
            .directional_delta
            .as_mut()
            .expect("directional delta state enabled");
        debug_assert!(state.journal.depth() < NUM_CELLS);
        let direction_deltas = state
            .journal
            .push_after(board.line_pattern_ids.as_ref(), &dirty);
        profile.add_frame_write(start);

        self.last_dirty_cells = dirty.len();
        self.last_direction_deltas = direction_deltas;
        profile
    }

    fn materialize_pending<W: QuantizedCodebookAccess>(
        &mut self,
        weights: &W,
        profile_enabled: bool,
    ) -> EvalStateStepProfile {
        if self.directional_delta.is_some() {
            return self.materialize_pending_directional_delta(weights, profile_enabled);
        }
        let mut profile = EvalStateStepProfile::default();
        let dim = weights.dim();
        for frame_idx in 0..self.stack_len {
            if self.stack[frame_idx].materialized {
                continue;
            }
            let undo = &mut self.stack[frame_idx];
            for undo_idx in 0..undo.len {
                let cell = undo.cells[undo_idx];
                let undo_base = undo_idx * dim;

                let start = EvalStateStepProfile::start(profile_enabled);
                undo.black[undo_base..undo_base + dim].copy_from_slice(quant_cell_slice(
                    &self.cell_black,
                    cell,
                    dim,
                ));
                undo.white[undo_base..undo_base + dim].copy_from_slice(quant_cell_slice(
                    &self.cell_white,
                    cell,
                    dim,
                ));
                profile.add_backup(start);

                let start = EvalStateStepProfile::start(profile_enabled);
                add_quant_cell_to_features(
                    &self.cell_black,
                    &mut self.features_black,
                    cell,
                    dim,
                    -1,
                );
                add_quant_cell_to_features(
                    &self.cell_white,
                    &mut self.features_white,
                    cell,
                    dim,
                    -1,
                );
                profile.add_aggregate(start);

                let start = EvalStateStepProfile::start(profile_enabled);
                compute_cell_quantized_from_pattern_ids(
                    &undo.pattern_ids[undo_idx],
                    weights,
                    Stone::Black,
                    quant_cell_slice_mut(&mut self.cell_black, cell, dim),
                );
                compute_cell_quantized_from_pattern_ids(
                    &undo.pattern_ids[undo_idx],
                    weights,
                    Stone::White,
                    quant_cell_slice_mut(&mut self.cell_white, cell, dim),
                );
                profile.add_recompute(start);

                let start = EvalStateStepProfile::start(profile_enabled);
                add_quant_cell_to_features(
                    &self.cell_black,
                    &mut self.features_black,
                    cell,
                    dim,
                    1,
                );
                add_quant_cell_to_features(
                    &self.cell_white,
                    &mut self.features_white,
                    cell,
                    dim,
                    1,
                );
                profile.add_aggregate(start);
            }
            undo.materialized = true;
        }
        profile
    }

    fn materialize_pending_directional_delta<W: QuantizedCodebookAccess>(
        &mut self,
        weights: &W,
        profile_enabled: bool,
    ) -> EvalStateStepProfile {
        let mut profile = EvalStateStepProfile::default();
        let Self {
            cell_black,
            cell_white,
            features_black,
            features_white,
            directional_delta,
            ..
        } = self;
        let state = directional_delta
            .as_mut()
            .expect("directional delta state enabled");
        let QuantDirectionalDeltaState {
            raw_black,
            raw_white,
            journal,
            ..
        } = state;
        let mut sink = QuantizedCodebookTokenSink::new(
            weights,
            raw_black,
            raw_white,
            cell_black,
            cell_white,
            features_black,
            features_white,
            &mut profile,
            profile_enabled,
            false,
        );
        journal.materialize_pending(&mut sink);
        profile
    }

    pub fn pop_move(&mut self, weights: &QuantizedCodebookWeights) {
        let _ = self.pop_move_profiled_with_access(weights, false);
    }

    pub fn pop_move_profiled(
        &mut self,
        weights: &QuantizedCodebookWeights,
        profile_enabled: bool,
    ) -> EvalStateStepProfile {
        self.pop_move_profiled_with_access(weights, profile_enabled)
    }

    pub(crate) fn pop_move_profiled_with_access<W: QuantizedCodebookAccess>(
        &mut self,
        weights: &W,
        profile_enabled: bool,
    ) -> EvalStateStepProfile {
        let mut profile = EvalStateStepProfile {
            pop_calls: 1,
            ..EvalStateStepProfile::default()
        };
        if let Some(state) = self.directional_delta.as_ref() {
            if state.journal.depth() == 0 {
                return profile;
            }
            return self.pop_move_directional_delta(weights, profile_enabled, profile);
        }
        if self.stack_len == 0 {
            return profile;
        }
        self.stack_len -= 1;
        let undo = &self.stack[self.stack_len];
        let dim = weights.dim();
        if !undo.materialized {
            self.last_dirty_cells = 0;
            return profile;
        }
        for undo_idx in (0..undo.len).rev() {
            let cell = undo.cells[undo_idx];
            let undo_base = undo_idx * dim;
            let start = EvalStateStepProfile::start(profile_enabled);
            add_quant_cell_to_features(&self.cell_black, &mut self.features_black, cell, dim, -1);
            add_quant_cell_to_features(&self.cell_white, &mut self.features_white, cell, dim, -1);
            profile.add_aggregate(start);

            let start = EvalStateStepProfile::start(profile_enabled);
            quant_cell_slice_mut(&mut self.cell_black, cell, dim)
                .copy_from_slice(&undo.black[undo_base..undo_base + dim]);
            quant_cell_slice_mut(&mut self.cell_white, cell, dim)
                .copy_from_slice(&undo.white[undo_base..undo_base + dim]);
            profile.add_restore(start);

            let start = EvalStateStepProfile::start(profile_enabled);
            add_quant_cell_to_features(&self.cell_black, &mut self.features_black, cell, dim, 1);
            add_quant_cell_to_features(&self.cell_white, &mut self.features_white, cell, dim, 1);
            profile.add_aggregate(start);
        }
        self.last_dirty_cells = 0;
        profile
    }

    fn pop_move_directional_delta<W: QuantizedCodebookAccess>(
        &mut self,
        weights: &W,
        profile_enabled: bool,
        mut profile: EvalStateStepProfile,
    ) -> EvalStateStepProfile {
        let Self {
            cell_black,
            cell_white,
            features_black,
            features_white,
            directional_delta,
            ..
        } = self;
        let state = directional_delta
            .as_mut()
            .expect("directional delta state enabled");
        let QuantDirectionalDeltaState {
            raw_black,
            raw_white,
            journal,
            ..
        } = state;
        let mut sink = QuantizedCodebookTokenSink::new(
            weights,
            raw_black,
            raw_white,
            cell_black,
            cell_white,
            features_black,
            features_white,
            &mut profile,
            profile_enabled,
            true,
        );
        let popped = journal.pop(&mut sink).expect("TokenDelta stack underflow");
        debug_assert!(popped.deltas() <= MAX_DIRECTION_DELTAS);
        debug_assert!(journal.materialized_depth() <= journal.depth());
        self.last_dirty_cells = 0;
        self.last_direction_deltas = 0;
        profile
    }

    pub fn value(&mut self, board: &Board, weights: &QuantizedCodebookWeights) -> f32 {
        self.value_profiled_with_access(board, weights, false).0
    }

    pub fn value_profiled(
        &mut self,
        board: &Board,
        weights: &QuantizedCodebookWeights,
        profile_enabled: bool,
    ) -> (f32, EvalStateStepProfile) {
        self.value_profiled_with_access(board, weights, profile_enabled)
    }

    pub(crate) fn value_profiled_with_access<W: QuantizedCodebookAccess>(
        &mut self,
        board: &Board,
        weights: &W,
        profile_enabled: bool,
    ) -> (f32, EvalStateStepProfile) {
        let mut profile = self.materialize_pending(weights, profile_enabled);
        let features = match board.side_to_move {
            Stone::Black => &self.features_black,
            Stone::White => &self.features_white,
        };
        let start = EvalStateStepProfile::start(profile_enabled);
        let value = quant_value_from_features(features, weights);
        profile.add_forward(start);
        (value, profile)
    }

    /// Materialize the quantized 9x16 cache and return its D4-invariant
    /// corner/edge/center orbit sums from an explicit color perspective.
    ///
    /// This deliberately does not consult side-to-move: the White root
    /// ordering model scores the position after a candidate White move, when
    /// the child itself is Black-to-move.
    #[allow(dead_code)]
    pub(crate) fn explicit_orbit48(
        &mut self,
        weights: &QuantizedCodebookWeights,
        perspective: Stone,
    ) -> Result<[i64; 48], String> {
        self.explicit_orbit48_with_access(weights, perspective)
    }

    pub(crate) fn explicit_orbit48_with_access<W: QuantizedCodebookAccess>(
        &mut self,
        weights: &W,
        perspective: Stone,
    ) -> Result<[i64; 48], String> {
        if weights.dim() != 16 {
            return Err(format!(
                "white root ordering requires codebook dim 16, got {}",
                weights.dim()
            ));
        }
        if weights.embedding_scale() != QUANT_EMBED_SCALE || weights.embedding_scale() != 32 {
            return Err(format!(
                "white root ordering requires embedding scale 32, got {}",
                weights.embedding_scale()
            ));
        }
        let expected_features = REGIONS * 16;
        if self.features_black.len() != expected_features
            || self.features_white.len() != expected_features
            || self.cell_black.len() != NUM_CELLS * 16
            || self.cell_white.len() != NUM_CELLS * 16
            || weights.feature_len() != expected_features
            || weights.pattern_count() != PATTERN_NUM_IDS
        {
            return Err("white root ordering codebook shape mismatch".to_string());
        }

        let _ = self.materialize_pending(weights, false);
        let features = match perspective {
            Stone::Black => &self.features_black,
            Stone::White => &self.features_white,
        };
        let mut result = [0i64; 48];
        const CORNERS: [usize; 4] = [0, 2, 6, 8];
        const EDGES: [usize; 4] = [1, 3, 5, 7];
        for dim in 0..16 {
            let mut corner = 0i64;
            for region in CORNERS {
                corner = corner
                    .checked_add(i64::from(features[region * 16 + dim]))
                    .ok_or_else(|| format!("corner orbit overflow at dimension {dim}"))?;
            }
            let mut edge = 0i64;
            for region in EDGES {
                edge = edge
                    .checked_add(i64::from(features[region * 16 + dim]))
                    .ok_or_else(|| format!("edge orbit overflow at dimension {dim}"))?;
            }
            result[dim] = corner;
            result[16 + dim] = edge;
            result[32 + dim] = i64::from(features[4 * 16 + dim]);
        }
        Ok(result)
    }

    pub fn last_dirty_cells(&self) -> usize {
        self.last_dirty_cells
    }

    pub fn last_direction_deltas(&self) -> usize {
        self.last_direction_deltas
    }
}

pub fn evaluate_full_quantized(board: &Board, weights: &QuantizedCodebookWeights) -> f32 {
    let mut inc = IncrementalQuantizedCodebookEval::new(weights);
    inc.refresh(board, weights);
    inc.value(board, weights)
}

/// CB-AL1-only full-refresh audit entry point for the deployed factored
/// access path. It is deliberately unavailable to ordinary product builds.
#[cfg(feature = "cb-al1-audit")]
#[doc(hidden)]
pub fn evaluate_full_factored_quantized_for_audit(
    board: &Board,
    weights: &FactoredQuantizedCodebookWeights,
) -> f32 {
    let mut inc = IncrementalQuantizedCodebookEval::new_with_access(weights, false);
    inc.refresh_with_access(board, weights);
    inc.value_profiled_with_access(board, weights, false).0
}

pub fn dirty_cells_for_move(mv: Move) -> Vec<usize> {
    const DIRS: [(i32, i32); 4] = [(1, 0), (0, 1), (1, 1), (1, -1)];
    let row = (mv / BOARD_SIZE) as i32;
    let col = (mv % BOARD_SIZE) as i32;
    let mut seen = [false; NUM_CELLS];
    let mut cells = Vec::with_capacity(MAX_DIRTY_CELLS);
    for &(dr, dc) in &DIRS {
        for offset in -5i32..=5 {
            let r = row + dr * offset;
            let c = col + dc * offset;
            if !in_bounds(r, c) {
                continue;
            }
            let cell = r as usize * BOARD_SIZE + c as usize;
            if !seen[cell] {
                seen[cell] = true;
                cells.push(cell);
            }
        }
    }
    cells
}

#[inline]
fn in_bounds(row: i32, col: i32) -> bool {
    row >= 0 && row < BOARD_SIZE as i32 && col >= 0 && col < BOARD_SIZE as i32
}

fn compute_cell(
    board: &Board,
    weights: &CodebookWeights,
    cell: usize,
    perspective: Stone,
    out: &mut [f32],
) {
    out.fill(0.0);
    let swap = perspective == Stone::White;
    for &pid in &board.line_pattern_ids[cell] {
        let pid = if swap { swap_mapped_id(pid) } else { pid };
        let emb_base = pid as usize * weights.dim;
        for d in 0..weights.dim {
            out[d] += weights.embeddings[emb_base + d];
        }
    }
    for x in out {
        *x = x.max(0.0);
    }
}

fn compute_cell_quantized<W: QuantizedCodebookAccess>(
    board: &Board,
    weights: &W,
    cell: usize,
    perspective: Stone,
    out: &mut [i32],
) {
    compute_cell_quantized_from_pattern_ids(
        &board.line_pattern_ids[cell],
        weights,
        perspective,
        out,
    );
}

fn compute_cell_quantized_from_pattern_ids<W: QuantizedCodebookAccess>(
    pattern_ids: &[u16; 4],
    weights: &W,
    perspective: Stone,
    out: &mut [i32],
) {
    compute_cell_quantized_raw_from_pattern_ids(pattern_ids, weights, perspective, out);
    for x in out {
        *x = (*x).max(0);
    }
}

fn compute_cell_quantized_raw_from_pattern_ids<W: QuantizedCodebookAccess>(
    pattern_ids: &[u16; 4],
    weights: &W,
    perspective: Stone,
    out: &mut [i32],
) {
    out.fill(0);
    let swap = perspective == Stone::White;
    for &pid in pattern_ids {
        let pid = if swap { swap_mapped_id(pid) } else { pid };
        weights.add_embedding_to(pid, out);
    }
}

#[inline]
fn apply_quantized_token_delta_to_raw<W: QuantizedCodebookAccess>(
    old_pattern_id: u16,
    new_pattern_id: u16,
    weights: &W,
    perspective: Stone,
    raw: &mut [i32],
) {
    let (old_pattern_id, new_pattern_id) = if perspective == Stone::White {
        (
            swap_mapped_id(old_pattern_id),
            swap_mapped_id(new_pattern_id),
        )
    } else {
        (old_pattern_id, new_pattern_id)
    };
    weights.add_embedding_delta_to(old_pattern_id, new_pattern_id, raw);
}

fn refresh_quantized_cell_activation(
    raw_cells: &[i32],
    activated_cells: &mut [i32],
    features: &mut [i32],
    cell: usize,
    dim: usize,
) {
    let region = region_of_cell(cell);
    let cell_base = cell * dim;
    let feature_base = region * dim;
    for d in 0..dim {
        let new_value = raw_cells[cell_base + d].max(0);
        let old_value = activated_cells[cell_base + d];
        activated_cells[cell_base + d] = new_value;
        features[feature_base + d] += new_value - old_value;
    }
}

fn add_cell_to_features(cells: &[f32], features: &mut [f32], cell: usize, dim: usize, scale: f32) {
    let region = region_of_cell(cell);
    let denom = region_cell_count(region) as f32;
    let cell_base = cell * dim;
    let feature_base = region * dim;
    for d in 0..dim {
        features[feature_base + d] += scale * cells[cell_base + d] / denom;
    }
}

fn add_quant_cell_to_features(
    cells: &[i32],
    features: &mut [i32],
    cell: usize,
    dim: usize,
    sign: i32,
) {
    let region = region_of_cell(cell);
    let cell_base = cell * dim;
    let feature_base = region * dim;
    for d in 0..dim {
        features[feature_base + d] += sign * cells[cell_base + d];
    }
}

fn value_from_features(features: &[f32], weights: &CodebookWeights) -> f32 {
    cb2vec::score_f32(features, weights).expect("FIGRID floating codebook shape is validated")
}

fn quant_value_from_features<W: QuantizedCodebookAccess>(features: &[i32], weights: &W) -> f32 {
    cb2vec::score_quantized_uniform(features, weights, region_cell_count(0))
        .expect("FIGRID quantized codebook shape is validated")
}

#[inline]
fn cell_slice(cells: &[f32], cell: usize, dim: usize) -> &[f32] {
    let start = cell * dim;
    &cells[start..start + dim]
}

#[inline]
fn cell_slice_mut(cells: &mut [f32], cell: usize, dim: usize) -> &mut [f32] {
    let start = cell * dim;
    &mut cells[start..start + dim]
}

#[inline]
fn quant_cell_slice(cells: &[i32], cell: usize, dim: usize) -> &[i32] {
    let start = cell * dim;
    &cells[start..start + dim]
}

#[inline]
fn quant_cell_slice_mut(cells: &mut [i32], cell: usize, dim: usize) -> &mut [i32] {
    let start = cell * dim;
    &mut cells[start..start + dim]
}

fn region_of_cell(cell: usize) -> usize {
    let row = cell / BOARD_SIZE;
    let col = cell % BOARD_SIZE;
    let rr = (row / 5).min(2);
    let cc = (col / 5).min(2);
    rr * 3 + cc
}

fn region_cell_count(_region: usize) -> usize {
    25
}

fn dequantize_vec_i16(values: &[i16], scale: i32) -> Vec<f32> {
    let denom = scale as f32;
    values.iter().map(|&x| x as f32 / denom).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::GameResult;
    use crate::factored_codebook::PackedCodebookArtifact;
    use std::path::Path;

    const TOL: f32 = 1e-4;

    #[test]
    fn codebook_incremental_matches_full_refresh_smoke() {
        let weights = CodebookWeights::deterministic(16, 8);
        let moves = [
            112, 113, 97, 98, 127, 128, 111, 114, 96, 99, 126, 129, 82, 83, 84, 85, 100, 101, 115,
            116,
        ];
        let mut board = Board::new();
        let mut inc = IncrementalCodebookEval::new(&weights);
        inc.refresh(&board, &weights);

        assert_close(inc.value(&board, &weights), evaluate_full(&board, &weights));

        for &mv in &moves {
            if !board.is_empty(mv) {
                continue;
            }
            board.make_move(mv);
            inc.push_move(&board, mv, &weights);
            assert_close(inc.value(&board, &weights), evaluate_full(&board, &weights));
        }

        for _ in 0..moves.len() {
            board.undo_move();
            inc.pop_move(&weights);
            assert_close(inc.value(&board, &weights), evaluate_full(&board, &weights));
        }
    }

    #[test]
    fn quantized_codebook_matches_fake_dequantized_smoke() {
        let weights = CodebookWeights::deterministic(16, 8);
        let quantized = weights.quantize_i16_s32_s64();
        let dequantized = quantized.dequantized();
        let moves = [
            112, 113, 97, 98, 127, 128, 111, 114, 96, 99, 126, 129, 82, 83, 84, 85, 100, 101, 115,
            116,
        ];
        let mut board = Board::new();
        let mut inc = IncrementalQuantizedCodebookEval::new(&quantized);
        inc.refresh(&board, &quantized);

        assert_close(
            inc.value(&board, &quantized),
            evaluate_full_quantized(&board, &quantized),
        );
        assert_close(
            inc.value(&board, &quantized),
            evaluate_full(&board, &dequantized),
        );

        for &mv in &moves {
            if !board.is_empty(mv) {
                continue;
            }
            board.make_move(mv);
            inc.push_move(&board, mv, &quantized);
            assert_close(
                inc.value(&board, &quantized),
                evaluate_full_quantized(&board, &quantized),
            );
            assert_close(
                inc.value(&board, &quantized),
                evaluate_full(&board, &dequantized),
            );
        }

        for _ in 0..moves.len() {
            board.undo_move();
            inc.pop_move(&quantized);
            assert_close(
                inc.value(&board, &quantized),
                evaluate_full_quantized(&board, &quantized),
            );
            assert_close(
                inc.value(&board, &quantized),
                evaluate_full(&board, &dequantized),
            );
        }
    }

    #[test]
    fn factored_quantized_incremental_is_bit_exact_to_reconstructed_flat() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("models/gomoku_codebook_v1_swapclosed_factored.cbf");
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        let factored = PackedCodebookArtifact::parse(&bytes)
            .expect("valid CB-F1 artifact")
            .into_factored_quantized()
            .expect("factored payload");
        let flat = factored.reconstruct_flat();

        for &(old, new) in &[(0u16, 1u16), (1, 2), (585, 586), (4096, 4265)] {
            for component in 0..factored.dim() {
                assert_eq!(
                    QuantizedCodebookAccess::embedding_delta(&factored, old, new, component),
                    QuantizedCodebookAccess::embedding_delta(&flat, old, new, component),
                    "embedding delta {old}->{new}, component {component}"
                );
            }
        }

        let moves = [
            112, 113, 97, 98, 127, 128, 111, 114, 96, 99, 126, 129, 82, 83, 84, 85, 100, 101, 115,
            116,
        ];
        let mut board = Board::new();
        let mut factored_inc = IncrementalQuantizedCodebookEval::new_with_access(&factored, true);
        let mut flat_inc = IncrementalQuantizedCodebookEval::new_with_access(&flat, true);
        factored_inc.refresh_with_access(&board, &factored);
        flat_inc.refresh_with_access(&board, &flat);
        assert_factored_matches_flat(&board, &mut factored_inc, &factored, &mut flat_inc, &flat);

        for &mv in &moves {
            board.make_move(mv);
            factored_inc.push_move_profiled_with_access(&board, mv, &factored, false);
            flat_inc.push_move_profiled_with_access(&board, mv, &flat, false);
            assert_factored_matches_flat(
                &board,
                &mut factored_inc,
                &factored,
                &mut flat_inc,
                &flat,
            );
        }
        for _ in 0..moves.len() {
            board.undo_move();
            factored_inc.pop_move_profiled_with_access(&factored, false);
            flat_inc.pop_move_profiled_with_access(&flat, false);
            assert_factored_matches_flat(
                &board,
                &mut factored_inc,
                &factored,
                &mut flat_inc,
                &flat,
            );
        }
    }

    #[test]
    #[ignore = "CB-F1 benchmark: run in release mode with x86-64-v3"]
    fn cb_f1_reusable_full_refresh_microbenchmark() {
        fn timed_refresh<W: QuantizedCodebookAccess>(
            roots: &[Board],
            weights: &W,
            eval: &mut IncrementalQuantizedCodebookEval,
            repeats: usize,
        ) -> (u128, i64) {
            let start = std::time::Instant::now();
            let mut checksum = 0i64;
            for _ in 0..repeats {
                for board in roots {
                    eval.refresh_with_access(board, weights);
                    std::hint::black_box(&eval.cell_black);
                    std::hint::black_box(&eval.cell_white);
                    std::hint::black_box(&eval.features_black);
                    std::hint::black_box(&eval.features_white);
                    let cell_last = eval.cell_black.len() - 1;
                    let feature_last = eval.features_black.len() - 1;
                    for value in [
                        eval.cell_black[0],
                        eval.cell_black[cell_last / 2],
                        eval.cell_black[cell_last],
                        eval.cell_white[17],
                        eval.cell_white[cell_last / 2],
                        eval.features_black[0],
                        eval.features_black[feature_last],
                        eval.features_white[feature_last / 2],
                    ] {
                        checksum = checksum
                            .wrapping_mul(0x517c_c1b7_2722_0a95)
                            .wrapping_add(i64::from(value))
                            .wrapping_add(1);
                    }
                }
            }
            (start.elapsed().as_nanos(), std::hint::black_box(checksum))
        }

        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let artifact_path = manifest_dir.join("models/gomoku_codebook_v1_swapclosed_factored.cbf");
        let artifact_bytes = std::fs::read(&artifact_path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", artifact_path.display()));
        let artifact =
            PackedCodebookArtifact::parse(&artifact_bytes).expect("valid CB-F1 factored artifact");
        let source_sha256 = artifact
            .source_sha256()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let factored = artifact
            .into_factored_quantized()
            .expect("factored payload");
        let flat = factored.reconstruct_flat();

        let holdout_path = std::env::var_os("CB_F1_HOLDOUT_JSONL")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                manifest_dir.join(
                    "../figrid-dp-campaign/experiments/2026-07-25/dp_a1_fresh_holdout_64g.jsonl",
                )
            });
        let trace = std::fs::read_to_string(&holdout_path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", holdout_path.display()));
        let mut roots = Vec::with_capacity(1_022);
        'games: for line in trace.lines().filter(|line| !line.trim().is_empty()) {
            let game: Value = serde_json::from_str(line).expect("valid holdout JSONL row");
            let black = game["black_engine"].as_str().expect("black_engine");
            let white = game["white_engine"].as_str().expect("white_engine");
            let product_side = match (
                black.to_ascii_lowercase().contains("figrid"),
                white.to_ascii_lowercase().contains("figrid"),
            ) {
                (true, false) => Stone::Black,
                (false, true) => Stone::White,
                other => panic!("expected exactly one figrid side, got {other:?}"),
            };
            let mut board = Board::new();
            for move_json in game["moves"].as_array().expect("moves array") {
                let source = move_json["source"].as_str().unwrap_or("unknown");
                if source == "engine" && board.side_to_move == product_side {
                    roots.push(board.clone());
                    if roots.len() == 1_022 {
                        break 'games;
                    }
                }
                let x = move_json["x"].as_u64().expect("move x") as usize;
                let y = move_json["y"].as_u64().expect("move y") as usize;
                let mv = y * BOARD_SIZE + x;
                assert!(board.is_empty(mv), "holdout contains an occupied move");
                board.make_move(mv);
            }
        }
        assert_eq!(roots.len(), 1_022, "frozen product-root count drift");

        let classes = factored.classes();
        let mut same_class = 0u64;
        let mut mixed_class = 0u64;
        for board in &roots {
            for pattern_ids in &board.line_pattern_ids[..] {
                for swap in [false, true] {
                    let ids = if swap {
                        pattern_ids.map(swap_mapped_id)
                    } else {
                        *pattern_ids
                    };
                    let class = classes[ids[0] as usize];
                    if ids[1..]
                        .iter()
                        .all(|&pattern_id| classes[pattern_id as usize] == class)
                    {
                        same_class += 1;
                    } else {
                        mixed_class += 1;
                    }
                }
            }
        }

        let mut flat_eval = IncrementalQuantizedCodebookEval::new_with_access(&flat, true);
        let mut factored_eval = IncrementalQuantizedCodebookEval::new_with_access(&factored, true);
        const WARMUP_REPEATS: usize = 2;
        const REPEATS: usize = 24;
        let _ = timed_refresh(&roots, &flat, &mut flat_eval, WARMUP_REPEATS);
        let _ = timed_refresh(&roots, &factored, &mut factored_eval, WARMUP_REPEATS);
        let (a1_ns, a1_checksum) = timed_refresh(&roots, &flat, &mut flat_eval, REPEATS);
        let (b1_ns, b1_checksum) = timed_refresh(&roots, &factored, &mut factored_eval, REPEATS);
        let (b2_ns, b2_checksum) = timed_refresh(&roots, &factored, &mut factored_eval, REPEATS);
        let (a2_ns, a2_checksum) = timed_refresh(&roots, &flat, &mut flat_eval, REPEATS);
        assert_eq!(a1_checksum, b1_checksum, "A1/B1 checksum mismatch");
        assert_eq!(a1_checksum, b2_checksum, "A1/B2 checksum mismatch");
        assert_eq!(a1_checksum, a2_checksum, "A1/A2 checksum mismatch");

        let total_class_cells = same_class + mixed_class;
        let ratio = (b1_ns + b2_ns) as f64 / (a1_ns + a2_ns) as f64;
        println!(
            "{}",
            serde_json::json!({
                "format": "cb-f1-full-refresh-v1",
                "kind": "result",
                "arm_order": ["a1", "b1", "b2", "a2"],
                "holdout": holdout_path.to_string_lossy(),
                "factored_artifact": artifact_path.to_string_lossy(),
                "source_sha256": source_sha256,
                "roots": roots.len(),
                "warmup_repeats": WARMUP_REPEATS,
                "repeats": REPEATS,
                "a1_ns": a1_ns,
                "b1_ns": b1_ns,
                "b2_ns": b2_ns,
                "a2_ns": a2_ns,
                "ratio_b_over_a": ratio,
                "checksum": a1_checksum,
                "same_class": same_class,
                "mixed_class": mixed_class,
                "same_class_ratio": same_class as f64 / total_class_cells as f64,
            })
        );
    }

    #[test]
    #[ignore = "CB-F1 release gate: run explicitly with --release --ignored"]
    fn quantized_factored_100k_mixed_make_undo_full_rebuild_equality() {
        const OPERATIONS: usize = 100_000;
        const FULL_REBUILD_PERIOD: usize = 97;

        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("models/gomoku_codebook_v1_swapclosed_factored.cbf");
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        let factored = PackedCodebookArtifact::parse(&bytes)
            .expect("valid CB-F1 artifact")
            .into_factored_quantized()
            .expect("factored payload");
        let flat = factored.reconstruct_flat();

        let mut board = Board::new();
        let mut factored_inc = IncrementalQuantizedCodebookEval::new_with_access(&factored, true);
        let mut flat_inc = IncrementalQuantizedCodebookEval::new_with_access(&flat, true);
        factored_inc.refresh_with_access(&board, &factored);
        flat_inc.refresh_with_access(&board, &flat);
        let mut rng = TestRng::new(0xCBD1_2026_0725_0001);
        let mut makes = 0usize;
        let mut undos = 0usize;
        let mut materializations = 0usize;
        let mut full_rebuilds = 0usize;

        for operation in 1..=OPERATIONS {
            let should_undo =
                !board.history.is_empty() && (board.move_count >= 180 || rng.usize(4) == 0);
            if should_undo {
                board.undo_move();
                factored_inc.pop_move_profiled_with_access(&factored, false);
                flat_inc.pop_move_profiled_with_access(&flat, false);
                undos += 1;
            } else {
                let moves = board.legal_moves();
                let mv = moves[rng.usize(moves.len())];
                board.make_move(mv);
                factored_inc.push_move_profiled_with_access(&board, mv, &factored, false);
                flat_inc.push_move_profiled_with_access(&board, mv, &flat, false);
                makes += 1;
            }

            {
                let factored_state = factored_inc.directional_delta.as_ref().unwrap();
                let flat_state = flat_inc.directional_delta.as_ref().unwrap();
                assert_eq!(
                    factored_state.logical_pattern_ids(),
                    board.line_pattern_ids.as_ref(),
                    "factored logical Pattern4 IDs at operation {operation}"
                );
                assert_eq!(
                    flat_state.logical_pattern_ids(),
                    board.line_pattern_ids.as_ref(),
                    "flat logical Pattern4 IDs at operation {operation}"
                );
            }

            let materialize = rng.usize(8) == 0
                || operation % FULL_REBUILD_PERIOD == 0
                || operation == OPERATIONS;
            if materialize {
                {
                    let factored_state = factored_inc.directional_delta.as_ref().unwrap();
                    let flat_state = flat_inc.directional_delta.as_ref().unwrap();
                    assert_eq!(factored_state.raw_black, flat_state.raw_black);
                    assert_eq!(factored_state.raw_white, flat_state.raw_white);
                }
                assert_factored_matches_flat(
                    &board,
                    &mut factored_inc,
                    &factored,
                    &mut flat_inc,
                    &flat,
                );
                materializations += 1;
            }

            if operation % FULL_REBUILD_PERIOD == 0 || operation == OPERATIONS {
                let mut factored_full =
                    IncrementalQuantizedCodebookEval::new_with_access(&factored, true);
                let mut flat_full = IncrementalQuantizedCodebookEval::new_with_access(&flat, true);
                factored_full.refresh_with_access(&board, &factored);
                flat_full.refresh_with_access(&board, &flat);

                assert_factored_matches_flat(
                    &board,
                    &mut factored_full,
                    &factored,
                    &mut flat_full,
                    &flat,
                );
                assert_eq!(factored_inc.cell_black, factored_full.cell_black);
                assert_eq!(factored_inc.cell_white, factored_full.cell_white);
                assert_eq!(factored_inc.features_black, factored_full.features_black);
                assert_eq!(factored_inc.features_white, factored_full.features_white);
                let factored_state = factored_inc.directional_delta.as_ref().unwrap();
                let factored_full_state = factored_full.directional_delta.as_ref().unwrap();
                assert_eq!(factored_state.raw_black, factored_full_state.raw_black);
                assert_eq!(factored_state.raw_white, factored_full_state.raw_white);
                assert_eq!(
                    factored_inc
                        .value_profiled_with_access(&board, &factored, false)
                        .0
                        .to_bits(),
                    factored_full
                        .value_profiled_with_access(&board, &factored, false)
                        .0
                        .to_bits()
                );
                full_rebuilds += 1;
            }
        }

        while !board.history.is_empty() {
            board.undo_move();
            factored_inc.pop_move_profiled_with_access(&factored, false);
            flat_inc.pop_move_profiled_with_access(&flat, false);
        }
        let mut factored_root = IncrementalQuantizedCodebookEval::new_with_access(&factored, true);
        let mut flat_root = IncrementalQuantizedCodebookEval::new_with_access(&flat, true);
        factored_root.refresh_with_access(&board, &factored);
        flat_root.refresh_with_access(&board, &flat);
        assert_factored_matches_flat(&board, &mut factored_inc, &factored, &mut flat_inc, &flat);
        assert_eq!(factored_inc.cell_black, factored_root.cell_black);
        assert_eq!(factored_inc.cell_white, factored_root.cell_white);
        assert_eq!(factored_inc.features_black, factored_root.features_black);
        assert_eq!(factored_inc.features_white, factored_root.features_white);
        assert_eq!(
            factored_inc.directional_delta.as_ref().unwrap().raw_black,
            factored_root.directional_delta.as_ref().unwrap().raw_black
        );
        assert_eq!(
            factored_inc.directional_delta.as_ref().unwrap().raw_white,
            factored_root.directional_delta.as_ref().unwrap().raw_white
        );
        assert_factored_matches_flat(&board, &mut factored_root, &factored, &mut flat_root, &flat);

        eprintln!(
            "CB-F1 operations={OPERATIONS} makes={makes} undos={undos} \
             materializations={materializations} full_rebuilds={full_rebuilds}"
        );
    }

    #[test]
    fn quantized_directional_delta_matches_legacy_and_full_smoke() {
        let weights = CodebookWeights::deterministic(16, 8).quantize_i16_s32_s64();
        let moves = [
            112, 113, 97, 98, 127, 128, 111, 114, 96, 99, 126, 129, 82, 83, 84, 85, 100, 101, 115,
            116,
        ];
        let mut board = Board::new();
        let mut legacy = IncrementalQuantizedCodebookEval::new(&weights);
        let mut delta =
            IncrementalQuantizedCodebookEval::new_with_directional_delta(&weights, true);
        legacy.refresh(&board, &weights);
        delta.refresh(&board, &weights);
        assert!(delta.directional_delta_enabled());
        assert_quantized_directional_state(&board, &mut delta, &weights);

        for (ply, &mv) in moves.iter().enumerate() {
            if !board.is_empty(mv) {
                continue;
            }
            board.make_move(mv);
            legacy.push_move(&board, mv, &weights);
            delta.push_move(&board, mv, &weights);
            assert_eq!(
                delta
                    .directional_delta
                    .as_ref()
                    .unwrap()
                    .logical_pattern_ids(),
                board.line_pattern_ids.as_ref()
            );
            if ply % 3 == 2 {
                let delta_value = delta.value(&board, &weights);
                assert_eq!(
                    delta_value.to_bits(),
                    legacy.value(&board, &weights).to_bits()
                );
                assert_quantized_directional_state(&board, &mut delta, &weights);
            }
        }
        assert_quantized_directional_state(&board, &mut delta, &weights);

        for ply in (0..moves.len()).rev() {
            if board.history.is_empty() {
                break;
            }
            board.undo_move();
            legacy.pop_move(&weights);
            delta.pop_move(&weights);
            assert_eq!(
                delta
                    .directional_delta
                    .as_ref()
                    .unwrap()
                    .logical_pattern_ids(),
                board.line_pattern_ids.as_ref()
            );
            if ply % 4 == 0 {
                let delta_value = delta.value(&board, &weights);
                assert_eq!(
                    delta_value.to_bits(),
                    legacy.value(&board, &weights).to_bits()
                );
                assert_quantized_directional_state(&board, &mut delta, &weights);
            }
        }
        assert_quantized_directional_state(&board, &mut delta, &weights);
    }

    #[test]
    #[ignore = "CB-D1 TokenDelta release gate: run explicitly with --release --ignored"]
    fn quantized_directional_delta_100k_mixed_make_undo_full_rebuild_equality() {
        const OPERATIONS: usize = 100_000;
        const FULL_REBUILD_PERIOD: usize = 97;

        let weights = CodebookWeights::deterministic(16, 8).quantize_i16_s32_s64();
        let mut board = Board::new();
        let mut legacy = IncrementalQuantizedCodebookEval::new(&weights);
        let mut delta =
            IncrementalQuantizedCodebookEval::new_with_directional_delta(&weights, true);
        legacy.refresh(&board, &weights);
        delta.refresh(&board, &weights);
        let mut rng = TestRng::new(0xCBD1_2026_0725_0001);
        let mut makes = 0usize;
        let mut undos = 0usize;
        let mut materializations = 0usize;
        let mut direction_deltas = 0usize;

        for operation in 1..=OPERATIONS {
            let should_undo =
                !board.history.is_empty() && (board.move_count >= 180 || rng.usize(4) == 0);
            if should_undo {
                board.undo_move();
                legacy.pop_move(&weights);
                delta.pop_move(&weights);
                undos += 1;
            } else {
                let moves = board.legal_moves();
                let mv = moves[rng.usize(moves.len())];
                board.make_move(mv);
                legacy.push_move(&board, mv, &weights);
                delta.push_move(&board, mv, &weights);
                direction_deltas += delta.last_direction_deltas();
                makes += 1;
            }

            let state = delta.directional_delta.as_ref().unwrap();
            assert_eq!(
                state.logical_pattern_ids(),
                board.line_pattern_ids.as_ref(),
                "logical Pattern4 IDs at operation {operation}"
            );

            if rng.usize(8) == 0 || operation % FULL_REBUILD_PERIOD == 0 || operation == OPERATIONS
            {
                let delta_value = delta.value(&board, &weights);
                let legacy_value = legacy.value(&board, &weights);
                assert_eq!(delta_value.to_bits(), legacy_value.to_bits());
                materializations += 1;
            }
            if operation % FULL_REBUILD_PERIOD == 0 || operation == OPERATIONS {
                assert_quantized_directional_state(&board, &mut delta, &weights);
            }
        }

        while !board.history.is_empty() {
            board.undo_move();
            legacy.pop_move(&weights);
            delta.pop_move(&weights);
        }
        assert_quantized_directional_state(&board, &mut delta, &weights);
        assert_eq!(
            delta.value(&board, &weights).to_bits(),
            legacy.value(&board, &weights).to_bits()
        );
        eprintln!(
            "CB-D1/TokenDelta operations={OPERATIONS} makes={makes} undos={undos} \
             materializations={materializations} direction_deltas={direction_deltas}"
        );
    }

    #[test]
    #[ignore = "RQ542 gate: run explicitly with --release --features codebook-eval"]
    fn codebook_incremental_100k_transition_gate() {
        let weights = CodebookWeights::deterministic(16, 8);
        let mut rng = TestRng::new(0x5420_0001);
        let mut board = Board::new();
        let mut inc = IncrementalCodebookEval::new(&weights);
        inc.refresh(&board, &weights);

        let mut transitions = 0usize;
        let mut mismatch = 0usize;
        let mut undo_fail = 0usize;
        let mut dirty_counts = Vec::with_capacity(100_000);

        while transitions < 100_000 {
            if board.move_count >= 160
                || board.game_result() != GameResult::Ongoing
                || board.candidate_moves().is_empty()
            {
                while board.move_count > 0 {
                    board.undo_move();
                    inc.pop_move(&weights);
                    if !close(inc.value(&board, &weights), evaluate_full(&board, &weights)) {
                        undo_fail += 1;
                    }
                }
                board = Board::new();
                inc.refresh(&board, &weights);
            }

            let moves = board.candidate_moves();
            let mv = moves[rng.usize(moves.len())];
            board.make_move(mv);
            inc.push_move(&board, mv, &weights);
            dirty_counts.push(inc.last_dirty_cells());
            transitions += 1;

            if !close(inc.value(&board, &weights), evaluate_full(&board, &weights)) {
                mismatch += 1;
            }
        }

        while board.move_count > 0 {
            board.undo_move();
            inc.pop_move(&weights);
            if !close(inc.value(&board, &weights), evaluate_full(&board, &weights)) {
                undo_fail += 1;
            }
        }

        dirty_counts.sort_unstable();
        let avg_dirty = dirty_counts.iter().sum::<usize>() as f32 / dirty_counts.len() as f32;
        let p95_dirty = dirty_counts[dirty_counts.len() * 95 / 100] as f32;
        let avg_ratio = avg_dirty / NUM_CELLS as f32;
        let p95_ratio = p95_dirty / NUM_CELLS as f32;
        eprintln!(
            "RQ542 transitions={transitions} mismatch={mismatch} undo_fail={undo_fail} \
             avg_dirty_ratio={avg_ratio:.6} p95_dirty_ratio={p95_ratio:.6}"
        );

        assert_eq!(mismatch, 0, "full-vs-increment mismatch count");
        assert_eq!(undo_fail, 0, "undo roundtrip failure count");
        assert!(
            (0.12..=0.16).contains(&avg_ratio),
            "avg dirty ratio {avg_ratio:.6} is outside random-play sanity range"
        );
        assert!(
            (0.17..=0.20).contains(&p95_ratio),
            "p95 dirty ratio {p95_ratio:.6} is outside random-play sanity range"
        );
    }

    #[test]
    #[ignore = "RQ542 gate: set FIGRID_RQ542_GAMES_JSONL to rq535_accept_off_100g_games.jsonl"]
    fn codebook_incremental_rq535_trace_gate() {
        let path = std::env::var("FIGRID_RQ542_GAMES_JSONL")
            .expect("set FIGRID_RQ542_GAMES_JSONL to rq535 game JSONL");
        let games = load_trace_games(&path);
        assert!(!games.is_empty(), "no games loaded from {path}");

        let weights = CodebookWeights::deterministic(16, 8);
        let mut transitions = 0usize;
        let mut mismatch = 0usize;
        let mut undo_fail = 0usize;
        let mut passes = 0usize;
        let mut dirty_counts = Vec::with_capacity(100_000);

        while transitions < 100_000 {
            passes += 1;
            for game in &games {
                if transitions >= 100_000 {
                    break;
                }
                let mut board = Board::new();
                let mut inc = IncrementalCodebookEval::new(&weights);
                inc.refresh(&board, &weights);
                let mut played = 0usize;
                for &mv in game {
                    if transitions >= 100_000 {
                        break;
                    }
                    assert!(board.is_empty(mv), "illegal trace move {mv} in {path}");
                    board.make_move(mv);
                    inc.push_move(&board, mv, &weights);
                    dirty_counts.push(inc.last_dirty_cells());
                    transitions += 1;
                    played += 1;

                    if !close(inc.value(&board, &weights), evaluate_full(&board, &weights)) {
                        mismatch += 1;
                    }
                }
                for _ in 0..played {
                    board.undo_move();
                    inc.pop_move(&weights);
                    if !close(inc.value(&board, &weights), evaluate_full(&board, &weights)) {
                        undo_fail += 1;
                    }
                }
            }
        }

        dirty_counts.sort_unstable();
        let avg_dirty = dirty_counts.iter().sum::<usize>() as f32 / dirty_counts.len() as f32;
        let p95_dirty = dirty_counts[dirty_counts.len() * 95 / 100] as f32;
        let avg_ratio = avg_dirty / NUM_CELLS as f32;
        let p95_ratio = p95_dirty / NUM_CELLS as f32;
        eprintln!(
            "RQ542 trace_gate path={path} passes={passes} transitions={transitions} \
             mismatch={mismatch} undo_fail={undo_fail} avg_dirty_ratio={avg_ratio:.6} \
             p95_dirty_ratio={p95_ratio:.6}"
        );

        assert_eq!(mismatch, 0, "full-vs-increment mismatch count");
        assert_eq!(undo_fail, 0, "undo roundtrip failure count");
        assert!(
            (0.12..=0.16).contains(&avg_ratio),
            "avg dirty ratio {avg_ratio:.6} drifted away from Board pattern-cache dirty set"
        );
        assert!(
            (0.17..=0.20).contains(&p95_ratio),
            "p95 dirty ratio {p95_ratio:.6} drifted away from Board pattern-cache dirty set"
        );
    }

    fn assert_close(a: f32, b: f32) {
        assert!(
            close(a, b),
            "left={a:.9} right={b:.9} diff={:.9}",
            (a - b).abs()
        );
    }

    fn assert_factored_matches_flat(
        board: &Board,
        factored_inc: &mut IncrementalQuantizedCodebookEval,
        factored: &FactoredQuantizedCodebookWeights,
        flat_inc: &mut IncrementalQuantizedCodebookEval,
        flat: &QuantizedCodebookWeights,
    ) {
        let factored_value = factored_inc
            .value_profiled_with_access(board, factored, false)
            .0;
        let flat_value = flat_inc.value_profiled_with_access(board, flat, false).0;
        assert_eq!(factored_value.to_bits(), flat_value.to_bits());
        assert_eq!(factored_inc.cell_black, flat_inc.cell_black);
        assert_eq!(factored_inc.cell_white, flat_inc.cell_white);
        assert_eq!(factored_inc.features_black, flat_inc.features_black);
        assert_eq!(factored_inc.features_white, flat_inc.features_white);
        for perspective in [Stone::Black, Stone::White] {
            assert_eq!(
                factored_inc
                    .explicit_orbit48_with_access(factored, perspective)
                    .unwrap(),
                flat_inc
                    .explicit_orbit48_with_access(flat, perspective)
                    .unwrap()
            );
        }
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= TOL
    }

    fn assert_quantized_directional_state(
        board: &Board,
        delta: &mut IncrementalQuantizedCodebookEval,
        weights: &QuantizedCodebookWeights,
    ) {
        let actual_value = delta.value(board, weights);
        let mut full = {
            let mut full = IncrementalQuantizedCodebookEval::new(weights);
            full.refresh(board, weights);
            full
        };
        assert_eq!(
            actual_value.to_bits(),
            quant_value_from_features(
                match board.side_to_move {
                    Stone::Black => &full.features_black,
                    Stone::White => &full.features_white,
                },
                weights
            )
            .to_bits()
        );
        assert_eq!(delta.cell_black, full.cell_black);
        assert_eq!(delta.cell_white, full.cell_white);
        assert_eq!(delta.features_black, full.features_black);
        assert_eq!(delta.features_white, full.features_white);
        for perspective in [Stone::Black, Stone::White] {
            assert_eq!(
                delta.explicit_orbit48(weights, perspective).unwrap(),
                full.explicit_orbit48(weights, perspective).unwrap()
            );
        }

        let state = delta.directional_delta.as_ref().unwrap();
        assert_eq!(state.logical_pattern_ids(), board.line_pattern_ids.as_ref());
        let mut expected = vec![0i32; weights.dim];
        for cell in 0..NUM_CELLS {
            compute_cell_quantized_raw_from_pattern_ids(
                &board.line_pattern_ids[cell],
                weights,
                Stone::Black,
                &mut expected,
            );
            assert_eq!(
                quant_cell_slice(&state.raw_black, cell, weights.dim),
                expected.as_slice(),
                "Black raw cell {cell}"
            );
            compute_cell_quantized_raw_from_pattern_ids(
                &board.line_pattern_ids[cell],
                weights,
                Stone::White,
                &mut expected,
            );
            assert_eq!(
                quant_cell_slice(&state.raw_white, cell, weights.dim),
                expected.as_slice(),
                "White raw cell {cell}"
            );
        }
    }

    struct TestRng(u64);

    impl TestRng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn usize(&mut self, n: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 as usize) % n
        }
    }

    fn load_trace_games(path: &str) -> Vec<Vec<Move>> {
        let text =
            std::fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
        let mut games = Vec::new();
        for (line_no, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("failed to parse {path}:{}: {e}", line_no + 1));
            let moves = value
                .get("moves")
                .and_then(Value::as_array)
                .unwrap_or_else(|| panic!("missing moves array in {path}:{}", line_no + 1));
            let mut out = Vec::with_capacity(moves.len());
            for mv in moves {
                let x = mv
                    .get("x")
                    .and_then(Value::as_u64)
                    .unwrap_or_else(|| panic!("missing move x in {path}:{}", line_no + 1))
                    as usize;
                let y = mv
                    .get("y")
                    .and_then(Value::as_u64)
                    .unwrap_or_else(|| panic!("missing move y in {path}:{}", line_no + 1))
                    as usize;
                assert!(
                    x < BOARD_SIZE && y < BOARD_SIZE,
                    "out-of-board move in {path}"
                );
                out.push(y * BOARD_SIZE + x);
            }
            games.push(out);
        }
        games
    }
}
