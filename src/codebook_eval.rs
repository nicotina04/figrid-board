//! Experimental no-message-passing codebook evaluator.
//!
//! This module mirrors the normal `IncrementalEval` shape for the RQ542
//! correctness gate. It is deliberately feature-gated and is not wired into
//! search by default.

use crate::board::{BOARD_SIZE, Board, Move, NUM_CELLS, Stone};
use crate::pattern_table::{PATTERN_NUM_IDS, swap_mapped_id};
pub use crate::search::EvalStateStepProfile;
use crate::token_delta::{ReversibleTokenJournal, TokenDelta, TokenDeltaSink};
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
        let mut state = 0xC0DE_B00C_F00D_0542u64;
        let embeddings = deterministic_vec(&mut state, PATTERN_NUM_IDS * dim, 0.02);
        let head = deterministic_vec(&mut state, REGIONS * dim, 0.02);
        let factors = deterministic_vec(&mut state, REGIONS * dim * fm_rank, 0.02);
        Self {
            dim,
            fm_rank,
            embeddings,
            head,
            factors,
            bias: 0.01,
        }
    }

    pub fn from_json_bytes(data: &[u8]) -> Result<Self, String> {
        let root: Value = serde_json::from_slice(data)
            .map_err(|e| format!("failed to parse codebook json: {e}"))?;
        Self::from_json_value(&root)
    }

    pub fn from_json_value(root: &Value) -> Result<Self, String> {
        let format = json_str(root, "format")?;
        if format != "noru-relation-fusion-eval-v1" && format != "noru-pattern4-codebook-eval-v1" {
            return Err(format!("unsupported codebook format: {format}"));
        }

        let model = json_str(root, "model")?;
        if model != "codebook-region-fm" && model != "region-codebook-fm" {
            return Err(format!(
                "unsupported codebook model: {model}; expected codebook-region-fm"
            ));
        }

        let metadata = root.get("metadata");
        let dim = metadata
            .and_then(|m| json_usize_opt(m, "embedding_dim"))
            .or_else(|| json_usize_opt(root, "embedding_dim"))
            .ok_or_else(|| "missing embedding_dim".to_string())?;
        let fm_rank = metadata
            .and_then(|m| json_usize_opt(m, "fm_rank"))
            .or_else(|| json_usize_opt(root, "fm_rank"))
            .ok_or_else(|| "missing fm_rank".to_string())?;
        let regions = metadata
            .and_then(|m| json_usize_opt(m, "regions"))
            .or_else(|| json_usize_opt(root, "regions"))
            .unwrap_or(REGIONS);
        if regions != REGIONS {
            return Err(format!("unsupported region count: {regions}"));
        }

        let weights = root
            .get("weights")
            .ok_or_else(|| "missing weights object".to_string())?;
        let embeddings = json_f32_array(weights, "embeddings")?;
        let head = json_f32_array(weights, "head")?;
        let factors = json_f32_array(weights, "factors")?;
        let bias = json_f32_opt(weights, "bias").ok_or_else(|| "missing bias".to_string())?;

        let expected_embeddings = PATTERN_NUM_IDS * dim;
        let expected_head = REGIONS * dim;
        let expected_factors = expected_head * fm_rank;
        if embeddings.len() != expected_embeddings {
            return Err(format!(
                "embedding length mismatch: got {}, expected {expected_embeddings}",
                embeddings.len()
            ));
        }
        if head.len() != expected_head {
            return Err(format!(
                "head length mismatch: got {}, expected {expected_head}",
                head.len()
            ));
        }
        if factors.len() != expected_factors {
            return Err(format!(
                "factor length mismatch: got {}, expected {expected_factors}",
                factors.len()
            ));
        }

        Ok(Self {
            dim,
            fm_rank,
            embeddings,
            head,
            factors,
            bias,
        })
    }

    #[inline]
    pub fn feature_len(&self) -> usize {
        REGIONS * self.dim
    }

    pub fn quantize_i16_s32_s64(&self) -> QuantizedCodebookWeights {
        self.validate();
        QuantizedCodebookWeights {
            dim: self.dim,
            fm_rank: self.fm_rank,
            embedding_scale: QUANT_EMBED_SCALE,
            head_scale: QUANT_HEAD_SCALE,
            factor_scale: QUANT_FACTOR_SCALE,
            embeddings: quantize_vec_i16(&self.embeddings, QUANT_EMBED_SCALE),
            head: quantize_vec_i16(&self.head, QUANT_HEAD_SCALE),
            factors: quantize_vec_i16(&self.factors, QUANT_FACTOR_SCALE),
            bias: self.bias,
        }
    }

    fn validate(&self) {
        debug_assert_eq!(self.embeddings.len(), PATTERN_NUM_IDS * self.dim);
        debug_assert_eq!(self.head.len(), self.feature_len());
        debug_assert_eq!(self.factors.len(), self.feature_len() * self.fm_rank);
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
    old_pattern_ids: [[u16; 4]; MAX_DIRTY_CELLS],
    black: Vec<i32>,
    white: Vec<i32>,
}

struct QuantDirectionalDeltaState {
    raw_black: Vec<i32>,
    raw_white: Vec<i32>,
    logical_pattern_ids: Option<Box<[[u16; 4]; NUM_CELLS]>>,
    token_journal: Option<ReversibleTokenJournal<u16, 4, MAX_DIRECTION_DELTAS>>,
}

impl QuantDirectionalDeltaState {
    fn new(dim: usize, token_journal: bool) -> Self {
        Self {
            raw_black: vec![0; NUM_CELLS * dim],
            raw_white: vec![0; NUM_CELLS * dim],
            logical_pattern_ids: (!token_journal).then(|| Box::new([[0u16; 4]; NUM_CELLS])),
            token_journal: token_journal.then(|| ReversibleTokenJournal::new(NUM_CELLS, NUM_CELLS)),
        }
    }

    #[cfg(test)]
    fn logical_pattern_ids(&self) -> &[[u16; 4]] {
        self.token_journal.as_ref().map_or_else(
            || {
                let logical: &[[u16; 4]] = self
                    .logical_pattern_ids
                    .as_deref()
                    .expect("direct directional token mirror");
                logical
            },
            |journal| journal.logical_tokens(),
        )
    }
}

struct QuantizedCodebookTokenSink<'a> {
    weights: &'a QuantizedCodebookWeights,
    raw_black: &'a mut [i32],
    raw_white: &'a mut [i32],
    cell_black: &'a mut [i32],
    cell_white: &'a mut [i32],
    features_black: &'a mut [i32],
    features_white: &'a mut [i32],
    profile: &'a mut EvalStateStepProfile,
    profile_enabled: bool,
    restore: bool,
    active_site: Option<usize>,
    numeric_start: Option<std::time::Instant>,
}

impl<'a> QuantizedCodebookTokenSink<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        weights: &'a QuantizedCodebookWeights,
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
            active_site: None,
            numeric_start: None,
        }
    }
}

impl TokenDeltaSink<u16> for QuantizedCodebookTokenSink<'_> {
    #[inline]
    fn begin_site(&mut self, site: u16) {
        debug_assert!(self.active_site.is_none());
        self.active_site = Some(site as usize);
        self.numeric_start = EvalStateStepProfile::start(self.profile_enabled);
    }

    #[inline]
    fn apply(&mut self, delta: TokenDelta<u16>) {
        let cell = self.active_site.expect("TokenDelta site not begun");
        debug_assert_eq!(cell, delta.site as usize);
        debug_assert!((delta.lane as usize) < 4);
        apply_quantized_token_delta_to_raw(
            delta.old,
            delta.new,
            self.weights,
            Stone::Black,
            quant_cell_slice_mut(self.raw_black, cell, self.weights.dim),
        );
        apply_quantized_token_delta_to_raw(
            delta.old,
            delta.new,
            self.weights,
            Stone::White,
            quant_cell_slice_mut(self.raw_white, cell, self.weights.dim),
        );
    }

    #[inline]
    fn end_site(&mut self, site: u16) {
        let cell = self.active_site.take().expect("TokenDelta site not begun");
        debug_assert_eq!(cell, site as usize);
        let numeric_start = self.numeric_start.take();
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
            self.weights.dim,
        );
        refresh_quantized_cell_activation(
            self.raw_white,
            self.cell_white,
            self.features_white,
            cell,
            self.weights.dim,
        );
        self.profile.add_aggregate(aggregate_start);
    }
}

impl QuantUndoRecord {
    fn new(dim: usize, store_cell_backups: bool) -> Self {
        let backup_len = if store_cell_backups {
            MAX_DIRTY_CELLS * dim
        } else {
            0
        };
        Self {
            len: 0,
            materialized: false,
            cells: [0; MAX_DIRTY_CELLS],
            pattern_ids: [[0u16; 4]; MAX_DIRTY_CELLS],
            old_pattern_ids: [[0u16; 4]; MAX_DIRTY_CELLS],
            black: vec![0; backup_len],
            white: vec![0; backup_len],
        }
    }

    fn clear(&mut self) {
        self.len = 0;
        self.materialized = false;
    }
}

impl IncrementalQuantizedCodebookEval {
    pub fn new(weights: &QuantizedCodebookWeights) -> Self {
        Self::new_with_directional_delta(weights, false)
    }

    pub fn new_with_directional_delta(
        weights: &QuantizedCodebookWeights,
        directional_delta: bool,
    ) -> Self {
        Self::new_with_directional_delta_and_token_journal(weights, directional_delta, false)
    }

    pub(crate) fn new_with_directional_delta_and_token_journal(
        weights: &QuantizedCodebookWeights,
        directional_delta: bool,
        token_journal: bool,
    ) -> Self {
        weights.validate();
        assert!(
            directional_delta || !token_journal,
            "TokenDelta journal requires directional delta"
        );
        Self {
            cell_black: vec![0; NUM_CELLS * weights.dim],
            cell_white: vec![0; NUM_CELLS * weights.dim],
            features_black: vec![0; weights.feature_len()],
            features_white: vec![0; weights.feature_len()],
            stack: if token_journal {
                Vec::new()
            } else {
                (0..NUM_CELLS)
                    .map(|_| QuantUndoRecord::new(weights.dim, !directional_delta))
                    .collect()
            },
            stack_len: 0,
            last_dirty_cells: 0,
            last_direction_deltas: 0,
            directional_delta: directional_delta
                .then(|| QuantDirectionalDeltaState::new(weights.dim, token_journal)),
        }
    }

    pub fn directional_delta_enabled(&self) -> bool {
        self.directional_delta.is_some()
    }

    pub(crate) fn token_delta_journal_enabled(&self) -> bool {
        self.directional_delta
            .as_ref()
            .is_some_and(|state| state.token_journal.is_some())
    }

    pub fn refresh(&mut self, board: &Board, weights: &QuantizedCodebookWeights) {
        weights.validate();
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
                quant_cell_slice_mut(&mut self.cell_black, cell, weights.dim),
            );
            add_quant_cell_to_features(
                &self.cell_black,
                &mut self.features_black,
                cell,
                weights.dim,
                1,
            );

            compute_cell_quantized(
                board,
                weights,
                cell,
                Stone::White,
                quant_cell_slice_mut(&mut self.cell_white, cell, weights.dim),
            );
            add_quant_cell_to_features(
                &self.cell_white,
                &mut self.features_white,
                cell,
                weights.dim,
                1,
            );
        }

        if let Some(state) = self.directional_delta.as_mut() {
            if let Some(logical_pattern_ids) = state.logical_pattern_ids.as_mut() {
                logical_pattern_ids.copy_from_slice(board.line_pattern_ids.as_ref());
            }
            if let Some(journal) = state.token_journal.as_mut() {
                journal.reset(board.line_pattern_ids.as_ref());
            }
            for cell in 0..NUM_CELLS {
                compute_cell_quantized_raw_from_pattern_ids(
                    &board.line_pattern_ids[cell],
                    weights,
                    Stone::Black,
                    quant_cell_slice_mut(&mut state.raw_black, cell, weights.dim),
                );
                compute_cell_quantized_raw_from_pattern_ids(
                    &board.line_pattern_ids[cell],
                    weights,
                    Stone::White,
                    quant_cell_slice_mut(&mut state.raw_white, cell, weights.dim),
                );
            }
        }

        self.stack_len = 0;
        self.last_dirty_cells = 0;
        self.last_direction_deltas = 0;
    }

    pub fn push_move(&mut self, board: &Board, mv: Move, weights: &QuantizedCodebookWeights) {
        let _ = self.push_move_profiled(board, mv, weights, false);
    }

    pub fn push_move_profiled(
        &mut self,
        board: &Board,
        mv: Move,
        weights: &QuantizedCodebookWeights,
        _profile_enabled: bool,
    ) -> EvalStateStepProfile {
        weights.validate();
        if self.token_delta_journal_enabled() {
            return self.push_move_token_delta(board, mv, weights, _profile_enabled);
        }
        if self.directional_delta.is_some() {
            return self.push_move_directional(board, mv, weights, _profile_enabled);
        }
        let mut profile = EvalStateStepProfile {
            push_calls: 1,
            ..EvalStateStepProfile::default()
        };
        let start = EvalStateStepProfile::start(_profile_enabled);
        let dirty = dirty_cells_for_move(mv);
        profile.add_dirty_list(start);
        debug_assert!(dirty.len() <= MAX_DIRTY_CELLS);
        debug_assert!(
            self.stack_len < self.stack.len(),
            "quantized codebook undo stack overflow"
        );
        let undo = &mut self.stack[self.stack_len];
        undo.clear();

        let start = EvalStateStepProfile::start(_profile_enabled);
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

    fn push_move_token_delta(
        &mut self,
        board: &Board,
        mv: Move,
        weights: &QuantizedCodebookWeights,
        profile_enabled: bool,
    ) -> EvalStateStepProfile {
        weights.validate();
        let mut profile = EvalStateStepProfile {
            push_calls: 1,
            ..EvalStateStepProfile::default()
        };
        let start = EvalStateStepProfile::start(profile_enabled);
        let dirty = dirty_cells_for_move(mv);
        profile.add_dirty_list(start);
        debug_assert!(dirty.len() <= MAX_DIRTY_CELLS);
        debug_assert!(
            self.stack_len < NUM_CELLS,
            "TokenDelta journal depth overflow"
        );

        let start = EvalStateStepProfile::start(profile_enabled);
        let direction_deltas = self
            .directional_delta
            .as_mut()
            .and_then(|state| state.token_journal.as_mut())
            .expect("TokenDelta journal enabled")
            .push_after(board.line_pattern_ids.as_ref(), &dirty);
        profile.add_frame_write(start);

        self.last_dirty_cells = dirty.len();
        self.last_direction_deltas = direction_deltas;
        self.stack_len += 1;
        debug_assert_eq!(
            self.directional_delta
                .as_ref()
                .and_then(|state| state.token_journal.as_ref())
                .expect("TokenDelta journal enabled")
                .depth(),
            self.stack_len
        );
        profile
    }

    fn push_move_directional(
        &mut self,
        board: &Board,
        mv: Move,
        weights: &QuantizedCodebookWeights,
        profile_enabled: bool,
    ) -> EvalStateStepProfile {
        weights.validate();
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
        let state = self
            .directional_delta
            .as_mut()
            .expect("directional delta state enabled");
        let logical_pattern_ids = state
            .logical_pattern_ids
            .as_mut()
            .expect("direct directional token mirror");
        let start = EvalStateStepProfile::start(profile_enabled);
        let mut direction_deltas = 0usize;
        for cell in dirty.iter().copied() {
            let old_ids = logical_pattern_ids[cell];
            let new_ids = board.line_pattern_ids[cell];
            let changed = old_ids
                .iter()
                .zip(new_ids.iter())
                .filter(|(old, new)| old != new)
                .count();
            if changed == 0 {
                continue;
            }
            let undo_idx = undo.len;
            undo.cells[undo_idx] = cell;
            undo.old_pattern_ids[undo_idx] = old_ids;
            undo.pattern_ids[undo_idx] = new_ids;
            undo.len += 1;
            direction_deltas += changed;
            logical_pattern_ids[cell] = new_ids;
        }
        profile.add_frame_write(start);

        self.last_dirty_cells = dirty.len();
        self.last_direction_deltas = direction_deltas;
        self.stack_len += 1;
        profile
    }

    fn materialize_pending(
        &mut self,
        weights: &QuantizedCodebookWeights,
        profile_enabled: bool,
    ) -> EvalStateStepProfile {
        if self.token_delta_journal_enabled() {
            return self.materialize_pending_token_delta(weights, profile_enabled);
        }
        if self.directional_delta.is_some() {
            return self.materialize_pending_directional(weights, profile_enabled);
        }
        let mut profile = EvalStateStepProfile::default();
        for frame_idx in 0..self.stack_len {
            if self.stack[frame_idx].materialized {
                continue;
            }
            let undo = &mut self.stack[frame_idx];
            for undo_idx in 0..undo.len {
                let cell = undo.cells[undo_idx];
                let undo_base = undo_idx * weights.dim;

                let start = EvalStateStepProfile::start(profile_enabled);
                undo.black[undo_base..undo_base + weights.dim].copy_from_slice(quant_cell_slice(
                    &self.cell_black,
                    cell,
                    weights.dim,
                ));
                undo.white[undo_base..undo_base + weights.dim].copy_from_slice(quant_cell_slice(
                    &self.cell_white,
                    cell,
                    weights.dim,
                ));
                profile.add_backup(start);

                let start = EvalStateStepProfile::start(profile_enabled);
                add_quant_cell_to_features(
                    &self.cell_black,
                    &mut self.features_black,
                    cell,
                    weights.dim,
                    -1,
                );
                add_quant_cell_to_features(
                    &self.cell_white,
                    &mut self.features_white,
                    cell,
                    weights.dim,
                    -1,
                );
                profile.add_aggregate(start);

                let start = EvalStateStepProfile::start(profile_enabled);
                compute_cell_quantized_from_pattern_ids(
                    &undo.pattern_ids[undo_idx],
                    weights,
                    Stone::Black,
                    quant_cell_slice_mut(&mut self.cell_black, cell, weights.dim),
                );
                compute_cell_quantized_from_pattern_ids(
                    &undo.pattern_ids[undo_idx],
                    weights,
                    Stone::White,
                    quant_cell_slice_mut(&mut self.cell_white, cell, weights.dim),
                );
                profile.add_recompute(start);

                let start = EvalStateStepProfile::start(profile_enabled);
                add_quant_cell_to_features(
                    &self.cell_black,
                    &mut self.features_black,
                    cell,
                    weights.dim,
                    1,
                );
                add_quant_cell_to_features(
                    &self.cell_white,
                    &mut self.features_white,
                    cell,
                    weights.dim,
                    1,
                );
                profile.add_aggregate(start);
            }
            undo.materialized = true;
        }
        profile
    }

    fn materialize_pending_token_delta(
        &mut self,
        weights: &QuantizedCodebookWeights,
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
            token_journal,
            ..
        } = state;
        let journal = token_journal.as_mut().expect("TokenDelta journal enabled");
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

    fn materialize_pending_directional(
        &mut self,
        weights: &QuantizedCodebookWeights,
        profile_enabled: bool,
    ) -> EvalStateStepProfile {
        let mut profile = EvalStateStepProfile::default();
        let Self {
            cell_black,
            cell_white,
            features_black,
            features_white,
            stack,
            stack_len,
            directional_delta,
            ..
        } = self;
        let state = directional_delta
            .as_mut()
            .expect("directional delta state enabled");

        for undo in stack.iter_mut().take(*stack_len) {
            if undo.materialized {
                continue;
            }
            for undo_idx in 0..undo.len {
                let cell = undo.cells[undo_idx];
                let old_ids = &undo.old_pattern_ids[undo_idx];
                let new_ids = &undo.pattern_ids[undo_idx];

                let start = EvalStateStepProfile::start(profile_enabled);
                apply_quantized_pattern_delta_to_raw(
                    old_ids,
                    new_ids,
                    weights,
                    Stone::Black,
                    quant_cell_slice_mut(&mut state.raw_black, cell, weights.dim),
                );
                apply_quantized_pattern_delta_to_raw(
                    old_ids,
                    new_ids,
                    weights,
                    Stone::White,
                    quant_cell_slice_mut(&mut state.raw_white, cell, weights.dim),
                );
                profile.add_recompute(start);

                let start = EvalStateStepProfile::start(profile_enabled);
                refresh_quantized_cell_activation(
                    &state.raw_black,
                    cell_black,
                    features_black,
                    cell,
                    weights.dim,
                );
                refresh_quantized_cell_activation(
                    &state.raw_white,
                    cell_white,
                    features_white,
                    cell,
                    weights.dim,
                );
                profile.add_aggregate(start);
            }
            undo.materialized = true;
        }
        profile
    }

    pub fn pop_move(&mut self, weights: &QuantizedCodebookWeights) {
        let _ = self.pop_move_profiled(weights, false);
    }

    pub fn pop_move_profiled(
        &mut self,
        weights: &QuantizedCodebookWeights,
        profile_enabled: bool,
    ) -> EvalStateStepProfile {
        let mut profile = EvalStateStepProfile {
            pop_calls: 1,
            ..EvalStateStepProfile::default()
        };
        if self.stack_len == 0 {
            return profile;
        }
        if self.token_delta_journal_enabled() {
            return self.pop_move_token_delta(weights, profile_enabled, profile);
        }
        if self.directional_delta.is_some() {
            return self.pop_move_directional(weights, profile_enabled, profile);
        }
        self.stack_len -= 1;
        let undo = &self.stack[self.stack_len];
        if !undo.materialized {
            self.last_dirty_cells = 0;
            return profile;
        }
        for undo_idx in (0..undo.len).rev() {
            let cell = undo.cells[undo_idx];
            let undo_base = undo_idx * weights.dim;
            let start = EvalStateStepProfile::start(profile_enabled);
            add_quant_cell_to_features(
                &self.cell_black,
                &mut self.features_black,
                cell,
                weights.dim,
                -1,
            );
            add_quant_cell_to_features(
                &self.cell_white,
                &mut self.features_white,
                cell,
                weights.dim,
                -1,
            );
            profile.add_aggregate(start);

            let start = EvalStateStepProfile::start(profile_enabled);
            quant_cell_slice_mut(&mut self.cell_black, cell, weights.dim)
                .copy_from_slice(&undo.black[undo_base..undo_base + weights.dim]);
            quant_cell_slice_mut(&mut self.cell_white, cell, weights.dim)
                .copy_from_slice(&undo.white[undo_base..undo_base + weights.dim]);
            profile.add_restore(start);

            let start = EvalStateStepProfile::start(profile_enabled);
            add_quant_cell_to_features(
                &self.cell_black,
                &mut self.features_black,
                cell,
                weights.dim,
                1,
            );
            add_quant_cell_to_features(
                &self.cell_white,
                &mut self.features_white,
                cell,
                weights.dim,
                1,
            );
            profile.add_aggregate(start);
        }
        self.last_dirty_cells = 0;
        profile
    }

    fn pop_move_token_delta(
        &mut self,
        weights: &QuantizedCodebookWeights,
        profile_enabled: bool,
        mut profile: EvalStateStepProfile,
    ) -> EvalStateStepProfile {
        self.stack_len -= 1;
        let Self {
            cell_black,
            cell_white,
            features_black,
            features_white,
            stack_len,
            directional_delta,
            ..
        } = self;
        let state = directional_delta
            .as_mut()
            .expect("directional delta state enabled");
        let QuantDirectionalDeltaState {
            raw_black,
            raw_white,
            token_journal,
            ..
        } = state;
        let journal = token_journal.as_mut().expect("TokenDelta journal enabled");
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
        debug_assert_eq!(journal.depth(), *stack_len);
        debug_assert!(popped.deltas <= MAX_DIRECTION_DELTAS);
        debug_assert!(journal.materialized_depth() <= journal.depth());
        self.last_dirty_cells = 0;
        self.last_direction_deltas = 0;
        profile
    }

    fn pop_move_directional(
        &mut self,
        weights: &QuantizedCodebookWeights,
        profile_enabled: bool,
        mut profile: EvalStateStepProfile,
    ) -> EvalStateStepProfile {
        self.stack_len -= 1;
        let Self {
            cell_black,
            cell_white,
            features_black,
            features_white,
            stack,
            stack_len,
            directional_delta,
            ..
        } = self;
        let undo = &stack[*stack_len];
        let state = directional_delta
            .as_mut()
            .expect("directional delta state enabled");

        if undo.materialized {
            for undo_idx in (0..undo.len).rev() {
                let cell = undo.cells[undo_idx];
                let old_ids = &undo.old_pattern_ids[undo_idx];
                let new_ids = &undo.pattern_ids[undo_idx];

                let start = EvalStateStepProfile::start(profile_enabled);
                apply_quantized_pattern_delta_to_raw(
                    new_ids,
                    old_ids,
                    weights,
                    Stone::Black,
                    quant_cell_slice_mut(&mut state.raw_black, cell, weights.dim),
                );
                apply_quantized_pattern_delta_to_raw(
                    new_ids,
                    old_ids,
                    weights,
                    Stone::White,
                    quant_cell_slice_mut(&mut state.raw_white, cell, weights.dim),
                );
                profile.add_restore(start);

                let start = EvalStateStepProfile::start(profile_enabled);
                refresh_quantized_cell_activation(
                    &state.raw_black,
                    cell_black,
                    features_black,
                    cell,
                    weights.dim,
                );
                refresh_quantized_cell_activation(
                    &state.raw_white,
                    cell_white,
                    features_white,
                    cell,
                    weights.dim,
                );
                profile.add_aggregate(start);
            }
        }

        let logical_pattern_ids = state
            .logical_pattern_ids
            .as_mut()
            .expect("direct directional token mirror");
        for undo_idx in (0..undo.len).rev() {
            let cell = undo.cells[undo_idx];
            logical_pattern_ids[cell] = undo.old_pattern_ids[undo_idx];
        }
        self.last_dirty_cells = 0;
        self.last_direction_deltas = 0;
        profile
    }

    pub fn value(&mut self, board: &Board, weights: &QuantizedCodebookWeights) -> f32 {
        self.value_profiled(board, weights, false).0
    }

    pub fn value_profiled(
        &mut self,
        board: &Board,
        weights: &QuantizedCodebookWeights,
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
    pub(crate) fn explicit_orbit48(
        &mut self,
        weights: &QuantizedCodebookWeights,
        perspective: Stone,
    ) -> Result<[i64; 48], String> {
        if weights.dim != 16 {
            return Err(format!(
                "white root ordering requires codebook dim 16, got {}",
                weights.dim
            ));
        }
        if weights.embedding_scale != QUANT_EMBED_SCALE || weights.embedding_scale != 32 {
            return Err(format!(
                "white root ordering requires embedding scale 32, got {}",
                weights.embedding_scale
            ));
        }
        let expected_features = REGIONS * 16;
        if self.features_black.len() != expected_features
            || self.features_white.len() != expected_features
            || self.cell_black.len() != NUM_CELLS * 16
            || self.cell_white.len() != NUM_CELLS * 16
            || weights.feature_len() != expected_features
            || weights.embeddings.len() != PATTERN_NUM_IDS * 16
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

fn compute_cell_quantized(
    board: &Board,
    weights: &QuantizedCodebookWeights,
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

fn compute_cell_quantized_from_pattern_ids(
    pattern_ids: &[u16; 4],
    weights: &QuantizedCodebookWeights,
    perspective: Stone,
    out: &mut [i32],
) {
    compute_cell_quantized_raw_from_pattern_ids(pattern_ids, weights, perspective, out);
    for x in out {
        *x = (*x).max(0);
    }
}

fn compute_cell_quantized_raw_from_pattern_ids(
    pattern_ids: &[u16; 4],
    weights: &QuantizedCodebookWeights,
    perspective: Stone,
    out: &mut [i32],
) {
    out.fill(0);
    let swap = perspective == Stone::White;
    for &pid in pattern_ids {
        let pid = if swap { swap_mapped_id(pid) } else { pid };
        let emb_base = pid as usize * weights.dim;
        for d in 0..weights.dim {
            out[d] += weights.embeddings[emb_base + d] as i32;
        }
    }
}

fn apply_quantized_pattern_delta_to_raw(
    old_pattern_ids: &[u16; 4],
    new_pattern_ids: &[u16; 4],
    weights: &QuantizedCodebookWeights,
    perspective: Stone,
    raw: &mut [i32],
) {
    let swap = perspective == Stone::White;
    for dir_idx in 0..4 {
        let old_pid = old_pattern_ids[dir_idx];
        let new_pid = new_pattern_ids[dir_idx];
        if old_pid == new_pid {
            continue;
        }
        let old_pid = if swap {
            swap_mapped_id(old_pid)
        } else {
            old_pid
        };
        let new_pid = if swap {
            swap_mapped_id(new_pid)
        } else {
            new_pid
        };
        let old_base = old_pid as usize * weights.dim;
        let new_base = new_pid as usize * weights.dim;
        for d in 0..weights.dim {
            raw[d] +=
                weights.embeddings[new_base + d] as i32 - weights.embeddings[old_base + d] as i32;
        }
    }
}

#[inline]
fn apply_quantized_token_delta_to_raw(
    old_pattern_id: u16,
    new_pattern_id: u16,
    weights: &QuantizedCodebookWeights,
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
    let old_base = old_pattern_id as usize * weights.dim;
    let new_base = new_pattern_id as usize * weights.dim;
    for d in 0..weights.dim {
        raw[d] += weights.embeddings[new_base + d] as i32 - weights.embeddings[old_base + d] as i32;
    }
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
    let mut logit = weights.bias;
    for (x, w) in features.iter().zip(&weights.head) {
        logit += x * w;
    }
    for rank in 0..weights.fm_rank {
        let mut sum = 0.0f32;
        let mut square_sum = 0.0f32;
        for (idx, &x) in features.iter().enumerate() {
            let vx = weights.factors[idx * weights.fm_rank + rank] * x;
            sum += vx;
            square_sum += vx * vx;
        }
        logit += 0.5 * (sum * sum - square_sum);
    }
    logit
}

fn quant_value_from_features(features: &[i32], weights: &QuantizedCodebookWeights) -> f32 {
    let region_denom = region_cell_count(0) as f64;
    let feature_denom = weights.embedding_scale as f64 * region_denom;
    let mut logit = weights.bias as f64;

    let head_denom = feature_denom * weights.head_scale as f64;
    for (&x, &w) in features.iter().zip(&weights.head) {
        logit += (x as f64 * w as f64) / head_denom;
    }

    let factor_denom = feature_denom * weights.factor_scale as f64;
    for rank in 0..weights.fm_rank {
        let mut sum = 0.0f64;
        let mut square_sum = 0.0f64;
        for (idx, &x) in features.iter().enumerate() {
            let vx =
                (x as f64 * weights.factors[idx * weights.fm_rank + rank] as f64) / factor_denom;
            sum += vx;
            square_sum += vx * vx;
        }
        logit += 0.5 * (sum * sum - square_sum);
    }

    logit as f32
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

fn deterministic_vec(state: &mut u64, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| deterministic_f32(state, scale)).collect()
}

fn deterministic_f32(state: &mut u64, scale: f32) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    let unit = ((*state >> 40) as u32) as f32 / (1u32 << 24) as f32;
    (unit * 2.0 - 1.0) * scale
}

fn quantize_vec_i16(values: &[f32], scale: i32) -> Vec<i16> {
    values
        .iter()
        .map(|&x| {
            (x * scale as f32)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16
        })
        .collect()
}

fn dequantize_vec_i16(values: &[i16], scale: i32) -> Vec<f32> {
    let denom = scale as f32;
    values.iter().map(|&x| x as f32 / denom).collect()
}

fn json_str<'a>(v: &'a Value, key: &str) -> Result<&'a str, String> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string field {key}"))
}

fn json_usize_opt(v: &Value, key: &str) -> Option<usize> {
    v.get(key)
        .and_then(Value::as_u64)
        .and_then(|x| usize::try_from(x).ok())
}

fn json_f32_opt(v: &Value, key: &str) -> Option<f32> {
    v.get(key).and_then(Value::as_f64).map(|x| x as f32)
}

fn json_f32_array(v: &Value, key: &str) -> Result<Vec<f32>, String> {
    let raw = v
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing array field {key}"))?;
    raw.iter()
        .map(|x| {
            x.as_f64()
                .map(|v| v as f32)
                .ok_or_else(|| format!("non-numeric item in {key}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::GameResult;

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
        let mut journal =
            IncrementalQuantizedCodebookEval::new_with_directional_delta_and_token_journal(
                &weights, true, true,
            );
        legacy.refresh(&board, &weights);
        delta.refresh(&board, &weights);
        journal.refresh(&board, &weights);
        assert!(delta.directional_delta_enabled());
        assert!(journal.token_delta_journal_enabled());
        assert_quantized_directional_state(&board, &mut delta, &weights);
        assert_quantized_directional_state(&board, &mut journal, &weights);

        for (ply, &mv) in moves.iter().enumerate() {
            if !board.is_empty(mv) {
                continue;
            }
            board.make_move(mv);
            legacy.push_move(&board, mv, &weights);
            delta.push_move(&board, mv, &weights);
            journal.push_move(&board, mv, &weights);
            assert_eq!(
                delta
                    .directional_delta
                    .as_ref()
                    .unwrap()
                    .logical_pattern_ids(),
                board.line_pattern_ids.as_ref()
            );
            assert_eq!(
                journal
                    .directional_delta
                    .as_ref()
                    .unwrap()
                    .logical_pattern_ids(),
                board.line_pattern_ids.as_ref()
            );
            if ply % 3 == 2 {
                let direct_value = delta.value(&board, &weights);
                assert_eq!(
                    direct_value.to_bits(),
                    legacy.value(&board, &weights).to_bits()
                );
                assert_eq!(
                    journal.value(&board, &weights).to_bits(),
                    direct_value.to_bits()
                );
                assert_quantized_directional_state(&board, &mut delta, &weights);
                assert_quantized_directional_state(&board, &mut journal, &weights);
            }
        }
        assert_quantized_directional_state(&board, &mut delta, &weights);
        assert_quantized_directional_state(&board, &mut journal, &weights);

        for ply in (0..moves.len()).rev() {
            if board.history.is_empty() {
                break;
            }
            board.undo_move();
            legacy.pop_move(&weights);
            delta.pop_move(&weights);
            journal.pop_move(&weights);
            assert_eq!(
                delta
                    .directional_delta
                    .as_ref()
                    .unwrap()
                    .logical_pattern_ids(),
                board.line_pattern_ids.as_ref()
            );
            assert_eq!(
                journal
                    .directional_delta
                    .as_ref()
                    .unwrap()
                    .logical_pattern_ids(),
                board.line_pattern_ids.as_ref()
            );
            if ply % 4 == 0 {
                let direct_value = delta.value(&board, &weights);
                assert_eq!(
                    direct_value.to_bits(),
                    legacy.value(&board, &weights).to_bits()
                );
                assert_eq!(
                    journal.value(&board, &weights).to_bits(),
                    direct_value.to_bits()
                );
                assert_quantized_directional_state(&board, &mut delta, &weights);
                assert_quantized_directional_state(&board, &mut journal, &weights);
            }
        }
        assert_quantized_directional_state(&board, &mut delta, &weights);
        assert_quantized_directional_state(&board, &mut journal, &weights);
    }

    #[test]
    #[ignore = "CB-D1/CB-TD1 release gate: run explicitly with --release --ignored"]
    fn quantized_directional_delta_100k_mixed_make_undo_full_rebuild_equality() {
        const OPERATIONS: usize = 100_000;
        const FULL_REBUILD_PERIOD: usize = 97;

        let weights = CodebookWeights::deterministic(16, 8).quantize_i16_s32_s64();
        let mut board = Board::new();
        let mut legacy = IncrementalQuantizedCodebookEval::new(&weights);
        let mut delta =
            IncrementalQuantizedCodebookEval::new_with_directional_delta(&weights, true);
        let mut journal =
            IncrementalQuantizedCodebookEval::new_with_directional_delta_and_token_journal(
                &weights, true, true,
            );
        legacy.refresh(&board, &weights);
        delta.refresh(&board, &weights);
        journal.refresh(&board, &weights);
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
                journal.pop_move(&weights);
                undos += 1;
            } else {
                let moves = board.legal_moves();
                let mv = moves[rng.usize(moves.len())];
                board.make_move(mv);
                legacy.push_move(&board, mv, &weights);
                delta.push_move(&board, mv, &weights);
                journal.push_move(&board, mv, &weights);
                assert_eq!(
                    delta.last_direction_deltas(),
                    journal.last_direction_deltas(),
                    "direction delta count at operation {operation}"
                );
                direction_deltas += delta.last_direction_deltas();
                makes += 1;
            }

            let state = delta.directional_delta.as_ref().unwrap();
            assert_eq!(
                state.logical_pattern_ids(),
                board.line_pattern_ids.as_ref(),
                "direct logical Pattern4 IDs at operation {operation}"
            );
            assert_eq!(
                journal
                    .directional_delta
                    .as_ref()
                    .unwrap()
                    .logical_pattern_ids(),
                board.line_pattern_ids.as_ref(),
                "journal logical Pattern4 IDs at operation {operation}"
            );

            if rng.usize(8) == 0 || operation % FULL_REBUILD_PERIOD == 0 || operation == OPERATIONS
            {
                let direct_value = delta.value(&board, &weights);
                let legacy_value = legacy.value(&board, &weights);
                let journal_value = journal.value(&board, &weights);
                assert_eq!(direct_value.to_bits(), legacy_value.to_bits());
                assert_eq!(
                    journal_value.to_bits(),
                    direct_value.to_bits(),
                    "journal value at operation {operation}"
                );
                materializations += 1;
            }
            if operation % FULL_REBUILD_PERIOD == 0 || operation == OPERATIONS {
                assert_quantized_directional_state(&board, &mut delta, &weights);
                assert_quantized_directional_state(&board, &mut journal, &weights);
            }
        }

        while !board.history.is_empty() {
            board.undo_move();
            legacy.pop_move(&weights);
            delta.pop_move(&weights);
            journal.pop_move(&weights);
        }
        assert_quantized_directional_state(&board, &mut delta, &weights);
        assert_quantized_directional_state(&board, &mut journal, &weights);
        assert_eq!(
            delta.value(&board, &weights).to_bits(),
            legacy.value(&board, &weights).to_bits()
        );
        assert_eq!(
            journal.value(&board, &weights).to_bits(),
            delta.value(&board, &weights).to_bits()
        );
        eprintln!(
            "CB-D1/CB-TD1 operations={OPERATIONS} makes={makes} undos={undos} \
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
