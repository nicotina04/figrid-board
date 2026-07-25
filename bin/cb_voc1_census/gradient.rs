use figrid_board::board::{Board, Stone, BOARD_SIZE, NUM_CELLS};
use figrid_board::codebook_eval::{
    QuantizedCodebookWeights, QUANT_EMBED_SCALE, QUANT_FACTOR_SCALE, QUANT_HEAD_SCALE,
};
use figrid_board::pattern_table::{
    canonicalize, lookup_mapped_id, pack_window, read_window, swap_mapped_id,
};
use std::collections::BTreeMap;

pub(crate) const DIM: usize = 16;
pub(crate) const FM_RANK: usize = 8;
pub(crate) const REGIONS: usize = 9;
pub(crate) const K6: usize = 6;
const REGION_CELLS: f64 = 25.0;
const FEATURE_DENOM: f64 = QUANT_EMBED_SCALE as f64 * REGION_CELLS;
const DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Neumaier {
    sum: f64,
    correction: f64,
}

impl Neumaier {
    pub(crate) fn add(&mut self, value: f64) {
        let next = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.correction += (self.sum - next) + value;
        } else {
            self.correction += (value - next) + self.sum;
        }
        self.sum = next;
    }

    pub(crate) fn total(self) -> f64 {
        self.sum + self.correction
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ChildAnalysis {
    pub(crate) ell: f32,
    pub(crate) precast: f64,
    raw_tokens: Box<[[u32; 4]; NUM_CELLS]>,
    cell_preactivations: Vec<i32>,
    beta: [f64; REGIONS * DIM],
    pub(crate) raw_slots: BTreeMap<u32, u32>,
    pub(crate) zero_preactivations: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct SlateGradient {
    pub(crate) ce: f64,
    pub(crate) policy: [f64; K6],
    pub(crate) alpha: [f64; K6],
    pub(crate) raw_gradient: BTreeMap<u32, [f64; DIM]>,
    pub(crate) raw_slots: BTreeMap<u32, u32>,
    pub(crate) raw_occurrence_residual: BTreeMap<u32, f64>,
}

pub(crate) fn analyze_product_child(
    board: &Board,
    weights: &QuantizedCodebookWeights,
) -> Result<ChildAnalysis, String> {
    validate_product_weights(weights)?;

    let mut raw_tokens = Box::new([[0u32; 4]; NUM_CELLS]);
    let mut raw_cells = vec![0i32; NUM_CELLS * DIM];
    let mut features = [0i32; REGIONS * DIM];
    let mut raw_slots = BTreeMap::<u32, u32>::new();
    let (mine, opponent) = match board.side_to_move {
        Stone::Black => (&board.black, &board.white),
        Stone::White => (&board.white, &board.black),
    };

    for cell in 0..NUM_CELLS {
        let row = (cell / BOARD_SIZE) as i32;
        let col = (cell % BOARD_SIZE) as i32;
        for (direction, &(dr, dc)) in DIRS.iter().enumerate() {
            let window = canonicalize(&read_window(mine, opponent, row, col, dr, dc));
            let raw = pack_window(&window);
            raw_tokens[cell][direction] = raw;
            *raw_slots.entry(raw).or_default() += 1;

            let mapped = lookup_mapped_id(raw);
            let cached = match board.side_to_move {
                Stone::Black => board.line_pattern_ids[cell][direction],
                Stone::White => swap_mapped_id(board.line_pattern_ids[cell][direction]),
            };
            if mapped != cached {
                return Err(format!(
                    "raw/current mapped token mismatch at cell {cell} direction {direction}: \
                     raw={raw:08X} mapped={mapped} cached={cached}"
                ));
            }
            let embedding_base = mapped as usize * DIM;
            let cell_base = cell * DIM;
            for dimension in 0..DIM {
                raw_cells[cell_base + dimension] +=
                    weights.embeddings[embedding_base + dimension] as i32;
            }
        }
        let feature_base = region_of_cell(cell) * DIM;
        let cell_base = cell * DIM;
        for dimension in 0..DIM {
            features[feature_base + dimension] += raw_cells[cell_base + dimension].max(0);
        }
    }

    let (precast, ell, beta) = independent_product_forward(&features, weights)?;
    let mut zero_preactivations = 0u64;
    for cell in 0..NUM_CELLS {
        let cell_base = cell * DIM;
        for dimension in 0..DIM {
            let preactivation = raw_cells[cell_base + dimension];
            zero_preactivations += u64::from(preactivation == 0);
        }
    }
    if raw_slots
        .values()
        .map(|&count| count as usize)
        .sum::<usize>()
        != NUM_CELLS * 4
    {
        return Err("raw slot census is not 225*4".to_string());
    }
    if !precast.is_finite() || !ell.is_finite() || beta.iter().any(|value| !value.is_finite()) {
        return Err("non-finite product child analysis".to_string());
    }

    Ok(ChildAnalysis {
        ell,
        precast,
        raw_tokens,
        cell_preactivations: raw_cells,
        beta,
        raw_slots,
        zero_preactivations,
    })
}

pub(crate) fn combine_k6_gradient(
    children: &[ChildAnalysis; K6],
    q_teacher: &[f64; K6],
) -> Result<SlateGradient, String> {
    let utilities = std::array::from_fn(|index| -(children[index].ell as f64));
    let policy = softmax6(&utilities)?;
    let mut alpha = [0.0f64; K6];
    let mut q_sum = Neumaier::default();
    let mut ce = Neumaier::default();
    for index in 0..K6 {
        let q = q_teacher[index];
        if !q.is_finite() || q <= 0.0 {
            return Err(format!("invalid q_teacher[{index}]={q}"));
        }
        q_sum.add(q);
        alpha[index] = q - policy[index];
        ce.add(-q * policy[index].ln());
    }
    if (q_sum.total() - 1.0).abs() > 1.0e-12 {
        return Err(format!("q_teacher sum is {}, expected one", q_sum.total()));
    }

    let mut gradient_sums = BTreeMap::<u32, [Neumaier; DIM]>::new();
    let mut residual_sums = BTreeMap::<u32, Neumaier>::new();
    let mut raw_slots = BTreeMap::<u32, u32>::new();
    for candidate in 0..K6 {
        for (&raw, &count) in &children[candidate].raw_slots {
            *raw_slots.entry(raw).or_default() += count;
        }
        for cell in 0..NUM_CELLS {
            let cell_base = cell * DIM;
            let feature_base = region_of_cell(cell) * DIM;
            for direction in 0..4 {
                let raw = children[candidate].raw_tokens[cell][direction];
                residual_sums.entry(raw).or_default().add(alpha[candidate]);
                let target = gradient_sums.entry(raw).or_default();
                for dimension in 0..DIM {
                    if children[candidate].cell_preactivations[cell_base + dimension] > 0 {
                        let sensitivity =
                            children[candidate].beta[feature_base + dimension] / FEATURE_DENOM;
                        target[dimension].add(alpha[candidate] * sensitivity);
                    }
                }
            }
        }
    }
    let raw_gradient = gradient_sums
        .into_iter()
        .map(|(raw, sums)| (raw, sums.map(Neumaier::total)))
        .collect::<BTreeMap<_, _>>();
    let raw_occurrence_residual = residual_sums
        .into_iter()
        .map(|(raw, sum)| (raw, sum.total()))
        .collect::<BTreeMap<_, _>>();
    if raw_slots
        .values()
        .map(|&count| count as usize)
        .sum::<usize>()
        != K6 * NUM_CELLS * 4
    {
        return Err("slate raw slot census is not 6*225*4".to_string());
    }
    if !ce.total().is_finite()
        || policy.iter().any(|value| !value.is_finite())
        || alpha.iter().any(|value| !value.is_finite())
        || raw_gradient
            .values()
            .flatten()
            .any(|value| !value.is_finite())
        || raw_occurrence_residual
            .values()
            .any(|value| !value.is_finite())
    {
        return Err("non-finite K6 gradient".to_string());
    }

    Ok(SlateGradient {
        ce: ce.total(),
        policy,
        alpha,
        raw_gradient,
        raw_slots,
        raw_occurrence_residual,
    })
}

fn validate_product_weights(weights: &QuantizedCodebookWeights) -> Result<(), String> {
    if weights.dim != DIM
        || weights.fm_rank != FM_RANK
        || weights.embedding_scale != QUANT_EMBED_SCALE
        || weights.head_scale != QUANT_HEAD_SCALE
        || weights.factor_scale != QUANT_FACTOR_SCALE
        || weights.embeddings.len() % DIM != 0
        || weights.head.len() != REGIONS * DIM
        || weights.factors.len() != REGIONS * DIM * FM_RANK
    {
        return Err(format!(
            "unsupported product weights: dim={} rank={} scales={}/{}/{} lens={}/{}/{}",
            weights.dim,
            weights.fm_rank,
            weights.embedding_scale,
            weights.head_scale,
            weights.factor_scale,
            weights.embeddings.len(),
            weights.head.len(),
            weights.factors.len()
        ));
    }
    Ok(())
}

fn independent_product_forward(
    features: &[i32; REGIONS * DIM],
    weights: &QuantizedCodebookWeights,
) -> Result<(f64, f32, [f64; REGIONS * DIM]), String> {
    let mut logit = weights.bias as f64;
    let mut normalized = [0.0f64; REGIONS * DIM];
    let head_denom = FEATURE_DENOM * QUANT_HEAD_SCALE as f64;
    for index in 0..normalized.len() {
        normalized[index] = features[index] as f64 / FEATURE_DENOM;
        logit += (features[index] as f64 * weights.head[index] as f64) / head_denom;
    }

    let mut analytic_factor_sums = [0.0f64; FM_RANK];
    let factor_denom = FEATURE_DENOM * QUANT_FACTOR_SCALE as f64;
    for rank in 0..FM_RANK {
        let mut sum = 0.0f64;
        let mut square_sum = 0.0f64;
        let mut analytic_sum = Neumaier::default();
        for (index, &x) in features.iter().enumerate() {
            let vx = (x as f64 * weights.factors[index * FM_RANK + rank] as f64) / factor_denom;
            sum += vx;
            square_sum += vx * vx;
            analytic_sum.add(vx);
        }
        logit += 0.5 * (sum * sum - square_sum);
        analytic_factor_sums[rank] = analytic_sum.total();
    }

    let mut beta = [0.0f64; REGIONS * DIM];
    for (index, slot) in beta.iter_mut().enumerate() {
        let mut value = Neumaier::default();
        value.add(weights.head[index] as f64 / QUANT_HEAD_SCALE as f64);
        for (rank, &sum) in analytic_factor_sums.iter().enumerate() {
            let factor = weights.factors[index * FM_RANK + rank] as f64 / QUANT_FACTOR_SCALE as f64;
            value.add(factor * (sum - factor * normalized[index]));
        }
        *slot = value.total();
    }
    if !logit.is_finite() || beta.iter().any(|value| !value.is_finite()) {
        return Err("non-finite independent product forward".to_string());
    }
    Ok((logit, logit as f32, beta))
}

fn softmax6(values: &[f64; K6]) -> Result<[f64; K6], String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err("non-finite K6 utility".to_string());
    }
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut out = [0.0f64; K6];
    let mut denom = Neumaier::default();
    for index in 0..K6 {
        out[index] = (values[index] - max).exp();
        denom.add(out[index]);
    }
    let denom = denom.total();
    if !denom.is_finite() || denom <= 0.0 {
        return Err("invalid K6 softmax denominator".to_string());
    }
    for value in &mut out {
        *value /= denom;
    }
    Ok(out)
}

#[inline]
fn region_of_cell(cell: usize) -> usize {
    let row = cell / BOARD_SIZE;
    let col = cell % BOARD_SIZE;
    (row / 5) * 3 + col / 5
}

#[cfg(test)]
mod tests {
    use super::*;
    use figrid_board::codebook_eval::{evaluate_full_quantized, CodebookWeights};

    #[test]
    fn independent_product_forward_matches_released_kernel() {
        let weights = CodebookWeights::deterministic(DIM, FM_RANK).quantize_i16_s32_s64();
        let mut board = Board::new();
        for &mv in &[112usize, 113, 97, 98, 127, 128, 111] {
            let analysis = analyze_product_child(&board, &weights).unwrap();
            assert_eq!(
                analysis.ell.to_bits(),
                evaluate_full_quantized(&board, &weights).to_bits()
            );
            board.make_move(mv);
        }
    }

    #[test]
    fn raw_natural_tokens_map_to_cached_perspective_for_white() {
        let weights = CodebookWeights::deterministic(DIM, FM_RANK).quantize_i16_s32_s64();
        let mut board = Board::new();
        board.make_move(112);
        assert_eq!(board.side_to_move, Stone::White);
        analyze_product_child(&board, &weights).unwrap();
    }

    #[test]
    fn identical_children_with_matching_policy_cancel_listwise_gradient() {
        let weights = CodebookWeights::deterministic(DIM, FM_RANK).quantize_i16_s32_s64();
        let child = analyze_product_child(&Board::new(), &weights).unwrap();
        let children: [ChildAnalysis; K6] = std::array::from_fn(|_| child.clone());
        let q = [1.0 / K6 as f64; K6];
        let slate = combine_k6_gradient(&children, &q).unwrap();
        assert!((slate.ce - (K6 as f64).ln()).abs() < 1.0e-12);
        assert!(slate.alpha.iter().all(|value| value.abs() < 1.0e-15));
        assert!(slate
            .raw_gradient
            .values()
            .flatten()
            .all(|value| value.abs() < 1.0e-15));
    }
}
