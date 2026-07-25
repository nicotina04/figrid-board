//! CB-VOC1 color-orbit geometry and exact fixed-capacity selector.
//!
//! This module intentionally has no dependency on the census corpus.  It
//! reconstructs the released Pattern4 vocabulary geometry from first
//! principles, validates the incumbent `topk.bin`, and solves the registered
//! cost-{1,2} vocabulary problem exactly.

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, HashSet};

pub(crate) const WINDOW_CELLS: usize = 11;
pub(crate) const RAW_BITS: usize = WINDOW_CELLS * 2;
pub(crate) const RAW_SPACE: usize = 1usize << RAW_BITS;

pub(crate) const RELEASED_UNIVERSE_ROWS: usize = 199_827;
pub(crate) const RELEASED_UNIVERSE_FIXED_ORBITS: usize = 215;
pub(crate) const RELEASED_UNIVERSE_PAIR_ORBITS: usize = 99_806;
pub(crate) const RELEASED_UNIVERSE_ORBITS: usize =
    RELEASED_UNIVERSE_FIXED_ORBITS + RELEASED_UNIVERSE_PAIR_ORBITS;
pub(crate) const RELEASED_ANCHOR_BOUNDARY_ROWS: usize = 537;

pub(crate) const INCUMBENT_ROWS: usize = 4_265;
pub(crate) const INCUMBENT_FIXED_ORBITS: usize = 29;
pub(crate) const INCUMBENT_PAIR_ORBITS: usize = 2_118;
pub(crate) const INCUMBENT_ORBITS: usize = INCUMBENT_FIXED_ORBITS + INCUMBENT_PAIR_ORBITS;
pub(crate) const INCUMBENT_FREQUENCY_ROWS: usize = 4_096;
pub(crate) const INCUMBENT_CLOSURE_TAIL_ROWS: usize = INCUMBENT_ROWS - INCUMBENT_FREQUENCY_ROWS;
pub(crate) const INCUMBENT_RARE_ID: usize = INCUMBENT_ROWS;

/// A mine/opponent color orbit after left/right reflection canonicalization.
///
/// `first <= second`.  A fixed orbit has `first == second` and costs one
/// vocabulary row.  Otherwise the orbit costs two independently parameterized
/// rows.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct ColorOrbit {
    pub(crate) first: u32,
    pub(crate) second: u32,
}

impl ColorOrbit {
    #[inline]
    pub(crate) fn new(row: u32) -> Self {
        let partner = color_swap_sigma(row);
        if row <= partner {
            Self {
                first: row,
                second: partner,
            }
        } else {
            Self {
                first: partner,
                second: row,
            }
        }
    }

    #[inline]
    pub(crate) fn cost(self) -> usize {
        if self.first == self.second {
            1
        } else {
            2
        }
    }

    #[inline]
    pub(crate) fn is_fixed(self) -> bool {
        self.first == self.second
    }

    #[inline]
    pub(crate) fn contains(self, row: u32) -> bool {
        row == self.first || row == self.second
    }

    #[inline]
    pub(crate) fn visit_rows(self, mut visit: impl FnMut(u32)) {
        visit(self.first);
        if self.second != self.first {
            visit(self.second);
        }
    }
}

/// The released incumbent vocabulary, retaining its frequency/tail ID order.
#[derive(Clone, Debug)]
pub(crate) struct IncumbentVocabulary {
    rows_by_id: Vec<u32>,
    id_by_row: HashMap<u32, usize>,
    row_set: HashSet<u32>,
    pub(crate) fixed_orbits: usize,
    pub(crate) pair_orbits: usize,
}

impl IncumbentVocabulary {
    #[inline]
    pub(crate) fn rows_by_id(&self) -> &[u32] {
        &self.rows_by_id
    }

    #[inline]
    pub(crate) fn contains(&self, row: u32) -> bool {
        self.row_set.contains(&row)
    }

    #[inline]
    pub(crate) fn id_of(&self, row: u32) -> Option<usize> {
        self.id_by_row.get(&row).copied()
    }

    #[inline]
    pub(crate) fn mapped_id_or_rare(&self, row: u32) -> usize {
        self.id_of(row).unwrap_or(INCUMBENT_RARE_ID)
    }

    pub(crate) fn sorted_rows(&self) -> Vec<u32> {
        let mut rows = self.rows_by_id.clone();
        rows.sort_unstable();
        rows
    }
}

/// Complete released universe, color-orbit partition, and incumbent table.
#[derive(Clone, Debug)]
pub(crate) struct OrbitGeometry {
    /// All realizable reflection-canonical packed rows, ascending.
    pub(crate) universe: Vec<u32>,
    /// All color orbits, ascending by `(first, second)`.
    pub(crate) orbits: Vec<ColorOrbit>,
    pub(crate) incumbent: IncumbentVocabulary,
    pub(crate) anchor_boundary_rows: usize,
    pub(crate) fixed_orbits: usize,
    pub(crate) pair_orbits: usize,
}

impl OrbitGeometry {
    #[inline]
    pub(crate) fn contains_universe_row(&self, row: u32) -> bool {
        self.universe.binary_search(&row).is_ok()
    }

    pub(crate) fn orbit_of(&self, row: u32) -> Option<ColorOrbit> {
        self.contains_universe_row(row)
            .then(|| ColorOrbit::new(row))
    }
}

/// Pack one 11-cell base-4 window into the released 22-bit representation.
#[inline]
pub(crate) fn pack_window(window: &[u8; WINDOW_CELLS]) -> u32 {
    let mut packed = 0u32;
    for &cell in window {
        debug_assert!(cell < 4);
        packed = (packed << 2) | u32::from(cell);
    }
    packed
}

/// Unpack one released 22-bit representation.
#[inline]
pub(crate) fn unpack_window(packed: u32) -> [u8; WINDOW_CELLS] {
    debug_assert!((packed as usize) < RAW_SPACE);
    let mut window = [0u8; WINDOW_CELLS];
    for (index, cell) in window.iter_mut().enumerate() {
        let shift = (WINDOW_CELLS - 1 - index) * 2;
        *cell = ((packed >> shift) & 3) as u8;
    }
    window
}

#[inline]
fn reverse_packed(mut packed: u32) -> u32 {
    let mut reversed = 0u32;
    for _ in 0..WINDOW_CELLS {
        reversed = (reversed << 2) | (packed & 3);
        packed >>= 2;
    }
    reversed
}

/// Apply the released left/right reflection quotient.
#[inline]
pub(crate) fn canonicalize_packed(packed: u32) -> u32 {
    debug_assert!((packed as usize) < RAW_SPACE);
    packed.min(reverse_packed(packed))
}

/// Swap mine (`1`) and opponent (`2`), then apply reflection canonicalization.
///
/// This is the registered color involution `sigma`.  Input may be either a
/// raw or already reflection-canonical packed window; the output is always
/// reflection-canonical.
#[inline]
pub(crate) fn color_swap_sigma(packed: u32) -> u32 {
    debug_assert!((packed as usize) < RAW_SPACE);
    let mut swapped = 0u32;
    for index in 0..WINDOW_CELLS {
        let shift = (WINDOW_CELLS - 1 - index) * 2;
        let cell = ((packed >> shift) & 3) as u8;
        let swapped_cell = match cell {
            1 => 2,
            2 => 1,
            other => other,
        };
        swapped = (swapped << 2) | u32::from(swapped_cell);
    }
    canonicalize_packed(swapped)
}

/// Exact released realizability predicate.
pub(crate) fn is_released_realizable(packed: u32) -> bool {
    if (packed as usize) >= RAW_SPACE {
        return false;
    }
    let window = unpack_window(packed);
    let mut left_boundary = 0usize;
    while left_boundary < WINDOW_CELLS && window[left_boundary] == 3 {
        left_boundary += 1;
    }
    let mut right_boundary = 0usize;
    while right_boundary < WINDOW_CELLS && window[WINDOW_CELLS - 1 - right_boundary] == 3 {
        right_boundary += 1;
    }
    if left_boundary + right_boundary > WINDOW_CELLS {
        return false;
    }
    window[left_boundary..WINDOW_CELLS - right_boundary]
        .iter()
        .all(|&cell| cell != 3)
}

/// Enumerate the released reflection-canonical universe in ascending order.
///
/// The middle non-boundary span is generated directly, avoiding a scan of all
/// 4^11 bit patterns while preserving exactly the released
/// `[boundary*, {0,1,2}+, boundary*]` language.
pub(crate) fn enumerate_released_universe() -> Vec<u32> {
    let mut seen = vec![false; RAW_SPACE];
    for left_boundary in 0..WINDOW_CELLS {
        for right_boundary in 0..(WINDOW_CELLS - left_boundary) {
            let middle_len = WINDOW_CELLS - left_boundary - right_boundary;
            let combinations = 3usize.pow(middle_len as u32);
            for mut code in 0..combinations {
                let mut window = [3u8; WINDOW_CELLS];
                for cell in window[left_boundary..WINDOW_CELLS - right_boundary]
                    .iter_mut()
                    .rev()
                {
                    *cell = (code % 3) as u8;
                    code /= 3;
                }
                let canonical = canonicalize_packed(pack_window(&window));
                seen[canonical as usize] = true;
            }
        }
    }
    seen.into_iter()
        .enumerate()
        .filter_map(|(packed, present)| present.then_some(packed as u32))
        .collect()
}

/// Parse and strictly validate the released 4,265-row `topk.bin`.
pub(crate) fn parse_released_topk(
    bytes: &[u8],
    universe: &[u32],
) -> Result<IncumbentVocabulary, String> {
    let expected_bytes = INCUMBENT_ROWS * std::mem::size_of::<u32>();
    if bytes.len() != expected_bytes {
        return Err(format!(
            "topk length mismatch: got {} bytes, expected {expected_bytes}",
            bytes.len()
        ));
    }
    let mut rows_by_id = Vec::with_capacity(INCUMBENT_ROWS);
    for chunk in bytes.chunks_exact(4) {
        rows_by_id.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    let row_set: HashSet<u32> = rows_by_id.iter().copied().collect();
    if row_set.len() != INCUMBENT_ROWS {
        return Err(format!(
            "topk contains duplicate rows: {} unique of {INCUMBENT_ROWS}",
            row_set.len()
        ));
    }
    for (id, &row) in rows_by_id.iter().enumerate() {
        if !is_released_realizable(row) {
            return Err(format!("topk ID {id} is not realizable: {row:08X}"));
        }
        if canonicalize_packed(row) != row {
            return Err(format!(
                "topk ID {id} is not reflection-canonical: {row:08X}"
            ));
        }
        if universe.binary_search(&row).is_err() {
            return Err(format!(
                "topk ID {id} is absent from released universe: {row:08X}"
            ));
        }
        let partner = color_swap_sigma(row);
        if !row_set.contains(&partner) {
            return Err(format!(
                "topk ID {id} is not color closed: {row:08X} -> {partner:08X}"
            ));
        }
        if color_swap_sigma(partner) != row {
            return Err(format!(
                "topk color involution failed: {row:08X} -> {partner:08X}"
            ));
        }
    }

    let fixed_orbits = rows_by_id
        .iter()
        .filter(|&&row| color_swap_sigma(row) == row)
        .count();
    let paired_rows = INCUMBENT_ROWS
        .checked_sub(fixed_orbits)
        .ok_or_else(|| "incumbent fixed-orbit underflow".to_string())?;
    if paired_rows % 2 != 0 {
        return Err(format!("incumbent paired row count is odd: {paired_rows}"));
    }
    let pair_orbits = paired_rows / 2;
    if fixed_orbits != INCUMBENT_FIXED_ORBITS || pair_orbits != INCUMBENT_PAIR_ORBITS {
        return Err(format!(
            "incumbent orbit census mismatch: fixed={fixed_orbits}, pairs={pair_orbits}, \
             expected {}+{}",
            INCUMBENT_FIXED_ORBITS, INCUMBENT_PAIR_ORBITS
        ));
    }

    let frequency_set: HashSet<u32> = rows_by_id[..INCUMBENT_FREQUENCY_ROWS]
        .iter()
        .copied()
        .collect();
    let missing_frequency_partners = rows_by_id[..INCUMBENT_FREQUENCY_ROWS]
        .iter()
        .filter(|&&row| !frequency_set.contains(&color_swap_sigma(row)))
        .count();
    if missing_frequency_partners != INCUMBENT_CLOSURE_TAIL_ROWS {
        return Err(format!(
            "frequency-table open-partner count mismatch: got \
             {missing_frequency_partners}, expected {INCUMBENT_CLOSURE_TAIL_ROWS}"
        ));
    }
    for (tail_offset, &row) in rows_by_id[INCUMBENT_FREQUENCY_ROWS..].iter().enumerate() {
        let partner = color_swap_sigma(row);
        if row == partner || !frequency_set.contains(&partner) {
            return Err(format!(
                "closure tail ID {} is not a non-fixed partner of a frequency row: \
                 {row:08X} -> {partner:08X}",
                INCUMBENT_FREQUENCY_ROWS + tail_offset
            ));
        }
    }

    let id_by_row = rows_by_id
        .iter()
        .enumerate()
        .map(|(id, &row)| (row, id))
        .collect();
    Ok(IncumbentVocabulary {
        rows_by_id,
        id_by_row,
        row_set,
        fixed_orbits,
        pair_orbits,
    })
}

/// Build the strict released geometry and prove all immutable CB-VOC1 counts.
pub(crate) fn released_geometry(topk_bytes: &[u8]) -> Result<OrbitGeometry, String> {
    let universe = enumerate_released_universe();
    if universe.len() != RELEASED_UNIVERSE_ROWS {
        return Err(format!(
            "released universe mismatch: got {}, expected {RELEASED_UNIVERSE_ROWS}",
            universe.len()
        ));
    }
    if universe.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("released universe is not strictly ascending".to_string());
    }

    let anchor_boundary_rows = universe
        .iter()
        .filter(|&&row| unpack_window(row)[WINDOW_CELLS / 2] == 3)
        .count();
    if anchor_boundary_rows != RELEASED_ANCHOR_BOUNDARY_ROWS {
        return Err(format!(
            "anchor-boundary census mismatch: got {anchor_boundary_rows}, expected \
             {RELEASED_ANCHOR_BOUNDARY_ROWS}"
        ));
    }

    let mut orbits = Vec::with_capacity(RELEASED_UNIVERSE_ORBITS);
    for &row in &universe {
        let partner = color_swap_sigma(row);
        if universe.binary_search(&partner).is_err() {
            return Err(format!(
                "universe is not color closed: {row:08X} -> {partner:08X}"
            ));
        }
        if color_swap_sigma(partner) != row {
            return Err(format!(
                "universe color involution failed: {row:08X} -> {partner:08X}"
            ));
        }
        if row <= partner {
            orbits.push(ColorOrbit {
                first: row,
                second: partner,
            });
        }
    }
    orbits.sort_unstable();
    if orbits.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("universe orbit partition is not unique".to_string());
    }
    let fixed_orbits = orbits.iter().filter(|orbit| orbit.is_fixed()).count();
    let pair_orbits = orbits.len() - fixed_orbits;
    if orbits.len() != RELEASED_UNIVERSE_ORBITS
        || fixed_orbits != RELEASED_UNIVERSE_FIXED_ORBITS
        || pair_orbits != RELEASED_UNIVERSE_PAIR_ORBITS
    {
        return Err(format!(
            "universe orbit census mismatch: total={}, fixed={fixed_orbits}, \
             pairs={pair_orbits}; expected {}={}+{}",
            orbits.len(),
            RELEASED_UNIVERSE_ORBITS,
            RELEASED_UNIVERSE_FIXED_ORBITS,
            RELEASED_UNIVERSE_PAIR_ORBITS
        ));
    }

    let incumbent = parse_released_topk(topk_bytes, &universe)?;
    Ok(OrbitGeometry {
        universe,
        orbits,
        incumbent,
        anchor_boundary_rows,
        fixed_orbits,
        pair_orbits,
    })
}

/// Compensated sum in the exact supplied order.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct NeumaierSum {
    sum: f64,
    correction: f64,
}

impl NeumaierSum {
    #[inline]
    pub(crate) fn add(&mut self, value: f64) {
        let next = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.correction += (self.sum - next) + value;
        } else {
            self.correction += (value - next) + self.sum;
        }
        self.sum = next;
    }

    #[inline]
    pub(crate) fn total(self) -> f64 {
        self.sum + self.correction
    }
}

pub(crate) fn neumaier_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = NeumaierSum::default();
    for value in values {
        sum.add(value);
    }
    sum.total()
}

#[derive(Clone, Debug)]
struct ScoredOrbit {
    orbit: ColorOrbit,
    value: f64,
    incumbent_rows_retained: usize,
}

fn scored_orbit_order(left: &ScoredOrbit, right: &ScoredOrbit) -> Ordering {
    right
        .value
        .total_cmp(&left.value)
        .then_with(|| {
            right
                .incumbent_rows_retained
                .cmp(&left.incumbent_rows_retained)
        })
        .then_with(|| left.orbit.cmp(&right.orbit))
}

fn prefix_values(orbits: &[ScoredOrbit]) -> Result<Vec<f64>, String> {
    let mut prefix = Vec::with_capacity(orbits.len() + 1);
    prefix.push(0.0);
    let mut sum = NeumaierSum::default();
    for orbit in orbits {
        sum.add(orbit.value);
        let value = sum.total();
        if !value.is_finite() {
            return Err("non-finite orbit prefix sum".to_string());
        }
        prefix.push(value);
    }
    Ok(prefix)
}

fn prefix_retained(orbits: &[ScoredOrbit]) -> Result<Vec<usize>, String> {
    let mut prefix = Vec::with_capacity(orbits.len() + 1);
    prefix.push(0);
    for orbit in orbits {
        let next = prefix
            .last()
            .copied()
            .unwrap_or(0usize)
            .checked_add(orbit.incumbent_rows_retained)
            .ok_or_else(|| "incumbent retained prefix overflow".to_string())?;
        prefix.push(next);
    }
    Ok(prefix)
}

#[derive(Clone, Debug)]
struct SelectionCandidate {
    fixed_count: usize,
    pair_count: usize,
    objective: f64,
    incumbent_rows_retained: usize,
    symmetric_difference_rows: usize,
    rows: Vec<u32>,
}

fn candidate_is_better(candidate: &SelectionCandidate, incumbent: &SelectionCandidate) -> bool {
    match candidate.objective.total_cmp(&incumbent.objective) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => {
            candidate
                .incumbent_rows_retained
                .cmp(&incumbent.incumbent_rows_retained)
                .then_with(|| {
                    incumbent
                        .symmetric_difference_rows
                        .cmp(&candidate.symmetric_difference_rows)
                })
                .then_with(|| incumbent.rows.cmp(&candidate.rows))
                == Ordering::Greater
        }
    }
}

fn select_scored_orbits_exact(
    mut fixed: Vec<ScoredOrbit>,
    mut pairs: Vec<ScoredOrbit>,
    capacity: usize,
    incumbent_row_count: usize,
) -> Result<SelectionCandidate, String> {
    fixed.sort_by(scored_orbit_order);
    pairs.sort_by(scored_orbit_order);
    let fixed_values = prefix_values(&fixed)?;
    let pair_values = prefix_values(&pairs)?;
    let fixed_retained = prefix_retained(&fixed)?;
    let pair_retained = prefix_retained(&pairs)?;

    let mut best: Option<SelectionCandidate> = None;
    for fixed_count in 0..=fixed.len().min(capacity) {
        let remaining = capacity - fixed_count;
        if remaining % 2 != 0 {
            continue;
        }
        let pair_count = remaining / 2;
        if pair_count > pairs.len() {
            continue;
        }
        let objective = fixed_values[fixed_count] + pair_values[pair_count];
        if !objective.is_finite() {
            return Err(format!(
                "non-finite selector objective at fixed={fixed_count}, pairs={pair_count}"
            ));
        }
        let incumbent_rows_retained = fixed_retained[fixed_count]
            .checked_add(pair_retained[pair_count])
            .ok_or_else(|| "selector retained-row overflow".to_string())?;
        let selected_plus_incumbent = capacity
            .checked_add(incumbent_row_count)
            .ok_or_else(|| "selector symmetric-difference overflow".to_string())?;
        let twice_retained = incumbent_rows_retained
            .checked_mul(2)
            .ok_or_else(|| "selector retained-row multiplication overflow".to_string())?;
        let symmetric_difference_rows = selected_plus_incumbent
            .checked_sub(twice_retained)
            .ok_or_else(|| "selector retained rows exceed set sizes".to_string())?;

        let mut rows = Vec::with_capacity(capacity);
        for orbit in fixed.iter().take(fixed_count) {
            orbit.orbit.visit_rows(|row| rows.push(row));
        }
        for orbit in pairs.iter().take(pair_count) {
            orbit.orbit.visit_rows(|row| rows.push(row));
        }
        rows.sort_unstable();
        if rows.len() != capacity || rows.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err("selector produced duplicate or wrong-capacity rows".to_string());
        }
        let candidate = SelectionCandidate {
            fixed_count,
            pair_count,
            objective,
            incumbent_rows_retained,
            symmetric_difference_rows,
            rows,
        };
        if best
            .as_ref()
            .map(|current| candidate_is_better(&candidate, current))
            .unwrap_or(true)
        {
            best = Some(candidate);
        }
    }
    best.ok_or_else(|| format!("no exact cost-{{1,2}} solution for capacity {capacity}"))
}

/// Exact registered selector output.
#[derive(Clone, Debug)]
pub(crate) struct VocabularySelection {
    pub(crate) capacity: usize,
    pub(crate) fixed_orbits: usize,
    pub(crate) pair_orbits: usize,
    /// Selected packed rows in ascending numeric order.
    pub(crate) rows: Vec<u32>,
    /// Prefix objective used by the registered exact optimizer.
    pub(crate) selector_objective: f64,
    /// `Phi(V*)`, re-summed over selected rows in ascending numeric order.
    pub(crate) phi_selected: f64,
    /// `Phi(V0)`, re-summed over incumbent rows in ascending numeric order.
    pub(crate) phi_incumbent: f64,
    pub(crate) incumbent_rows_retained: usize,
    pub(crate) symmetric_difference_rows: usize,
    pub(crate) gained_rows: Vec<u32>,
    pub(crate) lost_rows: Vec<u32>,
    pub(crate) gained_orbits: usize,
    pub(crate) lost_orbits: usize,
}

impl VocabularySelection {
    pub(crate) fn r_phi(&self) -> Result<f64, String> {
        if !self.phi_incumbent.is_finite() || self.phi_incumbent <= 0.0 {
            return Err(format!(
                "Phi(V0) must be finite and positive, got {}",
                self.phi_incumbent
            ));
        }
        let ratio = (self.phi_selected - self.phi_incumbent) / self.phi_incumbent;
        if ratio.is_finite() {
            Ok(ratio)
        } else {
            Err("R_phi is non-finite".to_string())
        }
    }
}

/// Solve the registered exact selector for values supplied by raw packed row.
///
/// `value_for` is called exactly once for every row in the 199,827-row
/// universe.  Unobserved rows should return zero.
pub(crate) fn select_exact_capacity(
    geometry: &OrbitGeometry,
    capacity: usize,
    mut value_for: impl FnMut(u32) -> f64,
) -> Result<VocabularySelection, String> {
    if geometry.universe.len() != RELEASED_UNIVERSE_ROWS
        || geometry.orbits.len() != RELEASED_UNIVERSE_ORBITS
    {
        return Err("selector requires strict released geometry".to_string());
    }

    let mut values = Vec::with_capacity(geometry.universe.len());
    for &row in &geometry.universe {
        let value = value_for(row);
        if !value.is_finite() || value < 0.0 {
            return Err(format!(
                "row value must be finite and non-negative: {row:08X} -> {value}"
            ));
        }
        values.push(value);
    }
    let value_at = |row: u32| -> f64 {
        let index = geometry
            .universe
            .binary_search(&row)
            .expect("validated orbit row belongs to universe");
        values[index]
    };

    let mut fixed = Vec::with_capacity(geometry.fixed_orbits);
    let mut pairs = Vec::with_capacity(geometry.pair_orbits);
    for &orbit in &geometry.orbits {
        let value = if orbit.is_fixed() {
            value_at(orbit.first)
        } else {
            neumaier_sum([value_at(orbit.first), value_at(orbit.second)])
        };
        if !value.is_finite() {
            return Err(format!(
                "non-finite orbit value: {:08X}/{:08X}",
                orbit.first, orbit.second
            ));
        }
        let mut incumbent_rows_retained = 0usize;
        orbit.visit_rows(|row| {
            incumbent_rows_retained += usize::from(geometry.incumbent.contains(row));
        });
        let scored = ScoredOrbit {
            orbit,
            value,
            incumbent_rows_retained,
        };
        if orbit.is_fixed() {
            fixed.push(scored);
        } else {
            pairs.push(scored);
        }
    }

    let selected =
        select_scored_orbits_exact(fixed, pairs, capacity, geometry.incumbent.rows_by_id.len())?;
    let selected_set: HashSet<u32> = selected.rows.iter().copied().collect();
    let gained_rows: Vec<u32> = selected
        .rows
        .iter()
        .copied()
        .filter(|&row| !geometry.incumbent.contains(row))
        .collect();
    let mut lost_rows: Vec<u32> = geometry
        .incumbent
        .rows_by_id
        .iter()
        .copied()
        .filter(|row| !selected_set.contains(row))
        .collect();
    lost_rows.sort_unstable();

    let gained_orbits = geometry
        .orbits
        .iter()
        .filter(|&&orbit| {
            selected_set.contains(&orbit.first) && !geometry.incumbent.contains(orbit.first)
        })
        .count();
    let lost_orbits = geometry
        .orbits
        .iter()
        .filter(|&&orbit| {
            geometry.incumbent.contains(orbit.first) && !selected_set.contains(&orbit.first)
        })
        .count();

    let phi_selected = neumaier_sum(selected.rows.iter().map(|&row| value_at(row)));
    let incumbent_sorted = geometry.incumbent.sorted_rows();
    let phi_incumbent = neumaier_sum(incumbent_sorted.iter().map(|&row| value_at(row)));
    if !phi_selected.is_finite() || !phi_incumbent.is_finite() {
        return Err("non-finite Phi value".to_string());
    }

    Ok(VocabularySelection {
        capacity,
        fixed_orbits: selected.fixed_count,
        pair_orbits: selected.pair_count,
        rows: selected.rows,
        selector_objective: selected.objective,
        phi_selected,
        phi_incumbent,
        incumbent_rows_retained: selected.incumbent_rows_retained,
        symmetric_difference_rows: selected.symmetric_difference_rows,
        gained_rows,
        lost_rows,
        gained_orbits,
        lost_orbits,
    })
}

/// Dense membership helper used by combined/color/ordinal slot censuses.
#[derive(Clone, Debug)]
pub(crate) struct SelectionMembership {
    incumbent: Vec<u8>,
    candidate: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MembershipClass {
    Neither,
    IncumbentOnly,
    CandidateOnly,
    Both,
}

impl SelectionMembership {
    pub(crate) fn new(
        incumbent: &IncumbentVocabulary,
        candidate_rows: &[u32],
    ) -> Result<Self, String> {
        let mut incumbent_dense = vec![0u8; RAW_SPACE];
        for &row in incumbent.rows_by_id() {
            if (row as usize) >= RAW_SPACE {
                return Err(format!("incumbent row is out of range: {row:08X}"));
            }
            incumbent_dense[row as usize] = 1;
        }
        let mut candidate = vec![0u8; RAW_SPACE];
        for &row in candidate_rows {
            if (row as usize) >= RAW_SPACE {
                return Err(format!("candidate row is out of range: {row:08X}"));
            }
            if candidate[row as usize] != 0 {
                return Err(format!("duplicate candidate row: {row:08X}"));
            }
            candidate[row as usize] = 1;
        }
        Ok(Self {
            incumbent: incumbent_dense,
            candidate,
        })
    }

    #[inline]
    pub(crate) fn classify(&self, row: u32) -> Result<MembershipClass, String> {
        if (row as usize) >= RAW_SPACE {
            return Err(format!("slot row is out of range: {row:08X}"));
        }
        Ok(
            match (
                self.incumbent[row as usize] != 0,
                self.candidate[row as usize] != 0,
            ) {
                (false, false) => MembershipClass::Neither,
                (true, false) => MembershipClass::IncumbentOnly,
                (false, true) => MembershipClass::CandidateOnly,
                (true, true) => MembershipClass::Both,
            },
        )
    }
}

/// Integer slot counts from which all registered addressability metrics follow.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct SelectionMetricAccumulator {
    pub(crate) total_slots: u64,
    pub(crate) incumbent_addressed: u64,
    pub(crate) candidate_addressed: u64,
    pub(crate) retained_addressed: u64,
    pub(crate) gained_addressed: u64,
    pub(crate) lost_addressed: u64,
}

impl SelectionMetricAccumulator {
    pub(crate) fn observe(
        &mut self,
        membership: &SelectionMembership,
        row: u32,
    ) -> Result<(), String> {
        self.observe_count(membership, row, 1)
    }

    pub(crate) fn observe_count(
        &mut self,
        membership: &SelectionMembership,
        row: u32,
        count: u64,
    ) -> Result<(), String> {
        self.total_slots = self
            .total_slots
            .checked_add(count)
            .ok_or_else(|| "slot total overflow".to_string())?;
        match membership.classify(row)? {
            MembershipClass::Neither => {}
            MembershipClass::IncumbentOnly => {
                self.incumbent_addressed = checked_add(self.incumbent_addressed, count)?;
                self.lost_addressed = checked_add(self.lost_addressed, count)?;
            }
            MembershipClass::CandidateOnly => {
                self.candidate_addressed = checked_add(self.candidate_addressed, count)?;
                self.gained_addressed = checked_add(self.gained_addressed, count)?;
            }
            MembershipClass::Both => {
                self.incumbent_addressed = checked_add(self.incumbent_addressed, count)?;
                self.candidate_addressed = checked_add(self.candidate_addressed, count)?;
                self.retained_addressed = checked_add(self.retained_addressed, count)?;
            }
        }
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<SelectionMetrics, String> {
        if self.total_slots == 0 {
            return Err("addressability stratum has zero slots".to_string());
        }
        if self.incumbent_addressed != self.retained_addressed + self.lost_addressed
            || self.candidate_addressed != self.retained_addressed + self.gained_addressed
        {
            return Err("addressability count decomposition failed".to_string());
        }
        let denominator = self.total_slots as f64;
        let incumbent_percent = 100.0 * self.incumbent_addressed as f64 / denominator;
        let candidate_percent = 100.0 * self.candidate_addressed as f64 / denominator;
        let gain_percentage_points = candidate_percent - incumbent_percent;
        let gross_loss_percentage_points = 100.0 * self.lost_addressed as f64 / denominator;
        Ok(SelectionMetrics {
            counts: self,
            incumbent_percent,
            candidate_percent,
            gain_percentage_points,
            gross_loss_percentage_points,
        })
    }
}

#[inline]
fn checked_add(left: u64, right: u64) -> Result<u64, String> {
    left.checked_add(right)
        .ok_or_else(|| "slot counter overflow".to_string())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SelectionMetrics {
    pub(crate) counts: SelectionMetricAccumulator,
    pub(crate) incumbent_percent: f64,
    pub(crate) candidate_percent: f64,
    pub(crate) gain_percentage_points: f64,
    pub(crate) gross_loss_percentage_points: f64,
}

/// Return selected rows as a deterministic ordered set for fold/Jaccard work.
pub(crate) fn selected_row_set(rows: &[u32]) -> BTreeSet<u32> {
    rows.iter().copied().collect()
}

pub(crate) fn jaccard_rows(left: &BTreeSet<u32>, right: &BTreeSet<u32>) -> Result<f64, String> {
    let union = left.union(right).count();
    if union == 0 {
        return Err("Jaccard is undefined for two empty vocabularies".to_string());
    }
    Ok(left.intersection(right).count() as f64 / union as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOPK: &[u8] = include_bytes!("../../data/topk.bin");

    #[test]
    fn released_geometry_matches_registered_invariants() {
        let geometry = released_geometry(TOPK).expect("released geometry");
        assert_eq!(geometry.universe.len(), 199_827);
        assert_eq!(geometry.anchor_boundary_rows, 537);
        assert_eq!(geometry.orbits.len(), 100_021);
        assert_eq!(geometry.fixed_orbits, 215);
        assert_eq!(geometry.pair_orbits, 99_806);
        assert_eq!(geometry.incumbent.rows_by_id().len(), 4_265);
        assert_eq!(geometry.incumbent.fixed_orbits, 29);
        assert_eq!(geometry.incumbent.pair_orbits, 2_118);

        for &row in &geometry.universe {
            let partner = color_swap_sigma(row);
            assert_eq!(color_swap_sigma(partner), row);
            assert!(geometry.contains_universe_row(partner));
            assert_eq!(canonicalize_packed(row), row);
            assert!(is_released_realizable(row));
        }
        for id in 0..INCUMBENT_FREQUENCY_ROWS {
            let row = geometry.incumbent.rows_by_id()[id];
            let partner = color_swap_sigma(row);
            let partner_id = geometry.incumbent.id_of(partner).expect("closed partner");
            assert_eq!(
                color_swap_sigma(geometry.incumbent.rows_by_id()[partner_id]),
                row
            );
        }
        for id in INCUMBENT_FREQUENCY_ROWS..INCUMBENT_ROWS {
            let row = geometry.incumbent.rows_by_id()[id];
            let partner_id = geometry
                .incumbent
                .id_of(color_swap_sigma(row))
                .expect("tail partner");
            assert!(partner_id < INCUMBENT_FREQUENCY_ROWS);
        }
    }

    #[test]
    fn exact_selector_matches_bruteforce_small_problem() {
        let incumbent: HashSet<u32> = [2u32, 10, 11, 14, 15].into_iter().collect();
        let fixed = vec![
            scored(
                ColorOrbit {
                    first: 1,
                    second: 1,
                },
                1.5,
                &incumbent,
            ),
            scored(
                ColorOrbit {
                    first: 2,
                    second: 2,
                },
                1.0,
                &incumbent,
            ),
            scored(
                ColorOrbit {
                    first: 3,
                    second: 3,
                },
                0.5,
                &incumbent,
            ),
        ];
        let pairs = vec![
            scored(
                ColorOrbit {
                    first: 10,
                    second: 11,
                },
                10.0,
                &incumbent,
            ),
            scored(
                ColorOrbit {
                    first: 12,
                    second: 13,
                },
                8.0,
                &incumbent,
            ),
            scored(
                ColorOrbit {
                    first: 14,
                    second: 15,
                },
                8.0,
                &incumbent,
            ),
        ];
        let exact =
            select_scored_orbits_exact(fixed.clone(), pairs.clone(), 5, incumbent.len()).unwrap();

        let all: Vec<ScoredOrbit> = fixed.into_iter().chain(pairs).collect();
        let mut brute: Option<SelectionCandidate> = None;
        for mask in 0usize..(1usize << all.len()) {
            let chosen: Vec<&ScoredOrbit> = all
                .iter()
                .enumerate()
                .filter_map(|(index, orbit)| ((mask >> index) & 1 == 1).then_some(orbit))
                .collect();
            if chosen.iter().map(|orbit| orbit.orbit.cost()).sum::<usize>() != 5 {
                continue;
            }
            let mut rows = Vec::new();
            let mut objective_sum = NeumaierSum::default();
            let mut retained = 0usize;
            let mut fixed_count = 0usize;
            let mut pair_count = 0usize;
            for orbit in chosen {
                objective_sum.add(orbit.value);
                retained += orbit.incumbent_rows_retained;
                if orbit.orbit.is_fixed() {
                    fixed_count += 1;
                } else {
                    pair_count += 1;
                }
                orbit.orbit.visit_rows(|row| rows.push(row));
            }
            rows.sort_unstable();
            let candidate = SelectionCandidate {
                fixed_count,
                pair_count,
                objective: objective_sum.total(),
                incumbent_rows_retained: retained,
                symmetric_difference_rows: 5 + incumbent.len() - 2 * retained,
                rows,
            };
            if brute
                .as_ref()
                .map(|current| candidate_is_better(&candidate, current))
                .unwrap_or(true)
            {
                brute = Some(candidate);
            }
        }
        let brute = brute.expect("brute-force feasible selection");
        assert_eq!(exact.rows, brute.rows);
        assert_eq!(exact.fixed_count, brute.fixed_count);
        assert_eq!(exact.pair_count, brute.pair_count);
        assert_eq!(exact.objective.to_bits(), brute.objective.to_bits());
        assert_eq!(exact.incumbent_rows_retained, brute.incumbent_rows_retained);
        assert_eq!(
            exact.symmetric_difference_rows,
            brute.symmetric_difference_rows
        );
        // The equal-value pair retaining incumbent rows must beat 12/13.
        assert!(exact.rows.contains(&14));
        assert!(exact.rows.contains(&15));
    }

    #[test]
    fn selection_metric_decomposition_is_exact() {
        let incumbent = synthetic_incumbent(&[1, 2]);
        let membership = SelectionMembership::new(&incumbent, &[2, 3]).unwrap();
        let mut metric = SelectionMetricAccumulator::default();
        metric.observe_count(&membership, 0, 2).unwrap();
        metric.observe_count(&membership, 1, 3).unwrap();
        metric.observe_count(&membership, 2, 5).unwrap();
        metric.observe_count(&membership, 3, 7).unwrap();
        let result = metric.finish().unwrap();
        assert_eq!(result.counts.total_slots, 17);
        assert_eq!(result.counts.incumbent_addressed, 8);
        assert_eq!(result.counts.candidate_addressed, 12);
        assert_eq!(result.counts.retained_addressed, 5);
        assert_eq!(result.counts.lost_addressed, 3);
        assert_eq!(result.counts.gained_addressed, 7);
        assert!((result.gain_percentage_points - 100.0 * 4.0 / 17.0).abs() < 1e-12);
        assert!((result.gross_loss_percentage_points - 100.0 * 3.0 / 17.0).abs() < 1e-12);
    }

    #[test]
    fn neumaier_recovers_small_residual() {
        assert_eq!(neumaier_sum([1.0e16, 1.0, -1.0e16]), 1.0);
    }

    fn scored(orbit: ColorOrbit, value: f64, incumbent: &HashSet<u32>) -> ScoredOrbit {
        let mut incumbent_rows_retained = 0usize;
        orbit.visit_rows(|row| {
            incumbent_rows_retained += usize::from(incumbent.contains(&row));
        });
        ScoredOrbit {
            orbit,
            value,
            incumbent_rows_retained,
        }
    }

    fn synthetic_incumbent(rows: &[u32]) -> IncumbentVocabulary {
        IncumbentVocabulary {
            rows_by_id: rows.to_vec(),
            id_by_row: rows
                .iter()
                .enumerate()
                .map(|(id, &row)| (row, id))
                .collect(),
            row_set: rows.iter().copied().collect(),
            fixed_orbits: 0,
            pair_orbits: 0,
        }
    }
}
