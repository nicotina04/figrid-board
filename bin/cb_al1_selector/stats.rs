//! Pure, deterministic statistics engine for the CB-AL1 P0B reveal.
//!
//! The caller owns file/schema replay and constructs the 500 answer-opaque
//! support units. This module validates the statistical contract, computes the
//! registered point gates, the finite-support random-control distribution, and
//! the selected-union dependence-cluster stress test. It never reads a file or
//! mutates product state.

use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) const SUPPORT_UNITS: usize = 500;
pub(crate) const UNITS_PER_ORDINAL: usize = 100;
pub(crate) const ARM_UNITS_PER_ORDINAL: usize = 25;
pub(crate) const ARM_UNITS: usize = 125;
pub(crate) const ARM_COLOR_SLOTS: usize = 250;
pub(crate) const RANDOM_CONTROL_REPLICATES: usize = 100_000;
pub(crate) const CLUSTER_REPLICATES: usize = 100_000;
pub(crate) const CLUSTER_MAX_ATTEMPTS: usize = 1_000_000;
pub(crate) const RANDOM_CONTROL_SEED: u64 = 2_026_727_102;
pub(crate) const CLUSTER_SEED: u64 = 2_026_727_101;

pub(crate) const GO_LABEL: &str = "GO_FRESH_AL1_PREREG_ONLY";
pub(crate) const NO_GO_SUPPORT_LABEL: &str = "NO_GO_MEASUREMENT_SUPPORT";
pub(crate) const NO_GO_UPPER_BOUND_LABEL: &str = "NO_GO_SELECTOR_UPPER_BOUND";

const ORDINALS: [u8; 5] = [1, 2, 4, 6, 8];
const MIN_COMPLETE_UNITS: u64 = 115;
const MIN_MEASURABLE_COLOR_SLOTS: u64 = 115;
const MIN_DEPENDENCE_CLUSTERS: usize = 30;
const MAX_ARM_UNITS_PER_CLUSTER: usize = 12;
const MAX_ARM_OVERLAP: usize = 50;
const MIN_DISTINCT_ARM_OPENINGS: usize = 63;

/// Array order used by [`Unit::observations`].
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum ChoiceFamily {
    StaticTop,
    ArchivedActual,
}

impl ChoiceFamily {
    pub(crate) const ALL: [Self; 2] = [Self::StaticTop, Self::ArchivedActual];

    pub(crate) const fn index(self) -> usize {
        match self {
            Self::StaticTop => 0,
            Self::ArchivedActual => 1,
        }
    }

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::StaticTop => "static_top",
            Self::ArchivedActual => "archived_actual",
        }
    }
}

/// Inner-array order used by [`Unit::observations`].
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum Color {
    Black,
    White,
}

impl Color {
    pub(crate) const ALL: [Self; 2] = [Self::Black, Self::White];

    pub(crate) const fn index(self) -> usize {
        match self {
            Self::Black => 0,
            Self::White => 1,
        }
    }

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Black => "black",
            Self::White => "white",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Observation {
    pub(crate) measurable: bool,
    pub(crate) error: bool,
    pub(crate) regret: f64,
    pub(crate) teacher_max_tied: bool,
}

impl Observation {
    pub(crate) const fn unmeasurable() -> Self {
        Self {
            measurable: false,
            error: false,
            regret: 0.0,
            teacher_max_tied: false,
        }
    }
}

/// One of the 500 frozen support units.
///
/// `observations[choice.index()][color.index()]` is the sole valid indexing
/// convention. `support_rank` is the zero-based rank inside this unit's
/// ordinal stratum under the preregistered support hash.
#[derive(Clone, Debug)]
pub(crate) struct Unit {
    pub(crate) uid: String,
    pub(crate) ordinal: u8,
    pub(crate) support_rank: u8,
    pub(crate) opening_group_hash: String,
    pub(crate) parent_d4_side_hashes: [String; 2],
    pub(crate) legal_child_d4_side_hashes: [Vec<String>; 2],
    pub(crate) matched_component_uid: Option<String>,
    pub(crate) complete_pair: bool,
    pub(crate) active: bool,
    pub(crate) deterministic_control: bool,
    pub(crate) observations: [[Observation; 2]; 2],
}

#[derive(Clone, Debug)]
pub(crate) struct AnalysisOutcome {
    pub(crate) final_label: &'static str,
    pub(crate) report: Value,
}

#[derive(Clone, Copy, Debug)]
struct AnalysisConfig {
    random_replicates: usize,
    cluster_replicates: usize,
    cluster_max_attempts: usize,
    random_seed: u64,
    cluster_seed: u64,
}

impl AnalysisConfig {
    const PRODUCTION: Self = Self {
        random_replicates: RANDOM_CONTROL_REPLICATES,
        cluster_replicates: CLUSTER_REPLICATES,
        cluster_max_attempts: CLUSTER_MAX_ATTEMPTS,
        random_seed: RANDOM_CONTROL_SEED,
        cluster_seed: CLUSTER_SEED,
    };
}

#[derive(Clone, Copy, Debug, Default)]
struct Neumaier {
    sum: f64,
    correction: f64,
}

impl Neumaier {
    fn add(&mut self, value: f64) {
        let next = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.correction += (self.sum - next) + value;
        } else {
            self.correction += (value - next) + self.sum;
        }
        self.sum = next;
    }

    fn total(self) -> f64 {
        self.sum + self.correction
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Metric {
    attempted: u64,
    measurable: u64,
    errors: u64,
    tied_teacher_maxima: u64,
    regret: Neumaier,
}

impl Metric {
    fn observe(&mut self, observation: Observation) -> Result<(), String> {
        self.attempted = checked_add(self.attempted, 1, "attempted-slot count")?;
        self.tied_teacher_maxima = checked_add(
            self.tied_teacher_maxima,
            u64::from(observation.teacher_max_tied),
            "tied-teacher-maximum count",
        )?;
        if observation.measurable {
            self.measurable = checked_add(self.measurable, 1, "measurable-slot count")?;
            self.errors = checked_add(
                self.errors,
                u64::from(observation.error),
                "teacher-error count",
            )?;
            self.regret.add(observation.regret);
        }
        Ok(())
    }

    fn regret_sum(self) -> Result<f64, String> {
        require_finite("regret sum", self.regret.total())
    }

    fn fixed_error_rate(self) -> Result<f64, String> {
        checked_rate(self.errors, self.attempted, "fixed usable-error discovery")
    }

    fn fixed_mean_regret(self) -> Result<f64, String> {
        checked_mean(
            self.regret_sum()?,
            self.attempted,
            "fixed usable mean regret",
        )
    }

    fn conditional_error_rate(self) -> Result<Option<f64>, String> {
        optional_rate(
            self.errors,
            self.measurable,
            "measurable-only conditional error",
        )
    }

    fn conditional_mean_regret(self) -> Result<Option<f64>, String> {
        optional_mean(
            self.regret_sum()?,
            self.measurable,
            "measurable-only conditional regret",
        )
    }

    fn json(self) -> Result<Value, String> {
        let fixed_error = self.fixed_error_rate()?;
        let conditional_error = self.conditional_error_rate()?;
        Ok(json!({
            "attempted_slots": self.attempted,
            "measurable_slots": self.measurable,
            "errors": self.errors,
            "tied_teacher_maxima": self.tied_teacher_maxima,
            "regret_sum": self.regret_sum()?,
            "fixed_slot_usable_error_discovery": fixed_error,
            "fixed_slot_usable_error_discovery_pp": 100.0 * fixed_error,
            "fixed_slot_usable_mean_regret": self.fixed_mean_regret()?,
            "measurable_only_conditional_error_rate": conditional_error,
            "measurable_only_conditional_error_rate_pp":
                conditional_error.map(|value| 100.0 * value),
            "measurable_only_conditional_mean_regret":
                self.conditional_mean_regret()?,
        }))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Arm {
    Active,
    Control,
}

impl Arm {
    const ALL: [Self; 2] = [Self::Active, Self::Control];

    const fn index(self) -> usize {
        match self {
            Self::Active => 0,
            Self::Control => 1,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Control => "deterministic_control",
        }
    }

    fn contains(self, unit: &Unit) -> bool {
        match self {
            Self::Active => unit.active,
            Self::Control => unit.deterministic_control,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Scope {
    Combined,
    Black,
    White,
}

impl Scope {
    const ALL: [Self; 3] = [Self::Combined, Self::Black, Self::White];
    const RANDOM: [Self; 2] = [Self::Combined, Self::White];

    const fn name(self) -> &'static str {
        match self {
            Self::Combined => "combined",
            Self::Black => "black",
            Self::White => "white",
        }
    }

    const fn random_index(self) -> usize {
        match self {
            Self::Combined => 0,
            Self::White => 1,
            Self::Black => panic!("Black is not a registered bootstrap scope"),
        }
    }
}

#[derive(Clone, Debug)]
struct ValidatedContext {
    fixed_order: Vec<usize>,
    strata: [Vec<usize>; 5],
    active_indices: Vec<usize>,
    control_indices: Vec<usize>,
    overlap_indices: Vec<usize>,
}

fn checked_add(left: u64, right: u64, label: &str) -> Result<u64, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("{label} overflow"))
}

fn require_finite(label: &str, value: f64) -> Result<f64, String> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("{label} is not finite: {value:?}"))
    }
}

fn checked_rate(numerator: u64, denominator: u64, label: &str) -> Result<f64, String> {
    if denominator == 0 {
        return Err(format!("{label} has a zero denominator"));
    }
    require_finite(label, numerator as f64 / denominator as f64)
}

fn optional_rate(numerator: u64, denominator: u64, label: &str) -> Result<Option<f64>, String> {
    if denominator == 0 {
        Ok(None)
    } else {
        checked_rate(numerator, denominator, label).map(Some)
    }
}

fn checked_mean(sum: f64, denominator: u64, label: &str) -> Result<f64, String> {
    if denominator == 0 {
        return Err(format!("{label} has a zero denominator"));
    }
    require_finite(label, sum / denominator as f64)
}

fn optional_mean(sum: f64, denominator: u64, label: &str) -> Result<Option<f64>, String> {
    if denominator == 0 {
        Ok(None)
    } else {
        checked_mean(sum, denominator, label).map(Some)
    }
}

fn ordinal_index(ordinal: u8) -> Option<usize> {
    ORDINALS
        .iter()
        .position(|&registered| registered == ordinal)
}

fn require_upper_hash(label: &str, value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'A'..=b'F').contains(&byte))
    {
        return Err(format!("{label} must be 64 uppercase hexadecimal bytes"));
    }
    Ok(())
}

fn validate_observation(
    uid: &str,
    choice: ChoiceFamily,
    color: Color,
    observation: Observation,
) -> Result<(), String> {
    require_finite(
        &format!(
            "observation regret {uid}/{}/{}",
            choice.name(),
            color.name()
        ),
        observation.regret,
    )?;
    if observation.regret < 0.0 {
        return Err(format!(
            "negative regret at {uid}/{}/{}",
            choice.name(),
            color.name()
        ));
    }
    if !observation.measurable {
        if observation.error || observation.regret.to_bits() != 0.0f64.to_bits() {
            return Err(format!(
                "unmeasurable observation carries a label at {uid}/{}/{}",
                choice.name(),
                color.name()
            ));
        }
    } else if observation.error != (observation.regret > 0.0) {
        return Err(format!(
            "error/regret disagreement at {uid}/{}/{}",
            choice.name(),
            color.name()
        ));
    }
    Ok(())
}

fn validate_units(units: &[Unit]) -> Result<ValidatedContext, String> {
    if units.len() != SUPPORT_UNITS {
        return Err(format!(
            "support has {} units, expected {SUPPORT_UNITS}",
            units.len()
        ));
    }

    let mut uids = BTreeSet::<String>::new();
    let mut strata: [Vec<usize>; 5] = std::array::from_fn(|_| Vec::new());
    for (index, unit) in units.iter().enumerate() {
        require_upper_hash("unit UID", &unit.uid)?;
        if !uids.insert(unit.uid.clone()) {
            return Err(format!("duplicate support UID {}", unit.uid));
        }
        let ordinal_slot = ordinal_index(unit.ordinal)
            .ok_or_else(|| format!("unregistered ordinal {} at {}", unit.ordinal, unit.uid))?;
        if usize::from(unit.support_rank) >= UNITS_PER_ORDINAL {
            return Err(format!(
                "support rank {} is out of range at {}",
                unit.support_rank, unit.uid
            ));
        }
        strata[ordinal_slot].push(index);

        require_upper_hash("opening_group_hash", &unit.opening_group_hash)?;
        for (color_index, parent) in unit.parent_d4_side_hashes.iter().enumerate() {
            require_upper_hash(
                &format!("parent_d4_side_hash[{color_index}] at {}", unit.uid),
                parent,
            )?;
        }
        for (color_index, children) in unit.legal_child_d4_side_hashes.iter().enumerate() {
            if children.is_empty() {
                return Err(format!(
                    "empty full-legal child identity list [{color_index}] at {}",
                    unit.uid
                ));
            }
            for child in children {
                require_upper_hash(
                    &format!("legal child hash [{color_index}] at {}", unit.uid),
                    child,
                )?;
            }
        }

        if unit.complete_pair != unit.matched_component_uid.is_some() {
            return Err(format!(
                "complete-pair/component presence mismatch at {}",
                unit.uid
            ));
        }
        if let Some(component) = &unit.matched_component_uid {
            require_upper_hash("matched component UID", component)?;
        }

        for choice in ChoiceFamily::ALL {
            for color in Color::ALL {
                let observation = unit.observations[choice.index()][color.index()];
                validate_observation(&unit.uid, choice, color, observation)?;
                if !unit.complete_pair && observation.measurable {
                    return Err(format!(
                        "incomplete pair is measurable at {}/{}/{}",
                        unit.uid,
                        choice.name(),
                        color.name()
                    ));
                }
                if !unit.complete_pair && observation.teacher_max_tied {
                    return Err(format!(
                        "incomplete pair carries a teacher-tie flag at {}/{}/{}",
                        unit.uid,
                        choice.name(),
                        color.name()
                    ));
                }
            }
        }
        for color in Color::ALL {
            let actual = unit.observations[ChoiceFamily::ArchivedActual.index()][color.index()];
            if actual.measurable != unit.complete_pair {
                return Err(format!(
                    "archived-actual measurability does not equal complete_pair at {}/{}",
                    unit.uid,
                    color.name()
                ));
            }
            let static_tied =
                unit.observations[ChoiceFamily::StaticTop.index()][color.index()].teacher_max_tied;
            if static_tied != actual.teacher_max_tied {
                return Err(format!(
                    "teacher-max tie flag differs by choice family at {}/{}",
                    unit.uid,
                    color.name()
                ));
            }
        }
    }

    for (slot, stratum) in strata.iter_mut().enumerate() {
        if stratum.len() != UNITS_PER_ORDINAL {
            return Err(format!(
                "ordinal {} has {} support units, expected {UNITS_PER_ORDINAL}",
                ORDINALS[slot],
                stratum.len()
            ));
        }
        stratum.sort_by_key(|&index| units[index].support_rank);
        for (expected_rank, &index) in stratum.iter().enumerate() {
            if usize::from(units[index].support_rank) != expected_rank {
                return Err(format!(
                    "ordinal {} support ranks are not exactly 0..99",
                    ORDINALS[slot]
                ));
            }
        }
        let active = stratum.iter().filter(|&&index| units[index].active).count();
        let control = stratum
            .iter()
            .filter(|&&index| units[index].deterministic_control)
            .count();
        if active != ARM_UNITS_PER_ORDINAL || control != ARM_UNITS_PER_ORDINAL {
            return Err(format!(
                "ordinal {} arm counts are active={active}, control={control}, expected 25/25",
                ORDINALS[slot]
            ));
        }
    }

    let mut fixed_order = (0..units.len()).collect::<Vec<_>>();
    fixed_order.sort_by(|&left, &right| {
        (units[left].ordinal, units[left].uid.as_str())
            .cmp(&(units[right].ordinal, units[right].uid.as_str()))
    });
    let active_indices = fixed_order
        .iter()
        .copied()
        .filter(|&index| units[index].active)
        .collect::<Vec<_>>();
    let control_indices = fixed_order
        .iter()
        .copied()
        .filter(|&index| units[index].deterministic_control)
        .collect::<Vec<_>>();
    let overlap_indices = fixed_order
        .iter()
        .copied()
        .filter(|&index| units[index].active && units[index].deterministic_control)
        .collect::<Vec<_>>();

    if active_indices.len() != ARM_UNITS || control_indices.len() != ARM_UNITS {
        return Err(format!(
            "arm counts are active={}, control={}, expected {ARM_UNITS}",
            active_indices.len(),
            control_indices.len()
        ));
    }
    if overlap_indices.len() > MAX_ARM_OVERLAP {
        return Err(format!(
            "active/control overlap {} exceeds {MAX_ARM_OVERLAP}",
            overlap_indices.len()
        ));
    }
    for arm in Arm::ALL {
        let indices = match arm {
            Arm::Active => &active_indices,
            Arm::Control => &control_indices,
        };
        let mut openings = BTreeMap::<&str, usize>::new();
        for &index in indices {
            *openings
                .entry(units[index].opening_group_hash.as_str())
                .or_default() += 1;
        }
        if openings.len() < MIN_DISTINCT_ARM_OPENINGS {
            return Err(format!(
                "{} has {} distinct openings, expected at least {MIN_DISTINCT_ARM_OPENINGS}",
                arm.name(),
                openings.len()
            ));
        }
        if let Some((opening, count)) = openings.iter().find(|(_, count)| **count > 2) {
            return Err(format!(
                "{} has {count} units for opening {opening}, maximum is 2",
                arm.name()
            ));
        }
    }

    Ok(ValidatedContext {
        fixed_order,
        strata,
        active_indices,
        control_indices,
        overlap_indices,
    })
}

fn aggregate(
    units: &[Unit],
    order: &[usize],
    arm: Arm,
    choice: ChoiceFamily,
    scope: Scope,
    ordinal: Option<u8>,
) -> Result<Metric, String> {
    let mut metric = Metric::default();
    for &index in order {
        let unit = &units[index];
        if !arm.contains(unit) || ordinal.is_some_and(|expected| expected != unit.ordinal) {
            continue;
        }
        match scope {
            Scope::Combined => {
                for color in Color::ALL {
                    metric.observe(unit.observations[choice.index()][color.index()])?;
                }
            }
            Scope::Black => {
                metric.observe(unit.observations[choice.index()][Color::Black.index()])?;
            }
            Scope::White => {
                metric.observe(unit.observations[choice.index()][Color::White.index()])?;
            }
        }
    }
    Ok(metric)
}

fn aggregate_component(
    units: &[Unit],
    order: &[usize],
    arm: Arm,
    choice: ChoiceFamily,
    scope: Scope,
    component_uid: &str,
) -> Result<Metric, String> {
    let mut metric = Metric::default();
    for &index in order {
        let unit = &units[index];
        if !arm.contains(unit) || unit.matched_component_uid.as_deref() != Some(component_uid) {
            continue;
        }
        match scope {
            Scope::Combined => {
                for color in Color::ALL {
                    metric.observe(unit.observations[choice.index()][color.index()])?;
                }
            }
            Scope::Black => {
                metric.observe(unit.observations[choice.index()][Color::Black.index()])?;
            }
            Scope::White => {
                metric.observe(unit.observations[choice.index()][Color::White.index()])?;
            }
        }
    }
    Ok(metric)
}

fn arm_report(units: &[Unit], context: &ValidatedContext, arm: Arm) -> Result<Value, String> {
    let indices = match arm {
        Arm::Active => &context.active_indices,
        Arm::Control => &context.control_indices,
    };
    let complete_units = indices
        .iter()
        .filter(|&&index| units[index].complete_pair)
        .count();
    let components = indices
        .iter()
        .filter_map(|&index| units[index].matched_component_uid.as_deref())
        .collect::<BTreeSet<_>>();
    let mut choices = Map::new();
    for choice in ChoiceFamily::ALL {
        let mut colors = Map::new();
        for scope in [Scope::Black, Scope::White] {
            colors.insert(
                scope.name().to_string(),
                aggregate(units, &context.fixed_order, arm, choice, scope, None)?.json()?,
            );
        }
        let mut ordinals = Map::new();
        for ordinal in ORDINALS {
            let mut ordinal_scopes = Map::new();
            let ordinal_complete_units = indices
                .iter()
                .filter(|&&index| units[index].ordinal == ordinal && units[index].complete_pair)
                .count();
            ordinal_scopes.insert(
                "complete_pair_units".to_string(),
                json!(ordinal_complete_units),
            );
            for scope in Scope::ALL {
                ordinal_scopes.insert(
                    scope.name().to_string(),
                    aggregate(
                        units,
                        &context.fixed_order,
                        arm,
                        choice,
                        scope,
                        Some(ordinal),
                    )?
                    .json()?,
                );
            }
            ordinals.insert(ordinal.to_string(), Value::Object(ordinal_scopes));
        }
        let mut component_reports = Map::new();
        for component_uid in &components {
            let mut component_scopes = Map::new();
            for scope in Scope::ALL {
                component_scopes.insert(
                    scope.name().to_string(),
                    aggregate_component(
                        units,
                        &context.fixed_order,
                        arm,
                        choice,
                        scope,
                        component_uid,
                    )?
                    .json()?,
                );
            }
            component_reports.insert(
                (*component_uid).to_string(),
                Value::Object(component_scopes),
            );
        }
        choices.insert(
            choice.name().to_string(),
            json!({
                "combined": aggregate(
                    units,
                    &context.fixed_order,
                    arm,
                    choice,
                    Scope::Combined,
                    None,
                )?.json()?,
                "colors": colors,
                "ordinals": ordinals,
                "components": component_reports,
            }),
        );
    }
    Ok(json!({
        "units": indices.len(),
        "color_slots": 2 * indices.len(),
        "complete_pair_units": complete_units,
        "matched_components": components.len(),
        "choices": choices,
    }))
}

#[derive(Default)]
struct GateCollector {
    records: Vec<Value>,
    coverage_pass: bool,
    decision_pass: bool,
}

impl GateCollector {
    fn new() -> Self {
        Self {
            records: Vec::new(),
            coverage_pass: true,
            decision_pass: true,
        }
    }

    fn add(
        &mut self,
        category: &'static str,
        name: impl Into<String>,
        observed: Value,
        threshold: impl Into<String>,
        pass: bool,
    ) {
        if category == "measurement_support" {
            self.coverage_pass &= pass;
        } else if category == "decision" {
            self.decision_pass &= pass;
        }
        self.records.push(json!({
            "category": category,
            "name": name.into(),
            "observed": observed,
            "threshold": threshold.into(),
            "pass": pass,
        }));
    }
}

fn conditional_error_gate(
    active_errors: u64,
    active_denominator: u64,
    control_errors: u64,
    control_denominator: u64,
    require_three_pp: bool,
) -> Result<bool, String> {
    if active_denominator == 0 || control_denominator == 0 {
        return Ok(false);
    }
    let err_a = u128::from(active_errors);
    let den_a = u128::from(active_denominator);
    let err_c = u128::from(control_errors);
    let den_c = u128::from(control_denominator);
    if require_three_pp {
        let left = 100u128
            .checked_mul(err_a)
            .and_then(|value| value.checked_mul(den_c))
            .ok_or_else(|| "conditional error gate left side overflow".to_string())?;
        let control_term = 100u128
            .checked_mul(err_c)
            .and_then(|value| value.checked_mul(den_a))
            .ok_or_else(|| "conditional error gate control term overflow".to_string())?;
        let margin_term = 3u128
            .checked_mul(den_a)
            .and_then(|value| value.checked_mul(den_c))
            .ok_or_else(|| "conditional error gate margin term overflow".to_string())?;
        let right = control_term
            .checked_add(margin_term)
            .ok_or_else(|| "conditional error gate right side overflow".to_string())?;
        Ok(left >= right)
    } else {
        let left = err_a
            .checked_mul(den_c)
            .ok_or_else(|| "Black conditional gate left side overflow".to_string())?;
        let right = err_c
            .checked_mul(den_a)
            .ok_or_else(|| "Black conditional gate right side overflow".to_string())?;
        Ok(left >= right)
    }
}

fn paired_fixed_difference(
    units: &[Unit],
    order: &[usize],
    choice: ChoiceFamily,
    scope: Scope,
) -> Result<(i64, f64), String> {
    let mut error_difference = 0i64;
    let mut regret_difference = Neumaier::default();
    for &index in order {
        let unit = &units[index];
        let weight = i8::from(unit.active) - i8::from(unit.deterministic_control);
        if weight == 0 {
            continue;
        }
        let colors: &[Color] = match scope {
            Scope::Combined => &Color::ALL,
            Scope::Black => &[Color::Black],
            Scope::White => &[Color::White],
        };
        for &color in colors {
            let observation = unit.observations[choice.index()][color.index()];
            let signed_error = i64::from(weight) * i64::from(observation.error);
            error_difference = error_difference
                .checked_add(signed_error)
                .ok_or_else(|| "paired fixed error difference overflow".to_string())?;
            regret_difference.add(f64::from(weight) * observation.regret);
        }
    }
    Ok((
        error_difference,
        require_finite("paired fixed regret difference", regret_difference.total())?,
    ))
}

fn difference(left: Option<f64>, right: Option<f64>, label: &str) -> Result<Option<f64>, String> {
    match (left, right) {
        (Some(left), Some(right)) => require_finite(label, left - right).map(Some),
        _ => Ok(None),
    }
}

fn point_analysis(
    units: &[Unit],
    context: &ValidatedContext,
    gates: &mut GateCollector,
) -> Result<Value, String> {
    let mut choice_reports = Map::new();
    for choice in ChoiceFamily::ALL {
        let mut scope_reports = Map::new();
        for scope in Scope::ALL {
            let active = aggregate(
                units,
                &context.fixed_order,
                Arm::Active,
                choice,
                scope,
                None,
            )?;
            let control = aggregate(
                units,
                &context.fixed_order,
                Arm::Control,
                choice,
                scope,
                None,
            )?;
            let separate_error_difference = i64::try_from(active.errors)
                .map_err(|_| "active error count does not fit i64".to_string())?
                - i64::try_from(control.errors)
                    .map_err(|_| "control error count does not fit i64".to_string())?;
            let (paired_error_difference, paired_regret_difference) =
                paired_fixed_difference(units, &context.fixed_order, choice, scope)?;
            if paired_error_difference != separate_error_difference {
                return Err(format!(
                    "overlap cancellation mismatch for {}/{}: paired={paired_error_difference}, separate={separate_error_difference}",
                    choice.name(),
                    scope.name()
                ));
            }

            let fixed_error_pass = match scope {
                Scope::Combined => paired_error_difference >= 8,
                Scope::White => paired_error_difference >= 4,
                Scope::Black => paired_error_difference >= 0,
            };
            gates.add(
                "decision",
                format!("{}_fixed_error_{}", choice.name(), scope.name()),
                json!({
                    "active_errors": active.errors,
                    "control_errors": control.errors,
                    "count_difference": paired_error_difference,
                    "delta_pp": 100.0
                        * paired_error_difference as f64
                        / active.attempted as f64,
                }),
                match scope {
                    Scope::Combined => "count_difference >= 8",
                    Scope::White => "count_difference >= 4",
                    Scope::Black => "count_difference >= 0",
                },
                fixed_error_pass,
            );

            let conditional_error_pass = conditional_error_gate(
                active.errors,
                active.measurable,
                control.errors,
                control.measurable,
                scope != Scope::Black,
            )?;
            let conditional_error_delta = difference(
                active.conditional_error_rate()?,
                control.conditional_error_rate()?,
                "conditional error-rate delta",
            )?;
            gates.add(
                "decision",
                format!(
                    "{}_measurable_only_conditional_error_{}",
                    choice.name(),
                    scope.name()
                ),
                json!({
                    "active_errors": active.errors,
                    "active_denominator": active.measurable,
                    "control_errors": control.errors,
                    "control_denominator": control.measurable,
                    "delta": conditional_error_delta,
                    "delta_pp": conditional_error_delta.map(|value| 100.0 * value),
                }),
                if scope == Scope::Black {
                    "err_A/den_A >= err_C/den_C"
                } else {
                    "err_A/den_A - err_C/den_C >= 3/100"
                },
                conditional_error_pass,
            );

            let fixed_regret_pass = match scope {
                Scope::Combined | Scope::White => paired_regret_difference > 0.0,
                Scope::Black => paired_regret_difference >= 0.0,
            };
            gates.add(
                "decision",
                format!("{}_fixed_regret_{}", choice.name(), scope.name()),
                json!({
                    "paired_regret_numerator_difference": paired_regret_difference,
                    "delta_mean": paired_regret_difference / active.attempted as f64,
                }),
                if scope == Scope::Black {
                    "active fixed mean regret >= control"
                } else {
                    "active fixed mean regret > control"
                },
                fixed_regret_pass,
            );

            let conditional_regret_delta = difference(
                active.conditional_mean_regret()?,
                control.conditional_mean_regret()?,
                "conditional mean-regret delta",
            )?;
            let conditional_regret_pass = conditional_regret_delta.is_some_and(|delta| {
                if scope == Scope::Black {
                    delta >= 0.0
                } else {
                    delta > 0.0
                }
            });
            gates.add(
                "decision",
                format!(
                    "{}_measurable_only_conditional_regret_{}",
                    choice.name(),
                    scope.name()
                ),
                json!({
                    "active_mean": active.conditional_mean_regret()?,
                    "control_mean": control.conditional_mean_regret()?,
                    "delta": conditional_regret_delta,
                }),
                if scope == Scope::Black {
                    "active conditional mean regret >= control"
                } else {
                    "active conditional mean regret > control"
                },
                conditional_regret_pass,
            );

            scope_reports.insert(
                scope.name().to_string(),
                json!({
                    "active": active.json()?,
                    "deterministic_control": control.json()?,
                    "fixed_error_count_difference": paired_error_difference,
                    "fixed_error_delta_pp":
                        100.0 * paired_error_difference as f64 / active.attempted as f64,
                    "paired_fixed_regret_numerator_difference": paired_regret_difference,
                    "fixed_mean_regret_delta":
                        paired_regret_difference / active.attempted as f64,
                    "conditional_error_rate_delta": conditional_error_delta,
                    "conditional_error_rate_delta_pp":
                        conditional_error_delta.map(|value| 100.0 * value),
                    "conditional_mean_regret_delta": conditional_regret_delta,
                    "overlap_contribution_to_paired_numerators": {
                        "error": 0,
                        "regret_bits": 0.0f64.to_bits(),
                        "pass": true,
                    },
                }),
            );
        }
        choice_reports.insert(choice.name().to_string(), Value::Object(scope_reports));
    }
    Ok(Value::Object(choice_reports))
}

fn membership_category_metric(
    units: &[Unit],
    order: &[usize],
    choice: ChoiceFamily,
    scope: Scope,
    ordinal: u8,
    active: bool,
    control: bool,
) -> Result<Metric, String> {
    let mut metric = Metric::default();
    for &index in order {
        let unit = &units[index];
        if unit.ordinal != ordinal || unit.active != active || unit.deterministic_control != control
        {
            continue;
        }
        match scope {
            Scope::Combined => {
                for color in Color::ALL {
                    metric.observe(unit.observations[choice.index()][color.index()])?;
                }
            }
            Scope::Black => {
                metric.observe(unit.observations[choice.index()][Color::Black.index()])?;
            }
            Scope::White => {
                metric.observe(unit.observations[choice.index()][Color::White.index()])?;
            }
        }
    }
    Ok(metric)
}

fn metric_json_allow_empty(metric: Metric) -> Result<Value, String> {
    if metric.attempted != 0 {
        return metric.json();
    }
    if metric.measurable != 0
        || metric.errors != 0
        || metric.tied_teacher_maxima != 0
        || metric.regret_sum()?.to_bits() != 0.0f64.to_bits()
    {
        return Err("empty membership metric carries observations".to_string());
    }
    Ok(json!({
        "attempted_slots": 0,
        "measurable_slots": 0,
        "errors": 0,
        "tied_teacher_maxima": 0,
        "regret_sum": 0.0,
        "fixed_slot_usable_error_discovery": Value::Null,
        "fixed_slot_usable_error_discovery_pp": Value::Null,
        "fixed_slot_usable_mean_regret": Value::Null,
        "measurable_only_conditional_error_rate": Value::Null,
        "measurable_only_conditional_error_rate_pp": Value::Null,
        "measurable_only_conditional_mean_regret": Value::Null,
    }))
}

fn membership_report(units: &[Unit], context: &ValidatedContext) -> Result<Value, String> {
    let mut by_ordinal = Map::new();
    for ordinal in ORDINALS {
        let mut active_only = 0usize;
        let mut control_only = 0usize;
        let mut overlap = 0usize;
        let mut neither = 0usize;
        for unit in units.iter().filter(|unit| unit.ordinal == ordinal) {
            match (unit.active, unit.deterministic_control) {
                (true, false) => active_only += 1,
                (false, true) => control_only += 1,
                (true, true) => overlap += 1,
                (false, false) => neither += 1,
            }
        }
        by_ordinal.insert(
            ordinal.to_string(),
            json!({
                "active_only": active_only,
                "control_only": control_only,
                "overlap": overlap,
                "neither": neither,
            }),
        );
    }
    let categories = [
        ("active_only", true, false),
        ("control_only", false, true),
        ("overlap", true, true),
        ("neither", false, false),
    ];
    let mut choices = Map::new();
    for choice in ChoiceFamily::ALL {
        let mut choice_ordinals = Map::new();
        for ordinal in ORDINALS {
            let mut ordinal_categories = Map::new();
            for (name, active, control) in categories {
                let mut scopes = Map::new();
                for scope in Scope::ALL {
                    let metric = membership_category_metric(
                        units,
                        &context.fixed_order,
                        choice,
                        scope,
                        ordinal,
                        active,
                        control,
                    )?;
                    scopes.insert(scope.name().to_string(), metric_json_allow_empty(metric)?);
                }
                ordinal_categories.insert(name.to_string(), Value::Object(scopes));
            }
            choice_ordinals.insert(ordinal.to_string(), Value::Object(ordinal_categories));
        }
        choices.insert(choice.name().to_string(), Value::Object(choice_ordinals));
    }
    Ok(json!({
        "active_units": context.active_indices.len(),
        "control_units": context.control_indices.len(),
        "overlap_units": context.overlap_indices.len(),
        "active_only": context.active_indices.len() - context.overlap_indices.len(),
        "control_only": context.control_indices.len() - context.overlap_indices.len(),
        "neither": units.len()
            - (context.active_indices.len() + context.control_indices.len()
                - context.overlap_indices.len()),
        "by_ordinal": by_ordinal,
        "choice_family_by_ordinal_and_membership": choices,
        "paired_numerator_cancellation_contract": true,
    }))
}

#[derive(Clone)]
struct Sha256 {
    state: [u32; 8],
    block: [u8; 64],
    block_len: usize,
    message_len: u64,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6A09_E667,
                0xBB67_AE85,
                0x3C6E_F372,
                0xA54F_F53A,
                0x510E_527F,
                0x9B05_688C,
                0x1F83_D9AB,
                0x5BE0_CD19,
            ],
            block: [0; 64],
            block_len: 0,
            message_len: 0,
        }
    }

    fn update(&mut self, mut bytes: &[u8]) -> Result<(), String> {
        self.message_len = self
            .message_len
            .checked_add(
                u64::try_from(bytes.len())
                    .map_err(|_| "SHA-256 input length does not fit u64".to_string())?,
            )
            .ok_or_else(|| "SHA-256 message length overflow".to_string())?;
        if self.block_len != 0 {
            let take = (64 - self.block_len).min(bytes.len());
            self.block[self.block_len..self.block_len + take].copy_from_slice(&bytes[..take]);
            self.block_len += take;
            bytes = &bytes[take..];
            if self.block_len == 64 {
                let block = self.block;
                self.compress(&block);
                self.block_len = 0;
            }
        }
        while bytes.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&bytes[..64]);
            self.compress(&block);
            bytes = &bytes[64..];
        }
        if !bytes.is_empty() {
            self.block[..bytes.len()].copy_from_slice(bytes);
            self.block_len = bytes.len();
        }
        Ok(())
    }

    fn finalize(mut self) -> Result<[u8; 32], String> {
        let bit_len = self
            .message_len
            .checked_mul(8)
            .ok_or_else(|| "SHA-256 bit length overflow".to_string())?;
        self.block[self.block_len] = 0x80;
        self.block_len += 1;
        if self.block_len > 56 {
            self.block[self.block_len..].fill(0);
            let block = self.block;
            self.compress(&block);
            self.block = [0; 64];
            self.block_len = 0;
        }
        self.block[self.block_len..56].fill(0);
        self.block[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.block;
        self.compress(&block);

        let mut digest = [0u8; 32];
        for (chunk, word) in digest.chunks_exact_mut(4).zip(self.state) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        Ok(digest)
    }

    fn compress(&mut self, block: &[u8; 64]) {
        const K: [u32; 64] = [
            0x428A_2F98,
            0x7137_4491,
            0xB5C0_FBCF,
            0xE9B5_DBA5,
            0x3956_C25B,
            0x59F1_11F1,
            0x923F_82A4,
            0xAB1C_5ED5,
            0xD807_AA98,
            0x1283_5B01,
            0x2431_85BE,
            0x550C_7DC3,
            0x72BE_5D74,
            0x80DE_B1FE,
            0x9BDC_06A7,
            0xC19B_F174,
            0xE49B_69C1,
            0xEFBE_4786,
            0x0FC1_9DC6,
            0x240C_A1CC,
            0x2DE9_2C6F,
            0x4A74_84AA,
            0x5CB0_A9DC,
            0x76F9_88DA,
            0x983E_5152,
            0xA831_C66D,
            0xB003_27C8,
            0xBF59_7FC7,
            0xC6E0_0BF3,
            0xD5A7_9147,
            0x06CA_6351,
            0x1429_2967,
            0x27B7_0A85,
            0x2E1B_2138,
            0x4D2C_6DFC,
            0x5338_0D13,
            0x650A_7354,
            0x766A_0ABB,
            0x81C2_C92E,
            0x9272_2C85,
            0xA2BF_E8A1,
            0xA81A_664B,
            0xC24B_8B70,
            0xC76C_51A3,
            0xD192_E819,
            0xD699_0624,
            0xF40E_3585,
            0x106A_A070,
            0x19A4_C116,
            0x1E37_6C08,
            0x2748_774C,
            0x34B0_BCB5,
            0x391C_0CB3,
            0x4ED8_AA4A,
            0x5B9C_CA4F,
            0x682E_6FF3,
            0x748F_82EE,
            0x78A5_636F,
            0x84C8_7814,
            0x8CC7_0208,
            0x90BE_FFFA,
            0xA450_6CEB,
            0xBEF9_A3F7,
            0xC671_78F2,
        ];

        let mut schedule = [0u32; 64];
        for (index, chunk) in block.chunks_exact(4).enumerate() {
            schedule[index] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for index in 16..64 {
            let s0 = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let s1 = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(s0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for index in 0..64 {
            let upper_e = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(upper_e)
                .wrapping_add(choose)
                .wrapping_add(K[index])
                .wrapping_add(schedule[index]);
            let upper_a = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = upper_a.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }
}

fn upper_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut output = String::with_capacity(2 * bytes.len());
    for &byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0F)]));
    }
    output
}

fn sha256_hex(bytes: &[u8]) -> Result<String, String> {
    let mut hasher = Sha256::new();
    hasher.update(bytes)?;
    Ok(upper_hex(&hasher.finalize()?))
}

struct AuditedSplitMix64 {
    state: u64,
    draws: u64,
    stream: Sha256,
}

impl AuditedSplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            draws: 0,
            stream: Sha256::new(),
        }
    }

    fn next_u64(&mut self) -> Result<u64, String> {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let output = z ^ (z >> 31);
        self.draws = self
            .draws
            .checked_add(1)
            .ok_or_else(|| "SplitMix64 draw count overflow".to_string())?;
        self.stream.update(&output.to_le_bytes())?;
        Ok(output)
    }

    fn bounded(&mut self, n: usize) -> Result<usize, String> {
        if n == 0 {
            return Err("bounded SplitMix64 called with n=0".to_string());
        }
        let bound = u64::try_from(n).map_err(|_| "bound does not fit u64".to_string())?;
        let threshold = 0u64.wrapping_sub(bound) % bound;
        loop {
            let value = self.next_u64()?;
            if value >= threshold {
                return usize::try_from(value % bound)
                    .map_err(|_| "bounded SplitMix64 result does not fit usize".to_string());
            }
        }
    }

    fn audit_json(&self, seed: u64) -> Result<Value, String> {
        Ok(json!({
            "algorithm": "SplitMix64 with rejection-based unbiased bounded sampler",
            "seed": seed,
            "raw_u64_draws": self.draws,
            "raw_u64_le_stream_sha256": upper_hex(&self.stream.clone().finalize()?),
        }))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SeriesMetric {
    ErrorDiscovery,
    MeanRegret,
}

impl SeriesMetric {
    const ALL: [Self; 2] = [Self::ErrorDiscovery, Self::MeanRegret];

    const fn index(self) -> usize {
        match self {
            Self::ErrorDiscovery => 0,
            Self::MeanRegret => 1,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::ErrorDiscovery => "fixed_slot_usable_error_discovery",
            Self::MeanRegret => "fixed_slot_usable_mean_regret",
        }
    }
}

struct SeriesBank {
    values: [Vec<f64>; 8],
}

impl SeriesBank {
    fn new(capacity: usize) -> Self {
        Self {
            values: std::array::from_fn(|_| Vec::with_capacity(capacity)),
        }
    }

    fn index(choice: ChoiceFamily, scope: Scope, metric: SeriesMetric) -> usize {
        choice.index() * 4 + scope.random_index() * 2 + metric.index()
    }

    fn push(
        &mut self,
        choice: ChoiceFamily,
        scope: Scope,
        metric: SeriesMetric,
        value: f64,
    ) -> Result<(), String> {
        require_finite(
            &format!(
                "bootstrap series {}/{}/{}",
                choice.name(),
                scope.name(),
                metric.name()
            ),
            value,
        )?;
        self.values[Self::index(choice, scope, metric)].push(value);
        Ok(())
    }

    fn get(&self, choice: ChoiceFamily, scope: Scope, metric: SeriesMetric) -> &[f64] {
        &self.values[Self::index(choice, scope, metric)]
    }
}

fn hash_f64_stream(values: &[f64]) -> Result<String, String> {
    let mut hasher = Sha256::new();
    for &value in values {
        require_finite("quantile input", value)?;
        hasher.update(&value.to_bits().to_le_bytes())?;
    }
    Ok(upper_hex(&hasher.finalize()?))
}

fn quantile_index(length: usize, numerator: usize, denominator: usize) -> Result<usize, String> {
    if length == 0 || denominator == 0 || numerator > denominator {
        return Err("invalid quantile request".to_string());
    }
    (length - 1)
        .checked_mul(numerator)
        .map(|value| value / denominator)
        .ok_or_else(|| "quantile index overflow".to_string())
}

#[derive(Clone, Copy)]
struct SeriesSummary {
    p10: f64,
}

fn series_report(
    values: &[f64],
    expected: usize,
    error_metric: bool,
) -> Result<(Value, SeriesSummary), String> {
    if values.len() != expected {
        return Err(format!(
            "quantile vector has {} values, expected {expected}",
            values.len()
        ));
    }
    let generation_hash = hash_f64_stream(values)?;
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let sorted_hash = hash_f64_stream(&sorted)?;
    let p10_index = quantile_index(sorted.len(), 1, 10)?;
    let p50_index = quantile_index(sorted.len(), 1, 2)?;
    let p90_index = quantile_index(sorted.len(), 9, 10)?;
    let p10 = sorted[p10_index];
    let mut report = json!({
        "count": sorted.len(),
        "generation_order_f64_le_sha256": generation_hash,
        "sorted_quantile_input_f64_le_sha256": sorted_hash,
        "minimum": sorted[0],
        "p10_index_zero_based": p10_index,
        "p10": p10,
        "p50": sorted[p50_index],
        "p90": sorted[p90_index],
        "maximum": sorted[sorted.len() - 1],
    });
    if error_metric {
        let object = report
            .as_object_mut()
            .ok_or_else(|| "series report is not an object".to_string())?;
        object.insert("p10_pp".to_string(), json!(100.0 * p10));
    }
    Ok((report, SeriesSummary { p10 }))
}

#[derive(Clone, Copy)]
struct P10Matrix {
    values: [[[f64; 2]; 2]; 2],
}

impl P10Matrix {
    fn from_bank(bank: &SeriesBank, expected: usize) -> Result<(Value, Self), String> {
        let mut values = [[[0.0; 2]; 2]; 2];
        let mut choices = Map::new();
        for choice in ChoiceFamily::ALL {
            let mut scopes = Map::new();
            for scope in Scope::RANDOM {
                let mut metrics = Map::new();
                for metric in SeriesMetric::ALL {
                    let (report, summary) = series_report(
                        bank.get(choice, scope, metric),
                        expected,
                        metric == SeriesMetric::ErrorDiscovery,
                    )?;
                    values[choice.index()][scope.random_index()][metric.index()] = summary.p10;
                    metrics.insert(metric.name().to_string(), report);
                }
                scopes.insert(scope.name().to_string(), Value::Object(metrics));
            }
            choices.insert(choice.name().to_string(), Value::Object(scopes));
        }
        Ok((Value::Object(choices), Self { values }))
    }

    fn get(self, choice: ChoiceFamily, scope: Scope, metric: SeriesMetric) -> f64 {
        self.values[choice.index()][scope.random_index()][metric.index()]
    }
}

fn signed_active_minus_mask(
    units: &[Unit],
    order: &[usize],
    other_mask: &[bool],
) -> Result<([[i64; 2]; 2], [[f64; 2]; 2]), String> {
    if other_mask.len() != units.len() {
        return Err("comparison mask length mismatch".to_string());
    }
    let mut errors = [[0i64; 2]; 2];
    let mut regrets = [[Neumaier::default(); 2]; 2];
    for &index in order {
        let unit = &units[index];
        let weight = i8::from(unit.active) - i8::from(other_mask[index]);
        if weight == 0 {
            continue;
        }
        for choice in ChoiceFamily::ALL {
            let black = unit.observations[choice.index()][Color::Black.index()];
            let white = unit.observations[choice.index()][Color::White.index()];
            let combined_errors = i64::from(black.error) + i64::from(white.error);
            errors[choice.index()][Scope::Combined.random_index()] = errors[choice.index()]
                [Scope::Combined.random_index()]
            .checked_add(i64::from(weight) * combined_errors)
            .ok_or_else(|| "random-control combined error difference overflow".to_string())?;
            errors[choice.index()][Scope::White.random_index()] = errors[choice.index()]
                [Scope::White.random_index()]
            .checked_add(i64::from(weight) * i64::from(white.error))
            .ok_or_else(|| "random-control White error difference overflow".to_string())?;
            regrets[choice.index()][Scope::Combined.random_index()]
                .add(f64::from(weight) * black.regret);
            regrets[choice.index()][Scope::Combined.random_index()]
                .add(f64::from(weight) * white.regret);
            regrets[choice.index()][Scope::White.random_index()]
                .add(f64::from(weight) * white.regret);
        }
    }
    let mut regret_totals = [[0.0; 2]; 2];
    for choice in ChoiceFamily::ALL {
        for scope in Scope::RANDOM {
            regret_totals[choice.index()][scope.random_index()] = require_finite(
                "random-control signed regret difference",
                regrets[choice.index()][scope.random_index()].total(),
            )?;
        }
    }
    Ok((errors, regret_totals))
}

struct BootstrapResult {
    report: Value,
    p10: Option<P10Matrix>,
    completed: bool,
}

fn finite_support_random_controls(
    units: &[Unit],
    context: &ValidatedContext,
    config: AnalysisConfig,
) -> Result<BootstrapResult, String> {
    if config.random_replicates == 0 {
        return Err("random-control replicate count must be nonzero".to_string());
    }
    let mut rng = AuditedSplitMix64::new(config.random_seed);
    let mut bank = SeriesBank::new(config.random_replicates);
    let mut selected_stream = Sha256::new();
    let mut selected = vec![false; units.len()];

    for replicate in 0..config.random_replicates {
        for selected_value in &mut selected {
            *selected_value = false;
        }
        for stratum in &context.strata {
            let mut work = [0usize; UNITS_PER_ORDINAL];
            work.copy_from_slice(stratum);
            for index in 0..ARM_UNITS_PER_ORDINAL {
                let swap_index = index + rng.bounded(UNITS_PER_ORDINAL - index)?;
                work.swap(index, swap_index);
            }
            for &unit_index in &work[..ARM_UNITS_PER_ORDINAL] {
                if std::mem::replace(&mut selected[unit_index], true) {
                    return Err(format!(
                        "random control selected duplicate unit {}",
                        units[unit_index].uid
                    ));
                }
                selected_stream.update(units[unit_index].uid.as_bytes())?;
                selected_stream.update(b"\n")?;
            }
        }
        let selected_count = selected.iter().filter(|&&value| value).count();
        if selected_count != ARM_UNITS {
            return Err(format!(
                "random control replicate {replicate} has {selected_count} units"
            ));
        }

        let (error_differences, regret_differences) =
            signed_active_minus_mask(units, &context.fixed_order, &selected)?;
        for choice in ChoiceFamily::ALL {
            for scope in Scope::RANDOM {
                let denominator = match scope {
                    Scope::Combined => ARM_COLOR_SLOTS as f64,
                    Scope::White => ARM_UNITS as f64,
                    Scope::Black => unreachable!(),
                };
                bank.push(
                    choice,
                    scope,
                    SeriesMetric::ErrorDiscovery,
                    error_differences[choice.index()][scope.random_index()] as f64 / denominator,
                )?;
                bank.push(
                    choice,
                    scope,
                    SeriesMetric::MeanRegret,
                    regret_differences[choice.index()][scope.random_index()] / denominator,
                )?;
            }
        }
    }

    let (series, p10) = P10Matrix::from_bank(&bank, config.random_replicates)?;
    Ok(BootstrapResult {
        report: json!({
            "interpretation":
                "finite-support percentile rank; not a population CI, p-value, or causal estimate",
            "replicates": config.random_replicates,
            "units_per_ordinal_per_control": ARM_UNITS_PER_ORDINAL,
            "control_units_per_replicate": ARM_UNITS,
            "rng": rng.audit_json(config.random_seed)?,
            "selected_control_stream": {
                "encoding":
                    "uppercase UID ASCII plus LF, direct append in replicate then ordinal [1,2,4,6,8] then partial-Fisher-Yates selected index 0..24 order",
                "sha256": upper_hex(&selected_stream.finalize()?),
            },
            "series": series,
        }),
        p10: Some(p10),
        completed: true,
    })
}

#[derive(Clone, Debug)]
struct Dsu {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl Dsu {
    fn new(length: usize) -> Self {
        Self {
            parent: (0..length).collect(),
            rank: vec![0; length],
        }
    }

    fn find(&mut self, index: usize) -> usize {
        if self.parent[index] != index {
            let root = self.find(self.parent[index]);
            self.parent[index] = root;
        }
        self.parent[index]
    }

    fn union(&mut self, left: usize, right: usize) {
        let left_root = self.find(left);
        let right_root = self.find(right);
        if left_root == right_root {
            return;
        }
        match self.rank[left_root].cmp(&self.rank[right_root]) {
            std::cmp::Ordering::Less => self.parent[left_root] = right_root,
            std::cmp::Ordering::Greater => self.parent[right_root] = left_root,
            std::cmp::Ordering::Equal => {
                self.parent[right_root] = left_root;
                self.rank[left_root] = self.rank[left_root].saturating_add(1);
            }
        }
    }
}

#[derive(Clone, Debug)]
struct DependenceCluster {
    key: String,
    members: Vec<usize>,
}

#[derive(Clone, Debug)]
struct ClusterSet {
    clusters: Vec<DependenceCluster>,
    active_cluster_count: usize,
    control_cluster_count: usize,
    maximum_active_units: usize,
    maximum_control_units: usize,
}

fn connect_identity(
    owners: &mut BTreeMap<String, usize>,
    dsu: &mut Dsu,
    token: String,
    local_index: usize,
) {
    if let Some(&owner) = owners.get(&token) {
        dsu.union(owner, local_index);
    } else {
        owners.insert(token, local_index);
    }
}

fn build_dependence_clusters(
    units: &[Unit],
    context: &ValidatedContext,
) -> Result<ClusterSet, String> {
    let selected = context
        .fixed_order
        .iter()
        .copied()
        .filter(|&index| units[index].active || units[index].deterministic_control)
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err("selected-arm union is empty".to_string());
    }
    let mut dsu = Dsu::new(selected.len());
    let mut owners = BTreeMap::<String, usize>::new();
    for (local_index, &global_index) in selected.iter().enumerate() {
        let unit = &units[global_index];
        connect_identity(
            &mut owners,
            &mut dsu,
            format!("O:{}", unit.opening_group_hash),
            local_index,
        );
        for parent in &unit.parent_d4_side_hashes {
            connect_identity(&mut owners, &mut dsu, format!("S:{parent}"), local_index);
        }
        for children in &unit.legal_child_d4_side_hashes {
            for child in children {
                connect_identity(&mut owners, &mut dsu, format!("S:{child}"), local_index);
            }
        }
        if let Some(component) = &unit.matched_component_uid {
            connect_identity(&mut owners, &mut dsu, format!("C:{component}"), local_index);
        }
    }

    let mut groups = BTreeMap::<usize, Vec<usize>>::new();
    for (local_index, &global_index) in selected.iter().enumerate() {
        groups
            .entry(dsu.find(local_index))
            .or_default()
            .push(global_index);
    }
    let mut clusters = Vec::with_capacity(groups.len());
    let mut keys = BTreeSet::new();
    for mut members in groups.into_values() {
        members.sort_by(|&left, &right| {
            (units[left].ordinal, units[left].uid.as_str())
                .cmp(&(units[right].ordinal, units[right].uid.as_str()))
        });
        let mut member_uids = members
            .iter()
            .map(|&index| units[index].uid.as_str())
            .collect::<Vec<_>>();
        member_uids.sort_unstable();
        let preimage = format!("CB-AL1|cluster-v1|{}", member_uids.as_slice().join(","));
        let key = sha256_hex(preimage.as_bytes())?;
        if !keys.insert(key.clone()) {
            return Err(format!("duplicate dependence-cluster key {key}"));
        }
        clusters.push(DependenceCluster { key, members });
    }
    clusters.sort_by(|left, right| left.key.cmp(&right.key));

    let mut active_cluster_count = 0usize;
    let mut control_cluster_count = 0usize;
    let mut maximum_active_units = 0usize;
    let mut maximum_control_units = 0usize;
    for cluster in &clusters {
        let active = cluster
            .members
            .iter()
            .filter(|&&index| units[index].active)
            .count();
        let control = cluster
            .members
            .iter()
            .filter(|&&index| units[index].deterministic_control)
            .count();
        active_cluster_count += usize::from(active != 0);
        control_cluster_count += usize::from(control != 0);
        maximum_active_units = maximum_active_units.max(active);
        maximum_control_units = maximum_control_units.max(control);
    }

    Ok(ClusterSet {
        clusters,
        active_cluster_count,
        control_cluster_count,
        maximum_active_units,
        maximum_control_units,
    })
}

fn cluster_membership_json(units: &[Unit], clusters: &ClusterSet) -> Value {
    Value::Array(
        clusters
            .clusters
            .iter()
            .map(|cluster| {
                let active_units = cluster
                    .members
                    .iter()
                    .filter(|&&index| units[index].active)
                    .count();
                let control_units = cluster
                    .members
                    .iter()
                    .filter(|&&index| units[index].deterministic_control)
                    .count();
                let mut member_uids = cluster
                    .members
                    .iter()
                    .map(|&index| units[index].uid.as_str())
                    .collect::<Vec<_>>();
                member_uids.sort_unstable();
                json!({
                    "cluster_key": cluster.key,
                    "member_uids_sorted_for_key": member_uids,
                    "active_units": active_units,
                    "deterministic_control_units": control_units,
                })
            })
            .collect(),
    )
}

fn dependence_cluster_stress(
    units: &[Unit],
    clusters: &ClusterSet,
    config: AnalysisConfig,
) -> Result<BootstrapResult, String> {
    if config.cluster_replicates == 0 || config.cluster_max_attempts < config.cluster_replicates {
        return Err("invalid cluster replicate/attempt contract".to_string());
    }
    let cluster_count = clusters.clusters.len();
    if cluster_count == 0 {
        return Err("dependence cluster set is empty".to_string());
    }

    let mut rng = AuditedSplitMix64::new(config.cluster_seed);
    let mut sample_stream = Sha256::new();
    let mut bank = SeriesBank::new(config.cluster_replicates);
    let mut accepted = 0usize;
    let mut attempts = 0usize;

    while accepted < config.cluster_replicates && attempts < config.cluster_max_attempts {
        let mut denominators = [[0u64; 2]; 2];
        let mut errors = [[[0u64; 2]; 2]; 2];
        let mut regrets = [[[Neumaier::default(); 2]; 2]; 2];

        for _ in 0..cluster_count {
            let sampled = rng.bounded(cluster_count)?;
            sample_stream.update(
                &u64::try_from(sampled)
                    .map_err(|_| "cluster index does not fit u64".to_string())?
                    .to_le_bytes(),
            )?;
            for &unit_index in &clusters.clusters[sampled].members {
                let unit = &units[unit_index];
                for arm in Arm::ALL {
                    if !arm.contains(unit) {
                        continue;
                    }
                    denominators[arm.index()][Scope::Combined.random_index()] = checked_add(
                        denominators[arm.index()][Scope::Combined.random_index()],
                        2,
                        "cluster combined attempted denominator",
                    )?;
                    denominators[arm.index()][Scope::White.random_index()] = checked_add(
                        denominators[arm.index()][Scope::White.random_index()],
                        1,
                        "cluster White attempted denominator",
                    )?;
                    for choice in ChoiceFamily::ALL {
                        let black = unit.observations[choice.index()][Color::Black.index()];
                        let white = unit.observations[choice.index()][Color::White.index()];
                        errors[arm.index()][choice.index()][Scope::Combined.random_index()] =
                            checked_add(
                                errors[arm.index()][choice.index()][Scope::Combined.random_index()],
                                u64::from(black.error) + u64::from(white.error),
                                "cluster combined error count",
                            )?;
                        errors[arm.index()][choice.index()][Scope::White.random_index()] =
                            checked_add(
                                errors[arm.index()][choice.index()][Scope::White.random_index()],
                                u64::from(white.error),
                                "cluster White error count",
                            )?;
                        regrets[arm.index()][choice.index()][Scope::Combined.random_index()]
                            .add(black.regret);
                        regrets[arm.index()][choice.index()][Scope::Combined.random_index()]
                            .add(white.regret);
                        regrets[arm.index()][choice.index()][Scope::White.random_index()]
                            .add(white.regret);
                    }
                }
            }
        }
        attempts += 1;
        let active_denominator = denominators[Arm::Active.index()][Scope::Combined.random_index()];
        let control_denominator =
            denominators[Arm::Control.index()][Scope::Combined.random_index()];
        if active_denominator == 0 || control_denominator == 0 {
            continue;
        }

        for choice in ChoiceFamily::ALL {
            for scope in Scope::RANDOM {
                let active_denominator = denominators[Arm::Active.index()][scope.random_index()];
                let control_denominator = denominators[Arm::Control.index()][scope.random_index()];
                if active_denominator == 0 || control_denominator == 0 {
                    return Err(format!(
                        "cluster replicate has zero {}/{} denominator after combined check",
                        choice.name(),
                        scope.name()
                    ));
                }
                let active_error = checked_rate(
                    errors[Arm::Active.index()][choice.index()][scope.random_index()],
                    active_denominator,
                    "cluster active error rate",
                )?;
                let control_error = checked_rate(
                    errors[Arm::Control.index()][choice.index()][scope.random_index()],
                    control_denominator,
                    "cluster control error rate",
                )?;
                bank.push(
                    choice,
                    scope,
                    SeriesMetric::ErrorDiscovery,
                    active_error - control_error,
                )?;
                let active_regret = checked_mean(
                    regrets[Arm::Active.index()][choice.index()][scope.random_index()].total(),
                    active_denominator,
                    "cluster active mean regret",
                )?;
                let control_regret = checked_mean(
                    regrets[Arm::Control.index()][choice.index()][scope.random_index()].total(),
                    control_denominator,
                    "cluster control mean regret",
                )?;
                bank.push(
                    choice,
                    scope,
                    SeriesMetric::MeanRegret,
                    active_regret - control_regret,
                )?;
            }
        }
        accepted += 1;
    }

    let completed = accepted == config.cluster_replicates;
    let (series, p10) = if completed {
        let (series, p10) = P10Matrix::from_bank(&bank, config.cluster_replicates)?;
        (series, Some(p10))
    } else {
        (Value::Null, None)
    };
    Ok(BootstrapResult {
        report: json!({
            "interpretation":
                "conditional stability of frozen active/deterministic-control sets; not a selector-vs-random CI",
            "requested_accepted_replicates": config.cluster_replicates,
            "maximum_attempts": config.cluster_max_attempts,
            "attempts": attempts,
            "accepted_replicates": accepted,
            "discarded_zero_arm_denominator": attempts - accepted,
            "completed": completed,
            "cluster_count": cluster_count,
            "active_touched_clusters": clusters.active_cluster_count,
            "deterministic_control_touched_clusters": clusters.control_cluster_count,
            "maximum_active_units_in_one_cluster": clusters.maximum_active_units,
            "maximum_control_units_in_one_cluster": clusters.maximum_control_units,
            "replicate_denominator":
                "attempted slots attached to each arm among sampled cluster occurrences",
            "rng": rng.audit_json(config.cluster_seed)?,
            "sampled_cluster_stream": {
                "encoding":
                    "u64_le(returned bounded cluster index) for each draw in attempt order, including discarded attempts, with no prefix",
                "sha256": upper_hex(&sample_stream.finalize()?),
            },
            "clusters": cluster_membership_json(units, clusters),
            "series": series,
        }),
        p10,
        completed,
    })
}

fn add_measurement_support_gates(
    units: &[Unit],
    context: &ValidatedContext,
    clusters: &ClusterSet,
    gates: &mut GateCollector,
) -> Result<Value, String> {
    let mut arms = Map::new();
    for arm in Arm::ALL {
        let indices = match arm {
            Arm::Active => &context.active_indices,
            Arm::Control => &context.control_indices,
        };
        let complete = indices
            .iter()
            .filter(|&&index| units[index].complete_pair)
            .count();
        let complete_pass = complete >= MIN_COMPLETE_UNITS as usize;
        gates.add(
            "measurement_support",
            format!("{}_complete_pair_units", arm.name()),
            json!(complete),
            format!(">= {MIN_COMPLETE_UNITS}"),
            complete_pass,
        );
        let mut choices = Map::new();
        for choice in ChoiceFamily::ALL {
            let mut colors = Map::new();
            for color in Color::ALL {
                let scope = match color {
                    Color::Black => Scope::Black,
                    Color::White => Scope::White,
                };
                let metric = aggregate(units, &context.fixed_order, arm, choice, scope, None)?;
                let pass = metric.measurable >= MIN_MEASURABLE_COLOR_SLOTS;
                gates.add(
                    "measurement_support",
                    format!(
                        "{}_{}_{}_measurable_slots",
                        arm.name(),
                        choice.name(),
                        color.name()
                    ),
                    json!(metric.measurable),
                    format!(">= {MIN_MEASURABLE_COLOR_SLOTS}"),
                    pass,
                );
                colors.insert(
                    color.name().to_string(),
                    json!({
                        "measurable_slots": metric.measurable,
                        "pass": pass,
                    }),
                );
            }
            choices.insert(choice.name().to_string(), Value::Object(colors));
        }
        arms.insert(
            arm.name().to_string(),
            json!({
                "complete_pair_units": complete,
                "complete_pair_pass": complete_pass,
                "choice_color_coverage": choices,
            }),
        );
    }

    let active_cluster_pass = clusters.active_cluster_count >= MIN_DEPENDENCE_CLUSTERS;
    let control_cluster_pass = clusters.control_cluster_count >= MIN_DEPENDENCE_CLUSTERS;
    let active_dominance_pass = clusters.maximum_active_units <= MAX_ARM_UNITS_PER_CLUSTER;
    let control_dominance_pass = clusters.maximum_control_units <= MAX_ARM_UNITS_PER_CLUSTER;
    gates.add(
        "measurement_support",
        "active_dependence_clusters",
        json!(clusters.active_cluster_count),
        format!(">= {MIN_DEPENDENCE_CLUSTERS}"),
        active_cluster_pass,
    );
    gates.add(
        "measurement_support",
        "deterministic_control_dependence_clusters",
        json!(clusters.control_cluster_count),
        format!(">= {MIN_DEPENDENCE_CLUSTERS}"),
        control_cluster_pass,
    );
    gates.add(
        "measurement_support",
        "maximum_active_units_in_one_cluster",
        json!(clusters.maximum_active_units),
        format!("<= {MAX_ARM_UNITS_PER_CLUSTER}"),
        active_dominance_pass,
    );
    gates.add(
        "measurement_support",
        "maximum_control_units_in_one_cluster",
        json!(clusters.maximum_control_units),
        format!("<= {MAX_ARM_UNITS_PER_CLUSTER}"),
        control_dominance_pass,
    );

    Ok(json!({
        "arms": arms,
        "dependence_clusters": {
            "union_clusters": clusters.clusters.len(),
            "active_touched": clusters.active_cluster_count,
            "deterministic_control_touched": clusters.control_cluster_count,
            "maximum_active_units": clusters.maximum_active_units,
            "maximum_deterministic_control_units": clusters.maximum_control_units,
        },
    }))
}

fn add_p10_gates(prefix: &str, matrix: Option<P10Matrix>, gates: &mut GateCollector) {
    for choice in ChoiceFamily::ALL {
        for scope in Scope::RANDOM {
            for metric in SeriesMetric::ALL {
                let observed = matrix.map(|values| values.get(choice, scope, metric));
                let pass = observed.is_some_and(|value| value > 0.0);
                gates.add(
                    "decision",
                    format!(
                        "{prefix}_{}_{}_{}_p10_gt_0",
                        choice.name(),
                        scope.name(),
                        metric.name()
                    ),
                    json!(observed),
                    "p10 > 0",
                    pass,
                );
            }
        }
    }
}

pub(crate) fn analyze(units: &[Unit]) -> Result<AnalysisOutcome, String> {
    analyze_with_config(units, AnalysisConfig::PRODUCTION)
}

fn analyze_with_config(units: &[Unit], config: AnalysisConfig) -> Result<AnalysisOutcome, String> {
    let context = validate_units(units)?;
    let clusters = build_dependence_clusters(units, &context)?;
    let mut gates = GateCollector::new();
    let measurement_support =
        add_measurement_support_gates(units, &context, &clusters, &mut gates)?;
    let point = point_analysis(units, &context, &mut gates)?;
    let random = finite_support_random_controls(units, &context, config)?;
    if !random.completed {
        return Err("finite-support random controls did not complete".to_string());
    }
    add_p10_gates("finite_support_random_control", random.p10, &mut gates);
    let cluster = dependence_cluster_stress(units, &clusters, config)?;
    gates.add(
        "measurement_support",
        "cluster_bootstrap_completed",
        json!({
            "completed": cluster.completed,
            "required_accepted_replicates": config.cluster_replicates,
        }),
        format!(
            "{} accepted within {} attempts",
            config.cluster_replicates, config.cluster_max_attempts
        ),
        cluster.completed,
    );
    add_p10_gates("dependence_cluster", cluster.p10, &mut gates);
    gates.add(
        "correctness",
        "paired_overlap_cancellation",
        json!(true),
        "exact paired numerator cancellation",
        true,
    );

    let final_label = if !gates.coverage_pass {
        NO_GO_SUPPORT_LABEL
    } else if gates.decision_pass {
        GO_LABEL
    } else {
        NO_GO_UPPER_BOUND_LABEL
    };
    let passed = gates
        .records
        .iter()
        .filter(|record| record.get("pass") == Some(&Value::Bool(true)))
        .count();
    let total = gates.records.len();
    let report = json!({
        "format": "cb-al1-p0b-stats-v1",
        "final_label": final_label,
        "array_index_contract": {
            "observations_outer_choice": ["static_top", "archived_actual"],
            "observations_inner_color": ["black", "white"],
        },
        "registered_constants": {
            "support_units": SUPPORT_UNITS,
            "ordinals": ORDINALS,
            "units_per_ordinal": UNITS_PER_ORDINAL,
            "arm_units_per_ordinal": ARM_UNITS_PER_ORDINAL,
            "arm_units": ARM_UNITS,
            "arm_color_slots": ARM_COLOR_SLOTS,
            "minimum_complete_units": MIN_COMPLETE_UNITS,
            "minimum_measurable_color_slots_per_family": MIN_MEASURABLE_COLOR_SLOTS,
            "minimum_dependence_clusters_per_arm": MIN_DEPENDENCE_CLUSTERS,
            "maximum_arm_units_per_cluster": MAX_ARM_UNITS_PER_CLUSTER,
            "random_control_replicates": config.random_replicates,
            "cluster_accepted_replicates": config.cluster_replicates,
            "cluster_max_attempts": config.cluster_max_attempts,
            "random_control_seed": config.random_seed,
            "cluster_seed": config.cluster_seed,
        },
        "membership": membership_report(units, &context)?,
        "measurement_support": measurement_support,
        "arms": {
            "active": arm_report(units, &context, Arm::Active)?,
            "deterministic_control": arm_report(units, &context, Arm::Control)?,
        },
        "point_comparison": point,
        "finite_support_random_controls": random.report,
        "dependence_cluster_stress": cluster.report,
        "gates": gates.records,
        "gate_summary": {
            "passed": passed,
            "total": total,
            "measurement_support_pass": gates.coverage_pass,
            "decision_pass": gates.decision_pass,
            "all_pass": gates.coverage_pass && gates.decision_pass,
        },
    });
    Ok(AnalysisOutcome {
        final_label,
        report,
    })
}

#[cfg(test)]
mod stats_tests {
    use super::*;

    fn hash(value: u64) -> String {
        format!("{value:064X}")
    }

    fn measured(error: bool) -> Observation {
        Observation {
            measurable: true,
            error,
            regret: if error { 0.25 } else { 0.0 },
            teacher_max_tied: false,
        }
    }

    fn synthetic_support() -> Vec<Unit> {
        let mut units = Vec::with_capacity(SUPPORT_UNITS);
        for (ordinal_slot, ordinal) in ORDINALS.into_iter().enumerate() {
            for rank in 0..UNITS_PER_ORDINAL {
                let index = ordinal_slot * UNITS_PER_ORDINAL + rank;
                let active = rank < ARM_UNITS_PER_ORDINAL;
                let control = (ARM_UNITS_PER_ORDINAL..2 * ARM_UNITS_PER_ORDINAL).contains(&rank);
                let observation = measured(active);
                units.push(Unit {
                    uid: hash(1 + index as u64),
                    ordinal,
                    support_rank: rank as u8,
                    opening_group_hash: hash(10_000 + index as u64),
                    parent_d4_side_hashes: [
                        hash(20_000 + 2 * index as u64),
                        hash(20_001 + 2 * index as u64),
                    ],
                    legal_child_d4_side_hashes: [
                        vec![hash(40_000 + 2 * index as u64)],
                        vec![hash(40_001 + 2 * index as u64)],
                    ],
                    matched_component_uid: Some(hash(60_000 + index as u64)),
                    complete_pair: true,
                    active,
                    deterministic_control: control,
                    observations: [[observation; 2]; 2],
                });
            }
        }
        units
    }

    #[test]
    fn sha256_known_answers_are_exact() {
        assert_eq!(
            sha256_hex(b"").unwrap(),
            "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855"
        );
        assert_eq!(
            sha256_hex(b"abc").unwrap(),
            "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD"
        );
    }

    #[test]
    fn splitmix64_and_bounded_sampler_are_deterministic() {
        let mut rng = AuditedSplitMix64::new(0);
        assert_eq!(rng.next_u64().unwrap(), 0xE220_A839_7B1D_CDAF);
        for bound in 1..=100 {
            let value = rng.bounded(bound).unwrap();
            assert!(value < bound);
        }
    }

    #[test]
    fn literal_conditional_error_predicate_has_exact_boundary() {
        assert!(conditional_error_gate(23, 100, 20, 100, true).unwrap());
        assert!(!conditional_error_gate(22, 100, 20, 100, true).unwrap());
        assert!(conditional_error_gate(20, 100, 20, 100, false).unwrap());
        assert!(!conditional_error_gate(19, 100, 20, 100, false).unwrap());
    }

    #[test]
    fn missing_unit_is_joined_by_full_legal_identity() {
        let shared_child = hash(900_000);
        let mut left = synthetic_support().remove(0);
        left.active = true;
        left.deterministic_control = false;
        left.complete_pair = false;
        left.matched_component_uid = None;
        left.observations = [[Observation::unmeasurable(); 2]; 2];
        left.legal_child_d4_side_hashes[0] = vec![shared_child.clone()];
        let mut right = synthetic_support().remove(1);
        right.active = false;
        right.deterministic_control = true;
        right.legal_child_d4_side_hashes[1] = vec![shared_child];
        let units = vec![left, right];
        let context = ValidatedContext {
            fixed_order: vec![0, 1],
            strata: std::array::from_fn(|_| Vec::new()),
            active_indices: vec![0],
            control_indices: vec![1],
            overlap_indices: Vec::new(),
        };
        let clusters = build_dependence_clusters(&units, &context).unwrap();
        assert_eq!(clusters.clusters.len(), 1);
        assert_eq!(clusters.clusters[0].members.len(), 2);
    }

    #[test]
    fn end_to_end_synthetic_go_exercises_all_statistics() {
        let units = synthetic_support();
        let outcome = analyze_with_config(
            &units,
            AnalysisConfig {
                random_replicates: 256,
                cluster_replicates: 256,
                cluster_max_attempts: 4_096,
                random_seed: RANDOM_CONTROL_SEED,
                cluster_seed: CLUSTER_SEED,
            },
        )
        .unwrap();
        assert_eq!(outcome.final_label, GO_LABEL);
        assert_eq!(
            outcome.report["gate_summary"]["all_pass"],
            Value::Bool(true)
        );
        assert_eq!(
            outcome.report["array_index_contract"]["observations_outer_choice"],
            json!(["static_top", "archived_actual"])
        );
    }

    #[test]
    fn incomplete_pair_with_measurable_label_fails_closed() {
        let mut units = synthetic_support();
        units[0].complete_pair = false;
        units[0].matched_component_uid = None;
        let error = validate_units(&units).unwrap_err();
        assert!(error.contains("incomplete pair is measurable"));
    }

    #[test]
    fn support_rank_gap_fails_closed() {
        let mut units = synthetic_support();
        units[0].support_rank = 99;
        let error = validate_units(&units).unwrap_err();
        assert!(error.contains("support ranks are not exactly 0..99"));
    }
}
