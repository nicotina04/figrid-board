use crate::corpus;
use crate::provenance;
use figrid_board::board::{Move, Stone};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) type UnitKey = (String, usize);

#[derive(Clone, Debug)]
pub(crate) struct LabelInventory {
    pub(crate) mv: Move,
    pub(crate) child_hash: String,
    pub(crate) legacy_black_logit_bits: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct LabelCandidate {
    pub(crate) mv: Move,
    pub(crate) child_hash: String,
    pub(crate) legacy_black_logit_bits: u32,
    pub(crate) teacher_top: bool,
    pub(crate) deployed_actual: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct LabelParent {
    pub(crate) row_uid: String,
    pub(crate) parent_hash: String,
    pub(crate) side: Stone,
    pub(crate) history: Vec<(Move, Stone)>,
    pub(crate) inventory: Vec<LabelInventory>,
    pub(crate) candidates: [LabelCandidate; corpus::K6],
    pub(crate) q_teacher: [f64; corpus::K6],
    pub(crate) repeat_scores: [[i64; corpus::K6]; 2],
}

impl LabelParent {
    pub(crate) fn deployed_actual_move(&self) -> Result<Move, String> {
        let moves = self
            .candidates
            .iter()
            .filter(|candidate| candidate.deployed_actual)
            .map(|candidate| candidate.mv)
            .collect::<Vec<_>>();
        match moves.as_slice() {
            [mv] => Ok(*mv),
            _ => Err(format!(
                "{} deployed_actual role count is {}",
                self.row_uid,
                moves.len()
            )),
        }
    }

    pub(crate) fn candidate_index(&self, mv: Move) -> Option<usize> {
        self.candidates
            .iter()
            .position(|candidate| candidate.mv == mv)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct LabelUnit {
    pub(crate) unit_uid: String,
    pub(crate) opening_hash: String,
    pub(crate) ordinal: usize,
    pub(crate) component_uid: String,
    pub(crate) black: LabelParent,
    pub(crate) white: LabelParent,
}

#[derive(Clone, Debug)]
pub(crate) struct LabelIndex {
    pub(crate) units: BTreeMap<UnitKey, LabelUnit>,
    pub(crate) diagnostics: Value,
}

pub(crate) fn load_label_index(paths: &corpus::InputPaths) -> Result<LabelIndex, String> {
    let corpus::CorpusBundle {
        slates,
        product_float: _,
        product: _,
        lineage: _,
        diagnostics,
    } = corpus::load_validate_and_replay(paths)?;

    let mut grouped = BTreeMap::<UnitKey, Vec<corpus::Slate>>::new();
    for slate in slates {
        grouped
            .entry((slate.opening_hash.clone(), slate.ordinal))
            .or_default()
            .push(slate);
    }
    if grouped.len() != 668 {
        return Err(format!(
            "train unit-key count {}, expected 668",
            grouped.len()
        ));
    }

    let mut units = BTreeMap::new();
    for (key, mut pair) in grouped {
        if pair.len() != 2 {
            return Err(format!("unit key {key:?} has {} rows", pair.len()));
        }
        pair.sort_by_key(|slate| match slate.root_side {
            Stone::Black => 0,
            Stone::White => 1,
        });
        if pair[0].root_side != Stone::Black
            || pair[1].root_side != Stone::White
            || pair[0].component_uid != pair[1].component_uid
            || pair[0].opening_hash != pair[1].opening_hash
            || pair[0].ordinal != pair[1].ordinal
            || pair[0].history.get(..4) != pair[1].history.get(..4)
        {
            return Err(format!("paired-color invariant failed for {key:?}"));
        }
        let black_parent_hash = pair[0].parent_hash.clone();
        let white_parent_hash = pair[1].parent_hash.clone();
        let unit_uid = provenance::sha256_hex(
            format!(
                "RQ615C|structural-unit|{}|{}|{}|{}",
                key.0, key.1, black_parent_hash, white_parent_hash
            )
            .as_bytes(),
        );
        let black = convert_parent(&pair[0])?;
        let white = convert_parent(&pair[1])?;
        let component_uid = pair[0].component_uid.clone();
        let unit = LabelUnit {
            unit_uid,
            opening_hash: key.0.clone(),
            ordinal: key.1,
            component_uid,
            black,
            white,
        };
        if units.insert(key, unit).is_some() {
            return Err("duplicate reconstructed label unit".to_string());
        }
    }

    let component_diagnostics = validate_components(&units)?;
    Ok(LabelIndex {
        units,
        diagnostics: json!({
            "shared_corpus": diagnostics,
            "al1_exact_q_tolerance": 1.0e-15,
            "reconstructed_units": 668,
            "component_replay": component_diagnostics,
        }),
    })
}

fn convert_parent(slate: &corpus::Slate) -> Result<LabelParent, String> {
    let q0 = softmax(&slate.repeat_scores_mover[0]);
    let q1 = softmax(&slate.repeat_scores_mover[1]);
    for index in 0..corpus::K6 {
        let expected = (q0[index] + q1[index]) / 2.0;
        if (slate.q_teacher[index] - expected).abs() > 1.0e-15 {
            return Err(format!(
                "{} q_teacher[{index}] fails 1e-15 replay: stored={:.17} expected={expected:.17}",
                slate.row_uid, slate.q_teacher[index]
            ));
        }
    }

    let inventory = slate
        .legal_inventory
        .iter()
        .map(|item| LabelInventory {
            mv: item.mv,
            child_hash: item.stored_child_hash.clone(),
            legacy_black_logit_bits: item.stored_black_logit.to_bits(),
        })
        .collect::<Vec<_>>();
    let candidates = slate
        .candidates
        .iter()
        .map(|candidate| LabelCandidate {
            mv: candidate.mv,
            child_hash: candidate.stored_child_hash.clone(),
            legacy_black_logit_bits: candidate.stored_black_logit.to_bits(),
            teacher_top: candidate.teacher_top,
            deployed_actual: candidate.deployed_actual,
        })
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| "candidate array conversion failed".to_string())?;
    let parent = LabelParent {
        row_uid: slate.row_uid.clone(),
        parent_hash: slate.parent_hash.clone(),
        side: slate.root_side,
        history: slate.history.clone(),
        inventory,
        candidates,
        q_teacher: slate.q_teacher,
        repeat_scores: slate.repeat_scores_mover,
    };
    let actual = parent.deployed_actual_move()?;
    if parent.candidate_index(actual).is_none() {
        return Err(format!("{} deployed actual is not in K=6", parent.row_uid));
    }
    if parent
        .candidates
        .iter()
        .filter(|candidate| candidate.teacher_top)
        .count()
        != 1
    {
        return Err(format!(
            "{} teacher_top role count mismatch",
            parent.row_uid
        ));
    }
    Ok(parent)
}

fn softmax(scores: &[i64; corpus::K6]) -> [f64; corpus::K6] {
    let mut max = f64::NEG_INFINITY;
    let mut logits = [0.0; corpus::K6];
    for index in 0..corpus::K6 {
        logits[index] = scores[index] as f64 / 400.0;
        max = max.max(logits[index]);
    }
    let mut weights = [0.0; corpus::K6];
    let mut total = 0.0;
    for index in 0..corpus::K6 {
        weights[index] = (logits[index] - max).exp();
        total += weights[index];
    }
    let mut output = [0.0; corpus::K6];
    for index in 0..corpus::K6 {
        output[index] = weights[index] / total;
    }
    output
}

fn validate_components(units: &BTreeMap<UnitKey, LabelUnit>) -> Result<Value, String> {
    let values = units.values().collect::<Vec<_>>();
    let mut dsu = Dsu::new(values.len());
    let mut first_by_identity = BTreeMap::<String, usize>::new();
    let mut tokens_by_unit = Vec::with_capacity(values.len());
    for (index, unit) in values.iter().enumerate() {
        let mut tokens = BTreeSet::new();
        tokens.insert(format!("O:{}", unit.opening_hash));
        tokens.insert(format!("S:{}", unit.black.parent_hash));
        tokens.insert(format!("S:{}", unit.white.parent_hash));
        for parent in [&unit.black, &unit.white] {
            for candidate in &parent.candidates {
                tokens.insert(format!("S:{}", candidate.child_hash));
            }
        }
        for token in &tokens {
            if let Some(&other) = first_by_identity.get(token) {
                dsu.union(index, other);
            } else {
                first_by_identity.insert(token.clone(), index);
            }
        }
        tokens_by_unit.push(tokens);
    }

    let mut groups = BTreeMap::<usize, Vec<usize>>::new();
    for index in 0..values.len() {
        let root = dsu.find(index);
        groups.entry(root).or_default().push(index);
    }
    if groups.len() != 388 {
        return Err(format!(
            "reconstructed train component count {}, expected 388",
            groups.len()
        ));
    }

    let mut expected_uids = BTreeSet::new();
    let mut max_units = 0usize;
    for members in groups.values() {
        let mut unit_uids = members
            .iter()
            .map(|&index| values[index].unit_uid.clone())
            .collect::<Vec<_>>();
        unit_uids.sort();
        unit_uids.dedup();
        let mut tokens = BTreeSet::new();
        for &index in members {
            tokens.extend(tokens_by_unit[index].iter().cloned());
        }
        let expected = provenance::sha256_hex(
            format!(
                "RQ615C|component-v1|units={}|identities={}",
                unit_uids.join(","),
                tokens.into_iter().collect::<Vec<_>>().join(",")
            )
            .as_bytes(),
        );
        for &index in members {
            if values[index].component_uid != expected {
                return Err(format!(
                    "component UID mismatch for unit {}: stored={} expected={expected}",
                    values[index].unit_uid, values[index].component_uid
                ));
            }
        }
        if !expected_uids.insert(expected) {
            return Err("duplicate reconstructed component UID".to_string());
        }
        max_units = max_units.max(members.len());
    }
    Ok(json!({
        "components": groups.len(),
        "component_uids_unique": expected_uids.len(),
        "max_units_per_component": max_units,
        "identity_tokens": first_by_identity.len(),
        "formula_replay_mismatches": 0,
    }))
}

#[derive(Clone, Debug)]
struct Dsu {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl Dsu {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            rank: vec![0; len],
        }
    }

    fn find(&mut self, value: usize) -> usize {
        if self.parent[value] != value {
            self.parent[value] = self.find(self.parent[value]);
        }
        self.parent[value]
    }

    fn union(&mut self, left: usize, right: usize) {
        let mut left = self.find(left);
        let mut right = self.find(right);
        if left == right {
            return;
        }
        if self.rank[left] < self.rank[right] {
            std::mem::swap(&mut left, &mut right);
        }
        self.parent[right] = left;
        if self.rank[left] == self.rank[right] {
            self.rank[left] += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_is_normalized() {
        let q = softmax(&[3000, 100, 0, -100, -2500, -3000]);
        assert!((q.iter().sum::<f64>() - 1.0).abs() <= 1.0e-15);
        assert!(q[0] > q[1]);
    }

    #[test]
    fn dsu_unions_transitively() {
        let mut dsu = Dsu::new(4);
        dsu.union(0, 1);
        dsu.union(1, 2);
        assert_eq!(dsu.find(0), dsu.find(2));
        assert_ne!(dsu.find(0), dsu.find(3));
    }
}
