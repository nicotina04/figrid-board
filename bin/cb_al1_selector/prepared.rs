use crate::hash::{
    ColoredMove, Digest, FileSeal, canonical_opening_hash, canonical_position_hash,
    historical_unit_rank_digest, parent_uid as expected_parent_uid, require_file_seal,
    selector_digest, split_bucket, stone_char, uid_stream_hash, unit_uid as expected_unit_uid,
};
use figrid_board::board::{BOARD_SIZE, Board, GameResult, Move, NUM_CELLS, RuleSet, Stone};
use figrid_board::codebook_eval::{
    CodebookWeights, QuantizedCodebookWeights, evaluate_full_factored_quantized_for_audit,
    evaluate_full_quantized,
};
use figrid_board::factored_codebook::{
    FactoredQuantizedCodebookWeights, PackedCodebookArtifact, PackedCodebookKind,
};
use figrid_board::pattern_table::{PATTERN_NUM_IDS, PATTERN_RARE_ID, lookup_mapped_id};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

const PREPARED_BYTES: u64 = 78_707_493;
const PREPARED_SHA256: &str = "2B5391DD9BB78969F119AD70162CDCA185E62B25FAB720CA7AB852030DDFC74B";
const PHASE2_MANIFEST_BYTES: u64 = 3_500;
const PHASE2_MANIFEST_SHA256: &str =
    "92D6BF8E6F42181F0A25BDDF41D839B59A9B758D25637081DC57A375C50F4C4D";
const PRODUCT_MODEL_BYTES: u64 = 1_410_562;
const PRODUCT_MODEL_SHA256: &str =
    "42968FDAB01BA8CCD1DE3DED05C532E4B237DD47EEFFD7AE1C2F264D77BA7DA2";
const PRODUCT_CBF_BYTES: u64 = 353_582;
const PRODUCT_CBF_SHA256: &str = "141014529417A73E58B210832AFD189AD970E045A8907F7D2879693C5B171A8D";
const TOPK_BYTES: u64 = 17_060;
const TOPK_SHA256: &str = "103891DCD1DCD978C654593ABE78EF32C56E2E350B500EE665BC45AC051AA16D";

const HISTORICAL_FIREWALL_BYTES: u64 = 292_573_334;
const HISTORICAL_FIREWALL_SHA256: &str =
    "3886D4645881531CEC0698B9BC9DCA8E27E12BB517F039A95C8625297B48D4E6";
const LINEAGE_MODEL_BYTES: u64 = 1_413_542;
const LINEAGE_MODEL_SHA256: &str =
    "69BB7C599ADA3A1151577CE3315356BC33C40EDB49A003C9BC4EB90A98F82E18";

const PREPARED_FORMAT: &str = "rq615c-preteacher-paired-unit-v1";
const MANIFEST_FORMAT: &str = "rq615c-phase2-prepared-manifest-v1";
const ORDINALS: [usize; 5] = [1, 2, 4, 6, 8];
const PRODUCT_DIM: usize = 16;
const PRODUCT_FM_RANK: usize = 8;
const REGIONS: usize = 9;
const LEGACY_EMBED_SCALE: i32 = 32;
const LEGACY_HEAD_SCALE: i32 = 2048;
const LEGACY_FACTOR_SCALE: i32 = 2048;
const CURRENT_EMBED_SCALE: i32 = 32;
const CURRENT_HEAD_SCALE: i32 = 64;
const CURRENT_FACTOR_SCALE: i32 = 64;
const SUPPORT_PER_ORDINAL: usize = 100;
const ARM_PER_ORDINAL: usize = 25;

pub(crate) const QUIET_REASON_ORDER: [&str; 4] = [
    "too_few_legal",
    "mover_immediate_five",
    "opponent_immediate_five",
    "actual_illegal",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PreparedPaths {
    pub(crate) prepared_units: PathBuf,
    pub(crate) phase2_manifest: PathBuf,
    pub(crate) product_model: PathBuf,
    pub(crate) product_cbf: PathBuf,
    pub(crate) topk: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Split {
    Train,
    Dev,
    Safety,
}

impl Split {
    pub(crate) fn parse(value: &str) -> Result<Self, String> {
        match value {
            "train" => Ok(Self::Train),
            "dev" => Ok(Self::Dev),
            "safety" => Ok(Self::Safety),
            _ => Err(format!("invalid split {value:?}")),
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Dev => "dev",
            Self::Safety => "safety",
        }
    }

    fn expected_from_opening(opening_hash: &str) -> Self {
        match split_bucket(opening_hash) {
            0..=69 => Self::Train,
            70..=84 => Self::Dev,
            _ => Self::Safety,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum QuietReason {
    TooFewLegal,
    MoverImmediateFive,
    OpponentImmediateFive,
    ActualIllegal,
}

impl QuietReason {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::TooFewLegal => QUIET_REASON_ORDER[0],
            Self::MoverImmediateFive => QUIET_REASON_ORDER[1],
            Self::OpponentImmediateFive => QUIET_REASON_ORDER[2],
            Self::ActualIllegal => QUIET_REASON_ORDER[3],
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct InventoryEntry {
    pub(crate) mv: Move,
    pub(crate) child_d4_side_hash: String,
    pub(crate) legacy_black_logit: f32,
    pub(crate) legacy_black_logit_bits: u32,
    pub(crate) current_child_logit: f32,
    pub(crate) current_child_logit_bits: u32,
    pub(crate) current_mover_utility: f32,
    pub(crate) current_mover_utility_bits: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct ScoredMove {
    pub(crate) mv: Move,
    pub(crate) utility: f32,
    pub(crate) utility_bits: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct ParentDiagnostics {
    pub(crate) static_top: ScoredMove,
    pub(crate) static_second: ScoredMove,
    pub(crate) archived_actual: ScoredMove,
    pub(crate) margin: f32,
    pub(crate) margin_bits: u32,
    pub(crate) actual_gap: f32,
    pub(crate) actual_gap_bits: u32,
    pub(crate) search_disagreement: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedParent {
    pub(crate) parent_uid: String,
    pub(crate) parent_d4_side_hash: String,
    pub(crate) side_to_move: Stone,
    pub(crate) history: Vec<ColoredMove>,
    pub(crate) figrid_actual_move: Move,
    pub(crate) current_root_logit: f32,
    pub(crate) current_root_logit_bits: u32,
    pub(crate) legal_inventory: Vec<InventoryEntry>,
    pub(crate) quiet_reasons: Vec<QuietReason>,
    pub(crate) diagnostics: ParentDiagnostics,
}

impl PreparedParent {
    pub(crate) fn quiet_eligible(&self) -> bool {
        self.quiet_reasons.is_empty()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedUnit {
    pub(crate) unit_uid: String,
    pub(crate) split: Split,
    pub(crate) opening_group_hash: String,
    pub(crate) figrid_ordinal: usize,
    pub(crate) black: PreparedParent,
    pub(crate) white: PreparedParent,
    pub(crate) mean_margin: f32,
    pub(crate) mean_margin_bits: u32,
    pub(crate) support_digest: Digest,
    pub(crate) active_digest: Digest,
    pub(crate) control_digest: Digest,
}

impl PreparedUnit {
    pub(crate) fn quiet_eligible(&self) -> bool {
        self.black.quiet_eligible() && self.white.quiet_eligible()
    }

    pub(crate) fn parent(&self, side: Stone) -> &PreparedParent {
        match side {
            Stone::Black => &self.black,
            Stone::White => &self.white,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SelectorSupportStatus {
    ReadyForReveal,
    NoGoSelectorSupport,
}

impl SelectorSupportStatus {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::ReadyForReveal => "P0A_READY_FOR_REVEAL",
            Self::NoGoSelectorSupport => "NO_GO_SELECTOR_SUPPORT",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ManifestSummary {
    pub(crate) selected_units: usize,
    pub(crate) selected_parents: usize,
    pub(crate) selected_inventory_entries: usize,
    pub(crate) split_units: BTreeMap<Split, usize>,
    pub(crate) distinct_openings_by_split: BTreeMap<Split, usize>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct AnalysisCounts {
    pub(crate) units: usize,
    pub(crate) parents: usize,
    pub(crate) inventory_entries: usize,
    pub(crate) current_root_parity_checks: usize,
    pub(crate) current_child_parity_checks: usize,
    pub(crate) legacy_child_replay_checks: usize,
    pub(crate) split_units: BTreeMap<Split, usize>,
    pub(crate) color_parents: BTreeMap<&'static str, usize>,
    pub(crate) exclusions: BTreeMap<(usize, Split, &'static str, &'static str), usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SelectorStreams {
    pub(crate) support_by_ordinal: BTreeMap<usize, Vec<String>>,
    pub(crate) active_by_ordinal: BTreeMap<usize, Vec<String>>,
    pub(crate) control_by_ordinal: BTreeMap<usize, Vec<String>>,
    pub(crate) support: Vec<String>,
    pub(crate) active: Vec<String>,
    pub(crate) control: Vec<String>,
    pub(crate) support_sha256: String,
    pub(crate) active_sha256: String,
    pub(crate) control_sha256: String,
    pub(crate) overlap_units: usize,
    pub(crate) active_distinct_openings: usize,
    pub(crate) control_distinct_openings: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedAnalysis {
    pub(crate) input_seals: BTreeMap<&'static str, FileSeal>,
    pub(crate) manifest: ManifestSummary,
    pub(crate) units: Vec<PreparedUnit>,
    pub(crate) counts: AnalysisCounts,
    pub(crate) selector: SelectorStreams,
    pub(crate) status: SelectorSupportStatus,
    pub(crate) product_source_payload_identity: bool,
    pub(crate) current_quantized_payload_identity: bool,
}

impl PreparedAnalysis {
    pub(crate) fn unit_by_uid(&self, uid: &str) -> Option<&PreparedUnit> {
        self.units.iter().find(|unit| unit.unit_uid == uid)
    }

    pub(crate) fn to_json(&self) -> Value {
        let artifacts = self
            .input_seals
            .iter()
            .map(|(name, seal)| {
                (
                    (*name).to_string(),
                    json!({"bytes": seal.bytes, "sha256": seal.sha256}),
                )
            })
            .collect::<Map<_, _>>();
        let units = self.units.iter().map(unit_summary_json).collect::<Vec<_>>();
        let exclusions = self
            .counts
            .exclusions
            .iter()
            .map(|(&(ordinal, split, color, reason), &count)| {
                json!({
                    "figrid_ordinal": ordinal,
                    "split": split.name(),
                    "color": color,
                    "reason": reason,
                    "count": count,
                })
            })
            .collect::<Vec<_>>();
        json!({
            "status": self.status.name(),
            "input_artifacts": artifacts,
            "manifest": {
                "selected_units": self.manifest.selected_units,
                "selected_parents": self.manifest.selected_parents,
                "selected_inventory_entries": self.manifest.selected_inventory_entries,
                "selected_units_by_split": split_map_json(&self.manifest.split_units),
                "selected_distinct_openings_by_split":
                    split_map_json(&self.manifest.distinct_openings_by_split),
            },
            "contract": {
                "ordinals": ORDINALS,
                "quiet_reason_order": QUIET_REASON_ORDER,
                "support_per_ordinal": SUPPORT_PER_ORDINAL,
                "arm_per_ordinal": ARM_PER_ORDINAL,
            },
            "payload_audit": {
                "product_json_cbf_source_bit_identity": self.product_source_payload_identity,
                "cbf_reconstructed_flat_fresh_quantized_identity":
                    self.current_quantized_payload_identity,
                "current_root_parity_checks": self.counts.current_root_parity_checks,
                "current_child_parity_checks": self.counts.current_child_parity_checks,
                "legacy_child_replay_checks": self.counts.legacy_child_replay_checks,
            },
            "validation_counts": {
                "units": self.counts.units,
                "parents": self.counts.parents,
                "legal_inventory_entries": self.counts.inventory_entries,
                "units_by_split": split_map_json(&self.counts.split_units),
                "parents_by_color": self.counts.color_parents,
                "quiet_exclusions": exclusions,
            },
            "selector": {
                "support_by_ordinal": ordinal_map_json(&self.selector.support_by_ordinal),
                "active_by_ordinal": ordinal_map_json(&self.selector.active_by_ordinal),
                "control_by_ordinal": ordinal_map_json(&self.selector.control_by_ordinal),
                "support_uids": self.selector.support,
                "active_uids": self.selector.active,
                "control_uids": self.selector.control,
                "support_uid_stream_sha256": self.selector.support_sha256,
                "active_uid_stream_sha256": self.selector.active_sha256,
                "control_uid_stream_sha256": self.selector.control_sha256,
                "active_control_overlap_units": self.selector.overlap_units,
                "active_distinct_openings": self.selector.active_distinct_openings,
                "control_distinct_openings": self.selector.control_distinct_openings,
            },
            "unit_diagnostics": units,
        })
    }
}

struct ProductModels {
    legacy: QuantizedCodebookWeights,
    current_flat: QuantizedCodebookWeights,
    current_factored: FactoredQuantizedCodebookWeights,
}

pub(crate) fn recheck_p0a_inputs(
    paths: &PreparedPaths,
) -> Result<BTreeMap<&'static str, FileSeal>, String> {
    let mut seals = BTreeMap::new();
    seals.insert(
        "prepared_units",
        require_file_seal(
            &paths.prepared_units,
            PREPARED_BYTES,
            PREPARED_SHA256,
            "RQ615C prepared units",
        )?,
    );
    seals.insert(
        "phase2_manifest",
        require_file_seal(
            &paths.phase2_manifest,
            PHASE2_MANIFEST_BYTES,
            PHASE2_MANIFEST_SHA256,
            "RQ615C Phase-2 manifest",
        )?,
    );
    seals.insert(
        "product_model",
        require_file_seal(
            &paths.product_model,
            PRODUCT_MODEL_BYTES,
            PRODUCT_MODEL_SHA256,
            "product FP32 model",
        )?,
    );
    seals.insert(
        "product_cbf",
        require_file_seal(
            &paths.product_cbf,
            PRODUCT_CBF_BYTES,
            PRODUCT_CBF_SHA256,
            "product factored CBF",
        )?,
    );
    seals.insert(
        "topk",
        require_file_seal(
            &paths.topk,
            TOPK_BYTES,
            TOPK_SHA256,
            "swap-closed Pattern4 vocabulary",
        )?,
    );
    Ok(seals)
}

pub(crate) fn analyze_p0a(paths: &PreparedPaths) -> Result<PreparedAnalysis, String> {
    let initial_seals = recheck_p0a_inputs(paths)?;

    let manifest_bytes = fs::read(&paths.phase2_manifest).map_err(|error| {
        format!(
            "failed to read Phase-2 manifest {}: {error}",
            paths.phase2_manifest.display()
        )
    })?;
    let manifest = validate_phase2_manifest(&manifest_bytes)?;

    let topk_bytes = fs::read(&paths.topk)
        .map_err(|error| format!("failed to read topk {}: {error}", paths.topk.display()))?;
    validate_topk(&topk_bytes)?;

    let (models, source_identity, quantized_identity) = load_product_models(paths)?;
    let file = File::open(&paths.prepared_units).map_err(|error| {
        format!(
            "failed to open prepared units {}: {error}",
            paths.prepared_units.display()
        )
    })?;
    let mut units = Vec::with_capacity(1_000);
    let mut counts = AnalysisCounts::default();
    let mut previous_order_key: Option<(u8, Digest, String, usize, String)> = None;
    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        let line =
            line.map_err(|error| format!("prepared line {} read failed: {error}", line_index + 1))?;
        if line.is_empty() {
            return Err(format!("prepared line {} is blank", line_index + 1));
        }
        let value: Value = serde_json::from_str(&line)
            .map_err(|error| format!("prepared line {} invalid JSON: {error}", line_index + 1))?;
        let unit = parse_prepared_unit(&value, &models, &mut counts)
            .map_err(|error| format!("prepared line {}: {error}", line_index + 1))?;
        let order_key = (
            match unit.split {
                Split::Train => 0,
                Split::Dev => 1,
                Split::Safety => 2,
            },
            historical_unit_rank_digest(&unit.opening_group_hash, unit.figrid_ordinal),
            unit.opening_group_hash.clone(),
            unit.figrid_ordinal,
            unit.unit_uid.clone(),
        );
        if previous_order_key
            .as_ref()
            .is_some_and(|previous| *previous >= order_key)
        {
            return Err(format!(
                "historical prepared row order is not strict at line {}",
                line_index + 1
            ));
        }
        previous_order_key = Some(order_key);
        units.push(unit);
    }

    validate_corpus_contract(&units, &counts, &manifest)?;
    let (selector, status) = select_arms(&units)?;

    let final_seals = recheck_p0a_inputs(paths)?;
    if final_seals != initial_seals {
        return Err("P0A input identities changed during analysis".to_string());
    }

    Ok(PreparedAnalysis {
        input_seals: final_seals,
        manifest,
        units,
        counts,
        selector,
        status,
        product_source_payload_identity: source_identity,
        current_quantized_payload_identity: quantized_identity,
    })
}

fn load_product_models(paths: &PreparedPaths) -> Result<(ProductModels, bool, bool), String> {
    let model_bytes = fs::read(&paths.product_model).map_err(|error| {
        format!(
            "failed to read product model {}: {error}",
            paths.product_model.display()
        )
    })?;
    let product = CodebookWeights::from_json_bytes(&model_bytes)
        .map_err(|error| format!("invalid product model: {error}"))?;
    validate_model_shape(&product)?;

    let cbf_bytes = fs::read(&paths.product_cbf).map_err(|error| {
        format!(
            "failed to read product CBF {}: {error}",
            paths.product_cbf.display()
        )
    })?;
    let artifact = PackedCodebookArtifact::parse(&cbf_bytes)
        .map_err(|error| format!("invalid product CBF: {error}"))?;
    if artifact.kind() != PackedCodebookKind::Factored {
        return Err("product CBF is not factored".to_string());
    }
    require_float_payload_identity(&product, artifact.source_weights(), "product JSON/CBF")?;
    let factored = artifact
        .factored_quantized()
        .ok_or_else(|| "product CBF omitted factored payload".to_string())?
        .clone();
    factored
        .validate()
        .map_err(|error| format!("invalid factored payload: {error}"))?;

    let reconstructed = factored.reconstruct_flat();
    let fresh_current = quantize_exact(
        &product,
        CURRENT_EMBED_SCALE,
        CURRENT_HEAD_SCALE,
        CURRENT_FACTOR_SCALE,
    )?;
    require_quantized_identity(
        &reconstructed,
        &fresh_current,
        "CBF reconstructed/current fresh quantized",
    )?;

    let legacy = quantize_exact(
        &product,
        LEGACY_EMBED_SCALE,
        LEGACY_HEAD_SCALE,
        LEGACY_FACTOR_SCALE,
    )?;
    Ok((
        ProductModels {
            legacy,
            current_flat: reconstructed,
            current_factored: factored,
        },
        true,
        true,
    ))
}

fn validate_model_shape(weights: &CodebookWeights) -> Result<(), String> {
    let valid = weights.dim == PRODUCT_DIM
        && weights.fm_rank == PRODUCT_FM_RANK
        && weights.embeddings.len() == PATTERN_NUM_IDS * PRODUCT_DIM
        && weights.head.len() == REGIONS * PRODUCT_DIM
        && weights.factors.len() == REGIONS * PRODUCT_DIM * PRODUCT_FM_RANK
        && weights.bias.is_finite()
        && weights.embeddings.iter().all(|value| value.is_finite())
        && weights.head.iter().all(|value| value.is_finite())
        && weights.factors.iter().all(|value| value.is_finite());
    if valid {
        Ok(())
    } else {
        Err("product FP32 model shape/finiteness mismatch".to_string())
    }
}

fn quantize_exact(
    weights: &CodebookWeights,
    embedding_scale: i32,
    head_scale: i32,
    factor_scale: i32,
) -> Result<QuantizedCodebookWeights, String> {
    validate_model_shape(weights)?;
    Ok(QuantizedCodebookWeights {
        dim: weights.dim,
        fm_rank: weights.fm_rank,
        embedding_scale,
        head_scale,
        factor_scale,
        embeddings: quantize_vector(&weights.embeddings, embedding_scale, "embeddings")?,
        head: quantize_vector(&weights.head, head_scale, "head")?,
        factors: quantize_vector(&weights.factors, factor_scale, "factors")?,
        bias: weights.bias,
    })
}

fn quantize_vector(values: &[f32], scale: i32, label: &str) -> Result<Vec<i16>, String> {
    if scale <= 0 {
        return Err(format!("{label} scale must be positive"));
    }
    values
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            if !value.is_finite() {
                return Err(format!("{label}[{index}] is non-finite"));
            }
            let scaled = value * scale as f32;
            if !scaled.is_finite() {
                return Err(format!("{label}[{index}] scaled value is non-finite"));
            }
            let rounded = scaled.round();
            if rounded < i16::MIN as f32 || rounded > i16::MAX as f32 {
                return Err(format!("{label}[{index}] overflows i16"));
            }
            Ok(rounded as i16)
        })
        .collect()
}

fn require_float_payload_identity(
    left: &CodebookWeights,
    right: &CodebookWeights,
    label: &str,
) -> Result<(), String> {
    if left.dim != right.dim
        || left.fm_rank != right.fm_rank
        || left.embeddings.len() != right.embeddings.len()
        || left.head.len() != right.head.len()
        || left.factors.len() != right.factors.len()
        || left.bias.to_bits() != right.bias.to_bits()
        || !left
            .embeddings
            .iter()
            .zip(&right.embeddings)
            .all(|(a, b)| a.to_bits() == b.to_bits())
        || !left
            .head
            .iter()
            .zip(&right.head)
            .all(|(a, b)| a.to_bits() == b.to_bits())
        || !left
            .factors
            .iter()
            .zip(&right.factors)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    {
        return Err(format!("{label} FP32 payload mismatch"));
    }
    Ok(())
}

fn require_quantized_identity(
    left: &QuantizedCodebookWeights,
    right: &QuantizedCodebookWeights,
    label: &str,
) -> Result<(), String> {
    if left.dim != right.dim
        || left.fm_rank != right.fm_rank
        || left.embedding_scale != right.embedding_scale
        || left.head_scale != right.head_scale
        || left.factor_scale != right.factor_scale
        || left.embeddings != right.embeddings
        || left.head != right.head
        || left.factors != right.factors
        || left.bias.to_bits() != right.bias.to_bits()
    {
        return Err(format!("{label} payload mismatch"));
    }
    Ok(())
}

fn validate_topk(bytes: &[u8]) -> Result<(), String> {
    if PATTERN_RARE_ID as usize != PATTERN_NUM_IDS - 1 {
        return Err("released Pattern4 rare-ID shape mismatch".to_string());
    }
    if bytes.len() != (PATTERN_NUM_IDS - 1) * 4 {
        return Err("topk length does not match PATTERN_NUM_IDS".to_string());
    }
    let mut seen = BTreeSet::new();
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        let packed = u32::from_le_bytes(chunk.try_into().expect("four-byte topk word"));
        if !seen.insert(packed) {
            return Err(format!("duplicate topk packed token at index {index}"));
        }
        if usize::from(lookup_mapped_id(packed)) != index {
            return Err(format!("topk mapped-ID order mismatch at index {index}"));
        }
    }
    Ok(())
}

fn validate_phase2_manifest(bytes: &[u8]) -> Result<ManifestSummary, String> {
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("invalid Phase-2 manifest JSON: {error}"))?;
    require_exact_keys(
        &value,
        &[
            "contract",
            "format",
            "inputs",
            "outputs",
            "representation",
            "selection",
            "status",
        ],
        "Phase-2 manifest",
    )?;
    if json_str(&value, "format")? != MANIFEST_FORMAT || json_str(&value, "status")? != "complete" {
        return Err("Phase-2 manifest format/status mismatch".to_string());
    }

    let contract = json_object(&value, "contract")?;
    require_exact_map_keys(
        contract,
        &[
            "answer_opaque_firewall_only",
            "exact_1000_units_2000_parents",
            "full_legal_inventory_present",
            "outcome_free",
        ],
        "Phase-2 contract",
    )?;
    for key in [
        "answer_opaque_firewall_only",
        "exact_1000_units_2000_parents",
        "full_legal_inventory_present",
        "outcome_free",
    ] {
        if json_bool_map(contract, key)? != true {
            return Err(format!("Phase-2 contract {key} is not true"));
        }
    }

    let representation = json_object(&value, "representation")?;
    require_exact_map_keys(
        representation,
        &[
            "base",
            "base_logit_perspective",
            "guard_k",
            "legal_inventory_order",
            "mover_utility_tie_break",
        ],
        "Phase-2 representation",
    )?;
    if json_str_map(representation, "base")? != "E32/H2048/F2048"
        || json_str_map(representation, "base_logit_perspective")? != "p(Black)"
        || json_usize_map(representation, "guard_k")? != 6
        || json_str_map(representation, "legal_inventory_order")? != "cell=y*15+x ascending"
        || json_str_map(representation, "mover_utility_tie_break")? != "cell=y*15+x ascending"
    {
        return Err("Phase-2 representation contract mismatch".to_string());
    }

    let inputs = json_object(&value, "inputs")?;
    require_exact_map_keys(
        inputs,
        &[
            "base_codebook",
            "historical_firewall",
            "phase2_export_seal",
            "structural_dev",
            "structural_manifest",
            "structural_safety",
            "structural_train",
        ],
        "Phase-2 inputs",
    )?;
    for key in [
        "base_codebook",
        "historical_firewall",
        "phase2_export_seal",
        "structural_dev",
        "structural_manifest",
        "structural_safety",
        "structural_train",
    ] {
        validate_manifest_artifact(
            inputs
                .get(key)
                .ok_or_else(|| format!("missing Phase-2 input {key}"))?,
            false,
            key,
        )?;
    }
    require_manifest_artifact_identity(
        inputs
            .get("historical_firewall")
            .expect("exact input keys established"),
        HISTORICAL_FIREWALL_BYTES,
        HISTORICAL_FIREWALL_SHA256,
        None,
        "historical firewall",
    )?;
    require_manifest_artifact_identity(
        inputs
            .get("base_codebook")
            .expect("exact input keys established"),
        LINEAGE_MODEL_BYTES,
        LINEAGE_MODEL_SHA256,
        None,
        "Phase-2 base codebook",
    )?;

    let outputs = json_object(&value, "outputs")?;
    require_exact_map_keys(outputs, &["prepared_units"], "Phase-2 outputs")?;
    let prepared = outputs
        .get("prepared_units")
        .ok_or_else(|| "missing prepared_units output".to_string())?;
    validate_manifest_artifact(prepared, true, "prepared_units")?;
    require_manifest_artifact_identity(
        prepared,
        PREPARED_BYTES,
        PREPARED_SHA256,
        Some(1_000),
        "prepared_units output",
    )?;

    let selection = json_object(&value, "selection")?;
    require_exact_map_keys(
        selection,
        &[
            "black_parents",
            "dropped_over_two_per_opening",
            "post_guard_units",
            "rejections",
            "selected_distinct_openings_by_split",
            "selected_legal_inventory_entries",
            "selected_parent_d4_side_hashes_unique",
            "selected_parents",
            "selected_unique_parent_d4_side_hashes",
            "selected_units",
            "selected_units_by_split",
            "structural_units",
            "white_parents",
        ],
        "Phase-2 selection",
    )?;
    for (key, expected) in [
        ("structural_units", 2_911usize),
        ("post_guard_units", 2_833),
        ("dropped_over_two_per_opening", 1_647),
        ("selected_legal_inventory_entries", 428_320),
        ("selected_parents", 2_000),
        ("selected_unique_parent_d4_side_hashes", 2_000),
        ("selected_units", 1_000),
        ("black_parents", 1_000),
        ("white_parents", 1_000),
    ] {
        let observed = json_usize_map(selection, key)?;
        if observed != expected {
            return Err(format!(
                "Phase-2 selection {key}={observed}, expected {expected}"
            ));
        }
    }
    if !json_bool_map(selection, "selected_parent_d4_side_hashes_unique")? {
        return Err("Phase-2 parent hash uniqueness flag is false".to_string());
    }

    let rejections = json_object_map(selection, "rejections")?;
    require_exact_map_keys(
        rejections,
        &[
            "base_guard_child",
            "historical_opening",
            "historical_parent",
        ],
        "Phase-2 rejections",
    )?;
    for (key, expected) in [
        ("base_guard_child", 78usize),
        ("historical_opening", 0),
        ("historical_parent", 0),
    ] {
        if json_usize_map(rejections, key)? != expected {
            return Err(format!("Phase-2 rejection count mismatch for {key}"));
        }
    }

    let split_units = parse_split_counts(
        json_object_map(selection, "selected_units_by_split")?,
        [700, 150, 150],
        "selected units",
    )?;
    let openings = parse_split_counts(
        json_object_map(selection, "selected_distinct_openings_by_split")?,
        [394, 85, 81],
        "selected distinct openings",
    )?;

    Ok(ManifestSummary {
        selected_units: 1_000,
        selected_parents: 2_000,
        selected_inventory_entries: 428_320,
        split_units,
        distinct_openings_by_split: openings,
    })
}

fn validate_manifest_artifact(value: &Value, has_rows: bool, label: &str) -> Result<(), String> {
    let keys: &[&str] = if has_rows {
        &["bytes", "path", "rows", "sha256"]
    } else {
        &["bytes", "path", "sha256"]
    };
    require_exact_keys(value, keys, label)?;
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    if json_str_map(object, "path")?.is_empty() {
        return Err(format!("{label} path is empty"));
    }
    let _ = json_u64_map(object, "bytes")?;
    require_upper_hash(json_str_map(object, "sha256")?, &format!("{label} sha256"))?;
    if has_rows {
        let _ = json_usize_map(object, "rows")?;
    }
    Ok(())
}

fn require_manifest_artifact_identity(
    value: &Value,
    expected_bytes: u64,
    expected_sha256: &str,
    expected_rows: Option<usize>,
    label: &str,
) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    if json_u64_map(object, "bytes")? != expected_bytes
        || json_str_map(object, "sha256")? != expected_sha256
        || expected_rows
            .map(|rows| json_usize_map(object, "rows").map(|got| got != rows))
            .transpose()?
            .unwrap_or(false)
    {
        return Err(format!("{label} identity mismatch"));
    }
    Ok(())
}

fn parse_split_counts(
    object: &Map<String, Value>,
    expected: [usize; 3],
    label: &str,
) -> Result<BTreeMap<Split, usize>, String> {
    require_exact_map_keys(object, &["dev", "safety", "train"], label)?;
    let pairs = [
        (Split::Train, "train", expected[0]),
        (Split::Dev, "dev", expected[1]),
        (Split::Safety, "safety", expected[2]),
    ];
    let mut result = BTreeMap::new();
    for (split, key, wanted) in pairs {
        let observed = json_usize_map(object, key)?;
        if observed != wanted {
            return Err(format!("{label}.{key}={observed}, expected {wanted}"));
        }
        result.insert(split, observed);
    }
    Ok(result)
}

fn parse_prepared_unit(
    value: &Value,
    models: &ProductModels,
    counts: &mut AnalysisCounts,
) -> Result<PreparedUnit, String> {
    require_exact_keys(
        value,
        &[
            "figrid_ordinal",
            "format",
            "opening_group_hash",
            "parents",
            "split",
            "unit_uid",
        ],
        "prepared unit",
    )?;
    if json_str(value, "format")? != PREPARED_FORMAT {
        return Err("prepared row format mismatch".to_string());
    }
    let unit_uid = upper_hash(json_str(value, "unit_uid")?, "unit UID")?;
    let opening_group_hash = upper_hash(json_str(value, "opening_group_hash")?, "opening hash")?;
    let split = Split::parse(json_str(value, "split")?)?;
    if split != Split::expected_from_opening(&opening_group_hash) {
        return Err("prepared split/opening hash assignment mismatch".to_string());
    }
    let figrid_ordinal = json_usize(value, "figrid_ordinal")?;
    if !ORDINALS.contains(&figrid_ordinal) {
        return Err(format!("unregistered figrid ordinal {figrid_ordinal}"));
    }
    let parents = json_object(value, "parents")?;
    require_exact_map_keys(
        parents,
        &["figrid_black", "figrid_white"],
        "prepared parents",
    )?;
    let black = parse_prepared_parent(
        parents
            .get("figrid_black")
            .ok_or_else(|| "missing figrid_black parent".to_string())?,
        Stone::Black,
        &unit_uid,
        &opening_group_hash,
        models,
        counts,
    )?;
    let white = parse_prepared_parent(
        parents
            .get("figrid_white")
            .ok_or_else(|| "missing figrid_white parent".to_string())?,
        Stone::White,
        &unit_uid,
        &opening_group_hash,
        models,
        counts,
    )?;
    if black.history[..4] != white.history[..4] {
        return Err("Black/White parents do not share literal first-four opening".to_string());
    }
    let computed_unit_uid = expected_unit_uid(
        &opening_group_hash,
        figrid_ordinal,
        &black.parent_d4_side_hash,
        &white.parent_d4_side_hash,
    );
    if computed_unit_uid != unit_uid {
        return Err("unit UID preimage mismatch".to_string());
    }

    let mean_margin = (white.diagnostics.margin + black.diagnostics.margin) / 2.0f32;
    if !mean_margin.is_finite() || mean_margin < 0.0 {
        return Err("mean margin is non-finite or negative".to_string());
    }
    for (color, parent) in [("Black", &black), ("White", &white)] {
        for reason in &parent.quiet_reasons {
            *counts
                .exclusions
                .entry((figrid_ordinal, split, color, reason.name()))
                .or_default() += 1;
        }
    }

    counts.units += 1;
    counts.parents += 2;
    counts.inventory_entries += black.legal_inventory.len() + white.legal_inventory.len();
    *counts.split_units.entry(split).or_default() += 1;
    *counts.color_parents.entry("Black").or_default() += 1;
    *counts.color_parents.entry("White").or_default() += 1;

    Ok(PreparedUnit {
        support_digest: selector_digest("CB-AL1|support-v1|", &unit_uid),
        active_digest: selector_digest("CB-AL1|active-v1|", &unit_uid),
        control_digest: selector_digest("CB-AL1|control-v1|", &unit_uid),
        unit_uid,
        split,
        opening_group_hash,
        figrid_ordinal,
        black,
        white,
        mean_margin,
        mean_margin_bits: mean_margin.to_bits(),
    })
}

fn parse_prepared_parent(
    value: &Value,
    expected_side: Stone,
    unit_uid: &str,
    opening_hash: &str,
    models: &ProductModels,
    counts: &mut AnalysisCounts,
) -> Result<PreparedParent, String> {
    require_exact_keys(
        value,
        &[
            "figrid_actual_move",
            "history",
            "legal_inventory",
            "parent_d4_side_hash",
            "parent_uid",
            "side_to_move",
        ],
        "prepared parent",
    )?;
    let parent_uid = upper_hash(json_str(value, "parent_uid")?, "parent UID")?;
    let parent_hash = upper_hash(
        json_str(value, "parent_d4_side_hash")?,
        "parent D4+side hash",
    )?;
    let side_to_move = parse_stone(json_str(value, "side_to_move")?)?;
    if side_to_move != expected_side {
        return Err(format!(
            "{} parent side mismatch",
            stone_char(expected_side)
        ));
    }

    let history_values = value
        .get("history")
        .and_then(Value::as_array)
        .ok_or_else(|| "history must be an array".to_string())?;
    if history_values.len() < 4 {
        return Err("parent history has fewer than four plies".to_string());
    }
    let mut history = Vec::with_capacity(history_values.len());
    let mut board = Board::new();
    if board.effective_rule_set() != RuleSet::Freestyle {
        return Err("new board is not Freestyle".to_string());
    }
    for (ply, item) in history_values.iter().enumerate() {
        require_exact_keys(item, &["color", "x", "y"], "history stone")?;
        let color = parse_stone(json_str(item, "color")?)?;
        let expected_color = if ply % 2 == 0 {
            Stone::Black
        } else {
            Stone::White
        };
        if color != expected_color || color != board.side_to_move {
            return Err(format!("history color/turn mismatch at ply {ply}"));
        }
        let mv = parse_xy_move(item)?;
        if !board.is_legal_move(mv) {
            return Err(format!("illegal history move at ply {ply}"));
        }
        board.make_move(mv);
        if board.game_result() != GameResult::Ongoing {
            return Err(format!("history is terminal at ply {ply}"));
        }
        history.push(ColoredMove { mv, color });
    }
    if board.side_to_move != side_to_move || board.game_result() != GameResult::Ongoing {
        return Err("parent side/terminal state mismatch".to_string());
    }
    if canonical_position_hash(&history, side_to_move) != parent_hash {
        return Err("parent D4+side hash mismatch".to_string());
    }
    if canonical_opening_hash(&history[..4])? != opening_hash {
        return Err("ordered opening hash mismatch".to_string());
    }
    if expected_parent_uid(unit_uid, side_to_move, &parent_hash) != parent_uid {
        return Err("parent UID preimage mismatch".to_string());
    }

    let actual_value = value
        .get("figrid_actual_move")
        .ok_or_else(|| "missing figrid_actual_move".to_string())?;
    require_exact_keys(actual_value, &["x", "y"], "figrid actual move")?;
    let actual_move = parse_xy_move(actual_value)?;

    let legal = board.legal_moves();
    if legal.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("board legal moves are not in strict cell order".to_string());
    }
    if legal.len() != NUM_CELLS - history.len() {
        return Err("root legal count does not equal empty-cell count".to_string());
    }
    if legal.len() < 6 {
        return Err("prepared parent has fewer than guard_k legal moves".to_string());
    }
    if !legal.contains(&actual_move) {
        return Err("archived actual move is illegal".to_string());
    }

    let root_factored =
        evaluate_full_factored_quantized_for_audit(&board, &models.current_factored);
    let root_flat = evaluate_full_quantized(&board, &models.current_flat);
    if !root_factored.is_finite()
        || !root_flat.is_finite()
        || root_factored.to_bits() != root_flat.to_bits()
    {
        return Err("current root factored/flat parity mismatch".to_string());
    }
    counts.current_root_parity_checks += 1;

    let inventory_values = value
        .get("legal_inventory")
        .and_then(Value::as_array)
        .ok_or_else(|| "legal_inventory must be an array".to_string())?;
    if inventory_values.len() != legal.len() {
        return Err("legal_inventory length mismatch".to_string());
    }
    let mut inventory = Vec::with_capacity(legal.len());
    for (index, (item, &expected_move)) in inventory_values.iter().zip(&legal).enumerate() {
        require_exact_keys(
            item,
            &[
                "base_logit_f32",
                "base_logit_f32_bits",
                "child_d4_side_hash",
                "move",
            ],
            "legal inventory entry",
        )?;
        let move_value = item
            .get("move")
            .ok_or_else(|| format!("inventory[{index}] missing move"))?;
        require_exact_keys(move_value, &["x", "y"], "inventory move")?;
        let mv = parse_xy_move(move_value)?;
        if mv != expected_move {
            return Err(format!("inventory[{index}] is not in exact cell order"));
        }
        let child_hash = upper_hash(
            json_str(item, "child_d4_side_hash")?,
            "inventory child hash",
        )?;
        let (stored_legacy, stored_bits) =
            parse_f32_with_bits(item, "base_logit_f32", "base_logit_f32_bits")?;

        let root_snapshot = board.clone();
        board.make_move(mv);
        let mut child_history = history.clone();
        child_history.push(ColoredMove {
            mv,
            color: side_to_move,
        });
        if canonical_position_hash(&child_history, side_to_move.opponent()) != child_hash {
            return Err(format!("inventory[{index}] child hash mismatch"));
        }

        let current_factored =
            evaluate_full_factored_quantized_for_audit(&board, &models.current_factored);
        let current_flat = evaluate_full_quantized(&board, &models.current_flat);
        if !current_factored.is_finite()
            || !current_flat.is_finite()
            || current_factored.to_bits() != current_flat.to_bits()
        {
            return Err(format!(
                "inventory[{index}] current factored/flat parity mismatch"
            ));
        }
        counts.current_child_parity_checks += 1;
        let current_utility = -current_factored;
        if !current_utility.is_finite() {
            return Err(format!("inventory[{index}] current utility is non-finite"));
        }

        let natural_side = board.side_to_move;
        board.side_to_move = Stone::Black;
        let replayed_legacy = evaluate_full_quantized(&board, &models.legacy);
        board.side_to_move = natural_side;
        if !replayed_legacy.is_finite() || replayed_legacy.to_bits() != stored_bits {
            return Err(format!(
                "inventory[{index}] legacy forced-Black replay mismatch: got {:08X}, expected {stored_bits:08X}",
                replayed_legacy.to_bits()
            ));
        }
        counts.legacy_child_replay_checks += 1;

        board.undo_move();
        require_board_restored(&board, &root_snapshot, index)?;
        inventory.push(InventoryEntry {
            mv,
            child_d4_side_hash: child_hash,
            legacy_black_logit: stored_legacy,
            legacy_black_logit_bits: stored_bits,
            current_child_logit: current_factored,
            current_child_logit_bits: current_factored.to_bits(),
            current_mover_utility: current_utility,
            current_mover_utility_bits: current_utility.to_bits(),
        });
    }

    let quiet_reasons = quiet_reasons(&board, side_to_move, actual_move, &legal);
    let diagnostics = parent_diagnostics(&inventory, actual_move)?;
    Ok(PreparedParent {
        parent_uid,
        parent_d4_side_hash: parent_hash,
        side_to_move,
        history,
        figrid_actual_move: actual_move,
        current_root_logit: root_factored,
        current_root_logit_bits: root_factored.to_bits(),
        legal_inventory: inventory,
        quiet_reasons,
        diagnostics,
    })
}

fn parent_diagnostics(
    inventory: &[InventoryEntry],
    actual_move: Move,
) -> Result<ParentDiagnostics, String> {
    if inventory.len() < 2 {
        return Err("at least two legal moves required for diagnostics".to_string());
    }
    let mut ordered = inventory.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        right
            .current_mover_utility
            .partial_cmp(&left.current_mover_utility)
            .expect("current utilities are finite")
            .then_with(|| left.mv.cmp(&right.mv))
    });
    let scored = |entry: &InventoryEntry| ScoredMove {
        mv: entry.mv,
        utility: entry.current_mover_utility,
        utility_bits: entry.current_mover_utility_bits,
    };
    let top = scored(ordered[0]);
    let second = scored(ordered[1]);
    let actual_entry = inventory
        .iter()
        .find(|entry| entry.mv == actual_move)
        .ok_or_else(|| "archived actual move absent from inventory".to_string())?;
    let actual = scored(actual_entry);
    let margin = top.utility - second.utility;
    let actual_gap = top.utility - actual.utility;
    if !margin.is_finite() || margin < 0.0 || !actual_gap.is_finite() || actual_gap < 0.0 {
        return Err("parent margin/gap is non-finite or negative".to_string());
    }
    Ok(ParentDiagnostics {
        search_disagreement: top.mv != actual_move,
        margin_bits: margin.to_bits(),
        actual_gap_bits: actual_gap.to_bits(),
        static_top: top,
        static_second: second,
        archived_actual: actual,
        margin,
        actual_gap,
    })
}

fn quiet_reasons(
    board: &Board,
    mover: Stone,
    actual_move: Move,
    legal: &[Move],
) -> Vec<QuietReason> {
    let mut reasons = Vec::new();
    if legal.len() < 6 {
        reasons.push(QuietReason::TooFewLegal);
    }
    if has_immediate_five(board, mover) {
        reasons.push(QuietReason::MoverImmediateFive);
    }
    if has_immediate_five(board, mover.opponent()) {
        reasons.push(QuietReason::OpponentImmediateFive);
    }
    if !legal.contains(&actual_move) {
        reasons.push(QuietReason::ActualIllegal);
    }
    reasons
}

fn has_immediate_five(board: &Board, stone: Stone) -> bool {
    const AXES: [(i32, i32); 4] = [(1, 0), (0, 1), (1, 1), (1, -1)];
    (0..NUM_CELLS).any(|mv| {
        if !board.is_empty(mv) {
            return false;
        }
        let x = (mv % BOARD_SIZE) as i32;
        let y = (mv / BOARD_SIZE) as i32;
        AXES.iter().any(|&(dx, dy)| {
            let count = 1
                + contiguous_stones(board, stone, x, y, dx, dy)
                + contiguous_stones(board, stone, x, y, -dx, -dy);
            count >= 5
        })
    })
}

fn contiguous_stones(board: &Board, stone: Stone, x: i32, y: i32, dx: i32, dy: i32) -> usize {
    let mut count = 0usize;
    let mut cx = x + dx;
    let mut cy = y + dy;
    while cx >= 0 && cy >= 0 && cx < BOARD_SIZE as i32 && cy < BOARD_SIZE as i32 {
        let cell = cy as usize * BOARD_SIZE + cx as usize;
        let occupied = match stone {
            Stone::Black => board.black.get(cell),
            Stone::White => board.white.get(cell),
        };
        if !occupied {
            break;
        }
        count += 1;
        cx += dx;
        cy += dy;
    }
    count
}

fn require_board_restored(board: &Board, root: &Board, index: usize) -> Result<(), String> {
    let restored = board.black == root.black
        && board.white == root.white
        && board.side_to_move == root.side_to_move
        && board.move_count == root.move_count
        && board.last_move == root.last_move
        && board.history == root.history
        && board.zobrist == root.zobrist
        && board.line_pattern_ids == root.line_pattern_ids
        && board.rule_set == root.rule_set
        && board.exact5 == root.exact5;
    if restored {
        Ok(())
    } else {
        Err(format!(
            "board/root restoration mismatch after inventory[{index}]"
        ))
    }
}

fn validate_corpus_contract(
    units: &[PreparedUnit],
    counts: &AnalysisCounts,
    manifest: &ManifestSummary,
) -> Result<(), String> {
    if units.len() != 1_000
        || counts.units != 1_000
        || counts.parents != 2_000
        || counts.inventory_entries != 428_320
        || counts.current_root_parity_checks != 2_000
        || counts.current_child_parity_checks != 428_320
        || counts.legacy_child_replay_checks != 428_320
        || counts.color_parents.get("Black").copied() != Some(1_000)
        || counts.color_parents.get("White").copied() != Some(1_000)
    {
        return Err("prepared global count/replay contract mismatch".to_string());
    }
    if counts.split_units != manifest.split_units
        || counts.units != manifest.selected_units
        || counts.parents != manifest.selected_parents
        || counts.inventory_entries != manifest.selected_inventory_entries
    {
        return Err("prepared corpus/manifest count mismatch".to_string());
    }

    let mut unit_uids = BTreeSet::new();
    let mut parent_uids = BTreeSet::new();
    let mut parent_hashes = BTreeSet::new();
    let mut openings: BTreeMap<String, usize> = BTreeMap::new();
    let mut openings_by_split: BTreeMap<Split, BTreeSet<String>> = BTreeMap::new();
    for unit in units {
        if !unit_uids.insert(unit.unit_uid.as_str()) {
            return Err(format!("duplicate unit UID {}", unit.unit_uid));
        }
        *openings.entry(unit.opening_group_hash.clone()).or_default() += 1;
        openings_by_split
            .entry(unit.split)
            .or_default()
            .insert(unit.opening_group_hash.clone());
        for parent in [&unit.black, &unit.white] {
            if !parent_uids.insert(parent.parent_uid.as_str()) {
                return Err(format!("duplicate parent UID {}", parent.parent_uid));
            }
            if !parent_hashes.insert(parent.parent_d4_side_hash.as_str()) {
                return Err(format!(
                    "duplicate parent D4+side hash {}",
                    parent.parent_d4_side_hash
                ));
            }
            if parent.legal_inventory.len() != NUM_CELLS - parent.history.len() {
                return Err("prepared parent inventory is not full legal".to_string());
            }
        }
    }
    if unit_uids.len() != 1_000
        || parent_uids.len() != 2_000
        || parent_hashes.len() != 2_000
        || openings.values().any(|&count| count > 2)
    {
        return Err("prepared uniqueness/opening-cap contract mismatch".to_string());
    }
    let observed_openings = [Split::Train, Split::Dev, Split::Safety]
        .into_iter()
        .map(|split| {
            (
                split,
                openings_by_split
                    .get(&split)
                    .map(BTreeSet::len)
                    .unwrap_or(0),
            )
        })
        .collect::<BTreeMap<_, _>>();
    if observed_openings != manifest.distinct_openings_by_split {
        return Err("prepared distinct-opening counts do not reproduce manifest".to_string());
    }
    Ok(())
}

fn select_arms(units: &[PreparedUnit]) -> Result<(SelectorStreams, SelectorSupportStatus), String> {
    let first = select_arms_once(units)?;
    let second = select_arms_once(units)?;
    if second != first {
        return Err("second in-process selector pass changed output".to_string());
    }
    Ok(first)
}

fn select_arms_once(
    units: &[PreparedUnit],
) -> Result<(SelectorStreams, SelectorSupportStatus), String> {
    let mut support_by_ordinal = BTreeMap::new();
    let mut active_by_ordinal = BTreeMap::new();
    let mut control_by_ordinal = BTreeMap::new();
    let mut support_ready = true;

    for ordinal in ORDINALS {
        let mut eligible = units
            .iter()
            .filter(|unit| {
                unit.split == Split::Train
                    && unit.figrid_ordinal == ordinal
                    && unit.quiet_eligible()
            })
            .collect::<Vec<_>>();
        require_unique_digest(
            &eligible,
            |unit| unit.support_digest,
            &format!("support ordinal {ordinal}"),
        )?;
        eligible.sort_by_key(|unit| unit.support_digest);
        if eligible.len() < SUPPORT_PER_ORDINAL {
            support_ready = false;
        }
        eligible.truncate(SUPPORT_PER_ORDINAL);
        let support_refs = eligible;
        support_by_ordinal.insert(
            ordinal,
            support_refs
                .iter()
                .map(|unit| unit.unit_uid.clone())
                .collect(),
        );

        require_unique_digest(
            &support_refs,
            |unit| unit.active_digest,
            &format!("active ordinal {ordinal}"),
        )?;
        let mut active = support_refs.clone();
        active.sort_by(|left, right| {
            right
                .white
                .diagnostics
                .search_disagreement
                .cmp(&left.white.diagnostics.search_disagreement)
                .then_with(|| {
                    right
                        .white
                        .diagnostics
                        .actual_gap
                        .total_cmp(&left.white.diagnostics.actual_gap)
                })
                .then_with(|| {
                    left.white
                        .diagnostics
                        .margin
                        .total_cmp(&right.white.diagnostics.margin)
                })
                .then_with(|| left.mean_margin.total_cmp(&right.mean_margin))
                .then_with(|| left.active_digest.cmp(&right.active_digest))
        });
        active.truncate(ARM_PER_ORDINAL);
        active_by_ordinal.insert(
            ordinal,
            active.iter().map(|unit| unit.unit_uid.clone()).collect(),
        );

        require_unique_digest(
            &support_refs,
            |unit| unit.control_digest,
            &format!("control ordinal {ordinal}"),
        )?;
        let mut control = support_refs.clone();
        control.sort_by_key(|unit| unit.control_digest);
        control.truncate(ARM_PER_ORDINAL);
        control_by_ordinal.insert(
            ordinal,
            control.iter().map(|unit| unit.unit_uid.clone()).collect(),
        );
    }

    let flatten = |map: &BTreeMap<usize, Vec<String>>| {
        ORDINALS
            .iter()
            .flat_map(|ordinal| map.get(ordinal).into_iter().flatten().cloned())
            .collect::<Vec<_>>()
    };
    let support = flatten(&support_by_ordinal);
    let active = flatten(&active_by_ordinal);
    let control = flatten(&control_by_ordinal);
    require_unique_uid_stream(&support, "support")?;
    require_unique_uid_stream(&active, "active")?;
    require_unique_uid_stream(&control, "control")?;

    let active_set = active.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let control_set = control.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let overlap_units = active_set.intersection(&control_set).count();
    let (active_distinct_openings, active_opening_cap) = arm_opening_audit(units, &active)?;
    let (control_distinct_openings, control_opening_cap) = arm_opening_audit(units, &control)?;

    let exact_cardinality = support.len() == ORDINALS.len() * SUPPORT_PER_ORDINAL
        && active.len() == ORDINALS.len() * ARM_PER_ORDINAL
        && control.len() == ORDINALS.len() * ARM_PER_ORDINAL
        && ORDINALS.iter().all(|ordinal| {
            support_by_ordinal.get(ordinal).map(Vec::len) == Some(SUPPORT_PER_ORDINAL)
                && active_by_ordinal.get(ordinal).map(Vec::len) == Some(ARM_PER_ORDINAL)
                && control_by_ordinal.get(ordinal).map(Vec::len) == Some(ARM_PER_ORDINAL)
        });
    let ready = support_ready
        && exact_cardinality
        && overlap_units <= 50
        && active_opening_cap <= 2
        && control_opening_cap <= 2
        && active_distinct_openings >= 63
        && control_distinct_openings >= 63;
    let selector = SelectorStreams {
        support_sha256: uid_stream_hash(support.iter().map(String::as_str)),
        active_sha256: uid_stream_hash(active.iter().map(String::as_str)),
        control_sha256: uid_stream_hash(control.iter().map(String::as_str)),
        support_by_ordinal,
        active_by_ordinal,
        control_by_ordinal,
        support,
        active,
        control,
        overlap_units,
        active_distinct_openings,
        control_distinct_openings,
    };
    Ok((
        selector,
        if ready {
            SelectorSupportStatus::ReadyForReveal
        } else {
            SelectorSupportStatus::NoGoSelectorSupport
        },
    ))
}

fn require_unique_digest<F>(units: &[&PreparedUnit], digest: F, label: &str) -> Result<(), String>
where
    F: Fn(&PreparedUnit) -> Digest,
{
    let mut seen = BTreeSet::new();
    for unit in units {
        if !seen.insert(digest(unit)) {
            return Err(format!("{label} SHA-256 collision"));
        }
    }
    Ok(())
}

fn require_unique_uid_stream(uids: &[String], label: &str) -> Result<(), String> {
    let unique = uids.iter().collect::<BTreeSet<_>>();
    if unique.len() != uids.len() {
        Err(format!("{label} UID stream contains a duplicate"))
    } else {
        Ok(())
    }
}

fn arm_opening_audit(units: &[PreparedUnit], uids: &[String]) -> Result<(usize, usize), String> {
    let lookup = units
        .iter()
        .map(|unit| (unit.unit_uid.as_str(), unit.opening_group_hash.as_str()))
        .collect::<BTreeMap<_, _>>();
    let mut counts = BTreeMap::<&str, usize>::new();
    for uid in uids {
        let opening = lookup
            .get(uid.as_str())
            .ok_or_else(|| format!("selected UID {uid} absent from prepared units"))?;
        *counts.entry(*opening).or_default() += 1;
    }
    Ok((
        counts.len(),
        counts.values().copied().max().unwrap_or_default(),
    ))
}

fn unit_summary_json(unit: &PreparedUnit) -> Value {
    json!({
        "unit_uid": unit.unit_uid,
        "split": unit.split.name(),
        "opening_group_hash": unit.opening_group_hash,
        "figrid_ordinal": unit.figrid_ordinal,
        "quiet_eligible": unit.quiet_eligible(),
        "mean_margin": f32_json_with_bits(unit.mean_margin, unit.mean_margin_bits),
        "selector_digests": {
            "support": crate::hash::hex_upper(&unit.support_digest),
            "active": crate::hash::hex_upper(&unit.active_digest),
            "control": crate::hash::hex_upper(&unit.control_digest),
        },
        "parents": {
            "figrid_black": parent_summary_json(&unit.black),
            "figrid_white": parent_summary_json(&unit.white),
        },
    })
}

fn parent_summary_json(parent: &PreparedParent) -> Value {
    json!({
        "parent_uid": parent.parent_uid,
        "parent_d4_side_hash": parent.parent_d4_side_hash,
        "side_to_move": stone_char(parent.side_to_move).to_string(),
        "history_plies": parent.history.len(),
        "history": parent.history.iter().enumerate().map(|(ply, item)| json!({
            "ply": ply,
            "color": stone_char(item.color).to_string(),
            "move": move_json(item.mv),
        })).collect::<Vec<_>>(),
        "current_root_logit": f32_json_with_bits(
            parent.current_root_logit,
            parent.current_root_logit_bits,
        ),
        "legal_moves": parent.legal_inventory.len(),
        "child_hash_stream_sha256": uid_stream_hash(
            parent.legal_inventory.iter().map(|entry| entry.child_d4_side_hash.as_str())
        ),
        "legal_inventory": parent.legal_inventory.iter().map(inventory_entry_json).collect::<Vec<_>>(),
        "quiet_eligible": parent.quiet_eligible(),
        "quiet_reasons": parent.quiet_reasons.iter().map(|reason| reason.name()).collect::<Vec<_>>(),
        "static_top": scored_move_json(&parent.diagnostics.static_top),
        "static_second": scored_move_json(&parent.diagnostics.static_second),
        "archived_actual": scored_move_json(&parent.diagnostics.archived_actual),
        "margin": f32_json_with_bits(
            parent.diagnostics.margin,
            parent.diagnostics.margin_bits,
        ),
        "actual_gap": f32_json_with_bits(
            parent.diagnostics.actual_gap,
            parent.diagnostics.actual_gap_bits,
        ),
        "search_disagreement": parent.diagnostics.search_disagreement,
    })
}

fn inventory_entry_json(value: &InventoryEntry) -> Value {
    json!({
        "move": move_json(value.mv),
        "child_d4_side_hash": value.child_d4_side_hash,
        "legacy_black_logit": f32_json_with_bits(
            value.legacy_black_logit,
            value.legacy_black_logit_bits,
        ),
        "current_child_logit": f32_json_with_bits(
            value.current_child_logit,
            value.current_child_logit_bits,
        ),
        "current_mover_utility": f32_json_with_bits(
            value.current_mover_utility,
            value.current_mover_utility_bits,
        ),
    })
}

fn scored_move_json(value: &ScoredMove) -> Value {
    json!({
        "move": move_json(value.mv),
        "utility": value.utility,
        "utility_bits": format!("{:08X}", value.utility_bits),
    })
}

fn f32_json_with_bits(value: f32, bits: u32) -> Value {
    json!({"value": value, "bits": format!("{bits:08X}")})
}

fn move_json(mv: Move) -> Value {
    json!({"x": mv % BOARD_SIZE, "y": mv / BOARD_SIZE})
}

fn split_map_json(values: &BTreeMap<Split, usize>) -> Value {
    json!({
        "train": values.get(&Split::Train).copied().unwrap_or_default(),
        "dev": values.get(&Split::Dev).copied().unwrap_or_default(),
        "safety": values.get(&Split::Safety).copied().unwrap_or_default(),
    })
}

fn ordinal_map_json(values: &BTreeMap<usize, Vec<String>>) -> Value {
    ORDINALS
        .iter()
        .map(|ordinal| {
            (
                ordinal.to_string(),
                json!(values.get(ordinal).cloned().unwrap_or_default()),
            )
        })
        .collect::<Map<_, _>>()
        .into()
}

fn require_exact_keys(value: &Value, expected: &[&str], label: &str) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    require_exact_map_keys(object, expected, label)
}

fn require_exact_map_keys(
    object: &Map<String, Value>,
    expected: &[&str],
    label: &str,
) -> Result<(), String> {
    let actual = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(format!(
            "{label} key mismatch: actual={actual:?}, expected={expected:?}"
        ));
    }
    Ok(())
}

fn json_object<'a>(value: &'a Value, key: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{key} must be an object"))
}

fn json_object_map<'a>(
    value: &'a Map<String, Value>,
    key: &str,
) -> Result<&'a Map<String, Value>, String> {
    value
        .get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{key} must be an object"))
}

fn json_str<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{key} must be a string"))
}

fn json_str_map<'a>(value: &'a Map<String, Value>, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{key} must be a string"))
}

fn json_bool_map(value: &Map<String, Value>, key: &str) -> Result<bool, String> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{key} must be a boolean"))
}

fn json_u64_map(value: &Map<String, Value>, key: &str) -> Result<u64, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{key} must be a nonnegative integer"))
}

fn json_usize(value: &Value, key: &str) -> Result<usize, String> {
    let raw = value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{key} must be a nonnegative integer"))?;
    usize::try_from(raw).map_err(|_| format!("{key} does not fit usize"))
}

fn json_usize_map(value: &Map<String, Value>, key: &str) -> Result<usize, String> {
    let raw = json_u64_map(value, key)?;
    usize::try_from(raw).map_err(|_| format!("{key} does not fit usize"))
}

fn parse_xy_move(value: &Value) -> Result<Move, String> {
    let x = json_usize(value, "x")?;
    let y = json_usize(value, "y")?;
    if x >= BOARD_SIZE || y >= BOARD_SIZE {
        return Err(format!("move ({x},{y}) is outside the 15x15 board"));
    }
    Ok(y * BOARD_SIZE + x)
}

fn parse_stone(value: &str) -> Result<Stone, String> {
    match value {
        "B" => Ok(Stone::Black),
        "W" => Ok(Stone::White),
        _ => Err(format!("invalid stone {value:?}")),
    }
}

fn require_upper_hash(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'A'..=b'F').contains(&byte))
    {
        return Err(format!(
            "{label} must be 64 uppercase hexadecimal characters"
        ));
    }
    Ok(())
}

fn upper_hash(value: &str, label: &str) -> Result<String, String> {
    require_upper_hash(value, label)?;
    Ok(value.to_string())
}

fn parse_f32_with_bits(
    value: &Value,
    float_key: &str,
    bits_key: &str,
) -> Result<(f32, u32), String> {
    let rendered = value
        .get(float_key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{float_key} must be a JSON number"))?;
    let float = rendered as f32;
    if !float.is_finite() {
        return Err(format!(
            "{float_key} is non-finite after binary32 conversion"
        ));
    }
    let bits_text = json_str(value, bits_key)?;
    if bits_text.len() != 8
        || !bits_text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'A'..=b'F').contains(&byte))
    {
        return Err(format!("{bits_key} must be eight uppercase hex characters"));
    }
    let bits = u32::from_str_radix(bits_text, 16)
        .map_err(|error| format!("{bits_key} is invalid hex: {error}"))?;
    if float.to_bits() != bits {
        return Err(format!(
            "{float_key}/{bits_key} mismatch: parsed {:08X}, stored {bits:08X}",
            float.to_bits()
        ));
    }
    Ok((float, bits))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_keys_and_f32_bits_fail_closed() {
        assert!(require_exact_keys(&json!({"a": 1, "b": 2}), &["a", "b"], "row").is_ok());
        assert!(require_exact_keys(&json!({"a": 1}), &["a", "b"], "row").is_err());
        assert!(
            parse_f32_with_bits(&json!({"value": 1.5, "bits": "3FC00000"}), "value", "bits")
                .is_ok()
        );
        assert!(
            parse_f32_with_bits(&json!({"value": 1.5, "bits": "3F800000"}), "value", "bits")
                .is_err()
        );
    }

    #[test]
    fn exact_quantizer_uses_ties_away_and_rejects_overflow() {
        assert_eq!(quantize_vector(&[0.5, -0.5], 1, "x").unwrap(), [1, -1]);
        assert!(quantize_vector(&[f32::MAX], 2048, "x").is_err());
    }

    #[test]
    fn quiet_audit_checks_both_colors_without_changing_turn() {
        let mut board = Board::new();
        for mv in [0, 15, 1, 16, 2, 17, 3, 30] {
            board.make_move(mv);
        }
        let side = board.side_to_move;
        assert!(has_immediate_five(&board, Stone::Black));
        assert!(!has_immediate_five(&board, Stone::White));
        assert_eq!(board.side_to_move, side);
    }
}
