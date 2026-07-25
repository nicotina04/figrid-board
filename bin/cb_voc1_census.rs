#![cfg(feature = "codebook-eval")]

//! CB-VOC1 Stage A0/A1 offline census.
//!
//! This binary is deliberately not linked into `pbrain-figrid`.  It accepts
//! only the five preregistered model/corpus inputs and has no CLI surface for
//! RQ615C dev, safety, the 64-game holdout, or game outcomes.

use figrid_board::board::{Board, GameResult, Move, Stone, BOARD_SIZE, NUM_CELLS};
use figrid_board::codebook_eval::{
    evaluate_full_quantized, CodebookWeights, QuantizedCodebookWeights,
};
use figrid_board::pattern_table::{
    canonicalize, lookup_mapped_id, pack_window, read_window, swap_mapped_id, PATTERN_NUM_IDS,
    PATTERN_RARE_ID,
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

#[path = "cb_voc1_census/a1.rs"]
mod a1;
#[path = "cb_voc1_census/gradient.rs"]
mod gradient;
#[path = "cb_voc1_census/orbit.rs"]
mod orbit;

const K6: usize = 6;
const DIM: usize = 16;
const FM_RANK: usize = 8;
const REGIONS: usize = 9;
const PRODUCT_EMBED_SCALE: i32 = 32;
const PRODUCT_HEAD_SCALE: i32 = 64;
const PRODUCT_FACTOR_SCALE: i32 = 64;
const LINEAGE_HEAD_SCALE: i32 = 2048;
const LINEAGE_FACTOR_SCALE: i32 = 2048;
const EXPECTED_ROWS: usize = 1_336;
const EXPECTED_UNITS: usize = 668;
const EXPECTED_COMPONENTS: usize = 388;
const EXPECTED_COLOR_ROWS: usize = 668;
const ROW_FORMAT: &str = "rq615c-k6-projected-slate-v1";
const MANIFEST_FORMAT: &str = "rq615c-phase4-projection-manifest-v1";

const PRODUCT_MODEL_SEAL: ArtifactSeal = ArtifactSeal {
    bytes: 1_410_562,
    sha256: "42968FDAB01BA8CCD1DE3DED05C532E4B237DD47EEFFD7AE1C2F264D77BA7DA2",
};
const TOPK_SEAL: ArtifactSeal = ArtifactSeal {
    bytes: 17_060,
    sha256: "103891DCD1DCD978C654593ABE78EF32C56E2E350B500EE665BC45AC051AA16D",
};
const TRAIN_SEAL: ArtifactSeal = ArtifactSeal {
    bytes: 54_991_200,
    sha256: "E00A2DA513B05D7631A01003C7E6274E9A3D7575E2C2BD92D5199F1B5385CEB6",
};
const MANIFEST_SEAL: ArtifactSeal = ArtifactSeal {
    bytes: 5_463,
    sha256: "579D1387D7E4DE8F5CB34DB168B6D15655DB229D992751B1DC17BB6CF4260AA7",
};
const LINEAGE_MODEL_SEAL: ArtifactSeal = ArtifactSeal {
    bytes: 1_413_542,
    sha256: "69BB7C599ADA3A1151577CE3315356BC33C40EDB49A003C9BC4EB90A98F82E18",
};

#[derive(Clone, Copy)]
struct ArtifactSeal {
    bytes: u64,
    sha256: &'static str,
}

#[derive(Debug)]
struct Args {
    product_model: PathBuf,
    topk: PathBuf,
    train: PathBuf,
    manifest: PathBuf,
    lineage_model: PathBuf,
    out_report: PathBuf,
}

#[derive(Clone, Debug)]
pub(crate) struct Candidate {
    pub(crate) mv: Move,
    pub(crate) stored_black_logit: f32,
    pub(crate) stored_child_hash: String,
}

#[derive(Clone)]
pub(crate) struct Slate {
    pub(crate) row_uid: String,
    pub(crate) component_uid: String,
    pub(crate) opening_hash: String,
    pub(crate) parent_hash: String,
    pub(crate) ordinal: usize,
    pub(crate) root_side: Stone,
    pub(crate) history: Vec<(Move, Stone)>,
    pub(crate) parent: Board,
    pub(crate) candidates: [Candidate; K6],
    pub(crate) q_teacher: [f64; K6],
}

#[derive(Clone)]
pub(crate) struct IndependentForward {
    pub(crate) precast: f64,
    pub(crate) value: f32,
    pub(crate) raw_tokens: Box<[[u32; 4]; NUM_CELLS]>,
    pub(crate) cell_preactivations: Vec<i32>,
    pub(crate) features: [i32; REGIONS * DIM],
    pub(crate) beta: [f64; REGIONS * DIM],
}

#[derive(Default)]
struct ReplayStats {
    product_mismatches: usize,
    lineage_mismatches: usize,
    raw_mapped_mismatches: usize,
    children: usize,
    product_precast_bits: Vec<u8>,
    product_f32_bits: Vec<u8>,
    forced_black_product_minus_lineage_abs: Vec<f64>,
}

#[derive(Default, Clone, Copy)]
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

fn main() {
    if let Err(error) = run() {
        eprintln!("CB-VOC1 INVALID_CB_VOC1: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    refuse_existing(&args.out_report)?;

    let product_bytes = read_sealed(&args.product_model, PRODUCT_MODEL_SEAL, "product model")?;
    let topk_bytes = read_sealed(&args.topk, TOPK_SEAL, "topk")?;
    let train_bytes = read_sealed(&args.train, TRAIN_SEAL, "RQ615C train")?;
    let manifest_bytes = read_sealed(&args.manifest, MANIFEST_SEAL, "RQ615C manifest")?;
    let lineage_bytes = read_sealed(
        &args.lineage_model,
        LINEAGE_MODEL_SEAL,
        "RQ569 lineage model",
    )?;

    validate_manifest(&manifest_bytes)?;
    validate_topk_shape(&topk_bytes)?;

    let product_float = CodebookWeights::from_json_bytes(&product_bytes)?;
    let lineage_float = CodebookWeights::from_json_bytes(&lineage_bytes)?;
    validate_model_shape(&product_float)?;
    validate_model_shape(&lineage_float)?;
    require_float_payload_identity(&product_float, &lineage_float)?;

    let product = product_float.quantize_i16_s32_s64();
    validate_quant_shape(
        &product,
        PRODUCT_EMBED_SCALE,
        PRODUCT_HEAD_SCALE,
        PRODUCT_FACTOR_SCALE,
    )?;
    let lineage = quantize_exact(
        &lineage_float,
        PRODUCT_EMBED_SCALE,
        LINEAGE_HEAD_SCALE,
        LINEAGE_FACTOR_SCALE,
    )?;

    let slates = load_slates(&train_bytes)?;
    let corpus = validate_corpus(&slates)?;
    let mut replay = replay_children(&slates, &product, &lineage)?;
    if replay.children != EXPECTED_ROWS * K6
        || replay.product_mismatches != 0
        || replay.lineage_mismatches != 0
        || replay.raw_mapped_mismatches != 0
    {
        return Err(format!(
            "replay gate failed: children={} product={} lineage={} raw_mapped={}",
            replay.children,
            replay.product_mismatches,
            replay.lineage_mismatches,
            replay.raw_mapped_mismatches
        ));
    }

    replay
        .forced_black_product_minus_lineage_abs
        .sort_by(f64::total_cmp);
    let a1 = a1::run_a1(&slates, &product, &topk_bytes)?;
    let report = json!({
        "format": "cb-voc1-stage-a0-a1-report-v1",
        "status": if a1.passed {
            "A1_PASS_A2_REQUIRED"
        } else {
            "NO_GO_PRECONDITION"
        },
        "claim_boundary": {
            "selector_coordinate": "natural child-to-move ell; root utility is -ell",
            "stored_lineage_coordinate": "forced Black value; diagnostics only",
            "gradient_orbit_optimizer_implemented": true,
            "stage_a2_a3_implemented": false,
            "strength_or_product_claim": false,
            "forbidden_row_inputs_opened": 0,
            "forbidden_inputs": ["RQ615C dev", "RQ615C safety_internal", "64-game holdout"]
        },
        "inputs": {
            "product_model": artifact_json(&args.product_model, PRODUCT_MODEL_SEAL),
            "topk": artifact_json(&args.topk, TOPK_SEAL),
            "train": artifact_json(&args.train, TRAIN_SEAL),
            "manifest": artifact_json(&args.manifest, MANIFEST_SEAL),
            "lineage_model": artifact_json(&args.lineage_model, LINEAGE_MODEL_SEAL)
        },
        "corpus": corpus,
        "quantization": {
            "product": {"embedding": 32, "head": 64, "factor": 64},
            "lineage": {"embedding": 32, "head": 2048, "factor": 2048},
            "float_payload_bit_identical": true
        },
        "a0": {
            "status": "A0_FORWARD_REPLAY_PASS",
            "replay": {
                "children": replay.children,
                "expected_children": EXPECTED_ROWS * K6,
                "product_public_vs_independent_bit_mismatches": replay.product_mismatches,
                "lineage_stored_vs_independent_bit_mismatches": replay.lineage_mismatches,
                "natural_raw_vs_mapped_id_mismatches": replay.raw_mapped_mismatches,
                "product_precast_f64_stream_sha256": sha256_hex(&replay.product_precast_bits),
                "product_f32_stream_sha256": sha256_hex(&replay.product_f32_bits),
                "abs_forced_black_product_minus_lineage": describe_sorted(
                    &replay.forced_black_product_minus_lineage_abs
                )
            }
        },
        "a1": a1.report,
        "downstream": if a1.passed {
            json!({
                "next_stage": "A2_COMPONENT_HELD_OUT_ROBUSTNESS",
                "a2_opened": true,
                "a3_opened": false,
                "retraining_opened": false,
                "artifact_build_opened": false,
                "benchmark_opened": false,
                "arena_opened": false
            })
        } else {
            json!({
                "next_stage": "STOP_CB_VOC1",
                "a2_opened": false,
                "a3_opened": false,
                "retraining_opened": false,
                "artifact_build_opened": false,
                "benchmark_opened": false,
                "arena_opened": false,
                "reason": "one or more preregistered A1 gates failed"
            })
        }
    });
    let report_bytes = serde_json::to_vec_pretty(&report)
        .map_err(|error| format!("failed to serialize report: {error}"))?;

    recheck_inputs(&args)?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.out_report)
        .map_err(|error| format!("failed to create {}: {error}", args.out_report.display()))?;
    output
        .write_all(&report_bytes)
        .and_then(|_| output.write_all(b"\n"))
        .map_err(|error| format!("failed to write {}: {error}", args.out_report.display()))?;
    println!(
        "CB-VOC1 {}: rows={} children={} product_mismatch=0 lineage_mismatch=0",
        if a1.passed {
            "A1_PASS_A2_REQUIRED"
        } else {
            "NO_GO_PRECONDITION"
        },
        slates.len(),
        replay.children
    );
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let mut values = BTreeMap::<String, String>::new();
    let mut iter = env::args().skip(1);
    while let Some(option) = iter.next() {
        if option == "-h" || option == "--help" {
            return Err(usage().to_string());
        }
        if !matches!(
            option.as_str(),
            "--product-model"
                | "--topk"
                | "--train"
                | "--manifest"
                | "--lineage-model"
                | "--out-report"
        ) {
            return Err(format!(
                "unknown or forbidden option {option:?}\n{}",
                usage()
            ));
        }
        let value = iter
            .next()
            .ok_or_else(|| format!("missing value for {option}"))?;
        if values.insert(option.clone(), value).is_some() {
            return Err(format!("duplicate option {option}"));
        }
    }
    let mut take = |name: &str| {
        values
            .remove(name)
            .map(PathBuf::from)
            .ok_or_else(|| format!("missing required {name}\n{}", usage()))
    };
    let args = Args {
        product_model: take("--product-model")?,
        topk: take("--topk")?,
        train: take("--train")?,
        manifest: take("--manifest")?,
        lineage_model: take("--lineage-model")?,
        out_report: take("--out-report")?,
    };
    debug_assert!(values.is_empty());
    Ok(args)
}

fn usage() -> &'static str {
    "usage: cb-voc1-census --product-model MODEL.json --topk topk.bin \
     --train rq615c_k6_train.jsonl --manifest rq615c_final_corpus_manifest.json \
     --lineage-model rq569_model.json --out-report NEW.json"
}

fn refuse_existing(path: &Path) -> Result<(), String> {
    if path.exists() {
        return Err(format!("refusing to overwrite {}", path.display()));
    }
    Ok(())
}

fn artifact_json(path: &Path, seal: ArtifactSeal) -> Value {
    json!({"path": path, "bytes": seal.bytes, "sha256": seal.sha256})
}

fn read_sealed(path: &Path, seal: ArtifactSeal, name: &str) -> Result<Vec<u8>, String> {
    let bytes = fs::read(path).map_err(|error| format!("failed to read {name}: {error}"))?;
    if bytes.len() as u64 != seal.bytes {
        return Err(format!(
            "{name} byte mismatch: got {}, expected {}",
            bytes.len(),
            seal.bytes
        ));
    }
    let observed = sha256_hex(&bytes);
    if observed != seal.sha256 {
        return Err(format!(
            "{name} SHA-256 mismatch: got {observed}, expected {}",
            seal.sha256
        ));
    }
    Ok(bytes)
}

fn recheck_inputs(args: &Args) -> Result<(), String> {
    read_sealed(
        &args.product_model,
        PRODUCT_MODEL_SEAL,
        "product model recheck",
    )?;
    read_sealed(&args.topk, TOPK_SEAL, "topk recheck")?;
    read_sealed(&args.train, TRAIN_SEAL, "train recheck")?;
    read_sealed(&args.manifest, MANIFEST_SEAL, "manifest recheck")?;
    read_sealed(
        &args.lineage_model,
        LINEAGE_MODEL_SEAL,
        "lineage model recheck",
    )?;
    Ok(())
}

fn validate_manifest(bytes: &[u8]) -> Result<(), String> {
    let value: Value =
        serde_json::from_slice(bytes).map_err(|error| format!("invalid manifest JSON: {error}"))?;
    if json_str(&value, "format")? != MANIFEST_FORMAT
        || json_str(&value, "status")? != "READY_FOR_RQ615D"
    {
        return Err("manifest format/status mismatch".to_string());
    }
    let counts = json_object(&value, "counts")?;
    require_nested_usize(counts, "final_slates_by_split", "train", EXPECTED_ROWS)?;
    require_nested_usize(counts, "final_units_by_split", "train", EXPECTED_UNITS)?;
    require_nested_usize(
        counts,
        "final_clusters_by_split",
        "train",
        EXPECTED_COMPONENTS,
    )?;
    let outputs = json_object(&value, "outputs")?;
    let train = json_object_map(outputs, "train")?;
    if json_usize_map(train, "rows")? != EXPECTED_ROWS
        || json_usize_map(train, "bytes")? as u64 != TRAIN_SEAL.bytes
        || json_str_map(train, "sha256")? != TRAIN_SEAL.sha256
    {
        return Err("manifest train output identity mismatch".to_string());
    }
    let overlap = json_object(&value, "overlap_audit")?;
    if !json_bool_map(overlap, "all_cross_split_overlaps_zero")?
        || !json_bool_map(overlap, "exact_paired_colors")?
    {
        return Err("manifest overlap/color contract failed".to_string());
    }
    let colors = json_object_map(overlap, "colors_by_split")?;
    let train_colors = json_object_map(colors, "train")?;
    if json_usize_map(train_colors, "black")? != EXPECTED_COLOR_ROWS
        || json_usize_map(train_colors, "white")? != EXPECTED_COLOR_ROWS
    {
        return Err("manifest train color count mismatch".to_string());
    }
    Ok(())
}

fn require_nested_usize(
    parent: &serde_json::Map<String, Value>,
    object: &str,
    key: &str,
    expected: usize,
) -> Result<(), String> {
    let object = json_object_map(parent, object)?;
    let got = json_usize_map(object, key)?;
    if got != expected {
        return Err(format!("{object:?}.{key}={got}, expected {expected}"));
    }
    Ok(())
}

fn validate_topk_shape(bytes: &[u8]) -> Result<(), String> {
    if bytes.len() != (PATTERN_NUM_IDS - 1) * 4 {
        return Err("topk length does not match released PATTERN_NUM_IDS".to_string());
    }
    let mut seen = BTreeSet::new();
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        let packed = u32::from_le_bytes(chunk.try_into().expect("four-byte chunk"));
        if !seen.insert(packed) {
            return Err(format!("duplicate topk packed token at id {index}"));
        }
        if lookup_mapped_id(packed) as usize != index {
            return Err(format!("topk id-order mismatch at id {index}"));
        }
    }
    Ok(())
}

fn validate_model_shape(weights: &CodebookWeights) -> Result<(), String> {
    if weights.dim != DIM
        || weights.fm_rank != FM_RANK
        || weights.embeddings.len() != PATTERN_NUM_IDS * DIM
        || weights.head.len() != REGIONS * DIM
        || weights.factors.len() != REGIONS * DIM * FM_RANK
    {
        return Err("codebook model shape mismatch".to_string());
    }
    Ok(())
}

fn validate_quant_shape(
    weights: &QuantizedCodebookWeights,
    embed_scale: i32,
    head_scale: i32,
    factor_scale: i32,
) -> Result<(), String> {
    if weights.dim != DIM
        || weights.fm_rank != FM_RANK
        || weights.embedding_scale != embed_scale
        || weights.head_scale != head_scale
        || weights.factor_scale != factor_scale
        || weights.embeddings.len() != PATTERN_NUM_IDS * DIM
        || weights.head.len() != REGIONS * DIM
        || weights.factors.len() != REGIONS * DIM * FM_RANK
    {
        return Err("quantized codebook shape/scale mismatch".to_string());
    }
    Ok(())
}

fn require_float_payload_identity(
    product: &CodebookWeights,
    lineage: &CodebookWeights,
) -> Result<(), String> {
    if product.dim != lineage.dim
        || product.fm_rank != lineage.fm_rank
        || product.bias.to_bits() != lineage.bias.to_bits()
        || !f32_slice_bits_equal(&product.embeddings, &lineage.embeddings)
        || !f32_slice_bits_equal(&product.head, &lineage.head)
        || !f32_slice_bits_equal(&product.factors, &lineage.factors)
    {
        return Err("product/RQ569 f32 payloads are not bit-identical".to_string());
    }
    Ok(())
}

fn f32_slice_bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(&left, &right)| left.to_bits() == right.to_bits())
}

fn quantize_exact(
    weights: &CodebookWeights,
    embed_scale: i32,
    head_scale: i32,
    factor_scale: i32,
) -> Result<QuantizedCodebookWeights, String> {
    validate_model_shape(weights)?;
    Ok(QuantizedCodebookWeights {
        dim: DIM,
        fm_rank: FM_RANK,
        embedding_scale: embed_scale,
        head_scale,
        factor_scale,
        embeddings: quantize_slice(&weights.embeddings, embed_scale, "embedding")?,
        head: quantize_slice(&weights.head, head_scale, "head")?,
        factors: quantize_slice(&weights.factors, factor_scale, "factor")?,
        bias: weights.bias,
    })
}

fn quantize_slice(values: &[f32], scale: i32, name: &str) -> Result<Vec<i16>, String> {
    values
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            if !value.is_finite() {
                return Err(format!("{name}[{index}] is non-finite"));
            }
            let rounded = (value * scale as f32).round();
            if rounded < i16::MIN as f32 || rounded > i16::MAX as f32 {
                return Err(format!("{name}[{index}] overflows i16"));
            }
            Ok(rounded as i16)
        })
        .collect()
}

fn load_slates(bytes: &[u8]) -> Result<Vec<Slate>, String> {
    let mut rows = Vec::with_capacity(EXPECTED_ROWS);
    let reader = BufReader::new(bytes);
    for (line_index, line) in reader.lines().enumerate() {
        let line = line.map_err(|error| format!("train line {}: {error}", line_index + 1))?;
        if line.is_empty() {
            return Err(format!("blank train line {}", line_index + 1));
        }
        let value: Value = serde_json::from_str(&line)
            .map_err(|error| format!("train line {} invalid JSON: {error}", line_index + 1))?;
        rows.push(
            parse_slate(&value)
                .map_err(|error| format!("train line {}: {error}", line_index + 1))?,
        );
    }
    if rows.len() != EXPECTED_ROWS {
        return Err(format!(
            "train row count {}, expected {EXPECTED_ROWS}",
            rows.len()
        ));
    }
    Ok(rows)
}

fn parse_slate(value: &Value) -> Result<Slate, String> {
    require_exact_keys(
        value,
        &[
            "format",
            "row_uid",
            "final_split",
            "component_uid",
            "opening_group_hash",
            "parent_d4_side_hash",
            "side_to_move",
            "figrid_ordinal",
            "history",
            "candidates",
            "legal_inventory",
            "repeat_scores_mover",
            "repeat_bands_mover",
            "q_teacher",
        ],
    )?;
    if json_str(value, "format")? != ROW_FORMAT || json_str(value, "final_split")? != "train" {
        return Err("row format/split mismatch".to_string());
    }
    let row_uid = upper_hash(json_str(value, "row_uid")?, "row_uid")?;
    let component_uid = upper_hash(json_str(value, "component_uid")?, "component_uid")?;
    let opening_hash = upper_hash(json_str(value, "opening_group_hash")?, "opening hash")?;
    let parent_hash = upper_hash(json_str(value, "parent_d4_side_hash")?, "parent hash")?;
    let ordinal = json_usize(value, "figrid_ordinal")?;
    if !matches!(ordinal, 1 | 2 | 4 | 6 | 8) {
        return Err(format!("unregistered ordinal {ordinal}"));
    }
    let root_side = parse_stone(json_str(value, "side_to_move")?)?;

    let history_values = value
        .get("history")
        .and_then(Value::as_array)
        .ok_or_else(|| "history must be an array".to_string())?;
    if history_values.len() < 4 {
        return Err("history shorter than four-stone opening".to_string());
    }
    let mut parent = Board::new();
    let mut history = Vec::with_capacity(history_values.len());
    for (ply, item) in history_values.iter().enumerate() {
        require_exact_keys(item, &["x", "y", "color"])?;
        let stone = parse_stone(json_str(item, "color")?)?;
        if stone != parent.side_to_move {
            return Err(format!("history turn mismatch at ply {ply}"));
        }
        let mv = parse_move(item)?;
        if !parent.is_legal_move(mv) {
            return Err(format!("illegal history move at ply {ply}"));
        }
        parent.make_move(mv);
        history.push((mv, stone));
    }
    if parent.side_to_move != root_side || parent.game_result() != GameResult::Ongoing {
        return Err("parent side/result invariant failed".to_string());
    }
    if canonical_position_hash(&history, root_side) != parent_hash
        || canonical_opening_hash(&history[..4]) != opening_hash
    {
        return Err("parent/opening hash mismatch".to_string());
    }
    let expected_uid = sha256_hex(
        format!("RQ615C|projected-slate-v1|{opening_hash}|{ordinal}|{parent_hash}").as_bytes(),
    );
    if row_uid != expected_uid {
        return Err("row UID preimage mismatch".to_string());
    }

    let legal = parent.legal_moves();
    let inventory_values = value
        .get("legal_inventory")
        .and_then(Value::as_array)
        .ok_or_else(|| "legal_inventory must be an array".to_string())?;
    if inventory_values.len() != legal.len() {
        return Err("legal inventory length mismatch".to_string());
    }
    let mut inventory = BTreeMap::<Move, (String, f32)>::new();
    for (index, (item, &expected_move)) in inventory_values.iter().zip(&legal).enumerate() {
        require_exact_keys(
            item,
            &[
                "move",
                "child_d4_side_hash",
                "base_logit_f32",
                "base_logit_f32_bits",
            ],
        )?;
        let move_value = item
            .get("move")
            .ok_or_else(|| format!("inventory[{index}] missing move"))?;
        require_exact_keys(move_value, &["x", "y"])?;
        let mv = parse_move(move_value)?;
        if mv != expected_move {
            return Err(format!("inventory[{index}] not in exact cell order"));
        }
        let child_hash = upper_hash(
            json_str(item, "child_d4_side_hash")?,
            "inventory child hash",
        )?;
        let logit = f32_with_bits(item, "base_logit_f32", "base_logit_f32_bits")?;
        let mut child_history = history.clone();
        child_history.push((mv, root_side));
        if canonical_position_hash(&child_history, root_side.opponent()) != child_hash {
            return Err(format!("inventory[{index}] child hash mismatch"));
        }
        inventory.insert(mv, (child_hash, logit));
    }

    let candidate_values = value
        .get("candidates")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidates must be an array".to_string())?;
    if candidate_values.len() != K6 {
        return Err("candidate count mismatch".to_string());
    }
    let mut candidate_moves = BTreeSet::new();
    let mut candidates = Vec::with_capacity(K6);
    let mut teacher_top = 0usize;
    let mut deployed_actual = 0usize;
    for (index, item) in candidate_values.iter().enumerate() {
        require_exact_keys(
            item,
            &[
                "candidate_index",
                "move",
                "roles",
                "child_d4_side_hash",
                "base_logit_f32",
                "base_logit_f32_bits",
            ],
        )?;
        if json_usize(item, "candidate_index")? != index {
            return Err("candidate index/order mismatch".to_string());
        }
        let move_value = item
            .get("move")
            .ok_or_else(|| "candidate missing move".to_string())?;
        require_exact_keys(move_value, &["x", "y"])?;
        let mv = parse_move(move_value)?;
        if !candidate_moves.insert(mv) || !parent.is_legal_move(mv) {
            return Err("duplicate or illegal candidate move".to_string());
        }
        let child_hash = upper_hash(json_str(item, "child_d4_side_hash")?, "child hash")?;
        let stored = f32_with_bits(item, "base_logit_f32", "base_logit_f32_bits")?;
        let inventory_entry = inventory
            .get(&mv)
            .ok_or_else(|| "candidate absent from inventory".to_string())?;
        if inventory_entry.0 != child_hash || inventory_entry.1.to_bits() != stored.to_bits() {
            return Err("candidate/inventory identity mismatch".to_string());
        }
        let roles = item
            .get("roles")
            .and_then(Value::as_array)
            .ok_or_else(|| "roles must be an array".to_string())?;
        if roles.is_empty() {
            return Err("candidate roles empty".to_string());
        }
        let mut unique_roles = BTreeSet::new();
        for role in roles {
            let role = role
                .as_str()
                .ok_or_else(|| "candidate role is not a string".to_string())?;
            if !matches!(
                role,
                "teacher_top"
                    | "deployed_actual"
                    | "base_rank_1"
                    | "base_rank_2"
                    | "base_rank_3"
                    | "base_rank_4"
                    | "base_rank_5"
                    | "base_rank_6"
            ) || !unique_roles.insert(role)
            {
                return Err(format!("invalid/duplicate role {role}"));
            }
        }
        teacher_top += usize::from(unique_roles.contains("teacher_top"));
        deployed_actual += usize::from(unique_roles.contains("deployed_actual"));
        candidates.push(Candidate {
            mv,
            stored_black_logit: stored,
            stored_child_hash: child_hash,
        });
    }
    if teacher_top != 1 || deployed_actual != 1 {
        return Err("candidate role census mismatch".to_string());
    }

    let scores = parse_two_by_six_i64(value, "repeat_scores_mover")?;
    let bands = parse_two_by_six_strings(value, "repeat_bands_mover")?;
    for repeat in 0..2 {
        for index in 0..K6 {
            if !(-3000..=3000).contains(&scores[repeat][index])
                || bands[repeat][index] != score_band(scores[repeat][index])
                || bands[0][index] != bands[1][index]
            {
                return Err("teacher score/band invariant failed".to_string());
            }
        }
    }
    let q_values = value
        .get("q_teacher")
        .and_then(Value::as_array)
        .ok_or_else(|| "q_teacher must be an array".to_string())?;
    if q_values.len() != K6 {
        return Err("q_teacher length mismatch".to_string());
    }
    let mut q_teacher = [0.0f64; K6];
    for (index, item) in q_values.iter().enumerate() {
        q_teacher[index] = item
            .as_f64()
            .filter(|value| value.is_finite() && *value > 0.0)
            .ok_or_else(|| "q_teacher contains invalid probability".to_string())?;
    }
    if (q_teacher.iter().sum::<f64>() - 1.0).abs() > 1.0e-12 {
        return Err("q_teacher normalization mismatch".to_string());
    }
    let q0 = softmax_teacher_scores(&scores[0]);
    let q1 = softmax_teacher_scores(&scores[1]);
    for index in 0..K6 {
        if (q_teacher[index] - 0.5 * (q0[index] + q1[index])).abs() > 1.0e-12 {
            return Err("q_teacher repeat-score replay mismatch".to_string());
        }
    }

    Ok(Slate {
        row_uid,
        component_uid,
        opening_hash,
        parent_hash,
        ordinal,
        root_side,
        history,
        parent,
        candidates: candidates
            .try_into()
            .map_err(|_| "candidate array conversion failed".to_string())?,
        q_teacher,
    })
}

fn validate_corpus(slates: &[Slate]) -> Result<Value, String> {
    let mut row_uids = BTreeSet::new();
    let mut parents = BTreeSet::new();
    let mut components = BTreeSet::new();
    let mut ordinals = BTreeSet::new();
    let mut colors = [0usize; 2];
    let mut units = BTreeMap::<(String, usize), Vec<&Slate>>::new();
    for slate in slates {
        if !row_uids.insert(&slate.row_uid) || !parents.insert(&slate.parent_hash) {
            return Err("duplicate row UID or parent hash".to_string());
        }
        components.insert(&slate.component_uid);
        ordinals.insert(slate.ordinal);
        colors[stone_index(slate.root_side)] += 1;
        units
            .entry((slate.opening_hash.clone(), slate.ordinal))
            .or_default()
            .push(slate);
    }
    if components.len() != EXPECTED_COMPONENTS
        || units.len() != EXPECTED_UNITS
        || colors != [EXPECTED_COLOR_ROWS, EXPECTED_COLOR_ROWS]
        || ordinals != BTreeSet::from([1, 2, 4, 6, 8])
    {
        return Err("corpus aggregate census mismatch".to_string());
    }
    for ((opening, ordinal), rows) in &units {
        if rows.len() != 2
            || rows[0].root_side == rows[1].root_side
            || rows[0].component_uid != rows[1].component_uid
        {
            return Err(format!(
                "paired-color unit invariant failed at {opening}/{ordinal}"
            ));
        }
    }
    Ok(json!({
        "rows": slates.len(),
        "units": units.len(),
        "components": components.len(),
        "black_rows": colors[0],
        "white_rows": colors[1],
        "ordinals": ordinals
    }))
}

fn replay_children(
    slates: &[Slate],
    product: &QuantizedCodebookWeights,
    lineage: &QuantizedCodebookWeights,
) -> Result<ReplayStats, String> {
    let mut stats = ReplayStats::default();
    for slate in slates {
        for candidate in &slate.candidates {
            let mut child = slate.parent.clone();
            child.make_move(candidate.mv);
            if child.side_to_move != slate.root_side.opponent() {
                return Err(format!("{} candidate child side mismatch", slate.row_uid));
            }
            let product_independent = independent_forward(&child, product)?;
            let product_public = evaluate_full_quantized(&child, product);
            if product_independent.value.to_bits() != product_public.to_bits() {
                stats.product_mismatches += 1;
            }
            stats
                .product_precast_bits
                .extend_from_slice(&product_independent.precast.to_bits().to_le_bytes());
            stats
                .product_f32_bits
                .extend_from_slice(&product_independent.value.to_bits().to_le_bytes());

            for cell in 0..NUM_CELLS {
                for dir in 0..4 {
                    let raw = product_independent.raw_tokens[cell][dir];
                    let mapped = lookup_mapped_id(raw);
                    let expected = match child.side_to_move {
                        Stone::Black => child.line_pattern_ids[cell][dir],
                        Stone::White => swap_mapped_id(child.line_pattern_ids[cell][dir]),
                    };
                    if mapped != expected {
                        stats.raw_mapped_mismatches += 1;
                    }
                }
            }

            let mut black_child = child.clone();
            black_child.side_to_move = Stone::Black;
            let forced_black_product = independent_forward(&black_child, product)?;
            let lineage_independent = independent_forward(&black_child, lineage)?;
            if lineage_independent.value.to_bits() != candidate.stored_black_logit.to_bits() {
                stats.lineage_mismatches += 1;
            }
            stats
                .forced_black_product_minus_lineage_abs
                .push((forced_black_product.value as f64 - lineage_independent.value as f64).abs());
            stats.children += 1;
        }
    }
    Ok(stats)
}

pub(crate) fn independent_forward(
    board: &Board,
    weights: &QuantizedCodebookWeights,
) -> Result<IndependentForward, String> {
    validate_quant_shape(
        weights,
        weights.embedding_scale,
        weights.head_scale,
        weights.factor_scale,
    )?;
    let (mine, opponent) = match board.side_to_move {
        Stone::Black => (&board.black, &board.white),
        Stone::White => (&board.white, &board.black),
    };
    let mut raw_tokens = Box::new([[0u32; 4]; NUM_CELLS]);
    let mut preactivations = vec![0i32; NUM_CELLS * DIM];
    let mut features = [0i32; REGIONS * DIM];
    const DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
    for cell in 0..NUM_CELLS {
        let row = (cell / BOARD_SIZE) as i32;
        let col = (cell % BOARD_SIZE) as i32;
        for (dir_index, &(dr, dc)) in DIRS.iter().enumerate() {
            let window = read_window(mine, opponent, row, col, dr, dc);
            let packed = pack_window(&canonicalize(&window));
            raw_tokens[cell][dir_index] = packed;
            let mapped = lookup_mapped_id(packed) as usize;
            let embedding_base = mapped * DIM;
            let cell_base = cell * DIM;
            for dimension in 0..DIM {
                preactivations[cell_base + dimension] +=
                    i32::from(weights.embeddings[embedding_base + dimension]);
            }
        }
        let feature_base = region_of_cell(cell) * DIM;
        let cell_base = cell * DIM;
        for dimension in 0..DIM {
            features[feature_base + dimension] += preactivations[cell_base + dimension].max(0);
        }
    }

    let feature_denom = weights.embedding_scale as f64 * 25.0;
    let head_denom = feature_denom * weights.head_scale as f64;
    let factor_denom = feature_denom * weights.factor_scale as f64;
    let mut precast = weights.bias as f64;
    for (&feature, &head) in features.iter().zip(&weights.head) {
        precast += (feature as f64 * head as f64) / head_denom;
    }
    let mut rank_sums = [0.0f64; FM_RANK];
    for (rank, sum_slot) in rank_sums.iter_mut().enumerate() {
        let mut sum = 0.0f64;
        let mut square_sum = 0.0f64;
        for (index, &feature) in features.iter().enumerate() {
            let vx =
                (feature as f64 * weights.factors[index * FM_RANK + rank] as f64) / factor_denom;
            sum += vx;
            square_sum += vx * vx;
        }
        precast += 0.5 * (sum * sum - square_sum);
        *sum_slot = sum;
    }

    let mut beta = [0.0f64; REGIONS * DIM];
    for index in 0..beta.len() {
        let x = features[index] as f64 / feature_denom;
        let mut derivative = weights.head[index] as f64 / weights.head_scale as f64;
        for (rank, &sum) in rank_sums.iter().enumerate() {
            let factor =
                weights.factors[index * FM_RANK + rank] as f64 / weights.factor_scale as f64;
            derivative += factor * (sum - factor * x);
        }
        beta[index] = derivative;
        if !beta[index].is_finite() {
            return Err(format!("non-finite beta at feature {index}"));
        }
    }
    if !precast.is_finite() {
        return Err("non-finite independent product forward".to_string());
    }
    Ok(IndependentForward {
        precast,
        value: precast as f32,
        raw_tokens,
        cell_preactivations: preactivations,
        features,
        beta,
    })
}

fn region_of_cell(cell: usize) -> usize {
    let row = cell / BOARD_SIZE;
    let col = cell % BOARD_SIZE;
    (row / 5).min(2) * 3 + (col / 5).min(2)
}

fn describe_sorted(values: &[f64]) -> Value {
    json!({
        "count": values.len(),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "max": values.last().copied().unwrap_or(0.0)
    })
}

fn percentile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let index = ((values.len() - 1) as f64 * q).round() as usize;
    values[index.min(values.len() - 1)]
}

fn parse_move(value: &Value) -> Result<Move, String> {
    let x = json_usize(value, "x")?;
    let y = json_usize(value, "y")?;
    if x >= BOARD_SIZE || y >= BOARD_SIZE {
        return Err(format!("move outside board: ({x},{y})"));
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

fn stone_index(stone: Stone) -> usize {
    match stone {
        Stone::Black => 0,
        Stone::White => 1,
    }
}

fn stone_char(stone: Stone) -> char {
    match stone {
        Stone::Black => 'B',
        Stone::White => 'W',
    }
}

fn canonical_position_hash(history: &[(Move, Stone)], side: Stone) -> String {
    let mut forms = Vec::with_capacity(8);
    for transform in 0..8 {
        let mut stones = history
            .iter()
            .map(|&(mv, stone)| {
                let (x, y) = (mv % BOARD_SIZE, mv / BOARD_SIZE);
                let (tx, ty) = transform_xy(x, y, transform);
                format!("{}{:03}", stone_char(stone), ty * BOARD_SIZE + tx)
            })
            .collect::<Vec<_>>();
        stones.sort();
        forms.push(format!(
            "rule=0|side={}|{}",
            stone_char(side),
            stones.join(",")
        ));
    }
    let canonical = forms.into_iter().min().expect("eight forms");
    sha256_hex(format!("RQ608-state-v1|{canonical}").as_bytes())
}

fn canonical_opening_hash(history: &[(Move, Stone)]) -> String {
    let mut forms = Vec::with_capacity(8);
    for transform in 0..8 {
        let encoded = history
            .iter()
            .enumerate()
            .map(|(ply, &(mv, stone))| {
                let (x, y) = (mv % BOARD_SIZE, mv / BOARD_SIZE);
                let (tx, ty) = transform_xy(x, y, transform);
                format!("{ply}:{}{:03}", stone_char(stone), ty * BOARD_SIZE + tx)
            })
            .collect::<Vec<_>>();
        forms.push(format!("rule=0|{}", encoded.join(",")));
    }
    let canonical = forms.into_iter().min().expect("eight forms");
    sha256_hex(format!("RQ608-ordered-opening-v1|{canonical}").as_bytes())
}

fn transform_xy(x: usize, y: usize, transform: usize) -> (usize, usize) {
    let n = BOARD_SIZE - 1;
    [
        (x, y),
        (n - y, x),
        (n - x, n - y),
        (y, n - x),
        (n - x, y),
        (x, n - y),
        (y, x),
        (n - y, n - x),
    ][transform]
}

fn parse_two_by_six_i64(value: &Value, key: &str) -> Result<[[i64; K6]; 2], String> {
    let outer = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{key} must be an array"))?;
    if outer.len() != 2 {
        return Err(format!("{key} outer length mismatch"));
    }
    let mut result = [[0i64; K6]; 2];
    for repeat in 0..2 {
        let inner = outer[repeat]
            .as_array()
            .ok_or_else(|| format!("{key}[{repeat}] must be an array"))?;
        if inner.len() != K6 {
            return Err(format!("{key}[{repeat}] length mismatch"));
        }
        for index in 0..K6 {
            result[repeat][index] = inner[index]
                .as_i64()
                .ok_or_else(|| format!("{key}[{repeat}][{index}] not integer"))?;
        }
    }
    Ok(result)
}

fn parse_two_by_six_strings(value: &Value, key: &str) -> Result<[[String; K6]; 2], String> {
    let outer = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{key} must be an array"))?;
    if outer.len() != 2 {
        return Err(format!("{key} outer length mismatch"));
    }
    let mut result: [[String; K6]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| String::new()));
    for repeat in 0..2 {
        let inner = outer[repeat]
            .as_array()
            .ok_or_else(|| format!("{key}[{repeat}] must be an array"))?;
        if inner.len() != K6 {
            return Err(format!("{key}[{repeat}] length mismatch"));
        }
        for index in 0..K6 {
            result[repeat][index] = inner[index]
                .as_str()
                .ok_or_else(|| format!("{key}[{repeat}][{index}] not string"))?
                .to_string();
        }
    }
    Ok(result)
}

fn score_band(score: i64) -> &'static str {
    if score >= 2500 {
        "forced_win"
    } else if score <= -2500 {
        "forced_loss"
    } else {
        "cp"
    }
}

fn softmax_teacher_scores(scores: &[i64; K6]) -> [f64; K6] {
    let scaled: [f64; K6] = std::array::from_fn(|index| scores[index] as f64 / 400.0);
    let max = scaled
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .expect("K6 nonempty");
    let exp: [f64; K6] = std::array::from_fn(|index| (scaled[index] - max).exp());
    let sum: f64 = exp.iter().sum();
    std::array::from_fn(|index| exp[index] / sum)
}

fn f32_with_bits(value: &Value, number_key: &str, bits_key: &str) -> Result<f32, String> {
    let number = value
        .get(number_key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("missing {number_key}"))? as f32;
    if !number.is_finite() {
        return Err(format!("{number_key} non-finite"));
    }
    let bits = json_str(value, bits_key)?;
    if bits.len() != 8
        || !bits
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_lowercase())
    {
        return Err(format!("{bits_key} must be eight uppercase hex digits"));
    }
    let expected =
        u32::from_str_radix(bits, 16).map_err(|error| format!("invalid {bits_key}: {error}"))?;
    if number.to_bits() != expected {
        return Err(format!("{number_key}/{bits_key} mismatch"));
    }
    Ok(number)
}

fn upper_hash(value: &str, name: &str) -> Result<String, String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_lowercase())
    {
        return Err(format!("{name} is not uppercase SHA-256 text"));
    }
    Ok(value.to_string())
}

fn require_exact_keys(value: &Value, expected: &[&str]) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| "expected JSON object".to_string())?;
    let actual = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(format!(
            "JSON key mismatch: actual={actual:?}, expected={expected:?}"
        ));
    }
    Ok(())
}

fn json_str<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string {key}"))
}

fn json_usize(value: &Value, key: &str) -> Result<usize, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("missing usize {key}"))
}

fn json_bool(value: &Value, key: &str) -> Result<bool, String> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("missing bool {key}"))
}

fn json_object<'a>(
    value: &'a Value,
    key: &str,
) -> Result<&'a serde_json::Map<String, Value>, String> {
    value
        .get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("missing object {key}"))
}

fn json_object_map<'a>(
    value: &'a serde_json::Map<String, Value>,
    key: &str,
) -> Result<&'a serde_json::Map<String, Value>, String> {
    value
        .get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("missing object {key}"))
}

fn json_str_map<'a>(
    value: &'a serde_json::Map<String, Value>,
    key: &str,
) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string {key}"))
}

fn json_bool_map(value: &serde_json::Map<String, Value>, key: &str) -> Result<bool, String> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("missing bool {key}"))
}

fn json_usize_map(value: &serde_json::Map<String, Value>, key: &str) -> Result<usize, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("missing usize {key}"))
}

fn sha256_hex(input: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut state = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut padded = input.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in padded.chunks_exact(64) {
        let mut schedule = [0u32; 64];
        for index in 0..16 {
            schedule[index] = u32::from_be_bytes(
                chunk[index * 4..index * 4 + 4]
                    .try_into()
                    .expect("four-byte word"),
            );
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
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let big1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(big1)
                .wrapping_add(choose)
                .wrapping_add(K[index])
                .wrapping_add(schedule[index]);
            let big0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = big0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }
    state.iter().map(|word| format!("{word:08X}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD"
        );
    }

    #[test]
    fn independent_product_forward_is_bit_exact_on_deterministic_position() {
        let float = CodebookWeights::deterministic(DIM, FM_RANK);
        let quant = float.quantize_i16_s32_s64();
        let mut board = Board::new();
        for mv in [112, 111, 97, 113, 127, 98, 128, 83] {
            board.make_move(mv);
        }
        let audit = independent_forward(&board, &quant).unwrap();
        let released = evaluate_full_quantized(&board, &quant);
        assert_eq!(audit.value.to_bits(), released.to_bits());
        assert!(audit.beta.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn natural_raw_ids_match_released_mapped_ids_for_both_colors() {
        let float = CodebookWeights::deterministic(DIM, FM_RANK);
        let quant = float.quantize_i16_s32_s64();
        let mut board = Board::new();
        for mv in [112, 111, 97, 113, 127] {
            board.make_move(mv);
        }
        assert_eq!(board.side_to_move, Stone::White);
        let audit = independent_forward(&board, &quant).unwrap();
        for cell in 0..NUM_CELLS {
            for direction in 0..4 {
                assert_eq!(
                    lookup_mapped_id(audit.raw_tokens[cell][direction]),
                    swap_mapped_id(board.line_pattern_ids[cell][direction])
                );
            }
        }
    }

    #[test]
    fn quantize_exact_can_reproduce_product_scales() {
        let float = CodebookWeights::deterministic(DIM, FM_RANK);
        let public = float.quantize_i16_s32_s64();
        let independent = quantize_exact(
            &float,
            PRODUCT_EMBED_SCALE,
            PRODUCT_HEAD_SCALE,
            PRODUCT_FACTOR_SCALE,
        )
        .unwrap();
        assert_eq!(public.embeddings, independent.embeddings);
        assert_eq!(public.head, independent.head);
        assert_eq!(public.factors, independent.factors);
        assert_eq!(public.bias.to_bits(), independent.bias.to_bits());
    }

    #[test]
    fn cli_surface_has_no_forbidden_row_inputs() {
        let usage = usage();
        assert!(!usage.contains("dev"));
        assert!(!usage.contains("safety"));
        assert!(!usage.contains("holdout"));
    }

    #[test]
    fn rare_id_is_not_a_selectable_topk_row() {
        assert_eq!(PATTERN_RARE_ID as usize, PATTERN_NUM_IDS - 1);
        assert_eq!(TOPK_SEAL.bytes as usize, PATTERN_RARE_ID as usize * 4);
    }
}
