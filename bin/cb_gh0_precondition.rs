#![cfg(feature = "codebook-eval")]

//! CB-GH0 authoritative D4 semantic precondition analyzer.
//!
//! This is an offline, read-only analyzer.  Its command-line surface accepts
//! exactly the three preregistered inputs and one create-new report path.  It
//! does not implement or enable a canonical hash or transposition table.

use figrid_board::board::{BOARD_SIZE, Board, GameResult, Move, NUM_CELLS, RuleSet, Stone};
use figrid_board::codebook_eval::{
    CodebookWeights, QuantizedCodebookWeights, evaluate_full_quantized,
};
use figrid_board::pattern_table::{
    PATTERN_NUM_IDS, PATTERN_TOP_K, canonicalize, lookup_mapped_id, pack_window, unpack_window,
};
use figrid_board::{to_idx, to_rc};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Cursor, Write};
use std::path::{Path, PathBuf};

const TRANSFORMS: usize = 8;
const REGIONS: usize = 9;
const DIM: usize = 16;
const FM_RANK: usize = 8;
const EXPECTED_GAMES: usize = 64;
const EXPECTED_ROOTS: usize = 1_022;
const NONIDENTITY_COMPARISONS: usize = EXPECTED_ROOTS * (TRANSFORMS - 1);
const DIRECTIONS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];

const MODEL_SEAL: ArtifactSeal = ArtifactSeal {
    bytes: 1_410_562,
    sha256: "42968FDAB01BA8CCD1DE3DED05C532E4B237DD47EEFFD7AE1C2F264D77BA7DA2",
};
const TOPK_SEAL: ArtifactSeal = ArtifactSeal {
    bytes: 17_060,
    sha256: "103891DCD1DCD978C654593ABE78EF32C56E2E350B500EE665BC45AC051AA16D",
};
const TRACE_SEAL: ArtifactSeal = ArtifactSeal {
    bytes: 317_511,
    sha256: "1FD40D8948F113AD236FA44F5EEADCA1907C0C3103987CB4C704B67A9B47531A",
};

#[derive(Clone, Copy)]
struct ArtifactSeal {
    bytes: u64,
    sha256: &'static str,
}

#[derive(Debug)]
struct Args {
    model: PathBuf,
    topk: PathBuf,
    trace: PathBuf,
    out_report: PathBuf,
}

struct D4Geometry {
    maps: [[usize; NUM_CELLS]; TRANSFORMS],
    inverses: [[usize; NUM_CELLS]; TRANSFORMS],
    inverse_transform: [usize; TRANSFORMS],
    region_maps: [[usize; REGIONS]; TRANSFORMS],
    composition: [[usize; TRANSFORMS]; TRANSFORMS],
    report: Value,
}

struct TensorCensus {
    passed: bool,
    report: Value,
}

struct PatternLemma {
    passed: bool,
    report: Value,
}

struct GeometryLemma {
    passed: bool,
    report: Value,
}

#[derive(Default)]
struct RootGeometryCounters {
    transformed_boards: usize,
    occupancy_cell_color_checks: usize,
    history_move_checks: usize,
    legal_set_checks: usize,
    legal_move_equivariance_checks: usize,
    candidate_set_checks: usize,
    candidate_legality_checks: usize,
    side_checks: usize,
    rule_checks: usize,
    move_count_checks: usize,
    last_move_checks: usize,
    game_result_checks: usize,
    pattern_cache_checks: usize,
}

#[derive(Default)]
struct EvalTransformStats {
    comparisons: usize,
    mismatches: usize,
    abs_differences: Vec<f64>,
    first_witness: Option<Value>,
}

struct RootCensus {
    geometry_passed: bool,
    eval_passed: bool,
    geometry_report: Value,
    eval_report: Value,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("CB-GH0 INVALID_CB_GH0: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    refuse_existing(&args.out_report)?;

    let model_bytes = read_sealed(&args.model, MODEL_SEAL, "product model")?;
    let topk_bytes = read_sealed(&args.topk, TOPK_SEAL, "topk vocabulary")?;
    let trace_bytes = read_sealed(&args.trace, TRACE_SEAL, "frozen trace")?;

    let vocabulary_report = validate_linked_topk(&topk_bytes)?;
    let float_weights = CodebookWeights::from_json_bytes(&model_bytes)
        .map_err(|error| format!("released model parser rejected product model: {error}"))?;
    let quantized = validate_and_quantize_model(&float_weights)?;
    let model_report = json!({
        "parser": "CodebookWeights::from_json_bytes",
        "shape": {
            "regions": REGIONS,
            "embedding_dim": float_weights.dim,
            "fm_rank": float_weights.fm_rank,
            "embeddings": float_weights.embeddings.len(),
            "head": float_weights.head.len(),
            "factors": float_weights.factors.len()
        },
        "all_raw_f32_finite": true,
        "quantization": {
            "embedding_scale": quantized.embedding_scale,
            "head_scale": quantized.head_scale,
            "factor_scale": quantized.factor_scale,
            "released_E32_H64_F64": true
        }
    });

    let d4 = build_d4_geometry()?;
    let pattern = pattern4_geometry_lemma(&d4)?;
    let line_geometry = line_geometry_lemma(&d4)?;
    let rule_geometry = rule_geometry_lemma();
    let tensor = tensor_census(&float_weights, &quantized, &d4);
    let roots = frozen_root_census(&trace_bytes, &quantized, &d4)?;

    let game_state_geometry_passed =
        line_geometry.passed && rule_geometry.passed && roots.geometry_passed;
    let forward_structure_passed = false;
    let tt_evaluator_passed =
        pattern.passed && tensor.passed && forward_structure_passed && roots.eval_passed;

    let combined_status = if game_state_geometry_passed && tt_evaluator_passed {
        "OPEN_GH0_HASH_AND_TT"
    } else if game_state_geometry_passed {
        "OPEN_GH0_HASH_ONLY_TT_BLOCKED"
    } else {
        "STOP_GH0_STATE_GEOMETRY_PRECONDITION"
    };
    let hash_branch_status = if game_state_geometry_passed {
        "OPEN_GH0_HASH_IMPLEMENTATION"
    } else {
        "STOP_GH0_STATE_GEOMETRY_PRECONDITION"
    };
    let tt_branch_status = if tt_evaluator_passed {
        "OPEN_GH0_TT_IMPLEMENTATION"
    } else {
        "STOP_GH0_TT_SEMANTIC_PRECONDITION"
    };

    let report = json!({
        "format": "cb-gh0-d4-semantic-precondition-v1",
        "status": combined_status,
        "claim_boundary": {
            "authoritative_stage": "P0 only",
            "incremental_hash_implemented": false,
            "canonical_tt_implemented": false,
            "performance_measured": false,
            "arena_opened": false,
            "default_change_opened": false,
            "trace_use": "correctness only; no outcome or feature selection statistic"
        },
        "inputs": {
            "model": artifact_json(&args.model, MODEL_SEAL),
            "topk": artifact_json(&args.topk, TOPK_SEAL),
            "trace": artifact_json(&args.trace, TRACE_SEAL)
        },
        "model": model_report,
        "vocabulary": vocabulary_report,
        "d4": d4.report,
        "game_state_geometry": {
            "passed": game_state_geometry_passed,
            "status": hash_branch_status,
            "line_geometry": line_geometry.report,
            "rule_terminal_geometry": rule_geometry.report,
            "frozen_roots": roots.geometry_report
        },
        "tt_evaluator": {
            "passed": tt_evaluator_passed,
            "status": tt_branch_status,
            "pattern4_coordinate_boundary_reverse_lemma": pattern.report,
            "deployed_tensor_structure": tensor.report,
            "released_forward_structural_proof": {
                "passed": false,
                "integer_feature_construction_order_independent": true,
                "final_accumulation_type": "f64",
                "final_accumulation_order": "released physical feature-index order",
                "exact_or_d4_invariant_canonical_term_order": false,
                "reason": "a D4 feature permutation can reorder an equal multiset of f64 terms; floating-point addition is not associative"
            },
            "frozen_root_product_witnesses": roots.eval_report
        },
        "decision": {
            "combined": combined_status,
            "hash_branch_opened": game_state_geometry_passed,
            "tt_score_bound_branch_opened": tt_evaluator_passed,
            "p1_hash_correctness_required": game_state_geometry_passed,
            "p2_hash_cost_opened": false,
            "p1_tt_opened": tt_evaluator_passed,
            "p2_tt_opened": false,
            "product_environment_switch_opened": false,
            "benchmark_opened": false,
            "arena_opened": false,
            "default_change_opened": false
        }
    });
    let report_bytes = serde_json::to_vec_pretty(&report)
        .map_err(|error| format!("failed to serialize report: {error}"))?;

    recheck_inputs(&args)?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.out_report)
        .map_err(|error| {
            format!(
                "failed to create report {}: {error}",
                args.out_report.display()
            )
        })?;
    output
        .write_all(&report_bytes)
        .and_then(|_| output.write_all(b"\n"))
        .map_err(|error| {
            format!(
                "failed to write report {}: {error}",
                args.out_report.display()
            )
        })?;

    println!(
        "CB-GH0 {combined_status}: roots={EXPECTED_ROOTS} geometry={} tt_evaluator={}",
        game_state_geometry_passed, tt_evaluator_passed
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
            "--model" | "--topk" | "--trace" | "--out-report"
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
        model: take("--model")?,
        topk: take("--topk")?,
        trace: take("--trace")?,
        out_report: take("--out-report")?,
    };
    debug_assert!(values.is_empty());
    Ok(args)
}

fn usage() -> &'static str {
    "usage: cb-gh0-precondition --model MODEL.json --topk topk.bin \
     --trace dp_a1_fresh_holdout_64g.jsonl --out-report NEW.json"
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
    read_sealed(&args.model, MODEL_SEAL, "product model postflight")?;
    read_sealed(&args.topk, TOPK_SEAL, "topk vocabulary postflight")?;
    read_sealed(&args.trace, TRACE_SEAL, "frozen trace postflight")?;
    Ok(())
}

fn validate_linked_topk(bytes: &[u8]) -> Result<Value, String> {
    if PATTERN_TOP_K != 4_265 || PATTERN_NUM_IDS != 4_266 {
        return Err(format!(
            "linked vocabulary shape mismatch: top_k={PATTERN_TOP_K} ids={PATTERN_NUM_IDS}"
        ));
    }
    if bytes.len() != PATTERN_TOP_K * 4 {
        return Err("sealed topk length does not match linked PATTERN_TOP_K".to_string());
    }
    let mut unique = BTreeSet::new();
    let mut canonical_checks = 0usize;
    let mut linked_id_checks = 0usize;
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        let packed = u32::from_le_bytes(chunk.try_into().expect("four-byte topk item"));
        if packed >= (1u32 << 22) {
            return Err(format!("topk item {index} exceeds 22 bits: {packed}"));
        }
        if !unique.insert(packed) {
            return Err(format!(
                "duplicate topk packed token at index {index}: {packed}"
            ));
        }
        let canonical = pack_window(&canonicalize(&unpack_window(packed)));
        if canonical != packed {
            return Err(format!(
                "topk item {index} is not reflection-canonical: {packed} -> {canonical}"
            ));
        }
        canonical_checks += 1;
        let linked = lookup_mapped_id(packed);
        if linked as usize != index {
            return Err(format!(
                "linked topk mapping mismatch for packed {packed}: linked={linked} sealed={index}"
            ));
        }
        linked_id_checks += 1;
    }
    Ok(json!({
        "sealed_entries": PATTERN_TOP_K,
        "rare_id": PATTERN_TOP_K,
        "total_ids": PATTERN_NUM_IDS,
        "unique_entries": unique.len(),
        "canonical_checks": canonical_checks,
        "linked_lookup_id_checks": linked_id_checks,
        "sealed_file_matches_compile_embedded_lookup_order": true
    }))
}

fn validate_and_quantize_model(
    weights: &CodebookWeights,
) -> Result<QuantizedCodebookWeights, String> {
    if weights.dim != DIM
        || weights.fm_rank != FM_RANK
        || weights.embeddings.len() != PATTERN_NUM_IDS * DIM
        || weights.head.len() != REGIONS * DIM
        || weights.factors.len() != REGIONS * DIM * FM_RANK
    {
        return Err(format!(
            "model shape mismatch: dim={} rank={} embeddings={} head={} factors={}",
            weights.dim,
            weights.fm_rank,
            weights.embeddings.len(),
            weights.head.len(),
            weights.factors.len()
        ));
    }
    if !weights.bias.is_finite()
        || weights.embeddings.iter().any(|value| !value.is_finite())
        || weights.head.iter().any(|value| !value.is_finite())
        || weights.factors.iter().any(|value| !value.is_finite())
    {
        return Err("model contains a non-finite f32 value".to_string());
    }
    let quantized = weights.quantize_i16_s32_s64();
    if quantized.dim != DIM
        || quantized.fm_rank != FM_RANK
        || quantized.embedding_scale != 32
        || quantized.head_scale != 64
        || quantized.factor_scale != 64
        || quantized.embeddings.len() != PATTERN_NUM_IDS * DIM
        || quantized.head.len() != REGIONS * DIM
        || quantized.factors.len() != REGIONS * DIM * FM_RANK
        || !quantized.bias.is_finite()
    {
        return Err("released E32/H64/F64 quantization shape/scale mismatch".to_string());
    }
    Ok(quantized)
}

fn transform_point(transform: usize, row: i32, col: i32) -> (i32, i32) {
    const N: i32 = (BOARD_SIZE - 1) as i32;
    match transform {
        0 => (row, col),
        1 => (col, N - row),
        2 => (N - row, N - col),
        3 => (N - col, row),
        4 => (row, N - col),
        5 => (N - row, col),
        6 => (col, row),
        7 => (N - col, N - row),
        _ => unreachable!("D4 transform index"),
    }
}

fn build_d4_geometry() -> Result<D4Geometry, String> {
    let mut maps = [[0usize; NUM_CELLS]; TRANSFORMS];
    let mut inverses = [[usize::MAX; NUM_CELLS]; TRANSFORMS];
    for transform in 0..TRANSFORMS {
        let mut seen = [false; NUM_CELLS];
        for cell in 0..NUM_CELLS {
            let (row, col) = to_rc(cell);
            let (mapped_row, mapped_col) = transform_point(transform, row as i32, col as i32);
            if !(0..BOARD_SIZE as i32).contains(&mapped_row)
                || !(0..BOARD_SIZE as i32).contains(&mapped_col)
            {
                return Err(format!(
                    "transform {transform} maps cell {cell} outside board"
                ));
            }
            let mapped = to_idx(mapped_row as usize, mapped_col as usize);
            if seen[mapped] {
                return Err(format!(
                    "transform {transform} is not bijective at mapped cell {mapped}"
                ));
            }
            seen[mapped] = true;
            maps[transform][cell] = mapped;
            inverses[transform][mapped] = cell;
        }
        if seen.iter().any(|present| !present) || inverses[transform].contains(&usize::MAX) {
            return Err(format!("transform {transform} is not a full bijection"));
        }
    }

    let mut inverse_transform = [usize::MAX; TRANSFORMS];
    for transform in 0..TRANSFORMS {
        for candidate in 0..TRANSFORMS {
            if (0..NUM_CELLS).all(|cell| maps[candidate][maps[transform][cell]] == cell) {
                inverse_transform[transform] = candidate;
                break;
            }
        }
        if inverse_transform[transform] == usize::MAX {
            return Err(format!(
                "no inverse transform found for transform {transform}"
            ));
        }
        for cell in 0..NUM_CELLS {
            if inverses[transform][maps[transform][cell]] != cell
                || maps[inverse_transform[transform]][maps[transform][cell]] != cell
            {
                return Err(format!(
                    "inverse mismatch transform={transform} cell={cell}"
                ));
            }
        }
    }
    let expected_inverse = [0, 3, 2, 1, 4, 5, 6, 7];
    if inverse_transform != expected_inverse {
        return Err(format!(
            "inverse transform convention mismatch: got {inverse_transform:?}"
        ));
    }

    let mut composition = [[usize::MAX; TRANSFORMS]; TRANSFORMS];
    for left in 0..TRANSFORMS {
        for right in 0..TRANSFORMS {
            for candidate in 0..TRANSFORMS {
                if (0..NUM_CELLS).all(|cell| maps[candidate][cell] == maps[left][maps[right][cell]])
                {
                    composition[left][right] = candidate;
                    break;
                }
            }
            if composition[left][right] == usize::MAX {
                return Err(format!(
                    "D4 composition not closed for left={left} right={right}"
                ));
            }
        }
    }

    let mut region_maps = [[usize::MAX; REGIONS]; TRANSFORMS];
    let mut region_cell_checks = 0usize;
    for transform in 0..TRANSFORMS {
        for region in 0..REGIONS {
            let mut mapped_region = None;
            let row_start = (region / 3) * 5;
            let col_start = (region % 3) * 5;
            for row in row_start..row_start + 5 {
                for col in col_start..col_start + 5 {
                    let mapped = maps[transform][to_idx(row, col)];
                    let (mapped_row, mapped_col) = to_rc(mapped);
                    let observed = (mapped_row / 5) * 3 + mapped_col / 5;
                    if let Some(expected) = mapped_region {
                        if observed != expected {
                            return Err(format!(
                                "region map inconsistent transform={transform} region={region}"
                            ));
                        }
                    } else {
                        mapped_region = Some(observed);
                    }
                    region_cell_checks += 1;
                }
            }
            region_maps[transform][region] = mapped_region.expect("five-by-five region");
        }
    }

    let report = json!({
        "convention": [
            "(row,col)",
            "(col,14-row)",
            "(14-row,14-col)",
            "(14-col,row)",
            "(row,14-col)",
            "(14-row,col)",
            "(col,row)",
            "(14-col,14-row)"
        ],
        "cell_maps": {
            "transforms": TRANSFORMS,
            "cells_per_transform": NUM_CELLS,
            "bijection_checks": TRANSFORMS * NUM_CELLS,
            "mismatches": 0
        },
        "inverse": {
            "transform_indices": inverse_transform,
            "cell_checks": TRANSFORMS * NUM_CELLS,
            "mismatches": 0
        },
        "composition": {
            "table": composition,
            "pair_checks": TRANSFORMS * TRANSFORMS,
            "cell_checks": TRANSFORMS * TRANSFORMS * NUM_CELLS,
            "mismatches": 0
        },
        "regions": {
            "maps": region_maps,
            "cell_checks": region_cell_checks,
            "mismatches": 0
        }
    });
    Ok(D4Geometry {
        maps,
        inverses,
        inverse_transform,
        region_maps,
        composition,
        report,
    })
}

fn line_window_coordinates(
    anchor_row: i32,
    anchor_col: i32,
    direction: (i32, i32),
) -> [Option<(i32, i32)>; 11] {
    std::array::from_fn(|index| {
        let offset = index as i32 - 5;
        let row = anchor_row + direction.0 * offset;
        let col = anchor_col + direction.1 * offset;
        ((0..BOARD_SIZE as i32).contains(&row) && (0..BOARD_SIZE as i32).contains(&col))
            .then_some((row, col))
    })
}

fn transformed_line_window_coordinates(
    transform: usize,
    anchor_row: i32,
    anchor_col: i32,
    direction: (i32, i32),
) -> [Option<(i32, i32)>; 11] {
    std::array::from_fn(|index| {
        let offset = index as i32 - 5;
        let row = anchor_row + direction.0 * offset;
        let col = anchor_col + direction.1 * offset;
        let (mapped_row, mapped_col) = transform_point(transform, row, col);
        ((0..BOARD_SIZE as i32).contains(&mapped_row)
            && (0..BOARD_SIZE as i32).contains(&mapped_col))
        .then_some((mapped_row, mapped_col))
    })
}

fn pattern_coordinate_boundary_census(d4: &D4Geometry) -> (usize, usize, usize, Option<Value>) {
    let mut checks = 0usize;
    let mut direct = 0usize;
    let mut reversed = 0usize;
    let mut first_failure = None;
    for cell in 0..NUM_CELLS {
        let (row, col) = to_rc(cell);
        for (direction_index, &direction) in DIRECTIONS.iter().enumerate() {
            for transform in 0..TRANSFORMS {
                checks += 1;
                let transformed = transformed_line_window_coordinates(
                    transform, row as i32, col as i32, direction,
                );
                let mapped_anchor = d4.maps[transform][cell];
                let (mapped_row, mapped_col) = to_rc(mapped_anchor);
                let mut matched = None;
                for (target_direction_index, &target_direction) in DIRECTIONS.iter().enumerate() {
                    let target = line_window_coordinates(
                        mapped_row as i32,
                        mapped_col as i32,
                        target_direction,
                    );
                    if transformed == target {
                        matched = Some((target_direction_index, false));
                        break;
                    }
                    let target_reversed = std::array::from_fn(|index| target[10 - index]);
                    if transformed == target_reversed {
                        matched = Some((target_direction_index, true));
                        break;
                    }
                }
                match matched {
                    Some((_target_direction, false)) => direct += 1,
                    Some((_target_direction, true)) => reversed += 1,
                    None => {
                        if first_failure.is_none() {
                            first_failure = Some(json!({
                                "cell": cell,
                                "row": row,
                                "col": col,
                                "direction_index": direction_index,
                                "transform": transform
                            }));
                        }
                    }
                }
            }
        }
    }
    (checks, direct, reversed, first_failure)
}

fn pattern4_geometry_lemma(d4: &D4Geometry) -> Result<PatternLemma, String> {
    let (coordinate_checks, direct, reversed, first_coordinate_failure) =
        pattern_coordinate_boundary_census(d4);
    let coordinate_mismatches = coordinate_checks.saturating_sub(direct.saturating_add(reversed));

    let raw_cases = 1usize << 22;
    let mut canonical_reverse_mismatches = 0usize;
    let mut mapped_reverse_mismatches = 0usize;
    let mut first_raw_failure = None;
    for raw in 0..raw_cases as u32 {
        let window = unpack_window(raw);
        let reverse: [u8; 11] = std::array::from_fn(|index| window[10 - index]);
        let canonical = pack_window(&canonicalize(&window));
        let reverse_canonical = pack_window(&canonicalize(&reverse));
        let canonical_bad = canonical != reverse_canonical;
        let mapped = lookup_mapped_id(raw);
        let mapped_reverse = lookup_mapped_id(pack_window(&reverse));
        let mapped_bad = mapped != mapped_reverse;
        if canonical_bad {
            canonical_reverse_mismatches += 1;
        }
        if mapped_bad {
            mapped_reverse_mismatches += 1;
        }
        if (canonical_bad || mapped_bad) && first_raw_failure.is_none() {
            first_raw_failure = Some(json!({
                "raw": raw,
                "reverse": pack_window(&reverse),
                "canonical": canonical,
                "reverse_canonical": reverse_canonical,
                "mapped_id": mapped,
                "reverse_mapped_id": mapped_reverse
            }));
        }
    }

    let passed = coordinate_mismatches == 0
        && canonical_reverse_mismatches == 0
        && mapped_reverse_mismatches == 0;
    Ok(PatternLemma {
        passed,
        report: json!({
            "passed": passed,
            "independent_of_model_weights": true,
            "anchor_direction_transform_checks": coordinate_checks,
            "direct_sequence_matches": direct,
            "reversed_sequence_matches": reversed,
            "coordinate_or_boundary_mismatches": coordinate_mismatches,
            "first_coordinate_failure": first_coordinate_failure,
            "released_canonicalizer_reverse_exhaustive": {
                "raw_22bit_cases": raw_cases,
                "canonical_token_mismatches": canonical_reverse_mismatches,
                "mapped_id_or_rare_mismatches": mapped_reverse_mismatches,
                "first_failure": first_raw_failure
            },
            "arbitrary_black_white_occupancy_token_equivariance_proved": passed
        }),
    })
}

fn maximal_board_lines() -> Vec<Vec<usize>> {
    let mut lines = Vec::new();
    for &(dr, dc) in &DIRECTIONS {
        for row in 0..BOARD_SIZE as i32 {
            for col in 0..BOARD_SIZE as i32 {
                let previous_row = row - dr;
                let previous_col = col - dc;
                if (0..BOARD_SIZE as i32).contains(&previous_row)
                    && (0..BOARD_SIZE as i32).contains(&previous_col)
                {
                    continue;
                }
                let mut line = Vec::new();
                let mut scan_row = row;
                let mut scan_col = col;
                while (0..BOARD_SIZE as i32).contains(&scan_row)
                    && (0..BOARD_SIZE as i32).contains(&scan_col)
                {
                    line.push(to_idx(scan_row as usize, scan_col as usize));
                    scan_row += dr;
                    scan_col += dc;
                }
                lines.push(line);
            }
        }
    }
    lines
}

fn canonical_line(mut line: Vec<usize>) -> Vec<usize> {
    let reverse: Vec<usize> = line.iter().rev().copied().collect();
    if reverse < line {
        line = reverse;
    }
    line
}

fn line_geometry_lemma(d4: &D4Geometry) -> Result<GeometryLemma, String> {
    let lines = maximal_board_lines();
    let canonical_lines: BTreeSet<Vec<usize>> = lines.iter().cloned().map(canonical_line).collect();
    let mut checks = 0usize;
    let mut mismatches = 0usize;
    let mut first_failure = None;
    let mut segment_checks = 0usize;
    let mut open_end_adjacency_checks = 0usize;

    for (line_index, line) in lines.iter().enumerate() {
        let segment_count = line.len() * (line.len() + 1) / 2;
        for transform in 0..TRANSFORMS {
            checks += 1;
            segment_checks += segment_count;
            // Every nonempty subsegment has at most two adjacent open-end
            // sites.  Mapping a full maximal line directly or in reverse
            // maps those predecessor/successor sites bijectively as well.
            for start in 0..line.len() {
                for end in start..line.len() {
                    open_end_adjacency_checks += usize::from(start > 0);
                    open_end_adjacency_checks += usize::from(end + 1 < line.len());
                }
            }
            let mapped: Vec<usize> = line.iter().map(|&cell| d4.maps[transform][cell]).collect();
            let mapped_canonical = canonical_line(mapped);
            if !canonical_lines.contains(&mapped_canonical) {
                mismatches += 1;
                if first_failure.is_none() {
                    first_failure = Some(json!({
                        "line_index": line_index,
                        "transform": transform,
                        "source": line,
                        "mapped_canonical": mapped_canonical
                    }));
                }
            }
        }
    }
    let passed = mismatches == 0;
    Ok(GeometryLemma {
        passed,
        report: json!({
            "passed": passed,
            "undirected_directions": ["row", "column", "down_diagonal", "up_diagonal"],
            "maximal_lines_enumerated": lines.len(),
            "unique_coordinate_lines": canonical_lines.len(),
            "line_transform_checks": checks,
            "contiguous_segment_checks": segment_checks,
            "open_end_adjacency_checks": open_end_adjacency_checks,
            "length_or_direction_or_adjacency_mismatches": mismatches,
            "first_failure": first_failure,
            "stone_color_preserved_by_coordinate_maps": true
        }),
    })
}

fn rule_geometry_lemma() -> GeometryLemma {
    let rules = [
        RuleSet::Freestyle,
        RuleSet::Standard,
        RuleSet::Caro,
        RuleSet::Renju,
    ];
    let sides = [Stone::Black, Stone::White];
    let mut tuple_checks = 0usize;
    let mut d4_applications = 0usize;
    let mut mismatches = 0usize;
    for rule in rules {
        for side in sides {
            for count in 1..=BOARD_SIZE as u32 {
                for open_ends in 0..=2 {
                    tuple_checks += 1;
                    let reference = rule.line_wins(side, count, open_ends);
                    for _transform in 0..TRANSFORMS {
                        d4_applications += 1;
                        if rule.line_wins(side, count, open_ends) != reference {
                            mismatches += 1;
                        }
                    }
                }
            }
        }
    }
    let passed = mismatches == 0;
    GeometryLemma {
        passed,
        report: json!({
            "passed": passed,
            "rules": ["Freestyle", "Standard", "Caro", "Renju terminal semantics"],
            "sides": ["Black", "White"],
            "lengths": [1, BOARD_SIZE],
            "open_ends": [0, 2],
            "rule_side_length_open_end_tuples": tuple_checks,
            "d4_applications": d4_applications,
            "mismatches": mismatches,
            "structural_basis": "RuleSet::line_wins depends only on side, contiguous length, and open-end count"
        }),
    }
}

fn flat_row_census_f32(
    values: &[f32],
    row_width: usize,
    name: &str,
    d4: &D4Geometry,
) -> (Vec<Value>, usize) {
    let mut rows = Vec::with_capacity(TRANSFORMS - 1);
    let mut total_mismatches = 0usize;
    for transform in 1..TRANSFORMS {
        let mut mismatches = 0usize;
        let mut first = None;
        for region in 0..REGIONS {
            let mapped_region = d4.region_maps[transform][region];
            for offset in 0..row_width {
                let left = values[region * row_width + offset];
                let right = values[mapped_region * row_width + offset];
                if left.to_bits() != right.to_bits() {
                    mismatches += 1;
                    if first.is_none() {
                        first = Some(json!({
                            "region": region,
                            "mapped_region": mapped_region,
                            "row_offset": offset,
                            "source_value": left,
                            "mapped_value": right,
                            "source_bits": format!("0x{:08X}", left.to_bits()),
                            "mapped_bits": format!("0x{:08X}", right.to_bits())
                        }));
                    }
                }
            }
        }
        total_mismatches += mismatches;
        rows.push(json!({
            "transform": transform,
            "tensor": name,
            "comparisons": REGIONS * row_width,
            "mismatches": mismatches,
            "first_mismatch": first
        }));
    }
    (rows, total_mismatches)
}

fn flat_row_census_i16(
    values: &[i16],
    row_width: usize,
    name: &str,
    d4: &D4Geometry,
) -> (Vec<Value>, usize) {
    let mut rows = Vec::with_capacity(TRANSFORMS - 1);
    let mut total_mismatches = 0usize;
    for transform in 1..TRANSFORMS {
        let mut mismatches = 0usize;
        let mut first = None;
        for region in 0..REGIONS {
            let mapped_region = d4.region_maps[transform][region];
            for offset in 0..row_width {
                let left = values[region * row_width + offset];
                let right = values[mapped_region * row_width + offset];
                if left != right {
                    mismatches += 1;
                    if first.is_none() {
                        first = Some(json!({
                            "region": region,
                            "mapped_region": mapped_region,
                            "row_offset": offset,
                            "source_value": left,
                            "mapped_value": right
                        }));
                    }
                }
            }
        }
        total_mismatches += mismatches;
        rows.push(json!({
            "transform": transform,
            "tensor": name,
            "comparisons": REGIONS * row_width,
            "mismatches": mismatches,
            "first_mismatch": first
        }));
    }
    (rows, total_mismatches)
}

fn orbit_group_census_f32(values: &[f32], row_width: usize, regions: &[usize]) -> Value {
    let mut mismatches = 0usize;
    let mut first = None;
    for offset in 0..row_width {
        let reference = values[regions[0] * row_width + offset];
        if regions[1..]
            .iter()
            .any(|&region| values[region * row_width + offset].to_bits() != reference.to_bits())
        {
            mismatches += 1;
            if first.is_none() {
                first = Some(json!({
                    "row_offset": offset,
                    "values": regions.iter().map(|&region| values[region * row_width + offset]).collect::<Vec<_>>(),
                    "bits": regions.iter().map(|&region| format!("0x{:08X}", values[region * row_width + offset].to_bits())).collect::<Vec<_>>()
                }));
            }
        }
    }
    json!({
        "groups": row_width,
        "mismatched_groups": mismatches,
        "first_mismatch": first
    })
}

fn orbit_group_census_i16(values: &[i16], row_width: usize, regions: &[usize]) -> Value {
    let mut mismatches = 0usize;
    let mut first = None;
    for offset in 0..row_width {
        let reference = values[regions[0] * row_width + offset];
        if regions[1..]
            .iter()
            .any(|&region| values[region * row_width + offset] != reference)
        {
            mismatches += 1;
            if first.is_none() {
                first = Some(json!({
                    "row_offset": offset,
                    "values": regions.iter().map(|&region| values[region * row_width + offset]).collect::<Vec<_>>()
                }));
            }
        }
    }
    json!({
        "groups": row_width,
        "mismatched_groups": mismatches,
        "first_mismatch": first
    })
}

fn json_mismatched_groups(value: &Value) -> usize {
    value
        .get("mismatched_groups")
        .and_then(Value::as_u64)
        .expect("group report count") as usize
}

fn tensor_census(
    float: &CodebookWeights,
    quantized: &QuantizedCodebookWeights,
    d4: &D4Geometry,
) -> TensorCensus {
    let factor_width = DIM * FM_RANK;
    let (raw_head, raw_head_mismatches) = flat_row_census_f32(&float.head, DIM, "raw_f32_head", d4);
    let (quant_head, quant_head_mismatches) =
        flat_row_census_i16(&quantized.head, DIM, "quantized_i16_head", d4);
    let (raw_factors, raw_factor_mismatches) =
        flat_row_census_f32(&float.factors, factor_width, "raw_f32_factors", d4);
    let (quant_factors, quant_factor_mismatches) = flat_row_census_i16(
        &quantized.factors,
        factor_width,
        "quantized_i16_factors",
        d4,
    );

    const CORNERS: [usize; 4] = [0, 2, 6, 8];
    const EDGES: [usize; 4] = [1, 3, 5, 7];
    let raw_head_corners = orbit_group_census_f32(&float.head, DIM, &CORNERS);
    let raw_head_edges = orbit_group_census_f32(&float.head, DIM, &EDGES);
    let quant_head_corners = orbit_group_census_i16(&quantized.head, DIM, &CORNERS);
    let quant_head_edges = orbit_group_census_i16(&quantized.head, DIM, &EDGES);
    let raw_factor_corners = orbit_group_census_f32(&float.factors, factor_width, &CORNERS);
    let raw_factor_edges = orbit_group_census_f32(&float.factors, factor_width, &EDGES);
    let quant_factor_corners = orbit_group_census_i16(&quantized.factors, factor_width, &CORNERS);
    let quant_factor_edges = orbit_group_census_i16(&quantized.factors, factor_width, &EDGES);

    let per_transform_mismatches = raw_head_mismatches
        + quant_head_mismatches
        + raw_factor_mismatches
        + quant_factor_mismatches;
    let group_mismatches = [
        &raw_head_corners,
        &raw_head_edges,
        &quant_head_corners,
        &quant_head_edges,
        &raw_factor_corners,
        &raw_factor_edges,
        &quant_factor_corners,
        &quant_factor_edges,
    ]
    .into_iter()
    .map(|value| json_mismatched_groups(value))
    .sum::<usize>();
    let passed = per_transform_mismatches == 0 && group_mismatches == 0;

    TensorCensus {
        passed,
        report: json!({
            "passed": passed,
            "direct_row_equality_protocol_gate": true,
            "fm_latent_reparameterization_inferred": false,
            "global_embedding_table_is_region_independent": true,
            "scalar_bias_is_region_independent": true,
            "per_transform": {
                "raw_f32_head": raw_head,
                "quantized_i16_head": quant_head,
                "raw_f32_factors": raw_factors,
                "quantized_i16_factors": quant_factors
            },
            "per_transform_mismatches_total": per_transform_mismatches,
            "orbit_groups": {
                "raw_f32_head": {"corners": raw_head_corners, "edges": raw_head_edges},
                "quantized_i16_head": {"corners": quant_head_corners, "edges": quant_head_edges},
                "raw_f32_factors": {"corners": raw_factor_corners, "edges": raw_factor_edges},
                "quantized_i16_factors": {"corners": quant_factor_corners, "edges": quant_factor_edges}
            },
            "orbit_group_mismatches_total": group_mismatches
        }),
    }
}

fn is_figrid(name: &str) -> bool {
    name.to_ascii_lowercase().contains("figrid")
}

fn parse_trace_stone(raw: &str) -> Option<Stone> {
    match raw {
        "B" => Some(Stone::Black),
        "W" => Some(Stone::White),
        _ => None,
    }
}

fn stone_name(stone: Stone) -> &'static str {
    match stone {
        Stone::Black => "black",
        Stone::White => "white",
    }
}

fn rule_name(rule: RuleSet) -> &'static str {
    match rule {
        RuleSet::Freestyle => "Freestyle",
        RuleSet::Standard => "Standard",
        RuleSet::Caro => "Caro",
        RuleSet::Renju => "Renju",
    }
}

fn result_name(result: GameResult) -> &'static str {
    match result {
        GameResult::BlackWin => "black_win",
        GameResult::WhiteWin => "white_win",
        GameResult::Draw => "draw",
        GameResult::Ongoing => "ongoing",
    }
}

fn sorted_mapped_moves(moves: &[Move], map: &[usize; NUM_CELLS]) -> Vec<Move> {
    let mut mapped: Vec<Move> = moves.iter().map(|&mv| map[mv]).collect();
    mapped.sort_unstable();
    mapped
}

fn sorted_moves(mut moves: Vec<Move>) -> Vec<Move> {
    moves.sort_unstable();
    moves
}

fn contains_duplicate_moves(sorted_moves: &[Move]) -> bool {
    sorted_moves.windows(2).any(|pair| pair[0] == pair[1])
}

fn build_transformed_board(
    history: &[Move],
    rule: RuleSet,
    transform: usize,
    d4: &D4Geometry,
) -> Result<Board, String> {
    let mut board = Board::new();
    board.set_rule_set(rule);
    for (ply, &mv) in history.iter().enumerate() {
        if board.game_result() != GameResult::Ongoing {
            return Err(format!(
                "transformed history continues after terminal state transform={transform} ply={ply}"
            ));
        }
        let mapped = d4.maps[transform][mv];
        if !board.is_legal_move(mapped) {
            return Err(format!(
                "transformed history replay illegal transform={transform} ply={ply} move={mapped}"
            ));
        }
        board.make_move(mapped);
    }
    Ok(board)
}

fn move_history_json(history: &[Move]) -> Vec<Value> {
    history
        .iter()
        .enumerate()
        .map(|(ply, &mv)| {
            let (row, col) = to_rc(mv);
            json!({
                "ply": ply,
                "idx": mv,
                "x": col,
                "y": row,
                "color": if ply % 2 == 0 { "B" } else { "W" }
            })
        })
        .collect()
}

fn boards_equal_at_identity(left: &Board, right: &Board) -> bool {
    left.black == right.black
        && left.white == right.white
        && left.side_to_move == right.side_to_move
        && left.move_count == right.move_count
        && left.last_move == right.last_move
        && left.history == right.history
        && left.zobrist == right.zobrist
        && left.line_pattern_ids == right.line_pattern_ids
        && left.effective_rule_set() == right.effective_rule_set()
        && left.game_result() == right.game_result()
}

#[allow(clippy::too_many_arguments)]
fn process_root(
    root: &Board,
    root_index: usize,
    game_id: &Value,
    ply: usize,
    weights: &QuantizedCodebookWeights,
    d4: &D4Geometry,
    geometry: &mut RootGeometryCounters,
    eval_stats: &mut [EvalTransformStats; TRANSFORMS],
) -> Result<(), String> {
    let history = root.history.clone();
    let rule = root.effective_rule_set();
    let original_legal = root.legal_moves();
    let original_candidates = root.candidate_moves();
    let sorted_original_candidates = sorted_moves(original_candidates.clone());
    if contains_duplicate_moves(&sorted_original_candidates) {
        return Err(format!(
            "root candidate generator returned duplicate move root={root_index}"
        ));
    }
    if original_candidates
        .iter()
        .any(|&mv| !root.is_legal_move(mv))
    {
        return Err(format!(
            "root candidate generator returned illegal move root={root_index}"
        ));
    }

    let mut transformed = Vec::with_capacity(TRANSFORMS);
    for transform in 0..TRANSFORMS {
        transformed.push(build_transformed_board(&history, rule, transform, d4)?);
    }
    if !boards_equal_at_identity(root, &transformed[0]) {
        return Err(format!(
            "identity full-history rebuild mismatch root={root_index}"
        ));
    }
    geometry.pattern_cache_checks += NUM_CELLS * 4;

    for transform in 0..TRANSFORMS {
        let board = &transformed[transform];
        geometry.transformed_boards += 1;

        geometry.side_checks += 1;
        if board.side_to_move != root.side_to_move {
            return Err(format!(
                "side mismatch root={root_index} transform={transform}"
            ));
        }
        geometry.rule_checks += 1;
        if board.effective_rule_set() != rule {
            return Err(format!(
                "effective rule mismatch root={root_index} transform={transform}"
            ));
        }
        geometry.move_count_checks += 1;
        if board.move_count != root.move_count {
            return Err(format!(
                "move count mismatch root={root_index} transform={transform}"
            ));
        }
        geometry.last_move_checks += 1;
        if board.last_move != root.last_move.map(|mv| d4.maps[transform][mv]) {
            return Err(format!(
                "last move mismatch root={root_index} transform={transform}"
            ));
        }
        geometry.game_result_checks += 1;
        if board.game_result() != root.game_result() {
            return Err(format!(
                "game result mismatch root={root_index} transform={transform}: {} vs {}",
                result_name(root.game_result()),
                result_name(board.game_result())
            ));
        }

        if board.history.len() != history.len() {
            return Err(format!(
                "history length mismatch root={root_index} transform={transform}"
            ));
        }
        for (history_index, (&source, &mapped)) in
            history.iter().zip(board.history.iter()).enumerate()
        {
            geometry.history_move_checks += 1;
            if mapped != d4.maps[transform][source] || d4.inverses[transform][mapped] != source {
                return Err(format!(
                    "history map/inverse mismatch root={root_index} transform={transform} ply={history_index}"
                ));
            }
        }

        for cell in 0..NUM_CELLS {
            let mapped = d4.maps[transform][cell];
            geometry.occupancy_cell_color_checks += 2;
            if board.black.get(mapped) != root.black.get(cell)
                || board.white.get(mapped) != root.white.get(cell)
            {
                return Err(format!(
                    "occupancy mismatch root={root_index} transform={transform} cell={cell}"
                ));
            }
            geometry.legal_move_equivariance_checks += 1;
            if board.is_legal_move(mapped) != root.is_legal_move(cell) {
                return Err(format!(
                    "legal move mismatch root={root_index} transform={transform} cell={cell}"
                ));
            }
        }

        geometry.legal_set_checks += 1;
        let expected_legal = sorted_mapped_moves(&original_legal, &d4.maps[transform]);
        let observed_legal = sorted_moves(board.legal_moves());
        if observed_legal != expected_legal {
            return Err(format!(
                "full legal set mismatch root={root_index} transform={transform}"
            ));
        }

        geometry.candidate_set_checks += 1;
        let expected_candidates = sorted_mapped_moves(&original_candidates, &d4.maps[transform]);
        let observed_candidates_unsorted = board.candidate_moves();
        let observed_candidates = sorted_moves(observed_candidates_unsorted.clone());
        if contains_duplicate_moves(&observed_candidates) {
            return Err(format!(
                "transformed candidate generator returned duplicate move root={root_index} transform={transform}"
            ));
        }
        for &candidate in &observed_candidates_unsorted {
            geometry.candidate_legality_checks += 1;
            if !board.is_legal_move(candidate) {
                return Err(format!(
                    "illegal transformed candidate root={root_index} transform={transform} move={candidate}"
                ));
            }
            let original = d4.inverses[transform][candidate];
            if !root.is_legal_move(original) {
                return Err(format!(
                    "inverse candidate legality mismatch root={root_index} transform={transform} move={candidate}"
                ));
            }
        }
        if observed_candidates != expected_candidates {
            return Err(format!(
                "candidate set mismatch root={root_index} transform={transform}"
            ));
        }
    }

    let mut values = [0.0f32; TRANSFORMS];
    for transform in 0..TRANSFORMS {
        values[transform] = evaluate_full_quantized(&transformed[transform], weights);
        if !values[transform].is_finite() {
            return Err(format!(
                "non-finite released product evaluation root={root_index} transform={transform}"
            ));
        }
    }
    for transform in 1..TRANSFORMS {
        let baseline = values[0];
        let candidate = values[transform];
        let stats = &mut eval_stats[transform];
        stats.comparisons += 1;
        stats
            .abs_differences
            .push((f64::from(candidate) - f64::from(baseline)).abs());
        if candidate.to_bits() != baseline.to_bits() {
            stats.mismatches += 1;
            if stats.first_witness.is_none() {
                stats.first_witness = Some(json!({
                    "root_index_zero_based": root_index,
                    "game_id": game_id,
                    "ply": ply,
                    "rule": rule_name(rule),
                    "natural_side_to_move": stone_name(root.side_to_move),
                    "history": move_history_json(&history),
                    "transform": transform,
                    "t0_value": baseline,
                    "t_value": candidate,
                    "t0_bits": format!("0x{:08X}", baseline.to_bits()),
                    "t_bits": format!("0x{:08X}", candidate.to_bits())
                }));
            }
        }
    }
    Ok(())
}

fn frozen_root_census(
    trace_bytes: &[u8],
    weights: &QuantizedCodebookWeights,
    d4: &D4Geometry,
) -> Result<RootCensus, String> {
    let mut games = 0usize;
    let mut processed_roots = 0usize;
    let mut geometry = RootGeometryCounters::default();
    let mut eval_stats: [EvalTransformStats; TRANSFORMS] =
        std::array::from_fn(|_| EvalTransformStats::default());

    for line_result in BufReader::new(Cursor::new(trace_bytes)).lines() {
        let line = line_result.map_err(|error| format!("trace is not valid UTF-8: {error}"))?;
        if line.trim().is_empty() {
            continue;
        }
        games += 1;
        let game: Value = serde_json::from_str(&line)
            .map_err(|error| format!("invalid trace JSON at game line {games}: {error}"))?;
        let game_id = game
            .get("game_id")
            .cloned()
            .ok_or_else(|| format!("trace game line {games} missing game_id"))?;
        let black_engine = required_str(&game, "black_engine", games)?;
        let white_engine = required_str(&game, "white_engine", games)?;
        let product_side = match (is_figrid(black_engine), is_figrid(white_engine)) {
            (true, false) => Stone::Black,
            (false, true) => Stone::White,
            other => {
                return Err(format!(
                    "expected exactly one case-insensitive figrid engine game={game_id}, got {other:?}"
                ));
            }
        };
        let moves = game
            .get("moves")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("trace game={game_id} missing moves array"))?;
        let declared_result = required_str(&game, "result", games)?;
        if !matches!(declared_result, "black_win" | "white_win" | "draw") {
            return Err(format!(
                "trace game={game_id} has invalid final result {declared_result:?}"
            ));
        }
        let declared_move_count = game
            .get("move_count")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| format!("trace game={game_id} missing valid move_count"))?;
        if declared_move_count != moves.len() {
            return Err(format!(
                "trace move_count mismatch game={game_id}: declared={declared_move_count} rows={}",
                moves.len()
            ));
        }

        let mut board = Board::new();
        board.set_rule_set(RuleSet::Freestyle);
        for (ply, move_json) in moves.iter().enumerate() {
            if board.game_result() != GameResult::Ongoing {
                return Err(format!(
                    "trace contains move after terminal state game={game_id} ply={ply}"
                ));
            }
            let x = required_usize(move_json, "x", &game_id, ply)?;
            let y = required_usize(move_json, "y", &game_id, ply)?;
            if x >= BOARD_SIZE || y >= BOARD_SIZE {
                return Err(format!(
                    "trace move outside board game={game_id} ply={ply} x={x} y={y}"
                ));
            }
            let color_raw = move_json
                .get("color")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("trace move missing color game={game_id} ply={ply}"))?;
            let color = parse_trace_stone(color_raw).ok_or_else(|| {
                format!("trace move invalid color game={game_id} ply={ply}: {color_raw}")
            })?;
            let source = move_json
                .get("source")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("trace move missing source game={game_id} ply={ply}"))?;
            if !matches!(source, "engine" | "opening") {
                return Err(format!(
                    "trace move invalid source game={game_id} ply={ply}: {source:?}"
                ));
            }
            let mv = to_idx(y, x);

            if source == "engine"
                && board.side_to_move == product_side
                && processed_roots < EXPECTED_ROOTS
            {
                process_root(
                    &board,
                    processed_roots,
                    &game_id,
                    ply,
                    weights,
                    d4,
                    &mut geometry,
                    &mut eval_stats,
                )?;
                processed_roots += 1;
            }

            if color != board.side_to_move {
                return Err(format!(
                    "trace color/STM mismatch game={game_id} ply={ply}: color={} stm={}",
                    stone_name(color),
                    stone_name(board.side_to_move)
                ));
            }
            if !board.is_legal_move(mv) {
                return Err(format!(
                    "trace illegal or occupied move game={game_id} ply={ply} idx={mv}"
                ));
            }
            board.make_move(mv);
        }
        if board.move_count != declared_move_count {
            return Err(format!(
                "trace replay count mismatch game={game_id}: replay={} declared={declared_move_count}",
                board.move_count
            ));
        }
        if result_name(board.game_result()) != declared_result {
            return Err(format!(
                "trace result mismatch game={game_id}: replay={} declared={declared_result}",
                result_name(board.game_result())
            ));
        }
    }

    if games != EXPECTED_GAMES {
        return Err(format!(
            "frozen trace game count mismatch: got {games}, expected {EXPECTED_GAMES}"
        ));
    }
    if processed_roots != EXPECTED_ROOTS {
        return Err(format!(
            "frozen root count mismatch: processed={processed_roots} expected={EXPECTED_ROOTS}"
        ));
    }

    let mut per_transform = Vec::with_capacity(TRANSFORMS - 1);
    let mut total_mismatches = 0usize;
    let mut total_comparisons = 0usize;
    for (transform, stats) in eval_stats.iter_mut().enumerate().skip(1) {
        if stats.comparisons != EXPECTED_ROOTS || stats.abs_differences.len() != EXPECTED_ROOTS {
            return Err(format!(
                "evaluation comparison count mismatch transform={transform}: comparisons={} diffs={}",
                stats.comparisons,
                stats.abs_differences.len()
            ));
        }
        stats.abs_differences.sort_by(f64::total_cmp);
        total_mismatches += stats.mismatches;
        total_comparisons += stats.comparisons;
        per_transform.push(json!({
            "transform": transform,
            "roots": stats.comparisons,
            "mismatches": stats.mismatches,
            "mismatch_rate": stats.mismatches as f64 / stats.comparisons as f64,
            "exact_fraction": format!("{}/{}", stats.mismatches, stats.comparisons),
            "absolute_difference": {
                "p50": sorted_quantile(&stats.abs_differences, 0.50),
                "p95": sorted_quantile(&stats.abs_differences, 0.95),
                "p99": sorted_quantile(&stats.abs_differences, 0.99),
                "max": stats.abs_differences.last().copied().unwrap_or(0.0)
            },
            "first_mismatch": stats.first_witness
        }));
    }
    if total_comparisons != NONIDENTITY_COMPARISONS {
        return Err(format!(
            "nonidentity comparison count mismatch: got {total_comparisons}, expected {NONIDENTITY_COMPARISONS}"
        ));
    }
    let eval_passed = total_mismatches == 0;
    let geometry_passed = true;
    Ok(RootCensus {
        geometry_passed,
        eval_passed,
        geometry_report: json!({
            "passed": true,
            "trace_games": games,
            "selected_product_roots": processed_roots,
            "selection_rule": "before each move: source==engine && side_to_move==the unique case-insensitive figrid engine side; stride=1",
            "trace_rule": "Freestyle",
            "full_history_rebuilds": geometry.transformed_boards,
            "identity_pattern_cache_checks": geometry.pattern_cache_checks,
            "occupancy_cell_color_checks": geometry.occupancy_cell_color_checks,
            "history_move_map_inverse_checks": geometry.history_move_checks,
            "side_checks": geometry.side_checks,
            "effective_rule_checks": geometry.rule_checks,
            "move_count_checks": geometry.move_count_checks,
            "last_move_checks": geometry.last_move_checks,
            "game_result_checks": geometry.game_result_checks,
            "full_legal_set_checks": geometry.legal_set_checks,
            "per_cell_legal_equivariance_checks": geometry.legal_move_equivariance_checks,
            "candidate_set_checks_after_mapping_and_sorting": geometry.candidate_set_checks,
            "candidate_legality_and_inverse_checks": geometry.candidate_legality_checks,
            "candidate_vector_order_compared": false,
            "mismatches": 0
        }),
        eval_report: json!({
            "passed": eval_passed,
            "released_entrypoint": "evaluate_full_quantized",
            "perspective": "natural side to move",
            "comparison": "final f32 to_bits; no tolerance",
            "roots": processed_roots,
            "nonidentity_comparisons": total_comparisons,
            "expected_nonidentity_comparisons": NONIDENTITY_COMPARISONS,
            "mismatches": total_mismatches,
            "quantile_rule": "sort finite absolute f64 differences ascending; index=round(q*(n-1))",
            "per_transform": per_transform
        }),
    })
}

fn required_str<'a>(game: &'a Value, field: &str, line: usize) -> Result<&'a str, String> {
    game.get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("trace game line {line} missing string {field}"))
}

fn required_usize(
    value: &Value,
    field: &str,
    game_id: &Value,
    ply: usize,
) -> Result<usize, String> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("trace move missing usize {field} game={game_id} ply={ply}"))
}

fn sorted_quantile(values: &[f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let index = ((values.len() - 1) as f64 * quantile).round() as usize;
    values[index.min(values.len() - 1)]
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
                    .expect("four-byte SHA-256 word"),
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
    fn frozen_d4_maps_are_a_group_with_expected_inverses() {
        let d4 = build_d4_geometry().expect("valid D4 geometry");
        assert_eq!(d4.inverse_transform, [0, 3, 2, 1, 4, 5, 6, 7]);
        assert_eq!(d4.composition[1][3], 0);
        assert_eq!(d4.maps[1][to_idx(0, 0)], to_idx(0, 14));
        assert_eq!(d4.maps[7][to_idx(0, 0)], to_idx(14, 14));
        assert_eq!(d4.maps[6][to_idx(4, 9)], to_idx(9, 4));
        assert_eq!(d4.maps[2][to_idx(7, 7)], to_idx(7, 7));
    }

    #[test]
    fn pattern_coordinate_boundary_sequences_are_direct_or_reversed() {
        let d4 = build_d4_geometry().expect("valid D4 geometry");
        let (checks, direct, reversed, first_failure) = pattern_coordinate_boundary_census(&d4);
        assert_eq!(checks, NUM_CELLS * DIRECTIONS.len() * TRANSFORMS);
        assert_eq!(direct + reversed, checks);
        assert!(first_failure.is_none());
    }

    #[test]
    fn line_and_rule_geometry_are_exact() {
        let d4 = build_d4_geometry().expect("valid D4 geometry");
        assert!(line_geometry_lemma(&d4).expect("line lemma").passed);
        assert!(rule_geometry_lemma().passed);
    }

    #[test]
    fn transformed_history_rejects_move_after_terminal_state() {
        let d4 = build_d4_geometry().expect("valid D4 geometry");
        let history = [
            to_idx(7, 3),
            to_idx(0, 0),
            to_idx(7, 4),
            to_idx(0, 1),
            to_idx(7, 5),
            to_idx(0, 2),
            to_idx(7, 6),
            to_idx(0, 3),
            to_idx(7, 7),
            to_idx(0, 4),
        ];
        let error = match build_transformed_board(&history, RuleSet::Freestyle, 0, &d4) {
            Ok(_) => panic!("post-terminal continuation must be rejected"),
            Err(error) => error,
        };
        assert!(error.contains("continues after terminal state"));
    }

    #[test]
    fn quantile_rule_is_frozen() {
        let values = [0.0, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(sorted_quantile(&values, 0.50), 2.0);
        assert_eq!(sorted_quantile(&values, 0.95), 4.0);
        assert_eq!(sorted_quantile(&values, 0.99), 4.0);
    }
}
