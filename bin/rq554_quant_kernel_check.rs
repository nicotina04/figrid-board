#![cfg(feature = "codebook-eval")]

use figrid_board::codebook_eval::{
    CodebookWeights, IncrementalQuantizedCodebookEval, evaluate_full, evaluate_full_quantized,
};
use figrid_board::{BOARD_SIZE, Board, Move};
use serde_json::{Value, json};
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

#[derive(Debug)]
struct Args {
    model: PathBuf,
    games_jsonl: PathBuf,
    out_json: PathBuf,
    max_transitions: usize,
    runtime_scale: f64,
}

#[derive(Default)]
struct Stats {
    games: usize,
    transitions: usize,
    incremental_mismatches: usize,
    undo_mismatches: usize,
    max_full_diff_cp: f64,
    max_dequant_diff_cp: f64,
    full_diffs_cp: Vec<f64>,
    dequant_diffs_cp: Vec<f64>,
    dirty_counts: Vec<usize>,
}

fn main() -> Result<(), String> {
    let args = parse_args()?;
    let bytes = std::fs::read(&args.model)
        .map_err(|e| format!("failed to read {}: {e}", args.model.display()))?;
    let fp32 = CodebookWeights::from_json_bytes(&bytes)?;
    let quant = fp32.quantize_i16_s32_s64();
    let dequant = quant.dequantized();
    let games = load_trace_games(&args.games_jsonl)?;
    if games.is_empty() {
        return Err(format!(
            "no games loaded from {}",
            args.games_jsonl.display()
        ));
    }

    let mut stats = Stats::default();
    'passes: while stats.transitions < args.max_transitions {
        for game in &games {
            if stats.transitions >= args.max_transitions {
                break 'passes;
            }
            stats.games += 1;
            let mut board = Board::new();
            let mut inc = IncrementalQuantizedCodebookEval::new(&quant);
            inc.refresh(&board, &quant);
            let mut played = 0usize;
            for &mv in game {
                if stats.transitions >= args.max_transitions {
                    break;
                }
                if !board.is_empty(mv) {
                    break;
                }
                board.make_move(mv);
                inc.push_move(&board, mv, &quant);
                stats.dirty_counts.push(inc.last_dirty_cells());
                stats.transitions += 1;
                played += 1;
                record_position(&mut stats, &args, &board, &mut inc, &quant, &dequant);
            }
            for _ in 0..played {
                board.undo_move();
                inc.pop_move(&quant);
                let inc_v = inc.value(&board, &quant);
                let full_v = evaluate_full_quantized(&board, &quant);
                if !close(inc_v, full_v) {
                    stats.undo_mismatches += 1;
                }
            }
        }
    }

    stats.full_diffs_cp.sort_by(|a, b| a.total_cmp(b));
    stats.dequant_diffs_cp.sort_by(|a, b| a.total_cmp(b));
    stats.dirty_counts.sort_unstable();

    let report = json!({
        "format": "rq554-quant-kernel-check-v1",
        "model": args.model,
        "games_jsonl": args.games_jsonl,
        "max_transitions": args.max_transitions,
        "runtime_scale": args.runtime_scale,
        "quant": {
            "embeddings": "i16_s32",
            "head": "i16_s64",
            "factors": "i16_s64",
            "bias": "f32"
        },
        "stats": {
            "games_passed": stats.games,
            "transitions": stats.transitions,
            "incremental_mismatches": stats.incremental_mismatches,
            "undo_mismatches": stats.undo_mismatches,
            "quant_full_diff_cp": describe(&stats.full_diffs_cp),
            "fake_dequant_diff_cp": describe(&stats.dequant_diffs_cp),
            "dirty_cells": describe_usize(&stats.dirty_counts),
        }
    });

    let mut out = File::create(&args.out_json)
        .map_err(|e| format!("failed to create {}: {e}", args.out_json.display()))?;
    writeln!(out, "{}", serde_json::to_string_pretty(&report).unwrap())
        .map_err(|e| format!("failed to write {}: {e}", args.out_json.display()))?;

    println!(
        "rq554-quant-kernel-check: transitions={} inc_mismatch={} undo_mismatch={} dequant_p99={:.6}cp dequant_max={:.6}cp",
        stats.transitions,
        stats.incremental_mismatches,
        stats.undo_mismatches,
        percentile(&stats.dequant_diffs_cp, 0.99),
        stats.max_dequant_diff_cp,
    );
    Ok(())
}

fn record_position(
    stats: &mut Stats,
    args: &Args,
    board: &Board,
    inc: &mut IncrementalQuantizedCodebookEval,
    quant: &figrid_board::codebook_eval::QuantizedCodebookWeights,
    dequant: &CodebookWeights,
) {
    let inc_v = inc.value(board, quant);
    let full_v = evaluate_full_quantized(board, quant);
    if !close(inc_v, full_v) {
        stats.incremental_mismatches += 1;
    }
    let full_diff = ((inc_v - full_v) as f64 * args.runtime_scale).abs();
    stats.max_full_diff_cp = stats.max_full_diff_cp.max(full_diff);
    stats.full_diffs_cp.push(full_diff);

    let fake_dequant_v = evaluate_full(board, dequant);
    let dequant_diff = ((inc_v - fake_dequant_v) as f64 * args.runtime_scale).abs();
    stats.max_dequant_diff_cp = stats.max_dequant_diff_cp.max(dequant_diff);
    stats.dequant_diffs_cp.push(dequant_diff);
}

fn parse_args() -> Result<Args, String> {
    let mut model = None;
    let mut games_jsonl = None;
    let mut out_json = None;
    let mut max_transitions = 100_000usize;
    let mut runtime_scale = 22.97f64;
    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => model = it.next().map(PathBuf::from),
            "--games-jsonl" => games_jsonl = it.next().map(PathBuf::from),
            "--out-json" => out_json = it.next().map(PathBuf::from),
            "--max-transitions" => {
                max_transitions = it
                    .next()
                    .ok_or_else(|| "missing --max-transitions value".to_string())?
                    .parse()
                    .map_err(|e| format!("bad --max-transitions: {e}"))?;
            }
            "--runtime-scale" => {
                runtime_scale = it
                    .next()
                    .ok_or_else(|| "missing --runtime-scale value".to_string())?
                    .parse()
                    .map_err(|e| format!("bad --runtime-scale: {e}"))?;
            }
            "-h" | "--help" => return Err(usage()),
            _ => return Err(format!("unknown arg `{arg}`\n{}", usage())),
        }
    }
    Ok(Args {
        model: model.ok_or_else(usage)?,
        games_jsonl: games_jsonl.ok_or_else(usage)?,
        out_json: out_json.ok_or_else(usage)?,
        max_transitions,
        runtime_scale,
    })
}

fn usage() -> String {
    "usage: rq554-quant-kernel-check --model MODEL.json --games-jsonl games.jsonl --out-json out.json [--max-transitions N] [--runtime-scale CP]".to_string()
}

fn load_trace_games(path: &PathBuf) -> Result<Vec<Vec<Move>>, String> {
    let file = File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let mut games = Vec::new();
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line =
            line.map_err(|e| format!("failed to read {}:{}: {e}", path.display(), line_no + 1))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)
            .map_err(|e| format!("failed to parse {}:{}: {e}", path.display(), line_no + 1))?;
        let moves = value
            .get("moves")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("missing moves array in {}:{}", path.display(), line_no + 1))?;
        let mut out = Vec::with_capacity(moves.len());
        for mv in moves {
            let x =
                mv.get("x").and_then(Value::as_u64).ok_or_else(|| {
                    format!("missing move x in {}:{}", path.display(), line_no + 1)
                })? as usize;
            let y =
                mv.get("y").and_then(Value::as_u64).ok_or_else(|| {
                    format!("missing move y in {}:{}", path.display(), line_no + 1)
                })? as usize;
            if x >= BOARD_SIZE || y >= BOARD_SIZE {
                return Err(format!(
                    "out-of-board move in {}:{}",
                    path.display(),
                    line_no + 1
                ));
            }
            out.push(y * BOARD_SIZE + x);
        }
        games.push(out);
    }
    Ok(games)
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-6
}

fn describe(xs: &[f64]) -> Value {
    json!({
        "count": xs.len(),
        "p50": percentile(xs, 0.50),
        "p90": percentile(xs, 0.90),
        "p99": percentile(xs, 0.99),
        "max": xs.last().copied().unwrap_or(0.0),
    })
}

fn describe_usize(xs: &[usize]) -> Value {
    json!({
        "count": xs.len(),
        "p50": percentile_usize(xs, 0.50),
        "p90": percentile_usize(xs, 0.90),
        "p95": percentile_usize(xs, 0.95),
        "max": xs.last().copied().unwrap_or(0),
    })
}

fn percentile(xs: &[f64], q: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let idx = ((xs.len() - 1) as f64 * q).round() as usize;
    xs[idx.min(xs.len() - 1)]
}

fn percentile_usize(xs: &[usize], q: f64) -> usize {
    if xs.is_empty() {
        return 0;
    }
    let idx = ((xs.len() - 1) as f64 * q).round() as usize;
    xs[idx.min(xs.len() - 1)]
}
