#![cfg(feature = "codebook-eval")]

use figrid_board::codebook_eval::{evaluate_full, CodebookWeights};
use figrid_board::eval::evaluate;
use figrid_board::{to_idx, Board, Stone, BOARD_SIZE, GOMOKU_NNUE_CONFIG};
use noru::network::NnueWeights;
use serde_json::{json, Value};
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

#[derive(Debug)]
struct Args {
    jsonl: PathBuf,
    flat_model: PathBuf,
    codebook_model: PathBuf,
    out_json: PathBuf,
    max_samples: usize,
}

struct Sample {
    board: Board,
}

#[derive(Default)]
struct Stats {
    records: usize,
    usable: usize,
    skipped_non_cp: usize,
    skipped_parse: usize,
}

fn main() -> Result<(), String> {
    let args = parse_args()?;
    let flat_weights = load_flat(&args.flat_model)?;
    let codebook_weights = load_codebook(&args.codebook_model)?;
    let (samples, stats) = load_samples(&args)?;
    if samples.is_empty() {
        return Err("no usable cp samples".to_string());
    }

    let mut flat = Vec::with_capacity(samples.len());
    let mut codebook = Vec::with_capacity(samples.len());
    for sample in &samples {
        flat.push(evaluate(&sample.board, &flat_weights) as f64);
        codebook.push(evaluate_full(&sample.board, &codebook_weights) as f64);
    }

    let flat_abs = abs_values(&flat);
    let codebook_abs = abs_values(&codebook);
    let flat_p50 = quantile(flat_abs.clone(), 0.50);
    let flat_p90 = quantile(flat_abs.clone(), 0.90);
    let code_p50 = quantile(codebook_abs.clone(), 0.50);
    let code_p90 = quantile(codebook_abs.clone(), 0.90);
    let scale_p50 = positive_ratio(flat_p50, code_p50)?;
    let scale_p90 = positive_ratio(flat_p90, code_p90)?;
    let chosen_scale = (scale_p50.ln() + scale_p90.ln()).mul_add(0.5, 0.0).exp();
    let codebook_scaled: Vec<f64> = codebook.iter().map(|v| v * chosen_scale).collect();
    let codebook_scaled_abs = abs_values(&codebook_scaled);

    let report = json!({
        "format": "rq544-eval-scale-v1",
        "jsonl": args.jsonl,
        "flat_model": args.flat_model,
        "codebook_model": args.codebook_model,
        "filter": "first max_samples rows with value.label_kind == cp and value.eval_cp != null",
        "max_samples": args.max_samples,
        "records_scanned": stats.records,
        "samples": stats.usable,
        "skipped": {
            "non_cp_or_no_eval_cp": stats.skipped_non_cp,
            "parse": stats.skipped_parse,
        },
        "flat_abs_cp": describe(&flat_abs),
        "codebook_raw_abs": describe(&codebook_abs),
        "scale": {
            "p50_ratio": scale_p50,
            "p90_ratio": scale_p90,
            "chosen_geomean_p50_p90": chosen_scale,
        },
        "codebook_scaled_abs_cp": describe(&codebook_scaled_abs),
        "signed": {
            "flat": describe_signed(&flat),
            "codebook_raw": describe_signed(&codebook),
            "codebook_scaled": describe_signed(&codebook_scaled),
            "sign_agreement": sign_agreement(&flat, &codebook),
            "pearson": pearson(&flat, &codebook),
        },
    });

    let mut out = File::create(&args.out_json)
        .map_err(|e| format!("failed to create {}: {e}", args.out_json.display()))?;
    writeln!(out, "{}", serde_json::to_string_pretty(&report).unwrap())
        .map_err(|e| format!("failed to write {}: {e}", args.out_json.display()))?;

    println!(
        "rq544-eval-scale: samples={} flat_abs_p50={:.3} flat_abs_p90={:.3} code_raw_abs_p50={:.6} code_raw_abs_p90={:.6} scale={:.6}",
        stats.usable, flat_p50, flat_p90, code_p50, code_p90, chosen_scale
    );
    Ok(())
}

fn load_samples(args: &Args) -> Result<(Vec<Sample>, Stats), String> {
    let file = File::open(&args.jsonl)
        .map_err(|e| format!("failed to open {}: {e}", args.jsonl.display()))?;
    let mut stats = Stats::default();
    let mut samples = Vec::with_capacity(args.max_samples);
    for line in BufReader::new(file).lines() {
        let line = line.map_err(|e| format!("failed to read jsonl line: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        stats.records += 1;
        let rec = match serde_json::from_str::<Value>(&line) {
            Ok(v) => v,
            Err(_) => {
                stats.skipped_parse += 1;
                continue;
            }
        };
        let value = match rec.get("value") {
            Some(v) => v,
            None => {
                stats.skipped_parse += 1;
                continue;
            }
        };
        if json_str(value, "label_kind") != Some("cp")
            || value.get("eval_cp").and_then(Value::as_f64).is_none()
        {
            stats.skipped_non_cp += 1;
            continue;
        }
        match parse_board(&rec) {
            Some(board) => {
                samples.push(Sample { board });
                stats.usable += 1;
                if samples.len() >= args.max_samples {
                    break;
                }
            }
            None => stats.skipped_parse += 1,
        }
    }
    Ok((samples, stats))
}

fn parse_board(rec: &Value) -> Option<Board> {
    if json_str(rec, "format") != Some("noru-rapfi-common-distill-v1") {
        return None;
    }
    let history = rec.get("history")?.as_array()?;
    let mut board = Board::new();
    for stone in history {
        let x = json_usize(stone, "x")?;
        let y = json_usize(stone, "y")?;
        if x >= BOARD_SIZE || y >= BOARD_SIZE {
            return None;
        }
        let color = json_str(stone, "color").and_then(parse_side)?;
        if color != board.side_to_move {
            return None;
        }
        let mv = to_idx(y, x);
        if !board.is_empty(mv) {
            return None;
        }
        board.make_move(mv);
    }
    if let Some(side) = json_str(rec, "side_to_move").and_then(parse_side) {
        if side != board.side_to_move {
            return None;
        }
    }
    Some(board)
}

fn load_flat(path: &PathBuf) -> Result<NnueWeights, String> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    NnueWeights::load_from_bytes(&bytes, Some(GOMOKU_NNUE_CONFIG))
        .map_err(|e| format!("failed to parse {}: {e}", path.display()))
}

fn load_codebook(path: &PathBuf) -> Result<CodebookWeights, String> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    CodebookWeights::from_json_bytes(&bytes)
}

fn describe(values: &[f64]) -> Value {
    json!({
        "mean": mean(values),
        "p50": quantile(values.to_vec(), 0.50),
        "p90": quantile(values.to_vec(), 0.90),
        "p95": quantile(values.to_vec(), 0.95),
        "max": values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    })
}

fn describe_signed(values: &[f64]) -> Value {
    json!({
        "mean": mean(values),
        "p10": quantile(values.to_vec(), 0.10),
        "p50": quantile(values.to_vec(), 0.50),
        "p90": quantile(values.to_vec(), 0.90),
        "min": values.iter().copied().fold(f64::INFINITY, f64::min),
        "max": values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    })
}

fn abs_values(values: &[f64]) -> Vec<f64> {
    values.iter().map(|v| v.abs()).collect()
}

fn quantile(mut values: Vec<f64>, q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let pos = ((values.len() - 1) as f64 * q).round() as usize;
    values[pos.min(values.len() - 1)]
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn positive_ratio(num: f64, den: f64) -> Result<f64, String> {
    if num <= 0.0 || den <= 0.0 {
        return Err(format!("non-positive quantile ratio: num={num}, den={den}"));
    }
    Ok(num / den)
}

fn sign_agreement(a: &[f64], b: &[f64]) -> f64 {
    let mut total = 0usize;
    let mut agree = 0usize;
    for (&x, &y) in a.iter().zip(b) {
        if x == 0.0 || y == 0.0 {
            continue;
        }
        total += 1;
        if x.signum() == y.signum() {
            agree += 1;
        }
    }
    agree as f64 / total.max(1) as f64
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let mean_a = mean(a);
    let mean_b = mean(b);
    let mut num = 0.0;
    let mut den_a = 0.0;
    let mut den_b = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        let dx = x - mean_a;
        let dy = y - mean_b;
        num += dx * dy;
        den_a += dx * dx;
        den_b += dy * dy;
    }
    if den_a <= 0.0 || den_b <= 0.0 {
        0.0
    } else {
        num / (den_a.sqrt() * den_b.sqrt())
    }
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        jsonl: PathBuf::new(),
        flat_model: PathBuf::new(),
        codebook_model: PathBuf::new(),
        out_json: PathBuf::from("rq544_eval_scale.json"),
        max_samples: 10_000,
    };
    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--jsonl" => args.jsonl = PathBuf::from(next_arg(&mut iter, "--jsonl")?),
            "--flat-model" => args.flat_model = PathBuf::from(next_arg(&mut iter, "--flat-model")?),
            "--codebook-model" => {
                args.codebook_model = PathBuf::from(next_arg(&mut iter, "--codebook-model")?)
            }
            "--out-json" => args.out_json = PathBuf::from(next_arg(&mut iter, "--out-json")?),
            "--max-samples" => {
                args.max_samples = next_arg(&mut iter, "--max-samples")?
                    .parse()
                    .map_err(|_| "invalid --max-samples".to_string())?
            }
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg {other}\n{}", usage())),
        }
    }
    if args.jsonl.as_os_str().is_empty() {
        return Err(format!("missing --jsonl\n{}", usage()));
    }
    if args.flat_model.as_os_str().is_empty() {
        return Err(format!("missing --flat-model\n{}", usage()));
    }
    if args.codebook_model.as_os_str().is_empty() {
        return Err(format!("missing --codebook-model\n{}", usage()));
    }
    if args.max_samples == 0 {
        return Err("--max-samples must be positive".to_string());
    }
    Ok(args)
}

fn next_arg(iter: &mut impl Iterator<Item = String>, name: &str) -> Result<String, String> {
    iter.next()
        .ok_or_else(|| format!("missing value for {name}"))
}

fn usage() -> &'static str {
    "usage: rq544-eval-scale --jsonl VAL.jsonl --flat-model MODEL.bin --codebook-model MODEL.json --out-json OUT.json [--max-samples N]"
}

fn json_usize(v: &Value, key: &str) -> Option<usize> {
    v.get(key).and_then(Value::as_u64).map(|u| u as usize)
}

fn json_str<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Value::as_str)
}

fn parse_side(raw: &str) -> Option<Stone> {
    match raw {
        "B" | "Black" | "black" => Some(Stone::Black),
        "W" | "White" | "white" => Some(Stone::White),
        _ => None,
    }
}
