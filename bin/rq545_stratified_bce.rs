#![cfg(feature = "codebook-eval")]

use figrid_board::codebook_eval::{evaluate_full, CodebookWeights};
use figrid_board::eval::evaluate;
use figrid_board::{to_idx, Board, Stone, BOARD_SIZE, GOMOKU_NNUE_CONFIG};
use noru::network::NnueWeights;
use serde_json::{json, Value};
use std::collections::BTreeMap;
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
    flat_pred_scale: f64,
}

struct Sample {
    board: Board,
    target: f64,
    label_kind: String,
    eval_cp: Option<f64>,
}

#[derive(Default)]
struct Bucket {
    count: usize,
    sum_target: f64,
    sum_flat_pred: f64,
    sum_codebook_pred: f64,
    sum_flat_bce: f64,
    sum_codebook_bce: f64,
    target_extreme: usize,
}

#[derive(Default)]
struct LoadStats {
    records: usize,
    usable: usize,
    skipped_parse: usize,
}

fn main() -> Result<(), String> {
    let args = parse_args()?;
    let flat_weights = load_flat(&args.flat_model)?;
    let codebook_weights = load_codebook(&args.codebook_model)?;
    let (samples, load_stats) = load_samples(&args)?;
    if samples.is_empty() {
        return Err("no usable samples".to_string());
    }

    let mut buckets: BTreeMap<String, Bucket> = BTreeMap::new();
    let mut overall = Bucket::default();
    for sample in &samples {
        let flat_raw = evaluate(&sample.board, &flat_weights) as f64;
        let codebook_raw = evaluate_full(&sample.board, &codebook_weights) as f64;
        let flat_pred = sigmoid(flat_raw / args.flat_pred_scale);
        let codebook_pred = sigmoid(codebook_raw);
        let key = bucket_key(sample);
        update_bucket(
            buckets.entry(key).or_default(),
            sample.target,
            flat_pred,
            codebook_pred,
        );
        update_bucket(&mut overall, sample.target, flat_pred, codebook_pred);
    }

    let bucket_json: Vec<Value> = buckets
        .iter()
        .map(|(name, bucket)| bucket_to_json(name, bucket, overall.count))
        .collect();

    let report = json!({
        "format": "rq545-stratified-bce-v1",
        "jsonl": args.jsonl,
        "flat_model": args.flat_model,
        "codebook_model": args.codebook_model,
        "flat_pred_scale": args.flat_pred_scale,
        "max_samples": args.max_samples,
        "records_scanned": load_stats.records,
        "samples": load_stats.usable,
        "skipped_parse": load_stats.skipped_parse,
        "overall": bucket_to_json("overall", &overall, overall.count),
        "buckets": bucket_json,
    });

    let mut out = File::create(&args.out_json)
        .map_err(|e| format!("failed to create {}: {e}", args.out_json.display()))?;
    writeln!(out, "{}", serde_json::to_string_pretty(&report).unwrap())
        .map_err(|e| format!("failed to write {}: {e}", args.out_json.display()))?;

    println!(
        "rq545-stratified-bce: samples={} flat_bce={:.6} codebook_bce={:.6} delta={:+.6}",
        overall.count,
        mean(overall.sum_flat_bce, overall.count),
        mean(overall.sum_codebook_bce, overall.count),
        mean(overall.sum_flat_bce, overall.count) - mean(overall.sum_codebook_bce, overall.count),
    );
    for (name, bucket) in &buckets {
        println!(
            "  {name}: n={} share={:.1}% flat={:.6} codebook={:.6} delta={:+.6}",
            bucket.count,
            bucket.count as f64 / overall.count.max(1) as f64 * 100.0,
            mean(bucket.sum_flat_bce, bucket.count),
            mean(bucket.sum_codebook_bce, bucket.count),
            mean(bucket.sum_flat_bce, bucket.count) - mean(bucket.sum_codebook_bce, bucket.count),
        );
    }
    Ok(())
}

fn load_samples(args: &Args) -> Result<(Vec<Sample>, LoadStats), String> {
    let file = File::open(&args.jsonl)
        .map_err(|e| format!("failed to open {}: {e}", args.jsonl.display()))?;
    let mut stats = LoadStats::default();
    let mut samples = Vec::new();
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
        match parse_sample(&rec) {
            Some(sample) => {
                samples.push(sample);
                stats.usable += 1;
                if args.max_samples > 0 && samples.len() >= args.max_samples {
                    break;
                }
            }
            None => stats.skipped_parse += 1,
        }
    }
    Ok((samples, stats))
}

fn parse_sample(rec: &Value) -> Option<Sample> {
    if json_str(rec, "format") != Some("noru-rapfi-common-distill-v1") {
        return None;
    }
    let value = rec.get("value")?;
    let target = value.get("target_prob")?.as_f64()?;
    let label_kind = json_str(value, "label_kind")?.to_string();
    let eval_cp = value.get("eval_cp").and_then(Value::as_f64);
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
    Some(Sample {
        board,
        target,
        label_kind,
        eval_cp,
    })
}

fn bucket_key(sample: &Sample) -> String {
    match sample.label_kind.as_str() {
        "mate_win" => "mate_win".to_string(),
        "mate_loss" => "mate_loss".to_string(),
        "cp" => match sample.eval_cp.map(f64::abs) {
            Some(v) if v <= 150.0 => "cp_live_abs_le_150".to_string(),
            Some(v) if v <= 300.0 => "cp_mid_abs_151_300".to_string(),
            Some(_) => "cp_abs_gt_300".to_string(),
            None => "cp_no_eval_cp".to_string(),
        },
        other => format!("other_{other}"),
    }
}

fn update_bucket(bucket: &mut Bucket, target: f64, flat_pred: f64, codebook_pred: f64) {
    bucket.count += 1;
    bucket.sum_target += target;
    bucket.sum_flat_pred += flat_pred;
    bucket.sum_codebook_pred += codebook_pred;
    bucket.sum_flat_bce += bce(target, flat_pred);
    bucket.sum_codebook_bce += bce(target, codebook_pred);
    if !(0.1..=0.9).contains(&target) {
        bucket.target_extreme += 1;
    }
}

fn bucket_to_json(name: &str, bucket: &Bucket, total: usize) -> Value {
    let flat_bce = mean(bucket.sum_flat_bce, bucket.count);
    let codebook_bce = mean(bucket.sum_codebook_bce, bucket.count);
    json!({
        "bucket": name,
        "count": bucket.count,
        "share": bucket.count as f64 / total.max(1) as f64,
        "target_extreme_share": bucket.target_extreme as f64 / bucket.count.max(1) as f64,
        "mean_target": mean(bucket.sum_target, bucket.count),
        "mean_flat_pred": mean(bucket.sum_flat_pred, bucket.count),
        "mean_codebook_pred": mean(bucket.sum_codebook_pred, bucket.count),
        "flat_bce": flat_bce,
        "codebook_bce": codebook_bce,
        "delta_flat_minus_codebook": flat_bce - codebook_bce,
    })
}

fn bce(target: f64, pred: f64) -> f64 {
    let p = pred.clamp(1e-6, 1.0 - 1e-6);
    -(target * p.ln() + (1.0 - target) * (1.0 - p).ln())
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn mean(sum: f64, count: usize) -> f64 {
    sum / count.max(1) as f64
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

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        jsonl: PathBuf::new(),
        flat_model: PathBuf::new(),
        codebook_model: PathBuf::new(),
        out_json: PathBuf::from("rq545_stratified_bce.json"),
        max_samples: 0,
        flat_pred_scale: 25.0,
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
            "--flat-pred-scale" => {
                args.flat_pred_scale = next_arg(&mut iter, "--flat-pred-scale")?
                    .parse()
                    .map_err(|_| "invalid --flat-pred-scale".to_string())?;
                if args.flat_pred_scale <= 0.0 {
                    return Err("--flat-pred-scale must be positive".to_string());
                }
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
    Ok(args)
}

fn next_arg(iter: &mut impl Iterator<Item = String>, name: &str) -> Result<String, String> {
    iter.next()
        .ok_or_else(|| format!("missing value for {name}"))
}

fn usage() -> &'static str {
    "usage: rq545-stratified-bce --jsonl VAL.jsonl --flat-model MODEL.bin --codebook-model MODEL.json --out-json OUT.json [--flat-pred-scale 25] [--max-samples N]"
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
