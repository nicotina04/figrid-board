//! Evaluate the current figrid eval path on a Rapfi-labelled JSONL set.
//!
//! This is a small harness for the env-gated Relation Lite sidecar:
//! run once normally, then run again with `NORU_RELATION_LITE_SIDECAR=...`.

use figrid_board::vct::{classify_move_fast, ThreatKind};
use figrid_board::{evaluate, to_idx, Board, Move, Stone, BOARD_SIZE, GOMOKU_NNUE_CONFIG};
use noru::network::NnueWeights;
use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

const Q_SCALE: f32 = 200.0;
const MATE_LOGIT: f32 = 5.0;

#[derive(Default, Clone)]
struct SliceStats {
    n: usize,
    cp: usize,
    mate: usize,
    sum_abs: f64,
    sum_sq: f64,
    sign_correct: usize,
    xs: Vec<f64>,
    ys: Vec<f64>,
}

#[derive(Clone, Copy)]
struct Sample {
    target: f32,
    pred: f32,
    is_mate: bool,
}

struct Args {
    jsonl: String,
    model: String,
    exact5: bool,
    pred_scale: f32,
}

fn main() {
    let args = parse_args();
    let weights = load_model(&args.model);
    let file = File::open(&args.jsonl).unwrap_or_else(|e| {
        eprintln!("error opening {}: {e}", args.jsonl);
        std::process::exit(1);
    });

    let mut records = 0usize;
    let mut usable = 0usize;
    let mut skipped = 0usize;
    let mut slices: BTreeMap<String, SliceStats> = BTreeMap::new();

    for line in BufReader::new(file).lines().filter_map(Result::ok) {
        if line.trim().is_empty() {
            continue;
        }
        records += 1;
        let Ok(rec) = serde_json::from_str::<serde_json::Value>(&line) else {
            skipped += 1;
            continue;
        };
        let Some((board, target, is_mate)) = parse_record(&rec, args.exact5) else {
            skipped += 1;
            continue;
        };
        let sample = Sample {
            target,
            pred: evaluate(&board, &weights) as f32 / args.pred_scale,
            is_mate,
        };
        usable += 1;
        add_sample(&mut slices, "global", sample);
        add_sample(
            &mut slices,
            if is_mate { "label:mate" } else { "label:cp" },
            sample,
        );
        if let Some(shape) = shape_key(&rec, &board) {
            add_sample(&mut slices, shape, sample);
        }
    }

    println!("relation-lite-val");
    println!("  jsonl   : {}", args.jsonl);
    println!("  model   : {}", args.model);
    println!("  pred_scale: {}", args.pred_scale);
    println!(
        "  sidecar : {}",
        env::var("NORU_RELATION_LITE_SIDECAR").unwrap_or_else(|_| "(off)".to_string())
    );
    println!("  records : {records} usable={usable} skipped={skipped}");
    for key in sorted_keys(&slices) {
        let s = &slices[&key];
        println!(
            "  [{key}] n={} cp={} mate={} mae={:.4} mse={:.4} sign={:.1}% corr={:.3}",
            s.n,
            s.cp,
            s.mate,
            s.mae(),
            s.mse(),
            s.sign_acc() * 100.0,
            pearson(&s.xs, &s.ys),
        );
    }
}

fn parse_args() -> Args {
    let mut jsonl = String::new();
    let mut model = String::new();
    let mut exact5 = false;
    let mut pred_scale = 1.0;
    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--jsonl" => jsonl = iter.next().unwrap_or_else(|| usage_err("missing --jsonl")),
            "--model" => model = iter.next().unwrap_or_else(|| usage_err("missing --model")),
            "--exact5" => exact5 = true,
            "--pred-scale" => {
                pred_scale = iter
                    .next()
                    .unwrap_or_else(|| usage_err("missing --pred-scale"))
                    .parse()
                    .unwrap_or_else(|_| usage_err("invalid --pred-scale"));
                if pred_scale <= 0.0 {
                    usage_err("--pred-scale must be positive");
                }
            }
            "--help" | "-h" => {
                eprintln!("{}", usage());
                std::process::exit(0);
            }
            other => usage_err(&format!("unknown arg {other}")),
        }
    }
    if jsonl.is_empty() {
        usage_err("missing --jsonl");
    }
    if model.is_empty() {
        usage_err("missing --model");
    }
    Args {
        jsonl,
        model,
        exact5,
        pred_scale,
    }
}

fn usage_err(msg: &str) -> ! {
    eprintln!("error: {msg}");
    eprintln!("{}", usage());
    std::process::exit(2);
}

fn usage() -> &'static str {
    "usage: relation-lite-val --jsonl VAL.jsonl --model MODEL.bin [--exact5] [--pred-scale F]"
}

fn load_model(path: &str) -> NnueWeights {
    let data = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("error reading model {path}: {e}");
        std::process::exit(1);
    });
    NnueWeights::load_from_bytes(&data, Some(GOMOKU_NNUE_CONFIG)).unwrap_or_else(|e| {
        eprintln!("error loading model {path}: {e}");
        std::process::exit(1);
    })
}

fn parse_record(rec: &serde_json::Value, exact5: bool) -> Option<(Board, f32, bool)> {
    let eval_cp = rec
        .get("rapfi_eval_cp")
        .and_then(|v| if v.is_null() { None } else { v.as_i64() });
    let mate_in = rec
        .get("rapfi_mate_in")
        .and_then(|v| if v.is_null() { None } else { v.as_i64() });
    let (target, is_mate) = target_from_label(eval_cp, mate_in)?;
    let position = rec.get("position")?.as_array()?;
    let mut board = Board::new();
    board.exact5 = exact5;
    for stone in position {
        let color = parse_side(json_str(stone, "color")?)?;
        if color != board.side_to_move {
            return None;
        }
        let mv = move_from_value(stone)?;
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
    Some((board, target, is_mate))
}

fn target_from_label(eval_cp: Option<i64>, mate_in: Option<i64>) -> Option<(f32, bool)> {
    if let Some(m) = mate_in {
        Some((if m > 0 { MATE_LOGIT } else { -MATE_LOGIT }, true))
    } else {
        eval_cp.map(|cp| (cp as f32 / Q_SCALE, false))
    }
}

fn shape_key(rec: &serde_json::Value, board: &Board) -> Option<&'static str> {
    let played = move_value(rec, "played_move")?;
    let rapfi = move_value(rec, "rapfi_move")?;
    if played == rapfi || !board.is_empty(played) || !board.is_empty(rapfi) {
        return None;
    }
    let side = board.side_to_move;
    let played_attack = classify_move_fast(board, played, side);
    let played_block = classify_move_fast(board, played, side.opponent());
    let rapfi_attack = classify_move_fast(board, rapfi, side);
    let rapfi_block = classify_move_fast(board, rapfi, side.opponent());
    let played_weak = is_weak_attack(played_attack, played_block);
    let rapfi_weak = is_weak_attack(rapfi_attack, rapfi_block);
    match (played_weak, rapfi_weak) {
        (true, false) => Some("shape:played_weak_rapfi_not"),
        (true, true) => Some("shape:both_weak_attack"),
        (false, true) => Some("shape:rapfi_weak_played_not"),
        (false, false) => Some("shape:neither_weak_attack"),
    }
}

fn is_weak_attack(attack: ThreatKind, block: ThreatKind) -> bool {
    matches!(attack, ThreatKind::ClosedFour | ThreatKind::OpenThree) && block == ThreatKind::None
}

fn add_sample(slices: &mut BTreeMap<String, SliceStats>, key: &str, sample: Sample) {
    slices.entry(key.to_string()).or_default().add(sample);
}

impl SliceStats {
    fn add(&mut self, sample: Sample) {
        self.n += 1;
        if sample.is_mate {
            self.mate += 1;
        } else {
            self.cp += 1;
        }
        let diff = (sample.pred - sample.target) as f64;
        self.sum_abs += diff.abs();
        self.sum_sq += diff * diff;
        if (sample.pred > 0.0) == (sample.target > 0.0) {
            self.sign_correct += 1;
        }
        self.xs.push(sample.target as f64);
        self.ys.push(sample.pred as f64);
    }

    fn mae(&self) -> f64 {
        self.sum_abs / self.n.max(1) as f64
    }

    fn mse(&self) -> f64 {
        self.sum_sq / self.n.max(1) as f64
    }

    fn sign_acc(&self) -> f64 {
        self.sign_correct as f64 / self.n.max(1) as f64
    }
}

fn sorted_keys(slices: &BTreeMap<String, SliceStats>) -> Vec<String> {
    let mut keys: Vec<_> = slices.keys().cloned().collect();
    keys.sort();
    if let Some(pos) = keys.iter().position(|k| k == "global") {
        keys.swap(0, pos);
    }
    keys
}

fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() < 2 || xs.len() != ys.len() {
        return 0.0;
    }
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;
    for (&x, &y) in xs.iter().zip(ys) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    let den = (den_x * den_y).sqrt();
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

fn move_value(rec: &serde_json::Value, key: &str) -> Option<Move> {
    rec.get(key).and_then(move_from_value)
}

fn move_from_value(v: &serde_json::Value) -> Option<Move> {
    let x = json_usize(v, "x")?;
    let y = json_usize(v, "y")?;
    if x >= BOARD_SIZE || y >= BOARD_SIZE {
        return None;
    }
    Some(to_idx(y, x))
}

fn json_usize(v: &serde_json::Value, key: &str) -> Option<usize> {
    v.get(key)?.as_u64().map(|x| x as usize)
}

fn json_str<'a>(v: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(|x| x.as_str())
}

fn parse_side(s: &str) -> Option<Stone> {
    match s {
        "B" | "b" | "black" | "Black" => Some(Stone::Black),
        "W" | "w" | "white" | "White" => Some(Stone::White),
        _ => None,
    }
}
