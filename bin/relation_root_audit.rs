//! Compare baseline/root-only game records and dump the root candidate table
//! at the first divergent position.

use figrid_board::vct::classify_move_fast;
use figrid_board::{to_idx, to_rc, Board, Move, Searcher, Stone, BOARD_SIZE, GOMOKU_NNUE_CONFIG};
use noru::network::NnueWeights;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Duration;

struct Args {
    baseline_jsonl: String,
    test_jsonl: String,
    model: String,
    seeds: Vec<i64>,
    max_depth: u32,
    time_ms: u64,
    top: usize,
    exact5: bool,
}

fn main() {
    let args = parse_args();
    let weights = load_model(&args.model);
    let baseline = load_games(&args.baseline_jsonl);
    let test = load_games(&args.test_jsonl);

    println!("relation-root-audit");
    println!("  baseline: {}", args.baseline_jsonl);
    println!("  test    : {}", args.test_jsonl);
    println!("  model   : {}", args.model);
    println!("  max_depth: {}", args.max_depth);
    println!("  time_ms : {}", args.time_ms);
    println!(
        "  sidecar : {}",
        env::var("NORU_RELATION_LITE_SIDECAR").unwrap_or_else(|_| "(off)".to_string())
    );
    println!(
        "  mode    : {}",
        env::var("NORU_RELATION_LITE_MODE").unwrap_or_else(|_| "(unset)".to_string())
    );
    println!(
        "  margin  : {}",
        env::var("NORU_RELATION_LITE_ROOT_MARGIN").unwrap_or_else(|_| "(default)".to_string())
    );
    println!(
        "  gate    : {}",
        env::var("NORU_RELATION_LITE_ROOT_GATE").unwrap_or_else(|_| "(unset)".to_string())
    );

    for seed in &args.seeds {
        let Some(base) = baseline.get(seed) else {
            eprintln!("warning: seed {seed} missing from baseline");
            continue;
        };
        let Some(other) = test.get(seed) else {
            eprintln!("warning: seed {seed} missing from test");
            continue;
        };
        audit_seed(*seed, base, other, &weights, &args);
    }
}

fn audit_seed(seed: i64, base: &Value, other: &Value, weights: &NnueWeights, args: &Args) {
    let base_moves = base["moves"].as_array().unwrap_or_else(|| {
        eprintln!("seed {seed}: baseline moves missing");
        std::process::exit(1);
    });
    let other_moves = other["moves"].as_array().unwrap_or_else(|| {
        eprintln!("seed {seed}: test moves missing");
        std::process::exit(1);
    });
    let div = first_divergence(base_moves, other_moves);
    if div >= base_moves.len() || div >= other_moves.len() {
        println!("\nseed {seed}: no divergent move inside both records");
        return;
    }

    let mut board = Board::new();
    board.exact5 = args.exact5;
    for mv in &base_moves[..div] {
        let color = parse_move_color(mv).unwrap_or_else(|| fail_seed(seed, "bad move color"));
        if color != board.side_to_move {
            fail_seed(seed, "move color does not match side_to_move");
        }
        let idx = move_from_value(mv).unwrap_or_else(|| fail_seed(seed, "bad move coord"));
        if !board.is_empty(idx) {
            fail_seed(seed, "occupied move while replaying position");
        }
        board.make_move(idx);
    }

    let base_mv =
        move_from_value(&base_moves[div]).unwrap_or_else(|| fail_seed(seed, "bad baseline move"));
    let test_mv =
        move_from_value(&other_moves[div]).unwrap_or_else(|| fail_seed(seed, "bad test move"));

    let mut search_board = board.clone();
    let mut searcher = Searcher::new();
    let audit = searcher.audit_root_candidates(
        &mut search_board,
        weights,
        args.max_depth,
        Some(Duration::from_millis(args.time_ms)),
    );

    let chosen = audit.result.best_move;
    let mut search_ranked = audit.candidates.clone();
    search_ranked.sort_by(|a, b| b.search_score.cmp(&a.search_score));
    let mut search_rank = HashMap::new();
    for (i, c) in search_ranked.iter().enumerate() {
        search_rank.entry(c.mv).or_insert(i + 1);
    }

    let mut relation_ranked: Vec<_> = audit
        .candidates
        .iter()
        .filter(|c| c.relation_score.is_some())
        .cloned()
        .collect();
    relation_ranked.sort_by(|a, b| b.relation_score.cmp(&a.relation_score));
    let mut relation_rank = HashMap::new();
    for (i, c) in relation_ranked.iter().enumerate() {
        relation_rank.entry(c.mv).or_insert(i + 1);
    }

    let raw_best = search_ranked
        .first()
        .map(|c| c.search_score)
        .unwrap_or_default();
    let mut rows = BTreeSet::new();
    for c in search_ranked.iter().take(args.top) {
        rows.insert(c.mv);
    }
    for c in relation_ranked.iter().take(args.top) {
        rows.insert(c.mv);
    }
    rows.insert(base_mv);
    rows.insert(test_mv);
    if let Some(mv) = chosen {
        rows.insert(mv);
    }

    println!("\nseed {seed}");
    println!(
        "  engines : baseline {} vs {}, test {} vs {}",
        json_str(base, "black_engine").unwrap_or("?"),
        json_str(base, "white_engine").unwrap_or("?"),
        json_str(other, "black_engine").unwrap_or("?"),
        json_str(other, "white_engine").unwrap_or("?"),
    );
    println!(
        "  result  : baseline {}, test {}",
        json_str(base, "result").unwrap_or("?"),
        json_str(other, "result").unwrap_or("?")
    );
    println!(
        "  ply     : {} side={} stones={}",
        div,
        stone_name(board.side_to_move),
        board.move_count
    );
    println!("  baseline move: {}", move_name(base_mv));
    println!("  test move    : {}", move_name(test_mv));
    println!(
        "  audit chosen : {} score={} depth={} nodes={}",
        chosen
            .map(move_name)
            .unwrap_or_else(|| "(none)".to_string()),
        audit.result.score,
        audit.result.depth,
        audit.result.nodes
    );
    println!(
        "  candidates: {} shown={} raw_best={}",
        audit.candidates.len(),
        rows.len(),
        raw_best
    );
    println!("  move\tmarks\ts_rank\tr_rank\tsearch\tdelta\trelation\tattack\tblock\tforcing");

    let by_move: HashMap<Move, _> = audit.candidates.iter().map(|c| (c.mv, c)).collect();
    for mv in rows {
        let Some(c) = by_move.get(&mv) else {
            println!("  {}\tmissing", move_name(mv));
            continue;
        };
        let mut marks = String::new();
        if mv == base_mv {
            marks.push('B');
        }
        if mv == test_mv {
            marks.push('T');
        }
        if Some(mv) == chosen {
            marks.push('C');
        }
        if search_rank.get(&mv) == Some(&1) {
            marks.push('S');
        }
        if relation_rank.get(&mv) == Some(&1) {
            marks.push('R');
        }
        if marks.is_empty() {
            marks.push('-');
        }

        let side = board.side_to_move;
        let attack = classify_move_fast(&board, mv, side);
        let block = classify_move_fast(&board, mv, side.opponent());
        println!(
            "  {}\t{}\t{}\t{}\t{}\t{}\t{}\t{:?}\t{:?}\t{}",
            move_name(mv),
            marks,
            search_rank
                .get(&mv)
                .map(usize::to_string)
                .unwrap_or_else(|| "-".to_string()),
            relation_rank
                .get(&mv)
                .map(usize::to_string)
                .unwrap_or_else(|| "-".to_string()),
            c.search_score,
            raw_best - c.search_score,
            c.relation_score
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string()),
            attack,
            block,
            c.is_forcing,
        );
    }
}

fn parse_args() -> Args {
    let mut baseline_jsonl = String::new();
    let mut test_jsonl = String::new();
    let mut model = String::new();
    let mut seeds = Vec::new();
    let mut max_depth = 20;
    let mut time_ms = 850;
    let mut top = 12;
    let mut exact5 = false;

    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--baseline-jsonl" => {
                baseline_jsonl = iter
                    .next()
                    .unwrap_or_else(|| usage_err("missing --baseline-jsonl"))
            }
            "--test-jsonl" => {
                test_jsonl = iter
                    .next()
                    .unwrap_or_else(|| usage_err("missing --test-jsonl"))
            }
            "--model" => model = iter.next().unwrap_or_else(|| usage_err("missing --model")),
            "--seeds" => {
                let raw = iter.next().unwrap_or_else(|| usage_err("missing --seeds"));
                seeds = raw
                    .split(',')
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| {
                        s.trim()
                            .parse()
                            .unwrap_or_else(|_| usage_err("invalid seed"))
                    })
                    .collect();
            }
            "--max-depth" => {
                max_depth = iter
                    .next()
                    .unwrap_or_else(|| usage_err("missing --max-depth"))
                    .parse()
                    .unwrap_or_else(|_| usage_err("invalid --max-depth"))
            }
            "--time-ms" => {
                time_ms = iter
                    .next()
                    .unwrap_or_else(|| usage_err("missing --time-ms"))
                    .parse()
                    .unwrap_or_else(|_| usage_err("invalid --time-ms"))
            }
            "--top" => {
                top = iter
                    .next()
                    .unwrap_or_else(|| usage_err("missing --top"))
                    .parse()
                    .unwrap_or_else(|_| usage_err("invalid --top"))
            }
            "--exact5" => exact5 = true,
            "--help" | "-h" => {
                eprintln!("{}", usage());
                std::process::exit(0);
            }
            other => usage_err(&format!("unknown arg {other}")),
        }
    }

    if baseline_jsonl.is_empty() {
        usage_err("missing --baseline-jsonl");
    }
    if test_jsonl.is_empty() {
        usage_err("missing --test-jsonl");
    }
    if model.is_empty() {
        usage_err("missing --model");
    }
    if seeds.is_empty() {
        usage_err("missing --seeds");
    }
    Args {
        baseline_jsonl,
        test_jsonl,
        model,
        seeds,
        max_depth,
        time_ms,
        top,
        exact5,
    }
}

fn usage_err(msg: &str) -> ! {
    eprintln!("error: {msg}");
    eprintln!("{}", usage());
    std::process::exit(2);
}

fn usage() -> &'static str {
    "usage: relation-root-audit --baseline-jsonl BASE.jsonl --test-jsonl TEST.jsonl --model MODEL.bin --seeds S1,S2 [--max-depth 20] [--time-ms 850] [--top 12] [--exact5]"
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

fn load_games(path: &str) -> BTreeMap<i64, Value> {
    let file = File::open(path).unwrap_or_else(|e| {
        eprintln!("error opening {path}: {e}");
        std::process::exit(1);
    });
    let mut out = BTreeMap::new();
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        if line.trim().is_empty() {
            continue;
        }
        let rec: Value = serde_json::from_str(&line).unwrap_or_else(|e| {
            eprintln!("error parsing {path}: {e}");
            std::process::exit(1);
        });
        if let Some(seed) = rec.get("seed").and_then(Value::as_i64) {
            out.insert(seed, rec);
        }
    }
    out
}

fn first_divergence(a: &[Value], b: &[Value]) -> usize {
    let n = a.len().min(b.len());
    for i in 0..n {
        if move_signature(&a[i]) != move_signature(&b[i]) {
            return i;
        }
    }
    n
}

fn move_signature(v: &Value) -> Option<(usize, usize, Stone)> {
    let mv = move_from_value(v)?;
    Some((mv % BOARD_SIZE, mv / BOARD_SIZE, parse_move_color(v)?))
}

fn move_from_value(v: &Value) -> Option<Move> {
    let x = v.get("x")?.as_u64()? as usize;
    let y = v.get("y")?.as_u64()? as usize;
    if x >= BOARD_SIZE || y >= BOARD_SIZE {
        return None;
    }
    Some(to_idx(y, x))
}

fn parse_move_color(v: &Value) -> Option<Stone> {
    match v.get("color")?.as_str()? {
        "B" | "b" => Some(Stone::Black),
        "W" | "w" => Some(Stone::White),
        _ => None,
    }
}

fn move_name(mv: Move) -> String {
    let (row, col) = to_rc(mv);
    format!("{col},{row}")
}

fn stone_name(stone: Stone) -> &'static str {
    match stone {
        Stone::Black => "B",
        Stone::White => "W",
    }
}

fn json_str<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Value::as_str)
}

fn fail_seed(seed: i64, msg: &str) -> ! {
    eprintln!("seed {seed}: {msg}");
    std::process::exit(1);
}
