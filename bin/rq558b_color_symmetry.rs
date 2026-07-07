#![cfg(feature = "codebook-eval")]

use figrid_board::board::{ZOBRIST_SIDE, zobrist_stone_key};
use figrid_board::codebook_eval::{CodebookWeights, evaluate_full};
use figrid_board::eval::{compute_active_features, evaluate, evaluate_base};
use figrid_board::pattern_table::{lookup_mapped_id, pack_window, read_window};
use figrid_board::{BOARD_SIZE, Board, GOMOKU_NNUE_CONFIG, NUM_CELLS, Stone, to_idx};
use noru::network::NnueWeights;
use serde_json::{Value, json};
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

#[derive(Debug)]
struct Args {
    games_jsonl: Vec<PathBuf>,
    flat_model: PathBuf,
    codebook_model: PathBuf,
    out_json: PathBuf,
    max_positions: usize,
    min_ply: usize,
}

struct Sample {
    source: String,
    game_id: i64,
    ply: usize,
    board: Board,
}

#[derive(Default)]
struct LoadStats {
    files: usize,
    games: usize,
    parsed_games: usize,
    skipped_games: usize,
    sampled_positions: usize,
}

struct Row {
    source: String,
    game_id: i64,
    ply: usize,
    flat_base: i32,
    flat_base_swapped: i32,
    flat_runtime: i32,
    flat_runtime_swapped: i32,
    codebook: f32,
    codebook_swapped: f32,
    stm_features_equal: bool,
    nstm_features_equal: bool,
    stm_len: usize,
    nstm_len: usize,
    stm_swapped_len: usize,
    nstm_swapped_len: usize,
}

fn main() -> Result<(), String> {
    let args = parse_args()?;
    let flat_weights = load_flat(&args.flat_model)?;
    let codebook_weights = load_codebook(&args.codebook_model)?;
    let (samples, load_stats) = load_samples(&args)?;
    if samples.is_empty() {
        return Err("no usable black-to-move positions sampled".to_string());
    }

    let mut rows = Vec::with_capacity(samples.len());
    for sample in &samples {
        let swapped = color_swapped_board(&sample.board);
        debug_assert_eq!(sample.board.side_to_move, Stone::Black);
        debug_assert_eq!(swapped.side_to_move, Stone::White);

        let (mut stm, mut nstm) = compute_active_features(&sample.board);
        let (mut stm_swapped, mut nstm_swapped) = compute_active_features(&swapped);
        stm.sort_unstable();
        nstm.sort_unstable();
        stm_swapped.sort_unstable();
        nstm_swapped.sort_unstable();

        rows.push(Row {
            source: sample.source.clone(),
            game_id: sample.game_id,
            ply: sample.ply,
            flat_base: evaluate_base(&sample.board, &flat_weights),
            flat_base_swapped: evaluate_base(&swapped, &flat_weights),
            flat_runtime: evaluate(&sample.board, &flat_weights),
            flat_runtime_swapped: evaluate(&swapped, &flat_weights),
            codebook: evaluate_full(&sample.board, &codebook_weights),
            codebook_swapped: evaluate_full(&swapped, &codebook_weights),
            stm_features_equal: stm == stm_swapped,
            nstm_features_equal: nstm == nstm_swapped,
            stm_len: stm.len(),
            nstm_len: nstm.len(),
            stm_swapped_len: stm_swapped.len(),
            nstm_swapped_len: nstm_swapped.len(),
        });
    }

    let flat_base_diffs: Vec<f64> = rows
        .iter()
        .map(|r| (r.flat_base - r.flat_base_swapped) as f64)
        .collect();
    let flat_runtime_diffs: Vec<f64> = rows
        .iter()
        .map(|r| (r.flat_runtime - r.flat_runtime_swapped) as f64)
        .collect();
    let codebook_diffs: Vec<f64> = rows
        .iter()
        .map(|r| (r.codebook - r.codebook_swapped) as f64)
        .collect();
    let feature_mismatches = rows
        .iter()
        .filter(|r| !r.stm_features_equal || !r.nstm_features_equal)
        .count();

    let report = json!({
        "format": "rq558b-color-symmetry-v1",
        "intent": "Assert eval(P, black-to-move) approximately equals eval(color_swap(P), white-to-move) on real game prefixes.",
        "inputs": {
            "games_jsonl": args.games_jsonl,
            "flat_model": args.flat_model,
            "codebook_model": args.codebook_model,
            "max_positions": args.max_positions,
            "min_ply": args.min_ply,
            "sample_filter": "real game prefixes where side_to_move is Black; swapped board flips black/white stones and side_to_move to White",
        },
        "load_stats": {
            "files": load_stats.files,
            "games": load_stats.games,
            "parsed_games": load_stats.parsed_games,
            "skipped_games": load_stats.skipped_games,
            "sampled_positions": load_stats.sampled_positions,
        },
        "summary": {
            "positions": rows.len(),
            "feature_mismatches": feature_mismatches,
            "flat_base_diff_cp": describe_signed(&flat_base_diffs),
            "flat_runtime_diff_cp": describe_signed(&flat_runtime_diffs),
            "codebook_raw_diff": describe_signed(&codebook_diffs),
        },
        "worst_examples": worst_examples(&rows, 8),
    });

    let mut out = File::create(&args.out_json)
        .map_err(|e| format!("failed to create {}: {e}", args.out_json.display()))?;
    writeln!(out, "{}", serde_json::to_string_pretty(&report).unwrap())
        .map_err(|e| format!("failed to write {}: {e}", args.out_json.display()))?;

    println!(
        "rq558b-color-symmetry: positions={} feature_mismatches={} flat_base_abs_p99={:.3} flat_runtime_abs_p99={:.3} codebook_abs_p99={:.6}",
        rows.len(),
        feature_mismatches,
        abs_quantile(&flat_base_diffs, 0.99),
        abs_quantile(&flat_runtime_diffs, 0.99),
        abs_quantile(&codebook_diffs, 0.99),
    );
    Ok(())
}

fn load_samples(args: &Args) -> Result<(Vec<Sample>, LoadStats), String> {
    let mut stats = LoadStats::default();
    let mut samples = Vec::with_capacity(args.max_positions);
    for path in &args.games_jsonl {
        stats.files += 1;
        let file =
            File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
        let source = path.display().to_string();
        for line in BufReader::new(file).lines() {
            let line = line.map_err(|e| format!("failed to read {}: {e}", path.display()))?;
            if line.trim().is_empty() {
                continue;
            }
            stats.games += 1;
            let rec = match serde_json::from_str::<Value>(&line) {
                Ok(v) => v,
                Err(_) => {
                    stats.skipped_games += 1;
                    continue;
                }
            };
            let game_id = rec.get("game_id").and_then(Value::as_i64).unwrap_or(-1);
            let moves = match rec.get("moves").and_then(Value::as_array) {
                Some(moves) => moves,
                None => {
                    stats.skipped_games += 1;
                    continue;
                }
            };
            stats.parsed_games += 1;
            let mut board = Board::new();
            for (ply, mv_json) in moves.iter().enumerate() {
                if ply >= args.min_ply && board.side_to_move == Stone::Black {
                    samples.push(Sample {
                        source: source.clone(),
                        game_id,
                        ply,
                        board: board.clone(),
                    });
                    stats.sampled_positions += 1;
                    if samples.len() >= args.max_positions {
                        return Ok((samples, stats));
                    }
                }
                let Some((mv, color)) = parse_move(mv_json) else {
                    stats.skipped_games += 1;
                    break;
                };
                if color != board.side_to_move || !board.is_empty(mv) {
                    stats.skipped_games += 1;
                    break;
                }
                board.make_move(mv);
            }
        }
    }
    Ok((samples, stats))
}

fn parse_move(v: &Value) -> Option<(usize, Stone)> {
    let x = v.get("x")?.as_u64()? as usize;
    let y = v.get("y")?.as_u64()? as usize;
    if x >= BOARD_SIZE || y >= BOARD_SIZE {
        return None;
    }
    let color = parse_side(v.get("color")?.as_str()?)?;
    Some((to_idx(y, x), color))
}

fn color_swapped_board(board: &Board) -> Board {
    let mut out = board.clone();
    std::mem::swap(&mut out.black, &mut out.white);
    out.side_to_move = board.side_to_move.opponent();
    rebuild_aux_state(&mut out);
    out
}

fn rebuild_aux_state(board: &mut Board) {
    const DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
    for cell in 0..NUM_CELLS {
        let row = (cell / BOARD_SIZE) as i32;
        let col = (cell % BOARD_SIZE) as i32;
        for (dir_idx, &(dr, dc)) in DIRS.iter().enumerate() {
            let window = read_window(&board.black, &board.white, row, col, dr, dc);
            board.line_pattern_ids[cell][dir_idx] = lookup_mapped_id(pack_window(&window));
        }
    }

    board.zobrist = 0;
    for cell in 0..NUM_CELLS {
        if board.black.get(cell) {
            board.zobrist ^= zobrist_stone_key(Stone::Black, cell);
        }
        if board.white.get(cell) {
            board.zobrist ^= zobrist_stone_key(Stone::White, cell);
        }
    }
    if board.side_to_move == Stone::White {
        board.zobrist ^= ZOBRIST_SIDE;
    }
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

fn describe_signed(values: &[f64]) -> Value {
    json!({
        "mean": mean(values),
        "p01": quantile(values.to_vec(), 0.01),
        "p10": quantile(values.to_vec(), 0.10),
        "p50": quantile(values.to_vec(), 0.50),
        "p90": quantile(values.to_vec(), 0.90),
        "p99": quantile(values.to_vec(), 0.99),
        "min": values.iter().copied().fold(f64::INFINITY, f64::min),
        "max": values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        "abs": {
            "p50": abs_quantile(values, 0.50),
            "p90": abs_quantile(values, 0.90),
            "p99": abs_quantile(values, 0.99),
            "max": values.iter().map(|v| v.abs()).fold(0.0, f64::max),
        }
    })
}

fn worst_examples(rows: &[Row], limit: usize) -> Vec<Value> {
    let mut idxs: Vec<usize> = (0..rows.len()).collect();
    idxs.sort_by(|&a, &b| {
        codebook_or_flat_abs(&rows[b]).total_cmp(&codebook_or_flat_abs(&rows[a]))
    });
    idxs.into_iter()
        .take(limit)
        .map(|i| {
            let r = &rows[i];
            json!({
                "source": r.source,
                "game_id": r.game_id,
                "ply": r.ply,
                "flat_base": r.flat_base,
                "flat_base_swapped": r.flat_base_swapped,
                "flat_base_diff": r.flat_base - r.flat_base_swapped,
                "flat_runtime": r.flat_runtime,
                "flat_runtime_swapped": r.flat_runtime_swapped,
                "flat_runtime_diff": r.flat_runtime - r.flat_runtime_swapped,
                "codebook": r.codebook,
                "codebook_swapped": r.codebook_swapped,
                "codebook_diff": r.codebook - r.codebook_swapped,
                "stm_features_equal": r.stm_features_equal,
                "nstm_features_equal": r.nstm_features_equal,
                "feature_lens": {
                    "stm": r.stm_len,
                    "nstm": r.nstm_len,
                    "stm_swapped": r.stm_swapped_len,
                    "nstm_swapped": r.nstm_swapped_len,
                },
            })
        })
        .collect()
}

fn codebook_or_flat_abs(row: &Row) -> f64 {
    let code = (row.codebook - row.codebook_swapped).abs() as f64;
    let flat = (row.flat_base - row.flat_base_swapped).abs() as f64;
    code.max(flat)
}

fn quantile(mut values: Vec<f64>, q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let pos = ((values.len() - 1) as f64 * q).round() as usize;
    values[pos.min(values.len() - 1)]
}

fn abs_quantile(values: &[f64], q: f64) -> f64 {
    let abs_values: Vec<f64> = values.iter().map(|v| v.abs()).collect();
    quantile(abs_values, q)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn parse_side(raw: &str) -> Option<Stone> {
    match raw {
        "B" | "Black" | "black" => Some(Stone::Black),
        "W" | "White" | "white" => Some(Stone::White),
        _ => None,
    }
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        games_jsonl: Vec::new(),
        flat_model: PathBuf::new(),
        codebook_model: PathBuf::new(),
        out_json: PathBuf::from("rq558b_color_symmetry.json"),
        max_positions: 100,
        min_ply: 4,
    };
    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--games-jsonl" => args
                .games_jsonl
                .push(PathBuf::from(next_arg(&mut iter, "--games-jsonl")?)),
            "--flat-model" => args.flat_model = PathBuf::from(next_arg(&mut iter, "--flat-model")?),
            "--codebook-model" => {
                args.codebook_model = PathBuf::from(next_arg(&mut iter, "--codebook-model")?)
            }
            "--out-json" => args.out_json = PathBuf::from(next_arg(&mut iter, "--out-json")?),
            "--max-positions" => {
                args.max_positions = next_arg(&mut iter, "--max-positions")?
                    .parse()
                    .map_err(|_| "invalid --max-positions".to_string())?
            }
            "--min-ply" => {
                args.min_ply = next_arg(&mut iter, "--min-ply")?
                    .parse()
                    .map_err(|_| "invalid --min-ply".to_string())?
            }
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg {other}\n{}", usage())),
        }
    }
    if args.games_jsonl.is_empty() {
        return Err(format!("missing --games-jsonl\n{}", usage()));
    }
    if args.flat_model.as_os_str().is_empty() {
        return Err(format!("missing --flat-model\n{}", usage()));
    }
    if args.codebook_model.as_os_str().is_empty() {
        return Err(format!("missing --codebook-model\n{}", usage()));
    }
    if args.max_positions == 0 {
        return Err("--max-positions must be positive".to_string());
    }
    Ok(args)
}

fn next_arg(iter: &mut impl Iterator<Item = String>, name: &str) -> Result<String, String> {
    iter.next()
        .ok_or_else(|| format!("missing value for {name}"))
}

fn usage() -> &'static str {
    "usage: rq558b-color-symmetry --games-jsonl GAMES.jsonl [--games-jsonl MORE.jsonl] --flat-model MODEL.bin --codebook-model MODEL.json --out-json OUT.json [--max-positions N] [--min-ply PLY]"
}
