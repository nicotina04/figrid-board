use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::time::{Duration, Instant};

#[cfg(feature = "codebook-eval")]
use figrid_board::codebook_eval::{CodebookWeights, QuantizedCodebookWeights};
use figrid_board::{Board, GOMOKU_NNUE_CONFIG, SearchProfileSnapshot, Searcher, to_idx, to_rc};
use noru::network::NnueWeights;
use serde_json::{Value, json};

#[derive(Debug)]
struct Args {
    input: String,
    output: String,
    limit: usize,
    sample_every: usize,
    depth: u32,
    time_ms: Option<u64>,
    eval: String,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut input = None;
        let mut output = None;
        let mut limit = 64usize;
        let mut sample_every = 8usize;
        let mut depth = 20u32;
        let mut time_ms = None;
        let mut eval = "flat".to_string();

        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--input" => input = it.next(),
                "--output" => output = it.next(),
                "--limit" => {
                    limit = it
                        .next()
                        .ok_or("--limit requires a value")?
                        .parse()
                        .map_err(|e| format!("bad --limit: {e}"))?;
                }
                "--sample-every" => {
                    sample_every = it
                        .next()
                        .ok_or("--sample-every requires a value")?
                        .parse()
                        .map_err(|e| format!("bad --sample-every: {e}"))?;
                }
                "--depth" => {
                    depth = it
                        .next()
                        .ok_or("--depth requires a value")?
                        .parse()
                        .map_err(|e| format!("bad --depth: {e}"))?;
                }
                "--time-ms" => {
                    time_ms = Some(
                        it.next()
                            .ok_or("--time-ms requires a value")?
                            .parse()
                            .map_err(|e| format!("bad --time-ms: {e}"))?,
                    );
                }
                "--eval" => eval = it.next().ok_or("--eval requires a value")?,
                "--help" | "-h" => return Err(usage()),
                other => return Err(format!("unknown argument `{other}`\n{}", usage())),
            }
        }

        if sample_every == 0 {
            return Err("--sample-every must be > 0".to_string());
        }

        Ok(Self {
            input: input.ok_or_else(usage)?,
            output: output.unwrap_or_else(|| "-".to_string()),
            limit,
            sample_every,
            depth,
            time_ms,
            eval,
        })
    }
}

fn usage() -> String {
    "usage: rq582-search-profile --input games.jsonl [--output out.jsonl] \
     [--eval flat|codebook-quant] [--depth N] [--time-ms MS] [--limit N] \
     [--sample-every N]\n\
     Set NORU_SEARCH_PROFILE=1 to record profile buckets."
        .to_string()
}

fn load_weights_bytes() -> Result<Vec<u8>, String> {
    let path = std::env::var("FIGRID_WEIGHTS")
        .unwrap_or_else(|_| "models/gomoku_v52_5stone_conv_93k.bin".into());
    std::fs::read(&path).map_err(|e| format!("failed to read weights from `{path}`: {e}"))
}

#[cfg(feature = "codebook-eval")]
fn load_quantized_codebook() -> Result<QuantizedCodebookWeights, String> {
    let bytes = match std::env::var("FIGRID_CODEBOOK_WEIGHTS")
        .or_else(|_| std::env::var("NORU_CODEBOOK_EVAL_MODEL"))
        .ok()
        .map(|s| s.trim().to_string())
    {
        Some(path) if !path.is_empty() => {
            std::fs::read(&path).map_err(|e| format!("failed to read codebook `{path}`: {e}"))?
        }
        _ => include_bytes!("../models/gomoku_codebook_v1_swapclosed.json").to_vec(),
    };
    let weights = CodebookWeights::from_json_bytes(&bytes)
        .map_err(|e| format!("failed to parse codebook weights: {e}"))?;
    Ok(weights.quantize_i16_s32_s64())
}

fn profile_json(profile: SearchProfileSnapshot) -> Value {
    json!({
        "enabled": profile.enabled,
        "total_ns": profile.total_ns,
        "eval_ns": profile.eval_ns,
        "eval_calls": profile.eval_calls,
        "movegen_order_ns": profile.movegen_order_ns,
        "movegen_order_calls": profile.movegen_order_calls,
        "make_undo_ns": profile.make_undo_ns,
        "make_undo_calls": profile.make_undo_calls,
        "board_make_undo_ns": profile.board_make_undo_ns,
        "board_make_undo_calls": profile.board_make_undo_calls,
        "eval_state_push_pop_ns": profile.eval_state_push_pop_ns,
        "eval_state_push_pop_calls": profile.eval_state_push_pop_calls,
        "eval_state_dirty_list_ns": profile.eval_state_dirty_list_ns,
        "eval_state_dirty_list_calls": profile.eval_state_dirty_list_calls,
        "eval_state_frame_write_ns": profile.eval_state_frame_write_ns,
        "eval_state_frame_write_calls": profile.eval_state_frame_write_calls,
        "eval_state_backup_ns": profile.eval_state_backup_ns,
        "eval_state_backup_calls": profile.eval_state_backup_calls,
        "eval_state_recompute_ns": profile.eval_state_recompute_ns,
        "eval_state_recompute_calls": profile.eval_state_recompute_calls,
        "eval_state_aggregate_ns": profile.eval_state_aggregate_ns,
        "eval_state_aggregate_calls": profile.eval_state_aggregate_calls,
        "eval_state_restore_ns": profile.eval_state_restore_ns,
        "eval_state_restore_calls": profile.eval_state_restore_calls,
        "eval_state_forward_ns": profile.eval_state_forward_ns,
        "eval_state_forward_calls": profile.eval_state_forward_calls,
        "eval_state_push_calls": profile.eval_state_push_calls,
        "eval_state_pop_calls": profile.eval_state_pop_calls,
        "tt_ns": profile.tt_ns,
        "tt_calls": profile.tt_calls,
        "root_vct_ns": profile.root_vct_ns,
        "root_vct_calls": profile.root_vct_calls,
        "qsearch_ns": profile.qsearch_ns,
        "qsearch_calls": profile.qsearch_calls,
    })
}

fn main() -> Result<(), String> {
    let args = Args::parse()?;
    let flat_bytes = load_weights_bytes()?;
    let flat_weights = NnueWeights::load_from_bytes(&flat_bytes, Some(GOMOKU_NNUE_CONFIG))
        .map_err(|e| format!("failed to parse flat weights: {e}"))?;

    #[cfg(feature = "codebook-eval")]
    let codebook_weights = if args.eval == "codebook-quant" {
        Some(load_quantized_codebook()?)
    } else {
        None
    };
    #[cfg(not(feature = "codebook-eval"))]
    if args.eval == "codebook-quant" {
        return Err("codebook-quant requires --features codebook-eval".to_string());
    }

    let input = File::open(&args.input).map_err(|e| format!("failed to open input: {e}"))?;
    let mut out: Box<dyn Write> = if args.output == "-" {
        Box::new(std::io::stdout())
    } else {
        Box::new(BufWriter::new(
            File::create(&args.output).map_err(|e| format!("failed to create output: {e}"))?,
        ))
    };
    let time_limit = args.time_ms.map(Duration::from_millis);
    let mut emitted = 0usize;
    let mut seen_engine_positions = 0usize;

    for line in BufReader::new(input).lines() {
        let line = line.map_err(|e| format!("failed to read input line: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let game: Value =
            serde_json::from_str(&line).map_err(|e| format!("failed to parse JSONL game: {e}"))?;
        let game_id = game.get("game_id").cloned().unwrap_or(Value::Null);
        let seed = game.get("seed").cloned().unwrap_or(Value::Null);
        let moves = game
            .get("moves")
            .and_then(Value::as_array)
            .ok_or("game row missing moves array")?;
        let mut board = Board::new();
        for (ply, mv_json) in moves.iter().enumerate() {
            let x = mv_json
                .get("x")
                .and_then(Value::as_u64)
                .ok_or("move missing x")? as usize;
            let y = mv_json
                .get("y")
                .and_then(Value::as_u64)
                .ok_or("move missing y")? as usize;
            let source = mv_json
                .get("source")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            if source == "engine" {
                if seen_engine_positions % args.sample_every == 0 {
                    let mut search_board = board.clone();
                    let mut searcher = Searcher::new();
                    let started = Instant::now();
                    let result = match args.eval.as_str() {
                        "flat" => searcher.search(
                            &mut search_board,
                            &flat_weights,
                            args.depth,
                            time_limit,
                        ),
                        #[cfg(feature = "codebook-eval")]
                        "codebook-quant" => searcher.search_codebook_eval_quantized(
                            &mut search_board,
                            &flat_weights,
                            codebook_weights.as_ref().expect("codebook weights loaded"),
                            args.depth,
                            time_limit,
                        ),
                        other => return Err(format!("unknown eval arm `{other}`")),
                    };
                    let elapsed_ms = started.elapsed().as_millis();
                    let best_move = result.best_move.map(|m| {
                        let (row, col) = to_rc(m);
                        json!({"x": col, "y": row})
                    });
                    let row = json!({
                        "game_id": game_id,
                        "seed": seed,
                        "ply": ply,
                        "move_count": board.move_count,
                        "eval": args.eval,
                        "depth": args.depth,
                        "time_ms_limit": args.time_ms,
                        "elapsed_ms": elapsed_ms,
                        "best_move": best_move,
                        "score": result.score,
                        "searched_depth": result.depth,
                        "nodes": result.nodes,
                        "profile": profile_json(searcher.search_profile()),
                    });
                    writeln!(out, "{row}").map_err(|e| format!("failed to write output: {e}"))?;
                    emitted += 1;
                    if emitted >= args.limit {
                        return Ok(());
                    }
                }
                seen_engine_positions += 1;
            }
            let idx = to_idx(y, x);
            if !board.is_empty(idx) {
                return Err(format!(
                    "non-empty move at game={game_id:?} ply={ply} ({x},{y})"
                ));
            }
            board.make_move(idx);
        }
    }
    Ok(())
}
