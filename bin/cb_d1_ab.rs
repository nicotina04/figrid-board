//! Same-binary product-policy A/B harness for CB-D1 directional deltas.

use figrid_board::codebook_eval::{CodebookWeights, QuantizedCodebookWeights};
use figrid_board::{Board, GOMOKU_NNUE_CONFIG, RuleSet, Searcher, Stone, to_idx, to_rc};
use noru::network::NnueWeights;
use serde_json::{Value, json};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::time::{Duration, Instant};

const BASELINE_COMMIT: &str = "8d7d47d60ec37313a0c459ffdd18616bb9df4be0";
const CODEBOOK_SHA256: &str = "42968fdab01ba8ccd1de3ded05c532e4b237dd47eeffd7ae1c2f264d77ba7da2";

#[derive(Debug)]
struct Args {
    input: String,
    output: String,
    flat_weights: String,
    max_searches: usize,
    sample_every: usize,
    depth: u32,
    time_ms: Option<u64>,
    directional_delta: bool,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut input = None;
        let mut output = None;
        let mut flat_weights = "models/gomoku_v52_5stone_conv_93k.bin".to_string();
        let mut max_searches = usize::MAX;
        let mut sample_every = 1usize;
        let mut depth = 4u32;
        let mut time_ms = None;
        let mut directional_delta = false;
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--input" => input = it.next(),
                "--output" => output = it.next(),
                "--flat-weights" => {
                    flat_weights = it.next().ok_or("--flat-weights requires a path")?
                }
                "--max-searches" => max_searches = parse_next(&mut it, "--max-searches")?,
                "--sample-every" => sample_every = parse_next(&mut it, "--sample-every")?,
                "--depth" => depth = parse_next(&mut it, "--depth")?,
                "--time-ms" => time_ms = Some(parse_next(&mut it, "--time-ms")?),
                "--directional-delta" => directional_delta = true,
                "--help" | "-h" => return Err(usage()),
                other => return Err(format!("unknown argument `{other}`\n{}", usage())),
            }
        }
        if depth == 0 || sample_every == 0 {
            return Err("--depth and --sample-every must be > 0".to_string());
        }
        Ok(Self {
            input: input.ok_or_else(usage)?,
            output: output.ok_or_else(usage)?,
            flat_weights,
            max_searches,
            sample_every,
            depth,
            time_ms,
            directional_delta,
        })
    }
}

fn parse_next<T: std::str::FromStr>(
    it: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    it.next()
        .ok_or_else(|| format!("{flag} requires a value"))?
        .parse()
        .map_err(|error| format!("invalid {flag}: {error}"))
}

fn usage() -> String {
    "usage: cb-d1-ab --input games.jsonl --output run.jsonl \
     [--directional-delta] [--max-searches N] [--sample-every N] \
     [--depth N] [--time-ms MS] [--flat-weights path]"
        .to_string()
}

fn load_flat(path: &str) -> Result<NnueWeights, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("failed to read flat weights `{path}`: {error}"))?;
    NnueWeights::load_from_bytes(&bytes, Some(GOMOKU_NNUE_CONFIG))
        .map_err(|error| format!("failed to parse flat weights: {error}"))
}

fn load_codebook() -> Result<QuantizedCodebookWeights, String> {
    CodebookWeights::from_json_bytes(include_bytes!(
        "../models/gomoku_codebook_v1_swapclosed.json"
    ))
    .map(|weights| weights.quantize_i16_s32_s64())
    .map_err(|error| format!("failed to parse embedded codebook: {error}"))
}

fn product_searcher(directional_delta: bool) -> Result<Searcher, String> {
    let mut searcher = Searcher::new();
    searcher.set_use_threat_field(false);
    searcher.set_use_lazy_threat_field(false);
    searcher.set_use_move_picker(false);
    searcher.set_use_tail_threat_materialize(false);
    searcher.set_use_packed_line_windows(true);
    searcher.set_use_candidate_frontier(true);
    searcher.set_use_codebook_directional_delta(directional_delta);
    searcher.set_white_root_order_enabled(true)?;
    Ok(searcher)
}

fn is_figrid(name: &str) -> bool {
    name.to_ascii_lowercase().contains("figrid")
}

fn root_vct_enabled() -> bool {
    std::env::var("NORU_ROOT_VCT")
        .map(|raw| {
            let value = raw.trim();
            !(value == "0"
                || value.eq_ignore_ascii_case("false")
                || value.eq_ignore_ascii_case("off")
                || value.eq_ignore_ascii_case("no"))
        })
        .unwrap_or(true)
}

fn main() -> Result<(), String> {
    let args = Args::parse()?;
    let flat = load_flat(&args.flat_weights)?;
    let codebook = load_codebook()?;
    let input = File::open(&args.input)
        .map_err(|error| format!("failed to open input `{}`: {error}", args.input))?;
    let output = File::create(&args.output)
        .map_err(|error| format!("failed to create output `{}`: {error}", args.output))?;
    let mut output = BufWriter::new(output);
    let time_limit = args.time_ms.map(Duration::from_millis);
    let root_vct_enabled = root_vct_enabled();
    let mut searches = 0usize;
    let mut seen_product_positions = 0usize;

    writeln!(
        output,
        "{}",
        json!({
            "kind": "seal",
            "format": "cb-d1-ab-v2",
            "baseline_commit": BASELINE_COMMIT,
            "codebook_sha256": CODEBOOK_SHA256,
            "input": args.input,
            "flat_weights": args.flat_weights,
            "depth": args.depth,
            "time_ms": args.time_ms,
            "root_vct_enabled": root_vct_enabled,
            "rules": "15x15-freestyle",
            "eval": "embedded-quantized-codebook",
            "white_root_order": true,
            "packed_windows": true,
            "candidate_frontier": true,
            "directional_delta": args.directional_delta,
        })
    )
    .map_err(|error| format!("failed to write seal: {error}"))?;

    for line in BufReader::new(input).lines() {
        let line = line.map_err(|error| format!("failed to read input: {error}"))?;
        if line.trim().is_empty() {
            continue;
        }
        if searches >= args.max_searches {
            break;
        }
        let game: Value = serde_json::from_str(&line)
            .map_err(|error| format!("failed to parse game JSONL: {error}"))?;
        let game_id = game.get("game_id").cloned().unwrap_or(Value::Null);
        let seed = game.get("seed").cloned().unwrap_or(Value::Null);
        let black_engine = game
            .get("black_engine")
            .and_then(Value::as_str)
            .ok_or("game missing black_engine")?;
        let white_engine = game
            .get("white_engine")
            .and_then(Value::as_str)
            .ok_or("game missing white_engine")?;
        let product_side = match (is_figrid(black_engine), is_figrid(white_engine)) {
            (true, false) => Stone::Black,
            (false, true) => Stone::White,
            other => return Err(format!("expected one figrid side, got {other:?}")),
        };
        let moves = game
            .get("moves")
            .and_then(Value::as_array)
            .ok_or("game missing moves")?;
        let mut board = Board::new();
        board.set_rule_set(RuleSet::Freestyle);
        let mut searcher = product_searcher(args.directional_delta)?;

        for (ply, move_json) in moves.iter().enumerate() {
            if searches >= args.max_searches {
                break;
            }
            let x = move_json
                .get("x")
                .and_then(Value::as_u64)
                .ok_or("move missing x")? as usize;
            let y = move_json
                .get("y")
                .and_then(Value::as_u64)
                .ok_or("move missing y")? as usize;
            let source = move_json
                .get("source")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let actual_move = to_idx(y, x);

            if source == "engine" && board.side_to_move == product_side {
                if seen_product_positions % args.sample_every == 0 {
                    let mut search_board = board.clone();
                    let started = Instant::now();
                    let result = searcher.search_codebook_eval_quantized(
                        &mut search_board,
                        &flat,
                        &codebook,
                        args.depth,
                        time_limit,
                    );
                    let elapsed_ns = started.elapsed().as_nanos();
                    let shape = searcher.search_shape_stats();
                    let best_move = result.best_move.map(|mv| {
                        let (row, col) = to_rc(mv);
                        json!({"x": col, "y": row, "idx": mv})
                    });
                    writeln!(
                        output,
                        "{}",
                        json!({
                            "kind": "position",
                            "game_id": game_id,
                            "seed": seed,
                            "ply": ply,
                            "move_count": board.move_count,
                            "root_zobrist": board.zobrist,
                            "actual_move": {"x": x, "y": y, "idx": actual_move},
                            "best_move": best_move,
                            "score": result.score,
                            "searched_depth": result.depth,
                            "nodes": result.nodes,
                            "elapsed_ns": elapsed_ns,
                            "eval": "codebook-quant",
                            "requested_depth": args.depth,
                            "time_ms": args.time_ms,
                            "root_vct_enabled": root_vct_enabled,
                            "product_defaults": root_vct_enabled,
                            "directional_delta": args.directional_delta,
                            "shape": {
                                "main_nodes": shape.main_nodes,
                                "qsearch_nodes": shape.qsearch_nodes,
                                "tt_probes": shape.tt_probes,
                                "tt_hits": shape.tt_hits,
                                "tt_cutoffs": shape.tt_cutoffs,
                            },
                        })
                    )
                    .map_err(|error| format!("failed to write position: {error}"))?;
                    searches += 1;
                }
                seen_product_positions += 1;
            }
            if !board.is_empty(actual_move) {
                return Err(format!("occupied move game={game_id:?} ply={ply}"));
            }
            board.make_move(actual_move);
        }
    }

    output
        .flush()
        .map_err(|error| format!("failed to flush output: {error}"))
}
