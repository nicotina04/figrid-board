//! P4-R0: census of forced nodes in the main search.
//!
//! Runs fixed-depth searches over sampled positions from a sparring JSONL and
//! reports how many interior nodes are *forced* (the opponent already threatens
//! an immediate five) versus how many moves the search actually visits there.
//! Fixed depth, not fixed time, so instrumentation overhead cannot distort the
//! counts.

use std::fs::File;
use std::io::{BufRead, BufReader, Write};

use figrid_board::{Board, GOMOKU_NNUE_CONFIG, Searcher};
use noru::network::NnueWeights;
use serde_json::{Value, json};

struct Args {
    input: String,
    output: String,
    depth: u32,
    limit: usize,
    sample_every: usize,
    node_budget: Option<u64>,
    restrict: bool,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut input = None;
        let mut output = "-".to_string();
        let mut depth = 8u32;
        let mut limit = usize::MAX;
        let mut sample_every = 8usize;
        let mut node_budget = Some(4_000_000u64);
        let mut restrict = false;
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            let mut next = |what: &str| -> Result<String, String> {
                it.next().ok_or_else(|| format!("{what} requires a value"))
            };
            match arg.as_str() {
                "--input" => input = Some(next("--input")?),
                "--output" => output = next("--output")?,
                "--depth" => depth = next("--depth")?.parse().map_err(|e| format!("{e}"))?,
                "--limit" => limit = next("--limit")?.parse().map_err(|e| format!("{e}"))?,
                "--sample-every" => {
                    sample_every = next("--sample-every")?.parse().map_err(|e| format!("{e}"))?
                }
                "--restrict" => restrict = true,
                "--node-budget" => {
                    let v: u64 = next("--node-budget")?.parse().map_err(|e| format!("{e}"))?;
                    node_budget = if v == 0 { None } else { Some(v) };
                }
                other => return Err(format!("unknown argument `{other}`")),
            }
        }
        Ok(Args {
            input: input.ok_or("--input is required")?,
            output,
            depth,
            limit,
            sample_every: sample_every.max(1),
            node_budget,
            restrict,
        })
    }
}

fn main() -> Result<(), String> {
    let args = Args::parse()?;
    let path = std::env::var("FIGRID_WEIGHTS")
        .unwrap_or_else(|_| "models/gomoku_v52_5stone_conv_93k.bin".into());
    let bytes = std::fs::read(&path).map_err(|e| format!("failed to read `{path}`: {e}"))?;
    let weights = NnueWeights::load_from_bytes(&bytes, Some(GOMOKU_NNUE_CONFIG))
        .map_err(|e| format!("failed to parse weights: {e}"))?;

    let input = File::open(&args.input).map_err(|e| format!("failed to open input: {e}"))?;
    let mut total = figrid_board::p4r0_census::ForcedCensus::default();
    let mut positions = 0usize;
    let mut games = 0usize;
    let mut seen_engine = 0usize;
    let mut search_nodes = 0u64;
    let started = std::time::Instant::now();

    for line in BufReader::new(input).lines() {
        let line = line.map_err(|e| format!("failed to read line: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        if games >= args.limit {
            break;
        }
        games += 1;
        let game: Value = serde_json::from_str(&line).map_err(|e| format!("bad JSONL: {e}"))?;
        let moves = game
            .get("moves")
            .and_then(Value::as_array)
            .ok_or("game row missing moves array")?;
        let mut board = Board::new();
        for mv_json in moves {
            let x = mv_json.get("x").and_then(Value::as_u64).ok_or("move x")? as usize;
            let y = mv_json.get("y").and_then(Value::as_u64).ok_or("move y")? as usize;
            let source = mv_json
                .get("source")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            if source == "engine" {
                if seen_engine % args.sample_every == 0 {
                    let mut search_board = board.clone();
                    let mut searcher = Searcher::new();
                    searcher.set_node_limit(args.node_budget);
                    searcher.set_use_forced_restriction(args.restrict);
                    searcher.census.enabled = true;
                    let result = searcher.search(&mut search_board, &weights, args.depth, None);
                    search_nodes += result.nodes;
                    total.merge(&searcher.census);
                    positions += 1;
                    if positions % 25 == 0 {
                        eprintln!(
                            "  {positions} positions, {} interior nodes, {:.0}s",
                            total.interior_nodes,
                            started.elapsed().as_secs_f64()
                        );
                    }
                }
                seen_engine += 1;
            }
            board.make_move(y * 15 + x);
        }
    }

    let report = total.report();
    print!("{report}");
    println!(
        "\npositions {} | games {} | depth {} | elapsed {:.1}s",
        positions,
        games,
        args.depth,
        started.elapsed().as_secs_f64()
    );

    if args.output != "-" {
        let summary = json!({
            "restrict": args.restrict,
            "search_nodes": search_nodes,
            "positions": positions,
            "games": games,
            "depth": args.depth,
            "sample_every": args.sample_every,
            "node_budget": args.node_budget,
            "interior_nodes": total.interior_nodes,
            "forced_nodes": total.forced_nodes,
            "forced_moves_searched": total.forced_moves_searched,
            "forced_sound_moves": total.forced_sound_moves,
            "forced_own_five_nodes": total.forced_own_five_nodes,
            "forced_lost_nodes": total.forced_lost_nodes,
            "forced_cut": total.forced_cut,
            "forced_all": total.forced_all,
            "forced_exact": total.forced_exact,
            "forced_searched_hist": total.forced_searched_hist,
            "quiet_nodes": total.quiet_nodes,
            "quiet_moves_searched": total.quiet_moves_searched,
            "quiet_cut": total.quiet_cut,
            "quiet_all": total.quiet_all,
            "quiet_exact": total.quiet_exact,
            "report": report,
        });
        let mut f =
            File::create(&args.output).map_err(|e| format!("failed to create output: {e}"))?;
        writeln!(f, "{}", serde_json::to_string_pretty(&summary).unwrap())
            .map_err(|e| format!("failed to write output: {e}"))?;
    }
    Ok(())
}
