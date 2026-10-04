//! Gomocup (Piskvork) protocol adapter for the noru-tactic NNUE engine.
//!
//! Ships as the `pbrain-figrid` binary.

use std::io::{self, BufRead, Write};

#[cfg(all(feature = "board20", not(feature = "codebook-eval")))]
compile_error!("the 20x20 pbrain needs `codebook-eval` (the flat NNUE feature layout is 15x15-only)");
use std::sync::OnceLock;
use std::time::{Duration, Instant};

#[cfg(feature = "codebook-eval")]
use figrid_board::codebook_eval::{CodebookWeights, QuantizedCodebookWeights};
#[cfg(feature = "codebook-eval")]
use figrid_board::factored_codebook::{FactoredQuantizedCodebookWeights, PackedCodebookArtifact};
use figrid_board::{BOARD_SIZE, Board, GOMOKU_NNUE_CONFIG, RuleSet, Searcher, book, to_idx};
use noru::network::NnueWeights;

/// Source the v52 NNUE weights. Two modes:
///
/// 1. `embed-weights` cargo feature (Gomocup submission build) — gzip-compressed
///    weights are baked into the binary at compile time and decompressed once
///    on startup. Yields a single self-contained executable.
///
/// 2. Default (crates.io publish, dev builds) — weights are read from disk at
///    startup. Path resolution order:
///       a. `$FIGRID_WEIGHTS` env var if set
///       b. `./models/gomoku_v52_5stone_conv_93k.bin` relative to cwd
///       c. error out with a hint
#[cfg(feature = "embed-weights")]
fn load_weights_bytes() -> Result<Vec<u8>, String> {
    use flate2::read::GzDecoder;
    use std::io::Read;
    const COMPRESSED: &[u8] = include_bytes!("../models/gomoku_v52_5stone_conv_93k.bin.gz");
    let mut decoder = GzDecoder::new(COMPRESSED);
    let mut out = Vec::with_capacity(15_000_000);
    decoder
        .read_to_end(&mut out)
        .map_err(|e| format!("failed to decompress embedded weights: {e}"))?;
    Ok(out)
}

#[cfg(not(feature = "embed-weights"))]
fn load_weights_bytes() -> Result<Vec<u8>, String> {
    let path = std::env::var("FIGRID_WEIGHTS")
        .unwrap_or_else(|_| "models/gomoku_v52_5stone_conv_93k.bin".into());
    std::fs::read(&path).map_err(|e| {
        format!(
            "failed to read weights from `{path}`: {e}\n\
             hint: set $FIGRID_WEIGHTS or place the file at ./models/, \
             or rebuild with `--features embed-weights` for a self-contained binary"
        )
    })
}

const MAX_DEPTH: u32 = 20;
const DEFAULT_TIMEOUT_MS: i64 = 30_000;
const DEFAULT_MATCH_MS: i64 = 1_000_000_000;
/// Headroom subtracted from the turn budget so the last node batch finishes
/// well before Piskvork's deadline. Without this, the 128-node deadline
/// check can overshoot by ~50 ms on NNUE-heavy positions.
const SAFETY_MARGIN_MS: i64 = 150;
const TELEMETRY_WIN_SCORE: i32 = 999_000;

fn pbrain_max_depth() -> u32 {
    static VALUE: OnceLock<u32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("NORU_PBRAIN_MAX_DEPTH")
            .ok()
            .and_then(|raw| raw.trim().parse::<u32>().ok())
            .filter(|depth| *depth > 0)
            .unwrap_or(MAX_DEPTH)
    })
}

fn pbrain_fixed_depth() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("NORU_PBRAIN_FIXED_DEPTH")
            .map(|raw| env_bool_default(&raw, false))
            .unwrap_or(false)
    })
}

/// Engine variables a release pbrain accepts. Any other non-empty `NORU_*`
/// / `FIGRID_*` variable is a stale or misspelled switch and aborts startup
/// (fail-closed: a removed flag must never be silently ignored in a match).
const KNOWN_ENGINE_VARS: &[&str] = &[
    // Model files and codebook.
    "FIGRID_WEIGHTS",
    "FIGRID_CODEBOOK_WEIGHTS",
    "NORU_CODEBOOK_EVAL_SCALE",
    // Optional per-rule codebooks (each needs its own scale).
    "FIGRID_CODEBOOK_WEIGHTS_STANDARD",
    "FIGRID_CODEBOOK_WEIGHTS_CARO",
    "FIGRID_CODEBOOK_WEIGHTS_RENJU",
    "NORU_CODEBOOK_EVAL_SCALE_STANDARD",
    "NORU_CODEBOOK_EVAL_SCALE_CARO",
    "NORU_CODEBOOK_EVAL_SCALE_RENJU",
    "NORU_CODEBOOK_FACTORED",
    "NORU_CODEBOOK_DIRECTIONAL_DELTA",
    "FIGRID_WHITE_ROOT_ORDER",
    // Search.
    "NORU_PACKED_LINE_WINDOWS",
    "NORU_CANDIDATE_FRONTIER",
    "NORU_POLICY_ORDER",
    "NORU_POLICY_REDUCE",
    "NORU_FORCED_REPLY_RESTRICTION",
    "NORU_PBRAIN_FIXED_DEPTH",
    "NORU_PBRAIN_MAX_DEPTH",
    // Tooling (profiling, tests, benches).
    "NORU_SEARCH_PROFILE",
    "NORU_TEST_WEIGHTS",
    "FIGRID_BENCH_WEIGHTS",
];
const KNOWN_ENGINE_VAR_PREFIXES: &[&str] = &["FIGRID_VCT_"];

/// Names of non-empty `NORU_*` / `FIGRID_*` variables outside the known
/// list. Names are compared ASCII-case-insensitively (Windows env lookups
/// are case-insensitive); empty values are ignored so wrappers can blank
/// old variables.
fn unknown_engine_vars() -> Vec<String> {
    unknown_engine_var_names(
        std::env::vars_os()
            .filter(|(_, value)| !value.is_empty())
            .map(|(name, _)| name.to_string_lossy().into_owned()),
    )
}

fn unknown_engine_var_names(names: impl Iterator<Item = String>) -> Vec<String> {
    let mut unknown: Vec<String> = names
        .filter(|name| {
            let upper = name.to_ascii_uppercase();
            if !(upper.starts_with("NORU_") || upper.starts_with("FIGRID_")) {
                return false;
            }
            let known = KNOWN_ENGINE_VARS.contains(&upper.as_str())
                || KNOWN_ENGINE_VAR_PREFIXES
                    .iter()
                    .any(|prefix| upper.starts_with(prefix));
            !known
        })
        .collect();
    unknown.sort();
    unknown
}

#[cfg(test)]
mod engine_var_tests {
    use super::*;

    #[test]
    fn unknown_engine_vars_fail_closed_on_removed_switches() {
        let names = [
            "NORU_ROOT_VCT",
            "noru_policy_order",
            "FIGRID_CODEBOOK_EVAL",
            "FIGRID_VCT_PROFILE",
            "FIGRID_CODEBOOK_WEIGHTS",
            "PATH",
            "NORU_CANDIDATE_RANKER",
            "NORU_CANDIDATE_FRONTIER",
        ]
        .map(String::from);
        assert_eq!(
            unknown_engine_var_names(names.into_iter()),
            vec!["FIGRID_CODEBOOK_EVAL", "NORU_CANDIDATE_RANKER", "NORU_ROOT_VCT"]
        );
    }

    #[test]
    fn empty_boolean_values_mean_default() {
        assert!(env_bool_default("", true));
        assert!(!env_bool_default("  ", false));
        assert!(env_bool_default("on", false));
        assert!(!env_bool_default("off", true));
    }
}

#[cfg(feature = "embed-weights")]
fn weights_label() -> String {
    "embedded".to_string()
}

#[cfg(not(feature = "embed-weights"))]
fn weights_label() -> String {
    std::env::var("FIGRID_WEIGHTS")
        .unwrap_or_else(|_| "models/gomoku_v52_5stone_conv_93k.bin".into())
}

#[cfg(all(feature = "codebook-eval", feature = "cb-f1-flat-asset"))]
const EMBEDDED_CODEBOOK_CBF: &[u8] =
    include_bytes!("../models/gomoku_codebook_v1_swapclosed_compact_flat.cbf");

#[cfg(all(feature = "codebook-eval", not(feature = "cb-f1-flat-asset")))]
const EMBEDDED_CODEBOOK_CBF: &[u8] =
    include_bytes!("../models/gomoku_codebook_v1_swapclosed_factored.cbf");

/// The codebook always runs through the quantized i16/s32/s64 kernel; the
/// float kernel remains available to library callers only.
#[cfg(feature = "codebook-eval")]
enum CodebookRuntimeWeights {
    Quantized {
        weights: QuantizedCodebookWeights,
        embedded: bool,
    },
    FactoredQuantized {
        weights: FactoredQuantizedCodebookWeights,
        embedded: bool,
    },
}

#[cfg(feature = "codebook-eval")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WhiteRootOrderMode {
    Auto,
    On,
    Off,
}

fn env_bool_default(raw: &str, default: bool) -> bool {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return default;
    }
    !(trimmed == "0"
        || trimmed.eq_ignore_ascii_case("false")
        || trimmed.eq_ignore_ascii_case("off")
        || trimmed.eq_ignore_ascii_case("no"))
}

/// Packed Pattern4 windows are the 0.8.2 product default. Setting
/// `NORU_PACKED_LINE_WINDOWS=off` restores the 0.8.1 updater.
fn packed_line_windows_enabled() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("NORU_PACKED_LINE_WINDOWS")
            .map(|raw| env_bool_default(&raw, true))
            .unwrap_or(true)
    })
}

/// Exact-order candidate-frontier maintenance is the 0.8.2 product default.
/// Setting `NORU_CANDIDATE_FRONTIER=off` restores the A2-only path.
fn candidate_frontier_enabled() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("NORU_CANDIDATE_FRONTIER")
            .map(|raw| env_bool_default(&raw, true))
            .unwrap_or(true)
    })
}

/// CB-D1 exact directional-delta evaluator is promoted for the pbrain.
/// Set `NORU_CODEBOOK_DIRECTIONAL_DELTA=off` for immediate rollback.
#[cfg(feature = "codebook-eval")]
fn codebook_directional_delta_enabled() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("NORU_CODEBOOK_DIRECTIONAL_DELTA")
            .map(|raw| env_bool_default(&raw, true))
            .unwrap_or(true)
    })
}

/// CB-F1 is an opt-in prototype. The embedded artifact is compact in either
/// mode, but the runtime keeps the established flat quantized representation
/// unless this selector is explicitly enabled.
#[cfg(feature = "codebook-eval")]
fn codebook_factored_enabled() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("NORU_CODEBOOK_FACTORED")
            .map(|raw| env_bool_default(&raw, false))
            .unwrap_or(false)
    })
}

#[cfg(not(feature = "codebook-eval"))]
fn reject_explicit_white_root_order_without_codebook() -> Result<(), String> {
    let Some(raw) = std::env::var_os("FIGRID_WHITE_ROOT_ORDER") else {
        return Ok(());
    };
    let value = raw
        .to_str()
        .ok_or_else(|| "FIGRID_WHITE_ROOT_ORDER is not valid Unicode".to_string())?
        .trim();
    if value.is_empty()
        || value.eq_ignore_ascii_case("auto")
        || value == "0"
        || value.eq_ignore_ascii_case("false")
        || value.eq_ignore_ascii_case("off")
        || value.eq_ignore_ascii_case("no")
    {
        return Ok(());
    }
    if value == "1"
        || value.eq_ignore_ascii_case("true")
        || value.eq_ignore_ascii_case("on")
        || value.eq_ignore_ascii_case("yes")
    {
        return Err(
            "FIGRID_WHITE_ROOT_ORDER=on requires a build with the codebook-eval feature"
                .to_string(),
        );
    }
    Err(format!(
        "invalid FIGRID_WHITE_ROOT_ORDER value `{value}`; expected auto, on, or off"
    ))
}

#[cfg(feature = "codebook-eval")]
fn parse_white_root_order_mode(raw: Option<&str>) -> Result<WhiteRootOrderMode, String> {
    let Some(raw) = raw else {
        return Ok(WhiteRootOrderMode::Auto);
    };
    let value = raw.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("auto") {
        Ok(WhiteRootOrderMode::Auto)
    } else if value == "1"
        || value.eq_ignore_ascii_case("true")
        || value.eq_ignore_ascii_case("on")
        || value.eq_ignore_ascii_case("yes")
    {
        Ok(WhiteRootOrderMode::On)
    } else if value == "0"
        || value.eq_ignore_ascii_case("false")
        || value.eq_ignore_ascii_case("off")
        || value.eq_ignore_ascii_case("no")
    {
        Ok(WhiteRootOrderMode::Off)
    } else {
        Err(format!(
            "invalid FIGRID_WHITE_ROOT_ORDER value `{raw}`; expected auto, on, or off"
        ))
    }
}

#[cfg(feature = "codebook-eval")]
fn configure_white_root_order(
    searcher: &mut Searcher,
    codebook: &Option<CodebookRuntimeWeights>,
) -> Result<(), String> {
    let configured = std::env::var_os("FIGRID_WHITE_ROOT_ORDER");
    let configured = configured
        .as_deref()
        .map(|value| {
            value
                .to_str()
                .ok_or_else(|| "FIGRID_WHITE_ROOT_ORDER is not valid Unicode".to_string())
        })
        .transpose()?;
    let mode = parse_white_root_order_mode(configured)?;
    configure_white_root_order_mode(searcher, codebook, mode)
}

#[cfg(feature = "codebook-eval")]
fn configure_white_root_order_mode(
    searcher: &mut Searcher,
    codebook: &Option<CodebookRuntimeWeights>,
    mode: WhiteRootOrderMode,
) -> Result<(), String> {
    let supported = matches!(
        codebook,
        Some(CodebookRuntimeWeights::Quantized { embedded: true, .. })
            | Some(CodebookRuntimeWeights::FactoredQuantized { embedded: true, .. })
    );
    match mode {
        WhiteRootOrderMode::Auto | WhiteRootOrderMode::On if supported => {
            searcher.set_white_root_order_enabled(true)
        }
        WhiteRootOrderMode::Auto | WhiteRootOrderMode::Off => {
            searcher.set_white_root_order_enabled(false)
        }
        WhiteRootOrderMode::On => Err(
            "FIGRID_WHITE_ROOT_ORDER=on requires the embedded quantized codebook evaluator"
                .to_string(),
        ),
    }
}

#[cfg(all(test, feature = "codebook-eval"))]
mod white_root_order_tests {
    use super::*;

    fn embedded_quantized() -> Option<CodebookRuntimeWeights> {
        Some(CodebookRuntimeWeights::Quantized {
            weights: CodebookWeights::deterministic(16, 8).quantize_i16_s32_s64(),
            embedded: true,
        })
    }

    #[test]
    fn mode_parser_accepts_public_values_and_rejects_unknown_values() {
        assert_eq!(
            parse_white_root_order_mode(None).unwrap(),
            WhiteRootOrderMode::Auto
        );
        assert_eq!(
            parse_white_root_order_mode(Some("auto")).unwrap(),
            WhiteRootOrderMode::Auto
        );
        assert_eq!(
            parse_white_root_order_mode(Some("ON")).unwrap(),
            WhiteRootOrderMode::On
        );
        assert_eq!(
            parse_white_root_order_mode(Some("0")).unwrap(),
            WhiteRootOrderMode::Off
        );
        assert!(parse_white_root_order_mode(Some("maybe")).is_err());
    }

    #[test]
    fn auto_enables_only_the_embedded_quantized_path() {
        let embedded = embedded_quantized();
        let mut searcher = Searcher::new();
        configure_white_root_order_mode(&mut searcher, &embedded, WhiteRootOrderMode::Auto)
            .unwrap();
        assert!(searcher.white_root_order_enabled());

        let custom = match embedded_quantized().unwrap() {
            CodebookRuntimeWeights::Quantized { weights, .. } => {
                Some(CodebookRuntimeWeights::Quantized {
                    weights,
                    embedded: false,
                })
            }
            CodebookRuntimeWeights::FactoredQuantized { .. } => unreachable!(),
        };
        let mut custom_searcher = Searcher::new();
        configure_white_root_order_mode(&mut custom_searcher, &custom, WhiteRootOrderMode::Auto)
            .unwrap();
        assert!(!custom_searcher.white_root_order_enabled());
        assert!(
            configure_white_root_order_mode(&mut custom_searcher, &custom, WhiteRootOrderMode::On,)
                .is_err()
        );
    }

    #[cfg(not(feature = "cb-f1-flat-asset"))]
    #[test]
    fn auto_and_on_support_the_embedded_factored_path() {
        let factored = Some(load_embedded_codebook_weights(EMBEDDED_CODEBOOK_CBF, true).unwrap());
        assert!(matches!(
            factored,
            Some(CodebookRuntimeWeights::FactoredQuantized { embedded: true, .. })
        ));

        for mode in [WhiteRootOrderMode::Auto, WhiteRootOrderMode::On] {
            let mut searcher = Searcher::new();
            configure_white_root_order_mode(&mut searcher, &factored, mode).unwrap();
            assert!(searcher.white_root_order_enabled());
        }
    }
}

#[cfg(feature = "codebook-eval")]
fn load_embedded_codebook_weights(
    bytes: &[u8],
    factored: bool,
) -> Result<CodebookRuntimeWeights, String> {
    let artifact = PackedCodebookArtifact::parse(bytes)
        .map_err(|e| format!("failed to parse embedded packed codebook: {e}"))?;
    if factored {
        let weights = artifact.into_factored_quantized().map_err(|e| {
            format!("NORU_CODEBOOK_FACTORED=on requires a factored embedded codebook: {e}")
        })?;
        Ok(CodebookRuntimeWeights::FactoredQuantized {
            weights,
            embedded: true,
        })
    } else {
        // Keep the OFF arm identical to the established product kernel: the
        // exact source floats are quantized into the existing flat table.
        let weights = artifact.into_source_weights().quantize_i16_s32_s64();
        Ok(CodebookRuntimeWeights::Quantized {
            weights,
            embedded: true,
        })
    }
}

#[cfg(feature = "codebook-eval")]
/// `FIGRID_CODEBOOK_WEIGHTS` is the single codebook loader knob: unset = the
/// embedded codebook, a path = that `.ngcb`/`.json` file, and
/// `""`/`0`/`off`/`false`/`no` = no codebook (flat NNUE eval).
fn load_codebook_weights() -> Result<Option<CodebookRuntimeWeights>, String> {
    let configured_path = match std::env::var_os("FIGRID_CODEBOOK_WEIGHTS") {
        Some(path) => Some(
            path.into_string()
                .map_err(|_| "FIGRID_CODEBOOK_WEIGHTS is not valid Unicode".to_string())?,
        ),
        None => None,
    };
    let Some(configured_path) = configured_path else {
        return load_embedded_codebook_weights(EMBEDDED_CODEBOOK_CBF, codebook_factored_enabled())
            .map(Some);
    };
    let path = configured_path.trim();
    if path.is_empty()
        || path == "0"
        || path.eq_ignore_ascii_case("off")
        || path.eq_ignore_ascii_case("false")
        || path.eq_ignore_ascii_case("no")
    {
        return Ok(None);
    }

    load_codebook_file(path).map(Some)
}

/// External model paths are JSON or NGCB1 binaries (detected by magic), in either the legacy or the
/// full-vocabulary id space. They never inherit the embedded CB-F1 representation selector.
#[cfg(feature = "codebook-eval")]
fn load_codebook_file(path: &str) -> Result<CodebookRuntimeWeights, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("failed to read codebook weights from `{path}`: {e}"))?;
    let weights = CodebookWeights::from_bytes_auto(&bytes)
        .map_err(|e| format!("failed to parse codebook weights: {e}"))?;
    if weights.is_full_vocab() {
        // Build the full-vocabulary lookup tables now so their one-off cost
        // never lands inside the first move's clock.
        figrid_board::pattern_table::warm_full_vocab();
    }
    Ok(CodebookRuntimeWeights::Quantized {
        weights: weights.quantize_i16_s32_s64(),
        embedded: false,
    })
}

/// A codebook used only under one rule, with the eval scale it was calibrated for.
#[cfg(feature = "codebook-eval")]
struct RuleCodebook {
    rule: RuleSet,
    weights: CodebookRuntimeWeights,
    scale: f32,
    label: String,
}

/// Optional per-rule codebooks: `FIGRID_CODEBOOK_WEIGHTS_<RULE>` with a mandatory
/// `NORU_CODEBOOK_EVAL_SCALE_<RULE>` (a model without its scale fails closed).
#[cfg(feature = "codebook-eval")]
fn load_rule_codebooks() -> Result<Vec<RuleCodebook>, String> {
    let mut out = Vec::new();
    for (rule, tag) in [(RuleSet::Standard, "STANDARD"), (RuleSet::Caro, "CARO"), (RuleSet::Renju, "RENJU")] {
        let path = std::env::var(format!("FIGRID_CODEBOOK_WEIGHTS_{tag}")).unwrap_or_default();
        let path = path.trim();
        if path.is_empty() {
            continue;
        }
        let raw = std::env::var(format!("NORU_CODEBOOK_EVAL_SCALE_{tag}"))
            .map_err(|_| format!("FIGRID_CODEBOOK_WEIGHTS_{tag} needs NORU_CODEBOOK_EVAL_SCALE_{tag}"))?;
        let scale = raw
            .trim()
            .parse::<f32>()
            .ok()
            .filter(|s| s.is_finite() && *s > 0.0)
            .ok_or_else(|| format!("invalid NORU_CODEBOOK_EVAL_SCALE_{tag}: {raw}"))?;
        out.push(RuleCodebook { rule, weights: load_codebook_file(path)?, scale, label: format!("{tag}={path}") });
    }
    Ok(out)
}

/// `codebook=...; ...` part of the startup config line.
#[cfg(feature = "codebook-eval")]
fn codebook_config_summary(codebook: &Option<CodebookRuntimeWeights>, searcher: &Searcher) -> String {
    let on_off = |enabled: bool| if enabled { "on" } else { "off" };
    let (source, kernel, ids) = match codebook {
        None => return "codebook=off".to_string(),
        Some(CodebookRuntimeWeights::Quantized { weights, embedded }) => (
            if *embedded {
                "embedded".to_string()
            } else {
                std::env::var("FIGRID_CODEBOOK_WEIGHTS")
                    .map(|path| path.trim().to_string())
                    .unwrap_or_default()
            },
            "quantized-flat",
            weights.num_ids(),
        ),
        Some(CodebookRuntimeWeights::FactoredQuantized { weights, .. }) => (
            "embedded".to_string(),
            "quantized-factored",
            weights.token_count(),
        ),
    };
    format!(
        "codebook={source}; codebook_kernel={kernel}; codebook_ids={ids}; scale={}; \
         directional_delta={}; white_root_order={}",
        figrid_board::search::codebook_eval_scale(),
        on_off(codebook_directional_delta_enabled()),
        on_off(searcher.white_root_order_enabled()),
    )
}

#[cfg(all(test, feature = "codebook-eval"))]
mod codebook_loader_tests {
    use super::*;

    #[test]
    fn embedded_factored_off_uses_the_existing_flat_quantizer() {
        let runtime = load_embedded_codebook_weights(EMBEDDED_CODEBOOK_CBF, false).unwrap();
        assert!(matches!(
            runtime,
            CodebookRuntimeWeights::Quantized { embedded: true, .. }
        ));
    }

    #[cfg(not(feature = "cb-f1-flat-asset"))]
    #[test]
    fn embedded_factored_on_keeps_only_the_factored_runtime() {
        let runtime = load_embedded_codebook_weights(EMBEDDED_CODEBOOK_CBF, true).unwrap();
        assert!(matches!(
            runtime,
            CodebookRuntimeWeights::FactoredQuantized { embedded: true, .. }
        ));
    }

    #[cfg(feature = "cb-f1-flat-asset")]
    #[test]
    fn flat_counterfactual_fails_closed_when_factored_is_requested() {
        let error = match load_embedded_codebook_weights(EMBEDDED_CODEBOOK_CBF, true) {
            Ok(_) => panic!("flat counterfactual must reject the factored selector"),
            Err(error) => error,
        };
        assert!(error.contains("requires a factored embedded codebook"));
    }
}

fn telemetry_score(score: i32) -> String {
    if score.abs() >= TELEMETRY_WIN_SCORE - 1_000 {
        let mate = (TELEMETRY_WIN_SCORE - score.abs()).max(1);
        if score >= 0 {
            format!("+M{mate}")
        } else {
            format!("-M{mate}")
        }
    } else {
        score.to_string()
    }
}

fn telemetry_count(value: u64) -> String {
    if value >= 1_000_000_000 {
        format!("{}G", value / 1_000_000_000)
    } else if value >= 1_000_000 {
        format!("{}M", value / 1_000_000)
    } else if value >= 1_000 {
        format!("{}K", value / 1_000)
    } else {
        value.to_string()
    }
}

fn emit_search_message(result: &figrid_board::SearchResult, elapsed: Duration) {
    let time_ms = elapsed.as_millis().max(1) as u64;
    let nps = result.nodes.saturating_mul(1_000) / time_ms;
    let depth = if result.depth == 0 {
        "0-0".to_string()
    } else {
        format!("{}-{}", result.depth, result.depth)
    };
    println!(
        "MESSAGE Speed {} | Depth {} | Eval {} | Node {} | Time {}ms",
        telemetry_count(nps),
        depth,
        telemetry_score(result.score),
        telemetry_count(result.nodes),
        time_ms
    );
}

struct ProtocolInfo {
    timeout_turn: i64,
    timeout_match: i64,
    time_left: i64,
    rule_exact5: bool,
    rule_continuous: bool,
    rule_renju: bool,
    rule_caro: bool,
}

impl ProtocolInfo {
    fn new() -> Self {
        Self {
            timeout_turn: DEFAULT_TIMEOUT_MS,
            timeout_match: DEFAULT_MATCH_MS,
            time_left: DEFAULT_MATCH_MS,
            rule_exact5: false,
            rule_continuous: false,
            rule_renju: false,
            rule_caro: false,
        }
    }

    fn rule_set(&self) -> Option<RuleSet> {
        // Supported now: Freestyle (0), Standard exact-5 (1), Renju (4: black
        // exact five with forbidden double-four / double-three / overline,
        // white five or more) and Caro as Gomocup plays it (9 = caro|exact5:
        // exactly five, not blocked at both ends by stones). Bare rule 8
        // (overline-wins Caro) and continuous games are not modeled.
        if self.rule_continuous || (self.rule_renju && self.rule_caro) {
            return None;
        }
        if self.rule_renju {
            return Some(RuleSet::Renju);
        }
        if self.rule_caro {
            if self.rule_exact5 {
                Some(RuleSet::Caro)
            } else {
                None
            }
        } else if self.rule_exact5 {
            Some(RuleSet::Standard)
        } else {
            Some(RuleSet::Freestyle)
        }
    }

    fn rule_supported(&self) -> bool {
        self.rule_set().is_some()
    }

    fn turn_budget(&self, move_count: usize) -> Duration {
        // No match budget announced (Piskvork running with `time_match=0`,
        // arena scripts that set only `timeout_turn`, etc.) — fall back to
        // the per-move cap. Anything below half of the sentinel default
        // means the controller actually told us a real budget.
        if self.timeout_match >= DEFAULT_MATCH_MS / 2 {
            return Duration::from_millis((self.timeout_turn - SAFETY_MARGIN_MS).max(50) as u64);
        }

        let time_left = self.time_left.max(0);
        if time_left <= 0 {
            // Out of time — respond instantly. Loses on time eventually but
            // never overshoots the controller's deadline.
            return Duration::from_millis(50);
        }

        // Real Gomocup games end well before the 225-cell board fills up;
        // 35 moves per side is a calibration-friendly midpoint between the
        // shortest decisive games (~25) and long late-mate fights (~50).
        // Old code divided the *whole match budget* by `15*15/2 = 112`, an
        // estimate that left ~70% of the time unused at game end.
        const EXPECTED_PER_SIDE: i64 = 35;
        let played_this_side = (move_count as i64 + 1) / 2;
        let remaining_this_side = (EXPECTED_PER_SIDE - played_this_side).max(5);
        let equal_share = time_left / remaining_this_side;

        // Phase-based multiplier. Spending more time in the tactical
        // midgame and less in the random opening / forced endgame matches
        // standard chess-engine practice and the diagnosis of figrid
        // losing in plies 8-25 (Phase A.1 white-loss analysis).
        // Multipliers stored in basis points / 100 to keep the math in i64.
        let phase_mul: i64 = match move_count {
            0..=5 => 30,    // opening — most engines waste time here
            6..=11 => 80,   // early — getting into tactics
            12..=24 => 150, // tactical peak — boost
            25..=34 => 100, // late midgame — equal share
            _ => 60,        // endgame — often forced
        };
        let phase_budget = (equal_share * phase_mul) / 100;

        // Hard caps:
        //   * `timeout_turn` is the controller-imposed per-move ceiling.
        //   * `time_left / 3` keeps a single move from blowing the budget;
        //     even the deepest tactical search rarely needs more than a
        //     third of remaining time.
        //   * 100 ms floor so abort logic still has a chance to fire.
        let safe_max = time_left / 3;
        let budget = phase_budget.min(self.timeout_turn).min(safe_max).max(100);

        Duration::from_millis((budget - SAFETY_MARGIN_MS).max(50) as u64)
    }

    fn update(&mut self, key: &str, val: &str) {
        let val = val.trim();
        match key {
            "timeout_turn" => {
                if let Ok(v) = val.parse() {
                    self.timeout_turn = v;
                }
            }
            "timeout_match" => {
                if let Ok(v) = val.parse() {
                    self.timeout_match = v;
                }
            }
            "time_left" => {
                if let Ok(v) = val.parse() {
                    self.time_left = v;
                }
            }
            "rule" => {
                if let Ok(b) = val.parse::<u8>() {
                    self.rule_exact5 = (b & 1) != 0;
                    self.rule_continuous = (b & 2) != 0;
                    self.rule_renju = (b & 4) != 0;
                    self.rule_caro = (b & 8) != 0;
                }
            }
            _ => {}
        }
    }
}

struct Engine {
    board: Board,
    weights: NnueWeights,
    #[cfg(feature = "codebook-eval")]
    codebook_weights: Option<CodebookRuntimeWeights>,
    #[cfg(feature = "codebook-eval")]
    rule_codebooks: Vec<RuleCodebook>,
    /// One-line effective configuration, announced before the first START
    /// reply.
    config_line: String,
    searcher: Searcher,
    info: ProtocolInfo,
    started: bool,
}

impl Engine {
    fn new() -> Result<Self, String> {
        #[cfg(not(feature = "codebook-eval"))]
        reject_explicit_white_root_order_without_codebook()?;
        let bytes = load_weights_bytes()?;
        let weights = NnueWeights::load_from_bytes(&bytes, Some(GOMOKU_NNUE_CONFIG))
            .map_err(|e| format!("failed to parse weights: {e}"))?;
        #[cfg(feature = "codebook-eval")]
        let codebook_weights = load_codebook_weights()?;
        // The flat NNUE feature layout is 15x15-only; the 20x20 engine always evaluates with a codebook.
        #[cfg(all(feature = "codebook-eval", feature = "board20"))]
        if codebook_weights.is_none() {
            return Err("the 20x20 build needs a codebook (FIGRID_CODEBOOK_WEIGHTS must not be off)".to_string());
        }
        #[cfg(feature = "codebook-eval")]
        let rule_codebooks = load_rule_codebooks()?;
        #[allow(unused_mut)]
        let mut searcher = Searcher::new();
        #[cfg(feature = "codebook-eval")]
        configure_white_root_order(&mut searcher, &codebook_weights)?;
        #[cfg(feature = "codebook-eval")]
        searcher.set_use_codebook_directional_delta(codebook_directional_delta_enabled());
        searcher.set_use_candidate_frontier(candidate_frontier_enabled());
        searcher.set_use_packed_line_windows(packed_line_windows_enabled());
        // Resolving the search switches here reports a bad NORU_POLICY_ORDER
        // (or NORU_POLICY_REDUCE without a table) at startup instead of at the
        // first search.
        let search_config = figrid_board::search::runtime_config_summary()?;
        #[cfg(feature = "codebook-eval")]
        let codebook_config = {
            let base = codebook_config_summary(&codebook_weights, &searcher);
            if rule_codebooks.is_empty() {
                base
            } else {
                let per_rule: Vec<String> =
                    rule_codebooks.iter().map(|r| format!("{} scale={}", r.label, r.scale)).collect();
                format!("{base}; rule_codebooks=[{}]", per_rule.join(", "))
            }
        };
        #[cfg(not(feature = "codebook-eval"))]
        let codebook_config = "codebook=unsupported".to_string();
        let on_off = |enabled: bool| if enabled { "on" } else { "off" };
        let config_line = format!(
            "version={}; weights={}; {codebook_config}; packed_line_windows={}; \
             candidate_frontier={}; {search_config}; pbrain_fixed_depth={}; pbrain_max_depth={}",
            env!("CARGO_PKG_VERSION"),
            weights_label(),
            on_off(packed_line_windows_enabled()),
            on_off(candidate_frontier_enabled()),
            on_off(pbrain_fixed_depth()),
            pbrain_max_depth(),
        );
        let board = Board::new();
        Ok(Self {
            board,
            weights,
            #[cfg(feature = "codebook-eval")]
            codebook_weights,
            #[cfg(feature = "codebook-eval")]
            rule_codebooks,
            config_line,
            searcher,
            info: ProtocolInfo::new(),
            started: false,
        })
    }

    fn reset_board(&mut self) {
        self.board = Board::new();
    }

    fn apply_opp_move(&mut self, x: u8, y: u8) -> Result<(), String> {
        let idx = xy_to_idx(x, y)?;
        if !self.board.is_empty(idx) {
            return Err(format!("cell ({x},{y}) already occupied"));
        }
        self.board.make_move(idx);
        Ok(())
    }

    fn no_move_error(&self) -> &'static str {
        if self.info.rule_supported() {
            "ERROR - no legal move"
        } else {
            "ERROR - unsupported rule"
        }
    }

    fn choose_move(&mut self) -> Option<(u8, u8)> {
        if !self.info.rule_supported() {
            return None;
        }
        // Sync the win rule into the board on every move. Gomocup sends
        // `START` *before* `INFO rule 1`, and `BOARD` calls `reset_board()`
        // (which clears `exact5` back to false), so setting this only in the
        // START handler leaves Standard games running in Freestyle mode —
        // the engine would score an overline as a win and fail the
        // `standard_specific` rule probe. Re-applying it here, right before
        // search, covers every command path regardless of ordering.
        self.board
            .set_rule_set(self.info.rule_set().unwrap_or(RuleSet::Freestyle));

        // Opening book hook is intentionally disabled — the Rapfi-distilled
        // book regressed 24 g vs Pela at Gomocup TC (3/24 with book vs 4/24
        // without; book-hit side fell to 1/12 = 8.3 % alone). The numbers
        // are within 24 g noise (σ ≈ 10 pp) but the per-side breakdown is
        // consistent with the well-known capacity-gap problem: the teacher's
        // best move leads to lines our search can't follow up. The book
        // tables stay in `crate::book` for a future self-play book attempt.
        let _ = book::lookup; // keep the symbol live for the next try.

        let max_depth = pbrain_max_depth();
        let time_limit = if pbrain_fixed_depth() {
            None
        } else {
            Some(self.info.turn_budget(self.board.move_count))
        };
        let search_start = Instant::now();
        #[cfg(feature = "codebook-eval")]
        let rule = self.board.effective_rule_set();
        #[cfg(feature = "codebook-eval")]
        let (codebook, scale) = match self.rule_codebooks.iter().find(|r| r.rule == rule) {
            Some(r) => (Some(&r.weights), Some(r.scale)),
            None => (self.codebook_weights.as_ref(), None),
        };
        #[cfg(feature = "codebook-eval")]
        self.searcher.set_codebook_eval_scale(scale);
        #[cfg(feature = "codebook-eval")]
        let result = match codebook {
            Some(CodebookRuntimeWeights::Quantized {
                weights: codebook_weights,
                ..
            }) => self.searcher.search_codebook_eval_quantized(
                &mut self.board,
                &self.weights,
                codebook_weights,
                max_depth,
                time_limit,
            ),
            Some(CodebookRuntimeWeights::FactoredQuantized {
                weights: codebook_weights,
                ..
            }) => self.searcher.search_codebook_eval_quantized_factored(
                &mut self.board,
                &self.weights,
                codebook_weights,
                max_depth,
                time_limit,
            ),
            None => self
                .searcher
                .search(&mut self.board, &self.weights, max_depth, time_limit),
        };
        #[cfg(not(feature = "codebook-eval"))]
        let result = self
            .searcher
            .search(&mut self.board, &self.weights, max_depth, time_limit);
        emit_search_message(&result, search_start.elapsed());
        // Never emit an illegal move (occupied, or a Renju forbidden point for black).
        let board = &self.board;
        let mv = result
            .best_move
            .filter(|&mv| board.is_legal_move(mv))
            .or_else(|| board.candidate_moves().into_iter().find(|&mv| board.is_legal_move(mv)))
            .or_else(|| (0..BOARD_SIZE * BOARD_SIZE).find(|&mv| board.is_legal_move(mv)))?;
        self.board.make_move(mv);
        Some(idx_to_xy(mv))
    }
}

fn xy_to_idx(x: u8, y: u8) -> Result<usize, String> {
    if (x as usize) >= BOARD_SIZE || (y as usize) >= BOARD_SIZE {
        return Err(format!("coord ({x},{y}) out of range"));
    }
    Ok(to_idx(y as usize, x as usize))
}

fn idx_to_xy(idx: usize) -> (u8, u8) {
    let row = idx / BOARD_SIZE;
    let col = idx % BOARD_SIZE;
    (col as u8, row as u8)
}

fn main() {
    // Fail closed on stale or misspelled switches before anything reads the
    // environment or the manager sends a command.
    let unknown = unknown_engine_vars();
    if !unknown.is_empty() {
        let mut stdout = io::stdout().lock();
        for name in &unknown {
            writeln!(stdout, "ERROR unknown engine variable {name}").ok();
        }
        stdout.flush().ok();
        std::process::exit(2);
    }

    let mut engine = match Engine::new() {
        Ok(e) => e,
        Err(e) => {
            println!("ERROR - {e}");
            std::process::exit(1);
        }
    };

    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    let mut reader = stdin.lock();
    let mut line = String::new();
    let mut config_announced = false;

    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => std::process::exit(0),
            Ok(_) => {}
            Err(_) => std::process::exit(0),
        }

        let trimmed = line.trim_end_matches(['\n', '\r']);
        let mut split = trimmed.split_whitespace();
        let Some(raw_cmd) = split.next() else {
            continue;
        };
        let command = raw_cmd.to_uppercase();

        match command.as_str() {
            "START" => {
                let Some(sz_str) = split.next() else {
                    writeln!(stdout, "ERROR - missing board size").ok();
                    continue;
                };
                let Ok(sz) = sz_str.parse::<usize>() else {
                    writeln!(stdout, "ERROR - cannot parse board size").ok();
                    continue;
                };
                if sz != BOARD_SIZE {
                    writeln!(stdout, "ERROR - unsupported board size ({sz})").ok();
                    continue;
                }
                if !engine.info.rule_supported() {
                    writeln!(stdout, "ERROR - unsupported rule").ok();
                    continue;
                }
                engine.reset_board();
                // Standard rule (rule=1): exactly-5 wins. Tell the board so
                // its check_win drops overlines from the win-set.
                engine
                    .board
                    .set_rule_set(engine.info.rule_set().unwrap_or(RuleSet::Freestyle));
                engine.started = true;
                // The reply to START must be the first line: managers (and
                // harnesses) read OK strictly. The config MESSAGE follows it,
                // once per process.
                writeln!(stdout, "OK").ok();
                if !config_announced {
                    writeln!(stdout, "MESSAGE config: {}", engine.config_line).ok();
                    config_announced = true;
                }
            }
            "BEGIN" => {
                if !engine.started {
                    writeln!(stdout, "ERROR - engine not started").ok();
                    continue;
                }
                if let Some((x, y)) = engine.choose_move() {
                    writeln!(stdout, "{x},{y}").ok();
                } else {
                    writeln!(stdout, "{}", engine.no_move_error()).ok();
                }
            }
            "TURN" => {
                if !engine.started {
                    writeln!(stdout, "ERROR - engine not started").ok();
                    continue;
                }
                let Some(payload) = split.next() else {
                    writeln!(stdout, "ERROR - missing coord").ok();
                    continue;
                };
                let mut parts = payload.split(',');
                let Some(x) = parts.next().and_then(|s| s.trim().parse::<u8>().ok()) else {
                    writeln!(stdout, "ERROR - bad coord").ok();
                    continue;
                };
                let Some(y) = parts.next().and_then(|s| s.trim().parse::<u8>().ok()) else {
                    writeln!(stdout, "ERROR - bad coord").ok();
                    continue;
                };
                if let Err(e) = engine.apply_opp_move(x, y) {
                    writeln!(stdout, "ERROR - {e}").ok();
                    continue;
                }
                if let Some((ox, oy)) = engine.choose_move() {
                    writeln!(stdout, "{ox},{oy}").ok();
                } else {
                    writeln!(stdout, "{}", engine.no_move_error()).ok();
                }
            }
            "BOARD" => {
                if !engine.started {
                    writeln!(stdout, "ERROR - engine not started").ok();
                    continue;
                }
                let mut coords_own: Vec<(u8, u8)> = Vec::new();
                let mut coords_opp: Vec<(u8, u8)> = Vec::new();
                loop {
                    line.clear();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 {
                        std::process::exit(0);
                    }
                    let t = line.trim();
                    if t.to_uppercase() == "DONE" {
                        break;
                    }
                    let nums: Vec<u8> = t
                        .split(',')
                        .filter_map(|s| s.trim().parse::<u8>().ok())
                        .collect();
                    if nums.len() < 3 {
                        writeln!(stdout, "ERROR - expected input: 'x,y,player'").ok();
                        continue;
                    }
                    let (x, y, who) = (nums[0], nums[1], nums[2]);
                    match who {
                        1 => coords_own.push((x, y)),
                        2 => coords_opp.push((x, y)),
                        _ => {}
                    }
                }

                // own이 흑인지 백인지 결정.
                // 총 돌 수가 짝수면 다음은 흑 차례 → own=흑. 홀수면 own=백.
                let total = coords_own.len() + coords_opp.len();
                let own_is_white = total % 2 == 1;
                let (coords_b, coords_w) = if own_is_white {
                    (&coords_opp, &coords_own)
                } else {
                    (&coords_own, &coords_opp)
                };

                engine.reset_board();
                let zip_len = coords_b.len().min(coords_w.len());
                for i in 0..zip_len {
                    match xy_to_idx(coords_b[i].0, coords_b[i].1) {
                        Ok(idx) if engine.board.is_empty(idx) => engine.board.make_move(idx),
                        _ => {
                            writeln!(stdout, "ERROR - invalid BOARD state").ok();
                            engine.reset_board();
                            break;
                        }
                    }
                    match xy_to_idx(coords_w[i].0, coords_w[i].1) {
                        Ok(idx) if engine.board.is_empty(idx) => engine.board.make_move(idx),
                        _ => {
                            writeln!(stdout, "ERROR - invalid BOARD state").ok();
                            engine.reset_board();
                            break;
                        }
                    }
                }
                for i in zip_len..coords_b.len() {
                    let (bx, by) = coords_b[i];
                    match xy_to_idx(bx, by) {
                        Ok(idx) if engine.board.is_empty(idx) => engine.board.make_move(idx),
                        _ => {
                            writeln!(stdout, "ERROR - invalid BOARD state").ok();
                            engine.reset_board();
                            break;
                        }
                    }
                }

                if let Some((x, y)) = engine.choose_move() {
                    writeln!(stdout, "{x},{y}").ok();
                } else {
                    writeln!(stdout, "{}", engine.no_move_error()).ok();
                }
            }
            "INFO" => {
                let Some(key) = split.next() else {
                    continue;
                };
                let Some(val) = split.next() else {
                    continue;
                };
                engine.info.update(key, val);
                // Managers send `INFO rule` after START, so the START check
                // cannot catch it; say so here instead of playing Freestyle.
                if key == "rule" && !engine.info.rule_supported() {
                    writeln!(stdout, "ERROR - unsupported rule {val}").ok();
                }
            }
            "END" => std::process::exit(0),
            "ABOUT" => {
                writeln!(
                    stdout,
                    "name=\"figrid\", version=\"{}\", author=\"Hogyung Choi\", country=\"KR\"",
                    env!("CARGO_PKG_VERSION")
                )
                .ok();
            }
            _ => {
                writeln!(stdout, "UNKNOWN").ok();
            }
        }
        stdout.flush().ok();
    }
}
