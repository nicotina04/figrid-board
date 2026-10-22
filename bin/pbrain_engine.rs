use super::ProtocolInfo;
// Gomocup (Piskvork) protocol adapter for the noru-tactic NNUE engine.
//
// Ships as the `pbrain-figrid` binary.


use std::sync::OnceLock;
use std::time::{Duration, Instant};

#[cfg(feature = "codebook-eval")]
use backend::codebook_eval::{CodebookWeights, ExactI8Codebook, QuantizedCodebookWeights};
#[cfg(feature = "codebook-eval")]
use backend::factored_codebook::{FactoredQuantizedCodebookWeights, PackedCodebookArtifact};
use backend::{BOARD_SIZE, Board, GOMOKU_NNUE_CONFIG, RuleSet, Searcher, book, to_idx};
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
    #[cfg(not(feature = "release-models"))]
    const COMPRESSED: &[u8] = include_bytes!("../models/gomoku_v52_5stone_conv_93k.bin.gz");
    #[cfg(feature = "release-models")]
    const COMPRESSED: &[u8] = include_bytes!(concat!(
        env!("FIGRID_RELEASE_MODEL_DIR"),
        "/ordering_flat_v52.bin.gz"
    ));
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
/// Headroom subtracted from the turn budget so the last node batch finishes
/// well before Piskvork's deadline. Without this, the 128-node deadline
/// check can overshoot by ~50 ms on NNUE-heavy positions.
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
#[cfg(test)]
use super::environment::unknown_engine_var_names;
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
            vec![
                "FIGRID_CODEBOOK_EVAL",
                "NORU_CANDIDATE_RANKER",
                "NORU_ROOT_VCT"
            ]
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
    ExactI8 {
        weights: ExactI8Codebook,
        embedded: bool,
    },
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
/// Pre-empting an opponent three is not forcing (1.1). `NORU_DEMOTE_PREEMPT_THREE=off` restores the 1.0 forcing set.
fn demote_preempt_three_enabled() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("NORU_DEMOTE_PREEMPT_THREE")
            .map(|raw| env_bool_default(&raw, true))
            .unwrap_or(true)
    })
}

/// Per-move quiet policy (1.1): unset or empty uses the embedded model, `off`/`0`/`false`/`no` disables it, anything
/// else is a `PGM1` file. Returns the model and its label for the config line.
fn load_policy_move() -> Result<
    (
        Option<std::sync::Arc<backend::policy_move::PolicyMove>>,
        String,
    ),
    String,
> {
    use backend::policy_move::PolicyMove;
    let raw = std::env::var("NORU_POLICY_MOVE").unwrap_or_default();
    let raw = raw.trim();
    if raw.is_empty() {
        #[cfg(feature = "codebook-eval")]
        return Ok((
            Some(std::sync::Arc::new(PolicyMove::embedded())),
            "embedded".to_string(),
        ));
        #[cfg(not(feature = "codebook-eval"))]
        return Ok((None, "unsupported".to_string()));
    }
    if !env_bool_default(raw, true) {
        return Ok((None, "off".to_string()));
    }
    let bytes = std::fs::read(raw).map_err(|e| format!("NORU_POLICY_MOVE `{raw}`: {e}"))?;
    let model =
        PolicyMove::from_bytes(&bytes).map_err(|e| format!("NORU_POLICY_MOVE `{raw}`: {e}"))?;
    Ok((Some(std::sync::Arc::new(model)), raw.to_string()))
}

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
            CodebookRuntimeWeights::FactoredQuantized { .. }
            | CodebookRuntimeWeights::ExactI8 { .. } => unreachable!(),
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
    #[cfg(feature = "release-models")]
    if configured_path.is_none() {
        let bytes: &[u8] = if BOARD_SIZE == 20 {
            include_bytes!(concat!(env!("FIGRID_RELEASE_MODEL_DIR"), "/ng3_board20.ngcb8.gz"))
        } else {
            include_bytes!(concat!(env!("FIGRID_RELEASE_MODEL_DIR"), "/ng3_standard_omok.ngcb8.gz"))
        };
        return embedded_exact_i8(bytes).map(Some);
    }
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
    if bytes.starts_with(ExactI8Codebook::MAGIC) {
        let weights = ExactI8Codebook::from_bytes(&bytes)?;
        backend::pattern_table::warm_full_vocab();
        return Ok(CodebookRuntimeWeights::ExactI8 {
            weights,
            embedded: false,
        });
    }
    let weights = CodebookWeights::from_bytes_auto(&bytes)
        .map_err(|e| format!("failed to parse codebook weights: {e}"))?;
    if weights.is_full_vocab() {
        // Build the full-vocabulary lookup tables now so their one-off cost
        // never lands inside the first move's clock.
        backend::pattern_table::warm_full_vocab();
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
    for (rule, tag) in [
        (RuleSet::Freestyle, "FREESTYLE"),
        (RuleSet::Standard, "STANDARD"),
        (RuleSet::Omok, "OMOK"),
        (RuleSet::Caro, "CARO"),
        (RuleSet::Renju, "RENJU"),
    ] {
        let configured = std::env::var_os(format!("FIGRID_CODEBOOK_WEIGHTS_{tag}"));
        if configured.is_none() {
            #[cfg(feature = "release-models")]
            if BOARD_SIZE == 15 && std::env::var_os("FIGRID_CODEBOOK_WEIGHTS").is_none() {
                let embedded: Option<(&[u8], f32)> = match rule {
                    RuleSet::Freestyle => Some((
                        include_bytes!(concat!(
                            env!("FIGRID_RELEASE_MODEL_DIR"),
                            "/kg11_freestyle15.ngcb8.gz"
                        )),
                        13.171443,
                    )),
                    RuleSet::Caro => Some((
                        include_bytes!(concat!(
                            env!("FIGRID_RELEASE_MODEL_DIR"),
                            "/kg14_caro.ngcb8.gz"
                        )),
                        14.939977,
                    )),
                    RuleSet::Renju => Some((
                        include_bytes!(concat!(
                            env!("FIGRID_RELEASE_MODEL_DIR"),
                            "/ng3_renju.ngcb8.gz"
                        )),
                        15.499594,
                    )),
                    _ => None,
                };
                if let Some((gz, scale)) = embedded {
                    out.push(RuleCodebook {
                        rule,
                        weights: embedded_exact_i8(gz)?,
                        scale,
                        label: format!("{tag}=embedded-release"),
                    });
                }
            }
            continue;
        }
        let path = configured
            .unwrap()
            .into_string()
            .map_err(|_| format!("FIGRID_CODEBOOK_WEIGHTS_{tag} is not Unicode"))?;
        let path = path.trim();
        if path.is_empty() {
            continue;
        }
        let raw = std::env::var(format!("NORU_CODEBOOK_EVAL_SCALE_{tag}"))
            .map_err(|_| format!("FIGRID_CODEBOOK_WEIGHTS_{tag} needs its scale"))?;
        let scale = raw
            .trim()
            .parse::<f32>()
            .ok()
            .filter(|s| s.is_finite() && *s > 0.)
            .ok_or_else(|| format!("invalid NORU_CODEBOOK_EVAL_SCALE_{tag}"))?;
        out.push(RuleCodebook {
            rule,
            weights: load_codebook_file(path)?,
            scale,
            label: format!("{tag}={path}"),
        });
    }
    Ok(out)
}

#[cfg(feature = "release-models")]
fn embedded_exact_i8(gz: &[u8]) -> Result<CodebookRuntimeWeights, String> {
    use std::io::Read;
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(gz)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    let weights = ExactI8Codebook::from_bytes(&bytes)?;
    backend::pattern_table::warm_full_vocab();
    Ok(CodebookRuntimeWeights::ExactI8 {
        weights,
        embedded: true,
    })
}

#[cfg(feature = "codebook-eval")]
fn embedded_default_scale() -> Option<f32> {
    #[cfg(feature = "release-models")]
    if std::env::var_os("FIGRID_CODEBOOK_WEIGHTS").is_none()
        && std::env::var("NORU_CODEBOOK_EVAL_SCALE")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|s| s.is_finite() && *s > 0.)
            .is_none()
    {
        return Some(13.171443);
    }
    None
}

/// `codebook=...; ...` part of the startup config line.
#[cfg(feature = "codebook-eval")]
fn codebook_config_summary(
    codebook: &Option<CodebookRuntimeWeights>,
    searcher: &Searcher,
) -> String {
    let on_off = |enabled: bool| if enabled { "on" } else { "off" };
    let (source, kernel, ids) = match codebook {
        None => return "codebook=off".to_string(),
        Some(CodebookRuntimeWeights::ExactI8 { weights, embedded }) => (
            if *embedded {
                "embedded-release".into()
            } else {
                "external-exact-i8".into()
            },
            "exact-int8",
            weights.table.len() / weights.scorer.dim,
        ),
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
        embedded_default_scale().unwrap_or_else(backend::search::codebook_eval_scale),
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

fn emit_search_message(result: &backend::SearchResult, elapsed: Duration) {
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

pub(super) struct Engine {
    pub(super) board: Board,
    weights: NnueWeights,
    #[cfg(feature = "codebook-eval")]
    codebook_weights: Option<CodebookRuntimeWeights>,
    #[cfg(feature = "codebook-eval")]
    rule_codebooks: Vec<RuleCodebook>,
    /// One-line effective configuration, announced before the first START
    /// reply.
    pub(super) config_line: String,
    searcher: Searcher,
    pub(super) info: ProtocolInfo,
    pub(super) started: bool,
}

impl Engine {
    pub(super) fn new() -> Result<Self, String> {
        #[cfg(not(feature = "codebook-eval"))]
        reject_explicit_white_root_order_without_codebook()?;
        let bytes = load_weights_bytes()?;
        let weights = NnueWeights::load_from_bytes(&bytes, Some(GOMOKU_NNUE_CONFIG))
            .map_err(|e| format!("failed to parse weights: {e}"))?;
        #[cfg(feature = "codebook-eval")]
        let codebook_weights = load_codebook_weights()?;
        // The flat NNUE feature layout is 15x15-only; the 20x20 engine always evaluates with a codebook.
        #[cfg(feature = "codebook-eval")]
        if BOARD_SIZE == 20 && codebook_weights.is_none() {
            return Err(
                "the 20x20 build needs a codebook (FIGRID_CODEBOOK_WEIGHTS must not be off)"
                    .to_string(),
            );
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
        let (policy_move, policy_move_label) = load_policy_move()?;
        searcher.set_policy_move(policy_move);
        searcher.set_demote_preempt_three(demote_preempt_three_enabled());
        // Resolving the search switches here reports a bad NORU_POLICY_ORDER
        // (or NORU_POLICY_REDUCE without a table) at startup instead of at the
        // first search.
        let search_config = backend::search::runtime_config_summary()?;
        #[cfg(feature = "codebook-eval")]
        let codebook_config = {
            let base = codebook_config_summary(&codebook_weights, &searcher);
            if rule_codebooks.is_empty() {
                base
            } else {
                let per_rule: Vec<String> = rule_codebooks
                    .iter()
                    .map(|r| format!("{} scale={}", r.label, r.scale))
                    .collect();
                format!("{base}; rule_codebooks=[{}]", per_rule.join(", "))
            }
        };
        #[cfg(not(feature = "codebook-eval"))]
        let codebook_config = "codebook=unsupported".to_string();
        let on_off = |enabled: bool| if enabled { "on" } else { "off" };
        let config_line = format!(
            "version={}; weights={}; {codebook_config}; packed_line_windows={}; \
             candidate_frontier={}; policy_move={policy_move_label}; demote_preempt_three={}; {search_config};              pbrain_fixed_depth={}; pbrain_max_depth={}",
            env!("CARGO_PKG_VERSION"),
            weights_label(),
            on_off(packed_line_windows_enabled()),
            on_off(candidate_frontier_enabled()),
            on_off(demote_preempt_three_enabled()),
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

    pub(super) fn reset_board(&mut self) {
        self.board = Board::new();
    }

    pub(super) fn apply_opp_move(&mut self, x: u8, y: u8) -> Result<(), String> {
        let idx = xy_to_idx(x, y)?;
        if !self.board.is_empty(idx) {
            return Err(format!("cell ({x},{y}) already occupied"));
        }
        self.board.make_move(idx);
        Ok(())
    }

    pub(super) fn no_move_error(&self) -> &'static str {
        if self.info.rule_supported() {
            "ERROR - no legal move"
        } else {
            "ERROR - unsupported rule"
        }
    }

    pub(super) fn choose_move(&mut self) -> Option<(u8, u8)> {
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
            None => (self.codebook_weights.as_ref(), embedded_default_scale()),
        };
        #[cfg(feature = "codebook-eval")]
        self.searcher.set_codebook_eval_scale(scale);
        #[cfg(feature = "codebook-eval")]
        let result = match codebook {
            Some(CodebookRuntimeWeights::ExactI8 { weights, .. }) => {
                self.searcher.search_codebook_eval_exact_i8(
                    &mut self.board,
                    &self.weights,
                    weights,
                    max_depth,
                    time_limit,
                )
            }
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
            .or_else(|| {
                board
                    .candidate_moves()
                    .into_iter()
                    .find(|&mv| board.is_legal_move(mv))
            })
            .or_else(|| (0..BOARD_SIZE * BOARD_SIZE).find(|&mv| board.is_legal_move(mv)))?;
        self.board.make_move(mv);
        Some(idx_to_xy(mv))
    }
}

pub(super) fn xy_to_idx(x: u8, y: u8) -> Result<usize, String> {
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

