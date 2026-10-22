const KNOWN_ENGINE_VARS: &[&str] = &[
    // Model files and codebook.
    "FIGRID_WEIGHTS",
    "FIGRID_CODEBOOK_WEIGHTS",
    "NORU_CODEBOOK_EVAL_SCALE",
    // Optional per-rule codebooks (each needs its own scale).
    "FIGRID_CODEBOOK_WEIGHTS_FREESTYLE",
    "FIGRID_CODEBOOK_WEIGHTS_OMOK",
    "FIGRID_CODEBOOK_WEIGHTS_STANDARD",
    "FIGRID_CODEBOOK_WEIGHTS_CARO",
    "FIGRID_CODEBOOK_WEIGHTS_RENJU",
    "NORU_CODEBOOK_EVAL_SCALE_FREESTYLE",
    "NORU_CODEBOOK_EVAL_SCALE_OMOK",
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
    "NORU_POLICY_MOVE",
    "NORU_DEMOTE_PREEMPT_THREE",
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
pub(super) fn unknown_engine_vars() -> Vec<String> {
    unknown_engine_var_names(
        std::env::vars_os()
            .filter(|(_, value)| !value.is_empty())
            .map(|(name, _)| name.to_string_lossy().into_owned()),
    )
}

pub(super) fn unknown_engine_var_names(names: impl Iterator<Item = String>) -> Vec<String> {
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
