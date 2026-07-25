use super::*;

#[test]
fn usage_separates_p0a_and_p0b_label_surfaces() {
    let text = usage();
    assert!(text.contains(" p0a "));
    assert!(text.contains(" p0b "));
    assert!(text.contains("--expected-p0a-bytes"));
    assert!(text.contains("--expected-p0a-sha256"));
}

#[test]
fn preregistration_commit_is_frozen() {
    assert_eq!(PREREGISTER_COMMIT, "5c63f04");
}

#[test]
fn cargo_exposes_the_registered_compile_environment() {
    assert_eq!(COMPILE_TIME_RUSTFLAGS, Some(CANONICAL_RUSTFLAGS));
    assert!(
        COMPILE_TIME_FORBIDDEN
            .iter()
            .all(|(_, value)| value.is_none()),
        "compile-time forbidden variables: {COMPILE_TIME_FORBIDDEN:?}"
    );
}

#[test]
fn registered_p0a_argument_vector_is_fail_closed() {
    let expected = [
        "p0a",
        "--prepared-units",
        PREPARED_UNITS_PATH,
        "--phase2-manifest",
        PHASE2_MANIFEST_PATH,
        "--product-model",
        PRODUCT_MODEL_PATH,
        "--product-cbf",
        PRODUCT_CBF_PATH,
        "--topk",
        TOPK_PATH,
        "--out-selector",
        P0A_OUTPUT_PATH,
    ];
    let mut arguments = expected.map(OsString::from);
    validate_registered_arguments(&arguments).unwrap();
    arguments.swap(1, 3);
    assert!(validate_registered_arguments(&arguments).is_err());
}

#[test]
fn registered_p0b_argument_vector_allows_only_the_two_literals_to_vary() {
    let expected = [
        "p0b",
        "--selector",
        P0A_OUTPUT_PATH,
        "--expected-p0a-bytes",
        "123",
        "--expected-p0a-sha256",
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "--prepared-units",
        PREPARED_UNITS_PATH,
        "--phase2-manifest",
        PHASE2_MANIFEST_PATH,
        "--product-model",
        PRODUCT_MODEL_PATH,
        "--product-cbf",
        PRODUCT_CBF_PATH,
        "--topk",
        TOPK_PATH,
        "--train",
        TRAIN_PATH,
        "--final-manifest",
        FINAL_MANIFEST_PATH,
        "--lineage-model",
        LINEAGE_MODEL_PATH,
        "--out-reveal",
        P0B_OUTPUT_PATH,
    ];
    let mut arguments = expected.map(OsString::from);
    validate_registered_arguments(&arguments).unwrap();
    arguments[2] = OsString::from("alternate-selector.json");
    assert!(validate_registered_arguments(&arguments).is_err());
}
