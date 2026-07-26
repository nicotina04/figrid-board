#![cfg(feature = "codebook-eval")]

use figrid_board::codebook_eval::{
    CodebookWeights, QUANT_EMBED_SCALE, QUANT_FACTOR_SCALE, QUANT_HEAD_SCALE,
    QuantizedCodebookWeights,
};
use figrid_board::factored_codebook::{
    FactoredQuantizedCodebookWeights, PackedCodebookArtifact, PackedCodebookKind,
    PackedQuantizedPayload,
};
use figrid_board::pattern_table::{PATTERN_NUM_IDS, PATTERN_RARE_ID};
use serde_json::json;

const REGIONS: usize = 9;
const FACTORED_ARTIFACT: &[u8] =
    include_bytes!("../models/gomoku_codebook_v1_swapclosed_factored.cbf");
const CB2VEC_JOURNAL_V0_1_0: &[u8] =
    include_bytes!("../audit/provenance/cb2vec-0.1.0-journal.rs.snapshot");

#[test]
fn public_weight_paths_fields_and_constructors_remain_compatible() {
    let mut literal = CodebookWeights {
        dim: 1,
        fm_rank: 1,
        embeddings: vec![0.0; PATTERN_NUM_IDS],
        head: vec![0.0; REGIONS],
        factors: vec![0.0; REGIONS],
        bias: 0.25,
    };
    literal.embeddings[0] = 1.0;
    assert_eq!(literal.feature_len(), REGIONS);

    let quantized = literal.quantize_i16_s32_s64();
    assert_quantized_shape(&quantized);
    assert_eq!(quantized.embeddings[0], QUANT_EMBED_SCALE as i16);

    let dequantized: CodebookWeights = quantized.dequantized();
    assert_eq!(dequantized.dim, 1);
    assert_eq!(dequantized.fm_rank, 1);
    assert_eq!(dequantized.embeddings[0], 1.0);
    assert_eq!(dequantized.bias, 0.25);
    drop(dequantized);
    drop(quantized);
    drop(literal);

    let literal_quantized = QuantizedCodebookWeights {
        dim: 1,
        fm_rank: 1,
        embedding_scale: QUANT_EMBED_SCALE,
        head_scale: QUANT_HEAD_SCALE,
        factor_scale: QUANT_FACTOR_SCALE,
        embeddings: vec![0; PATTERN_NUM_IDS],
        head: vec![0; REGIONS],
        factors: vec![0; REGIONS],
        bias: -0.5,
    };
    assert_quantized_shape(&literal_quantized);
    let literal_dequantized: CodebookWeights = literal_quantized.dequantized();
    assert_eq!(literal_dequantized.bias, -0.5);
    drop(literal_dequantized);
    drop(literal_quantized);

    let deterministic: CodebookWeights = CodebookWeights::deterministic(1, 1);
    assert_eq!(deterministic.dim, 1);
    assert_eq!(deterministic.fm_rank, 1);
    assert_eq!(deterministic.embeddings.len(), PATTERN_NUM_IDS);
    assert_eq!(deterministic.feature_len(), REGIONS);
    drop(deterministic);

    let root = json!({
        "format": "noru-pattern4-codebook-eval-v1",
        "model": "codebook-region-fm",
        "embedding_dim": 1,
        "fm_rank": 1,
        "regions": REGIONS,
        "weights": {
            "embeddings": vec![0.0; PATTERN_NUM_IDS],
            "head": vec![0.0; REGIONS],
            "factors": vec![0.0; REGIONS],
            "bias": 0.125,
        },
    });
    let from_value: CodebookWeights =
        CodebookWeights::from_json_value(&root).expect("legacy Value loader");
    assert_eq!(from_value.embeddings.len(), PATTERN_NUM_IDS);
    assert_eq!(from_value.bias, 0.125);
    drop(from_value);

    let json_bytes = serde_json::to_vec(&root).expect("serialize compatibility fixture");
    let from_bytes: CodebookWeights =
        CodebookWeights::from_json_bytes(&json_bytes).expect("legacy byte loader");
    assert_eq!(from_bytes.feature_len(), REGIONS);
    assert_eq!(from_bytes.bias, 0.125);

    let mut missing_model = root.clone();
    missing_model
        .as_object_mut()
        .expect("object fixture")
        .remove("model");
    assert!(CodebookWeights::from_json_value(&missing_model).is_err());

    let mut wrong_regions = root.clone();
    wrong_regions["regions"] = json!(REGIONS - 1);
    assert!(CodebookWeights::from_json_value(&wrong_regions).is_err());
    drop(root);
}

#[test]
fn packed_factored_paths_and_return_types_remain_compatible() {
    let _: fn(&[u8]) -> Result<PackedCodebookArtifact, String> = PackedCodebookArtifact::parse;
    let _: fn(PackedCodebookArtifact) -> PackedQuantizedPayload =
        PackedCodebookArtifact::into_quantized_payload;
    let _: fn(PackedCodebookArtifact) -> Result<FactoredQuantizedCodebookWeights, String> =
        PackedCodebookArtifact::into_factored_quantized;
    let _: fn(&FactoredQuantizedCodebookWeights) -> QuantizedCodebookWeights =
        FactoredQuantizedCodebookWeights::reconstruct_flat;

    let artifact: PackedCodebookArtifact =
        PackedCodebookArtifact::parse(FACTORED_ARTIFACT).expect("public factored artifact");
    assert_eq!(artifact.kind(), PackedCodebookKind::Factored);

    // Consuming the artifact drops its source-float payload before the only
    // flat reconstruction, keeping this compatibility test's peak memory low.
    let factored: FactoredQuantizedCodebookWeights = artifact
        .into_factored_quantized()
        .expect("factored payload through the legacy FIGRID path");
    factored.validate().expect("valid FIGRID factored weights");

    let flat: QuantizedCodebookWeights = factored.reconstruct_flat();
    assert_eq!(flat.dim, factored.dim());
    assert_eq!(flat.fm_rank, factored.fm_rank());
    assert_eq!(flat.embeddings.len(), PATTERN_NUM_IDS * factored.dim());
    assert_eq!(flat.head.len(), factored.feature_len());
    assert_eq!(
        flat.factors.len(),
        factored.feature_len() * factored.fm_rank()
    );

    for pattern_id in [0, 585, PATTERN_RARE_ID] {
        for component in 0..factored.dim() {
            assert_eq!(
                flat.embeddings[pattern_id as usize * factored.dim() + component],
                factored.embedding(pattern_id, component)
            );
        }
    }
}

#[test]
fn packaged_journal_provenance_snapshot_is_frozen() {
    // The standalone dependency now resolves from crates.io. This source
    // snapshot remains pinned to its published 0.1.0 provenance so historical
    // audit binaries can be rebuilt without a nested dependency checkout.
    assert_eq!(CB2VEC_JOURNAL_V0_1_0.len(), 16_090);
}

fn assert_quantized_shape(weights: &QuantizedCodebookWeights) {
    assert_eq!(weights.dim, 1);
    assert_eq!(weights.fm_rank, 1);
    assert_eq!(weights.embedding_scale, QUANT_EMBED_SCALE);
    assert_eq!(weights.head_scale, QUANT_HEAD_SCALE);
    assert_eq!(weights.factor_scale, QUANT_FACTOR_SCALE);
    assert_eq!(weights.embeddings.len(), PATTERN_NUM_IDS);
    assert_eq!(weights.feature_len(), REGIONS);
    assert_eq!(weights.factors.len(), REGIONS);
}
