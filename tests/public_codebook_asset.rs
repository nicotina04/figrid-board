#![cfg(feature = "codebook-eval")]

use figrid_board::codebook_eval::{
    CodebookWeights, QUANT_EMBED_SCALE, QUANT_FACTOR_SCALE, QUANT_HEAD_SCALE,
};
use serde_json::Value;

const ASSET: &[u8] = include_bytes!("../models/gomoku_codebook_v1_swapclosed.json");
const EXPECTED_SEMANTIC_SHA256: &str =
    "C88366A868FD7203944E64E6AE8056144E796ECA1EF1CDE913FAE08CD9EB96CA";
const EXPECTED_QUANTIZED_SHA256: &str =
    "BDEAF5A5156BA2B3AD2B29B7D33843E936763D844973F73A36E023CE83E5228A";

#[test]
fn public_asset_is_minimal_and_loader_semantics_are_frozen() {
    let root: Value = serde_json::from_slice(ASSET).expect("public codebook JSON");
    let object = root.as_object().expect("object root");
    let mut root_keys = object.keys().map(String::as_str).collect::<Vec<_>>();
    root_keys.sort_unstable();
    assert_eq!(
        root_keys,
        [
            "embedding_dim",
            "fm_rank",
            "format",
            "model",
            "regions",
            "weights",
        ]
    );
    assert_eq!(
        object.get("format").and_then(Value::as_str),
        Some("noru-pattern4-codebook-eval-v1")
    );
    assert_eq!(
        object.get("model").and_then(Value::as_str),
        Some("codebook-region-fm")
    );
    let weight_object = object
        .get("weights")
        .and_then(Value::as_object)
        .expect("weights object");
    let mut weight_keys = weight_object.keys().map(String::as_str).collect::<Vec<_>>();
    weight_keys.sort_unstable();
    assert_eq!(weight_keys, ["bias", "embeddings", "factors", "head"]);

    let text = std::str::from_utf8(ASSET)
        .expect("UTF-8")
        .to_ascii_lowercase();
    for forbidden in [
        ["ra", "pfi"].concat(),
        ["pe", "la"].concat(),
        ["sea", "son"].concat(),
        ["exper", "iments"].concat(),
        [".work", "trees"].concat(),
    ] {
        assert!(
            !text.contains(&forbidden),
            "forbidden literal {forbidden:?}"
        );
    }
    assert!(!contains_numbered_identifier(&text));
    assert!(!text.contains(":\\users\\"));
    for key in [
        "metadata",
        "normalization",
        "objective",
        "target_mode",
        "pattern_id_migration",
        "pattern_id_migration_source",
    ] {
        assert!(!object.contains_key(key), "forbidden root key {key}");
    }

    let weights = CodebookWeights::from_json_bytes(ASSET).expect("runtime loader accepts asset");
    assert_eq!(weights.dim, 16);
    assert_eq!(weights.fm_rank, 8);
    assert_eq!(weights.embeddings.len(), 68_256);
    assert_eq!(weights.head.len(), 144);
    assert_eq!(weights.factors.len(), 1_152);
    assert_eq!(weights.bias.to_bits(), 0xBE9C_5ADD);

    assert_eq!(
        sha256_hex(&semantic_payload(&weights)),
        EXPECTED_SEMANTIC_SHA256
    );

    let quantized = weights.quantize_i16_s32_s64();
    assert_eq!(quantized.embedding_scale, QUANT_EMBED_SCALE);
    assert_eq!(quantized.head_scale, QUANT_HEAD_SCALE);
    assert_eq!(quantized.factor_scale, QUANT_FACTOR_SCALE);
    assert_eq!(
        sha256_hex(&quantized_payload(&quantized)),
        EXPECTED_QUANTIZED_SHA256
    );
}

fn contains_numbered_identifier(text: &str) -> bool {
    let bytes = text.as_bytes();
    for index in 0..bytes.len().saturating_sub(4) {
        if bytes[index] == b'r'
            && bytes[index + 1] == b'q'
            && bytes[index + 2].is_ascii_digit()
            && bytes[index + 3].is_ascii_digit()
            && bytes[index + 4].is_ascii_digit()
        {
            return true;
        }
    }
    false
}

fn semantic_payload(weights: &CodebookWeights) -> Vec<u8> {
    let mut out = b"figrid-codebook-semantic-v1\0".to_vec();
    for value in [weights.dim, weights.fm_rank, 9] {
        out.extend_from_slice(&(value as u64).to_le_bytes());
    }
    for values in [&weights.embeddings, &weights.head, &weights.factors] {
        out.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            out.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }
    out.extend_from_slice(&weights.bias.to_bits().to_le_bytes());
    out
}

fn quantized_payload(weights: &figrid_board::codebook_eval::QuantizedCodebookWeights) -> Vec<u8> {
    let mut out = b"figrid-codebook-quantized-v1\0".to_vec();
    for value in [
        weights.dim as i64,
        weights.fm_rank as i64,
        9,
        weights.embedding_scale as i64,
        weights.head_scale as i64,
        weights.factor_scale as i64,
    ] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    for values in [&weights.embeddings, &weights.head, &weights.factors] {
        out.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    out.extend_from_slice(&weights.bias.to_bits().to_le_bytes());
    out
}

fn sha256_hex(input: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut data = input.to_vec();
    let bit_len = (data.len() as u64) * 8;
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&bit_len.to_be_bytes());
    let mut state = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    for chunk in data.chunks_exact(64) {
        let mut words = [0u32; 64];
        for (index, word) in words.iter_mut().take(16).enumerate() {
            *word = u32::from_be_bytes(chunk[index * 4..index * 4 + 4].try_into().unwrap());
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(sum1)
                .wrapping_add(choose)
                .wrapping_add(K[index])
                .wrapping_add(words[index]);
            let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = sum0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }
    state.iter().map(|word| format!("{word:08X}")).collect()
}
