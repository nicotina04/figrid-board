use cb2vec::{CB2VEC_ARTIFACT_MAGIC, CodebookWeights, PackedCodebookArtifact};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source = CodebookWeights::deterministic(12, 3, 8, 2);
    let quantized = source.quantize_i16_s32_s64();
    let artifact = PackedCodebookArtifact::new_flat(source, quantized, [0x42; 32])?;

    let bytes = artifact.to_bytes()?;
    assert_eq!(&bytes[..8], &CB2VEC_ARTIFACT_MAGIC);

    let parsed = PackedCodebookArtifact::parse(&bytes)?;
    assert_eq!(parsed.to_bytes()?, bytes);
    assert_eq!(parsed.source_sha256(), &[0x42; 32]);
    Ok(())
}
