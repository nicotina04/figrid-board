//! Converts codebook weights (JSON, or NGCB1) to the NGCB1 binary format.
//!
//! Usage: `ngcb-convert IN.json OUT.ngcb`. The written file is parsed back
//! and must match the input model bit for bit.

use figrid_board::codebook_eval::CodebookWeights;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let input = args.next().ok_or("usage: ngcb-convert IN.json OUT.ngcb")?;
    let output = args.next().ok_or("usage: ngcb-convert IN.json OUT.ngcb")?;
    let bytes = std::fs::read(&input).map_err(|e| format!("read {input}: {e}"))?;
    let weights = CodebookWeights::from_bytes_auto(&bytes)?;
    let out = weights.to_ngcb1_bytes();
    std::fs::write(&output, &out).map_err(|e| format!("write {output}: {e}"))?;
    // Roundtrip verification: the written file must parse back to an
    // identical model (bit-level f32 equality).
    let back = CodebookWeights::from_ngcb1_bytes(&out)?;
    let same = back.dim == weights.dim
        && back.fm_rank == weights.fm_rank
        && back.bias.to_bits() == weights.bias.to_bits()
        && back
            .embeddings
            .iter()
            .zip(&weights.embeddings)
            .all(|(a, b)| a.to_bits() == b.to_bits())
        && back
            .head
            .iter()
            .zip(&weights.head)
            .all(|(a, b)| a.to_bits() == b.to_bits())
        && back
            .factors
            .iter()
            .zip(&weights.factors)
            .all(|(a, b)| a.to_bits() == b.to_bits());
    if !same {
        return Err("roundtrip mismatch".into());
    }
    eprintln!(
        "ok: dim={} fm_rank={} num_ids={} bytes={}",
        weights.dim,
        weights.fm_rank,
        weights.num_ids(),
        out.len()
    );
    Ok(())
}
