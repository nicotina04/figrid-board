//! JSON/NGCB1 converter, with lossless INT16-to-INT8 embedding packing.
use figrid_board::codebook_eval::{CodebookWeights, ExactI8Codebook};
fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let exact = args.first().is_some_and(|s| s == "--exact-i8");
    if exact {
        args.remove(0);
    }
    if args.len() != 2 {
        return Err("usage: ngcb-convert [--exact-i8] INPUT OUTPUT".into());
    }
    let bytes = std::fs::read(&args[0]).map_err(|e| e.to_string())?;
    let weights = CodebookWeights::from_bytes_auto(&bytes)?;
    let out = if exact {
        let q = weights.quantize_i16_s32_s64();
        let p = ExactI8Codebook::from_quantized(&q)?;
        let out = p.to_bytes();
        if !ExactI8Codebook::from_bytes(&out)?.exact_reconstruction(&q) {
            return Err("exact INT8 reconstruction failed".into());
        }
        out
    } else {
        let out = weights.to_ngcb1_bytes();
        let back = CodebookWeights::from_ngcb1_bytes(&out)?;
        if back.to_ngcb1_bytes() != out {
            return Err("NGCB1 roundtrip failed".into());
        }
        out
    };
    std::fs::write(&args[1], &out).map_err(|e| e.to_string())?;
    eprintln!(
        "ok: exact_i8={exact} dim={} rank={} ids={} bytes={}",
        weights.dim,
        weights.fm_rank,
        weights.num_ids(),
        out.len()
    );
    Ok(())
}
