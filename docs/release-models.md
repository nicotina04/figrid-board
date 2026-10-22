# GitHub release model builds (1.3.0)

The published 1.3.0 Windows ZIP contains separate self-contained 15x15 and 20x20
Piskvork executables. The current source builds one executable supporting both
sizes; the existing release models remain separate and are selected at `START`.
It requires an x86-64-v3 (AVX2-class) CPU and uses the static Windows CRT.

| Rule | Embedded model | Evaluation scale |
| --- | --- | --- |
| Freestyle, 15x15 (0) | KG11, validated by KG12 | 13.171443 |
| Caro (9) | KG14 | 14.939977 |
| Standard (1), Korean Omok (17) | original NG3 Freestyle15 | 13.171443 |
| Renju (4) | original NG3 Renju | 15.499594 |
| 20x20 Freestyle | original NG3 board20 | 13.171443 |

Ordering weights and move policy are unchanged from 1.2.0. New Freestyle and Caro
weights do not apply to other rules. Renju's stopped experiment was not promoted.

## Rebuilding the binary distribution

Obtain the matching private `models-1.3.0` assets with authorized access. Put the five
`.ngcb8.gz` files and `ordering_flat_v52.bin.gz` in one directory. These private assets
are deliberately absent from public Git and the crate package. On Windows:

```powershell
$env:FIGRID_RELEASE_MODEL_DIR = 'C:/absolute/path/to/models'
$env:RUSTFLAGS = '-C target-cpu=x86-64-v3 -C target-feature=+crt-static'
cargo build --release --features release-models --bin pbrain-figrid
# The resulting pbrain-figrid.exe accepts both START 15 and START 20.
```

Normal crate builds retain their existing public assets. This GitHub release does
not publish a new crates.io package. Stop/progress APIs are off unless installed.

## Exact INT8 embedding format

`NGCB8E1\0` stores dim=32, rank=8 and 199827 rows, an FP32 bias, scales 32/64/64,
32 little-endian INT16 channel offsets, the row-major signed INT8 table, then the
9*32 INT16 head and 9*32*8 INT16 factors. Integer header fields are little-endian
32-bit. Reconstruct as `i16(table[row,channel]) + offset[channel]`.
Delta updates subtract table entries directly because offsets cancel.
The file is 6,399,748 bytes. The old FP32 master is 25,588,248 bytes; established
runtime weights were INT16. This is mixed INT8 embeddings / INT16 head, not an
all-INT8 evaluator or a second lossy quantization.

```text
ngcb-convert --exact-i8 model.ngcb model.ngcb8
```

Conversion applies the established INT16 quantizer and exhaustively verifies
reconstruction. A channel wider than 255 fails. The pbrain accepts `.ngcb8` via
`FIGRID_CODEBOOK_WEIGHTS` or `FIGRID_CODEBOOK_WEIGHTS_<RULE>`; per-rule external
models require `NORU_CODEBOOK_EVAL_SCALE_<RULE>`. A global external model suppresses
embedded per-rule defaults. Set its scale with `NORU_CODEBOOK_EVAL_SCALE`.

Use `ExactI8Codebook::from_bytes`, not `CodebookWeights::from_bytes_auto`, for this
format, and `Searcher::search_codebook_eval_exact_i8` for inference. Execution timing
can vary despite exact arithmetic; no speedup is claimed.
