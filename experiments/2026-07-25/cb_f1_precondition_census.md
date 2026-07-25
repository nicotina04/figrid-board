# CB-F1 factored residual codebook precondition census

Date: 2026-07-25 (Asia/Seoul)

This census was performed read-only at
`dc0d9afae658113747e5666c3864b381cc971582`, before any CB-F1 prototype
or product-path change. Its purpose is to choose one architecture and freeze
the implementation gates; it is not a post-hoc promotion result.

## Frozen deployed representation

- source model:
  `models/gomoku_codebook_v1_swapclosed.json`
- source SHA-256:
  `42968FDAB01BA8CCD1DE3DED05C532E4B237DD47EEFFD7AE1C2F264D77BA7DA2`
- Pattern4 vocabulary: 4,266 IDs, including the RARE ID
- embedding dimension: 16
- pooled features: 9 regions x 16 = 144
- FM rank: 8
- deployed quantization:
  embeddings S32 i16, head/factors S64 i16, bias f32

| Part | Values | FP32 raw | Deployed payload |
|---|---:|---:|---:|
| embeddings | 68,256 | 273,024 B | 136,512 B |
| linear head | 144 | 576 B | 288 B |
| FM factors | 1,152 | 4,608 B | 2,304 B |
| bias | 1 | 4 B | 4 B |
| total | 69,553 | 278,212 B | 139,108 B |

Embeddings are 98.13% of the deployed quantized weight payload. The embedded
JSON is 1,410,562 B; its embeddings array alone occupies 1,384,819 B.

## Candidate census

### Exact duplicates

There are 4,097 unique quantized embedding rows. IDs `4096..4265` contain 170
copies of the same RARE row. Tail collapse alone is lossless but saves only
5,408 B, or 3.96% of the embedding payload.

### Color-swap orbit sharing

The 4,266 IDs form 2,148 color-swap orbits: 30 fixed points and 2,118 pairs.
No paired quantized rows are equal and no pair is an exact negation. A
swap-orbit base plus residual needs about 111.2 KiB and saves only about 18.6%
of the embedding payload, so it is dominated by the selected candidate.

### Low-rank approximation

The 4,266 x 16 quantized embedding matrix has full rank 16.

| Rank | Raw energy retained | Centered energy retained |
|---:|---:|---:|
| 8 | 71.39% | 69.79% |
| 12 | 87.59% | 86.98% |

On-demand rank-8 reconstruction would also add 128 multiplies and roughly 112
adds per token vector. It is neither exact nor a credible hot-path upper bound
for this card.

### Selected exact hierarchy

Reversal canonicalization preserves the 11-cell window's anchor at index 5.
The vocabulary census is:

- anchor empty: 2,019 IDs
- anchor mine: 1,123 IDs
- anchor opponent: 1,123 IDs
- RARE: 1 ID

For each component, use the integer ceil-midpoint of the class minimum and
maximum as a base:

```text
base = min + ceil((max - min) / 2)
residual[token, component] = embedding[token, component] - base[class, component]
```

With the four semantic classes, only two residuals exceed i8:

- token 188, component 14: -132
- token 585, component 14: +132

Making token 585 one fixed singleton class changes the relevant midpoint and
puts every residual in `[-128, 127]`. The selected representation is therefore:

```text
E_q(token, component)
  = Base_q[Class[token], component]
  + Residual_q[token, component]
```

with five i16 base rows, a u8 class map, and i8 residuals. It reconstructs all
68,256 deployed i16 values exactly.

| Selected payload part | Bytes |
|---|---:|
| 5 x 16 i16 bases | 160 |
| 4,266 u8 class IDs | 4,266 |
| 4,266 x 16 i8 residuals | 68,256 |
| factored embeddings | 72,682 |
| unchanged head/factors/bias | 2,596 |
| total quantized weights | 75,278 |

This is a 63,830 B reduction from 139,108 B: 45.89% for all quantized weights
and 46.76% for the embedding part.

## Decision

`GO_PREREGISTER_EXACT_HIERARCHY`.

CB-F1 will test only the five-class exact integer hierarchy. It will not train
new weights, approximate the frozen table, tie color-swap or D4 orbits, add a
lane-specific signal, change quantization scales, or alter the vocabulary.
Those operations would change more than one factor or overlap later cards.

