# figrid v1.2 — Final Cross-Validated Implementation Roadmap (Complete Edition)

> **Status**: v1.2 (3-way LLM cross-validation 통과, complete edition)
> **Updates from v1.0/v1.1**: 확률 70% → 60%, full Mixnet-Small clone → MixNet-lite, leaf VCF/VCT 격상, Star Block 격상, day-level stop-loss 추가, asymmetric loss 추가
> **Target**: Gomocup 2026 (deadline 2026-05-29), Pela 100-game match ≥50%
> **Audience note**: 작성자는 alpha-beta search, NNUE, TT, quantization 등에 깊이 익숙하다고 가정. 본문은 영어 기술 용어 + 한국어 narrative 혼용.

---

## TL;DR

- **Pela는 4주 안에 잡을 수 있다 — 단, "Rapfi clone 완성"이 아니라 "MixNet-lite + Pattern4 hybrid + leaf VCF"라는 축소형 hybrid path로**. v1.0의 full Mixnet-Small clone 권고는 noru 라이브러리 (1D-sparse abstraction) 호환 비용을 underweight했음. ChatGPT와 Gemini가 독립적으로 같은 축소형을 권고함.

- **Rapfi의 진짜 영업비밀은 architecture가 아니라 training pipeline이다**. Mix9는 KataGomo (b28c512nbt)가 30 × RTX4090 × 6개월로 생성한 30.8M positions에서 ResNet 6b128f teacher로부터 knowledge-distilled (75% distillation + 25% true label)된 결과물. CPU-only로 from-scratch 학습으로 따라잡으려는 시도는 비대칭 게임이며 4주에 불가능 — **Rapfi binary를 teacher로 쓰는 distillation이 가장 현실적 shortcut**.

- **Honest probability that figrid beats Pela ≥50% in a 100-game match within 4 weeks of well-executed work: 55-65%** (v1.0 70% → 하향). Best-case ceiling은 "Pela + 100~250 Elo" (=1600~1750 freestyle Elo, SwineBox/Hewer/Tito 구간). Rapfi tier (2625+)는 4주는커녕 4개월에도 불가능. **Ship-it 권고**: deadline 시도 push, 단 "Pela 제거를 mandatory milestone, Rapfi-tier는 long-term roadmap"으로 분리.

- **Day-level stop-loss**: Day 3, 7, 14, 21, 28에 명시적 go/no-go 기준. Phase B (MixNet-lite) 실패 시 Phase A (Pattern4-injected v52) fallback ship.

---

## A. Deep Rapfi Source-Code / Architecture Analysis (Featured Section)

이것이 본 보고서의 가장 가치 있는 부분이며, figrid의 다음 단계 결정의 근거 전체를 여기에 고정한다.

### A.1 Repository Layout and Toolchain

`github.com/dhbloo/rapfi`는 4개의 sub-project로 구성된다 (`.gitmodules` 확인):

| Sub-project | Path | Role |
|---|---|---|
| **Rapfi** | `Rapfi/` | C++17 alpha-beta engine + NNUE inference |
| **Trainer** | `Trainer/` → `dhbloo/pytorch-nnue-trainer` | PyTorch training, exports `bin-lz4` weights |
| **Networks** | `Networks/` → `dhbloo/rapfi-networks` | 무료 CC0 라이선스 weight 배포소 |
| **Gomocalc** | `Gomocalc/` | WebAssembly 데모 GUI |

빌드는 CMake presets — `x64-clang-Native`, `x64-clang-AVX2`, `x64-clang-VNNI` 등 분리. 2025-Gomocup 릴리스 노트는 "AVX-VNNI, AVX-512, AVX-512 VNNI 지원 추가, weight size & memory footprint 감소"를 명시. 즉 **Rapfi는 figrid가 이미 하고 있는 AVX-512 VNNI 최적화를 거의 동일한 시점에 동일한 instruction set으로 채택한 상태**이며, 거기서는 더 이상 차별화 여지가 거의 없다.

핵심 weight format header는 `pytorch-nnue-trainer` README에서 직접 확인 가능 — magic `0xacd8cc6a = crc32("gomoku network weight version 1")`, arch_hash, rule_mask (1=gomoku, 2=standard, 4=renju), boardsize_mask. **figrid의 noru weight format을 호환되게 만들 필요는 없으며**, 그 대신 Rapfi binary를 그대로 teacher로 호출하는 게 더 빠르다.

### A.2 NNUE Topology — Mixnet (Mix6 → Mix7 → Mix8 → Mix9)

ICLR 2025-withdrawn submission `arXiv:2503.13178` (Jin, Duan, Hang)에 모든 architecture 디테일이 공개됨. **이게 figrid가 모방해야 하는 청사진이다.**

**Input**: one-hot binary `x ∈ {0,1}^(2×H×W)` (current player + opponent stones). **No global accumulator.**

**Step 1 — Line decomposition.** 모든 셀 `(i,j)`마다 길이-11 directional patterns 4개:
- `L^(0,1)` 가로, `L^(1,0)` 세로, `L^(1,1)` 정대각, `L^(1,-1)` 역대각
- 가능한 패턴 수 N = Σᵢ₊ⱼ≤5 3^(i+1+j) = **397,488** (보더 포함)

**Step 2 — Mapping network → Codebook.**
- 5-layer Dir Conv (3×3 kernel with non-zero weights only along the chosen direction; 가로/세로용 `M_hv`와 대각용 `M_di` 두 가지 weight set로 분리)
- Layers 사이에 1×1 point-wise conv 끼워넣기 + skip connection
- Internal channels `M`, output channels `C`
- Training 후 **397,488 패턴을 모두 enumerate하여 lossless하게 lookup table `f_CB ∈ R^(N×C)`로 export**. 이 codebook이 Rapfi-networks repo의 weight file 본체.
- `int16` quantization, scale factor 32, 값 클램프 [-16, 16]. (논문 §A.3)

**Step 3 — Aggregation per cell.**
```
f_(i,j) = ReLU( M_hv(L⁰¹) + M_hv(L¹⁰) + M_di(L¹¹) + M_di(L¹⁻¹) )
```
→ Feature map `F ∈ R^(C×15×15)`. **이게 per-cell accumulator**. (figrid의 "1D global 512-dim"과 정반대 구조.)

**Step 4 — Depth-wise 3×3 Conv on first C/2 channels** (나머지 절반은 identity)
```
F'(k,i,j) = Σ_{m,n=1..3} F(k, i+m-2, j+n-2) · W(c,m,n),  k ∈ [1, C/2]
F'(k,i,j) = F(k,i,j),                                     k ∈ [C/2+1, C]
```

**Step 5 — Heads** (Figure 4 in arXiv 2503.13178):

*Policy head*
- AvgPool → Linear → ReLU → Linear → produces (weights, bias) for **dynamic point-wise conv** applied to first P channels of F'
- 1×1 conv → raw policy `π̂ ∈ R^(15×15)`
- Ablation table 3에서 dynamic conv 제거 시 policy loss 1.213 → 1.431, MCTS-2s에서 -118 ELO

*Value head*
- F'을 3×3 region chunks로 분할 (4 corner + 4 edge + 1 center)
- 각 chunk → AvgPool → **Star block** (linear → ReLU → linear → multiply pairs → linear → ReLU; multiplication-pooling kernel trick)
- 4 group features + global mean을 Concat → 3-layer MLP → `(W, D, L) ∈ R³`

**Configurations** (논문 Table 1):

| Variant | M | C | P | V | CB params | FF params | Storage |
|---|---|---|---|---|---|---|---|
| Mixnet **Small** | 64 | 32 | 16 | 32 | 14,160k | 37k | 28.4 MiB |
| Mixnet **Medium** | 128 | 64 | 32 | 64 | 28,320k | 146k | 54.7 MiB |
| Mixnet **Large** | 256 | 128 | 64 | 128 | 56,640k | 580k | 111 MiB |

**중요 데이터 포인트 (논문 §5.3, Table 1):** α-β search throughput은 Small **428k nps**, Medium 257k, Large 104k — 즉 **Rapfi의 NNUE eval cost는 single-threaded 100k–400k positions/sec 범위**이며, ResNet-baseline은 4b64f이 1.7k nps로 약 250× 느림. Large가 raw eval은 정확하지만 incremental update 비용이 incremental advantage를 상쇄해서 Medium이 종종 α-β에서 최강. Gomocup 2024-2025 출전 production weight는 **Medium configuration (M=128, C=64) 추정**.

### A.3 Training Pipeline — 가장 결정적 차별점

**This is where figrid's 8-cycle retraining campaign was structurally outgunned.**

| 항목 | Rapfi Mix9 (논문 §4) | figrid v52→v72 |
|---|---|---|
| Data source | KataGomo (b28c512nbt) self-play, 30 × RTX4090 × 10시간/일 × 6개월 | 93k–9M Rapfi-vs-Rapfi self-play, CPU-only |
| Position count | **30.8M** filtered | 644k effective (9M까지 가도 saturate) |
| Label format | (B, V_t=(p_w,p_l,p_d) ∈ R³, π_t ∈ R^(H×W)) — **WDL + policy distribution** | cp + WDL hybrid, no policy |
| Teacher | **ResNet 6b128f** (KataGo-trained) | None (학습은 from-scratch) |
| Distillation | **75% distillation labels + 25% true label**, cross-entropy + KD | N/A |
| Optimizer | Adam lr=1e-3, β=(0.9, 0.999), batch=128, **600k iterations** | (similar order) |
| Quantization | 16-bit codebook (scale 32), 16-bit DWConv + accumulation in 32-bit, **8-bit matmul + 32-bit accumulate** in heads | int16 throughout (noru) |
| SIMD | AVX2 baseline, ~4× over scalar (논문 §A.3); 2025 release adds AVX-512 VNNI | AVX-512 VNNI already done |

**Key insight.** Rapfi의 30.8M dataset은 KataGomo (AlphaZero pipeline)이 **policy distribution (π)을 함께 생산**한다. Mixnet은 cross-entropy loss로 그 policy를 학습하며 (논문 Eq. 6), value loss는 categorical 3-class WDL (Eq. 7)이다. **figrid는 policy head가 아예 없다.** 이것이 search-side에서 NN-policy-driven move ordering / depth modulation을 못하는 이유.

**Loss decomposition from paper Table 2 (반드시 읽을 것):**
- Mixnet Medium: train value 0.7791, policy 1.213
- ResNet 6b96f baseline: value 0.7727, policy **1.035** ← Mixnet은 policy에서 못 따라감
- ResNet 20b256f: value 0.7264, policy 0.7979

즉 Mixnet의 raw evaluation accuracy는 ResNet 4b64f ~ 6b96f 사이; **search depth로 policy gap을 메운다.** figrid의 v72 loss 1.03 → "Mixnet Medium보다 약간 나은 value 영역, 하지만 policy 공간에서는 비교 불가" 정도의 신호로 해석해야 한다.

### A.4 Search Stack — Rapfi α-β (논문 §A.4 + 2023 release notes)

Rapfi의 α-β는 PVS variant + 다음 enhancements:

1. **VCF quiescence at leaf** — chess engine의 quiescence와 동일 역할. 공격자만 forcing-four(공격 후 단일 응수만 가능한 수)를 두며 horizon effect 회피. figrid의 quiescence(Five/OpenFour/DoubleFour/FourThree)는 broader — **forcing-four-only를 추가 옵션으로 분리하면 false-extension이 줄어들 가능성 있음**.
2. **Transposition table** with first-move ordering. (4-bucket lockless 추정; Stockfish/Embryo와 유사)
3. **Futility pruning, late move reduction (LMR), null move pruning, singular extension** — 모두 chess-style. Gomoku에서 null-move는 보통 약하지만 Rapfi가 적용한다고 명시. (figrid는 null-move 미적용 — Section E에서 권고.)
4. **NN policy로 move ranking + dynamic depth adjustment** — Mix9 정책 분포로 LMR margin과 ordering을 조정. **figrid에는 정책망 자체가 없으므로 즉시 도입 불가.**
5. **SPSA-tuned search params** (FineFishing tradition) — 2023 release notes에 명시. **figrid에 즉시 도입 가능, 효과 +20~50 Elo 예상.**
6. **Database integration** (Yixin format) — 2023+ Rapfi는 known book + database lookup 사용. Tournament 환경에서는 무시 가능 (Gomocup 룰상 제한).

### A.5 Concrete Gap Inventory (figrid 0.6.10 vs Rapfi Mix9), Ranked by Likely Strength Contribution

| Rank | Rapfi feature missing in figrid | Estimated Elo impact | 4-week feasibility |
|---|---|---|---|
| 1 | **Per-cell feature map (no global accumulator)** with line-pattern codebook | +200~400 Elo (constructive — figrid 현 NNUE를 갈아엎는 baseline) | **Required core change (Phase B)** |
| 2 | **Knowledge distillation from a stronger teacher** (Rapfi binary로 직접 가능) | +100~200 Elo | High (Phase A) |
| 3 | **Policy head + policy-driven move ordering / LMR** | +50~150 Elo | Medium (Phase B) |
| 4 | **WDL categorical value head with 3-class softmax** | +30~80 Elo (figrid의 cp+WDL hybrid보다 calibration 우수) | Easy (Phase A/B) |
| 5 | **Star block + value grouping** (kernel-trick non-linearity) | +30~50 Elo | Easy (Phase B 필수) |
| 6 | **Dynamic point-wise conv in policy head** | +30~80 Elo (논문 ablation: -118 Elo when removed) | Medium (Phase B optional, post-deadline) |
| 7 | **SPSA tuning of all search-side magic numbers** | +20~50 Elo | Easy (Phase C) |
| 8 | **Continuation history / countermove history** (Stockfish-style) | +10~30 Elo | Easy (Phase C, 1~2일) |
| 9 | **Forcing-four-only quiescence option** (vs figrid 현 broader QS) | +5~20 Elo | Trivial (Phase C) |
| 10 | **Leaf VCF gate via Pattern4-mini** ★ (v1.2에서 격상) | +40~80 Elo, **figrid 고유 advantage** | Easy (Phase C #1) |
| 11 | **Null move pruning** (adapted to Gomoku — passive opponent test) | +10~30 Elo, but high risk in tactical positions | Medium (post-deadline) |
| 12 | **Database / book integration** | Tournament 환경에서 무관 | N/A |

**Ranking 총평.** 1+2+10이 4주 budget의 80%를 차지해야 한다 (Phase A/B/C 핵심). 3~6은 새로운 architecture에 자연스럽게 follow되므로 sub-week. 7~9는 marginal gain (Phase C 부가).

---

## B. Diagnosis of Saturation (왜 9M 데이터에서 loss가 *오른다*?)

5가지 가설을 각각 점수화한다 (0–10, 10이 가장 그럴듯).

| # | Hypothesis | Score | Reasoning |
|---|---|---|---|
| 1 | **Information bottleneck (1D global 512-dim → 225-cell board)** | **9** | Information-theoretic 관점에서, 225 cells × 4 directions × 11-stone window의 위치-특이적 정보를 단일 1D 512-vector로 압축하는 것은 불가능하지 않으나 매우 비효율적. 모든 cell의 contribution이 동일 vector에 더해지므로 *position-specific* spatial localization이 사라짐. Rapfi가 per-cell로 가는 정확한 이유. |
| 2 | **Feature collisions in 14336 sparse features with hash-collision components** | 6 | "3×3/5×5 conv-kernel hash"와 "compound" 같은 hashed features는 collision이 부분적으로 학습 시 noise로 작용. 다만 14336 → 512 dense projection이 collision을 어느 정도 흡수하므로 이 자체로 loss UP은 설명 안 됨. |
| 3 | **Label noise from heuristic depth-4 / depth-16 in opening positions** | **8** | Opening positions에서 진짜 value가 거의 0 (균등 분포)에 가까운데, search가 임의의 한쪽으로 ±200cp를 산출하면 이는 *반-임의 라벨*. 30k 단위로는 통계적으로 wash out되지만, 9M 단위에서는 이 noise가 model이 fit하려는 *진짜* 신호와 동급 magnitude가 되어 loss가 오를 수 있다. SF NNUE pipeline이 `early-fen-skipping=28`을 쓰는 이유와 동일한 문제 (Stockfish PR #5149에서 명시적으로 다룸). |
| 4 | **Distribution mismatch (Rapfi-self-play vs Pela positions)** | **8** | Rapfi-vs-Rapfi 게임은 *Rapfi의 evaluation function이 매력적이라고 평가한* 위치들로 편향됨. Pela style positions는 hand-pattern 평가가 만든 다른 manifold에 위치. 9M 단위에서는 이 manifold-gap이 model의 *Pela 인근에서의 일반화 성능*을 갉아먹는다. |
| 5 | **Capacity-data Pareto: 512-dim acc + [128,64] 못 fit** | 7 | 9M positions에서 model이 fit하기 위한 effective parameter 요구량은 capacity 512×128 + 128×64 + ... ≈ 100k params로는 부족할 수 있다. 다만 "loss 오름"은 capacity 부족만으로는 일반적으로 *plateau*이지 *상승*은 아니므로, 이는 #3, #4와 결합되어야 설명 가능. |

**진단 결론.** Loss UP은 **단일 원인이 아니라 #1 (구조적 bottleneck) × #3 (label noise) × #4 (distribution mismatch)의 곱셈적 효과**다. 단순히 데이터를 더 넣어도, 더 큰 model로 가도 같은 한계에 부딪힌다. 이는 acc 1024로 갔을 때 noise가 들어왔다는 v71 결과와 정확히 일치. **architectural change 없이는 어떤 데이터/capacity tweak도 +30 Elo 이상 가져오지 못한다.**

가장 확실한 검증 실험 (1일): 같은 v52 weights로 학습하되, **input의 location index를 추가 feature로 인코딩**(9 그룹 broadcast: corner/edge/center class)하고 다시 학습. Loss가 즉시 떨어지면 hypothesis #1이 dominant. 떨어지지 않으면 #3/#4가 dominant. 이 결과에 따라 다음 단계가 갈린다.

---

## C. Recommendations (4-Week Budget) — v1.2 Phase A/B/C Plan

v1.0의 "Top-3 architecture-first" 구조에서 **Phase A/B/C 병행 + day-level stop-loss** 구조로 재편. 핵심 변경: full Mixnet-Small clone이 아니라 **MixNet-lite** (codebook은 살리고 DWConv는 optional). ChatGPT/Gemini 둘 다 독립 수렴한 처방.

### Phase A (Day 1-7) — Fallback Baseline 확보 [필수]

목표: v0.6.11 fallback ship-able 상태. 새 architecture가 실패해도 출전 가능한 안전망.

**구성:**

1. **Pattern4-mini threat-feature injection on v52**
   - Pattern4-mini의 4097-class를 64-dim learned embedding lookup
   - 기존 14336 sparse feature와 concat (= 14400 sparse + 64 dense)
   - **Why first**: figrid가 이미 Pattern4-mini를 보유 → 추가 인프라 비용 거의 0. NN capacity가 위협 인식 학습에 낭비되지 않음. **Rapfi/Stockfish/Lc0 어디에도 이 정도로 정교한 hand-crafted threat을 NN input으로 직접 주입하는 사례 없음 → 진짜 차별화.**

2. **Rapfi-Mix9 binary teacher labeling**
   - ~100k–300k positions에 (W, D, L) label 생산
   - Distribution mix:
     - 60-70%: Pela-vs-figrid + Pela-vs-Rapfi positions
     - 20-30%: Balanced/Swap2-like openings
     - ≤20%: Rapfi-vs-Rapfi generic
   - **Quiet-position filtering** (arXiv 2412.17948 권고): Pattern4-mini로 `forcing_threat_count >= 2`인 위치 제외. figrid의 Pattern4-mini를 noisy-position filter로 재사용 — 이미 만들어진 코드라 추가 비용 0.
   - Wall-clock: 16-thread × 250k nps × 3-5일

3. **Asymmetric loss for white-side**
   - 0/15 white-side observation을 직접 처방 (Gemini 단독 발견)
   - White 평가에서 false positive (lost position을 even/win으로 평가)에 weight 2-3×
   - 학습 코드에서: `if side == WHITE and label == LOSS and pred > 0: loss *= 2.5`
   - WDL 3-class softmax로 통일 (cp+WDL hybrid 폐기). Rapfi 정신 + Lc0 0.30 contempt 학습.

| Field | Value |
|---|---|
| Effort | 1.0 CPU-week |
| Risk | Low |
| Expected ROI | +30~80 Elo (단독, v52 기준) |
| Day-7 success criterion | Fixed Pela 30-game test suite에서 v52보다 나빠지지 않음 |

### Phase B (Day 7-21) — MixNet-lite 구현 [메인]

목표: per-cell feature map을 유지하는 축소형 Rapfi 구조. **Full clone이 아님.**

**Architecture:**

```
Input: stones[2][15][15] (one-hot: my_stones, opp_stones)
   │
   ├─ Line decomposition: 4 directions × length 11 per cell
   │     Patterns enumerated: N = 397,488 (border-aware 3-state index packing)
   │
   ├─ Codebook lookup (offline-precomputed, int16, scale=32, clamp=[-16,16])
   │     M_hv: weights for horizontal/vertical (boundary-aware)
   │     M_di: weights for diagonals (boundary-aware)
   │     Output per direction: f_dir ∈ R^C
   │
   ├─ Per-cell aggregation (incremental update friendly)
   │     F[c, i, j] = ReLU( M_hv(L_horiz) + M_hv(L_vert) + M_di(L_d1) + M_di(L_d2) )
   │     Shape: F ∈ R^(C × 15 × 15)         ← 핵심: 1D global accumulator NOT 사용
   │     C = 16 or 32 (start with C=16, scale up only if budget allows)
   │
   ├─ [Optional, Day 14 stop-loss] DWConv 1 layer × C/2 channels
   │     If integration unstable by Day 14: SKIP this layer
   │
   ├─ Region pooling (9 regions: 4 corners + 4 edges + 1 center)
   │     Each region: 5×5 area AvgPool → R^C
   │     Concatenate 9 regions: R^(9C)
   │
   ├─ Star Block (REQUIRED, +40~70 Elo, AVX-friendly)
   │     x → [Linear(2C) || Linear(2C, ReLU)] → element-wise multiply → Linear(C)
   │     Output: R^C
   │
   ├─ Value head: 3-layer MLP (C → 2C → 2C → 3) → softmax(W, D, L)
   │
   └─ Policy head: 1×1 conv (R^(C×15×15) → R^(1×15×15)) → 225 logits
        Used for move ordering only. NO dynamic conv (post-deadline).
```

**핵심 제약:**
- **DWConv는 optional**. Day 14까지 안정 안 되면 codebook + per-cell map + light heads로 끝.
- **Star Block은 필수** (Gemini가 단독으로 격상 — +40~70 Elo, 0.5주, AVX 친화적이라 짚은 것)
- **Codebook은 int16 quantized + offline-precomputed 정적 lookup table**. 학습은 PyTorch로 따로.
- **noru fork 또는 별도 inference path가 필요할 가능성 높음**. 이게 Phase B의 가장 큰 risk. 두 옵션:
  1. noru fork with 2D tensor inference path. 4주 budget 추가 1주 risk.
  2. Bypass noru, write inference directly with `std::arch::x86_64::*` intrinsics (vpdpbusd for VNNI). **권장.**

**Training (PyTorch, separate from Rust):**
- Phase A의 Rapfi-Mix9 distilled labels (100k-300k positions) 사용
- Adam lr=1e-3, batch=128, ~200k iterations
- Loss: CE on policy + CE on WDL (asymmetric weighting from Phase A 적용)
- Export: int16 codebook (binary lookup table) + int16 quantized heads

| Field | Value |
|---|---|
| Effort | 2.0 CPU-weeks |
| Risk | Medium-High (noru integration이 핵심 변수) |
| Expected ROI | +100~200 Elo (Phase A baseline 위에서) |
| Day-21 stop-loss | Pela 100-game match가 ≥45%가 아니면 Phase B 동결, Phase A fallback ship 준비 |

### Phase C (Day 7-28, parallel) — Search-side Surgical [필수]

목표: NNUE만으로 안 되는 tactical gap 직접 보강. ChatGPT/Gemini 둘 다 독립적으로 강조.

**구성 (priority order):**

1. **Leaf VCF/VCT via Pattern4-mini gate** ★ [Day 7-14]
   - α-β leaf node에서 Pattern4가 high-volatility (≥2 forcing threats)를 인식하면 NNUE 호출 대신 dedicated VCF solver 진입
   - 의사코드:
     ```
     if pattern4.high_volatility(pos):  // ≥2 forcing threats
         result = vcf_solver(pos, max_depth=20)
         if result.is_terminal:
             return result.value
     return nnue_eval(pos)
     ```
   - VCF solver는 forcing-four sequence만 따라가는 dedicated mini-search (DFPN or threat-space)
   - Pattern4-mini의 4097-class incremental cache가 gate 비용을 거의 0으로 만듦
   - **Expected +40~80 Elo. figrid 고유 advantage. v1.0/v1.1에서 underweight.**

2. **Continuation history** (1-ply countermove + 2-ply follow-up history tables) [Day 14-17]
   - Stockfish-style. Update formula (gravity): `cont_history += bonus - cont_history * |bonus| / MAX_HISTORY`
   - Move ordering bonus + LMR margin reduction
   - **Expected +20~30 Elo**

3. **SPSA tuning** [Day 17-25]
   - Tune: LMR margin, futility margin, aspiration window, IIR depth, VCT root time-budget, leaf VCF threshold
   - 200-500 self-play games per round, 3-5 rounds
   - OpenBench-style or chess-tune library
   - **Expected +20~50 Elo**

4. (Optional) **Singular extension** [Day 25-28 if time]
   - TT move가 reduced-search에서 다른 모든 move보다 명확히 강하면 1 ply 더
   - **Expected +10~20 Elo**

| Field | Value |
|---|---|
| Effort | 1.0 CPU-week (parallel with B) |
| Risk | Low |
| Expected ROI | +40~80 Elo |
| Critical | Leaf VCF는 Phase B 결과와 무관하게 반드시 진행. v1.0이 underweight했던 부분. |

---

## D. What NOT To Try in 4 Weeks

| 후보 | 거부 사유 |
|---|---|
| **Full Mixnet-Small clone with all bells and whistles** | v1.0 권고였으나 v1.2에서 "MixNet-lite"로 축소. noru 4주 budget 초과 risk. |
| **E(2)-equivariant 8-fold rotation/mirror conv** | Gomoku는 boundary conditions가 corner/edge/center symmetry를 깨뜨림. Rapfi가 명시적으로 `M_hv` (가로/세로용)와 `M_di` (대각용) 두 개로 나눈 이유가 정확히 이것 — full E(2) equivariance는 expressiveness loss. Implementation은 1주, debug는 무한정. |
| **Multi-scale conv (3+5+9 kernel mix)** | Receptive field는 5-stack DWConv로 이미 길이 11에 도달. 5×5/9×9는 incremental update시 3~9× 더 많은 cell 재계산 (per-stone update가 4×11에서 4×27로 폭증). Rapfi가 3×3 only를 선택한 이유. |
| **Per-line accumulator를 별도 디자인으로** | v1.1에서 재평가됨. Phase B의 alternative path로 검토 가치 있음. v1.2에서는 Phase B 권고 architecture로 통합. |
| **MoE with phase gating** | 좋은 idea지만 4주에 (training infra + gating function tuning + per-expert distillation)을 할 수 없다. Stockfish chess MoE 작업 (arXiv 2401.16852)은 +120 Elo였지만 그것도 기존 baseline이 매우 강한 상태에서 weeks of GPU. Cache coherency 파괴 risk도 큼. |
| **HalfKA-style relational features** | Chess의 KingxAll-pieces 매핑은 chess-specific (piece type 다양성). Gomoku는 piece가 1종이므로 HalfKA에 직접적 대응이 없음 (last-move pivot은 가능하나 inflated feature space는 학습 비용만 증가). |
| **Transformer / Mamba / GNN backbone** | CPU-only no-GPU 환경에서는 inference cost가 prohibitive. Lc0 BT4 (191M params)는 GPU에서만 의미. Rapfi 논문 결론에서도 "shallow network" 한계를 인정하면서 안 가는 길로 명시. |
| **AlphaZero-style policy MCTS** | α-β의 incremental update advantage를 잃음. Rapfi 자신이 MCTS variant를 시험한 후 α-β로 결정 (논문 §5.3 Figure 6). |
| **본인의 NNUE를 from-scratch 9M+로 다시 학습** | 8 cycles에서 saturation 확인됨. 같은 일을 더 하는 것은 budget 낭비. Distillation으로 가야 한다. |
| **Yixin / Embryo source 분석** | Embryo는 Stockfish-derivative이고 source가 일부만 공개 (Hexik/Embryo_engine은 binary release만). Yixin은 closed-source. 공개된 design 조각은 Rapfi보다 정보량이 적음. |
| **NUMA-aware lazy SMP rewrite** | WSL2 16-thread 환경에서는 NUMA 효과 거의 없음 (single-socket). 무시. |
| **Custom AMX path** | Sapphire Rapids 이상 필요. Rapfi 2025도 AMX는 추가하지 않음. CPU 호환성 문제로 deployment 위험. |
| **KataGomo b28c512nbt를 직접 teacher로** | GPU 필요. 4주 안에 access 어려움. Rapfi binary로 distillation 가능 (이미 Mix9가 KataGo로 distilled됨 → "transitive distillation"으로 충분). |

---

## E. Things You Haven't Mentioned (그리고 검색 결과가 직접 짚어준 것들)

### E.1 Stockfish 2024–2026 NNUE 진화 — figrid에 직접 이식 가능한 lessons

- **SFNNv9 (linrock, 2024)**: L1 size 1024 → **3072**. 핵심은 *pre-train → fine-tune 10-stage sequence*로 학습 안정성 확보. figrid가 단일 stage로 9M까지 늘리는 게 saturate한 vs SF가 multi-stage로 100B+ positions까지 가는 게 안 saturate한 차이. **권고**: figrid도 retrain 시 multi-stage curriculum (낮은 quality data로 wide pre-train → distilled labels로 fine-tune).
- **SFNNv10 (2026, Stockfish 18)**: **Threat Inputs** — `(piece_a_at_square_a, piece_b_at_square_b)` pairs where `b ∈ attack_set(a)`. Gomoku 대응: **(my_stone_at_square_a, my_stone_at_square_b) where they're co-linear within distance 4**. 이는 figrid의 "Compound" feature와 사실상 같은 idea의 잘 정제된 버전. SF18 release: +46 Elo. figrid가 Compound feature를 제대로 weight하면 +20~50 Elo 가능.
- **SFNNv13 (Nov 2025)**: L2 doubled 16→32 because **threat inputs reduced L1/accumulator cost enough to budget more L2**. 시사점: input feature engineering이 살아나면 head capacity tradeoff도 이동한다.
- **Continuation history**: jackk03의 talkchess 2024 thread가 implementation pitfall들을 정리. `cont_history += bonus - cont_history * |bonus| / MAX_HISTORY` (gravity). 1, 2, 4, 6 ply가 SF가 실제 사용. Gomoku에서 1-ply만 적용해도 +20~30 Elo 안전한 베팅.
- **Quantization**: SF는 input → int8, hidden activations → int8, output → int32. AVX-512 VNNI `vpdpbusd` 1개 명령어로 4× int8 dot product. **figrid noru가 int16 throughout이면 immediate 2× speed-up potential — 단 QAT (quantization-aware training)가 동반되어야 함** (arXiv 2509.22935 "Compute-Optimal QAT" 참조). 4주에는 무리, longer-term.

### E.2 Lc0 / BT4 — 모방 금지 목록 (CPU-only Gomoku에서)

Transformer encoder + smolgen은 GPU에서만 의미. 확실히 **figrid 4주에는 NOT-DO 리스트**. 다만 한 가지 lesson: **WDL output reformulation** (Lc0 0.30 contempt + WDL_mu) — 단순 cp 대신 W/D/L 3-class softmax를 model에서 직접 출력하면 **MCTS와 α-β 둘 다에서 calibration이 좋아지며 LMR/futility margin tuning이 쉬워진다**. figrid의 cp+WDL hybrid는 결국 둘 사이의 hybrid 매개변수 lambda_bce/lambda_mse를 추가로 tuning해야 하는 부담이 있다 — Rapfi처럼 **순수 3-class softmax로 통일**하면 +10~30 Elo 기대. (v1.2: Phase A에 통합됨)

### E.3 KataGo / KataGomo — Teacher 후보로서

`hzyhhzy/KataGomo`의 README는 명시적으로 "Gomoku-NNUE: A fast and small network ... Gomoku-NNUE is firstly created by me ([hzy]/gomoku_nnue/multiRules) and **improved by dblue and used in Rapfi**". 즉 Rapfi의 codebook NNUE는 KataGomo author와의 collaboration. 또한 `b28c512nbt` net (2024.5–2024.11 학습)이 Rapfi training data의 source. **figrid가 가능한 정신**:
1. 가장 강한 teacher = KataGomo b28c512nbt (GPU 필요, 일반 user는 access 어려움)
2. 그 다음 = Rapfi-Mix9 binary (CPU OK, 무료, license CC0 weights + GPL3 engine)
3. 그 다음 = AlphaGomoku (Kozarzewski) MCTS+ResNet (GPL, Gomocup binary)

**Pragmatic choice for figrid**: Rapfi-Mix9 binary as teacher. 그 다음 단계 (post-deadline)에서 KataGomo로 업그레이드.

### E.4 Recent 2024–2026 Gomoku/Renju papers

검색으로 확인된 2024–2026 Gomoku-direct NNUE 논문은 **Rapfi (arXiv 2503.13178, March 2025) 1편이 유일**. 외에는:
- arXiv 2503.21683 "LLM-Gomoku" — LLM이 Gomoku를 self-play로 학습 (engine strength irrelevant)
- arXiv 2309.01294 "AlphaZero Gomoku" — basic AlphaZero baseline, not competitive
- arXiv 2412.17948 "Study of the Proper NNUE Dataset" (Tan & Watkinson Medina) — Xiangqi-focused but **methodology directly transfers** to Gomoku (quiet-position filtering 권고, Phase A에 적용됨)
- KataGomo project (hzyhhzy github) — neural-net-only, GPU heavy, but hybrid NNUE+NN experiment failed ("not very success" per README)

**결론: 2024–2026의 SOTA로 figrid가 모방해야 할 architecture는 정확히 Rapfi Mix9 한 줄기로 압축됨.** 이는 어떻게 보면 figrid에게는 행운 — 검증된 단일 청사진이 완전히 공개되어 있다.

### E.5 noru-specific consideration

`lib.rs/crates/noru` 페이지는 "NnueConfig decouples feature_size, accumulator_size, hidden_sizes, and the activation function from the binary layout". **즉 noru 자체는 1D-global-accumulator를 가정하는 라이브러리이며, (C, H, W) per-cell tensor는 noru의 abstraction에 직접적으로 들어맞지 않는다.** 이게 Phase B의 hidden risk — noru를 fork하거나 별도 inference path를 작성해야 할 가능성. 4주 budget에 미리 계상 필요. (Pure Rust SIMD intrinsics (`std::arch::x86_64`)로 직접 작성하는 것이 가장 빠를 가능성 — Phase B 권장.)

### E.6 Other potentially-relevant search techniques

- **ProbCut (Buro 1996, Ehud's chess variant)**: shallow search로 대체로 cutoff가 날 위치를 미리 prune. Stockfish 사용 중. Gomoku에서 적용 사례는 명시적으로 없으나 잘 동작할 가능성 (threat-rich position에서는 risky). **Week-4 stretch (post-deadline)**.
- **History pruning**: history score < threshold이면 reduce/skip. Continuation history와 결합 시 +10~20 Elo. Easy. Phase C에 통합 가능.
- **Razoring**: lazy-eval pre-pruning at low depths. Stockfish 사용. Gomoku에서는 NNUE가 충분히 빠르므로 minor effect 예상.
- **Singular extension**: figrid는 미언급. Rapfi 사용. TT move가 다른 모든 move보다 명확히 강하면 그것만 1 ply 더 본다. Stockfish에서 +30 Elo. **Phase C optional**.
- **IIR (Internal Iterative Reduction)**: figrid 이미 사용. 충분.

### E.7 Conv-Light or Conv-Free Architectural Alternatives

본 보고서 v1에서 Section D ("don't try")에 묶었던 후보들 중 일부, 그리고 후속 brainstorm에서 발견된 후보들을 conv 사용량 관점에서 재평가한다.

| 후보 | NNUE incremental | Pattern4-mini 시너지 | 신규성 | 4주 ROI 추정 | v1.2 처분 |
|---|---|---|---|---|---|
| **Threat-feature injection** | ✓ | 매우 강함 | 중간 (figrid 고유) | +20~60 Elo | **Phase A에 통합** |
| **Star Block** (multiplication pooling) | ✓ | 약함 | 낮음 (Mixnet 표준) | +40~70 Elo | **Phase B 필수 항목으로 격상** |
| **Asymmetric loss for white-side** | ✓ (학습만) | 약함 | 중간 (white 0/15 직접 처방) | +20~40 Elo | **Phase A에 통합** |
| **Leaf VCF gate via Pattern4** | ✓ | 매우 강함 | 중간 (figrid 고유) | +40~80 Elo | **Phase C #1로 격상** |
| Auxiliary-task multi-head | ✓ (학습만) | 강함 | 중간 (KataGo) | +20~50 Elo | post-deadline (Phase B의 amplifier로 검토) |
| Position-encoded pointwise MLP | ✓ | 약함 | 낮음 | +30~80 Elo | Phase B의 9-region pooling으로 흡수 |
| Hyperedge / collinear-set | ✓ | 강함 | 낮음 | +30~70 Elo | post-deadline (Phase B의 alternative) |
| Per-line accumulator | ✓ | 강함 | 중간 | +50~100 Elo | Phase B core architecture로 흡수 |
| D4 weight tying (group conv 없이) | ✓ | 약함 | 중간 | +20~80 Elo | post-deadline |
| Linear attention along 4 lines | △ (sparse update tracking 필요) | 약함 | 높음 | unknown | post-deadline |
| Search-disagreement hard-negative mining | N/A (training pipeline) | 강함 | 중간 | +20~50 Elo | **Phase A의 quiet-filtering과 결합 권고** |

**가장 figrid-specific differentiation 큰 후보**: threat-feature injection (Phase A) + Leaf VCF gate (Phase C). 둘 다 Pattern4-mini의 4097-class incremental cache를 활용해야 가능. **이게 figrid 고유 advantage이자 Rapfi가 못 하는 영역**.

---

## F. Calibration Check — 어디까지 갈 수 있는가?

Gomocup Elo ratings (`gomocup.org/elo-ratings/`, "Best Versions Only") freestyle 기준:

| Rank | Engine (Best Version) | Elo |
|---|---|---|
| 1 | RAPFI 0.34.05 (2022) | **2625** |
| 2 | EMBRYO 0.6.4.2600 (2019) | 2437 |
| 3 | BARBAKAN 1.0 (2021) | 2321 |
| 4 | ALPHAGOMOKU (MK) 5.3.0 (2022) | 2256 |
| 5 | KATAGOMO 20210502 (2021) | 2254 |
| 6 | YIXIN 0.7.13 (2018) | 2192 |
| ... | ... | ... |
| 15 | HEWER 4.0.11.424 (2018) | 1817 |
| 16 | TITO 1.3.2 (2014) | 1796 |
| 22 | CARBON 2.4 (2017) | 1670 |
| 26 | **PELA 7.8 (2017)** | **1499** |

**Mix9 Rapfi 추정 Elo (2024 release)**: 2625 (2022 mix7-era) + 75~100 (mix7→mix8) + 20~55 (mix8→mix9) ≈ **2720~2780**.

**figrid가 4주 budget으로 도달 가능한 tier (best case, well-executed):**

| Tier | Elo range | 도달 조건 |
|---|---|---|
| Pela 미달 (현재 유지) | < 1500 | Architecture 안 바꾸고 search-side만 tuning |
| Pela 동등 (현 ship 시점에서 ~50% vs Pela) | 1450~1550 | 현재 v52 + SPSA tuning 정도 |
| Pela + epsilon (50%~60% vs Pela) | 1500~1650 | Phase A 단독 + Phase C 일부 |
| **★ "between Pela and Hewer/Tito"** | **1650~1800** | **Phase A + B + C fully executed** ← 현실적 best case |
| Hewer-tier (1800~) | 1800~1900 | Phase A+B+C + 추가 4주 (8주 총량) |
| Carbon/Pentazen tier | 1900~2200 | 별도 8~16주 + GPU 학습 access |
| Rapfi/Embryo tier | 2400+ | 별도 6개월~ + KataGomo-tier teacher |

**Anchor data points**:
- Rapfi 2018.02 (라피의 첫 NNUE 도입 직전 alpha-beta + classical eval) = 2020 Elo, 즉 *figrid의 4주 끝나도 Rapfi 7년 전 버전에는 못 미친다*.
- Pela 7.5 (2006)에서 7.8 (2017)까지 11년에 걸친 Elo 변화는 1492 → 1499로 사실상 stagnant — 즉 Pela는 **2010년대 알파-베타 + hand-pattern engine의 완성된 baseline**이며, figrid가 이미 6/30 (20%) 성능을 내고 있다는 것은 **bench Elo로 ~1300 → 1499 gap, 약 -130 Elo**.

**Realistic best-case ceiling for figrid v0.7 (May 29, 2026)**: **~1700 Elo, "between Pela and Hewer/Tito"**. 이는 Pela를 명확히 60%+로 이기지만 Embryo/Rapfi에는 손도 못 대는 tier.

**Why not higher?** Rapfi 2018.02도 알파베타 + 클래식 평가 시점에서 2020 Elo를 찍었다 — 즉 그것이 "alpha-beta + threat/pattern eval에 Mix-net 없이 도달 가능한 plateau"의 reasonable proxy. figrid가 4주 안에 거기까지 가려면 *Rapfi 6년분 search+eval tuning*을 압축해야 하는데, SPSA 한 번으로는 안 된다.

**Why not lower?** Rapfi-Mix9 distillation이 figrid의 80%를 *Rapfi 2018-tier evaluation accuracy*에 단번에 끌어올릴 잠재력이 있다. 그 80%가 1500 → 1700에 충분.

---

## G. Honest Probability — 60% (v1.0 70% → 하향 조정)

**Top recommendation (Phase A + B + C 통합) reasonable execution 시 Pela 100-game ≥50% 확률 = 60%**.

분해 (ChatGPT decomposition 채택):

| Path | P(Pela ≥50%) |
|---|---|
| 현재 v52에 search tuning만 | 20-30% |
| v52 + Pattern4 inj + Rapfi distill (Phase A only) | 35-45% |
| Gemini식 codebook → 기존 1D accumulator 직결 | 40-50% |
| **Phase A + B + C (v1.2 권고)** | **55-65%** |
| Full Mixnet-Small clone (v1.0 권고) | 65-70% if successful, but high impl risk |

**v1.0의 70%는 Phase B (full Mixnet) 성공 조건부에서는 가능하지만 unconditional은 over-confident**. Gemini 40%는 figrid가 이미 Pattern4-mini, root VCT, forcing quiescence를 보유한 점을 underweight. 60%가 honest median.

분해의 분해:
- P(Phase A 정상 작동, Day 7 success criterion 달성) = **85%**
  - Risk: Pattern4 4097-class를 64-dim embedding으로 lookup하는 inference path 정확성
- P(Phase B integrate 완료, Day 14 ≥ baseline) | Phase A = **65%**
  - Risk: noru가 (C, H, W) tensor 적합하지 않아 inference path 직접 작성 필요. 1주 추가 비용.
- P(Phase C가 Phase A/B 위에서 +40 Elo 이상) | Phase A+B = **80%**
  - Leaf VCF는 Pattern4 보유로 안전; SPSA는 검증된 기법
- P(통합 결과가 Pela 대비 50% 이상) | A+B+C = **~85%**
  - Pela 1499 Elo, best case 1700은 +200 Elo의 67% expected score
- 결합: 0.85 × 0.65 × 0.80 × 0.85 ≈ 0.38 (Phase B 성공 path)
- + Phase B 부분 성공 path (Phase A + C만) × 그것의 50% 도달률 ≈ 0.15
- + Phase A 단독 + Phase C path × 그것의 30% 도달률 ≈ 0.10
- **합계 ~0.60** (이게 honest 60%의 출처)

**1-σ downside scenarios** (왜 실패할 수 있나):
- noru fork 작업이 1.5주 잡아먹어서 Phase B가 week-3 끝까지 production 불가 → 새 NNUE 없는 상태로 마감 → Phase A fallback ship 시 ~35% vs Pela
- Rapfi distillation labels의 distribution이 Pela 위치와 너무 달라서 generalization gap 발생 → +50 Elo 정도만 (50% threshold 달성은 가능하지만 gap이 sober)
- 새 NNUE의 quantization int16→int8 SIMD path에서 numerical instability → search depth 손실이 evaluation 향상을 상쇄

**1-σ upside**:
- Phase A+B+C가 깔끔하게 결합되며 70% vs Pela (Hewer-tier 1700+에 근접)

---

## H. Day-Level Stop-Loss Schedule (v1.2 신규)

ChatGPT가 단독으로 격상시킨 management discipline. 4주 budget을 명시적 milestone으로 분할.

| Day | Milestone | Go criterion | No-go action |
|---|---|---|---|
| **3** | Phase A: Pattern4 injection 학습/추론 정상 작동 | Compile + 100-position eval pass | Phase A 디버깅, B 대기 |
| **7** | Phase A 완성: v0.6.11 fallback baseline | Fixed Pela 30-game suite에서 v52보다 나빠지지 않음 | Phase A polish, B 시작 1주 연기 |
| **14** | Phase B: MixNet-lite FP32 학습 + int16 quantization | v52+Pattern4 baseline보다 validation/test에서 유의미하게 좋음 | DWConv 단계 skip, codebook + light heads로 축소 |
| **21** | Phase B + C 통합: Pela 100-game match | ≥45% (50% 도달 시그널) | Phase B 동결, v0.6.11 fallback polish |
| **28** | Final ship | 100-game match ≥50% → v0.7 ship; 아니면 v0.6.11 + Phase C ship | — |

---

## H'. Ship-It-Now Check — Push, but Split the Goal

**Direct answer: Push through, but explicitly separate "Gomocup 2026 ship target" from "Rapfi-tier roadmap"**.

이유:
1. **Gomocup 2026 (May 29 deadline)에 figrid의 v0.6.10을 그냥 제출하는 것은 Pela보다 약한 상태로 등록됨을 의미** — 이미 v0.6.10도 Gomocup 2025에 "first version, really weak" 상태로 등록했으므로 같은 자리에서 정체 (Embarrassment 비용). Pela 분명히 잡는 것은 4주 budget으로 도달 가능한 의미 있는 마일스톤.
2. **현재 NNUE 캐피티 한계는 architectural** — 더 retraining cycle을 굴려도 의미 없다. 8 cycles + 9M corpus 결과가 그것을 정량적으로 증명. 따라서 *현재 architecture의 추가 마이크로 최적화는 budget 낭비*. 새 architecture로 전환은 어차피 해야 하는 일이므로 deadline을 강제로 forcing function으로 사용하는 것이 가장 효율적.
3. **그러나 "deadline 안에 Rapfi tier"를 노리지는 말 것**. 그것은 Gomocup 2027 또는 그 이후의 목표. 4주 안에 Rapfi tier를 노리면 Phase A+B+C 모두 부실하게 끝나서 *어떤 마일스톤도 명확히 달성하지 못하는 worst case*.

**Concrete ship plan**:
- Day 21: Pela 100-game 결과 확인
- < 45%: v0.6.11 + search tuning + telemetry ship
- 45-50%: v0.7-rc1 디버깅 1~2일, 안 되면 v0.6.11 ship
- ≥50%: v0.7 ship

**Long-term roadmap (Gomocup 2027 또는 이후 6~9개월)**:
1. KataGomo b28c512nbt 또는 더 강한 teacher 확보 (GPU access 또는 cloud rental ~$300의 hobbyist budget)
2. 30M+ position dataset (Pela-style + Rapfi-style mix), multi-stage curriculum (early-position skipping, quiet-position filtering)
3. Mixnet-Medium (M=128, C=64) — Rapfi production size
4. Star block + value group + dynamic conv 완전 구현
5. Lazy SMP + lockless TT + thread-voted root pondering
6. 6-month target: Embryo/Pentazen tier (~2200~2400 Elo)
7. 12-month target: Rapfi-Mix7 tier (~2400~2500 Elo). Rapfi-Mix9 (2700+)는 KataGomo-grade teacher data 확보 여부에 의존.

---

## Appendix: Red-Flag Items the Developer Should Verify Before Trusting This Report

- **Rapfi Mix9 production weight의 정확한 (M, C) configuration은 weight binary header에서 확인 필요** (이 보고서는 논문 §5 ablation 결과를 근거로 Medium 추정). Networks repo `dhbloo/rapfi-networks`에 LZ4 압축 binary가 있고 `arch_hash`로 구별 가능.
- **2025-Gomocup Rapfi 릴리스 노트의 "New NNUE architecture, smaller weight size"는 Mix10 또는 Mix9의 quantization 변경일 수 있음**. 현 시점 (2026-05-06)에서 2024-06-09 release 이후 Rapfi가 Mix9 → Mix10로 갔는지는 검색에서 명시 확인 안 됨 — 만약 Mix10이면 ablation/architecture 디테일이 더 진화했을 수 있다. **구현 시작 전 최신 Rapfi commit + paper의 v2/v3가 있는지 재확인**.
- **figrid의 self-play 측정에서 +13pp는 Pela 측정 0pp 변화와 모순되지 않으나**, "self-play sibling improvement"는 Pela 일반화에 대한 weak signal이다. Section B의 distribution mismatch 가설을 dataset가 실제로 풀고 있는지 검증하기 위해 **Pela-vs-figrid에서의 fixed test position set (예: 100 positions where Pela picked a specific move)**에서의 figrid eval correlation을 직접 metric으로 추적할 것을 권고. 이는 self-play sibling보다 의미 있는 진척도 지표.
- 본 보고서의 모든 정량적 Elo 추정 (예: "+200~400 Elo from per-cell architecture")은 Rapfi 자체 ablation (Table 2, 3, Figure 5/6) + Stockfish PR-level Elo gain reports의 inference이다. **figrid의 starting baseline이 Rapfi의 baseline (ResNet 4b64f, ~1800 self-play Elo)와 다르므로 Elo 변화량은 직접 transfer되지 않는다.** Pela 대비 측정만이 실제 ground truth.
- Yixin/Embryo 분석은 이 보고서에서 의도적으로 skip — 둘 다 closed/binary-only이며 Rapfi가 이미 더 강하고 source가 완전히 공개. Embryo가 Stockfish-derivative라는 점 외에 추가 정보 가치는 figrid의 4주 budget 내에서 거의 없다.

---

## v1.2 Provenance — 어떻게 이 보고서에 도달했나

본 보고서는 단일 LLM의 single pass가 아니라 다음 과정의 합성:

1. **v1.0 (Claude Opus 4.7 deep research)**: Rapfi gap 분석 + 진단 + 초기 Top-3 권고 + 70% 확률
2. **외부 cross-validation (ChatGPT Pro + Gemini Plus)**:
   - ChatGPT: "Compass 보고서 가장 타당, 단 70%는 over-confident → MixNet-lite로 축소 + day-level stop-loss"
   - Gemini: "Rapfi-style codebook 방향 정답, 단 full clone은 자살. Star Block, asymmetric loss, leaf VCF가 빠져있음 → 추가 권고"
3. **Three-way 합의 7개 항목 robust** (1D bottleneck, codebook, distillation, label noise, MoE 거부, Transformer 거부, Pela tier 도달 가능)
4. **발산 4개 항목 v1.2에서 결정**: full Mixnet → lite, leaf VCF 격상, Star Block 격상, asymmetric loss 격상, 70% → 60%
5. **본 v1.2 = v1.0 + ChatGPT phase 축소 처방 + Gemini 단독 발견 항목들의 합성**

가위바위보 cycle이 발생했다가 (v1.0 round) 정보 충분한 상태에서 무너진 결과 (v1.2 round)이며, 단일 LLM 누구도 이 보고서에 도달하지 못함. **사용자의 cross-validation discipline이 v1.2의 진짜 author**.

---

## Bottom Line for the Developer (한 문장)

> **"Burn the global accumulator, but build a *MixNet-lite* — not a full clone — with Rapfi-Mix9 distillation as your teacher and Pattern4-mini as both Phase A injection and Phase C leaf-VCF gate. Commit to day-level stop-loss. Pela 60% probability, MixNet-Medium은 deadline 이후로."**
