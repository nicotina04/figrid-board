# figrid v1.2 — Claude Code Implementation Brief

> **Goal**: Beat Pela ≥50% in 100-game match by 2026-05-29 (Gomocup deadline).
> **Probability**: 55-65% with disciplined execution.
> **Reference**: Full report at `figrid_nnue_v1.2_report.md` (read only when ambiguous).

---

## Context (1 줄씩)

- **Engine**: figrid v0.6.10, Rust, NNUE eval lib `noru`, 16-thread WSL2 CPU, no GPU.
- **Current NNUE**: v52, 14336 sparse → 1D global accumulator (512) → MLP [128, 64], CReLU.
- **Existing assets** (활용해야 함): Pattern4-mini (4097-class threat cache, incremental, O(1) lookup), root VCT, forcing-move quiescence, AVX-512 VNNI.
- **Known limit**: 1D global accumulator는 9M positions에서 saturate (loss UP). architecture 갈아엎어야 함.
- **Strategy**: Phase A (fallback) + Phase B (MixNet-lite) + Phase C (search) 병행.

---

## Phase A (Day 1-7) — Fallback Baseline

### Tasks (in order)

1. **Pattern4 threat-feature injection on v52** [Day 1-3]
   - Pattern4-mini의 4097-class를 64-dim learned embedding으로 lookup
   - 기존 14336 sparse feature와 concat (= 14400 sparse + 64 dense)
   - noru의 `NnueConfig`에 dense input branch 추가 필요
   - Sanity test: 100 fixed positions에서 eval 호출 정상

2. **Rapfi-Mix9 binary teacher labeling** [Day 2-5, parallel]
   - Rapfi binary로 ~100k–300k positions에 (W, D, L) label 생산
   - **Distribution mix**: 60-70% Pela-vs-figrid + Pela-vs-Rapfi positions, 20-30% balanced/Swap2-like openings, ≤20% Rapfi-vs-Rapfi generic
   - **Quiet-position filtering**: Pattern4-mini로 `forcing_threat_count >= 2`인 위치 제외
   - Wall-clock: 16-thread × 250k nps × 3-5일

3. **Asymmetric loss training** [Day 5-7]
   - White-side false positive (lost position을 even/win으로 평가)에 weight 2-3×
   - 학습 코드 loss 함수에서 `if side == WHITE and label == LOSS and pred > 0: loss *= 2.5`
   - WDL 3-class softmax로 통일 (cp+WDL hybrid 폐기)

### Day-7 Go Criterion
Fixed Pela 30-game test suite에서 v52보다 나빠지지 않을 것. 이게 fallback ship-able baseline.

---

## Phase B (Day 7-21) — MixNet-lite

### Architecture Spec

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

### Training (PyTorch, separate from Rust)

- Use Phase A labels (Rapfi-Mix9 distilled, 100k-300k positions)
- Adam lr=1e-3, batch=128, ~200k iterations
- Loss: CE on policy + CE on WDL (asymmetric weighting from Phase A)
- Export: int16 codebook (binary lookup table) + int16 quantized heads

### Rust Integration

**Critical risk**: noru는 1D-sparse abstraction 기반. (C, 15, 15) tensor 처리 SIMD path가 없을 가능성 높음. 두 옵션:
1. **noru fork** with 2D tensor inference path. 4주 budget 추가 1주 risk.
2. **Bypass noru**, write inference directly with `std::arch::x86_64::*` intrinsics (vpdpbusd for VNNI). **권장.**

**Day-14 checkpoint**: Codebook lookup + per-cell aggregation + region pooling + Star Block + heads의 Rust inference path가 v52+Pattern4 baseline보다 valid set에서 better. 안 되면 DWConv layer skip하고 minimum viable Mixnet-lite로 끝.

### Day-21 Stop-Loss
Pela 100-game match에서 ≥45% (50% 도달 시그널). 안 되면 Phase B 동결, Phase A fallback ship 준비.

---

## Phase C (Day 7-28, parallel) — Search-side

### Tasks (priority order)

1. **Leaf VCF/VCT via Pattern4-mini gate** ★ [Day 7-14]
   - α-β leaf node에서:
     ```
     if pattern4.high_volatility(pos):  // ≥2 forcing threats
         result = vcf_solver(pos, max_depth=20)
         if result.is_terminal:
             return result.value
     return nnue_eval(pos)
     ```
   - VCF solver는 forcing-four sequence만 따라가는 dedicated mini-search (DFPN or threat-space)
   - Pattern4-mini가 gate 비용을 거의 0으로 만듦
   - **Expected +40~80 Elo. figrid 고유 advantage. v1.0/v1.1에서 underweight.**

2. **Continuation history** [Day 14-17]
   - 1-ply countermove + 2-ply follow-up history tables
   - Update formula: `cont_history += bonus - cont_history * |bonus| / MAX_HISTORY` (gravity)
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

---

## Stop-Loss Decision Tree

```
Day 3:  Phase A injection 동작? → No: debug, B 대기. Yes: continue.
Day 7:  Phase A baseline ≥ v52? → No: A polish, B 1주 연기. Yes: B 시작.
Day 14: MixNet-lite > v52+Pattern4? → No: DWConv skip, lite-est. Yes: continue.
Day 21: vs Pela ≥45%? → No: B 동결, fallback prep. Yes: ship target locked.
Day 28: vs Pela ≥50%? → Yes: v0.7 ship. No: v0.6.11 (Phase A + Phase C) ship.
```

---

## Test Commands (구체화 필요)

```bash
# Build
cargo build --release --features avx512vnni

# Pela 100-game match (assumes pisqpipe protocol)
./benchmark/run_match.sh figrid pela --games=100 --time=2s --swap-sides

# Self-play sibling test (regression detection)
./benchmark/sibling_match.sh v52 vN --games=200

# Pela fixed-position test suite (Phase A Day-7 criterion)
./benchmark/fixed_position_eval.sh --suite=pela_30 --engine=figrid

# SPSA tuning round
./benchmark/spsa_round.sh --params=lmr,futility,iir,vct_budget --games=300
```

(실제 figrid repo의 benchmark 스크립트 이름은 다를 수 있음. 적절히 매핑.)

---

## Output Artifacts (Day 28 ship)

- **If success path** (vs Pela ≥50%): `figrid v0.7` with MixNet-lite NNUE + Phase C search improvements
- **If fallback path**: `figrid v0.6.11` with v52 + Pattern4 injection + Rapfi-distilled labels + Phase C search improvements

Either way: **Phase C work는 ship됨**. Phase B 실패해도 search-side gain은 보존.

---

## Do NOT Attempt (4-week budget waste)

- Full Mixnet-Small clone (모든 head + dynamic conv 포함). 4주 안 됨. → MixNet-lite만.
- E(2)-equivariant conv. Boundary가 corner/edge symmetry 깨뜨림.
- Multi-scale conv (3+5+9). Incremental update 폭증.
- MoE phase-gating. Cache coherency 파괴.
- Transformer/Mamba/GNN. NNUE incremental update 깨짐.
- AlphaZero-style policy MCTS. α-β advantage 손실.
- KataGomo b28c512nbt teacher (GPU 필요). Rapfi binary로 충분.
- Custom AMX path. Sapphire Rapids 미만 호환성 risk.
- 9M+ from-scratch retraining without distillation. 8 cycles에서 saturation 증명됨.

---

## When Ambiguous

Refer to `figrid_nnue_v1.2_report.md`:
- Section A: Rapfi internals (codebook structure, training pipeline)
- Section B: Saturation diagnosis (5 hypotheses ranking)
- Section E: Other techniques (Stockfish SFNNv9-v13 lessons, etc.)
- Section G: Probability decomposition

자체 판단 필요 시: **"이 변경이 Phase A/B/C 중 어디에 fit?"** 으로 자문. 분류 안 되면 post-deadline.
