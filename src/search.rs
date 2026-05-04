/// 알파-베타 탐색기 (NNUE 평가)
///
/// - 반복 심화 (Iterative Deepening)
/// - 알파-베타 가지치기
/// - 위협 탐지 기반 수 정렬 (4목/열린3 감지)
/// - 킬러 무브 + 히스토리 휴리스틱
/// - 시간 제한

use crate::board::{Board, GameResult, Move, Stone, BOARD_SIZE, NUM_CELLS};
use crate::eval::IncrementalEval;
use crate::heuristic::{scan_line, DIR};
use crate::transposition::{Bound, TranspositionTable, TtStats};
use crate::vct::{classify_move, search_vct, ThreatKind, VctConfig, THREAT_KIND_COUNT};
use noru::network::NnueWeights;
use std::time::{Duration, Instant};

const INF: i32 = 1_000_000;
const WIN_SCORE: i32 = 999_000;

/// Root VCT 기본 시간 예산 (time_limit이 없을 때 사용). 아레나 등 고정 depth 탐색용.
const ROOT_VCT_BUDGET_MS: u64 = 150;
/// Root VCT 최대 재귀 깊이 (공격-수비 쌍).
const ROOT_VCT_DEPTH: u32 = 14;
/// Root VCT가 턴 예산에서 차지할 비율. 5s 턴 → VCT 625ms, 30s 턴 → 3.75s.
const ROOT_VCT_BUDGET_FRACTION: u32 = 8;
/// Root VCT 예산 상한 (α-β 시간 확보). 2초.
const ROOT_VCT_BUDGET_CAP_MS: u64 = 2_000;
/// Root VCT 예산 하한 (너무 짧으면 TT warmup도 못 함).
const ROOT_VCT_BUDGET_FLOOR_MS: u64 = 100;

/// α-β TT 버킷 수 = 2^N. 18 → 262 144 버킷 = 8 MB.
/// 0.6.1 진단(2026-04-27)에서 16 bits(2 MB)로는 displaced 28.5% / always-replace
/// 사용 2.1%로 collision-driven eviction 발생. 18 bits면 같은 게임에서
/// displaced 5~10% 예상. Piskvork 대회 메모리 ≥350 MB 여유 안에서 무리 없음.
const TT_BUCKET_BITS: u32 = 18;

/// Aspiration windows: 이전 iteration score 주변 좁은 window. depth ≥ 이 값
/// 부터 적용. 1~3 ply는 score 변동이 커서 widening cost가 절약 효과 상회.
const ASPIRATION_MIN_DEPTH: u32 = 4;
/// 초기 window half-width (centipawn). 너무 작으면 widening 자주, 크면 효과
/// 작음. 검증된 chess engine 50~100 정도. 우리 BCE eval scale 기준 50.
const ASPIRATION_INITIAL_DELTA: i32 = 50;

/// Quiescence lite 최대 ply. depth==0 도달 시 NNUE static eval 반환 대신,
/// 강제수(즉시승/즉시방어/오픈4/더블4/4-3/오픈4 차단)만 이 ply 한도까지
/// 추가 탐색해서 horizon effect 완화. 일반 무브는 stand-pat. Codex 권장
/// (2026-04-26): "win/must-block/open-four/double-four/four-three 정도만
/// 2-4 ply 제한". 너무 크면 leaf 폭발, 작으면 효과 미미.
const QSEARCH_MAX_PLY: u32 = 4;

/// Threat-gated LMR (Late Move Reductions): non-PV / non-killer / non-forcing
/// 무브를 r ply 줄여서 빠르게 본 뒤 fail-high 시 full depth로 재탐색.
/// "naive LMR -43%p" 회귀의 원인은 강제수까지 reduce했기 때문 → tier 기반
/// gating으로 위협 라인은 절대 줄이지 않는다. 5초 예산에서 한 ply 더 들어가는
/// 효과가 직접적인 승률 향상.
const LMR_MIN_DEPTH: u32 = 3;
const LMR_MIN_MOVE_IDX: usize = 3;

/// IIR (Internal Iterative Reduction): TT-miss 노드 + non-PV + 충분히 깊을 때
/// 1 ply 줄여서 빠르게 본다. TT-miss는 좋은 PV 무브를 모르는 상태라 정상
/// search 시 ordering이 약함 → cutoff 비효율. 1 ply 줄여 빠르게 끝내고
/// store된 entry로 다음 iteration에서 PV 확보. chess engine에서 검증된
/// cheap 기법 (~+30 elo).
const IIR_MIN_DEPTH: u32 = 4;

/// LMP (Late Move Pruning): 얕은 비-PV 노드에서 move_idx가 임계값 이상인
/// 비-forcing / 비-killer 무브 skip. count 기반이라 razoring/futility가
/// 실패했던 NNUE eval scale 좁음 문제 회피. tier 정렬 끝에 있는 quiet
/// move는 좋은 후보일 가능성 매우 낮음.
const LMP_MIN_DEPTH: u32 = 1;
const LMP_MAX_DEPTH: u32 = 3;
const LMP_BASE: usize = 8;
const LMP_PER_DEPTH: usize = 4;

/// 탐색 결과
pub struct SearchResult {
    pub best_move: Option<Move>,
    pub score: i32,
    pub depth: u32,
    pub nodes: u64,
}

/// 탐색기
pub struct Searcher {
    pub nodes: u64,
    /// TT cutoff 횟수 — probe()로 가져온 entry가 depth/bound 충족해서
    /// 즉시 score 반환한 횟수. TT 효과 측정의 핵심 지표.
    pub tt_cutoffs: u64,
    killers: [[Option<Move>; 2]; 64],
    history: [[i32; NUM_CELLS]; 2],
    deadline: Option<Instant>,
    aborted: bool,
    /// α-β 노드 결과 캐시. 같은 포지션을 여러 번 탐색 안 하도록.
    /// search 호출 사이에 보존되어 iterative deepening 다음 iteration에서
    /// 이전 iteration의 PV/cutoff 정보를 그대로 활용.
    tt: TranspositionTable,
}

impl Searcher {
    pub fn new() -> Self {
        Self {
            nodes: 0,
            tt_cutoffs: 0,
            killers: [[None; 2]; 64],
            history: [[0; NUM_CELLS]; 2],
            deadline: None,
            aborted: false,
            tt: TranspositionTable::new(TT_BUCKET_BITS),
        }
    }

    /// TT 진단 카운터 반환 — search() 끝난 직후 호출 가능.
    pub fn tt_stats(&self) -> TtStats {
        self.tt.stats()
    }

    /// TT 점유율 — `(non-empty depth_pref, non-empty always_replace, 총 bucket)`.
    pub fn tt_occupancy(&self) -> (usize, usize, usize) {
        self.tt.occupancy()
    }

    /// 반복 심화 탐색
    pub fn search(
        &mut self,
        board: &mut Board,
        weights: &NnueWeights,
        max_depth: u32,
        time_limit: Option<Duration>,
    ) -> SearchResult {
        self.nodes = 0;
        self.tt_cutoffs = 0;
        self.aborted = false;
        self.killers = [[None; 2]; 64];
        self.history = [[0; NUM_CELLS]; 2];
        self.deadline = time_limit.map(|d| Instant::now() + d);
        // TT 진단 카운터도 리셋 — 한 search() 호출이 한 측정 단위.
        self.tt.reset_stats();
        // 다음 search 호출이 새 게임 포지션부터일 수 있으므로 TT 비움.
        // 같은 search() 호출 내 iterative deepening 사이에는 TT가 유지되어
        // 깊은 iteration이 얕은 iteration의 cutoff/PV 정보를 그대로 활용.
        self.tt.clear();

        // Root VCT: 짧은 시간 안에 강제 승리 수열을 찾으면 α-β 건너뜀.
        // Dynamic budget — 턴 예산의 1/ROOT_VCT_BUDGET_FRACTION (cap/floor 적용).
        let vct_budget = match time_limit {
            Some(d) => (d / ROOT_VCT_BUDGET_FRACTION)
                .max(Duration::from_millis(ROOT_VCT_BUDGET_FLOOR_MS))
                .min(Duration::from_millis(ROOT_VCT_BUDGET_CAP_MS)),
            None => Duration::from_millis(ROOT_VCT_BUDGET_MS),
        };
        let vct_cfg = VctConfig {
            max_depth: ROOT_VCT_DEPTH,
            time_budget: Some(vct_budget),
        };
        if let Some(seq) = search_vct(board, &vct_cfg) {
            if let Some(&first) = seq.first() {
                return SearchResult {
                    best_move: Some(first),
                    score: WIN_SCORE,
                    depth: seq.len() as u32,
                    nodes: self.nodes,
                };
            }
        }

        let mut best_result = SearchResult {
            best_move: None,
            score: 0,
            depth: 0,
            nodes: 0,
        };

        // Incremental NNUE state — 탐색 시작 시 한 번 full refresh, 이후
        // make_move/undo_move와 쌍으로 push/pop해서 매 leaf에서 full
        // compute_active_features를 돌지 않고 Accumulator forward만 수행.
        let mut inc = IncrementalEval::new(weights);
        inc.refresh(board, weights);

        // PV-move priority: the best move from iteration depth-1 becomes the
        // first move we try at iteration depth. Combined with PVS + Aspiration,
        // this drastically reduces re-search cost.
        let mut prev_best: Option<Move> = None;
        let mut prev_score: Option<i32> = None;

        for depth in 1..=max_depth {
            // Aspiration windows: depth ≥ 4 부터 이전 iteration score 주변
            // 좁은 [s-delta, s+delta] 로 시작. fail-low/high 시 재탐색.
            // 정확한 score는 보통 좁은 window 안 → 첫 시도에서 cutoff 효과
            // 극대화. window 빗나갈 때만 widening.
            let mut alpha_init: i32;
            let mut beta_init: i32;
            let aspirate = depth >= ASPIRATION_MIN_DEPTH && prev_score.is_some();
            if aspirate {
                let s = prev_score.unwrap();
                alpha_init = s - ASPIRATION_INITIAL_DELTA;
                beta_init = s + ASPIRATION_INITIAL_DELTA;
            } else {
                alpha_init = -INF;
                beta_init = INF;
            }

            let mut alpha = alpha_init;
            let mut beta = beta_init;
            let mut delta = ASPIRATION_INITIAL_DELTA;
            let mut iter_result: (Option<Move>, i32) = (None, 0);

            // Aspiration re-search loop. fail-high/low 마다 window widen.
            loop {
                iter_result = self.root_pvs_iteration(
                    board, weights, &mut inc, depth, alpha, beta, prev_best,
                );
                if self.aborted {
                    break;
                }
                let score = iter_result.1;
                if !aspirate {
                    break;
                }
                if score <= alpha {
                    // fail-low: alpha 더 낮춤
                    delta = (delta * 2).min(INF / 4);
                    alpha = (alpha - delta).max(-INF);
                    if alpha == -INF {
                        // 한 번 더 시도하면 full window라 break해도 됨,
                        // 그 결과는 다음 iteration에서 사용. 여기선 break 후
                        // 다시 -INF/INF로.
                        beta = INF;
                        // re-search with full window once
                        iter_result = self.root_pvs_iteration(
                            board, weights, &mut inc, depth,
                            -INF, INF, prev_best,
                        );
                        break;
                    }
                } else if score >= beta {
                    delta = (delta * 2).min(INF / 4);
                    beta = (beta + delta).min(INF);
                    if beta == INF {
                        alpha = -INF;
                        iter_result = self.root_pvs_iteration(
                            board, weights, &mut inc, depth,
                            -INF, INF, prev_best,
                        );
                        break;
                    }
                } else {
                    // window 안 → OK
                    break;
                }
            }

            if self.aborted {
                break;
            }

            let (best_move, score) = iter_result;
            best_result = SearchResult {
                best_move,
                score,
                depth,
                nodes: self.nodes,
            };

            if score.abs() > WIN_SCORE - 100 {
                break;
            }

            prev_best = best_move;
            prev_score = Some(score);
        }

        best_result
    }

    /// 한 iteration의 root-level PVS 탐색.
    /// `[alpha_init, beta_init]` window 안에서 모든 root move를 탐색하고
    /// best move + alpha 반환. Aspiration loop의 inner step.
    fn root_pvs_iteration(
        &mut self,
        board: &mut Board,
        weights: &NnueWeights,
        inc: &mut IncrementalEval,
        depth: u32,
        alpha_init: i32,
        beta_init: i32,
        prev_best: Option<Move>,
    ) -> (Option<Move>, i32) {
        let mut alpha = alpha_init;
        let beta = beta_init;
        let mut best_move: Option<Move> = None;

        let mut moves = self.order_moves(board, 0);
        if let Some(pv) = prev_best {
            if let Some(pos) = moves.iter().position(|&(m, _)| m == pv) {
                if pos != 0 {
                    moves.swap(0, pos);
                }
            }
        }

        for (move_idx, &(mv, is_forcing)) in moves.iter().enumerate() {
            // TT prefetch — same trick as in alpha_beta: warm the child's
            // TT bucket while make_move + accumulator delta runs.
            let next_zob = board.zobrist
                ^ crate::board::zobrist_stone_key(board.side_to_move, mv)
                ^ crate::board::ZOBRIST_SIDE;
            self.tt.prefetch(next_zob);

            board.make_move(mv);
            inc.push_move(board, mv, weights);

            let is_killer = self.killers[0][0] == Some(mv) || self.killers[0][1] == Some(mv);

            let score = if move_idx == 0 {
                -self.alpha_beta(board, weights, inc, depth - 1, 1, -beta, -alpha)
            } else {
                let reduction = lmr_reduction(depth, move_idx, is_forcing, is_killer);
                let reduced_depth = (depth - 1).saturating_sub(reduction);
                let mut null = -self.alpha_beta(
                    board, weights, inc, reduced_depth, 1, -alpha - 1, -alpha,
                );
                // LMR re-search (same null window): reduced 결과가 alpha 넘으면
                // full depth로 다시 본다 — 진짜 fail-high인지 검증.
                if !self.aborted && reduction > 0 && null > alpha {
                    null = -self.alpha_beta(
                        board, weights, inc, depth - 1, 1, -alpha - 1, -alpha,
                    );
                }
                if !self.aborted && null > alpha && null < beta {
                    -self.alpha_beta(board, weights, inc, depth - 1, 1, -beta, -alpha)
                } else {
                    null
                }
            };

            inc.pop_move();
            board.undo_move();

            if self.aborted {
                break;
            }

            if score > alpha {
                alpha = score;
                best_move = Some(mv);
            }
        }

        (best_move, alpha)
    }

    fn alpha_beta(
        &mut self,
        board: &mut Board,
        weights: &NnueWeights,
        inc: &mut IncrementalEval,
        depth: u32,
        ply: usize,
        mut alpha: i32,
        beta: i32,
    ) -> i32 {
        self.nodes += 1;

        // 시간 체크 (1024 노드마다)
        if self.nodes & 127 == 0 {
            if let Some(deadline) = self.deadline {
                if Instant::now() >= deadline {
                    self.aborted = true;
                    return 0;
                }
            }
        }

        // 게임 종료 체크
        match board.game_result() {
            GameResult::BlackWin | GameResult::WhiteWin => {
                return -(WIN_SCORE - ply as i32);
            }
            GameResult::Draw => return 0,
            GameResult::Ongoing => {}
        }

        // 깊이 도달 → quiescence lite로 forcing line만 마저 풀기.
        // stand-pat = NNUE static eval. 강제수가 stand-pat을 깰 때만 확장.
        if depth == 0 {
            return self.qsearch(board, weights, inc, 0, ply, alpha, beta);
        }

        // === TT lookup ===
        // 같은 zobrist key + 충분한 depth로 이미 탐색했다면 재사용.
        // bound 종류에 따라 alpha/beta cutoff 가능.
        let original_alpha = alpha;
        let tt_hit = self.tt.probe(board.zobrist);
        let mut tt_move: Option<Move> = None;
        if let Some(entry) = tt_hit {
            tt_move = if entry.best_move == u16::MAX {
                None
            } else {
                Some(entry.best_move as Move)
            };
            if entry.depth as u32 >= depth {
                let cached = entry.score;
                match entry.bound {
                    Bound::Exact => {
                        self.tt_cutoffs += 1;
                        return cached;
                    }
                    Bound::Lower if cached >= beta => {
                        self.tt_cutoffs += 1;
                        return cached;
                    }
                    Bound::Upper if cached <= alpha => {
                        self.tt_cutoffs += 1;
                        return cached;
                    }
                    _ => {}
                }
            }
        }

        // === IIR (Internal Iterative Reduction) ===
        // TT-miss + 깊은 비-PV 노드는 1 ply 줄여서 본다. ordering이 약한
        // 노드라 정상 깊이로는 cutoff 비효율. 줄여서 빨리 끝내고 store된
        // entry가 다음 iteration의 PV 가이드 역할.
        let is_pv = beta - alpha > 1;
        let depth = if depth >= IIR_MIN_DEPTH && tt_move.is_none() && !is_pv {
            depth - 1
        } else {
            depth
        };

        let mut moves = self.order_moves(board, ply);
        if moves.is_empty() {
            return 0;
        }

        // TT-best move를 맨 앞으로 (PVS에서 가장 큰 cutoff 효과).
        if let Some(tt_mv) = tt_move {
            if let Some(pos) = moves.iter().position(|&(m, _)| m == tt_mv) {
                if pos != 0 {
                    moves.swap(0, pos);
                }
            }
        }

        let mut best_score = -INF;
        let mut best_move_at_node: Option<Move> = None;
        let side = board.side_to_move as usize;

        // PVS + Threat-gated LMR:
        // order_moves[0] = PV 예측 → full window 탐색.
        // 이후 모든 무브는 null-window로 빠르게 본 뒤 fail-high만 full re-search.
        // LMR 추가: 비-PV / 비-killer / 비-forcing 무브는 reduction r ply 줄여서
        // 본다. 줄여서도 alpha 넘으면 full depth로 재탐색. tier 기반 gating으로
        // 강제수는 절대 reduce하지 않아 horizon effect 유지.
        for (move_idx, &(mv, is_forcing)) in moves.iter().enumerate() {
            let is_killer = ply < 64
                && (self.killers[ply][0] == Some(mv) || self.killers[ply][1] == Some(mv));

            // === LMP (Late Move Pruning) ===
            // 비-PV / 비-forcing / 비-killer / 얕은 depth에서 move_idx가
            // 임계값 이상이면 quiet move skip. count-based라 eval 분포
            // 무관, 항상 trigger 보장. tier 정렬 끝의 quiet move는 좋은
            // 후보 가능성 매우 낮음.
            if !is_pv && !is_forcing && !is_killer
                && depth >= LMP_MIN_DEPTH && depth <= LMP_MAX_DEPTH
            {
                let lmp_threshold = LMP_BASE + LMP_PER_DEPTH * depth as usize;
                if move_idx >= lmp_threshold {
                    continue;
                }
            }

            // TT prefetch: hint the CPU to load the child node's TT bucket
            // into L1 while we run the (cache-cold) make_move + accumulator
            // delta below. The child's first action is a TT probe, so by
            // the time it gets there the line is already warm. Worth ~5-10%
            // search throughput on cache-bound positions.
            let next_zob = board.zobrist
                ^ crate::board::zobrist_stone_key(board.side_to_move, mv)
                ^ crate::board::ZOBRIST_SIDE;
            self.tt.prefetch(next_zob);

            board.make_move(mv);
            inc.push_move(board, mv, weights);

            let score = if move_idx == 0 {
                -self.alpha_beta(board, weights, inc, depth - 1, ply + 1, -beta, -alpha)
            } else {
                let reduction = lmr_reduction(depth, move_idx, is_forcing, is_killer);
                let reduced_depth = (depth - 1).saturating_sub(reduction);
                let mut null_score = -self.alpha_beta(
                    board, weights, inc, reduced_depth, ply + 1, -alpha - 1, -alpha,
                );
                if !self.aborted && reduction > 0 && null_score > alpha {
                    null_score = -self.alpha_beta(
                        board, weights, inc, depth - 1, ply + 1, -alpha - 1, -alpha,
                    );
                }
                if !self.aborted && null_score > alpha && null_score < beta {
                    -self.alpha_beta(board, weights, inc, depth - 1, ply + 1, -beta, -alpha)
                } else {
                    null_score
                }
            };

            inc.pop_move();
            board.undo_move();

            if self.aborted {
                return 0;
            }

            if score > best_score {
                best_score = score;
                best_move_at_node = Some(mv);
            }
            if score > alpha {
                alpha = score;
                self.history[side][mv] += (depth * depth) as i32;
            }
            if alpha >= beta {
                if ply < 64 {
                    self.killers[ply][1] = self.killers[ply][0];
                    self.killers[ply][0] = Some(mv);
                }
                break;
            }
        }

        // === TT store ===
        // bound 분류:
        //   - best_score <= original_alpha → fail-low (Upper bound, true value ≤)
        //   - best_score >= beta            → fail-high (Lower bound, true value ≥)
        //   - 그 외                          → Exact PV node
        let bound = if best_score <= original_alpha {
            Bound::Upper
        } else if best_score >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        // depth가 u8로 들어가니 saturate. 우리 max_depth ≤ 20이라 문제 없음.
        self.tt.store(
            board.zobrist,
            best_score,
            depth.min(255) as u8,
            bound,
            best_move_at_node,
        );

        best_score
    }

    /// Quiescence lite. 강제수만 확장해 horizon effect 완화.
    /// - stand-pat (NNUE static eval) 로 fail-high 빠른 cutoff
    /// - 즉시승/즉시방어/오픈4/더블4/4-3/오픈4 차단 만 후보
    /// - QSEARCH_MAX_PLY 도달 시 stand-pat 반환
    fn qsearch(
        &mut self,
        board: &mut Board,
        weights: &NnueWeights,
        inc: &mut IncrementalEval,
        qply: u32,
        ply: usize,
        mut alpha: i32,
        beta: i32,
    ) -> i32 {
        self.nodes += 1;

        if self.nodes & 127 == 0 {
            if let Some(deadline) = self.deadline {
                if Instant::now() >= deadline {
                    self.aborted = true;
                    return 0;
                }
            }
        }

        match board.game_result() {
            GameResult::BlackWin | GameResult::WhiteWin => {
                return -(WIN_SCORE - ply as i32);
            }
            GameResult::Draw => return 0,
            GameResult::Ongoing => {}
        }

        let stand_pat = inc.eval(weights);
        if qply >= QSEARCH_MAX_PLY {
            return stand_pat;
        }
        if stand_pat >= beta {
            return stand_pat;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let candidates = board.candidate_moves();
        if candidates.is_empty() {
            return stand_pat;
        }

        let (my, opp) = match board.side_to_move {
            Stone::Black => (&board.black, &board.white),
            Stone::White => (&board.white, &board.black),
        };

        // 0.6.9: opp_kind를 candidates 한 번만 스캔해서 캐시. 기존 코드는
        // (1) opp_has_five 검사로 N회 + (2) 루프 내 OpenFour 차단 검사로 N회
        // → 같은 classify_move(opp,my,mv)를 최대 2N회 호출. 캐싱으로 N회로.
        let opp_kinds: Vec<ThreatKind> = candidates
            .iter()
            .map(|&m| classify_move(opp, my, m))
            .collect();
        let opp_has_five = opp_kinds.iter().any(|&k| matches!(k, ThreatKind::Five));

        let mut forcing: Vec<(Move, i32)> = Vec::new();
        for (i, &mv) in candidates.iter().enumerate() {
            let opp_kind = opp_kinds[i];
            let my_kind = classify_move(my, opp, mv);

            // 즉시 승리는 must-block 여부와 무관하게 항상 우선.
            if matches!(my_kind, ThreatKind::Five) {
                forcing.push((mv, 1_000_000));
                continue;
            }

            if opp_has_five {
                // Must-block 모드: 상대 Five 차단수만.
                if matches!(opp_kind, ThreatKind::Five) {
                    forcing.push((mv, 900_000));
                }
                continue;
            }

            // 공격 강제수 — 분기 대신 packed table.
            let attack = QS_ATTACK_TABLE[my_kind as usize];
            if attack > 0 {
                forcing.push((mv, attack));
                continue;
            }

            // 상대 OpenFour 차단도 강제수 (캐시된 opp_kind 재사용).
            if matches!(opp_kind, ThreatKind::OpenFour) {
                forcing.push((mv, 700_000));
            }
        }

        if forcing.is_empty() {
            return stand_pat;
        }

        forcing.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        let mut best = stand_pat;
        for &(mv, _) in &forcing {
            board.make_move(mv);
            inc.push_move(board, mv, weights);
            let score = -self.qsearch(board, weights, inc, qply + 1, ply + 1, -beta, -alpha);
            inc.pop_move();
            board.undo_move();

            if self.aborted {
                return 0;
            }

            if score > best {
                best = score;
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                break;
            }
        }

        best
    }

    /// 수 정렬: 위협 탐지 → 킬러 무브 → 히스토리 휴리스틱.
    /// 반환은 (mv, is_forcing) — is_forcing은 LMR gating에 사용되는 위협 태그.
    ///
    /// Packs (score, is_forcing, mv) into a single u64 so the hot sort path
    /// runs on a primitive-integer slice (pdqsort kernel) instead of a
    /// struct-comparison lambda. On 30-50 candidate moves this saves ~30%
    /// of the order_moves time vs the previous `Vec<(Move, i32, bool)>`
    /// + `sort_unstable_by(|a,b| b.1.cmp(&a.1))` form. Search throughput
    /// gain ~3-5%.
    fn order_moves(&self, board: &Board, ply: usize) -> Vec<(Move, bool)> {
        let candidates = board.candidate_moves();
        let side = board.side_to_move as usize;

        let (my, opp) = match board.side_to_move {
            Stone::Black => (&board.black, &board.white),
            Stone::White => (&board.white, &board.black),
        };

        // Layout (highest → lowest bit):
        //   [bits 16..64]: score + SCORE_BIAS (i32 range easily fits 48 bits)
        //   [bit  9]      : is_forcing flag
        //   [bits 0..9]   : mv index (0..225 → 9 bits)
        const SCORE_BIAS: i64 = 1 << 30;
        const MV_MASK: u64 = (1 << 9) - 1;
        const FORCING_BIT: u64 = 1 << 9;

        let mut packed: Vec<u64> = candidates
            .into_iter()
            .map(|m| {
                let (s, f) = self.move_score_and_forcing(m, ply, side, my, opp);
                let score_u = (s as i64 + SCORE_BIAS) as u64;
                (score_u << 16) | (if f { FORCING_BIT } else { 0 }) | (m as u64)
            })
            .collect();

        // Descending order = best score first. sort_unstable on u64 hits
        // the optimized pdqsort code path directly.
        packed.sort_unstable_by(|a, b| b.cmp(a));

        packed
            .into_iter()
            .map(|p| {
                let mv = (p & MV_MASK) as Move;
                let f = (p & FORCING_BIT) != 0;
                (mv, f)
            })
            .collect()
    }

    /// 한 무브의 정렬 점수 + LMR-gating 용 is_forcing 동시 산출.
    /// is_forcing = 어느 쪽이든 OpenThree 이상 위협이면 true.
    /// → 이 무브는 LMR로 reduce하지 않는다.
    fn move_score_and_forcing(
        &self,
        mv: Move,
        ply: usize,
        side: usize,
        my: &crate::board::BitBoard,
        opp: &crate::board::BitBoard,
    ) -> (i32, bool) {
        let row = (mv / BOARD_SIZE) as i32;
        let col = (mv % BOARD_SIZE) as i32;

        let my_kind = classify_move(my, opp, mv);
        let opp_kind = classify_move(opp, my, mv);

        // 0.6.9: 분기 없는 packed-table tier scoring. TIER 상수 간 buffer가
        // ≥ 100 000으로 잡혀있어 max() 결과가 if-else 체인과 동일함을 보장.
        // (예: my OpenFour=8M > opp OpenFour=7M > my DoubleFour=6M ...)
        let attack_tier = MOVE_ATTACK_TABLE[my_kind as usize];
        let block_tier = MOVE_BLOCK_TABLE[opp_kind as usize];
        let tier_score = attack_tier.max(block_tier);

        let is_forcing = is_forcing_kind(my_kind) || is_forcing_kind(opp_kind);

        // Five 케이스 early-return: 나머지 score 합산을 절약 (TIER_WIN/BLOCK_WIN
        // 자체로 다른 tier와 충돌 안 함, killer/history 더해도 의미 없음).
        if matches!(my_kind, ThreatKind::Five) {
            return (TIER_WIN, true);
        }
        if matches!(opp_kind, ThreatKind::Five) {
            return (TIER_BLOCK_WIN, true);
        }

        let mut score = tier_score;

        if ply < 64 {
            if self.killers[ply][0] == Some(mv) {
                score += 80_000;
            } else if self.killers[ply][1] == Some(mv) {
                score += 40_000;
            }
        }
        score += self.history[side][mv].min(50_000);

        for &(dr, dc) in &DIR {
            let my_info = scan_line(my, opp, row, col, dr, dc);
            if my_info.count == 2 && my_info.open_front && my_info.open_back {
                score += 200;
            }
            let opp_info = scan_line(opp, my, row, col, dr, dc);
            if opp_info.count == 2 && opp_info.open_front && opp_info.open_back {
                score += 150;
            }
        }

        // 0.6.5 (2026-04-27): center bonus 삭제. quiet move ordering에서
        // 안쪽 우대 → 자기 돌 뭉침 + 코너 opening (Pela swap2 등) 대응 약함
        // 진단 결과. 14 - center_dist는 보드 끝까지 14 차이라 한 라인을
        // 지배. 삭제 후 quiet 무브는 killer/history/scan-line(open-2)만으로
        // tie-break.

        (score, is_forcing)
    }

}

// === Move ordering tier 점수 ===
// 각 tier 사이 buffer가 ≥ 100 000이라 어떤 history/killer/center 합산도
// 다른 tier와 절대 충돌하지 않음.
const TIER_WIN: i32 = 10_000_000;
const TIER_BLOCK_WIN: i32 = 9_000_000;
const TIER_OPEN_FOUR: i32 = 8_000_000;
const TIER_BLOCK_OPEN_FOUR: i32 = 7_000_000;
const TIER_DOUBLE_FOUR: i32 = 6_000_000;
const TIER_BLOCK_DOUBLE_FOUR: i32 = 5_000_000;
const TIER_DOUBLE_THREE: i32 = 4_000_000;
const TIER_BLOCK_DOUBLE_THREE: i32 = 3_000_000;
const TIER_CLOSED_FOUR: i32 = 1_500_000;
const TIER_BLOCK_CLOSED_FOUR: i32 = 1_400_000;
const TIER_OPEN_THREE: i32 = 1_000_000;
const TIER_BLOCK_OPEN_THREE: i32 = 900_000;

// === Branchless threat-score tables (0.6.9) ===
// `ThreatKind as usize` 인덱스. 순서는 vct.rs의 #[repr(u8)] discriminant와 일치:
//   0=None  1=ClosedFour  2=OpenThree  3=Five
//   4=OpenFour  5=DoubleFour  6=FourThree  7=DoubleThree
// 변경 시 vct.rs의 ThreatKind discriminant도 함께 수정해야 함.

/// Move ordering: 내 위협이 만드는 attack tier 점수.
const MOVE_ATTACK_TABLE: [i32; THREAT_KIND_COUNT] = [
    0,                  // None
    TIER_CLOSED_FOUR,   // ClosedFour
    TIER_OPEN_THREE,    // OpenThree
    TIER_WIN,           // Five
    TIER_OPEN_FOUR,     // OpenFour
    TIER_DOUBLE_FOUR,   // DoubleFour
    TIER_DOUBLE_FOUR,   // FourThree
    TIER_DOUBLE_THREE,  // DoubleThree
];

/// Move ordering: 상대 위협 차단 tier 점수.
const MOVE_BLOCK_TABLE: [i32; THREAT_KIND_COUNT] = [
    0,                          // None
    TIER_BLOCK_CLOSED_FOUR,     // ClosedFour
    TIER_BLOCK_OPEN_THREE,      // OpenThree
    TIER_BLOCK_WIN,             // Five
    TIER_BLOCK_OPEN_FOUR,       // OpenFour
    TIER_BLOCK_DOUBLE_FOUR,     // DoubleFour
    TIER_BLOCK_DOUBLE_FOUR,     // FourThree
    TIER_BLOCK_DOUBLE_THREE,    // DoubleThree
];

/// `is_forcing` 비트마스크. bit i set ↔ ThreatKind discriminant i가 forcing.
/// 현행 forcing 정의: ClosedFour, OpenThree, Five, OpenFour, DoubleFour, FourThree.
/// (DoubleThree는 제외 — 기존 동작 유지.)
const FORCING_MASK: u8 = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6);

/// qsearch attack-tier 점수 (내 위협). Five는 별도 처리(반환 즉시 cutoff).
const QS_ATTACK_TABLE: [i32; THREAT_KIND_COUNT] = [
    0,        // None
    0,        // ClosedFour — qsearch attack 아님
    0,        // OpenThree
    0,        // Five — caller 별도 처리
    800_000,  // OpenFour
    600_000,  // DoubleFour
    600_000,  // FourThree
    0,        // DoubleThree — qsearch attack 아님
];

#[inline]
fn is_forcing_kind(kind: ThreatKind) -> bool {
    (FORCING_MASK >> (kind as u8)) & 1 != 0
}

/// Threat-gated LMR reduction 계산.
/// 강제수 / killer / 첫 LMR_MIN_MOVE_IDX 무브 / 얕은 depth는 0 (안 줄임).
/// 그 외 비-forcing tier 0 무브: depth/idx에 따라 1~2 ply.
/// reduction은 depth-2를 넘지 않도록 cap (qsearch 직행 방지).
fn lmr_reduction(depth: u32, move_idx: usize, is_forcing: bool, is_killer: bool) -> u32 {
    if depth < LMR_MIN_DEPTH || move_idx < LMR_MIN_MOVE_IDX || is_forcing || is_killer {
        return 0;
    }
    let mut r = 1u32;
    if depth >= 6 {
        r += 1;
    }
    if move_idx >= 6 {
        r += 1;
    }
    r.min(depth.saturating_sub(2))
}

#[allow(dead_code)]
fn threat_priority(kind: ThreatKind, defending: bool) -> i32 {
    // 호환용 stub — 새 move_score는 inline tier로 직접 처리.
    let base = match kind {
        ThreatKind::Five => 1_000_000,
        ThreatKind::OpenFour => 500_000,
        ThreatKind::DoubleFour | ThreatKind::FourThree => 300_000,
        ThreatKind::DoubleThree => 200_000,
        ThreatKind::ClosedFour => 100_000,
        ThreatKind::OpenThree => 30_000,
        ThreatKind::None => 0,
    };
    // 수비는 공격보다 살짝 낮게 (내가 이기는 게 상대 이김 막는 것보다 우선).
    if defending {
        base * 9 / 10
    } else {
        base
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{to_idx, Board};
    use crate::features::GOMOKU_NNUE_CONFIG;

    #[test]
    fn test_search_finds_winning_move() {
        let mut board = Board::new();
        let weights = NnueWeights::zeros(GOMOKU_NNUE_CONFIG);

        // 흑이 4목 만든 상태 — 5번째 수를 찾아야 함
        board.make_move(to_idx(7, 3));
        board.make_move(to_idx(8, 3));
        board.make_move(to_idx(7, 4));
        board.make_move(to_idx(8, 4));
        board.make_move(to_idx(7, 5));
        board.make_move(to_idx(8, 5));
        board.make_move(to_idx(7, 6));
        board.make_move(to_idx(8, 6));

        let mut searcher = Searcher::new();
        let result = searcher.search(&mut board, &weights, 2, None);

        let winning_moves = [to_idx(7, 7), to_idx(7, 2)];
        assert!(result.best_move.is_some());
        assert!(
            winning_moves.contains(&result.best_move.unwrap()),
            "should find the winning move, got {:?}",
            result.best_move
        );
    }

    #[test]
    fn test_search_depth_1() {
        let mut board = Board::new();
        let weights = NnueWeights::zeros(GOMOKU_NNUE_CONFIG);
        let mut searcher = Searcher::new();
        let result = searcher.search(&mut board, &weights, 1, None);
        assert!(result.best_move.is_some());
    }
}
