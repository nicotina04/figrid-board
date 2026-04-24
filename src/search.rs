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
use crate::vct::{classify_move, search_vct, ThreatKind, VctConfig};
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
    killers: [[Option<Move>; 2]; 64],
    history: [[i32; NUM_CELLS]; 2],
    deadline: Option<Instant>,
    aborted: bool,
}

impl Searcher {
    pub fn new() -> Self {
        Self {
            nodes: 0,
            killers: [[None; 2]; 64],
            history: [[0; NUM_CELLS]; 2],
            deadline: None,
            aborted: false,
        }
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
        self.aborted = false;
        self.killers = [[None; 2]; 64];
        self.history = [[0; NUM_CELLS]; 2];
        self.deadline = time_limit.map(|d| Instant::now() + d);

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
        // first move we try at iteration depth. Combined with PVS, this
        // drastically reduces re-search cost — if the PV is still best, the
        // null-window searches on the remaining moves almost all fail low.
        let mut prev_best: Option<Move> = None;

        for depth in 1..=max_depth {
            let mut best_move = None;
            let mut alpha = -INF;
            let beta = INF;
            let mut moves = self.order_moves(board, 0);

            if let Some(pv) = prev_best {
                if let Some(pos) = moves.iter().position(|&m| m == pv) {
                    if pos != 0 {
                        moves.swap(0, pos);
                    }
                }
            }

            for (move_idx, mv) in moves.iter().enumerate() {
                let mv = *mv;
                board.make_move(mv);
                inc.push_move(board, mv, weights);

                // Root PVS: first move uses the full window; every
                // subsequent move is proved-not-better via null window and
                // only re-searched with the full window on fail-high.
                let score = if move_idx == 0 {
                    -self.alpha_beta(board, weights, &mut inc, depth - 1, 1, -beta, -alpha)
                } else {
                    let null = -self.alpha_beta(
                        board,
                        weights,
                        &mut inc,
                        depth - 1,
                        1,
                        -alpha - 1,
                        -alpha,
                    );
                    if !self.aborted && null > alpha && null < beta {
                        -self.alpha_beta(board, weights, &mut inc, depth - 1, 1, -beta, -alpha)
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

            if self.aborted {
                break;
            }

            best_result = SearchResult {
                best_move,
                score: alpha,
                depth,
                nodes: self.nodes,
            };

            if alpha.abs() > WIN_SCORE - 100 {
                break;
            }

            prev_best = best_move;
        }

        best_result
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

        // 깊이 도달 → NNUE 정적 평가 (incremental accumulator의 forward만)
        if depth == 0 {
            return inc.eval(weights);
        }

        let moves = self.order_moves(board, ply);
        if moves.is_empty() {
            return 0;
        }

        let mut best_score = -INF;
        let side = board.side_to_move as usize;

        // PVS (Principal Variation Search):
        // order_moves[0] is our best guess for the PV. Search it with the
        // full [-beta, -alpha] window. For every later move, assume
        // order_moves got it right and first prove "this move is not
        // better than what we have" with a null window [-alpha-1, -alpha].
        // If the null-window search fails high (returns > alpha, < beta),
        // the move actually could be better — re-search with full window.
        // Large speedup when move ordering is good; costs a re-search
        // occasionally when ordering is wrong.
        for (move_idx, mv) in moves.iter().enumerate() {
            let mv = *mv;
            board.make_move(mv);
            inc.push_move(board, mv, weights);

            let score = if move_idx == 0 {
                -self.alpha_beta(board, weights, inc, depth - 1, ply + 1, -beta, -alpha)
            } else {
                let null_score = -self.alpha_beta(
                    board,
                    weights,
                    inc,
                    depth - 1,
                    ply + 1,
                    -alpha - 1,
                    -alpha,
                );
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

        best_score
    }

    /// 수 정렬: 위협 탐지 → 킬러 무브 → 히스토리 휴리스틱
    fn order_moves(&self, board: &Board, ply: usize) -> Vec<Move> {
        let mut moves = board.candidate_moves();
        let side = board.side_to_move as usize;

        let (my, opp) = match board.side_to_move {
            Stone::Black => (&board.black, &board.white),
            Stone::White => (&board.white, &board.black),
        };

        moves.sort_unstable_by(|&a, &b| {
            let score_a = self.move_score(a, ply, side, my, opp);
            let score_b = self.move_score(b, ply, side, my, opp);
            score_b.cmp(&score_a)
        });

        moves
    }

    fn move_score(
        &self,
        mv: Move,
        ply: usize,
        side: usize,
        my: &crate::board::BitBoard,
        opp: &crate::board::BitBoard,
    ) -> i32 {
        let mut score = self.history[side][mv];
        let row = (mv / BOARD_SIZE) as i32;
        let col = (mv % BOARD_SIZE) as i32;

        // === Composite threat (multi-direction) ===
        // DIR-local scan은 방향별 패턴만 보기 때문에 3-3, 4-3, double-four 같은
        // 다방향 복합 위협을 놓침. classify_move는 4방향 종합 등급을 반환.
        // 수비 측(opp_kind)을 크게 반영해야 Pela 같은 상대가 3-3으로 떡발리지 않음.
        score += threat_priority(classify_move(my, opp, mv), /*defending=*/ false);
        score += threat_priority(classify_move(opp, my, mv), /*defending=*/ true);

        // 위협 탐지: 이 수를 두면 어떤 패턴이 만들어지는지 체크
        for &(dr, dc) in &DIR {
            // 내 돌 연장
            let my_info = scan_line(my, opp, row, col, dr, dc);
            if my_info.count >= 4 {
                score += 500_000; // 승리수 또는 4목
            } else if my_info.count >= 3 {
                let open = my_info.open_front as u32 + my_info.open_back as u32;
                if open >= 2 {
                    score += 50_000; // 열린 3 완성
                } else if open >= 1 {
                    score += 5_000; // 닫힌 3
                }
            } else if my_info.count >= 2 {
                let open = my_info.open_front as u32 + my_info.open_back as u32;
                if open >= 2 {
                    score += 500; // 열린 2
                }
            }

            // 상대 위협 차단
            let opp_info = scan_line(opp, my, row, col, dr, dc);
            if opp_info.count >= 4 {
                score += 400_000; // 상대 5목 방어
            } else if opp_info.count >= 3 {
                let open = opp_info.open_front as u32 + opp_info.open_back as u32;
                if open >= 2 {
                    score += 40_000; // 상대 열린 3 방어
                } else if open >= 1 {
                    score += 4_000; // 상대 닫힌 3 방어
                }
            }
        }

        // 킬러 무브
        if ply < 64 {
            if self.killers[ply][0] == Some(mv) {
                score += 10_000;
            } else if self.killers[ply][1] == Some(mv) {
                score += 5_000;
            }
        }

        // 중앙 선호
        let center_dist = ((row - 7).abs() + (col - 7).abs()) as i32;
        score += (14 - center_dist) * 2;

        score
    }
}

/// `classify_move` 결과를 move ordering priority 점수로 매핑.
/// `defending=true`이면 상대 관점 위협 (차단 우선순위).
fn threat_priority(kind: ThreatKind, defending: bool) -> i32 {
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
