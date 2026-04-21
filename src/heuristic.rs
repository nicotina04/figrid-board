/// 휴리스틱 오목 평가 함수
///
/// 4방향(가로, 세로, 대각선×2)으로 연속 패턴을 스캔하여 점수 매김.
/// 패턴:
///   - 5목(승리): 1,000,000
///   - 열린 4 (양쪽 열림): 100,000
///   - 닫힌 4 (한쪽 막힘) / 4-1갭: 10,000
///   - 열린 3: 5,000
///   - 닫힌 3: 500
///   - 열린 2: 100
///   - 닫힌 2: 10

use crate::board::{BitBoard, Board, Stone, BOARD_SIZE, NUM_CELLS};

pub const DIR: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];

/// 한 방향으로 연속 돌 수 + 양쪽 열림 여부 스캔
#[derive(Debug)]
pub struct LineInfo {
    /// 연속 돌 수
    pub count: u32,
    /// 앞쪽 열림 (빈칸)
    pub open_front: bool,
    /// 뒤쪽 열림 (빈칸)
    pub open_back: bool,
}

pub fn scan_line(stones: &BitBoard, opp: &BitBoard, row: i32, col: i32, dr: i32, dc: i32) -> LineInfo {
    let mut count = 1u32;

    // 정방향
    let mut open_front = false;
    for step in 1..5 {
        let nr = row + dr * step;
        let nc = col + dc * step;
        if nr < 0 || nr >= BOARD_SIZE as i32 || nc < 0 || nc >= BOARD_SIZE as i32 {
            break;
        }
        let idx = nr as usize * BOARD_SIZE + nc as usize;
        if stones.get(idx) {
            count += 1;
        } else {
            open_front = !opp.get(idx); // 빈칸이면 열림
            break;
        }
    }

    // 역방향
    let mut open_back = false;
    for step in 1..5 {
        let nr = row - dr * step;
        let nc = col - dc * step;
        if nr < 0 || nr >= BOARD_SIZE as i32 || nc < 0 || nc >= BOARD_SIZE as i32 {
            break;
        }
        let idx = nr as usize * BOARD_SIZE + nc as usize;
        if stones.get(idx) {
            count += 1;
        } else {
            open_back = !opp.get(idx);
            break;
        }
    }

    LineInfo {
        count,
        open_front,
        open_back,
    }
}

/// 패턴별 점수 변환
fn pattern_score(info: &LineInfo) -> i32 {
    let open_ends = info.open_front as u32 + info.open_back as u32;

    match (info.count, open_ends) {
        (5.., _) => 1_000_000,     // 5목 이상 = 승리
        (4, 2) => 100_000,         // 열린 4 = 거의 승리
        (4, 1) => 10_000,          // 닫힌 4
        (3, 2) => 5_000,           // 열린 3
        (3, 1) => 500,             // 닫힌 3
        (2, 2) => 100,             // 열린 2
        (2, 1) => 10,              // 닫힌 2
        _ => 0,
    }
}

/// 보드 전체를 휴리스틱으로 평가
/// 리턴: 양수 = side_to_move 유리
pub fn heuristic_eval(board: &Board) -> i32 {
    let (my_stones, opp_stones) = match board.side_to_move {
        Stone::Black => (&board.black, &board.white),
        Stone::White => (&board.white, &board.black),
    };

    let mut my_score: i32 = 0;
    let mut opp_score: i32 = 0;

    // 각 돌에서 4방향으로 패턴 스캔
    // 중복 방지: 각 방향에서 시작점이 라인의 첫 번째 돌인 경우만 카운트
    for idx in 0..NUM_CELLS {
        let row = (idx / BOARD_SIZE) as i32;
        let col = (idx % BOARD_SIZE) as i32;

        for &(dr, dc) in &DIR {
            // 내 돌
            if my_stones.get(idx) {
                // 이 돌이 라인의 시작인지 확인 (이전 칸이 같은 돌이 아닌 경우)
                let pr = row - dr;
                let pc = col - dc;
                let is_start = if pr < 0 || pr >= BOARD_SIZE as i32 || pc < 0 || pc >= BOARD_SIZE as i32 {
                    true
                } else {
                    !my_stones.get(pr as usize * BOARD_SIZE + pc as usize)
                };

                if is_start {
                    let info = scan_line(my_stones, opp_stones, row, col, dr, dc);
                    if info.count >= 2 {
                        my_score += pattern_score(&info);
                    }
                }
            }

            // 상대 돌
            if opp_stones.get(idx) {
                let pr = row - dr;
                let pc = col - dc;
                let is_start = if pr < 0 || pr >= BOARD_SIZE as i32 || pc < 0 || pc >= BOARD_SIZE as i32 {
                    true
                } else {
                    !opp_stones.get(pr as usize * BOARD_SIZE + pc as usize)
                };

                if is_start {
                    let info = scan_line(opp_stones, my_stones, row, col, dr, dc);
                    if info.count >= 2 {
                        opp_score += pattern_score(&info);
                    }
                }
            }
        }
    }

    // 공격 가중 (내 위협이 약간 더 가치 있음 — 선공 이점)
    my_score * 11 / 10 - opp_score
}

/// 휴리스틱 + 알파-베타 탐색
pub struct HeuristicSearcher {
    pub nodes: u64,
    killers: [[Option<usize>; 2]; 64],
    history: [[i32; NUM_CELLS]; 2],
}

const INF: i32 = 10_000_000;
const WIN: i32 = 1_000_000;

impl HeuristicSearcher {
    pub fn new() -> Self {
        Self {
            nodes: 0,
            killers: [[None; 2]; 64],
            history: [[0; NUM_CELLS]; 2],
        }
    }

    pub fn search(&mut self, board: &mut Board, max_depth: u32) -> (Option<usize>, i32) {
        self.nodes = 0;
        self.killers = [[None; 2]; 64];
        self.history = [[0; NUM_CELLS]; 2];

        let mut best_move = None;
        let mut best_score = -INF;

        for depth in 1..=max_depth {
            let mut alpha = -INF;
            let moves = self.order_moves(board, 0);

            for mv in &moves {
                board.make_move(*mv);
                let score = -self.alpha_beta(board, depth - 1, 1, -INF, -alpha);
                board.undo_move();

                if score > alpha {
                    alpha = score;
                    best_move = Some(*mv);
                    best_score = score;
                }
            }
        }

        (best_move, best_score)
    }

    fn alpha_beta(
        &mut self,
        board: &mut Board,
        depth: u32,
        ply: usize,
        mut alpha: i32,
        beta: i32,
    ) -> i32 {
        self.nodes += 1;

        // 승리 체크
        if let Some(mv) = board.last_move {
            if board.check_win(mv) {
                return -(WIN + 100 - ply as i32);
            }
        }

        if depth == 0 {
            return heuristic_eval(board);
        }

        let moves = self.order_moves(board, ply);
        if moves.is_empty() {
            return 0;
        }

        let mut best = -INF;
        let side = board.side_to_move as usize;

        for mv in &moves {
            board.make_move(*mv);
            let score = -self.alpha_beta(board, depth - 1, ply + 1, -beta, -alpha);
            board.undo_move();

            if score > best {
                best = score;
            }
            if score > alpha {
                alpha = score;
                self.history[side][*mv] += (depth * depth) as i32;
            }
            if alpha >= beta {
                if ply < 64 {
                    self.killers[ply][1] = self.killers[ply][0];
                    self.killers[ply][0] = Some(*mv);
                }
                break;
            }
        }

        best
    }

    fn order_moves(&self, board: &Board, ply: usize) -> Vec<usize> {
        let mut moves = board.candidate_moves();
        let side = board.side_to_move as usize;

        // 즉시 승리/방어 수를 최우선으로
        let (my, opp) = match board.side_to_move {
            Stone::Black => (&board.black, &board.white),
            Stone::White => (&board.white, &board.black),
        };

        moves.sort_unstable_by(|&a, &b| {
            let score_a = self.move_priority(a, ply, side, my, opp);
            let score_b = self.move_priority(b, ply, side, my, opp);
            score_b.cmp(&score_a)
        });

        moves
    }

    fn move_priority(
        &self,
        mv: usize,
        ply: usize,
        side: usize,
        my: &BitBoard,
        opp: &BitBoard,
    ) -> i32 {
        let mut score = self.history[side][mv];
        let row = (mv / BOARD_SIZE) as i32;
        let col = (mv % BOARD_SIZE) as i32;

        // 이 수를 두면 내가 몇 목이 되는지 빠르게 체크
        for &(dr, dc) in &DIR {
            // 내 돌 연장
            let my_info = scan_line(my, opp, row, col, dr, dc);
            if my_info.count >= 4 {
                score += 500_000; // 승리수 또는 4목
            } else if my_info.count >= 3 {
                let open = my_info.open_front as u32 + my_info.open_back as u32;
                if open >= 2 {
                    score += 50_000; // 열린 3 완성
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::to_idx;

    #[test]
    fn test_heuristic_detects_open_four() {
        let mut board = Board::new();
        // 흑: (7,3) (7,4) (7,5) (7,6) — 4연속, 열린 4
        // 백: 떨어진 곳에 둬서 위협 없음
        board.make_move(to_idx(7, 3));
        board.make_move(to_idx(0, 0));
        board.make_move(to_idx(7, 4));
        board.make_move(to_idx(0, 14));
        board.make_move(to_idx(7, 5));
        board.make_move(to_idx(14, 0));
        board.make_move(to_idx(7, 6));
        // 현재 백 턴. 흑이 열린 4, 백은 위협 없음 → 백 관점 매우 불리
        let eval = heuristic_eval(&board);
        assert!(eval < -50_000, "should detect opponent's open four, got {eval}");
    }

    #[test]
    fn test_heuristic_search_finds_winning() {
        let mut board = Board::new();
        // 흑 4목 상태
        board.make_move(to_idx(7, 3));
        board.make_move(to_idx(8, 3));
        board.make_move(to_idx(7, 4));
        board.make_move(to_idx(8, 4));
        board.make_move(to_idx(7, 5));
        board.make_move(to_idx(8, 5));
        board.make_move(to_idx(7, 6));
        board.make_move(to_idx(8, 6));
        // 흑 턴: (7,2) 또는 (7,7)에 두면 5목

        let mut searcher = HeuristicSearcher::new();
        let (best, score) = searcher.search(&mut board, 4);
        let winning = [to_idx(7, 2), to_idx(7, 7)];
        assert!(best.is_some());
        assert!(winning.contains(&best.unwrap()), "got {:?}", best);
        assert!(score > WIN / 2, "should see winning score, got {score}");
    }

    #[test]
    fn test_heuristic_blocks_threat() {
        let mut board = Board::new();
        // 백이 열린 3 만든 상태: (7,4) (7,5) (7,6)
        board.make_move(to_idx(0, 0)); // 흑 아무데나
        board.make_move(to_idx(7, 4));
        board.make_move(to_idx(0, 1));
        board.make_move(to_idx(7, 5));
        board.make_move(to_idx(0, 2));
        board.make_move(to_idx(7, 6));
        // 흑 턴: (7,3) 또는 (7,7)로 막아야 함

        let mut searcher = HeuristicSearcher::new();
        let (best, _) = searcher.search(&mut board, 4);
        let blocking = [to_idx(7, 3), to_idx(7, 7)];
        assert!(best.is_some());
        assert!(
            blocking.contains(&best.unwrap()),
            "should block threat, got {:?}",
            best
        );
    }
}
