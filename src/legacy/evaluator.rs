use super::{
    CaroSerCalc, CheckerRec, FreeSerCalc, Rec, Rule, Stack, StdSerCalc, Tree,
};
use crate::{Coord, CoordState, Error};

use std::time::{Duration, Instant};

/// Trait for all possible evaluators. `SZ` is board size.
pub trait Eval<const SZ: usize>: Rec<SZ> {
    /// Sets timeout value for each turn, the execution time of `calc_next` must not exceed it.
    fn set_turn_timeout(&mut self, timeout: Duration);

    /// Sets the maximum RAM usage in bytes, the evaluator must not use more at any timepoint.
    fn set_max_ram(&mut self, ram_max: usize);

    /// Calculates the next move according to the rule.
    fn calc_next(&mut self) -> Coord<SZ>;

    /// Calculates and places the next move.
    fn place_next(&mut self) -> Coord<SZ> {
        let coord = self.calc_next();
        if coord.is_real() {
            self.add(coord).unwrap();
        }
        coord
    }
}

pub type StdEvaluator15 = Evaluator<15, { 15 * 15 + 16 }, StdSerCalc<15>>;
pub type StdEvaluator20 = Evaluator<20, { 20 * 20 + 16 }, StdSerCalc<20>>;
pub type FreeEvaluator15 = Evaluator<15, { 15 * 15 + 16 }, FreeSerCalc<15>>;
pub type FreeEvaluator20 = Evaluator<20, { 20 * 20 + 16 }, FreeSerCalc<20>>;
pub type CaroEvaluator15 = Evaluator<15, { 15 * 15 + 16 }, CaroSerCalc<15>>;
pub type CaroEvaluator20 = Evaluator<20, { 20 * 20 + 16 }, CaroSerCalc<20>>;

/// Currently, this is the only implementation of [Eval].
///
/// TODO: Implement alpha-beta algorithm.
#[derive(Clone, Debug)]
pub struct Evaluator<const SZ: usize, const LEN: usize, CH: Rule<SZ>> {
    rec: CheckerRec<SZ, LEN, CH>,
    t_max: Duration,
    ram_max: usize,
}

impl<const SZ: usize, const LEN: usize, CH: Rule<SZ>> Evaluator<SZ, LEN, CH> {
    /// Creates a new `Evaluator` with the given rule and turn timeout.
    pub fn new(rule: CH, turn_limit: Duration) -> Self {
        Self {
            rec: CheckerRec::new(rule),
            t_max: turn_limit,
            ram_max: 100 * 1024 * 1024,
        }
    }
}

/// Operates for all nodes under the relative depth beneath the current node;
/// then comes back to the original node. Custom `operation` must leave the tree depth
/// (after operation) the same as the depth before the operation; after the operation,
/// the tree cursor should be set back to the location before the `operation`; it shouldn't
/// try to delete nodes at (or above) this specified relative depth.
macro_rules! traversal_in_depth {
    ($tree_name:ident, $trav_dep:expr, $operation:block) => {
        let mut macro_stack = Stack::new();
        let mut rel_depth = 0_u16;
        loop {
            if $tree_name.cur_has_down() && rel_depth < $trav_dep {
                if $tree_name.cur_has_right() {
                    macro_stack.push($tree_name.cur_coord());
                }
                $tree_name.cur_go_down().unwrap();
                rel_depth += 1;
            } else if $tree_name.cur_has_right() {
                $tree_name.cur_go_right().unwrap();
            } else if let Some(upper) = macro_stack.pop() {
                let d_depth = $tree_name.cur_back_to(upper).unwrap();
                rel_depth -= d_depth;
                $tree_name.cur_go_right().unwrap();
            } else {
                for _ in 0..rel_depth {
                    $tree_name.cur_go_up().unwrap();
                }
                break;
            }
            if rel_depth == $trav_dep {
                $operation
            }
        }
    };
}

impl<const SZ: usize, const LEN: usize, CH: Rule<SZ>> Eval<SZ> for Evaluator<SZ, LEN, CH> {
    fn set_turn_timeout(&mut self, timeout: Duration) {
        self.t_max = timeout;
    }

    fn set_max_ram(&mut self, ram_max: usize) {
        self.ram_max = ram_max;
    }

    fn calc_next(&mut self) -> Coord<SZ> {
        if self.is_full() {
            return Coord::new();
        }

        let t_timeout = Instant::now() + self.t_max - Duration::from_millis(200);
        let t_near_timeout = t_timeout - self.t_max * 25 / 100;

        // The root of this tree corresponds to the last move in `self.rec`.
        let mut tree = EvalTree::new(&self.rec);

        let cnt_root_cands = tree.cur_expand(20);
        if cnt_root_cands == 0 {
            return Coord::new();
        } else if cnt_root_cands == 1 {
            tree.cur_go_down().unwrap();
            return tree.cur_coord();
        }
        traversal_in_depth!(tree, 1, { tree.cur_expand_depth(7, 4) });
        tree.cur_order_minimax();

        if Instant::now() >= t_timeout {
            tree.cur_go_down().unwrap();
            return tree.cur_coord();
        }

        tree.cur_reduce_branches(10);
        traversal_in_depth!(tree, 1, { tree.cur_cut_branches(3) });
        traversal_in_depth!(tree, 4 - 1, { tree.cur_reduce_branches(4) });
        traversal_in_depth!(tree, 5 - 1, { tree.cur_reduce_branches(4) });

        let mut depth_expand = 5;
        let mut cnt_loops = 0;
        let mut cycle_duration = None;
        loop {
            if tree.inner.ram_used() > self.ram_max / 2 {
                tree.compress();
            }

            let t_cycle_begin = Instant::now();
            if let Some(&dur) = cycle_duration.as_ref() {
                if t_cycle_begin + dur >= t_timeout {
                    break;
                }
            } else if t_cycle_begin >= t_near_timeout {
                break;
            }

            tree.cur_goto_root();
            traversal_in_depth!(tree, depth_expand, { tree.cur_expand_depth(7, 4) });
            traversal_in_depth!(tree, depth_expand - 2, {
                tree.cur_order_minimax();
                tree.cur_cut_branches(2);
            });
            traversal_in_depth!(tree, depth_expand, { tree.cur_cut_branches(2) });
            depth_expand += 4;

            cnt_loops += 1;
            if cnt_loops == 3 {
                tree.cur_goto_root(); // XXX: not needed if above functions are correct
                tree.cur_reduce_branches(5);
            }

            cycle_duration.replace(Instant::now() - t_cycle_begin);
        }

        tree.cur_goto_root();
        tree.cur_order_minimax();
        tree.cur_go_down().unwrap();
        tree.cur_coord()
    }
}

#[allow(dead_code)]
impl<const SZ: usize, const LEN: usize, CH: Rule<SZ>> Rec<SZ> for Evaluator<SZ, LEN, CH> {
    #[inline(always)]
    fn as_slice(&self) -> &[Coord<SZ>] {
        self.rec.as_slice()
    }

    #[inline(always)]
    fn coord_state(&self, coord: Coord<SZ>) -> CoordState {
        self.rec.coord_state(coord)
    }

    #[inline(always)]
    fn get_quad_rows(&self, coord: Coord<SZ>) -> Option<[super::Row; 4]> {
        self.rec.get_quad_rows(coord)
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.rec.len()
    }

    #[inline(always)]
    fn len_max(&self) -> usize {
        LEN
    }

    #[inline(always)]
    fn stones_count(&self) -> usize {
        self.rec.stones_count()
    }

    #[inline(always)]
    fn is_full(&self) -> bool {
        self.rec.is_full()
    }

    #[inline(always)]
    fn add(&mut self, coord: Coord<SZ>) -> Result<(), Error> {
        self.rec.add(coord)
    }

    #[inline(always)]
    fn undo(&mut self) -> Result<Coord<SZ>, Error> {
        self.rec.undo()
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.rec.clear()
    }
}

/// A tree with a record checker tied to the tree cursor. Each node contains
/// a unified evaluation value.
struct EvalTree<const SZ: usize, const LEN: usize, CH: Rule<SZ>> {
    inner: Tree<SZ, i16>,
    rec: CheckerRec<SZ, LEN, CH>,
    cand_tmp_list: [(Coord<SZ>, i16); LEN],
}

impl<const SZ: usize, const LEN: usize, CH: Rule<SZ>> EvalTree<SZ, LEN, CH> {
    pub fn new(rec: &CheckerRec<SZ, LEN, CH>) -> Self {
        let mut inner = Tree::new();
        let rec = rec.clone();
        *inner.cur_info_mut() = rec.score_unified();
        Self {
            inner,
            rec,
            cand_tmp_list: [(Coord::new(), 0); LEN],
        }
    }

    #[allow(unused)]
    #[inline(always)]
    pub fn inner(&self) -> &Tree<SZ, i16> {
        &self.inner
    }

    #[inline(always)]
    pub fn cur_depth(&self) -> u16 {
        self.inner.cur_depth()
    }

    #[inline(always)]
    pub fn cur_is_leaf(&self) -> bool {
        self.inner.cur_is_leaf()
    }

    #[inline(always)]
    pub fn cur_has_down(&self) -> bool {
        self.inner.cur_has_down()
    }

    #[inline(always)]
    pub fn cur_has_right(&self) -> bool {
        self.inner.cur_has_right()
    }

    #[inline(always)]
    #[allow(unused)]
    pub fn cur_val(&self) -> i16 {
        *self.inner.cur_info()
    }

    #[inline(always)]
    pub fn cur_set_val(&mut self, val: i16) {
        *self.inner.cur_info_mut() = val;
    }

    #[inline(always)]
    pub fn cur_coord(&self) -> Coord<SZ> {
        self.inner.cur_coord()
    }

    #[inline(always)]
    pub fn down_is_leaf(&self) -> Option<bool> {
        self.inner.down_is_leaf()
    }

    #[inline(always)]
    pub fn down_val(&self) -> Option<i16> {
        self.inner.down_info().copied()
    }

    #[inline]
    pub fn cur_goto_root(&mut self) {
        let prev_depth = self.cur_depth();
        for _ in 0..prev_depth {
            self.rec.undo().unwrap();
        }
        self.inner.cur_goto_root();
    }

    #[inline(always)]
    pub fn cur_go_up(&mut self) -> Result<(), Error> {
        self.inner.cur_go_up()?;
        self.rec.undo().unwrap();
        Ok(())
    }

    #[inline(always)]
    pub fn cur_back_to(&mut self, coord: Coord<SZ>) -> Result<u16, Error> {
        let d_depth = self.inner.cur_back_to(coord)?;
        for _ in 0..d_depth {
            self.rec.undo().unwrap();
        }
        Ok(d_depth)
    }

    #[inline(always)]
    pub fn cur_go_down(&mut self) -> Result<(), Error> {
        self.inner.cur_go_down()?;
        self.rec.add(self.inner.cur_coord()).unwrap();
        Ok(())
    }

    #[inline(always)]
    pub fn cur_go_right(&mut self) -> Result<(), Error> {
        self.inner.cur_go_right()?;
        self.rec.undo().unwrap();
        self.rec.add(self.inner.cur_coord()).unwrap();
        Ok(())
    }

    #[inline(always)]
    pub fn cur_set_next(&mut self, coord: Coord<SZ>) -> Result<(), Error> {
        self.rec.add(coord)?;
        let exists = self.inner.cur_find_next(coord).is_some();
        self.inner.cur_set_next(coord);
        if !exists {
            *self.inner.cur_info_mut() = self.rec.score_unified();
        }
        Ok(())
    }

    #[inline(always)]
    pub fn cur_delete(&mut self) {
        let prev_depth = self.cur_depth();
        self.inner.cur_delete();
        if prev_depth > 0 {
            self.rec.undo().unwrap();
        }
    }

    #[inline(always)]
    pub fn compress(&mut self) {
        self.inner.compress();
    }

    /// Sets choice candidates for the current node.
    #[inline(always)]
    pub fn cur_expand(&mut self, cnt_max: u16) -> u16 {
        let cnt = self
            .rec
            .write_candidates(&mut self.cand_tmp_list[..cnt_max as usize]);
        self.inner.cur_set_children(&self.cand_tmp_list[..cnt]);
        cnt as u16
    }

    /// Does [EvalTree::cur_expand] recursively to reach the specified relative depth.
    pub fn cur_expand_depth(&mut self, cnt_max: u16, depth: u16) {
        if depth == 0 {
            return;
        }
        self.cur_expand(cnt_max);
        if depth == 1 {
            return;
        }
        if self.cur_go_down().is_err() {
            return;
        }
        loop {
            self.cur_expand_depth(cnt_max, depth - 1);
            if self.cur_go_right().is_err() {
                break;
            }
        }
        self.cur_go_up().unwrap();
    }

    /// Does a postorder traversal under current node to determine all leftmost nodes,
    /// goes back to the original node at last.
    ///
    /// Note: this should be called right after `cur_expand_depth`.
    pub fn cur_order_minimax(&mut self) {
        if self.cur_is_leaf() {
            return;
        }
        let mut stack = Stack::new();
        loop {
            while !self.down_is_leaf().unwrap() {
                stack.push(self.cur_coord());
                self.cur_go_down().unwrap();
            }

            let val = self.down_val().unwrap();
            self.cur_set_val(val);

            while self.cur_go_right().is_err() {
                let Some(upper) = stack.pop() else {
                    return;
                };
                self.cur_back_to(upper)
                    .unwrap_or_else(|_| panic!("{upper}  {:?}", self.rec));

                let new_left = if self.rec.color_next().is_black() {
                    self.inner.cur_find_max_child()
                } else {
                    self.inner.cur_find_min_child()
                };
                if let Some((left_coord, &left_val)) = new_left {
                    self.inner.cur_adj_left_child(left_coord);
                    self.cur_set_val(left_val)
                }
            }
        }
    }

    /// Cuts right siblings at several levels (set by `cut_depth`) under current node,
    /// goes back to the original node at last.
    ///
    /// Note: this may be called after `cur_order_minimax`.
    pub fn cur_cut_branches(&mut self, cut_depth: u16) {
        let prev_depth = self.cur_depth();
        for _ in 0..cut_depth {
            if self.cur_go_down().is_err() {
                break;
            }
            self.inner.cur_delete_siblings();
        }
        while self.cur_depth() > prev_depth {
            self.cur_go_up().unwrap();
        }
    }

    /// Reduces choices with worst values under current node, until
    /// the degree of current node reaches `target_deg`.
    pub fn cur_reduce_branches(&mut self, target_deg: u16) {
        let mut cur_deg = self.inner.cur_get_degree();
        while cur_deg > target_deg {
            let worst = if self.rec.color_next().is_black() {
                self.inner.cur_find_min_child()
            } else {
                self.inner.cur_find_max_child()
            };
            self.cur_set_next(worst.unwrap().0).unwrap();
            self.cur_delete(); // goes up
            cur_deg -= 1;
        }
    }
}
