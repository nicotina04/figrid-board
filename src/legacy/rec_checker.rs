use std::{fmt::Display, ops::Index};

use super::{
    BaseRec,
    Rec, // trait
    Row,
    RowScore,
    Rule,
};
use crate::{Coord, CoordState, Error};

/// The sum of [RowScore]s for one side.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowsScore {
    /// Counts for connected or grouped stones with strengths 1, 2, 3, 4.
    pub cnts: [u16; 4],
    /// Count of rows where the count of strength 3 is greater than 1.
    pub cnt_live_3: u16,
    /// It is set to true when this side reaches strength 5 on any row.
    pub flag_5: bool,
}

impl RowsScore {
    /// Creates an empty score sum.
    pub fn new() -> Self {
        Self {
            cnts: [0; 4],
            cnt_live_3: 0,
            flag_5: false,
        }
    }

    /// Clears the score sum as if it's newly created.
    pub fn clear(&mut self) {
        *self = Self::new();
    }

    /// Updates the score sum for the change in a single row.
    pub fn update(&mut self, row_bef: RowScore, row_aft: RowScore) {
        for i in 0..4 {
            self.cnts[i] += row_aft.cnts[i] as u16;
            self.cnts[i] = self.cnts[i].saturating_sub(row_bef.cnts[i] as u16);
        }

        if row_aft.flag_live_3 && !row_bef.flag_live_3 {
            self.cnt_live_3 += 1;
        } else if row_bef.flag_live_3 && !row_aft.flag_live_3 {
            self.cnt_live_3 -= 1;
        }

        if row_aft.flag_5 && !row_bef.flag_5 {
            self.flag_5 = true;
        } else if row_bef.flag_5 && !row_aft.flag_5 {
            self.flag_5 = false;
        }
    }
}

impl Default for RowsScore {
    fn default() -> Self {
        Self::new()
    }
}

impl RowsScore {
    /// Calculates a static evaluation value with the current sums of both sides and the
    /// color of the next move. greater value means greater advantage for the black side.
    #[allow(clippy::eq_op)]
    pub fn unify(sum_b: &Self, sum_w: &Self, color_next: CoordState) -> i16 {
        if sum_b.flag_5 {
            return i16::MAX;
        }
        if sum_w.flag_5 {
            return i16::MIN;
        }
        // excluded connected 5 from now on
        if sum_b.cnts[4 - 1] > 0 && color_next.is_black() {
            return i16::MAX - 1;
        }
        if sum_w.cnts[4 - 1] > 0 && color_next.is_white() {
            return i16::MIN + 1;
        }
        // excluded connected 4 for self's turn from now on
        if sum_b.cnts[4 - 1] > 1 {
            return i16::MAX - 1; // here, color_next is white, sum_w.cnts[4 - 1] == 0
        }
        if sum_w.cnts[4 - 1] > 1 {
            return i16::MIN + 1; // here, color_next is black, sum_b.cnts[4 - 1] == 0
        }
        // excluded double 4 on opponent's turn from now on
        if sum_b.cnt_live_3 > 0 && sum_w.cnts[4 - 1] == 0 && color_next.is_black() {
            return i16::MAX - 2;
        }
        if sum_w.cnt_live_3 > 0 && sum_b.cnts[4 - 1] == 0 && color_next.is_white() {
            return i16::MIN + 2;
        }
        // excluded prior live 3 for self's turn from now on

        let mut sum = 0_i16;

        // 4 with live 3, not sure in a few cases
        if sum_b.cnts[4 - 1] > 0 && sum_b.cnt_live_3 > 0 {
            sum += 128; // here, color_next is white, sum_w.cnts[4 - 1] == 0
        }
        if sum_w.cnts[4 - 1] > 0 && sum_w.cnt_live_3 > 0 {
            sum -= 128; // here, color_next is black, sum_b.cnts[4 - 1] == 0
        }

        // prior double live 3, opponent's turn, not sure in a few cases
        if sum_b.cnt_live_3 > 1 {
            sum += 64; // here, color_next is white, sum_w.cnts[4 - 1] == 0
            if sum_w.cnts[3 - 1] == 0 {
                sum += 64;
            }
        }
        if sum_w.cnt_live_3 > 1 {
            sum -= 64; // here, color_next is black, sum_b.cnts[4 - 1] == 0
            if sum_b.cnts[3 - 1] == 0 {
                sum -= 64;
            }
        }

        // XXX: making improvement here may improve the overall performance significantly.
        sum += (sum_b.cnts[1 - 1]
            + (sum_b.cnts[2 - 1] << 2)
            + (sum_b.cnts[3 - 1] << 4)
            + (sum_b.cnt_live_3 << 5)
            + (sum_b.cnts[4 - 1] << 6)) as i16;
        sum -= (sum_w.cnts[1 - 1]
            + (sum_w.cnts[2 - 1] << 2)
            + (sum_w.cnts[3 - 1] << 4)
            + (sum_w.cnt_live_3 << 5)
            + (sum_w.cnts[4 - 1] << 6)) as i16;
        sum
    }
}

type RowScorePair = (RowScore, RowScore);

/// Record structure with a fixed [Rule] checker.
///
/// `SZ` is the board size, `LEN` is the maximum capacity.
#[derive(Clone, Debug)]
pub struct CheckerRec<const SZ: usize, const LEN: usize, CH: Rule<SZ>> {
    rec: BaseRec<SZ, LEN>,
    checker: CH,

    horizontal: [RowScorePair; SZ],
    vertical: [RowScorePair; SZ],
    diagonal_l1: [RowScorePair; SZ],
    diagonal_l2: [RowScorePair; SZ], // actual size is SZ - 1
    diagonal_r1: [RowScorePair; SZ],
    diagonal_r2: [RowScorePair; SZ], // actual size is SZ - 1

    undo_stack: [[RowScorePair; 4]; LEN],

    sum_b: RowsScore,
    sum_w: RowsScore,

    grid_cand: [[bool; SZ]; SZ],
    cand_tmp_list: [(Coord<SZ>, i16); LEN],
}

impl<const SZ: usize, const LEN: usize, CH: Rule<SZ>> CheckerRec<SZ, LEN, CH> {
    /// Returns an empty record.
    pub fn new(checker: CH) -> Self {
        assert!(SZ >= 5 && SZ <= 26);
        let new_pair = (RowScore::new(), RowScore::new());
        let mut grid_cand = [[false; SZ]; SZ];
        grid_cand[SZ / 2][SZ / 2] = true;
        Self {
            rec: BaseRec::new(),
            checker,

            horizontal: [new_pair; SZ],
            vertical: [new_pair; SZ],
            diagonal_l1: [new_pair; SZ],
            diagonal_l2: [new_pair; SZ],
            diagonal_r1: [new_pair; SZ],
            diagonal_r2: [new_pair; SZ],

            undo_stack: [[new_pair; 4]; LEN],

            sum_b: RowsScore::new(),
            sum_w: RowsScore::new(),

            grid_cand,
            cand_tmp_list: [(Coord::new(), 0); LEN],
        }
    }

    /// Returns a pair of row score sums for the black side and the white side.
    #[inline(always)]
    pub fn score_sum(&self) -> (RowsScore, RowsScore) {
        (self.sum_b.clone(), self.sum_w.clone())
    }

    /// Returns a static evaluation value for the current situation;
    /// greater value means greater advantage for the black side.
    #[inline(always)]
    pub fn score_unified(&self) -> i16 {
        RowsScore::unify(&self.sum_b, &self.sum_w, self.color_next())
    }

    /// Returns true if one side has won or the board is full of stones.
    pub fn is_finished(&self) -> bool {
        self.sum_b.flag_5 || self.sum_w.flag_5 || self.is_full()
    }

    /// Gets mutable references of pairs of row scores for horizontal, vertical,
    /// left and right diagonal rows passing through `coord`.
    #[inline(always)]
    fn quad_score_pairs(&mut self, coord: Coord<SZ>) -> Option<[&mut RowScorePair; 4]> {
        let (x, y) = coord.get()?;

        let row_diagonal_l = if x <= y {
            &mut self.diagonal_l1[SZ - 1 - (y - x) as usize]
        } else {
            &mut self.diagonal_l2[(x - y - 1) as usize]
        };

        let row_diagonal_r = if x + y <= SZ as u8 - 1 {
            &mut self.diagonal_r1[(x + y) as usize]
        } else {
            &mut self.diagonal_r2[(x + y) as usize - SZ]
        };

        Some([
            &mut self.horizontal[y as usize],
            &mut self.vertical[x as usize],
            row_diagonal_l,
            row_diagonal_r,
        ])
    }

    /// Writes candidates for the next move based on the static evaluation.
    /// Returns the amount of written candidates.
    pub fn write_candidates(&mut self, out_space: &mut [(Coord<SZ>, i16)]) -> usize {
        if out_space.is_empty() || self.is_finished() {
            return 0;
        }

        let mut cnt_raw = 0;
        for x in 0..SZ {
            for y in 0..SZ {
                if self.grid_cand[x][y] {
                    let coord = Coord::from(x as u8, y as u8);
                    self.cand_tmp_list[cnt_raw] = (coord, 0);
                    cnt_raw += 1;
                }
            }
        }

        for i in 0..cnt_raw {
            let coord = self.cand_tmp_list[i].0;
            self.add(coord).unwrap();
            let scr = self.score_unified();
            self.undo().unwrap();
            self.cand_tmp_list[i].1 = scr;
        }

        let next_is_black = self.color_next().is_black();

        let cand_list = &mut self.cand_tmp_list[..cnt_raw];
        if next_is_black {
            cand_list.sort_unstable_by(|a, b| b.1.cmp(&a.1)); // reverse ordering
        } else {
            cand_list.sort_unstable_by(|a, b| a.1.cmp(&b.1));
        }

        let mut selected_len = out_space.len().min(cand_list.len());
        if cnt_raw > 0 && (cand_list[0].1 >= i16::MAX - 2 || cand_list[0].1 <= i16::MIN + 2) {
            // the best choice means sure win or sure lose
            selected_len = 1;
        }

        let selected = &cand_list[..selected_len];
        out_space[..selected_len].copy_from_slice(selected);
        selected_len
    }
}

impl<const SZ: usize, const LEN: usize, CH: Rule<SZ>> Rec<SZ> for CheckerRec<SZ, LEN, CH> {
    #[inline(always)]
    fn as_slice(&self) -> &[Coord<SZ>] {
        self.rec.as_slice()
    }

    #[inline(always)]
    fn coord_state(&self, coord: Coord<SZ>) -> CoordState {
        self.rec.coord_state(coord)
    }

    #[inline(always)]
    fn get_quad_rows(&self, coord: Coord<SZ>) -> Option<[Row; 4]> {
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

    #[inline]
    #[allow(clippy::needless_range_loop)]
    fn add(&mut self, coord: Coord<SZ>) -> Result<(), Error> {
        if self.is_finished() {
            return Err(Error::RecIsFinished);
        }

        self.rec.add(coord)?;

        let Some((x, y)) = coord.get() else {
            return Ok(());
        };

        #[rustfmt::skip] const SIGNS: [(i8, i8); 8] = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ];
        self.grid_cand[x as usize][y as usize] = false;
        for (sign_x, sign_y) in SIGNS {
            for i in 1..=4 {
                if let Some(coord_off) = coord.offset(i * sign_x, i * sign_y) {
                    let (x, y) = coord_off.get().unwrap();
                    if self.coord_state(coord_off).is_empty() {
                        self.grid_cand[x as usize][y as usize] = true;
                    }
                }
            }
        }

        let quad_rows_aft = self.rec.get_quad_rows(coord).unwrap();
        for i in 0..4 {
            let score_pair_bef = *self.quad_score_pairs(coord).unwrap()[i];
            let score_pair_aft = self.checker.check_row(quad_rows_aft[i]);
            self.sum_b.update(score_pair_bef.0, score_pair_aft.0);
            self.sum_w.update(score_pair_bef.1, score_pair_aft.1);
            self.undo_stack[self.len() - 1][i] = score_pair_bef;
            *self.quad_score_pairs(coord).unwrap()[i] = score_pair_aft;
        }

        Ok(())
    }

    #[inline]
    #[allow(clippy::needless_range_loop)]
    fn undo(&mut self) -> Result<Coord<SZ>, Error> {
        let coord = self.last_coord().ok_or(Error::RecIsEmpty)?;
        self.rec.undo()?;
        let Some((x, y)) = coord.get() else {
            return Ok(coord);
        };
        self.grid_cand[x as usize][y as usize] = true;

        let undo_store = self.undo_stack[self.len()];
        for i in 0..4 {
            let score_pair_bef = *self.quad_score_pairs(coord).unwrap()[i];
            self.sum_b.update(score_pair_bef.0, undo_store[i].0);
            self.sum_w.update(score_pair_bef.1, undo_store[i].1);
            *self.quad_score_pairs(coord).unwrap()[i] = undo_store[i];
        }

        Ok(coord)
    }

    #[inline]
    fn clear(&mut self) {
        self.rec.clear();

        let new_pair = (RowScore::new(), RowScore::new());
        self.horizontal = [new_pair; SZ];
        self.vertical = [new_pair; SZ];
        self.diagonal_l1 = [new_pair; SZ];
        self.diagonal_l2 = [new_pair; SZ];
        self.diagonal_r1 = [new_pair; SZ];
        self.diagonal_r2 = [new_pair; SZ];

        self.sum_b.clear();
        self.sum_w.clear();

        self.grid_cand = [[false; SZ]; SZ];
        self.grid_cand[SZ / 2][SZ / 2] = true;
    }
}

// Note: these traits can't be implemented automatically for all types with trait Rec<SZ>

impl<const SZ: usize, const LEN: usize, CH: Rule<SZ>> Index<usize> for CheckerRec<SZ, LEN, CH> {
    type Output = Coord<SZ>;

    #[inline(always)]
    fn index(&self, i: usize) -> &Coord<SZ> {
        &self.as_slice()[i]
    }
}

impl<const SZ: usize, const LEN: usize, CH: Rule<SZ>> Display for CheckerRec<SZ, LEN, CH> {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.print_str(f, " ", true)
    }
}

impl<const SZ: usize, const LEN: usize, CH: Rule<SZ>> PartialEq for CheckerRec<SZ, LEN, CH> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<const SZ: usize, const LEN: usize, CH: Rule<SZ>> Eq for CheckerRec<SZ, LEN, CH> {}
