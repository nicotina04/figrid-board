use super::Row;

/// The score of one side on a single [Row] calculated by a [Rule].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowScore {
    /// Counts for connected or grouped stones with strengths 1, 2, 3, 4.
    pub cnts: [u8; 4],
    /// It is set to true when the count of strength 3 is greater than 1.
    pub flag_live_3: bool,
    /// It is set to true when strength 5 of this side occurs in the row.
    pub flag_5: bool,
}

impl RowScore {
    /// Creates an empty row score.
    pub fn new() -> Self {
        RowScore {
            cnts: [0; 4],
            flag_live_3: false,
            flag_5: false,
        }
    }

    /// Adds `cnt` for strength `len`. The valid range of `len` is `[1, 5)`,
    /// otherwise it will panic.
    #[inline(always)]
    pub fn add(&mut self, len: u8, cnt: u8) {
        self.cnts[(len - 1) as usize] += cnt;
    }

    /// Adds another `RowScore` to make a sum. This is used to combine
    /// scores of segments split from one row. Use [crate::RowsScore] when
    /// summing scores of multiple `Row`s to avoid possible overflow.
    #[inline(always)]
    pub fn enter(&mut self, other: RowScore) {
        if other.flag_5 {
            self.flag_5 = true;
        }
        if self.flag_5 {
            return;
        }
        if other.flag_live_3 {
            self.flag_live_3 = true;
        }
        for i in 0..4 {
            self.cnts[i] += other.cnts[i];
        }
    }

    /// Returns the count of strength `len`.
    #[inline(always)]
    pub fn score_of_len(&self, len: u8) -> u8 {
        self.cnts[(len - 1) as usize]
    }
}

impl Default for RowScore {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents a rule check method.
pub trait Rule<const SZ: usize>: Clone + std::fmt::Debug {
    /// The implementation should return a pair of properly calculated
    /// [RowScore]s for the black and white on the given row.
    fn check_row(&self, row: Row) -> (RowScore, RowScore);
}

pub use serial_calc::{CaroSerCalc, FreeSerCalc, StdSerCalc};

/// Rule implementations based on the serial counter running on CPU.
mod serial_calc {
    use super::*;
    use crate::CoordState;

    /// Counter for the lengths of continuous consistent states in the row.
    #[derive(PartialEq, Debug)]
    struct RowCounter<const SZ: usize> {
        len: u8,
        cnts: [(CoordState, u8); SZ],
    }

    impl<const SZ: usize> RowCounter<SZ> {
        /// Counts for the lengths of continuous consistent states in the row.
        /// This may panic if the length of the row is larger than `SZ`.
        pub fn count(row: Row) -> Self {
            let mut counter = RowCounter {
                len: 0,
                cnts: [(CoordState::Empty, 0); SZ],
            };
            let mut last_st = CoordState::Empty;
            let mut last_cnt = 0u8;
            for st in row.iter() {
                if st == last_st {
                    last_cnt += 1;
                } else {
                    if last_cnt > 0 {
                        counter.cnts[counter.len as usize] = (last_st, last_cnt);
                        counter.len += 1;
                    }
                    last_st = st;
                    last_cnt = 1;
                }
            }
            if last_cnt > 0 {
                counter.cnts[counter.len as usize] = (last_st, last_cnt);
                counter.len += 1;
            }
            counter
        }

        /// Returns the slice of counted values.
        #[inline(always)]
        pub fn get(&self) -> &[(CoordState, u8)] {
            &self.cnts[..self.len as usize]
        }

        /// Iterates through counted values.
        #[inline(always)]
        pub fn iter(&self) -> std::slice::Iter<'_, (CoordState, u8)> {
            self.get().iter()
        }

        /// Calculates a pair of `RowScore` with the `Segment` checker function.
        pub fn check(&self, f: impl Fn(&mut Segment<SZ>) -> RowScore) -> (RowScore, RowScore) {
            let mut sum_b = RowScore::new();
            let mut sum_w = RowScore::new();

            if self.len == 1 && self.get()[0].0.is_empty() {
                return (sum_b, sum_w);
            }

            let mut color = CoordState::Empty;
            let mut sum = &mut sum_b;
            let mut seg = Segment::<SZ>::new();

            for (i, &(st, cnt)) in self.iter().enumerate() {
                if st != color && !st.is_empty() {
                    if seg.count > 0 {
                        sum.enter(f(&mut seg));
                        seg.clear();
                    }
                    if i > 0 && self.get()[i - 1].0.is_empty() {
                        seg.spaces[0] = self.get()[i - 1].1;
                    }
                    color = st;
                    sum = if color.is_black() {
                        &mut sum_b
                    } else {
                        &mut sum_w
                    };
                }
                if !st.is_empty() {
                    seg.conns[seg.count].len = cnt;
                    seg.count += 1;
                } else {
                    seg.spaces[seg.count] = cnt;
                }
            }
            if seg.count > 0 {
                sum.enter(f(&mut seg));
            }
            (sum_b, sum_w)
        }
    }

    #[test]
    fn row_count_test() {
        use CoordState::*;
        type Counter = RowCounter<15>;

        let mut row = Row::new(15);
        assert_eq!(Counter::count(row).get(), &[(Empty, 15)]);

        row.set(1, Black);
        assert_eq!(
            Counter::count(row).get(),
            &[(Empty, 1), (Black, 1), (Empty, 13)]
        );
        row.set(0, Black);
        assert_eq!(Counter::count(row).get(), &[(Black, 2), (Empty, 13)]);
        row.set(3, White);
        row.set(14, White);
        assert_eq!(
            Counter::count(row).get(),
            &[(Black, 2), (Empty, 1), (White, 1), (Empty, 10), (White, 1)]
        );
    }

    #[derive(Clone, Copy, Debug)]
    struct Conn {
        len: u8,
        l_grp: bool,
        r_grp: bool,
    }
    impl Conn {
        pub fn new(len: u8) -> Self {
            Conn {
                len,
                l_grp: false,
                r_grp: false,
            }
        }
    }

    /// Represents a series of empty or non-empty positions cut from `RowCounter`,
    /// in which connected stones belongs to *one* side.
    #[derive(Debug)]
    struct Segment<const SZ: usize> {
        spaces: [u8; SZ],  // count of spaces = count + 1
        conns: [Conn; SZ], // lengths of connected stones
        count: usize,      // count of connections. index of array must be usize
    }

    impl<const SZ: usize> Segment<SZ> {
        pub fn new() -> Self {
            assert!(SZ >= 5 && SZ <= 26);
            Segment {
                spaces: [0; SZ],
                conns: [Conn::new(0); SZ],
                count: 0,
            }
        }

        pub fn clear(&mut self) {
            *self = Self::new();
        }
    }

    /// Serial (CPU) rule checker for the standard rule.
    #[derive(Clone, Debug)]
    pub struct StdSerCalc<const SZ: usize>;

    impl<const SZ: usize> StdSerCalc<SZ> {
        fn segment_check(seg: &mut Segment<SZ>, caro_mode: bool) -> RowScore {
            let mut score = RowScore::new();
            if seg.count == 0 {
                return score;
            }

            if !caro_mode {
                // space at boundary is unique (without risk of long connection)
                seg.spaces[0] += 1;
                seg.spaces[seg.count] += 1;
            }

            // spaces[i]          spaces[i + 1]              spaces[i + 2]
            //           conns[i]               conns[i + 1]               conns[i + 2]
            //           ------------ len_with_r -----------
            // note: range "0..n" in rust doesn't include n, while "0..=n" does include n

            // grouping
            for i in 0..seg.count - 1 {
                if seg.conns[i].len + seg.spaces[i + 1] + seg.conns[i + 1].len <= 5 {
                    seg.conns[i].r_grp = true;
                    seg.conns[i + 1].l_grp = true;
                }
            }

            // count score for each connection
            for i in 0..seg.count {
                if seg.conns[i].len > 5 {
                    continue;
                }
                if seg.conns[i].len == 5 {
                    if !caro_mode || (seg.spaces[i] > 0 && seg.spaces[i + 1] > 0) {
                        score.flag_5 = true;
                    }
                    continue; // return score;
                }

                let l_valid = !seg.conns[i].l_grp && seg.conns[i].len + (seg.spaces[i] - 1) >= 5;
                let r_valid =
                    !seg.conns[i].r_grp && seg.conns[i].len + (seg.spaces[i + 1] - 1) >= 5;

                if l_valid {
                    score.add(seg.conns[i].len, 1); // the left is valid
                }
                if r_valid {
                    score.add(seg.conns[i].len, 1); // the right is valid
                }
                if l_valid && r_valid && seg.conns[i].len == 3 {
                    score.flag_live_3 = true;
                }
                if !(l_valid || r_valid)
                    && (seg.spaces[i] - 1) + seg.conns[i].len + (seg.spaces[i + 1] - 1) >= 5
                {
                    // one of conns[i].l_grp and conns[i].r_grp should be false, except case of X...X...X
                    score.add(seg.conns[i].len, 1); // borrowing
                }
            }

            // count score for groups
            for i in 0..seg.count - 1 {
                if !seg.conns[i].r_grp {
                    continue;
                }
                let len_score = seg.conns[i].len + seg.conns[i + 1].len;
                let len_with_r = len_score + seg.spaces[i + 1];

                if len_with_r == 5 {
                    score.add(len_score, 1);
                    continue; // don't consider the left/right side
                }

                let l_valid = len_with_r + (seg.spaces[i] - 1) >= 5;
                // `r_valid` and `r_dbl_grp` cannot be both true
                let r_valid = len_with_r + (seg.spaces[i + 2] - 1) >= 5;
                let r_dbl_grp = i + 2 < seg.count
                    && (len_with_r + seg.spaces[i + 2] + seg.conns[i + 2].len) == 5;

                if l_valid {
                    score.add(len_score, 1); // the left is valid
                }
                if r_valid {
                    score.add(len_score, 1); // the right is valid
                }
                if r_dbl_grp {
                    // the only case of double grouping (occurs at the right side)
                    score.add(3, 1); // -X.X.X-
                }
                if l_valid && r_valid && len_score == 3 {
                    score.flag_live_3 = true;
                }
                if !(l_valid || r_valid || r_dbl_grp)
                    && len_with_r + (seg.spaces[i] - 1) + (seg.spaces[i + 2] - 1) >= 5
                {
                    score.add(len_score, 1); // borrowing
                }
            }

            score
        }
    }

    impl<const SZ: usize> Rule<SZ> for StdSerCalc<SZ> {
        #[inline(always)]
        fn check_row(&self, row: Row) -> (RowScore, RowScore) {
            let counter = RowCounter::count(row);
            counter.check(|seg| Self::segment_check(seg, false))
        }
    }

    #[test]
    fn std_ser_calc_test() {
        use crate::CoordState;
        use crate::legacy::StdSerCalc;

        const B: CoordState = CoordState::Black;
        const W: CoordState = CoordState::White;
        const E: CoordState = CoordState::Empty;

        let calc = StdSerCalc::<19>;

        let row = [W, E, W, E, W, E, E, E, E];
        assert_eq!(
            calc.check_row(row[..].into()).1,
            RowScore {
                cnts: [1, 1, 1, 0],
                flag_live_3: false,
                flag_5: false
            }
        );

        let row = [E, E, W, W, E, E, W, E, W];
        assert_eq!(
            calc.check_row(row[..].into()).1,
            RowScore {
                cnts: [0, 1, 1, 0],
                flag_live_3: false,
                flag_5: false
            }
        );

        let row = [E, E, W, W, E, E, W, W, E];
        assert_eq!(
            calc.check_row(row[..].into()).1,
            RowScore {
                cnts: [0, 1, 0, 0],
                flag_live_3: false,
                flag_5: false
            }
        );

        let row = [E, E, W, W, E, E, W, E, W, W, W, E, W, W, W];
        assert_eq!(
            calc.check_row(row[..].into()).1,
            RowScore {
                cnts: [0, 1, 1, 1],
                flag_live_3: false,
                flag_5: false
            }
        );

        let row = [E, E, W, W, E, E, W, W, W, E, E, W, E, E, E];
        assert_eq!(
            calc.check_row(row[..].into()).1,
            RowScore {
                cnts: [1, 1, 1, 0],
                flag_live_3: false,
                flag_5: false
            }
        );

        let row = [E, E, W, W, E, E, W, W, W, E, E, W, W, E, E];
        assert_eq!(
            calc.check_row(row[..].into()).1,
            RowScore {
                cnts: [0, 2, 1, 0],
                flag_live_3: false,
                flag_5: false
            }
        );

        let row = [E, E, W, E, E, E, W, W, W, E, E, E, W, E, E];
        assert_eq!(
            calc.check_row(row[..].into()).1,
            RowScore {
                cnts: [2, 0, 2, 0],
                flag_live_3: true,
                flag_5: false
            }
        );

        let row = [W, E, W, E, W, E, W, E, W, W, W, E, E, E, E];
        assert_eq!(
            calc.check_row(row[..].into()).1,
            RowScore {
                cnts: [0, 0, 3, 1],
                flag_live_3: false,
                flag_5: false
            }
        );

        let row = [E, E, W, W, E, E, E, E, W, W, E, E];
        assert_eq!(
            calc.check_row(row[..].into()).1,
            RowScore {
                cnts: [0, 2, 0, 0],
                flag_live_3: false,
                flag_5: false
            }
        );

        let row = [E, E, W, W, E, E, W, E, W, E, E, W];
        assert_eq!(
            calc.check_row(row[..].into()).1,
            RowScore {
                cnts: [0, 2, 1, 0],
                flag_live_3: false,
                flag_5: false
            }
        );

        let row = [E, E, W, W, W, W, W, E, W, E, E, W];
        assert_eq!(calc.check_row(row[..].into()).1.flag_5, true);

        let row = [W, W, W, W, W, W, B, E, E, E, E, W];
        assert_eq!(
            calc.check_row(row[..].into()).1,
            RowScore {
                cnts: [1, 0, 0, 0],
                flag_live_3: false,
                flag_5: false
            }
        );

        let row = [B, E, W, E, W, E, E, W, E, E, E, B, W, E, E, W, W, E, E];
        assert_eq!(
            calc.check_row(row[..].into()).1,
            RowScore {
                cnts: [1, 3, 1, 0],
                flag_live_3: false,
                flag_5: false
            }
        );

        let row = [E, E, W, B, E, E, W, E, E, E, W, E, E, E, W, B, B, B, B];
        assert_eq!(
            calc.check_row(row[..].into()).1,
            RowScore {
                cnts: [2, 2, 0, 0],
                flag_live_3: false,
                flag_5: false
            }
        );

        let row = [B, B, E, E, E, E, E, B, W, E, E, W, E, E, E, B, E];
        assert_eq!(
            calc.check_row(row[..].into()),
            (
                RowScore {
                    cnts: [2, 1, 0, 0],
                    flag_live_3: false,
                    flag_5: false
                },
                RowScore {
                    cnts: [1, 1, 0, 0],
                    flag_live_3: false,
                    flag_5: false
                }
            )
        );
    }

    /// Serial (CPU) rule checker for the caro rule.
    #[derive(Clone, Debug)]
    pub struct CaroSerCalc<const SZ: usize>;

    impl<const SZ: usize> Rule<SZ> for CaroSerCalc<SZ> {
        #[inline(always)]
        fn check_row(&self, row: Row) -> (RowScore, RowScore) {
            let counter = RowCounter::count(row);
            counter.check(|seg| StdSerCalc::<SZ>::segment_check(seg, true))
        }
    }

    /// Serial (CPU) rule checker for the freestyle rule.
    #[derive(Clone, Debug)]
    pub struct FreeSerCalc<const SZ: usize>;

    impl<const SZ: usize> FreeSerCalc<SZ> {
        fn segment_check(seg: &mut Segment<SZ>) -> RowScore {
            let mut score = RowScore::new();
            if seg.count == 0 {
                return score;
            }

            // spaces[i]          spaces[i + 1]              spaces[i + 2]
            //           conns[i]               conns[i + 1]               conns[i + 2]
            //           ------------ len_with_r -----------

            // grouping
            for i in 0..seg.count - 1 {
                if seg.spaces[i + 1] < 5 - seg.conns[i].len
                    && seg.spaces[i + 1] < 5 - seg.conns[i + 1].len
                {
                    seg.conns[i].r_grp = true;
                    seg.conns[i + 1].l_grp = true;
                }
            }

            // count score for each connection
            for i in 0..seg.count {
                if seg.conns[i].len >= 5 {
                    score.flag_5 = true;
                    continue; // return score;
                }

                let l_valid = !seg.conns[i].l_grp && seg.conns[i].len + seg.spaces[i] >= 5;
                let r_valid = !seg.conns[i].r_grp && seg.conns[i].len + seg.spaces[i + 1] >= 5;

                if l_valid {
                    score.add(seg.conns[i].len, 1); // the left is valid
                }
                if r_valid {
                    score.add(seg.conns[i].len, 1); // the right is valid
                }
                if l_valid && r_valid && seg.conns[i].len == 3 {
                    score.flag_live_3 = true;
                }
                if !(l_valid || r_valid)
                    && seg.spaces[i] + seg.conns[i].len + seg.spaces[i + 1] >= 5
                {
                    score.add(seg.conns[i].len, 1); // borrowing
                }
            }

            // count score for groups
            for i in 0..seg.count - 1 {
                if !seg.conns[i].r_grp {
                    continue;
                }
                let len_with_r_no_sp = seg.conns[i].len + seg.conns[i + 1].len;
                let len_with_r = len_with_r_no_sp + seg.spaces[i + 1];
                let len_score = if len_with_r <= 5 {
                    len_with_r_no_sp
                } else {
                    5_u8.saturating_sub(seg.spaces[i + 1])
                };
                if len_score == 0 {
                    continue;
                }

                if len_with_r >= 5 {
                    score.add(len_score, 1);
                    continue; // don't consider the left/right side
                }

                let l_valid = len_with_r + seg.spaces[i] >= 5;
                // `r_valid` and `r_dbl_grp` cannot be both true
                let r_valid = len_with_r + seg.spaces[i + 2] >= 5;
                let r_dbl_grp = i + 2 < seg.count
                    && (len_with_r + seg.spaces[i + 2] + seg.conns[i + 2].len) == 5;

                if l_valid {
                    score.add(len_score, 1); // the left is valid
                }
                if r_valid {
                    score.add(len_score, 1); // the right is valid
                }
                if r_dbl_grp {
                    // the only case of double grouping (occurs at the right side)
                    score.add(3, 1); // -X.X.X-
                }
                if l_valid && r_valid && len_score == 3 {
                    score.flag_live_3 = true;
                }
                if !(l_valid || r_valid || r_dbl_grp)
                    && len_with_r + seg.spaces[i] + seg.spaces[i + 2] >= 5
                {
                    score.add(len_score, 1); // borrowing
                }
            }

            score
        }
    }

    impl<const SZ: usize> Rule<SZ> for FreeSerCalc<SZ> {
        #[inline(always)]
        fn check_row(&self, row: Row) -> (RowScore, RowScore) {
            let counter = RowCounter::count(row);
            counter.check(Self::segment_check)
        }
    }
}
