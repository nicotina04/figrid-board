use crate::{Coord, CoordState};

/// A row on the board. The length is determined at runtime.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Row {
    data: u64,
}

impl Row {
    pub(crate) const E: u64 = 0b00;
    pub(crate) const B: u64 = 0b10;
    pub(crate) const W: u64 = 0b11;
    const LEN_OFFSET: u8 = 64 - 5;

    /// Creates an empty row.
    pub fn new(len: u8) -> Self {
        assert!((1u8..=26u8).contains(&len));
        Self {
            data: (len as u64) << Self::LEN_OFFSET,
        }
    }

    /// Returns the length of the row.
    #[inline(always)]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> u8 {
        (self.data >> Self::LEN_OFFSET) as u8
    }

    /// Clears all stones in the row.
    pub fn clear(&mut self) {
        self.data = (self.len() as u64) << Self::LEN_OFFSET;
    }

    /// Gets the raw value that represents the row, except the raw bits
    /// for the row length. Used internally.
    #[inline(always)]
    pub fn get_raw(&self) -> u64 {
        self.data & !(0b11111 << Self::LEN_OFFSET)
    }

    #[inline(always)]
    fn bit_pos(pos: u8) -> u8 {
        2 * pos
    }

    /// Gets the [CoordState] at given index in the row.
    /// The index is unchecked; caller should make sure of `pos < len()`.
    #[inline(always)]
    pub fn get(&self, pos: u8) -> CoordState {
        match (self.data >> Self::bit_pos(pos)) & 0b11 {
            Self::B => CoordState::Black,
            Self::W => CoordState::White,
            _ => CoordState::Empty,
        }
    }

    /// Iterates through [CoordState]s of the row.
    #[inline(always)]
    pub fn iter(
        &self,
    ) -> std::iter::Map<std::ops::Range<u8>, impl FnMut(u8) -> CoordState + use<'_>> {
        (0..self.len()).map(|i| self.get(i))
    }

    /// Sets the [CoordState] at given index in the row.
    /// The index is unchecked; caller should make sure of `pos < len()`.
    #[inline(always)]
    pub fn set(&mut self, pos: u8, state: CoordState) {
        let mask = u64::MAX ^ (0b11 << Self::bit_pos(pos));
        self.data &= mask;
        self.data |= match state {
            CoordState::Empty => Self::E,
            CoordState::Black => Self::B,
            CoordState::White => Self::W,
        } << Self::bit_pos(pos);
    }
}

impl std::fmt::Debug for Row {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::fmt::Write;
        for st in self.iter() {
            let ch = match st {
                CoordState::Black => 'X',
                CoordState::White => 'O',
                CoordState::Empty => '.',
            };
            f.write_char(ch)?;
        }
        Ok(())
    }
}

impl From<&[CoordState]> for Row {
    fn from(sts: &[CoordState]) -> Self {
        if sts.is_empty() {
            return Self::new(1);
        }
        let mut row = Self::new(sts.len().min(26) as u8);
        for i in 0..row.len() {
            row.set(i, sts[i as usize]);
        }
        row
    }
}

impl<const SZ: usize> Default for Rows<SZ> {
    fn default() -> Self {
        Self::new()
    }
}

#[test]
fn row_test() {
    let mut row = Row::new(15);
    assert_eq!(row.len(), 15);
    assert_eq!(row.get_raw(), 0b_00_00_00_00_00);
    row.set(0, CoordState::Black);
    assert_eq!(row.get(0), CoordState::Black);
    assert_eq!(row.get_raw(), 0b_00_00_00_00_10);
    row.set(2, CoordState::White);
    assert_eq!(row.get(2), CoordState::White);
    assert_eq!(row.get_raw(), 0b_00_00_11_00_10);
    row.set(2, CoordState::Black);
    assert_eq!(row.get(2), CoordState::Black);
    row.set(0, CoordState::Empty);
    assert_eq!(row.get(0), CoordState::Empty);
    assert_eq!(row.len(), 15);
    row.clear();
    assert_eq!(row.get_raw(), 0b_00_00_00_00_00);
}

/// Storage of all [Row]s on the square board of size `SZ`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rows<const SZ: usize> {
    horizontal: [Row; SZ],  // [a1, b1...o1], [a2, b2...o2]...[a15, b15...o15]
    vertical: [Row; SZ],    // [a1, a2...a15], [b1, b2...b15]...[o1, o2...o15]
    diagonal_l1: [Row; SZ], // [a15], [a14, b15], [a13, b14, c15]...[a1, b2...o15]  len: 1 to SZ             x <= y, y - x: SZ-1..0
    diagonal_l2: [Row; SZ], // [b1, c2...o14], [c1, d2...o13]...[n1, o2], [o1]      len: SZ-1 to 1 (then 0?) x >  y, x - y: 1..SZ-1
    diagonal_r1: [Row; SZ], // [a1] , [a2 , b1], [a3,  b2,  c1]...[a15, b14...o1]   len: 1 to SZ             x + y <= SZ-1
    diagonal_r2: [Row; SZ], // [b15, c14...o2], [c15, d14...o3]...[n15, o14], [o15] len: SZ-1 to 1 (then 0?) x + y >  SZ-1
}

impl<const SZ: usize> Rows<SZ> {
    /// Creates a `Rows` for an empty board.
    pub fn new() -> Self {
        assert!(SZ >= 5 && SZ <= 26);
        let fake = Row::new(SZ as u8);
        let mut rows = Self {
            horizontal: [fake; SZ],
            vertical: [fake; SZ],
            diagonal_l1: [fake; SZ],
            diagonal_l2: [fake; SZ],
            diagonal_r1: [fake; SZ],
            diagonal_r2: [fake; SZ],
        };
        rows.clear();
        rows
    }

    /// Clears all rows on the board.
    pub fn clear(&mut self) {
        self.horizontal = [Row::new(SZ as u8); SZ];
        self.vertical = [Row::new(SZ as u8); SZ];
        for len in 1..=SZ {
            self.diagonal_l1[len - 1] = Row::new(len as u8);
            self.diagonal_r1[len - 1] = Row::new(len as u8);
            if SZ > len {
                self.diagonal_l2[len - 1] = Row::new((SZ - len) as u8);
                self.diagonal_r2[len - 1] = Row::new((SZ - len) as u8);
            }
        }
    }

    /// Returns the state at a coordinate, or `CoordState::Empty` if `coord` is null.
    #[inline(always)]
    pub fn get(&self, coord: Coord<SZ>) -> CoordState {
        let Some((x, y)) = coord.get() else {
            return CoordState::Empty;
        };
        self.horizontal[y as usize].get(x)
    }

    /// Returns the horizontal, vertical, left and right diagonal [Row]s passing through
    /// `coord`, or `None` if `coord` is null.
    #[inline]
    pub fn get_quad_rows(&self, coord: Coord<SZ>) -> Option<[Row; 4]> {
        let (x, y) = coord.get()?;

        let row_diagonal_l = if x <= y {
            self.diagonal_l1[SZ - 1 - (y - x) as usize]
        } else {
            self.diagonal_l2[(x - y - 1) as usize]
        };

        let row_diagonal_r = if x + y <= SZ as u8 - 1 {
            self.diagonal_r1[(x + y) as usize]
        } else {
            self.diagonal_r2[(x + y) as usize - SZ]
        };

        Some([
            self.horizontal[y as usize],
            self.vertical[x as usize],
            row_diagonal_l,
            row_diagonal_r,
        ])
    }

    /// Sets the `CoordState` at `coord`, updating 4 rows passing through it.
    #[inline]
    pub fn set(&mut self, coord: Coord<SZ>, st: CoordState) {
        let Some((x, y)) = coord.get() else {
            return;
        };

        self.horizontal[y as usize].set(x, st);
        self.vertical[x as usize].set(y, st);

        if x <= y {
            self.diagonal_l1[SZ - 1 - (y - x) as usize].set(x, st);
        } else {
            self.diagonal_l2[(x - y - 1) as usize].set(y, st);
        }

        if x + y <= SZ as u8 - 1 {
            self.diagonal_r1[(x + y) as usize].set(x, st);
        } else {
            self.diagonal_r2[(x + y) as usize - SZ].set(SZ as u8 - 1 - y, st);
        }
    }
}

#[test]
fn rows_test() {
    let mut rows = Rows::<15>::new();
    let rec = ["h7", "i7", "j8", "k9", "i9", "h10", "k8", "i8"];
    let mut st = CoordState::Black;
    for coord in rec.iter() {
        rows.set(coord.parse().unwrap(), st);
        st = if st == CoordState::Black {
            CoordState::White
        } else {
            CoordState::Black
        };
    }

    let quad_rows = rows.get_quad_rows(Coord::from(7, 7)).unwrap();
    assert_eq!(quad_rows[0].len(), 15);
    assert_eq!(
        format!(
            "{:?} {:?} {:?} {:?}",
            quad_rows[0], quad_rows[1], quad_rows[2], quad_rows[3]
        ),
        "........OXX.... ......X..O..... ........X...... ........O......"
    );

    let quad_rows = rows.get_quad_rows(Coord::from(7, 5)).unwrap();
    assert_eq!(
        format!(
            "{:?} {:?} {:?} {:?}",
            quad_rows[0], quad_rows[1], quad_rows[2], quad_rows[3]
        ),
        "............... ......X..O..... ......OXO.... ............."
    );

    let quad_rows = rows.get_quad_rows(Coord::from(9, 8)).unwrap();
    assert_eq!(
        format!(
            "{:?} {:?} {:?} {:?}",
            quad_rows[0], quad_rows[1], quad_rows[2], quad_rows[3]
        ),
        "........X.O.... .......X....... ......XO...... .......X...."
    );

    let quad_rows = rows.get_quad_rows(Coord::from(6, 8)).unwrap();
    assert_eq!(
        format!(
            "{:?} {:?} {:?} {:?}",
            quad_rows[0], quad_rows[1], quad_rows[2], quad_rows[3]
        ),
        "........X.O.... ............... .......O..... ........O......"
    );

    rows.clear();
    rows.set(Coord::from(0, 0), CoordState::Black);
    rows.set(Coord::from(0, 14), CoordState::White);
    rows.set(Coord::from(14, 0), CoordState::Black);
    rows.set(Coord::from(14, 14), CoordState::White);
    let quad_rows = rows.get_quad_rows(Coord::from(0, 0)).unwrap();
    assert_eq!(
        format!(
            "{:?} {:?} {:?} {:?}",
            quad_rows[0], quad_rows[1], quad_rows[2], quad_rows[3]
        ),
        "X.............X X.............O X.............O X"
    );
    let quad_rows = rows.get_quad_rows(Coord::from(14, 0)).unwrap();
    assert_eq!(
        format!(
            "{:?} {:?} {:?} {:?}",
            quad_rows[0], quad_rows[1], quad_rows[2], quad_rows[3]
        ),
        "X.............X X.............O X O.............X"
    );
}
