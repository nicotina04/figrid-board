use std::{fmt::Display, ops::Index, str::FromStr};

use super::{
    Rec, // trait
    Row,
    Rows,
};
use crate::{Coord, CoordState, Error};

/// Basic record structure without any evaluator or rule checker.
///
/// `SZ` is the board size, `LEN` is the maximum capacity (it should be more than `SZ*SZ`).
#[derive(Clone, Debug)]
pub struct BaseRec<const SZ: usize, const LEN: usize> {
    len: usize,
    cnt_pass: u8,
    seq: [Coord<SZ>; LEN],
    rows: Rows<SZ>,
}

pub type BaseRec15 = BaseRec<15, { 15 * 15 + 16 }>; //maximum 16 pass moves
pub type BaseRec20 = BaseRec<20, { 20 * 20 + 16 }>;

impl<const SZ: usize, const LEN: usize> BaseRec<SZ, LEN> {
    const SZ_X_SZ: usize = SZ * SZ;

    /// Returns an empty record.
    #[inline]
    pub fn new() -> Self {
        assert!(SZ >= 5 && SZ <= 26);
        Self {
            len: 0,
            cnt_pass: 0u8,
            seq: [Coord::new(); LEN],
            rows: Rows::new(),
        }
    }
}

impl<const SZ: usize, const LEN: usize> Rec<SZ> for BaseRec<SZ, LEN> {
    #[inline(always)]
    fn as_slice(&self) -> &[Coord<SZ>] {
        &self.seq[..self.len]
    }

    #[inline(always)]
    fn coord_state(&self, coord: Coord<SZ>) -> CoordState {
        self.rows.get(coord)
    }

    #[inline(always)]
    fn get_quad_rows(&self, coord: Coord<SZ>) -> Option<[Row; 4]> {
        self.rows.get_quad_rows(coord)
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    fn len_max(&self) -> usize {
        LEN
    }

    #[inline]
    fn stones_count(&self) -> usize {
        self.len - self.cnt_pass as usize
    }

    #[inline(always)]
    fn is_full(&self) -> bool {
        // `>` should be impossible
        self.len >= LEN || self.stones_count() >= Self::SZ_X_SZ
    }

    #[inline(always)]
    fn add(&mut self, coord: Coord<SZ>) -> Result<(), Error> {
        if self.len >= LEN {
            return Err(Error::RecIsFull);
        }
        if !self.coord_state(coord).is_empty() {
            return Err(Error::CoordNotEmpty);
        }

        if coord.is_real() {
            self.rows.set(coord, self.color_next());
        } else {
            self.cnt_pass += 1;
        }
        // SAFETY: assumes self.len is valid
        *unsafe { self.seq.get_unchecked_mut(self.len) } = coord;
        self.len += 1;
        Ok(())
    }

    #[inline(always)]
    fn undo(&mut self) -> Result<Coord<SZ>, Error> {
        let coord = self.last_coord().ok_or(Error::RecIsEmpty)?;
        self.len -= 1;
        if coord.is_real() {
            self.rows.set(coord, CoordState::Empty);
        } else {
            self.cnt_pass -= 1;
        }
        Ok(coord)
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.len = 0;
        self.cnt_pass = 0;
        self.rows.clear();
    }
}

// Note: these traits can't be implemented automatically for all types with trait Rec<SZ>

impl<const SZ: usize, const LEN: usize> Default for BaseRec<SZ, LEN> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const SZ: usize, const LEN: usize> FromStr for BaseRec<SZ, LEN> {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut rec = Self::new();
        rec.append_str(s).map_err(|_| Error::ParseError)?;
        Ok(rec)
    }
}

impl<const SZ: usize, const LEN: usize> Index<usize> for BaseRec<SZ, LEN> {
    type Output = Coord<SZ>;

    #[inline(always)]
    fn index(&self, i: usize) -> &Coord<SZ> {
        &self.as_slice()[i]
    }
}

impl<const SZ: usize, const LEN: usize> Display for BaseRec<SZ, LEN> {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.print_str(f, " ", true)
    }
}

impl<const SZ: usize, const LEN: usize> PartialEq for BaseRec<SZ, LEN> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<const SZ: usize, const LEN: usize> Eq for BaseRec<SZ, LEN> {}
