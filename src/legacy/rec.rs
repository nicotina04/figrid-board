use std::{fmt, slice};

use super::Row;
use crate::{Coord, CoordState, Error, Rotation};

/// Trait for all types that store `Coord<SZ>` sequence in continuous memory
/// and don't allow any coordinate value (except null) to be repeated.
/// Any even index corresponds to a black stone, odd index means it is white.
pub trait Rec<const SZ: usize> {
    /// Returns a slice of all added coordinates.
    fn as_slice(&self) -> &[Coord<SZ>];
    /// Returns the state at a coordinate, or `CoordState::Empty` if `coord` is null.
    fn coord_state(&self, coord: Coord<SZ>) -> CoordState;
    /// Returns the horizontal, vertical, left and right diagonal [Row]s passing through
    /// `coord`, or `None` if `coord` is null.
    fn get_quad_rows(&self, coord: Coord<SZ>) -> Option<[Row; 4]>;
    /// Count of all added coordinates.
    fn len(&self) -> usize;
    /// Possible maximum amount of coordinates (including null).
    fn len_max(&self) -> usize;
    /// Count of all non-null coordinate.
    fn stones_count(&self) -> usize;
    /// Returns `false` if `len_max` isn't reached and the board isn't full.
    fn is_full(&self) -> bool;

    /// Adds a coordinate into the sequence. Returns `Error::CoordNotEmpty`
    /// if the coordinate exists in the stored sequence. Other rules can be defined.
    fn add(&mut self, coord: Coord<SZ>) -> Result<(), Error>;
    /// Undo the last move (removes the last coordinate). Returns `Error::RecIsEmpty`
    /// if there is no move. Other rules can be defined.
    fn undo(&mut self) -> Result<Coord<SZ>, Error>;
    /// Clears the record.
    fn clear(&mut self);

    /// Returns an iterator of all added coordinates (from first to last).
    #[inline(always)]
    fn iter(&self) -> slice::Iter<'_, Coord<SZ>> {
        self.as_slice().iter()
    }

    /// Checks if the record is empty.
    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the last coordinate (move), or `None` if the record is empty.
    #[inline(always)]
    fn last_coord(&self) -> Option<Coord<SZ>> {
        if !self.is_empty() {
            // SAFETY: assumes self.len is valid
            Some(*unsafe { self.as_slice().get_unchecked(self.len() - 1) })
        } else {
            None
        }
    }

    /// Returns the stone color of the next coordinate (move) to be added.
    /// It does not check if the record is full.
    #[inline(always)]
    fn color_next(&self) -> CoordState {
        if self.len() % 2 == 0 {
            CoordState::Black
        } else {
            CoordState::White
        }
    }

    /// Tries to append all coordinates got from iterator `coords`
    /// into the current record, stops at the first failure of `add()`.
    /// Always return the count of added coordinates.
    #[inline]
    fn append(&mut self, coords: impl Iterator<Item = Coord<SZ>>) -> Result<usize, usize>
    where
        Self: Sized,
    {
        let mut i: usize = 0;
        for c in coords {
            if self.add(c).is_err() {
                return Err(i);
            }
            i += 1;
        }
        Ok(i)
    }

    /// Tries to append all coordinates parsed from `str_coords`
    /// into the current record, stops at the first failure of `add()`.
    /// Always return the count of added coordinates.
    fn append_str(&mut self, str_coords: &str) -> Result<usize, usize> {
        let mut len_checked = 0;
        let mut cnt_added = 0;
        while let Some((c, l)) = Coord::<SZ>::parse_str(&str_coords[len_checked..]) {
            self.add(c).map_err(|_| cnt_added)?;
            cnt_added += 1;
            len_checked += l;
        }
        if cnt_added > 0 { Ok(cnt_added) } else { Err(0) }
    }

    /// Finds `coord` in the sequence reversely, returns the count of
    /// removed coordinates after `coord` if it's found, otherwise
    /// returns `Error::ItemNotExist`.
    #[inline]
    fn back_to(&mut self, coord: Coord<SZ>) -> Result<usize, Error> {
        if let Some(i) = self.iter().rposition(|&c| c == coord) {
            let steps = (self.len() - 1) - i;
            for _ in 0..steps {
                let _ = self.undo();
            }
            Ok(steps)
        } else {
            Err(Error::ItemNotExist)
        }
    }

    /// Rotates the board by `rotation`, changes all valid coordinates.
    #[inline(always)]
    fn rotate(&mut self, rotation: Rotation)
    where
        Self: Sized,
    {
        self.transform(|c| c.rotate(rotation))
    }

    /// Translates all stones by `x` and `y` offsets, changes all valid coordinates.
    /// Returns `Error::TransformFailed` if some stones are out of range after translation,
    /// in this case the original record should be kept.
    #[inline(always)]
    fn translate(&mut self, x: i8, y: i8) -> Result<(), Error>
    where
        Self: Sized,
    {
        self.transform_checked(|c| c.offset(x, y))
    }

    /// Changes values of all valid coordinates by `f` and refreshes the board.
    /// Do make sure `f` is one-to-one mapping and the returned value is always valid,
    /// otherwise it may generate an unknown result.
    fn transform(&mut self, f: impl Fn(Coord<SZ>) -> Coord<SZ>)
    where
        Self: Sized,
    {
        // Note: Vec is used
        let seq = self
            .iter()
            .map(|&c| if c.is_real() { f(c) } else { c })
            .collect::<Vec<_>>();

        self.clear();
        for c in seq {
            let _ = self.add(c);
        }
    }

    /// Changes values of all valid coordinates by `f` and refreshes the board.
    /// Returns `Error::TransformFailed` if failed, in such case the record sequence
    /// becomes dirty (has an unknown length).
    fn transform_checked<F>(&mut self, f: F) -> Result<(), Error>
    where
        F: Fn(Coord<SZ>) -> Option<Coord<SZ>>,
        Self: Sized,
    {
        let seq = self
            .iter()
            .map(|&c| if c.is_real() { f(c) } else { Some(c) })
            .collect::<Vec<_>>();

        for c in &seq {
            if c.is_none() {
                return Err(Error::TransformFailed);
            }
        }
        self.clear();
        for c in seq {
            self.add(c.unwrap()).map_err(|_| Error::TransformFailed)?;
        }
        Ok(())
    }

    /// Prints the current record (for example, to a `String`) in a readable format.
    /// It might be overrided if the format does not meet the requirement.
    ///
    /// ```
    /// use figrid_board::legacy::{ Rec, BaseRec15 };
    /// let rec: BaseRec15 = "l10j7k6m11m13".parse().unwrap();
    /// let mut s = String::new();
    /// rec.print_str(&mut s, ", ", false);
    /// assert_eq!(s, "l10, j7, k6, m11, m13");
    /// s.clear(); rec.print_str(&mut s, ",", true);
    /// assert_eq!(s, "1. l10,j7 2. k6,m11 3. m13");
    /// ```
    fn print_str(&self, f: &mut impl fmt::Write, divider: &str, print_no: bool) -> std::fmt::Result
    where
        Self: Sized,
    {
        if self.is_empty() {
            return Ok(());
        }
        if print_no {
            for (i, c) in self.iter().enumerate() {
                if i % 2 == 0 {
                    let div = if i < self.len() - 1 { divider } else { "" };
                    write!(f, "{}. {}{}", i / 2 + 1, c, div)?;
                } else {
                    write!(f, "{} ", c)?;
                }
            }
        } else {
            for c in self.iter().take(self.len() - 1) {
                write!(f, "{}{}", c, divider)?;
            }
            write!(f, "{}", self.last_coord().unwrap())?;
        }
        Ok(())
    }

    /// Draws the board by characters. `dots` gives extra information to be shown on the board,
    /// set to `&[]` if not needed; set `full_char` to `true` if non-ASCII output is possible
    /// (and ambiguous width characters should be fullwidth).
    fn print_board(
        &self,
        f: &mut impl fmt::Write,
        dots: &[Coord<SZ>],
        full_char: bool,
    ) -> std::fmt::Result
    where
        Self: Sized,
    {
        #[allow(non_snake_case)]
        let SIZE: u8 = SZ as u8;
        #[rustfmt::skip]
        let (
            // actually string literals, not chars
            ch_black, ch_black_last, ch_white, ch_white_last, ch_emp, ch_dot,
            ch_b, ch_u, ch_l, ch_r, ch_bl, ch_br, ch_ul, ch_ur,
        ) = if full_char {
            ("●", "◆", "○", "⊙", "┼", "·", "┴", "┬", "├", "┤", "└", "┘", "┌", "┐")
        } else {
            (" X", " #", " O", " Q", " .", " *", " .", " .", " .", " .", " .", " .", " .", " .")
        };

        for y in (0..SIZE).rev() {
            write!(f, " {:2}", y + 1)?;
            for x in 0..SIZE {
                let coord = Coord::from(x, y);
                let st = self.coord_state(coord);
                let ch = match st {
                    CoordState::Empty => {
                        if dots.iter().any(|&c| coord == c) {
                            ch_dot
                        } else if x == 0 && y == 0 {
                            ch_bl
                        } else if x == SIZE - 1 && y == 0 {
                            ch_br
                        } else if x == 0 && y == SIZE - 1 {
                            ch_ul
                        } else if x == SIZE - 1 && y == SIZE - 1 {
                            ch_ur
                        } else if x == 0 {
                            ch_l
                        } else if x == SIZE - 1 {
                            ch_r
                        } else if y == 0 {
                            ch_b
                        } else if y == SIZE - 1 {
                            ch_u
                        } else {
                            ch_emp
                        }
                    }
                    CoordState::Black => {
                        if coord != self.last_coord().unwrap_or(Coord::new()) {
                            ch_black
                        } else {
                            ch_black_last
                        }
                    }
                    CoordState::White => {
                        if coord != self.last_coord().unwrap_or(Coord::new()) {
                            ch_white
                        } else {
                            ch_white_last
                        }
                    }
                };
                write!(f, "{ch}")?;
            }
            writeln!(f)?;
        }
        write!(f, "    ")?;
        for h in 0..SIZE {
            write!(f, "{} ", crate::coord::coord_x_letter(h))?;
        }
        writeln!(f)?;
        Ok(())
    }
}
