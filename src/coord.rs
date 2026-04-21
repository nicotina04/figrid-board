use std::{fmt::Display, str::FromStr};

use crate::Error;

/// State of a coordinate (position) on the board.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum CoordState {
    Empty,
    Black,
    White,
}

impl CoordState {
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        *self == CoordState::Empty
    }

    #[inline(always)]
    pub fn is_stone(&self) -> bool {
        !self.is_empty()
    }

    #[inline(always)]
    pub fn is_black(&self) -> bool {
        *self == CoordState::Black
    }

    #[inline(always)]
    pub fn is_white(&self) -> bool {
        *self == CoordState::White
    }
}

/// Coordinate on the board of size `SZ`, or a null coordinate not on the board.
/// For example, coordinate of "a1" is `Coord {x: 0, y: 0}`.
///
/// Null coordinate is allowed for pass moves and possible wrapper of `Rec`
/// to store swapping info. Designed for making different amount of black and white
/// stones possible without storing color info (just check if the index is even).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Coord<const SZ: usize> {
    x: u8,
    y: u8,
}

/// The `x` or `y` value of a null coordinate.
pub(crate) const COORD_NULL_VAL: u8 = 0xff;

pub type Coord15 = Coord<15>;
pub type Coord20 = Coord<20>;

impl<const SZ: usize> Coord<SZ> {
    const SIZE: u8 = if SZ >= 5 && SZ <= 26 {
        SZ as u8
    } else {
        panic!()
    };

    const COORD_NULL: Self = Self {
        x: COORD_NULL_VAL,
        y: COORD_NULL_VAL,
    };

    /// Returns a null coordinate.
    ///
    /// ```
    /// assert!(figrid_board::Coord15::new().is_null());
    /// ```
    #[inline(always)]
    pub fn new() -> Self {
        Self::COORD_NULL
    }

    /// Returns a coordinate object if `x` and `y` are within the board size,
    /// otherwise returns a null coordinate.
    ///
    /// ```
    /// let coord = figrid_board::Coord15::from(3, 9);
    /// assert!(coord.is_real());
    /// assert_eq!(format!("{coord}"), "d10");
    ///
    /// let coord = figrid_board::Coord15::from(9, 15);
    /// assert!(coord.is_null());
    /// assert_eq!(format!("{coord}"), "-");
    /// ```
    #[inline(always)]
    pub fn from(x: u8, y: u8) -> Self {
        if x < Self::SIZE && y < Self::SIZE {
            Self { x, y }
        } else {
            Self::COORD_NULL
        }
    }

    /// Use it carefully, it allows building invalid coordinate of which
    /// the x and y can be set to values other than COORD_NULL_VAL (UB!)
    #[inline(always)]
    pub(crate) unsafe fn build_unchecked(x: u8, y: u8) -> Self {
        Self { x, y }
    }

    /// Check if it is a null coordinate.
    #[inline(always)]
    pub fn is_null(&self) -> bool {
        *self == Self::COORD_NULL
    }

    /// Check if it is a real coordinate on the board.
    #[inline(always)]
    pub fn is_real(&self) -> bool {
        !self.is_null()
    }

    /// Returns the coordinate (x, y), or `None` if it is null.
    ///
    /// ```
    /// let coord: figrid_board::Coord15 = "h8".parse().unwrap();
    /// assert_eq!(coord.get(), Some((7, 7)));
    ///
    /// let coord = figrid_board::Coord15::new();
    /// assert_eq!(coord.get(), None);
    /// ```
    #[inline(always)]
    pub fn get(&self) -> Option<(u8, u8)> {
        if self.is_real() {
            Some((self.x, self.y))
        } else {
            None
        }
    }

    /// Returns the coordinate (x, y), panics if it is null.
    #[inline(always)]
    pub fn unwrap(&self) -> (u8, u8) {
        self.get().unwrap()
    }

    /// Returns the coordinate (x, y), which contains invalid values if it is null.
    ///
    /// # Safety
    ///
    /// The caller must make sure that this coordinate is not null.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self) -> (u8, u8) {
        (self.x, self.y)
    }

    /// Set the coordinate if `x` and `y` are within the board size,
    /// otherwise returns `Err(figrid_board::Error::InvalidCoord)`.
    ///
    /// ```
    /// let mut coord = figrid_board::Coord15::new();
    /// assert_eq!(coord.set(9, 15), Err(figrid_board::Error::InvalidCoord));
    /// assert_eq!(coord.get(), None);
    /// assert_eq!(coord.set(9, 9), Ok(()));
    /// assert_eq!(format!("{coord}"), "j10");
    /// ```
    #[inline(always)]
    pub fn set(&mut self, x: u8, y: u8) -> Result<(), Error> {
        if x < Self::SIZE && y < Self::SIZE {
            self.x = x;
            self.y = y;
            Ok(())
        } else {
            Err(Error::InvalidCoord)
        }
    }

    /// Set it to a null coordinate.
    #[inline(always)]
    pub fn set_null(&mut self) {
        *self = Self::COORD_NULL;
    }

    /// Use it carefully, it allows setting invalid coordinate of which
    /// the x and y can be values other than COORD_NULL_VAL (UB!)
    #[inline(always)]
    #[allow(dead_code)]
    pub(crate) unsafe fn set_unchecked(&mut self, x: u8, y: u8) {
        self.x = x;
        self.y = y;
    }
}

impl<const SZ: usize> Default for Coord<SZ> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const SZ: usize> FromStr for Coord<SZ> {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::parse_str(s).ok_or(Error::ParseError)?.0)
    }
}

impl<const SZ: usize> Display for Coord<SZ> {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_real() {
            write!(f, "{}{}", coord_x_letter(self.x), self.y + 1)
        } else {
            write!(f, "-")
        }
    }
}

impl<const SZ: usize> Coord<SZ> {
    /// Used by `FromStr` implementation of `Coord` and the parser of `Record`.
    /// Returns a coordinate and the parsed string length on success.
    #[inline]
    pub(crate) fn parse_str(str_coords: &str) -> Option<(Self, usize)> {
        const ALPHABET: &str = "abcdefghijklmnopqrstuvwxyz";
        let alphabet = &ALPHABET[..SZ];

        let mut len_checked: usize = 0;
        loop {
            let str_rem = &str_coords[len_checked..];
            let mut itr = str_rem.chars().enumerate();
            // finds a possible coord. i: index in str_rem, x: possible coord x
            let loc_alpha = itr.find_map(|(i, c)| alphabet.find(c).map(|x| (i, x)));
            if let Some((i, x)) = loc_alpha {
                len_checked += i + 1;
                // parse the possible coord y
                let mut num: Option<u32> = None;
                for (_, ch) in itr {
                    if !ch.is_ascii_digit() {
                        break;
                    }
                    if num.is_none() {
                        num = ch.to_digit(10);
                    } else {
                        num.replace(10 * num.unwrap() + ch.to_digit(10).unwrap());
                    }
                    len_checked += 1;
                }
                if let Some(n) = num {
                    if n == 0 || n > SZ as u32 {
                        continue;
                    }
                    return Some((
                        Self {
                            x: x as u8,
                            y: (n - 1) as u8,
                        },
                        len_checked,
                    ));
                }
            } else {
                return None;
            }
        }
    }
}

#[inline(always)]
pub(crate) fn coord_x_letter(x: u8) -> char {
    if x < 26 {
        // SAFETY: x < 26
        unsafe { char::from_u32_unchecked(('a' as u32) + x as u32) }
    } else {
        '?'
    }
}

/// Possible rotation operations.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Rotation {
    Original,          // 0 00, don't rotate
    Clockwise,         // 0 01, rotate 90 degrees
    CentralSymmetric,  // 0 10, equals rotate 180 degrees clockwise
    Counterclockwise,  // 0 11, equals rotate 270 degrees clockwise
    FlipHorizontal,    // 1 00, reflect by vertical line in the middle
    FlipLeftDiagonal,  // 1 01, equals FlipHorizontal | Clockwise
    FlipVertical,      // 1 10, equals FlipHorizontal | CentralSymmetric
    FlipRightDiagonal, // 1 11, equals FlipHorizontal | Counterclockwise
}

impl<const SZ: usize> Coord<SZ> {
    /// Returns the coordinate rotated by `rotation`.
    ///
    /// ```
    /// use figrid_board::{ Coord15, Rotation::* };
    /// let coord = Coord15::from(0, 1); // a2
    /// assert_eq!(coord.rotate(Original), coord);
    /// assert_eq!(coord.rotate(Clockwise).unwrap(), (1, 14));
    /// assert_eq!(coord.rotate(CentralSymmetric).unwrap(), (14, 13));
    /// assert_eq!(coord.rotate(Counterclockwise).unwrap(), (13, 0));
    /// assert_eq!(coord.rotate(FlipHorizontal).unwrap(), (14, 1));
    /// assert_eq!(coord.rotate(FlipLeftDiagonal).unwrap(), (1, 0));
    /// assert_eq!(coord.rotate(FlipVertical).unwrap(), (0, 13));
    /// assert_eq!(coord.rotate(FlipRightDiagonal).unwrap(), (13, 14));
    /// assert_eq!(coord.rotate(FlipRightDiagonal).rotate(FlipRightDiagonal.reverse()), coord);
    /// assert_eq!(coord.rotate(FlipVertical.add(CentralSymmetric)),
    ///     coord.rotate(FlipHorizontal));
    /// ```
    #[inline(always)]
    pub fn rotate(&self, rotation: Rotation) -> Coord<SZ> {
        let (mut x, y) = if let Some(coord) = self.get() {
            coord
        } else {
            return *self;
        };

        let bnd = SZ as u8 - 1u8;
        let (fl, ro) = rotation.fl_ro();

        // flip before rotate
        if fl == 1_u8 {
            x = bnd - x;
        }
        let (x, y) = match ro {
            0b01_u8 => (y, bnd - x),
            0b10_u8 => (bnd - x, bnd - y),
            0b11_u8 => (bnd - y, x),
            _ => (x, y),
        };
        // SAFETY: assumes the algorithm is correct
        unsafe { Coord::<SZ>::build_unchecked(x, y) }
    }

    /// Returns the coordinate translated by `x` and `y` offsets, or `None` if out of range.
    ///
    /// ```
    /// let coord = figrid_board::Coord15::from(1, 2);
    /// assert_eq!(coord.offset(-1i8, 3i8).unwrap(), "a6".parse().unwrap());
    /// assert_eq!(coord.offset(-2i8, 3i8), None);
    /// ```
    #[inline(always)]
    pub fn offset(&self, offset_x: i8, offset_y: i8) -> Option<Coord<SZ>> {
        let (x, y) = if let Some(coord) = self.get() {
            coord
        } else {
            return Some(*self);
        };
        let (x_new, y_new) = (
            (x as i8).checked_add(offset_x)?,
            (y as i8).checked_add(offset_y)?,
        );
        if x_new < 0i8 || y_new < 0i8 {
            return None;
        }
        let (x_new, y_new) = (x_new as u8, y_new as u8);
        if x_new < Self::SIZE && y_new < Self::SIZE {
            Some(Coord { x: x_new, y: y_new })
        } else {
            None
        }
    }
}

use Rotation::*;
impl Rotation {
    /// Returns a new rotation operation equal to doing `self` and then `later`.
    #[inline]
    pub fn add(&self, later: Self) -> Self {
        // r1 + r2 = (fl1 + ro1) + (fl2 + ro2) = fl1 + (ro1 + fl2) + ro2
        // when fl2 = 0: ro1 + fl2 = fl2 + ro1
        // when fl2 = 1: ro1 + fl2 = fl2 + (0b100 - ro1)
        let (fl1, ro1) = self.fl_ro();
        let (fl2, ro2) = later.fl_ro();

        let fl = (fl1 + fl2) & 0b1_u8;
        let ro = if fl2 == 1 {
            0b100_u8 - ro1 + ro2
        } else {
            ro1 + ro2
        };
        let ro = ro & 0b11_u8;
        Self::from_fl_ro(fl, ro)
    }

    /// Returns the reverse operation of this operation.
    #[inline]
    pub fn reverse(&self) -> Self {
        let (fl, mut ro) = self.fl_ro();
        if fl == 0_u8 {
            ro = (0b100_u8 - ro) & 0b11_u8;
        }
        Self::from_fl_ro(fl, ro)
    }

    #[inline(always)]
    fn fl_ro(&self) -> (u8, u8) {
        let fl = match *self {
            FlipHorizontal | FlipLeftDiagonal | FlipVertical | FlipRightDiagonal => 0b1_u8,
            _ => 0b0_u8,
        };
        let ro = match *self {
            Clockwise | FlipLeftDiagonal => 0b01_u8,
            CentralSymmetric | FlipVertical => 0b10_u8,
            Counterclockwise | FlipRightDiagonal => 0b11_u8,
            _ => 0b00_u8,
        };
        (fl, ro)
    }

    #[inline(always)]
    fn from_fl_ro(fl: u8, ro: u8) -> Self {
        match (fl << 2_u8) | ro {
            0b001_u8 => Clockwise,
            0b010_u8 => CentralSymmetric,
            0b011_u8 => Counterclockwise,
            0b100_u8 => FlipHorizontal,
            0b101_u8 => FlipLeftDiagonal,
            0b110_u8 => FlipVertical,
            0b111_u8 => FlipRightDiagonal,
            _ => Original,
        }
    }
}
