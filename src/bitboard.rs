//! Fixed-size bitboard storage shared by both statically compiled engines.
//!
//! The trait uses static dispatch. The 15x15 implementation retains its two-word
//! representation; the 20x20 implementation retains its four-word representation.

/// Operations needed by a board or a tactical bitboard cache.
pub trait BitBoard: Copy + Eq + Send + Sync + 'static {
    const SIZE: usize;
    const EMPTY: Self;
    type Words: AsRef<[u128]> + Copy + Eq;
    type Iter: Iterator<Item = usize>;
    fn get(&self, idx: usize) -> bool;
    fn set(&mut self, idx: usize);
    fn clear(&mut self, idx: usize);
    fn or(&self, other: &Self) -> Self;
    fn and(&self, other: &Self) -> Self;
    fn and_not(&self, other: &Self) -> Self;
    fn intersects(&self, other: &Self) -> bool;
    fn is_zero(&self) -> bool;
    fn count_ones(&self) -> u32;
    fn words(&self) -> Self::Words;
    fn iter_ones(&self) -> Self::Iter;
}

mod small {
    const BITBOARD_WORDS: usize = 2;
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct BitBoard {
        pub lo: u128,
        pub hi: u128,
    }

    impl BitBoard {
        pub const EMPTY: Self = Self { lo: 0, hi: 0 };

        /// Words in ascending bit order (`[lo, hi]`).
        #[inline]
        pub const fn words(&self) -> [u128; BITBOARD_WORDS] {
            [self.lo, self.hi]
        }

        /// Copy with bit `idx` set (usable in const tables).
        pub const fn with(self, idx: usize) -> Self {
            if idx < 128 {
                Self {
                    lo: self.lo | 1u128 << idx,
                    hi: self.hi,
                }
            } else {
                Self {
                    lo: self.lo,
                    hi: self.hi | 1u128 << (idx - 128),
                }
            }
        }

        #[inline]
        pub fn and(&self, other: &BitBoard) -> BitBoard {
            BitBoard {
                lo: self.lo & other.lo,
                hi: self.hi & other.hi,
            }
        }

        #[inline]
        pub fn intersects(&self, other: &BitBoard) -> bool {
            (self.lo & other.lo) != 0 || (self.hi & other.hi) != 0
        }

        #[inline]
        pub fn and_not(&self, other: &BitBoard) -> BitBoard {
            BitBoard {
                lo: self.lo & !other.lo,
                hi: self.hi & !other.hi,
            }
        }

        #[inline]
        pub fn is_zero(&self) -> bool {
            self.lo == 0 && self.hi == 0
        }

        #[inline]
        pub fn get(&self, idx: usize) -> bool {
            if idx < 128 {
                (self.lo >> idx) & 1 != 0
            } else {
                (self.hi >> (idx - 128)) & 1 != 0
            }
        }

        #[inline]
        pub fn set(&mut self, idx: usize) {
            if idx < 128 {
                self.lo |= 1u128 << idx;
            } else {
                self.hi |= 1u128 << (idx - 128);
            }
        }

        #[inline]
        pub fn clear(&mut self, idx: usize) {
            if idx < 128 {
                self.lo &= !(1u128 << idx);
            } else {
                self.hi &= !(1u128 << (idx - 128));
            }
        }

        #[inline]
        pub fn or(&self, other: &BitBoard) -> BitBoard {
            BitBoard {
                lo: self.lo | other.lo,
                hi: self.hi | other.hi,
            }
        }

        #[inline]
        pub fn count_ones(&self) -> u32 {
            self.lo.count_ones() + self.hi.count_ones()
        }

        /// Iterate over the indices of set bits, lowest first.
        /// Enables feature extraction loops to skip empty cells entirely —
        /// critical when the board is sparse (early/midgame), since a
        /// stone-driven pass is ~6× cheaper than scanning all 225 cells.
        #[inline]
        pub fn iter_ones(&self) -> BitBoardIter {
            BitBoardIter {
                lo: self.lo,
                hi: self.hi,
            }
        }
    }

    pub struct BitBoardIter {
        lo: u128,
        hi: u128,
    }

    impl Iterator for BitBoardIter {
        type Item = usize;
        #[inline]
        fn next(&mut self) -> Option<usize> {
            if self.lo != 0 {
                let idx = self.lo.trailing_zeros() as usize;
                self.lo &= self.lo - 1;
                Some(idx)
            } else if self.hi != 0 {
                let idx = 128 + self.hi.trailing_zeros() as usize;
                self.hi &= self.hi - 1;
                Some(idx)
            } else {
                None
            }
        }
    }
}
mod large {
    const BITBOARD_WORDS: usize = 4;
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct BitBoard {
        pub w: [u128; BITBOARD_WORDS],
    }

    impl BitBoard {
        pub const EMPTY: Self = Self {
            w: [0; BITBOARD_WORDS],
        };

        /// Words in ascending bit order.
        #[inline]
        pub const fn words(&self) -> [u128; BITBOARD_WORDS] {
            self.w
        }

        /// Copy with bit `idx` set (usable in const tables).
        pub const fn with(self, idx: usize) -> Self {
            let mut w = self.w;
            w[idx >> 7] |= 1u128 << (idx & 127);
            Self { w }
        }

        #[inline]
        pub fn get(&self, idx: usize) -> bool {
            (self.w[idx >> 7] >> (idx & 127)) & 1 != 0
        }

        #[inline]
        pub fn set(&mut self, idx: usize) {
            self.w[idx >> 7] |= 1u128 << (idx & 127);
        }

        #[inline]
        pub fn clear(&mut self, idx: usize) {
            self.w[idx >> 7] &= !(1u128 << (idx & 127));
        }

        #[inline]
        pub fn or(&self, other: &BitBoard) -> BitBoard {
            BitBoard {
                w: std::array::from_fn(|i| self.w[i] | other.w[i]),
            }
        }

        #[inline]
        pub fn and(&self, other: &BitBoard) -> BitBoard {
            BitBoard {
                w: std::array::from_fn(|i| self.w[i] & other.w[i]),
            }
        }

        #[inline]
        pub fn intersects(&self, other: &BitBoard) -> bool {
            self.w.iter().zip(other.w.iter()).any(|(a, b)| a & b != 0)
        }

        #[inline]
        pub fn and_not(&self, other: &BitBoard) -> BitBoard {
            BitBoard {
                w: std::array::from_fn(|i| self.w[i] & !other.w[i]),
            }
        }

        #[inline]
        pub fn is_zero(&self) -> bool {
            self.w.iter().all(|&x| x == 0)
        }

        #[inline]
        pub fn count_ones(&self) -> u32 {
            self.w.iter().map(|x| x.count_ones()).sum()
        }

        /// Iterate over the indices of set bits, lowest first.
        #[inline]
        pub fn iter_ones(&self) -> BitBoardIter {
            BitBoardIter { w: self.w, i: 0 }
        }
    }

    pub struct BitBoardIter {
        w: [u128; BITBOARD_WORDS],
        i: usize,
    }

    impl Iterator for BitBoardIter {
        type Item = usize;
        #[inline]
        fn next(&mut self) -> Option<usize> {
            while self.i < BITBOARD_WORDS {
                let word = &mut self.w[self.i];
                if *word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    *word &= *word - 1;
                    return Some(self.i * 128 + bit);
                }
                self.i += 1;
            }
            None
        }
    }
}
pub use large::{BitBoard as BitBoard20, BitBoardIter as BitBoardIter20};
pub use small::{BitBoard as BitBoard15, BitBoardIter as BitBoardIter15};

macro_rules! impl_bitboard {
    ($ty:ty, $iter:ty, $size:literal, $words:literal) => {
        impl BitBoard for $ty {
            const SIZE: usize = $size;
            const EMPTY: Self = <$ty>::EMPTY;
            type Words = [u128; $words];
            type Iter = $iter;
            #[inline]
            fn get(&self, idx: usize) -> bool {
                <$ty>::get(self, idx)
            }
            #[inline]
            fn set(&mut self, idx: usize) {
                <$ty>::set(self, idx)
            }
            #[inline]
            fn clear(&mut self, idx: usize) {
                <$ty>::clear(self, idx)
            }
            #[inline]
            fn or(&self, rhs: &Self) -> Self {
                <$ty>::or(self, rhs)
            }
            #[inline]
            fn and(&self, rhs: &Self) -> Self {
                <$ty>::and(self, rhs)
            }
            #[inline]
            fn and_not(&self, rhs: &Self) -> Self {
                <$ty>::and_not(self, rhs)
            }
            #[inline]
            fn intersects(&self, rhs: &Self) -> bool {
                <$ty>::intersects(self, rhs)
            }
            #[inline]
            fn is_zero(&self) -> bool {
                <$ty>::is_zero(self)
            }
            #[inline]
            fn count_ones(&self) -> u32 {
                <$ty>::count_ones(self)
            }
            #[inline]
            fn words(&self) -> Self::Words {
                <$ty>::words(self)
            }
            #[inline]
            fn iter_ones(&self) -> Self::Iter {
                <$ty>::iter_ones(self)
            }
        }
    };
}
impl_bitboard!(BitBoard15, BitBoardIter15, 15, 2);
impl_bitboard!(BitBoard20, BitBoardIter20, 20, 4);
