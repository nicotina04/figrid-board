#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stone {
    Black,
    White,
}

impl Stone {
    pub fn opponent(self) -> Stone {
        match self {
            Stone::Black => Stone::White,
            Stone::White => Stone::Black,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleSet {
    Freestyle,
    Standard,
    Caro,
    /// Black wins with exactly five and may not play forbidden points (double-four, double-three, overline;
    /// see [`crate::renju`]); white wins with five or more.
    Renju,
    /// Korean omok rule: both sides win only with exactly five and neither may play a double-three
    /// ([`crate::renju::is_forbidden_omok`]); double-fours and overlines are legal. Not a Gomocup rule; the pbrain
    /// protocol uses `INFO rule 17` (16 | exact five) for it.
    Omok,
}

impl RuleSet {
    #[inline]
    pub const fn uses_exact_five(self) -> bool {
        matches!(self, RuleSet::Standard | RuleSet::Renju | RuleSet::Omok)
    }

    /// Win test when only the empty terminals are known. For Caro this treats
    /// a board edge as blocking, which is stricter than Gomocup; prefer
    /// [`RuleSet::line_wins_with_edges`] whenever the edge count is available.
    #[inline]
    pub const fn line_wins(self, side: Stone, count: u32, open_ends: u32) -> bool {
        self.line_wins_with_edges(side, count, open_ends, 0)
    }

    /// Win test with the GomocupJudge terminal semantics: `open_ends` counts
    /// empty in-board terminals and `edge_ends` counts terminals past the
    /// board. Caro (Gomocup `rule 9`) wins only with exactly five that is not
    /// blocked at both ends by opponent stones; a board edge does not block.
    #[inline]
    pub const fn line_wins_with_edges(
        self,
        side: Stone,
        count: u32,
        open_ends: u32,
        edge_ends: u32,
    ) -> bool {
        match self {
            RuleSet::Freestyle => count >= 5,
            RuleSet::Standard | RuleSet::Omok => count == 5,
            RuleSet::Caro => count == 5 && open_ends + edge_ends > 0,
            RuleSet::Renju => match side {
                Stone::Black => count == 5,
                Stone::White => count >= 5,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    BlackWin,
    WhiteWin,
    Draw,
    Ongoing,
}
