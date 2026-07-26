//! Exact, bounded DFPN used only by the CB-P1 audit.
//!
//! This module intentionally does not participate in the product search path.
//! Its OR vocabulary is the product-default forcing generator, while every
//! AND node contains every legal defender move.  A solved result is therefore
//! a statement about this finite, registered graph only.

use super::{VctScratch, bb_pair, gather_attack_moves, reach_mask_for_side};
use crate::board::{BOARD_SIZE, Board, Move, NUM_CELLS, RuleSet, Stone};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::Instant;

/// Saturating infinity used by the registered proof-number arithmetic.
pub const DFPN_INF: u64 = 1u64 << 60;

/// Conservative registered memory-accounting ceiling.
pub const REGISTERED_MEMORY_CAP_BYTES: u64 = 64 * 1024 * 1024;

/// Registered CB-P1 horizon in further plies.
pub const REGISTERED_HORIZON: u8 = 14;

const STATE_ACCOUNTING_BYTES: u64 = 144;
const EDGE_ACCOUNTING_BYTES: u64 = 24;
const FINGERPRINT_ACCOUNTING_BYTES: u64 = 32;
const COLLISION_ACCOUNTING_BYTES: u64 = 24;
const REGISTERED_POLICY_DIGEST: u64 = 0xCB01_DF01_0E00_0001;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DfpnStatus {
    ProvenWin,
    ExhaustedBounded,
    UnknownNodeBudget,
    UnknownMemory,
    UnknownAbort,
}

impl DfpnStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ProvenWin => "ProvenWin",
            Self::ExhaustedBounded => "ExhaustedBounded",
            Self::UnknownNodeBudget => "UnknownNodeBudget",
            Self::UnknownMemory => "UnknownMemory",
            Self::UnknownAbort => "UnknownAbort",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundedDfpnConfig {
    pub max_horizon: u8,
    pub memory_cap_bytes: u64,
}

impl BoundedDfpnConfig {
    pub const fn registered() -> Self {
        Self {
            max_horizon: REGISTERED_HORIZON,
            memory_cap_bytes: REGISTERED_MEMORY_CAP_BYTES,
        }
    }

    #[cfg(test)]
    const fn small(max_horizon: u8) -> Self {
        Self {
            max_horizon,
            memory_cap_bytes: REGISTERED_MEMORY_CAP_BYTES,
        }
    }
}

impl Default for BoundedDfpnConfig {
    fn default() -> Self {
        Self::registered()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DfpnWidthBin {
    pub width: u16,
    pub or_count: u64,
    pub and_count: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DfpnCheckpoint {
    pub expansion_cap: u64,
    pub pn: u64,
    pub dn: u64,
    pub status: DfpnStatus,
    pub expansions: u64,
    pub calls: u64,
    pub threshold_returns: u64,
    pub exact_states: u64,
    pub stored_edges: u64,
    pub or_expansions: u64,
    pub and_expansions: u64,
    pub width_histogram: Vec<DfpnWidthBin>,
    pub exact_transposition_hits: u64,
    pub fingerprint_collisions: u64,
    pub distinct_fingerprints: u64,
    pub collision_entries: u64,
    pub exact_alias_errors: u64,
    pub accounted_bytes: u64,
    pub elapsed_nanos: u128,
    pub root_state_digest: String,
    pub scientific_digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DfpnCertificateReplay {
    pub status: DfpnStatus,
    pub visited_nodes: u64,
    pub visited_edges: u64,
    pub certificate_digest: String,
    pub root_restored: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DfpnError {
    UnsupportedRule(RuleSet),
    InvalidRoot(&'static str),
    RootMismatch,
    DecreasingExpansionCap { current: u64, requested: u64 },
    NoSolvedCertificate(DfpnStatus),
    Certificate(String),
    RestorationMismatch,
}

impl fmt::Display for DfpnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedRule(rule) => {
                write!(f, "CB-P1 bounded DFPN requires Freestyle, got {rule:?}")
            }
            Self::InvalidRoot(reason) => write!(f, "invalid bounded-DFPN root: {reason}"),
            Self::RootMismatch => write!(f, "board does not match the registered DFPN root"),
            Self::DecreasingExpansionCap { current, requested } => write!(
                f,
                "expansion cap decreased below consumed work: current={current}, requested={requested}"
            ),
            Self::NoSolvedCertificate(status) => {
                write!(
                    f,
                    "certificate requested for unresolved status {}",
                    status.as_str()
                )
            }
            Self::Certificate(reason) => write!(f, "certificate replay failed: {reason}"),
            Self::RestorationMismatch => write!(f, "DFPN failed to restore the board exactly"),
        }
    }
}

impl std::error::Error for DfpnError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NodeRole {
    Or,
    And,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ExactStateKey {
    black_lo: u128,
    black_hi: u128,
    white_lo: u128,
    white_hi: u128,
    side_to_move: Stone,
    root_attacker: Stone,
    role: NodeRole,
    remaining_horizon: u8,
    rule_set: RuleSet,
    policy_digest: u64,
}

impl ExactStateKey {
    fn from_board(
        board: &Board,
        root_attacker: Stone,
        role: NodeRole,
        remaining_horizon: u8,
        policy_digest: u64,
    ) -> Self {
        Self {
            black_lo: board.black.lo,
            black_hi: board.black.hi,
            white_lo: board.white.lo,
            white_hi: board.white.hi,
            side_to_move: board.side_to_move,
            root_attacker,
            role,
            remaining_horizon,
            rule_set: board.effective_rule_set(),
            policy_digest,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LeafKind {
    AttackerFive,
    DefenderFive,
    FullBoard,
    Horizon,
    NoAttackerMove,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Edge {
    mv: u16,
    child: usize,
}

#[derive(Clone, Debug)]
struct Node {
    key: ExactStateKey,
    pn: u64,
    dn: u64,
    expanded: bool,
    terminal: Option<LeafKind>,
    edges: Vec<Edge>,
}

impl Node {
    fn frontier(key: ExactStateKey) -> Self {
        Self {
            key,
            pn: 1,
            dn: 1,
            expanded: false,
            terminal: None,
            edges: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PermanentHalt {
    Memory,
    Abort,
}

struct SessionStats {
    expansions: u64,
    calls: u64,
    threshold_returns: u64,
    stored_edges: u64,
    or_expansions: u64,
    and_expansions: u64,
    or_widths: [u64; NUM_CELLS + 1],
    and_widths: [u64; NUM_CELLS + 1],
    exact_transposition_hits: u64,
    fingerprint_collisions: u64,
    collision_entries: u64,
    exact_alias_errors: u64,
    elapsed_nanos: u128,
}

impl Default for SessionStats {
    fn default() -> Self {
        Self {
            expansions: 0,
            calls: 0,
            threshold_returns: 0,
            stored_edges: 0,
            or_expansions: 0,
            and_expansions: 0,
            or_widths: [0; NUM_CELLS + 1],
            and_widths: [0; NUM_CELLS + 1],
            exact_transposition_hits: 0,
            fingerprint_collisions: 0,
            collision_entries: 0,
            exact_alias_errors: 0,
            elapsed_nanos: 0,
        }
    }
}

/// A deterministic, cumulative bounded-DFPN session.
///
/// A session is tied to one exact root. `advance_to` may be called repeatedly
/// with increasing expansion caps; the node table and proof numbers are kept.
pub struct BoundedDfpnSession {
    config: BoundedDfpnConfig,
    root_snapshot: Board,
    root_attacker: Stone,
    policy_digest: u64,
    root_id: usize,
    nodes: Vec<Node>,
    fingerprints: HashMap<u64, Vec<usize>>,
    fingerprint_mask: u64,
    stats: SessionStats,
    halt: Option<PermanentHalt>,
    root_state_digest: String,
}

impl BoundedDfpnSession {
    pub fn new(board: &Board, config: BoundedDfpnConfig) -> Result<Self, DfpnError> {
        let rule = board.effective_rule_set();
        if rule != RuleSet::Freestyle {
            return Err(DfpnError::UnsupportedRule(rule));
        }
        if board.move_count != board.history.len() {
            return Err(DfpnError::InvalidRoot("move_count/history mismatch"));
        }
        if board.black.count_ones() as usize + board.white.count_ones() as usize != board.move_count
        {
            return Err(DfpnError::InvalidRoot("bitboard/move_count mismatch"));
        }
        if board.black.lo & board.white.lo != 0 || board.black.hi & board.white.hi != 0 {
            return Err(DfpnError::InvalidRoot("overlapping bitboards"));
        }
        let terminal = scan_terminal(board, board.side_to_move)?;
        if terminal.is_some() {
            return Err(DfpnError::InvalidRoot("root is already terminal"));
        }

        let policy_digest = policy_digest(config.max_horizon);
        let root_attacker = board.side_to_move;
        let root_key = ExactStateKey::from_board(
            board,
            root_attacker,
            NodeRole::Or,
            config.max_horizon,
            policy_digest,
        );
        let root_state_digest = digest_key(&root_key);
        let root_fp = fingerprint(&root_key) & u64::MAX;
        let mut fingerprints = HashMap::new();
        fingerprints.insert(root_fp, vec![0]);
        let mut session = Self {
            config,
            root_snapshot: board.clone(),
            root_attacker,
            policy_digest,
            root_id: 0,
            nodes: vec![Node::frontier(root_key)],
            fingerprints,
            fingerprint_mask: u64::MAX,
            stats: SessionStats::default(),
            halt: None,
            root_state_digest,
        };
        if session.accounted_bytes() > config.memory_cap_bytes {
            session.halt = Some(PermanentHalt::Memory);
        }
        Ok(session)
    }

    pub fn policy_digest(&self) -> u64 {
        self.policy_digest
    }

    pub fn policy_digest_hex(&self) -> String {
        format!("{:016X}", self.policy_digest)
    }

    pub fn root_state_digest(&self) -> &str {
        &self.root_state_digest
    }

    pub fn advance_to(
        &mut self,
        board: &mut Board,
        expansion_cap: u64,
    ) -> Result<DfpnCheckpoint, DfpnError> {
        if !board_exact_eq(board, &self.root_snapshot) {
            return Err(DfpnError::RootMismatch);
        }
        if expansion_cap < self.stats.expansions {
            return Err(DfpnError::DecreasingExpansionCap {
                current: self.stats.expansions,
                requested: expansion_cap,
            });
        }

        let status_before = self.status();
        let may_advance = matches!(status_before, DfpnStatus::UnknownNodeBudget)
            && self.stats.expansions < expansion_cap;
        if may_advance {
            let start = Instant::now();
            let call_limit = expansion_cap.saturating_mul(64).saturating_add(1_000_000);
            self.dfpn(
                self.root_id,
                DFPN_INF,
                DFPN_INF,
                board,
                expansion_cap,
                call_limit,
            )?;
            self.stats.elapsed_nanos = self
                .stats
                .elapsed_nanos
                .saturating_add(start.elapsed().as_nanos());
        }

        if !board_exact_eq(board, &self.root_snapshot) {
            return Err(DfpnError::RestorationMismatch);
        }
        Ok(self.checkpoint(expansion_cap))
    }

    pub fn checkpoint(&self, expansion_cap: u64) -> DfpnCheckpoint {
        let root = &self.nodes[self.root_id];
        let mut width_histogram = Vec::new();
        for width in 0..=NUM_CELLS {
            let or_count = self.stats.or_widths[width];
            let and_count = self.stats.and_widths[width];
            if or_count != 0 || and_count != 0 {
                width_histogram.push(DfpnWidthBin {
                    width: width as u16,
                    or_count,
                    and_count,
                });
            }
        }
        let mut checkpoint = DfpnCheckpoint {
            expansion_cap,
            pn: root.pn,
            dn: root.dn,
            status: self.status(),
            expansions: self.stats.expansions,
            calls: self.stats.calls,
            threshold_returns: self.stats.threshold_returns,
            exact_states: self.nodes.len() as u64,
            stored_edges: self.stats.stored_edges,
            or_expansions: self.stats.or_expansions,
            and_expansions: self.stats.and_expansions,
            width_histogram,
            exact_transposition_hits: self.stats.exact_transposition_hits,
            fingerprint_collisions: self.stats.fingerprint_collisions,
            distinct_fingerprints: self.fingerprints.len() as u64,
            collision_entries: self.stats.collision_entries,
            exact_alias_errors: self.stats.exact_alias_errors,
            accounted_bytes: self.accounted_bytes(),
            elapsed_nanos: self.stats.elapsed_nanos,
            root_state_digest: self.root_state_digest.clone(),
            scientific_digest: String::new(),
        };
        checkpoint.scientific_digest = digest_checkpoint(&checkpoint);
        checkpoint
    }

    pub fn status(&self) -> DfpnStatus {
        let root = &self.nodes[self.root_id];
        if root.pn == 0 {
            DfpnStatus::ProvenWin
        } else if root.dn == 0 {
            DfpnStatus::ExhaustedBounded
        } else {
            match self.halt {
                Some(PermanentHalt::Memory) => DfpnStatus::UnknownMemory,
                Some(PermanentHalt::Abort) => DfpnStatus::UnknownAbort,
                None => DfpnStatus::UnknownNodeBudget,
            }
        }
    }

    pub fn verify_terminal_certificate(
        &self,
        board: &mut Board,
    ) -> Result<DfpnCertificateReplay, DfpnError> {
        let status = self.status();
        let truth = match status {
            DfpnStatus::ProvenWin => true,
            DfpnStatus::ExhaustedBounded => false,
            other => return Err(DfpnError::NoSolvedCertificate(other)),
        };
        if !board_exact_eq(board, &self.root_snapshot) {
            return Err(DfpnError::RootMismatch);
        }

        let mut replay = ReplayState::default();
        let result = self.replay_node(self.root_id, truth, board, &mut replay);
        let restored = board_exact_eq(board, &self.root_snapshot);
        if !restored {
            return Err(DfpnError::RestorationMismatch);
        }
        result?;
        Ok(DfpnCertificateReplay {
            status,
            visited_nodes: replay.visited_nodes,
            visited_edges: replay.visited_edges,
            certificate_digest: replay.digest.finish_hex(),
            root_restored: restored,
        })
    }

    fn accounted_bytes(&self) -> u64 {
        accounted_bytes(
            self.nodes.len() as u64,
            self.stats.stored_edges,
            self.fingerprints.len() as u64,
            self.stats.collision_entries,
        )
    }

    fn dfpn(
        &mut self,
        node_id: usize,
        proof_threshold: u64,
        disproof_threshold: u64,
        board: &mut Board,
        expansion_cap: u64,
        call_limit: u64,
    ) -> Result<(), DfpnError> {
        if self.halt.is_some() || self.stats.expansions >= expansion_cap {
            return Ok(());
        }
        if self.stats.calls >= call_limit {
            self.halt = Some(PermanentHalt::Abort);
            return Ok(());
        }
        self.stats.calls += 1;
        self.assert_board_matches_node(board, node_id)?;

        if !self.nodes[node_id].expanded {
            self.expand_node(node_id, board, expansion_cap)?;
            if self.halt.is_some() || self.stats.expansions >= expansion_cap {
                return Ok(());
            }
        }
        self.recompute_node(node_id)?;

        loop {
            let (pn, dn) = {
                let node = &self.nodes[node_id];
                (node.pn, node.dn)
            };
            if pn >= proof_threshold || dn >= disproof_threshold {
                self.stats.threshold_returns = self.stats.threshold_returns.saturating_add(1);
                break;
            }
            if pn == 0 || dn == 0 || self.halt.is_some() || self.stats.expansions >= expansion_cap {
                break;
            }
            if self.stats.calls >= call_limit {
                self.halt = Some(PermanentHalt::Abort);
                break;
            }

            let (edge, child_pt, child_dt) =
                self.most_proving_child(node_id, proof_threshold, disproof_threshold)?;
            board.make_move(edge.mv as usize);
            let child_result = self.dfpn(
                edge.child,
                child_pt,
                child_dt,
                board,
                expansion_cap,
                call_limit,
            );
            board.undo_move();
            child_result?;
            self.assert_board_matches_node(board, node_id)?;
            self.recompute_node(node_id)?;
        }
        Ok(())
    }

    fn expand_node(
        &mut self,
        node_id: usize,
        board: &mut Board,
        expansion_cap: u64,
    ) -> Result<(), DfpnError> {
        if self.stats.expansions >= expansion_cap || self.halt.is_some() {
            return Ok(());
        }
        self.assert_board_matches_node(board, node_id)?;
        let key = self.nodes[node_id].key.clone();

        if let Some(leaf) = scan_terminal(board, self.root_attacker)? {
            self.commit_leaf(node_id, leaf);
            return Ok(());
        }
        if key.remaining_horizon == 0 {
            self.commit_leaf(node_id, LeafKind::Horizon);
            return Ok(());
        }

        let moves = match key.role {
            NodeRole::Or => registered_attack_moves(board),
            NodeRole::And => board.legal_moves(),
        };
        if moves.is_empty() {
            let leaf = match key.role {
                NodeRole::Or => LeafKind::NoAttackerMove,
                NodeRole::And => LeafKind::FullBoard,
            };
            self.commit_leaf(node_id, leaf);
            return Ok(());
        }

        let child_role = match key.role {
            NodeRole::Or => NodeRole::And,
            NodeRole::And => NodeRole::Or,
        };
        let mut planned = Vec::with_capacity(moves.len());
        for mv in moves {
            if !board.is_legal_move(mv) {
                return Err(DfpnError::InvalidRoot("generator emitted illegal move"));
            }
            board.make_move(mv);
            let child_key = ExactStateKey::from_board(
                board,
                self.root_attacker,
                child_role,
                key.remaining_horizon - 1,
                self.policy_digest,
            );
            board.undo_move();
            planned.push(PlannedChild {
                mv,
                fingerprint: fingerprint(&child_key) & self.fingerprint_mask,
                key: child_key,
                resolved: None,
            });
        }
        self.assert_board_matches_node(board, node_id)?;

        let plan = self.plan_ledger(&planned);
        let future_bytes = accounted_bytes(
            self.nodes.len() as u64 + plan.new_states,
            self.stats.stored_edges + planned.len() as u64,
            self.fingerprints.len() as u64 + plan.new_fingerprints,
            self.stats.collision_entries + plan.new_collision_entries,
        );
        if future_bytes > self.config.memory_cap_bytes {
            self.halt = Some(PermanentHalt::Memory);
            return Ok(());
        }

        let mut newly_inserted: Vec<(ExactStateKey, u64, usize)> = Vec::new();
        let mut edges = Vec::with_capacity(planned.len());
        let mut transposition_hits = 0u64;
        let mut collision_additions = 0u64;
        for child in &mut planned {
            let existing = self.lookup_exact(child.fingerprint, &child.key);
            let child_id = if let Some(id) = existing {
                transposition_hits += 1;
                id
            } else if let Some((_, _, id)) = newly_inserted
                .iter()
                .find(|(key, fp, _)| *fp == child.fingerprint && *key == child.key)
            {
                transposition_hits += 1;
                *id
            } else {
                let bucket_was_nonempty = self
                    .fingerprints
                    .get(&child.fingerprint)
                    .is_some_and(|bucket| !bucket.is_empty())
                    || newly_inserted
                        .iter()
                        .any(|(_, fp, _)| *fp == child.fingerprint);
                if bucket_was_nonempty {
                    collision_additions += 1;
                }
                let id = self.nodes.len();
                self.nodes.push(Node::frontier(child.key.clone()));
                self.fingerprints
                    .entry(child.fingerprint)
                    .or_default()
                    .push(id);
                newly_inserted.push((child.key.clone(), child.fingerprint, id));
                id
            };
            child.resolved = Some(child_id);
            edges.push(Edge {
                mv: child.mv as u16,
                child: child_id,
            });
        }

        debug_assert_eq!(newly_inserted.len() as u64, plan.new_states);
        debug_assert_eq!(collision_additions, plan.new_collision_entries);
        self.stats.exact_transposition_hits = self
            .stats
            .exact_transposition_hits
            .saturating_add(transposition_hits);
        self.stats.fingerprint_collisions = self
            .stats
            .fingerprint_collisions
            .saturating_add(collision_additions);
        self.stats.collision_entries = self
            .stats
            .collision_entries
            .saturating_add(collision_additions);
        self.stats.stored_edges = self.stats.stored_edges.saturating_add(edges.len() as u64);

        let width = edges.len();
        let node = &mut self.nodes[node_id];
        node.expanded = true;
        node.terminal = None;
        node.edges = edges;
        self.record_expansion(key.role, width);
        self.recompute_node(node_id)?;
        Ok(())
    }

    fn plan_ledger(&self, children: &[PlannedChild]) -> LedgerPlan {
        let mut new_exact: Vec<(&ExactStateKey, u64)> = Vec::new();
        let mut new_fps = HashSet::new();
        let mut new_collision_entries = 0u64;
        for child in children {
            if self.lookup_exact(child.fingerprint, &child.key).is_some()
                || new_exact
                    .iter()
                    .any(|(key, fp)| *fp == child.fingerprint && **key == child.key)
            {
                continue;
            }
            let existing_width = self
                .fingerprints
                .get(&child.fingerprint)
                .map_or(0, Vec::len);
            let pending_width = new_exact
                .iter()
                .filter(|(_, fp)| *fp == child.fingerprint)
                .count();
            if existing_width + pending_width != 0 {
                new_collision_entries += 1;
            }
            if existing_width == 0 && pending_width == 0 {
                new_fps.insert(child.fingerprint);
            }
            new_exact.push((&child.key, child.fingerprint));
        }
        LedgerPlan {
            new_states: new_exact.len() as u64,
            new_fingerprints: new_fps.len() as u64,
            new_collision_entries,
        }
    }

    fn lookup_exact(&self, fp: u64, key: &ExactStateKey) -> Option<usize> {
        self.fingerprints.get(&fp).and_then(|bucket| {
            bucket
                .iter()
                .copied()
                .find(|&node_id| self.nodes[node_id].key == *key)
        })
    }

    fn commit_leaf(&mut self, node_id: usize, leaf: LeafKind) {
        let role = self.nodes[node_id].key.role;
        let proved = leaf == LeafKind::AttackerFive;
        let node = &mut self.nodes[node_id];
        node.expanded = true;
        node.terminal = Some(leaf);
        node.edges.clear();
        if proved {
            node.pn = 0;
            node.dn = DFPN_INF;
        } else {
            node.pn = DFPN_INF;
            node.dn = 0;
        }
        self.record_expansion(role, 0);
    }

    fn record_expansion(&mut self, role: NodeRole, width: usize) {
        self.stats.expansions = self.stats.expansions.saturating_add(1);
        match role {
            NodeRole::Or => {
                self.stats.or_expansions = self.stats.or_expansions.saturating_add(1);
                self.stats.or_widths[width] = self.stats.or_widths[width].saturating_add(1);
            }
            NodeRole::And => {
                self.stats.and_expansions = self.stats.and_expansions.saturating_add(1);
                self.stats.and_widths[width] = self.stats.and_widths[width].saturating_add(1);
            }
        }
    }

    fn recompute_node(&mut self, node_id: usize) -> Result<(), DfpnError> {
        if !self.nodes[node_id].expanded || self.nodes[node_id].terminal.is_some() {
            return Ok(());
        }
        if self.nodes[node_id].edges.is_empty() {
            return Err(DfpnError::Certificate(
                "expanded nonterminal has no child ledger".to_owned(),
            ));
        }
        let role = self.nodes[node_id].key.role;
        let (pn, dn) = aggregate(
            role,
            self.nodes[node_id]
                .edges
                .iter()
                .map(|edge| (self.nodes[edge.child].pn, self.nodes[edge.child].dn)),
        );
        self.nodes[node_id].pn = pn;
        self.nodes[node_id].dn = dn;
        Ok(())
    }

    fn most_proving_child(
        &self,
        node_id: usize,
        proof_threshold: u64,
        disproof_threshold: u64,
    ) -> Result<(Edge, u64, u64), DfpnError> {
        let node = &self.nodes[node_id];
        if node.edges.is_empty() {
            return Err(DfpnError::Certificate(
                "most-proving child requested from empty ledger".to_owned(),
            ));
        }
        let selected_index = match node.key.role {
            NodeRole::Or => node
                .edges
                .iter()
                .enumerate()
                .min_by_key(|(_, edge)| self.nodes[edge.child].pn)
                .map(|(i, _)| i),
            NodeRole::And => node
                .edges
                .iter()
                .enumerate()
                .min_by_key(|(_, edge)| self.nodes[edge.child].dn)
                .map(|(i, _)| i),
        }
        .expect("nonempty ledger");
        let edge = node.edges[selected_index];
        let child = &self.nodes[edge.child];

        let (child_pt, child_dt) = match node.key.role {
            NodeRole::Or => {
                let second = node
                    .edges
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != selected_index)
                    .map(|(_, edge)| self.nodes[edge.child].pn)
                    .min()
                    .unwrap_or(DFPN_INF);
                (
                    proof_threshold.min(saturating_plus_one(second)),
                    threshold_minus_total_plus_selected(disproof_threshold, node.dn, child.dn)?,
                )
            }
            NodeRole::And => {
                let second = node
                    .edges
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != selected_index)
                    .map(|(_, edge)| self.nodes[edge.child].dn)
                    .min()
                    .unwrap_or(DFPN_INF);
                (
                    threshold_minus_total_plus_selected(proof_threshold, node.pn, child.pn)?,
                    disproof_threshold.min(saturating_plus_one(second)),
                )
            }
        };
        Ok((edge, child_pt, child_dt))
    }

    fn assert_board_matches_node(&self, board: &Board, node_id: usize) -> Result<(), DfpnError> {
        let node = &self.nodes[node_id];
        let observed = ExactStateKey::from_board(
            board,
            self.root_attacker,
            node.key.role,
            node.key.remaining_horizon,
            self.policy_digest,
        );
        if observed == node.key {
            Ok(())
        } else {
            Err(DfpnError::RootMismatch)
        }
    }

    fn replay_node(
        &self,
        node_id: usize,
        expect_proved: bool,
        board: &mut Board,
        replay: &mut ReplayState,
    ) -> Result<(), DfpnError> {
        self.assert_board_matches_node(board, node_id)?;
        if !replay.seen.insert((node_id, expect_proved)) {
            return Ok(());
        }
        let node = &self.nodes[node_id];
        if !node.expanded {
            return Err(DfpnError::Certificate(
                "certificate reached an unexpanded frontier".to_owned(),
            ));
        }
        if expect_proved && node.pn != 0 {
            return Err(DfpnError::Certificate(
                "proof certificate reached a node without pn=0".to_owned(),
            ));
        }
        if !expect_proved && node.dn != 0 {
            return Err(DfpnError::Certificate(
                "disproof certificate reached a node without dn=0".to_owned(),
            ));
        }
        replay.visited_nodes += 1;
        replay.digest.key(&node.key);
        replay.digest.u64(expect_proved as u64);

        let independent_terminal = replay_scan_terminal(board, self.root_attacker)?;
        let independent_terminal = independent_terminal
            .or_else(|| (node.key.remaining_horizon == 0).then_some(LeafKind::Horizon));
        if let Some(leaf) = independent_terminal {
            if node.terminal != Some(leaf) || !node.edges.is_empty() {
                return Err(DfpnError::Certificate(
                    "stored terminal does not match independent full-board scan".to_owned(),
                ));
            }
            if expect_proved != (leaf == LeafKind::AttackerFive) {
                return Err(DfpnError::Certificate(
                    "terminal truth does not match requested certificate".to_owned(),
                ));
            }
            replay.digest.u64(leaf as u64);
            return Ok(());
        }

        let regenerated = match node.key.role {
            NodeRole::Or => registered_attack_moves(board),
            NodeRole::And => board.legal_moves(),
        };
        if regenerated.is_empty() {
            let expected_leaf = match node.key.role {
                NodeRole::Or => LeafKind::NoAttackerMove,
                NodeRole::And => LeafKind::FullBoard,
            };
            if node.terminal != Some(expected_leaf) || expect_proved || !node.edges.is_empty() {
                return Err(DfpnError::Certificate(
                    "empty regenerated ledger does not match negative terminal".to_owned(),
                ));
            }
            replay.digest.u64(expected_leaf as u64);
            return Ok(());
        }
        if node.terminal.is_some() {
            return Err(DfpnError::Certificate(
                "nonterminal node carries a terminal marker".to_owned(),
            ));
        }
        if regenerated.len() != node.edges.len() {
            return Err(DfpnError::Certificate(format!(
                "ledger width mismatch: regenerated={}, stored={}",
                regenerated.len(),
                node.edges.len()
            )));
        }

        let child_role = match node.key.role {
            NodeRole::Or => NodeRole::And,
            NodeRole::And => NodeRole::Or,
        };
        for (&mv, edge) in regenerated.iter().zip(&node.edges) {
            if edge.mv as usize != mv {
                return Err(DfpnError::Certificate(
                    "registered child order mismatch".to_owned(),
                ));
            }
            board.make_move(mv);
            let regenerated_key = ExactStateKey::from_board(
                board,
                self.root_attacker,
                child_role,
                node.key.remaining_horizon - 1,
                self.policy_digest,
            );
            let key_matches = self.nodes[edge.child].key == regenerated_key;
            board.undo_move();
            if !key_matches {
                return Err(DfpnError::Certificate(
                    "stored child exact key mismatch".to_owned(),
                ));
            }
            replay.visited_edges += 1;
            replay.digest.u64(mv as u64);
            replay.digest.key(&regenerated_key);
        }
        self.assert_board_matches_node(board, node_id)?;

        let (recomputed_pn, recomputed_dn) = aggregate(
            node.key.role,
            node.edges
                .iter()
                .map(|edge| (self.nodes[edge.child].pn, self.nodes[edge.child].dn)),
        );
        if (node.pn, node.dn) != (recomputed_pn, recomputed_dn) {
            return Err(DfpnError::Certificate(
                "stored proof numbers do not match child aggregation".to_owned(),
            ));
        }

        let selected: Vec<usize> = match (node.key.role, expect_proved) {
            (NodeRole::Or, true) => vec![
                node.edges
                    .iter()
                    .position(|edge| self.nodes[edge.child].pn == 0)
                    .ok_or_else(|| {
                        DfpnError::Certificate("OR proof lacks a proved child".to_owned())
                    })?,
            ],
            (NodeRole::And, true) => {
                if node.edges.iter().any(|edge| self.nodes[edge.child].pn != 0) {
                    return Err(DfpnError::Certificate(
                        "AND proof does not cover every legal defense".to_owned(),
                    ));
                }
                (0..node.edges.len()).collect()
            }
            (NodeRole::Or, false) => {
                if node.edges.iter().any(|edge| self.nodes[edge.child].dn != 0) {
                    return Err(DfpnError::Certificate(
                        "OR disproof does not cover every attacker child".to_owned(),
                    ));
                }
                (0..node.edges.len()).collect()
            }
            (NodeRole::And, false) => vec![
                node.edges
                    .iter()
                    .position(|edge| self.nodes[edge.child].dn == 0)
                    .ok_or_else(|| {
                        DfpnError::Certificate("AND disproof lacks a disproved child".to_owned())
                    })?,
            ],
        };

        for index in selected {
            let edge = node.edges[index];
            board.make_move(edge.mv as usize);
            let result = self.replay_node(edge.child, expect_proved, board, replay);
            board.undo_move();
            result?;
        }
        self.assert_board_matches_node(board, node_id)
    }

    #[cfg(test)]
    fn force_fingerprint_mask(&mut self, mask: u64) {
        self.fingerprint_mask = mask;
        self.fingerprints.clear();
        self.stats.collision_entries = 0;
        self.stats.fingerprint_collisions = 0;
        for node_id in 0..self.nodes.len() {
            let fp = fingerprint(&self.nodes[node_id].key) & mask;
            let bucket = self.fingerprints.entry(fp).or_default();
            if !bucket.is_empty() {
                self.stats.collision_entries += 1;
                self.stats.fingerprint_collisions += 1;
            }
            bucket.push(node_id);
        }
    }
}

struct PlannedChild {
    mv: Move,
    fingerprint: u64,
    key: ExactStateKey,
    resolved: Option<usize>,
}

struct LedgerPlan {
    new_states: u64,
    new_fingerprints: u64,
    new_collision_entries: u64,
}

#[derive(Default)]
struct ReplayState {
    seen: HashSet<(usize, bool)>,
    visited_nodes: u64,
    visited_edges: u64,
    digest: StableDigest,
}

fn registered_attack_moves(board: &Board) -> Vec<Move> {
    let side = board.side_to_move;
    let (my, opp) = bb_pair(board, side);
    let reach = reach_mask_for_side(board, side);
    let mut scratch = VctScratch::default();
    gather_attack_moves(
        board,
        my,
        opp,
        side,
        board.effective_rule_set(),
        false,
        false,
        true,
        Some(&reach),
        None,
        &mut scratch,
        false,
    )
    .into_iter()
    .map(|(mv, _)| mv)
    .collect()
}

fn scan_terminal(board: &Board, attacker: Stone) -> Result<Option<LeafKind>, DfpnError> {
    let black_win = full_board_has_five(board, Stone::Black);
    let white_win = full_board_has_five(board, Stone::White);
    if black_win && white_win {
        return Err(DfpnError::InvalidRoot(
            "both colors have a winning line in one exact state",
        ));
    }
    if black_win || white_win {
        let winner = if black_win {
            Stone::Black
        } else {
            Stone::White
        };
        return Ok(Some(if winner == attacker {
            LeafKind::AttackerFive
        } else {
            LeafKind::DefenderFive
        }));
    }
    if board.black.count_ones() + board.white.count_ones() == NUM_CELLS as u32 {
        return Ok(Some(LeafKind::FullBoard));
    }
    Ok(None)
}

/// Deliberately separate, cell-anchored terminal scan used by certificate
/// replay.  Unlike the solver's maximal-run scanner, this checks every
/// occupied anchor and counts in both directions.
fn replay_scan_terminal(board: &Board, attacker: Stone) -> Result<Option<LeafKind>, DfpnError> {
    fn wins(board: &Board, side: Stone) -> bool {
        let stones = match side {
            Stone::Black => &board.black,
            Stone::White => &board.white,
        };
        let occupied = board.black.or(&board.white);
        const DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
        for anchor in stones.iter_ones() {
            let row = (anchor / BOARD_SIZE) as i32;
            let col = (anchor % BOARD_SIZE) as i32;
            for &(dr, dc) in &DIRS {
                let mut count = 1u32;
                let mut r = row + dr;
                let mut c = col + dc;
                while in_board(r, c) && stones.get(r as usize * BOARD_SIZE + c as usize) {
                    count += 1;
                    r += dr;
                    c += dc;
                }
                let end_r = r;
                let end_c = c;
                r = row - dr;
                c = col - dc;
                while in_board(r, c) && stones.get(r as usize * BOARD_SIZE + c as usize) {
                    count += 1;
                    r -= dr;
                    c -= dc;
                }
                let mut open_ends = 0u32;
                if in_board(r, c) && !occupied.get(r as usize * BOARD_SIZE + c as usize) {
                    open_ends += 1;
                }
                if in_board(end_r, end_c)
                    && !occupied.get(end_r as usize * BOARD_SIZE + end_c as usize)
                {
                    open_ends += 1;
                }
                if board.effective_rule_set().line_wins(side, count, open_ends) {
                    return true;
                }
            }
        }
        false
    }

    let black = wins(board, Stone::Black);
    let white = wins(board, Stone::White);
    if black && white {
        return Err(DfpnError::InvalidRoot(
            "both colors have a winning line in certificate replay",
        ));
    }
    if black || white {
        let winner = if black { Stone::Black } else { Stone::White };
        return Ok(Some(if winner == attacker {
            LeafKind::AttackerFive
        } else {
            LeafKind::DefenderFive
        }));
    }
    if board.black.count_ones() + board.white.count_ones() == NUM_CELLS as u32 {
        return Ok(Some(LeafKind::FullBoard));
    }
    Ok(None)
}

fn full_board_has_five(board: &Board, side: Stone) -> bool {
    let stones = match side {
        Stone::Black => &board.black,
        Stone::White => &board.white,
    };
    let occupied = board.black.or(&board.white);
    let rule = board.effective_rule_set();
    const DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
    for cell in stones.iter_ones() {
        let row = (cell / BOARD_SIZE) as i32;
        let col = (cell % BOARD_SIZE) as i32;
        for &(dr, dc) in &DIRS {
            let prev_r = row - dr;
            let prev_c = col - dc;
            if in_board(prev_r, prev_c)
                && stones.get(prev_r as usize * BOARD_SIZE + prev_c as usize)
            {
                continue;
            }
            let mut count = 0u32;
            let mut r = row;
            let mut c = col;
            while in_board(r, c) && stones.get(r as usize * BOARD_SIZE + c as usize) {
                count += 1;
                r += dr;
                c += dc;
            }
            let mut open_ends = 0u32;
            if in_board(prev_r, prev_c)
                && !occupied.get(prev_r as usize * BOARD_SIZE + prev_c as usize)
            {
                open_ends += 1;
            }
            if in_board(r, c) && !occupied.get(r as usize * BOARD_SIZE + c as usize) {
                open_ends += 1;
            }
            if rule.line_wins(side, count, open_ends) {
                return true;
            }
        }
    }
    false
}

#[inline]
fn in_board(row: i32, col: i32) -> bool {
    row >= 0 && row < BOARD_SIZE as i32 && col >= 0 && col < BOARD_SIZE as i32
}

fn policy_digest(horizon: u8) -> u64 {
    // The base value binds: fast classifier ON, reach mask ON, all registered
    // experimental vocabularies/shortcuts/index/scratch OFF, all-legal AND,
    // full-board terminals, exact equality, and the threshold formulas.
    mix64(REGISTERED_POLICY_DIGEST ^ horizon as u64)
}

fn fingerprint(key: &ExactStateKey) -> u64 {
    let mut h = 0x6A09_E667_F3BC_C909;
    for word in [
        key.black_lo as u64,
        (key.black_lo >> 64) as u64,
        key.black_hi as u64,
        (key.black_hi >> 64) as u64,
        key.white_lo as u64,
        (key.white_lo >> 64) as u64,
        key.white_hi as u64,
        (key.white_hi >> 64) as u64,
        stone_tag(key.side_to_move),
        stone_tag(key.root_attacker),
        role_tag(key.role),
        key.remaining_horizon as u64,
        rule_tag(key.rule_set),
        key.policy_digest,
    ] {
        h = mix64(h ^ word);
    }
    h
}

fn aggregate(role: NodeRole, children: impl Iterator<Item = (u64, u64)>) -> (u64, u64) {
    match role {
        NodeRole::Or => children.fold((DFPN_INF, 0), |(pn, dn), (cpn, cdn)| {
            (pn.min(cpn), saturating_add(dn, cdn))
        }),
        NodeRole::And => children.fold((0, DFPN_INF), |(pn, dn), (cpn, cdn)| {
            (saturating_add(pn, cpn), dn.min(cdn))
        }),
    }
}

#[inline]
fn saturating_add(lhs: u64, rhs: u64) -> u64 {
    lhs.saturating_add(rhs).min(DFPN_INF)
}

#[inline]
fn saturating_plus_one(value: u64) -> u64 {
    if value >= DFPN_INF {
        DFPN_INF
    } else {
        value + 1
    }
}

fn threshold_minus_total_plus_selected(
    threshold: u64,
    total: u64,
    selected: u64,
) -> Result<u64, DfpnError> {
    if threshold >= DFPN_INF {
        return Ok(DFPN_INF);
    }
    if total >= threshold {
        return Err(DfpnError::Certificate(
            "finite DFPN threshold invariant violated".to_owned(),
        ));
    }
    let value = threshold as u128 + selected as u128;
    let value = value.saturating_sub(total as u128);
    Ok(value.clamp(1, DFPN_INF as u128) as u64)
}

fn accounted_bytes(states: u64, edges: u64, fingerprints: u64, collisions: u64) -> u64 {
    STATE_ACCOUNTING_BYTES
        .saturating_mul(states)
        .saturating_add(EDGE_ACCOUNTING_BYTES.saturating_mul(edges))
        .saturating_add(FINGERPRINT_ACCOUNTING_BYTES.saturating_mul(fingerprints))
        .saturating_add(COLLISION_ACCOUNTING_BYTES.saturating_mul(collisions))
}

fn board_exact_eq(lhs: &Board, rhs: &Board) -> bool {
    lhs.black == rhs.black
        && lhs.white == rhs.white
        && lhs.side_to_move == rhs.side_to_move
        && lhs.move_count == rhs.move_count
        && lhs.last_move == rhs.last_move
        && lhs.history == rhs.history
        && lhs.zobrist == rhs.zobrist
        && lhs.line_pattern_ids == rhs.line_pattern_ids
        && lhs.rule_set == rhs.rule_set
        && lhs.exact5 == rhs.exact5
}

fn digest_key(key: &ExactStateKey) -> String {
    let mut digest = StableDigest::default();
    digest.key(key);
    digest.finish_hex()
}

fn digest_checkpoint(checkpoint: &DfpnCheckpoint) -> String {
    let mut digest = StableDigest::default();
    digest.bytes(checkpoint.root_state_digest.as_bytes());
    digest.u64(checkpoint.expansion_cap);
    digest.u64(checkpoint.pn);
    digest.u64(checkpoint.dn);
    digest.u64(status_tag(checkpoint.status));
    digest.u64(checkpoint.expansions);
    digest.u64(checkpoint.calls);
    digest.u64(checkpoint.threshold_returns);
    digest.u64(checkpoint.exact_states);
    digest.u64(checkpoint.stored_edges);
    digest.u64(checkpoint.or_expansions);
    digest.u64(checkpoint.and_expansions);
    for bin in &checkpoint.width_histogram {
        digest.u64(bin.width as u64);
        digest.u64(bin.or_count);
        digest.u64(bin.and_count);
    }
    digest.u64(checkpoint.exact_transposition_hits);
    digest.u64(checkpoint.fingerprint_collisions);
    digest.u64(checkpoint.distinct_fingerprints);
    digest.u64(checkpoint.collision_entries);
    digest.u64(checkpoint.exact_alias_errors);
    digest.u64(checkpoint.accounted_bytes);
    digest.finish_hex()
}

#[derive(Clone)]
struct StableDigest {
    lanes: [u64; 4],
}

impl Default for StableDigest {
    fn default() -> Self {
        Self {
            lanes: [
                0x243F_6A88_85A3_08D3,
                0x1319_8A2E_0370_7344,
                0xA409_3822_299F_31D0,
                0x082E_FA98_EC4E_6C89,
            ],
        }
    }
}

impl StableDigest {
    fn u64(&mut self, value: u64) {
        for (index, lane) in self.lanes.iter_mut().enumerate() {
            *lane = mix64(*lane ^ value.rotate_left((index * 13) as u32) ^ index as u64);
        }
    }

    fn bytes(&mut self, bytes: &[u8]) {
        self.u64(bytes.len() as u64);
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.u64(u64::from_le_bytes(word));
        }
    }

    fn key(&mut self, key: &ExactStateKey) {
        for word in [
            key.black_lo as u64,
            (key.black_lo >> 64) as u64,
            key.black_hi as u64,
            (key.black_hi >> 64) as u64,
            key.white_lo as u64,
            (key.white_lo >> 64) as u64,
            key.white_hi as u64,
            (key.white_hi >> 64) as u64,
            stone_tag(key.side_to_move),
            stone_tag(key.root_attacker),
            role_tag(key.role),
            key.remaining_horizon as u64,
            rule_tag(key.rule_set),
            key.policy_digest,
        ] {
            self.u64(word);
        }
    }

    fn finish_hex(&self) -> String {
        format!(
            "{:016X}{:016X}{:016X}{:016X}",
            self.lanes[0], self.lanes[1], self.lanes[2], self.lanes[3]
        )
    }
}

#[inline]
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

const fn stone_tag(stone: Stone) -> u64 {
    match stone {
        Stone::Black => 1,
        Stone::White => 2,
    }
}

const fn role_tag(role: NodeRole) -> u64 {
    match role {
        NodeRole::Or => 1,
        NodeRole::And => 2,
    }
}

const fn rule_tag(rule: RuleSet) -> u64 {
    match rule {
        RuleSet::Freestyle => 1,
        RuleSet::Standard => 2,
        RuleSet::Caro => 3,
        RuleSet::Renju => 4,
    }
}

const fn status_tag(status: DfpnStatus) -> u64 {
    match status {
        DfpnStatus::ProvenWin => 1,
        DfpnStatus::ExhaustedBounded => 2,
        DfpnStatus::UnknownNodeBudget => 3,
        DfpnStatus::UnknownMemory => 4,
        DfpnStatus::UnknownAbort => 5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::to_idx;

    fn board_from_moves(moves: &[(usize, usize)]) -> Board {
        let mut board = Board::new();
        for &(row, col) in moves {
            let mv = to_idx(row, col);
            assert!(board.is_legal_move(mv));
            assert!(
                replay_scan_terminal(&board, board.side_to_move)
                    .unwrap()
                    .is_none()
            );
            board.make_move(mv);
        }
        board
    }

    fn immediate_attacker_board() -> Board {
        board_from_moves(&[
            (7, 3),
            (0, 0),
            (7, 4),
            (0, 14),
            (7, 5),
            (14, 0),
            (7, 6),
            (14, 14),
        ])
    }

    fn open_three_attacker_board() -> Board {
        board_from_moves(&[(7, 5), (0, 0), (7, 6), (0, 14), (7, 7), (14, 0)])
    }

    fn defender_counter_board() -> Board {
        board_from_moves(&[
            (7, 6),
            (0, 1),
            (7, 7),
            (0, 2),
            (12, 12),
            (0, 3),
            (13, 10),
            (0, 4),
        ])
    }

    fn exhaustive_bounded(
        board: &mut Board,
        attacker: Stone,
        remaining: u8,
    ) -> Result<bool, DfpnError> {
        if let Some(leaf) = replay_scan_terminal(board, attacker)? {
            return Ok(leaf == LeafKind::AttackerFive);
        }
        if remaining == 0 {
            return Ok(false);
        }
        if board.side_to_move == attacker {
            let moves = registered_attack_moves(board);
            if moves.is_empty() {
                return Ok(false);
            }
            for mv in moves {
                board.make_move(mv);
                let proved = exhaustive_bounded(board, attacker, remaining - 1)?;
                board.undo_move();
                if proved {
                    return Ok(true);
                }
            }
            Ok(false)
        } else {
            let moves = board.legal_moves();
            if moves.is_empty() {
                return Ok(false);
            }
            for mv in moves {
                board.make_move(mv);
                let proved = exhaustive_bounded(board, attacker, remaining - 1)?;
                board.undo_move();
                if !proved {
                    return Ok(false);
                }
            }
            Ok(true)
        }
    }

    fn solve(board: &mut Board, horizon: u8) -> (BoundedDfpnSession, DfpnCheckpoint) {
        let mut session =
            BoundedDfpnSession::new(board, BoundedDfpnConfig::small(horizon)).unwrap();
        let checkpoint = session.advance_to(board, 100_000).unwrap();
        (session, checkpoint)
    }

    #[test]
    fn immediate_attacker_five_is_proved_and_replayed() {
        let mut board = immediate_attacker_board();
        let snapshot = board.clone();
        let (session, checkpoint) = solve(&mut board, 1);
        assert_eq!(checkpoint.status, DfpnStatus::ProvenWin);
        assert_eq!(checkpoint.pn, 0);
        assert!(board_exact_eq(&board, &snapshot));

        let replay = session.verify_terminal_certificate(&mut board).unwrap();
        assert_eq!(replay.status, DfpnStatus::ProvenWin);
        assert!(replay.root_restored);
        assert!(replay.visited_nodes >= 2);
        assert!(board_exact_eq(&board, &snapshot));
    }

    #[test]
    fn defender_immediate_five_counter_disproves_root() {
        let mut board = defender_counter_board();
        assert_eq!(board.side_to_move, Stone::Black);
        let (session, checkpoint) = solve(&mut board, 2);
        assert_eq!(checkpoint.status, DfpnStatus::ExhaustedBounded);
        assert_eq!(checkpoint.dn, 0);
        let replay = session.verify_terminal_certificate(&mut board).unwrap();
        assert_eq!(replay.status, DfpnStatus::ExhaustedBounded);
        assert!(replay.root_restored);
    }

    #[test]
    fn and_ledger_contains_irrelevant_distant_defenses() {
        let mut board = defender_counter_board();
        let root_move_count = board.move_count;
        let mut session = BoundedDfpnSession::new(&board, BoundedDfpnConfig::small(2)).unwrap();
        let checkpoint = session.advance_to(&mut board, 2).unwrap();
        assert_eq!(checkpoint.expansions, 2);
        let first_and = session.nodes[session.nodes[session.root_id].edges[0].child].clone();
        assert_eq!(first_and.key.role, NodeRole::And);
        assert_eq!(
            first_and.edges.len(),
            NUM_CELLS - (root_move_count + 1),
            "every legal defender move, including distant irrelevant moves, must be present"
        );
        assert_eq!(first_and.edges.first().unwrap().mv, 0);
    }

    #[test]
    fn open_four_proof_replays_every_legal_defense() {
        let mut board = open_three_attacker_board();
        let snapshot = board.clone();
        let (session, checkpoint) = solve(&mut board, 3);
        assert_eq!(checkpoint.status, DfpnStatus::ProvenWin);

        let proved_and = session.nodes[session.root_id]
            .edges
            .iter()
            .map(|edge| &session.nodes[edge.child])
            .find(|child| child.pn == 0)
            .expect("root proof must select a proved AND child");
        assert_eq!(proved_and.key.role, NodeRole::And);
        assert_eq!(
            proved_and.edges.len(),
            NUM_CELLS - (snapshot.move_count + 1),
            "the proof ledger must contain every legal defender move"
        );
        assert!(
            proved_and
                .edges
                .iter()
                .all(|edge| session.nodes[edge.child].pn == 0),
            "every legal defense must lead to a proved attacker continuation"
        );

        let replay = session.verify_terminal_certificate(&mut board).unwrap();
        assert!(
            replay.visited_edges >= proved_and.edges.len() as u64 + 1,
            "certificate replay must traverse the root edge and every defense"
        );
        assert!(board_exact_eq(&board, &snapshot));
    }

    #[test]
    fn no_forcing_attacker_move_is_bounded_disproof() {
        let mut board = Board::new();
        let (session, checkpoint) = solve(&mut board, 3);
        assert_eq!(checkpoint.status, DfpnStatus::ExhaustedBounded);
        assert_eq!(checkpoint.expansions, 1);
        session.verify_terminal_certificate(&mut board).unwrap();
    }

    #[test]
    fn budget_interrupt_resume_matches_fresh_solution() {
        let mut resumed_board = immediate_attacker_board();
        let snapshot = resumed_board.clone();
        let config = BoundedDfpnConfig::small(1);
        let mut resumed = BoundedDfpnSession::new(&resumed_board, config).unwrap();
        let first = resumed.advance_to(&mut resumed_board, 1).unwrap();
        assert_eq!(first.status, DfpnStatus::UnknownNodeBudget);
        assert!(matches!(
            resumed.verify_terminal_certificate(&mut resumed_board),
            Err(DfpnError::NoSolvedCertificate(
                DfpnStatus::UnknownNodeBudget
            ))
        ));
        let final_resumed = resumed.advance_to(&mut resumed_board, 10).unwrap();

        let mut fresh_board = snapshot.clone();
        let mut fresh = BoundedDfpnSession::new(&fresh_board, config).unwrap();
        let final_fresh = fresh.advance_to(&mut fresh_board, 10).unwrap();
        assert_eq!(final_resumed.status, final_fresh.status);
        assert_eq!(
            (final_resumed.pn, final_resumed.dn),
            (final_fresh.pn, final_fresh.dn)
        );
        assert_eq!(final_resumed.expansions, final_fresh.expansions);
        assert_eq!(
            resumed
                .verify_terminal_certificate(&mut resumed_board)
                .unwrap()
                .certificate_digest,
            fresh
                .verify_terminal_certificate(&mut fresh_board)
                .unwrap()
                .certificate_digest
        );
        assert!(board_exact_eq(&resumed_board, &snapshot));
        assert!(board_exact_eq(&fresh_board, &snapshot));
    }

    #[test]
    fn deliberate_fingerprint_alias_keeps_exact_states_separate() {
        let mut board = immediate_attacker_board();
        let mut session = BoundedDfpnSession::new(&board, BoundedDfpnConfig::small(1)).unwrap();
        session.force_fingerprint_mask(0);
        let checkpoint = session.advance_to(&mut board, 10).unwrap();
        assert_eq!(checkpoint.status, DfpnStatus::ProvenWin);
        assert_eq!(checkpoint.distinct_fingerprints, 1);
        assert!(checkpoint.exact_states > 1);
        assert!(checkpoint.fingerprint_collisions > 0);
        assert_eq!(checkpoint.exact_alias_errors, 0);
        session.verify_terminal_certificate(&mut board).unwrap();
    }

    #[test]
    fn memory_rejection_is_atomic_and_unknown() {
        let mut board = immediate_attacker_board();
        let config = BoundedDfpnConfig {
            max_horizon: 1,
            memory_cap_bytes: 200,
        };
        let mut session = BoundedDfpnSession::new(&board, config).unwrap();
        let checkpoint = session.advance_to(&mut board, 100).unwrap();
        assert_eq!(checkpoint.status, DfpnStatus::UnknownMemory);
        assert_eq!(checkpoint.expansions, 0);
        assert_eq!(checkpoint.exact_states, 1);
        assert_eq!(checkpoint.stored_edges, 0);
    }

    #[test]
    fn dfpn_matches_exhaustive_small_bounded_reference() {
        let cases = [
            (immediate_attacker_board(), 1u8),
            (defender_counter_board(), 2u8),
            (Board::new(), 3u8),
        ];
        for (mut board, horizon) in cases {
            let snapshot = board.clone();
            let attacker = board.side_to_move;
            let expected = exhaustive_bounded(&mut board, attacker, horizon).unwrap();
            assert!(board_exact_eq(&board, &snapshot));
            let (_, checkpoint) = solve(&mut board, horizon);
            let observed = match checkpoint.status {
                DfpnStatus::ProvenWin => true,
                DfpnStatus::ExhaustedBounded => false,
                status => panic!("small exhaustive case unresolved: {status:?}"),
            };
            assert_eq!(observed, expected);
            assert!(board_exact_eq(&board, &snapshot));
        }
    }

    #[test]
    fn full_board_terminal_scan_ignores_last_move() {
        let mut board = immediate_attacker_board();
        board.make_move(to_idx(7, 2));
        assert_eq!(
            replay_scan_terminal(&board, Stone::Black).unwrap(),
            Some(LeafKind::AttackerFive)
        );
        board.last_move = Some(to_idx(14, 14));
        assert_eq!(
            scan_terminal(&board, Stone::Black).unwrap(),
            Some(LeafKind::AttackerFive)
        );
    }
}
