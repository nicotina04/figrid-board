use figrid_board::d4_hash::{D4_COMPOSE, D4_MAP};
use figrid_board::pattern_table::{WindowThreat, lookup_mapped_id, pack_window, read_window};
use figrid_board::{
    BOARD_SIZE, BitBoard, Board, Move, NUM_CELLS, QuietThreatConfig, RuleSet, Stone,
    classify_move_with_directions,
};
use std::collections::BTreeMap;

const GRAPH_DOMAIN: &[u8] = b"CB-GH1-GRAPH-V1\0";
const ROLE_DOMAIN: &[u8] = b"CB-GH1-ROLE-V1\0";
const ROOTED_TRANSITION_TAG: u8 = 1;
const DETAIL_LIMIT: usize = 64;
const AXES: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
const THREAT_CONFIG: QuietThreatConfig = QuietThreatConfig {
    min_gain: 1,
    enable_jump_three: false,
    enable_gap_four: false,
};

/// Fixed-size census fields for one exact rooted transition graph.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GraphShape {
    pub(crate) affected_sites: u32,
    pub(crate) board_cells: u32,
    pub(crate) boundary_cells: u32,
    pub(crate) factors: u32,
    pub(crate) incidences: u32,
    pub(crate) bytes: u32,
}

/// The graph byte string, rather than either hash, is the authoritative code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CanonicalTransition {
    pub(crate) bytes: Vec<u8>,
    pub(crate) digest: [u8; 32],
    pub(crate) key64: u64,
    pub(crate) min_mask: u8,
    pub(crate) shape: GraphShape,
    pub(crate) exact_role_bytes: Vec<u8>,
}

/// Per-transition semantic audit counters. Counts are never hidden by the
/// bounded detail buffer, so a caller can aggregate the whole corpus without
/// retaining an unbounded list of diagnostics.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TransitionAudit {
    pub(crate) transformed_board_checks: u64,
    pub(crate) transformed_board_mismatches: u64,
    pub(crate) coordinate_graph_checks: u64,
    pub(crate) coordinate_graph_mismatches: u64,
    pub(crate) coordinate_role_checks: u64,
    pub(crate) coordinate_role_mismatches: u64,
    pub(crate) canonical_bytes_checks: u64,
    pub(crate) canonical_bytes_mismatches: u64,
    pub(crate) digest_checks: u64,
    pub(crate) digest_mismatches: u64,
    pub(crate) key64_checks: u64,
    pub(crate) key64_mismatches: u64,
    pub(crate) exact_role_checks: u64,
    pub(crate) exact_role_mismatches: u64,
    pub(crate) min_mask_checks: u64,
    pub(crate) min_mask_mismatches: u64,
    pub(crate) color_role_checks: u64,
    pub(crate) color_role_mismatches: u64,
    pub(crate) details: Vec<String>,
    pub(crate) details_truncated: u64,
}

impl TransitionAudit {
    pub(crate) fn mismatch_count(&self) -> u64 {
        self.transformed_board_mismatches
            + self.coordinate_graph_mismatches
            + self.coordinate_role_mismatches
            + self.canonical_bytes_mismatches
            + self.digest_mismatches
            + self.key64_mismatches
            + self.exact_role_mismatches
            + self.min_mask_mismatches
            + self.color_role_mismatches
    }

    fn detail(&mut self, message: String) {
        if self.details.len() < DETAIL_LIMIT {
            self.details.push(message);
        } else {
            self.details_truncated += 1;
        }
    }
}

/// One-time audit for the frozen coordinate convention.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct CoordinateFormulaAudit {
    pub(crate) in_board_checks: u64,
    pub(crate) in_board_mismatches: u64,
    pub(crate) virtual_checks: u64,
    pub(crate) virtual_mismatches: u64,
    pub(crate) details: Vec<String>,
    pub(crate) details_truncated: u64,
}

impl CoordinateFormulaAudit {
    pub(crate) fn mismatch_count(&self) -> u64 {
        self.in_board_mismatches + self.virtual_mismatches
    }

    pub(crate) fn is_clean(&self) -> bool {
        self.mismatch_count() == 0
    }

    fn detail(&mut self, message: String) {
        if self.details.len() < DETAIL_LIMIT {
            self.details.push(message);
        } else {
            self.details_truncated += 1;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Occupancy {
    Empty = 0,
    Mover = 1,
    Opponent = 2,
    Boundary = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum OwnerRole {
    Mover = 0,
    Opponent = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Relation {
    Anchor = 0,
    Support = 1,
    Blocker = 2,
    FootprintEmpty = 3,
    Boundary = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct NodeId {
    row: i8,
    col: i8,
    boundary: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NodeRecord {
    id: NodeId,
    parent: Occupancy,
    child: Occupancy,
    rooted: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FactorRecord {
    owner: OwnerRole,
    parent_kind: WindowThreat,
    child_kind: WindowThreat,
    source_row: i8,
    source_col: i8,
    axis_row: i8,
    axis_col: i8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IncidenceRecord {
    factor: usize,
    node: usize,
    absolute_distance: u8,
    parent: Relation,
    child: Relation,
}

#[derive(Clone, Debug)]
struct GraphData {
    affected_sites: usize,
    nodes: Vec<NodeRecord>,
    factors: Vec<FactorRecord>,
    incidences: Vec<IncidenceRecord>,
}

fn checked_i8(value: i32, label: &str) -> Result<i8, String> {
    i8::try_from(value).map_err(|_| format!("{label}={value} is outside the i8 domain"))
}

fn in_board(row: i32, col: i32) -> bool {
    (0..BOARD_SIZE as i32).contains(&row) && (0..BOARD_SIZE as i32).contains(&col)
}

fn occupancy(board: &Board, cell: usize, mover: Stone) -> Occupancy {
    if board.black.get(cell) {
        if mover == Stone::Black {
            Occupancy::Mover
        } else {
            Occupancy::Opponent
        }
    } else if board.white.get(cell) {
        if mover == Stone::White {
            Occupancy::Mover
        } else {
            Occupancy::Opponent
        }
    } else {
        Occupancy::Empty
    }
}

fn owner_stone(owner: OwnerRole, mover: Stone) -> Stone {
    match owner {
        OwnerRole::Mover => mover,
        OwnerRole::Opponent => mover.opponent(),
    }
}

fn kind_tag(kind: WindowThreat) -> Result<u8, String> {
    match kind {
        WindowThreat::None => Ok(0),
        WindowThreat::OpenTwo => Ok(1),
        WindowThreat::ClosedThree => Ok(2),
        WindowThreat::OpenThree => Ok(3),
        WindowThreat::ClosedFour => Ok(4),
        WindowThreat::OpenFour => Ok(5),
        WindowThreat::Five => Ok(6),
        WindowThreat::JumpThree => {
            Err("JumpThree escaped the frozen enable_jump_three=false graph semantics".to_string())
        }
    }
}

fn relation_at(
    board: &Board,
    row: i32,
    col: i32,
    offset: i32,
    owner: OwnerRole,
    mover: Stone,
) -> Relation {
    if !in_board(row, col) {
        return Relation::Boundary;
    }
    let cell = row as usize * BOARD_SIZE + col as usize;
    if board.is_empty(cell) {
        if offset == 0 {
            Relation::Anchor
        } else {
            Relation::FootprintEmpty
        }
    } else if occupancy(board, cell, mover)
        == match owner {
            OwnerRole::Mover => Occupancy::Mover,
            OwnerRole::Opponent => Occupancy::Opponent,
        }
    {
        Relation::Support
    } else {
        Relation::Blocker
    }
}

fn insert_node(
    nodes: &mut Vec<NodeRecord>,
    node_by_id: &mut BTreeMap<NodeId, usize>,
    record: NodeRecord,
) -> Result<usize, String> {
    if let Some(&index) = node_by_id.get(&record.id) {
        if nodes[index] != record {
            return Err(format!(
                "shared node {:?} acquired inconsistent labels {:?} and {:?}",
                record.id, nodes[index], record
            ));
        }
        return Ok(index);
    }
    let index = nodes.len();
    nodes.push(record);
    node_by_id.insert(record.id, index);
    Ok(index)
}

fn build_graph(parent: &Board, mv: Move) -> Result<(GraphData, Board), String> {
    if mv >= NUM_CELLS {
        return Err(format!("rooted move {mv} is outside 0..{NUM_CELLS}"));
    }
    if !parent.is_empty(mv) {
        return Err(format!("rooted move {mv} is occupied"));
    }

    let mover = parent.side_to_move;
    let mut child = parent.clone();
    child.make_move(mv);
    let root_row = (mv / BOARD_SIZE) as i32;
    let root_col = (mv % BOARD_SIZE) as i32;

    let mut factors = Vec::new();
    let mut affected_sites = 0usize;
    for (dir_idx, &(axis_row, axis_col)) in AXES.iter().enumerate() {
        for root_offset in -5..=5 {
            let source_row = root_row + root_offset * axis_row;
            let source_col = root_col + root_offset * axis_col;
            if !in_board(source_row, source_col) {
                continue;
            }
            affected_sites += 1;
            let source = source_row as usize * BOARD_SIZE + source_col as usize;
            for owner in [OwnerRole::Mover, OwnerRole::Opponent] {
                let side = owner_stone(owner, mover);
                let parent_kind = if parent.is_empty(source) {
                    classify_move_with_directions(parent, source, side, THREAT_CONFIG).1[dir_idx]
                } else {
                    WindowThreat::None
                };
                let child_kind = if child.is_empty(source) {
                    classify_move_with_directions(&child, source, side, THREAT_CONFIG).1[dir_idx]
                } else {
                    WindowThreat::None
                };
                kind_tag(parent_kind)?;
                kind_tag(child_kind)?;
                if parent_kind == WindowThreat::None && child_kind == WindowThreat::None {
                    continue;
                }
                factors.push(FactorRecord {
                    owner,
                    parent_kind,
                    child_kind,
                    source_row: checked_i8(source_row - root_row, "factor source row")?,
                    source_col: checked_i8(source_col - root_col, "factor source col")?,
                    axis_row: axis_row as i8,
                    axis_col: axis_col as i8,
                });
            }
        }
    }
    if affected_sites > 44 {
        return Err(format!(
            "affected site frontier exceeded the registered maximum: {affected_sites}"
        ));
    }

    let mut nodes = Vec::new();
    let mut node_by_id = BTreeMap::new();
    let root_id = NodeId {
        row: 0,
        col: 0,
        boundary: false,
    };
    insert_node(
        &mut nodes,
        &mut node_by_id,
        NodeRecord {
            id: root_id,
            parent: occupancy(parent, mv, mover),
            child: occupancy(&child, mv, mover),
            rooted: true,
        },
    )?;

    let mut incidences = Vec::with_capacity(factors.len() * 9);
    for (factor_index, factor) in factors.iter().enumerate() {
        let source_row = root_row + factor.source_row as i32;
        let source_col = root_col + factor.source_col as i32;
        for offset in -4i32..=4 {
            let row = source_row + offset * factor.axis_row as i32;
            let col = source_col + offset * factor.axis_col as i32;
            let boundary = !in_board(row, col);
            let relative_row = checked_i8(row - root_row, "node relative row")?;
            let relative_col = checked_i8(col - root_col, "node relative col")?;
            if !(-9..=9).contains(&relative_row) || !(-9..=9).contains(&relative_col) {
                return Err(format!(
                    "factor footprint escaped registered [-9,9]^2: ({relative_row},{relative_col})"
                ));
            }
            let id = NodeId {
                row: relative_row,
                col: relative_col,
                boundary,
            };
            let (parent_occupancy, child_occupancy) = if boundary {
                (Occupancy::Boundary, Occupancy::Boundary)
            } else {
                let cell = row as usize * BOARD_SIZE + col as usize;
                (
                    occupancy(parent, cell, mover),
                    occupancy(&child, cell, mover),
                )
            };
            let node = insert_node(
                &mut nodes,
                &mut node_by_id,
                NodeRecord {
                    id,
                    parent: parent_occupancy,
                    child: child_occupancy,
                    rooted: relative_row == 0 && relative_col == 0,
                },
            )?;
            incidences.push(IncidenceRecord {
                factor: factor_index,
                node,
                absolute_distance: offset.unsigned_abs() as u8,
                parent: relation_at(parent, row, col, offset, factor.owner, mover),
                child: relation_at(&child, row, col, offset, factor.owner, mover),
            });
        }
    }

    if incidences.len() != factors.len() * 9 {
        return Err(format!(
            "incidence cardinality {} != factors {} * 9",
            incidences.len(),
            factors.len()
        ));
    }

    Ok((
        GraphData {
            affected_sites,
            nodes,
            factors,
            incidences,
        },
        child,
    ))
}

/// Frozen relative-coordinate D4 formulas from the preregistration.
pub(crate) fn transform_relative(transform: usize, row: i8, col: i8) -> (i8, i8) {
    match transform {
        0 => (row, col),
        1 => (col, -row),
        2 => (-row, -col),
        3 => (-col, row),
        4 => (row, -col),
        5 => (-row, col),
        6 => (col, row),
        7 => (-col, -row),
        _ => panic!("D4 transform index must be in 0..8"),
    }
}

fn frozen_formula_reference(transform: usize, row: i8, col: i8) -> (i16, i16) {
    let row = row as i16;
    let col = col as i16;
    match transform {
        0 => (row, col),
        1 => (col, -row),
        2 => (-row, -col),
        3 => (-col, row),
        4 => (row, -col),
        5 => (-row, col),
        6 => (col, row),
        7 => (-col, -row),
        _ => panic!("D4 transform index must be in 0..8"),
    }
}

fn normalized_axis(row: i8, col: i8) -> (i8, i8) {
    debug_assert!(row != 0 || col != 0);
    if row < 0 || (row == 0 && col < 0) {
        (-row, -col)
    } else {
        (row, col)
    }
}

fn transformed_node(node: NodeRecord, transform: usize) -> NodeRecord {
    let (row, col) = transform_relative(transform, node.id.row, node.id.col);
    NodeRecord {
        id: NodeId {
            row,
            col,
            boundary: node.id.boundary,
        },
        ..node
    }
}

fn transformed_factor(factor: FactorRecord, transform: usize) -> FactorRecord {
    let (source_row, source_col) =
        transform_relative(transform, factor.source_row, factor.source_col);
    let (axis_row, axis_col) = transform_relative(transform, factor.axis_row, factor.axis_col);
    let (axis_row, axis_col) = normalized_axis(axis_row, axis_col);
    FactorRecord {
        source_row,
        source_col,
        axis_row,
        axis_col,
        ..factor
    }
}

fn node_bytes(node: NodeRecord) -> [u8; 6] {
    [
        node.id.boundary as u8,
        node.id.row as u8,
        node.id.col as u8,
        node.parent as u8,
        node.child as u8,
        node.rooted as u8,
    ]
}

fn factor_bytes(factor: FactorRecord) -> Result<[u8; 7], String> {
    Ok([
        factor.owner as u8,
        kind_tag(factor.parent_kind)?,
        kind_tag(factor.child_kind)?,
        factor.source_row as u8,
        factor.source_col as u8,
        factor.axis_row as u8,
        factor.axis_col as u8,
    ])
}

fn incidence_bytes(
    incidence: IncidenceRecord,
    factor_ordinals: &[u32],
    node_ordinals: &[u32],
) -> [u8; 11] {
    let mut bytes = [0u8; 11];
    bytes[0..4].copy_from_slice(&factor_ordinals[incidence.factor].to_le_bytes());
    bytes[4..8].copy_from_slice(&node_ordinals[incidence.node].to_le_bytes());
    bytes[8] = incidence.absolute_distance;
    bytes[9] = incidence.parent as u8;
    bytes[10] = incidence.child as u8;
    bytes
}

fn serialize_graph(data: &GraphData, transform: usize) -> Result<Vec<u8>, String> {
    if transform >= 8 {
        return Err(format!("invalid graph transform {transform}"));
    }
    let mut transformed_nodes = data
        .nodes
        .iter()
        .copied()
        .enumerate()
        .map(|(source, node)| (node_bytes(transformed_node(node, transform)), source))
        .collect::<Vec<_>>();
    transformed_nodes.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    for pair in transformed_nodes.windows(2) {
        if pair[0].0 == pair[1].0 {
            return Err(format!(
                "duplicate transformed node record under T{transform}: {:?}",
                pair[0].0
            ));
        }
    }
    let mut node_ordinals = vec![0u32; data.nodes.len()];
    for (ordinal, &(_, source)) in transformed_nodes.iter().enumerate() {
        node_ordinals[source] =
            u32::try_from(ordinal).map_err(|_| "node ordinal overflow".to_string())?;
    }

    let mut transformed_factors = data
        .factors
        .iter()
        .copied()
        .enumerate()
        .map(|(source, factor)| {
            factor_bytes(transformed_factor(factor, transform)).map(|bytes| (bytes, source))
        })
        .collect::<Result<Vec<_>, _>>()?;
    transformed_factors.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    for pair in transformed_factors.windows(2) {
        if pair[0].0 == pair[1].0 {
            return Err(format!(
                "duplicate transformed factor record under T{transform}: {:?}",
                pair[0].0
            ));
        }
    }
    let mut factor_ordinals = vec![0u32; data.factors.len()];
    for (ordinal, &(_, source)) in transformed_factors.iter().enumerate() {
        factor_ordinals[source] =
            u32::try_from(ordinal).map_err(|_| "factor ordinal overflow".to_string())?;
    }

    let mut transformed_incidences = data
        .incidences
        .iter()
        .copied()
        .map(|incidence| incidence_bytes(incidence, &factor_ordinals, &node_ordinals))
        .collect::<Vec<_>>();
    transformed_incidences.sort_unstable();

    let node_count =
        u32::try_from(transformed_nodes.len()).map_err(|_| "node count overflow".to_string())?;
    let factor_count = u32::try_from(transformed_factors.len())
        .map_err(|_| "factor count overflow".to_string())?;
    let incidence_count = u32::try_from(transformed_incidences.len())
        .map_err(|_| "incidence count overflow".to_string())?;
    let capacity = GRAPH_DOMAIN.len()
        + 1
        + 12
        + transformed_nodes.len() * 6
        + transformed_factors.len() * 7
        + transformed_incidences.len() * 11;
    let mut bytes = Vec::with_capacity(capacity);
    bytes.extend_from_slice(GRAPH_DOMAIN);
    bytes.push(ROOTED_TRANSITION_TAG);
    bytes.extend_from_slice(&node_count.to_le_bytes());
    bytes.extend_from_slice(&factor_count.to_le_bytes());
    bytes.extend_from_slice(&incidence_count.to_le_bytes());
    for (record, _) in transformed_nodes {
        bytes.extend_from_slice(&record);
    }
    for (record, _) in transformed_factors {
        bytes.extend_from_slice(&record);
    }
    for record in transformed_incidences {
        bytes.extend_from_slice(&record);
    }
    debug_assert_eq!(bytes.len(), capacity);
    Ok(bytes)
}

fn rule_tag(rule: RuleSet) -> u8 {
    match rule {
        RuleSet::Freestyle => 0,
        RuleSet::Standard => 1,
        RuleSet::Caro => 2,
        RuleSet::Renju => 3,
    }
}

fn transformed_bitboard(source: &BitBoard, transform: usize) -> BitBoard {
    let mut result = BitBoard::EMPTY;
    for cell in source.iter_ones() {
        result.set(D4_MAP[transform][cell] as usize);
    }
    result
}

fn append_bitboard(bytes: &mut Vec<u8>, board: BitBoard) {
    // Big-endian limbs preserve the production exact-state convention.
    bytes.extend_from_slice(&board.lo.to_be_bytes());
    bytes.extend_from_slice(&board.hi.to_be_bytes());
}

fn exact_role_lane(
    parent: &Board,
    child: &Board,
    mv: Move,
    transform: usize,
) -> Result<Vec<u8>, String> {
    if transform >= 8 {
        return Err(format!("invalid exact-role transform {transform}"));
    }
    let mover = parent.side_to_move;
    let (parent_mover, parent_opponent, child_mover, child_opponent) = match mover {
        Stone::Black => (parent.black, parent.white, child.black, child.white),
        Stone::White => (parent.white, parent.black, child.white, child.black),
    };
    let mut bytes = Vec::with_capacity(ROLE_DOMAIN.len() + 3 + 128);
    bytes.extend_from_slice(ROLE_DOMAIN);
    bytes.push(ROOTED_TRANSITION_TAG);
    bytes.push(rule_tag(parent.effective_rule_set()));
    bytes.push(D4_MAP[transform][mv]);
    append_bitboard(&mut bytes, transformed_bitboard(&parent_mover, transform));
    append_bitboard(
        &mut bytes,
        transformed_bitboard(&parent_opponent, transform),
    );
    append_bitboard(&mut bytes, transformed_bitboard(&child_mover, transform));
    append_bitboard(&mut bytes, transformed_bitboard(&child_opponent, transform));
    Ok(bytes)
}

fn exact_role_canonical(parent: &Board, child: &Board, mv: Move) -> Result<Vec<u8>, String> {
    let mut best = exact_role_lane(parent, child, mv, 0)?;
    for transform in 1..8 {
        let candidate = exact_role_lane(parent, child, mv, transform)?;
        if candidate < best {
            best = candidate;
        }
    }
    Ok(best)
}

fn canonical_from_data(
    parent: &Board,
    child: &Board,
    mv: Move,
    data: &GraphData,
) -> Result<CanonicalTransition, String> {
    let mut lanes = Vec::with_capacity(8);
    for transform in 0..8 {
        lanes.push(serialize_graph(data, transform)?);
    }
    let bytes = lanes
        .iter()
        .min()
        .cloned()
        .ok_or_else(|| "D4 lane set is empty".to_string())?;
    let mut min_mask = 0u8;
    for (transform, lane) in lanes.iter().enumerate() {
        if *lane == bytes {
            min_mask |= 1u8 << transform;
        }
    }
    if min_mask == 0 {
        return Err("canonical graph minimum mask is empty".to_string());
    }

    let mut digest_input = Vec::with_capacity(GRAPH_DOMAIN.len() + bytes.len());
    digest_input.extend_from_slice(GRAPH_DOMAIN);
    digest_input.extend_from_slice(&bytes);
    let digest = sha256(&digest_input);
    let key64 = u64::from_le_bytes(
        digest[0..8]
            .try_into()
            .expect("SHA-256 has at least eight bytes"),
    );
    let board_cells = data.nodes.iter().filter(|node| !node.id.boundary).count();
    let boundary_cells = data.nodes.len() - board_cells;
    let shape = GraphShape {
        affected_sites: u32::try_from(data.affected_sites)
            .map_err(|_| "affected-site count overflow".to_string())?,
        board_cells: u32::try_from(board_cells)
            .map_err(|_| "board-cell count overflow".to_string())?,
        boundary_cells: u32::try_from(boundary_cells)
            .map_err(|_| "boundary-cell count overflow".to_string())?,
        factors: u32::try_from(data.factors.len())
            .map_err(|_| "factor count overflow".to_string())?,
        incidences: u32::try_from(data.incidences.len())
            .map_err(|_| "incidence count overflow".to_string())?,
        bytes: u32::try_from(bytes.len()).map_err(|_| "byte count overflow".to_string())?,
    };
    Ok(CanonicalTransition {
        bytes,
        digest,
        key64,
        min_mask,
        shape,
        exact_role_bytes: exact_role_canonical(parent, child, mv)?,
    })
}

pub(crate) fn canonical_transition(
    parent: &Board,
    mv: Move,
) -> Result<CanonicalTransition, String> {
    let (data, child) = build_graph(parent, mv)?;
    canonical_from_data(parent, &child, mv, &data)
}

fn transformed_board_via_history(parent: &Board, transform: usize) -> Result<Board, String> {
    if transform >= 8 {
        return Err(format!("invalid semantic board transform {transform}"));
    }
    if parent.history.len() != parent.move_count {
        return Err(format!(
            "history length {} != move_count {}",
            parent.history.len(),
            parent.move_count
        ));
    }
    let mut transformed = Board::new();
    transformed.set_rule_set(parent.effective_rule_set());
    for (ply, &mv) in parent.history.iter().enumerate() {
        if mv >= NUM_CELLS {
            return Err(format!("history ply {ply} has out-of-range move {mv}"));
        }
        let mapped = D4_MAP[transform][mv] as usize;
        if !transformed.is_empty(mapped) {
            return Err(format!(
                "history ply {ply} maps to duplicate cell {mapped} under T{transform}"
            ));
        }
        transformed.make_move(mapped);
    }
    let expected_black = transformed_bitboard(&parent.black, transform);
    let expected_white = transformed_bitboard(&parent.white, transform);
    if transformed.black != expected_black
        || transformed.white != expected_white
        || transformed.side_to_move != parent.side_to_move
        || transformed.move_count != parent.move_count
    {
        return Err(format!(
            "history semantic rebuild disagrees with transformed state under T{transform}"
        ));
    }
    Ok(transformed)
}

fn color_role_swapped(parent: &Board) -> Board {
    let mut swapped = parent.clone();
    std::mem::swap(&mut swapped.black, &mut swapped.white);
    swapped.side_to_move = swapped.side_to_move.opponent();
    // `classify_move_with_directions` independently checks its directional
    // reference against the released mapped-ID aggregate in debug builds.
    // Rebuild that absolute-color cache after swapping rather than relying on
    // a release-only elision of the check.
    for cell in 0..NUM_CELLS {
        let row = (cell / BOARD_SIZE) as i32;
        let col = (cell % BOARD_SIZE) as i32;
        for (dir_index, &(axis_row, axis_col)) in AXES.iter().enumerate() {
            let window = read_window(&swapped.black, &swapped.white, row, col, axis_row, axis_col);
            swapped.line_pattern_ids[cell][dir_index] = lookup_mapped_id(pack_window(&window));
        }
    }
    // Graph construction and role identity do not consume the stale
    // absolute-color Zobrist fingerprint.
    swapped
}

/// Transform an all-minimum mask after first applying `outer` to the board.
///
/// A lane `inner` on the transformed board is
/// `T_inner(T_outer(board)) = T_D4_COMPOSE[inner][outer](board)`.
pub(crate) fn composed_min_mask(original: u8, outer: usize) -> u8 {
    assert!(outer < 8, "D4 transform index must be in 0..8");
    let mut result = 0u8;
    for inner in 0..8 {
        let original_lane = D4_COMPOSE[inner][outer] as usize;
        if original & (1u8 << original_lane) != 0 {
            result |= 1u8 << inner;
        }
    }
    result
}

pub(crate) fn audit_transition_equivariance(
    parent: &Board,
    mv: Move,
) -> Result<TransitionAudit, String> {
    let (base_data, base_child) = build_graph(parent, mv)?;
    let base = canonical_from_data(parent, &base_child, mv, &base_data)?;
    let mut audit = TransitionAudit::default();

    for outer in 0..8 {
        audit.transformed_board_checks += 1;
        let transformed_parent = match transformed_board_via_history(parent, outer) {
            Ok(board) => board,
            Err(error) => {
                audit.transformed_board_mismatches += 1;
                audit.detail(format!("T{outer} semantic board rebuild: {error}"));
                continue;
            }
        };
        let transformed_move = D4_MAP[outer][mv] as usize;
        let (rebuilt_data, rebuilt_child) = match build_graph(&transformed_parent, transformed_move)
        {
            Ok(value) => value,
            Err(error) => {
                audit.transformed_board_mismatches += 1;
                audit.detail(format!("T{outer} semantic graph rebuild: {error}"));
                continue;
            }
        };
        let rebuilt = match canonical_from_data(
            &transformed_parent,
            &rebuilt_child,
            transformed_move,
            &rebuilt_data,
        ) {
            Ok(value) => value,
            Err(error) => {
                audit.transformed_board_mismatches += 1;
                audit.detail(format!("T{outer} semantic canonicalization: {error}"));
                continue;
            }
        };

        audit.coordinate_graph_checks += 1;
        let coordinate_graph = serialize_graph(&base_data, outer)?;
        let semantic_identity = serialize_graph(&rebuilt_data, 0)?;
        if coordinate_graph != semantic_identity {
            audit.coordinate_graph_mismatches += 1;
            audit.detail(format!(
                "T{outer} coordinate-transformed graph bytes differ from semantic rebuild"
            ));
        }

        audit.coordinate_role_checks += 1;
        let coordinate_role = exact_role_lane(parent, &base_child, mv, outer)?;
        let semantic_role =
            exact_role_lane(&transformed_parent, &rebuilt_child, transformed_move, 0)?;
        if coordinate_role != semantic_role {
            audit.coordinate_role_mismatches += 1;
            audit.detail(format!(
                "T{outer} coordinate-transformed exact-role bytes differ from semantic rebuild"
            ));
        }

        audit.canonical_bytes_checks += 1;
        if rebuilt.bytes != base.bytes {
            audit.canonical_bytes_mismatches += 1;
            audit.detail(format!("T{outer} canonical exact graph bytes changed"));
        }
        audit.digest_checks += 1;
        if rebuilt.digest != base.digest {
            audit.digest_mismatches += 1;
            audit.detail(format!("T{outer} graph SHA-256 changed"));
        }
        audit.key64_checks += 1;
        if rebuilt.key64 != base.key64 {
            audit.key64_mismatches += 1;
            audit.detail(format!("T{outer} graph prospective u64 changed"));
        }
        audit.exact_role_checks += 1;
        if rebuilt.exact_role_bytes != base.exact_role_bytes {
            audit.exact_role_mismatches += 1;
            audit.detail(format!("T{outer} canonical exact-role transition changed"));
        }
        audit.min_mask_checks += 1;
        let expected_mask = composed_min_mask(base.min_mask, outer);
        if rebuilt.min_mask != expected_mask {
            audit.min_mask_mismatches += 1;
            audit.detail(format!(
                "T{outer} min-mask mismatch: rebuilt={:#04x}, expected={expected_mask:#04x}",
                rebuilt.min_mask
            ));
        }
    }

    let swapped_parent = color_role_swapped(parent);
    let swapped = canonical_transition(&swapped_parent, mv)?;
    for (label, equal) in [
        ("exact graph bytes", swapped.bytes == base.bytes),
        ("graph SHA-256", swapped.digest == base.digest),
        ("prospective graph u64", swapped.key64 == base.key64),
        (
            "exact role-transition bytes",
            swapped.exact_role_bytes == base.exact_role_bytes,
        ),
        ("all-min D4 mask", swapped.min_mask == base.min_mask),
    ] {
        audit.color_role_checks += 1;
        if !equal {
            audit.color_role_mismatches += 1;
            audit.detail(format!("color-role swap changed {label}"));
        }
    }

    Ok(audit)
}

pub(crate) fn audit_d4_coordinate_formulas() -> CoordinateFormulaAudit {
    let mut audit = CoordinateFormulaAudit::default();
    for root in 0..NUM_CELLS {
        let root_row = (root / BOARD_SIZE) as i32;
        let root_col = (root % BOARD_SIZE) as i32;
        for cell in 0..NUM_CELLS {
            let cell_row = (cell / BOARD_SIZE) as i32;
            let cell_col = (cell % BOARD_SIZE) as i32;
            let relative_row = (cell_row - root_row) as i8;
            let relative_col = (cell_col - root_col) as i8;
            for transform in 0..8 {
                audit.in_board_checks += 1;
                let mapped_root = D4_MAP[transform][root] as usize;
                let mapped_cell = D4_MAP[transform][cell] as usize;
                let map_relative = (
                    (mapped_cell / BOARD_SIZE) as i32 - (mapped_root / BOARD_SIZE) as i32,
                    (mapped_cell % BOARD_SIZE) as i32 - (mapped_root % BOARD_SIZE) as i32,
                );
                let formula = transform_relative(transform, relative_row, relative_col);
                if map_relative != (formula.0 as i32, formula.1 as i32) {
                    audit.in_board_mismatches += 1;
                    audit.detail(format!(
                        "T{transform} root={root} cell={cell}: map={map_relative:?}, formula={formula:?}"
                    ));
                }
            }
        }
    }

    // The graph's virtual coordinates occupy this entire registered domain.
    // Check the formulas directly over that full square, then independently
    // check range closure and inverse recovery.
    const INVERSE: [usize; 8] = [0, 3, 2, 1, 4, 5, 6, 7];
    for row in -9i8..=9 {
        for col in -9i8..=9 {
            for transform in 0..8 {
                audit.virtual_checks += 1;
                let mapped = transform_relative(transform, row, col);
                let frozen = frozen_formula_reference(transform, row, col);
                let recovered = transform_relative(INVERSE[transform], mapped.0, mapped.1);
                if (mapped.0 as i16, mapped.1 as i16) != frozen
                    || !(-9..=9).contains(&mapped.0)
                    || !(-9..=9).contains(&mapped.1)
                    || recovered != (row, col)
                {
                    audit.virtual_mismatches += 1;
                    audit.detail(format!(
                        "virtual T{transform} ({row},{col}) -> {mapped:?}, frozen={frozen:?}, inverse={recovered:?}"
                    ));
                }
            }
        }
    }
    audit
}

// Minimal dependency-free SHA-256, retained here so the graph artifact does
// not change Cargo dependencies or couple its identity to the harness.
pub(crate) fn sha256(input: &[u8]) -> [u8; 32] {
    const INITIAL: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity((input.len() + 72) & !63);
    padded.extend_from_slice(input);
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    let mut state = INITIAL;
    for chunk in padded.chunks_exact(64) {
        let mut words = [0u32; 64];
        for (index, word) in words.iter_mut().take(16).enumerate() {
            *word = u32::from_be_bytes(
                chunk[index * 4..index * 4 + 4]
                    .try_into()
                    .expect("SHA-256 word slice is four bytes"),
            );
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let big_s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(big_s1)
                .wrapping_add(choose)
                .wrapping_add(K[index])
                .wrapping_add(words[index]);
            let big_s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = big_s0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }

    let mut digest = [0u8; 32];
    for (index, word) in state.iter().enumerate() {
        digest[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn sample_board() -> Board {
        let mut board = Board::new();
        for mv in [
            7 * BOARD_SIZE + 7,
            4 * BOARD_SIZE + 5,
            7 * BOARD_SIZE + 8,
            9 * BOARD_SIZE + 3,
            6 * BOARD_SIZE + 8,
            2 * BOARD_SIZE + 12,
            8 * BOARD_SIZE + 7,
        ] {
            board.make_move(mv);
        }
        board
    }

    #[test]
    fn sha256_known_answers_are_exact() {
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn relative_formulas_match_frozen_examples() {
        let input = (2, 5);
        let expected = [
            (2, 5),
            (5, -2),
            (-2, -5),
            (-5, 2),
            (2, -5),
            (-2, 5),
            (5, 2),
            (-5, -2),
        ];
        for (transform, expected) in expected.into_iter().enumerate() {
            assert_eq!(transform_relative(transform, input.0, input.1), expected);
        }
        let audit = audit_d4_coordinate_formulas();
        assert_eq!(audit.mismatch_count(), 0, "{:?}", audit.details);
        assert_eq!(audit.in_board_checks, (NUM_CELLS * NUM_CELLS * 8) as u64);
        assert_eq!(audit.virtual_checks, (19 * 19 * 8) as u64);
    }

    #[test]
    fn graph_shape_and_fixed_record_widths_close() {
        let board = sample_board();
        let transition = canonical_transition(&board, 7 * BOARD_SIZE + 6).unwrap();
        assert!(transition.shape.affected_sites <= 44);
        assert!(transition.shape.factors <= 88);
        assert_eq!(transition.shape.incidences, transition.shape.factors * 9);
        assert_eq!(
            transition.shape.bytes as usize,
            GRAPH_DOMAIN.len()
                + 1
                + 12
                + (transition.shape.board_cells + transition.shape.boundary_cells) as usize * 6
                + transition.shape.factors as usize * 7
                + transition.shape.incidences as usize * 11
        );
        assert!(transition.bytes.starts_with(GRAPH_DOMAIN));
        assert_eq!(transition.bytes[GRAPH_DOMAIN.len()], ROOTED_TRANSITION_TAG);
        assert_eq!(
            transition.key64,
            u64::from_le_bytes(transition.digest[0..8].try_into().unwrap())
        );
    }

    #[test]
    fn center_opening_retains_all_minimum_transforms() {
        let board = Board::new();
        let transition = canonical_transition(&board, 7 * BOARD_SIZE + 7).unwrap();
        assert_eq!(transition.min_mask, 0xff);
        assert!(transition.shape.factors > 0);
        assert_eq!(transition.shape.boundary_cells, 0);
    }

    #[test]
    fn semantic_d4_and_color_role_rebuilds_are_exact() {
        let board = sample_board();
        let audit = audit_transition_equivariance(&board, 7 * BOARD_SIZE + 6).unwrap();
        assert_eq!(audit.mismatch_count(), 0, "{audit:#?}");
        assert_eq!(audit.transformed_board_checks, 8);
        assert_eq!(audit.coordinate_graph_checks, 8);
        assert_eq!(audit.coordinate_role_checks, 8);
        assert_eq!(audit.canonical_bytes_checks, 8);
        assert_eq!(audit.digest_checks, 8);
        assert_eq!(audit.key64_checks, 8);
        assert_eq!(audit.exact_role_checks, 8);
        assert_eq!(audit.min_mask_checks, 8);
        assert_eq!(audit.color_role_checks, 5);
    }

    #[test]
    fn boundary_nodes_survive_geometric_rebuilds() {
        let mut board = Board::new();
        for mv in [1, 100, 2, 101] {
            board.make_move(mv);
        }
        let transition = canonical_transition(&board, 4).unwrap();
        assert!(transition.shape.boundary_cells > 0);
        let audit = audit_transition_equivariance(&board, 4).unwrap();
        assert_eq!(audit.mismatch_count(), 0, "{audit:#?}");
    }

    #[test]
    fn graph_translation_abstraction_does_not_erase_exact_role_identity() {
        let mut first = Board::new();
        for mv in [
            7 * BOARD_SIZE + 7,
            5 * BOARD_SIZE + 5,
            7 * BOARD_SIZE + 8,
            5 * BOARD_SIZE + 6,
        ] {
            first.make_move(mv);
        }
        let mut shifted = Board::new();
        for mv in [
            8 * BOARD_SIZE + 7,
            6 * BOARD_SIZE + 5,
            8 * BOARD_SIZE + 8,
            6 * BOARD_SIZE + 6,
        ] {
            shifted.make_move(mv);
        }
        let first_code = canonical_transition(&first, 7 * BOARD_SIZE + 6).unwrap();
        let shifted_code = canonical_transition(&shifted, 8 * BOARD_SIZE + 6).unwrap();
        assert_eq!(first_code.bytes, shifted_code.bytes);
        assert_eq!(first_code.digest, shifted_code.digest);
        assert_ne!(first_code.exact_role_bytes, shifted_code.exact_role_bytes);
    }

    #[test]
    fn minimum_mask_composition_matches_lane_permutation() {
        for mask in 1u16..=255 {
            for outer in 0..8 {
                let composed = composed_min_mask(mask as u8, outer);
                assert_eq!(composed.count_ones(), (mask as u8).count_ones());
                for inner in 0..8 {
                    assert_eq!(
                        composed & (1 << inner) != 0,
                        mask as u8 & (1 << D4_COMPOSE[inner][outer]) != 0
                    );
                }
            }
        }
    }

    #[test]
    fn serialization_and_digest_golden() {
        let board = sample_board();
        let transition = canonical_transition(&board, 7 * BOARD_SIZE + 6).unwrap();
        // These values freeze enum tags, record ordering, count widths, the
        // joint D4 choice, and digest domain separation in one compact golden.
        assert_eq!(transition.shape.affected_sites, 44);
        assert_eq!(transition.shape.incidences, transition.shape.factors * 9);
        assert_eq!(
            hex(&transition.digest),
            "b0e403aca085b5f2f90a1ab11a6a188c9dbf927145b411331fabec0fc5866227"
        );
        assert_eq!(
            hex(&transition.bytes[..GRAPH_DOMAIN.len() + 13]),
            "43422d4748312d47524150482d563100012a0000000800000048000000"
        );
    }

    #[test]
    fn rejected_inputs_do_not_construct_partial_codes() {
        let mut board = Board::new();
        let mv = 7 * BOARD_SIZE + 7;
        board.make_move(mv);
        assert!(canonical_transition(&board, mv).is_err());
        assert!(canonical_transition(&board, NUM_CELLS).is_err());
    }
}
