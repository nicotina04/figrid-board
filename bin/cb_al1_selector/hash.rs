use figrid_board::board::{BOARD_SIZE, Move, Stone};
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub(crate) type Digest = [u8; 32];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ColoredMove {
    pub(crate) mv: Move,
    pub(crate) color: Stone,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FileSeal {
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
}

#[derive(Clone)]
pub(crate) struct Sha256 {
    state: [u32; 8],
    block: [u8; 64],
    block_len: usize,
    bytes: u64,
}

impl Sha256 {
    pub(crate) fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            block: [0; 64],
            block_len: 0,
            bytes: 0,
        }
    }

    pub(crate) fn update(&mut self, mut input: &[u8]) {
        self.bytes = self.bytes.wrapping_add(input.len() as u64);

        if self.block_len != 0 {
            let take = (64 - self.block_len).min(input.len());
            self.block[self.block_len..self.block_len + take].copy_from_slice(&input[..take]);
            self.block_len += take;
            input = &input[take..];
            if self.block_len == 64 {
                compress(&mut self.state, &self.block);
                self.block_len = 0;
            }
        }

        while input.len() >= 64 {
            let block: &[u8; 64] = input[..64].try_into().expect("64-byte SHA-256 block");
            compress(&mut self.state, block);
            input = &input[64..];
        }

        if !input.is_empty() {
            self.block[..input.len()].copy_from_slice(input);
            self.block_len = input.len();
        }
    }

    pub(crate) fn finalize(mut self) -> Digest {
        let bit_len = self.bytes.wrapping_mul(8);
        self.update(&[0x80]);
        let zero_count = (56 + 64 - self.block_len) % 64;
        if zero_count != 0 {
            self.update(&[0u8; 64][..zero_count]);
        }
        self.update(&bit_len.to_be_bytes());
        debug_assert_eq!(self.block_len, 0);

        let mut digest = [0u8; 32];
        for (index, word) in self.state.iter().enumerate() {
            digest[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        digest
    }
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn sha256(input: &[u8]) -> Digest {
    let mut hash = Sha256::new();
    hash.update(input);
    hash.finalize()
}

pub(crate) fn hex_upper(digest: &Digest) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::with_capacity(64);
    for &byte in digest {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

pub(crate) fn sha256_hex(input: &[u8]) -> String {
    hex_upper(&sha256(input))
}

pub(crate) fn seal_file(path: &Path) -> Result<FileSeal, String> {
    let mut file =
        File::open(path).map_err(|error| format!("failed to open {}: {error}", path.display()))?;
    let mut hash = Sha256::new();
    let mut bytes = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        bytes = bytes
            .checked_add(read as u64)
            .ok_or_else(|| format!("byte count overflow for {}", path.display()))?;
        hash.update(&buffer[..read]);
    }
    Ok(FileSeal {
        bytes,
        sha256: hex_upper(&hash.finalize()),
    })
}

pub(crate) fn require_file_seal(
    path: &Path,
    expected_bytes: u64,
    expected_sha256: &str,
    label: &str,
) -> Result<FileSeal, String> {
    let observed = seal_file(path)?;
    if observed.bytes != expected_bytes {
        return Err(format!(
            "{label} byte mismatch: got {}, expected {expected_bytes}",
            observed.bytes
        ));
    }
    if observed.sha256 != expected_sha256 {
        return Err(format!(
            "{label} SHA-256 mismatch: got {}, expected {expected_sha256}",
            observed.sha256
        ));
    }
    Ok(observed)
}

pub(crate) fn transform_xy(x: usize, y: usize, transform: usize) -> (usize, usize) {
    let n = BOARD_SIZE - 1;
    [
        (x, y),
        (n - y, x),
        (n - x, n - y),
        (y, n - x),
        (n - x, y),
        (x, n - y),
        (y, x),
        (n - y, n - x),
    ][transform]
}

pub(crate) fn canonical_position_hash(history: &[ColoredMove], side: Stone) -> String {
    let mut canonical: Option<String> = None;
    for transform in 0..8 {
        let mut stones = history
            .iter()
            .map(|stone| {
                let x = stone.mv % BOARD_SIZE;
                let y = stone.mv / BOARD_SIZE;
                let (tx, ty) = transform_xy(x, y, transform);
                format!("{}{:03}", stone_char(stone.color), ty * BOARD_SIZE + tx)
            })
            .collect::<Vec<_>>();
        stones.sort();
        let form = format!("rule=0|side={}|{}", stone_char(side), stones.join(","));
        if canonical.as_ref().is_none_or(|current| form < *current) {
            canonical = Some(form);
        }
    }
    sha256_hex(
        format!(
            "RQ608-state-v1|{}",
            canonical.expect("the D4 transform set is nonempty")
        )
        .as_bytes(),
    )
}

pub(crate) fn canonical_opening_hash(history: &[ColoredMove]) -> Result<String, String> {
    if history.len() != 4 {
        return Err(format!(
            "ordered opening requires exactly four plies, got {}",
            history.len()
        ));
    }
    let mut canonical: Option<String> = None;
    for transform in 0..8 {
        let stones = history
            .iter()
            .enumerate()
            .map(|(ply, stone)| {
                let x = stone.mv % BOARD_SIZE;
                let y = stone.mv / BOARD_SIZE;
                let (tx, ty) = transform_xy(x, y, transform);
                format!(
                    "{ply}:{}{:03}",
                    stone_char(stone.color),
                    ty * BOARD_SIZE + tx
                )
            })
            .collect::<Vec<_>>();
        let form = format!("rule=0|{}", stones.join(","));
        if canonical.as_ref().is_none_or(|current| form < *current) {
            canonical = Some(form);
        }
    }
    Ok(sha256_hex(
        format!(
            "RQ608-ordered-opening-v1|{}",
            canonical.expect("the D4 transform set is nonempty")
        )
        .as_bytes(),
    ))
}

pub(crate) fn split_bucket(opening_hash: &str) -> u8 {
    let digest = sha256(format!("RQ615C|opening-group|{opening_hash}").as_bytes());
    digest.iter().fold(0u16, |remainder, &byte| {
        (remainder * 256 + u16::from(byte)) % 100
    }) as u8
}

pub(crate) fn unit_uid(
    opening_hash: &str,
    ordinal: usize,
    black_parent_hash: &str,
    white_parent_hash: &str,
) -> String {
    sha256_hex(
        format!(
            "RQ615C|structural-unit|{opening_hash}|{ordinal}|{black_parent_hash}|{white_parent_hash}"
        )
        .as_bytes(),
    )
}

pub(crate) fn parent_uid(unit_uid: &str, side: Stone, parent_hash: &str) -> String {
    sha256_hex(
        format!(
            "RQ615C|structural-parent|{unit_uid}|{}|{parent_hash}",
            stone_char(side)
        )
        .as_bytes(),
    )
}

pub(crate) fn historical_unit_rank_digest(opening_hash: &str, ordinal: usize) -> Digest {
    sha256(format!("RQ615C|unit|{opening_hash}|{ordinal}").as_bytes())
}

pub(crate) fn selector_digest(domain: &str, uppercase_uid: &str) -> Digest {
    sha256(format!("{domain}{uppercase_uid}").as_bytes())
}

pub(crate) fn uid_stream_hash<'a>(uids: impl IntoIterator<Item = &'a str>) -> String {
    let mut hash = Sha256::new();
    for uid in uids {
        hash.update(uid.as_bytes());
        hash.update(b"\n");
    }
    hex_upper(&hash.finalize())
}

pub(crate) fn stone_char(stone: Stone) -> char {
    match stone {
        Stone::Black => 'B',
        Stone::White => 'W',
    }
}

fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
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

    let mut words = [0u32; 64];
    for (index, bytes) in block.chunks_exact(4).take(16).enumerate() {
        words[index] = u32::from_be_bytes(bytes.try_into().expect("four-byte SHA-256 word"));
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

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for index in 0..64 {
        let big1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let choose = (e & f) ^ ((!e) & g);
        let temp1 = h
            .wrapping_add(big1)
            .wrapping_add(choose)
            .wrapping_add(K[index])
            .wrapping_add(words[index]);
        let big0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let majority = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = big0.wrapping_add(majority);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vectors_and_streaming_agree() {
        assert_eq!(
            sha256_hex(b""),
            "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD"
        );
        let mut streaming = Sha256::new();
        streaming.update(b"a");
        streaming.update(b"b");
        streaming.update(b"c");
        assert_eq!(hex_upper(&streaming.finalize()), sha256_hex(b"abc"));
    }

    #[test]
    fn d4_state_and_opening_hashes_are_transform_invariant() {
        let history = [
            ColoredMove {
                mv: 112,
                color: Stone::Black,
            },
            ColoredMove {
                mv: 98,
                color: Stone::White,
            },
            ColoredMove {
                mv: 113,
                color: Stone::Black,
            },
            ColoredMove {
                mv: 82,
                color: Stone::White,
            },
        ];
        let rotated = history.map(|stone| {
            let (x, y) = (stone.mv % BOARD_SIZE, stone.mv / BOARD_SIZE);
            let (tx, ty) = transform_xy(x, y, 1);
            ColoredMove {
                mv: ty * BOARD_SIZE + tx,
                color: stone.color,
            }
        });
        assert_eq!(
            canonical_position_hash(&history, Stone::Black),
            canonical_position_hash(&rotated, Stone::Black)
        );
        assert_eq!(
            canonical_opening_hash(&history).unwrap(),
            canonical_opening_hash(&rotated).unwrap()
        );
    }
}
