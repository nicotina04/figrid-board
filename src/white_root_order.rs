//! Built-in White root move-ordering model.
//!
//! The model contributes only a residual score for the 48-coordinate,
//! D4-invariant codebook summary. Search is responsible for selecting eligible
//! White-root quiet runs and for keeping tactical, terminal, PV, and killer
//! moves as barriers. The original baseline position remains an explicit
//! negative-unit-span anchor, so this model cannot silently replace the
//! engine's baseline ordering contract.

/// Public identifier for the built-in model format.
#[cfg(test)]
pub const WHITE_ROOT_ORDER_FORMAT: &str = "figrid-white-root-order-v1";

/// Public identifier for the 48-coordinate input layout.
#[cfg(test)]
pub const WHITE_ROOT_ORDER_FEATURE_SCHEMA: &str = "figrid-codebook-d4-orbit48-v1";

/// Public identifier for the baseline-position anchor.
#[cfg(test)]
pub const WHITE_ROOT_ORDER_ANCHOR_SCHEMA: &str = "baseline-run-negative-unit-span-v1";

/// Divisor applied to every integer orbit coordinate before multiplication.
pub const WHITE_ROOT_ORDER_COORDINATE_SCALE: f32 = 800.0;

/// Stable names for the D4 orbit coordinates.
#[cfg(test)]
pub const WHITE_ROOT_ORDER_COORDINATE_NAMES: [&str; 48] = [
    "corner_00",
    "corner_01",
    "corner_02",
    "corner_03",
    "corner_04",
    "corner_05",
    "corner_06",
    "corner_07",
    "corner_08",
    "corner_09",
    "corner_10",
    "corner_11",
    "corner_12",
    "corner_13",
    "corner_14",
    "corner_15",
    "edge_00",
    "edge_01",
    "edge_02",
    "edge_03",
    "edge_04",
    "edge_05",
    "edge_06",
    "edge_07",
    "edge_08",
    "edge_09",
    "edge_10",
    "edge_11",
    "edge_12",
    "edge_13",
    "edge_14",
    "edge_15",
    "center_00",
    "center_01",
    "center_02",
    "center_03",
    "center_04",
    "center_05",
    "center_06",
    "center_07",
    "center_08",
    "center_09",
    "center_10",
    "center_11",
    "center_12",
    "center_13",
    "center_14",
    "center_15",
];

/// Exact IEEE-754 coefficient bits for the production model.
///
/// Storing bits instead of decimal literals avoids compiler- or serializer-
/// dependent rounding and makes the release model identity directly testable.
pub const WHITE_ROOT_ORDER_COEFFICIENT_BITS: [u32; 48] = [
    0xC0DC_2DC1,
    0x41C0_0E02,
    0xC178_ED03,
    0x4188_D07C,
    0xC22B_9C48,
    0xBF9C_6533,
    0xBFA5_4D45,
    0xC171_4CD2,
    0xC0A4_E7F8,
    0xBC9E_9326,
    0x3F8F_B123,
    0x40B6_A0DF,
    0x40D7_D8F4,
    0xC18E_69BC,
    0x4087_AD0A,
    0xC179_1D6A,
    0xC09D_DF86,
    0x4109_42D4,
    0x40E8_8ED0,
    0x4130_F5B5,
    0xC087_5163,
    0x4119_1E32,
    0x41CA_E87C,
    0x4060_676A,
    0xC098_4D81,
    0x413D_C29D,
    0xC14C_F719,
    0x4187_381B,
    0xC09E_D7C9,
    0xC12D_6638,
    0xBFCE_448E,
    0xC1EC_551A,
    0xC106_784B,
    0x40A4_89D4,
    0x410A_2859,
    0x41E1_D867,
    0xC1F0_C0BA,
    0xC168_36EC,
    0x40BC_9649,
    0x40BD_4BA0,
    0xC19E_9F34,
    0xC1A9_3C76,
    0x3F97_6D92,
    0x414F_A8A6,
    0xC146_17CE,
    0x400F_5186,
    0xC043_7E9D,
    0xC081_E913,
];

/// Immutable built-in White root ordering model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WhiteRootOrder;

impl WhiteRootOrder {
    /// Return the exact release model.
    pub const fn production() -> Self {
        Self
    }

    /// Return the exact coefficient bit pattern used by the model.
    #[cfg(test)]
    pub const fn coefficient_bits(&self) -> [u32; 48] {
        WHITE_ROOT_ORDER_COEFFICIENT_BITS
    }

    /// Compute only the learned residual.
    ///
    /// Conversion, division, multiplication, and accumulation order are part
    /// of the model contract. Keep these as separate f32 operations and do not
    /// replace the loop with a vectorized reduction or `mul_add`.
    #[inline(never)]
    pub fn score_orbit48(&self, numerators: &[i64; 48]) -> Result<f32, String> {
        let mut residual = 0.0f32;
        for index in 0..48 {
            let converted = numerators[index] as f32;
            if !converted.is_finite() {
                return Err(format!(
                    "white-root-order numerator conversion is non-finite at coordinate {index}"
                ));
            }
            let coordinate = converted / WHITE_ROOT_ORDER_COORDINATE_SCALE;
            if !coordinate.is_finite() {
                return Err(format!(
                    "white-root-order scaled coordinate is non-finite at coordinate {index}"
                ));
            }
            let coefficient = f32::from_bits(WHITE_ROOT_ORDER_COEFFICIENT_BITS[index]);
            let term = coefficient * coordinate;
            if !term.is_finite() {
                return Err(format!(
                    "white-root-order residual term is non-finite at coordinate {index}"
                ));
            }
            residual = residual + term;
            if !residual.is_finite() {
                return Err(format!(
                    "white-root-order residual accumulator is non-finite at coordinate {index}"
                ));
            }
        }
        Ok(residual)
    }

    /// Build `-(run_index / (run_len - 1))` in the fixed f32 operation order.
    #[inline(never)]
    pub fn anchor_for_run_slot(run_index: usize, run_len: usize) -> Result<f32, String> {
        if run_len < 2 {
            return Err("white-root-order anchor requires a run of at least two moves".to_string());
        }
        if run_index >= run_len {
            return Err(format!(
                "white-root-order run index {run_index} is outside run length {run_len}"
            ));
        }
        let converted_index = run_index as f32;
        let converted_span = (run_len - 1) as f32;
        if !converted_index.is_finite() || !converted_span.is_finite() || converted_span == 0.0 {
            return Err("white-root-order anchor conversion is invalid".to_string());
        }
        let fraction = converted_index / converted_span;
        if !fraction.is_finite() {
            return Err("white-root-order anchor division is non-finite".to_string());
        }
        let anchor = -fraction;
        if !anchor.is_finite() {
            return Err("white-root-order anchor is non-finite".to_string());
        }
        Ok(anchor)
    }

    /// Add the immutable baseline anchor exactly once, after residual scoring.
    #[inline(never)]
    pub fn add_anchor_to_residual(
        &self,
        residual: f32,
        run_index: usize,
        run_len: usize,
    ) -> Result<f32, String> {
        if !residual.is_finite() {
            return Err("white-root-order residual input is non-finite".to_string());
        }
        let anchor = Self::anchor_for_run_slot(run_index, run_len)?;
        let score = anchor + residual;
        if !score.is_finite() {
            return Err("white-root-order anchor-plus-residual score is non-finite".to_string());
        }
        Ok(score)
    }

    /// Score one move in a baseline run using the complete model contract.
    #[cfg(test)]
    #[inline(never)]
    pub fn score_run_slot(
        &self,
        numerators: &[i64; 48],
        run_index: usize,
        run_len: usize,
    ) -> Result<f32, String> {
        let residual = self.score_orbit48(numerators)?;
        self.add_anchor_to_residual(residual, run_index, run_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent textual fixture copied from the frozen promotion source.
    // Keeping it in a different representation catches accidental edits to
    // the production u32 array without publishing research-only identifiers.
    const PROMOTION_SOURCE_BITS_HEX: [&str; 48] = [
        "C0DC2DC1", "41C00E02", "C178ED03", "4188D07C", "C22B9C48", "BF9C6533", "BFA54D45",
        "C1714CD2", "C0A4E7F8", "BC9E9326", "3F8FB123", "40B6A0DF", "40D7D8F4", "C18E69BC",
        "4087AD0A", "C1791D6A", "C09DDF86", "410942D4", "40E88ED0", "4130F5B5", "C0875163",
        "41191E32", "41CAE87C", "4060676A", "C0984D81", "413DC29D", "C14CF719", "4187381B",
        "C09ED7C9", "C12D6638", "BFCE448E", "C1EC551A", "C106784B", "40A489D4", "410A2859",
        "41E1D867", "C1F0C0BA", "C16836EC", "40BC9649", "40BD4BA0", "C19E9F34", "C1A93C76",
        "3F976D92", "414FA8A6", "C14617CE", "400F5186", "C0437E9D", "C081E913",
    ];

    #[test]
    fn production_coefficients_match_promotion_source_bits() {
        let expected = PROMOTION_SOURCE_BITS_HEX.map(|text| u32::from_str_radix(text, 16).unwrap());
        assert_eq!(WhiteRootOrder::production().coefficient_bits(), expected);
        assert!(expected.into_iter().map(f32::from_bits).all(f32::is_finite));
    }

    #[test]
    fn public_schema_and_coordinate_names_are_complete_and_generic() {
        assert_eq!(WHITE_ROOT_ORDER_FORMAT, "figrid-white-root-order-v1");
        assert_eq!(
            WHITE_ROOT_ORDER_FEATURE_SCHEMA,
            "figrid-codebook-d4-orbit48-v1"
        );
        assert_eq!(
            WHITE_ROOT_ORDER_ANCHOR_SCHEMA,
            "baseline-run-negative-unit-span-v1"
        );
        assert_eq!(WHITE_ROOT_ORDER_COORDINATE_NAMES.len(), 48);
        assert_eq!(WHITE_ROOT_ORDER_COORDINATE_NAMES[0], "corner_00");
        assert_eq!(WHITE_ROOT_ORDER_COORDINATE_NAMES[16], "edge_00");
        assert_eq!(WHITE_ROOT_ORDER_COORDINATE_NAMES[32], "center_00");
        assert_eq!(WHITE_ROOT_ORDER_COORDINATE_NAMES[47], "center_15");
        assert_eq!(
            WHITE_ROOT_ORDER_COORDINATE_SCALE.to_bits(),
            800.0f32.to_bits()
        );
    }

    #[test]
    fn serial_residual_and_final_anchor_add_match_golden_f32_bits() {
        let numerators = [
            -800, -663, -526, -389, -252, -115, 22, 159, 296, 433, 570, 707, -757, -620, -483,
            -346, -209, -72, 65, 202, 339, 476, 613, 750, -714, -577, -440, -303, -166, -29, 108,
            245, 382, 519, 656, 793, -671, -534, -397, -260, -123, 14, 151, 288, 425, 562, 699,
            -765,
        ];
        let model = WhiteRootOrder::production();
        let residual = model.score_orbit48(&numerators).unwrap();
        assert_eq!(residual.to_bits(), 0x42CC_AEB8);
        let score = model.add_anchor_to_residual(residual, 3, 8).unwrap();
        assert_eq!(score.to_bits(), 0x42CB_D34A);
        assert_eq!(
            model.score_run_slot(&numerators, 3, 8).unwrap().to_bits(),
            score.to_bits()
        );
    }

    #[test]
    fn anchor_endpoints_and_negative_zero_are_exact() {
        assert_eq!(
            WhiteRootOrder::anchor_for_run_slot(0, 4).unwrap().to_bits(),
            (-0.0f32).to_bits()
        );
        assert_eq!(
            WhiteRootOrder::anchor_for_run_slot(3, 4).unwrap().to_bits(),
            (-1.0f32).to_bits()
        );
        assert_eq!(
            WhiteRootOrder::anchor_for_run_slot(1, 3).unwrap().to_bits(),
            (-0.5f32).to_bits()
        );
    }

    #[test]
    fn invalid_run_and_nonfinite_residual_fail_closed() {
        assert!(WhiteRootOrder::anchor_for_run_slot(0, 0).is_err());
        assert!(WhiteRootOrder::anchor_for_run_slot(0, 1).is_err());
        assert!(WhiteRootOrder::anchor_for_run_slot(2, 2).is_err());
        let model = WhiteRootOrder::production();
        assert!(model.add_anchor_to_residual(f32::NAN, 0, 2).is_err());
        assert!(model.add_anchor_to_residual(f32::INFINITY, 0, 2).is_err());
    }
}
