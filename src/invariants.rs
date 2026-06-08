use serde::{Deserialize, Serialize};

/// A quantity that is (approximately) invariant under renormalization-group flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleInvariant {
    pub name: String,
    pub value: f64,
    pub deviation_across_scales: f64,
}

impl ScaleInvariant {
    /// Create a new scale invariant from values measured at different scales.
    ///
    /// `value` is the mean; `deviation_across_scales` is the standard deviation.
    pub fn from_scale_values(name: impl Into<String>, values: &[f64]) -> Self {
        let n = values.len() as f64;
        let mean = if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / n
        };
        let deviation = if values.len() < 2 {
            0.0
        } else {
            let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
            var.sqrt()
        };
        Self {
            name: name.into(),
            value: mean,
            deviation_across_scales: deviation,
        }
    }

    /// Check whether this invariant is well-conserved (deviation below threshold).
    pub fn is_conserved(&self, threshold: f64) -> bool {
        self.deviation_across_scales < threshold
    }

    /// Relative deviation: deviation / |value|.
    pub fn relative_deviation(&self) -> f64 {
        if self.value.abs() < 1e-14 {
            self.deviation_across_scales
        } else {
            self.deviation_across_scales / self.value.abs()
        }
    }
}

/// Check a collection of invariants and return those that are not conserved.
pub fn find_violations(invariants: &[ScaleInvariant], threshold: f64) -> Vec<&ScaleInvariant> {
    invariants
        .iter()
        .filter(|inv| !inv.is_conserved(threshold))
        .collect()
}

#[cfg(test)]
mod invariant_tests {
    use super::*;

    #[test]
    fn test_exact_invariant() {
        let inv = ScaleInvariant::from_scale_values("energy", &[100.0, 100.0, 100.0]);
        assert!(inv.is_conserved(0.01));
        assert_eq!(inv.value, 100.0);
    }

    #[test]
    fn test_violated_invariant() {
        let inv = ScaleInvariant::from_scale_values("something", &[1.0, 10.0, 100.0]);
        assert!(!inv.is_conserved(0.5));
    }
}
