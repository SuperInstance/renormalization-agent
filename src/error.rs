use std::fmt;

/// Errors produced by renormalization-group operations.
#[derive(Debug)]
pub enum RgError {
    /// Input data is too short for the requested block size.
    InsufficientData { needed: usize, got: usize },
    /// A coupling vector length mismatch.
    CouplingMismatch { expected: usize, got: usize },
    /// Numerical divergence during RG flow integration.
    Divergence { step: usize, message: String },
    /// Fixed-point finder did not converge.
    NoConvergence { iterations: usize },
    /// Invalid scale parameters (e.g. non-positive, output ≥ input).
    InvalidScale { detail: String },
}

impl fmt::Display for RgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientData { needed, got } => {
                write!(f, "insufficient data: need {needed} samples, got {got}")
            }
            Self::CouplingMismatch { expected, got } => {
                write!(f, "coupling mismatch: expected {expected}, got {got}")
            }
            Self::Divergence { step, message } => {
                write!(f, "divergence at step {step}: {message}")
            }
            Self::NoConvergence { iterations } => {
                write!(f, "fixed-point finder did not converge in {iterations} iterations")
            }
            Self::InvalidScale { detail } => {
                write!(f, "invalid scale: {detail}")
            }
        }
    }
}

impl std::error::Error for RgError {}
