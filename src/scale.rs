use serde::{Deserialize, Serialize};

use crate::error::RgError;

/// A coarse-graining operator that maps fine-grained data to a coarser scale.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleTransform {
    pub input_scale: f64,
    pub output_scale: f64,
    pub block_size: usize,
    pub transform_matrix: Vec<Vec<f64>>,
}

impl ScaleTransform {
    /// Create a block-averaging transform.
    ///
    /// Each block of `block_size` consecutive samples is replaced by its mean.
    pub fn block_average(block_size: usize) -> Self {
        Self {
            input_scale: 1.0,
            output_scale: block_size as f64,
            block_size,
            transform_matrix: Vec::new(),
        }
    }

    /// Create a decimation transform (keep every `block_size`-th sample).
    pub fn decimation(block_size: usize) -> Self {
        let n = block_size;
        let mut mat = vec![vec![0.0; n]; 1];
        mat[0][0] = 1.0;
        Self {
            input_scale: 1.0,
            output_scale: block_size as f64,
            block_size,
            transform_matrix: mat,
        }
    }

    /// Apply block averaging to a slice of data.
    pub fn block_average_apply(&self, data: &[f64]) -> Result<Vec<f64>, RgError> {
        if data.len() < self.block_size {
            return Err(RgError::InsufficientData {
                needed: self.block_size,
                got: data.len(),
            });
        }
        let n_blocks = data.len() / self.block_size;
        let mut out = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            let start = i * self.block_size;
            let end = start + self.block_size;
            let sum: f64 = data[start..end].iter().copied().sum();
            out.push(sum / self.block_size as f64);
        }
        Ok(out)
    }

    /// Apply decimation — keep the first element of each block.
    pub fn decimation_apply(&self, data: &[f64]) -> Result<Vec<f64>, RgError> {
        if data.len() < self.block_size {
            return Err(RgError::InsufficientData {
                needed: self.block_size,
                got: data.len(),
            });
        }
        let n_blocks = data.len() / self.block_size;
        let mut out = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            out.push(data[i * self.block_size]);
        }
        Ok(out)
    }

    /// Wavelet-inspired coarse-graining: averages with alternating sign weights
    /// (Haar-like approximation coefficients).
    pub fn wavelet_apply(&self, data: &[f64]) -> Result<Vec<f64>, RgError> {
        if data.len() < self.block_size {
            return Err(RgError::InsufficientData {
                needed: self.block_size,
                got: data.len(),
            });
        }
        let n_blocks = data.len() / self.block_size;
        let mut approx = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            let start = i * self.block_size;
            let end = start + self.block_size;
            // Haar approximation coefficient: weighted sum with +1/-1
            let mut val = 0.0;
            for (j, &x) in data[start..end].iter().enumerate() {
                let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
                val += sign * x;
            }
            approx.push(val / self.block_size as f64);
        }
        Ok(approx)
    }
}
