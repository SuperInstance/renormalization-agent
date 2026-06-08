use crate::error::RgError;
use crate::observable::Observable;
use crate::scale::ScaleTransform;

/// Multi-scale map of agent behavior: tick data → coarse-grained observables.
pub struct AgentScaleMap {
    pub tick_data: Vec<f64>,
    pub scales: Vec<f64>,
    pub observables: Vec<Observable>,
}

impl AgentScaleMap {
    /// Create a new agent scale map from raw tick data.
    pub fn new(tick_data: Vec<f64>) -> Self {
        Self {
            tick_data,
            scales: vec![1.0],
            observables: Vec::new(),
        }
    }

    /// Apply a block-averaging transform and record the resulting scale.
    pub fn coarse_grain(&mut self, block_size: usize) -> Result<(), RgError> {
        let transform = ScaleTransform::block_average(block_size);
        let coarse = transform.block_average_apply(&self.tick_data)?;
        let new_scale = self.scales.last().copied().unwrap_or(1.0) * block_size as f64;
        self.scales.push(new_scale);
        self.tick_data = coarse;
        Ok(())
    }

    /// Apply a decimation transform.
    pub fn decimate(&mut self, block_size: usize) -> Result<(), RgError> {
        let transform = ScaleTransform::decimation(block_size);
        let coarse = transform.decimation_apply(&self.tick_data)?;
        let new_scale = self.scales.last().copied().unwrap_or(1.0) * block_size as f64;
        self.scales.push(new_scale);
        self.tick_data = coarse;
        Ok(())
    }

    /// Apply wavelet-based coarse-graining.
    pub fn wavelet_coarse_grain(&mut self, block_size: usize) -> Result<(), RgError> {
        let transform = ScaleTransform::block_average(block_size);
        let coarse = transform.wavelet_apply(&self.tick_data)?;
        let new_scale = self.scales.last().copied().unwrap_or(1.0) * block_size as f64;
        self.scales.push(new_scale);
        self.tick_data = coarse;
        Ok(())
    }

    /// Extract the mean as an observable at the current scale.
    pub fn extract_mean(&mut self, name: impl Into<String>) {
        let name = name.into();
        let m = if self.tick_data.is_empty() {
            0.0
        } else {
            self.tick_data.iter().sum::<f64>() / self.tick_data.len() as f64
        };
        match self.observables.iter_mut().find(|o| o.name == name) {
            Some(obs) => obs.values.push(m),
            None => {
                let obs = Observable::new(&name, vec![m], 0.0);
                self.observables.push(obs);
            }
        }
    }

    /// Extract the variance as an observable at the current scale.
    pub fn extract_variance(&mut self, name: impl Into<String>) {
        let name = name.into();
        let v = if self.tick_data.len() < 2 {
            0.0
        } else {
            let m = self.tick_data.iter().sum::<f64>() / self.tick_data.len() as f64;
            self.tick_data.iter().map(|x| (x - m).powi(2)).sum::<f64>()
                / self.tick_data.len() as f64
        };
        match self.observables.iter_mut().find(|o| o.name == name) {
            Some(obs) => obs.values.push(v),
            None => {
                let obs = Observable::new(&name, vec![v], -2.0); // variance scales as L^{-2}
                self.observables.push(obs);
            }
        }
    }

    /// Extract the total sum as an observable (should be approximately scale-invariant).
    pub fn extract_total(&mut self, name: impl Into<String>) {
        let name = name.into();
        let total: f64 = self.tick_data.iter().copied().sum();
        match self.observables.iter_mut().find(|o| o.name == name) {
            Some(obs) => obs.values.push(total),
            None => {
                let obs = Observable::new(&name, vec![total], 1.0);
                self.observables.push(obs);
            }
        }
    }

    /// Get the number of scales recorded.
    pub fn num_scales(&self) -> usize {
        self.scales.len()
    }

    /// Get the current finest data length.
    pub fn current_data_len(&self) -> usize {
        self.tick_data.len()
    }
}
