use serde::{Deserialize, Serialize};

/// Whether an observable grows, shrinks, or stays constant under RG flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Relevance {
    Relevant,
    Irrelevant,
    Marginal,
}

/// An observable quantity tracked across scales.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observable {
    pub name: String,
    pub values: Vec<f64>,
    pub relevance: Relevance,
    pub scaling_dimension: f64,
}

impl Observable {
    /// Create a new observable with unknown relevance.
    pub fn new(name: impl Into<String>, values: Vec<f64>, scaling_dimension: f64) -> Self {
        Self {
            name: name.into(),
            values,
            relevance: Relevance::Marginal,
            scaling_dimension,
        }
    }

    /// Classify relevance based on how values change across scales.
    ///
    /// Compares the first and last values: if the ratio exceeds `threshold`,
    /// the observable is relevant (growing); if below `1/threshold`, irrelevant
    /// (shrinking); otherwise marginal.
    pub fn classify(&mut self, threshold: f64) {
        if self.values.len() < 2 {
            self.relevance = Relevance::Marginal;
            return;
        }
        let first = self.values.first().copied().unwrap_or(0.0);
        let last = self.values.last().copied().unwrap_or(0.0);
        if first.abs() < 1e-14 {
            self.relevance = Relevance::Marginal;
            return;
        }
        let ratio = last / first;
        if ratio > threshold {
            self.relevance = Relevance::Relevant;
        } else if ratio < 1.0 / threshold {
            self.relevance = Relevance::Irrelevant;
        } else {
            self.relevance = Relevance::Marginal;
        }
    }

    /// Compute the mean of all values.
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    /// Compute the variance of all values.
    pub fn variance(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }
        let m = self.mean();
        let n = self.values.len() as f64;
        self.values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n
    }

    /// Apply a scaling dimension to transform the value at a given scale ratio.
    pub fn rescale(&self, scale_ratio: f64) -> f64 {
        self.values
            .last()
            .copied()
            .unwrap_or(0.0)
            * scale_ratio.powf(self.scaling_dimension)
    }
}
