use crate::error::RgError;

/// Type alias for beta functions to reduce type complexity.
type BetaFn = Box<dyn Fn(&[f64]) -> f64>;

/// Renormalization-group flow: coupling constants evolve under scale changes.
pub struct RGFlow {
    pub couplings: Vec<f64>,
    pub beta_functions: Vec<BetaFn>,
    pub fixed_points: Vec<Vec<f64>>,
}

impl RGFlow {
    /// Create an RG flow with the given initial couplings and beta functions.
    pub fn new(
        couplings: Vec<f64>,
        beta_functions: Vec<BetaFn>,
    ) -> Result<Self, RgError> {
        if couplings.len() != beta_functions.len() {
            return Err(RgError::CouplingMismatch {
                expected: couplings.len(),
                got: beta_functions.len(),
            });
        }
        Ok(Self {
            couplings,
            beta_functions,
            fixed_points: Vec::new(),
        })
    }

    /// Integrate the RG flow using Euler's method.
    ///
    /// `dt` is the step size (logarithmic scale change), `steps` is the number of
    /// Euler steps. Returns the trajectory of coupling vectors.
    pub fn integrate_euler(&mut self, dt: f64, steps: usize) -> Result<Vec<Vec<f64>>, RgError> {
        let _n = self.couplings.len();
        let mut trajectory = Vec::with_capacity(steps + 1);
        trajectory.push(self.couplings.clone());

        let mut current = self.couplings.clone();
        for step in 0..steps {
            let derivatives: Vec<f64> = self
                .beta_functions
                .iter()
                .map(|beta| beta(&current))
                .collect();

            for (i, d) in derivatives.iter().enumerate() {
                current[i] += dt * d;
                if !current[i].is_finite() {
                    return Err(RgError::Divergence {
                        step,
                        message: format!("coupling[{i}] diverged to {}", current[i]),
                    });
                }
            }
            trajectory.push(current.clone());
        }

        self.couplings = current;
        Ok(trajectory)
    }

    /// Find a fixed point starting from `initial_guess` using Newton's method.
    ///
    /// For simplicity, this uses a finite-difference Jacobian and requires
    /// `tolerance` on the norm of β(λ).
    pub fn find_fixed_point(
        &self,
        initial_guess: &[f64],
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<Vec<f64>, RgError> {
        let n = self.couplings.len();
        if initial_guess.len() != n {
            return Err(RgError::CouplingMismatch {
                expected: n,
                got: initial_guess.len(),
            });
        }

        let mut lambda = initial_guess.to_vec();
        let eps = 1e-8;

        for _iter in 0..max_iterations {
            // Evaluate beta at current point
            let beta: Vec<f64> = self
                .beta_functions
                .iter()
                .map(|bf| bf(&lambda))
                .collect();

            let norm: f64 = beta.iter().map(|b| b * b).sum::<f64>().sqrt();
            if norm < tolerance {
                return Ok(lambda);
            }

            // Build Jacobian via finite differences
            let mut jacobian = vec![vec![0.0; n]; n];
            for j in 0..n {
                let mut lambda_p = lambda.clone();
                let mut lambda_m = lambda.clone();
                lambda_p[j] += eps;
                lambda_m[j] -= eps;
                for (i, bf) in self.beta_functions.iter().enumerate() {
                    let bp = bf(&lambda_p);
                    let bm = bf(&lambda_m);
                    jacobian[i][j] = (bp - bm) / (2.0 * eps);
                }
            }

            // Solve J * delta = -beta via Gaussian elimination
            let mut aug = vec![vec![0.0; n + 1]; n];
            for (i, row) in aug.iter_mut().enumerate() {
                for (j, val) in row.iter_mut().enumerate().take(n) {
                    *val = jacobian[i][j];
                }
                row[n] = -beta[i];
            }

            // Forward elimination with partial pivoting
            #[allow(clippy::needless_range_loop)]
            for col in 0..n {
                // Pivot
                let mut max_row = col;
                let mut max_val = aug[col][col].abs();
                for row in (col + 1)..n {
                    if aug[row][col].abs() > max_val {
                        max_val = aug[row][col].abs();
                        max_row = row;
                    }
                }
                aug.swap(col, max_row);

                if aug[col][col].abs() < 1e-14 {
                    continue; // singular column, skip
                }

                let pivot_val = aug[col][col];
                for row in (col + 1)..n {
                    let factor = aug[row][col] / pivot_val;
                    for j in col..=n {
                        aug[row][j] -= factor * aug[col][j];
                    }
                }
            }

            // Back substitution
            let mut delta = vec![0.0; n];
            for i in (0..n).rev() {
                if aug[i][i].abs() < 1e-14 {
                    continue;
                }
                delta[i] = aug[i][n];
                for j in (i + 1)..n {
                    delta[i] -= aug[i][j] * delta[j];
                }
                delta[i] /= aug[i][i];
            }

            // Damped update
            let damping = 0.5;
            for i in 0..n {
                lambda[i] += damping * delta[i];
            }
        }

        Err(RgError::NoConvergence {
            iterations: max_iterations,
        })
    }

    /// Register a known fixed point.
    pub fn add_fixed_point(&mut self, point: Vec<f64>) {
        self.fixed_points.push(point);
    }

    /// Classify a perturbation direction as relevant, irrelevant, or marginal
    /// based on linearization around a fixed point.
    ///
    /// Returns eigenvalues of the stability matrix ∂βᵢ/∂λⱼ evaluated at the
    /// fixed point. Positive eigenvalue → relevant, negative → irrelevant,
    /// zero (within tolerance) → marginal.
    pub fn classify_perturbation(
        &self,
        fixed_point: &[f64],
        perturbation: &[f64],
        tolerance: f64,
    ) -> Vec<crate::observable::Relevance> {
        let n = self.couplings.len();
        let eps = 1e-6;

        // Stability matrix
        let mut stab = vec![vec![0.0; n]; n];
        for j in 0..n {
            let mut fp = fixed_point.to_vec();
            let mut fm = fixed_point.to_vec();
            fp[j] += eps;
            fm[j] -= eps;
            for (i, bf) in self.beta_functions.iter().enumerate() {
                let bp = bf(&fp);
                let bm = bf(&fm);
                stab[i][j] = (bp - bm) / (2.0 * eps);
            }
        }

        // Apply stability matrix to perturbation to get eigenvalue estimates
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let response: f64 = (0..n)
                .map(|j| stab[i][j] * perturbation[j])
                .sum();
            // Effective eigenvalue: response / perturbation (scalar projection)
            let eig = if perturbation[i].abs() > 1e-14 {
                response / perturbation[i]
            } else {
                0.0
            };

            if eig > tolerance {
                result.push(crate::observable::Relevance::Relevant);
            } else if eig < -tolerance {
                result.push(crate::observable::Relevance::Irrelevant);
            } else {
                result.push(crate::observable::Relevance::Marginal);
            }
        }
        result
    }
}
