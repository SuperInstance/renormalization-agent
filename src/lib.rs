//! # renormalization-agent
//!
//! Renormalization group (RG) tools for multi-scale agent behavior analysis.
//!
//! The renormalization group coarse-grains systems by integrating out short-scale
//! degrees of freedom. Applied to agents, this means zooming out from individual
//! ticks to hourly to daily behavior. Each coarse-graining step preserves relevant
//! observables and flows irrelevant ones toward fixed points.
//!
//! ## The Physics
//!
//! In statistical mechanics, the RG explains why vastly different microscopic
//! systems (water, magnets, fluids) exhibit the same critical behavior. They share
//! **universality classes** — they flow to the same fixed point under coarse-graining.
//! Applied to agents: two agents with very different tick-level behavior may be
//! indistinguishable at daily scales.
//!
//! ## Core concepts
//!
//! - **[`scale::ScaleTransform`]** — Coarse-graining operators (block averaging,
//!   decimation, wavelet-based).
//! - **[`flow::RGFlow`]** — RG flow with beta functions, Euler integration, and
//!   fixed-point finding.
//! - **[`observable::Observable`]** — Quantities tracked across scales with
//!   relevance classification.
//! - **[`agent_scale::AgentScaleMap`]** — Apply successive transforms to agent
//!   tick data.
//! - **[`invariants::ScaleInvariant`]** — Quantities conserved across scales.
//!
//! ## Quick Start
//!
//! ```
//! use renormalization_agent::AgentScaleMap;
//!
//! let ticks: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.1).sin()).collect();
//! let mut map = AgentScaleMap::new(ticks);
//! map.extract_mean("mean");
//! map.coarse_grain(4).unwrap();
//! map.extract_mean("mean");
//!
//! let obs = map.observables.iter().find(|o| o.name == "mean").unwrap();//! assert_eq!(obs.values.len(), 2); // one per scale
//! ```

pub mod agent_scale;
pub mod error;
pub mod flow;
pub mod invariants;
pub mod observable;
pub mod scale;

pub use agent_scale::AgentScaleMap;
pub use error::RgError;
pub use flow::RGFlow;
pub use invariants::ScaleInvariant;
pub use observable::{Observable, Relevance};
pub use scale::ScaleTransform;

#[cfg(test)]
mod tests {
    use super::*;

    // ── ScaleTransform tests ──────────────────────────────────────────

    #[test]
    fn block_average_constant_data() {
        let st = ScaleTransform::block_average(4);
        let data = vec![5.0; 16];
        let out = st.block_average_apply(&data).unwrap();
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|&v| (v - 5.0).abs() < 1e-12));
    }

    #[test]
    fn block_average_linear_data() {
        let st = ScaleTransform::block_average(2);
        let data: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let out = st.block_average_apply(&data).unwrap();
        // blocks: [0,1]->0.5, [2,3]->2.5, [4,5]->4.5, [6,7]->6.5
        assert_eq!(out, vec![0.5, 2.5, 4.5, 6.5]);
    }

    #[test]
    fn block_average_insufficient_data() {
        let st = ScaleTransform::block_average(10);
        let data = vec![1.0, 2.0, 3.0];
        let err = st.block_average_apply(&data);
        assert!(err.is_err());
    }

    #[test]
    fn decimation_keeps_first() {
        let st = ScaleTransform::decimation(3);
        let data: Vec<f64> = (0..9).map(|i| i as f64).collect();
        let out = st.decimation_apply(&data).unwrap();
        assert_eq!(out, vec![0.0, 3.0, 6.0]);
    }

    #[test]
    fn decimation_insufficient() {
        let st = ScaleTransform::decimation(5);
        let data = vec![1.0];
        assert!(st.decimation_apply(&data).is_err());
    }

    #[test]
    fn wavelet_constant_data() {
        let st = ScaleTransform::block_average(2);
        let data = vec![3.0; 8];
        // Haar: (3 - 3)/2 = 0 for every block
        let out = st.wavelet_apply(&data).unwrap();
        assert!(out.iter().all(|&v| v.abs() < 1e-12));
    }

    #[test]
    fn wavelet_alternating() {
        let st = ScaleTransform::block_average(2);
        let data = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        // (1 - (-1))/2 = 1.0 for each block
        let out = st.wavelet_apply(&data).unwrap();
        assert_eq!(out, vec![1.0, 1.0, 1.0]);
    }

    // ── RGFlow tests ──────────────────────────────────────────────────

    #[test]
    fn euler_trivial_beta() {
        // β = 0 → couplings don't change
        let mut flow = RGFlow::new(
            vec![1.0, 2.0],
            vec![
                Box::new(|_: &[f64]| 0.0),
                Box::new(|_: &[f64]| 0.0),
            ],
        )
        .unwrap();
        let traj = flow.integrate_euler(0.1, 10).unwrap();
        assert_eq!(traj.len(), 11);
        assert!((traj[10][0] - 1.0).abs() < 1e-12);
        assert!((traj[10][1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn euler_linear_beta() {
        // β₁ = -λ₁ (flow to zero)
        let mut flow = RGFlow::new(
            vec![1.0],
            vec![Box::new(|l: &[f64]| -l[0])],
        )
        .unwrap();
        let traj = flow.integrate_euler(0.01, 100).unwrap();
        // After t=1.0: λ ≈ e^{-1} ≈ 0.368
        let approx = traj[100][0];
        assert!((approx - 0.368).abs() < 0.05, "got {approx}");
    }

    #[test]
    fn coupling_mismatch_error() {
        let result = RGFlow::new(
            vec![1.0],
            vec![
                Box::new(|_: &[f64]| 0.0),
                Box::new(|_: &[f64]| 0.0),
            ],
        );
        assert!(result.is_err());
    }

    #[test]
    fn divergence_detection() {
        let mut flow = RGFlow::new(
            vec![1.0],
            vec![Box::new(|l: &[f64]| l[0] * l[0])], // β = λ², explodes
        )
        .unwrap();
        let result = flow.integrate_euler(1.0, 1000);
        assert!(result.is_err());
    }

    #[test]
    fn find_trivial_fixed_point() {
        // β = -λ → fixed point at 0
        let flow = RGFlow::new(
            vec![1.0],
            vec![Box::new(|l: &[f64]| -l[0])],
        )
        .unwrap();
        let fp = flow.find_fixed_point(&[0.5], 1e-8, 100).unwrap();
        assert!(fp[0].abs() < 1e-6);
    }

    #[test]
    fn find_nontrivial_fixed_point() {
        // β = λ - λ³, fixed points at 0 and ±1
        let flow = RGFlow::new(
            vec![0.5],
            vec![Box::new(|l: &[f64]| l[0] - l[0].powi(3))],
        )
        .unwrap();
        let fp = flow.find_fixed_point(&[0.8], 1e-8, 200).unwrap();
        assert!((fp[0] - 1.0).abs() < 0.01, "got {}", fp[0]);
    }

    #[test]
    fn fixed_point_no_convergence() {
        let flow = RGFlow::new(
            vec![0.0],
            vec![Box::new(|l: &[f64]| l[0] + 1.0)], // β = λ + 1, no fixed point
        )
        .unwrap();
        let result = flow.find_fixed_point(&[0.0], 1e-8, 5);
        assert!(result.is_err());
    }

    #[test]
    fn classify_perturbation_relevant() {
        // β = -λ, fixed point at 0
        // Perturbation along λ is actually IRRELEVANT (β = -λ flows to zero)
        // Let's use β = +λ which has a relevant direction
        let flow = RGFlow::new(
            vec![1.0],
            vec![Box::new(|l: &[f64]| l[0])],
        )
        .unwrap();
        let cls = flow.classify_perturbation(&[0.0], &[1.0], 0.1);
        assert_eq!(cls[0], Relevance::Relevant);
    }

    #[test]
    fn classify_perturbation_irrelevant() {
        // β = -λ, fixed point at 0
        let flow = RGFlow::new(
            vec![1.0],
            vec![Box::new(|l: &[f64]| -l[0])],
        )
        .unwrap();
        let cls = flow.classify_perturbation(&[0.0], &[1.0], 0.1);
        assert_eq!(cls[0], Relevance::Irrelevant);
    }

    // ── Observable tests ──────────────────────────────────────────────

    #[test]
    fn observable_mean_and_variance() {
        let obs = Observable::new("test", vec![2.0, 4.0, 6.0], 0.0);
        assert!((obs.mean() - 4.0).abs() < 1e-12);
        // variance = ((4-4)^2 + (2-4)^2 + (6-4)^2)/3 = 8/3
        assert!((obs.variance() - 8.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn observable_classify_relevant() {
        let mut obs = Observable::new("x", vec![1.0, 2.0, 4.0, 8.0], 0.0);
        obs.classify(2.0);
        assert_eq!(obs.relevance, Relevance::Relevant);
    }

    #[test]
    fn observable_classify_irrelevant() {
        let mut obs = Observable::new("x", vec![8.0, 4.0, 2.0, 1.0], 0.0);
        obs.classify(2.0);
        assert_eq!(obs.relevance, Relevance::Irrelevant);
    }

    #[test]
    fn observable_classify_marginal() {
        let mut obs = Observable::new("x", vec![1.0, 1.1, 0.9, 1.05], 0.0);
        obs.classify(2.0);
        assert_eq!(obs.relevance, Relevance::Marginal);
    }

    #[test]
    fn observable_classify_single_value() {
        let mut obs = Observable::new("x", vec![1.0], 0.0);
        obs.classify(2.0);
        assert_eq!(obs.relevance, Relevance::Marginal);
    }

    #[test]
    fn observable_rescale() {
        let obs = Observable::new("x", vec![10.0], 2.0); // scaling dim = 2
        let rescaled = obs.rescale(2.0); // 10 * 2^2 = 40
        assert!((rescaled - 40.0).abs() < 1e-12);
    }

    // ── AgentScaleMap tests ───────────────────────────────────────────

    #[test]
    fn agent_scale_coarse_grain_basic() {
        let data = vec![1.0; 16];
        let mut map = AgentScaleMap::new(data);
        map.coarse_grain(4).unwrap();
        assert_eq!(map.current_data_len(), 4);
        assert_eq!(map.scales.len(), 2);
        assert_eq!(map.scales, vec![1.0, 4.0]);
    }

    #[test]
    fn agent_scale_preserves_mean() {
        let data: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.01).sin()).collect();
        let expected_mean = data.iter().sum::<f64>() / data.len() as f64;
        let mut map = AgentScaleMap::new(data);
        map.extract_mean("mean");
        map.coarse_grain(4).unwrap();
        map.extract_mean("mean");
        let obs = map.observables.iter().find(|o| o.name == "mean").unwrap();
        // Mean should be approximately preserved
        assert!((obs.values[0] - obs.values[1]).abs() < 0.05);
        assert!((obs.values[0] - expected_mean).abs() < 0.01);
    }

    #[test]
    fn agent_scale_total_sum_invariant() {
        let data = vec![2.0; 64];
        let mut map = AgentScaleMap::new(data);
        map.extract_total("total");
        map.coarse_grain(4).unwrap(); // 64 → 16, each block sums to 8, mean=2, total = 16*2 = 32
        map.extract_total("total");
        // Block averaging preserves total: original total = 128, after: 16 blocks * 2.0 = 32
        // Wait — block averaging returns MEANS, not sums. So total = 16 * 2.0 = 32.
        // The total is NOT preserved under block averaging (we lose factor of block_size).
        let obs = map.observables.iter().find(|o| o.name == "total").unwrap();
        // After one coarse-grain: data = [2.0; 16], total = 32
        assert!((obs.values[0] - 128.0).abs() < 1e-12);
        assert!((obs.values[1] - 32.0).abs() < 1e-12);
    }

    #[test]
    fn agent_scale_multiple_levels() {
        let data: Vec<f64> = (0..1024).map(|i| i as f64).collect();
        let mut map = AgentScaleMap::new(data);
        assert_eq!(map.num_scales(), 1);
        map.coarse_grain(4).unwrap(); // 1024 → 256
        assert_eq!(map.num_scales(), 2);
        map.coarse_grain(4).unwrap(); // 256 → 64
        assert_eq!(map.num_scales(), 3);
        assert_eq!(map.current_data_len(), 64);
    }

    #[test]
    fn agent_scale_decimate() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mut map = AgentScaleMap::new(data);
        map.decimate(5).unwrap();
        assert_eq!(map.current_data_len(), 20);
        assert_eq!(map.tick_data[0], 0.0);
        assert_eq!(map.tick_data[1], 5.0);
    }

    // ── ScaleInvariant tests ──────────────────────────────────────────

    #[test]
    fn scale_invariant_from_constant() {
        let inv = ScaleInvariant::from_scale_values("energy", &[42.0, 42.0, 42.0]);
        assert!((inv.value - 42.0).abs() < 1e-12);
        assert!(inv.is_conserved(0.01));
    }

    #[test]
    fn scale_invariant_violated() {
        let inv = ScaleInvariant::from_scale_values("x", &[1.0, 5.0, 25.0]);
        assert!(!inv.is_conserved(0.1));
    }

    #[test]
    fn scale_invariant_relative_deviation() {
        let inv = ScaleInvariant::from_scale_values("y", &[10.0, 10.1, 9.9]);
        assert!(inv.relative_deviation() < 0.02);
    }

    #[test]
    fn find_violations_returns_bad() {
        let invariants = vec![
            ScaleInvariant::from_scale_values("good", &[1.0, 1.0, 1.0]),
            ScaleInvariant::from_scale_values("bad", &[1.0, 100.0, 1000.0]),
        ];
        let violations = invariants::find_violations(&invariants, 0.5);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].name, "bad");
    }

    // ── Integration tests ─────────────────────────────────────────────

    #[test]
    fn full_pipeline() {
        // Create tick data: noisy constant signal
        let tick_data: Vec<f64> = (0..4096)
            .map(|i| 5.0 + 0.1 * ((i as f64 * 0.37).sin()))
            .collect();

        let mut map = AgentScaleMap::new(tick_data);
        map.extract_mean("mean");
        map.extract_variance("var");

        // Coarse-grain through several levels
        for _ in 0..4 {
            map.coarse_grain(4).unwrap();
            map.extract_mean("mean");
            map.extract_variance("var");
        }

        // Mean should stay near 5.0
        let mean_obs = map.observables.iter().find(|o| o.name == "mean").unwrap();
        assert_eq!(mean_obs.values.len(), 5);
        for &v in &mean_obs.values {
            assert!((v - 5.0).abs() < 0.1, "mean = {v}");
        }

        // Variance should decrease under coarse-graining
        let var_obs = map.observables.iter().find(|o| o.name == "var").unwrap();
        for i in 1..var_obs.values.len() {
            assert!(
                var_obs.values[i] <= var_obs.values[i - 1] + 0.01,
                "variance increased at level {i}"
            );
        }
    }
}

