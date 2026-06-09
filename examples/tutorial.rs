//! # Renormalization Agent — World-Class Tutorial
//!
//! This tutorial teaches renormalization group (RG) analysis for agent behavior.
//! Each lesson builds on the last, from basic coarse-graining to full RG flows.
//!
//! ## The Big Idea
//!
//! The renormalization group explains *universality*: why wildly different
//! microscopic systems produce the same macroscopic behavior. Water molecules
//! and spin lattices share critical exponents because they flow to the same
//! fixed point under coarse-graining.
//!
//! For agents: two agents with very different tick-level behavior may be
//! indistinguishable at daily scales. RG tells us *which* details survive
//! coarse-graining (relevant) and which wash out (irrelevant).
//!
//! Run: `cargo run --example tutorial`

use renormalization_agent::{
    AgentScaleMap, Observable, RGFlow, Relevance, ScaleInvariant,
};

// ─────────────────────────────────────────────────────────────────────────────
// Lesson 1: Block Averaging — Coarse-Graining 101
// ─────────────────────────────────────────────────────────────────────────────
//
// The simplest RG transform: replace each block of B consecutive values with
// their mean. This is the "majority rule" of renormalization.
//
// Block averaging preserves the MEAN of the signal (it's a linear average)
// but reduces variance (high-frequency fluctuations are averaged out).
//
// PHYSICS: This is exactly how thermodynamics emerges. Individual molecule
// positions don't matter — only aggregate properties (temperature, pressure)
// survive coarse-graining.

fn lesson1_block_averaging() {
    println!("═ Lesson 1: Block Averaging — Coarse-Graining Signal ═\n");

    // Generate a noisy signal: DC + sinusoidal
    let data: Vec<f64> = (0..1024)
        .map(|i| 5.0 + 0.1 * ((i as f64 * 0.37).sin()))
        .collect();

    let original_mean = data.iter().sum::<f64>() / data.len() as f64;
    let original_var = {
        let m = original_mean;
        data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / data.len() as f64
    };

    println!("  Original signal: {} samples", data.len());
    println!("  Mean:   {:.6}", original_mean);
    println!("  Variance: {:.6}\n", original_var);

    // Successive coarse-graining: 1024 → 256 → 64 → 16 → 4
    let mut map = AgentScaleMap::new(data);
    map.extract_mean("mean");
    map.extract_variance("variance");

    println!("  {:>6}  {:>6}  {:>12}  {:>12}", "Scale", "N", "Mean", "Variance");

    for _ in 0..4 {
        map.coarse_grain(4).unwrap();
        map.extract_mean("mean");
        map.extract_variance("variance");
        let scale = *map.scales.last().unwrap();
        let n = map.current_data_len();
        println!("  {:>6.0}  {:>6}  {:>12.6}  {:>12.6}",
                 scale, n,
                 *map.observables.iter().find(|o| o.name == "mean").unwrap().values.last().unwrap(),
                 *map.observables.iter().find(|o| o.name == "variance").unwrap().values.last().unwrap());
    }

    let mean_obs = map.observables.iter().find(|o| o.name == "mean").unwrap();
    let var_obs = map.observables.iter().find(|o| o.name == "variance").unwrap();

    // Mean is preserved; variance decreases
    let first_mean = mean_obs.values[0];
    let last_mean = *mean_obs.values.last().unwrap();
    let first_var = var_obs.values[0];
    let last_var = *var_obs.values.last().unwrap();

    println!("\n  Mean drift: |Δ| = {:.2e} (preserved)", (last_mean - first_mean).abs());
    println!("  Variance ratio: {:.4} → {:.6} (decreases)\n", first_var, last_var);

    assert!((last_mean - first_mean).abs() < 0.1, "Mean should be preserved");
    assert!(last_var <= first_var + 0.01, "Variance should decrease");
}

// ─────────────────────────────────────────────────────────────────────────────
// Lesson 2: Decimation — Keep Every Nth Sample
// ─────────────────────────────────────────────────────────────────────────────
//
// Decimation is the other fundamental RG transform: keep every Nth sample,
// discard the rest. Unlike block averaging, decimation DOES NOT preserve
// the mean — it preserves the value at sampled points.
//
// PHYSICS: Decimation is "integrating out" degrees of freedom. In the Ising
// model, you keep every other spin and trace over the rest.

fn lesson2_decimation() {
    println!("═ Lesson 2: Decimation — Subsampling Degrees of Freedom ═\n");

    let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
    println!("  Original: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...] (100 samples)");

    let mut map = AgentScaleMap::new(data.clone());
    map.decimate(5).unwrap();

    println!("  After decimation (B=5): first 5 values = {:?}", &map.tick_data[..5.min(map.tick_data.len())]);
    println!("  Samples: {} → {}", 100, map.current_data_len());
    println!("  Scale: 1 → {}\n", map.scales[1]);

    assert_eq!(map.tick_data[0], 0.0);
    assert_eq!(map.tick_data[1], 5.0);
    assert_eq!(map.tick_data[2], 10.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Lesson 3: Observables and Relevance Classification
// ─────────────────────────────────────────────────────────────────────────────
//
// Not all quantities survive coarse-graining. RG classifies observables as:
//
//   RELEVANT:   Grows under coarse-graining → determines macroscopic behavior
//   IRRELEVANT:  Shrinks under coarse-graining → becomes negligible at large scales
//   MARGINAL:    Stays roughly constant → boundary case, subtle physics
//
// Example: In a magnet near the critical point:
//   - Magnetization (order parameter) is RELEVANT
//   - Short-range correlations are IRRELEVANT
//   - Logarithmic corrections are MARGINAL

fn lesson3_observable_relevance() {
    println!("═ Lesson 3: Observable Relevance — What Survives Coarse-Graining? ═\n");

    // Relevant: values grow with scale
    let mut relevant_obs = Observable::new("growing", vec![1.0, 2.0, 4.0, 8.0], 0.0);
    relevant_obs.classify(2.0);
    println!("  Growing values [1, 2, 4, 8]:  {:?}", relevant_obs.relevance);

    // Irrelevant: values shrink with scale
    let mut irrelevant_obs = Observable::new("shrinking", vec![8.0, 4.0, 2.0, 1.0], 0.0);
    irrelevant_obs.classify(2.0);
    println!("  Shrinking values [8, 4, 2, 1]: {:?}", irrelevant_obs.relevance);

    // Marginal: values stay roughly constant
    let mut marginal_obs = Observable::new("constant", vec![1.0, 1.1, 0.9, 1.05], 0.0);
    marginal_obs.classify(2.0);
    println!("  Constant values [1, 1.1, 0.9, 1.05]: {:?}\n", marginal_obs.relevance);

    assert_eq!(relevant_obs.relevance, Relevance::Relevant);
    assert_eq!(irrelevant_obs.relevance, Relevance::Irrelevant);
    assert_eq!(marginal_obs.relevance, Relevance::Marginal);
}

// ─────────────────────────────────────────────────────────────────────────────
// Lesson 4: RG Flow — Beta Functions and Fixed Points
// ─────────────────────────────────────────────────────────────────────────────
//
// The RG flow describes how coupling constants evolve under scale changes.
//
//   dλ/dt = β(λ)
//
// where t = ln(scale). Fixed points are where β(λ*) = 0.
//
// Example: β(λ) = -λ has a fixed point at λ* = 0.
//   - Starting from λ > 0: flows toward 0 (irrelevant)
//   - The Gaussian fixed point: all couplings are zero
//
// This is the most important concept in RG: near a fixed point, you can
// linearize β and classify perturbations.

fn lesson4_rg_flow_fixed_points() {
    println!("═ Lesson 4: RG Flow — Beta Functions and Fixed Points ═\n");

    // β(λ) = -λ: flows to zero (Gaussian fixed point)
    let mut flow = RGFlow::new(
        vec![1.0],
        vec![Box::new(|l: &[f64]| -l[0])],
    ).unwrap();

    println!("  β(λ) = -λ (Gaussian flow)");
    println!("  Initial coupling: λ = 1.0");

    let trajectory = flow.integrate_euler(0.1, 20).unwrap();
    println!("  Flow trajectory:");
    for (i, point) in trajectory.iter().enumerate() {
        if i % 5 == 0 || i == trajectory.len() - 1 {
            println!("    t = {:>4.1}: λ = {:.6}", i as f64 * 0.1, point[0]);
        }
    }

    // Find the fixed point
    let fp = flow.find_fixed_point(&[0.5], 1e-8, 100).unwrap();
    println!("  Fixed point: λ* = {:.2e}\n", fp[0]);

    assert!(fp[0].abs() < 1e-6, "Fixed point should be at 0");
}

// ─────────────────────────────────────────────────────────────────────────────
// Lesson 5: Non-Trivial Fixed Point and Universality
// ─────────────────────────────────────────────────────────────────────────────
//
// β(λ) = λ - λ³ has three fixed points:
//   λ* = 0  (unstable: relevant direction)
//   λ* = ±1 (stable: attractive)
//
// This is the Landau-Ginzburg beta function! It describes phase transitions.
// Starting from λ = 0.5, the flow goes to λ* = 1.
//
// CLASSIFYING PERTURBATIONS: Near a fixed point, expand β = M·δλ + ...
//   - Positive eigenvalue of M → RELEVANT perturbation (grows)
//   - Negative eigenvalue → IRRELEVANT (shrinks)
//   - Zero eigenvalue → MARGINAL (needs higher-order analysis)

fn lesson5_nontrivial_fixed_point() {
    println!("═ Lesson 5: Non-Trivial Fixed Point — Phase Transitions ═\n");

    // β(λ) = λ - λ³: Landau-Ginzburg flow
    let flow = RGFlow::new(
        vec![0.5],
        vec![Box::new(|l: &[f64]| l[0] - l[0].powi(3))],
    ).unwrap();

    // Find the non-trivial fixed point
    let fp = flow.find_fixed_point(&[0.8], 1e-8, 200).unwrap();
    println!("  β(λ) = λ - λ³");
    println!("  Non-trivial fixed point: λ* = {:.6} (exact: 1.0)", fp[0]);

    // Classify perturbations around λ* = 0
    let cls_at_zero = flow.classify_perturbation(&[0.0], &[1.0], 0.1);
    println!("  Perturbation around λ* = 0: {:?}", cls_at_zero[0]);
    println!("  → λ* = 0 is UNSTABLE (relevant direction)");

    // Classify perturbations around λ* = 1
    // β'(1) = 1 - 3(1)² = -2 < 0 → irrelevant
    let cls_at_one = flow.classify_perturbation(&[1.0], &[1.0], 0.1);
    println!("  Perturbation around λ* = 1: {:?}", cls_at_one[0]);
    println!("  → λ* = 1 is STABLE (irrelevant direction)\n");

    assert!((fp[0] - 1.0).abs() < 0.01, "Fixed point should be near 1.0");
    assert_eq!(cls_at_zero[0], Relevance::Relevant);
    assert_eq!(cls_at_one[0], Relevance::Irrelevant);
}

// ─────────────────────────────────────────────────────────────────────────────
// Lesson 6: Scale Invariants — What's Conserved Across Scales
// ─────────────────────────────────────────────────────────────────────────────
//
// A scale invariant is a quantity that stays constant under RG flow.
// In physics, conservation laws correspond to symmetries (Noether's theorem).
// In RG, scale invariants correspond to relevant directions at fixed points.
//
// Example: The mean of a signal is invariant under block averaging.
// The variance is NOT invariant — it scales as L^{-2}.

fn lesson6_scale_invariants() {
    println!("═ Lesson 6: Scale Invariants — Conservation Across Scales ═\n");

    let tick_data: Vec<f64> = (0..4096)
        .map(|i| 5.0 + 0.1 * ((i as f64 * 0.37).sin()))
        .collect();

    let mut map = AgentScaleMap::new(tick_data);
    map.extract_mean("mean");

    for _ in 0..4 {
        map.coarse_grain(4).unwrap();
        map.extract_mean("mean");
    }

    let mean_obs = map.observables.iter().find(|o| o.name == "mean").unwrap();

    // Check if mean is a scale invariant
    let inv = ScaleInvariant::from_scale_values("mean", &mean_obs.values);
    println!("  Observable: mean");
    println!("  Values across scales: {:?}", mean_obs.values.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>());
    println!("  Scale invariant value: {:.6}", inv.value);
    println!("  Deviation across scales: {:.2e}", inv.deviation_across_scales);
    println!("  Relative deviation: {:.2e}", inv.relative_deviation());
    println!("  Is conserved (threshold 0.01)? {}\n", inv.is_conserved(0.01));

    assert!(inv.is_conserved(0.01), "Mean should be approximately invariant");

    // Also show a violated invariant
    let mut map2 = AgentScaleMap::new((0..1024).map(|i| (i as f64 * 0.1).sin()).collect());
    map2.extract_variance("var");
    for _ in 0..3 {
        map2.coarse_grain(4).unwrap();
        map2.extract_variance("var");
    }
    let var_obs = map2.observables.iter().find(|o| o.name == "var").unwrap();
    let var_inv = ScaleInvariant::from_scale_values("variance", &var_obs.values);
    println!("  Observable: variance");
    println!("  Values across scales: {:?}", var_obs.values.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>());
    println!("  Is conserved? {} (variance is NOT scale-invariant)\n", var_inv.is_conserved(0.1));
}

// ─────────────────────────────────────────────────────────────────────────────
// Lesson 7: Full Pipeline — Agent Behavior from Ticks to Topology
// ─────────────────────────────────────────────────────────────────────────────
//
// The full pipeline for analyzing agent behavior:
//
//   1. Collect tick-level behavior data
//   2. Extract observables (mean, variance) at each scale
//   3. Classify observables as relevant/irrelevant/marginal
//   4. Find scale invariants
//   5. Identify universality classes
//
// Two agents with the SAME macroscopic observables but DIFFERENT tick-level
// behavior belong to the same universality class.

fn lesson7_full_pipeline() {
    println!("═ Lesson 7: Full Pipeline — Agent Universality Classes ═\n");

    // Agent A: sinusoidal behavior
    let ticks_a: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.1).sin()).collect();
    // Agent B: noisy sinusoidal behavior (different micro, same macro)
    let ticks_b: Vec<f64> = (0..1024).map(|i| {
        (i as f64 * 0.1).sin() + 0.01 * ((i as f64 * 17.3).sin())
    }).collect();

    let mut map_a = AgentScaleMap::new(ticks_a);
    let mut map_b = AgentScaleMap::new(ticks_b);

    map_a.extract_mean("mean");
    map_b.extract_mean("mean");

    for _ in 0..4 {
        map_a.coarse_grain(4).unwrap();
        map_b.coarse_grain(4).unwrap();
        map_a.extract_mean("mean");
        map_b.extract_mean("mean");
    }

    let mean_a = map_a.observables.iter().find(|o| o.name == "mean").unwrap();
    let mean_b = map_b.observables.iter().find(|o| o.name == "mean").unwrap();

    println!("  Agent A mean across scales: {:?}", mean_a.values.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>());
    println!("  Agent B mean across scales: {:?}", mean_b.values.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>());
    println!();

    let inv_a = ScaleInvariant::from_scale_values("mean_a", &mean_a.values);
    let inv_b = ScaleInvariant::from_scale_values("mean_b", &mean_b.values);

    println!("  Agent A invariant: {:.6} (dev: {:.2e})", inv_a.value, inv_a.relative_deviation());
    println!("  Agent B invariant: {:.6} (dev: {:.2e})", inv_b.value, inv_b.relative_deviation());

    let same_class = (inv_a.value - inv_b.value).abs() < 0.1;
    println!("  Same universality class? {} (invariant diff: {:.2e})\n",
             same_class, (inv_a.value - inv_b.value).abs());

    assert!(same_class, "Agents with same macro behavior should be in same class");
}

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║   Renormalization Agent — Interactive Tutorial               ║");
    println!("║   Multi-scale analysis for agent behavior                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    lesson1_block_averaging();
    lesson2_decimation();
    lesson3_observable_relevance();
    lesson4_rg_flow_fixed_points();
    lesson5_nontrivial_fixed_point();
    lesson6_scale_invariants();
    lesson7_full_pipeline();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   All lessons passed! ✓                                      ║");
    println!("║                                                              ║");
    println!("║   Key takeaways:                                             ║");
    println!("║   1. Block averaging preserves mean, kills variance          ║");
    println!("║   2. Observables: relevant > irrelevant > marginal           ║");
    println!("║   3. RG flow: β(λ) = 0 at fixed points                      ║");
    println!("║   4. Scale invariants survive coarse-graining                ║");
    println!("║   5. Different micro → same macro = universality class       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
}
