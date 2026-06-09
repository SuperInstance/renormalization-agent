//! Basic RG flow and coarse-graining.
//!
//! Run with: cargo run --example basic

use renormalization_agent::{AgentScaleMap, RGFlow, ScaleInvariant};

fn main() {
    // --- Coarse-Graining Pipeline ---
    println!("=== Coarse-Graining Agent Behavior ===\n");

    let ticks: Vec<f64> = (0..1024).map(|i| 5.0 + 0.1 * (i as f64 * 0.37).sin()).collect();
    println!("Initial: {} tick samples", ticks.len());

    let mut map = AgentScaleMap::new(ticks);
    map.extract_mean("mean");
    map.extract_variance("var");

    // Coarse-grain through 4 levels
    for level in 0..4 {
        map.coarse_grain(4).unwrap();
        map.extract_mean("mean");
        map.extract_variance("var");
        println!("Level {}: {} samples", level + 1, map.current_data_len());
    }

    // Results
    let mean_obs = map.observables.iter().find(|o| o.name == "mean").unwrap();
    let var_obs = map.observables.iter().find(|o| o.name == "var").unwrap();

    println!("\nMean across scales: {:?}", mean_obs.values);
    println!("Variance across scales:");
    for (i, v) in var_obs.values.iter().enumerate() {
        println!("  Scale {i}: {v:.6}");
    }

    // --- RG Flow ---
    println!("\n=== RG Flow ===\n");

    let mut flow = RGFlow::new(
        vec![1.0],
        vec![Box::new(|l: &[f64]| -l[0])],  // β = -λ → flows to 0
    ).unwrap();

    let trajectory = flow.integrate_euler(0.1, 20).unwrap();
    println!("Coupling λ under β = -λ:");
    for (step, c) in trajectory.iter().enumerate() {
        println!("  t={step}: λ = {:.4}", c[0]);
    }

    // Fixed point
    let fp = flow.find_fixed_point(&[0.5], 1e-8, 100).unwrap();
    println!("\nFixed point: λ* = {:.6}", fp[0]);

    // --- Scale Invariants ---
    println!("\n=== Scale Invariants ===\n");

    let inv = ScaleInvariant::from_scale_values("mean", &mean_obs.values);
    println!("Mean invariant:");
    println!("  Value: {:.4}", inv.value);
    println!("  Deviation: {:.6}", inv.deviation_across_scales);
    println!("  Conserved: {}", inv.is_conserved(0.1));
}
