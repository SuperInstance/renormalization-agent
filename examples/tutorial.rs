//! Tutorial: step-by-step renormalization group analysis.
//!
//! Run with: cargo run --example tutorial

use renormalization_agent::*;

fn main() {
    println!("=== Renormalization Agent Tutorial ===\n");

    // Step 1: The Problem
    println!("Step 1: The Multi-Scale Problem");
    println!("================================");
    println!("Agent behavior looks different at different scales.");
    println!("Tick-level: noisy. Hourly: smoother. Daily: essential.\n");

    // Step 2: Create tick data
    println!("Step 2: Raw Agent Data");
    println!("=======================");
    let ticks: Vec<f64> = (0..4096)
        .map(|i| {
            let signal = 5.0 + (i as f64 * 0.01).sin();
            let noise = 0.5 * ((i as f64 * 7.3).sin());
            signal + noise
        })
        .collect();
    println!("Generated {} tick samples: signal + noise", ticks.len());
    println!("Range: [{:.2}, {:.2}]", ticks.iter().cloned().fold(f64::INFINITY, f64::min),
        ticks.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

    // Step 3: Build scale map
    println!("\nStep 3: Multi-Scale Analysis");
    println!("=============================");
    let mut map = AgentScaleMap::new(ticks);
    map.extract_mean("mean");
    map.extract_variance("var");
    map.extract_total("total");

    println!("Scale 0: {} samples", map.current_data_len());

    for level in 1..=5 {
        map.coarse_grain(4).unwrap();
        map.extract_mean("mean");
        map.extract_variance("var");
        map.extract_total("total");
        println!("Scale {level}: {} samples (block_size=4)", map.current_data_len());
    }

    // Step 4: Analyze observables
    println!("\nStep 4: Observable Classification");
    println!("==================================");

    for obs in &mut map.observables {
        obs.classify(2.0);
        let relevance = match obs.relevance {
            Relevance::Relevant => "RELEVANT (grows at large scales)",
            Relevance::Irrelevant => "IRRELEVANT (vanishes at large scales)",
            Relevance::Marginal => "MARGINAL (scale-invariant)",
        };
        println!("  {} (dim={}): {}", obs.name, obs.scaling_dimension, relevance);
        println!("    Values: {:?}", obs.values.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>());
    }

    // Step 5: Scale invariants
    println!("\nStep 5: Scale Invariants");
    println!("========================");

    for obs in &map.observables {
        let inv = ScaleInvariant::from_scale_values(&obs.name, &obs.values);
        let conserved = inv.is_conserved(0.1);
        println!("  {}: value={:.4}, deviation={:.6}, conserved={conserved}",
            inv.name, inv.value, inv.deviation_across_scales);
    }

    // Step 6: RG Flow
    println!("\nStep 6: RG Flow (Beta Functions)");
    println!("=================================");
    println!("Beta functions describe how couplings change with scale.");
    println!("dλ/dt = β(λ)\n");

    let examples: Vec<(&str, Box<dyn Fn(&[f64]) -> f64>, Vec<f64>)> = vec![
        ("β = -λ (flow to 0)", Box::new(|l: &[f64]| -l[0]), vec![1.0]),
        ("β = λ (flow away from 0)", Box::new(|l: &[f64]| l[0]), vec![0.5]),
        ("β = λ - λ³ (Wilson-Fisher)", Box::new(|l: &[f64]| l[0] - l[0].powi(3)), vec![0.5]),
    ];

    for (desc, beta, initial) in examples {
        let mut flow = RGFlow::new(initial.clone(), vec![beta]).unwrap();
        let trajectory = flow.integrate_euler(0.05, 20).unwrap();
        let first = trajectory[0][0];
        let last = trajectory.last().unwrap()[0];
        println!("  {desc}");
        println!("    λ: {first:.4} → {last:.4}");
    }

    // Step 7: Fixed points
    println!("\nStep 7: Fixed Points");
    println!("====================");
    println!("Fixed points: β(λ*) = 0. These are universality classes.\n");

    // β = λ - λ³ → fixed points at 0 and ±1
    let flow = RGFlow::new(
        vec![0.5],
        vec![Box::new(|l: &[f64]| l[0] - l[0].powi(3))],
    ).unwrap();

    for guess in [0.1, 0.8, -0.8] {
        match flow.find_fixed_point(&[guess], 1e-8, 200) {
            Ok(fp) => {
                let cls = flow.classify_perturbation(&fp, &[1.0], 0.1);
                let stability = match cls[0] {
                    Relevance::Relevant => "unstable",
                    Relevance::Irrelevant => "stable",
                    Relevance::Marginal => "marginal",
                };
                println!("  From guess {guess:+.1}: λ* = {:+.4} ({stability})", fp[0]);
            }
            Err(e) => println!("  From guess {guess:+.1}: no convergence ({e})"),
        }
    }

    println!("\n=== Tutorial Complete ===");
}
