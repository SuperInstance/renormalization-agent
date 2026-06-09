//! Advanced: custom beta functions, wavelet coarse-graining, and multi-agent comparison.
//!
//! Run with: cargo run --example advanced

use renormalization_agent::*;

fn main() {
    println!("=== Advanced Renormalization Agent ===\n");

    // --- Different Coarse-Graining Methods ---
    println!("1. Coarse-Graining Methods Comparison");
    println!("======================================\n");

    let data: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.1).sin() + 0.5 * (i as f64 * 0.7).sin()).collect();

    let mut block_map = AgentScaleMap::new(data.clone());
    block_map.extract_mean("mean");
    block_map.coarse_grain(4).unwrap();
    block_map.extract_mean("mean");

    let mut dec_map = AgentScaleMap::new(data.clone());
    dec_map.extract_mean("mean");
    dec_map.decimate(4).unwrap();
    dec_map.extract_mean("mean");

    let mut wav_map = AgentScaleMap::new(data);
    wav_map.extract_mean("mean");
    wav_map.wavelet_coarse_grain(4).unwrap();
    wav_map.extract_mean("mean");

    println!("  Method       Original  Coarse-grained");
    println!("  {:12} {:.4}     {:.4}",
        "Block avg", block_map.observables[0].values[0], block_map.observables[0].values[1]);
    println!("  {:12} {:.4}     {:.4}",
        "Decimation", dec_map.observables[0].values[0], dec_map.observables[0].values[1]);
    println!("  {:12} {:.4}     {:.4}",
        "Wavelet", wav_map.observables[0].values[0], wav_map.observables[0].values[1]);

    // --- Multi-Agent Comparison ---
    println!("\n2. Multi-Agent Behavioral Comparison");
    println!("======================================\n");

    let agents: Vec<(&str, Vec<f64>)> = vec![
        ("Steady", (0..1024).map(|i| 5.0 + 0.01 * (i as f64 * 0.1).sin()).collect()),
        ("Noisy", (0..1024).map(|i| 5.0 + 2.0 * (i as f64 * 7.3).sin()).collect()),
        ("Periodic", (0..1024).map(|i| 5.0 + 3.0 * (i as f64 * 0.01).sin()).collect()),
    ];

    println!("  {:12} {:>10} {:>10} {:>12}",
        "Agent", "Var@tick", "Var@64", "Var reduction");
    println!("  {}", "-".repeat(50));

    for (name, ticks) in &agents {
        let mut map = AgentScaleMap::new(ticks.clone());
        map.extract_variance("var");
        for _ in 0..4 {
            map.coarse_grain(4).unwrap();
            map.extract_variance("var");
        }
        let obs = map.observables.iter().find(|o| o.name == "var").unwrap();
        let reduction = obs.values[0] / obs.values[4].max(1e-10);
        println!("  {name:12} {:10.4} {:10.4} {:12.1}×",
            obs.values[0], obs.values[4], reduction);
    }

    // --- Complex RG Flow ---
    println!("\n3. Multi-Parameter RG Flow");
    println!("===========================\n");

    // Two-parameter flow: β₁ = -λ₁ + λ₂, β₂ = -λ₂
    let mut flow = RGFlow::new(
        vec![1.0, 0.5],
        vec![
            Box::new(|l: &[f64]| -l[0] + l[1]),
            Box::new(|l: &[f64]| -l[1]),
        ],
    ).unwrap();

    let trajectory = flow.integrate_euler(0.1, 30).unwrap();
    println!("  {:>4}  {:>8}  {:>8}", "Step", "λ₁", "λ₂");
    for (step, couplings) in trajectory.iter().step_by(5).enumerate() {
        let actual_step = step * 5;
        println!("  {actual_step:4}  {:8.4}  {:8.4}", couplings[0], couplings[1]);
    }

    match flow.find_fixed_point(&[0.1, 0.1], 1e-8, 200) {
        Ok(fp) => {
            println!("\n  Fixed point: λ* = {:?}", fp);
            let cls = flow.classify_perturbation(&fp, &[1.0, 0.0], 0.1);
            println!("  Perturbation along λ₁: {:?}", cls);
        }
        Err(e) => println!("  No convergence: {e}"),
    }

    // --- Divergence Detection ---
    println!("\n4. Divergence Detection");
    println!("========================\n");

    let mut diverging = RGFlow::new(
        vec![0.1],
        vec![Box::new(|l: &[f64]| l[0] * l[0])],  // β = λ² → blows up
    ).unwrap();

    match diverging.integrate_euler(1.0, 1000) {
        Ok(_) => println!("  Unexpected: flow did not diverge"),
        Err(e) => println!("  Divergence detected: {e}"),
    }

    // --- Full Pipeline ---
    println!("\n5. Full Pipeline: Signal + Noise");
    println!("==================================\n");

    let signal: Vec<f64> = (0..4096)
        .map(|i| 3.0 + (i as f64 * 0.001).sin())
        .collect();

    let mut map = AgentScaleMap::new(signal);
    map.extract_mean("mean");
    map.extract_variance("var");

    for _ in 0..6 {
        map.coarse_grain(4).unwrap();
        map.extract_mean("mean");
        map.extract_variance("var");
    }

    let mean_obs = map.observables.iter().find(|o| o.name == "mean").unwrap();
    let var_obs = map.observables.iter().find(|o| o.name == "var").unwrap();

    println!("  Mean across 7 scales:");
    for (i, v) in mean_obs.values.iter().enumerate() {
        let bar: String = "●".repeat((*v * 5.0) as usize);
        println!("    Scale {i}: {v:.4} {bar}");
    }

    println!("\n  Variance across 7 scales:");
    for (i, v) in var_obs.values.iter().enumerate() {
        let bar_len = (*v * 100.0) as usize;
        let bar: String = "█".repeat(bar_len.min(50));
        println!("    Scale {i}: {v:.6} {bar}");
    }
}
