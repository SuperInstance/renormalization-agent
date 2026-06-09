# renormalization-agent

**Renormalization group for agent systems — zoom out and see what matters.**

When you look at agent behavior tick-by-tick, it's noisy and complex. Zoom out to hourly summaries, and patterns emerge. Zoom out further to daily summaries, and the noise vanishes — only the essential dynamics remain. This is the **renormalization group (RG)**, the most powerful idea in theoretical physics, applied to multi-scale agent analysis.

## The Key Idea

The RG works by **coarse-graining**: replacing fine-grained data with block averages (or wavelets, or decimation). At each scale, you track **observables** (mean, variance, total). Under coarse-graining:

- **Relevant observables** grow — they dominate at large scales
- **Irrelevant observables** shrink — they vanish at large scales
- **Marginal observables** stay constant — they're scale-invariant

The RG **flow** traces how coupling constants evolve across scales. Fixed points of the flow are **universality classes** — different agents that look the same when you zoom out.

## Architecture

```
Tick Data ──→ AgentScaleMap ──→ Observables ──→ Scale Invariants
                  │                  │
           coarse_grain()     classify()
           decimate()         (relevant /
           wavelet()          irrelevant /
                              marginal)
                  │
                  ▼
              RGFlow ──→ Fixed Points ──→ Universality Classes
             (β functions)
```

## Quick Start

```rust
use renormalization_agent::AgentScaleMap;

// Agent tick data (e.g., activity level over 4096 ticks)
let ticks: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.1).sin()).collect();

let mut map = AgentScaleMap::new(ticks);

// Extract observables at the finest scale
map.extract_mean("mean");
map.extract_variance("var");

// Coarse-grain: 1024 → 256 → 64 → 16 → 4
for _ in 0..4 {
    map.coarse_grain(4).unwrap();
    map.extract_mean("mean");
    map.extract_variance("var");
}

// Mean should be approximately scale-invariant
let mean_obs = map.observables.iter().find(|o| o.name == "mean").unwrap();
println!("Mean across scales: {:?}", mean_obs.values);

// Variance should decrease (irrelevant under RG)
let var_obs = map.observables.iter().find(|o| o.name == "var").unwrap();
println!("Variance across scales: {:?}", var_obs.values);
```

## API Walkthrough

### Scale Transforms

Three coarse-graining methods:

```rust
let mut map = AgentScaleMap::new(data);

// Block averaging: replace blocks of N with their mean
// Best for: preserving mean, reducing noise
map.coarse_grain(4).unwrap();

// Decimation: keep every Nth sample
// Best for: temporal subsampling
map.decimate(5).unwrap();

// Wavelet: Haar-like approximation coefficients
// Best for: edge detection in behavior
map.wavelet_coarse_grain(2).unwrap();
```

### Observables

```rust
// Built-in extractions
map.extract_mean("mean");         // Scale dimension 0
map.extract_variance("var");      // Scale dimension -2
map.extract_total("total");       // Scale dimension 1

// Classify relevance
let obs = map.observables.iter_mut().find(|o| o.name == "mean").unwrap();
obs.classify(2.0);  // threshold: ratio > 2 → relevant
println!("{:?}", obs.relevance);  // Relevant, Irrelevant, or Marginal
```

### RG Flow

```rust
use renormalization_agent::{RGFlow, RgError};

// Define beta functions: dλ/dt = β(λ)
let mut flow = RGFlow::new(
    vec![1.0],  // initial coupling
    vec![Box::new(|l: &[f64]| -l[0])],  // β = -λ (flows to zero)
)?;

// Integrate the flow
let trajectory = flow.integrate_euler(0.01, 100)?;
for (step, couplings) in trajectory.iter().enumerate() {
    println!("Step {step}: λ = {:.4}", couplings[0]);
}

// Find fixed points: β(λ*) = 0
let fp = flow.find_fixed_point(&[0.5], 1e-8, 100)?;
println!("Fixed point: λ* = {:.4}", fp[0]);

// Classify perturbations
let relevance = flow.classify_perturbation(&fp, &[1.0], 0.1);
println!("Perturbation: {:?}", relevance);
```

### Scale Invariants

```rust
use renormalization_agent::{ScaleInvariant, find_violations};

let inv = ScaleInvariant::from_scale_values("energy", &[42.0, 42.0, 42.0]);
println!("Conserved: {}", inv.is_conserved(0.01));
println!("Relative deviation: {:.4}", inv.relative_deviation());

// Check a collection
let invariants = vec![
    ScaleInvariant::from_scale_values("good", &[1.0, 1.0, 1.0]),
    ScaleInvariant::from_scale_values("bad", &[1.0, 100.0, 10000.0]),
];
let violations = find_violations(&invariants, 0.5);
```

## Performance

- **Block averaging**: O(n) — linear in data size
- **RG flow integration**: O(steps × couplings²) — Newton's method for fixed points
- **Observable extraction**: O(n) per observable per scale

Suitable for tick datasets up to ~100K points. For larger datasets, pre-aggregate.

## Ecosystem

Part of the **SuperInstance** family:
- `hodge-consensus` — Which disputes will resolve
- `persistence-agent` — Which behavioral patterns are signal vs noise
- `cosmic-web` — Fleet architecture as large-scale cosmic structure
- `constraint-hamiltonian` — Constraint dynamics with symplectic guarantees
- `renormalization-agent` — Multi-scale agent behavior analysis

## License

MIT
