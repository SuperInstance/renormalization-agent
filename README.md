# renormalization-agent

[![crates.io](https://img.shields.io/crates/v/renormalization-agent.svg)](https://crates.io/crates/renormalization-agent)
[![docs.rs](https://docs.rs/renormalization-agent/badge.svg)](https://docs.rs/renormalization-agent)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Renormalization group tools for multi-scale agent behavior analysis.**

The renormalization group (RG) coarse-grains systems by integrating out short-scale
degrees of freedom. Applied to agents, this means zooming out from individual tick
data to hourly to daily behavior. Each coarse-graining step preserves relevant
observables while flowing irrelevant ones toward fixed points.

Use this crate to analyze agent behavior at multiple timescales, classify which
behavioral patterns are scale-invariant (relevant) vs. transient (irrelevant),
and detect when agent dynamics undergo phase transitions.

## Features

- **Scale transforms** — `ScaleTransform` with block averaging, decimation, and
  Haar wavelet-based coarse-graining
- **RG flow** — `RGFlow` with beta functions, Euler integration, fixed-point
  finding (Newton's method), and divergence detection
- **Observable tracking** — `Observable` tracks quantities across scales with
  automatic relevance classification (`Relevant`, `Irrelevant`, `Marginal`)
- **Scale invariants** — `ScaleInvariant` identifies quantities conserved across
  all coarse-graining levels; `find_violations()` flags broken invariants
- **Agent scale maps** — `AgentScaleMap` applies successive transforms to agent
  tick data, extracting mean/variance/total at each scale level
- **Multi-level analysis** — chain multiple coarse-graining steps (e.g.,
  4096 ticks → 1024 → 256 → 64) and observe how statistics evolve

## Quick Start

```rust
use renormalization_agent::agent_scale::AgentScaleMap;

// Raw agent tick data
let ticks: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.1).sin()).collect();

let mut map = AgentScaleMap::new(ticks);

// Extract observables at the finest scale
map.extract_mean("mean");
map.extract_variance("var");

// Coarse-grain by factor of 4: 1024 → 256
map.coarse_grain(4).unwrap();
map.extract_mean("mean");
map.extract_variance("var");

// Coarse-grain again: 256 → 64
map.coarse_grain(4).unwrap();
map.extract_mean("mean");
map.extract_variance("var");

// Check scale invariants
let invariants = map.find_invariants(0.05);
for inv in &invariants {
    println!("{}: value={:.4}, deviation={:.4}", inv.name, inv.value, inv.relative_deviation());
}
```

## RG Flow Analysis

```rust
use renormalization_agent::RGFlow;

// Define coupling constants with beta functions
let mut flow = RGFlow::new(
    vec![1.0],                                    // initial couplings
    vec![Box::new(|l: &[f64]| -l[0])],           // β(λ) = -λ (irrelevant)
).unwrap();

let trajectory = flow.integrate_euler(0.1, 100).unwrap();
let fixed_point = flow.find_fixed_point(&[0.5], 1e-8, 100).unwrap();
```

## Module Overview

| Module | Description |
|---|---|
| `scale` | `ScaleTransform` — block average, decimation, wavelet |
| `flow` | `RGFlow` — beta functions, Euler integration, fixed points |
| `observable` | `Observable`, `Relevance` — cross-scale quantity tracking |
| `invariants` | `ScaleInvariant`, `find_violations()` — conserved quantities |
| `agent_scale` | `AgentScaleMap` — multi-level agent data analysis |
| `error` | `RgError` — error types |

## Links

- [Documentation](https://docs.rs/renormalization-agent)
- [Repository](https://github.com/nightshift-crates/renormalization-agent)
- [Crates.io](https://crates.io/crates/renormalization-agent)

## License

MIT
