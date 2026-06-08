# renormalization-agent

[![crates.io](https://img.shields.io/crates/v/renormalization-agent.svg)](https://crates.io/crates/renormalization-agent)
[![docs.rs](https://docs.rs/renormalization-agent/badge.svg)](https://docs.rs/renormalization-agent)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## The Idea

In statistical physics, the **renormalization group (RG)** describes how a system's behavior changes as you zoom out. At fine scales, every detail matters. At coarse scales, most details wash out and only a few "relevant" parameters control the physics.

Agent behavior is the same. An agent's minute-by-minute actions are noisy and complex. But zoom out to hourly, daily, or weekly patterns, and the noise disappears — you see the agent's "fixed points": stable behavioral modes that persist across scales.

`renormalization-agent` applies RG-style coarse-graining to agent behavior data. It identifies which behavioral parameters are **relevant** (grow under coarse-graining), **irrelevant** (shrink and disappear), or **marginal** (persist but don't grow). Relevant parameters define the agent's core personality. Irrelevant parameters are noise.

## How It Works

### 1. Define behavioral observables

An observable is any measurable aspect of agent behavior: response latency, task completion rate, creativity score, cooperation frequency, etc.

```rust
use renormalization_agent::{Observable, AgentBehavior};

let behavior = AgentBehavior::new(vec![
    Observable::new("response_time", values),
    Observable::new("cooperation_rate", values),
    Observable::new("creativity_score", values),
]);
```

### 2. Coarse-grain across scales

The RG works by **block transformation**: group adjacent data points, average them, and see how the statistics change.

```rust
use renormalization_agent::ScaleTransform;

// Block averaging: group pairs, quadruples, octuples...
let scales = ScaleTransform::block_averaging(&behavior, /* max_scale */ 5);
for (scale, obs) in &scales {
    println!("Scale {}: mean={:.3} std={:.3}", scale, obs.mean(), obs.std());
}
```

Alternative coarse-graining methods:
- **Decimation**: keep every kth point (fast, but aliasing)
- **Haar wavelet**: separate into approximation + detail at each scale

### 3. Compute beta functions

The **beta function** β(g) = dg/d(ln s) tells you how a parameter g changes with scale s:
- β > 0: parameter **grows** under coarse-graining → *relevant* (defines personality)
- β < 0: parameter **shrinks** → *irrelevant* (noise)
- β ≈ 0: parameter is **marginal** (borderline)

```rust
use renormalization_agent::RGFlow;

let flow = RGFlow::compute(&scales);
for param in &flow.parameters {
    println!("{}: beta={:.4} → {:?}", param.name, param.beta, param.classification);
    // Output: "cooperation_rate: beta=0.234 → Relevant"
    // Output: "noise_jitter: beta=-1.502 → Irrelevant"
}
```

### 4. Find fixed points

A **fixed point** is a behavioral state that's invariant under coarse-graining — the agent looks the same at all scales. These are the agent's stable personality modes.

```rust
let fixed_points = flow.find_fixed_points();
for fp in &fixed_points {
    println!("Fixed point at {:?} (stable: {})", fp.values, fp.is_stable);
}
```

Stability is determined by the Jacobian eigenvalues at the fixed point: all negative = stable attractor, any positive = unstable (saddle or repeller).

## The Physics Analogy

| Physics RG | Agent RG |
|---|---|
| Coupling constants | Behavioral parameters (cooperation, creativity, etc.) |
| Length scale | Time scale (minutes → hours → days) |
| Beta function β(g) | How parameter changes when zooming out |
| Fixed point | Stable behavioral mode (personality type) |
| Relevant operator | Core personality trait |
| Irrelevant operator | Behavioral noise |
| Critical point | Phase transition in agent behavior |

## Module Map

| Module | What it does |
|---|---|
| `observable` | `Observable` — measurable behavioral quantity with statistics |
| `scale` | `ScaleTransform` — block averaging, decimation, Haar wavelet |
| `flow` | `RGFlow` — beta functions, perturbation classification (relevant/irrelevant/marginal) |
| `fixed_point` | Fixed point finder with Newton's method + Jacobian stability analysis |
| `coarse_grain` | `AgentBehavior` — multi-observable container with scale transforms |
| `error` | `RGError` |

## When To Use This

- **Agent profiling**: extract core personality from noisy behavioral data
- **Anomaly detection**: if an agent's relevant parameters change, something fundamental shifted
- **Multi-agent comparison**: two agents with different fixed points have genuinely different "personalities" (not just noise)
- **Behavioral prediction**: near a fixed point, behavior is predictable. Near a phase transition, it's not.

## Links

- [Documentation](https://docs.rs/renormalization-agent)
- [Repository](https://github.com/SuperInstance/renormalization-agent)
- [crates.io](https://crates.io/crates/renormalization-agent)
- Wilson & Kogut (1974) — The renormalization group and the ε-expansion

## License

MIT
