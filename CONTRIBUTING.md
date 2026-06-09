# Contributing to renormalization-agent

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

```bash
git clone https://github.com/SuperInstance/renormalization-agent
cd renormalization-agent
cargo build
cargo test
```

## Architecture

The crate has 6 modules:

```
scale (coarse-graining transforms)
  ↓
agent_scale (multi-scale map: tick data → observables)
  ↓
observable (track + classify quantities across scales)
  ↓
flow (RG flow: beta functions, Euler integration, fixed points)
  ↓
invariants (scale-invariant quantities)
  ↓
error (unified error type)
```

- **`scale`** — `ScaleTransform`: block averaging, decimation, wavelet
- **`agent_scale`** — `AgentScaleMap`: applies transforms sequentially, extracts observables
- **`observable`** — `Observable`: values across scales with relevance classification
- **`flow`** — `RGFlow`: beta functions, Euler integration, Newton's fixed-point finder
- **`invariants`** — `ScaleInvariant`: quantities conserved across scales
- **`error`** — `RgError`: unified error type for all operations

## Making Changes

### Code Style

- Run `cargo fmt` before committing
- Run `cargo clippy` and fix all warnings
- All public items must have doc comments
- Use `serde::{Serialize, Deserialize}` on all public structs

### Tests

Every PR must:
1. Pass `cargo test` (all existing tests)
2. Add tests for new functionality
3. Maintain the variance-decrease property under coarse-graining

### Key Invariants

- Block averaging of constant data returns the same constant
- Mean should be approximately preserved under block averaging
- Variance should decrease (or stay constant) under block averaging
- RG flow with β = 0 must preserve couplings exactly
- Fixed-point finder must converge for well-behaved beta functions

## Adding New Features

### New Coarse-Graining Methods

Add to `ScaleTransform` in `scale.rs`. Each method needs:
1. A constructor (e.g., `ScaleTransform::gaussian(sigma)`)
2. An apply method (e.g., `gaussian_apply(&self, data: &[f64]) -> Result<Vec<f64>, RgError>`)
3. Integration into `AgentScaleMap` via a new method

### New Observable Extractors

Add to `AgentScaleMap` in `agent_scale.rs`. Follow the pattern:
```rust
pub fn extract_foo(&mut self, name: impl Into<String>) {
    let value = /* compute from self.tick_data */;
    match self.observables.iter_mut().find(|o| o.name == name) {
        Some(obs) => obs.values.push(value),
        None => self.observables.push(Observable::new(name, vec![value], scaling_dim)),
    }
}
```

### New Integration Methods

Add to `RGFlow` in `flow.rs`. Options:
- **RK4** (4th-order Runge-Kutta, not symplectic but more accurate)
- **Leapfrog** (symplectic, 2nd-order)
- **Adaptive step size** (dynamically adjust dt)

## Release Checklist

- [ ] `cargo test` passes
- [ ] `cargo clippy` clean
- [ ] `cargo fmt` applied
- [ ] README.md updated if API changed
- [ ] Version bumped in Cargo.toml

## Questions?

Open an issue at https://github.com/SuperInstance/renormalization-agent/issues
