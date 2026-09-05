# Nude-quad multi-GPU monolith

Single-file implementation of inhomogeneous spin–cavity **cumulant** dynamics
(`src/monolith_mgpu.jl`), with settings, tests, and a CLI. Physics matches the
`nude-quad` package (tip ~`42f1d3a`). There is **no** Volkov–Zon path; do not
use `volkov_zon.jl` as an oracle.

## How to run

```bash
julia --project=. scripts/nude_quad_monolith.jl --settings examples/monolith_forward.jl
julia --project=. scripts/nude_quad_monolith.jl -s examples/monolith_forward_bspline.jl
julia --project=. scripts/nude_quad_monolith.jl -s examples/monolith_order2.jl
julia --project=. scripts/nude_quad_monolith.jl -s examples/monolith_optimizer.jl
# override mode
julia --project=. scripts/nude_quad_monolith.jl -s examples/monolith_forward.jl -m order2
```

JSON settings are also accepted (`examples/monolith_forward.json`) if `JSON3` is available.

Tests (CPU, no GPU required):

```bash
julia --project=. test/monolith_mgpu.jl
```

## Modes

| Mode | What it does |
|---|---|
| `forward` | 1st-order ODE with the **raw** `PULSE_CONFIG` drive |
| `forward-bspline` | Fit raw pulse → `k` B-spline sub-pulses, then 1st-order |
| `order2` | 2nd-order (plus 1st-order fields in the same state) with the raw pulse |
| `order2-bspline` | B-spline parameterization, then 2nd-order |
| `optimizer` | Adam on **only** the B-spline parameter vector |

B-spline layout (same as `bspline.jl` + `composite_pulse.jl`):

```
n_params = 3k + k*n_coeff_A + k*n_coeff_f
         = [gap, dur, phi0] × k  +  cA[:,k]  +  cf[:,k]
```

`gap`/`dur`/`cA` are softplus-encoded and scaled (`gap_scale`, `dur_scale`, `amp_scale`).

## Settings schema

A `.jl` file assigns (package spirit):

- `MODE` — one of the modes above
- `SIM_SETTING` — `M_delta`, `M_g`, `Ttotal`, `Nt_save`, `reltol`, `abstol`,
  `initial_condition` (`:ground/:inverted/:equator/:weak/:weak_inverted/:custom`),
  `ensemble_method` (`:auto` default, `:quadrature`, `:histogram`),
  optional `ensemble_M_delta` / `ensemble_M_g`, `saved_file_name`
- `SYSTEM_CONFIG` — `C_ens`, `delta0`, `kappa_e`, `kappa_i`,
  `freq_inhomogeneity`, `g_inhomogeneity`
- `PULSE_CONFIG` — tuple of `:gaussian` / `:wurst` / `:custom` pulses
- `BSPLINE` — `k`, `n_coeff_A`, `n_coeff_f`, `degree`, `taper_frac`
- `OPTIMIZER` — `num_epochs`, `learning_rate`, `patience`, `w_time`, `w_power`,
  `w_tmax`, `target_F`, `I_min`, `kappa_I`, `S_min`, `kappa_S`, `track`
- `COMPUTE` — `backend` (`:auto/:cpu/:gpu`), `integrator` (`:tsit5/:ck45`), `nshards`

JSON uses the same keys under `mode`, `sim`, `system`, `pulse`, `bspline`,
`optimizer`, `compute` (see `examples/monolith_forward.json`).

## Loss (`optimizer`)

Copied from `src/pulse_optimizer2.jl` (`pulse_cost` / `_fidelity_physics_cost`):

```
ss      = 1 − (silencing − target_F)²          # silencing success
fid     = inversion × ss
J_phys  = (1−fid)² + (κ_I/2)[I_min−I]₊² + (κ_S/2)[S_min−ss]₊²
J       = J_phys + w_time (duration/T_max)
               + w_tmax (max(t_end−T_max,0)/T_max)²
               + w_power mean(|cA|/amp_scale)²
```

Defaults: `I_min=S_min=0.85`, `κ_I=κ_S=50`, `w_time=0.15`, `w_power=0.05`,
`w_tmax=1`, `track=:weak` (one `:weak` trajectory supplies both inversion and
silencing). Gradients are a **discrete Tsit5 adjoint / VJP** along the recorded
mesh (analytic `rhs1_vjp!`); ForwardDiff is used only for `E(t; θ)` and the
cheap direct penalties, not through the full trajectory.

## Ensemble

`:auto` (and explicit `:quadrature`) uses Gauss–Legendre quadrature whenever
both distributions are quad-friendly:

- frequency: `:lorentzian` (tan map) or `:gaussian`
- coupling: `:constant`, `:gaussian`, or `:powerlaw_g`

Otherwise histogram. **Order-2 uses the same rule** — it does not silently
fall back to histogram when the ensemble is quad-friendly.

### Modeling choices

- **κₜ = κₑ + κᵢ** everywhere the main package uses total cavity loss.
- **Lorentzian truncation**: nodes live on `δ = γ tan(θ)` with
  `|θ| ≤ atan(span_gamma)`. Default `renormalize=false` keeps the truncated
  mass; cooperativity still uses the analytic `N = C_ens κₜ FWHM / (4⟨g²⟩)`
  (Lorentzian) or the Gaussian analogue with `√(π ln 2)`.
- **Coupling renormalize**: default `true` (package `coupling_inhomogeneity.jl`).

## Correctness (fixes vs nude-quad ICs)

Ground + vacuum must be a **fixed point** of the 2nd-order RHS:

- `⟨S⁻S⁺⟩_same = Nⱼ` (or `Nⱼ/2 − ⟨Sᶻ⟩`), **not** `0`
- Product-state same-bin algebra for non-ground ICs:
  - `⟨S⁻S⁺⟩ = |Sp|²(1−1/N) + N/2 − Sz`
  - `⟨SᶻSᶻ⟩ = Sz²(1−1/N) + N/4`

See section comments in `src/monolith_mgpu.jl` (eqs. 1–5).

## Multi-GPU

1st-order shards ensemble bins across devices. Each RHS:

1. On-device local `S = Σ gⱼ Sⱼ⁻`
2. **NCCL Allreduce (sum)** of that scalar (or P2P device copies)
3. Fused `ȧ / ⟨Ṡ⁺⟩ / ⟨Ṡᶻ⟩` on device

There is **no** host-staged row-sum exchange on the hot path. NCCL Allgather
is used for 2nd-order O(M) row-sums when multiple GPUs are present; host
staging is only a last-resort fallback and is logged.

Integrators: Tsit5 (default, dense-ish stages, adjoint) or CK45 (low-storage)
with **pooled stage buffers**. Backend `:auto` uses GPU when CUDA is
functional and `M` is large enough.

## Files

| Path | Role |
|---|---|
| `src/monolith_mgpu.jl` | The monolith (physics, solvers, adjoint, CLI) |
| `scripts/nude_quad_monolith.jl` | Thin CLI wrapper |
| `examples/monolith_*.jl` | Settings for each mode |
| `examples/monolith_forward.json` | JSON schema example |
| `test/monolith_mgpu.jl` | Fixed-point, RHS, B-spline tests |
