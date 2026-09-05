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
julia --project=. scripts/nude_quad_monolith.jl -s examples/monolith_order2_bspline.jl
julia --project=. scripts/nude_quad_monolith.jl -s examples/monolith_optimizer.jl --grad adjoint
# override mode (underscore or hyphen)
julia --project=. scripts/nude_quad_monolith.jl -s examples/monolith_forward.jl -m order2
```

JSON settings are also accepted (`examples/monolith_forward.json`) if `JSON3` is available.

Tests (CPU, no GPU required):

```bash
julia --project=. test/monolith_mgpu.jl
```

## Modes API

```
settings → prepare(ensemble_method=:auto|:quadrature|:histogram)
         → forward / forward_bspline(u) / order2 / order2_bspline(u)
         → optimize(u; grad=:adjoint|:forward)
```

| Mode | What it does |
|---|---|
| `forward` | 1st-order ODE with the **raw** `PULSE_CONFIG` drive |
| `forward_bspline` | Fit raw pulse → `k` B-spline sub-pulses, then 1st-order |
| `order2` | 2nd-order (plus 1st-order fields) with the raw pulse |
| `order2_bspline` | B-spline parameterization, then 2nd-order |
| `optimizer` | Adam on **only** the B-spline parameter vector |

Hyphenated aliases (`forward-bspline`, `order2-bspline`) are accepted.

B-spline layout (same as `bspline.jl` + `composite_pulse.jl`):

```
n_params = 3k + k*n_coeff_A + k*n_coeff_f
unpack   = gaps | durs | φ0 | cA | cf
```

`gap`/`dur`/`cA` are softplus-encoded and scaled (`gap_scale`, `dur_scale`, `amp_scale`).
`build_E_of_t` applies the Gevrey taper from `composite_pulse.jl`.

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
  `w_tmax`, `target_F`, `I_min`, `kappa_I`, `S_min`, `kappa_S`, `track`,
  `grad` (`:adjoint` default, or `:forward`), `checkpoint_stride`
- `COMPUTE` — `backend` (`:auto/:cpu/:gpu`), `integrator` (`:tsit5/:ck45`), `nshards`

JSON uses the same keys under `mode`, `sim`, `system`, `pulse`, `bspline`,
`optimizer`, `compute` (see `examples/monolith_forward.json`).

## Loss (`optimizer`)

Copied from `src/pulse_optimizer2.jl` (`pulse_cost` / `_fidelity_physics_cost`).
Metrics come from weighted inversion / silencing on `:weak` (or `:dual`) tracks.

```
ss      = 1 − (silencing − target_F)²
fid     = inversion × ss
J_base  = (1−fid)²
J_pen   = (κ_I/2) max(I_min−I,0)² + (κ_S/2) max(S_min−ss,0)²
J       = J_base + J_pen
        + w_time (duration/T_max)
        + w_tmax (max(t_end−T_max,0)/T_max)²
        + w_power (‖cA/amp_scale‖² / n_cA)
```

Defaults: `I_min=S_min=0.85`, `κ_I=κ_S=50`, `w_time=0.15`, `w_power=0.05`,
`w_tmax=1`, `track=:weak`.

## Gradients

**Primary path:** checkpointed discrete Tsit5 adjoint (`pulse_adjoint.jl` +
`tsit5_discrete_adjoint.jl` style). Analytic `rhs1_vjp!`; ForwardDiff only for
`E(t;θ)` and the cheap direct penalties.

`optimize(u; grad=:adjoint)` is the hot path. `grad=:forward` Duals through the
full solve and is kept only for parity tests. Optimizer multi-GPU in
`pulse_optimizer2` is **job fanout** across devices, not RHS sharding.

## Ensemble

`:auto` (and explicit `:quadrature`) uses Gauss–Legendre quadrature whenever
both distributions are quad-friendly:

- frequency: `:lorentzian` (tan map) or `:gaussian`
- coupling: `:constant`, `:gaussian`, or `:powerlaw_g`

Otherwise histogram. **`order2` / `order2_bspline` use the same rule** — they
do not silently fall back to histogram when the ensemble is quad-friendly
(the package `run_simulation` only defaulted `:auto` for 1st-order).

### Modeling choices

- **κₜ = κₑ + κᵢ** everywhere the main package uses total cavity loss.
  Do **not** copy `sim_2nd_multi_gpu_opt.jl`, which drops `κᵢ`.
- **Lorentzian truncation**: nodes live on `δ = γ tan(θ)` with
  `|θ| ≤ atan(span_gamma)`. Default `renormalize=false` keeps the truncated
  mass; cooperativity still uses the analytic `N = C_ens κₜ FWHM / (4⟨g²⟩)`
  (Lorentzian) or the Gaussian analogue with `√(π ln 2)`.
- **Coupling renormalize**: default `true` (package `coupling_inhomogeneity.jl`).

## Correctness (fixes vs nude-quad ICs)

Package 2nd-order ICs set ground `SmSp_same = 0`. That is **wrong**. Ground +
vacuum is a fixed point of the 2nd-order RHS iff the finite-N product-state
closures hold:

```
SmSp_same = |Sp|² (1 − 1/Nⱼ) + Nⱼ/2 − Sz     # ground ⇒ Nⱼ
SzSz_same = Sz² (1 − 1/Nⱼ) + Nⱼ/4
SpSp_same = Sp² (1 − 1/Nⱼ)
SzSp_same = Sz Sp (1 − 1/Nⱼ)
cross j≠k = products of means
```

## Multi-GPU layout (NCCL / P2P, not host `exchange_rowsums!`)

Package `MGPUlayout.jl` target:

- **small** = `3+9M` (`a, a†a†, a†a | Sp, Sz, adSp, adSm, adSz | SpSp_s, SzSp_s, SmSp_s, SzSz_s`).
  Replicate carefully; do **not** re-integrate the small RHS independently on every GPU.
- **large** = `5 × M × mloc` column-contiguous shards:
  `SpSp`, `SzSp`, `SzSpT`, `SmSp`, `SzSz` (`SzSpT` is an explicit transpose).
- Collectives each RHS (on-device):
  - Allreduce(SUM) of O(1) cavity sums
  - Allgather of O(M) row-sums
- Host `exchange_rowsums!` (`MGPUproblem.jl`) is intentionally **not** used.
- **κₜ = κₑ + κᵢ always.**

1st-order shards ensemble bins; each RHS Allreduces `Σ gⱼ Sⱼ⁻` then fuses
`ȧ / ⟨Ṡ⁺⟩ / ⟨Ṡᶻ⟩` on device. Order-2 multi-GPU is **RHS sharding** with
on-device collectives (not optimizer job fanout).

Integrators: Tsit5 (default, adjoint) or CK45 (low-storage) with pooled stage
buffers. Backend `:auto` uses GPU when CUDA is functional and `M` is large enough.

## Files

| Path | Role |
|---|---|
| `src/monolith_mgpu.jl` | The monolith (physics, solvers, adjoint, CLI) |
| `scripts/nude_quad_monolith.jl` | Thin CLI wrapper |
| `examples/monolith_*.jl` | Settings for each mode |
| `examples/monolith_forward.json` | JSON schema example |
| `test/monolith_mgpu.jl` | Fixed-point, RHS, B-spline, IC, adjoint-parity tests |
