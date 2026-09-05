# Requirements → code

All implementations live in `src/SpinCavityMonolith.jl` unless noted.

| Requirement | Where |
|---|---|
| Settings file: cavity + ensemble + pulse | `load_settings` / `load_settings_jl` / `load_settings_json`; schema in §2 comments. Examples: `examples/monolith_*.jl`. Cavity+ensemble: `SYSTEM_CONFIG` + `SIM_SETTING`. Pulse: `PULSE_CONFIG`. |
| `prepare(ensemble_method=…)` | `prepare`, `prepare_derived`, `resolve_ensemble_method`, `ensemble_method_for` |
| Mode `forward` — 1st-order, raw pulse | `forward` → `solve_1st_order` + `build_E_of_t_raw` |
| Mode `forward_bspline` | `forward_bspline` → `fit_raw_pulse_bspline` + `build_E_of_t` + `solve_1st_order` |
| Mode `order2` — 1st+2nd-order fields | `order2` → `solve_2nd_order` (state holds cavity, 1st-order spins, 2nd-order correlators) |
| Mode `order2_bspline` | `order2_bspline` (same solve, B-spline drive) |
| Mode `optimizer` — GD on B-spline params only | `optimize` / `optimize_bspline!` (Adam on `u` of length `n_params`) |
| B-spline math/layout `n_params=3k+k nA+k nf` | `CompositePulse`, `n_params`, `pack`/`unpack`/`decode`, `build_E_of_t`, Gevrey `_taper_window` |
| Loss (fidelity physics + time/power) | `_fidelity_physics_cost`, `pulse_cost_theta` |
| Quadrature if Δ/g quad-friendly, else histogram | `ensemble_method_for`, `resolve_ensemble_method`; used by `prepare` for **all** modes including order2 |
| Discrete adjoint as primary gradient | `pulse_cost_grad_adjoint`, `rhs1_vjp!`, `tsit5_step_vjp!`, `reverse_tsit5_checkpoints!`. `optimize(..., grad=:adjoint)` default. `grad=:forward` is Dual-through-solve, parity only. |
| Product-state ICs; ground `SmSp=Nⱼ` | `smsp_same_product` et al.; `build_u0_1st_order`, `build_u0_2nd_order`, `build_u0_2nd_mgpu` |
| `κₜ = κₑ+κᵢ` | `prepare_derived` (`kappa_t`); `rhs1!`; `rhs2_small!` |
| Auditable equations | Header of `src/SpinCavityMonolith.jl` (eqs. for ȧ, spins, same-bin ICs, loss) |
| Live multi-GPU, no dense duplicate stack | `rhs2_sharded!` is the RHS. Small once; large on column shards. `rhs2!` only packs shards for tests/reporting — it calls `rhs2_sharded!`. |
| ≥2-GPU NCCL/P2P is the bar | `build_collectives` → `:nccl` (preferred) or `:p2p`. This VM has **no NVIDIA GPU**; CPU path is documented, not a substitute for live NCCL. |
| On-device small RHS + rowsums (no hot-path D2H) | `_rowsum_kernel!` + `allreduce_sum_group!` + `_small_rhs_kernel!` / `_cuda_launch_small!` in `solve_2nd_gpu`. Shared math: `_small_cavity_derivs!`, `_small_bin_deriv!`. |
| NCCL Allreduce **group**, not a naive loop | `allreduce_sum_group!` wraps ranks (and `sumP`/`sumM`/`sumZ`) in `NCCL.group` / `groupStart`+`groupEnd`. Refuses a serial per-rank Allreduce if the group API is missing. |
| Host collective fallback is loud | `_host_collective_error` `@error`s and `error`s. Never returns `:host` as a live multi-GPU path. |
| Homemade GL; FastGaussQuadrature later | `_gauss_legendre_pts` (Golub–Welsch). FastGaussQuadrature.jl is an optional later dependency, not wired in. |
| CLI | `scripts/run_monolith.jl` → `main` / `run_mode` |

## Rename map (this tip)

| Old | New |
|---|---|
| module / package `NudeQuadMonolith` | `SpinCavityMonolith` |
| uuid `d5b32329-…` | `230e6bac-f383-4415-ba98-aaf7da95e674` |
| authors “Qove Laboratory” | removed |
| `src/monolith_mgpu.jl` | `src/SpinCavityMonolith.jl` |
| `scripts/nude_quad_monolith.jl` | `scripts/run_monolith.jl` |
| `test/monolith_mgpu.jl` | `test/spin_cavity_monolith.jl` |
