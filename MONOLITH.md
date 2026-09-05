# SpinCavityMonolith

Single-file solver (`src/SpinCavityMonolith.jl`). Settings describe cavity,
ensemble, and drive. Modes run 1st-order, 2nd-order, B-spline parameterization,
or Adam on B-spline parameters.

## Tree

```
src/SpinCavityMonolith.jl
scripts/run_monolith.jl
examples/monolith_*.jl|.json
test/spin_cavity_monolith.jl
test/runtests.jl
MONOLITH.md  REQUIREMENTS.md  README.md  LICENSE  Project.toml  .gitignore
```

## Modes

```
settings → prepare(ensemble_method=:auto|:quadrature|:histogram)
         → forward / forward_bspline(u) / order2 / order2_bspline(u)
         → optimize(u; grad=:adjoint|:forward)
```

| Mode | What it does |
|---|---|
| `forward` | 1st-order ODE, raw `PULSE_CONFIG` |
| `forward_bspline` | Fit raw pulse → `k` B-spline sub-pulses, then 1st-order |
| `order2` | 2nd-order (includes 1st-order fields), raw pulse |
| `order2_bspline` | B-spline parameterization, then 2nd-order |
| `optimizer` | Adam on the B-spline vector only (`grad=:adjoint` default) |

Hyphenated aliases (`forward-bspline`, `order2-bspline`) are accepted.

B-spline: `n_params = 3k + k n_coeff_A + k n_coeff_f`, unpack `gaps|durs|φ0|cA|cf`,
softplus scales, Gevrey taper. `build_E_of_t` is in this file.

## Settings

A `.jl` file assigns `MODE`, `SIM_SETTING`, `SYSTEM_CONFIG`, `PULSE_CONFIG`,
and optionally `BSPLINE`, `OPTIMIZER`, `COMPUTE`. JSON uses the same keys
(`examples/monolith_forward.json`).

- **Cavity:** `delta0`, `kappa_e`, `kappa_i` (`κₜ = κₑ+κᵢ`)
- **Ensemble:** `C_ens`, `freq_inhomogeneity`, `g_inhomogeneity`, `M_delta`, `M_g`
- **Pulse:** tuple of `:gaussian` / `:wurst` / `:custom`

`:auto` uses Gauss–Legendre when frequency is `:lorentzian`/`:gaussian` and
coupling is `:constant`/`:gaussian`/`:powerlaw_g`; otherwise histogram.
**order2 uses the same prepare path.** Homemade Golub–Welsch GL is the
implementation; FastGaussQuadrature.jl is an optional later swap for the
same nodes/weights, not required now.

## Loss

```
ss = 1 − (silencing − target_F)²
fid = inversion × ss
J = (1−fid)² + (κ_I/2)[I_min−I]₊² + (κ_S/2)[S_min−ss]₊²
  + w_time (T/T_max) + w_tmax (tmax_excess/T_max)²
  + w_power (‖cA/amp_scale‖² / n_cA)
```

Defaults: `I_min=S_min=0.85`, `κ_I=κ_S=50`, `w_time=0.15`, `w_power=0.05`,
`w_tmax=1`, `track=:weak`. Primary gradient: checkpointed discrete Tsit5 adjoint.

## ICs and equations

Auditable at the top of `src/SpinCavityMonolith.jl`:

```
ȧ = √κₑ E − i δ₀ a − i Σ gⱼ Sⱼ⁻ − (κₜ/2) a
SmSp_same = |Sp|²(1−1/N)+N/2−Sz     # ground ⇒ Nⱼ
SzSz_same = Sz²(1−1/N)+N/4
SpSp_same = Sp²(1−1/N)
SzSp_same = Sz·Sp·(1−1/N)
cross j≠k = mean products
```

## CPU Threads (this tip)

Production path until Tuesday iron: **CPU multicore**, not fake GPUs.

- `nshards` on CPU is a cache partition (`resolve_cpu_nshards`, default **1**
  contiguous large buffer). It is **not** `gpu_count()`.
- `rhs1!` / `rhs2_sharded!` use `Threads.@threads` over bins/columns when
  `nthreads() > 1` and the work is large enough. Per-thread row-sum
  buffers avoid false sharing.
- `RHS2Work` preallocates `sumP/M/Z` + thread locals. Tsit5 reuses stage
  buffers; `_axpy_shards!` / `_lincomb_shards!` are scalar loops (no
  temporary Complex arrays).
- Run with `julia -t auto` / `JULIA_NUM_THREADS`.

## Multi-GPU — Tuesday iron gate

**Not claimed done.** Live ≥2-GPU NCCL/P2P proof is deferred to Tuesday
iron. This VM has no NVIDIA GPU (`nvidia-smi` absent). Do not treat a
CPU or single-device run as multi-GPU.

Wired (unproven here): on-device small RHS + row-sums, NCCL Allreduce
*group*, host collectives `error` (`HOST COLLECTIVE FALLBACK`).

```bash
julia --project=. scripts/run_monolith.jl -s examples/monolith_forward.jl
julia --project=. scripts/run_monolith.jl -s examples/monolith_order2.jl
julia --project=. scripts/run_monolith.jl -s examples/monolith_optimizer.jl --grad adjoint
julia --startup-file=no test/spin_cavity_monolith.jl
```
