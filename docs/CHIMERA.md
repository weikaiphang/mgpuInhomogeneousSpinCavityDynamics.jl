# Chimera architecture (package-first rebuild)

Chimera is the **only** production dynamics tree. Public simulation API
lives under `src/chimera/`. The package entry
`src/InhomogeneousSpinCavityDynamics.jl` includes chimera plus the
pulse-optimizer / datagen leftovers that call that API. There is **no**
parallel nude-quad implementation, **no** `src/legacy/`, and **no**
top-level shim that re-exports old filenames.

Physical scope is unchanged: a driven lossy cavity coupled to an
inhomogeneous spin-½ ensemble (Tavis–Cummings / Dicke-type), closed at
first- and second-order cumulants, with frequency and coupling
inhomogeneity, pulses, QRT noise postprocessing, and multi-GPU
integration.

There is **no Volkov–Zon** solver or oracle.

```
                    SIM_SETTING / SYSTEM_CONFIG / PULSE_CONFIG
                                      │
                                      ▼
                         src/chimera/api.jl  (run_simulation)
                                      │
          ┌─────────────┬─────────────┼──────────────┬──────────────┐
          ▼             ▼             ▼              ▼              ▼
   FastGauss      QuantumCumulants  OrdinaryDiffEq  CUDA.jl +     JLD2
   Quadrature     + SecondQuantized  / DiffEq        NCCL.jl
   + QuadGK       Algebra            (SciML)         Allreduce
          │             │             │              │
          ▼             ▼             ▼              ▼
   ensemble bins   1st/2nd EOMs    single-device   multi-GPU
   tan-GL/GL+pdf   small-M derive  Tsit5 path      shard + rowsums
   log-GL          + large-M
                   structured
                   backend
          │             │
          ▼             ▼
   QuantumToolbox  QRT Jacobian
   (tiny Hilbert   = factorized
    oracle)        1st-order EOM
```

## Who owns which layer

| Layer | Package that owns it | Chimera call site | What was deleted |
| --- | --- | --- | --- |
| Ensemble quadrature | [FastGaussQuadrature.jl](https://github.com/JuliaApproximation/FastGaussQuadrature.jl) (Hale & Townsend, *SISC* 2013) + [QuadGK.jl](https://github.com/JuliaMath/QuadGK.jl) | `src/chimera/quadrature.jl` (`fgq_gausslegendre` → `gausslegendre`); histogram bin means still `quadgk` | Homemade Golub–Welsch. Oracle only: `test/oracles/golub_welsch.jl` |
| Cumulant EOMs / operator algebra | [QuantumCumulants.jl](https://github.com/qojulia/QuantumCumulants.jl) (Plankensteiner et al., [Quantum 6, 617 (2022)](https://doi.org/10.22331/q-2022-03-21-617)) + [SecondQuantizedAlgebra](https://github.com/qojulia/SecondQuantizedAlgebra.jl) | `src/chimera/eoms/quantum_cumulants.jl` + `quantum_cumulants_impl.jl` (`derive_tc_meanfield`, `complete`); Hamiltonian in `eoms/hamiltonian.jl` | Hand EOMs as the *source of truth*. Closures in `eoms/closure_*.jl` are the large-M backend of the QC-derived equations |
| Time integration | [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) / OrdinaryDiffEq (Rackauckas & Nie) | `src/chimera/integrate/sciml.jl` (`chimera_solve` / `Tsit5`) | Custom single-device stepper as the supported path |
| Multi-GPU | [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) + [NCCL.jl](https://github.com/JuliaGPU/NCCL.jl) (NVIDIA NCCL) | `exchange_rowsums!` defaults to NCCL Allreduce of the 3M complex row-sums | Host-staged reductions as the design center |
| Small-N oracle | [QuantumToolbox.jl](https://github.com/qutip/QuantumToolbox.jl) (QuTiP-like, Quantum 2025) | `src/chimera/oracle/quantumtoolbox.jl`, `scripts/oracle_quantumtoolbox.jl` | No exact-space check at all. Optional skip if the package is missing |
| I/O | JLD2 | existing trajectory contract | — |
| QRT / noise | Same 1st-order EOMs (QC Hamiltonian) via SciML-style Jacobian action | `eoms/jacobian_1st.jl` → `noise/qrt.jl` | Divergent hand Jacobian. `QRT_CLOSURE_LEVEL = :factorized_first_order_jacobian` is honest |

## Deleted NIH / nude-quad modules

These paths are **gone**. They are not quarantined and are not shims.

| Old path | Why deleted |
| --- | --- |
| `src/legacy/` (`sim_2nd_multi_gpu_opt.jl`, `accel_solver_1st_order.jl`) | Wrong physics (`κᵢ` dropped, inverted IC, host-reduced cavity source). Not inherited by chimera |
| `src/sim_2nd_multi_gpu_opt.jl`, `src/accel_solver_1st_order.jl` | Quarantine stubs that only `error(...)` |
| `src/rhs_*.jl`, `src/state_layout_*.jl`, `src/initial_conditions_*.jl` | Dead `include` redirects to `chimera/eoms/` |
| `src/solver_*.jl` | Dead `include` redirects to `chimera/integrate/` |
| `src/MGPU*.jl` | Dead `include` redirects to `chimera/mgpu/` |
| `src/ensemble.jl`, `src/ensemble_quadrature.jl`, `src/frequency_inhomogeneity.jl`, `src/coupling_inhomogeneity.jl` | Dead `include` redirects to `chimera/{ensemble,quadrature,frequency,coupling}.jl` |
| `src/simulation_api.jl` | Dead `include` of `chimera/api.jl`. `run_simulation` / `SIM_SETTING` live there |
| `src/noise.jl` | Dead `include` of `chimera/noise/qrt.jl` |
| Volkov–Zon | Never restored |

Pulse optimization and PINN datagen were not rewritten (out of scope).
They call the chimera 1st-order closure / SciML path after
`include("chimera/include_all.jl")`.

## Hamiltonian (package-owned)

```
H = Δ₀ a†a + Σⱼ δⱼ Sᶻⱼ + Σⱼ gⱼ (a S⁺ⱼ + a† S⁻ⱼ)
  + i √κₑ (E(t) a† − E(t)* a)
jumps: √κₜ a ,   κₜ = κₑ + κᵢ
Sᶻ = ±1/2 per spin; bin operators scale with Nⱼ
```

QuantumCumulants encodes this with `Destroy` + `Transition`/`IndexedOperator`
(`σ(2,1)=S⁺`, `σ(2,2)=Sᶻ+1/2`). `scripts/derive_qc_eoms.jl` regenerates the
completed 1st- and 2nd-order sets for M=1 and the indexed M=2 order-1 set.

A full symbolic O(M²) expand at large M is intractable; that is why the
structured closures exist. They are not a parallel hand theory — they
implement the same operator, and small-M QC + SciML trajectories are the
reference.

## Physics contract preserved from stress-harden

- Product-state ICs; ground `SmSp_same = Nⱼ`; equator/weak algebra
- `diag_mask` (closures match kernels; unused cross diagonal excluded)
- Order-2 `:auto` quadrature default
- `physics_correctness` + rowsum/RHS/IC stress tests
- Public `SIM_SETTING` / `SYSTEM_CONFIG` / `PULSE_CONFIG` / `run_simulation`

## Conventions

- Cavity amplitude damping `-(κₑ+κᵢ)/2`
- Drive `+√κₑ E(t)` on `∂t⟨a⟩`
- `a_out = E − √κₑ ⟨a⟩`
- Lorentzian `N = C_ens κ FWHM / (4⟨g²⟩)`; Gaussian peak-matched with `√(π ln 2)`
- `renormalize` defaults to `false` on frequency bins (truncated mass kept)

## Multi-GPU (one public API)

Production is `src/chimera/mgpu/` (`assemble_problem`, `mgpu_run_simulation`).

`exchange_rowsums!` **defaults to NCCL Allreduce**. CUDA P2P is the fallback
when NCCL cannot be constructed. Host staging is last resort and is not
the documented production path.

## How to test

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test/physics_correctness.jl
julia --project=. test/package_first.jl
julia --project=. test/quantum_cumulants.jl
julia --project=. scripts/derive_qc_eoms.jl
julia --project=. scripts/oracle_quantumtoolbox.jl   # if QuantumToolbox is available
julia --project=. -e 'using Pkg; Pkg.test()'
```
