# Chimera architecture

`chimera` keeps the same physical problem as `nude-quad`: a driven lossy cavity
coupled to an inhomogeneous spin-½ ensemble (Tavis–Cummings / Dicke-type),
closed at first- and second-order cumulants, with frequency and coupling
inhomogeneity, pulses, QRT noise postprocessing, and a multi-GPU integrator.

It does **not** introduce a different physical model. The work is to (1) fix
verified correctness bugs, (2) replace NIH numerics with research-grade
packages where they are the best fit, and (3) move multi-GPU reductions off
the host.

There is **no Volkov–Zon solver** and no analytical Volkov–Zon oracle. That
name does not correspond to a validation method used here.

## Conventions (unchanged)

- Per-spin \(S^z = \pm 1/2\); collective bin operators scale with \(N_j\).
- Cavity amplitude damping \(-(\kappa_e+\kappa_i)/2\).
- Drive \(+\sqrt{\kappa_e}\,E(t)\) on \(\partial_t\langle a\rangle\).
- Output field \(a_\mathrm{out} = E - \sqrt{\kappa_e}\,\langle a\rangle\).
- Cooperativity → spin number:
  - Lorentzian: \(N = C_\mathrm{ens}\,\kappa\,\mathrm{FWHM}/(4\langle g^2\rangle)\)
  - Gaussian (peak-matched): extra \(\sqrt{\pi\ln 2}\) in the denominator.

## Physics fixes

### Same-bin product-state initial conditions

For an uncorrelated product of \(N_j\) spin-½ in one bin,

```
SpSp = Sp² (1 − 1/Nj)
SzSp = Sz Sp (1 − 1/Nj)
SmSp = |Sp|² (1 − 1/Nj) + Nj/2 − Sz
SzSz = |Sz|² (1 − 1/Nj) + Nj/4
```

Cross-bin (\(j\neq k\)) moments factorize. Vacuum cavity / cavity–spin
moments are zero.

Special cases:

| IC | \(\langle S^+\rangle\) | \(\langle S^z\rangle\) | `SmSp_same` |
| --- | --- | --- | --- |
| `:ground` | 0 | \(-N_j/2\) | \(N_j\) |
| `:inverted` | 0 | \(+N_j/2\) | 0 |
| `:equator` | \(N_j/2\) | 0 | \((N_j/2)^2(1-1/N_j)+N_j/2\) |
| `:weak` / `:weak_inverted` | \(\varepsilon N_j/2\) | \(\mp N_j/2\) | full product formula |

`nude-quad` left ground `SmSp_same = 0`, so vacuum+ground was **not** a
fixed point of the 2nd-order RHS (`dadSm` did not cancel). Equator / weak
ICs used raw mean products. Both monolith and MGPU builders now use the
helpers above.

### Other corrections

- Order-2 `run_simulation` / `mgpu_run_simulation` default to
  `ensemble_method = :auto` (quadrature when the distributions support it),
  matching order 1.
- Incomplete order-2 saves truncate to the filled prefix and record
  `n_saved` / `n_requested` instead of writing uninitialized tails.
- Accel `g2_avg` for a truncated Gaussian uses the truncated second moment
  on \([\max(0,\mu-\mathrm{span}\,\sigma),\,\mu+\mathrm{span}\,\sigma]\),
  not the infinite-support \(\mu^2+\sigma^2\).
- `renormalize` defaults to `false`: keep truncated histogram/quadrature
  mass. \(C_\mathrm{ens}\to N\) still uses the analytic FWHM formula, so
  `N_total = N * sum(p)` unless renormalize is enabled.

## QRT postprocessing (intentional approximation)

`src/noise.jl` seeds **connected** second moments from the saved trajectory
(\(n-|\langle a\rangle|^2\), \(\langle a^\dagger S^+\rangle-\langle a^\dagger\rangle\langle S^+\rangle\), …)
and evolves two-time correlators with the **Jacobian of the first-order**
Tavis–Cummings mean-field equations.

That is the standard linearized quantum regression theorem (Gardiner &
Zoller, *Quantum Noise*; Plankensteiner et al., QuantumCumulants.jl,
[Quantum 6, 617 (2022)](https://doi.org/10.22331/q-2022-03-21-617)).
A Jacobian of the full second-order cumulant vector would be a different,
much larger postprocessor. Chimera documents this hybrid and does not
silently replace it.

## Packages chosen

| Package | Role | Why | Replaces |
| --- | --- | --- | --- |
| [FastGaussQuadrature.jl](https://github.com/JuliaApproximation/FastGaussQuadrature.jl) (Hale & Townsend, *SISC* 2013; Glaser–Liu–Rokhlin) | Gauss–Legendre nodes for tan-GL, GL+pdf, log-GL ensembles | Peer-reviewed, maintained, more accurate than a dense Golub–Welsch eigen solve | Homemade `_gauss_legendre_pts` (kept as a parity oracle) |
| [QuadGK.jl](https://github.com/JuliaMath/QuadGK.jl) (Gauss–Kronrod; adaptive QUADPACK lineage) | Histogram bin means \(\int x\,p(x)\,dx\) | Already correct; QUADPACK-grade adaptive quadrature | — |
| [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) / OrdinaryDiffEq (Rackauckas & Nie, *J. Open Res. Softw.* 2017) | Single-GPU DiffEq path (`Tsit5`) | Research-standard Julia ODE stack | — |
| [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) (Besard et al., *Advances in Eng. Softw.* 2019) | Device kernels, arrays, streams | Official Julia GPU stack | — |
| [NCCL.jl](https://github.com/JuliaGPU/NCCL.jl) wrapping [NCCL](https://developer.nvidia.com/nccl) | In-library `Allreduce` of O(M) row-sums and error norms | Device-side collectives; no host gather of the cavity source sums | `exchange_rowsums!` host pin + `copyto!` |
| [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) | Lorentzian / Gaussian pdf, cdf | Standard statistical library | — |
| [JLD2.jl](https://github.com/JuliaIO/JLD2.jl) | Trajectory I/O | Existing public save format | — |

## Packages considered and rejected

| Package | Why not |
| --- | --- |
| QuantumCumulants.jl | Generates symbolic cumulant equations; we already have a specialized inhomogeneous TC/Dicke closure. Using it as the runtime would change the state layout and lose the MGPU shard design. Useful as an independent algebra check, not as the integrator. |
| QuantumOptics.jl | Exact Hilbert-space master equations. Exponentially impossible for \(N\sim 10^6\)–\(10^{12}\) spins. Wrong tool for this ensemble. |
| QuTiP / Dynamiqs | Same exact-space limitation. Python FFI would add copies without helping the O(M²) 2nd-order GPU path. |
| JAX / Diffrax / CuPy | Would rewrite the whole stack. No correctness gain over CUDA.jl + OrdinaryDiffEq for this Julia codebase. |
| DiffEqGPU.jl | Ensemble-of-ODEs kernel; our state is one tightly coupled O(M²) system, not many independent IVPs. |
| KernelAbstractions.jl / GPUArrays / Adapt | Portable backends we do not need (CUDA-only by design). The existing kernels are already CUDA.jl. |
| GSL QUADPACK / PETSc / SLEPc / FFTW | Histogram moments are already QuadGK; there is no sparse eigenproblem or spectral PDE here. |
| CUDA-Q | Circuit / QPU programming model, not cavity-QED cumulant PDEs. |
| mpi4py | Process-level MPI. Intra-node multi-GPU is NCCL’s job; we already have NCCL.jl. |

## Multi-GPU design

Logical 2nd-order state: length \(3+9M+4M^2\).

Shard (see `MGPUlayout.jl`):

- `small_length = 3+9M` — **replicated** on every GPU (cavity + all bin means and same-bin moments).
- `large_length = 5 M m_\mathrm{loc}` — **local columns** of `SpSp`, `SzSp`, `SzSpT`, `SmSp`, `SzSz`.
- `SzSpT` is a coalescing transpose copy, **not** an extra degree of freedom
  (`global_state_length = 3+9M+4M²`).
- `EnsemblePartition` assigns contiguous columns.

RHS pipeline:

1. `global_sums_kernel!` — device reduction of \(\sum_j g_j\langle S_j\rangle\) (no host).
2. `cross_rhs_kernel!` — local columns + partial row-sums (device block reduce).
3. `rowsum_finalize_kernel!` — finish per-bin row-sums on device.
4. `exchange_rowsums!` — **NCCL Allreduce** of the \(3M\) complex row-sum
   buffer (reinterpreted as `Float64`). Fallback: CUDA P2P copies, then
   host staging.
5. `small_rhs_kernel!` — duplicated on every GPU using the assembled row-sums.

Single-GPU skips the exchange. Multi-GPU must match single-GPU to solver
tolerances on small tests (same RHS, different reduction tree).

`scripts/bench_rowsum_exchange.jl` times host staging vs NCCL/P2P when
hardware allows, and otherwise records the structural argument: the O(M)
cavity source is no longer copied through a pinned host buffer on the
NCCL path.

## Public API

`SIM_SETTING`, `SYSTEM_CONFIG`, `PULSE_CONFIG`, `run_simulation`, and
`mgpu_run_simulation` are unchanged except that order 2 now defaults to
`:auto` ensembles. Saved `.jld2` trajectories gain `n_saved` /
`n_requested` when the callback is short.

## How to test

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test/physics_correctness.jl
julia --project=. scripts/bench_rowsum_exchange.jl
```

Physics tests cover vacuum+ground RHS ≈ 0, `SmSp_same ≈ Nj`,
`SmSp+2Sz ≈ 0` on ground, product-state equator/weak ICs, quadrature mass,
\(C_\mathrm{ens}\) formulas, layout lengths, and GPU↔CPU RHS parity when
CUDA is functional.

## Out of scope

- Exact-space / QuTiP validation of large ensembles.
- Full 2nd-order QRT Jacobian.
- Frequency power-law in the main `prepare_derived` path (accel solver
  already has Pearson-VII; main API remains Lorentzian/Gaussian).
- Rewriting pulse optimization or PINN datagen.
- Any Volkov–Zon / closed-form linear-response oracle.
