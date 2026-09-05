# NIH tear gate (Reader / Devil)

Tip verified: `8190bcb537fcb168e862282396bdf7f63be22f19` on `cursor/chimera-tear-nih-e50a` (PR #11, off purged chimera `8f8d8f0`).
This note is the last-gate evidence. Residuals and suite counts are filled after the run.

## 1. NIH stepper gone — PASS

Tree grep + `ls` on tip `8190bcb`:

| Torn target | Status |
| --- | --- |
| `src/chimera/mgpu/integrator.jl` | absent |
| `src/chimera/mgpu/tableaus.jl` | absent |
| `function Tsit5Tableau` / `CK45Tableau` | no production definition |
| `function solve_mgpu!` / `tsit5_step!` / `ck45_step!` | no definition |
| `function cross_rhs_kernel!` / `small_rhs_kernel!` | no definition |
| `MGPUProblem.rhs!` | no method; `MGPUProblem` is sharding + NCCL context only |

`include_all.jl` loads `layout` → `devices` → `kernels` → `problem` → `ics` → `observables` → `state_io` → `rhs_cpu` → `solver`. No integrator/tableaus include.

Production time path: `mgpu_run_sim_2nd_order` → `chimera_ode_problem(rhs_2nd_order_mgpu!, ...)` → `chimera_solve` → `OrdinaryDiffEq.Tsit5()`. `CHIMERA_INTEGRATOR === :sciml_ordinarydiffeq`.

Out of tear scope (not the torn MGPU second stepper; still homemade RK elsewhere):
- `src/tsit5_discrete_adjoint.jl` pulse-optimizer discrete adjoint
- `src/correlations.jl` `_rk4_step_one_gpu!` (ASE/RASE postprocess)
- `src/chimera/noise/qrt.jl` `_noise_rk4_step_gpu!`

Those are not reachable from `rhs_2nd_order!` / `chimera_solve` / `mgpu_run_sim_2nd_order`.

## 2. Architecture — PASS (DiffEqGPU not forced)

One VF = `rhs_2nd_order!` (optional `injected_cross_rowsums`) + shard + Allreduce of the 3M row-sums → OrdinaryDiffEq Tsit5.

DiffEqGPU docs confirm the claim: `EnsembleGPUArray` / `EnsembleGPUKernel` solve **many independent small ODEs**; a single large structured `f` is supposed to be `CuArray` + OrdinaryDiffEq. Chimera order-2 is one tightly coupled O(M²) system whose multi-device primitive is a 3M Allreduce, not an ensemble. Documented in `docs/CHIMERA.md`. Production does not depend on DiffEqGPU.

## 3. Sacred C1 / H1 / `diag_mask` — (run below)

Hard-fail if the existing `physics_correctness` sets fail:
- `C1 ground: SmSp_same = Nj and vacuum+ground is a fixed point`
- `H1 product-state ICs for equator/weak/inverted`
- `cross-block diagonal is excluded from monolith row-sums`

## 4. PRIMARY CHEW — sharded RHS ≡ monolith

Production path under test: `rhs_2nd_order_sharded!` (CPU stand-in of shard + `assemble_rowsums!` + inject) vs monolith `rhs_2nd_order!`.

Coverage added on this gate: ground / equator / weak / inverted / weak_inverted / random, plus uneven `EnsemblePartition` splits and `:nccl` / `:p2p` / `:host` CPU mirrors.

| Quantity | Value |
| --- | --- |
| max abs residual | *(pending run)* |
| max rel residual | *(pending run)* |

Live multi-GPU NCCL / device kernels: only if `CUDA.functional()`.

## 5. Suites (this worker)

| Suite | PR #11 claim | This run |
| --- | --- | --- |
| `test/package_first.jl` | ~85 | pending |
| `test/quantum_cumulants.jl` | ~5 | pending |
| `test/oracle_quantumtoolbox.jl` | ~2 | pending |
| `test/physics_correctness.jl` | ~2993 + 2 GPU skips | pending |

Hardware: *(pending)*
