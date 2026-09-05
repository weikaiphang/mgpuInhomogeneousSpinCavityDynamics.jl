# NIH tear gate (Reader / Devil)

**Verdict: PASS. The NIH stepper is gone. Sharded RHS ≡ monolith on this worker (exact 0 residual). Tear can be called done on CPU evidence. Live multi-GPU NCCL was not executed (0 devices).**

Tip verified: `8190bcb537fcb168e862282396bdf7f63be22f19` on `cursor/chimera-tear-nih-e50a` (PR #11, off purged chimera `8f8d8f0`).
This PR adds residual chew + this note. Does not revive the NIH stepper.

Worker: Julia 1.10.12, `nvidia-smi` absent, `CUDA.functional() == false`, 0 devices. NCCL.jl listed as a dep; live Allreduce not executed (`NCCL` artifact missing / failed to precompile, as expected without CUDA).

## 1. NIH stepper gone — PASS

Tree grep + `ls` on tip `8190bcb`:

| Torn target | Status |
| --- | --- |
| `src/chimera/mgpu/integrator.jl` | **absent** |
| `src/chimera/mgpu/tableaus.jl` | **absent** |
| `function Tsit5Tableau` / `CK45Tableau` | no production definition |
| `function solve_mgpu!` / `tsit5_step!` / `ck45_step!` | no definition |
| `function cross_rhs_kernel!` / `small_rhs_kernel!` | no definition |
| `function combine_kernel!` / `lowstorage_kernel!` | no definition |
| `MGPUProblem.rhs!` | no method; `MGPUProblem` is sharding + NCCL context only |

`include_all.jl` loads `layout` → `devices` → `kernels` → `problem` → `ics` → `observables` → `state_io` → `rhs_cpu` → `solver`. No integrator/tableaus include.

Production time path: `mgpu_run_sim_2nd_order` → `chimera_ode_problem(rhs_2nd_order_mgpu!, ...)` → `chimera_solve` → `OrdinaryDiffEq.Tsit5()`. `CHIMERA_INTEGRATOR === :sciml_ordinarydiffeq`.

Remaining kernels are row-sum only: `rowsum_partial_kernel!`, `rowsum_finalize_kernel!`, `fill_kernel!`.

Out of tear scope (not the torn MGPU second stepper; still homemade RK elsewhere):
- `src/tsit5_discrete_adjoint.jl` pulse-optimizer discrete adjoint
- `src/correlations.jl` `_rk4_step_one_gpu!` (ASE/RASE postprocess)
- `src/chimera/noise/qrt.jl` `_noise_rk4_step_gpu!`

Those are not reachable from `rhs_2nd_order!` / `chimera_solve` / `mgpu_run_sim_2nd_order`.

## 2. Architecture — PASS (DiffEqGPU not forced)

One VF = `rhs_2nd_order!` (optional `injected_cross_rowsums`) + shard + Allreduce of the 3M row-sums → OrdinaryDiffEq Tsit5.

Verified against [DiffEqGPU getting started](https://docs.sciml.ai/DiffEqGPU/stable/getting_started/):
- **Many independent small ODEs** → `EnsembleGPUArray` / `EnsembleGPUKernel`
- **One big structured `f`** → `CuArray` + OrdinaryDiffEq Tsit5

Chimera order-2 is one tightly coupled O(M²) system. The multi-device primitive is a 3M Allreduce, not an ensemble. Production does not depend on DiffEqGPU. Documented in `docs/CHIMERA.md`. Claim stands; DiffEqGPU was not forced.

## 3. Sacred C1 / H1 / `diag_mask` — PASS (hard)

| Set | Result |
| --- | --- |
| `C1 ground: SmSp_same = Nj and vacuum+ground is a fixed point` | **10 / 10 pass** |
| `H1 product-state ICs for equator/weak/inverted` | **16 / 16 pass** |
| `cross-block diagonal is excluded from monolith row-sums` | **5 / 5 pass** |

## 4. PRIMARY CHEW — sharded RHS ≡ monolith — PASS

Production path: `rhs_2nd_order_sharded!` (shard + `assemble_rowsums!` + inject) vs monolith `rhs_2nd_order!`.

Coverage on this gate: ground / equator / weak / inverted / weak_inverted / random, uneven `EnsemblePartition` splits, and `:nccl` / `:p2p` / `:host` CPU mirrors.

| Quantity | Value |
| --- | --- |
| residual rows | 138 (75 uneven) |
| **max abs residual** | **0.0** |
| **max rel residual** | **0.0** |
| ground | max_abs=0.0 max_rel=0.0 n=21 |
| equator | max_abs=0.0 max_rel=0.0 n=30 |
| weak | max_abs=0.0 max_rel=0.0 n=20 |
| inverted | max_abs=0.0 max_rel=0.0 n=17 |
| weak_inverted | max_abs=0.0 max_rel=0.0 n=7 |
| random | max_abs=0.0 max_rel=0.0 n=43 |

Exact zeros on CPU Float64: injected 3M row-sums match `(Cross .* diag_mask) * g` bit-for-bit on this worker. Live device kernels / NCCL Allreduce were **not** executed (0 GPUs).

## 5. Suites (this worker)

| Suite | PR #11 claim | This run |
| --- | --- | --- |
| `test/package_first.jl` | ~85 | **85 pass** (38+8+3+3+4+5+24) |
| `test/quantum_cumulants.jl` | ~5 | **5 pass** |
| `test/oracle_quantumtoolbox.jl` | ~2 | **2 pass** |
| `test/physics_correctness.jl` | ~2993 + 2 GPU skips | **3435 pass + 2 skip** |

Physics delta vs PR #11: +442 from inverted/random backend expansion (+20 on the existing production set) plus the new residual table (422). Skips = `<2 GPU` hardware inventory and `!CUDA.functional()` device RHS. No failures.

NIH-gone subset inside package_first: **24 / 24 pass**.
