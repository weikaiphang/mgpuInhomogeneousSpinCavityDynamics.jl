# Chimera NIH audit (Devil's knife)

**Tip audited:** `8f8d8f00dce1a69cfe251c33e4f6a0a067d56261` (`cursor/chimera-packages-c852`, PR #6).
**Auditor branch:** `cursor/chimera-audit-nih-7693`.

## Verdict (blunt)

`src/chimera/mgpu` (~2135 lines) is **relocated NIH custom multi-GPU**, not a package-backed multi-GPU backend.

QuantumCumulants owns **small-M symbolic derivation**. OrdinaryDiffEq owns the **single-device** stepper. NCCL.jl owns **Allreduce of a 3M row-sum buffer**. None of those packages integrate, shard, or evaluate the production 2nd-order RHS on multiple GPUs.

The `eoms/` tree is a QuantumCumulants **wrapper + structured large-M backend**. The `mgpu/` tree is a fused CUDA replica of that backend plus a homemade Tsit5/CK45.

## Who actually owns what

| Layer | Package that owns it | What chimera still writes |
| --- | --- | --- |
| Ensemble nodes | FastGaussQuadrature `gausslegendre` + QuadGK histogram means | `quadrature.jl` maps Lorentzian/Gaussian bins |
| Hamiltonian / small-M EOMs | QuantumCumulants + SecondQuantizedAlgebra `complete(meanfield(...))` | `eoms/quantum_cumulants_impl.jl` (37 lines). Checked-in image: `eoms/generated/tc_equations.txt` |
| Large-M 1st/2nd RHS | **Nobody.** Hand-structured closures | `eoms/closure_*.jl`. Honest comment: backend of the QC-derived equations, not a second theory |
| Single-device time step | OrdinaryDiffEq `Tsit5` via `chimera_solve` | `integrate/sciml*.jl`. Order-2 can feed a `CuArray` into SciML (single GPU) |
| Multi-GPU time step | **Nobody.** Homemade Tsit5/CK45 + PI controller | `mgpu/integrator.jl`, `mgpu/tableaus.jl` |
| Multi-GPU 2nd-order RHS | **Nobody.** Custom `@cuda` kernels | `mgpu/kernels.jl` (`cross_rhs_kernel!`, `small_rhs_kernel!`, warp/block reductions) |
| Row-sum collective | NCCL.jl `Allreduce!` when ≥2 unique devices | `exchange_rowsums_nccl!`. P2P `copyto!` and host pin are fallbacks |
| Tiny Hilbert oracle | QuantumToolbox | `oracle/quantumtoolbox.jl` |
| Order-1 "MGPU" | SciML single-device | `mgpu_run_simulation` **returns `run_sim_1st_order`** |

`src/chimera/mgpu` has **zero** mentions of `OrdinaryDiffEq`, `ODEProblem`, `chimera_solve`, `QuantumCumulants`, or `DiffEqGPU`. That is the ownership test.

`integrate/sciml.jl` already admits it:

> The custom MGPU Tsit5/CK45 stepper is only the multi-GPU performance path

`eoms/closure_2nd.jl` already admits it:

> MGPU kernels are a fused sharded replica, not a second equation set.

## Evidence (files / functions)

NIH stepper and kernels:

- `solve_mgpu!`, `tsit5_step!`, `ck45_step!`, `PIController` — `mgpu/integrator.jl`
- `Tsit5Tableau`, `CK45Tableau` (hardcoded coefficients) — `mgpu/tableaus.jl`
- `cross_rhs_kernel!`, `small_rhs_kernel!`, `global_sums_kernel!`, `combine_kernel!`, `errnorm_kernel!` — `mgpu/kernels.jl` (522 lines)
- `MGPUProblem`, `Shard`, `rhs!` (`@cuda` launch graph) — `mgpu/problem.jl`
- `Executor` / `each_shard` thread spawn — `mgpu/devices.jl`

Package primitives only:

- `NCCL.Allreduce!` on `reinterpret(real, rowsum)` and on `normout` — `exchange_rowsums_nccl!`, `gather_norm_nccl`
- `CUDA.can_access_peer` / `enable_peer_access` — `devices.jl`; used by `exchange_rowsums_p2p!`
- `rhs_cpu!` is a one-call wrapper around `rhs_2nd_order!` — `mgpu/rhs_cpu.jl`

QC wrapper, not a runtime backend:

- `_derive_tc_M1` / `_derive_tc_indexed` call `complete(meanfield(...))` and stop
- Production `run_sim_*` never builds a QC `ODESystem`
- `qc_rhs_1st_M1!` / `qc_rhs_2nd_M1_image!` are numeric images for tests

## Devil cut line (do not rewrite in this audit)

**Keep**

- `eoms/quantum_cumulants*.jl` + `generated/tc_equations.txt` as the derivation owner
- `eoms/closure_*.jl`, `ics_*.jl`, `state_*.jl` as the large-M structured backend
- `integrate/sciml*.jl` as the package-backed single-device (and single-GPU CuArray) path
- `exchange_rowsums_nccl!` / `_p2p!` as the collective primitive
- C1 / H1 product-state ICs and `diag_mask` row-sum exclusion

**Tear later (NIH that should become a package-backed backend)**

- Custom integrator: `mgpu/integrator.jl` + `mgpu/tableaus.jl`
- Custom fused RHS kernels: `mgpu/kernels.jl` `cross_rhs_kernel!` / `small_rhs_kernel!` / reductions
- Custom register file / launch graph: `MGPUProblem.rhs!`

The one equation set is already `rhs_2nd_order!`. A package-backed multi-GPU cut is:

1. Keep sharding + `assemble_rowsums!` / NCCL Allreduce of the 3M row-sums.
2. Evaluate **that** vector field (or a thin KernelAbstractions/CUDA port of it), not a second algebra.
3. Hand the stages to OrdinaryDiffEq or DiffEqGPU. Delete the homemade Tsit5.

No correctness rewrite is required for this audit. The CPU-sharded replica matches the monolith. The QC order-2 M=1 image matches the shared sector (`⟨a⟩`, `⟨σ21⟩`, `⟨σ22⟩`, `⟨a'a⟩`, `⟨a'σ21⟩`, `⟨a'a'⟩`). A naive `Sz↔σ22` map of `⟨a'σ22⟩` / `⟨a σ21⟩` does **not** match — that is the cut: do not `eval` the QC 8-vector as the production RHS.

## Hardware on this worker

- `nvidia-smi`: absent
- CUDA devices: 0 (`CUDA.functional() == false`)
- Live ≥2-GPU NCCL Allreduce / P2P / device `rhs!` was **not executed**
- CPU mirrors (`assemble_rowsums_*`, `_stress_mgpu_rhs`) remain the gate
- Explicit skips: hardware inventory (`ndev < 2`) and `GPU↔CPU 2nd-order RHS parity`

## Tests on this revision (Julia 1.10.10, CPU)

PR #6 claimed: package_first 61, QC 5, oracle 2, physics 2905 + 1 GPU skip.

This run:

| Suite | Result vs claim |
| --- | --- |
| `test/package_first.jl` | **78 pass** (61 prior + 17 NIH ownership) |
| `test/quantum_cumulants.jl` | **8 pass** (5 prior + 3 generated-file checks) |
| `test/oracle_quantumtoolbox.jl` | **2 pass** (matches claim) |
| `test/physics_correctness.jl` | **2912 pass + 2 skip** (2905 prior + 1 hardware pass + 6 QC/MGPU parity; skips = `<2 GPU` + `!CUDA.functional`) |

C1, H1, and `diag_mask` row-sum exclusion all passed. No production RHS rewrite.

## Sacred physics (do not regress)

- C1: ground `SmSp_same = Nⱼ`; vacuum+ground is a fixed point
- H1: product-state ICs for equator / weak / inverted / weak_inverted
- `diag_mask` (`.!I`): unused cross-block diagonal excluded from row-sums
