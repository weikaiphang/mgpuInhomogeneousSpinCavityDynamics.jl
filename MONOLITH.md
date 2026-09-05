# Standalone nude-quad monolith

`src/monolith_mgpu.jl` is the product. This branch does **not** ship the old
nude-quad module tree (`rhs_*.jl`, `MGPU*.jl`, `pulse_optimizer2.jl`,
`sim_2nd_multi_gpu_opt.jl`, …). Physics, B-splines, loss, adjoint, and
multi-GPU are self-contained. No Volkov–Zon.

## Tree

```
src/monolith_mgpu.jl          # the implementation
scripts/nude_quad_monolith.jl # CLI
examples/monolith_*.jl        # settings for each mode
test/monolith_mgpu.jl
MONOLITH.md  README.md  LICENSE  Project.toml
```

## Modes

```
settings → prepare(ensemble_method=:auto|:quadrature|:histogram)
         → forward / forward_bspline(u) / order2 / order2_bspline(u)
         → optimize(u; grad=:adjoint|:forward)
```

`:auto` uses Gauss–Legendre when frequency is `:lorentzian`/`:gaussian` and
coupling is `:constant`/`:gaussian`/`:powerlaw_g`. **order2 uses the same rule.**

B-spline: `n_params = 3k + k nA + k nf`, unpack `gaps|durs|φ0|cA|cf`, softplus
scales, Gevrey taper.

## Loss

```
ss = 1 − (silencing − target_F)²
fid = inversion × ss
J = (1−fid)² + (κ_I/2)[I_min−I]₊² + (κ_S/2)[S_min−ss]₊²
  + w_time (T/T_max) + w_tmax (tmax_excess/T_max)²
  + w_power (‖cA/amp_scale‖² / n_cA)
```

Defaults: `I_min=S_min=0.85`, `κ_I=κ_S=50`, `w_time=0.15`, `w_power=0.05`,
`w_tmax=1`, `track=:weak`. Optimizer updates **only** B-spline parameters.
No noise/QRT unless a future mode asks for it.

Primary gradient: checkpointed discrete Tsit5 adjoint. `grad=:forward` is a
parity-only Dual-through-solve path.

## ICs (package ground SmSp=0 is wrong)

```
SmSp_same = |Sp|²(1−1/N)+N/2−Sz     # ground ⇒ Nⱼ
SzSz_same = Sz²(1−1/N)+N/4
SpSp_same = Sp²(1−1/N)
SzSp_same = Sz·Sp·(1−1/N)
cross j≠k = mean products
```

`κₜ = κₑ + κᵢ` always.

## Multi-GPU (the hot path)

One 2nd-order RHS stack, sharded:

- **small** `3+9M` — RHS evaluated **once** (rank 0). Other GPUs do not re-integrate `ȧ`.
- **large** `5×M×mloc` columns: `SpSp, SzSp, SzSpT, SmSp, SzSz`.
- Collectives: on-device Allreduce of O(M) row-sums (NCCL, else P2P).
  Not host `exchange_rowsums!`.
- CUDA kernels for large-block RHS. CPU is the same `rhs2_sharded!` loops
  (correctness / CI fallback), not a second physics stack.

1st-order shards bins; Allreduce of `Σ g S⁻`; `a` updated on rank 0.

If CUDA is absent, `backend=:auto` uses the CPU sharded fallback. That is
**not** pretending to be multi-GPU: the GPU path is what production should hit.

## Run

```bash
julia --project=. scripts/nude_quad_monolith.jl -s examples/monolith_forward.jl
julia --project=. scripts/nude_quad_monolith.jl -s examples/monolith_order2.jl
julia --project=. scripts/nude_quad_monolith.jl -s examples/monolith_optimizer.jl --grad adjoint
julia --project=. test/monolith_mgpu.jl
```
