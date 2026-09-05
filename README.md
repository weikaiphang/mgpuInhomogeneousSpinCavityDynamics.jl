# NudeQuadMonolith

Standalone multi-GPU solver for inhomogeneous spin–cavity **cumulant** dynamics.
This branch is **not** a wrapper around the old nude-quad package: one implementation,
one RHS stack, settings in / results out.

See **[MONOLITH.md](MONOLITH.md)** for modes, loss, ICs, and the NCCL/P2P layout.

```bash
julia --project=. scripts/nude_quad_monolith.jl --settings examples/monolith_forward.jl
julia --project=. scripts/nude_quad_monolith.jl -s examples/monolith_optimizer.jl --grad adjoint
julia --project=. test/monolith_mgpu.jl
```

Optional: CUDA + NCCL for the live multi-GPU order-2 path. ForwardDiff is required
for the discrete-adjoint drive VJP. There is no Volkov–Zon path.
