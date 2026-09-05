# InhomogeneousSpinCavityDynamics.jl (`nude-quad`)

Cumulant dynamics of an inhomogeneous spin ensemble coupled to a cavity.
This branch adds a **multi-GPU monolith** for forward / order-2 / B-spline
optimization. See **[MONOLITH.md](MONOLITH.md)** for modes, the settings
schema, the `pulse_cost` loss, correctness notes, and how NCCL is used.

```bash
julia --project=. scripts/nude_quad_monolith.jl --settings examples/monolith_forward.jl
julia --project=. scripts/nude_quad_monolith.jl -s examples/monolith_optimizer.jl --grad adjoint
julia --project=. test/monolith_mgpu.jl
```
