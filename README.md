# SpinCavityMonolith

Multi-GPU cumulant dynamics for an inhomogeneous spin ensemble coupled to a
driven cavity. One module (`src/SpinCavityMonolith.jl`), settings in, modes out.

```bash
julia --project=. scripts/run_monolith.jl --settings examples/monolith_forward.jl
julia --project=. scripts/run_monolith.jl -s examples/monolith_optimizer.jl --grad adjoint
julia --startup-file=no test/spin_cavity_monolith.jl
```

See **[MONOLITH.md](MONOLITH.md)** for modes and settings, and
**[REQUIREMENTS.md](REQUIREMENTS.md)** for the requirement → code map.

CPU Threads is the production path until Tuesday iron (`julia -t auto`).
Live ≥2-GPU NCCL/P2P is **not claimed done**. Required: ForwardDiff
(discrete-adjoint drive VJP). This VM has no NVIDIA GPU.
