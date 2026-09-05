# InhomogeneousSpinCavityDynamics.jl

Julia / multi-GPU simulation of a driven lossy cavity coupled to an
inhomogeneous spin-½ ensemble (Tavis–Cummings / Dicke-type), closed at
first- and second-order cumulants.

This `chimera` branch matches the scientific scope of `nude-quad` and
integrates research-grade packages (OrdinaryDiffEq, CUDA.jl, NCCL.jl,
FastGaussQuadrature.jl, QuadGK.jl). See [docs/CHIMERA.md](docs/CHIMERA.md)
for citations, rejected alternatives, physics fixes, and the multi-GPU
design.

A CUDA-compatible GPU is required for the production solvers. Physics
correctness tests in `test/physics_correctness.jl` run on CPU.

## Features

* 1st-order means \(\langle a\rangle\), \(\langle S^+_j\rangle\), \(\langle S^z_j\rangle\)
* 2nd-order cumulants, including same-bin and cross-bin correlators
* Frequency and coupling inhomogeneity: Lorentzian / Gaussian / power-law \(g\), histogram or quadrature (tan–GL, GL+pdf, log-GL)
* Pulses, saved trajectories (`.jld2`), QRT noise and correlation postprocessing
* DiffEq GPU path and sharded multi-GPU path (NCCL Allreduce for O(M) row-sums)
* PINN first-order datagen catalog (`src/datagen/`; see [src/datagen/README.md](src/datagen/README.md))

## Conventions

* \(S^z = \pm 1/2\) per spin
* Cavity damping \(-(\kappa_e+\kappa_i)/2\); drive \(+\sqrt{\kappa_e} E(t)\)
* \(a_\mathrm{out} = E - \sqrt{\kappa_e}\langle a\rangle\)
* Lorentzian \(N = C_\mathrm{ens}\,\kappa\,\mathrm{FWHM}/(4\langle g^2\rangle)\); Gaussian peak-matched with \(\sqrt{\pi\ln 2}\)
* `renormalize` defaults to `false` (truncated mass kept)

## Installation

```bash
git clone https://github.com/weikaiphang/mgpuInhomogeneousSpinCavityDynamics.jl.git
cd mgpuInhomogeneousSpinCavityDynamics.jl
julia --project=.
```

```julia
using Pkg
Pkg.instantiate()
using CUDA
CUDA.functional()
```

## Running an example

```bash
julia --project=. examples/run_demo.jl
julia --project=. examples/rase_2nd_order.jl
julia --project=. examples/MGPUrase_2nd_order.jl
```

```julia
using InhomogeneousSpinCavityDynamics
run_simulation(SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG)
```

Times are in seconds; angular frequencies are in rad/s unless stated.

Order 1 and order 2 default to `ensemble_method = :auto` (quadrature when
the configured distributions have a rule; otherwise histogram).

## Tests

```bash
julia --project=. test/physics_correctness.jl
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. scripts/bench_rowsum_exchange.jl
```

Physics tests check vacuum+ground as a 2nd-order fixed point, product-state
same-bin moments, quadrature mass, and \(C_\mathrm{ens}\) consistency.

## License

See `LICENSE`.
