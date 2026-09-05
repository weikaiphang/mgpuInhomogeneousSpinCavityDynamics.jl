# InhomogeneousSpinCavityDynamics.jl

Julia / multi-GPU simulation of a driven lossy cavity coupled to an
inhomogeneous spin-½ ensemble (Tavis–Cummings / Dicke-type), closed at
first- and second-order cumulants.

**This `chimera` tree is a package-first rebuild**, not a polished
nude-quad monolith. Production code lives in `src/chimera/`. Established
packages own each layer:

| Layer | Owner |
| --- | --- |
| Quadrature / inhomogeneity | FastGaussQuadrature.jl + QuadGK.jl |
| Cumulant EOMs | QuantumCumulants.jl + SecondQuantizedAlgebra |
| Time integration | DifferentialEquations / OrdinaryDiffEq (SciML) |
| Multi-GPU | CUDA.jl + NCCL.jl (Allreduce is the design center) |
| Small-N oracle | QuantumToolbox.jl |
| Trajectories | JLD2 |

See [docs/CHIMERA.md](docs/CHIMERA.md) for the architecture diagram,
deleted NIH modules, and the physics contract.

There is **no Volkov–Zon** solver.

## Features

* 1st-order means \(\langle a\rangle\), \(\langle S^+_j\rangle\), \(\langle S^z_j\rangle\)
* 2nd-order cumulants, including same-bin and cross-bin correlators
* Frequency and coupling inhomogeneity: Lorentzian / Gaussian / power-law \(g\), histogram or quadrature (tan–GL, GL+pdf, log-GL via FastGaussQuadrature)
* Pulses, saved trajectories (`.jld2`), QRT noise from the factorized 1st-order Jacobian
* SciML single-device path; sharded multi-GPU path (NCCL Allreduce for O(M) row-sums)
* PINN first-order datagen catalog (`src/datagen/`)

## Conventions

* \(S^z = \pm 1/2\) per spin
* Cavity damping \(-(\kappa_e+\kappa_i)/2\); drive \(+\sqrt{\kappa_e} E(t)\)
* \(a_\mathrm{out} = E - \sqrt{\kappa_e}\langle a\rangle\)
* Lorentzian \(N = C_\mathrm{ens}\,\kappa\,\mathrm{FWHM}/(4\langle g^2\rangle)\); Gaussian peak-matched with \(\sqrt{\pi\ln 2}\)
* `renormalize` defaults to `false` on frequency bins
* `QRT_CLOSURE_LEVEL = :factorized_first_order_jacobian` (same EOMs as QuantumCumulants order-1, not the 2nd-order Jacobian)
* Production multi-GPU is `src/chimera/mgpu`. `src/legacy/sim_2nd_multi_gpu_opt.jl` is quarantined

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

Order 1 and order 2 default to `ensemble_method = :auto`.

## Tests

```bash
julia --project=. test/physics_correctness.jl
julia --project=. test/package_first.jl
julia --project=. test/quantum_cumulants.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

## License

See `LICENSE`.
