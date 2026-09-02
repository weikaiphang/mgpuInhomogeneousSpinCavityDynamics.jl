# InhomogeneousSpinCavityDynamics.jl

Julia simulation code for **"Inhomogeneous Light-Matter Coupling as a Resource for Noiseless Quantum Memories"**.

`InhomogeneousSpinCavityDynamics.jl` is a GPU-based Julia package for simulating spin ensembles coupled to a cavity, including both spin-frequency inhomogeneity and spin–cavity coupling-strength inhomogeneity. A CUDA-compatible GPU is required to run the simulations.

## Features

* GPU-based simulation with first- and second-order cumulant expansion
* Frequency and spin–cavity coupling-strength inhomogeneity
* Configurable pulse sequences and ensemble distributions
* Noise and correlation calculations using the Quantum Regression Theorem
* PINN first-order datagen catalog (`src/datagen/`; see [src/datagen/README.md](src/datagen/README.md))
* Standalone analytical Volkov–Zon solver (`src/volkov_zon.jl`) — closed-form linear-response dynamics with no ensemble discretisation and no ODE solve; CPU only, no CUDA required

## Installation

Clone the repository:

```bash
git clone https://github.com/Qovelab/InhomogeneousSpinCavityDynamics.jl.git
```

Move into the repository:

```bash
cd InhomogeneousSpinCavityDynamics.jl
```

Start Julia with the repository environment activated:

```bash
julia --project=.
```

Install the required packages:

```julia
using Pkg
Pkg.instantiate()
```

Check whether CUDA is available:

```julia
using CUDA
CUDA.functional()
```

## Running an Example

Run an example from the repository directory:

```bash
julia --project=. examples/run_demo.jl
```

Inside a Julia script, load the package with:

```julia
using InhomogeneousSpinCavityDynamics
```

A simulation is run using:

```julia
InhomogeneousSpinCavityDynamics.run_simulation(
    SIM_SETTING,
    SYSTEM_CONFIG,
    PULSE_CONFIG,
)
```

All times are specified in seconds, and angular frequencies are specified in radians per second unless otherwise stated.

## User Interface

The simulation is controlled through three configuration objects:

* `SIM_SETTING`: numerical simulation and saving settings
* `SYSTEM_CONFIG`: cavity and spin-ensemble parameters
* `PULSE_CONFIG`: input pulse sequence

Unless stated otherwise:

* Times are in seconds.
* Frequencies, detunings, coupling strengths, and decay rates are angular frequencies in rad/s.
* Phases are in radians.

### `SIM_SETTING`

`SIM_SETTING` controls the simulation order, ensemble discretization, solver, and output file.

```julia
SIM_SETTING = (
    simulation_order = :order1,

    M_delta = 250,
    M_g     = 1,

    initial_condition = :ground,

    Ttotal = 100e-6,

    Nt_save = 5001,
    reltol  = 1e-8,
    abstol  = 1e-8,

    saved_file_name = joinpath(OUTDIR, "demo.jld2"),
)
```

#### Parameters

* `simulation_order`: Cumulant-expansion order.

  * `:order1`: first-order cumulant expansion
  * `:order2`: second-order cumulant expansion

* `M_delta`: Number of spin-frequency detuning bins.

* `M_g`: Number of spin–cavity coupling-strength bins.

* `initial_condition`: Initial spin state.

  * `:ground`: all spins initially in the ground state
  * `:inverted`: all spins initially in the excited state
  * `:custom`: user-defined initial state

* `Ttotal`: Total simulation duration.

* `Nt_save`: Number of uniformly spaced time points stored in the output file. The ODE solver may use additional internal time steps.

* `reltol`: Relative error tolerance of the ODE solver.

* `abstol`: Absolute error tolerance of the ODE solver.

* `saved_file_name`: Path and filename of the saved JLD2 data.

### `SYSTEM_CONFIG`

`SYSTEM_CONFIG` defines the ensemble cooperativity, cavity parameters, spin-frequency inhomogeneity, and spin–cavity coupling-strength inhomogeneity.

Only one `freq_inhomogeneity` and one `g_inhomogeneity` should be included in each `SYSTEM_CONFIG`.

```julia
SYSTEM_CONFIG = (
    C_ens = 0.6,

    delta0 = 0.0,
    kappa_e = 2π * 1e6,
    kappa_i = 0.0,

    freq_inhomogeneity = (
        kind = :lorentzian,
        FWHM = 2π * 1e6,
        span_gamma = 2.5,
        renormalize = false,
    ),

    g_inhomogeneity = (
        kind = :constant,
        g_value = 2π * 100,
    ),
)
```

#### General system parameters

* `C_ens`: Ensemble cooperativity.

* `delta0`: Cavity detuning relative to the rotating-frame reference frequency.

* `kappa_e`: External cavity decay rate.

* `kappa_i`: Internal cavity loss rate.

The total cavity decay rate is

```math
\kappa = \kappa_e + \kappa_i.
```

#### Spin-frequency inhomogeneity

`freq_inhomogeneity` defines the distribution of spin-frequency detunings.

##### Lorentzian distribution

```julia
freq_inhomogeneity = (
    kind = :lorentzian,
    FWHM = 2π * 1e6,
    span_gamma = 2.5,
    renormalize = false,
)
```

* `kind`: Selects the Lorentzian distribution.

* `FWHM`: Full width at half maximum of the detuning distribution.

* `span_gamma`: Simulated detuning range in units of the Lorentzian half width
  ($\gamma=\mathrm{FWHM}/2$). The distribution is discretized approximately over

  ```math
  -\text{span\_gamma}\cdot\gamma
  \leq \Delta \leq
  \text{span\_gamma}\cdot\gamma.
  ```

* `renormalize`: Controls the treatment of the truncated distribution.

  * `true`: rescales the probability inside the simulated range so that it sums to one
  * `false`: preserves the probability weight removed by truncation

##### Gaussian distribution

```julia
freq_inhomogeneity = (
    kind = :gaussian,
    FWHM = 2π * 1e6,
    span_sigma = 3.0,
    renormalize = false,
)
```

* `kind`: Selects the Gaussian distribution.

* `FWHM`: Full width at half maximum of the Gaussian distribution.

* `span_sigma`: Simulated detuning range on each side of the distribution center, measured in standard deviations. The distribution is discretized approximately over

  ```math
  -\text{span\_sigma}\cdot\sigma
  \leq \Delta \leq
  \text{span\_sigma}\cdot\sigma.
  ```

* `renormalize`: Controls whether the truncated Gaussian distribution is rescaled to have total probability one.

#### Spin–cavity coupling-strength inhomogeneity

`g_inhomogeneity` defines the distribution of spin–cavity coupling strengths.

##### Constant coupling

```julia
g_inhomogeneity = (
    kind = :constant,
    g_value = 2π * 100,
)
```

* `kind`: Selects a constant coupling strength.

* `g_value`: Coupling strength assigned to every spin.

For this option, `M_g` should normally be set to `1`.

##### Gaussian coupling distribution

```julia
g_inhomogeneity = (
    kind = :gaussian,
    mean = 2π * 100,
    std  = 2π * 0.00001,
    span_sigma = 3.0,
    renormalize = true,
)
```

* `kind`: Selects a Gaussian coupling-strength distribution.

* `mean`: Mean spin–cavity coupling strength.

* `std`: Standard deviation of the coupling strengths.

* `span_sigma`: Simulated coupling range on each side of the mean, measured in standard deviations.

* `renormalize`: Controls whether the truncated coupling distribution is rescaled to have total probability one.

##### Power-law coupling distribution

```julia
g_inhomogeneity = (
    kind = :powerlaw_g,
    alpha = 5 / 3,

    g_min = 2π * 1.0,
    g_max = 2π * 1000.0,

    binning = :log,
    renormalize = true,
)
```

The distribution follows

```math
p(g) \propto g^{-\alpha}.
```

* `kind`: Selects the power-law coupling distribution.

* `alpha`: Power-law exponent.

* `g_min`: Minimum coupling strength included in the simulation.

* `g_max`: Maximum coupling strength included in the simulation.

* `binning`: Spacing of the coupling bins.

  * `:log`: logarithmically spaced bins
  * `:linear`: equally spaced bins

* `renormalize`: Controls whether the distribution between `g_min` and `g_max` is normalized to have total probability one.

Logarithmic binning is generally more suitable when the coupling strengths cover several orders of magnitude.

##### User-defined coupling distribution

```julia
g_inhomogeneity = (
    kind = :user_defined,
    filename = "data/g_distribution.jld2",
    renormalize = true,
)
```

* `kind`: Selects a coupling distribution loaded from a file.

* `filename`: Path to the file containing the user-defined coupling distribution.

* `renormalize`: Controls whether the loaded distribution is rescaled to have total probability one.

The input file must follow the format expected by the user-defined distribution loader.

### `PULSE_CONFIG`

`PULSE_CONFIG` is a tuple containing all pulses applied during the simulation.

```julia
PULSE_CONFIG = (
    gaussian_pulse,
    wurst_pulse,
)
```

Each pulse is defined by a named tuple. Multiple pulses of the same or different types can be included.

#### Gaussian pulse

```julia
(
    name  = "Gaussian input signal",
    kind  = :gaussian,

    t0    = 20e-6,
    sigma = (10 / 6) * 1e-6,
    amp   = 0.0,
    omega = 0.0,
    phase = 0.0,
)
```

* `name`: Descriptive name used to identify the pulse.

* `kind`: Pulse type, here `:gaussian`.

* `t0`: Center time of the Gaussian pulse.

* `sigma`: Temporal standard deviation of the Gaussian envelope.

* `amp`: Maximum input-field amplitude.

* `omega`: Angular-frequency offset relative to the rotating-frame reference frequency.

* `phase`: Initial phase of the pulse.

#### WURST pulse

```julia
(
    name       = "First WURST pulse",
    kind       = :wurst,

    t_center   = 75e-6,
    duration   = 10e-6,
    amp        = 0.5 * sqrt(SYSTEM_CONFIG.kappa_e) * 2.0e4,
    bandwidth  = 5.0 * SYSTEM_CONFIG.freq_inhomogeneity.FWHM,
    n          = 20.0,
    omega0     = 0.0,
    chirp_sign = +1.0,
    phase0     = 0.0,
    edge_frac  = 1e-4,
)
```

* `name`: Descriptive name used to identify the pulse.

* `kind`: Pulse type, here `:wurst`.

* `t_center`: Center time of the WURST pulse.

* `duration`: Total duration of the pulse.

* `amp`: Maximum input-field amplitude.

* `bandwidth`: Total angular-frequency sweep bandwidth.

* `n`: Order of the WURST envelope. Larger values produce a flatter central region and sharper pulse edges.

* `omega0`: Center angular-frequency offset of the sweep.

* `chirp_sign`: Direction of the frequency sweep.

  * `+1.0`: increasing frequency
  * `-1.0`: decreasing frequency

* `phase0`: Initial phase of the pulse.

* `edge_frac`: Small cutoff applied near the pulse edges to avoid numerical problems where the WURST envelope approaches zero.

## Saved Data

Simulation results, input configurations, and ensemble information are saved in a JLD2 file.

### First-order saved data

#### Configurations

* `SIM_SETTING`: Simulation, solver, and saving settings.
* `SYSTEM_CONFIG`: Cavity and spin-ensemble parameters.
* `PULSE_CONFIG`: Pulse sequence used in the simulation.

#### Time-dependent quantities

* `t_saved`: Saved simulation times.
* `a_sol`: Cavity amplitude $\langle a\rangle$.
* `Σp_sol`: Total spin coherence $\sum_j \langle S_j^+\rangle$.
* `Σz_sol`: Total spin inversion $\sum_j \langle S_j^z\rangle$.
* `E_of_t_arr`: Input field evaluated at every saved time.

#### Ensemble discretization

* `M_delta`: Number of detuning bins.
* `M_g`: Number of coupling-strength bins.
* `M_total`: Total number of ensemble bins, $M_\delta M_g$.
* `delta_b_1d`: Detuning-bin centers.
* `g_b_1d`: Coupling-strength-bin centers.
* `Nj_2d`: Number of spins in each detuning–coupling bin.

#### Resonant-detuning trajectories

For every coupling-strength bin, the code selects the detuning bin closest to resonance and saves its trajectory.

* `idelta_res`: Selected detuning-bin index for each coupling-strength bin.
* `delta_res`: Selected detuning value for each coupling-strength bin.
* `keep_bins`: Flattened indices of the selected two-dimensional bins.
* `g_keep`: Coupling strength of each selected bin.
* `delta_keep`: Detuning of each selected bin.
* `Sp_keep`: Selected spin-coherence trajectories with shape $M_g \times N_{\mathrm{t,save}}$.
* `Sz_keep`: Selected spin-inversion trajectories with shape $M_g \times N_{\mathrm{t,save}}$.

#### Optional peak detection

* `peak_detection_config`: Settings used for automatic peak detection.
* `peak_detection_results`: Detected peak times, amplitudes, and related results.

#### Additional information

* `N_total`: Total number of spins represented by the ensemble.
* `elapsed_seconds`: Total simulation runtime in seconds.

### Second-order saved data

#### Configurations

* `SIM_SETTING`: Simulation, solver, and saving settings.
* `SYSTEM_CONFIG`: Cavity and spin-ensemble parameters.
* `PULSE_CONFIG`: Pulse sequence used in the simulation.

#### Cavity quantities

* `t_saved`: Saved simulation times.
* `a_sol`: Cavity amplitude $\langle a\rangle$.
* `n_sol`: Intracavity photon number $\langle a^\dagger a\rangle$.
* `adad_sol`: Second-order cavity term $\langle a^\dagger a^\dagger\rangle$.
* `E_of_t_arr`: Input field evaluated at every saved time.

#### Spin quantities

* `Sp_sol`: Spin coherence $\langle S_j^+\rangle$ for every ensemble bin.
* `Sz_sol`: Spin inversion $\langle S_j^z\rangle$ for every ensemble bin.

#### Cavity–spin correlations

* `adSp_sol`: Spin-cavity correlation $\langle a^\dagger S_j^+\rangle$.
* `adSm_sol`: Spin-cavity correlation $\langle a^\dagger S_j^-\rangle$.
* `adSz_sol`: Spin-cavity correlation $\langle a^\dagger S_j^z\rangle$.

These spin quantities and correlations are saved for every ensemble bin and every saved time.

#### Ensemble discretization

* `delta_b_1d`: Detuning-bin centers.
* `g_b_1d`: Coupling-strength-bin centers.
* `Nj_2d`: Number of spins in each detuning–coupling bin.

#### Additional information

* `N_total`: Total number of spins represented by the ensemble.
* `elapsed_seconds`: Total simulation runtime in seconds.

## Analysis scripts

The `scripts/` directory contains the following data-analysis programs:

- `plot_1st_order.jl`: Loads first-order simulation data and plots the cavity field, collective spin quantities, ensemble distributions, resonant-bin trajectories, and detected echoes.

- `plot_2nd_order.jl`: Loads second-order simulation data and plots the first- and second-order cavity and spin moments.

- `compute_noise.jl`: Uses second-order simulation data to calculate the output-field noise.

- `compute_correlations.jl`: Uses the Quantum Regression Theorem to calculate two-time correlations, including ASE–ASE, RASE–RASE, and ASE–RASE correlations.

Run a script from the repository directory using, for example:

```bash
julia --project=. scripts/compute_noise.jl
```

## Analytical Volkov–Zon solver

`src/volkov_zon.jl` is a **fully independent, standalone** solver for the same
physical system, using the analytical Volkov–Zon method instead of numerical
integration. It depends only on the Julia standard library — no CUDA, no
`DifferentialEquations`, no `Distributions` — so it runs on any machine:

```julia
include("src/volkov_zon.jl")
using .VolkovZon

sys = VZSystem(
    kappa_e   = 2π * 1e6,
    C_ens     = 0.6,
    detuning  = DetuningLorentzian(2π * 1e6),
    coupling  = CouplingPowerLaw(5/3, 2π * 1.0, 2π * 1000.0),
    drive     = gaussian_drive(t0 = 20e-6, sigma = (10/6) * 1e-6, amp = 1.0),
)

sol = vz_solve(sys; Ttotal = 100e-6, Nt_save = 5001)
sol.a, sol.a_out, sol.Sigma_p, sol.Sigma_z, sol.poles
```

### How it works

The method linearises the spin inversion (`⟨S^z⟩ → s₀/2`), which makes the spin
equation solvable in closed form. Substituting back, the whole inhomogeneous
ensemble collapses onto two closed-form scalars: the coupling second moment
`⟨g²⟩` and the detuning memory kernel

```math
K(t) = \int \rho(\Delta)\, e^{-i\Delta t}\, d\Delta ,
\qquad
\tilde K(s) = \int \frac{\rho(\Delta)}{s + i\Delta}\, d\Delta ,
```

obtained by contour integration. The cavity transform is then algebraic,

```math
\hat a(s) = \frac{a(0) + \sqrt{\kappa_e}\,\hat E(s) - i N \hat J(s)}{s + \kappa_t/2 + i\delta_0 - s_0 N\langle g^2\rangle \tilde K(s)} .
```

Two inversion routes are provided and selected automatically:

* `:residue` — **exact**. For a Lorentzian (and for the power-law family below)
  `K̃` is rational, so the poles of the transform are the roots of a
  polynomial and `⟨a⟩(t)` is an exact finite sum of exponentials — the
  cavity/spin polariton branches, returned in `sol.poles`.
* `:fft` — Bromwich inversion of the exact transform, used for the Gaussian
  line shape, whose `K̃` is the Faddeeva function.

**There is no ensemble discretisation to converge and no ODE is integrated**, so
the cost depends only on `Nt_save`. This matters most for heavy-tailed line
shapes: reproducing a Lorentzian ensemble by binning needs very many bins, since
even ±400 half-widths captures only ~96% of the probability.

### Distributions

Detuning `ρ(Δ)` — Lorentzian, Gaussian, or power law:

| Constructor | Density |
| --- | --- |
| `DetuningLorentzian(FWHM; center)` | `∝ 1/((Δ-c)² + w²)` |
| `DetuningGaussian(FWHM; center)` | `∝ exp(-(Δ-c)²/2σ²)` |
| `DetuningPowerLaw(FWHM, n; center)` | `∝ [1 + ((Δ-c)/w)²]^(-n)`, algebraic tails of exponent `2n` |

`DetuningPowerLaw` is the Pearson-VII / generalised-Lorentzian family: `n = 1`
reproduces the Lorentzian exactly and larger `n` interpolates towards the
Gaussian while keeping algebraic tails. Integer `n` keeps the kernel rational,
so the exact residue solution is available for the power-law case too.

Coupling `p(g)` — Lorentzian, Gaussian, or power law (plus a constant):

| Constructor | Density |
| --- | --- |
| `CouplingConstant(g)` | single value |
| `CouplingGaussian(mean, std; span_sigma)` | truncated normal |
| `CouplingLorentzian(center, FWHM; span_gamma)` | truncated Cauchy |
| `CouplingPowerLaw(alpha, g_min, g_max)` | `∝ g^(-alpha)`, as `:powerlaw_g` |

At linear order the coupling distribution enters the cavity dynamics *only*
through `⟨g²⟩`, so its shape is degenerate there; `⟨g⟩` additionally sets the
collective coherence. Coupling distributions are truncated and renormalised —
an untruncated Cauchy has no second moment.

### Output

`vz_solve` returns a `NamedTuple` with `t`, `a` (`⟨a⟩`), `a_out`
(`E(t) - √κ_e ⟨a⟩`, the package's convention), `E`, `Sigma_p` / `Sigma_m`
(`Σⱼ⟨S_j^±⟩`), `Sigma_z`, `Ngsigma_m`, `poles`, `residues`, `N_total`, `C_ens`
and `linear_validity`. Passing `bin_deltas` (and optionally `bin_gs`) also
returns the resolved single-spin coherences `sigma_minus` / `sigma_plus`;
`vz_ensemble_grid` supplies matching quadrature nodes and weights.

### Validity

The method is exact for the *linearised* ensemble. Every solution reports
`linear_validity`, the largest fractional change of the total inversion. It is
an ensemble average — resonant bins deplete considerably more — so keep it well
below `1e-3` for quantitative work. Measured against a direct nonlinear
integration, a `linear_validity` of `2.5e-7` gave `~2e-5` relative error in
`⟨a⟩`, and `2.5e-5` gave `~1.6e-3`.

### Running it

```bash
julia --project=. examples/volkov_zon_demo.jl
julia --project=. test/volkov_zon.jl
```

## PINN datagen

`src/datagen/` is a separate script package that builds a first-order
trajectory catalog for PINN training. It does not change solver physics.
Full documentation: [src/datagen/README.md](src/datagen/README.md).

`--phase` is required (`configs`, `simulate`, or `all`). Running the
script with no arguments is an error. `configs` **replaces**
`data/datagen/configs/`; do not use it to resume. Resume a killed
simulate with `--phase simulate` and the same run flags. Skip is
filename-based (JLD2 + nonempty pulsemat). The process exits `1` if any
catalog entry or trajectory failed.

Default `--M-cap 60000` and `--NT-save 5001` need on the order of 10 GB
host RAM per job. Smoke one stem with a smaller cap before a full
campaign. Start Julia with at least as many threads as CUDA GPUs
(`-t auto`).

```bash
julia --project=. src/datagen/datagen_run.jl --help
julia --project=. src/datagen/datagen_run.jl --phase configs --dry-run
julia --project=. src/datagen/datagen_run.jl --phase configs
julia -t auto --project=. src/datagen/datagen_run.jl --phase simulate --start 1 --stop 1 --tracks ground
julia -t auto --project=. src/datagen/datagen_run.jl --phase simulate --tracks precess
julia --project=. src/datagen/datagen_selftest.jl
```

## Repository Structure

```text
InhomogeneousSpinCavityDynamics.jl/
├── src/              Main simulation source code
│   ├── volkov_zon.jl Standalone analytical Volkov-Zon solver (stdlib only)
│   └── datagen/      PINN first-order datagen catalog (see src/datagen/README.md)
├── examples/         Example scripts for running simulations
├── scripts/          Plotting, noise, and correlation analysis scripts
├── paper/            Scripts for Obtaining data for figures in the paper (see citation)
├── data/             Saved JLD2 simulation data (including data/datagen/)
├── Project.toml      Julia package dependencies and compatibility
└── README.md         Repository documentation
```

- `src/`: Contains the first-order and second-order solvers, ensemble discretization, pulse definitions, validation functions, data-saving functions, and data-analysis functions. `src/datagen/` is the PINN first-order datagen catalog (see [src/datagen/README.md](src/datagen/README.md)). `src/volkov_zon.jl` is the standalone analytical Volkov–Zon solver (see [Analytical Volkov–Zon solver](#analytical-volkovzon-solver)).

- `examples/`: Contains example scripts showing how to define `SIM_SETTING`, `SYSTEM_CONFIG`, and `PULSE_CONFIG` and run a simulation.

- `scripts/`: Contains the main data-analysis programs:
  - `plot_1st_order.jl`
  - `plot_rose_1st_order.jl`
  - `plot_2nd_order.jl`
  - `compute_noise.jl`
  - `compute_correlations.jl`

- `paper/`: Contains scripts to get the data in the figures of the paper we published.

- `data/`: Default location for saved simulation results in JLD2 format.

- `Project.toml`: Lists the required Julia packages and supported Julia version.

## Reproducing the Paper Results

Run the following scripts from the root directory of the repository.

### Figure 3(b)

**Run:**

```bash
julia --project=. paper/fig_3_b/ace_under_various_c.jl
```

**Expected result:**  
Generates the simulation data $\langle \hat{a}_{out}(t) \rangle$ used for Figure 3(b).

### Figure 3(c)

**Run:**

```bash
julia --project=. paper/fig_3_c/silencing_factor.jl
```

**Expected result:**  
Generates the ACE amplitude used for Figure 3(c).

### Figure 3(d)

First generate the second-order simulation data.

**Run:**

For constant spin–cavity coupling:

```bash
julia --project=. paper/fig_3_d/rase_2nd_order_const_g.jl
```

For a Gaussian spin–cavity coupling distribution:

```bash
julia --project=. paper/fig_3_d/rase_2nd_order_gaussian_g.jl
```

Then set `INPUTS` in `scripts/compute_noise.jl` to the generated file or output folder and run:

```bash
julia --project=. scripts/compute_noise.jl
```

**Expected result:**  
Generates the main simulation data and the corresponding noise data used for Figure 3(d).

### Figure 4(c)

First generate the second-order simulation data.

**Run:**

```bash
julia --project=. paper/fig_4_c/rase_2nd_order_sweep_arp_duration.jl
```

Then set `INPUTS` in `scripts/compute_correlations.jl` to the generated file or output folder (also including main simulation data in Figure 3(d)) and run:

```bash
julia --project=. scripts/compute_correlations.jl
```

**Expected result:**  
Generates the main simulation data and the corresponding correlation data used for Figure 4(c).

### Figure 4(d)

First generate the second-order simulation data.

**Run:**

```bash
julia --project=. paper/fig_4_d/three_arp_pulses_c_0d6.jl
```

Then set `INPUTS` in `scripts/compute_correlations.jl` to the generated file or output folder (also including main simulation data in Figure 3(d)) and run:

```bash
julia --project=. scripts/compute_correlations.jl
```

**Expected result:**  
Generates the main simulation data and the corresponding correlation data used for Figure 4(d).

The generated JLD2 files are saved in the output directories specified in the scripts.

## Citation

If you use `InhomogeneousSpinCavityDynamics.jl` in your research, please cite the associated paper:

```bibtex
@article{hanamura2026inhomogeneous,
  title={Inhomogeneous Light-Matter Coupling as a Resource for Noiseless Quantum Memories},
  author={Hanamura, Fumiya and Bao, Sicheng and Yoo, Jie Lerk and Auff{\`e}ves, Alexia and Touzard, Steven},
  journal={arXiv preprint arXiv:2605.26783},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions, bug reports, or feature requests, please open an issue in the [GitHub repository](https://github.com/Qovelab/InhomogeneousSpinCavityDynamics.jl/issues).