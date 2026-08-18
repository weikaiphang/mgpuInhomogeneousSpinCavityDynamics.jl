module InhomogeneousSpinCavityDynamics

using CUDA
using DiffEqCallbacks
using DifferentialEquations
using Distributions
using JLD2
using LinearAlgebra
using Printf
using QuadGK

# Configurations
include("config.jl")
include("frequency_inhomogeneity.jl")
include("coupling_inhomogeneity.jl")
include("pulses.jl")
include("ensemble.jl")

# 1st-order simulation
include("state_layout_1st_order.jl")
include("initial_conditions_1st_order.jl")
include("rhs_1st_order.jl")
include("peak_detection_helpers.jl")
include("solver_1st_order.jl")

# 2nd-order simulation (single-GPU)
include("state_layout_2nd_order.jl")
include("initial_conditions_2nd_order.jl")
include("rhs_2nd_order.jl")
include("solver_2nd_order.jl")

# 2nd-order simulation (multi-GPU, ported from mgpu_InhomogeneousSpinCavityDynamics.jl).
# Load order matches that package: layout/devices/kernels/tableaus define the
# primitives problem.jl builds on; integrator.jl drives problem.jl; the rest are
# independent of each other. Names that collided with the single-GPU API above
# (run_simulation, run_sim_2nd_order, build_full_config) were renamed with an
# mgpu_ prefix in MGPUsolver.jl; nothing else needed renaming.
include("MGPUlayout.jl")
include("MGPUdevices.jl")
include("MGPUkernels.jl")
include("MGPUtableaus.jl")
include("MGPUproblem.jl")
include("MGPUintegrator.jl")
include("MGPUinitial_conditions.jl")
include("MGPUobservables.jl")
include("MGPUstate_io.jl")
include("MGPUrhs_cpu.jl")

# Quantum Regression Theorem for noise and correlations
include("noise.jl")
include("correlations.jl")

# User-facing API
include("simulation_api.jl")
include("MGPUsolver.jl")

# Main simulation exports (single-GPU)
export run_simulation
export run_sim_1st_order
export run_sim_2nd_order

# Multi-GPU 2nd-order simulation exports (ported from
# mgpu_InhomogeneousSpinCavityDynamics.jl; mgpu_ prefix marks the two names
# that collided with the single-GPU API above)
export mgpu_run_simulation
export mgpu_run_sim_2nd_order
export assemble_problem
export free_shards!
export set_initial_condition!
export scatter_state!, gather_state
export rhs!, rhs_cpu!
export solve_mgpu!
export memory_report, max_bins, EnsemblePartition
export SolverOptions, ObservableStore

# Pulse exports
export build_E_of_t
export sample_E_of_t

# Noise exports
export NoiseSimulationData
export load_noise_data
export print_noise_data_summary
export compute_output_mode_noise
export compute_noise_windows
export compute_noise_from_file
export print_noise_result

# Correlations exports
export CorrelationSettings
export CorrelationWorkspace
export load_correlation_workspace
export compute_ase_rase_correlations_gpu
export save_correlation_results

end
