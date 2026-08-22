module InhomogeneousSpinCavityDynamics

using CUDA
using DiffEqCallbacks
using DifferentialEquations
using Distributions
using ForwardDiff
using JLD2
using LinearAlgebra
using Plots
using Printf
using QuadGK
using Random

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

# Differentiable composite pi-pulse optimisation (ForwardDiff/Adam +
# basin-hopping port of InhomogeneousSpinCavityDynamics.py's
# pulse_optimized_spline.py, driving THIS package's own rhs_1st_order!/
# prepare_derived rather than the Python side's simplified toy model)
include("bspline.jl")
include("composite_pulse.jl")
include("canon_pulses.jl")
include("pulse_optimizer2.jl")
include("pulse_optimizer2_RJMCMC.jl")
include("multi_seed_pulse_optimizer.jl")
include("jld2_pulse_loader.jl")
include("composite_arp_pulses.jl")

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
export save_E_samples
export save_run_data
export plot_E_of_t

# Differentiable composite pi-pulse optimisation exports
export build_full_config, prepare_derived  # needed to build the `d` argument below
export make_clamped_knots, bspline_basis, bspline_eval, bspline_area, bspline_antiderivative
export CompositePulse, n_params, decode, initial_guess, total_area, pulse_duration
export k_of_seed_kind, seed_hs1, seed_composite_with_ghosts, seed_corpse, seed_bb1, seed_canonical
export run_sim_1st_order_pure, pulse_metrics, pulse_cost
export AdamState, adam_step!, run_local_adam, optimise_composite_pulse
export optimise_composite_pulse_over_k

# Trans-dimensional (RJMCMC-flavoured) k-hopping extension exports
# (pulse_optimizer2_RJMCMC.jl) -- distinct names from the fixed-k API
# above since `pulse.k` is NOT guaranteed to equal the input `k` for
# these (see that file's own module docstring)
export optimise_composite_pulse_rjmcmc
export optimise_composite_pulse_over_k_rjmcmc

# Multi-seed wrapper (multi_seed_pulse_optimizer.jl): one RJMCMC run per
# starting k against the same ensemble, each canonically seeded where a
# named form exists
export multi_seed_optimise_pulse_rjmcmc

# JLD2-driven signal/control pulse optimisation exports
export load_jld2_run, split_signal_control, build_signal_E_of_t
export run_sim_1st_order_trajectory, reconcile_against_jld2
export optrunlog_paths, save_optimisation_run_log, save_optimised_pulse_parameters
export optimise_control_pulse_from_jld2
export optimise_control_pulse_from_jld2_over_k

# Analytic composite ARP pulse exports
export generate_3arp_pi_pulse

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
