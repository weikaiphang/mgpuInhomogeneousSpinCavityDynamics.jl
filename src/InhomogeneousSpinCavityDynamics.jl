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

# NCCL.jl / NCCL_jll need a working libnccl. Keep the package loadable
# on CPU-only hosts; multi-GPU then falls back to P2P or host staging.
const _NCCL = try
    Base.require(Base.PkgId(
        Base.UUID("3fe64909-d7a1-4096-9b7d-7a0f12cf0f6b"), "NCCL"))
catch
    nothing
end

include("config.jl")
include("frequency_inhomogeneity.jl")
include("coupling_inhomogeneity.jl")
include("pulses.jl")
include("ensemble.jl")
include("ensemble_quadrature.jl")

include("state_layout_1st_order.jl")
include("initial_conditions_1st_order.jl")
include("rhs_1st_order.jl")
include("rhs_1st_order_real.jl")
include("rhs_1st_order_ip.jl")
include("peak_detection_helpers.jl")
include("solver_1st_order.jl")

include("bspline.jl")
include("composite_pulse.jl")
include("canon_pulses.jl")
include("pulse_optimizer2.jl")
include("tsit5_discrete_adjoint.jl")
include("pulse_adjoint.jl")
include("pulse_optimizer2_RJMCMC.jl")
include("multi_seed_pulse_optimizer.jl")
include("jld2_pulse_loader.jl")
include("composite_arp_pulses.jl")
include("pulse_report.jl")

include("state_layout_2nd_order.jl")
include("initial_conditions_2nd_order.jl")
include("rhs_2nd_order.jl")
include("MGPUtableaus.jl")
include("solver_2nd_workspace.jl")
include("solver_2nd_order.jl")

include("MGPUlayout.jl")
include("MGPUdevices.jl")
include("MGPUkernels.jl")
include("MGPUproblem.jl")
include("MGPUintegrator.jl")
include("MGPUinitial_conditions.jl")
include("MGPUobservables.jl")
include("MGPUstate_io.jl")
include("MGPUrhs_cpu.jl")

include("noise.jl")
include("correlations.jl")

include("simulation_api.jl")
include("MGPUsolver.jl")

export run_simulation
export run_sim_1st_order
export run_sim_2nd_order
export build_u0_cpu_2nd_order
export rhs_2nd_order!
export rhs_cpu!
export Solver2Workspace, solve_cpu_2nd!, record_save2!, tsit5_cpu_step!, attach_u0!

export mgpu_run
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

export build_E_of_t
export sample_E_of_t
export save_E_samples
export save_run_data
export plot_E_of_t

export build_full_config, prepare_derived
export prepare_derived_quadrature, ensemble_method_for, resolve_ensemble_method
export make_clamped_knots, bspline_basis, bspline_eval, bspline_area, bspline_antiderivative
export CompositePulse, n_params, decode, initial_guess, total_area, pulse_duration
export points_per_segment_for_budget
export k_of_seed_kind, seed_hs1, seed_composite_with_ghosts, seed_corpse, seed_bb1, seed_bir4, seed_canonical
export run_sim_1st_order_pure, pulse_metrics, pulse_cost
export generate_interior_seed
export solve_optimal_x_start
export pulse_cost_grad_adjoint
export pulse_gpu_count
export AdamState, adam_step!, run_local_adam, optimise_composite_pulse
export optimise_composite_pulse_over_k

export optimise_composite_pulse_rjmcmc
export optimise_composite_pulse_over_k_rjmcmc

export multi_seed_optimise_pulse_rjmcmc

export load_jld2_run, split_signal_control, build_signal_E_of_t
export try_parse_pulse_config, load_jld2_reference, run_reference_forward
export reconcile_reference, fit_linear_seed
export jld2_pipeline_defaults, jld2_optimizer_defaults
export run_sim_1st_order_trajectory, run_sim_1st_order_final
export optrunlog_paths, save_optimisation_run_log, save_optimised_pulse_parameters
export optimise_control_pulse_from_jld2
export write_pulse_report
export segment_signal_control, segment_signal_control_from_trace
export identified_signal_control, control_envelope_E_of_t, signal_envelope_E_of_t
export SignalControlRejected

export generate_2n1_arp_pi_pulse
export generate_2n1_arp_from_jld2
export generate_3arp_pi_pulse

export NoiseSimulationData
export load_noise_data
export print_noise_data_summary
export compute_output_mode_noise
export compute_noise_windows
export compute_noise_from_file
export print_noise_result

export CorrelationSettings
export CorrelationWorkspace
export load_correlation_workspace
export compute_ase_rase_correlations_gpu
export save_correlation_results

end
