module InhomogeneousSpinCavityDynamics

using CUDA
using DiffEqCallbacks
using DifferentialEquations
using Distributions
using FastGaussQuadrature
using ForwardDiff
using JLD2
using LinearAlgebra
using OrdinaryDiffEq
using Plots
using Printf
using QuadGK
using Random

using QuantumCumulants
using SecondQuantizedAlgebra

const HAVE_NCCL = try
    @eval using NCCL
    true
catch
    false
end

include("chimera/packages.jl")

# QRT postprocessor: Jacobian of the QuantumCumulants / factorized 1st-order
# Tavis–Cummings EOMs, not a linearization of the 2nd-order cumulant RHS.
const QRT_CLOSURE_LEVEL = :factorized_first_order_jacobian

include("config.jl")
include("pulses.jl")
include("chimera/include_all.jl")

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

include("correlations.jl")

export run_simulation
export run_sim_1st_order
export run_sim_2nd_order
export product_SmSp_same, product_SzSz_same, product_SpSp_same, product_SzSp_same
export product_state_samebin, build_u0_2nd_order

export mgpu_run_simulation
export mgpu_run_sim_2nd_order
export assemble_problem
export choose_rowsum_exchange, assemble_rowsums!
export assemble_rowsums_nccl!, assemble_rowsums_p2p!, assemble_rowsums_host!
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
export truncation_cooperativity, maybe_print_truncation_cooperativity
export QRT_CLOSURE_LEVEL
export CHIMERA_HAMILTONIAN, chimera_hamiltonian_symbols
export fgq_gausslegendre
export derive_tc_meanfield, qc_algebra_selftest
export compare_qc_to_closure_1st, solve_qc_sciml_1st_M1
export factorized_first_order_jacobian_action!
export chimera_solve, chimera_ode_problem, CHIMERA_INTEGRATOR
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
