using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using InhomogeneousSpinCavityDynamics
using Printf

try
    Base.buffer_writes(Base.stdout, 0)
catch
end

const JLD2_PATH = joinpath(@__DIR__, "..", "data", "data_1st_order", "3ARP_pi_gstd_1em06Hz.jld2")
isfile(JLD2_PATH) || error("missing $JLD2_PATH")

t0 = time()

# ---------------------------------------------------------------------------
# Branch: new-silencing @ 16270da.  Present:
#   annealing, gradient-wall with moving target, interior-seeding (now
#   OPT-IN; pipeline default use_interior=false), updated paper silencing
#   |F|_* (per frequency slice, weak :weak seed), single-track (track),
#   hop0_phyonly gate, and auto :ground winner re-check for track=:weak.
#
# This run finally does the "polish a near-optimal pulse" setup:
#
#   use_interior = false  -> warm start = the FAITHFUL linear seed of the
#     3ARP (rel_l2_complex ~ 0.007), i.e. inversion ~ 1, silencing = 1,
#     cost ~ 0.2.  No interior-seed amplitude-halving hole to dig out of.
#
#   hop0_phyonly = false  -> hop 0 is a NORMAL scheduled hop: it calibrates
#     x_tune and runs the full annealed cost (fidelity + walls + w_time*
#     duration + w_tmax + w_power) from epoch 1, instead of the default
#     physics-only sandbox (w_time pinned 0) that would then hand a jarring
#     schedule switch to hop 1.  From a near-optimal start the task is a
#     well-posed local polish: hold inv ~ 1 / sil = 1 while w_time trims
#     the pulse.
#
# Requested: param_budget=120, num_epochs=10, n_hops=3, patience=5,
#            grad_mode=:adjoint, track=:weak, use_interior=false,
#            hop0_phyonly=false.
#
# learning_rate = 0.015 (suggested).  Rationale:
#   * Every prior run's "lr too hot -> sawtooth" was from starting DEEP in
#     the infeasible region (cost 8-27) with STEEP ACTIVE walls.  Here the
#     walls are SLACK (inv, sil both above the 0.85 floors) so the gradient
#     is the smooth (1-fid)^2 + w_time*duration + w_power terms -- much
#     better conditioned.  A larger step than the 0.008-0.012 of the
#     deep-hole runs is safe and needed to make the pulse-trim visible in
#     10 epochs.  0.015 stays well under the 0.05 default so a step won't
#     fling inversion below 0.85 and re-activate the wall.
#   * cf_lr_scale = 0.3 : 18 chirp coeffs/segment at param_budget=120, but
#     near a good optimum the chirp is already right -> less 2*pi-runaway
#     risk than the deep-hole runs (which used 0.1).
#   * Tuning read, hop-0 ep 1-3: inversion holds ~1 and cost creeps down
#     (pulse shortening) -> can raise to 0.02;  inversion drops below ~0.9
#     (wall activating) -> drop to 0.008.
#
# reltol=1e-6 / abstol=1e-8 : resolve the ~0.28/bin weak :weak track.
# Everything else = branch defaults (kappa=50, target_F=1.0, w_time=0.15,
#   hop_step_size=0.5, x_tune_alpha=0.025, anneal on).
# ---------------------------------------------------------------------------
res = optimise_control_pulse_from_jld2(
    JLD2_PATH;
    verbose       = true,
    param_budget  = 120,
    num_epochs    = 10,
    n_hops        = 3,
    patience      = 5,
    grad_mode     = :adjoint,
    track         = :weak,
    use_interior  = false,
    hop0_phyonly  = false,
    learning_rate = 0.015,
    cf_lr_scale   = 0.3,
    reltol        = 1e-6,
    abstol        = 1e-8,
)

best_u, best_cost, pulse, signal_E_of_t, d, data, seed_fit_report, reference_metrics = res

println()
println("=== DONE in $(round(time() - t0; digits=1)) s ===")
@printf("  pulse shape        : k=%d  n_coeff_A=%d  n_coeff_f=%d  n_params=%d\n",
        pulse.k, pulse.n_coeff_A, pulse.n_coeff_f, n_params(pulse))
@printf("  reference metrics  : inversion=%.6g  silencing=%.6g  coherence=%.6g  duration=%.6g s\n",
        reference_metrics.inversion, reference_metrics.silencing,
        reference_metrics.coherence, reference_metrics.duration)
@printf("  optimised best_cost (track=:weak): %.6g\n", best_cost)
flush(stdout)
