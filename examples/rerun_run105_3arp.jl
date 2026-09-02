using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using InhomogeneousSpinCavityDynamics
using Printf

try
    Base.buffer_writes(Base.stdout, 0)
catch
end

const JLD2_PATH = joinpath(@__DIR__, "..", "data", "data_1st_order", "run_105_3ARP_M20000.jld2")
isfile(JLD2_PATH) || error("missing $JLD2_PATH")

t0 = time()

# ---------------------------------------------------------------------------
# Branch: new-silencing @ 16270da  (annealing, gradient-wall + moving target,
#   interior-seeding [opt-in], updated paper |F|_* silencing, single-track,
#   hop0_phyonly gate, auto :ground winner re-check).
#
# Target: run_105_3ARP_M20000.jld2 -- a datagen 3ARP run with REAL g-spread
#   (g_std/g_mean = 0.5%, C_ens 0.33).  Reference metrics under the new formula:
#     :ground  inversion   = 1.0
#     :weak    silencing|F|*= 0.0665     <- far below 1
#   So this is the genuine job: hold inversion = 1 (slack) while driving |F|_*
#   from 0.066 toward 1.  Starting cost ~14 (silencing_success wall active).
#   No Gaussian signal -> all 3 WURSTs are the control envelope.
#
# Requested: param_budget=120, num_epochs=15, n_hops=4, patience=5,
#            grad_mode=:adjoint, track=:weak, use_interior=false,
#            hop0_phyonly=true  (hop 0 = physics-only sandbox: descend the
#              silencing_success wall with no time penalty, then hops 1-3
#              add the annealed w_time -- a good split here because silencing
#              must climb 0.29 -> 1 and w_time would otherwise fight it).
#            reltol=1e-5  (loosened from 1e-6 to ~halve the ~10-14h runtime
#              the 1100us window projected; abstol=1e-8 kept as the floor).
#            fit_N=20001  (default resampled the analytic 3ARP at only the
#              sibling CSV's 5001 rows -> points_per_segment=100 -> a poor
#              chirp fit: rel_l2_complex=1.42, phi_rms=0.54 rad, and the seed
#              landed at inversion 0.50 / silencing 1.0 instead of the
#              reference's 1.0 / 0.066.  20001 gives ~400 pts/segment so the
#              18-coeff B-spline fit is faithful and the warm start IS the
#              3ARP -> the run is the intended "climb |F|_* from 0.066".)
#
# learning_rate = 0.02 (suggested).  Between regimes:
#   * NOT 0.015 (that was for a cost-~1.2 near-optimal start; here there's a
#     real ~14 -> ? descent to make in 15 epochs/hop).
#   * NOT the 0.05 default (diverged in every deep-infeasible run).
#   * track=:weak = one coherent exact gradient; |F|_* is well-conditioned
#     (per-slice, [0,1]); ONLY silencing needs to move (inversion slack) ->
#     no two-objective conflict -> 0.02 should descend cleanly.
#   cf_lr_scale = 0.3 : the CHIRP drives |F|_* (g-bin phase ~ integral Omega dt),
#     so the cf coeffs need real room -- not the tight 0.1 of the polish run.
#   Hop-0 read: silencing climbs monotone -> hold / try 0.03;  cost or
#     silencing swings > ~0.5 -> drop to 0.01.
#
# reltol=1e-6 / abstol=1e-8 : resolve the ~0.28/bin weak :weak track.
#   NOTE 1100us window => ~600s/epoch; ~40-60 epochs after early stops => ~6-9h.
# Everything else = branch defaults (kappa=50, target_F=1.0, w_time=0.15,
#   hop_step_size=0.5, x_tune_alpha=0.025, anneal on).
# ---------------------------------------------------------------------------
res = optimise_control_pulse_from_jld2(
    JLD2_PATH;
    verbose       = true,
    param_budget  = 120,
    num_epochs    = 15,
    n_hops        = 4,
    patience      = 5,
    grad_mode     = :adjoint,
    track         = :weak,
    use_interior  = false,
    hop0_phyonly  = true,
    fit_N         = 20001,
    learning_rate = 0.02,
    cf_lr_scale   = 0.3,
    reltol        = 1e-5,
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
