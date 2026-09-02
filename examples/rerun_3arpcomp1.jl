using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using InhomogeneousSpinCavityDynamics
using Printf

try
    Base.buffer_writes(Base.stdout, 0)
catch
end

const JLD2_PATH = joinpath(@__DIR__, "..", "data", "data_1st_order",
                           "run_105_3ARP_M20000__3arpcomp1.jld2")
isfile(JLD2_PATH) || error("missing $JLD2_PATH")

t0 = time()

# ---------------------------------------------------------------------------
# Branch: new-silencing @ 7f61df7.
#
# Target: run_105_3ARP_M20000__3arpcomp1.jld2 -- the EQUAL-AMPLITUDE 3-segment
#   ARP built by generate_2n1_arp_from_jld2(...; target_silencing=1, n_pairs=1).
#   Reference metrics (from that build): inversion = 0.999999, |F|_* = 1.0.
#   So this is a POLISH target -- hold (inv, |F|_*) = (1, 1) while w_time trims
#   the 660us pulse -- the same regime as the successful 3ARP_pi_gstd run.
#   (The linear seed may land slightly off (1,1) due to a per-segment phi0
#   fit offset seen on the sibling file; if so, hop 0 + the basin-hop recover
#   it, as they did for run_105 v3.)
#
# Requested: param_budget=120, num_epochs=15, n_hops=4, patience=5,
#            grad_mode=:adjoint, track=:weak, use_interior=false,
#            hop0_phyonly=false.
#
# learning_rate = 0.018 (suggested).  Between:
#   * 0.015 -- the polish value that gave a clean monotone descent on the
#     (1,1)-start 3ARP_pi_gstd file.
#   * 0.02  -- the value that worked on run_105 v3's HARDER start (|F|_* 0.01
#     -> 1 via hop-0 physics + basin-hop).
#   0.018 covers both: the reference here is a polish target, but the seed
#   may start low-silencing.
#   cf_lr_scale = 0.3 : the chirp drives |F|_* (g-bin phase ~ integral Omega dt).
#
# fit_N=20001 : no _pulsemat sibling; force a fine analytic resample so the
#   18-coeff B-spline seed is faithful.
# reltol=1e-5 / abstol=1e-8 : 1100us window; run_105 v3 confirmed 1e-5 ~halves
#   per-epoch time with results intact.
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
    hop0_phyonly  = false,
    fit_N         = 20001,
    learning_rate = 0.018,
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
