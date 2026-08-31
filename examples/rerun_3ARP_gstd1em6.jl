using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using InhomogeneousSpinCavityDynamics
using Printf

# Best-effort unbuffered stdout. The launcher also runs this under a PTY
# (`script`), which is what actually makes Julia line-buffer.
try
    Base.buffer_writes(Base.stdout, 0)
catch
end

const JLD2_PATH = joinpath(@__DIR__, "..", "data", "data_1st_order", "3ARP_pi_gstd_1em06Hz.jld2")
isfile(JLD2_PATH) || error("missing $JLD2_PATH")

t0 = time()

# --- run knobs -----------------------------------------------------------------
# Changes vs the 2-hop / 8-epoch run (final cost 21.47, inversion only 0.355 --
# under-converged, and hop 1 threw away hop 0's progress):
#
#   num_epochs   8   -> 20      hop 0's descent was clean and still improving at ep 7
#   hop_step_size 0.5 -> 0.15   the perturbation that kicked hop 1 out of hop 0's
#                               basin (6 straight worsening epochs, rejected).
#                               Smaller kick keeps annealing + moving-w_time
#                               target (hop 1 only) working on a NEARBY basin.
#   learning_rate 0.03 -> 0.05  hop 0 climbed ~0.015 inversion/epoch, monotone,
#                               not oscillatory -- room to step harder.
#   kappa_I/kappa_S 50 -> 15    soften the squared-hinge walls so the fidelity /
#                               time / power terms are not ~98% swamped while
#                               inversion is still climbing to the 0.85 floor.
#   patience      5   -> 8      more slack across a 20-epoch hop.
#
# This run: ONLY param_budget changes, 60 -> 120 (k=3 => nA=nf=18, 117 params,
# ~2x the B-spline resolution/segment).  Everything else identical to the
# pb60 run (final cost 7.09, inv 0.31) so the budget effect is isolated.
#
# Kept: grad_mode=:adjoint, num_epochs=20, n_hops=2, patience=8,
#       hop_step_size=0.15, learning_rate=0.05, cf_lr_scale=0.3,
#       kappa_I=kappa_S=15, reltol=abstol=1e-6, use_interior=true.
res = optimise_control_pulse_from_jld2(
    JLD2_PATH;
    verbose               = true,
    param_budget          = 120,
    num_epochs            = 20,
    n_hops                = 2,
    patience              = 8,
    hop_step_size          = 0.15,
    grad_mode             = :adjoint,
    learning_rate         = 0.05,
    cf_lr_scale           = 0.3,
    kappa_I               = 15.0,
    kappa_S               = 15.0,
    anneal_direct_weights = true,
    use_interior          = true,
    reltol                = 1e-6,
    abstol                = 1e-6,
)

best_u, best_cost, pulse, signal_E_of_t, d, data, seed_fit_report, reference_metrics = res

println()
println("=== DONE in $(round(time() - t0; digits=1)) s ===")
@printf("  pulse shape        : k=%d  n_coeff_A=%d  n_coeff_f=%d  n_params=%d\n",
        pulse.k, pulse.n_coeff_A, pulse.n_coeff_f, n_params(pulse))
@printf("  reference metrics  : inversion=%.6g  silencing=%.6g  coherence=%.6g  duration=%.6g s\n",
        reference_metrics.inversion, reference_metrics.silencing,
        reference_metrics.coherence, reference_metrics.duration)
@printf("  optimised best_cost: %.6g\n", best_cost)
flush(stdout)
