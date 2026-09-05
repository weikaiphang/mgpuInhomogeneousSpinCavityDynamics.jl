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
