using InhomogeneousSpinCavityDynamics, JLD2, Printf
const MOD = InhomogeneousSpinCavityDynamics
Base.eval(MOD, :(include("/home/weika/projects/InhomogeneousSpinCavityDynamics.jl/src/composite_arp_pulses___mod.jl")))

base = load("/home/weika/projects/InhomogeneousSpinCavityDynamics.jl/data/sweep_1st_order/run_105.jld2")["data"]
SIM  = merge(base.SIM_SETTING, (M_delta = 500, M_g = 20, saved_file_name = "unused.jld2"))
SYS  = base.SYSTEM_CONFIG
d = prepare_derived(build_full_config(SIM, SYS))
@assert d.M == 10000

println("M_delta x M_g            = ", SIM.M_delta, " x ", SIM.M_g, " = ", d.M)
println("Ttotal                  = ", SIM.Ttotal)
println("freq_inhomogeneity      = ", d.freq_inhomogeneity)
println("g_inhomogeneity         = ", d.g_inhomogeneity)
println("d.FWHM (mesh)            = ", d.FWHM)
println("d.freq_inhomogeneity.FWHM = ", d.freq_inhomogeneity.FWHM)
println("d.g_mean (mesh)          = ", d.g_mean)
println("d.g_inhomogeneity.mean   = ", d.g_inhomogeneity.mean)
println("kappa_e / kappa_i        = ", d.kappa_e, " / ", d.kappa_i)
println("N_total                  = ", d.N_total)

for tsil in (1, 0)
    println("\n", "="^64)
    println("target_silencing = ", tsil, "   (n_pairs = 1)")
    println("="^64)
    t0 = time()
    pc, r = generate_2n1_arp_pi_pulse(d; n_pairs = 1, target_silencing = tsil, compute = :cpu)
    dt = time() - t0
    println("PULSE_CONFIG (", length(pc), " segment", length(pc) == 1 ? "" : "s", "):")
    for (i, s) in enumerate(pc)
        @printf("  [%d] %-34s  t_center=%8.2f us  dur=%8.2f us  amp=%.6e  bw=%.6e\n",
                i, s.name, s.t_center*1e6, s.duration*1e6, s.amp, s.bandwidth)
    end
    println("report:")
    for (k, v) in pairs(r)
        println("  ", rpad(k, 20), " => ", v)
    end
    @printf("solve wall time: %.1f s\n", dt)
end
println("\nDONE")
