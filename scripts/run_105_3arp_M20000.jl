using InhomogeneousSpinCavityDynamics
using JLD2

const BASE   = joinpath(@__DIR__, "..", "data", "sweep_1st_order", "run_105.jld2")
const OUTPUT = joinpath(@__DIR__, "..", "data", "data_1st_order", "run_105_3ARP_M20000.jld2")

base = load(BASE)["data"]
SIM_SETTING   = merge(base.SIM_SETTING, (saved_file_name = OUTPUT,))
SYSTEM_CONFIG = base.SYSTEM_CONFIG

@assert SIM_SETTING.simulation_order === :order1
@assert SIM_SETTING.M_delta * SIM_SETTING.M_g == 20000 "M = $(SIM_SETTING.M_delta)*$(SIM_SETTING.M_g) != 20000"

d = prepare_derived(build_full_config(SIM_SETTING, SYSTEM_CONFIG))
@assert d.M == 20000

println("Base run      : ", BASE)
println("Output        : ", OUTPUT)
println("M_delta x M_g : ", SIM_SETTING.M_delta, " x ", SIM_SETTING.M_g, " = ", d.M)
println("Ttotal        : ", SIM_SETTING.Ttotal)
println("C_ens         : ", SYSTEM_CONFIG.C_ens)
println("freq_inhomog  : ", SYSTEM_CONFIG.freq_inhomogeneity)
println("g_inhomog     : ", SYSTEM_CONFIG.g_inhomogeneity)
println("N_total       : ", d.N_total)

bw       = 5.0 * d.FWHM
Tb       = 0.6 * (d.timespan[2] - d.timespan[1])
dur_odd  = Tb / 4
dur_even = 2 * dur_odd
amp_scale    = d.kappa_t / (4 * d.g_mean * d.sqrt_kappa_e)
Omega_target = pi * d.FWHM
amp_odd  = amp_scale * Omega_target
amp_even = amp_odd / sqrt(2)

t_start = 0.0
starts  = (t_start, t_start + dur_odd, t_start + dur_odd + dur_even)
durs    = (dur_odd, dur_even, dur_odd)
amps    = (amp_odd, amp_even, amp_odd)
names   = ("3ARP segment 1 (+k)", "3ARP segment 2 (+k/2)", "3ARP segment 3 (+k)")
common  = (n = 20.0, omega0 = 0.0, chirp_sign = 1.0, phase0 = 0.0, edge_frac = 1e-4)

ARP_PULSE_CONFIG = ntuple(3) do i
    merge((name = names[i], kind = :wurst,
           t_center = starts[i] + durs[i] / 2, duration = durs[i],
           amp = amps[i], bandwidth = bw), common)
end
t_end = starts[3] + durs[3]
@assert t_end <= d.timespan[2] "3-ARP ends at $(t_end) past window $(d.timespan[2])"

println("\n3-ARP pulse  (bw = ", bw, ",  span [0, ", t_end * 1e6, "] us):")
for (i, s) in enumerate(ARP_PULSE_CONFIG)
    println("  seg $i  ", s.name,
            "  t_center=", s.t_center * 1e6, " us  duration=", s.duration * 1e6,
            " us  amp=", s.amp)
end

stub = (
    SIM_SETTING   = SIM_SETTING,
    SYSTEM_CONFIG = SYSTEM_CONFIG,
    PULSE_CONFIG  = ARP_PULSE_CONFIG,
    base_run      = "run_105",
    arp_params    = (bandwidth = bw, T_budget = Tb, dur_odd = dur_odd,
                     dur_even = dur_even, amp_odd = amp_odd, amp_even = amp_even,
                     t_start = t_start, t_end = t_end),
    status        = "config-only; simulation pending",
)
mkpath(dirname(OUTPUT))
jldsave(OUTPUT; data = stub)
println("\nStub written : ", OUTPUT, "  (", filesize(OUTPUT), " bytes)")

println("\nStarting 1st-order GPU simulation (M = 20000, Ttotal = ",
        SIM_SETTING.Ttotal, ")...")
flush(stdout)
data = run_simulation(SIM_SETTING, SYSTEM_CONFIG, ARP_PULSE_CONFIG; clean_gpu = true)

save_run_data(OUTPUT, merge(data, (
    base_run   = "run_105",
    arp_params = stub.arp_params,
)))

inv_collective = (real(data.Σz_sol[end]) + data.N_total / 2) / data.N_total
println("\nDone. Final file : ", OUTPUT, "  (", filesize(OUTPUT), " bytes)")
println("Σz_sol[end]              = ", data.Σz_sol[end])
println("Σp_sol[end]              = ", data.Σp_sol[end])
println("collective inversion     = ", inv_collective, "   (0 = ground, 1 = fully inverted)")
println("|a_sol[end]|             = ", abs(data.a_sol[end]))
