using InhomogeneousSpinCavityDynamics
using Printf


OUTDIR = joinpath(@__DIR__, "..", "..", "data", "fig_3_b_mgpu")
mkpath(OUTDIR)


C_VALUES = [
    0.1,
    0.6,
]



SIM_SETTING_BASE = (
    simulation_order = :order1,  

    M_delta = 3000,
    M_g     = 1,

    initial_condition = :ground,  

    Ttotal = 1100e-6,

    Nt_save = 5001,
    reltol  = 1e-8,
    abstol  = 1e-8,

    saved_file_name = joinpath(OUTDIR, "temporary.jld2"),

    peak_detection = (
        labels = [:echo1, :echo2],
        times = [550e-6, 1050e-6],
        half_window = 10e-6,
    ),
)


SYSTEM_CONFIG_BASE = (
    C_ens = first(C_VALUES),

    delta0 = 0.0,
    kappa_e = 2 * pi * 1e6,
    kappa_i = 2 * pi * 0,

    freq_inhomogeneity = (
        kind = :lorentzian,
        FWHM = 2 * pi * 1e6,
        span_gamma = 2.5,
        renormalize = false,
    ),

    g_inhomogeneity = (
        kind = :gaussian,
        mean = 2 * pi * 100,
        std  = 2 * pi * 1,
        span_sigma = 3.0,
        renormalize = true,
    ),
)


PULSE_CONFIG = (
    (
        name  = "Gaussian input signal",
        kind  = :gaussian,

        t0    = 50e-6,
        sigma = 3e-6,
        amp   = 0.5 * sqrt(SYSTEM_CONFIG_BASE.kappa_e) * 0.332,
        omega = 0.0,
        phase = 0.0,
    ),

    (
        name       = "First WURST pulse",
        kind       = :wurst,

        t_center   = 300e-6,
        duration   = 400e-6,
        amp        = 0.5 * sqrt(SYSTEM_CONFIG_BASE.kappa_e) * 2.0e4,
        bandwidth  = 5.0 * SYSTEM_CONFIG_BASE.freq_inhomogeneity.FWHM,
        n          = 20.0,
        omega0     = 0.0,
        chirp_sign = +1.0,
        phase0     = 0.0,
        edge_frac  = 1e-4,
    ),

    (
        name       = "Second WURST pulse",
        kind       = :wurst,

        t_center   = 800e-6,
        duration   = 400e-6,
        amp        = 0.5 * sqrt(SYSTEM_CONFIG_BASE.kappa_e) * 2.0e4,
        bandwidth  = 5.0 * SYSTEM_CONFIG_BASE.freq_inhomogeneity.FWHM,
        n          = 20.0,
        omega0     = 0.0,
        chirp_sign = +1.0,
        phase0     = 0.0,
        edge_frac  = 1e-4,
    ),
)


function run_C_sweep(
    C_VALUES,
    SIM_SETTING_BASE,
    SYSTEM_CONFIG_BASE,
    PULSE_CONFIG,
    OUTDIR,
)
    println("Starting cooperativity sweep.")
    println("Number of simulations: $(length(C_VALUES))")
    println("Output directory: $OUTDIR")
    println()

    for (run_index, C_value) in enumerate(C_VALUES)

        C_label = replace(@sprintf("%.3f", C_value), "." => "p")

        saved_file_name = joinpath(
            OUTDIR,
            "order1_C_$(C_label).jld2",
        )

        SYSTEM_CONFIG_RUN = merge(
            SYSTEM_CONFIG_BASE,
            (
                C_ens = C_value,
            ),
        )

        SIM_SETTING_RUN = merge(
            SIM_SETTING_BASE,
            (
                saved_file_name = saved_file_name,
            ),
        )

        println("============================================================")
        println("Run $run_index / $(length(C_VALUES))")
        println("C_ens = $C_value")
        println("Saving to: $saved_file_name")
        println("============================================================")

        mgpu_run_simulation(
            SIM_SETTING_RUN,
            SYSTEM_CONFIG_RUN,
            PULSE_CONFIG,
        )

        println("Finished C_ens = $C_value")
        println()
    end

    println("All cooperativity simulations finished.")
    println("Saved files are in:")
    println(OUTDIR)

    return nothing
end

run_C_sweep(
    C_VALUES,
    SIM_SETTING_BASE,
    SYSTEM_CONFIG_BASE,
    PULSE_CONFIG,
    OUTDIR,
)

nothing