using InhomogeneousSpinCavityDynamics
using Printf



SWEEP_OUTDIR = joinpath(
    @__DIR__,
    "..",
    "..",
    "data",
    "fig_4_c_mgpu",
)

mkpath(SWEEP_OUTDIR)

WURST_duration_us_values = [
    10.0,
    30.0,
    100.0,
]

SKIP_EXISTING = true



BASE_SIM_SETTING = (
    simulation_order = :order2,

    M_delta = 375,
    M_g     = 1,

    initial_condition = :inverted,

    Ttotal = 150e-6,

    Nt_save = 5001,
    reltol  = 1e-8,
    abstol  = 1e-8,
)



SYSTEM_CONFIG = (
    C_ens = 0.6,

    delta0 = 0.0,
    kappa_e = 2π * 1e6,
    kappa_i = 2π * 0.0,

    freq_inhomogeneity = (
        kind = :lorentzian,
        FWHM = 2π * 1e6,
        span_gamma = 2.5,
        renormalize = false,
    ),

    g_inhomogeneity = (
        kind = :constant,
        g_value = 2π * 100.0,
        renormalize = false,
    ),
)



function make_duration_tag(duration_us)
    return @sprintf("%03dus", round(Int, duration_us))
end



function build_pulse_config(system_config, duration_us)
    return (
        (
            name = "WURST pulse",
            kind = :wurst,

            t_center = 75e-6,
            duration = duration_us * 1e-6,

            amp = (
                0.5 *
                sqrt(system_config.kappa_e) *
                2.0e4
            ),

            bandwidth = (
                5.0 *
                system_config.freq_inhomogeneity.FWHM
            ),

            n          = 20.0,
            omega0     = 0.0,
            chirp_sign = +1.0,
            phase0     = 0.0,
            edge_frac  = 1e-4,
        ),
    )
end



function run_duration_sweep(
    duration_values,
    base_sim_setting,
    system_config,
    sweep_outdir;
    skip_existing = true,
)
    N_runs = length(duration_values)

    println("============================================================")
    println("Starting WURST-duration sweep")
    println("Number of simulations: $N_runs")
    println("Fixed C_ens:           $(system_config.C_ens)")
    println(
        "Fixed g / 2π:          ",
        system_config.g_inhomogeneity.g_value / (2π),
        " Hz",
    )
    println("M_g:                   $(base_sim_setting.M_g)")
    println("Output directory:      $sweep_outdir")
    println("============================================================")

    for (run_index, duration_us) in enumerate(duration_values)
        duration_tag = make_duration_tag(duration_us)

        filename = "WURST_duration_$(duration_tag).jld2"
        saved_file_name = joinpath(sweep_outdir, filename)

        println()
        println("------------------------------------------------------------")
        println("Run $run_index / $N_runs")
        println("WURST duration = $duration_us μs")
        println("C_ens          = $(system_config.C_ens)")
        println(
            "g / 2π         = ",
            system_config.g_inhomogeneity.g_value / (2π),
            " Hz",
        )
        println("Output file    = $saved_file_name")
        println("------------------------------------------------------------")

        if skip_existing && isfile(saved_file_name)
            println("Output file already exists. Skipping this run.")
            continue
        end

        sim_setting = merge(
            base_sim_setting,
            (
                saved_file_name = saved_file_name,
            ),
        )

        pulse_config = build_pulse_config(
            system_config,
            duration_us,
        )

        mgpu_run_simulation(
            sim_setting,
            system_config,
            pulse_config,
        )

        println("Run $run_index finished.")
    end

    println()
    println("============================================================")
    println("WURST-duration sweep finished.")
    println("Results saved in:")
    println(sweep_outdir)
    println("============================================================")

    return nothing
end



run_duration_sweep(
    WURST_duration_us_values,
    BASE_SIM_SETTING,
    SYSTEM_CONFIG,
    SWEEP_OUTDIR;
    skip_existing = SKIP_EXISTING,
)

nothing