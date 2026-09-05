using InhomogeneousSpinCavityDynamics
using Printf


SWEEP_OUTDIR = joinpath(
    @__DIR__,
    "..",
    "..",
    "data",
    "fig_3_d_mgpu",
)

mkpath(SWEEP_OUTDIR)

C_values = [
    0.05,
    0.2,
    0.4,
    0.6,
    0.8,
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



BASE_SYSTEM_CONFIG = (



    delta0  = 0.0,
    kappa_e = 2π * 1e6,
    kappa_i = 2π * 0.0,


    freq_inhomogeneity = (
        kind        = :lorentzian,
        FWHM        = 2π * 1e6,
        span_gamma  = 2.5,
        renormalize = false,
    ),


    g_inhomogeneity = (
        kind        = :constant,
        g_value     = 2π * 100.0,
        renormalize = false,
    ),
)



function make_number_tag(x)



    return replace(@sprintf("%.3f", x), "." => "p")
end



function build_pulse_config(SYSTEM_CONFIG)
    return (
        (
            name = "WURST pulse",
            kind = :wurst,

            t_center = 75e-6,
            duration = 10e-6,

            amp = (
                0.5 *
                sqrt(SYSTEM_CONFIG.kappa_e) *
                2.0e4
            ),

            bandwidth = (
                5.0 *
                SYSTEM_CONFIG.freq_inhomogeneity.FWHM
            ),

            n          = 20.0,
            omega0     = 0.0,
            chirp_sign = +1.0,
            phase0     = 0.0,
            edge_frac  = 1e-4,
        ),
    )
end



function run_sweep()
    N_runs = length(C_values)


    g_value_Hz = (
        BASE_SYSTEM_CONFIG.g_inhomogeneity.g_value /
        (2π)
    )

    println("============================================================")
    println("Starting C_ens sweep")
    println("Number of simulations: $N_runs")
    println("Constant g / 2π:       $g_value_Hz Hz")
    println("M_g:                   $(BASE_SIM_SETTING.M_g)")
    println("Output directory:      $SWEEP_OUTDIR")
    println("============================================================")

    for (run_index, C_ens) in enumerate(C_values)





        C_tag = make_number_tag(C_ens)

        filename = "C_$(C_tag).jld2"

        saved_file_name = joinpath(
            SWEEP_OUTDIR,
            filename,
        )

        println()
        println("------------------------------------------------------------")
        println("Run $run_index / $N_runs")
        println("C_ens       = $C_ens")
        println("g / 2π      = $g_value_Hz Hz")
        println("Output file = $saved_file_name")
        println("------------------------------------------------------------")





        if SKIP_EXISTING && isfile(saved_file_name)
            println("Output file already exists. Skipping this run.")
            continue
        end





        SIM_SETTING = merge(
            BASE_SIM_SETTING,
            (
                saved_file_name = saved_file_name,
            ),
        )





        SYSTEM_CONFIG = merge(
            BASE_SYSTEM_CONFIG,
            (
                C_ens = C_ens,
            ),
        )





        PULSE_CONFIG = build_pulse_config(SYSTEM_CONFIG)





        mgpu_run_simulation(
            SIM_SETTING,
            SYSTEM_CONFIG,
            PULSE_CONFIG,
        )

        println("Run $run_index finished.")
    end





    println()
    println("============================================================")
    println("C_ens sweep finished.")
    println("Results saved in:")
    println(SWEEP_OUTDIR)
    println("============================================================")

    return nothing
end


run_sweep()

nothing