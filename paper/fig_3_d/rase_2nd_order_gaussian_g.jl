using InhomogeneousSpinCavityDynamics
using Printf



SWEEP_OUTDIR = joinpath(
    @__DIR__,
    "..",
    "..",
    "data",
    "fig_3_d",
)

mkpath(SWEEP_OUTDIR)

C_values = [
    0.05,
    0.2,
    0.4,
    0.6,
    0.8,
]

g_std_Hz_values = [
    0.5,
    1.0,
    5.0,
]

SKIP_EXISTING = true



BASE_SIM_SETTING = (

    simulation_order = :order2,


    M_delta = 375,
    M_g     = 20,


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
        kind        = :gaussian,
        mean        = 2π * 100.0,
        std         = 2π * 1.0,
        span_sigma  = 3.0,
        renormalize = PAPER_G_RENORMALIZE,
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
    N_C     = length(C_values)
    N_std   = length(g_std_Hz_values)
    N_runs  = N_C * N_std

    println("============================================================")
    println("Starting C_ens and g_std sweep")
    println("Number of C values:     $N_C")
    println("Number of g_std values: $N_std")
    println("Total simulations:      $N_runs")
    println("Output directory:       $SWEEP_OUTDIR")
    println("============================================================")


    sweep_points = Iterators.product(
        C_values,
        g_std_Hz_values,
    )

    for (run_index, sweep_point) in enumerate(sweep_points)
        C_ens, g_std_Hz = sweep_point





        C_tag   = make_number_tag(C_ens)
        std_tag = make_number_tag(g_std_Hz)

        filename = "C_$(C_tag)_gstd_$(std_tag)Hz.jld2"

        saved_file_name = joinpath(
            SWEEP_OUTDIR,
            filename,
        )

        println()
        println("------------------------------------------------------------")
        println("Run $run_index / $N_runs")
        println("C_ens        = $C_ens")
        print_cooperativity_honesty(C_ens, BASE_SYSTEM_CONFIG.freq_inhomogeneity,
                                    merge(BASE_SYSTEM_CONFIG.g_inhomogeneity, (std = 2π * g_std_Hz,)))
        println("g_std / 2π   = $g_std_Hz Hz")
        println("Output file  = $saved_file_name")
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





        g_inhomogeneity = merge(
            BASE_SYSTEM_CONFIG.g_inhomogeneity,
            (

                std = 2π * g_std_Hz,
            ),
        )





        SYSTEM_CONFIG = merge(
            BASE_SYSTEM_CONFIG,
            (
                C_ens = C_ens,
                g_inhomogeneity = g_inhomogeneity,
            ),
        )







        PULSE_CONFIG = build_pulse_config(SYSTEM_CONFIG)





        run_simulation(
            SIM_SETTING,
            SYSTEM_CONFIG,
            PULSE_CONFIG,
        )

        println("Run $run_index finished.")
    end





    println()
    println("============================================================")
    println("C_ens and g_std sweep finished.")
    println("Results saved in:")
    println(SWEEP_OUTDIR)
    println("============================================================")

    return nothing
end


run_sweep()

nothing