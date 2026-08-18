using InhomogeneousSpinCavityDynamics
using Printf


# ============================================================
# 1) USER SETTINGS
# ============================================================

SWEEP_OUTDIR = joinpath(
    @__DIR__,
    "..",
    "..",
    "data",
    "fig_4_c",
)

mkpath(SWEEP_OUTDIR)

# WURST durations to sweep in μs.
WURST_duration_us_values = [
    10.0,
    30.0,
    100.0,
]

# Skip simulations whose output files already exist.
SKIP_EXISTING = true


# ============================================================
# 2) BASE SIMULATION SETTINGS
# ============================================================

BASE_SIM_SETTING = (
    # --- simulation order ---
    simulation_order = :order2,

    # --- discretization ---
    M_delta = 375,
    M_g     = 1,

    # --- initial condition ---
    initial_condition = :inverted,

    # --- simulation time ---
    Ttotal = 150e-6,

    # --- solver ---
    Nt_save = 5001,
    reltol  = 1e-8,
    abstol  = 1e-8,
)


# ============================================================
# 3) FIXED SYSTEM PARAMETERS
# ============================================================

SYSTEM_CONFIG = (
    # --- cooperativity ---
    C_ens = 0.6,

    # --- cavity ---
    delta0 = 0.0,
    kappa_e = 2π * 1e6,
    kappa_i = 2π * 0.0,

    # --- detuning distribution ---
    freq_inhomogeneity = (
        kind = :lorentzian,
        FWHM = 2π * 1e6,
        span_gamma = 2.5,
        renormalize = false,
    ),

    # --- coupling distribution ---
    g_inhomogeneity = (
        kind = :constant,
        g_value = 2π * 100.0,
        renormalize = false,
    ),
)


# ============================================================
# 4) OUTPUT-FILENAME HELPER
# ============================================================

function make_duration_tag(duration_us)
    return @sprintf("%03dus", round(Int, duration_us))
end


# ============================================================
# 5) PULSE-CONFIGURATION HELPER
# ============================================================

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


# ============================================================
# 6) WURST-DURATION SWEEP
# ============================================================

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

        run_simulation(
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


# ============================================================
# 7) RUN THE SWEEP
# ============================================================

run_duration_sweep(
    WURST_duration_us_values,
    BASE_SIM_SETTING,
    SYSTEM_CONFIG,
    SWEEP_OUTDIR;
    skip_existing = SKIP_EXISTING,
)

nothing