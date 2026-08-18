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
    "fig_3_d",
)

mkpath(SWEEP_OUTDIR)

# Cooperativity values to sweep.
C_values = [
    0.05,
    0.2,
    0.4,
    0.6,
    0.8,
]

# Coupling standard deviations g_std / 2π in Hz.
# They are converted to rad/s when SYSTEM_CONFIG is constructed.
g_std_Hz_values = [
    0.5,
    1.0,
    5.0,
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
    M_g     = 20,

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

BASE_SYSTEM_CONFIG = (
    # C_ens is inserted separately for every sweep point.

    # --- cavity ---
    delta0  = 0.0,
    kappa_e = 2π * 1e6,
    kappa_i = 2π * 0.0,

    # --- detuning distribution ---
    freq_inhomogeneity = (
        kind        = :lorentzian,
        FWHM        = 2π * 1e6,
        span_gamma  = 2.5,
        renormalize = false,
    ),

    # --- coupling distribution ---
    # std is replaced separately for every sweep point.
    g_inhomogeneity = (
        kind        = :gaussian,
        mean        = 2π * 100.0,
        std         = 2π * 1.0,
        span_sigma  = 3.0,
        renormalize = true,
    ),
)


# ============================================================
# 4) OUTPUT-FILENAME HELPER
# ============================================================

function make_number_tag(x)
    # Examples:
    # 0.600  -> "0p600"
    # 10.000 -> "10p000"
    return replace(@sprintf("%.3f", x), "." => "p")
end


# ============================================================
# 5) PULSE-CONFIGURATION HELPER
# ============================================================

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


# ============================================================
# 6) TWO-DIMENSIONAL SWEEP
# ============================================================

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

    # Create all combinations of C_ens and g_std_Hz.
    sweep_points = Iterators.product(
        C_values,
        g_std_Hz_values,
    )

    for (run_index, sweep_point) in enumerate(sweep_points)
        C_ens, g_std_Hz = sweep_point

        # ----------------------------------------------------
        # Construct a unique output filename
        # ----------------------------------------------------

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
        println("g_std / 2π   = $g_std_Hz Hz")
        println("Output file  = $saved_file_name")
        println("------------------------------------------------------------")

        # ----------------------------------------------------
        # Optionally skip completed simulations
        # ----------------------------------------------------

        if SKIP_EXISTING && isfile(saved_file_name)
            println("Output file already exists. Skipping this run.")
            continue
        end

        # ----------------------------------------------------
        # Construct simulation settings
        # ----------------------------------------------------

        SIM_SETTING = merge(
            BASE_SIM_SETTING,
            (
                saved_file_name = saved_file_name,
            ),
        )

        # ----------------------------------------------------
        # Construct the coupling distribution
        # ----------------------------------------------------

        g_inhomogeneity = merge(
            BASE_SYSTEM_CONFIG.g_inhomogeneity,
            (
                # Convert g_std / 2π from Hz to g_std in rad/s.
                std = 2π * g_std_Hz,
            ),
        )

        # ----------------------------------------------------
        # Construct system configuration
        # ----------------------------------------------------

        SYSTEM_CONFIG = merge(
            BASE_SYSTEM_CONFIG,
            (
                C_ens = C_ens,
                g_inhomogeneity = g_inhomogeneity,
            ),
        )

        # ----------------------------------------------------
        # Construct pulse configuration
        # ----------------------------------------------------

        # Rebuild the pulse configuration after updating
        # SYSTEM_CONFIG so that it uses the current parameters.
        PULSE_CONFIG = build_pulse_config(SYSTEM_CONFIG)

        # ----------------------------------------------------
        # Run simulation
        # ----------------------------------------------------

        run_simulation(
            SIM_SETTING,
            SYSTEM_CONFIG,
            PULSE_CONFIG,
        )

        println("Run $run_index finished.")
    end

    # ========================================================
    # 7) FINISH
    # ========================================================

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