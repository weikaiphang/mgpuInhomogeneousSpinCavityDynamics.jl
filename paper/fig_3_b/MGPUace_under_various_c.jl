using InhomogeneousSpinCavityDynamics
using Printf

# ============================================================
# 1) OUTPUT DIRECTORY
# ============================================================

OUTDIR = joinpath(@__DIR__, "..", "..", "data", "fig_3_b_mgpu")
mkpath(OUTDIR)

# ============================================================
# 2) COOPERATIVITY VALUES TO SCAN
# ============================================================

C_VALUES = [
    0.1,
    0.6,
]

# Alternatively, use an evenly spaced scan:
# C_VALUES = collect(range(0.1, 1.0; length=10))

# ============================================================
# 3) BASE SIMULATION SETTINGS
# ============================================================

SIM_SETTING_BASE = (
    # --- simulation order ---
    simulation_order = :order1,  

    # --- discretization ---
    M_delta = 3000,
    M_g     = 1,

    # --- initial condition ---
    initial_condition = :ground,  

    # --- simulation time ---
    Ttotal = 1100e-6,

    # --- solver ---
    Nt_save = 5001,
    reltol  = 1e-8,
    abstol  = 1e-8,

    # --- temporary saving path; replaced during the sweep ---
    saved_file_name = joinpath(OUTDIR, "temporary.jld2"),

    # --- echo peak detection for phase recording ---
    peak_detection = (
        labels = [:echo1, :echo2],
        times = [550e-6, 1050e-6],
        half_window = 10e-6,
    ),
)

# ============================================================
# 4) BASE SYSTEM CONFIGURATION
# ============================================================

SYSTEM_CONFIG_BASE = (
    # --- cooperativity ---
    # This value is replaced during the sweep.
    C_ens = first(C_VALUES),

    # --- cavity ---
    delta0 = 0.0,
    kappa_e = 2 * pi * 1e6,
    kappa_i = 2 * pi * 0,

    # --- detuning distribution ---
    freq_inhomogeneity = (
        kind = :lorentzian,
        FWHM = 2 * pi * 1e6,
        span_gamma = 2.5,
        renormalize = false,
    ),

    # --- coupling distribution ---
    g_inhomogeneity = (
        kind = :gaussian,
        mean = 2 * pi * 100,
        std  = 2 * pi * 1,
        span_sigma = 3.0,
        renormalize = true,
    ),
)

# ============================================================
# 5) PULSE CONFIGURATION
# ============================================================

PULSE_CONFIG = (
    (
        name  = "Gaussian input signal",
        kind  = :gaussian,

        t0    = 50e-6,
        sigma = 3e-6,
        amp   = 0.5 * sqrt(SYSTEM_CONFIG_BASE.kappa_e) * 0.332, # This value is equivalent to 0.001 * (amp of the Gaussian pi-pulse)
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

# ============================================================
# 6) COOPERATIVITY SWEEP
# ============================================================

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

        # Filename-safe cooperativity label:
        # 0.600 -> 0p600
        C_label = replace(@sprintf("%.3f", C_value), "." => "p")

        saved_file_name = joinpath(
            OUTDIR,
            "order1_C_$(C_label).jld2",
        )

        # Configuration used only for this run
        SYSTEM_CONFIG_RUN = merge(
            SYSTEM_CONFIG_BASE,
            (
                C_ens = C_value,
            ),
        )

        # Simulation settings used only for this run
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