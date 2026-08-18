using InhomogeneousSpinCavityDynamics

OUTDIR = joinpath(@__DIR__, "..", "data")
mkpath(OUTDIR)

SIM_SETTING = (
    # --- simulation order ---
    simulation_order = :order1,   # :order1 (first-order), :order2 (second-order)

    # --- discretization ---
    M_delta = 3000,
    M_g     = 1,

    # --- initial condition ---
    initial_condition = :ground,        # :ground, :inverted, or :custom

    # --- simulation time ---
    Ttotal = 1100e-6,

    # --- solver ---
    Nt_save = 5001,
    reltol  = 1e-8,
    abstol  = 1e-8,

    # --- saving ---
    saved_file_name = joinpath(OUTDIR, "demo_1st_mgpu.jld2"),

    # --- echo peak detection for phase recording ---
    peak_detection = (
        labels = [:echo1, :echo2],
        times = [550e-6, 1050e-6],
        half_window = 10e-6,
    ),
)

SYSTEM_CONFIG = (
    # --- cooperativity ---
    C_ens   = 0.6,

    # --- cavity ---
    delta0 = 0.0,
    kappa_e = 2*pi*1e6,
    kappa_i = 2*pi*0,

    # --- detuning distribution ---
    freq_inhomogeneity = (
        kind = :lorentzian,
        FWHM = 2*pi*1e6,
        span_gamma = 2.5,
        renormalize = false,
    ),

    # --- coupling distribution ---
    g_inhomogeneity = (
        kind = :constant,
        g_value = 2π * 100,
    )
)

PULSE_CONFIG = (
    (
        name  = "Gaussian input signal",
        kind  = :gaussian,

        t0    = 50e-6,
        sigma = 3e-6,
        amp   = 0.5 * sqrt(SYSTEM_CONFIG.kappa_e) * 0.332,
        omega = 0.0,
        phase = 0.0,
    ),

    (
        name       = "First WURST pulse",
        kind       = :wurst,

        t_center   = 300e-6,
        duration   = 400e-6,
        amp        = 0.5 * sqrt(SYSTEM_CONFIG.kappa_e) * 2.0e4,
        bandwidth  = 5.0 * SYSTEM_CONFIG.freq_inhomogeneity.FWHM,
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
        amp        = 0.5 * sqrt(SYSTEM_CONFIG.kappa_e) * 2.0e4,
        bandwidth  = 5.0 * SYSTEM_CONFIG.freq_inhomogeneity.FWHM,
        n          = 20.0,
        omega0     = 0.0,
        chirp_sign = +1.0,
        phase0     = 0.0,
        edge_frac  = 1e-4,
    ),
)

mgpu_run_simulation(SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG)

println("Run finished.")

nothing