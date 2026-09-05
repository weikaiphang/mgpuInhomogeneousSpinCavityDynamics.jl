using InhomogeneousSpinCavityDynamics

OUTDIR = joinpath(@__DIR__, "..", "..", "data", "fig_4_d")
mkpath(OUTDIR)

SIM_SETTING = (
    simulation_order = :order2,

    M_delta = 375,
    M_g     = 20,

    initial_condition = :inverted,

    Ttotal = 150e-6,

    Nt_save = 5001,
    reltol  = 1e-8,
    abstol  = 1e-8,

    saved_file_name = joinpath(OUTDIR, "three_arp_pulses_c0d6.jld2"),
)

SYSTEM_CONFIG = (
    C_ens   = 0.6,

    delta0 = 0.0,
    kappa_e = 2*pi*1e6,
    kappa_i = 2*pi*0,

    freq_inhomogeneity = (
        kind = :lorentzian,
        FWHM = 2*pi*1e6,
        span_gamma = 2.5,
        renormalize = false,
    ),

    g_inhomogeneity = (
        kind = :gaussian,
        mean = 2*pi*100,
        std  = 2*pi*5,
        span_sigma = 3.0,
        renormalize = true,
    ),
)

PULSE_CONFIG = (
    (
        name       = "WURST pulse 1",
        kind       = :wurst,

        t_center   = 55e-6,
        duration   = 10e-6,
        amp        = 0.5 * sqrt(SYSTEM_CONFIG.kappa_e) * 2.0e4,
        bandwidth  = 5.0 * SYSTEM_CONFIG.freq_inhomogeneity.FWHM,
        n          = 20.0,
        omega0     = 0.0,
        chirp_sign = +1.0,
        phase0     = 0.0,
        edge_frac  = 1e-4,
    ),

    (
        name       = "WURST pulse 2",
        kind       = :wurst,

        t_center   = 75e-6,
        duration   = 20e-6,
        amp        = 0.5 * sqrt(SYSTEM_CONFIG.kappa_e) * 2.0e4,
        bandwidth  = 5.0 * SYSTEM_CONFIG.freq_inhomogeneity.FWHM,
        n          = 20.0,
        omega0     = 0.0,
        chirp_sign = +1.0,
        phase0     = 0.0,
        edge_frac  = 1e-4,
    ),

    (
        name       = "WURST pulse 3",
        kind       = :wurst,

        t_center   = 95e-6,
        duration   = 10e-6,
        amp        = 0.5 * sqrt(SYSTEM_CONFIG.kappa_e) * 2.0e4,
        bandwidth  = 5.0 * SYSTEM_CONFIG.freq_inhomogeneity.FWHM,
        n          = 20.0,
        omega0     = 0.0,
        chirp_sign = +1.0,
        phase0     = 0.0,
        edge_frac  = 1e-4,
    ),
)

run_simulation(SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG)

println("Run finished.")

nothing