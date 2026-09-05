using InhomogeneousSpinCavityDynamics

OUTDIR = joinpath(@__DIR__, "..", "data")
mkpath(OUTDIR)


SIM_SETTING = (
    simulation_order = :order2,
    M_delta = 375,
    M_g     = 1,
    initial_condition = :inverted,
    Ttotal = 150e-6,
    Nt_save = 5001,
    reltol  = 1e-8,
    abstol  = 1e-8,
    saved_file_name = joinpath(OUTDIR, "rase_2nd_mgpu.jld2"),

    nshards    = 1,
    integrator = :tsit5,
    save_mode  = :tstops,
)

SYSTEM_CONFIG = (
    C_ens   = 0.6,
    delta0  = 0.0,
    kappa_e = 2*pi*1e6,
    kappa_i = 2*pi*0,
    freq_inhomogeneity = (
        kind = :lorentzian,
        FWHM = 2*pi*1e6,
        span_gamma = 2.5,
        renormalize = false,
    ),
    g_inhomogeneity = (
        kind = :constant,
        g_value = 2π * 100,
    ),
)

PULSE_CONFIG = (
    (
        name       = "WURST pulse",
        kind       = :wurst,
        t_center   = 75e-6,
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

mgpu_run_simulation(SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG; progress_every = 200)
println("Run finished.")
nothing
