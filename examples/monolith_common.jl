# Shared cavity + ensemble characterisation for monolith examples.
# Lorentzian frequency + constant g is quadrature-friendly (:auto → quadrature).

const κe = 2π * 1e6
const FWHM = 2π * 1e6

SYSTEM_CONFIG = (
    C_ens = 0.6,
    delta0 = 0.0,
    kappa_e = κe,
    kappa_i = 2π * 0.0,   # included in κₜ = κₑ + κᵢ wherever the package uses κ
    freq_inhomogeneity = (
        kind = :lorentzian,
        FWHM = FWHM,
        span_gamma = 2.5,
        renormalize = false,   # truncated Lorentzian; mass is not forced to 1
    ),
    g_inhomogeneity = (
        kind = :constant,
        g_value = 2π * 100,
    ),
)

PULSE_CONFIG = (
    (
        name = "WURST",
        kind = :wurst,
        t_center = 50e-6,
        duration = 50e-6,
        amp = 0.5 * sqrt(κe) * 2.0e4,
        bandwidth = 5.0 * FWHM,
        n = 20.0,
        omega0 = 0.0,
        chirp_sign = +1.0,
        phase0 = 0.0,
        edge_frac = 1e-4,
    ),
)

BSPLINE = (k = 1, n_coeff_A = 4, n_coeff_f = 4, degree = 3, taper_frac = 0.1)

OPTIMIZER = (
    num_epochs = 5,
    learning_rate = 0.05,
    patience = 4,
    seed = 42,
    w_time = 0.15,
    w_power = 0.05,
    w_tmax = 1.0,
    target_F = 1.0,
    I_min = 0.85,
    kappa_I = 50.0,
    S_min = 0.85,
    kappa_S = 50.0,
    track = :weak,
)

COMPUTE = (backend = :auto, integrator = :tsit5, nshards = nothing)
