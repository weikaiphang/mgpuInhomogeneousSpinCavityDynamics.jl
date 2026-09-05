const G_OVER_KAPPA_MAX = 1e-2
const G_OVER_FWHM_MAX = 1e-2
const FWHM_OVER_KAPPA_RANGE = (0.2, 5.0)
const OVERCOUPLING_RANGE = (0.5, 1.0)
const C_ENS_RANGE = (0.05, 1.0)
const N_IMPLIED_RANGE = (1e4, 1e16)
const GAUSSIAN_G_SPAN_SIGMA = 3.0

const CANONICAL_KAPPA_T_HZ = 1.0e6
const CANONICAL_R = 1.0
const CANONICAL_GAMMA = 1.0
const CANONICAL_C_ENS = 0.6
const CANONICAL_G_HZ = 100.0
const CANONICAL_DELTA0_OVER_KT = 0.0

function kappa_t_of(sys)
    return sys.kappa_e + sys.kappa_i
end

function physical_g_mean(sys)
    g = sys.g_inhomogeneity
    if g.kind === :constant
        return Float64(g.g_value)
    elseif g.kind === :gaussian
        return Float64(g.mean)
    else
        error("datagen supports only :constant and :gaussian g, got $(g.kind).")
    end
end

function physical_g2_avg(sys)
    g = sys.g_inhomogeneity
    if g.kind === :constant
        return abs2(Float64(g.g_value))
    elseif g.kind === :gaussian
        return abs2(Float64(g.mean)) + abs2(Float64(g.std))
    else
        error("datagen supports only :constant and :gaussian g, got $(g.kind).")
    end
end

function build_freq_inhomogeneity(kind::Symbol, FWHM::Float64)
    if kind === :lorentzian
        return (
            kind = :lorentzian,
            FWHM = FWHM,
            span_gamma = 2.5,
            renormalize = false,
        )
    elseif kind === :gaussian
        return (
            kind = :gaussian,
            FWHM = FWHM,
            span_sigma = 3.0,
            renormalize = false,
        )
    else
        error("Unknown freq kind $kind.")
    end
end

function build_g_inhomogeneity(kind::Symbol, g_rad::Float64, eta::Float64)
    if kind === :constant
        return (
            kind = :constant,
            g_value = g_rad,
        )
    elseif kind === :gaussian
        std = eta * g_rad
        return (
            kind = :gaussian,
            mean = g_rad,
            std = std,
            span_sigma = GAUSSIAN_G_SPAN_SIGMA,
            renormalize = true,
        )
    else
        error("Unknown g kind $kind.")
    end
end

function system_from_physical(;
    kappa_t_hz::Float64,
    r::Float64,
    gamma::Float64,
    C_ens::Float64,
    g_hz::Float64,
    delta0_over_kt::Float64,
    freq_kind::Symbol,
    g_kind::Symbol,
    eta::Float64 = 0.0,
)
    kappa_t = TWO_PI * kappa_t_hz
    kappa_e = r * kappa_t
    kappa_i = (1 - r) * kappa_t
    FWHM = gamma * kappa_t
    g_rad = TWO_PI * g_hz
    delta0 = delta0_over_kt * kappa_t

    return (
        C_ens = C_ens,
        delta0 = delta0,
        kappa_e = kappa_e,
        kappa_i = kappa_i,
        freq_inhomogeneity = build_freq_inhomogeneity(freq_kind, FWHM),
        g_inhomogeneity = build_g_inhomogeneity(g_kind, g_rad, eta),
    )
end

function physics_gate_reason(sys)::Union{Nothing,String}
    kappa_e = sys.kappa_e
    kappa_i = sys.kappa_i
    kappa_t = kappa_t_of(sys)
    C_ens = sys.C_ens
    freq = sys.freq_inhomogeneity
    gcfg = sys.g_inhomogeneity
    FWHM = freq.FWHM
    g_mean = physical_g_mean(sys)

    kappa_e > 0 || return "kappa_e must be positive."
    kappa_i >= 0 || return "kappa_i must be non-negative."
    kappa_t > 0 || return "kappa_t must be positive."
    C_ens > 0 || return "C_ens must be positive."
    FWHM > 0 || return "FWHM must be positive."
    g_mean > 0 || return "g must be positive."

    freq.kind in (:lorentzian, :gaussian) || return "freq kind $(freq.kind) is not allowed."
    gcfg.kind in (:constant, :gaussian) || return "g kind $(gcfg.kind) is not allowed."

    C_ENS_RANGE[1] <= C_ens <= C_ENS_RANGE[2] ||
        return "C_ens=$C_ens outside $(C_ENS_RANGE)."

    r = kappa_e / kappa_t
    OVERCOUPLING_RANGE[1] <= r <= OVERCOUPLING_RANGE[2] ||
        return "overcoupling r=$r outside $(OVERCOUPLING_RANGE)."

    ratio_fwhm = FWHM / kappa_t
    FWHM_OVER_KAPPA_RANGE[1] <= ratio_fwhm <= FWHM_OVER_KAPPA_RANGE[2] ||
        return "FWHM/kappa_t=$ratio_fwhm outside $(FWHM_OVER_KAPPA_RANGE)."

    g_mean / kappa_t <= G_OVER_KAPPA_MAX ||
        return "g/kappa_t=$(g_mean / kappa_t) exceeds $G_OVER_KAPPA_MAX."
    g_mean / FWHM <= G_OVER_FWHM_MAX ||
        return "g/FWHM=$(g_mean / FWHM) exceeds $G_OVER_FWHM_MAX."

    abs(sys.delta0) <= max(kappa_t, FWHM) + eps(Float64) ||
        return "|delta0| exceeds max(kappa_t, FWHM)."

    if gcfg.kind === :gaussian
        gcfg.std >= 0 || return "gaussian g std must be non-negative."
        g_low = gcfg.mean - gcfg.span_sigma * gcfg.std
        g_low > 0 || return "gaussian g truncated onto g<=0 (mean - span_sigma*std <= 0)."
    end

    g2 = physical_g2_avg(sys)
    N = ISC.total_spin_number_from_cooperativity(C_ens, kappa_t, g2, freq)
    N_IMPLIED_RANGE[1] <= N <= N_IMPLIED_RANGE[2] ||
        return "implied N=$N outside $(N_IMPLIED_RANGE)."

    dummy = merge(
        (
            simulation_order = RULE_SIMULATION_ORDER,
            M_delta = 8,
            M_g = gcfg.kind === :constant ? 1 : 8,
            Ttotal = 1e-6,
            Nt_save = 8,
            reltol = RULE_RELTOL,
            abstol = RULE_ABSTOL,
        ),
        sys,
    )
    try
        ISC.validate_config(dummy)
    catch err
        rethrow_interrupt(err)
        return "validate_config: $(sprint(showerror, err))"
    end

    return nothing
end

function admit_system(sys)
    reason = physics_gate_reason(sys)
    return reason === nothing
end

function system_key(sys)
    fi = sys.freq_inhomogeneity
    gi = sys.g_inhomogeneity
    fspan = fi.kind === :lorentzian ? fi.span_gamma : fi.span_sigma
    gpart = if gi.kind === :constant
        (:constant, gi.g_value)
    else
        (:gaussian, gi.mean, gi.std, gi.span_sigma, gi.renormalize)
    end
    return (
        sys.C_ens,
        sys.delta0,
        sys.kappa_e,
        sys.kappa_i,
        fi.kind,
        fi.FWHM,
        fspan,
        fi.renormalize,
        gpart,
    )
end

function push_unique_system!(out, seen, sys)
    admit_system(sys) || return false
    k = system_key(sys)
    k in seen && return false
    push!(seen, k)
    push!(out, sys)
    return true
end

function canonical_system()
    return system_from_physical(;
        kappa_t_hz = CANONICAL_KAPPA_T_HZ,
        r = CANONICAL_R,
        gamma = CANONICAL_GAMMA,
        C_ens = CANONICAL_C_ENS,
        g_hz = CANONICAL_G_HZ,
        delta0_over_kt = CANONICAL_DELTA0_OVER_KT,
        freq_kind = :lorentzian,
        g_kind = :constant,
        eta = 0.0,
    )
end

const C_ENS_GRID = [0.05, 0.2, 0.4, 0.6, 0.8, 1.0]
const FREQ_KIND_GRID = [:lorentzian, :gaussian]
const GAMMA_GRID = [0.5, 1.0, 2.0]
const G_SPEC_GRID = (
    (:constant, 0.0),
    (:gaussian, 1e-8),
    (:gaussian, 0.005),
    (:gaussian, 0.05),
    (:gaussian, 0.15),
)

const KAPPA_T_HZ_GRID = [0.5e6, 1.0e6, 2.0e6]
const R_GRID = [1.0, 0.9, 0.7]
const DELTA0_OVER_KT_GRID = [0.0, 0.5, -0.5]
const G_HZ_GRID = [50.0, 100.0, 250.0]

function enumerate_system_catalog()
    out = Any[]
    seen = Set{Any}()

    for C_ens in C_ENS_GRID, freq_kind in FREQ_KIND_GRID, (g_kind, eta) in G_SPEC_GRID, gamma in GAMMA_GRID
        push_unique_system!(out, seen, system_from_physical(;
            kappa_t_hz = CANONICAL_KAPPA_T_HZ,
            r = CANONICAL_R,
            gamma = gamma,
            C_ens = C_ens,
            g_hz = CANONICAL_G_HZ,
            delta0_over_kt = CANONICAL_DELTA0_OVER_KT,
            freq_kind = freq_kind,
            g_kind = g_kind,
            eta = eta,
        ))
    end

    for kappa_t_hz in KAPPA_T_HZ_GRID, r in R_GRID, dlt in DELTA0_OVER_KT_GRID
        push_unique_system!(out, seen, system_from_physical(;
            kappa_t_hz = kappa_t_hz,
            r = r,
            gamma = CANONICAL_GAMMA,
            C_ens = CANONICAL_C_ENS,
            g_hz = CANONICAL_G_HZ,
            delta0_over_kt = dlt,
            freq_kind = :lorentzian,
            g_kind = :constant,
            eta = 0.0,
        ))
    end

    for g_hz in G_HZ_GRID, C_ens in C_ENS_GRID, freq_kind in FREQ_KIND_GRID
        push_unique_system!(out, seen, system_from_physical(;
            kappa_t_hz = CANONICAL_KAPPA_T_HZ,
            r = CANONICAL_R,
            gamma = CANONICAL_GAMMA,
            C_ens = C_ens,
            g_hz = g_hz,
            delta0_over_kt = CANONICAL_DELTA0_OVER_KT,
            freq_kind = freq_kind,
            g_kind = :constant,
            eta = 0.0,
        ))
    end

    return out
end
