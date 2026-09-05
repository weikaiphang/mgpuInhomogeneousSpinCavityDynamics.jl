


function _gauss_legendre_pts(n::Integer)
    n >= 1 || error("_gauss_legendre_pts: n must be >= 1")
    n == 1 && return ([0.0], [2.0])
    k = collect(1.0:(n - 1))
    beta = k ./ sqrt.(4 .* k .^ 2 .- 1.0)
    E = eigen(SymTridiagonal(zeros(Float64, n), beta))
    x = E.values
    w = 2.0 .* (E.vectors[1, :] .^ 2)
    p = sortperm(x)
    return x[p], w[p]
end


function _gauss_legendre_on(n::Integer, a::Real, b::Real)
    x, w = _gauss_legendre_pts(n)
    half = 0.5 * (b - a)
    return half .* x .+ 0.5 * (a + b), half .* w
end


function _quad_frequency_nodes(freq_cfg, M_delta::Integer)
    validate_frequency_inhomogeneity(freq_cfg)
    M_delta >= 1 || error("_quad_frequency_nodes: M_delta must be >= 1")
    kind = freq_cfg.kind
    FWHM = freq_cfg.FWHM

    if kind === :lorentzian
        gamma = lorentzian_gamma_from_FWHM(FWHM)
        span  = freq_cfg.span_gamma
        theta_max = atan(span)
        x, w = _gauss_legendre_pts(M_delta)
        theta = theta_max .* x
        delta = gamma .* tan.(theta)

        p = w .* (theta_max / pi)
        return delta, p

    elseif kind === :gaussian
        sigma = gaussian_sigma_from_FWHM(FWHM)
        span  = freq_cfg.span_sigma
        L = span * sigma
        delta, w = _gauss_legendre_on(M_delta, -L, L)
        p = w .* pdf.(Normal(0.0, sigma), delta)
        return delta, p

    else
        error("_quad_frequency_nodes: freq kind $kind has no quadrature rule " *
              "(supported: :lorentzian, :gaussian).")
    end
end

function _quad_coupling_bins(g_cfg, M_g::Integer)
    validate_coupling_inhomogeneity(g_cfg)
    M_g >= 1 || error("_quad_coupling_bins: M_g must be >= 1")
    kind = g_cfg.kind

    if kind === :constant
        gv = Float64(g_cfg.g_value)
        g_info = (kind = :constant, g_value = gv, M_g = 1,
                  renormalize = false, method = :quadrature)
        return ([prevfloat(gv), nextfloat(gv)], [gv], [1.0], gv, 0.0, abs2(gv), g_info)

    elseif kind === :gaussian
        mu   = Float64(g_cfg.mean)
        sd   = Float64(g_cfg.std)
        span = Float64(g_cfg.span_sigma)
        lo = max(0.0, mu - span * sd)
        hi = mu + span * sd
        g, w = _gauss_legendre_on(M_g, lo, hi)
        p = w .* pdf.(Normal(mu, sd), g)
        maybe_renormalize_coupling_probs!(p, g_cfg)
        g_mean, g_std, g2_avg = weighted_g_stats_from_bins(g, p)
        g_info = (kind = :gaussian, mean_input = mu, std_input = sd, span_sigma = span,
                  renormalize = coupling_renormalize_enabled(g_cfg), p_sum = sum(p),
                  g_low = lo, g_high = hi,
                  g_mean = g_mean, g_std = g_std, g2_avg = g2_avg, method = :quadrature)
        return ([lo, hi], g, p, g_mean, g_std, g2_avg, g_info)

    elseif kind === :powerlaw_g
        alpha = Float64(g_cfg.alpha)
        gmin  = Float64(g_cfg.g_min)
        gmax  = Float64(g_cfg.g_max)
        lx, w = _gauss_legendre_on(M_g, log(gmin), log(gmax))
        g = exp.(lx)
        p = w .* g .* g .^ (-alpha)
        maybe_renormalize_coupling_probs!(p, g_cfg)
        g_mean, g_std, g2_avg = weighted_g_stats_from_bins(g, p)
        g_info = (kind = :powerlaw_g, alpha = alpha, g_min = gmin, g_max = gmax,
                  binning = :log_gauss_legendre,
                  renormalize = coupling_renormalize_enabled(g_cfg), p_sum = sum(p),
                  g_low = minimum(g), g_high = maximum(g),
                  g_mean = g_mean, g_std = g_std, g2_avg = g2_avg, method = :quadrature)
        return ([gmin, gmax], g, p, g_mean, g_std, g2_avg, g_info)

    else
        error("_quad_coupling_bins: g kind $kind has no quadrature rule " *
              "(supported: :constant, :gaussian, :powerlaw_g).")
    end
end


_eq_optprop(x, k::Symbol, default) = hasproperty(x, k) ? getproperty(x, k) : default
_ens_quad_M_delta(CONFIG) = Int(_eq_optprop(CONFIG, :ensemble_M_delta, CONFIG.M_delta))
_ens_quad_M_g(CONFIG)     = Int(_eq_optprop(CONFIG, :ensemble_M_g, CONFIG.M_g))



function prepare_derived_quadrature(CONFIG; plan = nothing)
    plan === nothing && (plan = ensemble_method_for(CONFIG.freq_inhomogeneity, CONFIG.g_inhomogeneity))
    plan.method === :quadrature || error(
        "prepare_derived_quadrature called for a non-quadrature config: $(plan.reason)")

    C_ens   = CONFIG.C_ens
    M_delta_req = CONFIG.M_delta
    M_g_req     = CONFIG.M_g
    M_delta = _ens_quad_M_delta(CONFIG)
    M_g     = _ens_quad_M_g(CONFIG)

    kappa_e = CONFIG.kappa_e
    kappa_i = CONFIG.kappa_i
    kappa_t = kappa_e + kappa_i
    sqrt_kappa_e = sqrt(kappa_e)


    freq_cfg = CONFIG.freq_inhomogeneity

    delta_b_1d, p_delta = _quad_frequency_nodes(freq_cfg, M_delta)
    maybe_renormalize_frequency_probs!(p_delta, freq_cfg)

    edges_delta = build_frequency_edges(freq_cfg, M_delta)
    freq_info = build_frequency_info(freq_cfg, edges_delta)
    FWHM = freq_info.FWHM


    g_inhomogeneity = CONFIG.g_inhomogeneity
    edges_g, g_b_1d, p_g, g_mean, g_std, g2_avg, g_info =
        _quad_coupling_bins(g_inhomogeneity, M_g)
    M_g = length(g_b_1d)


    N = total_spin_number_from_cooperativity(C_ens, kappa_t, g2_avg, freq_cfg)
    println("Total spin number N = $N")


    Nj, delta_b, g_b, N_total, Nj_2d = build_2d_bins(N, delta_b_1d, p_delta, g_b_1d, p_g)

    M = M_delta * M_g
    _log_ensemble_choice(plan, M_delta_req, M_g_req, M_delta, M_g, M)


    timespan = (0.0, CONFIG.Ttotal)
    t_save = collect(range(0, CONFIG.Ttotal; length = CONFIG.Nt_save))
    Nt = length(t_save)

    return (
        C_ens = C_ens,

        M_delta = M_delta,
        M_g = M_g,
        M = M,

        freq_inhomogeneity = freq_cfg,
        freq_info = freq_info,
        FWHM = FWHM,

        g_inhomogeneity = g_inhomogeneity,
        g_info = g_info,

        g_mean = g_mean,
        g_std = g_std,
        g2_avg = g2_avg,

        kappa_e = kappa_e,
        kappa_i = kappa_i,
        kappa_t = kappa_t,
        sqrt_kappa_e = sqrt_kappa_e,

        delta0 = CONFIG.delta0,

        N = N,
        N_total = N_total,

        Nj = Nj,
        delta_b = delta_b,
        g_b = g_b,

        delta_b_1d = delta_b_1d,
        p_delta = p_delta,

        edges_g = edges_g,
        g_b_1d = g_b_1d,
        p_g = p_g,

        Nj_2d = Nj_2d,

        timespan = timespan,
        t_save = t_save,
        Nt = Nt,

        ensemble_method = :quadrature,
    )
end
