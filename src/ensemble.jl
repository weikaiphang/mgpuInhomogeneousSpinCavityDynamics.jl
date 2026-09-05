
function bin_means_and_probs(dist, edges)
    Mloc = length(edges) - 1
    probs = zeros(Float64, Mloc)
    means = zeros(Float64, Mloc)

    for j in 1:Mloc
        low, high = edges[j], edges[j+1]

        pj = cdf(dist, high) - cdf(dist, low)
        probs[j] = pj

        if pj > 0
            num, _ = quadgk(x -> x * pdf(dist, x), low, high)
            means[j] = num / pj
        else
            means[j] = 0.0
        end
    end

    return means, probs
end

function second_moment_from_bins(x_b_1d, p_x)
    p_sum = sum(p_x)

    if p_sum <= 0
        error("Cannot compute second moment because sum(p_x) <= 0.")
    end

    return sum((x_b_1d .^ 2) .* p_x) / p_sum
end


function build_2d_bins(N, delta_b_1d, p_delta, g_b_1d, p_g)
    M_delta = length(delta_b_1d)
    M_g     = length(g_b_1d)

    Nj_2d    = zeros(Float64, M_delta, M_g)
    delta_2d = zeros(Float64, M_delta, M_g)
    g_2d     = zeros(Float64, M_delta, M_g)

    for i in 1:M_delta
        for k in 1:M_g
            pij = p_delta[i] * p_g[k]

            Nj_2d[i, k]    = N * pij
            delta_2d[i, k] = delta_b_1d[i]
            g_2d[i, k]     = g_b_1d[k]
        end
    end

    Nj_flat    = vec(Nj_2d)
    delta_flat = vec(delta_2d)
    g_flat     = vec(g_2d)

    N_total = sum(Nj_2d)

    return Nj_flat, delta_flat, g_flat, N_total, Nj_2d
end




function ensemble_method_for(freq_cfg, g_cfg)
    fk = hasproperty(freq_cfg, :kind) ? freq_cfg.kind : :unknown
    gk = hasproperty(g_cfg, :kind)    ? g_cfg.kind    : :unknown
    freq_rule = fk === :lorentzian ? :tan_gauss_legendre :
                fk === :gaussian   ? :gauss_legendre_pdf  : nothing
    g_rule = gk === :constant   ? :single_node        :
             gk === :gaussian   ? :gauss_legendre_pdf  :
             gk === :powerlaw_g ? :log_gauss_legendre  : nothing
    ok = (freq_rule !== nothing) && (g_rule !== nothing)
    reason = ok ? "" :
        freq_rule === nothing ? "freq_inhomogeneity.kind=$fk has no quadrature rule" :
                                "g_inhomogeneity.kind=$gk has no quadrature rule"
    return (; method = ok ? :quadrature : :histogram,
              freq_kind = fk, freq_rule = freq_rule,
              g_kind = gk, g_rule = g_rule, reason = reason)
end


function resolve_ensemble_method(CONFIG, want::Symbol = :config)
    if want === :config
        want = hasproperty(CONFIG, :ensemble_method) ? Symbol(CONFIG.ensemble_method) : :auto
    end
    want in (:histogram, :quadrature, :auto) ||
        error("ensemble_method must be :histogram, :quadrature, or :auto; got :$want")

    want === :histogram && return (; method = :histogram,
        freq_kind = CONFIG.freq_inhomogeneity.kind, freq_rule = nothing,
        g_kind = CONFIG.g_inhomogeneity.kind, g_rule = nothing, reason = "forced")

    plan = ensemble_method_for(CONFIG.freq_inhomogeneity, CONFIG.g_inhomogeneity)
    if want === :quadrature && plan.method !== :quadrature
        error("ensemble_method=:quadrature requested but $(plan.reason); " *
              "use :histogram or :auto (which falls back).")
    end
    return plan
end

function _log_ensemble_choice(plan, M_delta_req, M_g_req, M_delta, M_g, M)
    if plan.method === :quadrature
        println("[ensemble] method = QUADRATURE  " *
                "(freq $(plan.freq_kind) -> $(plan.freq_rule),  g $(plan.g_kind) -> $(plan.g_rule))")
        println("[ensemble] discretisation: M_delta x M_g = $M_delta x $M_g  (M = $M)" *
                ((M_delta_req == M_delta && M_g_req == M_g) ? "" :
                 "   [requested $M_delta_req x $M_g_req]"))
    else
        println("[ensemble] method = HISTOGRAM" *
                (isempty(plan.reason) ? "" : "  ($(plan.reason))"))
        println("[ensemble] discretisation: M_delta x M_g = $M_delta x $M_g  (M = $M)")
    end
    return nothing
end

function prepare_derived(CONFIG; ensemble_method::Symbol = :config)
    _ens_plan = resolve_ensemble_method(CONFIG, ensemble_method)
    if _ens_plan.method === :quadrature
        @isdefined(prepare_derived_quadrature) || error(
            "prepare_derived: ensemble_method resolved to :quadrature but " *
            "ensemble_quadrature.jl is not loaded -- include it next to ensemble.jl.")
        return prepare_derived_quadrature(CONFIG; plan = _ens_plan)
    end

    C_ens   = CONFIG.C_ens
    M_delta = CONFIG.M_delta
    M_g     = CONFIG.M_g
    M       = M_delta * M_g

    kappa_e = CONFIG.kappa_e
    kappa_i = CONFIG.kappa_i
    kappa_t = kappa_e + kappa_i

    sqrt_kappa_e = sqrt(kappa_e)





    freq_cfg = CONFIG.freq_inhomogeneity

    dist_delta = build_frequency_distribution(freq_cfg)

    edges_delta = build_frequency_edges(
        freq_cfg,
        M_delta,
    )

    delta_b_1d, p_delta = bin_means_and_probs(
        dist_delta,
        edges_delta,
    )

    maybe_renormalize_frequency_probs!(
        p_delta,
        freq_cfg,
    )

    freq_info = build_frequency_info(
        freq_cfg,
        edges_delta,
    )

    FWHM = freq_info.FWHM





    g_inhomogeneity = CONFIG.g_inhomogeneity

    edges_g, g_b_1d, p_g, g_mean, g_std, g2_avg, g_info =
        build_coupling_bins(
            g_inhomogeneity,
            M_g,
        )





    N = total_spin_number_from_cooperativity(
        C_ens,
        kappa_t,
        g2_avg,
        freq_cfg,
    )

    println("Total spin number N = $N")
    _trunc = truncation_cooperativity(C_ens, p_delta, p_g)
    maybe_print_truncation_cooperativity(C_ens, p_delta, p_g, freq_cfg)





    Nj, delta_b, g_b, N_total, Nj_2d = build_2d_bins(
        N,
        delta_b_1d,
        p_delta,
        g_b_1d,
        p_g,
    )





    timespan = (0.0, CONFIG.Ttotal)

    t_save = collect(
        range(
            0,
            CONFIG.Ttotal;
            length = CONFIG.Nt_save,
        )
    )

    Nt = length(t_save)

    _log_ensemble_choice(_ens_plan, M_delta, M_g, M_delta, M_g, M)

    return (
        C_ens = C_ens,
        C_eff = _trunc.C_eff,
        sum_p_delta = _trunc.sum_p_delta,
        sum_p_g = _trunc.sum_p_g,

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

        ensemble_method = :histogram,
    )
end