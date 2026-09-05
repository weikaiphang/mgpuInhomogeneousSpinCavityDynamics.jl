
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
        want = hasproperty(CONFIG, :ensemble_method) ? Symbol(CONFIG.ensemble_method) : :histogram
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

# Discrete optical-depth factors after binning / quadrature.
# N is inverted from full-line C_ens; the ODE sees C_eff = C_ens × ∑p_δ × ∑p_g.
# Do not rescale N. Print both C_ens and C_eff (no silent ~0.76×).
function ensemble_optical_depth(C_ens, p_delta, p_g)
    pδ = sum(p_delta)
    pg = sum(p_g)
    return (p_delta_sum = pδ, p_g_sum = pg, C_eff = Float64(C_ens) * pδ * pg)
end

function _g_cfg_or_nothing(sys)
    return hasproperty(sys, :g_inhomogeneity) ? sys.g_inhomogeneity : nothing
end

function cooperativity_honesty(C_ens, freq_inhomogeneity, g_inhomogeneity=nothing)
    pδ = frequency_truncation_mass(freq_inhomogeneity)
    pg = g_inhomogeneity === nothing ? 1.0 : coupling_truncation_mass(g_inhomogeneity)
    return (
        C_ens = Float64(C_ens),
        p_delta_sum = pδ,
        p_g_sum = pg,
        C_eff = Float64(C_ens) * pδ * pg,
    )
end

function cooperativity_honesty(sys)
    return cooperativity_honesty(sys.C_ens, sys.freq_inhomogeneity, _g_cfg_or_nothing(sys))
end

function print_cooperativity_honesty(io::IO, C_ens, freq_inhomogeneity, g_inhomogeneity=nothing)
    h = cooperativity_honesty(C_ens, freq_inhomogeneity, g_inhomogeneity)
    freq_renorm = renormalize_frequency_probs_enabled(freq_inhomogeneity)
    println(io, "C_ens = $(h.C_ens)  (full-line; N is built from this)")
    println(io, "∑p_δ  = $(h.p_delta_sum)  (freq renormalize=$(freq_renorm))")
    if g_inhomogeneity !== nothing
        g_renorm = coupling_renormalize_enabled(g_inhomogeneity)
        pg_str = isnan(h.p_g_sum) ? "unknown (user_defined, renormalize=false)" : string(h.p_g_sum)
        println(io, "∑p_g  = $pg_str  (g renormalize=$(g_renorm))")
    end
    println(io, "C_eff = $(h.C_eff)  (ODE optical depth = C_ens × ∑p_δ × ∑p_g)")
    return h
end

print_cooperativity_honesty(C_ens, freq_inhomogeneity, g_inhomogeneity=nothing) =
    print_cooperativity_honesty(stdout, C_ens, freq_inhomogeneity, g_inhomogeneity)

print_cooperativity_honesty(io::IO, sys) =
    print_cooperativity_honesty(io, sys.C_ens, sys.freq_inhomogeneity, _g_cfg_or_nothing(sys))

print_cooperativity_honesty(sys) = print_cooperativity_honesty(stdout, sys)

function _log_ensemble_truncation(p_delta, p_g, freq_cfg, g_cfg, C_ens)
    od = ensemble_optical_depth(C_ens, p_delta, p_g)
    freq_renorm = renormalize_frequency_probs_enabled(freq_cfg)
    g_renorm = coupling_renormalize_enabled(g_cfg)
    println("[ensemble] C_ens = $C_ens  (full-line; N is built from this)")
    println("[ensemble] ∑p_δ = $(od.p_delta_sum), ∑p_g = $(od.p_g_sum)  " *
            "(renormalize freq=$(freq_renorm), g=$(g_renorm))")
    println("[ensemble] C_eff = $(od.C_eff)  (ODE optical depth = C_ens × ∑p_δ × ∑p_g)")
    return od
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

    M_g = length(g_b_1d)
    M   = M_delta * M_g





    N = total_spin_number_from_cooperativity(
        C_ens,
        kappa_t,
        g2_avg,
        freq_cfg,
    )

    println("Total spin number N = $N")





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

    _log_ensemble_choice(_ens_plan, CONFIG.M_delta, CONFIG.M_g, M_delta, M_g, M)
    od = _log_ensemble_truncation(p_delta, p_g, freq_cfg, g_inhomogeneity, C_ens)

    return (
        C_ens = C_ens,
        p_delta_sum = od.p_delta_sum,
        p_g_sum = od.p_g_sum,
        C_eff = od.C_eff,

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