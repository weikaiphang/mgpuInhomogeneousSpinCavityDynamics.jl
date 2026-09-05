


function validate_frequency_inhomogeneity(freq_inhomogeneity)
    kind = freq_inhomogeneity.kind
    FWHM = freq_inhomogeneity.FWHM

    if !(kind in (:gaussian, :lorentzian))
        error(
            "Unknown freq_inhomogeneity.kind = $(kind). " *
            "Use :gaussian or :lorentzian."
        )
    end

    if FWHM <= 0
        error("freq_inhomogeneity.FWHM must be positive.")
    end

    if kind == :gaussian
        if !hasproperty(freq_inhomogeneity, :span_sigma)
            error("For Gaussian frequency inhomogeneity, provide span_sigma.")
        end

        if freq_inhomogeneity.span_sigma <= 0
            error("freq_inhomogeneity.span_sigma must be positive.")
        end

    elseif kind == :lorentzian
        if !hasproperty(freq_inhomogeneity, :span_gamma)
            error("For Lorentzian frequency inhomogeneity, provide span_gamma.")
        end

        if freq_inhomogeneity.span_gamma <= 0
            error("freq_inhomogeneity.span_gamma must be positive.")
        end
    end

    return nothing
end



function gaussian_sigma_from_FWHM(FWHM)
    return FWHM / (2 * sqrt(2 * log(2)))
end

function lorentzian_gamma_from_FWHM(FWHM)
    return FWHM / 2
end



function build_frequency_distribution(freq_inhomogeneity)
    validate_frequency_inhomogeneity(freq_inhomogeneity)

    kind = freq_inhomogeneity.kind
    FWHM = freq_inhomogeneity.FWHM

    if kind == :gaussian
        σ = gaussian_sigma_from_FWHM(FWHM)
        return Normal(0.0, σ)

    elseif kind == :lorentzian
        γL = lorentzian_gamma_from_FWHM(FWHM)
        return Cauchy(0.0, γL)
    end
end



function build_frequency_edges(freq_inhomogeneity, M_delta)
    validate_frequency_inhomogeneity(freq_inhomogeneity)

    kind = freq_inhomogeneity.kind
    FWHM = freq_inhomogeneity.FWHM

    if kind == :gaussian
        σ = gaussian_sigma_from_FWHM(FWHM)
        span_sigma = freq_inhomogeneity.span_sigma

        return range(
            -span_sigma * σ,
             span_sigma * σ;
            length = M_delta + 1,
        )

    elseif kind == :lorentzian
        γL = lorentzian_gamma_from_FWHM(FWHM)
        span_gamma = freq_inhomogeneity.span_gamma

        return range(
            -span_gamma * γL,
             span_gamma * γL;
            length = M_delta + 1,
        )
    end
end



# Default renormalize=false: keep the truncated histogram/quadrature mass.
# C_ens → N still uses the analytic FWHM formula (not the truncated mass),
# so N_total = N * sum(p) when renormalize is left off. Set renormalize=true
# to force sum(p)=1 and N_total=N.
function renormalize_frequency_probs_enabled(freq_inhomogeneity)
    if hasproperty(freq_inhomogeneity, :renormalize)
        return freq_inhomogeneity.renormalize
    else
        return false
    end
end

function maybe_renormalize_frequency_probs!(p_delta, freq_inhomogeneity)
    if renormalize_frequency_probs_enabled(freq_inhomogeneity)
        s = sum(p_delta)

        if s <= 0
            error("Cannot renormalize detuning probabilities because sum(p_delta) <= 0.")
        end

        p_delta ./= s
    end

    return p_delta
end


# Claimed C_ens is converted to N via the analytic FWHM formula (infinite
# support). Truncated bins keep mass ∑p_δ ∑p_g < 1 when renormalize=false,
# so the realized cooperativity is C_eff = C_ens * ∑p_δ * ∑p_g.
function truncation_cooperativity(C_ens, p_delta, p_g=nothing)
    sδ = sum(p_delta)
    sg = p_g === nothing ? 1.0 : sum(p_g)
    return (
        sum_p_delta = float(sδ),
        sum_p_g = float(sg),
        C_ens = float(C_ens),
        C_eff = float(C_ens) * float(sδ) * float(sg),
    )
end

function maybe_print_truncation_cooperativity(C_ens, p_delta, p_g, freq_inhomogeneity; io::IO=stdout)
    kind = hasproperty(freq_inhomogeneity, :kind) ? freq_inhomogeneity.kind : :unknown
    kind === :lorentzian || return nothing
    renormalize_frequency_probs_enabled(freq_inhomogeneity) && return nothing
    info = truncation_cooperativity(C_ens, p_delta, p_g)
    println(io, "[ensemble] truncated Lorentzian (renormalize=false): ∑p_δ = $(info.sum_p_delta)")
    println(io, "[ensemble] ∑p_g = $(info.sum_p_g)")
    println(io, "[ensemble] effective C = $(info.C_eff)  vs claimed C_ens = $(info.C_ens)")
    return info
end



function total_spin_number_from_cooperativity(
    C_ens,
    kappa_t,
    g2_avg,
    freq_inhomogeneity,
)
    validate_frequency_inhomogeneity(freq_inhomogeneity)

    kind = freq_inhomogeneity.kind
    FWHM = freq_inhomogeneity.FWHM

    if C_ens < 0
        error("C_ens must be non-negative.")
    end

    if kappa_t <= 0
        error("kappa_t must be positive.")
    end

    if g2_avg <= 0
        error("g2_avg must be positive.")
    end

    if kind == :lorentzian
        return C_ens * kappa_t * FWHM / (4 * g2_avg)

    elseif kind == :gaussian
        return C_ens * kappa_t * FWHM / (4 * sqrt(pi * log(2)) * g2_avg)
    end
end



function build_frequency_info(freq_inhomogeneity, edges_delta)
    validate_frequency_inhomogeneity(freq_inhomogeneity)

    kind = freq_inhomogeneity.kind
    FWHM = freq_inhomogeneity.FWHM

    if kind == :gaussian
        σ = gaussian_sigma_from_FWHM(FWHM)

        return (
            kind = :gaussian,
            FWHM = FWHM,
            sigma = σ,
            span_sigma = freq_inhomogeneity.span_sigma,
            edges_min = first(edges_delta),
            edges_max = last(edges_delta),
            renormalize = renormalize_frequency_probs_enabled(freq_inhomogeneity),
        )

    elseif kind == :lorentzian
        γL = lorentzian_gamma_from_FWHM(FWHM)

        return (
            kind = :lorentzian,
            FWHM = FWHM,
            gammaL = γL,
            span_gamma = freq_inhomogeneity.span_gamma,
            edges_min = first(edges_delta),
            edges_max = last(edges_delta),
            renormalize = renormalize_frequency_probs_enabled(freq_inhomogeneity),
        )
    end
end