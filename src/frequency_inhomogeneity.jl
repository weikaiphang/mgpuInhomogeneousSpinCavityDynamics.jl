# ============================================================
# Frequency inhomogeneity
#
# Supported choices:
#
#     :gaussian
#     :lorentzian
#
# This file handles:
#
#     1. validation
#     2. frequency distribution construction
#     3. detuning-bin edge construction
#     4. cooperativity -> total spin number conversion
# ============================================================


# ============================================================
# Validation
# ============================================================

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


# ============================================================
# Width conversion
# ============================================================

function gaussian_sigma_from_FWHM(FWHM)
    return FWHM / (2 * sqrt(2 * log(2)))
end

function lorentzian_gamma_from_FWHM(FWHM)
    return FWHM / 2
end


# ============================================================
# Build detuning distribution
# ============================================================

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


# ============================================================
# Build detuning bin edges
# ============================================================

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


# ============================================================
# Optional probability renormalization
#
# If renormalize = true, the probability inside the chosen
# finite simulation range is normalized to 1.
#
# If renormalize = false, the probability outside the finite
# range is excluded, so sum(p_delta) may be less than 1.
# ============================================================

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


# ============================================================
# Cooperativity -> total spin number
#
# General formula:
#
#     C_ens = 2π N <g²> ρ(0) / κt
#
# where:
#
#     ρ(0) is the normalized detuning density at resonance.
#
# Lorentzian:
#
#     ρ(0) = 2 / (π FWHM)
#
#     C_ens = 4 N <g²> / (κt FWHM)
#
#     N = C_ens κt FWHM / (4 <g²>)
#
# Gaussian:
#
#     ρ(0) = 2 sqrt(log(2)) / (sqrt(π) FWHM)
#
#     C_ens = 4 sqrt(π log(2)) N <g²> / (κt FWHM)
#
#     N = C_ens κt FWHM / (4 sqrt(π log(2)) <g²>)
# ============================================================

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


# ============================================================
# Frequency info for saving / debugging
# ============================================================

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