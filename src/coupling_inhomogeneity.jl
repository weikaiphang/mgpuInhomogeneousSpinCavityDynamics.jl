


function coupling_renormalize_enabled(g_inhomogeneity)
    if hasproperty(g_inhomogeneity, :renormalize)
        return g_inhomogeneity.renormalize
    else
        return true
    end
end

# Paper convention (tip 3): truncated g meshes keep native mass
# (renormalize=false), matching frequency honesty (span_gamma=2.5,
# renormalize=false). Constant g has mass 1 either way. ⟨g²⟩, C_eff,
# and ARP Ω_rms all see that mesh. fig_3_c / fig_4_c / fig_4_d / rose
# / 3ARP must use this flag — do not silently switch one figure.
const PAPER_G_RENORMALIZE = false

# RMS coupling for drive ↔ Rabi. Bin-wise
#   Ω_j = 4 g_j √κ_e / κ_t · |E|
# so Ω_rms = √⟨Ω_j²⟩ = 4 √⟨g²⟩ √κ_e / κ_t · |E|.
# ⟨g²⟩ is d.g2_avg from prepare_derived — the same moment that
# inverts N from C_ens (total_spin_number_from_cooperativity) and
# that C_eff honesty already prints. Targeting Ω(⟨g⟩) instead of
# Ω_rms can fake F≈0/1 when g is inhomogeneous. Mean-g-only is
# not the paper ARP path.
function coupling_rms(g2_avg::Real)
    g2 = Float64(g2_avg)
    g2 > 0 || error("coupling_rms: g2_avg must be positive, got $g2_avg.")
    return sqrt(g2)
end

function coupling_rms(d)
    hasproperty(d, :g2_avg) || error(
        "coupling_rms: no g2_avg (pass prepare_derived output, not bare SYSTEM_CONFIG)."
    )
    return coupling_rms(d.g2_avg)
end

function arp_amp_scale(kappa_e::Real, kappa_t::Real, g_rms::Real)
    ke = Float64(kappa_e)
    kt = Float64(kappa_t)
    g = Float64(g_rms)
    (ke > 0 && kt > 0 && g > 0) || error(
        "arp_amp_scale: need positive kappa_e, kappa_t, g_rms (got $ke, $kt, $g)."
    )
    return kt / (4 * g * sqrt(ke))
end

function arp_drive_amplitude(kappa_e::Real, kappa_t::Real, g2_avg::Real, Omega_target::Real)
    Ω = Float64(Omega_target)
    Ω > 0 || error("arp_drive_amplitude: Omega_target must be positive, got $Omega_target.")
    return arp_amp_scale(kappa_e, kappa_t, coupling_rms(g2_avg)) * Ω
end


function maybe_renormalize_coupling_probs!(p_g, g_inhomogeneity)
    if coupling_renormalize_enabled(g_inhomogeneity)
        s = sum(p_g)

        if s <= 0
            error("Cannot renormalize g probabilities because sum(p_g) <= 0.")
        end

        p_g ./= s
    end

    return p_g
end


# Mass kept on the g mesh when renormalize=false. Constant / power-law
# support is already the full definition interval (mass 1). Gaussian is
# truncated to [max(0, μ − span σ), μ + span σ].
function coupling_truncation_mass(g_inhomogeneity)
    validate_coupling_inhomogeneity(g_inhomogeneity)
    coupling_renormalize_enabled(g_inhomogeneity) && return 1.0
    kind = g_inhomogeneity.kind
    kind === :constant && return 1.0
    kind === :powerlaw_g && return 1.0
    if kind === :user_defined
        return NaN
    end
    if kind === :gaussian
        μ = Float64(g_inhomogeneity.mean)
        σ = Float64(g_inhomogeneity.std)
        span = Float64(g_inhomogeneity.span_sigma)
        σ <= 0 && return 1.0
        lo = max(0.0, μ - span * σ)
        hi = μ + span * σ
        dist = Normal(μ, σ)
        return cdf(dist, hi) - cdf(dist, lo)
    end
    error("coupling_truncation_mass: unsupported g kind $(kind).")
end


function weighted_g_stats_from_bins(g_b_1d, p_g)
    p_sum = sum(p_g)

    if p_sum <= 0
        error("Cannot compute g statistics because sum(p_g) <= 0.")
    end

    w = p_g ./ p_sum

    g_mean = sum(w .* g_b_1d)
    g2_avg = sum(w .* g_b_1d.^2)
    g_std  = sqrt(sum(w .* (g_b_1d .- g_mean).^2))

    return g_mean, g_std, g2_avg
end



function load_user_g_distribution(filename; renormalize=true)
    data = load(filename)

    edges_g    = Float64.(data["edges_g"])
    g_b_1d_tmp = Float64.(data["g_b_1d_tmp"])
    p_g_tmp    = Float64.(data["p_g_tmp"])

    if length(g_b_1d_tmp) != length(p_g_tmp)
        error(
            "Loaded g_b_1d_tmp and p_g_tmp have different lengths: " *
            "length(g_b_1d_tmp) = $(length(g_b_1d_tmp)), " *
            "length(p_g_tmp) = $(length(p_g_tmp))."
        )
    end

    if length(edges_g) != length(g_b_1d_tmp) + 1
        error(
            "Loaded edges_g has inconsistent length. " *
            "Expected length(edges_g) = length(g_b_1d_tmp) + 1."
        )
    end

    p_sum = sum(p_g_tmp)

    if p_sum <= 0
        error("Cannot use user-defined g distribution because sum(p_g_tmp) <= 0.")
    end

    if renormalize
        p_g_tmp = p_g_tmp ./ p_sum
    end

    g_mean_tmp, g_std_tmp, g2_avg_tmp = weighted_g_stats_from_bins(
        g_b_1d_tmp,
        p_g_tmp,
    )

    return edges_g, g_b_1d_tmp, p_g_tmp, g_mean_tmp, g_std_tmp, g2_avg_tmp
end



function validate_coupling_inhomogeneity(g_inhomogeneity)
    if !hasproperty(g_inhomogeneity, :kind)
        error("g_inhomogeneity must contain a kind field.")
    end

    kind = g_inhomogeneity.kind

    allowed_kinds = (
        :constant,
        :gaussian,
        :powerlaw_g,
        :user_defined,
    )

    if !(kind in allowed_kinds)
        error(
            "Unknown g_inhomogeneity.kind = $(kind). " *
            "Use :constant, :gaussian, :powerlaw_g, or :user_defined."
        )
    end





    if kind == :constant
        if !hasproperty(g_inhomogeneity, :g_value)
            error(
                "For constant coupling, provide g_value. For example:\n" *
                "g_inhomogeneity = (\n" *
                "    kind = :constant,\n" *
                "    g_value = 2*pi*100,\n" *
                ")"
            )
        end

        g_value = g_inhomogeneity.g_value

        if !(g_value isa Real)
            error("For constant coupling, g_value must be a real number.")
        end

        if !isfinite(g_value)
            error("For constant coupling, g_value must be finite.")
        end

        if g_value <= 0
            error("For constant coupling, g_value must be positive.")
        end





    elseif kind == :gaussian
        if !hasproperty(g_inhomogeneity, :mean)
            error("For Gaussian g inhomogeneity, provide mean.")
        end

        if !hasproperty(g_inhomogeneity, :std)
            error("For Gaussian g inhomogeneity, provide std.")
        end

        if !hasproperty(g_inhomogeneity, :span_sigma)
            error("For Gaussian g inhomogeneity, provide span_sigma.")
        end

        if g_inhomogeneity.mean <= 0
            error("g_inhomogeneity.mean must be positive.")
        end

        if g_inhomogeneity.std < 0
            error("g_inhomogeneity.std must be non-negative.")
        end

        if g_inhomogeneity.span_sigma <= 0
            error("g_inhomogeneity.span_sigma must be positive.")
        end





    elseif kind == :powerlaw_g
        if !hasproperty(g_inhomogeneity, :alpha)
            error("For powerlaw_g inhomogeneity, provide alpha.")
        end

        if !hasproperty(g_inhomogeneity, :g_min)
            error("For powerlaw_g inhomogeneity, provide g_min.")
        end

        if !hasproperty(g_inhomogeneity, :g_max)
            error("For powerlaw_g inhomogeneity, provide g_max.")
        end

        if g_inhomogeneity.alpha <= 0
            error("g_inhomogeneity.alpha must be positive.")
        end

        if g_inhomogeneity.g_min <= 0
            error("g_inhomogeneity.g_min must be positive.")
        end

        if g_inhomogeneity.g_max <= g_inhomogeneity.g_min
            error(
                "g_inhomogeneity.g_max must be larger than " *
                "g_inhomogeneity.g_min."
            )
        end

        if hasproperty(g_inhomogeneity, :binning)
            if !(g_inhomogeneity.binning in (:log, :linear))
                error("For powerlaw_g, binning must be :log or :linear.")
            end
        end





    elseif kind == :user_defined
        if !hasproperty(g_inhomogeneity, :filename)
            error(
                "For user-defined g inhomogeneity, provide filename."
            )
        end

        if !isfile(g_inhomogeneity.filename)
            error(
                "Cannot find user-defined g-distribution file: " *
                "$(g_inhomogeneity.filename)"
            )
        end
    end





    if hasproperty(g_inhomogeneity, :renormalize)
        if !(g_inhomogeneity.renormalize isa Bool)
            error("g_inhomogeneity.renormalize must be true or false.")
        end
    end

    return nothing
end



function build_gaussian_coupling_bins(g_inhomogeneity, M_g)
    g_mean_input = g_inhomogeneity.mean
    g_std_input  = g_inhomogeneity.std
    span_sigma   = g_inhomogeneity.span_sigma

    g_low  = max(0.0, g_mean_input - span_sigma * g_std_input)
    g_high = g_mean_input + span_sigma * g_std_input

    dist_g = Normal(g_mean_input, g_std_input)

    edges_g = collect(range(
        g_low,
        g_high;
        length = M_g + 1,
    ))

    g_b_1d, p_g = bin_means_and_probs(
        dist_g,
        edges_g,
    )

    maybe_renormalize_coupling_probs!(
        p_g,
        g_inhomogeneity,
    )

    g_mean, g_std, g2_avg = weighted_g_stats_from_bins(
        g_b_1d,
        p_g,
    )

    g_info = (
        kind = :gaussian,

        mean_input = g_mean_input,
        std_input = g_std_input,
        span_sigma = span_sigma,

        renormalize = coupling_renormalize_enabled(g_inhomogeneity),
        p_sum = sum(p_g),

        g_low = g_low,
        g_high = g_high,

        g_mean = g_mean,
        g_std = g_std,
        g2_avg = g2_avg,
    )

    return edges_g, g_b_1d, p_g, g_mean, g_std, g2_avg, g_info
end



function powerlaw_integral(g_low, g_high, power)

    if abs(power + 1) < 1e-14
        return log(g_high / g_low)
    else
        return (g_high^(power + 1) - g_low^(power + 1)) / (power + 1)
    end
end


function powerlaw_bin_means_and_probs(edges_g, alpha)
    M_g = length(edges_g) - 1

    g_b_1d = zeros(Float64, M_g)
    p_g    = zeros(Float64, M_g)


    norm = powerlaw_integral(first(edges_g), last(edges_g), -alpha)

    if norm <= 0
        error("Power-law normalization is non-positive.")
    end

    for k in 1:M_g
        low  = edges_g[k]
        high = edges_g[k+1]

        prob_integral = powerlaw_integral(low, high, -alpha)
        mean_integral = powerlaw_integral(low, high, 1 - alpha)

        if prob_integral <= 0
            error("Power-law probability in bin $k is non-positive.")
        end

        p_g[k] = prob_integral / norm
        g_b_1d[k] = mean_integral / prob_integral
    end

    return g_b_1d, p_g
end


function build_powerlaw_coupling_bins(g_inhomogeneity, M_g)
    alpha = g_inhomogeneity.alpha
    g_min = g_inhomogeneity.g_min
    g_max = g_inhomogeneity.g_max

    binning = hasproperty(g_inhomogeneity, :binning) ?
        g_inhomogeneity.binning :
        :log

    if binning == :log
        edges_g = exp.(range(
            log(g_min),
            log(g_max);
            length = M_g + 1,
        ))

    elseif binning == :linear
        edges_g = collect(range(
            g_min,
            g_max;
            length = M_g + 1,
        ))

    else
        error("Unknown powerlaw_g binning = $(binning). Use :log or :linear.")
    end

    g_b_1d, p_g = powerlaw_bin_means_and_probs(
        edges_g,
        alpha,
    )

    maybe_renormalize_coupling_probs!(
        p_g,
        g_inhomogeneity,
    )

    g_mean, g_std, g2_avg = weighted_g_stats_from_bins(
        g_b_1d,
        p_g,
    )

    g_info = (
        kind = :powerlaw_g,

        alpha = alpha,
        g_min = g_min,
        g_max = g_max,
        binning = binning,

        renormalize = coupling_renormalize_enabled(g_inhomogeneity),
        p_sum = sum(p_g),

        g_low = minimum(g_b_1d),
        g_high = maximum(g_b_1d),

        g_mean = g_mean,
        g_std = g_std,
        g2_avg = g2_avg,
    )

    return edges_g, g_b_1d, p_g, g_mean, g_std, g2_avg, g_info
end


function build_constant_coupling_bins(g_inhomogeneity, M_g)
    @assert hasproperty(g_inhomogeneity, :g_value)

    g_value = Float64(g_inhomogeneity.g_value)

    @assert isfinite(g_value)
    @assert g_value >= 0.0


    if M_g != 1
        error("g_inhomogeneity.kind = :constant requires CONFIG.M_g = 1; " *
              "got M_g = $M_g. A constant coupling has a single g node. " *
              "Set M_g = 1 (prepare_derived also sets M_g = length(g_b_1d) " *
              "after bin construction).")
    end


    g_b_1d = [g_value]
    p_g    = [1.0]



    edges_g = [
        prevfloat(g_value),
        nextfloat(g_value),
    ]


    g_mean = g_value
    g_std  = 0.0
    g2_avg = abs2(g_value)

    g_info = (
        kind = :constant,
        g_value = g_value,
        M_g = 1,
        renormalize = false,
    )

    return (
        edges_g,
        g_b_1d,
        p_g,
        g_mean,
        g_std,
        g2_avg,
        g_info,
    )
end


function build_user_defined_coupling_bins(g_inhomogeneity, M_g)
    filename = g_inhomogeneity.filename
    renormalize = coupling_renormalize_enabled(g_inhomogeneity)

    edges_g, g_b_1d, p_g, g_mean, g_std, g2_avg =
        load_user_g_distribution(
            filename;
            renormalize = renormalize,
        )

    if length(g_b_1d) != M_g
        error(
            "SIM_SETTING.M_g = $(M_g), but the user-defined g distribution " *
            "contains $(length(g_b_1d)) bins. " *
            "Set SIM_SETTING.M_g = $(length(g_b_1d))."
        )
    end

    g_info = (
        kind = :user_defined,

        filename = filename,
        renormalize = renormalize,
        p_sum = sum(p_g),

        g_low = minimum(g_b_1d),
        g_high = maximum(g_b_1d),

        g_mean = g_mean,
        g_std = g_std,
        g2_avg = g2_avg,
    )

    return edges_g, g_b_1d, p_g, g_mean, g_std, g2_avg, g_info
end



function build_coupling_bins(g_inhomogeneity, M_g)
    validate_coupling_inhomogeneity(g_inhomogeneity)

    kind = g_inhomogeneity.kind

    if kind == :gaussian
        return build_gaussian_coupling_bins(g_inhomogeneity, M_g)

    elseif kind == :powerlaw_g
        return build_powerlaw_coupling_bins(g_inhomogeneity, M_g)

    elseif kind == :constant
        return build_constant_coupling_bins(g_inhomogeneity, M_g)

    elseif kind == :user_defined
        return build_user_defined_coupling_bins(g_inhomogeneity, M_g)

    else
        error("Unknown g_inhomogeneity.kind = $(kind).")
    end
end