


# Default renormalize=false except when a user file loader passes renormalize=true.
# Same convention as frequency bins: truncated mass is kept unless requested.
function coupling_renormalize_enabled(g_inhomogeneity)
    if hasproperty(g_inhomogeneity, :renormalize)
        return g_inhomogeneity.renormalize
    else
        return true
    end
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
        @warn """
        kind = :constant uses one effective g bin.
        Requested M_g = $M_g will be replaced by M_g = 1.
        """
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