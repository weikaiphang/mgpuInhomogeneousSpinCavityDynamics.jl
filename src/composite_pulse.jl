

_gevrey_bump(x) = ForwardDiff.value(x) > 0 ? exp(-one(x) / x) : zero(x)


function _smooth_step(x)
    a = _gevrey_bump(x)
    b = _gevrey_bump(one(x) - x)
    return a / (a + b)
end


function _taper_window(t, ts, te, taper_frac)
    dur = te - ts
    edge = taper_frac * dur
    rise = _smooth_step((t - ts) / edge)
    fall = _smooth_step((te - t) / edge)
    return rise * fall
end


struct CompositePulse
    k::Int
    n_coeff_A::Int
    n_coeff_f::Int
    degree::Int
    T_max::Float64
    gap_scale::Float64
    dur_scale::Float64
    dur_floor::Float64
    amp_scale::Float64
    freq_scale::Float64
    taper_frac::Float64
end

function CompositePulse(k::Integer, n_coeff_A::Integer, n_coeff_f::Integer, d;
                         degree::Integer=3, taper_frac::Real=0.1)
    k >= 1 || error("k must be a positive integer, got $k.")
    0 < taper_frac <= 0.5 || error(
        "taper_frac must be in (0, 0.5] (each sub-pulse has one taper region at each " *
        "edge; 0.5 is the largest that still leaves room for both), got $taper_frac."
    )






    n_coeff_A >= degree + 1 || error(
        "n_coeff_A must be >= degree+1 = $(degree+1) for a clamped B-spline of degree $degree, got $n_coeff_A."
    )
    n_coeff_f >= degree + 1 || error(
        "n_coeff_f must be >= degree+1 = $(degree+1) for a clamped B-spline of degree $degree, got $n_coeff_f."
    )
    T_max = d.timespan[2] - d.timespan[1]
    gap_scale = T_max / (2k)
    dur_scale = T_max / (2k)
    dur_floor = T_max * 1e-3





    typical_duration = max(dur_scale, 1e-30)


































    Omega_naive = pi / typical_duration
    Omega_power = d.FWHM
    Omega_adiabatic = sqrt(d.FWHM / typical_duration)
    Omega_target = max(Omega_naive, Omega_power, Omega_adiabatic)

















    amp_scale = (d.kappa_t / (4 * d.g_mean * d.sqrt_kappa_e)) * Omega_target
    freq_scale = d.FWHM
    return CompositePulse(k, n_coeff_A, n_coeff_f, degree, T_max,
                           gap_scale, dur_scale, dur_floor, amp_scale, freq_scale, Float64(taper_frac))
end

n_params(pulse::CompositePulse) = 3 * pulse.k + pulse.k * pulse.n_coeff_A + pulse.k * pulse.n_coeff_f

_softplus(x) = x > 30 ? x : log1p(exp(x))
_softplus_inv(y) = y + log(-expm1(-y))


function unpack(pulse::CompositePulse, u::AbstractVector)
    n = n_params(pulse)
    length(u) == n || error(
        "raw parameter vector has length $(length(u)), but this CompositePulse " *
        "(k=$(pulse.k), n_coeff_A=$(pulse.n_coeff_A), n_coeff_f=$(pulse.n_coeff_f)) needs $n."
    )
    k = pulse.k
    nA = k * pulse.n_coeff_A
    nf = k * pulse.n_coeff_f
    length(u) == 3k + nA + nf || error(
        "unpack: internal n_params mismatch (length(u)=$(length(u)), 3k+nA+nf=$(3k+nA+nf))."
    )
    raw_gap = u[1:k]
    raw_dur = u[k+1:2k]
    raw_phi0 = u[2k+1:3k]
    raw_cA = reshape(u[3k+1:3k+nA], pulse.n_coeff_A, k)
    raw_cf = reshape(u[3k+nA+1:3k+nA+nf], pulse.n_coeff_f, k)
    size(raw_cA) == (pulse.n_coeff_A, k) || error(
        "unpack: raw_cA size $(size(raw_cA)) != ($(pulse.n_coeff_A), $k)."
    )
    size(raw_cf) == (pulse.n_coeff_f, k) || error(
        "unpack: raw_cf size $(size(raw_cf)) != ($(pulse.n_coeff_f), $k)."
    )
    return raw_gap, raw_dur, raw_phi0, raw_cA, raw_cf
end

function _coeff_len(raw, n_coeff::Integer, k::Integer, name::AbstractString)
    if raw isa AbstractMatrix
        size(raw) == (n_coeff, k) || error(
            "pack: $name size $(size(raw)) != ($n_coeff, $k)."
        )
        return n_coeff * k
    end
    length(raw) == n_coeff * k || error(
        "pack: $name has length $(length(raw)), expected $n_coeff×$k = $(n_coeff * k)."
    )
    return length(raw)
end

function pack(pulse::CompositePulse, raw_gap, raw_dur, raw_phi0, raw_cA, raw_cf)
    k = pulse.k
    length(raw_gap) == k || error("pack: raw_gap length $(length(raw_gap)) != k=$k.")
    length(raw_dur) == k || error("pack: raw_dur length $(length(raw_dur)) != k=$k.")
    length(raw_phi0) == k || error("pack: raw_phi0 length $(length(raw_phi0)) != k=$k.")
    _coeff_len(raw_cA, pulse.n_coeff_A, k, "raw_cA")
    _coeff_len(raw_cf, pulse.n_coeff_f, k, "raw_cf")
    packed = vcat(vec(raw_gap), vec(raw_dur), vec(raw_phi0), vec(raw_cA), vec(raw_cf))
    length(packed) == n_params(pulse) || error(
        "pack: packed length $(length(packed)) != n_params=$(n_params(pulse))."
    )
    return packed
end


function decode(pulse::CompositePulse, u::AbstractVector)
    raw_gap, raw_dur, raw_phi0, raw_cA, raw_cf = unpack(pulse, u)
    k = pulse.k
    T = eltype(u)
    gap = pulse.gap_scale .* _softplus.(raw_gap)
    dur = pulse.dur_scale .* _softplus.(raw_dur) .+ pulse.dur_floor
    t_start = Vector{T}(undef, k)
    t_end = Vector{T}(undef, k)
    t = zero(T)
    @inbounds for i in 1:k
        t += gap[i]
        t_start[i] = t
        t += dur[i]
        t_end[i] = t
    end
    phi0 = raw_phi0
    cA = pulse.amp_scale .* _softplus.(raw_cA)
    cf = pulse.freq_scale .* raw_cf
    length(t_start) == k && length(t_end) == k || error(
        "decode: t_start/t_end lengths $(length(t_start))/$(length(t_end)) != k=$k."
    )
    size(cA) == (pulse.n_coeff_A, k) || error(
        "decode: cA size $(size(cA)) != ($(pulse.n_coeff_A), $k)."
    )
    size(cf) == (pulse.n_coeff_f, k) || error(
        "decode: cf size $(size(cf)) != ($(pulse.n_coeff_f), $k)."
    )
    return t_start, t_end, phi0, cA, cf
end


function initial_guess(pulse::CompositePulse; seed::Integer=42)
    rng = Random.Xoshiro(seed)
    k, nA, nf = pulse.k, pulse.n_coeff_A, pulse.n_coeff_f
    raw_gap = _softplus_inv.(0.3 .+ 0.7 .* rand(rng, k))
    raw_dur = _softplus_inv.(0.5 .+ 0.7 .* rand(rng, k))
    raw_phi0 = 2 * pi .* rand(rng, k) .- pi
    raw_cA = _softplus_inv.(0.5 .+ 1.0 .* rand(rng, nA, k))
    raw_cf = 0.3 .* randn(rng, nf, k)
    return pack(pulse, raw_gap, raw_dur, raw_phi0, raw_cA, raw_cf)
end


function _subpulse_knots(pulse::CompositePulse, t_start::AbstractVector, t_end::AbstractVector)
    k = pulse.k
    length(t_start) == k && length(t_end) == k || error(
        "_subpulse_knots: t_start/t_end lengths $(length(t_start))/$(length(t_end)) != k=$k."
    )
    degree = pulse.degree
    T = eltype(t_start)
    knots_A_list = Vector{Vector{T}}(undef, k)
    knots_f_list = Vector{Vector{T}}(undef, k)
    @inbounds for i in 1:k
        knots_A_list[i] = make_clamped_knots(pulse.n_coeff_A, t_start[i], t_end[i], degree)
        knots_f_list[i] = make_clamped_knots(pulse.n_coeff_f, t_start[i], t_end[i], degree)
    end
    return knots_A_list, knots_f_list
end


function build_E_of_t(pulse::CompositePulse, u::AbstractVector)
    t_start, t_end, phi0, cA, cf = decode(pulse, u)
    k = pulse.k
    degree = pulse.degree
    T = eltype(t_start)
    knots_A_list, knots_f_list = _subpulse_knots(pulse, t_start, t_end)

    knots_fp_list = Vector{Vector{T}}(undef, k)
    d_f_list = Vector{Vector{T}}(undef, k)
    phase_offset = Vector{T}(undef, k)
    running = zero(T)
    @inbounds for i in 1:k
        knots_fp, d_f = bspline_antiderivative(view(cf, :, i), knots_f_list[i], degree)
        knots_fp_list[i] = knots_fp
        d_f_list[i] = d_f
        phase_offset[i] = running + phi0[i]
        running = phase_offset[i] + d_f[end]
    end

    taper_frac = pulse.taper_frac
    return function E_of_t(t)
        @inbounds for i in 1:k
            if t >= t_start[i] && t <= t_end[i]
                A_spline = bspline_eval(t, view(cA, :, i), knots_A_list[i], degree)
                A = A_spline * _taper_window(t, t_start[i], t_end[i], taper_frac)
                phi = bspline_eval(t, d_f_list[i], knots_fp_list[i], degree + 1) + phase_offset[i]
                return A * cis(phi)
            end
        end
        return zero(Complex{T})
    end
end


function build_A_f_of_t(pulse::CompositePulse, u::AbstractVector)
    t_start, t_end, _, cA, cf = decode(pulse, u)
    k = pulse.k
    degree = pulse.degree
    T = eltype(t_start)
    knots_A_list, knots_f_list = _subpulse_knots(pulse, t_start, t_end)

    taper_frac = pulse.taper_frac
    A_of_t = function (t)
        @inbounds for i in 1:k
            if t >= t_start[i] && t <= t_end[i]
                A_spline = bspline_eval(t, view(cA, :, i), knots_A_list[i], degree)
                return A_spline * _taper_window(t, t_start[i], t_end[i], taper_frac)
            end
        end
        return zero(T)
    end
    f_of_t = function (t)
        @inbounds for i in 1:k
            if t >= t_start[i] && t <= t_end[i]
                return bspline_eval(t, view(cf, :, i), knots_f_list[i], degree)
            end
        end
        return zero(T)
    end
    return A_of_t, f_of_t
end


function total_area(pulse::CompositePulse, u::AbstractVector)
    t_start, t_end, _, cA, _ = decode(pulse, u)
    T = eltype(t_start)
    area = zero(T)
    @inbounds for i in 1:pulse.k
        knots = make_clamped_knots(pulse.n_coeff_A, t_start[i], t_end[i], pulse.degree)
        area += bspline_area(view(cA, :, i), knots, pulse.degree)
    end
    return area
end


function pulse_duration(pulse::CompositePulse, u::AbstractVector)
    _, t_end, _, _, _ = decode(pulse, u)
    return t_end[end]
end
