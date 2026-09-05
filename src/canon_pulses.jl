function k_of_seed_kind(kind::Symbol)
    kind === :hs1 && return 1
    kind === :corpse && return 5
    kind === :bb1 && return 7
    kind === :bir4 && return 4
    kind === :random && error(
        ":random has no fixed k -- pass an explicit k (e.g. specs=((k, :random), ...))."
    )
    error("Unknown seed kind $(kind). Use :hs1, :corpse, :bb1, :bir4, or :random.")
end

function seed_hs1(pulse::CompositePulse, Omega_max::Real, beta::Real, mu::Real)
    pulse.k == 1 || error("HS1 seed requires k=1, got k=$(pulse.k).")
    nA, nf = pulse.n_coeff_A, pulse.n_coeff_f

    dur = 6.0 / beta

    raw_gap = [_softplus_inv(1e-5 / pulse.gap_scale)]
    raw_dur = [_softplus_inv(max(1e-5, dur - pulse.dur_floor) / pulse.dur_scale)]
    raw_phi0 = [0.0]

    raw_cA = zeros(nA, 1)
    raw_cf = zeros(nf, 1)

    t_nodes = range(-dur / 2, dur / 2; length=nA)
    for i in 1:nA
        A_val = Omega_max * sech(beta * t_nodes[i])
        raw_cA[i, 1] = _softplus_inv(max(A_val, 1e-10) / pulse.amp_scale)
    end

    f_nodes = range(-dur / 2, dur / 2; length=nf)
    for i in 1:nf
        f_val = mu * beta * tanh(beta * f_nodes[i])
        raw_cf[i, 1] = f_val / pulse.freq_scale
    end

    return pack(pulse, raw_gap, raw_dur, raw_phi0, raw_cA, raw_cf)
end

function seed_composite_with_ghosts(
    pulse::CompositePulse,
    Omega_max::Real,
    areas::AbstractVector{<:Real},
    phase_jumps::AbstractVector{<:Real},
)
    expected_k = 2 * length(areas) - 1
    pulse.k == expected_k || error("Composite seed needs k=$(expected_k), got k=$(pulse.k).")
    length(phase_jumps) == length(areas) - 1 || error(
        "Need $(length(areas)-1) phase jumps for $(length(areas)) active pulses, got $(length(phase_jumps))."
    )

    nA, nf = pulse.n_coeff_A, pulse.n_coeff_f

    raw_gap = zeros(pulse.k)
    raw_dur = zeros(pulse.k)
    raw_phi0 = zeros(pulse.k)
    raw_cA = zeros(nA, pulse.k)
    raw_cf = zeros(nf, pulse.k)

    ghost_dur_target = pulse.dur_floor + 1e-6

    for i in 1:pulse.k
        raw_gap[i] = _softplus_inv(1e-6 / pulse.gap_scale)

        if isodd(i)
            idx = div(i, 2) + 1
            dur = areas[idx] / Omega_max
            raw_dur[i] = _softplus_inv(max(1e-6, dur - pulse.dur_floor) / pulse.dur_scale)
            raw_cA[:, i] .= _softplus_inv(Omega_max / pulse.amp_scale)
            raw_cf[:, i] .= 0.0
        else
            idx = div(i, 2)
            raw_dur[i] = _softplus_inv((ghost_dur_target - pulse.dur_floor) / pulse.dur_scale)
            raw_cA[:, i] .= -30.0
            req_f = phase_jumps[idx] / ghost_dur_target
            raw_cf[:, i] .= req_f / pulse.freq_scale
        end
    end

    return pack(pulse, raw_gap, raw_dur, raw_phi0, raw_cA, raw_cf)
end

function seed_corpse(pulse::CompositePulse, Omega_max::Real)
    areas = [7pi / 3, 5pi / 3, pi / 3]
    phase_jumps = [pi, -pi]
    return seed_composite_with_ghosts(pulse, Omega_max, areas, phase_jumps)
end

function seed_bb1(pulse::CompositePulse, Omega_max::Real)
    phi = acos(-0.25)
    areas = [pi, pi, 2pi, pi]
    phase_jumps = [phi, 2 * phi, -2 * phi]
    return seed_composite_with_ghosts(pulse, Omega_max, areas, phase_jumps)
end

function seed_bir4(
    pulse::CompositePulse,
    Omega_max::Real;
    flip_angle::Real=pi,
    duration=nothing,
    beta::Real=10.0,
    kappa::Real=atan(20.0),
    dwmax::Real=pulse.freq_scale,
)
    pulse.k == 4 || error("BIR-4 seed requires k=4, got k=$(pulse.k).")
    beta > 0 || error("BIR-4 beta must be > 0, got $(beta).")
    0 < kappa < pi / 2 || error("BIR-4 kappa must be in (0, pi/2), got $(kappa).")
    nA, nf = pulse.n_coeff_A, pulse.n_coeff_f

    T_p = duration === nothing ? pulse.T_max / 2 : Float64(duration)
    T_p > 0 || error("BIR-4 duration must be > 0, got $(T_p).")
    seg_dur = T_p / 4

    raw_gap = fill(_softplus_inv(1e-5 / pulse.gap_scale), 4)
    raw_dur = fill(
        _softplus_inv(max(1e-5, seg_dur - pulse.dur_floor) / pulse.dur_scale),
        4,
    )

    dphi = pi + flip_angle / 2
    raw_phi0 = [0.0, dphi, 0.0, -dphi]

    raw_cA = zeros(nA, 4)
    raw_cf = zeros(nf, 4)

    tan_k = tan(kappa)
    amp_falling(s) = Omega_max * tanh(beta * (1 - s))
    amp_rising(s) = Omega_max * tanh(beta * s)
    fm_up(s) = dwmax * tan(kappa * s) / tan_k
    fm_down(s) = dwmax * tan(kappa * (s - 1)) / tan_k

    sA = range(0.0, 1.0; length=nA)
    sf = range(0.0, 1.0; length=nf)
    seg_forms = ((amp_falling, fm_up), (amp_rising, fm_down),
                 (amp_falling, fm_up), (amp_rising, fm_down))
    for (col, (amp_fn, fm_fn)) in enumerate(seg_forms)
        for i in 1:nA
            A_val = amp_fn(sA[i])
            raw_cA[i, col] = _softplus_inv(max(A_val, 1e-10) / pulse.amp_scale)
        end
        for i in 1:nf
            raw_cf[i, col] = fm_fn(sf[i]) / pulse.freq_scale
        end
    end

    return pack(pulse, raw_gap, raw_dur, raw_phi0, raw_cA, raw_cf)
end

function seed_canonical(
    pulse::CompositePulse,
    kind::Symbol;
    Omega_max::Real=pulse.amp_scale,
    beta=nothing,
    mu=nothing,
    flip_angle::Real=pi,
    seed::Integer=42,
)
    if kind === :random
        return initial_guess(pulse; seed=seed)
    elseif kind === :hs1
        β = beta === nothing ? 6 / max(pulse.T_max / 2, 1e-30) : Float64(beta)
        μ = mu === nothing ? pulse.freq_scale / β : Float64(mu)
        return seed_hs1(pulse, Omega_max, β, μ)
    elseif kind === :corpse
        return seed_corpse(pulse, Omega_max)
    elseif kind === :bb1
        return seed_bb1(pulse, Omega_max)
    elseif kind === :bir4
        return beta === nothing ?
               seed_bir4(pulse, Omega_max; flip_angle=flip_angle) :
               seed_bir4(pulse, Omega_max; flip_angle=flip_angle, beta=Float64(beta))
    else
        error("Unknown seed kind $(kind). Use :hs1, :corpse, :bb1, :bir4, or :random.")
    end
end
