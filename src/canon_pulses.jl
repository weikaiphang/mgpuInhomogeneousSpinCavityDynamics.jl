# ============================================================
# CANONICAL WARM-STARTS (HS1 / CORPSE / BB1)
#
# These build a raw parameter vector `u` for an already-constructed
# CompositePulse. They are optimizer INITIALISATIONS, not drop-in
# replacements for analytic HS1/CORPSE/BB1 on the cavity: build_E_of_t
# still applies the C^∞ taper window, so "rectangular" composite
# envelopes are windowed, and HS1 coefficients are collocation samples
# rather than an L2 projection onto the B-spline basis.
#
# k is discrete -- each seed requires a fixed pulse.k (1 / 5 / 7).
# Use optimise_composite_pulse_over_k to run independent continuous
# optimisations in parallel, one per k, and keep the best.
# ============================================================

"""
    k_of_seed_kind(kind::Symbol) -> Int

Sub-pulse count required by a canonical seed: `:hs1 => 1`, `:corpse => 5`,
`:bb1 => 7`. `:random` has no fixed `k` (the caller chooses it).
"""
function k_of_seed_kind(kind::Symbol)
    kind === :hs1 && return 1
    kind === :corpse && return 5
    kind === :bb1 && return 7
    kind === :random && error(
        ":random has no fixed k -- pass an explicit k (e.g. specs=((k, :random), ...))."
    )
    error("Unknown seed kind $(kind). Use :hs1, :corpse, :bb1, or :random.")
end

"""
    seed_hs1(pulse, Omega_max, beta, mu)

Raw `u` for a complex hyperbolic-secant (HS1) adiabatic passage.
Requires `pulse.k == 1`. `Omega_max` is a cavity-input peak for `E(t)`
(same units as `pulse.amp_scale`), not a spin Rabi frequency.
"""
function seed_hs1(pulse::CompositePulse, Omega_max::Real, beta::Real, mu::Real)
    pulse.k == 1 || error("HS1 seed requires k=1, got k=$(pulse.k).")
    nA, nf = pulse.n_coeff_A, pulse.n_coeff_f

    # Cover ±3/β of the sech envelope (sech(3) ≈ 0.1 at the edge)
    dur = 6.0 / beta

    raw_gap = [_softplus_inv(1e-5 / pulse.gap_scale)]
    raw_dur = [_softplus_inv(max(1e-5, dur - pulse.dur_floor) / pulse.dur_scale)]
    raw_phi0 = [0.0]

    raw_cA = zeros(nA, 1)
    raw_cf = zeros(nf, 1)

    # Collocate sech / tanh at equally spaced local times (not an L2
    # projection onto the B-spline basis; clamped cubics interpolate
    # endpoints only).
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

"""
    seed_composite_with_ghosts(pulse, Omega_max, areas, phase_jumps)

Active rectangular (constant-coefficient) sub-pulses separated by
near-zero-amplitude "ghost" sub-pulses whose frequency-spline integral
injects `phase_jumps[i]` into the next active pulse's phase offset.
Requires `pulse.k == 2*length(areas)-1` and
`length(phase_jumps) == length(areas)-1`.
"""
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
            # Constant-coefficient spline: ∫f dt = c * duration
            req_f = phase_jumps[idx] / ghost_dur_target
            raw_cf[:, i] .= req_f / pulse.freq_scale
        end
    end

    return pack(pulse, raw_gap, raw_dur, raw_phi0, raw_cA, raw_cf)
end

"""
    seed_corpse(pulse, Omega_max)

k=5. Net π with phases 0, π, 0 (jumps +π, −π). Cancels first-order
detuning error in the ideal rectangular limit.
"""
function seed_corpse(pulse::CompositePulse, Omega_max::Real)
    areas = [7pi / 3, 5pi / 3, pi / 3]
    phase_jumps = [pi, -pi]
    return seed_composite_with_ghosts(pulse, Omega_max, areas, phase_jumps)
end

"""
    seed_bb1(pulse, Omega_max)

k=7. Wimperis BB1 around a π pulse: areas π, π, 2π, π with
φ = arccos(−1/4) and phases 0, φ, 3φ, φ.
"""
function seed_bb1(pulse::CompositePulse, Omega_max::Real)
    phi = acos(-0.25)
    areas = [pi, pi, 2pi, pi]
    phase_jumps = [phi, 2 * phi, -2 * phi]
    return seed_composite_with_ghosts(pulse, Omega_max, areas, phase_jumps)
end

"""
    seed_canonical(pulse, kind; Omega_max=pulse.amp_scale, beta=nothing, mu=nothing, seed=42)

Dispatch on `kind` (`:hs1`, `:corpse`, `:bb1`, `:random`) to a raw `u`
for `pulse`. Defaults: `Omega_max` is this pulse's own `amp_scale`
(cavity-input units); HS1 `beta` is chosen so the sech width is `T_max/2`;
HS1 `mu` is chosen so the peak chirp is `freq_scale` (the ensemble FWHM).
"""
function seed_canonical(
    pulse::CompositePulse,
    kind::Symbol;
    Omega_max::Real=pulse.amp_scale,
    beta=nothing,
    mu=nothing,
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
    else
        error("Unknown seed kind $(kind). Use :hs1, :corpse, :bb1, or :random.")
    end
end
