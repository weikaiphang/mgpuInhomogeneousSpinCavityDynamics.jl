# ============================================================
# PULSE_SPEC design tables, system binding, and Ttotal derivation.
#
# Pulse-intrinsic times (leading delays, durations, gaps) are catalogued
# first. Ttotal is derived afterwards from drive support, echo times
# (if any), and a short SYSTEM-dependent settling tail.
# ============================================================

const N_SIGMA_SUPPORT = 5.0
const ECHO_HALF_WINDOW = 10e-6
const SETTLE_N_KAPPA = 10.0
const SETTLE_N_FWHM = 3.0
const SETTLE_MIN = 1e-6

const PAPER_WURST_AMP_PREFACTOR = 2.0e4
const PAPER_SIGNAL_AMP_PREFACTOR = 0.332
const WURST_EDGE_FRAC = 1e-4

# ============================================================
# Physical helpers from SYSTEM_CONFIG
# ============================================================

function system_kappa_t(sys)
    return sys.kappa_e + sys.kappa_i
end

function system_fwhm(sys)
    return sys.freq_inhomogeneity.FWHM
end

function composite_dummy_d(sys, T_max::Float64)
    kappa_t = system_kappa_t(sys)
    g_mean = physical_g_mean(sys)
    return (
        timespan = (0.0, T_max),
        FWHM = system_fwhm(sys),
        kappa_t = kappa_t,
        g_mean = g_mean,
        sqrt_kappa_e = sqrt(sys.kappa_e),
        kappa_e = sys.kappa_e,
    )
end

function default_3arp_omega_max(sys)
    kappa_t = system_kappa_t(sys)
    g_mean = physical_g_mean(sys)
    return pi * kappa_t / (4 * g_mean * sqrt(sys.kappa_e)) * system_fwhm(sys)
end

# ============================================================
# Drive support and Ttotal
# ============================================================

function segment_support(seg)
    kind = seg.kind
    if kind === :gaussian
        half = N_SIGMA_SUPPORT * Float64(seg.sigma)
        return (Float64(seg.t0) - half, Float64(seg.t0) + half)
    elseif kind === :wurst
        half = Float64(seg.duration) / 2
        return (Float64(seg.t_center) - half, Float64(seg.t_center) + half)
    elseif kind === :composite_record
        return (Float64(seg.t_start), Float64(seg.t_end))
    else
        error("Unknown segment kind $(kind) in support calculation.")
    end
end

function pulse_drive_span(segments)
    t0 = Inf
    t1 = -Inf
    for seg in segments
        a, b = segment_support(seg)
        t0 = min(t0, a)
        t1 = max(t1, b)
    end
    return t0, t1
end

function gaussian_t0(segments)
    for seg in segments
        if seg.kind === :gaussian
            return Float64(seg.t0)
        end
    end
    return nothing
end

function control_centers(segments)
    centers = Float64[]
    for seg in segments
        if seg.kind === :wurst
            push!(centers, Float64(seg.t_center))
        elseif seg.kind === :composite_record
            push!(centers, 0.5 * (Float64(seg.t_start) + Float64(seg.t_end)))
        end
    end
    return centers
end

function pulse_echo_end(family::Symbol, segments)
    family in (:rose, :arp3_signal) || return 0.0
    t_sig = gaussian_t0(segments)
    t_sig === nothing && return 0.0
    centers = control_centers(segments)
    isempty(centers) && return 0.0
    if family === :rose && length(centers) >= 2
        echo1 = 2 * centers[1] - t_sig
        echo2 = centers[1] + centers[2] - t_sig
        return max(echo1, echo2) + ECHO_HALF_WINDOW
    end
    t_pi = sum(centers) / length(centers)
    return (2 * t_pi - t_sig) + ECHO_HALF_WINDOW
end

function t_settle(sys)
    kappa_t = system_kappa_t(sys)
    FWHM = system_fwhm(sys)
    return max(SETTLE_N_KAPPA / kappa_t, SETTLE_N_FWHM / FWHM, SETTLE_MIN)
end

function derive_ttotal(sys, PULSE_SPEC)
    t_drive0, t_drive_end = pulse_drive_span(PULSE_SPEC.segments)
    t_drive0 >= -1e-15 || error(
        "pulse support starts at $(t_drive0) s < 0 (Gaussian left tail or leading delay)."
    )
    t_echo_end = pulse_echo_end(PULSE_SPEC.family, PULSE_SPEC.segments)
    t_protocol_end = max(t_drive_end, t_echo_end)
    Ttotal = t_protocol_end + t_settle(sys)
    Ttotal > 0 || error("derived Ttotal must be positive, got $Ttotal.")
    return Ttotal
end

# ============================================================
# Binding analytic pulses to SYSTEM_CONFIG
# ============================================================

function wurst_segment(;
    name,
    t_center,
    duration,
    amp,
    bandwidth,
    n = 20.0,
    omega0 = 0.0,
    chirp_sign = 1.0,
    phase0 = 0.0,
    edge_frac = WURST_EDGE_FRAC,
)
    return (
        name = name,
        kind = :wurst,
        t_center = Float64(t_center),
        duration = Float64(duration),
        amp = Float64(amp),
        bandwidth = Float64(bandwidth),
        n = Float64(n),
        omega0 = Float64(omega0),
        chirp_sign = Float64(chirp_sign),
        phase0 = Float64(phase0),
        edge_frac = Float64(edge_frac),
    )
end

function gaussian_segment(; name, t0, sigma, amp, omega = 0.0, phase = 0.0)
    return (
        name = name,
        kind = :gaussian,
        t0 = Float64(t0),
        sigma = Float64(sigma),
        amp = Float64(amp),
        omega = Float64(omega),
        phase = Float64(phase),
    )
end

function paper_wurst_amp(sys, amp_mult)
    return amp_mult * 0.5 * sqrt(sys.kappa_e) * PAPER_WURST_AMP_PREFACTOR
end

function paper_signal_amp(sys, amp_mult)
    return amp_mult * 0.5 * sqrt(sys.kappa_e) * PAPER_SIGNAL_AMP_PREFACTOR
end

function make_pulse_spec(; family::Symbol, canonical::Bool, design, segments)
    return (
        family = family,
        canonical = canonical,
        design = design,
        segments = segments,
    )
end

function bind_rase_wurst(design, sys)
    dur = design.duration
    t_center = design.t_start + dur / 2
    seg = wurst_segment(;
        name = "RASE WURST",
        t_center = t_center,
        duration = dur,
        amp = paper_wurst_amp(sys, design.amp_mult),
        bandwidth = design.bw_fwhm_mult * system_fwhm(sys),
        n = design.n,
        omega0 = design.omega0_over_fwhm * system_fwhm(sys),
        chirp_sign = design.chirp_sign,
        phase0 = design.phase0,
    )
    return make_pulse_spec(;
        family = :rase_wurst,
        canonical = design.canonical,
        design = design,
        segments = (seg,),
    )
end

function bind_rose(design, sys)
    t0 = design.t0
    sigma = design.sigma
    dur = design.wurst_duration
    sig_tail = t0 + N_SIGMA_SUPPORT * sigma
    w1_start = sig_tail + design.gap_after_signal
    t_center1 = w1_start + dur / 2
    w1_end = w1_start + dur
    w2_start = w1_end + design.gap_between
    t_center2 = w2_start + dur / 2

    sig = gaussian_segment(;
        name = "Gaussian input signal",
        t0 = t0,
        sigma = sigma,
        amp = paper_signal_amp(sys, design.signal_amp_mult),
        omega = design.signal_omega_over_fwhm * system_fwhm(sys),
        phase = design.signal_phase,
    )
    w1 = wurst_segment(;
        name = "First WURST pulse",
        t_center = t_center1,
        duration = dur,
        amp = paper_wurst_amp(sys, design.wurst_amp_mult),
        bandwidth = design.bw_fwhm_mult * system_fwhm(sys),
        n = design.n,
        omega0 = design.omega0_over_fwhm * system_fwhm(sys),
        chirp_sign = design.chirp_sign,
        phase0 = design.phase0,
    )
    w2 = wurst_segment(;
        name = "Second WURST pulse",
        t_center = t_center2,
        duration = dur,
        amp = paper_wurst_amp(sys, design.wurst_amp_mult),
        bandwidth = design.bw_fwhm_mult * system_fwhm(sys),
        n = design.n,
        omega0 = design.omega0_over_fwhm * system_fwhm(sys),
        chirp_sign = design.chirp_sign,
        phase0 = design.phase0,
    )
    return make_pulse_spec(;
        family = :rose,
        canonical = design.canonical,
        design = design,
        segments = (sig, w1, w2),
    )
end

function bind_rose_paper(design, sys)
    sig = gaussian_segment(;
        name = "Gaussian input signal",
        t0 = design.t0,
        sigma = design.sigma,
        amp = paper_signal_amp(sys, design.signal_amp_mult),
        omega = 0.0,
        phase = 0.0,
    )
    w1 = wurst_segment(;
        name = "First WURST pulse",
        t_center = design.t_center1,
        duration = design.wurst_duration,
        amp = paper_wurst_amp(sys, 1.0),
        bandwidth = 5.0 * system_fwhm(sys),
        n = 20.0,
        chirp_sign = 1.0,
    )
    w2 = wurst_segment(;
        name = "Second WURST pulse",
        t_center = design.t_center2,
        duration = design.wurst_duration,
        amp = paper_wurst_amp(sys, 1.0),
        bandwidth = 5.0 * system_fwhm(sys),
        n = 20.0,
        chirp_sign = 1.0,
    )
    return make_pulse_spec(;
        family = :rose,
        canonical = true,
        design = design,
        segments = (sig, w1, w2),
    )
end

function bind_3arp(design, sys)
    Tb = design.T_budget
    dur1 = Tb / 4
    dur2 = 2 * dur1
    bw = design.bw_fwhm_mult * system_fwhm(sys)
    amp1 = design.omega_mult * default_3arp_omega_max(sys)
    amp2 = amp1 / sqrt(2)
    t1_start = design.t_start
    t1_end = t1_start + dur1
    t2_end = t1_end + dur2
    t3_end = t2_end + dur1
    common_n = 20.0
    segs = (
        wurst_segment(;
            name = "3ARP segment 1 (+k)",
            t_center = (t1_start + t1_end) / 2,
            duration = dur1,
            amp = amp1,
            bandwidth = bw,
            n = common_n,
            chirp_sign = 1.0,
        ),
        wurst_segment(;
            name = "3ARP segment 2 (+k/2)",
            t_center = (t1_end + t2_end) / 2,
            duration = dur2,
            amp = amp2,
            bandwidth = bw,
            n = common_n,
            chirp_sign = 1.0,
        ),
        wurst_segment(;
            name = "3ARP segment 3 (+k)",
            t_center = (t2_end + t3_end) / 2,
            duration = dur1,
            amp = amp1,
            bandwidth = bw,
            n = common_n,
            chirp_sign = 1.0,
        ),
    )
    if design.with_signal
        sig = gaussian_segment(;
            name = "Gaussian input signal",
            t0 = design.signal_t0,
            sigma = design.signal_sigma,
            amp = paper_signal_amp(sys, design.signal_amp_mult),
        )
        family = :arp3_signal
        segments = (sig, segs...)
    else
        family = :arp3
        segments = segs
    end
    return make_pulse_spec(;
        family = family,
        canonical = design.canonical,
        design = design,
        segments = segments,
    )
end

function composite_record_from(pulse::ISC.CompositePulse, u, name, canonical, family, design)
    t_start, t_end, _, _, _ = ISC.decode(pulse, u)
    return make_pulse_spec(;
        family = family,
        canonical = canonical,
        design = design,
        segments = ((
            name = name,
            kind = :composite_record,
            t_start = Float64(t_start[1]),
            t_end = Float64(t_end[end]),
            k = pulse.k,
            n_coeff_A = pulse.n_coeff_A,
            n_coeff_f = pulse.n_coeff_f,
            degree = pulse.degree,
            T_max = pulse.T_max,
            gap_scale = pulse.gap_scale,
            dur_scale = pulse.dur_scale,
            dur_floor = pulse.dur_floor,
            amp_scale = pulse.amp_scale,
            freq_scale = pulse.freq_scale,
            taper_frac = pulse.taper_frac,
            u = collect(Float64.(u)),
        ),),
    )
end

function reconstruct_composite_pulse(seg)
    return ISC.CompositePulse(
        seg.k,
        seg.n_coeff_A,
        seg.n_coeff_f,
        seg.degree,
        seg.T_max,
        seg.gap_scale,
        seg.dur_scale,
        seg.dur_floor,
        seg.amp_scale,
        seg.freq_scale,
        seg.taper_frac,
    )
end

function bind_canonical_composite(design, sys)
    d = composite_dummy_d(sys, design.T_max)
    nA = design.n_coeff
    nf = design.n_coeff
    k = ISC.k_of_seed_kind(design.seed_kind)
    pulse = ISC.CompositePulse(k, nA, nf, d; taper_frac = design.taper_frac)
    kwargs = (; Omega_max = design.omega_mult * pulse.amp_scale)
    if design.seed_kind === :hs1 && design.beta !== nothing
        kwargs = merge(kwargs, (; beta = design.beta))
    end
    if design.seed_kind === :hs1 && design.mu !== nothing
        kwargs = merge(kwargs, (; mu = design.mu))
    end
    u = ISC.seed_canonical(pulse, design.seed_kind; kwargs...)
    family = design.seed_kind
    return composite_record_from(
        pulse, u, String(design.seed_kind), design.canonical, family, design,
    )
end

function bind_block_pi(design, sys)
    d = composite_dummy_d(sys, design.T_max)
    pulse = ISC.CompositePulse(1, design.n_coeff, design.n_coeff, d; taper_frac = design.taper_frac)
    Omega_max = design.omega_mult * pulse.amp_scale
    gap = max(design.t_start, pulse.gap_scale * 1e-3)
    dur = max(design.duration, pulse.dur_floor + pulse.dur_scale * 1e-3)
    raw_gap = [ISC._softplus_inv(gap / pulse.gap_scale)]
    raw_dur = [ISC._softplus_inv((dur - pulse.dur_floor) / pulse.dur_scale)]
    raw_phi0 = [0.0]
    raw_cA = fill(ISC._softplus_inv(Omega_max / pulse.amp_scale), pulse.n_coeff_A, 1)
    raw_cf = zeros(pulse.n_coeff_f, 1)
    u = ISC.pack(pulse, raw_gap, raw_dur, raw_phi0, raw_cA, raw_cf)
    return composite_record_from(
        pulse, u, "block pi", design.canonical, :block_pi, design,
    )
end

function bind_random_composite(design, sys)
    d = composite_dummy_d(sys, design.T_max)
    pulse = ISC.CompositePulse(
        design.k, design.n_coeff, design.n_coeff, d; taper_frac = design.taper_frac,
    )
    u = ISC.initial_guess(pulse; seed = design.seed)
    return composite_record_from(
        pulse, u, "random composite seed=$(design.seed)", design.canonical,
        :random_composite, design,
    )
end

function bind_pulse(design, sys)
    fam = design.family
    if fam === :rase_wurst
        return bind_rase_wurst(design, sys)
    elseif fam === :rose
        return hasproperty(design, :t_center1) ? bind_rose_paper(design, sys) : bind_rose(design, sys)
    elseif fam === :arp3
        return bind_3arp(design, sys)
    elseif fam === :hs1 || fam === :corpse || fam === :bb1
        return bind_canonical_composite(design, sys)
    elseif fam === :block_pi
        return bind_block_pi(design, sys)
    elseif fam === :random_composite
        return bind_random_composite(design, sys)
    else
        error("Unknown pulse family $(fam).")
    end
end

# ============================================================
# Solver-legal PULSE_CONFIG (closures built only here)
# ============================================================

function materialize_pulse_config(PULSE_SPEC)
    segs = Any[]
    for seg in PULSE_SPEC.segments
        if seg.kind === :composite_record
            pulse = reconstruct_composite_pulse(seg)
            f = ISC.build_E_of_t(pulse, seg.u)
            push!(segs, (name = seg.name, kind = :custom, f = f))
        else
            push!(segs, seg)
        end
    end
    return tuple(segs...)
end

function pulse_config_is_valid(PULSE_CONFIG)
    try
        ISC.validate_pulse_config(PULSE_CONFIG)
        return true, ""
    catch err
        return false, sprint(showerror, err)
    end
end

# ============================================================
# Design tables
# ============================================================

function rase_designs()
    out = Any[]
    push!(out, (
        family = :rase_wurst,
        canonical = true,
        t_start = 70e-6,
        duration = 10e-6,
        amp_mult = 1.0,
        bw_fwhm_mult = 5.0,
        n = 20.0,
        chirp_sign = 1.0,
        omega0_over_fwhm = 0.0,
        phase0 = 0.0,
    ))
    t_starts = [5e-6, 20e-6, 50e-6]
    durations = [10e-6, 30e-6, 100e-6]
    amp_mults = [0.5, 1.0, 2.0]
    bw_mults = [3.0, 5.0, 8.0]
    for t_start in t_starts, duration in durations, amp_mult in amp_mults, bw in bw_mults
        push!(out, (
            family = :rase_wurst,
            canonical = false,
            t_start = t_start,
            duration = duration,
            amp_mult = amp_mult,
            bw_fwhm_mult = bw,
            n = 20.0,
            chirp_sign = 1.0,
            omega0_over_fwhm = 0.0,
            phase0 = 0.0,
        ))
    end
    base = (
        family = :rase_wurst, canonical = false,
        t_start = 20e-6, duration = 10e-6, amp_mult = 1.0, bw_fwhm_mult = 5.0,
        n = 20.0, chirp_sign = 1.0, omega0_over_fwhm = 0.0, phase0 = 0.0,
    )
    for n in (10.0, 40.0)
        push!(out, merge(base, (n = n,)))
    end
    for s in (-1.0,)
        push!(out, merge(base, (chirp_sign = s,)))
    end
    for o in (0.25,)
        push!(out, merge(base, (omega0_over_fwhm = o,)))
    end
    for ph in (pi / 2,)
        push!(out, merge(base, (phase0 = ph,)))
    end
    return out
end

function rose_designs()
    out = Any[]
    push!(out, (
        family = :rose,
        canonical = true,
        t0 = 50e-6,
        sigma = 3e-6,
        signal_amp_mult = 1.0,
        wurst_duration = 400e-6,
        t_center1 = 300e-6,
        t_center2 = 800e-6,
    ))
    t0s = [20e-6, 50e-6]
    sigmas = [1.5e-6, 3e-6, 6e-6]
    durs = [100e-6, 400e-6]
    gaps_sig = [20e-6, 50e-6]
    gaps_12 = [50e-6, 100e-6]
    sig_amps = [0.5, 1.0, 2.0]
    w_amps = [0.5, 1.0]
    bw_mults = [5.0, 8.0]
    for t0 in t0s, sigma in sigmas, dur in durs, g1 in gaps_sig, g2 in gaps_12,
        sa in sig_amps, wa in w_amps, bw in bw_mults
        push!(out, (
            family = :rose,
            canonical = false,
            t0 = t0,
            sigma = sigma,
            signal_amp_mult = sa,
            signal_omega_over_fwhm = 0.0,
            signal_phase = 0.0,
            wurst_duration = dur,
            gap_after_signal = g1,
            gap_between = g2,
            wurst_amp_mult = wa,
            bw_fwhm_mult = bw,
            n = 20.0,
            chirp_sign = 1.0,
            omega0_over_fwhm = 0.0,
            phase0 = 0.0,
        ))
    end
    base = (
        family = :rose, canonical = false,
        t0 = 50e-6, sigma = 3e-6, signal_amp_mult = 1.0,
        signal_omega_over_fwhm = 0.0, signal_phase = 0.0,
        wurst_duration = 100e-6, gap_after_signal = 50e-6, gap_between = 100e-6,
        wurst_amp_mult = 1.0, bw_fwhm_mult = 5.0, n = 20.0,
        chirp_sign = 1.0, omega0_over_fwhm = 0.0, phase0 = 0.0,
    )
    push!(out, merge(base, (chirp_sign = -1.0,)))
    push!(out, merge(base, (n = 10.0,)))
    push!(out, merge(base, (signal_omega_over_fwhm = 0.25,)))
    push!(out, merge(base, (signal_phase = pi / 2,)))
    push!(out, merge(base, (omega0_over_fwhm = 0.25,)))
    push!(out, merge(base, (phase0 = pi / 2,)))
    return out
end

function arp3_designs()
    out = Any[]
    t_starts = [20e-6, 50e-6, 80e-6]
    budgets = [20e-6, 40e-6, 100e-6]
    bw_mults = [5.0, 8.0]
    omega_mults = [1.0, 2.0]
    for t_start in t_starts, Tb in budgets, bw in bw_mults, om in omega_mults
        canonical = (
            t_start == 50e-6 && Tb == 40e-6 && bw == 5.0 && om == 1.0
        )
        push!(out, (
            family = :arp3,
            canonical = canonical,
            t_start = t_start,
            T_budget = Tb,
            bw_fwhm_mult = bw,
            omega_mult = om,
            with_signal = false,
            signal_t0 = 15e-6,
            signal_sigma = 3e-6,
            signal_amp_mult = 1.0,
        ))
    end
    for t_start in (50e-6,), Tb in (40e-6, 100e-6), sa in (0.5, 1.0)
        t0 = 15e-6
        sigma = 3e-6
        tail = t0 + N_SIGMA_SUPPORT * sigma
        t_start >= tail + 5e-6 || continue
        push!(out, (
            family = :arp3,
            canonical = false,
            t_start = t_start,
            T_budget = Tb,
            bw_fwhm_mult = 5.0,
            omega_mult = 1.0,
            with_signal = true,
            signal_t0 = t0,
            signal_sigma = sigma,
            signal_amp_mult = sa,
        ))
    end
    return out
end

function canonical_composite_designs()
    out = Any[]
    T_maxs = [50e-6, 100e-6, 200e-6]
    omega_mults = [1.0, 2.0]
    for seed_kind in (:hs1, :corpse, :bb1)
        n_coeff = seed_kind === :hs1 ? 8 : 4
        for T_max in T_maxs, om in omega_mults
            canonical = (T_max == 100e-6 && om == 1.0)
            push!(out, (
                family = seed_kind,
                seed_kind = seed_kind,
                canonical = canonical,
                T_max = T_max,
                n_coeff = n_coeff,
                omega_mult = om,
                taper_frac = 0.1,
                beta = nothing,
                mu = nothing,
            ))
        end
    end
    push!(out, (
        family = :hs1, seed_kind = :hs1, canonical = false,
        T_max = 100e-6, n_coeff = 8, omega_mult = 1.0, taper_frac = 0.1,
        beta = 6 / 30e-6, mu = nothing,
    ))
    return out
end

function block_pi_designs()
    out = Any[]
    t_starts = [5e-6, 20e-6, 50e-6]
    durs = [5e-6, 15e-6, 40e-6]
    omega_mults = [1.0, 2.0]
    for t_start in t_starts, duration in durs, om in omega_mults
        T_max = max(100e-6, t_start + duration + 20e-6)
        canonical = (t_start == 20e-6 && duration == 15e-6 && om == 1.0)
        push!(out, (
            family = :block_pi,
            canonical = canonical,
            t_start = t_start,
            duration = duration,
            T_max = T_max,
            n_coeff = 4,
            omega_mult = om,
            taper_frac = 0.1,
        ))
    end
    push!(out, (
        family = :block_pi, canonical = false,
        t_start = 20e-6, duration = 15e-6, T_max = 100e-6,
        n_coeff = 4, omega_mult = 1.0, taper_frac = 0.05,
    ))
    return out
end

function random_composite_designs()
    out = Any[]
    ks = [1, 3, 5]
    n_coeffs = [4, 8]
    seeds = [1, 2, 3, 4, 5]
    T_maxs = [50e-6, 100e-6]
    for k in ks, n_coeff in n_coeffs, seed in seeds, T_max in T_maxs
        canonical = (k == 1 && n_coeff == 4 && seed == 1 && T_max == 100e-6)
        push!(out, (
            family = :random_composite,
            canonical = canonical,
            k = k,
            n_coeff = n_coeff,
            seed = seed,
            T_max = T_max,
            taper_frac = 0.1,
        ))
    end
    return out
end

function all_pulse_designs()
    return vcat(
        rase_designs(),
        rose_designs(),
        arp3_designs(),
        canonical_composite_designs(),
        block_pi_designs(),
        random_composite_designs(),
    )
end

function pulse_family_of(design)
    if design.family === :arp3 && hasproperty(design, :with_signal) && design.with_signal
        return :arp3_signal
    end
    return design.family
end
