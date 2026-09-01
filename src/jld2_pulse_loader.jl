# ============================================================
# JLD2-DRIVEN SIGNAL/CONTROL PULSE OPTIMISATION
#
# Identification (waveforms, not tuple position) then optimisation of
# ONE control envelope. Signal is a fixed background, never in `u`.
#
# Linear pipeline (optimise_control_pulse_from_jld2, and the same
# helpers in examples/reference_run_workflow.ipynb):
#
#   1. Open a *.jld2 file.
#   2. Extract SYSTEM_CONFIG, PULSE_CONFIG, SIM_SETTING.
#   3. Parse them into the reference configs (ensemble `d`, pulse specs).
#      Signal vs control is identified by waveform (WURST / π / amplitude).
#   4. If PULSE_CONFIG is absent, load the sibling *_pulsemat.csv, identify
#      on that trace, and fit only the masked control envelope. If
#      PULSE_CONFIG is present but identification rejects the sequence, error
#      (do not fit a mixed drive into u).
#   5. Forward-simulate the recorded drive (:ground and the weak-excitation
#      :weak seed, Sp = eps*Nj/2). Those metrics are the reference
#      metrics: bright-mode (Nj g^2) weighted inversion I (paper App. H)
#      and the per-frequency-slice silencing factor |F|_* = <|F(omega)|>
#      (paper Eq. 5 / A.132).
#   6. If the file stores results, reconcile this run's final-state
#      outputs and metrics against them. If it stores none, auto-PASS.
#      FAIL stops. PASS continues to 7 then 8 (pulse_optimizer2.jl).
#   7. On PASS, linear-fit (fit_mode=:linear) a CompositePulse seed of the
#      CONTROL envelope only. Signal is never in the fit target.
#   8. optimise_composite_pulse (pulse_optimizer2.jl) optimises that same
#      CONTROL pulse. Identified signal is a fixed background (never in `u`).
# ============================================================

# ============================================================
# SIGNAL vs CONTROL IDENTIFICATION
#
# From each sub-pulse's explicit waveform E(t), not from tuple position
# and not from `name`. Pipeline (no circularity):
#
#   1. Per sub-pulse features: WURST?, π-phase?, π-area?, E[A], peak.
#   2. Control_0 = WURST OR π (phase OR area/flip-angle).
#   3. Remaining: signal iff peak ≤ a_signal * min{E[A_j] | Control_0},
#      with a_signal = 0.1. Otherwise those leftovers are extra control.
#   4. Time order must be signal* control+. Any signal after a control
#      rejects the whole sequence. Signal support must end before the
#      first control starts.
#
# When a PULSE_CONFIG is given, each entry is one sub-pulse (analytic
# 3ARP stays three WURSTs; do not re-split). When only a sampled I/Q
# trace is given, sub-pulses come from `_detect_subpulse_segments`
# (same defaults as the seed fitter), with a multiscale pass so a
# weak Gaussian is not hidden by a WURST peak.
# ============================================================

const _SC_A_SIGNAL = 0.1
const _SC_N_SIGMA_SUPPORT = 5.0
const _SC_N_SAMPLES = 4096
const _SC_REL_THRESH = 1e-3
const _SC_MIN_ACTIVE = 5
const _SC_MIN_SILENCE = 3
const _SC_PI_PHASE_FRAC = 0.15
const _SC_PI_AREA_REL = 0.20
const _SC_PI_N_MAX = 4
const _SC_WURST_ENV_CORR = 0.85
const _SC_WURST_CHIRP_R2 = 0.85
const _SC_WURST_SWEEP_RAD = Float64(π)
const _SC_FORM_FACTOR_CHIRP = 0.25
const _SC_A_FLOOR = 1e-30

struct SignalControlRejected <: Exception
    reason::String
end

function Base.showerror(io::IO, e::SignalControlRejected)
    print(io, "Signal/control identification rejected the sequence: ", e.reason)
end

"""
    signal_control_defaults() -> NamedTuple

Thresholds for [`segment_signal_control`](@ref). `a_signal=0.1` is the
locked amplitude ratio. Detection knobs match
[`_detect_subpulse_segments`](@ref).
"""
function signal_control_defaults()
    return (
        a_signal=_SC_A_SIGNAL,
        n_sigma_support=_SC_N_SIGMA_SUPPORT,
        n_samples=_SC_N_SAMPLES,
        rel_thresh=_SC_REL_THRESH,
        min_active_samples=_SC_MIN_ACTIVE,
        min_silence_samples=_SC_MIN_SILENCE,
        pi_phase_frac=_SC_PI_PHASE_FRAC,
        pi_area_rel=_SC_PI_AREA_REL,
        pi_n_max=_SC_PI_N_MAX,
        wurst_env_corr=_SC_WURST_ENV_CORR,
        wurst_chirp_r2=_SC_WURST_CHIRP_R2,
        wurst_sweep_rad=_SC_WURST_SWEEP_RAD,
        form_factor_chirp=_SC_FORM_FACTOR_CHIRP,
    )
end

# ------------------------------------------------------------
# Small numeric helpers (no Statistics.jl)
# ------------------------------------------------------------

_sc_mean(x) = sum(x) / length(x)

function _sc_corr(x::AbstractVector, y::AbstractVector)
    n = length(x)
    n == length(y) || return 0.0
    n < 3 && return 0.0
    mx = _sc_mean(x)
    my = _sc_mean(y)
    sx = 0.0
    sy = 0.0
    sxy = 0.0
    @inbounds for i in 1:n
        dx = x[i] - mx
        dy = y[i] - my
        sx += dx * dx
        sy += dy * dy
        sxy += dx * dy
    end
    (sx > 0.0 && sy > 0.0) || return 0.0
    return sxy / sqrt(sx * sy)
end

function _sc_trapz(t::AbstractVector, y::AbstractVector)
    n = length(t)
    n == length(y) || error("_sc_trapz: t/y lengths $(n)/$(length(y)).")
    n < 2 && return 0.0
    acc = 0.0
    @inbounds for i in 1:n-1
        acc += 0.5 * (y[i] + y[i+1]) * (t[i+1] - t[i])
    end
    return acc
end

function _sc_weighted_linear_fit(x::AbstractVector, y::AbstractVector, w::AbstractVector)
    n = length(x)
    n == length(y) == length(w) || error("_sc_weighted_linear_fit: length mismatch.")
    W = sum(w)
    W <= 0 && return (slope=0.0, intercept=0.0, r2=0.0)
    xbar = 0.0
    ybar = 0.0
    @inbounds for i in 1:n
        xbar += w[i] * x[i]
        ybar += w[i] * y[i]
    end
    xbar /= W
    ybar /= W
    sxx = 0.0
    sxy = 0.0
    syy = 0.0
    @inbounds for i in 1:n
        dx = x[i] - xbar
        dy = y[i] - ybar
        sxx += w[i] * dx * dx
        sxy += w[i] * dx * dy
        syy += w[i] * dy * dy
    end
    slope = sxx > 0.0 ? sxy / sxx : 0.0
    intercept = ybar - slope * xbar
    ss_res = 0.0
    @inbounds for i in 1:n
        res = y[i] - (intercept + slope * x[i])
        ss_res += w[i] * res * res
    end
    r2 = syy > 0.0 ? max(0.0, 1.0 - ss_res / syy) : 0.0
    return (slope=slope, intercept=intercept, r2=r2)
end

_sc_phase_abs_diff(a, b) = abs(rem2pi(Float64(a) - Float64(b), RoundNearest))

function _sc_omega_per_E(d)
    d === nothing && return nothing
    hasproperty(d, :g_mean) || return nothing
    hasproperty(d, :sqrt_kappa_e) || return nothing
    hasproperty(d, :kappa_t) || return nothing
    kt = Float64(d.kappa_t)
    kt > 0 || return nothing
    g = Float64(d.g_mean)
    s = Float64(d.sqrt_kappa_e)
    (g > 0 && s > 0) || return nothing
    # Steady-state |a| = 2√κe |E| / κt, Ω_spin = 2 g |a|  (rhs_1st_order!,
    # CompositePulse amp_scale). Same map as amp_scale's Ω → E conversion.
    return 4.0 * g * s / kt
end

# ------------------------------------------------------------
# Support and sampling of one PULSE_CONFIG entry
# ------------------------------------------------------------

function _sc_cfg_kind(cfg)
    hasproperty(cfg, :kind) || return :unknown
    return cfg.kind
end

function _sc_cfg_support(cfg; n_sigma=_SC_N_SIGMA_SUPPORT)
    kind = _sc_cfg_kind(cfg)
    if kind === :gaussian
        t0 = Float64(cfg.t0)
        sig = Float64(cfg.sigma)
        half = n_sigma * sig
        return (t0 - half, t0 + half)
    elseif kind === :wurst
        tc = Float64(cfg.t_center)
        dur = Float64(cfg.duration)
        edge = hasproperty(cfg, :edge_frac) ?
            max(dur * Float64(cfg.edge_frac), eps(Float64)) : max(dur * 1e-4, eps(Float64))
        pad = 8.0 * edge
        return (tc - dur / 2 - pad, tc + dur / 2 + pad)
    elseif hasproperty(cfg, :t_start) && hasproperty(cfg, :t_end)
        return (Float64(cfg.t_start), Float64(cfg.t_end))
    else
        return nothing
    end
end

function _sc_sample_callable(E_of_t, t_lo::Float64, t_hi::Float64, n_samples::Integer)
    t_hi > t_lo || error("_sc_sample_callable: empty interval [$t_lo, $t_hi].")
    n = Int(n_samples)
    n >= 8 || error("_sc_sample_callable: n_samples must be >= 8, got $n.")
    t = collect(range(t_lo, t_hi; length=n))
    I = Vector{Float64}(undef, n)
    Q = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        z = ComplexF64(E_of_t(t[i]))
        I[i] = real(z)
        Q[i] = imag(z)
    end
    return t, I, Q
end

# ------------------------------------------------------------
# Waveform tests
# ------------------------------------------------------------

function _sc_wurst_envelope_corr(t::AbstractVector, A::AbstractVector, t_s::Float64, t_e::Float64)
    T = t_e - t_s
    T > 0 || return 0.0
    peak = maximum(A)
    peak <= _SC_A_FLOOR && return 0.0
    An = A ./ peak
    τ = t .- t_s
    best = -1.0
    for n in (4.0, 8.0, 10.0, 16.0, 20.0, 32.0, 40.0)
        e = similar(An)
        @inbounds for i in eachindex(τ)
            s = sin(pi * (τ[i] - T / 2) / T)
            e[i] = 1.0 - abs(s)^n
        end
        c = _sc_corr(An, e)
        best = max(best, c)
    end
    return best
end

function _sc_is_wurst_waveform(t, A, f, t_s, t_e, defs; form_factor::Real=1.0)
    T = t_e - t_s
    T > 0 || return false
    env = _sc_wurst_envelope_corr(t, A, t_s, t_e)
    w = A .^ 2
    fit = _sc_weighted_linear_fit(t, f, w)
    sweep = abs(fit.slope) * T
    is_env = env >= defs.wurst_env_corr
    is_chirp = (fit.r2 >= defs.wurst_chirp_r2) && (sweep >= defs.wurst_sweep_rad)
    # Chirped ARP also has |∫E| ≪ ∫|E|; use that as a third vote so a
    # Gaussian (form_factor ~ 1, sweep ~ 0) cannot pass on envelope luck.
    is_stationary_phase = form_factor <= defs.form_factor_chirp
    return is_env && is_chirp && is_stationary_phase
end

function _sc_is_pi_phase(phi::AbstractVector, A::AbstractVector, defs)
    peak = maximum(A)
    peak <= _SC_A_FLOOR && return false
    floor = 0.05 * peak
    i0 = findfirst(a -> a >= floor, A)
    i1 = findlast(a -> a >= floor, A)
    (i0 === nothing || i1 === nothing || i1 <= i0) && return false
    ΔΦ = Float64(phi[i1] - phi[i0])
    return _sc_phase_abs_diff(ΔΦ, π) <= defs.pi_phase_frac * π
end

function _sc_is_pi_area(t::AbstractVector, A::AbstractVector, d, defs)
    ΩE = _sc_omega_per_E(d)
    ΩE === nothing && return false
    θ = ΩE * _sc_trapz(t, A)
    θ <= 0 && return false
    best_rel = Inf
    @inbounds for n in 1:Int(defs.pi_n_max)
        rel = abs(θ - n * π) / (n * π)
        best_rel = min(best_rel, rel)
    end
    return best_rel <= defs.pi_area_rel
end

# ------------------------------------------------------------
# Features of one sampled sub-pulse
# ------------------------------------------------------------

"""
    subpulse_waveform_features(t, I, Q; d=nothing, t_start=nothing, t_end=nothing, defs=signal_control_defaults())

Waveform features for one already-isolated sub-pulse (one detected
segment, or one `PULSE_CONFIG` entry sampled on its own support).
Uses [`_instantaneous_frequency`](@ref) for phase/frequency.
"""
function subpulse_waveform_features(
    t::AbstractVector, I::AbstractVector, Q::AbstractVector;
    d=nothing,
    t_start::Union{Nothing,Real}=nothing,
    t_end::Union{Nothing,Real}=nothing,
    kind::Symbol=:unknown,
    defs=signal_control_defaults(),
    source_index::Integer=0,
)
    n = length(t)
    n == length(I) == length(Q) || error(
        "subpulse_waveform_features: t/I/Q lengths $(n)/$(length(I))/$(length(Q))."
    )
    n >= 2 || error("subpulse_waveform_features: need >= 2 samples, got $n.")
    A = hypot.(I, Q)
    peak = maximum(A)
    ts = t_start === nothing ? Float64(t[1]) : Float64(t_start)
    te = t_end === nothing ? Float64(t[end]) : Float64(t_end)
    T = max(te - ts, eps(Float64))
    mean_A = _sc_trapz(t, A) / T
    area_A = _sc_trapz(t, A)
    E_int_re = _sc_trapz(t, I)
    E_int_im = _sc_trapz(t, Q)
    form_factor = area_A > _SC_A_FLOOR ? hypot(E_int_re, E_int_im) / area_A : 0.0

    phi, f = _instantaneous_frequency(t, I, Q)
    is_wurst_kind = kind === :wurst
    is_wurst_wave = peak > _SC_A_FLOOR &&
        _sc_is_wurst_waveform(t, A, f, ts, te, defs; form_factor=form_factor)
    is_wurst = is_wurst_kind || is_wurst_wave
    is_pi_phase = peak > _SC_A_FLOOR && _sc_is_pi_phase(phi, A, defs)
    is_pi_area = _sc_is_pi_area(t, A, d, defs)
    is_pi = is_pi_phase || is_pi_area
    ΩE = _sc_omega_per_E(d)
    flip_angle = ΩE === nothing ? NaN : ΩE * area_A

    return (
        source_index=Int(source_index),
        kind=kind,
        t_start=ts,
        t_end=te,
        peak=Float64(peak),
        mean_A=Float64(mean_A),
        area_A=Float64(area_A),
        form_factor=Float64(form_factor),
        flip_angle=Float64(flip_angle),
        is_wurst_kind=is_wurst_kind,
        is_wurst_waveform=is_wurst_wave,
        is_wurst=is_wurst,
        is_pi_phase=is_pi_phase,
        is_pi_area=is_pi_area,
        is_pi=is_pi,
    )
end

# ------------------------------------------------------------
# Assemble labels: Control_0, amplitude, sequence
# ------------------------------------------------------------

function _sc_assemble(features::Vector, a_signal::Real)
    n = length(features)
    n >= 1 || return (
        ok=false,
        reason="No sub-pulses to classify.",
        labels=Symbol[],
        control0_idx=Int[],
        signal_idx=Int[],
        control_idx=Int[],
        order=Int[],
        min_control_mean_A=NaN,
    )

    control0 = Int[]
    undetermined = Int[]
    @inbounds for i in 1:n
        f = features[i]
        if f.is_wurst || f.is_pi
            push!(control0, i)
        else
            push!(undetermined, i)
        end
    end

    if isempty(control0)
        return (
            ok=false,
            reason="No control sub-pulse: none is WURST or a π-pulse (phase or area/flip-angle). " *
                   "Refusing to invent a signal from position.",
            labels=fill(:rejected, n),
            control0_idx=Int[],
            signal_idx=Int[],
            control_idx=Int[],
            order=Int[],
            min_control_mean_A=NaN,
        )
    end

    min_ctrl_mean = minimum(features[j].mean_A for j in control0)
    thresh = Float64(a_signal) * min_ctrl_mean
    labels = Vector{Symbol}(undef, n)
    @inbounds for i in control0
        labels[i] = :control
    end
    signal_idx = Int[]
    control_idx = copy(control0)
    @inbounds for i in undetermined
        if features[i].peak <= thresh + eps(Float64)
            labels[i] = :signal
            push!(signal_idx, i)
        else
            labels[i] = :control
            push!(control_idx, i)
        end
    end
    sort!(control_idx)
    sort!(signal_idx)

    order = sort(collect(1:n); by=i -> (features[i].t_start, i))
    seen_control = false
    @inbounds for i in order
        if labels[i] === :control
            seen_control = true
        elseif labels[i] === :signal && seen_control
            seq = join(string(labels[j]) * "@" * string(round(features[j].t_start * 1e6; digits=3)) * "us"
                       for j in order)
            return (
                ok=false,
                reason="Signal follows a control pulse (forbidden prefix rule). Time order: $seq.",
                labels=labels,
                control0_idx=control0,
                signal_idx=signal_idx,
                control_idx=control_idx,
                order=order,
                min_control_mean_A=min_ctrl_mean,
            )
        end
    end

    if !isempty(signal_idx) && !isempty(control_idx)
        t_sig_end = maximum(features[i].t_end for i in signal_idx)
        t_ctrl0 = minimum(features[i].t_start for i in control_idx)
        if t_sig_end >= t_ctrl0
            return (
                ok=false,
                reason="A signal's support overlaps or outlasts the first control start " *
                       "(t_signal_end=$(t_sig_end)s, t_control_start=$(t_ctrl0)s). " *
                       "Control must start strictly after the signal has ended.",
                labels=labels,
                control0_idx=control0,
                signal_idx=signal_idx,
                control_idx=control_idx,
                order=order,
                min_control_mean_A=min_ctrl_mean,
            )
        end
    end

    isempty(control_idx) && return (
        ok=false,
        reason="No control sub-pulse left after labelling.",
        labels=labels,
        control0_idx=control0,
        signal_idx=signal_idx,
        control_idx=control_idx,
        order=order,
        min_control_mean_A=min_ctrl_mean,
    )

    return (
        ok=true,
        reason="ok",
        labels=labels,
        control0_idx=control0,
        signal_idx=signal_idx,
        control_idx=control_idx,
        order=order,
        min_control_mean_A=min_ctrl_mean,
    )
end

function _sc_finish(features, assembled, PULSE_CONFIG, a_signal; error_on_reject::Bool=false)
    result = (
        ok=assembled.ok,
        reason=assembled.reason,
        a_signal=Float64(a_signal),
        min_control_mean_A=assembled.min_control_mean_A,
        features=features,
        labels=assembled.labels,
        signal_idx=assembled.signal_idx,
        control_idx=assembled.control_idx,
        control0_idx=assembled.control0_idx,
        order=assembled.order,
        signal_cfg=PULSE_CONFIG === nothing ? nothing :
            (isempty(assembled.signal_idx) ? () : tuple((PULSE_CONFIG[i] for i in assembled.signal_idx)...)),
        control_cfg=PULSE_CONFIG === nothing ? nothing :
            tuple((PULSE_CONFIG[i] for i in assembled.control_idx)...),
    )
    if !result.ok && error_on_reject
        throw(SignalControlRejected(result.reason))
    end
    return result
end

"""
    _sc_detect_subpulses_multiscale(t, A; rel_thresh, min_active_samples, min_silence_samples, max_passes=4)

Calls [`_detect_subpulse_segments`](@ref) repeatedly. After each pass the
detected samples are zeroed so a later pass can see a weaker pulse that
the first pass treated as silence (the detector's threshold is
`rel_thresh * maximum(A)` of the array it is given). This is required
for ROSE-like traces: the Gaussian peak is ~10^{-4} of a WURST peak,
below the default `1e-3` global floor, so a single call finds only the
WURST. Each pass is still the existing detector, unchanged.
"""
function _sc_detect_subpulses_multiscale(
    t::AbstractVector, A::AbstractVector;
    rel_thresh::Real=_SC_REL_THRESH,
    min_active_samples::Integer=_SC_MIN_ACTIVE,
    min_silence_samples::Integer=_SC_MIN_SILENCE,
    max_passes::Integer=4,
)
    Awork = Float64.(A)
    found = Tuple{Int,Int}[]
    for _ in 1:max_passes
        maximum(Awork) <= _SC_A_FLOOR && break
        segs = _detect_subpulse_segments(
            t, Awork;
            rel_thresh=rel_thresh,
            min_active_samples=min_active_samples,
            min_silence_samples=min_silence_samples,
        )
        isempty(segs) && break
        for (i0, i1) in segs
            push!(found, (i0, i1))
            @inbounds Awork[i0:i1] .= 0.0
        end
    end
    sort!(found; by=first)
    return found
end

# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

"""
    segment_signal_control(PULSE_CONFIG; d=nothing, a_signal=0.1, error_on_reject=false, kwargs...)

Identify signal vs control from a `PULSE_CONFIG` tuple. Each entry is
one sub-pulse. Optional `d` (`prepare_derived`) enables the physical
flip-angle test `θ = (4 g_mean √κe / κt) ∫|E| dt`.

Returns a NamedTuple: `ok`, `reason`, `signal_cfg`, `control_cfg`,
`signal_idx`, `control_idx`, `features`, `labels`. If `ok=false` the
sequence must not be optimised. `error_on_reject=true` throws
[`SignalControlRejected`](@ref) instead of returning.
"""
function segment_signal_control(
    PULSE_CONFIG;
    d=nothing,
    a_signal::Real=signal_control_defaults().a_signal,
    n_sigma_support::Real=signal_control_defaults().n_sigma_support,
    n_samples::Integer=signal_control_defaults().n_samples,
    error_on_reject::Bool=false,
    defs=signal_control_defaults(),
)
    n = length(PULSE_CONFIG)
    n >= 1 || error("segment_signal_control: PULSE_CONFIG is empty.")
    a_signal > 0 || error("a_signal must be positive, got $a_signal.")
    defs = merge(defs, (a_signal=Float64(a_signal), n_sigma_support=Float64(n_sigma_support),
                         n_samples=Int(n_samples)))

    t_span_hi = 0.0
    if d !== nothing && hasproperty(d, :timespan)
        t_span_hi = Float64(d.timespan[2] - d.timespan[1])
    end

    features = Vector{Any}(undef, n)
    for i in 1:n
        cfg = PULSE_CONFIG[i]
        kind = _sc_cfg_kind(cfg)
        supp = _sc_cfg_support(cfg; n_sigma=defs.n_sigma_support)
        if supp === nothing
            t_span_hi > 0 || error(
                "PULSE_CONFIG[$i] kind=$kind has no analytic support; pass d with timespan " *
                "or give the entry t_start/t_end."
            )
            t_lo, t_hi = 0.0, t_span_hi
        else
            t_lo, t_hi = supp
            if t_span_hi > 0
                t_lo = max(t_lo, 0.0)
                t_hi = min(t_hi, t_span_hi)
            else
                t_lo = max(t_lo, 0.0)
            end
        end
        t_hi > t_lo || error("PULSE_CONFIG[$i] has empty support [$t_lo, $t_hi].")
        E = build_drive_pulse(cfg)
        t, I, Q = _sc_sample_callable(E, t_lo, t_hi, defs.n_samples)
        if kind !== :gaussian && kind !== :wurst
            A = hypot.(I, Q)
            segs = _detect_subpulse_segments(
                t, A;
                rel_thresh=defs.rel_thresh,
                min_active_samples=defs.min_active_samples,
                min_silence_samples=defs.min_silence_samples,
            )
            if !isempty(segs)
                i0 = segs[1][1]
                i1 = segs[end][2]
                t = t[i0:i1]
                I = I[i0:i1]
                Q = Q[i0:i1]
                t_lo = Float64(t[1])
                t_hi = Float64(t[end])
            end
        end
        features[i] = subpulse_waveform_features(
            t, I, Q; d=d, t_start=t_lo, t_end=t_hi, kind=kind, defs=defs, source_index=i,
        )
    end
    assembled = _sc_assemble(features, defs.a_signal)
    return _sc_finish(features, assembled, PULSE_CONFIG, defs.a_signal; error_on_reject=error_on_reject)
end

"""
    segment_signal_control_from_trace(t, I, Q; d=nothing, a_signal=0.1, error_on_reject=false, kwargs...)

Same labelling as [`segment_signal_control`](@ref), but sub-pulses are
exactly the segments of [`_detect_subpulse_segments`](@ref) on
`A=hypot.(I,Q)` (seed-fitter defaults unless overridden). Back-to-back
3ARP with unresolved edge zeros may merge into one segment -- that is
this detector's own behaviour; pass `PULSE_CONFIG` when the analytic
entries are known.
"""
function segment_signal_control_from_trace(
    t::AbstractVector, I::AbstractVector, Q::AbstractVector;
    d=nothing,
    a_signal::Real=signal_control_defaults().a_signal,
    error_on_reject::Bool=false,
    rel_thresh::Real=signal_control_defaults().rel_thresh,
    min_active_samples::Integer=signal_control_defaults().min_active_samples,
    min_silence_samples::Integer=signal_control_defaults().min_silence_samples,
    defs=signal_control_defaults(),
)
    n = length(t)
    n == length(I) == length(Q) || error(
        "segment_signal_control_from_trace: t/I/Q lengths $(n)/$(length(I))/$(length(Q))."
    )
    n >= 2 || error("segment_signal_control_from_trace: need >= 2 samples, got $n.")
    a_signal > 0 || error("a_signal must be positive, got $a_signal.")
    defs = merge(defs, (a_signal=Float64(a_signal), rel_thresh=Float64(rel_thresh),
                         min_active_samples=Int(min_active_samples),
                         min_silence_samples=Int(min_silence_samples)))

    A = hypot.(I, Q)
    segments = _sc_detect_subpulses_multiscale(
        t, A;
        rel_thresh=defs.rel_thresh,
        min_active_samples=defs.min_active_samples,
        min_silence_samples=defs.min_silence_samples,
    )
    isempty(segments) && return _sc_finish(
        Any[],
        (
            ok=false,
            reason="Existing sub-pulse detector found no active segments.",
            labels=Symbol[],
            control0_idx=Int[],
            signal_idx=Int[],
            control_idx=Int[],
            order=Int[],
            min_control_mean_A=NaN,
        ),
        nothing,
        defs.a_signal;
        error_on_reject=error_on_reject,
    )

    features = Vector{Any}(undef, length(segments))
    for (k, (i0, i1)) in enumerate(segments)
        ts = view(t, i0:i1)
        Is = view(I, i0:i1)
        Qs = view(Q, i0:i1)
        features[k] = subpulse_waveform_features(
            collect(ts), collect(Is), collect(Qs);
            d=d,
            t_start=Float64(t[i0]),
            t_end=Float64(t[i1]),
            kind=:unknown,
            defs=defs,
            source_index=k,
        )
    end
    assembled = _sc_assemble(features, defs.a_signal)
    return _sc_finish(features, assembled, nothing, defs.a_signal; error_on_reject=error_on_reject)
end

"""
    identified_signal_control(PULSE_CONFIG; kwargs...) -> (signal_cfg, control_cfg)

Throws [`SignalControlRejected`](@ref) unless identification and the
prefix rule both pass. Replacement for positional `split_signal_control`
that does not use `n_signal`.
"""
function identified_signal_control(PULSE_CONFIG; kwargs...)
    r = segment_signal_control(PULSE_CONFIG; error_on_reject=true, kwargs...)
    return r.signal_cfg, r.control_cfg
end

"""
    control_envelope_E_of_t(result) -> (t -> Complex)

ONE cavity-input envelope equal to the sum of every identified **control**
sub-pulse. This is the only drive that may be sampled, fit, or stored in
`u`. Signal lobes are not in this callable.
"""
function control_envelope_E_of_t(result)
    cfg = result.control_cfg
    (cfg === nothing || isempty(cfg)) && error(
        "control_envelope_E_of_t: identification produced no control pulses. " *
        (result.ok ? "" : result.reason)
    )
    return build_E_of_t(cfg)
end

"""
    signal_envelope_E_of_t(result; use_signal=true) -> (t -> Complex)

Fixed background equal to the sum of identified **signal** sub-pulses.
Never parameterized. `use_signal=false` returns [`_zero_drive`](@ref).
Empty signal (3ARP, RASE) is also `_zero_drive`.
"""
function signal_envelope_E_of_t(result; use_signal::Bool=true)
    (!use_signal || result.signal_cfg === nothing || isempty(result.signal_cfg)) &&
        return _zero_drive
    return build_E_of_t(result.signal_cfg)
end

"""
    mask_control_envelope_samples(t, I, Q, result) -> (Ex, Ep)

From a mixed I/Q trace, keep samples that fall inside an identified
control support and zero the rest (signal times and silence). The
result is one control envelope on the original grid, suitable as a
seed-fit target. Does not copy signal samples into the envelope.
"""
function mask_control_envelope_samples(
    t::AbstractVector, I::AbstractVector, Q::AbstractVector, result,
)
    n = length(t)
    n == length(I) == length(Q) || error(
        "mask_control_envelope_samples: t/I/Q lengths $(n)/$(length(I))/$(length(Q))."
    )
    Ex = zeros(Float64, n)
    Ep = zeros(Float64, n)
    isempty(result.control_idx) && return Ex, Ep
    @inbounds for j in 1:n
        tj = t[j]
        keep = false
        for i in result.control_idx
            f = result.features[i]
            if tj >= f.t_start && tj <= f.t_end
                keep = true
                break
            end
        end
        if keep
            Ex[j] = Float64(I[j])
            Ep[j] = Float64(Q[j])
        end
    end
    return Ex, Ep
end

"""
    load_jld2_run(path) -> data

Loads a saved run's `data` NamedTuple (`SIM_SETTING`, `SYSTEM_CONFIG`,
`PULSE_CONFIG`, the saved trajectory, ...) from a `.jld2` file written by
[`save_run_data`](@ref)/`run_sim_1st_order`'s own convention (`@save
filename data`).
"""
function load_jld2_run(path::AbstractString)
    return JLD2.load(path, "data")
end

"""
    _sibling_pulsemat_n_samples(jld2_path) -> Union{Nothing,Int}

Row count of `jld2_path`'s own sibling `_pulsemat.csv` (same directory,
same basename -- the convention [`save_run_data`](@ref) writes both files
under), or `nothing` if `jld2_path` doesn't end in `.jld2` or no such
sibling file exists. Counts lines directly (`# t_end_us,...` metadata line
+ `Re,Im` header + one row per sample -- [`save_E_samples`](@ref)'s own
format) rather than parsing every value via [`load_E_samples`](@ref),
since only the COUNT is needed here -- used to size a seed fit's own
`N_samples` to match the density the source run was ACTUALLY recorded at,
rather than an arbitrary fixed default (see
[`optimise_control_pulse_from_jld2`](@ref)'s own `fit_N`).
"""
function _sibling_pulsemat_n_samples(jld2_path::AbstractString)
    endswith(jld2_path, ".jld2") || return nothing
    sibling = jld2_path[1:end-length(".jld2")] * "_pulsemat.csv"
    isfile(sibling) || return nothing
    n = countlines(sibling) - 2
    n >= 2 || return nothing
    return n
end

"""
    split_signal_control(PULSE_CONFIG; n_signal=1) -> (signal_cfg, control_cfg)

Deprecated positional split (leading `n_signal` entries). The jld2
pipeline uses [`segment_signal_control`](@ref) / [`identified_signal_control`](@ref)
instead: one control envelope, signal never in `u`.
"""
function split_signal_control(PULSE_CONFIG; n_signal::Integer=1)
    n_signal >= 1 || error("n_signal must be >= 1, got $n_signal.")
    length(PULSE_CONFIG) > n_signal || error(
        "PULSE_CONFIG has only $(length(PULSE_CONFIG)) pulse(s); need more than " *
        "n_signal=$n_signal to have anything left over as the control pulse."
    )
    return PULSE_CONFIG[1:n_signal], PULSE_CONFIG[n_signal+1:end]
end

"""
    build_signal_E_of_t(signal_cfg, use_signal::Bool) -> (t -> Complex)

Builds the FIXED signal drive from `signal_cfg` (a `PULSE_CONFIG`-shaped
tuple, via the existing [`build_E_of_t`](@ref)) when `use_signal` is
`true`; otherwise returns [`_zero_drive`](@ref), identically zero at
every `t`, so the signal pulse has NO effect on anything downstream (the
ensemble never sees it at all) while keeping the same `t -> Complex`
calling convention every other drive in this package uses -- this is the
`USE_SIGNAL` mode flag: set `use_signal=false` to zero the signal pulse
out completely for whatever uses the returned closure next.
"""
function build_signal_E_of_t(signal_cfg, use_signal::Bool)
    (!use_signal || signal_cfg === nothing || isempty(signal_cfg)) && return _zero_drive
    return build_E_of_t(signal_cfg)
end

"""
    _sample_control_cfg(control_cfg, d, N_samples) -> (t, Ex, Ep)

Samples `control_cfg`'s own combined drive (`build_E_of_t(control_cfg)`) at
`N_samples` evenly spaced points over `[0, d.timespan[2]-d.timespan[1]]` --
the shared "get a raw `(t, I, Q)` trace out of a `PULSE_CONFIG`-shaped
segment list" step used by [`load_jld2_reference`](@ref) when
`PULSE_CONFIG` parses.
"""
function _sample_control_cfg(control_cfg, d, N_samples::Integer)
    N_samples >= 2 || error("_sample_control_cfg: N_samples must be >= 2, got $N_samples.")
    E_target = build_E_of_t(control_cfg)
    T_max = d.timespan[2] - d.timespan[1]
    T_max > 0 || error("_sample_control_cfg: ensemble timespan span must be positive, got $T_max.")
    Ex, Ep = sample_E_of_t(E_target, T_max, N_samples)
    length(Ex) == N_samples && length(Ep) == N_samples || error(
        "_sample_control_cfg: sample_E_of_t returned lengths $(length(Ex))/$(length(Ep)), expected $N_samples."
    )
    t = collect(range(0.0, T_max; length=N_samples))
    length(t) == N_samples || error("_sample_control_cfg: t length $(length(t)) != N_samples=$N_samples.")
    return t, Ex, Ep
end

# ============================================================
# REFERENCE PIPELINE (shared with reference_run_workflow.ipynb)
#
# load_jld2_reference → run_reference_forward → reconcile_reference
#   PASS → fit_linear_seed (CONTROL) → optimise_composite_pulse (CONTROL)
# FAIL at reconcile_reference refuses steps 7–8.
# jld2_pipeline_defaults / jld2_optimizer_defaults are the knobs.
# Changing a default here changes BOTH the loader and the notebook.
# ============================================================

"""
    jld2_pipeline_defaults() -> NamedTuple

Canonical knobs for [`load_jld2_reference`](@ref)/
[`optimise_control_pulse_from_jld2`](@ref) and for
`examples/reference_run_workflow.ipynb`. Changing a default here
changes BOTH entry points (`optimise_control_pulse_from_jld2` merges
this NamedTuple; it does not hardcode a second copy).
"""
function jld2_pipeline_defaults()
    return (
        n_signal=nothing,
        use_signal=false,
        use_interior=false,
        rtol_check=1e-3,
        atol_check=nothing,
        check_reltol=nothing,
        check_abstol=nothing,
        fit_seed_from_file=true,
        fit_N=nothing,
        param_budget=60,
        save_log=true,
        log_out_dir=nothing,
        pulsemat_N=nothing,
    )
end

"""
    jld2_optimizer_defaults() -> NamedTuple

Canonical kwargs forwarded to [`optimise_composite_pulse`](@ref) by
[`optimise_control_pulse_from_jld2`](@ref) and by the reference-run
notebook. Values match `optimise_composite_pulse`'s own signature
defaults (so an un-overridden jld2 run is bit-identical to calling
that function with only `signal_E_of_t` / `warm_start_u` extra).
`fit_mode` is not an optimiser knob -- the jld2 seed is always the
closed-form linear least-squares fitter.
"""
function jld2_optimizer_defaults()
    return (
        num_epochs=30,
        learning_rate=0.05,
        cf_lr_scale=1.0,
        patience=5,
        tol=1e-3,
        n_hops=3,
        hop_patience=2,
        hop_step_size=0.5,
        temperature=1.0,
        degree=3,
        taper_frac=0.1,
        w_tmax=1.0,
        w_power=0.05,
        target_F=1.0,
        w_time=0.15,
        seed=42,
        threaded_grad=true,
        compute=:auto,
        grad_mode=:forwarddiff,
        track=:dual,
        anneal_direct_weights=true,
        hop0_phyonly=true,
        x_tune_alpha=_DEFAULT_X_TUNE_ALPHA,
        recalibrate_optima_x=true,
        I_min=_DEFAULT_PENALTY_MIN,
        kappa_I=_DEFAULT_PENALTY_KAPPA,
        S_min=_DEFAULT_PENALTY_MIN,
        kappa_S=_DEFAULT_PENALTY_KAPPA,
    )
end

# Control-trace sample count when no sibling *_pulsemat.csv exists and the
# caller did not pass `fit_N`. Must be >= 2.
const _DEFAULT_CONTROL_FIT_N = 10001

# Names `optimise_composite_pulse` accepts beyond `jld2_optimizer_defaults()`.
const _JLD2_OPTIMISER_EXTRA_KEYS = (:warm_start_u, :label_prefix)
const _JLD2_SOLVER_KEYS = (:reltol, :abstol, :alg, :tstops)

function _jld2_allowed_keys()
    allowed = Set{Symbol}(keys(jld2_pipeline_defaults()))
    union!(allowed, keys(jld2_optimizer_defaults()))
    union!(allowed, _JLD2_OPTIMISER_EXTRA_KEYS)
    union!(allowed, _JLD2_SOLVER_KEYS)
    push!(allowed, :verbose)
    return allowed
end

function _jld2_split_kwargs(kwargs)
    caller = NamedTuple(kwargs)
    verbose = haskey(caller, :verbose) ? caller.verbose : true
    verbose isa Bool || error("verbose must be a Bool, got $(typeof(verbose)).")
    merged = merge(jld2_pipeline_defaults(), jld2_optimizer_defaults(), caller)
    allowed = _jld2_allowed_keys()
    for k in keys(merged)
        k in allowed || error(
            "unknown keyword $(k) for the jld2 pipeline / optimise_control_pulse_from_jld2. " *
            "Pipeline knobs: $(keys(jld2_pipeline_defaults())). " *
            "Optimiser knobs: $(keys(jld2_optimizer_defaults())) plus $(_JLD2_OPTIMISER_EXTRA_KEYS). " *
            "Solver extras: $(_JLD2_SOLVER_KEYS)."
        )
    end
    pipe_keys = keys(jld2_pipeline_defaults())
    pipe = NamedTuple{pipe_keys}(map(k -> getfield(merged, k), pipe_keys))
    opt_pairs = Pair{Symbol,Any}[
        k => v for (k, v) in pairs(merged) if k !== :verbose && !(k in pipe_keys)
    ]
    opt = (; opt_pairs...)
    return pipe, opt, verbose
end

function _call_optimise_composite_pulse(k, n_coeff_A, n_coeff_f, d, signal_E_of_t, opt)
    return optimise_composite_pulse(
        Int(k), Int(n_coeff_A), Int(n_coeff_f), d;
        signal_E_of_t=signal_E_of_t, opt...,
    )
end

"""
    try_parse_pulse_config(data; d=nothing, n_signal=nothing)
        -> (ok, signal_cfg, control_cfg, message, identification)

Reads `data.PULSE_CONFIG` and identifies signal vs control with
[`segment_signal_control`](@ref) (waveforms, not tuple position).
`n_signal` is an optional count check only. On success the pipeline
parameterizes one control envelope; signal is never in `u`.
When `PULSE_CONFIG` is present but identification rejects the sequence,
[`load_jld2_reference`](@ref) errors rather than fitting a mixed CSV.
"""
function try_parse_pulse_config(data; n_signal::Union{Nothing,Integer}=nothing, d=nothing)
    fail(msg) = (
        ok=false, signal_cfg=nothing, control_cfg=nothing, message=msg, identification=nothing,
    )
    hasproperty(data, :PULSE_CONFIG) || return fail(
        "PULSE_CONFIG is unavailable in the .jld2 data NamedTuple.",
    )
    pc = data.PULSE_CONFIG
    pc === nothing && return fail("PULSE_CONFIG is `nothing`.")
    local ident
    try
        ident = segment_signal_control(pc; d=d)
    catch e
        e isa InterruptException && rethrow()
        return fail("segment_signal_control failed: $e")
    end
    ident.ok || return fail("signal/control identification rejected: $(ident.reason)")
    if n_signal !== nothing && Int(n_signal) != _sc_n_signal(ident)
        return fail(
            "n_signal=$n_signal does not match identified signal count " *
            "$(_sc_n_signal(ident)) (identification is by waveform, not position).",
        )
    end
    signal_cfg = ident.signal_cfg
    control_cfg = ident.control_cfg
    try
        control_envelope_E_of_t(ident)
        signal_envelope_E_of_t(ident)
    catch e
        e isa InterruptException && rethrow()
        return fail("failed to build signal/control envelopes: $e")
    end
    n_sig = signal_cfg === nothing ? 0 : length(signal_cfg)
    n_ctrl = length(control_cfg)
    return (
        ok=true, signal_cfg=signal_cfg, control_cfg=control_cfg, identification=ident,
        message="Identified PULSE_CONFIG: $n_sig signal pulse(s), $n_ctrl control pulse(s) " *
                "(one control envelope; signal not in u).",
    )
end

"""
    _resolve_run_paths(path) -> (jld2_path, pulsemat_path)

Accepts a `.jld2` path or a `*_pulsemat.csv` path and returns the
paired files (`pulsemat_path` is `nothing` if the sibling CSV is
absent). Errors if a CSV is given without its sibling `.jld2`.
"""
function _resolve_run_paths(path::AbstractString)
    if endswith(path, ".jld2")
        pulsemat = path[1:end-length(".jld2")] * "_pulsemat.csv"
        return path, isfile(pulsemat) ? pulsemat : nothing
    elseif endswith(path, "_pulsemat.csv")
        jld2_path = path[1:end-length("_pulsemat.csv")] * ".jld2"
        isfile(jld2_path) || error(
            "Expected the sibling run $jld2_path (same directory/basename as $path, " *
            "the convention save_run_data writes both files under) -- not found."
        )
        return jld2_path, path
    else
        error("Expected a .jld2 or *_pulsemat.csv path, got $path.")
    end
end

"""
    build_E_of_t_from_samples(t, Ex, Ep) -> (t -> Complex)

Linear interpolant of a uniformly sampled I/Q trace (`sample_E_of_t`/
`load_E_samples` grid). Used as the recorded drive when `PULSE_CONFIG`
cannot be parsed and the sibling `_pulsemat.csv` is the only source.
Clamps to the first/last sample outside `[t[1], t[end]]`.
"""
function build_E_of_t_from_samples(t::AbstractVector, Ex::AbstractVector, Ep::AbstractVector)
    n = length(t)
    n >= 2 || error("build_E_of_t_from_samples: need at least 2 samples, got $n.")
    length(Ex) == n && length(Ep) == n || error(
        "build_E_of_t_from_samples: t/Ex/Ep lengths $(n)/$(length(Ex))/$(length(Ep)) must match."
    )
    t0 = Float64(t[1])
    t1 = Float64(t[end])
    Tspan = t1 - t0
    Tspan > 0 || error("build_E_of_t_from_samples: t span must be positive, got $Tspan.")
    Ex_c = collect(Float64, Ex)
    Ep_c = collect(Float64, Ep)
    return function (tt)
        x = (Float64(tt) - t0) / Tspan * (n - 1)
        if x <= 0
            return complex(Ex_c[1], Ep_c[1])
        elseif x >= n - 1
            return complex(Ex_c[n], Ep_c[n])
        end
        i = floor(Int, x) + 1
        α = x - (i - 1)
        return complex((1 - α) * Ex_c[i] + α * Ex_c[i + 1], (1 - α) * Ep_c[i] + α * Ep_c[i + 1])
    end
end

function _sc_n_signal(ident)
    ident === nothing && return 0
    idx = ident.signal_idx
    return idx === nothing ? 0 : length(idx)
end

function _recorded_control_duration(ident, t, Ex, Ep;
    rel_thresh::Real=1e-3, min_active_samples::Integer=5, min_silence_samples::Integer=3,
)
    if ident !== nothing && !isempty(ident.control_idx)
        return maximum(Float64(ident.features[i].t_end) for i in ident.control_idx)
    end
    A = hypot.(Ex, Ep)
    segments = _detect_subpulse_segments(
        t, A; rel_thresh=rel_thresh, min_active_samples=min_active_samples, min_silence_samples=min_silence_samples,
    )
    isempty(segments) && return Float64(t[end])
    return Float64(t[segments[end][2]])
end

function _getfield_or(data, name::Symbol, default=nothing)
    if data isa NamedTuple
        return haskey(data, name) ? getfield(data, name) : default
    end
    return hasproperty(data, name) ? getproperty(data, name) : default
end

function _last_or_nothing(x)
    x === nothing && return nothing
    try
        n = length(x)
        n < 1 && return nothing
        return x[end]
    catch e
        e isa InterruptException && rethrow()
        return nothing
    end
end

function _stored_final_outputs(data)
    a = _last_or_nothing(_getfield_or(data, :a_sol))
    p = _last_or_nothing(_getfield_or(data, :Σp_sol))
    z = _last_or_nothing(_getfield_or(data, :Σz_sol))
    (a === nothing || p === nothing || z === nothing) && return nothing
    return (a=a, Sigma_p=p, Sigma_z=z)
end

function _stored_named_metrics(data)
    bm = _getfield_or(data, :benchmark_metrics)
    src = bm !== nothing ? bm : data
    inv = _getfield_or(src, :inversion)
    sil = _getfield_or(src, :silencing)
    coh = _getfield_or(src, :coherence)
    dur = _getfield_or(src, :duration)
    all(x -> x === nothing, (inv, sil, coh, dur)) && return nothing
    return (inversion=inv, silencing=sil, coherence=coh, duration=dur)
end

function _print_reference_audit(ref)
    println("================================================================================")
    println("load_jld2_reference")
    println("================================================================================")
    println("  jld2_path             = $(ref.jld2_path)")
    println("  pulsemat_path         = $(ref.pulsemat_path === nothing ? "<none>" : ref.pulsemat_path)")
    println("  SIM_SETTING           = parsed")
    println("  SYSTEM_CONFIG         = parsed")
    println("  PULSE_CONFIG parse    = $(ref.parse_ok ? "OK" : "FAIL")")
    println("  parse message         = $(ref.parse_message)")
    println("  pulse source          = $(ref.pulse_source)")
    println("  pulse source reason   = $(ref.pulse_source_reason)")
    println("  n_signal (identified) = $(ref.n_signal)")
    println("  n_signal_check        = $(ref.n_signal_check)")
    println("  USE_SIGNAL            = $(ref.use_signal)")
    println("  USE_SIGNAL note       = $(ref.use_signal_note)")
    println("  use_interior          = $(ref.use_interior)")
    println("  control trace         = $(ref.control_trace_note)")
    println("  fit_N                 = $(ref.fit_N) ($(ref.fit_N_source))")
    println("  ensemble M            = $(ref.d.M)  (M_delta=$(ref.d.M_delta) x M_g=$(ref.d.M_g))")
    println("  timespan              = $(ref.d.timespan)")
    println("  reltol / abstol       = $(ref.data.SIM_SETTING.reltol) / $(ref.data.SIM_SETTING.abstol)")
    println("================================================================================")
    return nothing
end

"""
    load_jld2_reference(path; n_signal=nothing, use_signal=false, use_interior=false,
                        fit_N=nothing, verbose=true)
        -> NamedTuple

Steps 1–4 of the jld2 pipeline. `use_interior` is stored on the returned
`ref` and consumed after step 7 by [`optimise_control_pulse_from_jld2`](@ref)
(it does not change this loader's own I/Q / `d` / signal split):

  1. Open `path` (a `.jld2`, or a `*_pulsemat.csv` whose sibling `.jld2` is used).
  2. Extract `SIM_SETTING`, `SYSTEM_CONFIG`, `PULSE_CONFIG`.
  3. Parse them into the reference configs (`d` via `build_full_config` /
     `prepare_derived`; pulse specs via [`try_parse_pulse_config`](@ref)).
  4. If `PULSE_CONFIG` is absent, load the sibling `*_pulsemat.csv`, identify
     signal vs control on that trace, and fit only the masked control envelope.

When `PULSE_CONFIG` is present but identification fails, this errors (no mixed
CSV fit). Seed I/Q is the summed control envelope; `signal_E_of_t` is the
fixed background (never in `u`). `n_signal` is an optional identified-count
check, not a positional split.
"""
function load_jld2_reference(
    path::AbstractString;
    n_signal::Union{Nothing,Integer}=jld2_pipeline_defaults().n_signal,
    use_signal::Bool=jld2_pipeline_defaults().use_signal,
    use_interior::Bool=jld2_pipeline_defaults().use_interior,
    fit_N::Union{Nothing,Integer}=jld2_pipeline_defaults().fit_N,
    verbose::Bool=true,
)
    jld2_path, pulsemat_path = _resolve_run_paths(path)
    isfile(jld2_path) || error("JLD2 file not found: $jld2_path")
    data = load_jld2_run(jld2_path)

    hasproperty(data, :SIM_SETTING) || error("$jld2_path has no SIM_SETTING.")
    hasproperty(data, :SYSTEM_CONFIG) || error("$jld2_path has no SYSTEM_CONFIG.")
    hasproperty(data.SIM_SETTING, :reltol) || error("$jld2_path SIM_SETTING has no reltol.")
    hasproperty(data.SIM_SETTING, :abstol) || error("$jld2_path SIM_SETTING has no abstol.")
    CONFIG = build_full_config(data.SIM_SETTING, data.SYSTEM_CONFIG)
    d = prepare_derived(CONFIG)

    pc_present = hasproperty(data, :PULSE_CONFIG) && data.PULSE_CONFIG !== nothing
    parse_result = try_parse_pulse_config(data; n_signal=n_signal, d=d)
    if pc_present && !parse_result.ok
        error(
            "PULSE_CONFIG is present but signal/control identification refused to " *
            "build a control envelope: $(parse_result.message) — not falling back to a " *
            "mixed CSV fit (that would parameterize the signal)."
        )
    end

    if parse_result.ok
        pulse_source = :pulse_config
        pulse_source_reason = "identified PULSE_CONFIG from $(basename(jld2_path)): $(parse_result.message)"
        sibling_n = _sibling_pulsemat_n_samples(jld2_path)
        fit_N_use = fit_N !== nothing ? Int(fit_N) : something(sibling_n, _DEFAULT_CONTROL_FIT_N)
        if fit_N !== nothing
            fit_N_source = "explicit fit_N"
        elseif sibling_n !== nothing
            fit_N_source = "from sibling _pulsemat.csv row count"
        else
            fit_N_source = "no sibling _pulsemat.csv found, default"
        end
        fit_N_use >= 2 || error("fit_N must be >= 2, got $fit_N_use.")
        t, Ex, Ep = _sample_control_cfg(parse_result.control_cfg, d, fit_N_use)
        signal_E_rec = signal_envelope_E_of_t(parse_result.identification; use_signal=true)
        control_E_rec = control_envelope_E_of_t(parse_result.identification)
        recorded_E_of_t = (tt -> signal_E_rec(tt) + control_E_rec(tt))
        control_trace_note = "one control envelope (sum of identified control sub-pulses; signal excluded)"
        signal_E_always = signal_E_rec
        signal_E_of_t = signal_envelope_E_of_t(parse_result.identification; use_signal=use_signal)
        use_signal_note = use_signal ?
            "USE_SIGNAL=true: identified signal is a fixed background (not in optimised u)" :
            "USE_SIGNAL=false: signal zeroed for optimisation; reference forward-sim still uses the recorded signal"
    else
        pulsemat_path !== nothing && isfile(pulsemat_path) || error(
            "PULSE_CONFIG unavailable ($(parse_result.message)) and no sibling " *
            "_pulsemat.csv next to $jld2_path — cannot build a recorded drive."
        )
        t_end, Ex_full, Ep_full = load_E_samples(pulsemat_path)
        N = length(Ex_full)
        N >= 2 || error("load_jld2_reference: $pulsemat_path has $N sample(s); need >= 2.")
        length(Ep_full) == N || error("load_jld2_reference: I/Q length mismatch $(N)/$(length(Ep_full)).")
        t = collect(range(0.0, t_end; length=N))
        isapprox(d.timespan[2] - d.timespan[1], t_end; rtol=1e-9) || error(
            "T_max mismatch: $jld2_path's own SIM_SETTING.Ttotal implies T_max=" *
            "$(d.timespan[2]-d.timespan[1])s, but $pulsemat_path's own metadata says " *
            "t_end=$(t_end)s — these two files don't actually belong together."
        )
        ident_tr = segment_signal_control_from_trace(t, Ex_full, Ep_full; d=d)
        ident_tr.ok || error(
            "CSV fallback: signal/control identification rejected the recorded trace: " *
            "$(ident_tr.reason) — refusing to fit the mixed drive into u."
        )
        if n_signal !== nothing && Int(n_signal) != _sc_n_signal(ident_tr)
            error(
                "n_signal=$n_signal does not match identified signal count " *
                "$(_sc_n_signal(ident_tr)) (CSV fallback identification is by waveform)."
            )
        end
        Ex, Ep = mask_control_envelope_samples(t, Ex_full, Ep_full, ident_tr)
        pulse_source = :pulsemat_csv
        pulse_source_reason =
            "PULSE_CONFIG unavailable; control envelope extracted from sibling _pulsemat.csv. " *
            "Parse message: $(parse_result.message)"
        if fit_N !== nothing && Int(fit_N) != N
            println("load_jld2_reference: fit_N=$fit_N ignored in CSV-fallback mode; using the file's own $N samples.")
        end
        fit_N_use = N
        fit_N_source = "CSV file sample count"
        recorded_E_of_t = build_E_of_t_from_samples(t, Ex_full, Ep_full)
        control_trace_note = "CSV I/Q with signal times zeroed (one control envelope only)"
        if _sc_n_signal(ident_tr) > 0
            Ex_sig = Ex_full .- Ex
            Ep_sig = Ep_full .- Ep
            signal_E_always = build_E_of_t_from_samples(t, Ex_sig, Ep_sig)
        else
            signal_E_always = _zero_drive
        end
        signal_E_of_t = use_signal ? signal_E_always : _zero_drive
        use_signal_note = use_signal ?
            "USE_SIGNAL=true: CSV-identified signal is a fixed interpolant (not in optimised u)" :
            "USE_SIGNAL=false: signal zeroed for optimisation; reconcile still uses the mixed recorded drive"
        parse_result = merge(parse_result, (
            signal_cfg=nothing, control_cfg=nothing, identification=ident_tr,
        ))
    end

    ref = (
        path=path,
        jld2_path=jld2_path,
        pulsemat_path=pulsemat_path,
        SIM_SETTING=data.SIM_SETTING,
        SYSTEM_CONFIG=data.SYSTEM_CONFIG,
        PULSE_CONFIG=parse_result.ok ? data.PULSE_CONFIG : nothing,
        parse_ok=parse_result.ok,
        parse_message=parse_result.message,
        pulse_source=pulse_source,
        pulse_source_reason=pulse_source_reason,
        signal_cfg=parse_result.signal_cfg,
        control_cfg=parse_result.control_cfg,
        data=data,
        d=d,
        identification=parse_result.identification,
        n_signal=_sc_n_signal(parse_result.identification),
        n_signal_check=n_signal,
        use_signal=use_signal,
        use_signal_note=use_signal_note,
        use_interior=use_interior,
        signal_E_of_t=signal_E_of_t,
        signal_E_always=signal_E_always,
        recorded_E_of_t=recorded_E_of_t,
        control_t=t,
        control_Ex=Ex,
        control_Ep=Ep,
        control_trace_note=control_trace_note,
        fit_N=fit_N_use,
        fit_N_source=fit_N_source,
    )
    verbose && _print_reference_audit(ref)
    return ref
end

"""
    run_reference_forward(ref; reltol=nothing, abstol=nothing, compute=:auto, verbose=true)
        -> NamedTuple

Step 5: final-state forward simulation of `ref.recorded_E_of_t` on
`:ground` and the weak-excitation `:weak` seed. Returns `ground` outputs
(`a`, `Σp`, `Σz`), the `weak` track's `Sp`, and
`metrics = (inversion, silencing, coherence, weak_seed_retention, duration)`
(everything after `silencing` is diagnostic -- see [`pulse_metrics`](@ref)).
Those metrics are the reference metrics for later optimisation comparison.
`reltol`/`abstol` default to `ref.SIM_SETTING`.
"""
function run_reference_forward(
    ref;
    reltol=nothing,
    abstol=nothing,
    compute::Symbol=jld2_optimizer_defaults().compute,
    verbose::Bool=true,
)
    data = ref.data
    d = ref.d
    reltol_s = reltol === nothing ? data.SIM_SETTING.reltol : reltol
    abstol_s = abstol === nothing ? data.SIM_SETTING.abstol : abstol

    a, Sp, Sz = run_sim_1st_order_final(
        ref.recorded_E_of_t, d;
        initial_condition=:ground, reltol=reltol_s, abstol=abstol_s, compute=compute,
    )
    _, Sp_eq, _ = run_sim_1st_order_final(
        ref.recorded_E_of_t, d;
        initial_condition=:weak, reltol=reltol_s, abstol=abstol_s, compute=compute,
    )

    Sigma_p = sum(Sp)
    Sigma_z = sum(Sz)
    inversion = _weighted_inversion(Sz, d.g_b, d.Nj, Float64)
    silencing = _weighted_silencing_factor(Sp_eq, d.g_b, d.Nj, d.delta_b, Float64)
    coherence = _weighted_coherence(Sp_eq, d.g_b, d.Nj, d.delta_b, Float64)
    weak_seed_retention = _weak_seed_retention(Sp_eq, d.g_b, d.Nj, d.delta_b, Float64)
    duration = _recorded_control_duration(
        ref.identification, ref.control_t, ref.control_Ex, ref.control_Ep,
    )
    metrics = (
        inversion=inversion, silencing=silencing, coherence=coherence,
        weak_seed_retention=weak_seed_retention, duration=duration,
    )
    forward = (
        ground=(a=a, Sigma_p=Sigma_p, Sigma_z=Sigma_z, Sp=Sp, Sz=Sz),
        weak=(Sp=Sp_eq,),
        metrics=metrics,
        reltol=reltol_s,
        abstol=abstol_s,
        compute=compute,
    )
    if verbose
        println("Reference forward simulation ($(ref.jld2_path), final state):")
        println("  :ground   inversion=$(round(inversion, sigdigits=6))  a=$(a)  Σz=$(Sigma_z)")
        println(
            "  :weak     silencing=$(round(silencing, sigdigits=6))  " *
            "coherence=$(round(coherence, sigdigits=6))  " *
            "weak_seed_retention=$(round(weak_seed_retention, sigdigits=6))  " *
            "duration=$(round(duration, sigdigits=6))s"
        )
    end
    return forward
end

"""
    reconcile_reference(ref, forward; rtol=1e-3, atol=nothing, verbose=true)
        -> (ok, report)

Step 6: if the `.jld2` stores results, compare this run's **final state**
against them. If it stores none, auto-PASS.

Stored outputs (`a_sol`/`Σp_sol`/`Σz_sol`) are compared to the `:ground`
forward solve. Inversion derived from stored `Σz` is compared too.
Named stored metrics (`inversion`/`silencing`/`coherence`/`duration`,
including inside `benchmark_metrics` if present) are compared when they
exist. Missing pieces are skipped, not failed.
"""
function reconcile_reference(
    ref, forward;
    rtol::Real=jld2_pipeline_defaults().rtol_check,
    atol=jld2_pipeline_defaults().atol_check,
    verbose::Bool=true,
)
    data = ref.data
    d = ref.d
    stored_out = _stored_final_outputs(data)
    stored_met = _stored_named_metrics(data)
    atol_use = atol === nothing ? _default_final_state_atol(forward.abstol) : atol

    if stored_out === nothing && stored_met === nothing
        verbose && println(
            "Reconciliation against $(ref.jld2_path): auto-PASS " *
            "(file stores no a_sol/Σp_sol/Σz_sol and no named metrics)."
        )
        report = (
            auto_pass=true, ok=true, reason="no stored results",
            rel_a=NaN, rel_p=NaN, rel_z=NaN, rel_inv=NaN,
            rel_sil=NaN, rel_coh=NaN, rel_dur=NaN,
        )
        return true, report
    end

    ok = true
    rel_a = rel_p = rel_z = rel_inv = rel_sil = rel_coh = rel_dur = NaN
    err_a = err_p = err_z = NaN

    if stored_out !== nothing
        g = forward.ground
        err_a = abs(g.a - stored_out.a)
        err_p = abs(g.Sigma_p - stored_out.Sigma_p)
        err_z = abs(g.Sigma_z - stored_out.Sigma_z)
        rel_a = err_a / (abs(stored_out.a) + atol_use)
        rel_p = err_p / (abs(stored_out.Sigma_p) + atol_use)
        rel_z = err_z / (abs(stored_out.Sigma_z) + atol_use)
        our_inv = _final_inversion(g.Sigma_z, d.N_total)
        stored_inv = _final_inversion(stored_out.Sigma_z, d.N_total)
        rel_inv = abs(our_inv - stored_inv) / (abs(stored_inv) + atol_use)
        ok = ok && rel_a < rtol && rel_p < rtol && rel_z < rtol && rel_inv < rtol
    end

    if stored_met !== nothing
        m = forward.metrics
        if stored_met.inversion !== nothing
            rel_inv_m = abs(m.inversion - stored_met.inversion) / (abs(stored_met.inversion) + atol_use)
            isnan(rel_inv) && (rel_inv = rel_inv_m)
            ok = ok && rel_inv_m < rtol
        end
        if stored_met.silencing !== nothing
            rel_sil = abs(m.silencing - stored_met.silencing) / (abs(stored_met.silencing) + atol_use)
            ok = ok && rel_sil < rtol
        end
        if stored_met.coherence !== nothing
            # Recorded for information only -- NOT gated into `ok`. `coherence`
            # is a pure diagnostic (like `field_amp`/`weak_seed_retention`,
            # neither of which is compared here), and its definition changed
            # with the paper alignment (now the per-frequency-slice magnitude
            # companion to |F|_⋆), so a run saved under the old per-bin
            # definition would spuriously fail an equality check.
            rel_coh = abs(m.coherence - stored_met.coherence) / (abs(stored_met.coherence) + atol_use)
        end
        if stored_met.duration !== nothing
            rel_dur = abs(m.duration - stored_met.duration) / (abs(stored_met.duration) + atol_use)
            ok = ok && rel_dur < rtol
        end
    end

    if verbose
        status = ok ? "PASS" : "FAIL"
        println("Reconciliation against $(ref.jld2_path) (final state): $status (rtol=$rtol, atol=$atol_use)")
        stored_out !== nothing && println("  a:  abs_err=$err_a  rel_err=$rel_a")
        stored_out !== nothing && println("  Σz: abs_err=$err_z  rel_err=$rel_z")
        stored_out !== nothing && println("  Σp: abs_err=$err_p  rel_err=$rel_p")
        !isnan(rel_inv) && println("  inversion rel_err=$rel_inv")
        !isnan(rel_sil) && println("  silencing rel_err=$rel_sil")
        !isnan(rel_coh) && println("  coherence rel_err=$rel_coh")
        !isnan(rel_dur) && println("  duration  rel_err=$rel_dur")
        stored_out === nothing && println("  (no stored a_sol/Σp_sol/Σz_sol — output check skipped)")
        stored_met === nothing && println("  (no stored named metrics — metric check skipped)")
    end

    report = (
        auto_pass=false, ok=ok,
        rel_a=rel_a, rel_p=rel_p, rel_z=rel_z, rel_inv=rel_inv,
        rel_sil=rel_sil, rel_coh=rel_coh, rel_dur=rel_dur,
        err_a=err_a, err_p=err_p, err_z=err_z,
        inversion=forward.metrics.inversion,
    )
    return ok, report
end

"""
    fit_linear_seed(ref; k=nothing, n_coeff_A=nothing, n_coeff_f=nothing,
        param_budget=60, degree=3, taper_frac=0.1, verbose=true)
        -> (pulse, u_fit, fit_report, segments)

Step 7: closed-form `fit_mode=:linear` seed of the **control pulse** from
`ref`'s control I/Q (PULSE_CONFIG control segments, or the CSV when
PULSE_CONFIG was unavailable). The signal is not in this fit.
AUTO (all of `k`/`n_coeff_A`/`n_coeff_f` omitted) sizes the shape from
`param_budget`. EXACT (all three given) fits that shape.
"""
function fit_linear_seed(
    ref;
    k::Union{Nothing,Integer}=nothing,
    n_coeff_A::Union{Nothing,Integer}=nothing,
    n_coeff_f::Union{Nothing,Integer}=nothing,
    param_budget::Integer=jld2_pipeline_defaults().param_budget,
    degree::Integer=jld2_optimizer_defaults().degree,
    taper_frac::Real=jld2_optimizer_defaults().taper_frac,
    rel_thresh::Real=1e-3,
    min_active_samples::Integer=5,
    min_silence_samples::Integer=3,
    verbose::Bool=true,
)
    n_given = count(!isnothing, (k, n_coeff_A, n_coeff_f))
    n_given == 0 || n_given == 3 || error(
        "k/n_coeff_A/n_coeff_f must be given ALL THREE (EXACT seed shape) or NONE " *
        "(AUTO mode: detected k, n_coeff sized from param_budget=$param_budget) — got " *
        "k=$k, n_coeff_A=$n_coeff_A, n_coeff_f=$n_coeff_f."
    )

    t, Ex, Ep, d = ref.control_t, ref.control_Ex, ref.control_Ep, ref.d
    length(t) == length(Ex) == length(Ep) || error(
        "fit_linear_seed: control trace lengths t/Ex/Ep = $(length(t))/$(length(Ex))/$(length(Ep)) must match."
    )
    length(t) >= 2 || error("fit_linear_seed: control trace has $(length(t)) sample(s); need >= 2.")
    if n_given == 0
        pulse, u_fit, fit_report, segments = fit_composite_pulse_from_samples(
            t, Ex, Ep, d; fit_mode=:linear, param_budget=param_budget,
            degree=degree, taper_frac=taper_frac, rel_thresh=rel_thresh,
            min_active_samples=min_active_samples, min_silence_samples=min_silence_samples,
        )
        if verbose
            println(
                "Auto-detected k=$(pulse.k) sub-pulse(s) from $(ref.pulse_source) control trace; " *
                "sized n_coeff_A=n_coeff_f=$(pulse.n_coeff_A) to keep n_params=$(n_params(pulse)) " *
                "<= param_budget=$param_budget."
            )
        end
    else
        pulse, u_fit, fit_report, segments = fit_composite_pulse_seed_linear_exact(
            t, Ex, Ep, d, k, n_coeff_A, n_coeff_f;
            degree=degree, taper_frac=taper_frac, rel_thresh=rel_thresh,
            min_active_samples=min_active_samples, min_silence_samples=min_silence_samples,
        )
    end
    if verbose
        println(
            "Fitted a k=$(pulse.k) CompositePulse seed from $(ref.pulse_source) " *
            "(linear least-squares): rel_l2_complex=$(round(fit_report.rel_l2_complex, sigdigits=4)) " *
            "rel_l2_A=$(round(fit_report.rel_l2_A, sigdigits=4)) " *
            "rel_l2_f=$(round(fit_report.rel_l2_f, sigdigits=4)) " *
            "phi_rms_rad=$(round(fit_report.phi_rms_rad, sigdigits=4))"
        )
    end
    return pulse, u_fit, fit_report, segments
end

"""
    fit_composite_pulse_seed_linear_exact(t, Ex, Ep, d, k, n_coeff_A, n_coeff_f;
        degree=3, taper_frac=0.1, rel_thresh=1e-3, min_active_samples=5,
        min_silence_samples=3, cA_floor_frac, cf_clip_mult)
        -> (pulse::CompositePulse, u_fit, fit_report, segments)

Closed-form `fit_mode=:linear` fit of an already-sampled control trace
`(t, Ex, Ep)` into the EXACT `(k, n_coeff_A, n_coeff_f)` shape
[`fit_linear_seed`](@ref) / [`optimise_control_pulse_from_jld2`](@ref)
will hand to `optimise_composite_pulse` as `warm_start_u`. Requires
`n_coeff_A == n_coeff_f`. Detected segment count must equal `k`.
`cA_floor_frac` / `cf_clip_mult` are forwarded to
[`_fit_composite_pulse_from_samples_linear`](@ref).
"""
function fit_composite_pulse_seed_linear_exact(
    t::AbstractVector, Ex::AbstractVector, Ep::AbstractVector, d,
    k::Integer, n_coeff_A::Integer, n_coeff_f::Integer;
    degree::Integer=3, taper_frac::Real=0.1,
    rel_thresh::Real=1e-3, min_active_samples::Integer=5, min_silence_samples::Integer=3,
    cA_floor_frac::Real=_GRAD_SAFE_FRAC, cf_clip_mult::Real=20.0,
)
    n_coeff_A == n_coeff_f || error(
        "fit_composite_pulse_seed_linear_exact requires n_coeff_A == n_coeff_f " *
        "(got n_coeff_A=$n_coeff_A, n_coeff_f=$n_coeff_f) -- " *
        "fit_composite_pulse_from_samples's fit_mode=:linear implementation sizes one shared n_coeff for both."
    )

    A = sqrt.(Ex .^ 2 .+ Ep .^ 2)
    segments = _detect_subpulse_segments(
        t, A; rel_thresh=rel_thresh, min_active_samples=min_active_samples, min_silence_samples=min_silence_samples,
    )
    length(segments) == k || error(
        "Detected $(length(segments)) sub-pulse(s) in this trace (rel_thresh=$rel_thresh, " *
        "min_active_samples=$min_active_samples, min_silence_samples=$min_silence_samples), " *
        "but k=$k was requested -- pass a k matching the trace's own structure, or adjust the " *
        "segment-detection kwargs so it does."
    )

    n_pieces_target = n_coeff_A - degree
    n_pieces_target >= 1 || error(
        "n_coeff_A=$n_coeff_A is too small for degree=$degree (need n_coeff_A >= degree+1 = " *
        "$(degree+1))."
    )
    n_samples_max = maximum(i_end - i_start + 1 for (i_start, i_end) in segments)
    points_per_segment = cld(n_samples_max, n_pieces_target)

    # points_per_segment = cld(n_samples_max, n_pieces_target) is the
    # SMALLEST pps with _spline_coeff_count's own ceil(n_samples_max/pps)
    # <= n_pieces_target -- not necessarily one that hits n_pieces_target
    # EXACTLY. Because ceil-division skips values as pps decreases by 1 at
    # a time, some (n_samples_max, n_pieces_target) pairs have NO integer
    # pps landing on n_pieces_target exactly (e.g. n_samples_max=16,
    # n_pieces_target=5: achievable n_pieces are {..., 4, 6, ...}, 5 is
    # skipped) -- a real mathematical gap, not a bug to search harder
    # around. Check for this BEFORE the fit (not just via the post-hoc
    # pulse.n_coeff_A==n_coeff_A check below) so a caller who hits it gets
    # an actionable "these are the nearest achievable n_coeff_A values"
    # message instead of a confusing "internal inconsistency" one.
    n_pieces_lo = cld(n_samples_max, points_per_segment)
    if n_pieces_lo != n_pieces_target
        n_coeff_lo = n_pieces_lo + degree
        hi_msg = if points_per_segment > 1
            n_pieces_hi = cld(n_samples_max, points_per_segment - 1)
            "or n_coeff_A=$(n_pieces_hi + degree)"
        else
            "(no larger achievable value -- n_coeff_A=$n_coeff_lo is already the finest resolution this segment's $n_samples_max samples support)"
        end
        error(
            "n_coeff_A=$n_coeff_A is not achievable for a segment of $n_samples_max samples " *
            "at degree=$degree (points_per_segment-based sizing has a gap here) -- nearest " *
            "achievable values are n_coeff_A=$n_coeff_lo $hi_msg. Pick one of those, or use " *
            "AUTO mode (omit k/n_coeff_A/n_coeff_f, pass param_budget) to size n_coeff " *
            "automatically instead."
        )
    end

    pulse, u_fit, fit_report, segments = _fit_composite_pulse_from_samples_linear(
        t, Ex, Ep, d; points_per_segment=points_per_segment, degree=degree, taper_frac=taper_frac,
        segments=segments, cA_floor_frac=cA_floor_frac, cf_clip_mult=cf_clip_mult,
    )
    (pulse.k == k && pulse.n_coeff_A == n_coeff_A && pulse.n_coeff_f == n_coeff_f) || error(
        "internal inconsistency: fit_composite_pulse_seed_linear_exact computed " *
        "points_per_segment=$points_per_segment expecting (k=$k, n_coeff_A=$n_coeff_A, " *
        "n_coeff_f=$n_coeff_f), but the fit actually produced (k=$(pulse.k), " *
        "n_coeff_A=$(pulse.n_coeff_A), n_coeff_f=$(pulse.n_coeff_f))."
    )

    return pulse, u_fit, fit_report, segments
end

"""
    _final_inversion(Sigma_z, N_total) -> Float64

Collective inversion fraction `(Re[Sigma_z] + N_total/2) / N_total` at a
SINGLE (final) state -- `0` for the fully-ground ensemble, `1` for fully
inverted, matching `Sigma_z = Σ_j Sz_j`'s own `[-N_total/2, N_total/2]`
physical range (each bin's own `Sz_j` ranges over `[-Nj/2, Nj/2]`).
Deliberately final-state-only, not a trajectory summary -- see
[`run_sim_1st_order_final`](@ref)/[`reconcile_reference`](@ref)'s own
docstrings for why this package's `.jld2` reconciliation checks avoid a
full-trajectory CPU solve wherever the endpoint alone answers the question.
"""
_final_inversion(Sigma_z, N_total::Real) = (real(Sigma_z) + N_total / 2) / N_total

"""
    run_sim_1st_order_trajectory(E_of_t, d; initial_condition=:ground, alg=Tsit5(), reltol=1e-8, abstol=1e-8, tstops=Float64[]) -> (t, a, Sp, Sz)

Forward-only (not `ForwardDiff`-differentiated) CPU analogue of
`run_sim_1st_order` returning the FULL trajectory at `d.t_save`, unlike
[`run_sim_1st_order_pure`](@ref) which only returns the final state --
reuses the SAME `rhs_1st_order!`/ensemble machinery, just saving every
requested time point instead of one. `Sp`/`Sz` are returned as `(Nt, M)`
matrices (one row per saved time point, matching this package's own
`Sp_sol`/`Sz_sol` orientation from `run_sim_1st_order`). Used by
[`generate_2n1_arp_pi_pulse`](@ref) and datagen; the jld2 reconcile path
uses [`run_sim_1st_order_final`](@ref) (endpoint only).
"""
function run_sim_1st_order_trajectory(
    E_of_t, d;
    initial_condition::Symbol=:ground, alg=Tsit5(),
    reltol=1e-8, abstol=1e-8, tstops=Float64[],
    compute::Symbol=:auto,
)
    M = _assert_ensemble_shapes(d)
    hasproperty(d, :t_save) || error("run_sim_1st_order_trajectory: derived ensemble `d` is missing t_save.")
    compute_eff = _resolve_compute(compute, M)
    u0 = build_u0_1st_order_cpu(M, d.Nj, Float64, initial_condition)
    t, a, Sp, Sz = _run_sim_1st_order_from_u0(
        u0, E_of_t, d;
        alg=alg, reltol=reltol, abstol=abstol, tstops=tstops,
        save_mode=:trajectory, t_save=d.t_save, compute=compute_eff,
    )
    size(Sp) == (length(t), M) && size(Sz) == (length(t), M) || error(
        "run_sim_1st_order_trajectory: Sp/Sz shapes $(size(Sp))/$(size(Sz)) != ($((length(t), M)))."
    )
    return t, a, Sp, Sz
end

"""
    _default_final_state_atol(abstol_solve) -> Float64

Default absolute-error floor (`100 * abstol_solve`) for a final-state
`a`/`Σp`/`Σz` comparison against a saved run -- shared by
[`reconcile_reference`](@ref) and [`optimise_control_pulse_from_jld2`](@ref),
both of which compare a freshly-simulated FINAL
state against a recorded one. An endpoint that happens to be near-zero
(e.g. the cavity field after a completed pi-pulse) makes a PURELY relative
comparison hypersensitive to ordinary solver-noise-level absolute error --
tying the floor to `abstol_solve` (what the ODE solver itself already
considers "converged"), rather than to `0` or an arbitrary hardcoded
constant, keeps the comparison meaningful across runs with very different
tolerances/state scales.
"""
_default_final_state_atol(abstol_solve::Real) = 100 * abstol_solve

"""
    run_sim_1st_order_final(E_of_t, d; initial_condition=:ground, alg=Tsit5(), reltol=1e-8, abstol=1e-8, tstops=Float64[]) -> (a, Sp, Sz)

Forward-only (not `ForwardDiff`-differentiated) CPU analogue of
`run_sim_1st_order` returning ONLY the FINAL state at `d.timespan[2]`, unlike
[`run_sim_1st_order_trajectory`](@ref) which saves every point in
`d.t_save` -- reuses the SAME `rhs_1st_order!`/ensemble machinery, just with
`save_everystep=false, save_start=false` so the solver keeps only the last
step. `Sp`/`Sz` are the per-bin, length-`M` state AT THE FINAL TIME (same
shape [`run_sim_1st_order_pure`](@ref) returns for a `CompositePulse`, here
for an arbitrary `E_of_t` closure instead). Cheaper than
[`run_sim_1st_order_trajectory`](@ref) whenever only the ENDPOINT is
needed -- e.g. [`reconcile_reference`](@ref)'s own sanity check against
a saved run's final `a_sol[end]`/`Σp_sol[end]`/`Σz_sol[end]`, which does not
need every intermediate saved timepoint the way a peak-inversion-timing
check does.
"""
function run_sim_1st_order_final(
    E_of_t, d;
    initial_condition::Symbol=:ground, alg=Tsit5(),
    reltol=1e-8, abstol=1e-8, tstops=Float64[],
    compute::Symbol=:auto,
)
    M = _assert_ensemble_shapes(d)
    compute_eff = _resolve_compute(compute, M)
    u0 = build_u0_1st_order_cpu(M, d.Nj, Float64, initial_condition)
    a, Sp, Sz = _run_sim_1st_order_from_u0(
        u0, E_of_t, d;
        alg=alg, reltol=reltol, abstol=abstol, tstops=tstops,
        save_mode=:final, compute=compute_eff,
    )
    return a, Sp, Sz
end

"""
    optrunlog_paths(path; out_dir=nothing) -> (optrunlog_path, pulsemat_path, pulsepara_path)

Derives this run's `<basename>_optrunlog.jld2`/`<basename>_opt_pulsemat.csv`/
`<basename>_opt_pulsepara.jld2` output paths from the SOURCE `.jld2` path's
own basename -- the same "same directory, same basename + suffix"
convention `save_run_data` already uses for its own `_pulsemat.csv`
sibling (see `pulses.jl`). Pass `out_dir` to write all three files
elsewhere instead of alongside `path`; either way the containing directory
is created if it doesn't exist yet.
"""
function optrunlog_paths(path::AbstractString; out_dir=nothing)
    endswith(path, ".jld2") || error("Expected a .jld2 path, got $path.")
    base = path[1:end-length(".jld2")]
    dir = out_dir === nothing ? dirname(base) : out_dir
    isempty(dir) || mkpath(dir)
    base_name = basename(base)
    return joinpath(dir, base_name * "_optrunlog.jld2"),
           joinpath(dir, base_name * "_opt_pulsemat.csv"),
           joinpath(dir, base_name * "_opt_pulsepara.jld2")
end

"""
    save_optimised_pulse_parameters(pulsepara_path, source_path, pulse, best_u; final_metrics=nothing) -> pulsepara_path

Writes ONLY the FINAL optimised control pulse's exact parameters to
`pulsepara_path` (`<basename>_opt_pulsepara.jld2`, see
[`optrunlog_paths`](@ref)) -- deliberately separate from, and much
smaller than, `_optrunlog.jld2`'s full per-epoch `history`/settings/
output record, so a caller who only wants the converged pulse can load a
small, self-contained file without pulling in the whole run log. No
per-epoch trail is kept here, just this one converged result.

The `.jld2` file holds, under the top-level key `"data"` (same convention
[`load_jld2_run`](@ref) already reads):
  - `source_path` -- the ORIGINAL `.jld2` run this optimisation started from
  - `k`, `n_coeff_A`, `n_coeff_f`, `degree`, `taper_frac`, `T_max`,
    `amp_scale` -- `pulse`'s own defining fields, everything needed to
    reconstruct an identical `CompositePulse`
  - `final_u` -- the optimised raw parameter vector (`best_u`)
  - `t_start`, `t_end`, `phi0`, `cA`, `cf` -- the DECODED pulse parameters
    (see [`decode`](@ref)): each sub-pulse's start/end time, its own
    discrete additive phase jump, and its amplitude/frequency B-spline
    coefficients, i.e. the actual physical pulse shape `best_u` encodes
  - `final_metrics` -- `(cost, inversion, silencing, duration, coherence)`
    from [`pulse_cost`](@ref) at `best_u`, if supplied (`nothing`
    otherwise); context only, not required to reconstruct the pulse
"""
function save_optimised_pulse_parameters(
    pulsepara_path::AbstractString, source_path::AbstractString,
    pulse::CompositePulse, best_u::AbstractVector;
    final_metrics=nothing,
)
    t_start, t_end, phi0, cA, cf = decode(pulse, best_u)
    pulsepara = (
        source_path=source_path,
        k=pulse.k, n_coeff_A=pulse.n_coeff_A, n_coeff_f=pulse.n_coeff_f,
        degree=pulse.degree, taper_frac=pulse.taper_frac,
        T_max=pulse.T_max, amp_scale=pulse.amp_scale,
        final_u=collect(best_u),
        t_start=collect(t_start), t_end=collect(t_end), phi0=collect(phi0),
        cA=collect(cA), cf=collect(cf),
        final_metrics=final_metrics,
    )
    JLD2.save(pulsepara_path, "data", pulsepara)
    println("Saved optimised pulse parameters to $pulsepara_path")
    return pulsepara_path
end

"""
    save_optimisation_run_log(path, data, d, pulse, signal_E_of_t,
                               n_signal, use_signal, u0, initial_metrics,
                               best_u, final_metrics, history, optimizer_settings;
                               out_dir=nothing, pulsemat_N=nothing, benchmark_metrics=nothing)
        -> (optrunlog_path, pulsemat_path, pulsepara_path)

Writes this run's full record to `<basename>_optrunlog.jld2` (see
[`optrunlog_paths`](@ref)), samples the FINAL/optimal CONTROL pulse's own
drive (`build_E_of_t(pulse, best_u)` -- the control pulse alone, NOT
combined with the signal, since the signal is a separate FIXED input that
was never part of what got optimised) to `<basename>_opt_pulsemat.csv`
via the existing [`sample_E_of_t`](@ref)/[`save_E_samples`](@ref) (same
format/read pattern as every other `_pulsemat.csv` in this package), and
writes the same final pulse's exact/decoded parameters to
`<basename>_opt_pulsepara.jld2` via
[`save_optimised_pulse_parameters`](@ref) (a small, standalone file --
see its own docstring for what it holds; `final_u` also still lives
inside `_optrunlog.jld2` below, unchanged, since
[`optimise_composite_pulse`](@ref)'s `warm_start_u` continues to read it
from there).
`pulsemat_N` defaults to the source run's own `SIM_SETTING.Nt_save`, for
comparability with the original file's own sampling density.

The `.jld2` log holds, all under the top-level key `"data"` (same
convention [`load_jld2_run`](@ref) already reads):
  - `source_path`, `n_signal`, `use_signal`
  - `SIM_SETTING`, `SYSTEM_CONFIG` (copied from the source run, for
    provenance/reproducibility -- enough, together with `k`/`n_coeff_A`/
    `n_coeff_f` below, to rebuild `d`/`pulse` deterministically later)
  - `k`, `n_coeff_A`, `n_coeff_f`
  - `optimizer_settings` -- every setting that actually affects
    replication of this run: `USE_SIGNAL`/`n_signal` plus every knob
    [`optimise_composite_pulse`](@ref) exposes (`num_epochs`,
    `learning_rate`, `patience`, `tol`, `n_hops`, `hop_patience`,
    `hop_step_size`, `temperature`, `w_tmax`, `seed`, `degree`,
    `taper_frac`, and any numeric
    `solve_kwargs` override such as `reltol`/`abstol`/`target_F`/`w_time`)
    -- see [`optimise_composite_pulse`](@ref)'s own
    docstring for exactly what it captures and why (and what it
    deliberately excludes, e.g. non-serialisable closures)
  - `benchmark_metrics` -- `(inversion, duration, silencing, coherence)` of
    the SOURCE FILE's own original, unfitted pulse (never the optimised
    CompositePulse), as computed by
    [`run_reference_forward`](@ref); `nothing` if the caller omitted it
  - `initial_u` (the candidate pulse's own raw parameterisation, `u0`,
    used only after reconciliation passed)
  - `initial_metrics` (`(cost, inversion, silencing, duration, coherence)`
    at `u0`, from [`pulse_cost`](@ref) -- `cost` depends only on
    inversion and the collective silencing factor `|F|` (see
    [`_weighted_silencing_factor`](@ref)) plus the time/power penalties;
    `coherence` rides along in the same tuple but, like `duration`, is
    NOT part of `cost` -- it is the OLDER per-bin `Nj`-weighted mean of
    `|Sp|/(ε·Nj/2)` (see [`_weighted_coherence`](@ref)), DIAGNOSTIC ONLY,
    recorded purely for comparison against the collective `|F|` actually
    being optimised)
  - `initial_coherence` -- same `coherence` value as `initial_metrics`
    above, evaluated once at `u0` from a `:weak` solve, kept as its
    own top-level key purely for convenience so a saved run can be
    compared against that simpler metric without digging into the
    `initial_metrics` tuple
  - `initial_output` (`(a, Sigma_p, Sigma_z)` at `t1`, from actually
    simulating `u0` -- the candidate pulse's raw simulated output)
  - `history` -- one row per optimiser epoch, across every hop (see
    [`run_local_adam`](@ref)): `hop, epoch, k, cost, inversion, silencing,
    duration, coherence, improved`. `coherence` here is the SAME
    diagnostic-only per-bin metric as `initial_coherence`/`final_coherence`
    below, recorded for every epoch (reusing the `:weak` solve already
    run for `silencing`, so it costs nothing extra) -- never fed into
    `cost`, which the optimiser actually descends on. History rows do NOT
    carry the raw pulse parameter vector `u` for that epoch -- only the
    FINAL optimised pulse's exact parameters are saved, and in a separate
    file (see [`save_optimised_pulse_parameters`](@ref)/
    `_opt_pulsepara.jld2` below), not embedded here
  - `final_u` (the optimised control pulse's raw parameterisation,
    `best_u`)
  - `final_metrics`/`final_output`/`final_coherence` -- same shape as
    the initial ones, for `best_u`
"""
function save_optimisation_run_log(
    path::AbstractString, data, d, pulse::CompositePulse, signal_E_of_t,
    n_signal::Union{Nothing,Integer}, use_signal::Bool,
    u0::AbstractVector, initial_metrics,
    best_u::AbstractVector, final_metrics, history, optimizer_settings;
    out_dir=nothing, pulsemat_N=nothing, benchmark_metrics=nothing,
)
    optrunlog_path, pulsemat_path, pulsepara_path = optrunlog_paths(path; out_dir=out_dir)

    a0, Sp0, Sz0, _ = run_sim_1st_order_pure(u0, pulse, d; signal_E_of_t=signal_E_of_t)
    initial_output = (a=a0, Sigma_p=sum(Sp0), Sigma_z=sum(Sz0))

    a1, Sp1, Sz1, _ = run_sim_1st_order_pure(best_u, pulse, d; signal_E_of_t=signal_E_of_t)
    final_output = (a=a1, Sigma_p=sum(Sp1), Sigma_z=sum(Sz1))

    # Diagnostic only -- NOT part of the optimised cost (pulse_cost uses
    # the silencing factor |F|_⋆). Logged as top-level keys for convenience:
    # two extra :weak solves at u0/best_u, done once here, not every epoch.
    # `coherence` is the per-slice magnitude companion to |F|_⋆; `retention`
    # is its un-clamped value (weak-excitation validity check).
    _, Sp0_eq, _, Nj0_eq = run_sim_1st_order_pure(u0, pulse, d; signal_E_of_t=signal_E_of_t, initial_condition=:weak)
    initial_coherence = _weighted_coherence(Sp0_eq, d.g_b, Nj0_eq, d.delta_b, Float64)
    initial_weak_seed_retention = _weak_seed_retention(Sp0_eq, d.g_b, Nj0_eq, d.delta_b, Float64)
    _, Sp1_eq, _, Nj1_eq = run_sim_1st_order_pure(best_u, pulse, d; signal_E_of_t=signal_E_of_t, initial_condition=:weak)
    final_coherence = _weighted_coherence(Sp1_eq, d.g_b, Nj1_eq, d.delta_b, Float64)
    final_weak_seed_retention = _weak_seed_retention(Sp1_eq, d.g_b, Nj1_eq, d.delta_b, Float64)

    full_settings = merge((n_signal=n_signal, USE_SIGNAL=use_signal), optimizer_settings)

    run_log = (
        source_path=path, n_signal=n_signal, use_signal=use_signal,
        SIM_SETTING=data.SIM_SETTING, SYSTEM_CONFIG=data.SYSTEM_CONFIG,
        k=pulse.k, n_coeff_A=pulse.n_coeff_A, n_coeff_f=pulse.n_coeff_f,
        optimizer_settings=full_settings,
        benchmark_metrics=benchmark_metrics,
        initial_u=collect(u0), initial_metrics=initial_metrics, initial_output=initial_output,
        initial_coherence=initial_coherence,
        initial_weak_seed_retention=initial_weak_seed_retention,
        history=history,
        final_u=collect(best_u), final_metrics=final_metrics, final_output=final_output,
        final_coherence=final_coherence,
        final_weak_seed_retention=final_weak_seed_retention,
    )
    JLD2.save(optrunlog_path, "data", run_log)
    println("Saved optimisation run log to $optrunlog_path")

    N = if pulsemat_N === nothing
        hasproperty(data.SIM_SETTING, :Nt_save) || error(
            "save_optimisation_run_log: pulsemat_N was not given and SIM_SETTING has no Nt_save."
        )
        data.SIM_SETTING.Nt_save
    else
        Int(pulsemat_N)
    end
    N >= 2 || error("save_optimisation_run_log: pulsemat sample count must be >= 2, got $N.")
    control_E_of_t = build_E_of_t(pulse, best_u)
    sample_E_of_t(control_E_of_t, pulse.T_max, N; savepath=pulsemat_path)

    save_optimised_pulse_parameters(pulsepara_path, path, pulse, best_u; final_metrics=final_metrics)

    return optrunlog_path, pulsemat_path, pulsepara_path
end

"""
    optimise_control_pulse_from_jld2(path, k=nothing, n_coeff_A=nothing, n_coeff_f=nothing; kwargs...)
        -> (best_u, best_cost, pulse, signal_E_of_t, d, data, seed_fit_report, reference_metrics)

Linear pipeline:

  1–4. [`load_jld2_reference`](@ref) — open the `.jld2`, extract/parse
       `SIM_SETTING`/`SYSTEM_CONFIG`/`PULSE_CONFIG`, fall back to
       `*_pulsemat.csv` only if `PULSE_CONFIG` cannot be parsed.
  5.   [`run_reference_forward`](@ref) — `:ground` and `:weak` final-state
       solves. Those metrics are the reference metrics.
  6.   [`reconcile_reference`](@ref) — if the file stores results, compare
       this run's final-state outputs and metrics against them; if it stores
       none, auto-PASS. FAIL refuses to continue. PASS is the gate that
       lets steps 7–8 call `pulse_optimizer2.jl`.
  7.   [`fit_linear_seed`](@ref) — `fit_mode=:linear` of the **control
       pulse** (`PULSE_CONFIG` control segments, or the CSV). Signal is
       not fit. Skipped if `fit_seed_from_file=false` or `warm_start_u`
       is passed.
  7b.  [`generate_interior_seed`](@ref) — when `use_interior=true`,
       rewrite that `u_fit` toward inversion≈silencing≈0.5 using
       step 5's `inversion`/`silencing`. Default `use_interior=false`
       leaves the linear seed unchanged.
  8.   [`optimise_composite_pulse`](@ref) (`pulse_optimizer2.jl`) optimises
       that same **control pulse** on the reference ensemble `d`,
       warm-started from the seed. Parsed signal is a fixed background
       (`signal_E_of_t`), never in `u`.

Knobs: [`jld2_pipeline_defaults`](@ref) and [`jld2_optimizer_defaults`](@ref).
Caller kwargs overlay both; unknown names error rather than being splat
into the optimiser. The notebook calls the same helpers in this order.
"""
function optimise_control_pulse_from_jld2(
    path::AbstractString,
    k::Union{Nothing,Integer}=nothing,
    n_coeff_A::Union{Nothing,Integer}=nothing,
    n_coeff_f::Union{Nothing,Integer}=nothing;
    kwargs...,
)
    n_given = count(!isnothing, (k, n_coeff_A, n_coeff_f))
    n_given == 0 || n_given == 3 || error(
        "k/n_coeff_A/n_coeff_f must be given ALL THREE (EXACT seed shape) or NONE " *
        "(AUTO mode) — got k=$k, n_coeff_A=$n_coeff_A, n_coeff_f=$n_coeff_f."
    )

    pipe, opt_kwargs, verbose = _jld2_split_kwargs(kwargs)

    seed_fit_will_run = pipe.fit_seed_from_file && !haskey(opt_kwargs, :warm_start_u)
    seed_fit_will_run || n_given == 3 || error(
        "k/n_coeff_A/n_coeff_f must all be given explicitly when the seed fit won't run " *
        "(warm_start_u was provided, or fit_seed_from_file=false)."
    )

    verbose && println("=== 1–4  load_jld2_reference ===")
    ref = load_jld2_reference(
        path; n_signal=pipe.n_signal, use_signal=pipe.use_signal,
        use_interior=pipe.use_interior, fit_N=pipe.fit_N, verbose=verbose,
    )
    data = ref.data
    d = ref.d
    signal_E_of_t = ref.signal_E_of_t

    verbose && println("=== 5  run_reference_forward (:ground + :weak) ===")
    forward = run_reference_forward(
        ref; reltol=pipe.check_reltol, abstol=pipe.check_abstol,
        compute=opt_kwargs.compute, verbose=verbose,
    )
    reference_metrics = forward.metrics

    verbose && println("=== 6  reconcile_reference ===")
    ok, _ = reconcile_reference(
        ref, forward; rtol=pipe.rtol_check, atol=pipe.atol_check, verbose=verbose,
    )
    ok || error(
        "Reconciliation against $(ref.jld2_path) FAILED — refusing to optimise " *
        "against physics this port hasn't verified it can reproduce. " *
        "Inspect the printed errors above."
    )

    seed_fit_report = nothing
    if seed_fit_will_run
        verbose && println("=== 7  fit_linear_seed (CONTROL pulse, fit_mode=:linear) ===")
        seed_pulse, u_fit, fit_report, _ = fit_linear_seed(
            ref; k=k, n_coeff_A=n_coeff_A, n_coeff_f=n_coeff_f,
            param_budget=pipe.param_budget,
            degree=opt_kwargs.degree, taper_frac=opt_kwargs.taper_frac,
            verbose=verbose,
        )
        k, n_coeff_A, n_coeff_f = seed_pulse.k, seed_pulse.n_coeff_A, seed_pulse.n_coeff_f
        opt_kwargs = merge(opt_kwargs, (warm_start_u=u_fit,))
        seed_fit_report = (u_fit=u_fit, fit_report=fit_report)

        if pipe.use_interior
            verbose && println(
                "=== 7b  generate_interior_seed (I,S from step 5 → ~0.5,~0.5) ===",
            )
            I = reference_metrics.inversion
            S = reference_metrics.silencing
            pulse_int, u_int, interior_report, _ = generate_interior_seed(
                u_fit, I, S, seed_pulse, d;
                param_budget=pipe.param_budget,
                degree=opt_kwargs.degree, taper_frac=opt_kwargs.taper_frac,
                preserve_shape=true,
                N_samples=max(length(ref.control_t), 2),
            )
            k, n_coeff_A, n_coeff_f = pulse_int.k, pulse_int.n_coeff_A, pulse_int.n_coeff_f
            opt_kwargs = merge(opt_kwargs, (warm_start_u=u_int,))
            seed_fit_report = (
                u_fit=u_int, u_fit_linear=u_fit, fit_report=fit_report,
                interior_report=interior_report,
            )
            if verbose
                println(
                    "  inversion=$(round(I, sigdigits=6))  silencing=$(round(S, sigdigits=6))  " *
                    "amp_scale_factor=$(round(interior_report.amp_scale_factor, sigdigits=6))  " *
                    "chirp_bandwidth=$(interior_report.chirp_bandwidth)",
                )
            end
        end
    elseif pipe.use_interior
        error(
            "use_interior=true requires step 7 (fit_linear_seed). It is skipped when " *
            "fit_seed_from_file=false or warm_start_u is passed — set use_interior=false " *
            "to keep that warm start unchanged.",
        )
    end
    k === nothing && error("internal error: k is unset after the seed-fit branch.")
    n_coeff_A === nothing && error("internal error: n_coeff_A is unset after the seed-fit branch.")
    n_coeff_f === nothing && error("internal error: n_coeff_f is unset after the seed-fit branch.")

    verbose && println("=== 8  optimise_composite_pulse (CONTROL pulse, pulse_optimizer2.jl) ===")
    best_u, best_cost, pulse, u0, initial_metrics, history, final_metrics, optimizer_settings =
        _call_optimise_composite_pulse(k, n_coeff_A, n_coeff_f, d, signal_E_of_t, opt_kwargs)

    if pipe.save_log
        full_settings = merge(
            (pulse_source=ref.pulse_source, rtol_check=pipe.rtol_check, atol_check=pipe.atol_check,
             check_reltol=pipe.check_reltol, check_abstol=pipe.check_abstol,
             fit_seed_from_file=pipe.fit_seed_from_file, fit_N=ref.fit_N,
             param_budget=pipe.param_budget, n_signal_check=pipe.n_signal,
             use_interior=pipe.use_interior),
            optimizer_settings,
        )
        save_optimisation_run_log(
            ref.jld2_path, data, d, pulse, signal_E_of_t, ref.n_signal, pipe.use_signal,
            u0, initial_metrics, best_u, final_metrics, history, full_settings;
            out_dir=pipe.log_out_dir, pulsemat_N=pipe.pulsemat_N,
            benchmark_metrics=reference_metrics,
        )
    end

    return best_u, best_cost, pulse, signal_E_of_t, d, data, seed_fit_report, reference_metrics
end
