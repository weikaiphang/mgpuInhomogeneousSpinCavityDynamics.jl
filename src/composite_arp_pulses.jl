# ============================================================
# ANALYTIC ARP PI-PULSE: SINGLE (F ≈ 0) OR (2N+1)-SEGMENT (F ≈ 1)
#
# Two design targets, selected by `target_silencing`:
#
#   target_silencing = 1  (default)
#     Paper-style (2*n_pairs + 1)-segment composite. Every segment has
#     the SAME drive amplitude; odd segments (n_pairs+1 of them) chirp at
#     +k, even segments (n_pairs of them) at +(n_pairs/(n_pairs+1))*k
#     (exactly +k/2 when n_pairs == 1) and last longer by
#     (n_pairs+1)/n_pairs. With equal amplitude the leading dynamical
#     phase scales as 1/k_j and the signed segment sum
#         sum_j sigma_j / k_j = (n_pairs+1)/k - n_pairs/k_even = 0
#     cancels BOTH the omega^2/k chirp phase and the g^2/k silencing
#     phase, so this targets inversion ≈ 1 AND silencing factor F ≈ 1.
#     n_pairs == 1 is the classic "+k, +k/2, +k" train (durations 1:2:1,
#     amplitudes 1:1:1).
#
#   target_silencing = 0
#     A single WURST ARP spending the whole T_budget on one slow chirp.
#     The g-dependent ARP phase is left intact, so F ≈ 0
#     (|F| ~ exp(-r^2 Q^2 / 2), Q = Omega^2/k). `n_pairs` is ignored.
#
# Neither limit is an exact guarantee -- finite WURST edges, chirp
# restarts, the cavity-filtered envelope A(t), the discrete {g_j, omega_j}
# mesh and non-adiabatic leftovers all leave residuals. The function
# therefore MEASURES the outcome with ONE forward simulation from the
# `:weak` initial condition (Sz = -Nj/2, Sp = eps*Nj/2) and reads every
# reported metric off that single END STATE -- `inversion` from its own
# `Sz`, `silencing` / `coherence` / `weak_seed_retention` from its `Sp`.
# This is exactly `pulse_optimizer2.jl`'s `track = :weak` contract (one
# `:weak` solve, inversion read from it too, an O(eps) bias vs a separate
# `:ground` solve -- see `_assert_track` / `_pulse_cost_grad_threaded`).
# The endpoint-only [`run_sim_1st_order_final`](@ref) is used, so no
# (Nt x M) trajectory is ever materialised. If inversion is well short of
# 1, raise `T_budget` or `Omega_max`.
#
# System scales that drive the pulse come from the SYSTEM_CONFIG values
# `prepare_derived` carries on `d` -- `d.freq_inhomogeneity.FWHM`,
# `d.g_inhomogeneity` (`.g_value` / `.mean`), `d.kappa_e`, `d.kappa_i` --
# i.e. the same field contract `config.jl` validates, NOT the
# M_delta/M_g-discretised `d.FWHM` / `d.g_mean` (see `_arp_system_scales`).
# A pulse built this way is reproducible from a saved run's SYSTEM_CONFIG
# alone, independent of the ensemble-mesh resolution used for the solve:
#
#   bw           = bandwidth_fwhm_mult * FWHM
#   amp          = amp_scale * Omega_target          (same on every segment)
#   amp_scale    = kappa_t / (4 * g_scale * sqrt(kappa_e))
#   Omega_target = Omega_max  (if given)  else  pi * FWHM
#   T_budget     = given, else 0.6 * (d.timespan[2] - d.timespan[1])
#
# `amp` is duration-INDEPENDENT, so the adiabaticity margin
# amp^2 * dur / bw grows with `T_budget`. A pulse built for one
# cavity/ensemble is not expected to invert a different one at the same
# chirp rate / amplitude.
#
# [`generate_2n1_arp_from_jld2`](@ref) wraps all of this: read a package
# `.jld2` run file, reuse its SYSTEM_CONFIG and its (M_delta, M_g), build
# the pulse, run the one `:weak` solve, and write a sibling
# `*__<n_seg>arpcomp<target_silencing>.jld2` holding the new PULSE_CONFIG,
# a fresh SIM_SETTING, and the FINAL STATE ONLY.
# ============================================================

"""
    _arp_system_scales(d) -> (FWHM, kappa_e, kappa_t, g_scale)

Pulse-shaping system scales pulled from the SYSTEM_CONFIG values that
`prepare_derived` stores on `d` (the `config.jl` field contract), NOT
from the mesh-discretised `d.FWHM` / `d.g_mean`:

- `FWHM`    = `d.freq_inhomogeneity.FWHM` (required, positive; both
             `:gaussian` and `:lorentzian` carry it -- see
             `validate_frequency_inhomogeneity`).
- `kappa_e` = `d.kappa_e`;  `kappa_t = d.kappa_e + d.kappa_i`.
- `g_scale` = the coupling distribution's NOMINAL mean coupling:
             `d.g_inhomogeneity.g_value` (`:constant`) or `.mean`
             (`:gaussian`). `:powerlaw_g` / `:user_defined` carry no
             config-level mean, so those fall back to the discretised
             `d.g_mean`.
"""
function _arp_system_scales(d)
    fi = d.freq_inhomogeneity
    hasproperty(fi, :FWHM) || error(
        "_arp_system_scales: d.freq_inhomogeneity has no `FWHM` field " *
        "(expected per validate_frequency_inhomogeneity)."
    )
    FWHM = Float64(fi.FWHM)

    kappa_e = Float64(d.kappa_e)
    kappa_i = Float64(d.kappa_i)
    kappa_t = kappa_e + kappa_i

    gi = d.g_inhomogeneity
    g_scale = if gi.kind === :constant
        Float64(gi.g_value)
    elseif gi.kind === :gaussian
        Float64(gi.mean)
    else
        Float64(d.g_mean)  # :powerlaw_g / :user_defined -- no nominal mean in SYSTEM_CONFIG
    end

    (FWHM > 0 && kappa_e > 0 && kappa_t > 0 && g_scale > 0) || error(
        "_arp_system_scales: non-positive scale (FWHM=$FWHM, kappa_e=$kappa_e, " *
        "kappa_t=$kappa_t, g_scale=$g_scale)."
    )
    return FWHM, kappa_e, kappa_t, g_scale
end

"""
    _arp_segment_plan(target_silencing, n_pairs, Tb)
        -> (durs, labels, dur_odd, dur_even, n_pairs_eff)

Per-segment WURST durations (seconds) and names for the two ARP designs.
Drive amplitude is UNIFORM across segments in BOTH designs, so only the
durations and the chirp-rate labels branch here. Callers guarantee
`n_pairs >= 1`.

- `target_silencing == 0`: one segment spanning the whole budget --
  `durs == [Tb]`, `dur_odd == dur_even == Tb`, `n_pairs_eff == 0`.
- `target_silencing == 1`: `2*n_pairs+1` segments. The odd family
  (`n_pairs+1` segments, chirp `+k`) and the even family (`n_pairs`
  segments, chirp `+(n_pairs/(n_pairs+1))*k`) each fill `Tb/2`, so
  `dur_odd  = Tb / (2*(n_pairs+1))` and
  `dur_even = dur_odd * (n_pairs+1)/n_pairs`.
"""
function _arp_segment_plan(target_silencing::Integer, n_pairs::Integer, Tb::Real)
    Tbf = Float64(Tb)

    if target_silencing == 0
        return [Tbf], ["1ARP (single slow chirp, target F ≈ 0)"], Tbf, Tbf, 0
    end

    n = Int(n_pairs)
    n_seg = 2n + 1

    dur_odd = Tbf / (2 * (n + 1))
    dur_even = dur_odd * (n + 1) / n

    durs = [isodd(i) ? dur_odd : dur_even for i in 1:n_seg]
    even_lbl = n == 1 ? "+k/2" : "+($n/$(n + 1))*k"
    labels = ["$(n_seg)ARP segment $i ($(isodd(i) ? "+k" : even_lbl))" for i in 1:n_seg]

    return durs, labels, dur_odd, dur_even, n
end

"""
    generate_2n1_arp_pi_pulse(d; n_pairs=1, target_silencing=1,
                              bandwidth_fwhm_mult=5.0,
                              T_budget=nothing, Omega_max=nothing,
                              t_start=0.0, wurst_n=20.0, edge_frac=1e-4,
                              signal_E_of_t=_zero_drive, compute=:cpu,
                              reltol=1e-8, abstol=1e-8, keep_final_state=false)
        -> (PULSE_CONFIG, report)   [or (PULSE_CONFIG, report, final_state)]

Builds an analytic ARP π-pulse for the cavity/ensemble `d`
(`prepare_derived(CONFIG)`'s return value), then runs it forward **once**
from the `:weak` initial condition (endpoint only,
[`run_sim_1st_order_final`](@ref), no differentiation, CPU by default)
and reads every physics metric off that single end-state -- matching
`pulse_optimizer2.jl`'s `track = :weak` (one `:weak` solve; `inversion`
taken from that solve's own `Sz`, an O(ε) bias relative to a dedicated
`:ground` solve -- see [`_assert_track`](@ref)).

# Targets (`target_silencing`, only `0` or `1`)

- `1` (default): paper-style `(2*n_pairs+1)`-segment composite,
  **equal amplitude** on every segment, chirp rates
  `+k, +(n/(n+1))k, +k, ...` whose signed `1/k` phase sum vanishes.
  Aims for `inversion ≈ 1` and silencing factor **F ≈ 1**. `n_pairs = 1`
  is `+k, +k/2, +k`, durations `1:2:1`, amplitudes `1:1:1`.
- `0`: a **single** WURST ARP over the whole `T_budget`, leaving the
  g-dependent ARP phase intact. Aims for `inversion ≈ 1` and **F ≈ 0**.
  `n_pairs` is ignored (but must still be `>= 1`).

These are targets, not identities -- read `report.inversion` and
`report.silencing`. If inversion is short of 1, raise `T_budget` (slower
sweep at the same `Omega_max`) or raise `Omega_max`.

# System scales (from SYSTEM_CONFIG, not the mesh)

Pulse arguments are driven by the SYSTEM_CONFIG values `prepare_derived`
carries on `d`, using the same field names `config.jl` validates -- NOT
the `M_delta`/`M_g`-discretised `d.FWHM` / `d.g_mean` (see
[`_arp_system_scales`](@ref)):

- `bw = bandwidth_fwhm_mult * FWHM`, `FWHM = d.freq_inhomogeneity.FWHM`
  (default multiplier 5.0).
- `amp = amp_scale * Omega_target` on **every** segment, with
  `amp_scale = kappa_t / (4*g_scale*sqrt(kappa_e))`,
  `kappa_t = d.kappa_e + d.kappa_i`, and `g_scale` the coupling
  distribution's nominal mean (`d.g_inhomogeneity.g_value` for
  `:constant`, `.mean` for `:gaussian`; else `d.g_mean`).
- `Omega_target = Omega_max` if given (a Rabi frequency in `d`'s
  angular-frequency units, put through `amp_scale`), else `pi * FWHM`.
- `T_budget` defaults to `0.6 * (d.timespan[2]-d.timespan[1])`
  (`d.timespan = (0, CONFIG.Ttotal)`).

# Placement

Segments are back-to-back from `t_start` (default `0.0`), no gaps, no
shared phase/frequency continuity -- the chirp flies back to `-bw/2` at
each boundary while each WURST envelope still vanishes at its own edges
(no amplitude discontinuity). `signal_E_of_t` (default
[`_zero_drive`](@ref)) is added on top only for the metrics solve; it is
NOT part of the returned `PULSE_CONFIG`. `compute` is forwarded to
[`run_sim_1st_order_final`](@ref) (`:cpu` default; `:auto` / `:gpu` to
use CUDA).

# Returns

`(PULSE_CONFIG, report)` -- or `(PULSE_CONFIG, report, final_state)` when
`keep_final_state = true`:
- `PULSE_CONFIG`: a `(2*n_pairs+1)`-tuple (or a 1-tuple when
  `target_silencing == 0`) of `:wurst` specs, ready to concatenate with
  a signal tuple and simulate/save.
- `report`: NamedTuple `(inversion, coherence, silencing,
  weak_seed_retention, target_silencing, total_duration, t_start, t_end,
  bandwidth, duration_odd, duration_even, amp_odd, amp_even, n_pairs,
  total_segments)`. All four physics metrics come from the SINGLE `:weak`
  solve ([`_weighted_inversion`](@ref) on its `Sz`;
  [`_weighted_silencing_factor`](@ref) / [`_weighted_coherence`](@ref) /
  [`_weak_seed_retention`](@ref) on its `Sp`). Amplitude is uniform, so
  `amp_odd == amp_even`; for `target_silencing == 0` also `n_pairs == 0`,
  `total_segments == 1`, `duration_odd == duration_even == T_budget`.
  `t_end` is the pulse's own end (`t_start + T_budget`); `total_duration
  == t_end - t_start == T_budget`.
- `final_state` (only with `keep_final_state = true`): NamedTuple
  `(t_final, a, Sp, Sz)` -- the simulated-window end time
  `d.timespan[2]`, the cavity amplitude there, and the per-bin `S+` /
  `Sz` vectors (length `M`) at that time, from the SAME single `:weak`
  solve (no extra simulation). Used by
  [`generate_2n1_arp_from_jld2`](@ref) to write a final-state-only result.
"""
function generate_2n1_arp_pi_pulse(
    d;
    n_pairs::Integer=1,
    target_silencing::Integer=1,
    bandwidth_fwhm_mult::Real=5.0,
    T_budget::Union{Real,Nothing}=nothing,
    Omega_max::Union{Real,Nothing}=nothing,
    t_start::Real=0.0,
    wurst_n::Real=20.0,
    edge_frac::Real=1e-4,
    signal_E_of_t=_zero_drive,
    compute::Symbol=:cpu,
    reltol::Real=1e-8,
    abstol::Real=1e-8,
    keep_final_state::Bool=false,
)
    target_silencing in (0, 1) || error(
        "target_silencing must be 0 (single ARP, F ≈ 0) or 1 " *
        "((2n+1) composite, F ≈ 1), got $target_silencing."
    )
    n_pairs >= 1 || error("n_pairs must be a positive integer (>= 1), got $n_pairs.")

    # --- total interaction time ---
    T_window = d.timespan[2] - d.timespan[1]
    Tb = T_budget === nothing ? 0.6 * T_window : Float64(T_budget)
    Tb > 0 || error(
        T_budget === nothing ?
        "derived T_budget = 0.6*(d.timespan[2]-d.timespan[1]) = $Tb is not positive; check d.timespan." :
        "T_budget must be positive, got $Tb."
    )

    # --- system scales from SYSTEM_CONFIG (not the M_delta/M_g mesh) ---
    FWHM, kappa_e, kappa_t, g_scale = _arp_system_scales(d)

    bw = bandwidth_fwhm_mult * FWHM
    bw > 0 || error(
        "bandwidth_fwhm_mult*FWHM must be positive (got bandwidth_fwhm_mult=$bandwidth_fwhm_mult, FWHM=$FWHM)."
    )

    amp_scale = kappa_t / (4 * g_scale * sqrt(kappa_e))
    Omega_target = Omega_max === nothing ? pi * FWHM : Float64(Omega_max)
    Omega_target > 0 || error(
        Omega_max === nothing ?
        "default Rabi target pi*FWHM = $Omega_target is not positive; check d.freq_inhomogeneity.FWHM." :
        "Omega_max must be positive, got $Omega_target."
    )
    amp = amp_scale * Omega_target
    amp > 0 || error(
        "derived segment amplitude amp_scale*Omega_target = $amp is not positive; " *
        "check d.kappa_e / d.kappa_i / the coupling mean."
    )

    # --- segment layout: only durations + labels depend on the design ---
    durs, seg_labels, dur_odd, dur_even, n = _arp_segment_plan(target_silencing, n_pairs, Tb)
    n_seg = length(durs)

    t1_start = Float64(t_start)
    starts = t1_start .+ cumsum(durs) .- durs   # exclusive prefix sum
    t_end = t1_start + sum(durs)

    common = (n=wurst_n, omega0=0.0, chirp_sign=1.0, phase0=0.0, edge_frac=edge_frac)
    PULSE_CONFIG = Tuple(
        merge(
            (name=seg_labels[i], kind=:wurst,
             t_center=starts[i] + durs[i] / 2, duration=durs[i],
             amp=amp, bandwidth=bw),
            common,
        )
        for i in 1:n_seg
    )

    t_end <= d.timespan[2] || @warn(
        "$(n_seg)-segment ARP pulse ends at $(t_end*1e6)us, past d's own simulated window " *
        "end ($(d.timespan[2]*1e6)us) -- the solver will silently truncate it and the " *
        "reported metrics will reflect that truncated pulse. Increase d's own Ttotal, or " *
        "shrink T_budget."
    )

    # --- ONE :weak forward solve, endpoint only; every metric off it ---
    composite_E_of_t = build_E_of_t(PULSE_CONFIG)
    E_of_t(t) = composite_E_of_t(t) + signal_E_of_t(t)

    T = Float64
    a_final, Sp_dev, Sz_dev = run_sim_1st_order_final(
        E_of_t, d; initial_condition=:weak, compute=compute, reltol=reltol, abstol=abstol,
    )
    Sp_final = collect(Sp_dev)   # length M; collect() also lands any GPU output on the host
    Sz_final = collect(Sz_dev)
    inversion = _weighted_inversion(Sz_final, d.g_b, d.Nj, T)
    coherence = _weighted_coherence(Sp_final, d.g_b, d.Nj, d.delta_b, T)
    silencing = _weighted_silencing_factor(Sp_final, d.g_b, d.Nj, d.delta_b, T)
    weak_seed_retention = _weak_seed_retention(Sp_final, d.g_b, d.Nj, d.delta_b, T)

    report = (
        inversion=inversion, coherence=coherence, silencing=silencing,
        weak_seed_retention=weak_seed_retention,
        target_silencing=Int(target_silencing),
        total_duration=t_end - t1_start,
        t_start=t1_start, t_end=t_end,
        bandwidth=bw, duration_odd=dur_odd, duration_even=dur_even,
        amp_odd=amp, amp_even=amp,
        n_pairs=n, total_segments=n_seg,
    )

    if keep_final_state
        return PULSE_CONFIG, report,
               (t_final=d.timespan[2], a=a_final, Sp=Sp_final, Sz=Sz_final)
    end
    return PULSE_CONFIG, report
end

"""
    generate_3arp_pi_pulse(d; n=20.0, kwargs...) -> (PULSE_CONFIG, report)

DEPRECATED. The classic "+k, +k/2, +k" 3-segment ARP π-pulse is exactly
[`generate_2n1_arp_pi_pulse`](@ref)`(d; n_pairs=1, target_silencing=1, kwargs...)`.
This alias forwards to it (renaming the old `n` WURST-exponent keyword to
`wurst_n`) and will be removed in a future revision.

Behavioural notes for callers migrating off it:
- Default is now the paper equal-amplitude train (F ≈ 1), not the old
  `1 : 1/sqrt(2)` amplitude split. Pass `target_silencing=0` through
  `kwargs` for a single ARP (F ≈ 0) instead.
- Only ONE `:weak` forward solve is run now; `report.inversion` is read
  from it (O(ε) bias vs a `:ground` solve), matching
  `pulse_optimizer2.jl`'s `track = :weak`.
- `report` uses `duration_odd`/`duration_even`/`amp_odd`/`amp_even`
  (plus `n_pairs`/`total_segments`/`target_silencing`) in place of
  `duration1`/`duration2`/`amp1`/`amp2`.
- An explicit `Omega_max` is a Rabi-frequency target (put through the
  cavity `amp_scale`), not a raw WURST drive amplitude. Pass
  `Omega_max = pi*d.freq_inhomogeneity.FWHM` to reproduce the default.
"""
function generate_3arp_pi_pulse(d; n::Real=20.0, kwargs...)
    Base.depwarn(
        "generate_3arp_pi_pulse is deprecated; call " *
        "generate_2n1_arp_pi_pulse(d; n_pairs=1, target_silencing=1, ...) instead. " *
        "The `n` (WURST exponent) keyword is now `wurst_n`; `report` field names " *
        "changed (duration1/amp1 -> duration_odd/amp_odd, ...); default amplitudes " *
        "are now equal across segments (paper F ≈ 1); metrics come from a single " *
        "`:weak` solve; and an explicit `Omega_max` is a Rabi-frequency target " *
        "rather than a raw drive amplitude.",
        :generate_3arp_pi_pulse,
    )
    return generate_2n1_arp_pi_pulse(d; n_pairs=1, target_silencing=1, wurst_n=n, kwargs...)
end

# ============================================================
# JLD2 WRAPPER
# ============================================================

"""
    _load_run_payload(path) -> NamedTuple

The `data` payload of a package `.jld2` run file: the top-level `"data"`
NamedTuple if present (sweep / optrunlog / datagen layout), otherwise a
flat file whose keys include `"SYSTEM_CONFIG"`, reassembled into a
NamedTuple.
"""
function _load_run_payload(path::AbstractString)
    isfile(path) || error("generate_2n1_arp_from_jld2: no such file: $path")
    raw = JLD2.load(path)
    if haskey(raw, "data")
        return raw["data"]
    elseif haskey(raw, "SYSTEM_CONFIG")
        return NamedTuple(Symbol(k) => v for (k, v) in raw)
    else
        error(
            "generate_2n1_arp_from_jld2: $path has neither a top-level `data` NamedTuple " *
            "nor a `SYSTEM_CONFIG` key (keys: $(collect(keys(raw)))). Point at a package run file."
        )
    end
end

"""
    _arp_out_path(src, n_seg, target_silencing; out_dir=nothing) -> String

`<src-stem>__<n_seg>arpcomp<target_silencing>.jld2` -- next to `src`, or
in `out_dir` (using only `src`'s basename). Pure: creates no directories.
"""
function _arp_out_path(src::AbstractString, n_seg::Integer, target_silencing::Integer;
                       out_dir::Union{AbstractString,Nothing}=nothing)
    endswith(src, ".jld2") || error("generate_2n1_arp_from_jld2: source must be a .jld2 path, got $src.")
    stem = src[1:end - length(".jld2")]
    name = "$(n_seg)arpcomp$(target_silencing).jld2"
    return out_dir === nothing ? "$(stem)__$(name)" : joinpath(out_dir, "$(basename(stem))__$(name)")
end

"""
    generate_2n1_arp_from_jld2(jld2_path; target_silencing=1, n_pairs=1,
                               M_delta=nothing, M_g=nothing,
                               out_dir=nothing, compute=:cpu,
                               Ttotal=nothing, Nt_save=nothing,
                               bandwidth_fwhm_mult=5.0, T_budget=nothing,
                               Omega_max=nothing, t_start=0.0, wurst_n=20.0,
                               edge_frac=1e-4, reltol=1e-8, abstol=1e-8)
        -> outpath::String

End-to-end wrapper: read a package `.jld2` run file, take its
`SYSTEM_CONFIG` verbatim, build the analytic ARP π-pulse via
[`generate_2n1_arp_pi_pulse`](@ref), run **one** forward simulation on
the `:weak` track, and write a new `.jld2` holding the new
`PULSE_CONFIG`, a fresh `SIM_SETTING`, and the FINAL STATE ONLY.

Bootstrapping:
- `SYSTEM_CONFIG` -- lifted unchanged from `jld2_path` (`config.jl` field
  contract; validated via `validate_config`).
- `M_delta`, `M_g` -- taken straight from the source `SIM_SETTING`
  (the `M_delta` / `M_g` keywords override). No mesh sizing is done here.
- `Ttotal` -- the `Ttotal` keyword if given, else the source
  `SIM_SETTING.Ttotal`. `Nt_save` similarly (default `5001`; it does not
  affect the final-state result). `simulation_order = :order1`.
- All pulse-shaping scalars (`bw`, `amp`, default `T_budget`) come from
  the SYSTEM_CONFIG values on `d` -- see [`_arp_system_scales`](@ref).
- `bandwidth_fwhm_mult` / `T_budget` / `Omega_max` / `t_start` /
  `wurst_n` / `edge_frac` / `reltol` / `abstol` / `compute` pass straight
  through to [`generate_2n1_arp_pi_pulse`](@ref).

Output file: `<source-stem>__<n_seg>arpcomp<target_silencing>.jld2` next
to the source (or in `out_dir`), where `n_seg = 2*n_pairs+1` for
`target_silencing = 1` and `1` for `target_silencing = 0`. Its top-level
`data` NamedTuple carries `SIM_SETTING`, `SYSTEM_CONFIG`, `PULSE_CONFIG`,
`initial_condition = :weak`, the final-state fields `t_final` / `a_final`
/ `Sp_final` / `Sz_final` (per-bin, length `M`) / `Sigma_p_final` /
`Sigma_z_final`, the metrics `inversion` / `coherence` / `silencing` /
`weak_seed_retention` (+ the full `arp_report`), and provenance
(`target_silencing`, `n_pairs`, `total_segments`, `M_total`,
`source_jld2`). Returns the output path.
"""
function generate_2n1_arp_from_jld2(
    jld2_path::AbstractString;
    target_silencing::Integer=1,
    n_pairs::Integer=1,
    M_delta::Union{Integer,Nothing}=nothing,
    M_g::Union{Integer,Nothing}=nothing,
    out_dir::Union{AbstractString,Nothing}=nothing,
    compute::Symbol=:cpu,
    Ttotal::Union{Real,Nothing}=nothing,
    Nt_save::Union{Integer,Nothing}=nothing,
    bandwidth_fwhm_mult::Real=5.0,
    T_budget::Union{Real,Nothing}=nothing,
    Omega_max::Union{Real,Nothing}=nothing,
    t_start::Real=0.0,
    wurst_n::Real=20.0,
    edge_frac::Real=1e-4,
    reltol::Real=1e-8,
    abstol::Real=1e-8,
)
    payload = _load_run_payload(jld2_path)
    hasproperty(payload, :SYSTEM_CONFIG) || error(
        "generate_2n1_arp_from_jld2: $jld2_path payload has no SYSTEM_CONFIG."
    )
    sys = payload.SYSTEM_CONFIG
    src_sim = hasproperty(payload, :SIM_SETTING) ? payload.SIM_SETTING : nothing

    _from_src(key, override, default) =
        override !== nothing ? override :
        (src_sim !== nothing && hasproperty(src_sim, key)) ? getproperty(src_sim, key) :
        default === nothing ?
            error("generate_2n1_arp_from_jld2: $jld2_path has no SIM_SETTING.$key; pass $key=...") :
            default

    Ttot = Float64(_from_src(:Ttotal, Ttotal, nothing))
    Ttot > 0 || error("generate_2n1_arp_from_jld2: Ttotal must be positive, got $Ttot.")
    ntsave = Int(_from_src(:Nt_save, Nt_save, 5001))
    Md = Int(_from_src(:M_delta, M_delta, nothing))
    Mg = Int(_from_src(:M_g, M_g, nothing))
    (Md > 0 && Mg > 0) || error(
        "generate_2n1_arp_from_jld2: M_delta=$Md and M_g=$Mg must both be positive."
    )

    # --- fresh SIM_SETTING (config.jl contract) ---
    n_seg = target_silencing == 0 ? 1 : (2 * Int(n_pairs) + 1)
    outpath = _arp_out_path(String(jld2_path), n_seg, Int(target_silencing); out_dir=out_dir)
    SIM_SETTING = (
        simulation_order = :order1,
        M_delta = Md,
        M_g = Mg,
        initial_condition = :weak,
        Ttotal = Ttot,
        Nt_save = ntsave,
        reltol = reltol,
        abstol = abstol,
        saved_file_name = outpath,
    )

    CONFIG = build_full_config(SIM_SETTING, sys)
    validate_config(CONFIG)
    d = prepare_derived(CONFIG)
    d.M == Md * Mg || error(
        "generate_2n1_arp_from_jld2: derived M=$(d.M) != M_delta*M_g=$(Md * Mg)."
    )

    # --- analytic ARP pulse + ONE :weak forward solve (final state kept) ---
    PULSE_CONFIG, report, fs = generate_2n1_arp_pi_pulse(
        d;
        n_pairs=n_pairs, target_silencing=target_silencing,
        bandwidth_fwhm_mult=bandwidth_fwhm_mult, T_budget=T_budget, Omega_max=Omega_max,
        t_start=t_start, wurst_n=wurst_n, edge_frac=edge_frac,
        compute=compute, reltol=reltol, abstol=abstol,
        keep_final_state=true,
    )
    validate_pulse_config(PULSE_CONFIG)
    report.total_segments == n_seg || error(
        "generate_2n1_arp_from_jld2: n_seg mismatch (planned $n_seg, got $(report.total_segments))."
    )

    data = (
        SIM_SETTING = SIM_SETTING,
        SYSTEM_CONFIG = sys,
        PULSE_CONFIG = PULSE_CONFIG,
        initial_condition = :weak,

        # forward-simulation results: FINAL STATE ONLY (one :weak solve)
        t_final = fs.t_final,
        a_final = fs.a,
        Sp_final = fs.Sp,
        Sz_final = fs.Sz,
        Sigma_p_final = sum(fs.Sp),
        Sigma_z_final = sum(fs.Sz),

        # metrics from that same end-state (paper track = :weak)
        inversion = report.inversion,
        coherence = report.coherence,
        silencing = report.silencing,
        weak_seed_retention = report.weak_seed_retention,
        arp_report = report,

        # provenance
        target_silencing = Int(target_silencing),
        n_pairs = report.n_pairs,
        total_segments = report.total_segments,
        M_total = d.M,
        source_jld2 = abspath(String(jld2_path)),
    )

    dir = dirname(outpath)
    isempty(dir) || mkpath(dir)
    JLD2.jldsave(outpath; data = data)
    return outpath
end
