# ============================================================
# ANALYTIC (2N+1)-SEGMENT ARP COMPOSITE PI-PULSE
#
# Generates a fixed (non-optimised) analytic composite adiabatic-rapid-
# passage (ARP) pi-pulse: `2*n_pairs + 1` back-to-back WURST segments (see
# pulses.jl's own `wurst_drive`) all sweeping the SAME bandwidth. Odd
# segments (indices 1, 3, ..., 2*n_pairs+1 -- there are `n_pairs+1` of
# them) run at chirp rate `+k`; even segments (indices 2, 4, ...,
# 2*n_pairs -- `n_pairs` of them) at `+(n_pairs/(n_pairs+1))*k`. Since
# duration = bandwidth/chirp_rate at fixed bandwidth, the even segments
# are correspondingly longer (ratio `(n_pairs+1)/n_pairs`), and the
# budget splits so odd and even segments each take exactly `T_budget/2`.
#
# `n_pairs == 1` is the classic "+k, +k/2, +k" 3-segment refocusing
# sequence (durations 1:2:1, amplitudes 1 : 1/sqrt(2) : 1) used in
# InhomogeneousSpinCavityDynamics.jl's own reference material ("inversion
# and silencing") for cancelling dispersive phase picked up during a
# single fast chirp, at the cost of a longer total interaction time.
# Larger `n_pairs` is the natural subdivision of that template into more,
# finer segments.
#
# NOT system-independent: the segment RATIOS (bandwidth equal across all
# segments, duration ratio 1 : (n_pairs+1)/n_pairs, amplitude ratio
# 1 : sqrt(n_pairs/(n_pairs+1))) are a fixed template, but the ABSOLUTE
# bandwidth/amplitude are derived from the cavity/ensemble in `d`
# (`prepare_derived(CONFIG)`'s own return value), and the ABSOLUTE
# duration from `T_budget`. Amplitude is deliberately NOT tied to
# duration: `amp_odd` is a duration-INDEPENDENT drive-amplitude cap
# (`amp_scale * Omega_target`, with `Omega_target` a Rabi-frequency
# target -- `Omega_max` if given, else a cavity/ensemble-informed default
# off `d.FWHM`; see `CompositePulse`'s own `Omega_power` bound in
# composite_pulse.jl for why a duration-dependent, on-resonance-pi-pulse
# amplitude estimate is the wrong scale here), so that the per-segment
# adiabaticity margin (amp^2/chirp_rate) grows linearly with `T_budget`
# instead of shrinking against it. A pulse built for one cavity/ensemble
# is not expected to also invert a different one at the same chirp
# rate/amplitude.
# ============================================================

"""
    generate_2n1_arp_pi_pulse(d; n_pairs=1, bandwidth_fwhm_mult=5.0,
                              T_budget=nothing, Omega_max=nothing,
                              t_start=0.0, wurst_n=20.0, edge_frac=1e-4,
                              signal_E_of_t=_zero_drive, compute=:cpu,
                              reltol=1e-8, abstol=1e-8)
        -> (PULSE_CONFIG, report)

Builds a `2*n_pairs + 1`-segment composite ARP pi-pulse for the
cavity/ensemble `d` (`prepare_derived(CONFIG)`'s own return value -- the
same argument every other pulse-building/simulation function in this
package takes), runs it forward (no differentiation, CPU by default --
[`run_sim_1st_order_trajectory`](@ref)) from both `:ground` and `:weak`,
and reports how well it actually performed.

`n_pairs = 1` (the default) is the classic "+k, +k/2, +k" 3-segment
refocusing pulse: durations in the ratio 1:2:1, amplitudes
1 : 1/sqrt(2) : 1. Larger `n_pairs` subdivides the same idea into more
segments.

Segment structure (universal, system-independent RATIOS):
- All `2*n_pairs + 1` segments sweep the SAME bandwidth `bw =
  bandwidth_fwhm_mult * d.FWHM` (default multiplier 5.0, matching this
  package's own `duration_100us_gstd_1em06Hz.jld2` reference pulse's
  bandwidth-to-FWHM ratio) -- wide enough to cover the ensemble's
  inhomogeneous spread with margin, narrow enough not to waste
  interaction time on frequencies no spin actually occupies.
- Odd segments (indices 1, 3, ..., `2*n_pairs+1`; `n_pairs+1` of them)
  have duration `dur_odd` and chirp rate `+k = bw/dur_odd`. Even
  segments (indices 2, 4, ..., `2*n_pairs`; `n_pairs` of them) have
  duration `dur_even = dur_odd*(n_pairs+1)/n_pairs` and chirp rate
  `+(n_pairs/(n_pairs+1))*k` -- exactly `+k/2` when `n_pairs == 1`. The
  budget splits as `(n_pairs+1)*dur_odd + n_pairs*dur_even = T_budget`,
  i.e. odd and even segments each take exactly `T_budget/2`, so
  `dur_odd = T_budget / (2*(n_pairs+1))`.
- `T_budget` is the total interaction time of the whole segment train.
  If it is not given it defaults to
  `(d.timespan[2]-d.timespan[1]) * 0.6`, leaving margin in `d`'s own
  simulated window for a leading signal pulse and post-pulse settling --
  override it explicitly if that default doesn't fit your own use.
- Segment amplitudes are in the ratio `1 : sqrt(n_pairs/(n_pairs+1))`
  (odd : even) -- `1 : 1/sqrt(2)` for `n_pairs == 1`. This is the
  relationship that holds the adiabaticity margin
  (amplitude^2/chirp_rate) constant ACROSS segments within one run,
  independent of any system parameter:
  `amp_even^2 / ((n/(n+1))*k) == amp_odd^2 / k`.
- `amp_odd = amp_scale * Omega_target`, where
  `amp_scale = d.kappa_t / (4*d.g_mean*d.sqrt_kappa_e)` is the same
  cavity-informed steady-state input-flux-per-Rabi-rate prefactor
  `CompositePulse`'s own constructor uses (see that function's docstring
  for the derivation), and `Omega_target` is a duration-INDEPENDENT
  Rabi-frequency target:
    * `Omega_max` if you pass it -- interpreted as a Rabi frequency in
      `d`'s own angular-frequency units, NOT a raw drive amplitude: it
      is put through `amp_scale` exactly as the default is, so
      `Omega_max = pi*d.FWHM` reproduces the default `amp_odd`, and
      `CompositePulse` / datagen's own amplitude knob stay comparable.
    * otherwise `pi * d.FWHM`, sizing the drive off the ensemble's FWHM
      (`CompositePulse`'s `Omega_power` bound) rather than off `pi/dur`
      (its `Omega_naive` bound). `CompositePulse`'s own docstring
      documents that the `Omega_naive` on-resonance-pi-pulse scale
      systematically undersizes the drive for a broadband ensemble, and
      it additionally has the wrong DURATION dependence for this use: an
      amplitude that shrinks as `1/dur` makes the per-segment
      adiabaticity margin `amp^2/chirp_rate = amp^2*dur/bw` shrink as
      `1/dur` too, so raising `T_budget` would make the sweep LESS
      adiabatic, not more. A `dur`-independent `amp_odd` instead makes
      that margin `amp_odd^2*dur_odd/bw`, growing linearly with
      `dur_odd` (hence with `T_budget`) as a slower sweep should.
  This is a reasoned starting point, not an exact adiabaticity
  calculation for a CHIRPED (as opposed to resonant) drive -- that is
  exactly why this function measures and reports the actual achieved
  `inversion`/`silencing`/`coherence` rather than assuming the formula
  is exactly right. If `inversion` comes back well short of 1, increase
  `T_budget` (slower sweep at the same `Omega_max`, safer adiabatic
  following) and re-run, or raise `Omega_max` directly if you know your
  hardware/cavity can sustain a larger drive.

Segments are placed back-to-back starting at `t_start` (default `0.0`;
set this to land after any signal pulse of your own, e.g. `t_start =
t0_signal + few*sigma_signal`), with NO gaps and NO shared phase/
frequency continuity between segments (frequency "flies back" to `-bw/2`
at each segment boundary -- amplitude still glues smoothly to 0 there via
each WURST segment's own envelope, which vanishes at its edges, so there
is no envelope discontinuity, only a chirp restart). `signal_E_of_t`
(default [`_zero_drive`](@ref), i.e. none) is added on top of the
composite pulse for the metrics simulation ONLY -- it is NOT included in
the returned `PULSE_CONFIG`, since a signal pulse belongs in your own
PULSE_CONFIG tuple, not baked into this pulse-generation utility.

`compute` (default `:cpu`) is forwarded to
[`run_sim_1st_order_trajectory`](@ref): `:cpu` forces the CPU solver,
`:auto` lets it pick CUDA for a large ensemble if a functional device is
present, `:gpu` forces CUDA (erroring if none is usable). The default is
`:cpu` because this is a lightweight analytic-pulse utility, not a large
optimisation run.

Returns `(PULSE_CONFIG, report)`:
- `PULSE_CONFIG`: a `2*n_pairs+1`-tuple of `:wurst`-kind pulse specs (the
  same shape `pulses.jl`'s [`build_E_of_t`](@ref)/`run_sim_1st_order`
  already expect), ready to concatenate with a signal pulse tuple and
  simulate/save as-is.
- `report`: a NamedTuple
  `(inversion, coherence, silencing, weak_seed_retention, total_duration,
  t_start, t_end, bandwidth, duration_odd, duration_even, amp_odd,
  amp_even, n_pairs, total_segments)` -- the SAME metrics
  `pulse_optimizer2.jl` computes: `inversion`
  ([`_weighted_inversion`](@ref)), the paper silencing factor `silencing`
  ([`_weighted_silencing_factor`](@ref)), its per-slice magnitude
  companion `coherence` ([`_weighted_coherence`](@ref)) and the
  un-clamped `weak_seed_retention` ([`_weak_seed_retention`](@ref)).
  `total_duration` is `t_end - t_start` (the composite pulse's own span),
  NOT `d`'s full simulated window; it equals `T_budget` by construction.
"""
function generate_2n1_arp_pi_pulse(
    d;
    n_pairs::Integer=1,
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
)
    n_pairs >= 1 || error("n_pairs must be a positive integer (>= 1) for a valid (2n+1) sequence, got $n_pairs.")
    n = Int(n_pairs)
    n_seg = 2n + 1

    T_window = d.timespan[2] - d.timespan[1]
    Tb = T_budget === nothing ? 0.6 * T_window : Float64(T_budget)
    Tb > 0 || error(
        T_budget === nothing ?
        "derived T_budget = 0.6*(d.timespan[2]-d.timespan[1]) = $Tb is not positive; check d.timespan." :
        "T_budget must be positive, got $Tb."
    )

    bw = bandwidth_fwhm_mult * d.FWHM
    bw > 0 || error(
        "bandwidth_fwhm_mult*d.FWHM must be positive (got bandwidth_fwhm_mult=$bandwidth_fwhm_mult, d.FWHM=$(d.FWHM))."
    )

    # Time budget: `n+1` odd segments + `n` even segments, with the odd
    # half and the even half each taking exactly `Tb/2`:
    #   (n+1)*dur_odd  = Tb/2                 -> dur_odd = Tb/(2*(n+1))
    #   n*dur_even     = Tb/2, and dur_even/dur_odd = (n+1)/n
    dur_odd = Tb / (2 * (n + 1))
    dur_even = dur_odd * (n + 1) / n

    # `amp_scale`: steady-state cavity input-flux amplitude per unit Rabi
    # rate (the same prefactor `CompositePulse`'s constructor uses).
    # `Omega_max`, if given, is a Rabi-frequency TARGET in `d`'s own
    # angular-frequency units and is put through `amp_scale` exactly as
    # the default `pi*d.FWHM` target is -- so the amplitude knob here,
    # in `CompositePulse`, and in datagen's `bind_3arp` all mean the
    # same thing.
    amp_scale = d.kappa_t / (4 * d.g_mean * d.sqrt_kappa_e)
    Omega_target = Omega_max === nothing ? pi * d.FWHM : Float64(Omega_max)
    Omega_target > 0 || error(
        Omega_max === nothing ?
        "default Rabi target pi*d.FWHM = $Omega_target is not positive; check d.FWHM." :
        "Omega_max must be positive, got $Omega_target."
    )
    amp_odd = amp_scale * Omega_target
    amp_odd > 0 || error(
        "derived amp_odd = amp_scale*Omega_target = $amp_odd is not positive; " *
        "check d.kappa_t/d.g_mean/d.sqrt_kappa_e."
    )
    # `amp_even^2 / ((n/(n+1))*k) == amp_odd^2 / k`: adiabaticity margin
    # held constant across odd/even segments (1 : 1/sqrt(2) for n == 1).
    amp_even = amp_odd * sqrt(n / (n + 1))

    t1_start = Float64(t_start)

    # Per-segment duration/amplitude and exclusive-prefix start times.
    durs = [isodd(i) ? dur_odd : dur_even for i in 1:n_seg]
    amps = [isodd(i) ? amp_odd : amp_even for i in 1:n_seg]
    starts = t1_start .+ cumsum(durs) .- durs
    t_end = t1_start + sum(durs)

    common = (n=wurst_n, omega0=0.0, chirp_sign=1.0, phase0=0.0, edge_frac=edge_frac)
    even_rate_str = n == 1 ? "k/2" : "($n/$(n + 1))*k"
    segments = [
        merge(
            (name="$(n_seg)ARP segment $i ($(isodd(i) ? "+k" : "+" * even_rate_str))",
             kind=:wurst, t_center=starts[i] + durs[i] / 2, duration=durs[i],
             amp=amps[i], bandwidth=bw),
            common,
        )
        for i in 1:n_seg
    ]
    PULSE_CONFIG = Tuple(segments)

    t_end <= d.timespan[2] || @warn(
        "$(n_seg)-segment ARP pulse ends at $(t_end*1e6)us, past d's own simulated window " *
        "end ($(d.timespan[2]*1e6)us) -- run_sim_1st_order_trajectory will silently " *
        "truncate it, and the reported metrics will reflect that truncated pulse, not the " *
        "full composite. Increase d's own Ttotal, or shrink T_budget."
    )

    composite_E_of_t = build_E_of_t(PULSE_CONFIG)
    E_of_t(t) = composite_E_of_t(t) + signal_E_of_t(t)

    T = Float64
    _, _, _, Sz_g = run_sim_1st_order_trajectory(
        E_of_t, d; initial_condition=:ground, compute=compute, reltol=reltol, abstol=abstol,
    )
    inversion = _weighted_inversion(Sz_g[end, :], d.g_b, d.Nj, T)

    _, _, Sp_e, _ = run_sim_1st_order_trajectory(
        E_of_t, d; initial_condition=:weak, compute=compute, reltol=reltol, abstol=abstol,
    )
    coherence = _weighted_coherence(Sp_e[end, :], d.g_b, d.Nj, d.delta_b, T)
    silencing = _weighted_silencing_factor(Sp_e[end, :], d.g_b, d.Nj, d.delta_b, T)
    weak_seed_retention = _weak_seed_retention(Sp_e[end, :], d.g_b, d.Nj, d.delta_b, T)

    report = (
        inversion=inversion, coherence=coherence, silencing=silencing,
        weak_seed_retention=weak_seed_retention,
        total_duration=t_end - t1_start,
        t_start=t1_start, t_end=t_end,
        bandwidth=bw, duration_odd=dur_odd, duration_even=dur_even,
        amp_odd=amp_odd, amp_even=amp_even,
        n_pairs=n, total_segments=n_seg,
    )

    return PULSE_CONFIG, report
end

"""
    generate_3arp_pi_pulse(d; n=20.0, kwargs...) -> (PULSE_CONFIG, report)

DEPRECATED. The classic "+k, +k/2, +k" 3-segment ARP pi-pulse is exactly
[`generate_2n1_arp_pi_pulse`](@ref)`(d; n_pairs=1, kwargs...)`; this alias
forwards to it (renaming the old `n` WURST-exponent keyword to `wurst_n`)
and will be removed in a future revision.

Two behavioural notes for callers migrating off it:
- The returned `report` now uses the generalised field names
  `duration_odd`/`duration_even`/`amp_odd`/`amp_even` (plus `n_pairs`/
  `total_segments`) in place of the old `duration1`/`duration2`/`amp1`/
  `amp2`. For `n_pairs == 1` the values are identical
  (`duration1 == duration_odd`, `amp2 == amp_even`, etc.).
- An explicit `Omega_max` is now interpreted as a Rabi-frequency target
  (put through the cavity `amp_scale`), not as a raw WURST drive
  amplitude. Pass `Omega_max = pi*d.FWHM` to reproduce the old default,
  or `Omega_max = old_value / (d.kappa_t/(4*d.g_mean*d.sqrt_kappa_e))`
  to reproduce a specific old override.
"""
function generate_3arp_pi_pulse(d; n::Real=20.0, kwargs...)
    Base.depwarn(
        "generate_3arp_pi_pulse is deprecated; call " *
        "generate_2n1_arp_pi_pulse(d; n_pairs=1, ...) instead. The `n` (WURST exponent) " *
        "keyword is now `wurst_n`; `report` field names changed (duration1/amp1 -> " *
        "duration_odd/amp_odd, ...); and an explicit `Omega_max` is now a Rabi-frequency " *
        "target rather than a raw drive amplitude.",
        :generate_3arp_pi_pulse,
    )
    return generate_2n1_arp_pi_pulse(d; n_pairs=1, wurst_n=n, kwargs...)
end
