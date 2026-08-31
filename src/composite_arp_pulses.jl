# ============================================================
# ANALYTIC 3-SEGMENT (+k, +k/2, +k) ARP COMPOSITE PI-PULSE
#
# Generates a fixed (non-optimised) analytic composite adiabatic-rapid-
# passage (ARP) pi-pulse: three back-to-back WURST segments (see
# pulses.jl's own wurst_drive) sweeping the SAME bandwidth with chirp
# rates in the ratio k : k/2 : k (hence durations in the ratio 1:2:1,
# since duration = bandwidth/chirp_rate). This is the "+k, +k/2, +k"
# refocusing sequence used in InhomogeneousSpinCavityDynamics.jl's own
# reference material for cancelling dispersive phase picked up during a
# single fast chirp, at the cost of a longer total interaction time.
#
# NOT system-independent: the segment RATIOS (bandwidth equal across all
# three, duration 1:2:1, amplitude 1:1/sqrt(2):1) are a fixed, universal
# template, but the ABSOLUTE bandwidth/amplitude are derived from the
# cavity/ensemble in `d` (prepare_derived(CONFIG)'s own return value),
# and the ABSOLUTE duration from `T_budget`. Amplitude is deliberately
# NOT tied to duration: `amp1` is a duration-INDEPENDENT Rabi-frequency
# cap (`Omega_max`, overridable, else a cavity/ensemble-informed default
# off d.FWHM -- see CompositePulse's own `Omega_power` bound in
# composite_pulse.jl for why a duration-dependent, on-resonance-pi-pulse
# amplitude estimate is the wrong scale here), so that the per-segment
# adiabaticity margin (amp^2/chirp_rate) grows linearly with `T_budget`
# instead of shrinking against it. A pulse built for one cavity/ensemble
# is not expected to also invert a different one at the same chirp
# rate/amplitude.
# ============================================================

"""
    generate_3arp_pi_pulse(d; bandwidth_fwhm_mult=5.0, T_budget=nothing,
                            Omega_max=nothing, t_start=0.0, n=20.0, edge_frac=1e-4,
                            signal_E_of_t=_zero_drive, reltol=1e-8, abstol=1e-8)
        -> (PULSE_CONFIG, report)

Builds a (+k, +k/2, +k) composite ARP pi-pulse for the cavity/ensemble
`d` (`prepare_derived(CONFIG)`'s own return value -- the same argument
every other pulse-building/simulation function in this package takes),
runs it forward (no differentiation, no CUDA --
[`run_sim_1st_order_trajectory`](@ref)) from both `:ground` and
`:equator`, and reports how well it actually performed.

Segment structure (universal, system-independent RATIOS):
- All three segments sweep the SAME bandwidth `bw = bandwidth_fwhm_mult *
  d.FWHM` (default multiplier 5.0, matching this package's own
  `duration_100us_gstd_1em06Hz.jld2` reference pulse's bandwidth-to-FWHM
  ratio) -- wide enough to cover the ensemble's inhomogeneous spread with
  margin, narrow enough not to waste interaction time on frequencies no
  spin actually occupies.
- Segment durations are in the ratio 1:2:1 (`dur1, 2*dur1, dur1`), since
  chirp_rate = bandwidth/duration and segment 2's rate is HALF segment
  1/3's -- at fixed bandwidth, halving the rate doubles the duration.
  `dur1 = T_budget/4` (`T_budget` is the total 3-segment interaction
  time, `4*dur1` since 1+2+1=4); if `T_budget` is not given it defaults
  to `(d.timespan[2]-d.timespan[1]) * 0.6`, leaving margin in `d`'s own
  simulated window for a leading signal pulse and post-pulse settling --
  override it explicitly if that default doesn't fit your own use.
- Segment amplitudes are in the ratio `1 : 1/sqrt(2) : 1`: this is the
  relationship that holds the adiabaticity margin (amplitude^2/chirp_rate)
  constant ACROSS segments within one run, independent of any system
  parameter. `amp1` itself is `Omega_max` (a duration-INDEPENDENT Rabi-
  frequency cap, deliberately NOT a function of `dur1`): if `Omega_max`
  is not given, it defaults to `pi*d.kappa_t/(4*d.g_mean*d.sqrt_kappa_e) *
  d.FWHM`, the same cavity-informed "amp_scale" prefactor
  `CompositePulse`'s own constructor uses (see that function's docstring
  for the full derivation), but sized off the ensemble's FWHM (its
  `Omega_power` bound) rather than off `pi/dur1` (its `Omega_naive`
  bound) -- `CompositePulse`'s own docstring documents that the
  `Omega_naive` on-resonance-pi-pulse scale systematically undersizes the
  drive for a broadband ensemble, and it additionally has the wrong
  DURATION dependence for this use: an amplitude that shrinks as
  `1/dur1` makes the per-segment adiabaticity margin `amp^2/chirp_rate =
  amp^2*dur1/bw` shrink as `1/dur1` too, so raising `T_budget` would make
  the sweep LESS adiabatic, not more. Holding `amp1` constant in `dur1`
  instead makes that margin `Omega_max^2*dur1/bw`, growing linearly with
  `dur1` (hence with `T_budget`) as a slower sweep should. This is a
  reasoned starting point, not an exact adiabaticity calculation for a
  CHIRPED (as opposed to resonant) drive -- that is exactly why this
  function measures and reports the actual achieved `inversion` rather
  than assuming the formula is exactly right; if `inversion` comes back
  well short of 1, increase `T_budget` (slower sweep at the same
  `Omega_max`, safer adiabatic following) and re-run, or raise
  `Omega_max` directly if you know your hardware/cavity can sustain a
  larger drive.

Segments are placed back-to-back starting at `t_start` (default `0.0`;
set this to land after any signal pulse of your own, e.g. `t_start =
t0_signal + few*sigma_signal`), with NO gaps and NO shared phase/
frequency continuity between segments (frequency "flybacks" back to
`-bw/2` at each segment boundary -- amplitude still glues smoothly to 0
there via each WURST segment's own tanh gate, so there is no envelope
discontinuity, only a chirp restart). `signal_E_of_t` (default
[`_zero_drive`](@ref), i.e. none) is added on top of the composite pulse
for the metrics simulation ONLY -- it is NOT included in the returned
`PULSE_CONFIG`, since a signal pulse belongs in your own PULSE_CONFIG
tuple, not baked into this pulse-generation utility.

Returns `(PULSE_CONFIG, report)`:
- `PULSE_CONFIG`: a 3-tuple of `:wurst`-kind pulse specs (the same shape
  `pulses.jl`'s [`build_E_of_t`](@ref)/`run_sim_1st_order` already
  expect), ready to concatenate with a signal pulse tuple and simulate/
  save as-is.
- `report`: `(inversion, coherence, silencing, total_duration, t_start,
  t_end, bandwidth, duration1, duration2, amp1, amp2)` --
  `inversion`/`coherence` are the SAME per-bin metrics
  `pulse_optimizer2.jl` computes ([`_weighted_inversion`](@ref)/
  [`_weighted_coherence`](@ref)); `silencing` is the collective,
  cooperativity-weighted mode-overlap factor
  ([`_weighted_silencing_factor`](@ref)); `total_duration` is
  `t_end - t_start` (the composite pulse's own span), NOT `d`'s full
  simulated window.
"""
function generate_3arp_pi_pulse(
    d;
    bandwidth_fwhm_mult::Real=5.0,
    T_budget::Union{Real,Nothing}=nothing,
    Omega_max::Union{Real,Nothing}=nothing,
    t_start::Real=0.0,
    n::Real=20.0,
    edge_frac::Real=1e-4,
    signal_E_of_t=_zero_drive,
    reltol::Real=1e-8,
    abstol::Real=1e-8,
)
    T_window = d.timespan[2] - d.timespan[1]
    Tb = T_budget === nothing ? 0.6 * T_window : Float64(T_budget)
    Tb > 0 || error("T_budget must be positive, got $Tb.")

    bw = bandwidth_fwhm_mult * d.FWHM
    bw > 0 || error("bandwidth_fwhm_mult*d.FWHM must be positive (got FWHM=$(d.FWHM)).")

    dur1 = Tb / 4
    dur2 = 2 * dur1
    amp1 = Omega_max === nothing ?
        pi * d.kappa_t / (4 * d.g_mean * d.sqrt_kappa_e) * d.FWHM :
        Float64(Omega_max)
    amp1 > 0 || error("Omega_max must be positive, got $amp1.")
    amp2 = amp1 / sqrt(2)

    t1_start = Float64(t_start)
    t1_end = t1_start + dur1
    t2_end = t1_end + dur2
    t3_end = t2_end + dur1

    common = (n=n, omega0=0.0, chirp_sign=1.0, phase0=0.0, edge_frac=edge_frac)
    seg1 = merge((name="3ARP segment 1 (+k)", kind=:wurst,
                  t_center=(t1_start + t1_end) / 2, duration=dur1, amp=amp1, bandwidth=bw), common)
    seg2 = merge((name="3ARP segment 2 (+k/2)", kind=:wurst,
                  t_center=(t1_end + t2_end) / 2, duration=dur2, amp=amp2, bandwidth=bw), common)
    seg3 = merge((name="3ARP segment 3 (+k)", kind=:wurst,
                  t_center=(t2_end + t3_end) / 2, duration=dur1, amp=amp1, bandwidth=bw), common)
    PULSE_CONFIG = (seg1, seg2, seg3)

    t3_end <= d.timespan[2] || @warn(
        "3ARP pulse ends at $(t3_end*1e6)us, past d's own simulated window end " *
        "($(d.timespan[2]*1e6)us) -- run_sim_1st_order_trajectory will silently " *
        "truncate it, and the reported metrics will reflect that truncated pulse, " *
        "not the full composite. Increase d's own Ttotal, or shrink T_budget."
    )

    composite_E_of_t = build_E_of_t(PULSE_CONFIG)
    E_of_t(t) = composite_E_of_t(t) + signal_E_of_t(t)

    T = Float64
    _, _, Sp_g, Sz_g = run_sim_1st_order_trajectory(E_of_t, d; initial_condition=:ground, reltol=reltol, abstol=abstol)
    inversion = _weighted_inversion(Sz_g[end, :], d.g_b, d.Nj, T)

    _, _, Sp_e, Sz_e = run_sim_1st_order_trajectory(E_of_t, d; initial_condition=:equator, reltol=reltol, abstol=abstol)
    coherence = _weighted_coherence(Sp_e[end, :], d.Nj, T)
    silencing = _weighted_silencing_factor(Sp_e[end, :], d.g_b, d.Nj, d.delta_b, T)

    report = (
        inversion=inversion, coherence=coherence, silencing=silencing,
        total_duration=t3_end - t1_start,
        t_start=t1_start, t_end=t3_end,
        bandwidth=bw, duration1=dur1, duration2=dur2, amp1=amp1, amp2=amp2,
    )

    return PULSE_CONFIG, report
end
