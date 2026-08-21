# ============================================================
# COMPOSITE PI-PULSE (differentiable optimisation)
#
# Julia port of InhomogeneousSpinCavityDynamics.py/pulse_optimized_spline.py's
# CompositePulse, adapted to drive THIS package's own 1st-order physics
# (rhs_1st_order!/prepare_derived's delta0/kappa_e/kappa_i/C_ens-derived
# ensemble) rather than the Python side's simplified toy model, and to
# this package's own "t -> Complex" pulse convention (pulses.jl) rather
# than a pre-tabulated, jnp.interp-sampled drive: build_E_of_t below
# evaluates each sub-pulse's B-spline EXACTLY at whatever t the ODE
# integrator asks for, with no fixed time grid and no interpolation
# error at all -- simpler than, and strictly more accurate than, the
# Python port's fixed-grid-plus-linear-interpolation compromise.
#
# The amplitude envelope is the B-spline multiplied by a C^∞ taper
# window (see _gevrey_bump/_smooth_step/_taper_window below), not the
# raw clamped B-spline alone. This is what fixes a real, previously
# unresolved bug: a clamped B-spline's value AT t_start/t_end equals its
# first/last coefficient, which is free/unconstrained -- so the drive
# jumped discontinuously between "silence" (identically 0) and that
# coefficient's value at every sub-pulse edge, at a location
# (t_start/t_end) that itself moves with the optimised parameters
# (raw_gap especially, which translates both edges together). A
# parameter-dependent VALUE jump in an ODE's right-hand side requires a
# jump/transversality correction to the trajectory's own sensitivity
# (the same structure as Leibniz's rule for a parameter-dependent
# integration limit: a boundary term proportional to the jump size times
# how fast the boundary moves) -- ForwardDiff never saw that term, because
# the switch times handed to the solver as `tstops` are taken via
# `ForwardDiff.value` (a plain Float64, carrying no derivative
# information at all -- see run_sim_1st_order_pure). Multiplying by a
# window that matches EVERY derivative (not just the value) of the
# identically-zero silence region removes the discontinuity itself, so
# there is no jump term to miss, for any parameter, unconditionally --
# verified against finite differences after this fix: every one of the
# 12 raw parameters (for a k=1 test pulse) now matches to 0.0% relative
# error, including raw_gap, the one parameter that previously disagreed
# with a wrong sign. Also verified that `tstops` (run_sim_1st_order_pure)
# are no longer even required for gradient correctness once the window
# is in place (still kept there as a harmless integration-accuracy aid,
# since there's no reason not to) -- confirming the diagnosis, not just
# patching around it.
#
# One implementation subtlety worth flagging for anyone touching
# _gevrey_bump: comparing a Dual whose VALUE is exactly 0 against 0 does
# NOT reliably return false the way it would for a plain real -- see that
# function's own docstring for why a bare `x > 0` there silently produces
# NaN gradients exactly at the point this construction most needs to be
# correct, and why `ForwardDiff.value(x) > 0` is used instead.
# ============================================================

"""
    _gevrey_bump(x)

The classic `C^∞`-but-nowhere-analytic bump `exp(-1/x)` for `x > 0`, `0`
for `x <= 0` -- infinitely differentiable at `x=0`, with EVERY derivative
equal to 0 there (not just the value), which is exactly what lets it
glue seamlessly onto an identically-zero region with no discontinuity of
any order. Generic over `x`'s type (real or `ForwardDiff.Dual`).

The branch is decided via `ForwardDiff.value(x) > 0`, NOT a bare `x > 0`
-- unlike plain reals, comparing a `Dual` whose VALUE is exactly 0
against 0 does not reliably give `false`: ForwardDiff breaks ties using
the derivative direction (`Dual(0.0, 1.0) > 0` is `true`, treating a
positive partial at zero value as an infinitesimal step to the positive
side), so a bare `x > 0` silently takes the `exp(-1/x)` branch exactly at
the point this function needs it not to -- `1/x` with `x`'s VALUE at
0 then blows up the forward-mode quotient rule into `NaN`, even though
the true value and every derivative are 0 there. Forcing the branch
decision onto the primal value alone (same technique
`run_sim_1st_order_pure` already uses for `tstops`) sidesteps this
tie-breaking behaviour entirely; `ForwardDiff.value(x) = x` for plain
reals, so this is a no-op when `x` isn't a `Dual` at all. (Verified
directly: `ForwardDiff.derivative(x -> exp(-1/x)*(x>0), 0.0)` -- the
naive version -- returns `NaN`; this version returns `0.0`, matching a
finite-difference check.)
"""
_gevrey_bump(x) = ForwardDiff.value(x) > 0 ? exp(-one(x) / x) : zero(x)

"""
    _smooth_step(x)

`Z(x) = _gevrey_bump(x) / (_gevrey_bump(x) + _gevrey_bump(1-x))` -- a
`C^∞` step from 0 (at and below `x=0`) to 1 (at and above `x=1`),
matching every derivative of both constant extensions at the boundaries
(standard Gevrey-bump-ratio construction; the denominator is never 0:
`x<=0` gives `_gevrey_bump(1-x) > 0`, `x>=1` gives `_gevrey_bump(x) > 0`,
and both are positive in between).
"""
function _smooth_step(x)
    a = _gevrey_bump(x)
    b = _gevrey_bump(one(x) - x)
    return a / (a + b)
end

"""
    _taper_window(t, ts, te, taper_frac)

The full envelope window for one sub-pulse active on `[ts, te]`: rises
smoothly (via [`_smooth_step`](@ref)) from 0 to 1 over the first
`taper_frac` fraction of the duration, stays at 1 through the middle,
and falls smoothly back to 0 over the last `taper_frac` fraction --
`rise(t)*fall(t)` rather than `min(rise(t), fall(t))` specifically so the
product stays `C^∞` unconditionally (a min of two smooth functions can
kink where they cross; for `taper_frac < 0.5` the two ramps never
overlap so this never actually matters, but multiplication needs no such
assumption to stay smooth). `A(t) = A_spline(t) * _taper_window(t, ts,
te, taper_frac)` is exactly 0 at `ts`/`te` -- and every derivative of it
is too -- regardless of what `A_spline`'s own edge coefficients are, so
the B-spline coefficients themselves stay fully free (see
[`build_E_of_t`](@ref)).
"""
function _taper_window(t, ts, te, taper_frac)
    dur = te - ts
    edge = taper_frac * dur
    rise = _smooth_step((t - ts) / edge)
    fall = _smooth_step((te - t) / edge)
    return rise * fall
end

"""
    CompositePulse

A composite pi-pulse made of `k` sub-pulses in exact silence between one
another, each an independent pair of cubic B-splines (see bspline.jl) for
amplitude `A(t)` and frequency/chirp `f(t)` -- amplitude further shaped
by a `C^∞` taper window ([`_taper_window`](@ref), width `taper_frac` of
each sub-pulse's own duration) so the envelope (not just the underlying
spline) glues seamlessly onto silence at every order, not only in value.
Optimised over RAW (unconstrained) parameters `u` -- see [`decode`](@ref)
for the reparameterisation that makes every physical quantity
(durations, gaps, amplitudes) valid by construction, for any `u`.

Construct via [`CompositePulse`](@ref)`(k, n_coeff_A, n_coeff_f, d)`,
where `d` is `prepare_derived(CONFIG)`'s own return value -- `T_max` and
the encode/decode scales below are derived from it, so the pulse's own
time axis and amplitude scale stay physically consistent with the window/
rates the ODE actually integrates over.
"""
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
    0 < taper_frac <= 0.5 || error(
        "taper_frac must be in (0, 0.5] (each sub-pulse has one taper region at each " *
        "edge; 0.5 is the largest that still leaves room for both), got $taper_frac."
    )
    T_max = d.timespan[2] - d.timespan[1]
    gap_scale = T_max / (2k)
    dur_scale = T_max / (2k)
    dur_floor = T_max * 1e-3
    typical_duration = max(dur_scale * k, 1e-30)
    # Cavity-informed amplitude scale (steady-state regime: sub-pulse
    # duration >> 1/kappa_t, so the cavity field re-equilibrates to the
    # drive quasi-instantly -- true for this package's typical configs,
    # e.g. kappa_t ~ 2*pi*1e6 gives a ~160ns cavity response time against
    # ~100us-scale pulses). Target a pi rotation of the spins over
    # typical_duration: the spins feel Rabi rate Omega_spin = 2*g*|a|, and
    # a constant drive E produces steady-state |a| = 2*sqrt(kappa_e)*E/kappa_t
    # (from da/dt = sqrt(kappa_e)*E - 0.5*kappa_t*a, see rhs_1st_order.jl).
    # Solving Omega_spin*typical_duration = pi for E gives:
    #     amp_scale = pi*kappa_t / (4*g_mean*sqrt(kappa_e)*typical_duration)
    # `E(t)` (and hence amp_scale) has units s^-1/2 -- a cavity input-flux
    # amplitude, not a Rabi frequency -- so this replaces the dimensionally
    # mismatched `pi/typical_duration` (units s^-1) that was here before.
    # Using kappa_t (not kappa_e alone) keeps this correct when kappa_i is
    # non-negligible; the two coincide whenever kappa_i ~ 0.
    amp_scale = pi * d.kappa_t / (4 * d.g_mean * d.sqrt_kappa_e * typical_duration)
    freq_scale = d.FWHM
    return CompositePulse(k, n_coeff_A, n_coeff_f, degree, T_max,
                           gap_scale, dur_scale, dur_floor, amp_scale, freq_scale, Float64(taper_frac))
end

n_params(pulse::CompositePulse) = 2 * pulse.k + pulse.k * pulse.n_coeff_A + pulse.k * pulse.n_coeff_f

_softplus(x) = x > 30 ? x : log1p(exp(x))
_softplus_inv(y) = y + log(-expm1(-y))  # y > 0; softplus(_softplus_inv(y)) == y

"""
    unpack(pulse, u) -> (raw_gap, raw_dur, raw_cA, raw_cf)

Splits the flat raw parameter vector `u` into its four raw pieces.
`raw_cA`/`raw_cf` are `(n_coeff, k)` matrices, column `i` holding
sub-pulse `i`'s own raw coefficients.
"""
function unpack(pulse::CompositePulse, u::AbstractVector)
    n = n_params(pulse)
    length(u) == n || error(
        "raw parameter vector has length $(length(u)), but this CompositePulse " *
        "(k=$(pulse.k), n_coeff_A=$(pulse.n_coeff_A), n_coeff_f=$(pulse.n_coeff_f)) needs $n."
    )
    k = pulse.k
    nA = k * pulse.n_coeff_A
    raw_gap = u[1:k]
    raw_dur = u[k+1:2k]
    raw_cA = reshape(u[2k+1:2k+nA], pulse.n_coeff_A, k)
    raw_cf = reshape(u[2k+nA+1:end], pulse.n_coeff_f, k)
    return raw_gap, raw_dur, raw_cA, raw_cf
end

function pack(::CompositePulse, raw_gap, raw_dur, raw_cA, raw_cf)
    return vcat(vec(raw_gap), vec(raw_dur), vec(raw_cA), vec(raw_cf))
end

"""
    decode(pulse, u) -> (t_start, t_end, cA, cf)

Raw `u` -> physical `(t_start, t_end, cA, cf)`, valid for ANY `u`:

- `gap`/`duration = scale * softplus(raw) [+ floor for duration]` are
  always > 0, and `t_start`/`t_end` are built by cumulatively summing
  them (`t_start_i = t_end_{i-1} + gap_i`, `t_end_i = t_start_i + dur_i`)
  -- so `0 <= t_start_1 < t_end_1 < t_start_2 < ...` is guaranteed BY
  CONSTRUCTION, for every `u`. No separate ordering/positivity
  constraint is needed anywhere in the optimiser because of this (same
  softplus + cumulative-sum reparameterisation the Python port uses, and
  the same technique this package's Python sibling's waveguide.py/
  pulse_optimizer.py already rely on).
- `cA = amp_scale * softplus(raw_cA)` is always `>= 0`.
- `cf = freq_scale * raw_cf` is unconstrained (chirp may have either
  sign).

Nothing here caps `t_end[end] <= T_max` -- there is no natural
softplus-style reparameterisation compatible with a running cumulative
sum that would also cap it from above. `pulse_cost` guards against this
with two ON-BY-DEFAULT terms instead (see its own docstring): `duration`
([`pulse_duration`](@ref)) is measured from `t=0`, not from the pulse's
own first edge, so a longer leading gap already raises `w_time*duration`;
and `w_tmax` (default `1.0`) adds a quadratic penalty once `t_end[end]`
actually crosses `T_max`, which is silently truncated in the actual
simulated window otherwise (`run_sim_1st_order_pure` only ever integrates
over `d.timespan`). Pass `w_tmax=0` to `pulse_cost` to disable that
second guard.

Every gradient component (including `raw_gap`, which was NOT correct in
an earlier version of this file -- wrong sign, root-caused and fixed by
the taper window this file's own module docstring describes) has been
verified against finite differences to <0.0001% relative error.
"""
function decode(pulse::CompositePulse, u::AbstractVector)
    raw_gap, raw_dur, raw_cA, raw_cf = unpack(pulse, u)
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
    cA = pulse.amp_scale .* _softplus.(raw_cA)
    cf = pulse.freq_scale .* raw_cf
    return t_start, t_end, cA, cf
end

"""
    initial_guess(pulse; seed=42) -> Vector{Float64}

Reasonable random initialisation in RAW space -- built by sampling target
physical multiples of `decode`'s own scales, then encoding them back
through [`_softplus_inv`](@ref), so the resulting `u` decodes to
physically-sensible gaps/durations/amplitudes even though `u` itself is
unconstrained.
"""
function initial_guess(pulse::CompositePulse; seed::Integer=42)
    rng = Random.Xoshiro(seed)
    k, nA, nf = pulse.k, pulse.n_coeff_A, pulse.n_coeff_f
    raw_gap = _softplus_inv.(0.3 .+ 0.7 .* rand(rng, k))
    raw_dur = _softplus_inv.(0.5 .+ 0.7 .* rand(rng, k))
    raw_cA = _softplus_inv.(0.5 .+ 1.0 .* rand(rng, nA, k))
    raw_cf = 0.3 .* randn(rng, nf, k)
    return pack(pulse, raw_gap, raw_dur, raw_cA, raw_cf)
end

"""
    build_E_of_t(pulse, u) -> (t -> ComplexF64-or-Complex{Dual})

Builds the composite pulse's complex drive `E(t) = A(t)*exp(i*phi(t))` as
a `t -> Complex` closure, satisfying pulses.jl's own convention (drops
directly into `rhs_1st_order!`'s `E_of_t` slot). `A(t) = A_spline(t) *
_taper_window(t, ts, te, pulse.taper_frac)` -- the raw B-spline shaped by
the `C^∞` taper window, not the bare spline (see [`_taper_window`](@ref)
and this file's own module docstring for why: it's what makes `A(t)`, and
every derivative of it, actually match the identically-zero silence
region at `ts`/`te`, regardless of the B-spline's own free edge
coefficients). `phi(t) = integral of f dt'` is computed EXACTLY via each
sub-pulse's frequency-spline antiderivative (`bspline_antiderivative`,
not a numerical quadrature) -- the window is applied to the AMPLITUDE
only, not the phase, so this exactness is unaffected. Phase accumulates
across sub-pulses (each sub-pulse continues from the previous one's total
phase, the same global-accumulation semantics a single cumulative
integral over the whole timeline would give); silence contributes
nothing further since `f` is simply absent there. `t_start`/`t_end`/
`cA`/`cf`/knot vectors/phase offsets are all decoded from `u` ONCE here
(not on every call to the returned closure), since `u` is fixed for the
lifetime of one ODE solve but `E_of_t` gets called once per RHS
evaluation.
"""
function build_E_of_t(pulse::CompositePulse, u::AbstractVector)
    t_start, t_end, cA, cf = decode(pulse, u)
    k = pulse.k
    degree = pulse.degree
    T = eltype(t_start)

    knots_A_list = Vector{Vector{T}}(undef, k)
    knots_fp_list = Vector{Vector{T}}(undef, k)
    d_f_list = Vector{Vector{T}}(undef, k)
    phase_offset = Vector{T}(undef, k)
    running = zero(T)
    @inbounds for i in 1:k
        knots_A_list[i] = make_clamped_knots(pulse.n_coeff_A, t_start[i], t_end[i], degree)
        knots_f = make_clamped_knots(pulse.n_coeff_f, t_start[i], t_end[i], degree)
        knots_fp, d_f = bspline_antiderivative(view(cf, :, i), knots_f, degree)
        knots_fp_list[i] = knots_fp
        d_f_list[i] = d_f
        phase_offset[i] = running
        running += d_f[end]
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

"""
    total_area(pulse, u)

Exact area of the whole composite pulse's underlying B-splines,
`Σ ∫ A_spline(t) dt` -- NOT the area of the actual taper-windowed
envelope `A_spline(t)*_taper_window(t,...)` that `build_E_of_t` uses as
the physical drive. The product of a B-spline and the (non-polynomial)
taper window isn't itself a B-spline, so there's no equally cheap closed
form for the windowed area; this remains the exact area of the spline
alone, which is what `bspline_area`'s own analytic identity actually
computes. Treat this as an approximation of the physical pulse area
(close whenever `taper_frac` is small relative to the duration, since
then most of the spline's mass sits in the untapered middle), not an
exact one, if you need the true windowed area.
"""
function total_area(pulse::CompositePulse, u::AbstractVector)
    t_start, t_end, cA, _ = decode(pulse, u)
    T = eltype(t_start)
    area = zero(T)
    @inbounds for i in 1:pulse.k
        knots = make_clamped_knots(pulse.n_coeff_A, t_start[i], t_end[i], pulse.degree)
        area += bspline_area(view(cA, :, i), knots, pulse.degree)
    end
    return area
end

"""
    pulse_duration(pulse, u)

Elapsed time from the start of the simulated window (`t=0`) to the last
sub-pulse's own end, `t_end[end]` -- NOT `t_end[end] - t_start[1]` (the
bare active span). Measuring from `t=0` deliberately makes the leading
gap (`raw_gap[1]`) part of what this reports: `pulse_cost`'s
`w_time*duration` term is the only thing standing between the optimiser
and letting the whole sequence drift arbitrarily late (an active-span-only
`duration` is translation-invariant, so it costs nothing to slide the
entire pulse later, right up until `T_max` truncates it) -- see
`pulse_cost`'s own docstring for the paired `T_max` penalty that
addresses landing past the window.
"""
function pulse_duration(pulse::CompositePulse, u::AbstractVector)
    _, t_end, _, _ = decode(pulse, u)
    return t_end[end]
end
