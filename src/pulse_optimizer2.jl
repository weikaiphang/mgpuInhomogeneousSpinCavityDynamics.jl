# ============================================================
# DIFFERENTIABLE PULSE OPTIMISATION (dual-trajectory)
#
# Julia port of InhomogeneousSpinCavityDynamics.py/pulse_optimized_spline.py,
# extended with a DUAL-TRAJECTORY cost (see pulse_metrics/pulse_cost below):
# where pulse_optimizer.jl scores a candidate pulse from a single :ground
# solve, this file runs the SAME pulse u from two independent initial
# conditions -- :ground (inversion) and :equator (collective silencing
# factor |F|, a cooperativity-weighted mode-overlap integral, NOT a
# per-bin coherence average -- see _weighted_silencing_factor) -- and
# optimises both simultaneously via a target_F-driven penalty (target_F=1
# for RASE-style revival, target_F=0 for ROSE-style silencing), plus an
# L2 power penalty on the decoded amplitude coefficients (w_power). This
# is the finalised, actively-maintained pulse-optimisation entry point
# for this package (see InhomogeneousSpinCavityDynamics.jl's own include
# list); pulse_optimizer.jl predates the dual-trajectory cost and is no
# longer included in the module.
#
# Ports the algorithmic structure (B-spline composite pulse, Adam descent
# with early stopping, basin-hopping outer loop) while driving THIS
# package's own real physics (rhs_1st_order!, prepare_derived) rather
# than the Python side's simplified toy model. Gradients are computed via
# ForwardDiff (forward-mode) -- the only AD backend available in this
# package's dependency tree, and one well-suited here: rhs_1st_order!'s
# ComplexF64 state differentiates natively and correctly through
# ForwardDiff.Dual (verified against finite differences; ordinary Julia
# generic-complex-arithmetic code has none of diffrax's own documented
# "complex dtype backward-mode may not yet produce correct results"
# caveat, since forward-mode propagates dual numbers through ordinary
# arithmetic rather than needing a custom reverse-mode/checkpointing rule
# for complex state).
#
# Cost is expensive: each epoch differentiates through TWO full ODE
# solves (one per initial condition) via a single combined
# ForwardDiff.gradient call. On the M=20000 real ensemble
# (data/data_1st_order/duration_100us_gstd_1em06Hz.jld2), a 15-epoch,
# single-hop run took ~6 hours and produced a genuinely good result under
# this file's PREDECESSOR cost (a per-bin |Sp| coherence average, not
# the silencing factor below): inversion 0.91, coherence 0.93, up from
# 0.47/0.88 at init -- verified end-to-end, not just gradient-checked on
# the small toy config. That result is evidence the dual-trajectory
# (:ground + :equator) approach itself works end-to-end; the silencing
# factor replacing coherence has been gradient-checked on the small toy
# config only (see _weighted_silencing_factor's own docstring for why a
# naive per-bin phasor version was rejected) -- a full real-ensemble run
# under the new cost has not been repeated yet.
# ============================================================

"""
    build_u0_1st_order_cpu(M, Nj, ::Type{T}, initial_condition=:ground) -> Vector{Complex{T}}

Plain-`Vector` (not `CuArray`), element-type-generic analogue of
`build_u0_gpu_1st_order` -- `T` should be `eltype` of whatever pulse
parameter vector `u` is being solved/differentiated with (`Float64` for
an ordinary forward solve, `ForwardDiff.Dual` when differentiating),
so the initial state promotes correctly alongside the ODE's `E_of_t`-
driven trajectory.
"""
function build_u0_1st_order_cpu(M::Integer, Nj::AbstractVector, ::Type{T},
    initial_condition::Symbol=:ground) where {T}
    u0 = zeros(Complex{T}, state_length_1st_order(M))
    sp = IDX1_Sp_start:idx1_Sz_start(M)-1
    sz = idx1_Sz_start(M):state_length_1st_order(M)
    if initial_condition == :ground
        # South pole: Sz = -Nj/2, Sp = 0
        u0[sz] .= .-Nj ./ 2
    elseif initial_condition == :inverted
        # North pole: Sz = +Nj/2, Sp = 0
        u0[sz] .= Nj ./ 2
    elseif initial_condition == :equator
        # +x equator: Sz = 0, Sp = Nj/2 (real). Same Bloch radius Nj/2 as
        # :ground / :inverted, so a π_x pulse can invert z AND leave +x
        # on the equator (the dual-trajectory π-pulse cost).
        u0[sp] .= Nj ./ 2
    elseif initial_condition == :custom
        # already zero
    else
        error("Unknown initial_condition = $(initial_condition). Use :ground, :inverted, :equator, or :custom.")
    end
    return u0
end

"""
    _zero_drive(t) -> ComplexF64

The always-off signal drive: identically zero at every `t`, regardless
of type of `t` (real or `ForwardDiff.Dual`, since `t` itself is the ODE
integrator's own time variable, never a differentiated quantity). Default
`signal_E_of_t` for [`run_sim_1st_order_pure`](@ref) when no fixed signal
pulse is being layered under the (optimised) control pulse.
"""
_zero_drive(t) = zero(ComplexF64)

struct PulseSolveFailed <: Exception
    retcode
end

function Base.showerror(io::IO, e::PulseSolveFailed)
    print(io, "1st-order pulse ODE solve failed (retcode=$(e.retcode))")
end

function _successful_solve(sol)
    rc = sol.retcode
    name = rc isa Symbol ? rc : Symbol(string(rc))
    return name === :Success || name === :Terminated
end

"""
    run_sim_1st_order_pure(u, pulse::CompositePulse, d; signal_E_of_t=_zero_drive, initial_condition=:ground, alg=Tsit5(), reltol=1e-8, abstol=1e-8) -> (a, Sp, Sz, Nj)

Pure, CPU, ForwardDiff-differentiable analogue of `run_sim_1st_order`:
builds the CONTROL drive from the composite pulse `u` ([`build_E_of_t`](@ref)),
adds a FIXED background `signal_E_of_t(t)` on top of it (any `t -> Complex`
callable -- e.g. [`build_signal_E_of_t`](@ref)'s output when driving this
from a loaded .jld2 run; defaults to [`_zero_drive`](@ref), i.e. no
signal at all), and integrates the SAME `rhs_1st_order!` the GPU
production solver uses (completely unmodified) over `d.timespan`,
returning the final `(a, Sp, Sz)` state at `t1` plus `d.Nj` (handed back
so callers can build population-weighted ensemble metrics without
re-deriving the ensemble). `signal_E_of_t` is a plain closure captured by
value here, NOT part of `u` -- it is therefore structurally impossible
for `ForwardDiff.gradient(uu -> ..., u)` to ever differentiate through it,
which is what guarantees an optimisation built on top of this only ever
touches the control pulse (see `optimise_control_pulse_from_jld2` in
jld2_pulse_loader.jl). No callback, no file I/O, no CUDA -- `d` is
`prepare_derived(CONFIG)`'s own return value, exactly what
`run_sim_1st_order` itself builds internally, just constructed once by
the caller and reused across many calls (e.g. across an outer
optimisation loop) instead of rebuilt from `CONFIG` on every call.
"""
function run_sim_1st_order_pure(
    u::AbstractVector,
    pulse::CompositePulse,
    d;
    signal_E_of_t = _zero_drive,
    initial_condition::Symbol=:ground,
    alg=Tsit5(),
    reltol=1e-8,
    abstol=1e-8,
)
    T = eltype(u)
    M = d.M
    control_E_of_t = build_E_of_t(pulse, u)
    E_of_t(t) = control_E_of_t(t) + signal_E_of_t(t)
    u0 = build_u0_1st_order_cpu(M, d.Nj, T, initial_condition)
    p = (d.delta0, d.kappa_e, d.kappa_i, d.delta_b, d.g_b, M, E_of_t)

    # Historical note (the underlying issue is now fixed a different way,
    # see below, but this is why `tstops` are still here): each sub-pulse
    # used to have a HARD (exact-silence) edge at its own t_start/t_end --
    # a genuine discontinuity in the RHS at a time that MOVES as the
    # differentiated parameters (raw_gap/raw_dur) move it, which pinning
    # every known edge as an explicit `tstop` (standard SciML practice for
    # parameter-dependent switching times) mostly, but not entirely, fixed
    # -- raw_gap (which translates BOTH t_start and t_end together) still
    # disagreed with a finite-difference cross-check even with `tstops`
    # set, because `tstops` only forces the SOLVER to land on the switch;
    # it does nothing about the missing jump/transversality term a
    # parameter-dependent VALUE discontinuity requires in the trajectory's
    # OWN sensitivity (see CompositePulse's module docstring in
    # composite_pulse.jl for the full explanation). That's fixed properly
    # now: CompositePulse.build_E_of_t multiplies the amplitude by a C^∞
    # taper window that matches every derivative of the identically-zero
    # silence region at the edges, removing the discontinuity itself
    # (verified: raw_gap now agrees with finite differences to <0.0001%,
    # and in fact `tstops` are no longer even required for correctness at
    # all once the window is in place -- also verified directly). They're
    # kept here anyway as a harmless hint for the adaptive stepper near
    # each taper region, not because correctness depends on them anymore.
    t_start, t_end, _, _, _ = decode(pulse, u)
    tstops = ForwardDiff.value.(vcat(t_start, t_end))

    prob = ODEProblem(rhs_1st_order!, u0, d.timespan, p)
    sol = solve(prob, alg; reltol=reltol, abstol=abstol, save_everystep=false, save_start=false, tstops=tstops)
    _successful_solve(sol) || throw(PulseSolveFailed(sol.retcode))
    a, Sp, Sz = unpack_state_1st_order_u(sol.u[end], M)
    return a, collect(Sp), collect(Sz), d.Nj
end

function _forbid_initial_condition(kwargs)
    :initial_condition in keys(kwargs) && error(
        "dual-trajectory cost fixes initial conditions to :ground (inversion) and " *
        ":equator (silencing). Do not pass initial_condition into pulse_cost / " *
        "pulse_metrics / optimise_composite_pulse."
    )
    return nothing
end

function _weighted_inversion(Sz, Nj, ::Type{T}) where {T}
    weight = Nj ./ sum(Nj)
    Sz_fraction = real.(Sz) ./ (Nj ./ 2 .+ 1e-30)
    return sum(weight .* clamp.((Sz_fraction .+ 1) ./ 2, zero(T), one(T)))
end

"""
    _weighted_coherence(Sp, Nj, ::Type{T}) -> T

Per-bin `Nj`-weighted mean of `|Sp_j|/(Nj_j/2)`, in `[0, 1]`. NOT used by
[`pulse_cost`](@ref)/[`pulse_metrics`](@ref) (superseded there by
[`_weighted_silencing_factor`](@ref)'s collective mode-overlap `|F|`) --
kept only so [`save_optimisation_run_log`](@ref) can log this simpler,
per-bin-average metric alongside the silencing factor actually being
optimised, for comparison. Same `1e-30` epsilon pattern as
`_weighted_inversion`'s denominator (`abs(Dual(0))` is `0/0` otherwise).
"""
function _weighted_coherence(Sp, Nj, ::Type{T}) where {T}
    weight = Nj ./ sum(Nj)
    Sp_abs = sqrt.(abs2.(Sp) .+ 1e-30)
    Sp_fraction = Sp_abs ./ (Nj ./ 2 .+ 1e-30)
    return sum(weight .* clamp.(Sp_fraction, zero(T), one(T)))
end

"""
    _weighted_silencing_factor(Sp, g_b, Nj, ::Type{T}) -> T

Collective mode-overlap silencing factor `|F| ∈ [0, 1]` from the
equatorial track's end-state `Sp = <S^+>`:

    F = (Σ_j Nj_j g_j² Sp_j) / (Σ_j Nj_j g_j² (Nj_j/2))

The denominator is the maximum magnitude the numerator could reach if
every bin retained full coherence (`|Sp_j| = Nj_j/2`, the standard
Dicke-state bound this file already assumes elsewhere -- see
[`_weighted_inversion`](@ref)'s `Sz_fraction` clamp) AND every bin's
phase were perfectly aligned; `|Σ w_j Sp_j| <= Σ w_j |Sp_j| <= Σ w_j
(Nj_j/2)` by the triangle inequality, so `|F| <= 1` always, matching a
`|Sp|`-based coherence metric's own `[0, 1]` scale. `Nj_j g_j²`
weighting (not plain `Nj_j`) is the standard cooperativity-style
weighting (`g2_avg` in ensemble.jl) so bins that actually couple
strongly to the cavity mode dominate the collective sum.

The denominator is a FIXED (`u`-independent) constant built only from
`g_b`/`Nj` -- deliberately not `Σ_j Nj_j g_j² |Sp_j|` or any other
per-bin normalisation. Dividing each bin's `Sp_j` by ITS OWN `|Sp_j|`
(extracting a per-bin unit phasor) was considered and rejected: `d(|z|^2)
= 2*Re(z^* dz)`, which is exactly `0` AT `z=0` regardless of `dz`, so
`Sp_j / sqrt(abs2(Sp_j)+eps)`'s derivative scales like `1/sqrt(eps) ~
1e15` for any bin whose `Sp_j` passes near `0` -- routine across a wide
inhomogeneous ensemble (many bins, wide detuning spread), not a rare
edge case, and it would inject an enormously amplified, physically
meaningless gradient component with no `NaN`/`Inf` to flag it (silent
corruption, not a clean failure). Summing the raw (non-unit-normalised)
`Sp_j` first and normalising the WHOLE sum by a constant afterward has
no such singularity anywhere: the only division is by a fixed number
that never depends on `u`.
"""
function _weighted_silencing_factor(Sp::AbstractVector, g_b::AbstractVector, Nj::AbstractVector, ::Type{T}) where {T}
    weight = Nj .* abs2.(g_b)
    max_coherent_sum = sum(weight .* Nj) / 2
    F_complex = sum(weight .* Sp) / (max_coherent_sum + 1e-30)
    abs_F = sqrt(abs2(F_complex) + 1e-30)
    return clamp(abs_F, zero(T), one(T))
end

"""
    pulse_metrics(u, pulse, d; kwargs...) -> (inversion, silencing, coherence)

Dual-trajectory metrics used by [`pulse_cost`](@ref). `inversion` and
`silencing` are both in `[0, 1]`, higher = better, and are NOT two
coordinates of one Bloch vector:

- `inversion`: from `:ground` (`Sz = -Nj/2`, `Sp = 0`). `Nj`-weighted mean
  of `real(Sz)/(Nj/2)` mapped from `[-1, 1]` to `[0, 1]`. A π pulse scores
  near 1.
- `silencing`: from `:equator` (`Sz = 0`, `Sp = Nj/2` along +x). Collective,
  cooperativity-weighted mode-overlap factor `|F|` (see
  [`_weighted_silencing_factor`](@ref)) -- how much of the ensemble's
  equatorial coherence survives IN PHASE with the cavity mode, not a
  per-bin magnitude average. `F=1`: every bin coherent and phase-aligned
  (RASE-style revival). `F=0`: fully decohered OR fully destructively
  interfering (ROSE-style silencing) -- these are NOT distinguishable
  from `|F|` alone, by design (both `target_F=0` and this metric only
  ever measure the SIZE of the collective coherent sum).
- `coherence`: the OLDER, simpler per-bin `Nj`-weighted mean of
  `|Sp|/(Nj/2)` (see [`_weighted_coherence`](@ref)), from the SAME
  `:equator` solve as `silencing` (no extra ODE solve). DIAGNOSTIC ONLY --
  never fed into [`pulse_cost`](@ref)'s optimised objective, recorded
  purely so callers/logs can compare it against the collective `|F|`
  actually being optimised.

`Nj`-weighting for `inversion` (`Nj ./ sum(Nj)`, rather than the 1D
`p_delta` this package's `prepare_derived` also returns) is used so this
generalises correctly whether `M_g == 1` or not. Do not pass
`initial_condition` — both ICs are fixed here. Solver kwargs
(`signal_E_of_t`, `reltol`, ...) are forwarded to both solves.
"""
function pulse_metrics(u::AbstractVector, pulse::CompositePulse, d; kwargs...)
    _forbid_initial_condition(kwargs)
    T = eltype(u)
    _, _, Sz, Nj = run_sim_1st_order_pure(u, pulse, d; kwargs..., initial_condition=:ground)
    inversion = _weighted_inversion(Sz, Nj, T)
    _, Sp, _, Nj_eq = run_sim_1st_order_pure(u, pulse, d; kwargs..., initial_condition=:equator)
    silencing = _weighted_silencing_factor(Sp, d.g_b, Nj_eq, T)
    coherence = _weighted_coherence(Sp, Nj_eq, T)
    return inversion, silencing, coherence
end

"""
    pulse_cost(u, pulse, d; w_inv=1.0, w_sil=0.7, target_F=1.0, w_time=0.15, w_power=0.05, w_tmax=1.0, kwargs...)
        -> (cost, inversion, silencing, duration, coherence)

Scalar cost to be minimised. Inversion and silencing are scored on
**two independent ODE solves** of the same pulse `u` (see
[`pulse_metrics`](@ref)): `:ground` → inversion, `:equator` → collective
silencing factor `|F|`. `target_F` picks which cavity-QED protocol this
pulse is being optimised for: `target_F=1.0` (default) rewards
preserving collective coherence (RASE-style revival); `target_F=0.0`
rewards destroying it (ROSE-style echo silencing). Do not set `w_inv`/
`w_sil` to 0 unless you want only one of the two objectives.

    J = -w_inv*inversion + w_sil*(silencing - target_F)²
        + w_time*(duration/T_max) + w_tmax*max(t_end[end]-T_max, 0)²/T_max²
        + w_power*mean(|cA/amp_scale|²)

`w_power` is an L2 penalty on the decoded, scale-normalised amplitude
coefficients (i.e. `softplus.(raw_cA)`). Failed solves return `Inf`.
Do not pass `initial_condition`.

`coherence` is the OLDER, simpler per-bin `Nj`-weighted mean of
`|Sp|/(Nj/2)` (see [`_weighted_coherence`](@ref)/[`pulse_metrics`](@ref)),
computed from the SAME `:equator` solve as `silencing` whenever
`w_sil > 0` (no extra ODE solve); it stays `zero(T)` when `w_sil <= 0`
disables that solve, exactly mirroring `silencing`'s own on/off behaviour
in that regime. It is DIAGNOSTIC ONLY -- recorded for comparison, never
part of `cost`, which depends only on `inversion` and `silencing`.
"""
function pulse_cost(u::AbstractVector, pulse::CompositePulse, d;
                     w_inv=1.0, w_sil=0.7, target_F=1.0, w_time=0.15, w_power=0.05, w_tmax=1.0, kwargs...)
    _forbid_initial_condition(kwargs)
    T = eltype(u)
    duration = pulse_duration(pulse, u)
    _, t_end, _, cA, _ = decode(pulse, u)

    tmax_excess = max(t_end[end] - pulse.T_max, zero(T))
    tmax_penalty = w_tmax * (tmax_excess / pulse.T_max)^2

    normalized_cA = cA ./ pulse.amp_scale
    power_penalty = w_power * (sum(abs2, normalized_cA) / length(normalized_cA))

    inversion = zero(T)
    silencing = zero(T)
    coherence = zero(T)

    try
        if w_inv > 0.0
            _, _, Sz, Nj = run_sim_1st_order_pure(u, pulse, d; kwargs..., initial_condition=:ground)
            inversion = _weighted_inversion(Sz, Nj, T)
        end
        if w_sil > 0.0
            _, Sp, _, Nj_eq = run_sim_1st_order_pure(u, pulse, d; kwargs..., initial_condition=:equator)
            silencing = _weighted_silencing_factor(Sp, d.g_b, Nj_eq, T)
            # Diagnostic only -- reuses the equator solve already run above
            # for `silencing`, so this costs nothing extra; NOT part of `cost`.
            coherence = _weighted_coherence(Sp, Nj_eq, T)
        end
    catch e
        e isa PulseSolveFailed || rethrow()
        infT = convert(T, Inf)
        nanT = convert(T, NaN)
        return infT, nanT, nanT, duration, nanT
    end

    silencing_penalty = w_sil * (silencing - convert(T, target_F))^2
    cost = -w_inv * inversion + silencing_penalty + w_time * (duration / pulse.T_max) + tmax_penalty + power_penalty
    return cost, inversion, silencing, duration, coherence
end

# ============================================================
# ADAM (hand-rolled -- Optimisers.jl/Optim.jl are not dependencies of
# this package; Adam's update rule is short enough that adding either
# purely for this would be more overhead than benefit)
# ============================================================

mutable struct AdamState
    m::Vector{Float64}
    v::Vector{Float64}
    t::Int
end

AdamState(n::Integer) = AdamState(zeros(n), zeros(n), 0)

"""
    adam_step!(u, grad, state::AdamState; lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, lr_scale=nothing)

One in-place Adam update of `u` given `grad = ∇cost(u)`, mutating `state`.
Standard bias-corrected Adam (Kingma & Ba 2015). `lr_scale`, if given, is a
per-parameter multiplier (same length as `u`) applied to the FINAL step
only (`u[i] -= lr*lr_scale[i]*m_hat/(sqrt(v_hat)+eps)`) -- the moment
estimates `state.m`/`state.v` themselves still accumulate the raw
`grad`, unscaled, so `lr_scale` reshapes the effective per-parameter step
size without distorting Adam's own gradient-magnitude bookkeeping. Default
`nothing` reproduces the original uniform-`lr` update exactly (skips the
per-element multiply entirely, not just multiplies by an implicit `1.0`).
See [`run_local_adam`](@ref)'s `cf_lr_scale` for why a composite-pulse
optimisation in particular benefits from decoupling the chirp
coefficients' own step size from the rest of `u`.
"""
function adam_step!(u::AbstractVector, grad::AbstractVector, state::AdamState;
                     lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, lr_scale::Union{AbstractVector,Nothing}=nothing)
    state.t += 1
    @inbounds for i in eachindex(u)
        state.m[i] = beta1 * state.m[i] + (1 - beta1) * grad[i]
        state.v[i] = beta2 * state.v[i] + (1 - beta2) * grad[i]^2
        m_hat = state.m[i] / (1 - beta1^state.t)
        v_hat = state.v[i] / (1 - beta2^state.t)
        step = lr * m_hat / (sqrt(v_hat) + eps)
        u[i] -= lr_scale === nothing ? step : lr_scale[i] * step
    end
    return u
end

# ============================================================
# SHAPE FITTING: warm-starting a CompositePulse from a FIXED target drive
#
# Separate from pulse_cost/run_local_adam below (which fit against this
# package's own PHYSICS via a full ODE solve): this fits purely against a
# target WAVEFORM, no ODE solve at all -- minimising mean squared error
# between build_E_of_t(pulse, u) and a target t -> Complex drive at many
# sample points, via the same ForwardDiff.gradient + hand-rolled Adam this
# file already uses elsewhere. Orders of magnitude cheaper than a physics
# fit (no ODE solve per epoch), which is the point: it turns an arbitrary
# recorded pulse (e.g. a jld2 run's own analytic control pulse -- see
# jld2_pulse_loader.jl's fit_composite_pulse_seed) into a REASONED
# CompositePulse seed for the physics optimisation below, rather than a
# shape-blind random/canonical guess.
# ============================================================

"""
    fit_composite_pulse(pulse::CompositePulse, E_target; N_fit=4000, num_epochs=1000,
                         learning_rate=0.002, seed=42, u_init=nothing) -> (u_fit, fit_report)

Fits `pulse`'s raw parameters `u` so that `build_E_of_t(pulse, u)`
approximates a target drive `E_target(t)` (any `t -> Complex` callable --
e.g. `pulses.jl`'s `build_E_of_t(PULSE_CONFIG)` applied to an existing
recorded pulse) over `[0, pulse.T_max]`, by minimising mean squared error
at `N_fit` evenly spaced sample points -- sampled via `pulses.jl`'s own
[`sample_E_of_t`](@ref)`(E_target, pulse.T_max, N_fit)`, the SAME sampling
function every other reconstructed-curve consumer in this package uses
(`plot_E_of_t`, `save_run_data`'s `_pulsemat.csv`), rather than a second,
independent sampling loop. Purely a SHAPE fit -- it says nothing about the
resulting physics (inversion/silencing); follow up with a real 1st-order
solve (e.g. [`run_sim_1st_order_trajectory`](@ref)) to check that
separately.

`learning_rate` defaults to `0.002`, NOT `run_local_adam`/`pulse_cost`'s
own `0.05` -- that value is tuned for `pulse_cost`'s dimensionless,
already-normalised cost gradient, whereas `mse_only` here is raw squared
error in `E_target`'s own physical units (order `amp_scale^2 ~ 1e14` for
this package's typical cavity-input-flux scale), and Adam's per-parameter
step size, while adaptive to gradient MAGNITUDE, is still set in absolute
raw-parameter units by `lr` itself. Verified on this package's own 3-ARP
reference pulse (k=3): `learning_rate=0.05` (and even `0.005`) overshoots
on the very first step and never recovers within hundreds of epochs
(`rel_l2` stays pinned near its EPOCH-1 value or worse), while `0.002`
converges steadily to `rel_l2~0.09-0.12` over ~1000 epochs -- if fitting a
very differently-scaled target/ensemble, re-check this the same way (watch
`fit_report.history` for a non-monotonic first few epochs, a sign `lr` is
too large for that target's own scale).

`u_init` defaults to [`initial_guess`](@ref)`(pulse; seed=seed)` (same
random-but-physically-sensible starting point the physics optimiser
itself uses) rather than all-zeros, since an all-zero `raw_gap`/`raw_dur`
still decodes to a valid (if arbitrary) placement via `decode`'s
softplus + cumulative-sum reparameterisation, but starting from a point
already spread out over `[0, T_max]` gives every sub-pulse's gradient a
chance to find its own share of the target from the first epoch, rather
than all `k` sub-pulses initially overlapping near `t=0`. This is still a
GENERIC, timing-blind default -- verified (again on the 3-ARP reference)
to plateau at `rel_l2~0.998` (no better than an all-zero pulse) after 200
epochs, because its amplitude scale and sub-pulse timings have no relation
to the target's own; [`fit_composite_pulse_seed`](@ref) in
jld2_pulse_loader.jl passes a much better `u_init`
([`_segment_matched_seed_init`](@ref)) whenever the target's own segment
count matches `pulse.k`.

Returns `(u_fit, fit_report)`: `u_fit` is the LOWEST-mse `u` seen across
all epochs (not necessarily the last, same "track the best, don't just
return wherever descent stopped" pattern [`run_local_adam`](@ref) uses).
`fit_report = (mse, rel_l2, history)`: `rel_l2 = sqrt(mse*N_fit /
sum(abs2, target))` is a scale-free fit-quality number (`0` = perfect,
`~1` = no better than an all-zero pulse) that stays meaningful across
different targets/ensembles, unlike raw `mse` (whose scale depends on
`E_target`'s own amplitude). `history` is a `Vector{<:NamedTuple}` with
one `(epoch, mse)` row per epoch actually run.
"""
function fit_composite_pulse(
    pulse::CompositePulse, E_target;
    N_fit::Integer=4000, num_epochs::Integer=1000,
    learning_rate::Real=0.002, seed::Integer=42,
    u_init::Union{AbstractVector,Nothing}=nothing,
)
    t_grid = range(0.0, pulse.T_max; length=N_fit)
    Ex, Ep = sample_E_of_t(E_target, pulse.T_max, N_fit)
    target = complex.(Ex, Ep)
    target_energy = sum(abs2, target) + 1e-30

    u = u_init === nothing ? initial_guess(pulse; seed=seed) : collect(Float64, u_init)
    length(u) == n_params(pulse) || error(
        "u_init has length $(length(u)), but this CompositePulse (k=$(pulse.k), " *
        "n_coeff_A=$(pulse.n_coeff_A), n_coeff_f=$(pulse.n_coeff_f)) needs $(n_params(pulse))."
    )
    n = length(u)
    adam = AdamState(n)
    history = NamedTuple[]

    function mse_only(uu)
        E_of_t = build_E_of_t(pulse, uu)
        s = zero(eltype(uu))
        @inbounds for i in eachindex(t_grid)
            s += abs2(E_of_t(t_grid[i]) - target[i])
        end
        return s / N_fit
    end

    best_u, best_mse = copy(u), Inf
    for epoch in 1:num_epochs
        grad = ForwardDiff.gradient(mse_only, u)
        mse = Float64(mse_only(u))
        if mse < best_mse
            best_mse, best_u = mse, copy(u)
        end
        push!(history, (epoch=epoch, mse=mse))
        adam_step!(u, grad, adam; lr=learning_rate)
    end

    rel_l2 = sqrt(best_mse * N_fit / target_energy)
    return best_u, (mse=best_mse, rel_l2=rel_l2, history=history)
end

# ============================================================
# AMPLITUDE/FREQUENCY-SPACE FITTING FROM A RAW SAMPLED (t, I, Q) TRACE
#
# A second, more general route to a CompositePulse seed than
# fit_composite_pulse above: given ONLY a sampled I/Q waveform (no
# PULSE_CONFIG, no known sub-pulse count), discover how many sub-pulses it
# contains (silence-thresholding), extract its own amplitude/frequency
# decomposition, size ONE shared n_coeff_A/n_coeff_f from how many raw
# samples the densest detected sub-pulse actually has (~20-25 raw points
# per cubic B-spline piece), and fit build_A_f_of_t's amplitude/frequency
# curves against that decomposition directly -- rather than
# fit_composite_pulse's complex-valued MSE, which entangles amplitude and
# phase error in a way that is especially unstable exactly where the
# target is near-silent (phase becomes numerically meaningless as
# amplitude -> 0).
# ============================================================

"""
    _instantaneous_frequency(t, I, Q) -> (phi, f)

Unwrapped phase `phi(t_j) = atan(Q,I)` (continuous, no `2π` jumps at
consecutive samples) and its instantaneous angular frequency `f(t_j) =
dphi/dt` from a sampled I/Q trace: central difference on the unwrapped
`phi` (one-sided at the two endpoints). Both are returned at EVERY sample,
including where `I=Q=0` (phase, and hence `f`, is numerically meaningless
there) -- this function does not special-case or mask those points;
callers (see [`fit_composite_pulse_af`](@ref)/
[`fit_composite_pulse_from_samples`](@ref) with `fit_mode=:linear`) are expected to
down-weight them via an amplitude-based weight instead, since the
physically meaningful thing ("this region carries no drive") is already
fully captured by `A~0` there, not by discarding samples.

`phi` is returned (not just discarded internally the way an earlier
version of this function did) because it is itself a valid, DIRECT fit
target: `phi`'s own scale-invariant reference point is arbitrary (whatever
`atan` happens to return at the first sample, unrelated to
[`build_E_of_t`](@ref)'s own `phase_offset` convention), but its SHAPE over
one active sub-pulse is exactly the quantity a `cf`-fit that targets
[`build_E_of_t`](@ref)'s own EXACT phase integral `Φ(t) = ∫f dτ` should
match -- fitting `cf` against `f` alone (the OLDER approach) can leave
`Φ`'s accumulated integral drifting even when the per-point frequency
residual is tiny, since integration does not average away a small but
STRUCTURED (not i.i.d.) residual the way an RMS frequency comparison
would; see [`_fit_composite_pulse_from_samples_linear`](@ref)'s own
docstring for a real, measured case of exactly this (a `rel_l2_f~1e-13`
per-point frequency fit still producing a ~43% full-complex-trace
reconstruction error, traced to phase drift concentrated at the trace's
own peak amplitude).
"""
function _instantaneous_frequency(t::AbstractVector, I::AbstractVector, Q::AbstractVector)
    n = length(t)
    n == length(I) == length(Q) || error(
        "t/I/Q must have the same length, got $(n)/$(length(I))/$(length(Q))."
    )
    phi = atan.(Q, I)
    @inbounds for j in 2:n
        d = phi[j] - phi[j-1]
        while d > pi
            phi[j] -= 2 * pi
            d = phi[j] - phi[j-1]
        end
        while d < -pi
            phi[j] += 2 * pi
            d = phi[j] - phi[j-1]
        end
    end
    f = Vector{Float64}(undef, n)
    f[1] = (phi[2] - phi[1]) / (t[2] - t[1])
    f[n] = (phi[n] - phi[n-1]) / (t[n] - t[n-1])
    @inbounds for j in 2:n-1
        f[j] = (phi[j+1] - phi[j-1]) / (t[j+1] - t[j-1])
    end
    return phi, f
end

"""
    _detect_subpulse_segments(t, A; rel_thresh=1e-3, min_active_samples=5, min_silence_samples=3)
        -> Vector{Tuple{Int,Int}}

Detects contiguous "active" (non-silent) runs in a sampled amplitude trace
`A(t_j)`, returning `(i_start, i_end)` SAMPLE-INDEX ranges (inclusive
`t` indices), one per detected sub-pulse -- found purely from the sampled
waveform, with no assumed segment count or `PULSE_CONFIG` structure.

A sample counts as silent when `A[j] < rel_thresh * maximum(A)`. Two
robustness guards against noise: a candidate active run shorter than
`min_active_samples` is discarded (not a real sub-pulse, just a blip
poking above threshold in what's otherwise silence); a silent gap shorter
than `min_silence_samples` is merged back into "active" rather than
splitting one true sub-pulse into two (guards against a single noisy dip
near a target's own smooth near-zero region being mistaken for a genuine
separator).
"""
function _detect_subpulse_segments(
    t::AbstractVector, A::AbstractVector;
    rel_thresh::Real=1e-3, min_active_samples::Integer=5, min_silence_samples::Integer=3,
)
    n = length(A)
    thresh = rel_thresh * maximum(A)
    active = A .>= thresh

    j = 1
    while j <= n
        if !active[j]
            j0 = j
            while j <= n && !active[j]
                j += 1
            end
            gap_len = j - j0
            if gap_len < min_silence_samples && j0 > 1 && j <= n
                active[j0:j-1] .= true
            end
        else
            j += 1
        end
    end

    segments = Tuple{Int,Int}[]
    j = 1
    while j <= n
        if active[j]
            j0 = j
            while j <= n && active[j]
                j += 1
            end
            if j - j0 >= min_active_samples
                push!(segments, (j0, j - 1))
            end
        else
            j += 1
        end
    end
    return segments
end

"""
    _spline_coeff_count(n_samples; points_per_segment=22, degree=3) -> Int

Number of B-spline coefficients so each piecewise-cubic segment spans
roughly `points_per_segment` raw sample points: `n_pieces =
ceil(n_samples/points_per_segment)`, and (since a degree-`degree` clamped
B-spline with `n_coeff` coefficients has `n_coeff-degree` pieces --
[`make_clamped_knots`](@ref)) `n_coeff = n_pieces + degree`, floored at
`degree+1` (the minimum [`CompositePulse`](@ref) accepts).
"""
function _spline_coeff_count(n_samples::Integer; points_per_segment::Integer=22, degree::Integer=3)
    n_pieces = max(cld(n_samples, points_per_segment), 1)
    return max(n_pieces + degree, degree + 1)
end

"""
    points_per_segment_for_budget(n_samples_max, k; degree=3, param_budget=60) -> Int

Largest-detail `points_per_segment` (in [`_spline_coeff_count`](@ref)'s own
sense) that still keeps a `CompositePulse`'s total raw parameter count
(`n_params = 3*k + 2*k*n_coeff`, since `n_coeff_A = n_coeff_f = n_coeff` --
see [`n_params`](@ref)) at or under `param_budget`, given `k` sub-pulses and
`n_samples_max` (the LONGEST detected segment's own sample count --
`_spline_coeff_count` sizes the single shared `n_coeff` off this one value).

Inverts `_spline_coeff_count`'s own `n_pieces = ceil(n_samples/pps)`,
`n_coeff = n_pieces + degree`: first finds the largest feasible `n_coeff`
(`n_coeff_max = (param_budget - 3*k) ÷ (2*k)`, floor division), then the
SMALLEST `pps` that achieves at most `n_coeff_max - degree` pieces (`pps =
ceil(n_samples_max / n_pieces_max)`) -- ceiling division only ever pushes
the resulting `n_pieces` DOWN relative to `n_pieces_max`, never up, so the
ACTUAL `n_coeff` `_spline_coeff_count` returns for this `pps` is guaranteed
`<= n_coeff_max`, hence `n_params <= param_budget` exactly (checked below,
not just argued), not merely approximately.

Throws if `param_budget` cannot be met even at `CompositePulse`'s own
minimum coefficient count (`degree+1` per sub-pulse -- `_spline_coeff_count`'s
own floor), i.e. `param_budget < 3*k + 2*k*(degree+1)`: no `points_per_segment`,
however large, can go lower than that floor, so reduce `k` (fewer detected
sub-pulses) or `degree`, or raise `param_budget`, instead.
"""
function points_per_segment_for_budget(n_samples_max::Integer, k::Integer; degree::Integer=3, param_budget::Integer=60)
    k >= 1 || error("k must be a positive integer, got $k.")
    n_samples_max >= 1 || error("n_samples_max must be a positive integer, got $n_samples_max.")

    # n_params_for MUST match CompositePulse's own n_params(pulse) =
    # 3*pulse.k + pulse.k*pulse.n_coeff_A + pulse.k*pulse.n_coeff_f
    # (composite_pulse.jl) for the case n_coeff_A=n_coeff_f=n_coeff this
    # function always builds -- kept as ONE local closure (rather than
    # re-typed at each use below) specifically so there is only one place
    # to update if that formula ever changes again.
    n_params_for(n_coeff) = 3 * k + 2 * k * n_coeff

    n_coeff_floor = degree + 1
    min_budget = n_params_for(n_coeff_floor)
    param_budget >= min_budget || error(
        "param_budget=$param_budget cannot be met for k=$k sub-pulses at degree=$degree: " *
        "even the minimum coefficient count per sub-pulse ($n_coeff_floor) already needs " *
        "$min_budget total parameters (3*k + 2*k*(degree+1)). Reduce k/degree, or raise param_budget."
    )

    n_coeff_max = (param_budget - 3 * k) ÷ (2 * k)
    n_pieces_max = n_coeff_max - degree
    pps = cld(n_samples_max, n_pieces_max)

    n_coeff_actual = _spline_coeff_count(n_samples_max; points_per_segment=pps, degree=degree)
    n_params_actual = n_params_for(n_coeff_actual)
    n_params_actual <= param_budget || error(
        "internal inconsistency: computed points_per_segment=$pps still gives " *
        "n_params=$n_params_actual > param_budget=$param_budget (n_coeff=$n_coeff_actual)."
    )

    return pps
end

"""
    points_per_segment_for_budget(t, I, Q; degree=3, param_budget=60,
        rel_thresh=1e-3, min_active_samples=5, min_silence_samples=3) -> (pps::Int, segments)

Convenience wrapper over the `(n_samples_max, k)` method above: runs the
SAME segment detection ([`_detect_subpulse_segments`](@ref)) that
[`fit_composite_pulse_from_samples`](@ref)'s `fit_mode=:learned`/`:linear`
implementations use internally, then sizes
`points_per_segment` against the resulting `k` and longest-segment sample
count. Returns `(pps, segments)` so a caller can pass `pps` straight into
either fitter's own `points_per_segment` keyword -- or just pass
`param_budget` directly to those fitters, which do exactly this internally
(see their own docstrings).
"""
function points_per_segment_for_budget(
    t::AbstractVector, I::AbstractVector, Q::AbstractVector;
    degree::Integer=3, param_budget::Integer=60,
    rel_thresh::Real=1e-3, min_active_samples::Integer=5, min_silence_samples::Integer=3,
)
    A = sqrt.(I .^ 2 .+ Q .^ 2)
    segments = _detect_subpulse_segments(
        t, A; rel_thresh=rel_thresh, min_active_samples=min_active_samples, min_silence_samples=min_silence_samples,
    )
    isempty(segments) && error(
        "No active sub-pulses detected (rel_thresh=$rel_thresh too high relative to the " *
        "trace's own peak, or the trace really is all silence)."
    )
    k = length(segments)
    n_samples_max = maximum(i_end - i_start + 1 for (i_start, i_end) in segments)
    pps = points_per_segment_for_budget(n_samples_max, k; degree=degree, param_budget=param_budget)
    return pps, segments
end

"""
    _resolve_points_per_segment(points_per_segment, param_budget, n_samples_each, k, degree, caller_name) -> Int

Shared `param_budget` OVERRIDE step behind [`fit_composite_pulse_from_samples`](@ref)'s
two `fit_mode` implementations ([`_fit_composite_pulse_from_samples_learned`](@ref)
and [`_fit_composite_pulse_from_samples_linear`](@ref)): when `param_budget`
is given, replaces the caller's own `points_per_segment` with
[`points_per_segment_for_budget`](@ref)'s own sizing (against the ALREADY-
DETECTED `k`/`n_samples_each`, never a caller guess), printing what it
picked (`caller_name` names which of the two callers is logging, so the
message stays traceable back to its own call site); otherwise returns
`points_per_segment` unchanged.
"""
function _resolve_points_per_segment(
    points_per_segment::Integer, param_budget::Union{Nothing,Integer},
    n_samples_each, k::Integer, degree::Integer, caller_name::AbstractString,
)
    param_budget === nothing && return points_per_segment
    pps = points_per_segment_for_budget(maximum(n_samples_each), k; degree=degree, param_budget=param_budget)
    println("$caller_name: param_budget=$param_budget -> points_per_segment=$pps (k=$k)")
    return pps
end

"""
    fit_composite_pulse_af(pulse::CompositePulse, t_samples, A_target, f_target;
                            weight=A_target.^2, num_epochs=1000, learning_rate=0.002,
                            seed=42, u_init=nothing) -> (u_fit, fit_report)

Fits `pulse`'s raw parameters so its OWN amplitude/frequency curves
([`build_A_f_of_t`](@ref)) match `A_target`/`f_target` at `t_samples`,
instead of [`fit_composite_pulse`](@ref)'s complex-valued MSE. Minimises

    mean((A_of_t(t_j) - A_target[j])^2) +
    mean(weight[j] * (f_of_t(t_j) - f_target[j])^2) / mean(weight)

`weight` defaults to `A_target.^2`: instantaneous frequency extracted from
a sampled I/Q trace ([`_instantaneous_frequency`](@ref)) is numerically
meaningless wherever the amplitude is near 0 (phase is undefined at the
origin), so the frequency term is amplitude-weighted rather than given
equal weight everywhere; the amplitude term needs no such weighting since
`A_target` stays well-defined, and physically meaningful, all the way
down to 0. Dividing the frequency term by `mean(weight)` keeps the two
terms on comparable absolute scale regardless of `A_target`'s own units,
rather than letting whichever term happens to have larger raw magnitude
dominate the gradient purely because of a units mismatch.

Returns `(u_fit, fit_report)`: `u_fit` is the LOWEST-loss `u` seen across
all epochs. `fit_report = (loss, rel_l2_A, history)`: `rel_l2_A` is the
amplitude term's OWN scale-free residual (`0`=perfect), reported
separately from the combined `loss` since that's the more interpretable
single number for comparing fits (frequency error has no natural `[0,1]`
scale the way `fit_composite_pulse`'s `rel_l2` does).
"""
function fit_composite_pulse_af(
    pulse::CompositePulse, t_samples::AbstractVector, A_target::AbstractVector, f_target::AbstractVector;
    weight::AbstractVector=A_target .^ 2,
    num_epochs::Integer=1000, learning_rate::Real=0.002, seed::Integer=42,
    u_init::Union{AbstractVector,Nothing}=nothing,
)
    N = length(t_samples)
    (length(A_target) == N && length(f_target) == N && length(weight) == N) || error(
        "t_samples/A_target/f_target/weight must all have the same length, got " *
        "$(N)/$(length(A_target))/$(length(f_target))/$(length(weight))."
    )
    weight_mean = sum(weight) / N + 1e-30
    A_energy = sum(abs2, A_target) + 1e-30

    u = u_init === nothing ? initial_guess(pulse; seed=seed) : collect(Float64, u_init)
    length(u) == n_params(pulse) || error(
        "u_init has length $(length(u)), but this CompositePulse (k=$(pulse.k), " *
        "n_coeff_A=$(pulse.n_coeff_A), n_coeff_f=$(pulse.n_coeff_f)) needs $(n_params(pulse))."
    )
    n = length(u)
    adam = AdamState(n)
    history = NamedTuple[]

    function loss_only(uu)
        A_of_t, f_of_t = build_A_f_of_t(pulse, uu)
        sA = zero(eltype(uu))
        sf = zero(eltype(uu))
        @inbounds for j in 1:N
            sA += abs2(A_of_t(t_samples[j]) - A_target[j])
            sf += weight[j] * abs2(f_of_t(t_samples[j]) - f_target[j])
        end
        return sA / N + (sf / N) / weight_mean
    end

    best_u, best_loss = copy(u), Inf
    for epoch in 1:num_epochs
        grad = ForwardDiff.gradient(loss_only, u)
        loss = Float64(loss_only(u))
        if loss < best_loss
            best_loss, best_u = loss, copy(u)
        end
        push!(history, (epoch=epoch, loss=loss))
        adam_step!(u, grad, adam; lr=learning_rate)
    end

    A_of_t_best, _ = build_A_f_of_t(pulse, best_u)
    A_resid = sum(j -> abs2(A_of_t_best(t_samples[j]) - A_target[j]), 1:N)
    rel_l2_A = sqrt(A_resid / A_energy)

    return best_u, (loss=best_loss, rel_l2_A=rel_l2_A, history=history)
end

# ============================================================
# GRADIENT-SAFETY FLOOR/CLIP FOR SAMPLE-DERIVED SEEDS
#
# `decode`'s `gap`/`duration`/`cA` all go through `_softplus_inv` to reach
# RAW space, and `d(softplus)/d(raw) = sigmoid(raw)` -- for
# `raw = _softplus_inv(y)` with small `y = physical/scale`,
# `sigmoid(raw) ≈ y` (verified: `y=1e-6` -> `sigmoid(raw)≈1e-6`, `y=1e-30`
# -> `sigmoid(raw)≈1e-30`). Encoding a fitted quantity at a tiny fraction
# of its own scale -- this file previously did so via `max(peak_amp,
# 1e-30)` for `cA` in `fit_composite_pulse_from_samples`'s (`fit_mode=:learned`)
# seed, and `cA_floor_frac=1e-6` in its `fit_mode=:linear` implementation --
# therefore lands that raw parameter at a point where `decode`'s own
# gradient is attenuated by that same tiny factor from the very first
# optimisation epoch onward: `run_local_adam`'s Adam step can move such a
# parameter only as fast as this near-zero local sensitivity allows,
# regardless of how many epochs run, i.e. a practically dead parameter at
# the seed. `_GRAD_SAFE_FRAC=1e-2` keeps that attenuation to about 100x
# instead of 1e6-1e30x -- a deliberate seed-generation trade-off (a truly
# near-zero target coefficient gets encoded slightly too high) in exchange
# for guaranteeing every seeded parameter starts somewhere gradient
# descent can actually move it from.
# ============================================================

const _GRAD_SAFE_FRAC = 1e-2

"""
    _encode_scaled_softplus(physical, scale) -> Float64

Gradient-safe RAW encoding of a `decode`-softplus-reparameterised quantity
(`gap`, `duration`, or `cA`) fitted/derived directly from sampled data:
`_softplus_inv(max(physical, _GRAD_SAFE_FRAC*scale) / scale)`. See
[`_GRAD_SAFE_FRAC`](@ref) for why the floor is relative to `scale` rather
than an absolute numerical epsilon.
"""
_encode_scaled_softplus(physical::Real, scale::Real) = _softplus_inv(max(physical, _GRAD_SAFE_FRAC * scale) / scale)

"""
    _clip_cf_raw(cf_raw, cf_clip_mult) -> raw value(s), clamped

Clamps a fitted, already-scale-normalised chirp coefficient (`cf/freq_scale`)
to `± cf_clip_mult`. Unlike `gap`/`duration`/`cA`, `cf` is UNCONSTRAINED in
RAW space (no softplus, see [`decode`](@ref)), so it carries no vanishing-
gradient tail -- but a weighted least-squares (or MSE) fit against a noisy
instantaneous-frequency estimate can still occasionally return an outlier
coefficient far beyond any physically sensible chirp rate (the `A²`
weighting in [`fit_composite_pulse_af`](@ref)/[`_fit_composite_pulse_from_samples_linear`](@ref)
suppresses, but does not eliminate, noise from low-but-above-threshold-
amplitude samples). An extreme `cf` makes the very first physics-cost ODE
solve unnecessarily stiff or prone to outright failure -- and a failed
solve makes `pulse_cost` return a constant `Inf`/`NaN` with a ZERO
gradient (see `run_local_adam`), stalling the optimiser at the seed with
no way to move at all. `cf_clip_mult=20` (this section's default) is
generous relative to `pulse.freq_scale` (the ensemble's own `FWHM`, already
the natural chirp scale for this package's physics -- see
[`CompositePulse`](@ref)'s own `Omega_adiabatic` derivation) so it only
ever engages on genuine noise-driven outliers, not ordinary chirped
sub-pulses.
"""
_clip_cf_raw(cf_raw, cf_clip_mult::Real) = clamp.(cf_raw, -cf_clip_mult, cf_clip_mult)

"""
    _fit_composite_pulse_from_samples_learned(t, I, Q, d;
        points_per_segment=22, degree=3, taper_frac=0.1,
        rel_thresh=1e-3, min_active_samples=5, min_silence_samples=3,
        cf_clip_mult=20.0, num_epochs=1000, learning_rate=0.002, seed=42)
        -> (pulse::CompositePulse, u_fit, fit_report, segments)

Private `fit_mode=:learned` implementation behind
[`fit_composite_pulse_from_samples`](@ref) -- see that function for the
public entry point. Builds a [`CompositePulse`](@ref) seed DIRECTLY from a sampled I/Q trace,
with NO prior knowledge of how many sub-pulses it contains or where they
are:

  1. Detects sub-pulses via [`_detect_subpulse_segments`](@ref) (silence
     thresholding on `A=sqrt(I^2+Q^2)`) -- `k` is however many segments
     that finds, not a caller-supplied count.
  2. Extracts the target amplitude/frequency decomposition for the WHOLE
     trace ([`_instantaneous_frequency`](@ref)).
  3. Sizes ONE shared `n_coeff_A`/`n_coeff_f` (used for every sub-pulse --
     `CompositePulse` has no per-sub-pulse coefficient count) via
     [`_spline_coeff_count`](@ref), from whichever detected segment has
     the MOST samples -- every segment gets AT LEAST ~`points_per_segment`
     raw points per cubic piece this way, though a shorter segment ends up
     with more spline resolution than its own sample count would strictly
     need.
  4. Builds a segment-matched `u_init`: each sub-pulse placed at its
     detected segment's own `t`-span, amplitude flat at that segment's own
     peak `A`, frequency ramped linearly across that segment's own
     `extrema(f)` -- the same idea as
     [`_segment_matched_seed_init`](@ref) (jld2_pulse_loader.jl), but
     derived from the DETECTED segments' own sample data rather than a
     labelled `PULSE_CONFIG`.
  5. Fits via [`fit_composite_pulse_af`](@ref) (amplitude/frequency-space).

`d` is `prepare_derived(CONFIG)`'s own return value (only used for
`CompositePulse`'s own `T_max`/scale fields -- `t`'s own span need not
equal `d.timespan`, though for a physically meaningful seed it should).

Seed encoding is GRADIENT-SAFE by construction (see
[`_encode_scaled_softplus`](@ref)/[`_clip_cf_raw`](@ref)): `gap`/`duration`/
`cA` are floored at `_GRAD_SAFE_FRAC` (`1e-2`) of their own scale rather
than at an arbitrarily small numerical epsilon, and `cf` is clipped to
`± cf_clip_mult * pulse.freq_scale` -- both guard against handing
[`optimise_composite_pulse`](@ref)/`run_local_adam` a seed with a
practically-dead (vanishing-decode-gradient) parameter or a noise-driven
chirp outlier that makes the very first physics ODE solve fail (a failed
solve makes `pulse_cost` return `Inf`/`NaN` with a ZERO gradient, stalling
the optimiser at the seed with no way to move at all -- see
[`_clip_cf_raw`](@ref)'s own docstring).

Returns `(pulse, u_fit, fit_report, segments)` -- `segments` is the raw
`(i_start, i_end)` sample-index list from step 1, for inspection/plotting.

Pass `param_budget` (e.g. `60`) instead of hand-picking `points_per_segment`
to instead cap the resulting `n_params = 3*k + 2*k*n_coeff` directly -- see
[`points_per_segment_for_budget`](@ref), which this calls internally (after
step 1 determines `k`) to override `points_per_segment` when `param_budget`
is given. Useful because `ForwardDiff.gradient`+Adam (this function's own
descent) becomes impractically slow well before `n_coeff` reaches the tens,
per this docstring's own opening paragraph -- capping `n_params` up front is
the practical way to keep this route usable on a densely-sampled real trace.
"""
function _fit_composite_pulse_from_samples_learned(
    t::AbstractVector, I::AbstractVector, Q::AbstractVector, d;
    points_per_segment::Integer=22, degree::Integer=3, taper_frac::Real=0.1,
    rel_thresh::Real=1e-3, min_active_samples::Integer=5, min_silence_samples::Integer=3,
    cf_clip_mult::Real=20.0, num_epochs::Integer=1000, learning_rate::Real=0.002, seed::Integer=42,
    param_budget::Union{Nothing,Integer}=nothing,
)
    A = sqrt.(I .^ 2 .+ Q .^ 2)
    phi, f = _instantaneous_frequency(t, I, Q)

    segments = _detect_subpulse_segments(
        t, A; rel_thresh=rel_thresh, min_active_samples=min_active_samples, min_silence_samples=min_silence_samples,
    )
    isempty(segments) && error(
        "No active sub-pulses detected (rel_thresh=$rel_thresh too high relative to the " *
        "trace's own peak, or the trace really is all silence)."
    )
    k = length(segments)

    n_samples_each = [i_end - i_start + 1 for (i_start, i_end) in segments]
    points_per_segment = _resolve_points_per_segment(
        points_per_segment, param_budget, n_samples_each, k, degree, "fit_composite_pulse_from_samples (fit_mode=:learned)",
    )
    n_coeff = _spline_coeff_count(maximum(n_samples_each); points_per_segment=points_per_segment, degree=degree)

    pulse = CompositePulse(k, n_coeff, n_coeff, d; degree=degree, taper_frac=taper_frac)

    raw_gap = Vector{Float64}(undef, k)
    raw_dur = Vector{Float64}(undef, k)
    raw_phi0 = Vector{Float64}(undef, k)
    raw_cA = Matrix{Float64}(undef, n_coeff, k)
    raw_cf = Matrix{Float64}(undef, n_coeff, k)
    t_prev_end = 0.0
    running_seed = 0.0
    for (idx, (i_start, i_end)) in enumerate(segments)
        t_s, t_e = t[i_start], t[i_end]
        duration = t_e - t_s
        gap = max(t_s - t_prev_end, 0.0)
        dur_arg = max(duration - pulse.dur_floor, 0.0)
        raw_gap[idx] = _encode_scaled_softplus(gap, pulse.gap_scale)
        raw_dur[idx] = _encode_scaled_softplus(dur_arg, pulse.dur_scale)

        # build_A_f_of_t's own loss (fit_composite_pulse_af, below) never
        # references phi0 at all, so ForwardDiff/Adam leaves this component
        # untouched (zero gradient) -- this seed IS the final fitted value,
        # not just a starting point. Approximate `running` (build_E_of_t's
        # own accumulator) with the TARGET trace's own raw phase at each
        # segment's boundary, since this path's `raw_cf` seed is only a
        # crude linear ramp (not an exact antiderivative fit) and so has no
        # equally cheap EXACT `d_f[end]` to accumulate instead.
        raw_phi0[idx] = phi[i_start] - running_seed
        running_seed = phi[i_end]

        peak_amp = maximum(view(A, i_start:i_end))
        raw_cA[:, idx] .= _encode_scaled_softplus(peak_amp, pulse.amp_scale)

        f_lo, f_hi = extrema(view(f, i_start:i_end))
        raw_cf[:, idx] .= _clip_cf_raw(range(f_lo, f_hi; length=n_coeff) ./ pulse.freq_scale, cf_clip_mult)

        t_prev_end = t_e
    end
    u_init = pack(pulse, raw_gap, raw_dur, raw_phi0, raw_cA, raw_cf)

    u_fit, fit_report = fit_composite_pulse_af(
        pulse, t, A, f; num_epochs=num_epochs, learning_rate=learning_rate, seed=seed, u_init=u_init,
    )
    return pulse, u_fit, fit_report, segments
end

"""
    _fit_composite_pulse_from_samples_linear(t, I, Q, d;
        points_per_segment=6, degree=3, taper_frac=0.1,
        rel_thresh=1e-3, min_active_samples=5, min_silence_samples=3,
        cA_floor_frac=_GRAD_SAFE_FRAC, cf_clip_mult=20.0,
        param_budget=nothing, segments=nothing)
        -> (pulse::CompositePulse, u_fit, fit_report, segments)

Private `fit_mode=:linear` implementation behind
[`fit_composite_pulse_from_samples`](@ref) -- see that function for the
public entry point. Closed-form alternative to
[`_fit_composite_pulse_from_samples_learned`](@ref) (`fit_mode=:learned`):
same segment detection / `n_coeff` sizing (steps 1-3 of that function's own
docstring), but NO `ForwardDiff`/Adam descent at all. This matters at the
resolution the 20-25-points-per-segment rule implies for a real, densely
sampled trace: a single ~200us sub-pulse sampled at ~5000 points over
~600us needs `n_coeff~80`, and for `k=3` that is `n_params~480` --
verified impractical for `ForwardDiff.gradient`+Adam as currently
implemented (`bspline_basis` allocates fresh temporaries on every call,
with no caching; a real attempt at this scale did not finish a single
epoch in 10 minutes, ~9x10^8 allocations in). This function instead
exploits that MOST of what's being fit is actually linear, given the
segmentation this file already computes independently of any pulse
parameter:

  - `t_start`/`t_end` per sub-pulse are taken DIRECTLY from the detected
    segment's own sample boundaries -- no fitting needed, since
    segmentation already locates them exactly (to sample resolution).
  - Given those (hence given the B-spline's knot vector), `A_spline(t) =
    Σ cA_i B_i(t)` is LINEAR in `cA` -- the exact solution of a
    taper-weighted (folding the KNOWN, FIXED `_taper_window` multiplier
    into the design matrix so the amplitude term being solved for is
    `pulse`'s own actual physical envelope, not the bare untapered spline)
    linear least-squares problem, via `\\`.
  - `cf` is fit against the ACCUMULATED PHASE `phi` (the raw unwrapped
    `atan(Q,I)` from [`_instantaneous_frequency`](@ref)), NOT the
    pointwise frequency curve `f_spline(t) = Σ cf_i B_i(t)` an earlier
    version of this function fit directly. `Φ(t) = ∫f dτ` is itself LINEAR
    in `cf` (de Boor's antiderivative construction -- the same one
    [`bspline_antiderivative`](@ref) uses, expressed here as an explicit
    `(n_coeff+1, n_coeff)` linear map so the fit stays a single weighted
    least-squares solve, verified bit-for-bit equivalent to that
    function's own cumulative-sum recursion), so this is still one
    `n_samples x (n_coeff+1)` solve per sub-pulse, exact and just as cheap
    -- the `+1` column is an unconstrained additive constant absorbing
    `phi`'s own arbitrary reference point (unrelated to
    [`build_E_of_t`](@ref)'s own `phase_offset` bookkeeping), discarded
    after the solve. This matters because a per-point frequency fit can
    have an excellent RMS residual while its INTEGRAL still drifts
    (integration doesn't average away a small but structured, rather than
    i.i.d., residual) -- measured directly on this package's own 3-ARP
    reference: fitting `cf` against `f` gave `rel_l2_f~1e-13` (an
    essentially perfect per-point frequency fit) yet the reconstructed
    COMPLEX pulse (`build_E_of_t(pulse,u_fit)`, resampled and compared
    point-by-point against the original trace) was off by `rel_l2~0.43`,
    with >99.9999% of that error concentrated at the trace's own peak
    amplitude sample -- exactly where a modest phase error costs the most
    in a squared-error sum. Fitting `cf` against `phi` directly targets the
    quantity that actually enters `E(t)=A(t)*exp(iΦ(t))`, closing that gap.
    Both `cf`'s solve and the (retained, diagnostic-only) frequency-domain
    comparison are WEIGHTED by `A_target.^2` (same rationale as
    [`fit_composite_pulse_af`](@ref)'s `weight` default: phase/frequency
    are numerically meaningless wherever the target amplitude is near 0).

`points_per_segment` defaults to `6` (not the `fit_mode=:learned`
implementation's 20-25 spec) -- verified on this package's own 3-ARP reference pulse to
still improve `rel_l2_A` noticeably over the 20-25 range (0.0013 at 22
points/segment down to 0.00055 at 6), with sharply diminishing returns
below that (a further halving to 4 points/segment only reached 0.00046)
and negligible runtime cost either way (all of 22/12/8/6/4 fit in under
1.5s combined on that reference case, since this is a handful of small
linear solves, not an iterative descent).

`cA_floor_frac`: ordinary least squares has no non-negativity constraint,
but `CompositePulse`'s own parameterisation requires `cA >= 0`
(`decode`'s `cA = amp_scale*softplus(raw_cA)`, always non-negative, so
`_softplus_inv` needs a positive argument) -- any solved coefficient below
`cA_floor_frac * pulse.amp_scale` is clamped up to that floor before
encoding. This doubles as this seed's GRADIENT-SAFETY floor (see
[`_GRAD_SAFE_FRAC`](@ref)): `decode`'s own softplus-reparameterisation
gradient at the seed is attenuated by roughly `cA_floor_frac` itself, so
the previous default (`1e-6`, chosen only to keep `_softplus_inv`'s
argument positive) left any floored coefficient with an effectively DEAD
decode-gradient for the whole physics optimisation that follows --
`_GRAD_SAFE_FRAC=1e-2` keeps that attenuation to about 100x instead of
1e6x. This is still a pragmatic guard against small negative undershoots
near sharp features, NOT a proper non-negative least-squares solve; on the
package's own 3-ARP reference case this floor starts triggering (a small
handful of coefficients, out of hundreds) right around `points_per_segment
= 6`, so `fit_report.n_cA_floored` is worth checking at this default --
a persistently nonzero count is a sign a true NNLS solve would do better
than this clamp. `cf_clip_mult` guards the frequency side the same way
[`_clip_cf_raw`](@ref) does for [`_fit_composite_pulse_from_samples_learned`](@ref):
the weighted per-segment linear solve for `cf` has no such floor issue
(unconstrained, no softplus) but can still return a noise-driven outlier
coefficient that makes the very first physics ODE solve stiff or prone to
failure -- clipped to `± cf_clip_mult * pulse.freq_scale`.

Returns `(pulse, u_fit, fit_report, segments)` -- `fit_report =
(rel_l2_A, rel_l2_f, phi_rms_rad, rel_l2_complex, n_cA_floored,
n_cf_clipped)`: `rel_l2_A` is the same scale-free `[0,1]`-ish residual
`fit_composite_pulse`/`_af` report (`0` = perfect); `rel_l2_f` is now a
DIAGNOSTIC-ONLY pointwise frequency-curve residual (`Basis*cf_seg` vs
`f_seg`, evaluated at the `cf_seg` the PHASE fit produced) -- kept for
comparison, no longer what `cf` is actually fit against; `phi_rms_rad` is
the `A²`-weighted RMS phase residual in RADIANS (`Φ(t)` vs `phi`, the
quantity `cf`'s solve DOES target) -- unlike `rel_l2_A`/`rel_l2_f` this
has no natural `[0,1]` scale (it's an absolute angle), so judge it against
how many radians of phase error would actually matter for your own physics
(a fraction of a radian is generally fine; an O(1) value at high amplitude
is not). `rel_l2_complex` is the full-trace COMPLEX reconstruction error
(`build_E_of_t(pulse,u_fit)` resampled at `t`, compared against
`complex.(I,Q)` directly, `0`=perfect) -- unlike the other three, which
each check one decoupled piece, this is what a caller of the fitted pulse
actually experiences; see the "RESOLVED" note below for why this can (and
used to) disagree sharply with `phi_rms_rad` alone. `n_cA_floored`/
`n_cf_clipped` are the total counts of amplitude/frequency coefficients
(across all sub-pulses) that hit the non-negativity floor / clip bound
above.

RESOLVED (previously a known limitation of this function) -- a
near-perfect `phi_rms_rad` does NOT by itself guarantee a near-perfect
reconstructed COMPLEX pulse: each sub-pulse's phase fit here includes a
free additive constant (`phase_const` in the loop below) that absorbs
`phi`'s own arbitrary reference point. An earlier version of this function
discarded that constant after the solve, and `CompositePulse` itself had
no parameter to receive it back -- `build_E_of_t` hardcoded sub-pulse 1's
own phase to start at exactly `0` and accumulated every later sub-pulse's
own `phase_offset[i]` purely from the FITTED sub-pulses' internal `∫f`,
never from anything about the target's own true absolute phase reference.
Measured directly on this package's own 3-ARP reference before the fix:
sub-pulse 1's own raw phase at its detected start was `0.4466` rad,
predicting a `2*sin(0.4466/2) = 0.443` relative full-trace error --
matching the then-measured `rel_l2_complex = 0.434` almost exactly, despite
`phi_rms_rad ~ 1e-13`. `CompositePulse` now has an explicit per-sub-pulse
`raw_phi0` (see [`decode`](@ref)/[`build_E_of_t`](@ref)), and `phase_const`
is recovered into it EXACTLY here (not re-fit, not approximated -- see
`raw_phi0[idx] = phase_const - running_phase` in the loop below): on the
same 3-ARP reference, `rel_l2_complex` now measures `0.0004`, matching
`rel_l2_A` rather than sitting two orders of magnitude worse.

Pass `param_budget` (e.g. `60`) instead of hand-picking `points_per_segment`
to instead cap the resulting `n_params = 3*k + 2*k*n_coeff` directly -- see
[`points_per_segment_for_budget`](@ref), which this calls internally (after
step 1 determines `k`) to override `points_per_segment` when `param_budget`
is given. Since this route is a closed-form linear solve (not iterative),
raising `param_budget` costs only a bigger (still cheap) linear system, not
a slower descent -- unlike [`_fit_composite_pulse_from_samples_learned`](@ref), where
`param_budget` exists mainly to keep `ForwardDiff`/Adam tractable at all.

Pass `segments` (the same `Vector{Tuple{Int,Int}}` [`_detect_subpulse_segments`](@ref)
itself returns) to SKIP step 1's own detection and use the given segments
directly -- for a caller (e.g. [`fit_composite_pulse_seed_linear_exact`](@ref))
that already ran detection itself against this SAME `(t, A)` to validate
`k`/size `points_per_segment` before calling this function, avoiding a
second identical `O(N)` amplitude-threshold scan. `rel_thresh`/
`min_active_samples`/`min_silence_samples` are ignored when `segments` is
given (nothing left for them to control).
"""
function _fit_composite_pulse_from_samples_linear(
    t::AbstractVector, I::AbstractVector, Q::AbstractVector, d;
    points_per_segment::Integer=6, degree::Integer=3, taper_frac::Real=0.1,
    rel_thresh::Real=1e-3, min_active_samples::Integer=5, min_silence_samples::Integer=3,
    cA_floor_frac::Real=_GRAD_SAFE_FRAC, cf_clip_mult::Real=20.0,
    param_budget::Union{Nothing,Integer}=nothing,
    segments::Union{Nothing,Vector{Tuple{Int,Int}}}=nothing,
)
    A = sqrt.(I .^ 2 .+ Q .^ 2)
    phi, f = _instantaneous_frequency(t, I, Q)

    segments = segments === nothing ? _detect_subpulse_segments(
        t, A; rel_thresh=rel_thresh, min_active_samples=min_active_samples, min_silence_samples=min_silence_samples,
    ) : segments
    isempty(segments) && error(
        "No active sub-pulses detected (rel_thresh=$rel_thresh too high relative to the " *
        "trace's own peak, or the trace really is all silence)."
    )
    k = length(segments)

    n_samples_each = [i_end - i_start + 1 for (i_start, i_end) in segments]
    points_per_segment = _resolve_points_per_segment(
        points_per_segment, param_budget, n_samples_each, k, degree, "fit_composite_pulse_from_samples (fit_mode=:linear)",
    )
    n_coeff = _spline_coeff_count(maximum(n_samples_each); points_per_segment=points_per_segment, degree=degree)

    pulse = CompositePulse(k, n_coeff, n_coeff, d; degree=degree, taper_frac=taper_frac)
    cA_floor = cA_floor_frac * pulse.amp_scale

    raw_gap = Vector{Float64}(undef, k)
    raw_dur = Vector{Float64}(undef, k)
    raw_phi0 = Vector{Float64}(undef, k)
    raw_cA = Matrix{Float64}(undef, n_coeff, k)
    raw_cf = Matrix{Float64}(undef, n_coeff, k)
    n_cA_floored = 0
    n_cf_clipped = 0
    A_resid = 0.0
    f_resid = 0.0
    phi_resid = 0.0
    f_weight_sum = 0.0
    t_prev_end = 0.0
    running_phase = 0.0

    for (idx, (i_start, i_end)) in enumerate(segments)
        t_s, t_e = t[i_start], t[i_end]
        duration = t_e - t_s
        gap = max(t_s - t_prev_end, 0.0)
        dur_arg = max(duration - pulse.dur_floor, 0.0)
        raw_gap[idx] = _encode_scaled_softplus(gap, pulse.gap_scale)
        raw_dur[idx] = _encode_scaled_softplus(dur_arg, pulse.dur_scale)

        t_seg = view(t, i_start:i_end)
        A_seg = view(A, i_start:i_end)
        f_seg = view(f, i_start:i_end)
        phi_seg = view(phi, i_start:i_end)
        n_seg = length(t_seg)

        knots = make_clamped_knots(n_coeff, t_s, t_e, degree)
        Basis = Matrix{Float64}(undef, n_seg, n_coeff)
        @inbounds for j in 1:n_seg
            Basis[j, :] .= bspline_basis(t_seg[j], knots, degree)
        end

        taper_w = [_taper_window(t_seg[j], t_s, t_e, taper_frac) for j in 1:n_seg]
        M_A = Basis .* taper_w
        cA_seg = M_A \ collect(A_seg)
        n_cA_floored += count(<(cA_floor), cA_seg)
        cA_seg = max.(cA_seg, cA_floor)
        raw_cA[:, idx] .= _softplus_inv.(cA_seg ./ pulse.amp_scale)

        # Fit cf against the ACCUMULATED PHASE (the exact quantity
        # build_E_of_t's Φ(t)=∫f dτ reconstructs), not the pointwise
        # frequency curve -- see this function's own docstring for why a
        # pointwise-f fit can leave Φ drifting even when its own per-point
        # RMS looks excellent. `bspline_antiderivative`'s own degree-(p+1)
        # antiderivative construction, expressed here as an explicit
        # (n_coeff+1, n_coeff) linear map `L` (`d = L*cf`) so the whole
        # fit stays a single weighted linear least-squares solve; verified
        # bit-for-bit equivalent to `bspline_antiderivative`'s own
        # cumulative-sum recursion. The augmented constant column absorbs
        # `phi`'s own arbitrary reference point (whatever `atan` returned
        # at the trace's first sample) -- unrelated to and decoupled from
        # `build_E_of_t`'s own `phase_offset` bookkeeping, exactly the way
        # an intercept term in ordinary linear regression decouples a
        # slope fit from an unknown additive offset.
        knots_p1 = vcat(knots[1:1], knots, knots[end:end])
        Basis_p1 = Matrix{Float64}(undef, n_seg, n_coeff + 1)
        @inbounds for j in 1:n_seg
            Basis_p1[j, :] .= bspline_basis(t_seg[j], knots_p1, degree + 1)
        end
        L = zeros(Float64, n_coeff + 1, n_coeff)
        for j in 1:n_coeff
            width = (knots[j+degree+1] - knots[j]) / (degree + 1)
            L[j+1:end, j] .= width
        end
        M_Phi_base = Basis_p1 * L
        M_Phi = hcat(M_Phi_base, ones(n_seg))

        f_weight = A_seg .^ 2 .+ 1e-30
        sw = sqrt.(f_weight)
        M_Phi_weighted = M_Phi .* sw
        phi_target_weighted = collect(phi_seg) .* sw
        sol = M_Phi_weighted \ phi_target_weighted
        cf_seg = sol[1:n_coeff]
        phase_const = sol[end]

        cf_seg_raw = cf_seg ./ pulse.freq_scale
        n_cf_clipped += count(x -> abs(x) > cf_clip_mult, cf_seg_raw)
        raw_cf[:, idx] .= _clip_cf_raw(cf_seg_raw, cf_clip_mult)

        A_pred = M_A * cA_seg
        f_pred = Basis * cf_seg
        Phi_pred = M_Phi_base * cf_seg .+ phase_const
        A_resid += sum(abs2, A_pred .- A_seg)
        f_resid += sum(f_weight .* abs2.(f_pred .- f_seg))
        phi_resid += sum(f_weight .* abs2.(Phi_pred .- phi_seg))
        f_weight_sum += sum(f_weight)

        # `phase_const` IS the fitted value of build_E_of_t's own
        # `phase_offset[i]` (bspline_antiderivative references `d_f[1]=0`
        # at each sub-pulse's own t_start, so `Phi_pred` at t_start is
        # exactly `phase_const`) -- recover the DISCRETE JUMP `raw_phi0`
        # build_E_of_t actually adds (`phase_offset[i] = running+phi0[i]`)
        # by subtracting off what `running` will be at this point, tracked
        # here with the SAME recipe (`running = phase_offset[i]+d_f[end]`),
        # using `Phi_pred[end]-phase_const` as `d_f[end]` -- exact, since
        # `t_seg[end] == t_e` is precisely the antiderivative spline's own
        # right knot (bspline_basis's closed-right-boundary convention).
        raw_phi0[idx] = phase_const - running_phase
        d_f_end = Phi_pred[end] - phase_const
        running_phase = phase_const + d_f_end

        t_prev_end = t_e
    end

    u_fit = pack(pulse, raw_gap, raw_dur, raw_phi0, raw_cA, raw_cf)

    A_energy = sum(abs2, A) + 1e-30
    rel_l2_A = sqrt(A_resid / A_energy)
    rel_l2_f = sqrt(f_resid / (f_weight_sum + 1e-30)) / pulse.freq_scale
    phi_rms_rad = sqrt(phi_resid / (f_weight_sum + 1e-30))

    # Full-trace COMPLEX reconstruction error -- the one number that
    # actually reflects what a caller of build_E_of_t(pulse, u_fit) will
    # see, as opposed to rel_l2_A/rel_l2_f/phi_rms_rad, each of which only
    # checks one decoupled piece. Deliberately built from build_E_of_t
    # itself (not from any of this loop's own per-segment intermediates,
    # e.g. A_pred/Basis_p1/cf_seg, which are LOCAL to a single sub-pulse
    # and don't include phi0/taper/silence) -- this is the same quantity
    # smoke_test_fit_from_pulsemat's own diff_report.rel_l2 measures via a
    # CSV round-trip; computing it here too means a direct caller of this
    # function (no file I/O involved) gets it for free.
    E_fit = build_E_of_t(pulse, u_fit)
    E_pred = ComplexF64[E_fit(tt) for tt in t]
    E_tar = complex.(I, Q)
    complex_energy = sum(abs2, E_tar) + 1e-30
    rel_l2_complex = sqrt(sum(abs2, E_pred .- E_tar) / complex_energy)

    fit_report = (
        rel_l2_A=rel_l2_A, rel_l2_f=rel_l2_f, phi_rms_rad=phi_rms_rad,
        rel_l2_complex=rel_l2_complex, n_cA_floored=n_cA_floored, n_cf_clipped=n_cf_clipped,
    )
    return pulse, u_fit, fit_report, segments
end

"""
    fit_composite_pulse_from_samples(t, I, Q, d;
        fit_mode=:linear, points_per_segment=nothing, degree=3, taper_frac=0.1,
        rel_thresh=1e-3, min_active_samples=5, min_silence_samples=3,
        cf_clip_mult=20.0, param_budget=nothing, kwargs...)
        -> (pulse::CompositePulse, u_fit, fit_report, segments)

Builds a [`CompositePulse`](@ref) seed DIRECTLY from a sampled I/Q trace,
with NO prior knowledge of how many sub-pulses it contains or where they
are (segment detection via [`_detect_subpulse_segments`](@ref) -- `k` is
however many segments that finds, not a caller-supplied count). Dispatches
to one of two implementations, selected by `fit_mode`:

  - `fit_mode=:linear` (default): [`_fit_composite_pulse_from_samples_linear`](@ref)
    -- a closed-form, per-segment weighted least-squares solve (`cA` from the
    tapered amplitude trace, `cf` from the accumulated phase). No iterative
    descent, so it stays cheap even at hundreds of coefficients; this is the
    route this package's own optimisation pipeline
    ([`optimise_control_pulse_from_jld2`](@ref)) uses exclusively. Accepts
    `cA_floor_frac`/`segments` via `kwargs...`.
  - `fit_mode=:learned`: [`_fit_composite_pulse_from_samples_learned`](@ref)
    -- `ForwardDiff.gradient`+Adam descent (via [`fit_composite_pulse_af`](@ref)).
    Verified impractically slow once `n_coeff` reaches the tens (see that
    function's own docstring) -- kept for small/low-resolution fits or
    comparison against the closed-form route, not for a densely sampled real
    trace. Accepts `num_epochs`/`learning_rate`/`seed` via `kwargs...`.

`points_per_segment=nothing` resolves to each mode's own prior default (`6`
for `:linear`, `22` for `:learned`) -- pass an explicit value to override
either. `degree`/`taper_frac`/`rel_thresh`/`min_active_samples`/
`min_silence_samples`/`cf_clip_mult`/`param_budget` are shared by both modes
and forwarded as-is; see either implementation's own docstring for the full
mathematical detail and rationale (segment detection, spline construction,
gradient-safety floors/clips, `param_budget` sizing) -- none of that changed
by this dispatcher, which only selects which implementation runs.

Returns `(pulse, u_fit, fit_report, segments)` in both modes, though
`fit_report`'s own field set DIFFERS between them (see each implementation's
docstring) since the two fits report genuinely different diagnostics.
"""
function fit_composite_pulse_from_samples(
    t::AbstractVector, I::AbstractVector, Q::AbstractVector, d;
    fit_mode::Symbol=:linear,
    points_per_segment::Union{Nothing,Integer}=nothing,
    degree::Integer=3, taper_frac::Real=0.1,
    rel_thresh::Real=1e-3, min_active_samples::Integer=5, min_silence_samples::Integer=3,
    cf_clip_mult::Real=20.0,
    param_budget::Union{Nothing,Integer}=nothing,
    kwargs...,
)
    pps = points_per_segment === nothing ? (fit_mode === :linear ? 6 : 22) : points_per_segment
    if fit_mode === :linear
        return _fit_composite_pulse_from_samples_linear(
            t, I, Q, d; points_per_segment=pps, degree=degree, taper_frac=taper_frac,
            rel_thresh=rel_thresh, min_active_samples=min_active_samples, min_silence_samples=min_silence_samples,
            cf_clip_mult=cf_clip_mult, param_budget=param_budget, kwargs...,
        )
    elseif fit_mode === :learned
        return _fit_composite_pulse_from_samples_learned(
            t, I, Q, d; points_per_segment=pps, degree=degree, taper_frac=taper_frac,
            rel_thresh=rel_thresh, min_active_samples=min_active_samples, min_silence_samples=min_silence_samples,
            cf_clip_mult=cf_clip_mult, param_budget=param_budget, kwargs...,
        )
    else
        error("fit_mode must be :linear or :learned, got $(repr(fit_mode)).")
    end
end

# ============================================================
# TASK-PARALLEL GRADIENT (2-way, across the 2 initial conditions)
#
# pulse_cost's own scalar `cost` is LITERALLY additively separable into a
# :ground-only term, an :equator-only term, and a direct (no-ODE-solve)
# term -- see _pulse_cost_grad_threaded's own docstring. This exploits
# that separability to run the two EXPENSIVE, independent, differentiated
# ODE solves concurrently via Threads.@threads, using the SAME pattern
# optimise_composite_pulse_over_k (below) already uses for N independent
# optimise_composite_pulse runs -- proven safe in this exact codebase, not
# a new concurrency pattern. This is deliberately SCOPED to the 2
# initial-condition split only: also parallelising ForwardDiff.gradient's
# own internal chunk loop would require reimplementing its internal
# seed/extract machinery by hand, a real reimplementation with genuine
# silent-wrong-gradient risk if done incorrectly -- out of scope here.
# ============================================================

"""
    _pulse_cost_grad_threaded(u, pulse, d; w_inv=1.0, w_sil=0.7, target_F=1.0, w_time=0.15, w_power=0.05, w_tmax=1.0, kwargs...)
        -> (grad::Vector{Float64}, cost, inversion, silencing, duration, coherence)

Task-parallel drop-in for `ForwardDiff.gradient(uu -> pulse_cost(uu, pulse,
d; kwargs...)[1], u)` plus `pulse_cost`'s own aux outputs -- mathematically
EXACT (not an approximation), and never touches `ForwardDiff`'s own
internal chunking machinery. Exploits that [`pulse_cost`](@ref)'s own
scalar `cost` is LITERALLY additively separable into three pieces that
never share any ODE-solve state:

    cost(u) = [-w_inv*inversion(u)]                             (the :ground-track solve, ALONE)
            + [w_sil*(silencing(u)-target_F)^2]                 (the :equator-track solve, ALONE)
            + [w_time*(duration(u)/T_max) + tmax_penalty(u) + power_penalty(u)]  (no ODE solve at all)

so `∇cost = ∇[ground term] + ∇[equator term] + ∇[direct term]` EXACTLY, by
linearity of differentiation -- each piece is handed to its own,
completely ordinary `ForwardDiff.gradient` call, using ForwardDiff's own
PUBLIC `GradientConfig`/`Chunk` API (never its internal seeding/extraction
machinery) to force a single chunk of width `min(60, length(u))` instead
of `ForwardDiff.pickchunksize`'s own default (capped at 12 regardless of
`n` -- e.g. 5 chunks for a 57-parameter gradient): each chunk re-solves
the SAME adaptive ODE integration from scratch (paying its own step-size-
control/primal-recomputation overhead again), so one wide chunk trades
that redundant-re-solve overhead for wider (more expensive per elementary
operation) `Dual` arithmetic -- a real, measured net win for THIS
ODE-solve-dominated cost (chunk width is purely a performance knob here,
never a correctness one -- see the `chunk` variable's own comment below
for the full reasoning and the `n_params > 60` EXACT-mode fallback). The
two EXPENSIVE terms (:ground, :equator -- each a full differentiated ODE
solve, the dominant cost of one [`run_local_adam`](@ref) epoch) are
ADDITIONALLY dispatched across 2 `Threads.@threads` iterations to run
concurrently -- the chunk-width change and the 2-way threading are
independent, stacking optimisations.

Thread safety verified directly (not assumed): `rhs_1st_order!`
(rhs_1st_order.jl) and everything [`build_E_of_t`](@ref)/
[`bspline_basis`](@ref) (composite_pulse.jl/bspline.jl/pulses.jl) touch
build fresh, per-call objects with no module-level `const`/`global`/`Ref`
mutable buffer (checked directly via grep across those files) -- each
thread's [`run_sim_1st_order_pure`](@ref) call constructs its own
`ODEProblem`/`u0`/closures from scratch, so there is no shared mutable
state for two concurrent solves to race on. This is also the SAME
concurrency pattern [`optimise_composite_pulse_over_k`](@ref) (this file)
already uses for N independent [`optimise_composite_pulse`](@ref) runs via
`Threads.@threads` -- this function applies the identical, already-proven
pattern one level deeper (per-epoch, across the 2 initial conditions,
rather than per-`k`).

Only activates real parallelism when Julia was started with
`-t N`/`JULIA_NUM_THREADS=N>=2` (same activation model
[`optimise_composite_pulse_over_k`](@ref)'s own docstring already
documents) -- with `Threads.nthreads()==1` this still runs correctly, just
serially (via `Threads.@threads`'s own single-thread fallback), so calling
this with no extra threads available is safe, just not faster. For
ODE-heavy work sharing a process with BLAS-using code, consider
`LinearAlgebra.BLAS.set_num_threads(1)` so Julia threads do not
oversubscribe physical cores (same caveat
[`optimise_composite_pulse_over_k`](@ref) already documents).

`w_inv<=0`/`w_sil<=0` skip that track's ODE solve entirely (matching
[`pulse_cost`](@ref)'s own exact behaviour), returning `0.0` for that
term/metric without spinning up a thread for it. A `PulseSolveFailed` on
EITHER track is caught per-thread and reproduces [`pulse_cost`](@ref)'s
own exact failure contract: `(fill(NaN, n), Inf, NaN, NaN, duration,
NaN)` -- `duration` still valid (computed before any solve, exactly as
`pulse_cost` does), `grad` unusable but present so this function's return
type stays fixed regardless. Any OTHER exception is rethrown (via
`Threads.@threads`'s own `@sync`, arriving at the caller wrapped in a
`TaskFailedException` rather than bare -- the one, narrow behavioural
difference from calling [`pulse_cost`](@ref) directly).

Verified numerically equivalent to calling
`ForwardDiff.gradient(uu -> pulse_cost(uu, pulse, d; kwargs...)[1], u)`
plus `pulse_cost(u, pulse, d; kwargs...)` separately, to machine precision
across multiple random `u`/seeds (see `test/runtests.jl`'s own "threaded
gradient matches serial" testset) -- this function's whole purpose is to
be a faster, EXACT substitute for that combined computation, never an
approximation of it.
"""
function _pulse_cost_grad_threaded(u::AbstractVector, pulse::CompositePulse, d;
                                    w_inv=1.0, w_sil=0.7, target_F=1.0, w_time=0.15, w_power=0.05, w_tmax=1.0,
                                    kwargs...)
    _forbid_initial_condition(kwargs)
    n = length(u)
    duration = pulse_duration(pulse, u)

    # Force a single ForwardDiff chunk covering ALL n_params whenever
    # n_params <= 60 -- this package's own established param_budget cap
    # (see fit_composite_pulse_seed_auto/points_per_segment_for_budget),
    # so this covers this function's own intended caller
    # (run_local_adam(threaded_grad=true)) by construction in the common
    # AUTO-mode case. ForwardDiff.pickchunksize's own DEFAULT caps chunk
    # width at 12 regardless of n, splitting e.g. a 57-parameter gradient
    # into 5 chunks -- 5 FULLY REDUNDANT re-solves of the SAME adaptive
    # ODE integration (only which directions carry non-zero Dual partials
    # differs between chunks), each paying the solver's own step-size-
    # control/primal-recomputation overhead again. A single width-60 (or
    # width-n when n<60) chunk pays that ODE-solver control-flow overhead
    # ONCE per initial condition instead. `min(60, n)` (never a bare `60`)
    # is required for CORRECTNESS, not just tidiness -- ForwardDiff errors
    # outright if the requested chunk size exceeds `length(x)` (verified
    # directly: `Chunk{60}` on a length-57 input raises `ArgumentError:
    # chunk size cannot be greater than ForwardDiff.structural_length(x)`)
    # -- EXACT-mode callers (fit_composite_pulse_seed_linear_exact-derived
    # shapes) do not go through param_budget at all and can produce
    # n_params > 60, where this clamp falls back to (multiple) width-60
    # chunks rather than erroring. Chunk size is purely a performance
    # knob, never a correctness one -- verified bit-for-bit identical to
    # ForwardDiff's own default chunking in `test/runtests.jl`'s "threaded
    # gradient matches serial" testset.
    chunk = ForwardDiff.Chunk{min(60, n)}()

    function direct_only(uu)
        dur = pulse_duration(pulse, uu)
        _, t_end, _, cA, _ = decode(pulse, uu)
        Tu = eltype(uu)
        tmax_excess = max(t_end[end] - pulse.T_max, zero(Tu))
        tmax_penalty = w_tmax * (tmax_excess / pulse.T_max)^2
        normalized_cA = cA ./ pulse.amp_scale
        power_penalty = w_power * (sum(abs2, normalized_cA) / length(normalized_cA))
        return w_time * (dur / pulse.T_max) + tmax_penalty + power_penalty
    end
    grad_direct = ForwardDiff.gradient(direct_only, u, ForwardDiff.GradientConfig(direct_only, u, chunk))
    direct_val = direct_only(u)

    aux_ground = Ref{NTuple{2,Float64}}((0.0, 0.0))    # (cost term value, inversion)
    aux_equator = Ref{NTuple{3,Float64}}((0.0, 0.0, 0.0))  # (cost term value, silencing, coherence)
    grads = Vector{Vector{Float64}}(undef, 2)
    failed = fill(false, 2)

    Threads.@threads for i in 1:2
        try
            if i == 1
                if w_inv > 0.0
                    function ground_only(uu)
                        _, _, Sz, Nj = run_sim_1st_order_pure(uu, pulse, d; kwargs..., initial_condition=:ground)
                        inv_ = _weighted_inversion(Sz, Nj, eltype(uu))
                        val = -w_inv * inv_
                        aux_ground[] = (Float64(ForwardDiff.value(val)), Float64(ForwardDiff.value(inv_)))
                        return val
                    end
                    grads[1] = ForwardDiff.gradient(ground_only, u, ForwardDiff.GradientConfig(ground_only, u, chunk))
                else
                    grads[1] = zeros(n)
                end
            else
                if w_sil > 0.0
                    function equator_only(uu)
                        _, Sp, _, Nj_eq = run_sim_1st_order_pure(uu, pulse, d; kwargs..., initial_condition=:equator)
                        Tu = eltype(uu)
                        sil_ = _weighted_silencing_factor(Sp, d.g_b, Nj_eq, Tu)
                        coh_ = _weighted_coherence(Sp, Nj_eq, Tu)
                        val = w_sil * (sil_ - convert(Tu, target_F))^2
                        aux_equator[] = (Float64(ForwardDiff.value(val)), Float64(ForwardDiff.value(sil_)), Float64(ForwardDiff.value(coh_)))
                        return val
                    end
                    grads[2] = ForwardDiff.gradient(equator_only, u, ForwardDiff.GradientConfig(equator_only, u, chunk))
                else
                    grads[2] = zeros(n)
                end
            end
        catch e
            e isa PulseSolveFailed || rethrow()
            failed[i] = true
        end
    end

    any(failed) && return fill(NaN, n), Inf, NaN, NaN, duration, NaN

    ground_val, inversion = aux_ground[]
    equator_val, silencing, coherence = aux_equator[]
    grad = grads[1] .+ grads[2] .+ grad_direct
    cost = ground_val + equator_val + direct_val
    return grad, cost, inversion, silencing, duration, coherence
end

# ============================================================
# LOCAL DESCENT WITH EARLY STOPPING + BASIN-HOPPING OUTER LOOP
#
# Same two-level structure as the Python port's _run_local_adam/
# optimise_composite_pulse: an early-stopped local Adam descent inside
# each "hop", wrapped in a random-restart-plus-Metropolis-acceptance
# outer loop (standard basin-hopping) with its own early stopping.
# ============================================================

"""
    run_local_adam(u_start, pulse, d, cost_kwargs; hop=0, num_epochs=30, patience=5, tol=1e-3, learning_rate=0.05, label="", threaded_grad=false, kwargs...) -> (best_u, best_cost, best_inversion, best_silencing, best_duration, history)

One basin's local descent: Adam from `u_start`, stopped either after
`num_epochs` or after `patience` consecutive epochs without a cost
improvement of at least `tol` (whichever comes first). Each epoch takes
cost, metrics, and gradient from a single `ForwardDiff.gradient` sweep
(metrics via `ForwardDiff.value`) -- or, when `threaded_grad=true`, from
[`_pulse_cost_grad_threaded`](@ref) instead: a mathematically EXACT,
task-parallel substitute (see that function's own docstring) that
dispatches the epoch's two independent ODE solves (`:ground`, `:equator`)
across `Threads.@threads`, for roughly a 2x wall-clock speedup on the
gradient step -- the dominant cost of an epoch at a large ensemble --
when Julia is started with `-t N`/`JULIA_NUM_THREADS=N>=2` (a no-op,
still-correct fallback to serial execution otherwise). Default `false`:
zero behaviour change unless a caller opts in. A failed ODE (`PulseSolveFailed`,
reported as `Inf` cost by [`pulse_cost`](@ref)) skips the Adam step,
reverts `u` to the last point that DID solve successfully, and halves the
step size for the next attempt -- a standard backtracking-line-search
response. Reverting alone (without also shrinking the step) would not
help: `Adam`'s own momentum (`state.m`/`state.v`) is untouched while a
failing step is skipped, so recomputing the gradient at the same reverted
point with that same frozen momentum would just reproduce the identical
failing step deterministically, looping until `patience` runs out for no
benefit. The step size regrows only GRADUALLY on success (`*1.5`, capped
at `learning_rate`), not snapped straight back -- snapping back
immediately has the identical looping failure mode, since the very next
epoch after a revert is just a re-confirmation of the same already-known-
good point: an instant full-strength reset there would hand `adam_step!`
the same gradient and the same frozen momentum that caused the failure in
the first place, stepping right back into it. Gradual regrowth lets the
step size settle at whatever scale actually stays inside the feasible
region near a hard boundary (e.g. repeatedly overshooting `pulse.T_max`),
so the basin can keep making progress up to that boundary instead of
stalling.

Retrying is NOT re-evaluated from scratch: `last_good_u` is the exact
point already solved (one ODE solve + AD gradient) the moment it was
first accepted, so every backoff retry from it reuses that cached
`(grad, cost, inversion, silencing, duration)` instead of re-running an
identical, deterministic, and potentially expensive (`MaxIters`-bound
before it even reports failure) computation for a point already known.
`Adam`'s own `(m, v, t)` are similarly snapshotted the moment `last_good_u`
is accepted and restored before every retry from it, so a retry blends
that one cached gradient into momentum exactly once, the same as an
ordinary step would -- not once per retry attempt (repeatedly blending an
identical gradient into the EMA on every backoff would let momentum drift
toward that single gradient direction, discarding the history a real
sequence of steps would have preserved). Returns this basin's own
best point -- the caller is responsible for tracking the GLOBAL best across
basins (a basin's local best is not necessarily better than a previous
basin's) -- plus `history`, a `Vector{<:NamedTuple}` with one entry per
epoch actually run
(`hop, epoch, k, cost, inversion, silencing, duration, coherence,
improved`), tagged with the caller-supplied `hop` index so a caller
accumulating history across many basins can tell which hop each row came
from. `k` (the sub-pulse count, from `pulse.k`) is recorded on every row
too, even though it's constant within a single call, so history rows stay
self-describing if ever concatenated across runs with a different `k`
(e.g. a later warm-started continuation using a different `CompositePulse`
shape). `coherence` is [`pulse_cost`](@ref)'s DIAGNOSTIC-ONLY per-bin
`|Sp|/(Nj/2)` average from the same `:equator` solve as `silencing` (see
[`_weighted_coherence`](@ref)) -- recorded for comparison alongside the
collective `|F|` actually being optimised, never part of `cost` itself.

`cf_lr_scale` (default `1.0`, i.e. no change from the original uniform-`lr`
behaviour) multiplies the effective step size for the CHIRP/frequency
coefficients (`raw_cf`) only -- `gap`/`duration`/`phi0`/`cA` always step at
the full `learning_rate` (`phi0` is a one-shot discrete phase jump, not an
integrated periodic quantity like `cf`, so the periodic-runaway motivation
below doesn't apply to it either). Motivation: `raw_cf` enters the physical drive
through an EXACT phase integral (`build_E_of_t`'s `Φ(t) = ∫f dτ`, via
`bspline_antiderivative`), and the cost depends on that phase only through
`exp(iΦ(t))` -- a periodic, non-convex function of `raw_cf`. `adam_step!`'s
own per-parameter second-moment normalisation already keeps every raw
parameter's step size close to `lr` in magnitude regardless of its raw
gradient scale, so this is NOT compensating for `cf`'s gradient being
larger or smaller than `gap`/`dur`/`cA`'s -- it is deliberately slowing
descent along the periodic sub-manifold specifically, so a single epoch's
step is less likely to carry `Φ(t)` across a `2π` boundary and land in an
entirely different (and possibly worse) local phase-alignment than the one
the rest of `u` was descending toward. Pass e.g. `cf_lr_scale=0.1` to
soften this; the right value is problem-dependent (how large `pulse.
freq_scale*duration` is relative to `2π` for your own config), so no
non-`1.0` value is asserted as a universal default here -- watch
`history`'s `cost`/`silencing` columns for erratic (non-monotone,
large-swing) epoch-to-epoch behaviour as the signal that a smaller
`cf_lr_scale` is worth trying.
"""
function run_local_adam(u_start::AbstractVector, pulse::CompositePulse, d, cost_kwargs::NamedTuple;
                         hop::Integer=0, num_epochs::Integer=30, patience::Integer=5, tol::Real=1e-3,
                         learning_rate::Real=0.05, cf_lr_scale::Real=1.0, label::AbstractString="",
                         threaded_grad::Bool=false, solve_kwargs...)
    _forbid_initial_condition(solve_kwargs)
    u = copy(u_start)
    n = length(u)
    adam = AdamState(n)
    lr_scale = cf_lr_scale == 1.0 ? nothing : pack(
        pulse, ones(pulse.k), ones(pulse.k), ones(pulse.k), ones(pulse.n_coeff_A, pulse.k),
        fill(cf_lr_scale, pulse.n_coeff_f, pulse.k),
    )
    aux = Ref{NTuple{5,Float64}}((NaN, NaN, NaN, NaN, NaN))
    function cost_only(uu)
        c, inv_, sil_, dur_, coh_ = pulse_cost(uu, pulse, d; cost_kwargs..., solve_kwargs...)
        aux[] = (
            Float64(ForwardDiff.value(c)),
            Float64(ForwardDiff.value(inv_)),
            Float64(ForwardDiff.value(sil_)),
            Float64(ForwardDiff.value(dur_)),
            Float64(ForwardDiff.value(coh_)),
        )
        return c
    end

    best_u = copy(u_start)
    best_cost, best_inv, best_sil, best_dur = Inf, 0.0, 0.0, 0.0
    epochs_since_improve = 0
    history = NamedTuple[]
    last_good_u = copy(u_start)
    last_good_grad = zeros(n)
    last_good_aux = (NaN, NaN, NaN, NaN, NaN)
    adam_m0 = zeros(n)
    adam_v0 = zeros(n)
    adam_t0 = 0
    lr = learning_rate
    just_reverted = false

    for epoch in 1:num_epochs
        t_wall = time()
        # After a revert, `u` is exactly `last_good_u` -- a point already
        # fully evaluated (one ODE solve + AD gradient) the first time it
        # was accepted. Re-running that identical, expensive, deterministic
        # computation again on every backoff retry would waste exactly the
        # work the retry loop is trying to make productive; reuse the
        # cached result instead. `adam.m`/`adam.v`/`adam.t` are also reset
        # to their snapshot from right when `last_good_u` was accepted, so
        # each retry blends the SAME cached gradient into momentum exactly
        # once (as a normal step would), rather than accumulating it again
        # on top of whatever the previous failed attempt already blended in.
        if just_reverted
            grad = last_good_grad
            cost, inv_, sil_, dur_, coh_ = last_good_aux
            adam.m .= adam_m0
            adam.v .= adam_v0
            adam.t = adam_t0
        elseif threaded_grad
            grad, cost, inv_, sil_, dur_, coh_ = _pulse_cost_grad_threaded(u, pulse, d; cost_kwargs..., solve_kwargs...)
        else
            grad = ForwardDiff.gradient(cost_only, u)
            cost, inv_, sil_, dur_, coh_ = aux[]
        end
        if !isfinite(cost)
            epochs_since_improve += 1
            push!(history, (hop=hop, epoch=epoch, k=pulse.k, cost=cost, inversion=inv_,
                             silencing=sil_, duration=dur_, coherence=coh_, improved=false))
            elapsed = time() - t_wall
            u .= last_good_u
            lr /= 2
            just_reverted = true
            println(
                "$label epoch $(lpad(epoch, 3)): $(round(elapsed, digits=1))s  " *
                "cost=Inf   ODE solve failed -- reverted to last valid point, " *
                "halved step size to $(round(lr, sigdigits=3))"
            )
            if epochs_since_improve >= patience
                println("$label early stop at epoch $epoch (no improvement > $tol for $patience epochs)")
                break
            end
            continue
        end

        lr = min(lr * 1.5, learning_rate)
        last_good_u .= u
        last_good_grad .= grad
        last_good_aux = (cost, inv_, sil_, dur_, coh_)
        adam_m0 .= adam.m
        adam_v0 .= adam.v
        adam_t0 = adam.t
        just_reverted = false

        improved = cost < best_cost - tol
        if improved
            best_cost, best_u = cost, copy(u)
            best_inv, best_sil, best_dur = inv_, sil_, dur_
            epochs_since_improve = 0
        else
            epochs_since_improve += 1
        end

        push!(history, (hop=hop, epoch=epoch, k=pulse.k, cost=cost, inversion=inv_,
                         silencing=sil_, duration=dur_, coherence=coh_, improved=improved))

        adam_step!(u, grad, adam; lr=lr, lr_scale=lr_scale)

        elapsed = time() - t_wall
        mark = improved ? "*" : " "
        println(
            "$label epoch $(lpad(epoch, 3)): $(round(elapsed, digits=1))s  " *
            "cost=$(round(cost, digits=4)) $mark inversion=$(round(inv_, digits=4)) " *
            "silencing=$(round(sil_, digits=4)) duration=$(round(dur_, sigdigits=4))s"
        )

        if epochs_since_improve >= patience
            println("$label early stop at epoch $epoch (no improvement > $tol for $patience epochs)")
            break
        end
    end

    return best_u, best_cost, best_inv, best_sil, best_dur, history
end

"""
    optimise_composite_pulse(k, n_coeff_A, n_coeff_f, d;
        num_epochs=30, learning_rate=0.05, patience=5, tol=1e-3,
        n_hops=3, hop_patience=2, hop_step_size=0.5, temperature=1.0,
        degree=3, taper_frac=0.1, w_tmax=1.0, w_power=0.05,
        w_inv=1.0, w_sil=0.7, target_F=1.0, w_time=0.15, seed=42,
        warm_start_u=nothing, label_prefix="", threaded_grad=false, solve_kwargs...)
        -> (best_u, best_cost, pulse::CompositePulse, u0, initial_metrics, history, final_metrics, optimizer_settings)

Basin-hopping global search over composite-pulse solutions for THIS
package's own 1st-order physics, each basin explored by
[`run_local_adam`](@ref)'s early-stopped Adam descent (or its
[`_pulse_cost_grad_threaded`](@ref) task-parallel substitute when
`threaded_grad=true` -- forwarded to every `run_local_adam` call, hop 0
and every subsequent hop; default `false`, zero behaviour change unless a
caller opts in; see that function's own docstring for what it does and
why it is mathematically exact, not an approximation):

  1. Run a local Adam descent from a random initial guess (hop 0).
  2. For each further hop: perturb the CURRENT accepted point with
     isotropic Gaussian noise (`hop_step_size`, in raw/pre-softplus
     units) to get a new basin's starting point, run a local Adam
     descent from there, then apply the standard basin-hopping
     Metropolis test -- always accept an improving basin, accept a worse
     one with probability `exp(-(cost_new-cost_old)/temperature)` -- to
     decide whether the NEXT hop perturbs from this new point or falls
     back to the previous one. The globally best `(u, cost)` seen over
     ALL hops is tracked separately and is what gets returned.
  3. Stop hopping early once `hop_patience` consecutive hops produce no
     global improvement `> tol`.

`d` is `prepare_derived(CONFIG)`'s own return value (build it once via
this package's existing `build_full_config`/`prepare_derived`, e.g. from
`SIM_SETTING`/`SYSTEM_CONFIG` NamedTuples the same way `run_sim_1st_order`
does). Each epoch differentiates through one full forward solve of
`rhs_1st_order!` via `ForwardDiff.gradient` -- genuinely expensive for a
fine ensemble/tight tolerance, same caveat the Python port's own
docstrings carry: `num_epochs`/`n_hops` default to modest smoke-test
budgets, not a converged global optimum.

`degree` / `taper_frac` are forwarded to [`CompositePulse`](@ref) (defaults
3 and 0.1, same as constructing the pulse by hand). `w_tmax`, `w_power`,
`w_inv`, `w_sil`, `target_F`, and `w_time` are forwarded to
[`pulse_cost`](@ref). Defaults keep both dual-trajectory weights on
(`w_inv=1`, `w_sil=0.7`) targeting `target_F=1.0` (RASE-style revival;
pass `target_F=0.0` for ROSE-style silencing instead). These are explicit
keywords so they are NOT passed through to the ODE solver. Do not pass
`initial_condition` — the cost fixes `:ground` and `:equator` itself.

Besides the optimised `(best_u, best_cost, pulse)`, also returns: `u0`
(the initial/candidate parameterisation hop 0 actually started from --
either a fresh random guess or `warm_start_u`, see below),
`initial_metrics` (`(cost, inversion, silencing, duration, coherence)` at
`u0`, from [`pulse_cost`](@ref) -- `coherence` is diagnostic only, never
part of `cost`), `history` -- the [`run_local_adam`](@ref) per-epoch
log from EVERY hop, concatenated in run order (each row tagged with its
own `hop`/`epoch`) -- `final_metrics` (the same `(cost, inversion,
silencing, duration, coherence)` shape, for `best_u` specifically -- NOT necessarily
`history[end]`, since `best_u` can come from an earlier epoch than the
last one run), and `optimizer_settings`, a `NamedTuple` of every setting
that actually affected this run: `k`/`n_coeff_A`/`n_coeff_f`/`degree`/
`taper_frac` plus every one of this function's own explicit keyword
arguments (`num_epochs`, `learning_rate`, `patience`, `tol`, `n_hops`,
`hop_patience`, `hop_step_size`, `temperature`, `w_tmax`, `w_power`,
`w_inv`, `w_sil`, `target_F`, `w_time`, `seed`), plus
any of `solve_kwargs` whose value isn't a `Function` (so e.g. a numeric
`reltol`/`abstol`/`w_inv`/`w_sil`/`w_time` override is captured, while a
non-serialisable closure like `signal_E_of_t` is deliberately excluded --
that one is captured separately, as `use_signal`/`n_signal`, by
[`optimise_control_pulse_from_jld2`](@ref), since those two scalars are
enough to rebuild the exact same closure deterministically). All of
these are exactly what [`optimise_control_pulse_from_jld2`](@ref) needs
to write a full, replicable run log; ordinary callers that only want the
optimised pulse can simply ignore the extra return values.

`warm_start_u`: if given (e.g. a previous run's saved `final_u`, from a
loaded `_optrunlog.jld2` -- see [`load_jld2_run`](@ref)), hop 0 starts
from THIS point instead of a fresh `initial_guess(pulse; seed=seed)`, so
a later call can pick up and continue optimising from where an earlier
one left off. Must have length `n_params(pulse)`, i.e. be a raw parameter
vector for a `CompositePulse` with the SAME `(k, n_coeff_A, n_coeff_f)`
as this call's -- an error is raised otherwise, since a length mismatch
would silently decode into a nonsensical pulse rather than fail loudly.

`cf_lr_scale` (default `1.0`) is forwarded unchanged to every
[`run_local_adam`](@ref) call (hop 0 and every subsequent hop) -- see that
function's own docstring for what it does and why.
"""
function optimise_composite_pulse(
    k::Integer, n_coeff_A::Integer, n_coeff_f::Integer, d;
    num_epochs::Integer=30, learning_rate::Real=0.05, cf_lr_scale::Real=1.0, patience::Integer=5, tol::Real=1e-3,
    n_hops::Integer=3, hop_patience::Integer=2, hop_step_size::Real=0.5, temperature::Real=1.0,
    degree::Integer=3, taper_frac::Real=0.1, w_tmax::Real=1.0, w_power::Real=0.05,
    w_inv::Real=1.0, w_sil::Real=0.7, target_F::Real=1.0, w_time::Real=0.15,
    seed::Integer=42, warm_start_u=nothing, label_prefix::AbstractString="",
    threaded_grad::Bool=false, solve_kwargs...,
)
    _forbid_initial_condition(solve_kwargs)
    pulse = CompositePulse(k, n_coeff_A, n_coeff_f, d; degree=degree, taper_frac=taper_frac)
    cost_kwargs = (w_tmax=w_tmax, w_power=w_power, w_inv=w_inv, w_sil=w_sil, target_F=target_F, w_time=w_time)
    rng = Random.Xoshiro(seed)

    # threaded_grad is deliberately kept OUT of solve_kwargs (unlike every
    # other extra keyword here): solve_kwargs also flows straight into
    # pulse_cost's own plain initial_metrics/final_metrics calls below,
    # which forward it to run_sim_1st_order_pure -- a function with NO
    # catch-all kwargs..., so an unrecognised keyword there is a hard
    # MethodError, not a silent no-op. Only run_local_adam (which DOES
    # know about threaded_grad) should ever see it.
    solve_settings = NamedTuple(kv for kv in pairs(solve_kwargs) if !(kv[2] isa Function))
    optimizer_settings = merge(
        (k=k, n_coeff_A=n_coeff_A, n_coeff_f=n_coeff_f, degree=degree, taper_frac=taper_frac,
         num_epochs=num_epochs, learning_rate=learning_rate, cf_lr_scale=cf_lr_scale, patience=patience, tol=tol,
         n_hops=n_hops, hop_patience=hop_patience, hop_step_size=hop_step_size, temperature=temperature,
         w_tmax=w_tmax, w_power=w_power, w_inv=w_inv, w_sil=w_sil, target_F=target_F, w_time=w_time, seed=seed,
         threaded_grad=threaded_grad),
        solve_settings,
    )

    println(
        "$(label_prefix)Optimising k=$k pulses, $(n_params(pulse)) raw parameters (ForwardDiff/Adam + " *
        "basin-hopping, physics: InhomogeneousSpinCavityDynamics.jl rhs_1st_order!) ..."
    )

    if warm_start_u === nothing
        u0 = initial_guess(pulse; seed=seed)
    else
        length(warm_start_u) == n_params(pulse) || error(
            "warm_start_u has length $(length(warm_start_u)), but this CompositePulse " *
            "(k=$k, n_coeff_A=$n_coeff_A, n_coeff_f=$n_coeff_f) needs $(n_params(pulse))."
        )
        u0 = collect(Float64, warm_start_u)
        println("$(label_prefix)Warm-starting hop 0 from a supplied raw vector.")
    end
    initial_metrics = pulse_cost(u0, pulse, d; cost_kwargs..., solve_kwargs...)
    history = NamedTuple[]

    current_u, current_cost, _, _, _, hop0_history = run_local_adam(
        u0, pulse, d, cost_kwargs; hop=0, num_epochs, patience, tol, learning_rate, cf_lr_scale,
        label="$(label_prefix)[hop 0]", threaded_grad=threaded_grad, solve_kwargs...
    )
    append!(history, hop0_history)
    global_best_u, global_best_cost = current_u, current_cost
    hops_since_improve = 0

    for hop in 1:(n_hops-1)
        perturbation = hop_step_size .* randn(rng, length(current_u))
        candidate_u0 = current_u .+ perturbation

        cand_u, cand_cost, _, _, _, hop_history = run_local_adam(
            candidate_u0, pulse, d, cost_kwargs;
            hop, num_epochs, patience, tol, learning_rate, cf_lr_scale,
            label="$(label_prefix)[hop $hop]", threaded_grad=threaded_grad, solve_kwargs...,
        )
        append!(history, hop_history)

        if cand_cost < global_best_cost - tol
            global_best_u, global_best_cost = cand_u, cand_cost
            hops_since_improve = 0
        else
            hops_since_improve += 1
        end

        delta = cand_cost - current_cost
        accept = delta < 0.0 || rand(rng) < exp(-delta / max(temperature, 1e-12))
        if accept
            current_u, current_cost = cand_u, cand_cost
        end

        accept_str = accept ? "accepted" : "rejected"
        println(
            "$(label_prefix)hop $hop: local best cost=$(round(cand_cost, digits=4)) " *
            "($accept_str as new basin, delta=$(round(delta, digits=4))) " *
            "global best cost=$(round(global_best_cost, digits=4))"
        )

        if hops_since_improve >= hop_patience
            println("$(label_prefix)Basin-hopping stopped early after $hop hops (no global improvement > $tol for $hop_patience hops).")
            break
        end
    end

    final_metrics = pulse_cost(global_best_u, pulse, d; cost_kwargs..., solve_kwargs...)

    println("$(label_prefix)Optimisation complete. Global best cost: $(round(global_best_cost, digits=4)).")
    return global_best_u, global_best_cost, pulse, u0, initial_metrics, history, final_metrics, optimizer_settings
end

function _normalise_k_specs(kinds, specs)
    if specs !== nothing
        return collect(specs)
    end
    return [(k_of_seed_kind(kind), kind) for kind in kinds]
end

"""
    optimise_composite_pulse_over_k(n_coeff_A, n_coeff_f, d;
        kinds=(:hs1, :corpse, :bb1), specs=nothing, threaded=true,
        Omega_max=nothing, beta=nothing, mu=nothing, seed=42,
        optimizer_kwargs...)
        -> NamedTuple

Discrete search over sub-pulse count `k`. `k` is not a continuous
decision variable: each `k` gets its own `CompositePulse`, a canonical
warm-start ([`seed_canonical`](@ref)), and an independent
[`optimise_composite_pulse`](@ref) run on that pulse's continuous raw
parameters. The runs are independent and are threaded when
`threaded=true` and `Threads.nthreads() > 1` (start Julia with
`JULIA_NUM_THREADS=N`; for ODE-heavy work consider
`BLAS.set_num_threads(1)` so threads do not oversubscribe).

Default `kinds` is HS1 (`k=1`), CORPSE (`k=5`), BB1 (`k=7`). Pass
`specs=((k, kind), ...)` to choose `k` and seed explicitly, including
`:random` for `initial_guess` (e.g. `specs=((3, :random), (5, :corpse))`).

`Omega_max` defaults to each pulse's own `amp_scale` (cavity-input
units). HS1 `beta`/`mu` default as in [`seed_canonical`](@ref).
`optimizer_kwargs` are forwarded to [`optimise_composite_pulse`](@ref)
(`num_epochs`, `signal_E_of_t`, `w_tmax`, ...). Do not pass
`warm_start_u` -- the seed is built per `k`.

Returns a NamedTuple: `best_kind`, `best_k`, `best_u`, `best_cost`,
`pulse`, `u0`, `initial_metrics`, `history`, `final_metrics`,
`optimizer_settings` (the winning run, same payload as
[`optimise_composite_pulse`](@ref) plus kind), and `per_k` (one
NamedTuple per spec, in input order).
"""
function optimise_composite_pulse_over_k(
    n_coeff_A::Integer, n_coeff_f::Integer, d;
    kinds=(:hs1, :corpse, :bb1),
    specs=nothing,
    threaded::Bool=true,
    Omega_max=nothing,
    beta=nothing,
    mu=nothing,
    seed::Integer=42,
    optimizer_kwargs...,
)
    :warm_start_u in keys(optimizer_kwargs) && error(
        "optimise_composite_pulse_over_k builds a per-k canonical seed; do not pass warm_start_u."
    )
    _forbid_initial_condition(optimizer_kwargs)

    job_specs = _normalise_k_specs(kinds, specs)
    isempty(job_specs) && error("No (k, kind) specs to optimise.")
    seen = Dict{Int,Symbol}()
    for (k, kind) in job_specs
        k isa Integer && k >= 1 || error("k must be a positive integer, got $k.")
        kind isa Symbol || error("kind must be a Symbol, got $kind.")
        haskey(seen, k) && error("Duplicate k=$k (kinds $(seen[k]) and $kind). Each k can run once.")
        seen[k] = kind
        if kind !== :random
            k_of_seed_kind(kind) == k || error(
                "kind $kind requires k=$(k_of_seed_kind(kind)), got k=$k."
            )
        end
    end

    n = length(job_specs)
    nthreads = Threads.nthreads()
    use_threads = threaded && nthreads > 1 && n > 1
    println(
        "Discrete-k search: $n independent continuous optimisations " *
        (use_threads ? "on $nthreads threads" : "serially") *
        ". kinds=$(collect(spec[2] for spec in job_specs))."
    )

    function run_spec(spec)
        k, kind = spec
        prefix = "[$kind k=$k] "
        deg = get(optimizer_kwargs, :degree, 3)
        tfrac = get(optimizer_kwargs, :taper_frac, 0.1)
        pulse_seed = CompositePulse(k, n_coeff_A, n_coeff_f, d; degree=deg, taper_frac=tfrac)
        Ω = Omega_max === nothing ? pulse_seed.amp_scale : Omega_max
        u0 = seed_canonical(pulse_seed, kind; Omega_max=Ω, beta=beta, mu=mu, seed=seed)
        best_u, best_cost, pulse_out, u0_out, initial_metrics, history, final_metrics, optimizer_settings =
            optimise_composite_pulse(
                k, n_coeff_A, n_coeff_f, d;
                optimizer_kwargs...,
                seed=seed + 1000 * Int(k),
                warm_start_u=u0,
                label_prefix=prefix,
            )
        optimizer_settings = merge(optimizer_settings, (seed_kind=kind,))
        return (
            kind=kind, k=Int(k),
            best_u=best_u, best_cost=best_cost, pulse=pulse_out,
            u0=u0_out, initial_metrics=initial_metrics, history=history,
            final_metrics=final_metrics, optimizer_settings=optimizer_settings,
        )
    end

    per_k = Vector{NamedTuple}(undef, n)
    if use_threads
        Threads.@threads for i in 1:n
            per_k[i] = run_spec(job_specs[i])
        end
    else
        for i in 1:n
            per_k[i] = run_spec(job_specs[i])
        end
    end

    best = per_k[1]
    for r in per_k
        if r.best_cost < best.best_cost
            best = r
        end
    end

    println("Discrete-k search complete.")
    for r in per_k
        mark = r.k == best.k ? "*" : " "
        println(
            "  $mark k=$(r.k) $(r.kind): cost=$(round(r.best_cost, digits=4))"
        )
    end
    println("  winner: k=$(best.k) $(best.kind)  cost=$(round(best.best_cost, digits=4))")

    return (
        best_kind=best.kind, best_k=best.k,
        best_u=best.best_u, best_cost=best.best_cost, pulse=best.pulse,
        u0=best.u0, initial_metrics=best.initial_metrics, history=best.history,
        final_metrics=best.final_metrics, optimizer_settings=best.optimizer_settings,
        per_k=per_k,
    )
end

