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
    t_start, t_end, _, _ = decode(pulse, u)
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
    _, t_end, cA, _ = decode(pulse, u)

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
    adam_step!(u, grad, state::AdamState; lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8)

One in-place Adam update of `u` given `grad = ∇cost(u)`, mutating `state`.
Standard bias-corrected Adam (Kingma & Ba 2015).
"""
function adam_step!(u::AbstractVector, grad::AbstractVector, state::AdamState;
                     lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8)
    state.t += 1
    @inbounds for i in eachindex(u)
        state.m[i] = beta1 * state.m[i] + (1 - beta1) * grad[i]
        state.v[i] = beta2 * state.v[i] + (1 - beta2) * grad[i]^2
        m_hat = state.m[i] / (1 - beta1^state.t)
        v_hat = state.v[i] / (1 - beta2^state.t)
        u[i] -= lr * m_hat / (sqrt(v_hat) + eps)
    end
    return u
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
    run_local_adam(u_start, pulse, d, cost_kwargs; hop=0, num_epochs=30, patience=5, tol=1e-3, learning_rate=0.05, label="", kwargs...) -> (best_u, best_cost, best_inversion, best_silencing, best_duration, history)

One basin's local descent: Adam from `u_start`, stopped either after
`num_epochs` or after `patience` consecutive epochs without a cost
improvement of at least `tol` (whichever comes first). Each epoch takes
cost, metrics, and gradient from a single `ForwardDiff.gradient` sweep
(metrics via `ForwardDiff.value`). A failed ODE (`PulseSolveFailed`,
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
"""
function run_local_adam(u_start::AbstractVector, pulse::CompositePulse, d, cost_kwargs::NamedTuple;
                         hop::Integer=0, num_epochs::Integer=30, patience::Integer=5, tol::Real=1e-3,
                         learning_rate::Real=0.05, label::AbstractString="", solve_kwargs...)
    _forbid_initial_condition(solve_kwargs)
    u = copy(u_start)
    n = length(u)
    adam = AdamState(n)
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

        adam_step!(u, grad, adam; lr=lr)

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
        warm_start_u=nothing, label_prefix="", solve_kwargs...)
        -> (best_u, best_cost, pulse::CompositePulse, u0, initial_metrics, history, final_metrics, optimizer_settings)

Basin-hopping global search over composite-pulse solutions for THIS
package's own 1st-order physics, each basin explored by
[`run_local_adam`](@ref)'s early-stopped Adam descent:

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
"""
function optimise_composite_pulse(
    k::Integer, n_coeff_A::Integer, n_coeff_f::Integer, d;
    num_epochs::Integer=30, learning_rate::Real=0.05, patience::Integer=5, tol::Real=1e-3,
    n_hops::Integer=3, hop_patience::Integer=2, hop_step_size::Real=0.5, temperature::Real=1.0,
    degree::Integer=3, taper_frac::Real=0.1, w_tmax::Real=1.0, w_power::Real=0.05,
    w_inv::Real=1.0, w_sil::Real=0.7, target_F::Real=1.0, w_time::Real=0.15,
    seed::Integer=42, warm_start_u=nothing, label_prefix::AbstractString="", solve_kwargs...,
)
    _forbid_initial_condition(solve_kwargs)
    pulse = CompositePulse(k, n_coeff_A, n_coeff_f, d; degree=degree, taper_frac=taper_frac)
    cost_kwargs = (w_tmax=w_tmax, w_power=w_power, w_inv=w_inv, w_sil=w_sil, target_F=target_F, w_time=w_time)
    rng = Random.Xoshiro(seed)

    solve_settings = NamedTuple(kv for kv in pairs(solve_kwargs) if !(kv[2] isa Function))
    optimizer_settings = merge(
        (k=k, n_coeff_A=n_coeff_A, n_coeff_f=n_coeff_f, degree=degree, taper_frac=taper_frac,
         num_epochs=num_epochs, learning_rate=learning_rate, patience=patience, tol=tol,
         n_hops=n_hops, hop_patience=hop_patience, hop_step_size=hop_step_size, temperature=temperature,
         w_tmax=w_tmax, w_power=w_power, w_inv=w_inv, w_sil=w_sil, target_F=target_F, w_time=w_time, seed=seed),
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
        u0, pulse, d, cost_kwargs; hop=0, num_epochs, patience, tol, learning_rate,
        label="$(label_prefix)[hop 0]", solve_kwargs...
    )
    append!(history, hop0_history)
    global_best_u, global_best_cost = current_u, current_cost
    hops_since_improve = 0

    for hop in 1:(n_hops-1)
        perturbation = hop_step_size .* randn(rng, length(current_u))
        candidate_u0 = current_u .+ perturbation

        cand_u, cand_cost, _, _, _, hop_history = run_local_adam(
            candidate_u0, pulse, d, cost_kwargs;
            hop, num_epochs, patience, tol, learning_rate,
            label="$(label_prefix)[hop $hop]", solve_kwargs...,
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

