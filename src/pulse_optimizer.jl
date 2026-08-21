# ============================================================
# DIFFERENTIABLE PULSE OPTIMISATION
#
# Julia port of InhomogeneousSpinCavityDynamics.py/pulse_optimized_spline.py.
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
    if initial_condition == :ground
        u0[idx1_Sz_start(M):end] .= .-Nj ./ 2
    elseif initial_condition == :inverted
        u0[idx1_Sz_start(M):end] .= Nj ./ 2
    elseif initial_condition == :custom
        # already zero
    else
        error("Unknown initial_condition = $(initial_condition). Use :ground, :inverted, or :custom.")
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

"""
    pulse_metrics(u, pulse, d; kwargs...) -> (inversion, coherence)

Runs [`run_sim_1st_order_pure`](@ref) and reduces the final state to two
scalars in `[0, 1]`, higher = better:
- `inversion`: `Nj`-population-weighted mean fractional population
  inversion (`real(Sz)/(Nj/2)` mapped from `[-1,1]` to `[0,1]`).
- `coherence`: `Nj`-population-weighted mean normalised `|Sp|`.

`Nj`-weighting (`Nj ./ sum(Nj)`, rather than the 1D `p_delta` this
package's `prepare_derived` also returns) is used so this generalises
correctly whether `M_g == 1` or not, without needing to reshape the
flattened `(M_delta*M_g,)` bins back to `(M_delta, M_g)`.

`|Sp|` is computed as `sqrt(abs2(Sp) + 1e-30)`, not `abs(Sp)`: `abs` of a
complex Dual has a removable-singularity derivative at exactly `z=0`
(`d|z| = (re*d(re)+im*d(im))/|z|`, i.e. `0/0`), and `Sp` lands exactly at
`0` (to every order) whenever the composite pulse's active window has no
overlap with `d.timespan` at all -- e.g. an optimiser step pushed far
enough past `pulse.T_max` that nothing in `run_sim_1st_order_pure` ever
sees a nonzero drive. `ForwardDiff.gradient` returns `NaN` there
otherwise, which would corrupt the whole `u` vector on the next `Adam`
step even though every other term (including the `w_tmax` penalty
pulling back toward the window) still has a perfectly good gradient at
that point. The `1e-30` epsilon matches the one already used below for
the `Nj/2` denominator and is negligible at any `Sp` scale this ensemble
actually produces.
"""
function pulse_metrics(u::AbstractVector, pulse::CompositePulse, d; kwargs...)
    _, Sp, Sz, Nj = run_sim_1st_order_pure(u, pulse, d; kwargs...)
    T = eltype(u)
    weight = Nj ./ sum(Nj)
    Sz_fraction = real.(Sz) ./ (Nj ./ 2 .+ 1e-30)
    inversion = sum(weight .* clamp.((Sz_fraction .+ 1) ./ 2, zero(T), one(T)))
    Sp_abs = sqrt.(abs2.(Sp) .+ 1e-30)
    Sp_fraction = Sp_abs ./ (Nj ./ 2 .+ 1e-30)
    coherence = sum(weight .* clamp.(Sp_fraction, zero(T), one(T)))
    return inversion, coherence
end

"""
    pulse_cost(u, pulse, d; w_inv=1.0, w_coh=0.7, w_time=0.15, w_tmax=1.0, kwargs...) -> (cost, inversion, coherence, duration)

Scalar cost to be minimised:

    J = -w_inv*inversion - w_coh*coherence + w_time*(duration/T_max)
        + w_tmax*max(t_end[end]-T_max, 0)^2/T_max^2

Both duration-related terms are normalised by `pulse.T_max` so they sit
on the same `O(1)` scale as `inversion`/`coherence`, rather than in
seconds (`~1e-4`-`1e-7` for these pulses): `w_time*duration` alone (no
`/T_max`) is numerically inert at any sane `w_time` -- it never
meaningfully competed with the other two terms for gradient signal, so
in practice it did nothing.

`w_tmax` is ON by default (`1.0`, not `0`): [`pulse_duration`](@ref) is
measured from `t=0` (so a longer leading gap already raises the
`w_time*duration` term), but nothing else stops `t_end[end]` drifting
past `pulse.T_max` during optimisation, which is silently truncated in
`run_sim_1st_order_pure` (only `d.timespan` is ever integrated) --
`w_tmax`'s quadratic penalty is what actually keeps the optimiser inside
the simulated window; without it there is nothing but the (much softer)
`w_time` pressure standing between Adam and a pulse the ODE never fully
sees. Set `w_tmax=0` to disable it and go back to the un-penalised
behaviour.

If the ODE solve fails, this returns `Inf` for the cost rather than
using a partial `sol.u[end]`. Direct callers of
[`run_sim_1st_order_pure`](@ref) still see a thrown
`PulseSolveFailed`.
"""
function pulse_cost(u::AbstractVector, pulse::CompositePulse, d;
                     w_inv=1.0, w_coh=0.7, w_time=0.15, w_tmax=1.0, kwargs...)
    T = eltype(u)
    duration = pulse_duration(pulse, u)
    _, t_end, _, _ = decode(pulse, u)
    tmax_excess = max(t_end[end] - pulse.T_max, zero(T))
    tmax_penalty = w_tmax * (tmax_excess / pulse.T_max)^2

    inversion, coherence = try
        pulse_metrics(u, pulse, d; kwargs...)
    catch e
        e isa PulseSolveFailed || rethrow()
        infT = convert(T, Inf)
        nanT = convert(T, NaN)
        return infT, nanT, nanT, duration
    end

    cost = -w_inv * inversion - w_coh * coherence + w_time * (duration / pulse.T_max) + tmax_penalty
    return cost, inversion, coherence, duration
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
    run_local_adam(u_start, pulse, d, cost_kwargs; hop=0, num_epochs=30, patience=5, tol=1e-3, learning_rate=0.05, label="", kwargs...) -> (best_u, best_cost, best_inversion, best_coherence, best_duration, history)

One basin's local descent: Adam from `u_start`, stopped either after
`num_epochs` or after `patience` consecutive epochs without a cost
improvement of at least `tol` (whichever comes first). Each epoch takes
cost, metrics, and gradient from a single `ForwardDiff.gradient` sweep
(metrics via `ForwardDiff.value`). A failed ODE (`PulseSolveFailed`,
reported as `Inf` cost by [`pulse_cost`](@ref)) skips the Adam step
rather than differentiating `sol.u[end]`. Returns this basin's own best
point -- the caller is responsible for tracking the GLOBAL best across
basins (a basin's local best is not necessarily better than a previous
basin's) -- plus `history`, a `Vector{<:NamedTuple}` with one entry per
epoch actually run
(`hop, epoch, k, cost, inversion, coherence, duration, improved`),
tagged with the caller-supplied `hop` index so a caller accumulating
history across many basins can tell which hop each row came from. `k`
(the sub-pulse count, from `pulse.k`) is recorded on every row too, even
though it's constant within a single call, so history rows stay
self-describing if ever concatenated across runs with a different `k`
(e.g. a later warm-started continuation using a different `CompositePulse`
shape).
"""
function run_local_adam(u_start::AbstractVector, pulse::CompositePulse, d, cost_kwargs::NamedTuple;
                         hop::Integer=0, num_epochs::Integer=30, patience::Integer=5, tol::Real=1e-3,
                         learning_rate::Real=0.05, label::AbstractString="", solve_kwargs...)
    u = copy(u_start)
    n = length(u)
    adam = AdamState(n)
    aux = Ref{NTuple{4,Float64}}((NaN, NaN, NaN, NaN))
    function cost_only(uu)
        c, inv_, coh_, dur_ = pulse_cost(uu, pulse, d; cost_kwargs..., solve_kwargs...)
        aux[] = (
            Float64(ForwardDiff.value(c)),
            Float64(ForwardDiff.value(inv_)),
            Float64(ForwardDiff.value(coh_)),
            Float64(ForwardDiff.value(dur_)),
        )
        return c
    end

    best_u = copy(u_start)
    best_cost, best_inv, best_coh, best_dur = Inf, 0.0, 0.0, 0.0
    epochs_since_improve = 0
    history = NamedTuple[]

    for epoch in 1:num_epochs
        t_wall = time()
        grad = ForwardDiff.gradient(cost_only, u)
        cost, inv_, coh_, dur_ = aux[]

        if !isfinite(cost)
            epochs_since_improve += 1
            push!(history, (hop=hop, epoch=epoch, k=pulse.k, cost=cost, inversion=inv_,
                             coherence=coh_, duration=dur_, improved=false))
            elapsed = time() - t_wall
            println(
                "$label epoch $(lpad(epoch, 3)): $(round(elapsed, digits=1))s  " *
                "cost=Inf   ODE solve failed -- skipping Adam step"
            )
            if epochs_since_improve >= patience
                println("$label early stop at epoch $epoch (no improvement > $tol for $patience epochs)")
                break
            end
            continue
        end

        improved = cost < best_cost - tol
        if improved
            best_cost, best_u = cost, copy(u)
            best_inv, best_coh, best_dur = inv_, coh_, dur_
            epochs_since_improve = 0
        else
            epochs_since_improve += 1
        end

        push!(history, (hop=hop, epoch=epoch, k=pulse.k, cost=cost, inversion=inv_,
                         coherence=coh_, duration=dur_, improved=improved))

        adam_step!(u, grad, adam; lr=learning_rate)

        elapsed = time() - t_wall
        mark = improved ? "*" : " "
        println(
            "$label epoch $(lpad(epoch, 3)): $(round(elapsed, digits=1))s  " *
            "cost=$(round(cost, digits=4)) $mark inversion=$(round(inv_, digits=4)) " *
            "coherence=$(round(coh_, digits=4)) duration=$(round(dur_, sigdigits=4))s"
        )

        if epochs_since_improve >= patience
            println("$label early stop at epoch $epoch (no improvement > $tol for $patience epochs)")
            break
        end
    end

    return best_u, best_cost, best_inv, best_coh, best_dur, history
end

"""
    optimise_composite_pulse(k, n_coeff_A, n_coeff_f, d;
        num_epochs=30, learning_rate=0.05, patience=5, tol=1e-3,
        n_hops=3, hop_patience=2, hop_step_size=0.5, temperature=1.0,
        degree=3, taper_frac=0.1, w_tmax=1.0, seed=42, warm_start_u=nothing,
        label_prefix="", solve_kwargs...)
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
3 and 0.1, same as constructing the pulse by hand). `w_tmax` is forwarded
to [`pulse_cost`](@ref) and defaults to `1.0`, matching that function's
own default (ON: penalise `t_end[end]` drifting past `pulse.T_max`) --
pass `w_tmax=0` here to disable it. These three are explicit keywords so
they are NOT passed through to the ODE solver.

Besides the optimised `(best_u, best_cost, pulse)`, also returns: `u0`
(the initial/candidate parameterisation hop 0 actually started from --
either a fresh random guess or `warm_start_u`, see below),
`initial_metrics` (`(cost, inversion, coherence, duration)` at `u0`, from
[`pulse_cost`](@ref)), `history` -- the [`run_local_adam`](@ref) per-epoch
log from EVERY hop, concatenated in run order (each row tagged with its
own `hop`/`epoch`) -- `final_metrics` (the same `(cost, inversion,
coherence, duration)` shape, for `best_u` specifically -- NOT necessarily
`history[end]`, since `best_u` can come from an earlier epoch than the
last one run), and `optimizer_settings`, a `NamedTuple` of every setting
that actually affected this run: `k`/`n_coeff_A`/`n_coeff_f`/`degree`/
`taper_frac` plus every one of this function's own explicit keyword
arguments (`num_epochs`, `learning_rate`, `patience`, `tol`, `n_hops`,
`hop_patience`, `hop_step_size`, `temperature`, `w_tmax`, `seed`), plus
any of `solve_kwargs` whose value isn't a `Function` (so e.g. a numeric
`reltol`/`abstol`/`w_inv`/`w_coh`/`w_time` override is captured, while a
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
    degree::Integer=3, taper_frac::Real=0.1, w_tmax::Real=1.0,
    seed::Integer=42, warm_start_u=nothing, label_prefix::AbstractString="", solve_kwargs...,
)
    pulse = CompositePulse(k, n_coeff_A, n_coeff_f, d; degree=degree, taper_frac=taper_frac)
    cost_kwargs = (w_tmax=w_tmax,)
    rng = Random.Xoshiro(seed)

    # Every setting that actually affects this run, for a replicable log
    # entry -- see this function's own docstring. solve_kwargs is filtered
    # to non-Function values only, so a numeric override (reltol, abstol,
    # w_inv, ...) is captured but a closure (signal_E_of_t) is not.
    solve_settings = NamedTuple(kv for kv in pairs(solve_kwargs) if !(kv[2] isa Function))
    optimizer_settings = merge(
        (k=k, n_coeff_A=n_coeff_A, n_coeff_f=n_coeff_f, degree=degree, taper_frac=taper_frac,
         num_epochs=num_epochs, learning_rate=learning_rate, patience=patience, tol=tol,
         n_hops=n_hops, hop_patience=hop_patience, hop_step_size=hop_step_size, temperature=temperature,
         w_tmax=w_tmax, seed=seed),
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

