# ============================================================
# Discrete Tsit5 adjoint of the dual-trajectory pulse cost.
#
# Additive: does not replace ForwardDiff. Default Adam remains Dual.
# Forward state stays ComplexF64; the real split is only inside the VJP.
# signal_E_of_t is not differentiated (same contract as Dual).
# ============================================================

"""
    inversion_pullback!(λx, Sz, Nj)

Seeds `λx` with the RAW `∂inversion/∂Re(Sz_j)` gradient (i.e. the pullback
of [`_weighted_inversion`](@ref) alone, coefficient `+1` -- NOT weighted by
any `w_inv`, and not the pullback of a `cost` that also depends on
`silencing`). [`pulse_cost`](@ref)'s `physics_cost = (1 -
inversion*silencing_success)^2` couples the two dual-trajectory tracks
multiplicatively, so it has no well-defined single-track cost gradient any
more; [`pulse_cost_grad_adjoint`](@ref) combines this RAW `∇inversion`
with [`silencing_pullback!`](@ref)'s RAW `∇silencing` via the same
analytical chain rule [`_pulse_cost_grad_threaded`](@ref) uses, only AFTER
both tracks' adjoint sweeps have produced their own independent Jacobian.
"""
function inversion_pullback!(λx::AbstractVector, Sz::AbstractVector, Nj::AbstractVector)
    M = length(Nj)
    length(Sz) == M || error("inversion_pullback!: Sz length $(length(Sz)) != Nj length $M.")
    length(λx) == real_state_length_1st_order(M) || error(
        "inversion_pullback!: λx length $(length(λx)) != $(real_state_length_1st_order(M))."
    )
    fill!(λx, 0)
    wsum = sum(Nj)
    @inbounds for j in 1:M
        den = Nj[j] / 2 + 1e-30
        frac = real(Sz[j]) / den
        s = (frac + 1) / 2
        # clamp(s,0,1) Dual derivative: 1 on [0,1], 0 strictly outside.
        ds = (0 <= s <= 1) ? 1.0 : 0.0
        wj = Nj[j] / wsum
        # I = sum(wj * s),  dI/dRe(Sz_j) = wj * ds * (1/2) / den
        λx[_real_idx_zr(j, M)] = wj * ds * (0.5 / den)
    end
    return λx
end

"""
    silencing_pullback!(λx, Sp, g_b, Nj)

Seeds `λx` with the RAW `∂silencing/∂Re(Sp_j)`/`∂silencing/∂Im(Sp_j)`
gradient (the pullback of [`_weighted_silencing_factor`](@ref) alone,
`|F|` itself -- NOT a `w_sil*(silencing-target_F)^2` cost term). See
[`inversion_pullback!`](@ref)'s docstring for why the combining weight/
`target_F` chain-rule factor now lives in [`pulse_cost_grad_adjoint`](@ref)
instead of here.
"""
function silencing_pullback!(λx::AbstractVector, Sp::AbstractVector, g_b::AbstractVector,
                             Nj::AbstractVector)
    M = length(Nj)
    length(Sp) == length(g_b) == M || error(
        "silencing_pullback!: Sp/g_b/Nj lengths $(length(Sp))/$(length(g_b))/$(length(Nj)) must match."
    )
    length(λx) == real_state_length_1st_order(M) || error(
        "silencing_pullback!: λx length $(length(λx)) != $(real_state_length_1st_order(M))."
    )
    fill!(λx, 0)
    weight = Nj .* abs2.(g_b)
    den = sum(weight .* Nj) / 2 + 1e-30
    Fr = 0.0
    Fi = 0.0
    @inbounds for j in 1:M
        Fr += weight[j] * real(Sp[j])
        Fi += weight[j] * imag(Sp[j])
    end
    Fr /= den
    Fi /= den
    abs_F = sqrt(Fr * Fr + Fi * Fi + 1e-30)
    dsil = (0 <= abs_F <= 1) ? 1.0 : 0.0
    # silencing = clamp(abs_F, 0, 1), dsilencing/dRe(Sp_j) = dsil * (Fr/abs_F) * weight[j]/den
    scale = dsil / abs_F
    @inbounds for j in 1:M
        wj_den = weight[j] / den
        λx[_real_idx_pr(j, M)] = scale * Fr * wj_den
        λx[_real_idx_pi(j, M)] = scale * Fi * wj_den
    end
    return λx
end

function _assert_tsit5_alg(kwargs)
    if haskey(kwargs, :alg)
        nameof(typeof(kwargs[:alg])) === :Tsit5 || error(
            "pulse_cost_grad_adjoint only supports Tsit5, got $(typeof(kwargs[:alg]))."
        )
    end
    return nothing
end

function _control_plus_signal_E(pulse::CompositePulse, u, signal_E_of_t)
    control = build_E_of_t(pulse, u)
    return t -> control(t) + signal_E_of_t(t)
end

function reverse_tsit5_on_states!(
    gθ::AbstractVector,
    λx::AbstractVector,
    states::Vector{Vector{ComplexF64}},
    t::Vector{Float64},
    dts::AbstractVector,
    p,
    pulse::CompositePulse,
    u_pulse::AbstractVector,
    ws::Tsit5DiscAdjWorkspace,
)
    nstep = length(dts)
    nstep == length(states) - 1 || error(
        "reverse_tsit5_on_states!: $(length(states)) states for $(nstep) steps."
    )
    @inbounds for n in nstep:-1:1
        Δt = Float64(dts[n])
        Δt == 0 && continue
        tsit5_step_vjp!(λx, gθ, λx, states[n], t[n], Δt, p, pulse, u_pulse, ws)
    end
    return gθ
end

function reverse_tsit5_on_checkpoints!(
    gθ::AbstractVector,
    λx::AbstractVector,
    mesh::FrozenTsit5Mesh,
    stack::HostCheckpointStack,
    p,
    pulse::CompositePulse,
    u_pulse::AbstractVector,
    ws::Tsit5DiscAdjWorkspace,
)
    nchk = length(stack.index)
    nchk >= 2 || error("reverse_tsit5_on_checkpoints!: need at least 2 checkpoints.")
    @inbounds for w in (nchk - 1):-1:1
        i0 = stack.index[w]
        i1 = stack.index[w + 1]
        dts = @view mesh.dt[i0:(i1 - 1)]
        us = replay_tsit5_window(stack.u[w], p, mesh.t[i0], dts)
        tloc = mesh.t[i0:i1]
        reverse_tsit5_on_states!(gθ, λx, us, collect(tloc), dts, p, pulse, u_pulse, ws)
        us = nothing
        GC.gc(false)
    end
    return gθ
end

function _adjoint_one_track(
    u::AbstractVector,
    pulse::CompositePulse,
    d,
    initial_condition::Symbol,
    pullback!,
    reltol, abstol, tstops, signal_E_of_t, checkpoint_stride, use_checkpoints::Bool,
)
    M = Int(d.M)
    E_of_t = _control_plus_signal_E(pulse, u, signal_E_of_t)
    p = _host_ode_p(d, E_of_t)
    u0 = build_u0_1st_order_cpu(M, d.Nj, Float64, initial_condition)
    # record_full_u=!use_checkpoints: when the caller wants the checkpointed
    # reverse sweep, record_adaptive_tsit5_mesh's own memory-bounded mode
    # never materialises the full per-node state list (mesh.u) IN THE
    # FIRST PLACE -- this is what actually bounds memory at a large
    # ensemble; dropping mesh.u only AFTER building it in full (an earlier
    # version of this function did that, right before the backward sweep)
    # still pays the full O(steps) peak during recording itself, which is
    # exactly where a real M=20000 run was observed to exhaust memory.
    mesh, stack, u_end = record_adaptive_tsit5_mesh(
        u0, p, d.timespan;
        reltol=reltol, abstol=abstol, tstops=tstops, checkpoint_stride=checkpoint_stride,
        record_full_u=!use_checkpoints,
    )
    a, Sp, Sz = unpack_state_1st_order_u(u_end, M)
    nR = real_state_length_1st_order(M)
    λx = zeros(Float64, nR)
    pullback!(λx, a, Sp, Sz)
    gθ = zeros(Float64, length(u))
    ws = _tsit5_adj_workspace(M)
    if use_checkpoints
        # reverse_tsit5_on_checkpoints! reads only mesh.t/mesh.dt/stack.u
        # (it replays each window from stack's own downsampled snapshots,
        # never from mesh.u -- see that function's own body), so it is
        # correct here even in the degenerate case where checkpoint_stride
        # was left large enough that stack ends up with only the first/last
        # node (one window covering the whole trajectory, replayed in one
        # shot -- correct, just without the memory saving a smaller stride
        # would give).
        reverse_tsit5_on_checkpoints!(gθ, λx, mesh, stack, p, pulse, collect(Float64, u), ws)
    else
        reverse_tsit5_on_states!(gθ, λx, mesh.u, mesh.t, mesh.dt, p, pulse, collect(Float64, u), ws)
    end
    return gθ, a, collect(Sp), collect(Sz), mesh, stack
end

"""
    pulse_cost_on_frozen_mesh(u, pulse, d, mesh_ground, mesh_equator; ...) -> cost

Test-facing primal: replay each track's frozen `dt` sequence with Φ
(`tsit5_forced_step`) and score the same scalar as [`pulse_cost`](@ref).
Both `mesh_ground` and `mesh_equator` are always required now: the
multiplicative `fidelity_phys = inversion*silencing_success` has no
well-defined value with either track missing.
"""
function pulse_cost_on_frozen_mesh(
    u::AbstractVector,
    pulse::CompositePulse,
    d,
    mesh_ground,
    mesh_equator;
    target_F=1.0,
    w_time=0.15,
    w_power=0.05,
    w_tmax=1.0,
    signal_E_of_t=_zero_drive,
)
    T = eltype(u)
    E_of_t = _control_plus_signal_E(pulse, u, signal_E_of_t)
    p = _host_ode_p(d, E_of_t)
    duration = pulse_duration(pulse, u)

    mesh_ground === nothing && error("pulse_cost_on_frozen_mesh: ground mesh required.")
    u0g = build_u0_1st_order_cpu(Int(d.M), d.Nj, T, :ground)
    usg = replay_tsit5_window(u0g, p, mesh_ground.t[1], mesh_ground.dt)
    _, _, Sz = unpack_state_1st_order_u(usg[end], Int(d.M))
    inversion = _weighted_inversion(Sz, d.Nj, T)

    mesh_equator === nothing && error("pulse_cost_on_frozen_mesh: equator mesh required.")
    u0e = build_u0_1st_order_cpu(Int(d.M), d.Nj, T, :equator)
    use = replay_tsit5_window(u0e, p, mesh_equator.t[1], mesh_equator.dt)
    _, Sp, _ = unpack_state_1st_order_u(use[end], Int(d.M))
    silencing = _weighted_silencing_factor(Sp, d.g_b, d.Nj, T)
    coherence = _weighted_coherence(Sp, d.Nj, T)

    direct = _direct_cost_term(u, pulse, w_time, w_power, w_tmax)
    silencing_success = one(T) - (silencing - convert(T, target_F))^2
    fidelity_phys = inversion * silencing_success
    physics_cost = (one(T) - fidelity_phys)^2
    cost = physics_cost + direct
    return cost, inversion, silencing, duration, coherence
end

"""
    pulse_cost_grad_adjoint(u, pulse, d; kwargs...)
        -> (grad, cost, inversion, silencing, duration, coherence)

Frozen-mesh discrete Tsit5 adjoint of [`pulse_cost`](@ref). Dual-Tsit5
gradients are a different object; this is not a drop-in bit-identical
replacement. `signal_E_of_t` is not differentiated. CPU reverse.
Defaults `use_checkpoints=true, checkpoint_stride=300` -- this is the ONLY
gradient backend [`run_local_adam`](@ref)'s `grad_mode=:adjoint` selects,
so these defaults are effectively "the `grad_mode=:adjoint` defaults":
memory-bounded windowed Φ-replay from host checkpoints out of the box
(same gradient as the unbounded path when L1 replay matches the stored
mesh), rather than the full per-step state list every accepted point
would otherwise pin in memory for the whole trajectory. Pass
`use_checkpoints=false` (unbounded, reverses on the full stored adaptive
snapshots) or an explicit different `checkpoint_stride` to override.

`use_checkpoints=true` ALONE bounds only the forward recording pass
(`record_adaptive_tsit5_mesh`'s `mesh.u` is never materialised). It does
NOT by itself bound the reverse sweep: with `checkpoint_stride` left at
`typemax(Int)`, there is exactly ONE checkpoint window spanning the whole
trajectory, and `replay_tsit5_window` rebuilds the full per-step state
list for that one window anyway -- no memory saving over
`use_checkpoints=false`. That degenerate combination triggers a `@warn`;
it cannot happen under the defaults, only if `checkpoint_stride` is
explicitly reset to `typemax(Int)` while `use_checkpoints=true`.

Both tracks are always run (see [`inversion_pullback!`](@ref)/
[`silencing_pullback!`](@ref)'s docstrings): each adjoint sweep seeds
`λx` with the RAW `∂inversion/∂x`/`∂silencing/∂x` pullback (coefficient
`+1`, no `target_F`/weight baked in), producing `grad_I = ∇inversion(u)`
and `grad_F = ∇silencing(u)` -- the same two Jacobians
[`_pulse_cost_grad_threaded`](@ref) computes via Dual ODEs. This function
then applies the IDENTICAL analytical chain rule that function uses to
combine them into `∇physics_cost`, so `pulse_cost_grad_adjoint` and
`_pulse_cost_grad_threaded` are adjoints of the exact same scalar
[`pulse_cost`](@ref), just via two different ODE-sensitivity backends.
"""
function pulse_cost_grad_adjoint(
    u::AbstractVector,
    pulse::CompositePulse,
    d;
    target_F=1.0,
    w_time=0.15,
    w_power=0.05,
    w_tmax=1.0,
    compute::Symbol=:cpu,
    checkpoint_stride::Integer=300,
    use_checkpoints::Bool=true,
    kwargs...,
)
    _forbid_initial_condition(kwargs)
    _assert_tsit5_alg(kwargs)
    sk = _solver_kwargs(kwargs)
    n = length(u)
    length(u) == n_params(pulse) || error(
        "pulse_cost_grad_adjoint: u has length $(length(u)), expected n_params=$(n_params(pulse))."
    )
    if use_checkpoints && checkpoint_stride == typemax(Int)
        @warn "pulse_cost_grad_adjoint: use_checkpoints=true but checkpoint_stride was left " *
              "at its default (typemax(Int)) -- this produces exactly ONE checkpoint window " *
              "spanning the ENTIRE trajectory, so the reverse sweep still replays/holds the " *
              "FULL per-step state list (no memory saving over use_checkpoints=false; the " *
              "forward recording pass is bounded, but the reverse pass is not). Pass an " *
              "explicit, finite checkpoint_stride (e.g. a few hundred) to actually bound memory."
    end
    duration = pulse_duration(pulse, u)
    uθ = collect(Float64, u)
    signal_E_of_t = haskey(sk, :signal_E_of_t) ? sk.signal_E_of_t : _zero_drive
    reltol = haskey(sk, :reltol) ? sk.reltol : 1e-8
    abstol = haskey(sk, :abstol) ? sk.abstol : 1e-8
    t_start, t_end, _, _, _ = decode(pulse, uθ)
    tstops = collect(Float64, vcat(t_start, t_end))
    if haskey(sk, :tstops)
        tstops = collect(Float64, sk.tstops)
    end

    chunk = ForwardDiff.Chunk{min(60, n)}()
    direct_only(uu) = _direct_cost_term(uu, pulse, w_time, w_power, w_tmax)
    grad_direct = ForwardDiff.gradient(direct_only, uθ, ForwardDiff.GradientConfig(direct_only, uθ, chunk))
    direct_val = direct_only(uθ)

    compute_eff = compute
    compute_eff === :auto && (compute_eff = :cpu)
    compute_eff === :gpu && @warn "pulse_cost_grad_adjoint v1 reverses on CPU; ignoring compute=:gpu"

    local inversion, silencing, coherence, grad_I, grad_F
    try
        function pb_inv!(λx, a, Sp, Sz)
            inversion_pullback!(λx, Sz, d.Nj)
            return λx
        end
        grad_I, _, _, Sz, _, _ = _adjoint_one_track(
            uθ, pulse, d, :ground, pb_inv!,
            reltol, abstol, tstops, signal_E_of_t, checkpoint_stride, use_checkpoints,
        )
        inversion = Float64(_weighted_inversion(Sz, d.Nj, Float64))

        function pb_sil!(λx, a, Sp, Sz)
            silencing_pullback!(λx, Sp, d.g_b, d.Nj)
            return λx
        end
        grad_F, _, Sp, _, _, _ = _adjoint_one_track(
            uθ, pulse, d, :equator, pb_sil!,
            reltol, abstol, tstops, signal_E_of_t, checkpoint_stride, use_checkpoints,
        )
        silencing = Float64(_weighted_silencing_factor(Sp, d.g_b, d.Nj, Float64))
        coherence = Float64(_weighted_coherence(Sp, d.Nj, Float64))
    catch e
        e isa PulseSolveFailed || rethrow()
        GC.gc(false)
        return fill(NaN, n), Inf, NaN, NaN, duration, NaN
    end

    GC.gc(false)

    # Same analytical chain rule as _pulse_cost_grad_threaded: the raw
    # per-track Jacobians grad_I/grad_F combine through physics_cost's
    # multiplicative coupling here, not inside either pullback.
    silencing_success = 1.0 - (silencing - Float64(target_F))^2
    fidelity_phys = inversion * silencing_success
    physics_cost = (1.0 - fidelity_phys)^2

    grad_S = -2.0 * (silencing - Float64(target_F)) .* grad_F
    grad_physics = -2.0 * (1.0 - fidelity_phys) .* (silencing_success .* grad_I .+ inversion .* grad_S)

    grad = grad_physics .+ grad_direct
    cost = physics_cost + direct_val
    return grad, cost, inversion, silencing, duration, coherence
end
