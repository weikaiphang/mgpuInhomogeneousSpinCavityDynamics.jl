# ============================================================
# Discrete Tsit5 adjoint of the dual-trajectory pulse cost.
#
# Additive: does not replace ForwardDiff. Default Adam remains Dual.
# Forward state stays ComplexF64; the real split is only inside the VJP.
# signal_E_of_t is not differentiated (same contract as Dual).
# ============================================================

function inversion_pullback!(λx::AbstractVector, Sz::AbstractVector, Nj::AbstractVector, w_inv::Real)
    M = length(Nj)
    length(Sz) == M || error("inversion_pullback!: Sz length $(length(Sz)) != Nj length $M.")
    length(λx) == real_state_length_1st_order(M) || error(
        "inversion_pullback!: λx length $(length(λx)) != $(real_state_length_1st_order(M))."
    )
    fill!(λx, 0)
    w_inv == 0 && return λx
    wsum = sum(Nj)
    @inbounds for j in 1:M
        den = Nj[j] / 2 + 1e-30
        frac = real(Sz[j]) / den
        s = (frac + 1) / 2
        # clamp(s,0,1) Dual derivative: 1 on [0,1], 0 strictly outside.
        ds = (0 <= s <= 1) ? 1.0 : 0.0
        wj = Nj[j] / wsum
        # cost = -w_inv * I,  dI/dRe(Sz_j) = wj * ds * (1/2) / den
        λx[_real_idx_zr(j, M)] = -Float64(w_inv) * wj * ds * (0.5 / den)
    end
    return λx
end

function silencing_pullback!(λx::AbstractVector, Sp::AbstractVector, g_b::AbstractVector,
                             Nj::AbstractVector, w_sil::Real, target_F::Real)
    M = length(Nj)
    length(Sp) == length(g_b) == M || error(
        "silencing_pullback!: Sp/g_b/Nj lengths $(length(Sp))/$(length(g_b))/$(length(Nj)) must match."
    )
    length(λx) == real_state_length_1st_order(M) || error(
        "silencing_pullback!: λx length $(length(λx)) != $(real_state_length_1st_order(M))."
    )
    fill!(λx, 0)
    w_sil == 0 && return λx
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
    sil = abs_F < 0 ? 0.0 : (abs_F > 1 ? 1.0 : abs_F)
    dsil = (0 <= abs_F <= 1) ? 1.0 : 0.0
    dcost_dsil = 2 * Float64(w_sil) * (sil - Float64(target_F))
    scale = dcost_dsil * dsil / abs_F
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
    mesh, stack, u_end = record_adaptive_tsit5_mesh(
        u0, p, d.timespan;
        reltol=reltol, abstol=abstol, tstops=tstops, checkpoint_stride=checkpoint_stride,
    )
    a, Sp, Sz = unpack_state_1st_order_u(u_end, M)
    nR = real_state_length_1st_order(M)
    λx = zeros(Float64, nR)
    pullback!(λx, a, Sp, Sz)
    gθ = zeros(Float64, length(u))
    ws = _tsit5_adj_workspace(M)
    use_checkpoints_eff = use_checkpoints && length(stack.index) >= 2 && stack.stride < length(mesh.u)
    if use_checkpoints_eff
        # reverse_tsit5_on_checkpoints! reads only mesh.t/mesh.dt/stack.u
        # (it replays each window from stack's own downsampled snapshots,
        # never from mesh.u -- see that function's own body) -- mesh.u
        # itself (the FULL per-accepted-step state list, easily hundreds
        # of MB at a real ensemble, see FrozenTsit5Mesh's own docstring)
        # is therefore provably dead from this point on. Dropping the
        # reference here, before the (also allocation-heavy, replay-based)
        # backward sweep starts, lets the GC reclaim it during that sweep
        # instead of holding both simultaneously. Safe regardless of what
        # the caller does with `mesh` afterward: _adjoint_one_track's own
        # two call sites (below) both discard the returned `mesh`.
        mesh.u = Vector{ComplexF64}[]
        GC.gc(false)
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
`mesh_*` may be `nothing` when that weight is zero.
"""
function pulse_cost_on_frozen_mesh(
    u::AbstractVector,
    pulse::CompositePulse,
    d,
    mesh_ground,
    mesh_equator;
    w_inv=1.0,
    w_sil=0.7,
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
    inversion = zero(T)
    silencing = zero(T)
    coherence = zero(T)
    if w_inv > 0.0
        mesh_ground === nothing && error("pulse_cost_on_frozen_mesh: ground mesh required when w_inv>0.")
        u0 = build_u0_1st_order_cpu(Int(d.M), d.Nj, T, :ground)
        us = replay_tsit5_window(u0, p, mesh_ground.t[1], mesh_ground.dt)
        _, _, Sz = unpack_state_1st_order_u(us[end], Int(d.M))
        inversion = _weighted_inversion(Sz, d.Nj, T)
    end
    if w_sil > 0.0
        mesh_equator === nothing && error("pulse_cost_on_frozen_mesh: equator mesh required when w_sil>0.")
        u0 = build_u0_1st_order_cpu(Int(d.M), d.Nj, T, :equator)
        us = replay_tsit5_window(u0, p, mesh_equator.t[1], mesh_equator.dt)
        _, Sp, _ = unpack_state_1st_order_u(us[end], Int(d.M))
        silencing = _weighted_silencing_factor(Sp, d.g_b, d.Nj, T)
        coherence = _weighted_coherence(Sp, d.Nj, T)
    end
    direct = _direct_cost_term(u, pulse, w_time, w_power, w_tmax)
    cost = -w_inv * inversion + w_sil * (silencing - convert(T, target_F))^2 + direct
    return cost, inversion, silencing, duration, coherence
end

"""
    pulse_cost_grad_adjoint(u, pulse, d; kwargs...)
        -> (grad, cost, inversion, silencing, duration, coherence)

Frozen-mesh discrete Tsit5 adjoint of [`pulse_cost`](@ref). Dual-Tsit5
gradients are a different object; this is not a drop-in bit-identical
replacement. `signal_E_of_t` is not differentiated. CPU reverse.
`checkpoint_stride=typemax(Int)` (default) reverses on the stored
adaptive snapshots; a finite stride uses windowed Φ-replay from host
checkpoints (same gradient when L1 replay matches the stored mesh).
"""
function pulse_cost_grad_adjoint(
    u::AbstractVector,
    pulse::CompositePulse,
    d;
    w_inv=1.0,
    w_sil=0.7,
    target_F=1.0,
    w_time=0.15,
    w_power=0.05,
    w_tmax=1.0,
    compute::Symbol=:cpu,
    checkpoint_stride::Integer=typemax(Int),
    use_checkpoints::Bool=false,
    kwargs...,
)
    _forbid_initial_condition(kwargs)
    _assert_tsit5_alg(kwargs)
    sk = _solver_kwargs(kwargs)
    n = length(u)
    length(u) == n_params(pulse) || error(
        "pulse_cost_grad_adjoint: u has length $(length(u)), expected n_params=$(n_params(pulse))."
    )
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
    grad = ForwardDiff.gradient(direct_only, uθ, ForwardDiff.GradientConfig(direct_only, uθ, chunk))
    direct_val = direct_only(uθ)

    inversion = 0.0
    silencing = 0.0
    coherence = 0.0
    compute_eff = compute
    compute_eff === :auto && (compute_eff = :cpu)
    compute_eff === :gpu && @warn "pulse_cost_grad_adjoint v1 reverses on CPU; ignoring compute=:gpu"

    try
        if w_inv > 0.0
            function pb_inv!(λx, a, Sp, Sz)
                inversion_pullback!(λx, Sz, d.Nj, w_inv)
                return λx
            end
            g_g, _, _, Sz, _, _ = _adjoint_one_track(
                uθ, pulse, d, :ground, pb_inv!,
                reltol, abstol, tstops, signal_E_of_t, checkpoint_stride, use_checkpoints,
            )
            inversion = Float64(_weighted_inversion(Sz, d.Nj, Float64))
            grad .+= g_g
        end
        if w_sil > 0.0
            function pb_sil!(λx, a, Sp, Sz)
                silencing_pullback!(λx, Sp, d.g_b, d.Nj, w_sil, target_F)
                return λx
            end
            g_e, _, Sp, _, _, _ = _adjoint_one_track(
                uθ, pulse, d, :equator, pb_sil!,
                reltol, abstol, tstops, signal_E_of_t, checkpoint_stride, use_checkpoints,
            )
            silencing = Float64(_weighted_silencing_factor(Sp, d.g_b, d.Nj, Float64))
            coherence = Float64(_weighted_coherence(Sp, d.Nj, Float64))
            grad .+= g_e
        end
    catch e
        e isa PulseSolveFailed || rethrow()
        GC.gc(false)
        return fill(NaN, n), Inf, NaN, NaN, duration, NaN
    end

    GC.gc(false)
    cost = -w_inv * inversion + w_sil * (silencing - Float64(target_F))^2 + direct_val
    return grad, cost, inversion, silencing, duration, coherence
end
