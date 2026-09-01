# ============================================================
# Discrete Tsit5 adjoint of the dual-trajectory pulse cost.
#
# Additive: does not replace ForwardDiff. Default Adam remains Dual.
# Forward state stays ComplexF64; the real split is only inside the VJP.
# signal_E_of_t is not differentiated (same contract as Dual).
# ============================================================

"""
    inversion_pullback!(λx, Sz, g_b, Nj)

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

Matches [`_weighted_inversion`](@ref)'s paper (App. H) bright-mode /
cooperativity weight `w_j = Nj_j g_j²` (was plain `Nj_j`).
"""
function inversion_pullback!(λx::AbstractVector, Sz::AbstractVector, g_b::AbstractVector,
                             Nj::AbstractVector)
    M = length(Nj)
    length(Sz) == length(g_b) == M || error(
        "inversion_pullback!: Sz/g_b lengths $(length(Sz))/$(length(g_b)) != Nj length $M."
    )
    length(λx) == real_state_length_1st_order(M) || error(
        "inversion_pullback!: λx length $(length(λx)) != $(real_state_length_1st_order(M))."
    )
    fill!(λx, 0)
    w = Nj .* abs2.(g_b)
    wsum = sum(w) + 1e-30
    @inbounds for j in 1:M
        den = Nj[j] / 2 + 1e-30
        frac = real(Sz[j]) / den
        s = (frac + 1) / 2
        # clamp(s,0,1) Dual derivative: 1 on [0,1], 0 strictly outside.
        ds = (0 <= s <= 1) ? 1.0 : 0.0
        wj = w[j] / wsum
        # I = sum(wj * s),  dI/dRe(Sz_j) = wj * ds * (1/2) / den
        λx[_real_idx_zr(j, M)] = wj * ds * (0.5 / den)
    end
    return λx
end

"""
    silencing_pullback!(λx, Sp, g_b, Nj, delta_b; eps_seed=_WEAK_SEED)

Seeds `λx` with the RAW `∂silencing/∂Re(Sp_j)`/`∂silencing/∂Im(Sp_j)`
gradient (the pullback of [`_weighted_silencing_factor`](@ref) alone,
`|F|_⋆` itself -- NOT a `w_sil*(silencing-target_F)^2` cost term). See
[`inversion_pullback!`](@ref)'s docstring for why the combining weight/
`target_F` chain-rule factor now lives in [`pulse_cost_grad_adjoint`](@ref)
instead of here.

Analytic chain rule for the paper per-frequency-slice metric
(`_weighted_silencing_factor`): with `B(ω) = _frequency_slice_indices`,
`F(ω) = (Σ_{k∈B(ω)} g_k² Sp_k) / F_den(ω)`,
`F_den(ω) = ε Σ_{k∈B(ω)} g_k² Nj_k/2` (real, `u`-independent),
`n(ω) = Σ_{k∈B(ω)} Nj_k g_k²`, and
`|F|_⋆ = clamp(Σ_ω n(ω)|F(ω)| / Σ_ω n(ω), 0, 1)`, so for a bin `j` in
slice `ω(j)`:

    ∂|F|_⋆/∂Re(Sp_j) = 𝟙_clamp · (n(ω(j))/Σn) · Re(F(ω(j)))/|F(ω(j))| · g_j²/F_den(ω(j))

(and `Im` with `Im(F)` in place of `Re(F)`). No per-bin unit phasor
anywhere -- the only divisions are by `F_den(ω)` and `|F(ω)|`, both
bounded away from 0 by the `1e-30` floors.
"""
function silencing_pullback!(λx::AbstractVector, Sp::AbstractVector, g_b::AbstractVector,
                             Nj::AbstractVector, delta_b::AbstractVector;
                             eps_seed::Real=_WEAK_SEED)
    M = length(Nj)
    length(Sp) == length(g_b) == length(delta_b) == M || error(
        "silencing_pullback!: Sp/g_b/delta_b lengths $(length(Sp))/$(length(g_b))/$(length(delta_b)) != Nj length $M."
    )
    length(λx) == real_state_length_1st_order(M) || error(
        "silencing_pullback!: λx length $(length(λx)) != $(real_state_length_1st_order(M))."
    )
    fill!(λx, 0)
    slices = _frequency_slice_indices(delta_b)
    nω = [sum(Nj[idx] .* abs2.(g_b[idx])) for idx in slices]
    Nsum = sum(nω) + 1e-30

    # First pass: per-slice F(ω) components + F_den(ω), and the aggregate
    # |F|_⋆ needed to gate the outer clamp's derivative.
    Fr = zeros(Float64, length(slices))
    Fi = zeros(Float64, length(slices))
    Fden = zeros(Float64, length(slices))
    absF_star = 0.0
    @inbounds for (s, idx) in enumerate(slices)
        wg = abs2.(g_b[idx])
        fd = eps_seed * sum(wg .* (Nj[idx] ./ 2)) + 1e-30
        fr = 0.0
        fi = 0.0
        for (t, j) in enumerate(idx)
            fr += wg[t] * real(Sp[j])
            fi += wg[t] * imag(Sp[j])
        end
        fr /= fd
        fi /= fd
        Fr[s], Fi[s], Fden[s] = fr, fi, fd
        absF_star += nω[s] * sqrt(fr * fr + fi * fi + 1e-30)
    end
    absF_star /= Nsum
    dstar = (0 <= absF_star <= 1) ? 1.0 : 0.0

    @inbounds for (s, idx) in enumerate(slices)
        absF = sqrt(Fr[s] * Fr[s] + Fi[s] * Fi[s] + 1e-30)
        pref = dstar * (nω[s] / Nsum) / absF
        for j in idx
            gj2 = abs2(g_b[j])
            λx[_real_idx_pr(j, M)] = pref * Fr[s] * gj2 / Fden[s]
            λx[_real_idx_pi(j, M)] = pref * Fi[s] * gj2 / Fden[s]
        end
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
    pulse_cost_on_frozen_mesh(u, pulse, d, mesh_ground, mesh_weak; ...) -> cost

Test-facing primal: replay each track's frozen `dt` sequence with Φ
(`tsit5_forced_step`) and score the same scalar as [`pulse_cost`](@ref),
via the SAME [`_fidelity_physics_cost`](@ref) helper (so `I_min`/`kappa_I`/
`S_min`/`kappa_S`, defaults `_DEFAULT_PENALTY_MIN`/`_DEFAULT_PENALTY_KAPPA`,
track `pulse_cost`'s own formula exactly -- see that function's docstring).
Both `mesh_ground` and `mesh_weak` are always required now: the
multiplicative `fidelity_phys = inversion*silencing_success` has no
well-defined value with either track missing.
"""
function pulse_cost_on_frozen_mesh(
    u::AbstractVector,
    pulse::CompositePulse,
    d,
    mesh_ground,
    mesh_weak;
    target_F=1.0,
    w_time=0.15,
    w_power=0.05,
    w_tmax=1.0,
    I_min::Real=_DEFAULT_PENALTY_MIN, kappa_I::Real=_DEFAULT_PENALTY_KAPPA,
    S_min::Real=_DEFAULT_PENALTY_MIN, kappa_S::Real=_DEFAULT_PENALTY_KAPPA,
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
    inversion = _weighted_inversion(Sz, d.g_b, d.Nj, T)

    mesh_weak === nothing && error("pulse_cost_on_frozen_mesh: weak-excitation (:weak) mesh required.")
    u0e = build_u0_1st_order_cpu(Int(d.M), d.Nj, T, :weak)
    use = replay_tsit5_window(u0e, p, mesh_weak.t[1], mesh_weak.dt)
    _, Sp, _ = unpack_state_1st_order_u(use[end], Int(d.M))
    silencing = _weighted_silencing_factor(Sp, d.g_b, d.Nj, d.delta_b, T)
    coherence = _weighted_coherence(Sp, d.g_b, d.Nj, d.delta_b, T)
    field_amp = _weighted_field_amplitude(Sp, d.g_b, d.Nj, T)
    weak_seed_retention = _weak_seed_retention(Sp, d.g_b, d.Nj, d.delta_b, T)

    direct = _direct_cost_term(u, pulse, w_time, w_power, w_tmax)
    physics_cost, _, _ = _fidelity_physics_cost(inversion, silencing, target_F, I_min, kappa_I, S_min, kappa_S)
    cost = physics_cost + direct
    return cost, inversion, silencing, duration, coherence, field_amp, weak_seed_retention
end

"""
    pulse_cost_grad_adjoint(u, pulse, d; kwargs...)
        -> (grad, cost, inversion, silencing, duration, coherence, field_amp, weak_seed_retention)

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
then applies the IDENTICAL analytical chain rule that function uses (via
the SAME [`_fidelity_physics_cost`](@ref)/[`_fidelity_gradient_coefficients`](@ref)
helpers, so `I_min`/`kappa_I`/`S_min`/`kappa_S`, defaults
`_DEFAULT_PENALTY_MIN`/`_DEFAULT_PENALTY_KAPPA`, are handled identically
on both sides) to combine them into `∇physics_cost`, so
`pulse_cost_grad_adjoint` and `_pulse_cost_grad_threaded` are adjoints
of the exact same scalar [`pulse_cost`](@ref), just via two different
ODE-sensitivity backends.
"""
function pulse_cost_grad_adjoint(
    u::AbstractVector,
    pulse::CompositePulse,
    d;
    target_F=1.0,
    w_time=0.15,
    w_power=0.05,
    w_tmax=1.0,
    I_min::Real=_DEFAULT_PENALTY_MIN, kappa_I::Real=_DEFAULT_PENALTY_KAPPA,
    S_min::Real=_DEFAULT_PENALTY_MIN, kappa_S::Real=_DEFAULT_PENALTY_KAPPA,
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

    local inversion, silencing, coherence, field_amp, weak_seed_retention, grad_I, grad_F
    try
        function pb_inv!(λx, a, Sp, Sz)
            inversion_pullback!(λx, Sz, d.g_b, d.Nj)
            return λx
        end
        grad_I, _, _, Sz, _, _ = _adjoint_one_track(
            uθ, pulse, d, :ground, pb_inv!,
            reltol, abstol, tstops, signal_E_of_t, checkpoint_stride, use_checkpoints,
        )
        inversion = Float64(_weighted_inversion(Sz, d.g_b, d.Nj, Float64))

        function pb_sil!(λx, a, Sp, Sz)
            silencing_pullback!(λx, Sp, d.g_b, d.Nj, d.delta_b)
            return λx
        end
        grad_F, _, Sp, _, _, _ = _adjoint_one_track(
            uθ, pulse, d, :weak, pb_sil!,
            reltol, abstol, tstops, signal_E_of_t, checkpoint_stride, use_checkpoints,
        )
        silencing = Float64(_weighted_silencing_factor(Sp, d.g_b, d.Nj, d.delta_b, Float64))
        coherence = Float64(_weighted_coherence(Sp, d.g_b, d.Nj, d.delta_b, Float64))
        field_amp = Float64(_weighted_field_amplitude(Sp, d.g_b, d.Nj, Float64))
        weak_seed_retention = Float64(_weak_seed_retention(Sp, d.g_b, d.Nj, d.delta_b, Float64))
    catch e
        e isa PulseSolveFailed || rethrow()
        GC.gc(false)
        return fill(NaN, n), Inf, NaN, NaN, duration, NaN, NaN, NaN
    end

    GC.gc(false)

    # Same analytical chain rule as _pulse_cost_grad_threaded (both call the
    # SAME _fidelity_physics_cost/_fidelity_gradient_coefficients helpers):
    # the raw per-track Jacobians grad_I/grad_F combine through
    # physics_cost's multiplicative coupling (plus the squared-hinge
    # penalty's own restoring gradient) here, not inside either pullback.
    physics_cost, fidelity_phys, silencing_success =
        _fidelity_physics_cost(inversion, silencing, Float64(target_F), I_min, kappa_I, S_min, kappa_S)
    coeff_I, coeff_S = _fidelity_gradient_coefficients(inversion, silencing_success,
                                                        fidelity_phys, I_min, kappa_I, S_min, kappa_S)

    grad_S = -2.0 * (silencing - Float64(target_F)) .* grad_F
    grad_physics = coeff_I .* grad_I .+ coeff_S .* grad_S

    grad = grad_physics .+ grad_direct
    cost = physics_cost + direct_val
    return grad, cost, inversion, silencing, duration, coherence, field_amp, weak_seed_retention
end
