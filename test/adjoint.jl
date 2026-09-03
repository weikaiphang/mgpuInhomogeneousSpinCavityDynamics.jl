# Discrete Tsit5 adjoint tests. Existing Dual @testsets above are the
# production CI contract and are not relaxed here.

# Mixed absolute/relative max error. The 1e-6 absolute floor (raised from
# 1e-8) covers STRUCTURAL zeros of the pulse_cost gradient -- notably the
# global-drive-phase parameter, to which both the paper bright-mode
# inversion and the per-frequency-slice silencing factor |F|_⋆ are exactly
# invariant. Once |F|_⋆ clamps (a coherence-preserving pulse drives the
# :weak track's coherence well above its ε seed), the silencing track
# contributes an exact 0 to that component, so the two backends differ only
# by reverse- vs forward-mode roundoff on a genuine 0 (~1e-16). Every
# component with a non-negligible gradient (|g| ≫ 1e-6) is still compared at
# full relative strictness.
function _relmax(g, fd)
    return maximum(abs.(g .- fd) ./ max.(abs.(g), abs.(fd), 1e-6))
end

function _toy_p(d; E=t -> 1.0 + 0.0im)
    M = Int(d.M)
    return (Float64(d.delta0), Float64(d.kappa_e), Float64(d.kappa_i),
            collect(Float64, d.delta_b), collect(Float64, d.g_b), M, E)
end

@testset "real-split F matches rhs_1st_order!" begin
    d = FAKE_D_ODE
    M = Int(d.M)
    p = _toy_p(d)
    rng = Random.Xoshiro(1)
    u = randn(rng, ComplexF64, state_length_1st_order(M))
    du = similar(u)
    rhs_1st_order!(du, u, p, 0.0)
    x = Vector{Float64}(undef, real_state_length_1st_order(M))
    dx = similar(x)
    pack_state_real!(x, u, M)
    rhs_1st_order_real!(dx, x, p, 0.0)
    dx_ref = Vector{Float64}(undef, length(x))
    pack_state_real!(dx_ref, du, M)
    @test dx ≈ dx_ref atol=1e-14
end

# Pins rhs_1st_order! to its pre-optimisation formulation, elementwise.
# `_rhs_1st_order_preopt!` is the RHS exactly as it stood before the
# GPU-hot-path work (scalar `a = u[1]` / `du[1] = ...`, and a fresh
# `sum(g_b .* conj.(Sp))` temporary each call). The current
# rhs_1st_order! uses a 1-element view for `a`/`da` and a reused reduction
# buffer; both are meant to be BIT-IDENTICAL on the Vector / Dual
# backends. If a future change alters the vector field on these paths,
# this fails even when the looser real-split `atol` test above would not.
function _rhs_1st_order_preopt!(du, u, p, t)
    delta0, kappa_e, kappa_i, delta_b, g_b, M, E_of_t = p
    a = u[1]
    Sp = @view u[2:1+M]
    Sz = @view u[2+M:1+2M]
    dSp = @view du[2:1+M]
    dSz = @view du[2+M:1+2M]
    κe = kappa_e
    κt = kappa_e + kappa_i
    E_t = E_of_t(t)
    du[1] = sqrt(κe) * E_t - 1im * delta0 * a -
            1im * sum(g_b .* conj.(Sp)) - 0.5 * κt * a
    dSp .= 1im .* delta_b .* Sp .- 2im .* g_b .* conj(a) .* Sz
    dSz .= -1im .* g_b .* a .* Sp .+ 1im .* g_b .* conj(a) .* conj.(Sp)
    return nothing
end

@testset "rhs_1st_order! bit-identical to pre-optimisation formulation" begin
    d = FAKE_D_ODE
    M = Int(d.M)
    p = _toy_p(d; E = t -> 1.3 + 0.7im)

    # ComplexF64 path: exact bits, cold AND warm reduction buffer.
    for seed in (1, 2, 3), t in (0.0, 1.3e-6, 4.1e-5)
        rng = Random.Xoshiro(seed)
        u = randn(rng, ComplexF64, state_length_1st_order(M))
        a_new = similar(u); a_ref = similar(u)
        rhs_1st_order!(a_new, u, p, t)
        rhs_1st_order!(a_new, u, p, t)          # warm buffer
        _rhs_1st_order_preopt!(a_ref, u, p, t)
        @test reinterpret(UInt8, a_new) == reinterpret(UInt8, a_ref)
    end

    # ForwardDiff.Dual path: derivative of a scalar functional of `du`
    # w.r.t. a knob threaded through E(t) must match to the last bit.
    rng = Random.Xoshiro(7)
    u = randn(rng, ComplexF64, state_length_1st_order(M))
    w = randn(rng, ComplexF64, state_length_1st_order(M))
    tprobe = 2.7e-5
    func(rhs!, θ) = begin
        E = tt -> (1.3 + 0.7im) * (1 + θ * sin(2π * 1e4 * tt))
        pp = (p[1], p[2], p[3], p[4], p[5], M, E)
        du = Vector{Complex{eltype(θ)}}(undef, length(u))
        rhs!(du, convert(Vector{Complex{eltype(θ)}}, u), pp, tprobe)
        real(sum(w .* du))
    end
    g_new = ForwardDiff.derivative(θ -> func(rhs_1st_order!, θ), 0.37)
    g_ref = ForwardDiff.derivative(θ -> func(_rhs_1st_order_preopt!, θ), 0.37)
    @test g_new === g_ref
end

@testset "VJP adjoint identity and vs ForwardDiff λ·F" begin
    d = FAKE_D_ODE
    M = Int(d.M)
    p = _toy_p(d)
    rng = Random.Xoshiro(2)
    u = randn(rng, ComplexF64, state_length_1st_order(M))
    x = pack_state_real(u[1], u[2:1+M], u[2+M:end])
    λ = randn(rng, length(x))
    v = randn(rng, length(x))
    J = ForwardDiff.jacobian(xx -> begin
        dx = similar(xx)
        rhs_1st_order_real!(dx, xx, p, 1.3e-6)
        dx
    end, x)
    x̄ = similar(x)
    rhs_1st_order_vjp!(x̄, λ, x, p, 1.3e-6)
    @test abs(sum(λ .* (J * v)) - sum(x̄ .* v)) < 1e-10
    g_fd = ForwardDiff.gradient(xx -> sum(λ .* (begin
        dx = similar(xx)
        rhs_1st_order_real!(dx, xx, p, 1.3e-6)
        dx
    end)), x)
    @test x̄ ≈ g_fd atol=1e-10
end

@testset "∂F/∂θ sparsity is only the cavity drive" begin
    d = FAKE_D_ODE
    M = Int(d.M)
    pulse = CompositePulse(1, 4, 4, d)
    θ = seed_canonical(pulse, :hs1)
    rng = Random.Xoshiro(3)
    u = randn(rng, ComplexF64, state_length_1st_order(M))
    t = 0.5 * (d.timespan[1] + d.timespan[2])
    function Fθ(θθ)
        E = build_E_of_t(pulse, θθ)
        T = eltype(θθ)
        uC = map(z -> complex(T(real(z)), T(imag(z))), u)
        du = similar(uC)
        p = (d.delta0, d.kappa_e, d.kappa_i, collect(d.delta_b), collect(d.g_b), M, E)
        rhs_1st_order!(du, uC, p, t)
        xout = Vector{typeof(real(du[1]))}(undef, real_state_length_1st_order(M))
        pack_state_real!(xout, du, M)
        return xout
    end
    Jθ = ForwardDiff.jacobian(Fθ, θ)
    @test all(isapprox.(Jθ[3:end, :], 0; atol=1e-12))
    sqrt_κe = sqrt(d.kappa_e)
    gE = ForwardDiff.gradient(θθ -> sqrt_κe * real(build_E_of_t(pulse, θθ)(t)), θ)
    gEi = ForwardDiff.gradient(θθ -> sqrt_κe * imag(build_E_of_t(pulse, θθ)(t)), θ)
    @test Jθ[1, :] ≈ gE atol=1e-10
    @test Jθ[2, :] ≈ gEi atol=1e-10
end

@testset "inversion/silencing pullbacks vs ForwardDiff" begin
    d = FAKE_D_ODE
    M = Int(d.M)
    rng = Random.Xoshiro(4)
    Sz = randn(rng, ComplexF64, M)
    Sp = randn(rng, ComplexF64, M)
    nR = real_state_length_1st_order(M)

    # Pullbacks now seed the RAW ∂inversion/∂x, ∂silencing/∂x gradients
    # (no w_inv/w_sil/target_F baked in -- pulse_cost's multiplicative
    # fidelity couples the two tracks outside the pullback, see
    # pulse_cost_grad_adjoint).
    λI = zeros(nR)
    inversion_pullback!(λI, Sz, d.g_b, d.Nj)
    zr0 = real.(Sz)
    gI = ForwardDiff.gradient(zr -> begin
        SzD = complex.(zr, imag.(Sz))
        _weighted_inversion(SzD, d.g_b, d.Nj, eltype(zr))
    end, zr0)
    @test [λI[_real_idx_zr(j, M)] for j in 1:M] ≈ gI atol=1e-10
    @test all(λI[_real_idx_zi(j, M)] == 0 for j in 1:M)
    @test λI[1] == 0 && λI[2] == 0

    λS = zeros(nR)
    silencing_pullback!(λS, Sp, d.g_b, d.Nj, d.delta_b)
    spr = vcat(real.(Sp), imag.(Sp))
    gS = ForwardDiff.gradient(q -> begin
        SpD = complex.(q[1:M], q[M+1:2M])
        _weighted_silencing_factor(SpD, d.g_b, d.Nj, d.delta_b, eltype(q))
    end, spr)
    @test [λS[_real_idx_pr(j, M)] for j in 1:M] ≈ gS[1:M] atol=1e-10
    @test [λS[_real_idx_pi(j, M)] for j in 1:M] ≈ gS[M+1:2M] atol=1e-10
end

@testset "Tsit5 forced step matches OrdinaryDiffEq Tsit5" begin
    d = FAKE_D_ODE
    M = Int(d.M)
    pulse = CompositePulse(1, 4, 4, d)
    θ = seed_canonical(pulse, :hs1)
    E = build_E_of_t(pulse, θ)
    p = _toy_p(d; E=E)
    u0 = build_u0_1st_order_cpu(M, d.Nj, Float64, :ground)
    dt = 1e-7
    u_ours = tsit5_forced_step(u0, p, 0.0, dt)
    prob = ODEProblem(rhs_1st_order!, copy(u0), (0.0, dt), p)
    sol = solve(prob, Tsit5(); adaptive=false, dt=dt, save_everystep=true, dense=false)
    @test Array(sol.u[end]) ≈ u_ours atol=1e-12 rtol=1e-12
    tab = TSIT5_DISC_ADJ_TAB
    @test tab.c1 == 0.161
    @test tab.b1 == 0.09646076681806523
    @test tab.b6 == 2.324710524099774
end

@testset "one-step discrete VJP vs Dual" begin
    d = FAKE_D_ODE
    M = Int(d.M)
    pulse = CompositePulse(1, 4, 4, d)
    θ = seed_canonical(pulse, :hs1)
    u0 = build_u0_1st_order_cpu(M, d.Nj, Float64, :ground)
    t = 0.0
    dt = 1e-7
    E = build_E_of_t(pulse, θ)
    p = _toy_p(d; E=E)
    u1 = tsit5_forced_step(u0, p, t, dt)
    _, _, Sz = unpack_state_1st_order_u(u1, M)
    λx = zeros(real_state_length_1st_order(M))
    inversion_pullback!(λx, Sz, d.g_b, d.Nj)
    gθ = zeros(length(θ))
    ws = Tsit5DiscAdjWorkspace(M)
    tsit5_step_vjp!(copy(λx), gθ, λx, u0, t, dt, p, pulse, θ, ws)
    g_dual = ForwardDiff.gradient(θθ -> begin
        Ed = build_E_of_t(pulse, θθ)
        pd = (d.delta0, d.kappa_e, d.kappa_i, collect(d.delta_b), collect(d.g_b), M, Ed)
        u0d = build_u0_1st_order_cpu(M, d.Nj, eltype(θθ), :ground)
        u1d = tsit5_forced_step(u0d, pd, t, dt)
        _, _, Szd = unpack_state_1st_order_u(u1d, M)
        _weighted_inversion(Szd, d.g_b, d.Nj, eltype(θθ))
    end, θ)
    @test all(isfinite, gθ)
    @test all(isfinite, g_dual)
    @test _relmax(gθ, g_dual) < 1e-8
end

@testset "record_adaptive_tsit5_mesh: lean recording matches full recording" begin
    d = FAKE_D_ODE
    M = Int(d.M)
    pulse = CompositePulse(1, 4, 4, d)
    θ = seed_canonical(pulse, :hs1)
    t_start, t_end, _, _, _ = decode(pulse, θ)
    tstops = collect(Float64, vcat(t_start, t_end))
    E = build_E_of_t(pulse, θ)
    p = _host_ode_p(d, E)
    u0 = build_u0_1st_order_cpu(M, d.Nj, Float64, :ground)

    for stride in (1, 2, 5, 1000)
        mesh_full, stack_full, uend_full = record_adaptive_tsit5_mesh(
            u0, p, d.timespan; reltol=1e-8, abstol=1e-8, tstops=tstops,
            checkpoint_stride=stride, record_full_u=true,
        )
        mesh_lean, stack_lean, uend_lean = record_adaptive_tsit5_mesh(
            u0, p, d.timespan; reltol=1e-8, abstol=1e-8, tstops=tstops,
            checkpoint_stride=stride, record_full_u=false,
        )
        @test mesh_full.t == mesh_lean.t
        @test mesh_full.dt == mesh_lean.dt
        @test isempty(mesh_lean.u)
        @test length(mesh_full.u) == length(mesh_full.t)
        @test stack_full.index == stack_lean.index
        @test stack_full.t == stack_lean.t
        @test stack_full.stride == stack_lean.stride
        @test all(isapprox(stack_full.u[i], stack_lean.u[i]; atol=1e-12) for i in eachindex(stack_full.u))
        @test uend_full ≈ uend_lean atol=1e-12
        @test uend_full == stack_full.u[end]
        @test uend_lean == stack_lean.u[end]
    end
end

@testset "forced replay matches adaptive primal" begin
    d = FAKE_D_ODE
    M = Int(d.M)
    pulse = CompositePulse(1, 4, 4, d)
    θ = seed_canonical(pulse, :hs1)
    t_start, t_end, _, _, _ = decode(pulse, θ)
    tstops = collect(Float64, vcat(t_start, t_end))
    E = build_E_of_t(pulse, θ)
    p = _host_ode_p(d, E)
    u0 = build_u0_1st_order_cpu(M, d.Nj, Float64, :ground)
    mesh, _, u_end = record_adaptive_tsit5_mesh(
        u0, p, d.timespan; reltol=1e-8, abstol=1e-8, tstops=tstops,
    )
    us = replay_tsit5_window(u0, p, mesh.t[1], mesh.dt)
    @test us[end] ≈ u_end atol=1e-10 rtol=1e-10
end

@testset "frozen-mesh FD vs pulse_cost_grad_adjoint" begin
    d = FAKE_D_ODE
    pulse = CompositePulse(1, 4, 4, d)
    θ = seed_canonical(pulse, :hs1)
    t_start, t_end, _, _, _ = decode(pulse, θ)
    tstops = collect(Float64, vcat(t_start, t_end))
    E = build_E_of_t(pulse, θ)
    p = _host_ode_p(d, E)
    mesh_g, _, _ = record_adaptive_tsit5_mesh(
        build_u0_1st_order_cpu(Int(d.M), d.Nj, Float64, :ground), p, d.timespan;
        tstops=tstops,
    )
    mesh_e, _, _ = record_adaptive_tsit5_mesh(
        build_u0_1st_order_cpu(Int(d.M), d.Nj, Float64, :weak), p, d.timespan;
        tstops=tstops,
    )

    for cost_kw in ((target_F=1.0, w_time=0.15, w_power=0.05, w_tmax=1.0),
                     (target_F=1.0, w_time=0.15, w_power=0.05, w_tmax=1.0, I_min=0.99, kappa_I=5.0, kappa_S=0.0),
                     (target_F=1.0, w_time=0.15, w_power=0.05, w_tmax=1.0, S_min=0.99, kappa_S=5.0, kappa_I=0.0),
                     (target_F=1.0, w_time=0.15, w_power=0.05, w_tmax=1.0, I_min=0.9, kappa_I=3.0, S_min=0.9, kappa_S=8.0))
        # Both pulse_cost_grad_adjoint and pulse_cost_on_frozen_mesh get the
        # SAME I_min/kappa_I/S_min/kappa_S -- this is the test that would go
        # silently meaningless if pulse_cost_on_frozen_mesh had been
        # forgotten when the squared-hinge penalty was added.
        g_adj, cost_adj, inv_a, sil_a, dur_a, coh_a = pulse_cost_grad_adjoint(θ, pulse, d; cost_kw...)
        @test all(isfinite, g_adj)
        @test isfinite(cost_adj)

        J0, = pulse_cost_on_frozen_mesh(θ, pulse, d, mesh_g, mesh_e; cost_kw...)
        ε = 1e-7
        fd = similar(g_adj)
        for i in eachindex(θ)
            up = copy(θ)
            um = copy(θ)
            up[i] += ε
            um[i] -= ε
            Jp, = pulse_cost_on_frozen_mesh(up, pulse, d, mesh_g, mesh_e; cost_kw...)
            Jm, = pulse_cost_on_frozen_mesh(um, pulse, d, mesh_g, mesh_e; cost_kw...)
            fd[i] = (Jp - Jm) / (2ε)
        end
        @test _relmax(g_adj, fd) < 1e-4

        # Richardson on raw_gap (index 1): shrinking ε should not explode.
        ε2 = ε / 2
        up = copy(θ); um = copy(θ); up[1] += ε2; um[1] -= ε2
        Jp, = pulse_cost_on_frozen_mesh(up, pulse, d, mesh_g, mesh_e; cost_kw...)
        Jm, = pulse_cost_on_frozen_mesh(um, pulse, d, mesh_g, mesh_e; cost_kw...)
        fd2 = (Jp - Jm) / (2ε2)
        @test isfinite(fd2)
        @test abs(g_adj[1] - fd2) / max(abs(g_adj[1]), abs(fd2), 1e-8) < 1e-3
    end
end

@testset "checkpoint windows match full store" begin
    d = FAKE_D_ODE
    pulse = CompositePulse(1, 4, 4, d)
    θ = seed_canonical(pulse, :hs1)
    cost_kw = (target_F=1.0, w_time=0.15, w_power=0.05, w_tmax=1.0)
    # use_checkpoints=false explicitly here: the reference "full store, no
    # checkpointing" reverse path, deliberately NOT exercising checkpointing
    # at all (use_checkpoints now defaults to true, per grad_mode=:adjoint's
    # own default -- see pulse_cost_grad_adjoint's docstring).
    g_full, = pulse_cost_grad_adjoint(θ, pulse, d; cost_kw..., use_checkpoints=false)
    g_win, = pulse_cost_grad_adjoint(θ, pulse, d; cost_kw..., checkpoint_stride=2, use_checkpoints=true)
    @test g_win ≈ g_full atol=1e-10 rtol=1e-10

    # use_checkpoints=true with checkpoint_stride explicitly reset to
    # typemax(Int) (its old default) gives exactly one window spanning the
    # whole trajectory -- still correct (same gradient), but should warn
    # since it provides no memory saving. This degenerate combination no
    # longer happens under pulse_cost_grad_adjoint's OWN defaults (300, not
    # typemax(Int)) -- only when checkpoint_stride is reset explicitly.
    g_nostride = @test_logs (:warn, r"no memory saving") pulse_cost_grad_adjoint(
        θ, pulse, d; cost_kw..., use_checkpoints=true, checkpoint_stride=typemax(Int),
    )[1]
    @test g_nostride ≈ g_full atol=1e-10 rtol=1e-10
end

@testset "run_local_adam grad_mode=:adjoint smoke" begin
    pulse = CompositePulse(1, 4, 4, FAKE_D_ODE)
    u0 = seed_canonical(pulse, :hs1)
    cost_kwargs = (w_tmax=1.0, w_power=0.05, w_time=0.15)
    best_u, best_cost, _, _, _, history = run_local_adam(
        u0, pulse, FAKE_D_ODE, cost_kwargs;
        num_epochs=3, patience=3, learning_rate=0.05, label="[adj]",
        grad_mode=:adjoint, threaded_grad=false, compute=:cpu,
    )
    @test length(best_u) == n_params(pulse)
    @test isfinite(best_cost)
    @test length(history) >= 1
    @test_throws ErrorException run_local_adam(
        u0, pulse, FAKE_D_ODE, cost_kwargs; grad_mode=:nope
    )
end

@testset "pulse_cost_grad_adjoint vs _pulse_cost_grad_threaded: independent backends agree" begin
    # Two INDEPENDENTLY implemented exact differentiation methods (discrete
    # reverse-mode adjoint vs forward-mode Dual AD) of the SAME pulse_cost --
    # unlike every other adjoint test here, this is not checked against a
    # finite-difference approximation (limited to ~1e-4 by FD truncation
    # error) but directly against the OTHER production gradient backend, so
    # it can assert much tighter agreement. cost/inversion/silencing/duration/
    # coherence are expected to match EXACTLY (not just approximately): both
    # backends solve the identical ODE (rhs_1st_order!, Tsit5, same
    # tolerances) for the primal, and ForwardDiff.Dual's own value component
    # is ordinary floating-point arithmetic mirroring the primal computation
    # exactly, not a perturbed one. Only the GRADIENT itself, computed via
    # genuinely different algorithms, is expected to differ -- at the
    # Float64 roundoff floor, not at any looser tolerance.
    pulse = CompositePulse(1, 4, 4, FAKE_D_ODE)
    θ = seed_canonical(pulse, :hs1)
    for kw in ((w_tmax=1.0, w_power=0.05, w_time=0.15, target_F=1.0),
               (w_tmax=0.0, w_power=0.0, w_time=0.0, target_F=1.0),
               (w_tmax=2.0, w_power=0.2, w_time=0.5, target_F=0.0),
               (w_tmax=1.0, w_power=0.05, w_time=0.15, target_F=1.0, I_min=0.99, kappa_I=5.0, kappa_S=0.0),
               (w_tmax=1.0, w_power=0.05, w_time=0.15, target_F=1.0, S_min=0.99, kappa_S=5.0, kappa_I=0.0),
               (w_tmax=1.0, w_power=0.05, w_time=0.15, target_F=1.0, I_min=0.9, kappa_I=3.0, S_min=0.9, kappa_S=8.0),
               (w_tmax=2.0, w_power=0.2, w_time=0.5, target_F=0.0, I_min=0.9, kappa_I=6.0, S_min=0.9, kappa_S=2.0),
               # single-track: adjoint (1 fwd + 2 rev on :weak) vs threaded (one Jacobian on :weak)
               (w_tmax=1.0, w_power=0.05, w_time=0.15, target_F=1.0, track=:weak),
               (w_tmax=2.0, w_power=0.2, w_time=0.5, target_F=0.0, track=:weak, I_min=0.9, kappa_I=6.0, S_min=0.9, kappa_S=2.0))
        g_adj, cost_adj, inv_a, sil_a, dur_a, coh_a, famp_a, ret_a = pulse_cost_grad_adjoint(θ, pulse, FAKE_D_ODE; kw...)
        g_thr, cost_thr, inv_t, sil_t, dur_t, coh_t, famp_t, ret_t = _pulse_cost_grad_threaded(θ, pulse, FAKE_D_ODE; kw...)
        @test cost_adj == cost_thr
        @test inv_a == inv_t
        @test sil_a == sil_t
        @test dur_a == dur_t
        @test coh_a == coh_t
        @test famp_a == famp_t
        @test ret_a == ret_t
        # redefinition invariant: coherence is the triangle-inequality bound on |F|_⋆
        @test coh_a >= sil_a - 1e-12
        @test _relmax(g_adj, g_thr) < 1e-9
    end
end

@testset "run_local_adam grad_mode=:adjoint + anneal_direct_weights=true" begin
    # Coverage gap otherwise: anneal_direct_weights=true is exercised
    # elsewhere only under the default (:forwarddiff) grad_mode. The
    # schedule/moving-target logic in run_local_adam is grad_mode-agnostic
    # by construction (it only touches epoch_cost_kwargs/dyn_cost, whichever
    # of the three branches produced them), but that wiring deserves its own
    # direct check, not just an inference from the other two branches'
    # coverage. hop=1 is REQUIRED here -- hop=0 (the default) never anneals,
    # so this would silently stop testing anneal_direct_weights at all
    # without it.
    pulse = CompositePulse(1, 4, 4, FAKE_D_ODE)
    u0 = seed_canonical(pulse, :hs1)
    cost_kwargs = (w_tmax=1.0, w_power=0.05, w_time=0.15)
    adj = run_local_adam(u0, pulse, FAKE_D_ODE, cost_kwargs;
        hop=1, num_epochs=3, patience=3, learning_rate=0.05, label="[adj-anneal]",
        grad_mode=:adjoint, threaded_grad=false, compute=:cpu,
        anneal_direct_weights=true)
    @test length(adj[1]) == n_params(pulse)
    @test isfinite(adj[2])
    @test length(adj[6]) >= 1
    @test any(h.schedule_factor != 0.0 for h in adj[6])  # annealing genuinely active at hop=1

    # Epoch 1 starts from the SAME schedule state regardless of grad_mode
    # (the calibration seed -> schedule_factor = x_tune_alpha, applied
    # identically in all three branches), so all THREE
    # grad_mode/threaded_grad branches must reproduce the SAME sandbox
    # cost/inversion/silencing exactly (all solve the identical primal ODE
    # at epoch 1 -- see the cross-backend testset above for why cost, as
    # opposed to gradient, matches exactly rather than approximately). The
    # squared-hinge penalty (I_min/kappa_I/S_min/kappa_S, defaulted here
    # since cost_kwargs doesn't override them) is static, so this equality
    # holds trivially at every epoch, not just epoch 1 -- checked below too.
    fwd = run_local_adam(u0, pulse, FAKE_D_ODE, cost_kwargs;
        hop=1, num_epochs=1, patience=1, learning_rate=0.05, label="[fwd-anneal]",
        threaded_grad=false, anneal_direct_weights=true)
    thr = run_local_adam(u0, pulse, FAKE_D_ODE, cost_kwargs;
        hop=1, num_epochs=1, patience=1, learning_rate=0.05, label="[thr-anneal]",
        threaded_grad=true, anneal_direct_weights=true)
    @test adj[6][1].cost == fwd[6][1].cost == thr[6][1].cost
    @test adj[6][1].inversion == fwd[6][1].inversion == thr[6][1].inversion
    @test adj[6][1].silencing == fwd[6][1].silencing == thr[6][1].silencing

    # cost/inversion/silencing stay CLOSE across all three branches over
    # several epochs (not bit-exact past epoch 1: each backend's own
    # roundoff-level gradient difference feeds into Adam's next step, which
    # a nonlinear ODE solve then amplifies slightly -- so this is a loose
    # relative check, not the epoch-1 exact-equality one above).
    adj3 = run_local_adam(u0, pulse, FAKE_D_ODE, cost_kwargs;
        hop=1, num_epochs=3, patience=3, learning_rate=0.05, label="[adj-anneal3]",
        grad_mode=:adjoint, threaded_grad=false)
    fwd3 = run_local_adam(u0, pulse, FAKE_D_ODE, cost_kwargs;
        hop=1, num_epochs=3, patience=3, learning_rate=0.05, label="[fwd-anneal3]",
        threaded_grad=false)
    @test length(adj3[6]) == length(fwd3[6])
    for i in eachindex(adj3[6])
        @test adj3[6][i].cost ≈ fwd3[6][i].cost rtol=1e-4
        @test adj3[6][i].inversion ≈ fwd3[6][i].inversion rtol=1e-4
        @test adj3[6][i].silencing ≈ fwd3[6][i].silencing rtol=1e-4
    end
end
