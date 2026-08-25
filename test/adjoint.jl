# Discrete Tsit5 adjoint tests. Existing Dual @testsets above are the
# production CI contract and are not relaxed here.

function _relmax(g, fd)
    return maximum(abs.(g .- fd) ./ max.(abs.(g), abs.(fd), 1e-8))
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
    w_inv, w_sil, target_F = 1.3, 0.7, 0.4

    λI = zeros(nR)
    inversion_pullback!(λI, Sz, d.Nj, w_inv)
    zr0 = real.(Sz)
    gI = ForwardDiff.gradient(zr -> begin
        SzD = complex.(zr, imag.(Sz))
        -w_inv * _weighted_inversion(SzD, d.Nj, eltype(zr))
    end, zr0)
    @test [λI[_real_idx_zr(j, M)] for j in 1:M] ≈ gI atol=1e-10
    @test all(λI[_real_idx_zi(j, M)] == 0 for j in 1:M)
    @test λI[1] == 0 && λI[2] == 0

    λS = zeros(nR)
    silencing_pullback!(λS, Sp, d.g_b, d.Nj, w_sil, target_F)
    spr = vcat(real.(Sp), imag.(Sp))
    gS = ForwardDiff.gradient(q -> begin
        SpD = complex.(q[1:M], q[M+1:2M])
        sil = _weighted_silencing_factor(SpD, d.g_b, d.Nj, eltype(q))
        w_sil * (sil - target_F)^2
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
    inversion_pullback!(λx, Sz, d.Nj, 1.0)
    gθ = zeros(length(θ))
    ws = Tsit5DiscAdjWorkspace(M)
    tsit5_step_vjp!(copy(λx), gθ, λx, u0, t, dt, p, pulse, θ, ws)
    g_dual = ForwardDiff.gradient(θθ -> begin
        Ed = build_E_of_t(pulse, θθ)
        pd = (d.delta0, d.kappa_e, d.kappa_i, collect(d.delta_b), collect(d.g_b), M, Ed)
        u0d = build_u0_1st_order_cpu(M, d.Nj, eltype(θθ), :ground)
        u1d = tsit5_forced_step(u0d, pd, t, dt)
        _, _, Szd = unpack_state_1st_order_u(u1d, M)
        -_weighted_inversion(Szd, d.Nj, eltype(θθ))
    end, θ)
    @test all(isfinite, gθ)
    @test all(isfinite, g_dual)
    @test _relmax(gθ, g_dual) < 1e-8
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
    cost_kw = (w_inv=1.0, w_sil=0.7, target_F=1.0, w_time=0.15, w_power=0.05, w_tmax=1.0)
    g_adj, cost_adj, inv_a, sil_a, dur_a, coh_a = pulse_cost_grad_adjoint(θ, pulse, d; cost_kw...)
    @test all(isfinite, g_adj)
    @test isfinite(cost_adj)

    t_start, t_end, _, _, _ = decode(pulse, θ)
    tstops = collect(Float64, vcat(t_start, t_end))
    E = build_E_of_t(pulse, θ)
    p = _host_ode_p(d, E)
    mesh_g, _, _ = record_adaptive_tsit5_mesh(
        build_u0_1st_order_cpu(Int(d.M), d.Nj, Float64, :ground), p, d.timespan;
        tstops=tstops,
    )
    mesh_e, _, _ = record_adaptive_tsit5_mesh(
        build_u0_1st_order_cpu(Int(d.M), d.Nj, Float64, :equator), p, d.timespan;
        tstops=tstops,
    )
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

@testset "checkpoint windows match full store" begin
    d = FAKE_D_ODE
    pulse = CompositePulse(1, 4, 4, d)
    θ = seed_canonical(pulse, :hs1)
    cost_kw = (w_inv=1.0, w_sil=0.7, target_F=1.0, w_time=0.15, w_power=0.05, w_tmax=1.0)
    g_full, = pulse_cost_grad_adjoint(θ, pulse, d; cost_kw..., checkpoint_stride=typemax(Int))
    g_win, = pulse_cost_grad_adjoint(θ, pulse, d; cost_kw..., checkpoint_stride=2, use_checkpoints=true)
    @test g_win ≈ g_full atol=1e-10 rtol=1e-10
end

@testset "w_inv=0 / w_sil=0 skip tracks" begin
    d = FAKE_D_ODE
    pulse = CompositePulse(1, 4, 4, d)
    θ = seed_canonical(pulse, :hs1)
    skip_kw = (w_tmax=1.0, w_power=0.05, w_inv=0.0, w_sil=0.0, target_F=1.0, w_time=0.15)
    g_adj, cost_a, inv_a, sil_a, dur_a, coh_a = pulse_cost_grad_adjoint(θ, pulse, d; skip_kw...)
    g_fd = ForwardDiff.gradient(uu -> pulse_cost(uu, pulse, d; skip_kw...)[1], θ)
    @test g_adj ≈ g_fd atol=1e-10 rtol=1e-10
    @test inv_a == 0.0
    @test sil_a == 0.0
    @test coh_a == 0.0
    @test cost_a ≈ pulse_cost(θ, pulse, d; skip_kw...)[1] atol=1e-12
end

@testset "run_local_adam grad_mode=:adjoint smoke" begin
    pulse = CompositePulse(1, 4, 4, FAKE_D_ODE)
    u0 = seed_canonical(pulse, :hs1)
    cost_kwargs = (w_tmax=1.0, w_power=0.05, w_inv=1.0, w_sil=0.7, w_time=0.15)
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
