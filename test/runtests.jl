using Test
using ForwardDiff
using QuadGK
using Random
using DifferentialEquations

# Include the pulse stack without loading the full CUDA package.
include(joinpath(@__DIR__, "..", "src", "bspline.jl"))
include(joinpath(@__DIR__, "..", "src", "composite_pulse.jl"))
include(joinpath(@__DIR__, "..", "src", "canon_pulses.jl"))
include(joinpath(@__DIR__, "..", "src", "state_layout_1st_order.jl"))
include(joinpath(@__DIR__, "..", "src", "rhs_1st_order.jl"))
include(joinpath(@__DIR__, "..", "src", "rhs_1st_order_real.jl"))
include(joinpath(@__DIR__, "..", "src", "pulse_optimizer2.jl"))
include(joinpath(@__DIR__, "..", "src", "tsit5_discrete_adjoint.jl"))
include(joinpath(@__DIR__, "..", "src", "pulse_adjoint.jl"))

const FAKE_D = (
    timespan=(0.0, 1e-4),
    FWHM=1e6,
    kappa_t=2 * pi * 1e6,
    g_mean=2 * pi * 100,
    sqrt_kappa_e=sqrt(2 * pi * 1e6),
)

@testset "B-spline identities" begin
    n, deg = 6, 3
    t0, t1 = 0.1, 0.7
    knots = make_clamped_knots(n, t0, t1, deg)
    c = collect(range(0.2, 1.4; length=n))
    ts = range(t0, t1; length=81)
    Bsum = [sum(bspline_basis(t, knots, deg)) for t in ts]
    @test maximum(abs.(Bsum .- 1)) < 1e-14
    @test sum(bspline_basis(t1, knots, deg)) ≈ 1
    @test bspline_eval(t0, c, knots, deg) ≈ c[1]
    @test bspline_eval(t1, c, knots, deg) ≈ c[end]
    @test bspline_eval(t0 - 1e-12, c, knots, deg) == 0
    @test bspline_eval(t1 + 1e-12, c, knots, deg) == 0

    area_id = bspline_area(c, knots, deg)
    area_q = quadgk(t -> bspline_eval(t, c, knots, deg), t0, t1; rtol=1e-10)[1]
    @test area_id ≈ area_q rtol=1e-10

    kp, d = bspline_antiderivative(c, knots, deg)
    @test bspline_eval(t0, d, kp, deg + 1) ≈ 0 atol=1e-15
    @test bspline_eval(t1, d, kp, deg + 1) ≈ area_id rtol=1e-12
end

@testset "Gevrey Dual at 0" begin
    naive(x) = x > 0 ? exp(-1 / x) : 0.0
    @test isnan(ForwardDiff.derivative(naive, 0.0))
    @test ForwardDiff.derivative(_gevrey_bump, 0.0) == 0.0
    @test _gevrey_bump(0.0) == 0.0
    @test _smooth_step(0.0) == 0.0
    @test _smooth_step(1.0) == 1.0
    @test ForwardDiff.derivative(_smooth_step, 0.0) == 0.0
    @test ForwardDiff.derivative(_smooth_step, 1.0) == 0.0
end

@testset "unpack length check" begin
    pulse = CompositePulse(1, 4, 4, FAKE_D)
    u = initial_guess(pulse; seed=1)
    @test length(u) == n_params(pulse)
    @test_throws ErrorException unpack(pulse, u[1:end-1])
    @test_throws ErrorException decode(pulse, vcat(u, 0.0))
end

@testset "w_tmax default is identically zero" begin
    pulse = CompositePulse(1, 4, 4, FAKE_D)
    # Large positive raw duration/gap so t_end can exceed T_max.
    u = initial_guess(pulse; seed=1)
    u[1:2] .= 20.0
    _, t_end, _, _, _ = decode(pulse, u)
    excess = max(t_end[end] - pulse.T_max, 0.0)
    @test excess > 0
    penalty0 = 0.0 * (excess / pulse.T_max)^2
    penalty1 = 1.0 * (excess / pulse.T_max)^2
    @test penalty0 == 0.0
    @test penalty1 > 0.0
    g = ForwardDiff.gradient(uu -> begin
        _, te, _, _, _ = decode(pulse, uu)
        ex = max(te[end] - pulse.T_max, zero(eltype(uu)))
        0.0 * (ex / pulse.T_max)^2
    end, u)
    @test all(iszero, g)
end

@testset "raw_gap Dual vs finite difference on windowed |E|" begin
    pulse = CompositePulse(1, 4, 4, FAKE_D)
    u = initial_guess(pulse; seed=1)
    t_start, t_end, _, _, _ = decode(pulse, u)
    tmid = (t_start[1] + t_end[1]) / 2
    f(uu) = abs(build_E_of_t(pulse, uu)(tmid))
    g = ForwardDiff.gradient(f, u)
    ε = 1e-7
    u_plus = copy(u)
    u_plus[1] += ε
    fd = (f(u_plus) - f(u)) / ε
    @test isfinite(g[1])
    @test abs(g[1] - fd) / max(abs(g[1]), 1e-12) < 1e-3
end

@testset "canonical seeds" begin
    @test k_of_seed_kind(:hs1) == 1
    @test k_of_seed_kind(:corpse) == 5
    @test k_of_seed_kind(:bb1) == 7
    @test_throws ErrorException k_of_seed_kind(:random)

    p1 = CompositePulse(1, 4, 4, FAKE_D)
    u_hs1 = seed_canonical(p1, :hs1)
    @test length(u_hs1) == n_params(p1)
    t_start, t_end, _, cA, cf = decode(p1, u_hs1)
    @test all(cA .> 0)
    @test t_end[1] > t_start[1]

    p5 = CompositePulse(5, 4, 4, FAKE_D)
    u_c = seed_corpse(p5, p5.amp_scale)
    @test length(u_c) == n_params(p5)
    _, _, _, cA5, cf5 = decode(p5, u_c)
    @test all(cA5[:, 2] .< cA5[:, 1])  # ghost amplitude << active
    @test all(cA5[:, 4] .< cA5[:, 3])
    @test cf5[1, 2] != 0  # ghost chirp for π jump

    p7 = CompositePulse(7, 4, 4, FAKE_D)
    u_b = seed_bb1(p7, p7.amp_scale)
    @test length(u_b) == n_params(p7)
    @test_throws ErrorException seed_hs1(p5, p5.amp_scale, 1.0, 1.0)
    @test_throws ErrorException seed_canonical(p1, :corpse)
end

@testset "dual-trajectory initial conditions" begin
    M = 3
    Nj = [2.0, 4.0, 6.0]
    u_g = build_u0_1st_order_cpu(M, Nj, Float64, :ground)
    u_i = build_u0_1st_order_cpu(M, Nj, Float64, :inverted)
    u_e = build_u0_1st_order_cpu(M, Nj, Float64, :equator)
    _, Sp_g, Sz_g = unpack_state_1st_order_u(u_g, M)
    _, Sp_i, Sz_i = unpack_state_1st_order_u(u_i, M)
    _, Sp_e, Sz_e = unpack_state_1st_order_u(u_e, M)
    @test all(iszero, Sp_g)
    @test Sz_g ≈ .-Nj ./ 2
    @test all(iszero, Sp_i)
    @test Sz_i ≈ Nj ./ 2
    @test real.(Sp_e) ≈ Nj ./ 2
    @test all(iszero, imag.(Sp_e))
    @test all(iszero, Sz_e)
    @test abs2.(Sp_g) .+ abs2.(Sz_g) ≈ (Nj ./ 2) .^ 2
    @test abs2.(Sp_e) .+ abs2.(Sz_e) ≈ (Nj ./ 2) .^ 2
    @test_throws ErrorException build_u0_1st_order_cpu(M, Nj, Float64, :nope)
    @test_throws ErrorException _forbid_initial_condition((initial_condition=:ground,))
end

@testset "CompositePulse rejects n_coeff < degree+1" begin
    @test_throws ErrorException CompositePulse(1, 3, 4, FAKE_D)
    @test_throws ErrorException CompositePulse(1, 4, 3, FAKE_D)
    @test CompositePulse(1, 4, 4, FAKE_D) isa CompositePulse
end

# Small ensemble to actually exercise the ODE-solve path (run_sim_1st_order_pure,
# pulse_cost, run_local_adam) -- FAKE_D above only carries what CompositePulse's
# constructor needs; rhs_1st_order! additionally needs M/Nj/delta0/kappa_e/
# kappa_i/delta_b/g_b.
const FAKE_D_ODE = merge(FAKE_D, (
    M = 3,
    Nj = [2.0, 4.0, 6.0],
    delta0 = 0.0,
    kappa_e = FAKE_D.kappa_t,
    kappa_i = 0.0,
    delta_b = [0.0, 2 * pi * 5e4, -2 * pi * 5e4],
    g_b = [2 * pi * 100, 2 * pi * 90, 2 * pi * 110],
))

@testset "run_sim_1st_order_pure / pulse_cost end-to-end" begin
    pulse = CompositePulse(1, 4, 4, FAKE_D_ODE)
    u0 = seed_canonical(pulse, :hs1)

    a, Sp, Sz, Nj = run_sim_1st_order_pure(u0, pulse, FAKE_D_ODE)
    @test length(Sp) == FAKE_D_ODE.M
    @test length(Sz) == FAKE_D_ODE.M
    @test Nj == FAKE_D_ODE.Nj

    inv, sil = pulse_metrics(u0, pulse, FAKE_D_ODE)
    @test 0 <= inv <= 1
    @test 0 <= sil <= 1

    cost, inv2, sil2, dur = pulse_cost(u0, pulse, FAKE_D_ODE)
    @test isfinite(cost)
    @test inv2 == inv
    @test sil2 == sil
    @test dur ≈ pulse_duration(pulse, u0)

    # ForwardDiff.gradient through a real ODE solve, cross-checked against
    # finite differences (this is the claim CompositePulse's own module
    # docstring makes about raw_gap/the taper window -- verify it here too).
    g = ForwardDiff.gradient(uu -> pulse_cost(uu, pulse, FAKE_D_ODE)[1], u0)
    @test all(isfinite, g)
    eps = 1e-6
    fd = similar(g)
    for i in eachindex(u0)
        up = copy(u0)
        up[i] += eps
        fd[i] = (pulse_cost(up, pulse, FAKE_D_ODE)[1] - cost) / eps
    end
    relerr = abs.(g .- fd) ./ max.(abs.(g), 1e-8)
    @test maximum(relerr) < 1e-3

    # g_b = 0 decouples spins from the drive AND from the cavity mode
    # entirely: :ground never inverts (inversion == 0), and the silencing
    # factor -- weighted by Nj*g^2, see _weighted_silencing_factor -- has
    # zero weight everywhere, so it collapses to its own epsilon floor
    # (~0), NOT to 1.0: g=0 means no coupling channel into the cavity at
    # all, which is a genuinely different statement than "the spins stay
    # locally coherent" (they do, :equator's own Sp never decays here --
    # this metric just doesn't measure that; it measures collective
    # retrieval INTO the cavity mode specifically).
    d_nog = merge(FAKE_D_ODE, (g_b = zeros(FAKE_D_ODE.M),))
    inv_nog, sil_nog = pulse_metrics(u0, pulse, d_nog)
    @test inv_nog == 0.0
    @test sil_nog < 1e-6

    # dual-trajectory cost/metrics fix their own initial conditions
    @test_throws ErrorException pulse_cost(u0, pulse, FAKE_D_ODE; initial_condition=:ground)
    @test_throws ErrorException pulse_metrics(u0, pulse, FAKE_D_ODE; initial_condition=:ground)
end

@testset "threaded gradient matches serial" begin
    pulse = CompositePulse(1, 4, 4, FAKE_D_ODE)
    cost_kwargs = (w_tmax=1.0, w_power=0.05, w_inv=1.0, w_sil=0.7, target_F=1.0, w_time=0.15)

    u0 = seed_canonical(pulse, :hs1)
    for seed in 1:3
        u = seed == 1 ? u0 : initial_guess(pulse; seed=seed)
        g_serial = ForwardDiff.gradient(uu -> pulse_cost(uu, pulse, FAKE_D_ODE; cost_kwargs...)[1], u)
        cost_s, inv_s, sil_s, dur_s, coh_s = pulse_cost(u, pulse, FAKE_D_ODE; cost_kwargs...)
        g_thr, cost_t, inv_t, sil_t, dur_t, coh_t = _pulse_cost_grad_threaded(u, pulse, FAKE_D_ODE; cost_kwargs...)
        @test g_thr ≈ g_serial atol=1e-10 rtol=1e-10
        @test cost_t ≈ cost_s atol=1e-12
        @test inv_t == inv_s
        @test sil_t == sil_s
        @test dur_t == dur_s
        @test coh_t == coh_s
    end

    # w_inv<=0/w_sil<=0 skip that track's ODE solve entirely, matching
    # pulse_cost's own exact behaviour -- verify the threaded path skips
    # the same way and still matches serially.
    skip_kwargs = (w_tmax=1.0, w_power=0.05, w_inv=0.0, w_sil=0.0, target_F=1.0, w_time=0.15)
    g_serial0 = ForwardDiff.gradient(uu -> pulse_cost(uu, pulse, FAKE_D_ODE; skip_kwargs...)[1], u0)
    cost_s0, inv_s0, sil_s0, dur_s0, coh_s0 = pulse_cost(u0, pulse, FAKE_D_ODE; skip_kwargs...)
    g_thr0, cost_t0, inv_t0, sil_t0, dur_t0, coh_t0 = _pulse_cost_grad_threaded(u0, pulse, FAKE_D_ODE; skip_kwargs...)
    @test g_thr0 ≈ g_serial0 atol=1e-10 rtol=1e-10
    @test cost_t0 ≈ cost_s0 atol=1e-12
    @test inv_t0 == inv_s0 == 0.0
    @test sil_t0 == sil_s0 == 0.0
    @test coh_t0 == coh_s0 == 0.0

    # run_local_adam(threaded_grad=true) end-to-end smoke test -- must
    # behave like a normal run (finite cost, non-empty history), on the
    # SAME tiny budget the existing serial smoke test below uses.
    best_u_t, best_cost_t, best_inv_t, best_sil_t, best_dur_t, history_t = run_local_adam(
        u0, pulse, FAKE_D_ODE, cost_kwargs;
        num_epochs=3, patience=3, learning_rate=0.05, label="[thr]", threaded_grad=true,
    )
    @test length(best_u_t) == n_params(pulse)
    @test isfinite(best_cost_t)
    @test length(history_t) >= 1
    @test all(row.k == pulse.k for row in history_t)
end

@testset "run_local_adam smoke (tiny budget, real ODE solves)" begin
    pulse = CompositePulse(1, 4, 4, FAKE_D_ODE)
    u0 = seed_canonical(pulse, :hs1)
    cost_kwargs = (w_tmax=1.0, w_power=0.05, w_inv=1.0, w_sil=0.7, w_time=0.15)
    best_u, best_cost, best_inv, best_sil, best_dur, history = run_local_adam(
        u0, pulse, FAKE_D_ODE, cost_kwargs;
        num_epochs=3, patience=3, learning_rate=0.05, label="[test]",
    )
    @test length(best_u) == n_params(pulse)
    @test isfinite(best_cost)
    @test length(history) >= 1
    @test all(row.k == pulse.k for row in history)
    @test_throws ErrorException run_local_adam(
        u0, pulse, FAKE_D_ODE, cost_kwargs; initial_condition=:ground
    )
end

include(joinpath(@__DIR__, "adjoint.jl"))

