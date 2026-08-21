using Test
using ForwardDiff
using QuadGK
using Random

# Include the pulse stack without loading the full CUDA package.
include(joinpath(@__DIR__, "..", "src", "bspline.jl"))
include(joinpath(@__DIR__, "..", "src", "composite_pulse.jl"))
include(joinpath(@__DIR__, "..", "src", "canon_pulses.jl"))

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
    _, t_end, _, _ = decode(pulse, u)
    excess = max(t_end[end] - pulse.T_max, 0.0)
    @test excess > 0
    penalty0 = 0.0 * (excess / pulse.T_max)^2
    penalty1 = 1.0 * (excess / pulse.T_max)^2
    @test penalty0 == 0.0
    @test penalty1 > 0.0
    g = ForwardDiff.gradient(uu -> begin
        _, te, _, _ = decode(pulse, uu)
        ex = max(te[end] - pulse.T_max, zero(eltype(uu)))
        0.0 * (ex / pulse.T_max)^2
    end, u)
    @test all(iszero, g)
end

@testset "raw_gap Dual vs finite difference on windowed |E|" begin
    pulse = CompositePulse(1, 4, 4, FAKE_D)
    u = initial_guess(pulse; seed=1)
    t_start, t_end, _, _ = decode(pulse, u)
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
    t_start, t_end, cA, cf = decode(p1, u_hs1)
    @test all(cA .> 0)
    @test t_end[1] > t_start[1]

    p5 = CompositePulse(5, 4, 4, FAKE_D)
    u_c = seed_corpse(p5, p5.amp_scale)
    @test length(u_c) == n_params(p5)
    _, _, cA5, cf5 = decode(p5, u_c)
    @test all(cA5[:, 2] .< cA5[:, 1])  # ghost amplitude << active
    @test all(cA5[:, 4] .< cA5[:, 3])
    @test cf5[1, 2] != 0  # ghost chirp for π jump

    p7 = CompositePulse(7, 4, 4, FAKE_D)
    u_b = seed_bb1(p7, p7.amp_scale)
    @test length(u_b) == n_params(p7)
    @test_throws ErrorException seed_hs1(p5, p5.amp_scale, 1.0, 1.0)
    @test_throws ErrorException seed_canonical(p1, :corpse)
end

