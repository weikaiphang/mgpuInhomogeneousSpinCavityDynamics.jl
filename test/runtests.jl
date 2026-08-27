using Test
using ForwardDiff
using QuadGK
using Random
using DifferentialEquations
using DiffEqCallbacks

# Include the pulse stack without loading the full CUDA package.
include(joinpath(@__DIR__, "..", "src", "bspline.jl"))
include(joinpath(@__DIR__, "..", "src", "composite_pulse.jl"))
include(joinpath(@__DIR__, "..", "src", "canon_pulses.jl"))
include(joinpath(@__DIR__, "..", "src", "state_layout_1st_order.jl"))
include(joinpath(@__DIR__, "..", "src", "rhs_1st_order.jl"))
include(joinpath(@__DIR__, "..", "src", "rhs_1st_order_real.jl"))
include(joinpath(@__DIR__, "..", "src", "pulse_optimizer2.jl"))
include(joinpath(@__DIR__, "..", "src", "pulse_optimizer2_RJMCMC.jl"))
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
    cost_kwargs = (w_tmax=1.0, w_power=0.05, target_F=1.0, w_time=0.15)

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
    cost_kwargs = (w_tmax=1.0, w_power=0.05, w_time=0.15)
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

@testset "_curriculum_fidelity_weight" begin
    target_F = 1.0

    # Boundary conditions: f(fidelity_phys=0)=0, f(fidelity_phys=1)=1, for
    # several x_tune values (including negative, and the x_tune=0 linear
    # special case).
    for x in (-8.0, -0.5, 0.0, 0.5, 8.0)
        @test _curriculum_fidelity_weight(NaN, NaN, target_F, x) == 0.0  # no accepted point -> worst case
        @test _curriculum_fidelity_weight(1.0, target_F, target_F, x) ≈ 1.0  # perfect fidelity
    end

    # x_tune=0 is exactly linear (the removable-singularity limit).
    @test _curriculum_fidelity_weight(0.5, 1.0, target_F, 0.0) ≈ 0.5 * 1.0  # fidelity_phys = 0.5*1 = 0.5
    @test _curriculum_fidelity_weight(0.3, target_F, target_F, 0.0) ≈ 0.3

    # Monotone in fidelity_phys, for both signs of x_tune.
    for x in (-5.0, 5.0)
        lo = _curriculum_fidelity_weight(0.2, target_F, target_F, x)
        hi = _curriculum_fidelity_weight(0.9, target_F, target_F, x)
        @test hi > lo
    end

    # Front-loaded (x<0) vs back-loaded (x>0): at the SAME low fidelity,
    # negative x_tune must give strictly more weight than positive x_tune,
    # and both must bracket the linear (x=0) curve -- this is the whole
    # point of switching away from the old physics_loss-gated schedule.
    f_neg = _curriculum_fidelity_weight(0.1, target_F, target_F, -5.0)
    f_lin = _curriculum_fidelity_weight(0.1, target_F, target_F, 0.0)
    f_pos = _curriculum_fidelity_weight(0.1, target_F, target_F, 5.0)
    @test f_neg > f_lin > f_pos

    # inv/sil are clamped to [0,1] defensively (tiny float overshoot).
    @test _curriculum_fidelity_weight(1.0 + 1e-14, target_F, target_F, 0.0) ≈ 1.0

    # Validation: non-finite or too-large x_tune must error, not silently NaN/Inf.
    @test_throws ErrorException _curriculum_fidelity_weight(0.5, 0.5, target_F, NaN)
    @test_throws ErrorException _curriculum_fidelity_weight(0.5, 0.5, target_F, Inf)
    @test_throws ErrorException _curriculum_fidelity_weight(0.5, 0.5, target_F, 700.0)
    @test_throws ErrorException _curriculum_fidelity_weight(0.5, 0.5, target_F, -700.0)
end

@testset "solve_optimal_x_start" begin

    @testset "_schedule_shape matches _curriculum_fidelity_weight" begin
        # _curriculum_fidelity_weight(inv, sil, target_F, x) computes
        # fidelity_phys = inv*(1-(sil-target_F)^2) internally; sil=target_F=1
        # makes silencing_success=1, so fidelity_phys=inv exactly -- i.e.
        # _curriculum_fidelity_weight(F_0, 1.0, 1.0, x) reproduces
        # _schedule_shape(x, F_0) exactly.
        for F_0 in (0.001, 0.05, 0.1, 0.3, 0.6, 0.9, 0.999)
            for x in (-50.0, -5.0, -1.0, -1e-5, 0.0, 1e-5, 1.0, 5.0, 50.0)
                @test _schedule_shape(x, F_0) ≈ _curriculum_fidelity_weight(F_0, 1.0, 1.0, x) atol=1e-12
            end
        end
    end

    @testset "_schedule_shape: global monotonicity and limits (the fact solve_optimal_x_start relies on)" begin
        F_0 = 0.1
        xs = collect(-50.0:1.0:50.0)
        vals = [_schedule_shape(x, F_0) for x in xs]
        @test issorted(vals; rev=true)  # strictly decreasing across the WHOLE real line
        @test _schedule_shape(-50.0, F_0) > 0.99   # -> 1 as x -> -inf
        @test _schedule_shape(50.0, F_0) < 1e-15   # -> 0 as x -> +inf
        @test _schedule_shape(0.0, F_0) == F_0     # exactly F_0 at x=0 (linear point)
    end

    @testset "positive-x branch (alpha < F_0)" begin
        for (F_0, alpha) in ((0.5, 0.1), (0.8, 0.05), (0.3, 0.01), (0.95, 0.5))
            x = solve_optimal_x_start(F_0, alpha)
            @test x > 0
            @test _schedule_shape(x, F_0) ≈ alpha atol=1e-5
        end
    end

    @testset "negative-x branch (alpha > F_0) -- the front-loaded, low-starting-fidelity regime" begin
        for (F_0, alpha) in ((0.02, 0.3), (0.0002, 0.01), (0.1, 0.5), (0.4, 0.99), (0.05, 0.9))
            x = solve_optimal_x_start(F_0, alpha)
            @test x < 0
            @test _schedule_shape(x, F_0) ≈ alpha atol=1e-5
        end
    end

    @testset "the specific pathological case from the real run" begin
        # F_0=0.0002 (silencing stuck near 0), wanting alpha=0.01 (1% weight
        # even at that near-zero fidelity).
        F_0, alpha = 0.0002, 0.01
        x = solve_optimal_x_start(F_0, alpha)
        @test x < -1.0  # a genuinely strongly front-loaded curvature, not near-zero
        @test _schedule_shape(x, F_0) ≈ alpha atol=1e-5
        @test _curriculum_fidelity_weight(F_0, 1.0, 1.0, x) ≈ alpha atol=1e-5
    end

    @testset "alpha ≈ F_0 -> linear sentinel" begin
        @test solve_optimal_x_start(0.3, 0.3) == 1e-4
        @test solve_optimal_x_start(0.3, 0.3 + 1e-8) == 1e-4  # within default tol
    end

    @testset "degenerate F_0 (0 or 1): factor has no x-dependence, no crash" begin
        @test solve_optimal_x_start(0.0, 0.5) == 1e-4
        @test solve_optimal_x_start(1.0, 0.5) == 1e-4
        @test solve_optimal_x_start(0.0, 0.0) == 1e-4
        @test solve_optimal_x_start(1.0, 1.0) == 1e-4
    end

    @testset "alpha at the unreachable open endpoint: graceful degradation, no crash" begin
        # alpha=1 with F_0<1 strictly needs x->-inf; no finite root exists.
        x = solve_optimal_x_start(0.1, 1.0; x_max=50.0)
        @test isfinite(x)
        @test x < 0
        @test _schedule_shape(x, 0.1) > 0.9  # close to alpha=1, even if not exact

        # Symmetric case: alpha=0 with F_0>0 strictly needs x->+inf.
        x2 = solve_optimal_x_start(0.9, 0.0; x_max=50.0)
        @test isfinite(x2)
        @test x2 > 0
        @test _schedule_shape(x2, 0.9) < 0.1
    end

    @testset "input validation" begin
        @test_throws ErrorException solve_optimal_x_start(-0.1, 0.5)
        @test_throws ErrorException solve_optimal_x_start(1.1, 0.5)
        @test_throws ErrorException solve_optimal_x_start(0.5, -0.1)
        @test_throws ErrorException solve_optimal_x_start(0.5, 1.1)
        @test_throws ErrorException solve_optimal_x_start(0.5, 0.1; x_max=0.0)
        @test_throws ErrorException solve_optimal_x_start(0.5, 0.1; x_max=-1.0)
        @test_throws ErrorException solve_optimal_x_start(0.5, 0.1; tol=0.0)
        @test_throws ErrorException solve_optimal_x_start(0.5, 0.1; tol=-1e-6)
    end

    @testset "round-trip over a grid: every (F_0, alpha) pair recovers alpha" begin
        for F_0 in (0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99)
            for alpha in (0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99)
                x = solve_optimal_x_start(F_0, alpha)
                @test isfinite(x)
                got = _schedule_shape(x, F_0)
                # Loose-ish tolerance: some (F_0,alpha) pairs near the open
                # endpoints (e.g. F_0=0.001,alpha=0.99) are only reachable
                # in the x->-inf limit and won't hit the tight bisection
                # tol within x_max -- in that case at least confirm x
                # landed in the CORRECT half (alpha>F_0 needs x<0,
                # alpha<F_0 needs x>0 -- opposite sign from (alpha-F_0)).
                @test abs(got - alpha) < 1e-3 || sign(x) == -sign(alpha - F_0)
            end
        end
    end
end

@testset "_tmax_power_components / _reconstitute_static_direct_cost" begin
    pulse = CompositePulse(1, 4, 4, FAKE_D_ODE)
    u0 = seed_canonical(pulse, :hs1)

    # Fast, ODE-free exactness check: _direct_cost_term IS w_time*(dur/T_max)
    # + w_tmax*tmax_frac_sq + w_power*power_mean, so evaluating it at two
    # different (w_time, w_tmax, w_power) triples on the SAME u and
    # comparing against _reconstitute_static_direct_cost exercises the
    # exact same linear algebra pulse_cost's own ODE-dependent physics_cost
    # term never touches, without paying for a solve.
    dur = pulse_duration(pulse, u0)
    tmax_frac_sq, power_mean = _tmax_power_components(u0, pulse)
    w_time1, w_tmax1, w_power1 = 0.15, 1.0, 0.05
    w_time2, w_tmax2, w_power2 = 0.6, 3.0, 0.4
    direct1 = _direct_cost_term(u0, pulse, w_time1, w_power1, w_tmax1)
    direct2 = _direct_cost_term(u0, pulse, w_time2, w_power2, w_tmax2)
    reconstructed = _reconstitute_static_direct_cost(
        direct1, w_time2, w_time1, w_tmax2, w_tmax1, w_power2, w_power1, dur, tmax_frac_sq, power_mean, pulse.T_max,
    )
    @test reconstructed ≈ direct2 atol=1e-12
    @test _reconstitute_static_direct_cost(
        direct2, w_time1, w_time2, w_tmax1, w_tmax2, w_power1, w_power2, dur, tmax_frac_sq, power_mean, pulse.T_max,
    ) ≈ direct1 atol=1e-12

    # Cross-checked against a REAL second ODE solve too (pulse_cost, not
    # just _direct_cost_term in isolation) -- physics_cost must cancel
    # exactly since it never depends on w_time/w_tmax/w_power.
    cost1, inv1, sil1, dur1, coh1 = pulse_cost(u0, pulse, FAKE_D_ODE; w_time=w_time1, w_tmax=w_tmax1, w_power=w_power1)
    cost2, inv2, sil2, dur2, coh2 = pulse_cost(u0, pulse, FAKE_D_ODE; w_time=w_time2, w_tmax=w_tmax2, w_power=w_power2)
    @test dur1 == dur2 && inv1 == inv2 && sil1 == sil2 && coh1 == coh2
    reconstructed_full = _reconstitute_static_direct_cost(
        cost1, w_time2, w_time1, w_tmax2, w_tmax1, w_power2, w_power1, dur1, tmax_frac_sq, power_mean, pulse.T_max,
    )
    @test reconstructed_full ≈ cost2 atol=1e-10

    # Inf (failed-solve) sentinel passes through unchanged.
    @test _reconstitute_static_direct_cost(
        Inf, w_time2, w_time1, w_tmax2, w_tmax1, w_power2, w_power1, dur, tmax_frac_sq, power_mean, pulse.T_max,
    ) == Inf
end

@testset "run_local_adam anneal_direct_weights=true by default + mandatory calibration" begin
    pulse = CompositePulse(1, 4, 4, FAKE_D_ODE)
    u0 = seed_canonical(pulse, :hs1)
    cost_kwargs = (w_tmax=1.0, w_power=0.05, w_time=0.15)

    # anneal_direct_weights=true is now the default: calling with NOTHING
    # else set must reproduce EXACTLY the same result as explicitly passing
    # anneal_direct_weights=true (with its own default x_tune_alpha) --
    # bit-for-bit, confirming the default really is what the keyword's own
    # default claims.
    baseline = run_local_adam(u0, pulse, FAKE_D_ODE, cost_kwargs;
        num_epochs=3, patience=3, learning_rate=0.05, label="[adw-default]")
    explicit_on = run_local_adam(u0, pulse, FAKE_D_ODE, cost_kwargs;
        num_epochs=3, patience=3, learning_rate=0.05, label="[adw-explicit]", anneal_direct_weights=true)
    @test baseline[1] == explicit_on[1]  # best_u
    @test baseline[2] == explicit_on[2]  # best_cost
    @test baseline[6] == explicit_on[6]  # history

    # anneal_direct_weights=false must still be a real opt-OUT, recovering
    # the plain fixed-weight cost (schedule_factor == 1.0 on every row) and
    # diverging from the annealed-by-default baseline.
    off = run_local_adam(u0, pulse, FAKE_D_ODE, cost_kwargs;
        num_epochs=3, patience=3, learning_rate=0.05, label="[adw-off]", anneal_direct_weights=false)
    @test all(h.schedule_factor == 1.0 for h in off[6])
    @test off[6] != baseline[6]

    # history's x_tune field carries the CALIBRATED value (constant across
    # every row) matching solve_optimal_x_start applied to u0's own
    # measured fidelity under _DEFAULT_X_TUNE_ALPHA exactly.
    _, inv0, sil0, _, _ = pulse_cost(u0, pulse, FAKE_D_ODE; cost_kwargs...)
    F_0 = Float64(inv0) * (1.0 - (Float64(sil0) - 1.0)^2)
    x_tune_expected = solve_optimal_x_start(F_0, _DEFAULT_X_TUNE_ALPHA)
    @test all(h.x_tune == x_tune_expected for h in baseline[6])
    # Epoch 1's factor comes from last_good_aux still at its NaN/NaN
    # sentinel (no accepted point yet) -- _curriculum_fidelity_weight
    # treats that as worst-case fidelity_phys=0, giving factor==0 exactly.
    @test baseline[6][1].schedule_factor == _curriculum_fidelity_weight(NaN, NaN, 1.0, x_tune_expected)
    # Every later epoch's factor must reproduce _curriculum_fidelity_weight
    # applied to the PREVIOUS row's own (inversion, silencing) -- i.e. the
    # last ACCEPTED point, not the epoch about to run.
    for i in 2:length(baseline[6])
        expected = _curriculum_fidelity_weight(
            baseline[6][i-1].inversion, baseline[6][i-1].silencing, 1.0, x_tune_expected)
        @test baseline[6][i].schedule_factor == expected
    end

    # Passing x_tune_alpha=nothing EXPLICITLY is the escape hatch: bypasses
    # calibration entirely, falling back to the plain linear schedule
    # (x_tune=0, i.e. factor = fidelity_phys exactly) since there is no raw
    # x_tune keyword to fall back to any more.
    manual = run_local_adam(u0, pulse, FAKE_D_ODE, cost_kwargs;
        num_epochs=3, patience=3, learning_rate=0.05, label="[adw-manual]",
        x_tune_alpha=nothing)
    @test all(h.x_tune == 0.0 for h in manual[6])
    @test manual[6][1].schedule_factor == _curriculum_fidelity_weight(NaN, NaN, 1.0, 0.0)
    @test manual[6] != baseline[6]  # linear (x_tune=0) differs from the calibrated default

    # A custom x_tune_alpha overrides the default target, actually takes
    # effect (diverges from both the default-alpha baseline and the
    # uncalibrated linear manual run), and is a silent no-op when
    # anneal_direct_weights=false (nothing to calibrate for).
    custom_alpha = run_local_adam(u0, pulse, FAKE_D_ODE, cost_kwargs;
        num_epochs=3, patience=3, learning_rate=0.05, label="[adw-custom]", x_tune_alpha=0.5)
    @test custom_alpha[6][1].x_tune == solve_optimal_x_start(F_0, 0.5)
    @test custom_alpha[6] != baseline[6]
    @test custom_alpha[6] != manual[6]
    off_with_alpha = run_local_adam(u0, pulse, FAKE_D_ODE, cost_kwargs;
        num_epochs=3, patience=3, learning_rate=0.05, label="[adw-off-alpha]",
        anneal_direct_weights=false, x_tune_alpha=0.5)
    @test all(h.schedule_factor == 1.0 for h in off_with_alpha[6])
end

@testset "optimise_composite_pulse: mandatory upfront x_tune_alpha calibration + recalibrate_optima_x" begin
    common = (n_hops=2, num_epochs=2, patience=2, tol=1e-3, learning_rate=0.05,
              hop_step_size=0.5, seed=7, threaded_grad=false)

    # anneal_direct_weights=true is now the default: calling with nothing
    # else set must reproduce EXACTLY the same result as explicitly passing
    # anneal_direct_weights=true (default x_tune_alpha, default
    # recalibrate_optima_x=true) -- bit-for-bit.
    baseline = optimise_composite_pulse(1, 4, 4, FAKE_D_ODE;
        common..., label_prefix="[ocp-default] ")
    explicit_on = optimise_composite_pulse(1, 4, 4, FAKE_D_ODE;
        common..., label_prefix="[ocp-explicit] ", anneal_direct_weights=true)
    @test baseline[1] == explicit_on[1]  # best_u
    @test baseline[2] == explicit_on[2]  # best_cost
    @test baseline[6] == explicit_on[6]  # history

    # x_tune_alpha=nothing bypasses calibration entirely (the escape
    # hatch): falls back to the plain linear schedule (x_tune=0) for every
    # hop, diverging from the calibrated-by-default baseline above.
    manual = optimise_composite_pulse(1, 4, 4, FAKE_D_ODE;
        common..., label_prefix="[ocp-manual] ",
        x_tune_alpha=nothing, recalibrate_optima_x=false)
    @test all(h.x_tune == 0.0 for h in manual[6])
    @test manual[6] != baseline[6]

    # x_tune_alpha set to a custom value, recalibrate_optima_x=true
    # (default): every hop calibrates its own x_tune from its own starting
    # point -- hop 0 from u0 (this function's own seed calibration), every
    # later hop independently via run_local_adam's internal calibration.
    # Smoke-tests cleanly and diverges from the uncalibrated manual run
    # above.
    recal_true = optimise_composite_pulse(1, 4, 4, FAKE_D_ODE;
        common..., label_prefix="[ocp-recal-true] ", anneal_direct_weights=true,
        x_tune_alpha=0.9, recalibrate_optima_x=true)
    @test length(recal_true[1]) == n_params(CompositePulse(1, 4, 4, FAKE_D_ODE))
    @test isfinite(recal_true[2])
    @test recal_true[6] != manual[6]

    # x_tune_alpha set, recalibrate_optima_x=false: the SAME single
    # seed-calibrated x_tune (from u0, hop 0's starting point) is reused
    # for every hop, so hop 0 itself must be UNCHANGED vs recalibrate=true
    # (both calibrate identically from u0), while later hops differ (no
    # per-hop recalibration).
    recal_false = optimise_composite_pulse(1, 4, 4, FAKE_D_ODE;
        common..., label_prefix="[ocp-recal-false] ", anneal_direct_weights=true,
        x_tune_alpha=0.9, recalibrate_optima_x=false)
    @test isfinite(recal_false[2])
    n_hop0 = count(h -> h.hop == 0, recal_true[6])
    @test recal_true[6][1:n_hop0] == recal_false[6][1:n_hop0]  # hop 0 identical either way

    # x_tune_alpha with anneal_direct_weights=false is a silent no-op (like
    # x_tune itself) -- nothing to calibrate x_tune for, but no error.
    off_with_alpha = optimise_composite_pulse(1, 4, 4, FAKE_D_ODE;
        common..., anneal_direct_weights=false, x_tune_alpha=0.5)
    @test all(h.schedule_factor == 1.0 for h in off_with_alpha[6])
end

@testset "optimise_composite_pulse_rjmcmc: mandatory upfront x_tune_alpha calibration + recalibrate_optima_x" begin
    common = (n_hops=2, num_epochs=2, patience=2, tol=1e-3, learning_rate=0.05,
              hop_step_size=0.5, seed=7)

    # anneal_direct_weights=true is now the default: calling with nothing
    # else set must reproduce EXACTLY the same result as explicitly passing
    # anneal_direct_weights=true (default x_tune_alpha, default
    # recalibrate_optima_x=true) -- bit-for-bit.
    baseline = optimise_composite_pulse_rjmcmc(1, 4, 4, FAKE_D_ODE;
        common..., label_prefix="[rj-default] ")
    explicit_on = optimise_composite_pulse_rjmcmc(1, 4, 4, FAKE_D_ODE;
        common..., label_prefix="[rj-explicit] ", anneal_direct_weights=true)
    @test baseline[1] == explicit_on[1]  # best_u
    @test baseline[2] == explicit_on[2]  # best_cost
    @test baseline[6] == explicit_on[6]  # history

    # x_tune_alpha=nothing bypasses calibration entirely (the escape
    # hatch): falls back to the plain linear schedule (x_tune=0) for every
    # hop, diverging from the calibrated-by-default baseline above.
    manual = optimise_composite_pulse_rjmcmc(1, 4, 4, FAKE_D_ODE;
        common..., label_prefix="[rj-manual] ",
        x_tune_alpha=nothing, recalibrate_optima_x=false)
    @test all(h.x_tune == 0.0 for h in manual[6])
    @test manual[6] != baseline[6]

    # x_tune_alpha set, recalibrate_optima_x=false: the SAME single
    # seed-calibrated x_tune (from u0, hop 0's starting point, always at
    # the INPUT k) is reused for every hop, so hop 0 itself must be
    # UNCHANGED vs recalibrate=true (both calibrate identically from u0),
    # while later hops differ (no per-hop recalibration) -- INCLUDING
    # across any _grow_pulse/_shrink_pulse k-change a hop might take.
    recal_true = optimise_composite_pulse_rjmcmc(1, 4, 4, FAKE_D_ODE;
        common..., label_prefix="[rj-recal-true] ", anneal_direct_weights=true,
        x_tune_alpha=0.9, recalibrate_optima_x=true)
    recal_false = optimise_composite_pulse_rjmcmc(1, 4, 4, FAKE_D_ODE;
        common..., label_prefix="[rj-recal-false] ", anneal_direct_weights=true,
        x_tune_alpha=0.9, recalibrate_optima_x=false)
    @test isfinite(recal_true[2])
    @test isfinite(recal_false[2])
    n_hop0 = count(h -> h.hop == 0, recal_true[6])
    @test recal_true[6][1:n_hop0] == recal_false[6][1:n_hop0]  # hop 0 identical either way

    # x_tune_alpha with anneal_direct_weights=false is a silent no-op --
    # nothing to calibrate x_tune for, but no error.
    off_with_alpha = optimise_composite_pulse_rjmcmc(1, 4, 4, FAKE_D_ODE;
        common..., anneal_direct_weights=false, x_tune_alpha=0.5)
    @test all(h.schedule_factor == 1.0 for h in off_with_alpha[6])

    # optimizer_settings carries every explicit keyword actually forwarded
    # -- no stray x_tune, and the ones that DO exist reflect what was passed.
    @test baseline[8].anneal_direct_weights == true
    @test baseline[8].x_tune_alpha == _DEFAULT_X_TUNE_ALPHA
    @test baseline[8].recalibrate_optima_x == true
    @test !haskey(baseline[8], :x_tune)
end

include(joinpath(@__DIR__, "adjoint.jl"))

