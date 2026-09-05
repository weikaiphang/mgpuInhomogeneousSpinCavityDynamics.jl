# Standalone monolith tests (no CUDA / DifferentialEquations required).
#   julia --project=. test/monolith_mgpu.jl
#
# Covers: (a) vacuum⊗ground 2nd-order fixed point,
#         (b) 1st-order RHS vs package rhs_1st_order!,
#         (c) B-spline pack/unpack / n_params.

using Test
using LinearAlgebra
using Random
using ForwardDiff

include(joinpath(@__DIR__, "..", "src", "monolith_mgpu.jl"))
using .NudeQuadMonolith

const M = NudeQuadMonolith

@testset "B-spline pack/unpack / param count" begin
    d = (
        timespan = (0.0, 1e-4),
        FWHM = 1e6,
        kappa_t = 2π * 1e6,
        g_mean = 2π * 100,
        sqrt_kappa_e = sqrt(2π * 1e6),
    )
    for (k, nA, nf) in ((1, 4, 4), (2, 5, 4), (3, 4, 6))
        pulse = M.CompositePulse(k, nA, nf, d)
        n = M.n_params(pulse)
        @test n == 3k + k * nA + k * nf
        u = M.initial_guess(pulse; seed=7)
        @test length(u) == n
        raw_gap, raw_dur, raw_phi0, raw_cA, raw_cf = M.unpack(pulse, u)
        u2 = M.pack(pulse, raw_gap, raw_dur, raw_phi0, raw_cA, raw_cf)
        @test u2 ≈ u
        ts, te, phi0, cA, cf = M.decode(pulse, u)
        @test length(ts) == k && length(te) == k
        @test size(cA) == (nA, k) && size(cf) == (nf, k)
        @test all(te .> ts)
        @test_throws ErrorException M.unpack(pulse, u[1:end-1])
    end
end

@testset "1st-order RHS vs package rhs_1st_order!" begin
    Mbin = 3
    Nj = [2.0, 4.0, 6.0]
    delta_b = [0.0, 2π * 5e4, -2π * 5e4]
    g_b = [2π * 100, 2π * 90, 2π * 110]
    kappa_e = 2π * 1e6
    kappa_i = 2π * 1e5   # internal loss must enter κₜ
    delta0 = 1e4
    E_of_t = t -> 0.3 + 0.1im
    p = (delta0, kappa_e, kappa_i, delta_b, g_b, Mbin, E_of_t)
    u = M.build_u0_1st_order(Mbin, Nj, Float64, :weak)
    u[1] = 0.02 + 0.01im
    du_mono = zeros(ComplexF64, M.state_length_1st_order(Mbin))
    du_pkg = zeros(ComplexF64, M.state_length_1st_order(Mbin))
    M.rhs1!(du_mono, u, p, 1.7e-6)
    M.rhs_1st_order!(du_pkg, u, p, 1.7e-6)
    @test du_mono ≈ du_pkg rtol=1e-12 atol=1e-14
    # κᵢ present: turning it off must change ȧ
    p0 = (delta0, kappa_e, 0.0, delta_b, g_b, Mbin, E_of_t)
    du0 = zeros(ComplexF64, length(u))
    M.rhs1!(du0, u, p0, 1.7e-6)
    @test du_mono[1] != du0[1]
end

@testset "vacuum+ground 2nd-order fixed point" begin
    Mbin = 4
    Nj = [3.0, 5.0, 2.0, 8.0]
    delta_b = [0.0, 1e5, -2e5, 3e4]
    g_b = [2π * 80, 2π * 100, 2π * 90, 2π * 110]
    kappa_e = 2π * 1e6
    kappa_i = 2π * 2e5
    E0 = t -> 0.0 + 0.0im
    mask = M.make_diag_mask_host(Mbin)
    p = (0.0, kappa_e, kappa_i, delta_b, g_b, Mbin, mask, E0)
    u = M.build_u0_2nd_order(Mbin, Nj, Float64, :ground)
    st = M.unpack_state_2nd_order_u(u, Mbin)
    a, ad_ad, ad_a, Sp, Sz, adSp, adSm, adSz,
        SpSp_s, SzSp_s, SmSp_s, SzSz_s = st[1:12]
    @test a == 0 && ad_ad == 0 && ad_a == 0
    @test all(iszero, Sp) && all(iszero, adSp) && all(iszero, adSm) && all(iszero, adSz)
    @test Sz ≈ .-Nj ./ 2
    # CORRECTED IC: ⟨S⁻S⁺⟩ = Nj  (vacuum⊗ground), not 0
    @test real.(SmSp_s) ≈ Nj atol=1e-12
    @test all(iszero, imag.(SmSp_s))
    # Polarized: ⟨SᶻSᶻ⟩ = Nj²/4  (matches Sz²(1-1/N)+N/4)
    @test real.(SzSz_s) ≈ (Nj .^ 2) ./ 4 atol=1e-12
    du = zeros(ComplexF64, M.state_length_2nd_order(Mbin))
    M.rhs2!(du, u, p, 0.0)
    @test maximum(abs, du) < 1e-10
end

@testset "product-state same-bin algebra" begin
    N = 10.0
    # equator
    Sp = N / 2; Sz = 0.0
    @test M.smsp_same_product(Sp, Sz, N) ≈ abs2(Sp) * (1 - 1/N) + N/2 - Sz
    @test M.szsz_same_product(Sz, N) ≈ Sz^2 * (1 - 1/N) + N/4
    u = M.build_u0_2nd_order(1, [N], Float64, :equator)
    st = M.unpack_state_2nd_order_u(u, 1)
    @test real(st[11][1]) ≈ M.smsp_same_product(N/2, 0.0, N)
    @test real(st[12][1]) ≈ M.szsz_same_product(0.0, N)
end

@testset "ensemble :auto uses quadrature for Lorentzian×constant" begin
    CONFIG = (
        C_ens = 0.6, M_delta = 5, M_g = 1, Ttotal = 1e-5, Nt_save = 5,
        delta0 = 0.0, kappa_e = 2π * 1e6, kappa_i = 0.0,
        freq_inhomogeneity = (kind=:lorentzian, FWHM=2π*1e6, span_gamma=2.5, renormalize=false),
        g_inhomogeneity = (kind=:constant, g_value=2π*100),
        ensemble_method = :auto,
    )
    plan = M.resolve_ensemble_method(CONFIG, :auto)
    @test plan.method === :quadrature
    d = M.prepare_derived(CONFIG)
    @test d.ensemble_method === :quadrature
    @test d.kappa_t == d.kappa_e + d.kappa_i
    @test d.M == 5
end

println("monolith_mgpu tests finished.")
