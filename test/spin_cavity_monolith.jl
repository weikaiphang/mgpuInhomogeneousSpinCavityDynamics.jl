#   julia --startup-file=no test/spin_cavity_monolith.jl
#
# Covers: vacuum⊗ground fixed point, 1st-order κₜ, sharded RHS parity,
#         B-spline pack/unpack, product-state ICs, adjoint vs forward.

using Test
using LinearAlgebra
using Random
using ForwardDiff

include(joinpath(@__DIR__, "..", "src", "SpinCavityMonolith.jl"))
using .SpinCavityMonolith

const M = SpinCavityMonolith

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

@testset "1st-order RHS (κₜ = κₑ+κᵢ)" begin
    Mbin = 3
    Nj = [2.0, 4.0, 6.0]
    delta_b = [0.0, 2π * 5e4, -2π * 5e4]
    g_b = [2π * 100, 2π * 90, 2π * 110]
    kappa_e = 2π * 1e6
    kappa_i = 2π * 1e5
    delta0 = 1e4
    E_of_t = t -> 0.3 + 0.1im
    p = (delta0, kappa_e, kappa_i, delta_b, g_b, Mbin, E_of_t)
    u = M.build_u0_1st_order(Mbin, Nj, Float64, :weak)
    u[1] = 0.02 + 0.01im
    du = zeros(ComplexF64, M.state_length_1st_order(Mbin))
    M.rhs1!(du, u, p, 1.7e-6)
    a = u[1]
    s = sum(g_b .* conj.(u[2:4]))
    @test du[1] ≈ sqrt(kappa_e) * E_of_t(1.7e-6) - 1im * delta0 * a - 1im * s - 0.5 * (kappa_e + kappa_i) * a
    p0 = (delta0, kappa_e, 0.0, delta_b, g_b, Mbin, E_of_t)
    du0 = zeros(ComplexF64, length(u))
    M.rhs1!(du0, u, p0, 1.7e-6)
    @test du[1] != du0[1]
end

@testset "sharded RHS nshards=1 vs 2 parity" begin
    Mbin = 4
    Nj = [3.0, 5.0, 2.0, 8.0]
    delta_b = [0.0, 1e5, -2e5, 3e4]
    g_b = [2π * 80, 2π * 100, 2π * 90, 2π * 110]
    kappa_e = 2π * 1e6
    kappa_i = 2π * 2e5
    E0 = t -> 0.1 + 0.05im
    u = M.build_u0_2nd_order(Mbin, Nj, Float64, :equator)
    u[1] = 0.01 + 0.002im
    s1, L1, c1, o1 = M.dense_to_shards(u, Mbin, 1)
    s2, L2, c2, o2 = M.dense_to_shards(u, Mbin, 2)
    ds1 = zero(s1); dL1 = [zero(L) for L in L1]
    ds2 = zero(s2); dL2 = [zero(L) for L in L2]
    M.rhs2_sharded!(ds1, dL1, s1, L1, c1, o1, 0.0, kappa_e, kappa_i, delta_b, g_b, Mbin, E0(0.0))
    M.rhs2_sharded!(ds2, dL2, s2, L2, c2, o2, 0.0, kappa_e, kappa_i, delta_b, g_b, Mbin, E0(0.0))
    @test ds1 ≈ ds2 rtol=1e-12 atol=1e-14
    du1 = zeros(ComplexF64, M.state_length_2nd_order(Mbin))
    du2 = zeros(ComplexF64, length(du1))
    M.shards_to_dense!(du1, ds1, dL1, c1, o1, Mbin)
    M.shards_to_dense!(du2, ds2, dL2, c2, o2, Mbin)
    @test du1 ≈ du2 rtol=1e-12 atol=1e-14
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
    # equator: Sp = N/2, Sz = 0
    Sp = N / 2; Sz = 0.0
    @test M.smsp_same_product(Sp, Sz, N) ≈ abs2(Sp) * (1 - 1/N) + N/2 - Sz
    @test M.szsz_same_product(Sz, N) ≈ Sz^2 * (1 - 1/N) + N/4
    @test M.spsp_same_product(Sp, N) ≈ Sp^2 * (1 - 1/N)
    @test M.szsp_same_product(Sz, Sp, N) ≈ Sz * Sp * (1 - 1/N)
    u = M.build_u0_2nd_order(1, [N], Float64, :equator)
    st = M.unpack_state_2nd_order_u(u, 1)
    @test real(st[9][1]) ≈ M.spsp_same_product(N/2, N)          # SpSp_same
    @test real(st[10][1]) ≈ M.szsp_same_product(0.0, N/2, N)    # SzSp_same
    @test real(st[11][1]) ≈ M.smsp_same_product(N/2, 0.0, N)
    @test real(st[12][1]) ≈ M.szsz_same_product(0.0, N)
    # weak seed: Sp = ε N/2, Sz = −N/2
    uw = M.build_u0_2nd_order(1, [N], Float64, :weak)
    stw = M.unpack_state_2nd_order_u(uw, 1)
    Spw = M.WEAK_SEED * N / 2
    Szw = -N / 2
    @test real(stw[9][1]) ≈ M.spsp_same_product(Spw, N)
    @test real(stw[10][1]) ≈ M.szsp_same_product(Szw, Spw, N)
    @test real(stw[11][1]) ≈ M.smsp_same_product(Spw, Szw, N)
    # cross j≠k = mean products
    Nj = [4.0, 6.0]
    u2 = M.build_u0_2nd_order(2, Nj, Float64, :equator)
    st2 = M.unpack_state_2nd_order_u(u2, 2)
    @test st2[13][1, 2] ≈ (Nj[1] / 2) * (Nj[2] / 2)   # SpSp
    @test st2[14][1, 2] ≈ 0.0 * (Nj[2] / 2)           # SzSp
    @test st2[15][1, 2] ≈ (Nj[1] / 2) * (Nj[2] / 2)   # SmSp (real Sp)
    @test st2[16][1, 2] ≈ 0.0
    @test st2[13][1, 1] == 0 && st2[13][2, 2] == 0    # same-bin lives in small
end

@testset "5-block product-state ICs" begin
    Mbin = 4
    Nj = [3.0, 5.0, 2.0, 8.0]
    small, larges, counts, offsets = M.build_u0_2nd_mgpu(Mbin, Nj, :ground, 2)
    @test length(small) == M.mg_small_length(Mbin)
    @test length(larges) == 2
    @test sum(counts) == Mbin
    for j in 1:Mbin
        @test real(small[M.MG_NSCALAR + 7 * Mbin + j]) ≈ Nj[j] atol=1e-12  # SmSp_same
        @test real(small[M.MG_NSCALAR + 8 * Mbin + j]) ≈ (Nj[j]^2) / 4 atol=1e-12
        @test small[M.MG_NSCALAR + 5 * Mbin + j] == 0  # SpSp_same (ground)
        @test small[M.MG_NSCALAR + 6 * Mbin + j] == 0  # SzSp_same
    end
    for (p, L) in enumerate(larges)
        mloc = counts[p]; lo = offsets[p]
        for jl in 1:mloc
            k = lo + jl
            for j in 1:Mbin
                i = (jl - 1) * Mbin + j
                if j == k
                    @test L[i] == 0 && L[Mbin * mloc + i] == 0
                    @test L[3 * Mbin * mloc + i] == 0 && L[4 * Mbin * mloc + i] == 0
                else
                    @test real(L[4 * Mbin * mloc + i]) ≈ (Nj[j] * Nj[k]) / 4 atol=1e-12  # SzSz
                    @test L[2 * Mbin * mloc + i] == 0  # SzSpT ground
                end
            end
        end
    end
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
    # order2 must use the same :auto → quadrature rule
    @test M.prepare_derived(CONFIG; ensemble_method=:auto).ensemble_method === :quadrature
end

@testset "Modes API aliases + prepare" begin
    @test M._canon_mode(:forward) === :forward
    @test M._canon_mode(:forward_bspline) === :forward_bspline
    @test M._canon_mode(Symbol("forward-bspline")) === :forward_bspline
    @test M._canon_mode(:order2_bspline) === :order2_bspline
    @test M._canon_mode(Symbol("order2-bspline")) === :order2_bspline
    settings = M.load_settings(joinpath(@__DIR__, "..", "examples", "monolith_order2.jl"))
    prep = M.prepare(settings; ensemble_method=:auto)
    @test prep.d.ensemble_method === :quadrature
    @test prep.d.kappa_t == prep.d.kappa_e + prep.d.kappa_i
end

function _tiny_pulse_problem(; Ttotal=4e-8)
    κe = 2π * 1e6
    d = (
        timespan = (0.0, Ttotal),
        FWHM = 2π * 1e6,
        kappa_t = κe,
        kappa_e = κe,
        kappa_i = 0.0,
        g_mean = 2π * 100,
        sqrt_kappa_e = sqrt(κe),
        delta0 = 0.0,
        M = 1,
        Nj = [8.0],
        delta_b = [0.0],
        g_b = [2π * 100],
    )
    pulse = M.CompositePulse(1, 4, 4, d; degree=3, taper_frac=0.1)
    u = M.initial_guess(pulse; seed=3)
    return pulse, d, u, Ttotal
end

@testset "host collectives are loud; GPU bar is honest" begin
    col = M.Collective(:host, nothing, nothing)
    @test_throws ErrorException M.allreduce_sum!(col, [ones(ComplexF64, 2), ones(ComplexF64, 2)])
    @test_throws ErrorException M.allgather_shards!(
        col, [ones(ComplexF64, 2)], [ones(ComplexF64, 4)], [2], [0], 1)
    if M.gpu_count() == 0
        @test !M.cuda_functional()
        @test_throws ErrorException M.build_collectives(2)
    end
end

@testset "adjoint vs forward-mode gradient parity" begin
    pulse, d, u, Ttotal = _tiny_pulse_problem()
    # One accepted Tsit5 step so Dual-through-solve and discrete adjoint share a mesh.
    kw = (reltol=1e2, abstol=1e2, dt0=Ttotal, track=:weak)
    g_adj, c_adj, = M.pulse_cost_grad_adjoint(u, pulse, d; kw..., checkpoint_stride=typemax(Int))
    g_chk, c_chk, = M.pulse_cost_grad_adjoint(u, pulse, d; kw..., checkpoint_stride=1)
    @test c_chk ≈ c_adj rtol=1e-12
    @test g_chk ≈ g_adj rtol=1e-10 atol=1e-10
    g_fwd, c_fwd, = M.pulse_cost_grad_forward(u, pulse, d; kw..., backend=:cpu)
    @test c_fwd ≈ c_adj rtol=1e-8
    @test g_fwd ≈ g_adj rtol=5e-3 atol=1e-6
end

println("SpinCavityMonolith tests finished.")
