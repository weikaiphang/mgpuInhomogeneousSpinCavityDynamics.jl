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
    @test real(du[1] - du0[1]) ≈ real(-0.5 * kappa_i * a) atol=1e-12
end

@testset "order-2 κₜ = κₑ+κᵢ in ȧ" begin
    Mbin = 3
    Nj = [2.0, 4.0, 6.0]
    delta_b = [0.0, 1e5, -2e5]
    g_b = [2π * 100, 2π * 90, 2π * 110]
    kappa_e = 2π * 1e6
    kappa_i = 2π * 1e5
    u = M.build_u0_2nd_order(Mbin, Nj, Float64, :weak)
    u[1] = 0.02 + 0.01im
    mask = M.make_diag_mask_host(Mbin)
    E = t -> 0.3 + 0.1im
    du = zeros(ComplexF64, length(u))
    du0 = zeros(ComplexF64, length(u))
    M.rhs2!(du, u, (0.0, kappa_e, kappa_i, delta_b, g_b, Mbin, mask, E), 0.0)
    M.rhs2!(du0, u, (0.0, kappa_e, 0.0, delta_b, g_b, Mbin, mask, E), 0.0)
    @test du[1] != du0[1]
    @test du[1] - du0[1] ≈ -0.5 * kappa_i * u[1] rtol=1e-12 atol=1e-8
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
    # 1st-order: vacuum⊗ground and vacuum⊗inverted are fixed points (E=0).
    # 2nd-order: only ground is — inverted leaves `+2i g Sz` in dadSm because
    # SmSp_same = N/2−Sz is 0, not N, so the product-state cancel fails.
    for kind in (:ground, :inverted)
        u1 = M.build_u0_1st_order(Mbin, Nj, Float64, kind)
        du1 = zeros(ComplexF64, length(u1))
        M.rhs1!(du1, u1, (0.0, kappa_e, kappa_i, delta_b, g_b, Mbin, E0), 0.0)
        @test maximum(abs, du1) < 1e-12
    end
    ui = M.build_u0_2nd_order(Mbin, Nj, Float64, :inverted)
    dui = zeros(ComplexF64, length(ui))
    M.rhs2!(dui, ui, p, 0.0)
    dblk = M.unpack_state_2nd_order_du(dui, Mbin)
    @test maximum(abs, dblk[4]) > 1.0          # dadSm
    @test maximum(abs, dblk[1]) < 1e-12        # dSp
    @test maximum(abs, dblk[2]) < 1e-12        # dSz
    @test abs(dui[1]) < 1e-12
    # equator is not a free-evolution fixed point
    ue = M.build_u0_2nd_order(Mbin, Nj, Float64, :equator)
    due = zeros(ComplexF64, length(ue))
    M.rhs2!(due, ue, p, 0.0)
    @test maximum(abs, due) > 1e-8
end

@testset "product-state IC algebra, all kinds" begin
    kinds = (:ground, :inverted, :equator, :weak, :weak_inverted)
    Nj = [3.0, 5.0, 2.0, 8.0]
    Mbin = length(Nj)
    for kind in kinds
        Sp0, Sz0 = M._spin_means(Nj, kind)
        u1 = M.build_u0_1st_order(Mbin, Nj, Float64, kind)
        u2 = M.build_u0_2nd_order(Mbin, Nj, Float64, kind)
        a1, Sp1, Sz1 = M.unpack_state_1st_order_u(u1, Mbin)
        st = M.unpack_state_2nd_order_u(u2, Mbin)
        a, _, _, Sp2, Sz2, adSp, adSm, adSz,
            SpSp_s, SzSp_s, SmSp_s, SzSz_s,
            SpSp_x, SzSp_x, SmSp_x, SzSz_x = st
        @test a1 == 0 && a == 0
        @test collect(Sp1) ≈ Sp0
        @test collect(Sz1) ≈ Sz0
        @test collect(Sp2) ≈ Sp0
        @test collect(Sz2) ≈ Sz0
        @test all(iszero, adSp) && all(iszero, adSm) && all(iszero, adSz)
        for j in 1:Mbin
            N = Nj[j]; Sp = Sp0[j]; Sz = Sz0[j]
            @test real(SpSp_s[j]) ≈ real(M.spsp_same_product(Sp, N)) atol=1e-12
            @test real(SzSp_s[j]) ≈ real(M.szsp_same_product(Sz, Sp, N)) atol=1e-12
            @test real(SmSp_s[j]) ≈ real(M.smsp_same_product(Sp, Sz, N)) atol=1e-12
            @test real(SzSz_s[j]) ≈ real(M.szsz_same_product(Sz, N)) atol=1e-12
            @test SpSp_x[j, j] == 0 && SzSp_x[j, j] == 0
            @test SmSp_x[j, j] == 0 && SzSz_x[j, j] == 0
        end
        for k in 1:Mbin, j in 1:Mbin
            j == k && continue
            @test SpSp_x[j, k] ≈ Sp0[j] * Sp0[k] atol=1e-12
            @test SzSp_x[j, k] ≈ Sz0[j] * Sp0[k] atol=1e-12
            @test SmSp_x[j, k] ≈ conj(Sp0[j]) * Sp0[k] atol=1e-12
            @test SzSz_x[j, k] ≈ Sz0[j] * Sz0[k] atol=1e-12
        end
        small, larges, counts, offsets = M.build_u0_2nd_mgpu(Mbin, Nj, kind, 2)
        ud = zeros(ComplexF64, M.state_length_2nd_order(Mbin))
        M.shards_to_dense!(ud, small, larges, counts, offsets, Mbin)
        @test ud ≈ u2 atol=1e-12
    end
    @test M.smsp_same_product(0.0, -10.0 / 2, 10.0) ≈ 10.0
    @test M.smsp_same_product(0.0, 10.0 / 2, 10.0) ≈ 0.0
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
    hist = M.ensemble_method_for((kind=:uniform, FWHM=1.0), (kind=:constant, g_value=1.0))
    @test hist.method === :histogram
    @test M.resolve_ensemble_method(
        (freq_inhomogeneity=(kind=:uniform,), g_inhomogeneity=(kind=:constant,), ensemble_method=:auto),
        :auto).method === :histogram
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

@testset "1st vs 2nd mean-field consistency" begin
    Mbin = 4
    Nj = [3.0, 5.0, 2.0, 8.0]
    delta_b = [0.0, 1e5, -2e5, 3e4]
    g_b = [2π * 80, 2π * 100, 2π * 90, 2π * 110]
    kappa_e = 2π * 1e6
    kappa_i = 2π * 2e5
    delta0 = 1e4
    E = t -> 0.2 + 0.05im
    u1 = M.build_u0_1st_order(Mbin, Nj, Float64, :equator)
    u1[1] = 0.03 - 0.01im
    u2 = M.build_u0_2nd_order(Mbin, Nj, Float64, :equator)
    u2[1] = u1[1]
    a = u1[1]; ca = conj(a)
    @inbounds for j in 1:Mbin
        Sp = u2[M.IDX2_Sp_start + j - 1]
        Sz = u2[M.idx2_Sz_start(Mbin) + j - 1]
        u2[M.idx2_adSp_start(Mbin) + j - 1] = ca * Sp
        u2[M.idx2_adSm_start(Mbin) + j - 1] = ca * conj(Sp)
        u2[M.idx2_adSz_start(Mbin) + j - 1] = ca * Sz
    end
    du1 = zeros(ComplexF64, length(u1))
    du2 = zeros(ComplexF64, length(u2))
    M.rhs1!(du1, u1, (delta0, kappa_e, kappa_i, delta_b, g_b, Mbin, E), 1.2e-6)
    M.rhs2!(du2, u2, (delta0, kappa_e, kappa_i, delta_b, g_b, Mbin, M.make_diag_mask_host(Mbin), E), 1.2e-6)
    @test du2[1] ≈ du1[1] rtol=1e-12 atol=1e-14
    @test du2[4:3+Mbin] ≈ du1[2:1+Mbin] rtol=1e-12 atol=1e-14          # dSp
    @test du2[4+Mbin:3+2Mbin] ≈ du1[2+Mbin:1+2Mbin] rtol=1e-12 atol=1e-14  # dSz
end

@testset "CPU nshards is not fake-GPU; RHS hot path is allocation-free" begin
    @test M.resolve_cpu_nshards(16) == 1
    @test M.resolve_cpu_nshards(16; nshards=4) == 4
    @test M.resolve_cpu_nshards(3; nshards=8) == 3
    Mbin = 8
    Nj = fill(4.0, Mbin)
    delta_b = [2π * 1e4 * (j - 4) for j in 1:Mbin]
    g_b = fill(2π * 100.0, Mbin)
    kappa_e = 2π * 1e6
    kappa_i = 2π * 1e5
    u = M.build_u0_2nd_order(Mbin, Nj, Float64, :equator)
    u[1] = 0.01 + 0.002im
    Et = 0.1 + 0.05im
    s1, L1, c1, o1 = M.dense_to_shards(u, Mbin, 1)
    ds1 = zero(s1); dL1 = [zero(L) for L in L1]
    M.rhs2_sharded!(ds1, dL1, s1, L1, c1, o1, 0.0, kappa_e, kappa_i, delta_b, g_b, Mbin, Et)
    ds2 = zero(s1); dL2 = [zero(L) for L in L1]
    allocs = @allocated M.rhs2_sharded!(ds2, dL2, s1, L1, c1, o1, 0.0, kappa_e, kappa_i, delta_b, g_b, Mbin, Et)
    @test ds1 ≈ ds2 rtol=1e-15 atol=1e-15
    if Threads.nthreads() == 1
        @test allocs == 0
    else
        @test allocs < 8192
    end
    s4, L4, c4, o4 = M.dense_to_shards(u, Mbin, 4)
    ds4 = zero(s4); dL4 = [zero(L) for L in L4]
    M.rhs2_sharded!(ds4, dL4, s4, L4, c4, o4, 0.0, kappa_e, kappa_i, delta_b, g_b, Mbin, Et)
    du1 = zeros(ComplexF64, M.state_length_2nd_order(Mbin))
    du4 = zeros(ComplexF64, length(du1))
    M.shards_to_dense!(du1, ds1, dL1, c1, o1, Mbin)
    M.shards_to_dense!(du4, ds4, dL4, c4, o4, Mbin)
    @test du1 ≈ du4 rtol=1e-12 atol=1e-14
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
