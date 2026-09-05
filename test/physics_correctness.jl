using Test
using LinearAlgebra
using Random
using CUDA
using Distributions
using FastGaussQuadrature

if !@isdefined(state_length_2nd_order)
    include(joinpath(@__DIR__, "..", "src", "state_layout_2nd_order.jl"))
end
if !@isdefined(_unknown_initial_condition)
    include(joinpath(@__DIR__, "..", "src", "initial_conditions_1st_order.jl"))
end
if !@isdefined(product_SmSp_same)
    include(joinpath(@__DIR__, "..", "src", "initial_conditions_2nd_order.jl"))
end
if !@isdefined(rhs_2nd_order!)
    include(joinpath(@__DIR__, "..", "src", "rhs_2nd_order.jl"))
end
if !@isdefined(small_length)
    include(joinpath(@__DIR__, "..", "src", "MGPUlayout.jl"))
end
if !@isdefined(rhs_cpu!)
    include(joinpath(@__DIR__, "..", "src", "MGPUrhs_cpu.jl"))
end
if !@isdefined(total_spin_number_from_cooperativity)
    include(joinpath(@__DIR__, "..", "src", "frequency_inhomogeneity.jl"))
end
if !@isdefined(prepare_derived)
    include(joinpath(@__DIR__, "..", "src", "coupling_inhomogeneity.jl"))
    include(joinpath(@__DIR__, "..", "src", "ensemble.jl"))
    include(joinpath(@__DIR__, "..", "src", "ensemble_quadrature.jl"))
end
if !@isdefined(_with_default_ensemble_method)
    include(joinpath(@__DIR__, "..", "src", "simulation_api.jl"))
end
if !@isdefined(QRT_CLOSURE_LEVEL)
    const QRT_CLOSURE_LEVEL = :factorized_first_order_jacobian
end

const PHYS_M = 3
const PHYS_NJ = [2.0, 4.0, 6.0]

function _phys_unpack_same(u, M)
    st = unpack_state_2nd_order_u(u, M)
    Sp, Sz = st[4], st[5]
    SpSp_same, SzSp_same, SmSp_same, SzSz_same = st[9], st[10], st[11], st[12]
    return Sp, Sz, SmSp_same, SzSz_same, SpSp_same, SzSp_same
end

function _phys_rhs_ground(u, M, Nj)
    du = zero(u)
    delta = [0.0, 2π * 5e4, -2π * 5e4]
    g = [2π * 100, 2π * 90, 2π * 110]
    mask = ComplexF64.(.!Matrix(I, M, M))
    p = (0.0, 2π * 1e6, 0.0, delta, g, M, mask, t -> 0.0 + 0.0im)
    rhs_2nd_order!(du, u, p, 0.0)
    return du
end

@testset "product-state same-bin helpers" begin
    Nj, Sp, Sz = 4.0, 0.3 + 0.1im, -1.2
    @test product_SmSp_same(Nj, Sp, Sz) ≈ abs2(Sp) * (1 - 1/Nj) + Nj/2 - Sz
    @test product_SzSz_same(Nj, Sz) ≈ abs2(Sz) * (1 - 1/Nj) + Nj/4
    @test product_SpSp_same(Nj, Sp) ≈ Sp * Sp * (1 - 1/Nj)
    @test product_SzSp_same(Nj, Sz, Sp) ≈ Sz * Sp * (1 - 1/Nj)

    @test product_SmSp_same(5.0, 0.0, -2.5) ≈ 5.0
    @test product_SmSp_same(5.0, 0.0, +2.5) ≈ 0.0
    @test product_SzSz_same(5.0, -2.5) ≈ 25.0 / 4
end

@testset "C1 ground: SmSp_same = Nj and vacuum+ground is a fixed point" begin
    u = build_u0_2nd_order(PHYS_M, PHYS_NJ, :ground)
    Sp, Sz, SmSp, SzSz, SpSp, SzSp = _phys_unpack_same(u, PHYS_M)
    @test all(iszero, Sp)
    @test Sz ≈ .-PHYS_NJ ./ 2
    @test real.(SmSp) ≈ PHYS_NJ
    @test maximum(abs.(imag.(SmSp))) < 1e-15
    @test SmSp .+ 2 .* Sz ≈ zero(SmSp) atol=1e-14
    @test real.(SzSz) ≈ (PHYS_NJ .^ 2) ./ 4
    @test all(iszero, SpSp)
    @test all(iszero, SzSp)

    du = _phys_rhs_ground(u, PHYS_M, PHYS_NJ)
    @test maximum(abs.(du)) < 1e-12

    u_bad = copy(u)
    idx_smsp = 3 + 7 * PHYS_M + 1
    u_bad[idx_smsp:idx_smsp+PHYS_M-1] .= 0
    du_bad = _phys_rhs_ground(u_bad, PHYS_M, PHYS_NJ)
    @test maximum(abs.(du_bad)) > 1e-8
end

@testset "H1 product-state ICs for equator/weak/inverted" begin
    u_i = build_u0_2nd_order(PHYS_M, PHYS_NJ, :inverted)
    _, Sz_i, SmSp_i, SzSz_i, _, _ = _phys_unpack_same(u_i, PHYS_M)
    @test Sz_i ≈ PHYS_NJ ./ 2
    @test real.(SmSp_i) ≈ zero(PHYS_NJ) atol=1e-14
    @test real.(SzSz_i) ≈ (PHYS_NJ .^ 2) ./ 4

    u_e = build_u0_2nd_order(PHYS_M, PHYS_NJ, :equator)
    Sp_e, Sz_e, SmSp_e, SzSz_e, SpSp_e, SzSp_e = _phys_unpack_same(u_e, PHYS_M)
    @test Sp_e ≈ PHYS_NJ ./ 2
    @test all(iszero, Sz_e)
    @test SmSp_e ≈ product_SmSp_same.(PHYS_NJ, Sp_e, Sz_e)
    @test SzSz_e ≈ product_SzSz_same.(PHYS_NJ, Sz_e)
    @test SpSp_e ≈ product_SpSp_same.(PHYS_NJ, Sp_e)
    @test maximum(abs.(SmSp_e .- abs2.(Sp_e))) > 1e-8
    @test real.(SzSz_e) ≈ PHYS_NJ ./ 4

    u_w = build_u0_2nd_order(PHYS_M, PHYS_NJ, :weak)
    Sp_w, Sz_w, SmSp_w, SzSz_w, _, _ = _phys_unpack_same(u_w, PHYS_M)
    @test Sp_w ≈ _WEAK_SEED .* PHYS_NJ ./ 2
    @test Sz_w ≈ .-PHYS_NJ ./ 2
    @test SmSp_w ≈ product_SmSp_same.(PHYS_NJ, Sp_w, Sz_w)
    @test SzSz_w ≈ product_SzSz_same.(PHYS_NJ, Sz_w)
    @test real.(SmSp_w) ≈ PHYS_NJ .+ abs2.(Sp_w) .* (1 .- 1 ./ PHYS_NJ) atol=1e-12

    u_wi = build_u0_2nd_order(PHYS_M, PHYS_NJ, :weak_inverted)
    Sp_wi, Sz_wi, SmSp_wi, _, _, _ = _phys_unpack_same(u_wi, PHYS_M)
    @test SmSp_wi ≈ product_SmSp_same.(PHYS_NJ, Sp_wi, Sz_wi)
end

@testset "MGPU small-block prefix matches monolith product-state ICs" begin
    for kind in (:ground, :inverted, :equator, :weak, :weak_inverted)
        u = build_u0_2nd_order(PHYS_M, PHYS_NJ, kind)
        @test length(u) >= small_length(PHYS_M)
        Sp0, Sz0 = _spin_means_2nd_order(PHYS_NJ, kind)
        @test u[small_range(PHYS_M, F_Sp)] ≈ ComplexF64.(Sp0)
        @test u[small_range(PHYS_M, F_Sz)] ≈ ComplexF64.(Sz0)
        @test u[small_range(PHYS_M, F_SmSp_s)] ≈
            ComplexF64.(product_SmSp_same.(PHYS_NJ, Sp0, Sz0))
        @test u[small_range(PHYS_M, F_SzSz_s)] ≈
            ComplexF64.(product_SzSz_same.(PHYS_NJ, Sz0))
    end
end

@testset "C_ens → N conventions" begin
    κt = 2π * 1e6
    FWHM = 2π * 1e6
    g2 = (2π * 100.0)^2
    C = 0.6
    N_lor = total_spin_number_from_cooperativity(C, κt, g2, (kind=:lorentzian, FWHM=FWHM, span_gamma=2.5))
    N_gau = total_spin_number_from_cooperativity(C, κt, g2, (kind=:gaussian, FWHM=FWHM, span_sigma=3.0))
    @test N_lor ≈ C * κt * FWHM / (4 * g2)
    @test N_gau ≈ C * κt * FWHM / (4 * sqrt(π * log(2)) * g2)
    @test N_lor / N_gau ≈ sqrt(π * log(2))
end

@testset "quadrature mass" begin
    freq_L = (kind=:lorentzian, FWHM=2π * 1e6, span_gamma=2.5, renormalize=false)
    δ, pδ = _quad_frequency_nodes(freq_L, 64)
    @test length(δ) == 64
    @test sum(pδ) ≈ 2 * atan(2.5) / π rtol=1e-12
    @test maximum(abs.(δ)) ≤ tan(atan(2.5)) * (2π * 1e6 / 2) * (1 + 1e-12)

    freq_G = (kind=:gaussian, FWHM=2π * 1e6, span_sigma=3.0, renormalize=false)
    σ = gaussian_sigma_from_FWHM(freq_G.FWHM)
    δg, pg = _quad_frequency_nodes(freq_G, 48)
    @test sum(pg) ≈ 0.9973002039367398 rtol=1e-6

    g_cfg = (kind=:constant, g_value=2π * 100.0)
    _, g, p, _, _, g2, _ = _quad_coupling_bins(g_cfg, 1)
    @test g ≈ [2π * 100.0]
    @test p ≈ [1.0]
    @test g2 ≈ (2π * 100.0)^2

    x_fgq, w_fgq = _gauss_legendre_pts(16)
    x_gw, w_gw = _gauss_legendre_pts_golub_welsch(16)
    @test x_fgq ≈ x_gw rtol=1e-12 atol=1e-12
    @test w_fgq ≈ w_gw rtol=1e-12 atol=1e-12
end

@testset "order-2 default ensemble method is :auto" begin
    setting = (simulation_order=:second_order, M_delta=4, M_g=2)
    @test !hasproperty(setting, :ensemble_method)
    merged = _with_default_ensemble_method(setting, :second_order)
    @test merged.ensemble_method === :auto
    merged1 = _with_default_ensemble_method(setting, :first_order)
    @test merged1.ensemble_method === :auto
    forced = _with_default_ensemble_method(merge(setting, (ensemble_method=:histogram,)), :second_order)
    @test forced.ensemble_method === :histogram
end

@testset "layout lengths" begin
    M = 5
    @test state_length_2nd_order(M) == 3 + 9M + 4M*M
    @test global_state_length(M) == 3 + 9M + 4M*M
    @test small_length(M) == 3 + 9M
    @test large_length(M, 2) == 5 * M * 2
    @test shard_length(M, 2) == small_length(M) + large_length(M, 2)
end

@testset "CPU RHS vacuum+ground (MGPU packing)" begin
    M = PHYS_M
    Nj = PHYS_NJ
    u_mono = build_u0_2nd_order(M, Nj, :ground)
    u_cpu = zeros(ComplexF64, small_length(M) + 4 * M * M)
    u_cpu[1:small_length(M)] .= u_mono[1:small_length(M)]
    nsmall = small_length(M)
    SzSz_cross = reshape(@view(u_mono[nsmall + 3M*M + 1:nsmall + 4M*M]), M, M)
    reshape(@view(u_cpu[nsmall + 3M*M + 1:nsmall + 4M*M]), M, M) .= SzSz_cross

    du = zero(u_cpu)
    delta = Float64[0.0, 2π * 5e4, -2π * 5e4]
    g = Float64[2π * 100, 2π * 90, 2π * 110]
    rhs_cpu!(du, u_cpu, 0.0, 2π * 1e6, 0.0, delta, g, 0.0 + 0.0im)
    @test maximum(abs.(du)) < 1e-12
end

function _phys_rhs_pair(u; ke=2π * 1e6, ki=2π * 1e5, Et=0.25 + 0.1im)
    M = PHYS_M
    delta = Float64[0.0, 2π * 5e4, -2π * 5e4]
    g = Float64[2π * 100, 2π * 90, 2π * 110]
    mask = ComplexF64.(.!Matrix(I, M, M))
    p = (0.0, ke, ki, delta, g, M, mask, t -> Et)
    du2 = zero(u)
    rhs_2nd_order!(du2, u, p, 1e-6)
    du_cpu = zero(u)
    rhs_cpu!(du_cpu, u, 0.0, ke, ki, delta, g, Et)
    return du2, du_cpu
end

@testset "one 2nd-order RHS: rhs_cpu! matches rhs_2nd_order!" begin
    for kind in (:ground, :equator, :weak, :inverted)
        u = build_u0_2nd_order(PHYS_M, PHYS_NJ, kind)
        du2, du_cpu = _phys_rhs_pair(u)
        @test du_cpu ≈ du2 rtol=1e-13 atol=1e-13
    end

    rng = MersenneTwister(2710)
    u_rand = randn(rng, ComplexF64, state_length_2nd_order(PHYS_M))
    du2, du_cpu = _phys_rhs_pair(u_rand)
    @test du_cpu ≈ du2 rtol=1e-12 atol=1e-12
    @test maximum(abs.(du2)) > 1e-8
end

@testset "truncated Lorentzian C_eff vs claimed C_ens" begin
    C = 0.6
    freq = (kind=:lorentzian, FWHM=2π * 1e6, span_gamma=2.5, renormalize=false)
    _, pδ = _quad_frequency_nodes(freq, 64)
    p_g = [1.0]
    info = truncation_cooperativity(C, pδ, p_g)
    @test info.sum_p_delta ≈ 2 * atan(2.5) / π rtol=1e-12
    @test info.sum_p_g ≈ 1.0
    @test info.C_ens ≈ C
    @test info.C_eff ≈ C * info.sum_p_delta
    @test info.C_eff < C

    buf = IOBuffer()
    maybe_print_truncation_cooperativity(C, pδ, p_g, freq; io=buf)
    printed = String(take!(buf))
    @test occursin("∑p_δ", printed)
    @test occursin("effective C", printed)
    @test occursin("claimed C_ens", printed)
    @test occursin("0.4", printed)

    silent_buf = IOBuffer()
    maybe_print_truncation_cooperativity(
        C, pδ, p_g, merge(freq, (renormalize=true,)); io=silent_buf)
    @test isempty(String(take!(silent_buf)))

    cfg = (
        C_ens = C,
        M_delta = 8,
        M_g = 1,
        kappa_e = 2π * 1e6,
        kappa_i = 0.0,
        delta0 = 0.0,
        Ttotal = 1e-6,
        Nt_save = 3,
        freq_inhomogeneity = freq,
        g_inhomogeneity = (kind=:constant, g_value=2π * 100.0),
        ensemble_method = :quadrature,
    )
    d = prepare_derived(cfg; ensemble_method=:quadrature)
    @test d.C_eff ≈ C * d.sum_p_delta * d.sum_p_g
    @test d.C_eff < d.C_ens
    @test d.sum_p_delta ≈ info.sum_p_delta rtol=1e-10
end

@testset "QRT closure level is the factorized 1st-order Jacobian" begin
    @test QRT_CLOSURE_LEVEL === :factorized_first_order_jacobian
    @test QRT_CLOSURE_LEVEL !== :full_second_order_jacobian
end

# ---------------------------------------------------------------------------
# Stress helpers: CPU mirrors of the sharded MGPU layout / kernels.
# These let the three Devil's-ask areas run without a GPU.
# ---------------------------------------------------------------------------

@inline _stress_muli(z::Complex) = Complex(-imag(z), real(z))

function _stress_local_rowsums(part::EnsemblePartition, rng=MersenneTwister(1))
    ns = nshards(part)
    M = part.M
    bufs = [zeros(ComplexF64, 3M) for _ in 1:ns]
    truth = zeros(ComplexF64, 3M)
    for p in 1:ns
        r = rowsum_owned_range(part, p)
        vals = randn(rng, ComplexF64, length(r))
        bufs[p][r] .= vals
        truth[r] .= vals
    end
    return bufs, truth
end

function _stress_scatter(u_full, part::EnsemblePartition)
    M = part.M
    nsmall = small_length(M)
    oP = nsmall
    oZ = nsmall + M * M
    oM = nsmall + 2 * M * M
    oZZ = nsmall + 3 * M * M
    shards = Vector{Vector{ComplexF64}}(undef, nshards(part))
    for p in 1:nshards(part)
        mloc = part.counts[p]
        joff = part.offsets[p]
        lo = nsmall
        n = shard_length(M, mloc)
        u = zeros(ComplexF64, n)
        u[1:nsmall] .= u_full[1:nsmall]
        bs = M * mloc
        @inbounds for jl in 1:mloc
            j = joff + jl
            col = (jl - 1) * M
            for k in 1:M
                lin = (k - 1) * M + j
                u[lo + col + k]        = u_full[oP + lin]
                u[lo + bs + col + k]   = u_full[oZ + lin]
                u[lo + 2bs + col + k]  = u_full[oZ + (j - 1) * M + k]
                u[lo + 3bs + col + k]  = u_full[oM + lin]
                u[lo + 4bs + col + k]  = u_full[oZZ + lin]
            end
        end
        shards[p] = u
    end
    return shards
end

function _stress_gather(shards, part::EnsemblePartition)
    M = part.M
    nsmall = small_length(M)
    u_full = zeros(ComplexF64, global_state_length(M))
    u_full[1:nsmall] .= shards[1][1:nsmall]
    oP = nsmall
    oZ = nsmall + M * M
    oM = nsmall + 2 * M * M
    oZZ = nsmall + 3 * M * M
    for p in 1:nshards(part)
        s = shards[p]
        mloc = part.counts[p]
        joff = part.offsets[p]
        lo = nsmall
        bs = M * mloc
        @inbounds for jl in 1:mloc
            j = joff + jl
            col = (jl - 1) * M
            for k in 1:M
                lin = (k - 1) * M + j
                u_full[oP + lin]  = s[lo + col + k]
                u_full[oZ + lin]  = s[lo + bs + col + k]
                u_full[oM + lin]  = s[lo + 3bs + col + k]
                u_full[oZZ + lin] = s[lo + 4bs + col + k]
            end
        end
    end
    return u_full
end

function _stress_cross_and_rowsum!(du, u, delta, g, M, mloc, joff, lo)
    rowsum = zeros(ComplexF64, 3M)
    @inbounds a = u[1]
    ca = conj(a)
    oSp = 3
    oSz = 3 + M
    oadSp = 3 + 2M
    oadSm = 3 + 3M
    oadSz = 3 + 4M
    bs = M * mloc
    for jl in 1:mloc
        j = joff + jl
        gj = g[j]
        dj = delta[j]
        Spj = u[oSp + j]
        Szj = u[oSz + j]
        adSpj = u[oadSp + j]
        adSmj = u[oadSm + j]
        adSzj = u[oadSz + j]
        cSpj = conj(Spj)
        cadSmj = conj(adSmj)
        cadSzj = conj(adSzj)
        colbase = lo + (jl - 1) * M
        accP = zero(ComplexF64)
        accM = zero(ComplexF64)
        accZ = zero(ComplexF64)
        for k in 1:M
            i1 = colbase + k
            P = u[i1]
            Z = u[i1 + bs]
            ZT = u[i1 + 2bs]
            Mm = u[i1 + 3bs]
            ZZ = u[i1 + 4bs]
            if k == j
                z = zero(ComplexF64)
                du[i1] = z
                du[i1 + bs] = z
                du[i1 + 2bs] = z
                du[i1 + 3bs] = z
                du[i1 + 4bs] = z
            else
                gk = g[k]
                dk = delta[k]
                Spk = u[oSp + k]
                Szk = u[oSz + k]
                adSpk = u[oadSp + k]
                adSmk = u[oadSm + k]
                adSzk = u[oadSz + k]
                cSpk = conj(Spk)
                cadSmk = conj(adSmk)
                cadSzk = conj(adSzk)

                W = Spk * adSzj + ca * Z + adSpk * Szj - 2 * Spk * ca * Szj
                Ws = Spj * adSzk + ca * ZT + adSpj * Szk - 2 * Spj * ca * Szk
                du[i1] = _stress_muli((dj + dk) * P - 2 * gj * W - 2 * gk * Ws)

                U1 = Spk * cadSmj + Spj * cadSmk + a * P - 2 * Spk * Spj * a
                U2 = Spk * adSmj + ca * Mm + adSpk * cSpj - 2 * Spk * ca * cSpj
                U2s = Spj * adSmk + ca * conj(Mm) + adSpj * cSpk - 2 * Spj * ca * cSpk
                U3 = ZZ * ca + Szk * adSzj + Szj * adSzk - 2 * ca * Szk * Szj
                du[i1 + bs] = _stress_muli(dk * Z - gj * U1 + gj * U2 - 2 * gk * U3)
                du[i1 + 2bs] = _stress_muli(dj * ZT - gk * U1 + gk * U2s - 2 * gj * U3)

                Y1 = cadSzj * Spk + cadSmk * Szj + a * Z - 2 * Spk * a * Szj
                Y1s = cadSzk * Spj + cadSmj * Szk + a * ZT - 2 * Spj * a * Szk
                Y2 = ca * conj(ZT) + Szk * adSmj + cSpj * adSzk - 2 * ca * Szk * cSpj
                Y2s = ca * conj(Z) + Szj * adSmk + cSpk * adSzj - 2 * ca * Szj * cSpk
                du[i1 + 3bs] = _stress_muli((dk - dj) * Mm + 2 * gj * Y1 - 2 * gk * Y2)
                du[i1 + 4bs] = _stress_muli(gj * (Y2 - Y1s) + gk * (Y2s - Y1))

                accP += gk * P
                accM += gk * Mm
                accZ += gk * Z
            end
        end
        rb = 3 * (j - 1)
        rowsum[rb + 1] = accP
        rowsum[rb + 2] = accM
        rowsum[rb + 3] = accZ
    end
    return rowsum
end

function _stress_small_rhs!(du, u, rowsum, gsums, delta, g, Et, delta0, kappa_t, sqrt_ke, M)
    @inbounds begin
        a = u[1]
        ad_ad = u[2]
        ad_a = u[3]
    end
    ca = conj(a)
    cEt = conj(Et)
    half = 0.5
    S1, S2, S3 = gsums[1], gsums[2], gsums[3]
    du[1] = sqrt_ke * Et - _stress_muli(delta0 * a) - _stress_muli(S1) - half * kappa_t * a
    du[2] = 2 * _stress_muli(delta0 * ad_ad) + 2 * _stress_muli(S2) -
            kappa_t * ad_ad + 2 * sqrt_ke * ca * cEt
    du[3] = _stress_muli(conj(S3)) - _stress_muli(S3) - kappa_t * ad_a +
            sqrt_ke * Et * ca + sqrt_ke * cEt * a

    oSp = 3
    oSz = 3 + M
    oadSp = 3 + 2M
    oadSm = 3 + 3M
    oadSz = 3 + 4M
    oSpSp_s = 3 + 5M
    oSzSp_s = 3 + 6M
    oSmSp_s = 3 + 7M
    oSzSz_s = 3 + 8M
    @inbounds for j in 1:M
        gj = g[j]
        dj = delta[j]
        Sp = u[oSp + j]
        Sz = u[oSz + j]
        adSp = u[oadSp + j]
        adSm = u[oadSm + j]
        adSz = u[oadSz + j]
        SpSp_s = u[oSpSp_s + j]
        SzSp_s = u[oSzSp_s + j]
        SmSp_s = u[oSmSp_s + j]
        SzSz_s = u[oSzSz_s + j]
        rb = 3 * (j - 1)
        sum_gSpSp = gj * SpSp_s + rowsum[rb + 1]
        sum_gSmSp = gj * SmSp_s + rowsum[rb + 2]
        sum_gSzSp = gj * SzSp_s + rowsum[rb + 3]
        cSp = conj(Sp)
        cadSm = conj(adSm)
        cadSz = conj(adSz)
        du[oSp + j] = _stress_muli(dj * Sp - 2 * gj * adSz)
        du[oSz + j] = _stress_muli(gj * adSm - gj * cadSm)
        du[oadSp + j] = _stress_muli(delta0 * adSp + dj * adSp + sum_gSpSp -
                                    2 * gj * (2 * ca * adSz + ad_ad * Sz - 2 * ca * ca * Sz)) -
                        half * kappa_t * adSp + sqrt_ke * cEt * Sp
        du[oadSm + j] = _stress_muli(delta0 * adSm - dj * adSm + 2 * gj * Sz + sum_gSmSp +
                                    2 * gj * (cadSz * ca + a * adSz + Sz * ad_a - 2 * ca * a * Sz)) -
                        half * kappa_t * adSm + sqrt_ke * cEt * cSp
        du[oadSz + j] = _stress_muli(delta0 * adSz + sum_gSzSp -
                                    gj * (Sp + Sp * ad_a + ca * cadSm + a * adSp - 2 * Sp * ca * a) +
                                    gj * (2 * ca * adSm + ad_ad * cSp - 2 * ca * ca * cSp)) -
                        half * kappa_t * adSz + sqrt_ke * cEt * Sz
        du[oSpSp_s + j] = _stress_muli(2 * dj * SpSp_s + 2 * gj * adSp -
                                      4 * gj * (Sp * adSz + SzSp_s * ca + adSp * Sz - 2 * Sp * ca * Sz))
        du[oSzSp_s + j] = _stress_muli(dj * SzSp_s -
                                      gj * (2 * Sp * cadSm + a * SpSp_s - 2 * Sp * Sp * a) +
                                      gj * (Sp * adSm + ca * SmSp_s + adSp * cSp - 2 * Sp * ca * cSp) -
                                      2 * gj * (SzSz_s * ca + 2 * adSz * Sz - 2 * ca * Sz * Sz))
        Q1 = cadSz * Sp + SzSp_s * a + cadSm * Sz - 2 * Sp * a * Sz
        Q2 = ca * conj(SzSp_s) + cSp * adSz + adSm * Sz - 2 * ca * cSp * Sz
        du[oSmSp_s + j] = _stress_muli(2 * gj * Q1 - 2 * gj * Q2)
        du[oSzSz_s + j] = _stress_muli(gj * cadSm - gj * adSm - 2 * gj * Q1 + 2 * gj * Q2)
    end
    return nothing
end

function _stress_gsums(u, g, M)
    s1 = zero(ComplexF64)
    s2 = zero(ComplexF64)
    s3 = zero(ComplexF64)
    @inbounds for j in 1:M
        gj = g[j]
        s1 += gj * conj(u[3 + j])
        s2 += gj * u[3 + 2M + j]
        s3 += gj * u[3 + 3M + j]
    end
    return (s1, s2, s3)
end

function _stress_mgpu_rhs(u_full, delta, g, Et, delta0, kappa_e, kappa_i, part, mode::Symbol)
    M = part.M
    nsmall = small_length(M)
    shards = _stress_scatter(u_full, part)
    dus = [zeros(ComplexF64, length(s)) for s in shards]
    locals = Vector{Vector{ComplexF64}}(undef, nshards(part))
    gsums = _stress_gsums(shards[1], g, M)
    kappa_t = kappa_e + kappa_i
    sqrt_ke = sqrt(kappa_e)
    for p in 1:nshards(part)
        locals[p] = _stress_cross_and_rowsum!(
            dus[p], shards[p], delta, g, M, part.counts[p], part.offsets[p], nsmall)
    end
    assemble_rowsums!(mode, locals, part)
    for p in 1:nshards(part)
        _stress_small_rhs!(dus[p], shards[p], locals[p], gsums, delta, g,
                           Et, delta0, kappa_t, sqrt_ke, M)
    end
    return _stress_gather(dus, part)
end

function _stress_mgpu_cross_init(M, Nj, kind::Symbol)
    u = zeros(ComplexF64, state_length_2nd_order(M))
    u_mono = build_u0_2nd_order(M, Nj, kind)
    nsmall0 = small_length(M)
    u[1:nsmall0] .= u_mono[1:nsmall0]
    fill_szsz = kind === :ground || kind === :inverted ||
                kind === :weak || kind === :weak_inverted
    fill_sp_cross = kind === :equator || kind === :weak || kind === :weak_inverted
    nsmall = small_length(M)
    SpSp = reshape(@view(u[nsmall + 1:nsmall + M * M]), M, M)
    SzSp = reshape(@view(u[nsmall + M * M + 1:nsmall + 2M * M]), M, M)
    SmSp = reshape(@view(u[nsmall + 2M * M + 1:nsmall + 3M * M]), M, M)
    SzSz = reshape(@view(u[nsmall + 3M * M + 1:nsmall + 4M * M]), M, M)
    if fill_szsz
        @inbounds for k in 1:M, j in 1:M
            j == k && continue
            SzSz[j, k] = Nj[j] * Nj[k] / 4
        end
    end
    if fill_sp_cross
        sp_scale = kind === :equator ? 0.5 : _WEAK_SEED / 2
        sz_scale = kind === :equator ? 0.0 : (kind === :weak ? -0.5 : 0.5)
        @inbounds for k in 1:M, j in 1:M
            j == k && continue
            SpSp[j, k] = (sp_scale * Nj[j]) * (sp_scale * Nj[k])
            SmSp[j, k] = (sp_scale * Nj[j]) * (sp_scale * Nj[k])
            if sz_scale != 0
                SzSp[j, k] = (sz_scale * Nj[j]) * (sp_scale * Nj[k])
            end
        end
    end
    return u
end

function _stress_phys_p(M, Nj; ke=2π * 1e6, ki=2π * 1e5, Et=0.25 + 0.1im, delta0=0.0)
    delta = [2π * 5e4 * sin(2π * j / max(M, 1)) for j in 1:M]
    g = [2π * (80.0 + 15.0 * cos(2π * j / max(M, 1))) for j in 1:M]
    if M == length(PHYS_NJ) && Nj == PHYS_NJ
        delta = Float64[0.0, 2π * 5e4, -2π * 5e4]
        g = Float64[2π * 100, 2π * 90, 2π * 110]
    end
    return delta, g, ke, ki, ComplexF64(Et), delta0
end

function _stress_monolith_rhs(u, delta, g, ke, ki, Et, delta0)
    M = length(delta)
    mask = ComplexF64.(.!Matrix(I, M, M))
    p = (delta0, ke, ki, delta, g, M, mask, Returns(Et))
    du = zero(u)
    rhs_2nd_order!(du, u, p, 0.0)
    return du
end

@testset "EnsemblePartition edge sizes and uneven splits" begin
    for (M, ns) in ((1, 1), (2, 1), (2, 2), (3, 2), (5, 3), (7, 5), (8, 3),
                    (11, 4), (16, 5), (17, 16), (64, 7))
        part = EnsemblePartition(M, ns)
        @test nshards(part) == ns
        @test sum(part.counts) == M
        @test all(c -> c >= 1, part.counts)
        @test maximum(part.counts) - minimum(part.counts) <= 1
        covered = vcat((part[p] for p in 1:ns)...)
        @test covered == 1:M
        @test part.offsets[1] == 0
        for p in 2:ns
            @test part.offsets[p] == part.offsets[p-1] + part.counts[p-1]
        end
        @test last(rowsum_owned_range(part, ns)) == 3M
    end
    @test_throws ErrorException EnsemblePartition(0, 1)
    @test_throws ErrorException EnsemblePartition(3, 0)
    @test_throws ErrorException EnsemblePartition(3, 4)
    @test_throws ErrorException EnsemblePartition(5, -1)
end

@testset "rowsum backend chooser (NCCL / P2P / host ladder)" begin
    @test choose_rowsum_exchange(1; have_nccl=true, nunique_devices=1, nccl_ok=true, p2p_ok=true) === :none
    @test choose_rowsum_exchange(2; have_nccl=true, nunique_devices=2, nccl_ok=true, p2p_ok=true) === :nccl
    @test choose_rowsum_exchange(2; have_nccl=true, nunique_devices=2, nccl_ok=false, p2p_ok=true) === :p2p
    @test choose_rowsum_exchange(2; have_nccl=false, nunique_devices=2, nccl_ok=false, p2p_ok=true) === :p2p
    @test choose_rowsum_exchange(2; have_nccl=true, nunique_devices=1, nccl_ok=true, p2p_ok=true) === :p2p
    @test choose_rowsum_exchange(3; have_nccl=true, nunique_devices=2, nccl_ok=true, p2p_ok=false) === :host
    @test choose_rowsum_exchange(4; have_nccl=false, nunique_devices=4, nccl_ok=false, p2p_ok=false) === :host
    @test choose_rowsum_exchange(2; have_nccl=true, nunique_devices=2, nccl_ok=false, p2p_ok=false) === :host
end

@testset "rowsum NCCL / P2P / host agree on clean uneven partitions" begin
    for (M, ns) in ((1, 1), (2, 2), (3, 2), (5, 3), (7, 5), (8, 3), (11, 4), (17, 5), (64, 7))
        part = EnsemblePartition(M, ns)
        rng = MersenneTwister(1000 + 17 * M + ns)
        src, truth = _stress_local_rowsums(part, rng)
        for mode in (:nccl, :p2p, :host)
            bufs = [copy(b) for b in src]
            assemble_rowsums!(mode, bufs, part)
            for b in bufs
                @test b ≈ truth rtol=0 atol=0
            end
        end
        if ns == 1
            bufs = [copy(src[1])]
            assemble_rowsums!(:none, bufs, part)
            @test bufs[1] == src[1]
        end
    end
end

@testset "rowsum dirty non-owned slots: NCCL sums, P2P/host assign" begin
    part = EnsemblePartition(11, 4)
    rng = MersenneTwister(99)
    clean, truth = _stress_local_rowsums(part, rng)
    dirty = [copy(b) for b in clean]
    for p in 1:nshards(part)
        dirty[p] .+= randn(rng, ComplexF64, 3 * part.M)
        dirty[p][rowsum_owned_range(part, p)] .= clean[p][rowsum_owned_range(part, p)]
    end

    nccl = [copy(b) for b in dirty]
    assemble_rowsums_nccl!(nccl)
    p2p = [copy(b) for b in dirty]
    assemble_rowsums_p2p!(p2p, part)
    host = [copy(b) for b in dirty]
    assemble_rowsums_host!(host, part)

    @test p2p[1] ≈ truth
    @test host[1] ≈ truth
    @test p2p[1] ≈ host[1]
    @test maximum(abs.(nccl[1] .- truth)) > 1e-8
    for b in p2p
        @test b ≈ truth
    end
    for b in host
        @test b ≈ truth
    end
end

@testset "rowsum single-shard vs multi-shard owned coverage" begin
    M = 13
    full = randn(MersenneTwister(4), ComplexF64, 3M)
    one = EnsemblePartition(M, 1)
    b1 = [copy(full)]
    assemble_rowsums!(:none, b1, one)
    @test b1[1] == full

    for ns in (2, 3, 5, 12, 13)
        part = EnsemblePartition(M, ns)
        bufs = [zeros(ComplexF64, 3M) for _ in 1:ns]
        for p in 1:ns
            r = rowsum_owned_range(part, p)
            bufs[p][r] .= full[r]
        end
        for mode in (:nccl, :p2p, :host)
            got = [copy(b) for b in bufs]
            assemble_rowsums!(mode, got, part)
            for b in got
                @test b ≈ full
            end
        end
    end
end

@testset "order-2 ground / equator IC sweeps (weird Nj, M)" begin
    kinds = (:ground, :inverted, :equator, :weak, :weak_inverted)
    Nj_cases = (
        [1.0],
        [2.0],
        [0.5],
        [1e12],
        [1.0, 2.0, 3.0],
        [1.0, 1.0, 1.0],
        [0.25, 4.0, 1e6],
        [2.0, 4.0, 6.0],
        collect(1.0:7.0),
        [1e-3, 1.0, 50.0, 1e8],
    )
    for Nj in Nj_cases
        M = length(Nj)
        for kind in kinds
            u = build_u0_2nd_order(M, Nj, kind)
            Sp, Sz, SmSp, SzSz, SpSp, SzSp = _phys_unpack_same(u, M)
            Sp0, Sz0 = _spin_means_2nd_order(Nj, kind)
            @test Sp ≈ ComplexF64.(Sp0)
            @test Sz ≈ ComplexF64.(Sz0)
            @test SmSp ≈ ComplexF64.(product_SmSp_same.(Nj, Sp0, Sz0))
            @test SzSz ≈ ComplexF64.(product_SzSz_same.(Nj, Sz0))
            @test SpSp ≈ ComplexF64.(product_SpSp_same.(Nj, Sp0))
            @test SzSp ≈ ComplexF64.(product_SzSp_same.(Nj, Sz0, Sp0))
            @test u[small_range(M, F_SmSp_s)] ≈ ComplexF64.(product_SmSp_same.(Nj, Sp0, Sz0))

            st = unpack_state_2nd_order_u(u, M)
            SpSp_x, SzSp_x, SmSp_x, SzSz_x = st[13], st[14], st[15], st[16]
            @test all(iszero, diag(SpSp_x))
            @test all(iszero, diag(SzSp_x))
            @test all(iszero, diag(SmSp_x))
            @test all(iszero, diag(SzSz_x))
            @inbounds for k in 1:M, j in 1:M
                j == k && continue
                @test SpSp_x[j, k] ≈ ComplexF64(Sp0[j] * Sp0[k])
                @test SzSp_x[j, k] ≈ ComplexF64(Sz0[j] * Sp0[k])
                @test SmSp_x[j, k] ≈ ComplexF64(conj(Sp0[j]) * Sp0[k])
                @test SzSz_x[j, k] ≈ ComplexF64(Sz0[j] * Sz0[k])
            end

            u_mgpu = _stress_mgpu_cross_init(M, Nj, kind)
            @test u_mgpu ≈ u rtol=1e-13 atol=1e-13
        end

        u_g = build_u0_2nd_order(M, Nj, :ground)
        _, Sz_g, SmSp_g, _, _, _ = _phys_unpack_same(u_g, M)
        @test real.(SmSp_g) ≈ Nj
        @test SmSp_g .+ 2 .* Sz_g ≈ zero(SmSp_g) atol=1e-12 * max(1.0, maximum(abs.(Nj)))
        @test all(iszero, imag.(SmSp_g))

        u_e = build_u0_2nd_order(M, Nj, :equator)
        Sp_e, Sz_e, SmSp_e, SzSz_e, _, _ = _phys_unpack_same(u_e, M)
        @test all(iszero, Sz_e)
        @test Sp_e ≈ Nj ./ 2
        @test real.(SzSz_e) ≈ Nj ./ 4 atol=1e-12 * max(1.0, maximum(abs.(Nj)))
        @test maximum(abs.(SmSp_e .- abs2.(Sp_e))) > 0 || all(Nj .== 1)
    end
    @test_throws ErrorException build_u0_2nd_order(2, [1.0], :ground)
    @test_throws ErrorException build_u0_2nd_order(1, [1.0], :nope)
end

@testset "order-2 ground is a fixed point; equator is not" begin
    for (M, Nj) in ((1, [4.0]), (2, [1.0, 3.0]), (3, PHYS_NJ), (5, [1.0, 2.0, 0.5, 8.0, 1e3]),
                    (7, collect(range(0.75, 6.25; length=7))))
        delta, g, ke, ki, Et0, delta0 = _stress_phys_p(M, Nj; ki=2π * 2e5, Et=0)
        u_g = build_u0_2nd_order(M, Nj, :ground)
        du_g = _stress_monolith_rhs(u_g, delta, g, ke, ki, Et0, delta0)
        @test maximum(abs.(du_g)) < 1e-10 * max(1.0, maximum(abs.(g)) * maximum(abs.(Nj)))

        u_e = build_u0_2nd_order(M, Nj, :equator)
        du_e = _stress_monolith_rhs(u_e, delta, g, ke, 0.0, 0.0 + 0.0im, 0.0)
        @test all(isfinite, du_e)
        @test maximum(abs.(du_e)) > 1e-8

        u_step = copy(u_g)
        for _ in 1:8
            du = _stress_monolith_rhs(u_step, delta, g, ke, ki, Et0, delta0)
            u_step .+= 1e-12 .* du
        end
        @test maximum(abs.(u_step .- u_g)) < 1e-18 * length(u_g) + 1e-15
        @test all(isfinite, u_step)

        u_estep = copy(u_e)
        du1 = _stress_monolith_rhs(u_estep, delta, g, ke, 0.0, 0.0 + 0.0im, 0.0)
        u_estep .+= 1e-14 .* du1
        @test all(isfinite, u_estep)
        @test maximum(abs.(u_estep .- u_e)) > 0 || maximum(abs.(du1)) == 0
    end
end

@testset "CPU-sharded MGPU RHS vs monolith (single vs multi, all backends)" begin
    cases = (
        (3, PHYS_NJ, :ground),
        (3, PHYS_NJ, :equator),
        (3, PHYS_NJ, :weak),
        (3, PHYS_NJ, :inverted),
        (1, [5.0], :ground),
        (1, [5.0], :equator),
        (5, [1.0, 2.0, 0.5, 8.0, 3.0], :equator),
        (7, collect(1.0:7.0), :weak),
        (8, fill(2.0, 8), :ground),
    )
    for (M, Nj, kind) in cases
        delta, g, ke, ki, Et, delta0 = _stress_phys_p(M, Nj)
        u = build_u0_2nd_order(M, Nj, kind)
        du_mono = _stress_monolith_rhs(u, delta, g, ke, ki, Et, delta0)
        du_cpu = zero(u)
        rhs_cpu!(du_cpu, u, delta0, ke, ki, delta, g, Et)
        @test du_cpu ≈ du_mono rtol=1e-13 atol=1e-13

        shard_counts = unique(filter(ns -> 1 <= ns <= M, (1, 2, 3, 5, M)))
        dus = Dict{Tuple{Int,Symbol},Vector{ComplexF64}}()
        for ns in shard_counts
            part = EnsemblePartition(M, ns)
            modes = ns == 1 ? (:none,) : (:nccl, :p2p, :host)
            for mode in modes
                du = _stress_mgpu_rhs(u, delta, g, Et, delta0, ke, ki, part, mode)
                dus[(ns, mode)] = du
                @test du ≈ du_mono rtol=1e-11 atol=1e-11
                @test all(isfinite, du)
            end
        end
        if M >= 2
            @test dus[(1, :none)] ≈ dus[(min(3, M), :host)] rtol=1e-12 atol=1e-12
            @test dus[(min(3, M), :nccl)] ≈ dus[(min(3, M), :p2p)] rtol=0 atol=0
            @test dus[(min(3, M), :p2p)] ≈ dus[(min(3, M), :host)] rtol=0 atol=0
        end
    end

    rng = MersenneTwister(2710)
    for M in (3, 5, 8)
        delta, g, ke, ki, Et, delta0 = _stress_phys_p(M, ones(M))
        u = randn(rng, ComplexF64, state_length_2nd_order(M))
        du_mono = _stress_monolith_rhs(u, delta, g, ke, ki, Et, delta0)
        @test maximum(abs.(du_mono)) > 1e-8
        for ns in unique((1, 2, min(3, M), M))
            part = EnsemblePartition(M, ns)
            mode = ns == 1 ? :none : :host
            du = _stress_mgpu_rhs(u, delta, g, Et, delta0, ke, ki, part, mode)
            @test du ≈ du_mono rtol=1e-10 atol=1e-10
        end
        if M >= 3
            part = EnsemblePartition(M, 3)
            d_nccl = _stress_mgpu_rhs(u, delta, g, Et, delta0, ke, ki, part, :nccl)
            d_p2p = _stress_mgpu_rhs(u, delta, g, Et, delta0, ke, ki, part, :p2p)
            d_host = _stress_mgpu_rhs(u, delta, g, Et, delta0, ke, ki, part, :host)
            @test d_nccl ≈ d_p2p rtol=0 atol=0
            @test d_p2p ≈ d_host rtol=0 atol=0
        end
    end
end

@testset "rhs_cpu! vs rhs_2nd_order! numerical edges" begin
    edges = (
        (1, [1.0], :ground, 0.0, 2π * 1e6, 0.0, 0.0 + 0.0im),
        (1, [1.0], :equator, 2π * 1e5, 2π * 1e6, 2π * 1e5, 1.0 + 0.3im),
        (2, [1e-2, 1e6], :weak, 0.0, 1e-8, 1e-8, 0.0 + 0.0im),
        (4, [1.0, 1.0, 1.0, 1.0], :inverted, -2π * 1e4, 2π * 2e6, 0.0, -0.4 + 0.9im),
    )
    for (M, Nj, kind, delta0, ke, ki, Et) in edges
        u = build_u0_2nd_order(M, Nj, kind)
        delta = collect(range(-2π * 1e5, 2π * 1e5; length=M))
        g = collect(range(2π * 50, 2π * 150; length=M))
        mask = ComplexF64.(.!Matrix(I, M, M))
        p = (delta0, ke, ki, delta, g, M, mask, Returns(Et))
        du2 = zero(u)
        rhs_2nd_order!(du2, u, p, 0.0)
        du_cpu = zero(u)
        rhs_cpu!(du_cpu, u, delta0, ke, ki, delta, g, Et)
        @test du_cpu ≈ du2 rtol=1e-13 atol=1e-13
        @test all(isfinite, du2)
    end
end

@testset "GPU↔CPU 2nd-order RHS parity" begin
    gpu_ok = false
    try
        gpu_ok = CUDA.functional()
    catch
        gpu_ok = false
    end
    if !gpu_ok
        @test_skip "CUDA.functional() is false on this worker; device RHS not executed"
        return
    end

    for (M, Nj, kind) in ((PHYS_M, PHYS_NJ, :weak), (PHYS_M, PHYS_NJ, :ground),
                          (PHYS_M, PHYS_NJ, :equator), (5, [1.0, 2.0, 3.0, 4.0, 5.0], :weak))
        u = build_u0_2nd_order(M, Nj, kind)
        delta = [2π * 5e4 * sin(2π * j / M) for j in 1:M]
        g = [2π * (90.0 + 10.0 * j) for j in 1:M]
        if M == PHYS_M
            delta = [0.0, 2π * 5e4, -2π * 5e4]
            g = [2π * 100, 2π * 90, 2π * 110]
        end
        mask = ComplexF64.(.!Matrix(I, M, M))
        E = t -> 0.25 + 0.1im
        p_cpu = (0.0, 2π * 1e6, 2π * 1e5, delta, g, M, mask, E)
        du_cpu = zero(u)
        rhs_2nd_order!(du_cpu, u, p_cpu, 1e-6)

        p_gpu = (0.0, 2π * 1e6, 2π * 1e5, CuArray(delta), CuArray(g), M, CuArray(mask), E)
        du_gpu = CUDA.zeros(ComplexF64, length(u))
        rhs_2nd_order!(du_gpu, CuArray(u), p_gpu, 1e-6)
        @test Array(du_gpu) ≈ du_cpu rtol=1e-12 atol=1e-12
    end
end
