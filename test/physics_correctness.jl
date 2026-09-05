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

    printed = sprint() do io
        redirect_stdout(io) do
            maybe_print_truncation_cooperativity(C, pδ, p_g, freq)
        end
    end
    @test occursin("∑p_δ", printed)
    @test occursin("effective C", printed)
    @test occursin("claimed C_ens", printed)
    @test occursin(string(info.C_eff), printed) || occursin("0.4", printed)

    silent = sprint() do io
        redirect_stdout(io) do
            maybe_print_truncation_cooperativity(
                C, pδ, p_g, merge(freq, (renormalize=true,)))
        end
    end
    @test isempty(silent)

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

if isdefined(Main, :CUDA) || isdefined(@__MODULE__, :CUDA)
    try
        using CUDA
        if CUDA.functional()
            @testset "GPU↔CPU 2nd-order RHS parity" begin
                M = PHYS_M
                Nj = PHYS_NJ
                u = build_u0_2nd_order(M, Nj, :weak)
                delta = [0.0, 2π * 5e4, -2π * 5e4]
                g = [2π * 100, 2π * 90, 2π * 110]
                mask = ComplexF64.(.!Matrix(I, M, M))
                E = t -> 0.25 + 0.1im
                p_cpu = (0.0, 2π * 1e6, 0.0, delta, g, M, mask, E)
                du_cpu = zero(u)
                rhs_2nd_order!(du_cpu, u, p_cpu, 1e-6)

                p_gpu = (0.0, 2π * 1e6, 0.0, CuArray(delta), CuArray(g), M, CuArray(mask), E)
                du_gpu = CUDA.zeros(ComplexF64, length(u))
                rhs_2nd_order!(du_gpu, CuArray(u), p_gpu, 1e-6)
                @test Array(du_gpu) ≈ du_cpu rtol=1e-12 atol=1e-12
            end
        end
    catch
    end
end
