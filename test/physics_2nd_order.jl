using LinearAlgebra

const _PHYS_SRC = joinpath(@__DIR__, "..", "src")
for (pred, rel) in (
    (!isdefined(@__MODULE__, :state_length_2nd_order), "state_layout_2nd_order.jl"),
    (!isdefined(@__MODULE__, :small_length), "MGPUlayout.jl"),
    (!isdefined(@__MODULE__, :rhs_cpu!), "MGPUrhs_cpu.jl"),
    (!isdefined(@__MODULE__, :rhs_2nd_order!), "rhs_2nd_order.jl"),
    (!isdefined(@__MODULE__, :build_u0_cpu_2nd_order), "initial_conditions_2nd_order.jl"),
    (!isdefined(@__MODULE__, :_with_default_ensemble_method), "simulation_api.jl"),
    (!isdefined(@__MODULE__, :build_constant_coupling_bins), "coupling_inhomogeneity.jl"),
    (!isdefined(@__MODULE__, :_quad_frequency_nodes), "ensemble_quadrature.jl"),
)
    pred && include(joinpath(_PHYS_SRC, rel))
end

@inline function _smsp_product(Sp, Sz, Nj)
    invN = Nj > 0 ? (1 - 1 / Nj) : 0.0
    return abs2(Sp) * invN + Nj / 2 - Sz
end

@inline function _szsz_product(Sz, Nj)
    invN = Nj > 0 ? (1 - 1 / Nj) : 0.0
    return Sz * Sz * invN + Nj / 4
end

@testset "C1/H1: 2nd-order product initial conditions" begin
    M = 3
    Nj = [2.0, 4.0, 6.0]

    u_g = build_u0_cpu_2nd_order(M, Nj, :ground)
    st_g = unpack_state_2nd_order_u(u_g, M)
    @test all(iszero, st_g.Sp)
    @test real.(st_g.Sz) ≈ .-Nj ./ 2
    @test real.(st_g.SmSp_same) ≈ Nj
    @test real.(st_g.SzSz_same) ≈ (Nj .^ 2) ./ 4
    @test all(iszero, st_g.a) && all(iszero, st_g.ad_a) && all(iszero, st_g.ad_ad)

    u_i = build_u0_cpu_2nd_order(M, Nj, :inverted)
    st_i = unpack_state_2nd_order_u(u_i, M)
    @test real.(st_i.Sz) ≈ Nj ./ 2
    @test real.(st_i.SmSp_same) ≈ zeros(M) atol = 1e-14
    @test real.(st_i.SzSz_same) ≈ (Nj .^ 2) ./ 4

    u_e = build_u0_cpu_2nd_order(M, Nj, :equator)
    st_e = unpack_state_2nd_order_u(u_e, M)
    @test real.(st_e.Sp) ≈ Nj ./ 2
    @test all(iszero, st_e.Sz)
    @test real.(st_e.SmSp_same) ≈ _smsp_product.(st_e.Sp, st_e.Sz, Nj)
    @test real.(st_e.SzSz_same) ≈ _szsz_product.(real.(st_e.Sz), Nj)
    @test maximum(abs.(real.(st_e.SmSp_same) .- abs2.(st_e.Sp))) > 0.1

    u_w = build_u0_cpu_2nd_order(M, Nj, :weak)
    st_w = unpack_state_2nd_order_u(u_w, M)
    @test real.(st_w.SmSp_same) ≈ _smsp_product.(st_w.Sp, st_w.Sz, Nj)
    @test real.(st_w.SzSz_same) ≈ _szsz_product.(real.(st_w.Sz), Nj)

    u_wi = build_u0_cpu_2nd_order(M, Nj, :weak_inverted)
    st_wi = unpack_state_2nd_order_u(u_wi, M)
    @test real.(st_wi.SmSp_same) ≈ _smsp_product.(st_wi.Sp, st_wi.Sz, Nj)
    @test real.(st_wi.SzSz_same) ≈ _szsz_product.(real.(st_wi.Sz), Nj)

end

@testset "C1: vacuum ⊗ ground is a 2nd-order fixed point" begin
    M = 4
    Nj = [1.0, 2.0, 3.5, 8.0]
    u = build_u0_cpu_2nd_order(M, Nj, :ground)
    du = zero(u)
    delta_b = [0.0, 1.2e6, -8.0e5, 3.0e4]
    g_b = [1.0e3, 1.1e3, 9.0e2, 1.05e3]
    p = (0.0, 2π * 1e6, 2π * 2e5, delta_b, g_b, M, nothing, t -> 0.0 + 0im)
    rhs_2nd_order!(du, u, p, 0.0)
    @test maximum(abs.(du)) < 1e-10

    u_bad = copy(u)
    _, _, _, _, _, _, _, _, _, _, SmSp, _ = unpack_state_2nd_order_u(u_bad, M)
    SmSp .= 0
    du_bad = zero(u_bad)
    rhs_2nd_order!(du_bad, u_bad, p, 0.0)
    @test maximum(abs.(du_bad)) > 1e-6
end

@testset "H5: monolith rhs_2nd_order! ↔ rhs_cpu! parity" begin
    M = 3
    Nj = [2.0, 3.0, 5.0]
    u = build_u0_cpu_2nd_order(M, Nj, :weak)
    u[1] = 0.02 + 0.01im
    u[2] = 0.001
    u[3] = 0.03
    Random.seed!(1)
    u[4:end] .+= 1e-4 .* randn.(ComplexF64)
    delta_b = Float64[0.4, -0.2, 0.1]
    g_b = Float64[1.1, 0.9, 1.0]
    Et = 0.3 + 0.2im
    du_a = zero(u)
    du_b = zero(u)
    rhs_cpu!(du_a, u, 0.1, 1.5, 0.25, delta_b, g_b, Et)
    rhs_2nd_order!(du_b, u, (0.1, 1.5, 0.25, delta_b, g_b, M, nothing, _ -> Et), 0.0)
    @test du_a ≈ du_b atol = 1e-14
end

@testset "H4: :constant coupling rejects M_g ≠ 1" begin
    gcfg = (kind = :constant, g_value = 2π * 100)
    @test_throws ErrorException build_constant_coupling_bins(gcfg, 4)
    edges, g, p, gm, gs, g2, info = build_constant_coupling_bins(gcfg, 1)
    @test length(g) == 1
    @test p == [1.0]
    @test g2 ≈ abs2(gcfg.g_value)
end

@testset "H5: quadrature Lorentzian mass ∑p_δ ≈ 2 atan(span)/π" begin
    span = 2.5
    freq = (kind = :lorentzian, FWHM = 2π * 1e6, span_gamma = span, renormalize = false)
    δ, p = _quad_frequency_nodes(freq, 17)
    @test sum(p) ≈ 2 * atan(span) / π rtol = 1e-12
    maybe_renormalize_frequency_probs!(copy(p), freq)
    @test sum(p) < 1
end

@testset "order-2 ensemble_method defaults to :auto" begin
    sim = (simulation_order = :second_order, Ttotal = 1e-6)
    out = _with_default_ensemble_method(sim, :second_order)
    @test out.ensemble_method === :auto
    sim2 = merge(sim, (ensemble_method = :histogram,))
    @test _with_default_ensemble_method(sim2, :second_order).ensemble_method === :histogram
end

@testset "H3: κt = κe + κi in CPU / monolith 2nd-order RHS" begin
    M = 1
    Nj = [4.0]
    u = build_u0_cpu_2nd_order(M, Nj, :inverted)
    u[1] = 0.1
    du_e = zero(u)
    du_t = zero(u)
    g_b = [1.0]
    δ = [0.0]
    rhs_cpu!(du_e, u, 0.0, 2.0, 0.0, δ, g_b, 0.0im)
    rhs_cpu!(du_t, u, 0.0, 2.0, 3.0, δ, g_b, 0.0im)
    @test real(du_t[1] - du_e[1]) ≈ -0.5 * 3.0 * real(u[1]) atol = 1e-14
end
