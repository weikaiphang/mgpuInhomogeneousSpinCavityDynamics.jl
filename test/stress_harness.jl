# Stress harness for tip 05bed847a8fa7e5b8bf3e92aca2d2977796e9d73 (PR #4).
# Physics/math first, then CPU layout / allocs / API footguns.
#   julia --project=. --startup-file=no test/stress_harness.jl
#
# Label: stress-harness (not a product fix). Does not touch src/.
# Gate (1) live ≥2-GPU NCCL/P2P is DEFERRED — this file is CPU-only.
#
# Ported from PR #7 (tip 298060b) and re-pointed at 05bed84 Demiurge claims:
#   RHS2Work zero-alloc hot path, @threads, Tsit5 no Complex temps, nshards=1.

using Test
using LinearAlgebra
using Random
using ForwardDiff

if !isdefined(Main, :SpinCavityMonolith)
    include(joinpath(@__DIR__, "..", "src", "SpinCavityMonolith.jl"))
    using .SpinCavityMonolith
end
const SCM = Main.SpinCavityMonolith

const IC_KINDS = (:ground, :inverted, :equator, :weak, :weak_inverted, :custom)
const TIP_SHA = "05bed847a8fa7e5b8bf3e92aca2d2977796e9d73"

@testset "stress harness tip $TIP_SHA" begin

function _cfg(; fk=:lorentzian, gk=:constant, Md=5, Mg=1, ke=2π * 1e6, ki=2π * 1e5, C=0.6)
    freq = (kind=fk, FWHM=2π * 1e6, span_gamma=2.5, span_sigma=3.0, renormalize=false)
    g = if gk === :constant
        (kind=:constant, g_value=2π * 100)
    elseif gk === :gaussian
        (kind=:gaussian, mean=2π * 100, std=2π * 10, span_sigma=3.0, renormalize=true)
    elseif gk === :powerlaw_g
        (kind=:powerlaw_g, alpha=1.5, g_min=2π * 50, g_max=2π * 200, renormalize=true)
    else
        (kind=gk,)
    end
    return (C_ens=C, M_delta=Md, M_g=Mg, Ttotal=1e-5, Nt_save=5,
            delta0=0.0, kappa_e=ke, kappa_i=ki,
            freq_inhomogeneity=freq, g_inhomogeneity=g)
end

function _product_lift_2nd(a, Sp, Sz, Nj)
    Mb = length(Nj)
    u = zeros(ComplexF64, SCM.state_length_2nd_order(Mb))
    u[1] = a
    u[2] = conj(a)^2
    u[3] = abs2(a)
    ca = conj(a)
    @inbounds for j in 1:Mb
        N = Float64(Nj[j]); sp = ComplexF64(Sp[j]); sz = ComplexF64(Sz[j])
        u[SCM.IDX2_Sp_start + j - 1] = sp
        u[SCM.idx2_Sz_start(Mb) + j - 1] = sz
        u[SCM.idx2_adSp_start(Mb) + j - 1] = ca * sp
        u[SCM.idx2_adSm_start(Mb) + j - 1] = ca * conj(sp)
        u[SCM.idx2_adSz_start(Mb) + j - 1] = ca * sz
        u[SCM.idx2_adSz_start(Mb) + Mb + j - 1] = SCM.spsp_same_product(sp, N)
        u[SCM.idx2_adSz_start(Mb) + 2Mb + j - 1] = SCM.szsp_same_product(sz, sp, N)
        u[SCM.idx2_adSz_start(Mb) + 3Mb + j - 1] = SCM.smsp_same_product(sp, sz, N)
        u[SCM.idx2_adSz_start(Mb) + 4Mb + j - 1] = SCM.szsz_same_product(sz, N)
    end
    base = 3 + 9Mb
    @inbounds for k in 1:Mb, j in 1:Mb
        if j != k
            u[base + (k - 1) * Mb + j] = Sp[j] * Sp[k]
            u[base + Mb * Mb + (k - 1) * Mb + j] = Sz[j] * Sp[k]
            u[base + 2Mb * Mb + (k - 1) * Mb + j] = conj(Sp[j]) * Sp[k]
            u[base + 3Mb * Mb + (k - 1) * Mb + j] = Sz[j] * Sz[k]
        end
    end
    return u
end

function _rhs2_setup(Mb)
    Nj = fill(4.0, Mb)
    delta_b = [1e5 * sin(0.7 * j) for j in 1:Mb]
    g_b = fill(2π * 100.0, Mb)
    u = SCM.build_u0_2nd_order(Mb, Nj, Float64, :equator)
    u[1] = 0.01 + 0.002im
    s, L, c, o = SCM.dense_to_shards(u, Mb, 1)
    ds = zero(s); dL = [zero(x) for x in L]
    return (ds, dL, s, L, c, o, 0.0, 2π * 1e6, 2π * 2e5, delta_b, g_b, Mb, 0.1 + 0.05im)
end

# ---------------------------------------------------------------------------
# C1: same-bin product-state closures for every IC kind (+ edges)
# ---------------------------------------------------------------------------
@testset "C1 same-bin product closures — all IC kinds" begin
    sweeps = (
        [3.0, 5.0, 2.0, 8.0],
        [1.0],
        [0.0, 4.0],
        [1e-16, 3.0],
        [10.0, 10.0, 10.0],
    )
    for Nj in sweeps, kind in IC_KINDS
        Mb = length(Nj)
        u = SCM.build_u0_2nd_order(Mb, Nj, Float64, kind)
        st = SCM.unpack_state_2nd_order_u(u, Mb)
        Sp0, Sz0 = SCM._spin_means(Nj, kind)
        @test st[1] == 0 && st[2] == 0 && st[3] == 0
        @test all(iszero, st[6]) && all(iszero, st[7]) && all(iszero, st[8])
        for j in 1:Mb
            @test real(st[9][j]) ≈ real(SCM.spsp_same_product(Sp0[j], Nj[j])) atol=1e-12
            @test real(st[10][j]) ≈ real(SCM.szsp_same_product(Sz0[j], Sp0[j], Nj[j])) atol=1e-12
            @test real(st[11][j]) ≈ real(SCM.smsp_same_product(Sp0[j], Sz0[j], Nj[j])) atol=1e-12
            @test real(st[12][j]) ≈ real(SCM.szsz_same_product(Sz0[j], Nj[j])) atol=1e-12
            @test st[13][j, j] == 0 && st[14][j, j] == 0
            @test st[15][j, j] == 0 && st[16][j, j] == 0
        end
        if kind === :ground
            @test real.(st[11]) ≈ Nj atol=1e-12
        elseif kind === :inverted
            @test real.(st[11]) ≈ zero.(Nj) atol=1e-12
        end
    end
end

# ---------------------------------------------------------------------------
# H1: host/dense ↔ mgpu IC + Hermitian cross-bin SmSp / symmetric SzSz
# ---------------------------------------------------------------------------
@testset "H1 host↔mgpu ICs + cross-bin hermiticity — all IC kinds" begin
    Nj = [3.0, 5.0, 2.0, 8.0]
    Mb = length(Nj)
    for kind in IC_KINDS
        u = SCM.build_u0_2nd_order(Mb, Nj, Float64, kind)
        st = SCM.unpack_state_2nd_order_u(u, Mb)
        small, larges, counts, offsets = SCM.build_u0_2nd_mgpu(Mb, Nj, kind, 2)
        u2 = zeros(ComplexF64, SCM.state_length_2nd_order(Mb))
        SCM.shards_to_dense!(u2, small, larges, counts, offsets, Mb)
        st2 = SCM.unpack_state_2nd_order_u(u2, Mb)
        for idx in (4, 5, 9, 10, 11, 12, 13, 14, 15, 16)
            @test maximum(abs, st[idx] .- st2[idx]) < 1e-12
        end
        Sm, ZZ = st[15], st[16]
        @test maximum(abs, Sm .- conj.(transpose(Sm))) < 1e-14
        @test maximum(abs, ZZ .- transpose(ZZ)) < 1e-14
        u1 = SCM.build_u0_1st_order(Mb, Nj, Float64, kind)
        a1, Sp1, Sz1 = SCM.unpack_state_1st_order_u(u1, Mb)
        @test a1 == st[1] && Sp1 ≈ st[4] && Sz1 ≈ st[5]
    end
end

# ---------------------------------------------------------------------------
# Fixed points (E = 0)
# ---------------------------------------------------------------------------
@testset "fixed points E=0" begin
    Nj = [3.0, 5.0, 2.0, 8.0]
    Mb = 4
    delta_b = [0.0, 1e5, -2e5, 3e4]
    g_b = [2π * 80, 2π * 100, 2π * 90, 2π * 110]
    ke, ki = 2π * 1e6, 2π * 2e5
    p1 = (0.0, ke, ki, delta_b, g_b, Mb, t -> 0.0 + 0.0im)
    p2 = (0.0, ke, ki, delta_b, g_b, Mb, SCM.make_diag_mask_host(Mb), t -> 0.0 + 0.0im)
    for kind in (:ground, :inverted, :custom)
        u = SCM.build_u0_1st_order(Mb, Nj, Float64, kind)
        du = zeros(ComplexF64, length(u))
        SCM.rhs1!(du, u, p1, 0.0)
        @test maximum(abs, du) < 1e-12
    end
    u2 = SCM.build_u0_2nd_order(Mb, Nj, Float64, :ground)
    du2 = zeros(ComplexF64, length(u2))
    SCM.rhs2!(du2, u2, p2, 0.0)
    @test maximum(abs, du2) < 1e-12
    # inverted is a 1st-order fixed point but NOT 2nd-order: leftover +2i g Sz
    # because SmSp_same = N/2−Sz = 0, not N.
    uinv = SCM.build_u0_2nd_order(Mb, Nj, Float64, :inverted)
    duinv = zeros(ComplexF64, length(uinv))
    SCM.rhs2!(duinv, uinv, p2, 0.0)
    dblk = SCM.unpack_state_2nd_order_du(duinv, Mb)
    @test maximum(abs, dblk[4]) > 1.0          # dadSm leftover is physical
    @test maximum(abs, dblk[1]) < 1e-12        # dSp
    @test maximum(abs, dblk[2]) < 1e-12        # dSz
    @test abs(duinv[1]) < 1e-12
end

# ---------------------------------------------------------------------------
# 1st-order ↔ 2nd-order product-state lift (means)
# ---------------------------------------------------------------------------
@testset "RHS product-lift 1st↔2nd mean-field" begin
    Nj = [4.0, 6.0, 3.0, 10.0]
    Mb = 4
    delta_b = [0.0, 1e5, -2e5, 4e4]
    g_b = [2π * 80, 2π * 100, 2π * 90, 2π * 110]
    ke, ki, d0 = 2π * 1e6, 2π * 2e5, 1.2e4
    a = 0.03 + 0.01im
    E = t -> 0.2 + 0.05im
    p1 = (d0, ke, ki, delta_b, g_b, Mb, E)
    p2 = (d0, ke, ki, delta_b, g_b, Mb, SCM.make_diag_mask_host(Mb), E)
    for kind in IC_KINDS
        Sp0, Sz0 = SCM._spin_means(Nj, kind)
        u1 = SCM.build_u0_1st_order(Mb, Nj, Float64, kind)
        u1[1] = a
        du1 = zeros(ComplexF64, length(u1))
        SCM.rhs1!(du1, u1, p1, 0.0)
        u2 = _product_lift_2nd(a, Sp0, Sz0, Nj)
        du2 = zeros(ComplexF64, length(u2))
        SCM.rhs2!(du2, u2, p2, 0.0)
        da1, dSp1, dSz1 = SCM.unpack_state_1st_order_u(du1, Mb)
        dSp2, dSz2 = SCM.unpack_state_2nd_order_du(du2, Mb)[1:2]
        @test da1 ≈ du2[1] atol=1e-12
        @test dSp1 ≈ dSp2 atol=1e-12
        @test dSz1 ≈ dSz2 atol=1e-12
    end
end

# ---------------------------------------------------------------------------
# κₜ = κₑ + κᵢ
# ---------------------------------------------------------------------------
@testset "κₜ = κₑ+κᵢ in prepare / rhs1 / rhs2 / vjp" begin
    for (fk, gk) in ((:lorentzian, :constant), (:gaussian, :gaussian), (:lorentzian, :powerlaw_g))
        d = SCM.prepare_derived(_cfg(; fk=fk, gk=gk, Md=4, Mg=gk === :constant ? 1 : 2); ensemble_method=:auto)
        @test d.kappa_t == d.kappa_e + d.kappa_i
    end
    ke, ki = 2π * 1e6, 2π * 3e5
    a = 0.02 + 0.01im
    u = SCM.build_u0_2nd_order(1, [5.0], Float64, :ground)
    u[1] = a
    du = zeros(ComplexF64, length(u))
    SCM.rhs2!(du, u, (0.0, ke, ki, [0.0], [2π * 100], 1, SCM.make_diag_mask_host(1), t -> 0im), 0.0)
    @test du[1] ≈ -0.5 * (ke + ki) * a
    du0 = zeros(ComplexF64, length(u))
    SCM.rhs2!(du0, u, (0.0, ke, 0.0, [0.0], [2π * 100], 1, SCM.make_diag_mask_host(1), t -> 0im), 0.0)
    @test du[1] != du0[1]
    Mb = 3
    x = zeros(Float64, SCM.real_state_length_1st_order(Mb))
    u1 = SCM.build_u0_1st_order(Mb, [2.0, 4.0, 6.0], Float64, :ground)
    u1[1] = 0.01
    SCM.pack_state_real!(x, u1, Mb)
    λ = zeros(length(x)); λ[1] = 1.0
    x̄ = zeros(length(x))
    SCM.rhs1_vjp!(x̄, λ, x, (0.0, ke, ki, zeros(Mb), fill(2π * 100, Mb), Mb, t -> 0im), 0.0)
    @test x̄[1] ≈ -0.5 * (ke + ki)
end

# ---------------------------------------------------------------------------
# order-2 :auto → quadrature (hist mismatch hunt)
# ---------------------------------------------------------------------------
@testset "order-2 :auto → quadrature; hist nodes differ" begin
    for (fk, gk) in ((:lorentzian, :constant), (:gaussian, :constant),
                     (:lorentzian, :gaussian), (:lorentzian, :powerlaw_g),
                     (:gaussian, :gaussian))
        Mg = gk === :constant ? 1 : 3
        c = _cfg(; fk=fk, gk=gk, Md=6, Mg=Mg)
        plan = SCM.resolve_ensemble_method(c, :auto)
        d = SCM.prepare_derived(c; ensemble_method=:auto)
        @test plan.method === :quadrature
        @test d.ensemble_method === :quadrature
    end
    c = _cfg()
    dq = SCM.prepare_derived(c; ensemble_method=:auto)
    dh = SCM.prepare_derived(c; ensemble_method=:histogram)
    @test dq.ensemble_method === :quadrature
    @test dh.ensemble_method === :histogram
    @test !(dq.delta_b ≈ dh.delta_b)
    @test dq.N ≈ dh.N
    settings = SCM.load_settings(joinpath(@__DIR__, "..", "examples", "monolith_order2.jl"))
    prep = SCM.prepare(settings; ensemble_method=:auto)
    @test prep.d.ensemble_method === :quadrature
end

# ---------------------------------------------------------------------------
# sharded RHS + short integrate parity
# ---------------------------------------------------------------------------
@testset "CPU nshards=1/2/3 RHS and integrate parity" begin
    Mb = 4
    Nj = [3.0, 5.0, 2.0, 8.0]
    delta_b = [0.0, 1e5, -2e5, 3e4]
    g_b = [2π * 80, 2π * 100, 2π * 90, 2π * 110]
    ke, ki = 2π * 1e6, 2π * 2e5
    u = SCM.build_u0_2nd_order(Mb, Nj, Float64, :equator)
    u[1] = 0.01 + 0.002im
    refs = Dict{Int,Vector{ComplexF64}}()
    for ns in (1, 2, 3)
        s, L, c, o = SCM.dense_to_shards(u, Mb, ns)
        ds = zero(s); dL = [zero(x) for x in L]
        SCM.rhs2_sharded!(ds, dL, s, L, c, o, 0.0, ke, ki, delta_b, g_b, Mb, 0.1 + 0.05im)
        du = zeros(ComplexF64, SCM.state_length_2nd_order(Mb))
        SCM.shards_to_dense!(du, ds, dL, c, o, Mb)
        refs[ns] = du
    end
    @test refs[1] ≈ refs[2] rtol=1e-12 atol=1e-14
    @test refs[1] ≈ refs[3] rtol=1e-12 atol=1e-14

    function o2_end(ns)
        s, L, c, o = SCM.build_u0_2nd_mgpu(Mb, Nj, :equator, ns)
        s, L, _ = SCM.integrate_order2_sharded!(
            s, L, c, o, 0.0, ke, ki, delta_b, g_b, Mb, t -> 0.05 + 0.02im, (0.0, 4e-7);
            reltol=1e-7, abstol=1e-7)
        out = zeros(ComplexF64, SCM.state_length_2nd_order(Mb))
        SCM.shards_to_dense!(out, s, L, c, o, Mb)
        return out
    end
    u1, u2, u3 = o2_end(1), o2_end(2), o2_end(3)
    @test u1 ≈ u2 rtol=1e-11 atol=1e-12
    @test u1 ≈ u3 rtol=1e-11 atol=1e-12
end

# ---------------------------------------------------------------------------
# rhs1 VJP vs finite-difference
# ---------------------------------------------------------------------------
@testset "rhs1_vjp! vs finite-difference Jᵀλ" begin
    Mb = 3
    Nj = [2.0, 4.0, 6.0]
    delta_b = [0.0, 2π * 5e4, -2π * 5e4]
    g_b = [2π * 100, 2π * 90, 2π * 110]
    p = (2π * 0.0 + 1e4, 2π * 1e6, 2π * 1e5, delta_b, g_b, Mb, t -> 0.3 + 0.1im)
    u = SCM.build_u0_1st_order(Mb, Nj, Float64, :weak)
    u[1] = 0.02 + 0.01im
    x = zeros(Float64, SCM.real_state_length_1st_order(Mb))
    SCM.pack_state_real!(x, u, Mb)
    λ = randn(MersenneTwister(1), length(x))
    function f_real(x)
        uu = zeros(ComplexF64, SCM.state_length_1st_order(Mb))
        SCM.real_to_complex!(uu, x, Mb)
        du = zeros(ComplexF64, length(uu))
        SCM.rhs1!(du, uu, p, 1.7e-6)
        xr = zeros(Float64, length(x))
        SCM.pack_state_real!(xr, du, Mb)
        return xr
    end
    ε = 1e-6
    fx = f_real(x)
    J = zeros(length(x), length(x))
    for i in eachindex(x)
        xp = copy(x); xp[i] += ε
        J[:, i] = (f_real(xp) .- fx) ./ ε
    end
    x̄ = zeros(length(x))
    SCM.rhs1_vjp!(x̄, λ, x, p, 1.7e-6)
    @test x̄ ≈ J' * λ rtol=1e-8 atol=1e-5
end

# ---------------------------------------------------------------------------
# adjoint vs forward at M=4 (not the M=1 / reltol=1e2 cheat)
# ---------------------------------------------------------------------------
@testset "adjoint vs forward-mode parity at M=4" begin
    κe = 2π * 1e6
    d = (
        timespan=(0.0, 8e-8), FWHM=2π * 1e6, kappa_t=κe + 2π * 1e5,
        kappa_e=κe, kappa_i=2π * 1e5, g_mean=2π * 100, sqrt_kappa_e=sqrt(κe),
        delta0=0.0, M=4, Nj=fill(8.0, 4),
        delta_b=collect(2π * 1e5 .* range(-1, 1; length=4)),
        g_b=fill(2π * 100, 4),
    )
    pulse = SCM.CompositePulse(1, 4, 4, d; degree=3, taper_frac=0.1)
    u = SCM.initial_guess(pulse; seed=3)
    kw = (reltol=1e-4, abstol=1e-4, dt0=2e-8, track=:weak)
    g_adj, c_adj = SCM.pulse_cost_grad_adjoint(u, pulse, d; kw..., checkpoint_stride=typemax(Int))
    g_fwd, c_fwd = SCM.pulse_cost_grad_forward(u, pulse, d; kw..., backend=:cpu)
    @test c_fwd ≈ c_adj rtol=1e-10
    @test g_fwd ≈ g_adj rtol=1e-10 atol=1e-10
end

# ---------------------------------------------------------------------------
# CPU efficiency — measured against 05bed84 claims
# ---------------------------------------------------------------------------
@testset "CPU efficiency B1–B6 on $TIP_SHA" begin
    src = read(joinpath(@__DIR__, "..", "src", "SpinCavityMonolith.jl"), String)
    nt = Threads.nthreads()
    println("[stress] tip=", TIP_SHA, "  nthreads=", nt)

    # B4: Threads present; nshards default = 1
    @test occursin("@threads", src)
    @test occursin("struct RHS2Work", src)
    @test SCM.resolve_cpu_nshards(16) == 1
    @test SCM.resolve_cpu_nshards(16; nshards=4) == 4
    @test SCM.resolve_cpu_nshards(3; nshards=8) == 3
    println("[stress] B4 @threads present; nshards default=", SCM.resolve_cpu_nshards(48))

    # B5: large-block layout still 5 × (M·mloc) with stride bs
    @test occursin("bs = M * mloc", src)
    @test occursin("dlarge[i+4bs]", src)
    println("[stress] B5 layout still 5-block stride bs=M*mloc (OPEN)")

    # B1: explicit args (no splat) after warmup — matches product test
    function _b1_bytes(Mb)
        Nj = fill(4.0, Mb)
        delta_b = [1e5 * sin(0.7 * j) for j in 1:Mb]
        g_b = fill(2π * 100.0, Mb)
        u = SCM.build_u0_2nd_order(Mb, Nj, Float64, :equator)
        u[1] = 0.01 + 0.002im
        s, L, c, o = SCM.dense_to_shards(u, Mb, 1)
        ds = zero(s); dL = [zero(x) for x in L]
        Et = 0.1 + 0.05im
        ke, ki = 2π * 1e6, 2π * 2e5
        SCM.rhs2_sharded!(ds, dL, s, L, c, o, 0.0, ke, ki, delta_b, g_b, Mb, Et)
        return @allocated SCM.rhs2_sharded!(ds, dL, s, L, c, o, 0.0, ke, ki, delta_b, g_b, Mb, Et)
    end
    n_rhs = _b1_bytes(8)
    n_rhs48 = _b1_bytes(48)
    println("[stress] B1 rhs2_sharded! bytes/RHS M=8 = ", n_rhs, "  PRIOR=1232  TARGET=0")
    println("[stress] B1 rhs2_sharded! bytes/RHS M=48 = ", n_rhs48, "  TARGET=0 (1 thread)")
    u1 = SCM.build_u0_1st_order(8, fill(4.0, 8), Float64, :weak)
    du1 = zeros(ComplexF64, length(u1))
    p1 = (0.0, 2π * 1e6, 2π * 2e5, [1e5 * sin(0.7 * j) for j in 1:8], fill(2π * 100.0, 8), 8, t -> 0.1 + 0.0im)
    SCM.rhs1!(du1, u1, p1, 0.0)
    n1 = @allocated SCM.rhs1!(du1, u1, p1, 0.0)
    println("[stress] rhs1! bytes/RHS M=8 = ", n1, "  TARGET=0")
    @test n1 == 0
    if nt == 1
        @test n_rhs == 0
        @test n_rhs48 == 0
    else
        @test n_rhs < 8192
        @test n_rhs48 < 8192
    end

    # B2: replay one Tsit5 step with preallocated stage buffers
    function _b2_replay(Mb)
        Nj = fill(4.0, Mb)
        delta_b = [1e5 * sin(0.7 * j) for j in 1:Mb]
        g_b = fill(2π * 100.0, Mb)
        s, L, c, o = SCM.build_u0_2nd_mgpu(Mb, Nj, :equator, 1)
        ns = length(c)
        work = SCM.rhs2_work(eltype(s), Mb)
        tab = SCM.Tsit5Tab(Float64)
        kS = [zero(s) for _ in 1:6]
        kL = [[zero(L[p]) for p in 1:ns] for _ in 1:6]
        yS = zero(s); yL = [zero(L[p]) for p in 1:ns]
        u1S = zero(s); u1L = [zero(L[p]) for p in 1:ns]
        eS = zero(s); eL = [zero(L[p]) for p in 1:ns]
        dt = 1e-7
        E = 0.05 + 0.02im
        rhs_at!(dS, dL, ss, LL) = SCM.rhs2_sharded!(dS, dL, ss, LL, c, o, 0.0, 2π*1e6, 2π*2e5, delta_b, g_b, Mb, E, work)
        rhs_at!(kS[1], kL[1], s, L)
        SCM._axpy_shards!(yS, yL, s, L, dt * tab.a21, kS[1], kL[1])
        bytes = @allocated begin
            rhs_at!(kS[1], kL[1], s, L)
            SCM._axpy_shards!(yS, yL, s, L, dt * tab.a21, kS[1], kL[1])
            rhs_at!(kS[2], kL[2], yS, yL)
            SCM._lincomb_shards!(yS, yL, s, L, dt .* (tab.a31, tab.a32), kS[1:2], kL[1:2])
            rhs_at!(kS[3], kL[3], yS, yL)
            SCM._lincomb_shards!(yS, yL, s, L, dt .* (tab.a41, tab.a42, tab.a43), kS[1:3], kL[1:3])
            rhs_at!(kS[4], kL[4], yS, yL)
            SCM._lincomb_shards!(yS, yL, s, L, dt .* (tab.a51, tab.a52, tab.a53, tab.a54), kS[1:4], kL[1:4])
            rhs_at!(kS[5], kL[5], yS, yL)
            SCM._lincomb_shards!(yS, yL, s, L, dt .* (tab.a61, tab.a62, tab.a63, tab.a64, tab.a65), kS[1:5], kL[1:5])
            rhs_at!(kS[6], kL[6], yS, yL)
            SCM._lincomb_shards!(u1S, u1L, s, L, dt .* tab.b, kS, kL)
            SCM._lincomb_from_zero_shards!(eS, eL, dt .* tab.e, kS, kL)
        end
        return bytes
    end
    per_step = _b2_replay(8)
    println("[stress] B2 replay Tsit5 step M=8 bytes = ", per_step, "  PRIOR~19456  leftover = kS[1:n] slices")
    @test_broken per_step == 0

    # B3: B-spline E(t)
    d = (timespan=(0.0, 1e-4), FWHM=1e6, kappa_t=2π * 1e6, g_mean=2π * 100,
         sqrt_kappa_e=sqrt(2π * 1e6))
    pulse = SCM.CompositePulse(1, 4, 4, d)
    uu = SCM.initial_guess(pulse; seed=1)
    EE = SCM.build_E_of_t(pulse, uu)
    ts, te, = SCM.decode(pulse, uu)
    tmid = (ts[1] + te[1]) / 2
    EE(tmid)
    nE = @allocated EE(tmid)
    println("[stress] B3 bspline E(t) bytes/eval = ", nE, "  PRIOR=704  TARGET=0")
    @test_broken nE == 0

    # B6: order-2 still hardcodes Tsit5
    m = match(r"function integrate_order2_sharded![\s\S]+?function solve_2nd_order", src)
    body = m === nothing ? "" : m.match
    println("[stress] B6 order2 integrator body mentions ck45? ", occursin("ck45", lowercase(body)))
    @test_broken occursin("ck45", lowercase(body))
end

# ---------------------------------------------------------------------------
# B7 / B8 API footguns (still product bugs — report only)
# ---------------------------------------------------------------------------
@testset "B7/B8 ensemble API footguns (still open)" begin
    c_gauss_g = _cfg(; fk=:lorentzian, gk=:gaussian, Md=5, Mg=3)
    err7 = try
        SCM.prepare_derived(c_gauss_g; ensemble_method=:histogram)
        ""
    catch e
        sprint(showerror, e)
    end
    println("[stress] B7 forced hist+nonconst g error: ", err7)
    @test occursin("constant g or quad-friendly", err7)

    c_unk = (
        C_ens=0.6, M_delta=3, M_g=1, Ttotal=1e-5, Nt_save=3,
        delta0=0.0, kappa_e=2π * 1e6, kappa_i=0.0,
        freq_inhomogeneity=(kind=:uniform, FWHM=2π * 1e6, span_gamma=2.5, renormalize=false),
        g_inhomogeneity=(kind=:constant, g_value=2π * 100),
    )
    plan = SCM.resolve_ensemble_method(c_unk, :auto)
    @test plan.method === :histogram
    err8 = try
        SCM.prepare_derived(c_unk; ensemble_method=:auto)
        ""
    catch e
        sprint(showerror, e)
    end
    println("[stress] B8 :auto+unknown Δ kind plan=", plan.method, " error=", err8)
    @test !isempty(err8)
end

# ---------------------------------------------------------------------------
# B9 / B10 packaging (static)
# ---------------------------------------------------------------------------
@testset "B9/B10 packaging / docs" begin
    proj = read(joinpath(@__DIR__, "..", "Project.toml"), String)
    readme = read(joinpath(@__DIR__, "..", "README.md"), String)
    mono = read(joinpath(@__DIR__, "..", "MONOLITH.md"), String)
    @test occursin("ForwardDiff", proj)
    @test !occursin("LinearAlgebra", proj)
    println("[stress] B9 Project.toml still lists only ForwardDiff (OPEN)")
    doc_bare = occursin("julia --startup-file=no test/spin_cavity_monolith.jl", readme) ||
               occursin("julia --startup-file=no test/spin_cavity_monolith.jl", mono)
    println("[stress] B10 documented bare test invoke (needs --project=.)? ", doc_bare)
    @test doc_bare
end

end # parent testset

println("SpinCavityMonolith stress harness finished (tip ", TIP_SHA, ").")
