# Re-bench tip dc70114e8a9be6181d6a54f47b3bd88a078c5f09 (PR #4 / monolith).
# Harness only — no src/ product changes.
#   julia --project=. --startup-file=no -t 1 test/rebench_dc70114.jl
#
# Product path, 1 thread, after warmup.
# B3 is measured through a typed `_ass_E_bytes` (PulseDrive{Float64}, Float64).
# A bare `@testset` / Any-boxed `@allocated EE(t)` can count 32 B for the
# ComplexF64 return and is reported separately as a harness pitfall.

using Test
using LinearAlgebra
using Random
using ForwardDiff

if !isdefined(Main, :SpinCavityMonolith)
    include(joinpath(@__DIR__, "..", "src", "SpinCavityMonolith.jl"))
    using .SpinCavityMonolith
end
const SCM = Main.SpinCavityMonolith

const TIP_SHA = "dc70114e8a9be6181d6a54f47b3bd88a078c5f09"

_bench_Et(t) = ComplexF64(0.1, 0.05)

function _cfg(; fk=:lorentzian, gk=:constant, Md=5, Mg=1, ke=2π * 1e6, ki=2π * 1e5, C=0.6)
    freq = (kind=fk, FWHM=2π * 1e6, span_gamma=2.5, span_sigma=3.0, renormalize=false)
    g = if gk === :constant
        (kind=:constant, g_value=2π * 100)
    elseif gk === :gaussian
        (kind=:gaussian, mean=2π * 100, std=2π * 10, span_sigma=3.0, renormalize=true)
    else
        (kind=gk,)
    end
    return (C_ens=C, M_delta=Md, M_g=Mg, Ttotal=1e-5, Nt_save=5,
            delta0=0.0, kappa_e=ke, kappa_i=ki,
            freq_inhomogeneity=freq, g_inhomogeneity=g)
end

function _tiny_pulse()
    κe = 2π * 1e6
    d = (
        timespan = (0.0, 4e-8),
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
    pulse = SCM.CompositePulse(1, 4, 4, d; degree=3, taper_frac=0.1)
    u = SCM.initial_guess(pulse; seed=3)
    return pulse, d, u
end

# Ass-style typed helper. Must not live as an untyped `@testset` local.
function _ass_E_bytes(EE::SCM.PulseDrive{Float64}, t::Float64)
    EE(t)
    return @allocated EE(t)
end

function _boxed_E_bytes(EE, t)
    EE(t)
    return @allocated EE(t)
end

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
    SCM.rhs2_sharded!(ds, dL, s, L, c, o, 0.0, ke, ki, delta_b, g_b, Mb, Et)
    return @allocated SCM.rhs2_sharded!(ds, dL, s, L, c, o, 0.0, ke, ki, delta_b, g_b, Mb, Et)
end

@testset "rebench tip $TIP_SHA" begin
    nt = Threads.nthreads()
    println("[rebench] tip=", TIP_SHA, "  nthreads=", nt)
    src = read(joinpath(@__DIR__, "..", "src", "SpinCavityMonolith.jl"), String)

    @testset "B3 typed warm EE(tmid) 0 B/eval" begin
        pulse, _, uu = _tiny_pulse()
        EE = SCM.build_E_of_t(pulse, uu)
        @test EE isa SCM.PulseDrive{Float64}
        @test EE.scratch isa SCM.BSplineScratch{Float64}
        ts, te, = SCM.decode(pulse, uu)
        tmid = (ts[1] + te[1]) / 2
        E0 = EE(tmid)
        nE = _ass_E_bytes(EE, tmid)
        nE2 = _ass_E_bytes(EE, tmid)
        boxed = _boxed_E_bytes(EE, tmid)
        println("[rebench] B3 typed `_ass_E_bytes` bytes/eval = ", nE, "  (repeat ", nE2, ")")
        println("[rebench] B3 Any-boxed `_boxed_E_bytes` bytes/eval = ", boxed,
                "  (harness pitfall; ignore for PASS/FAIL)")
        println("[rebench] B3 |E(tmid)|=", abs(E0), "  type=", typeof(EE))
        @test abs(E0) > 0
        @test nE == 0
        @test nE2 == 0
        @test occursin("struct PulseDrive", src)
        @test occursin("struct BSplineScratch", src)
        @test occursin("function bspline_dot!", src)

        # Dual-through-u owns its own scratch; primal cache stays 0 after Dual.
        g_ad = ForwardDiff.gradient(θ -> real(SCM.build_E_of_t(pulse, θ)(tmid)), uu)
        EE_d = SCM.build_E_of_t(pulse, ForwardDiff.Dual{Nothing}.(uu, one.(uu)))
        @test EE_d isa SCM.PulseDrive
        @test EE_d.scratch isa SCM.BSplineScratch
        @test eltype(EE_d.scratch.a) !== Float64
        nE_after = _ass_E_bytes(EE, tmid)
        println("[rebench] B3 Dual-through-u scratch eltype=", eltype(EE_d.scratch.a),
                "  primal bytes/eval after Dual=", nE_after)
        ε = 1e-6
        g_fd = similar(uu)
        @inbounds for i in eachindex(uu)
            up = copy(uu); um = copy(uu)
            up[i] += ε; um[i] -= ε
            g_fd[i] = (real(SCM.build_E_of_t(pulse, up)(tmid)) -
                       real(SCM.build_E_of_t(pulse, um)(tmid))) / (2ε)
        end
        rel = maximum(abs, g_ad .- g_fd) / max(maximum(abs, g_fd), 1e-30)
        println("[rebench] B3 Dual-through-u vs FD max-rel=", rel)
        @test g_ad ≈ g_fd rtol=1e-5 atol=1e-7
        @test nE_after == 0
    end

    @testset "B1 0 B/RHS @1 thread (product rhs2_sharded!)" begin
        n8 = _b1_bytes(8)
        n48 = _b1_bytes(48)
        println("[rebench] B1 rhs2_sharded! bytes/RHS M=8 = ", n8)
        println("[rebench] B1 rhs2_sharded! bytes/RHS M=48 = ", n48)
        if nt == 1
            @test n8 == 0
            @test n48 == 0
        else
            @test n8 < 8192
            @test n48 < 8192
        end
    end

    @testset "B2 0 B/step product path (_rk6_order2! / Order2Pool)" begin
        Mbin = 8
        Nj = fill(4.0, Mbin)
        delta_b = [2π * 1e4 * (j - 4) for j in 1:Mbin]
        g_b = fill(2π * 100.0, Mbin)
        kappa_e = 2π * 1e6
        kappa_i = 2π * 1e5
        u2 = SCM.build_u0_2nd_order(Mbin, Nj, Float64, :equator)
        u2[1] = 0.01 + 0.002im
        s, L, c, o = SCM.dense_to_shards(u2, Mbin, 1)
        pool = SCM.Order2Pool(s, L)
        tab5 = SCM.Tsit5Tab(Float64)
        tabck = SCM.CK45Tab(Float64)
        SCM._rk6_order2!(pool, s, L, c, o, 0.0, kappa_e, kappa_i, delta_b, g_b, Mbin,
                         _bench_Et, 0.0, 1e-8, tab5)
        a2 = @allocated SCM._rk6_order2!(pool, s, L, c, o, 0.0, kappa_e, kappa_i, delta_b, g_b,
                                        Mbin, _bench_Et, 0.0, 1e-8, tab5)
        SCM._rk6_order2!(pool, s, L, c, o, 0.0, kappa_e, kappa_i, delta_b, g_b, Mbin,
                         _bench_Et, 0.0, 1e-8, tabck)
        a2c = @allocated SCM._rk6_order2!(pool, s, L, c, o, 0.0, kappa_e, kappa_i, delta_b, g_b,
                                         Mbin, _bench_Et, 0.0, 1e-8, tabck)
        println("[rebench] B2 product `_rk6_order2!` Tsit5 bytes/step = ", a2)
        println("[rebench] B2 product `_rk6_order2!` CK45  bytes/step = ", a2c)
        @test a2 == 0
        @test a2c == 0

        function _ass_integrate(tspan, pool)
            s0, L0, c0, o0 = SCM.build_u0_2nd_mgpu(Mbin, Nj, :equator, 1)
            _, _, nst = SCM.integrate_order2_sharded!(
                s0, L0, c0, o0, 0.0, kappa_e, kappa_i, delta_b, g_b, Mbin,
                _bench_Et, tspan; reltol=1e-6, abstol=1e-6, dt0=1e-7, pool=pool)
            return nst
        end
        _ass_integrate((0.0, 2e-7), pool)
        nst_s = _ass_integrate((0.0, 2e-7), pool)
        b_s = @allocated _ass_integrate((0.0, 2e-7), pool)
        nst_l = _ass_integrate((0.0, 8e-7), pool)
        b_l = @allocated _ass_integrate((0.0, 8e-7), pool)
        per = (b_l - b_s) / max(nst_l - nst_s, 1)
        println("[rebench] B2 integrate reused-pool incremental bytes/step = ", per,
                "  (short nsteps=", nst_s, " bytes=", b_s,
                "  long nsteps=", nst_l, " bytes=", b_l, ")")
        @test nst_l > nst_s
        @test per == 0
    end

    @testset "B5 pair-interleaved large layout" begin
        Mbin = 8
        @test SCM.mg_large_length(Mbin, Mbin) == 5 * Mbin * Mbin
        @test SCM.mg_pair(Mbin, 1, 1, 1) == 1
        @test SCM.mg_pair(Mbin, 1, 1, 5) == 5
        @test SCM.mg_pair(Mbin, 1, 2, 1) == 6
        @test SCM.mg_pair(Mbin, 2, 1, 1) == 5 * Mbin + 1
        # Adjacent 5-tuple: f=1..5 for one (jl,j) pair are contiguous.
        for jl in 1:3, j in 1:4, f in 1:4
            @test SCM.mg_pair(Mbin, jl, j, f + 1) == SCM.mg_pair(Mbin, jl, j, f) + 1
        end
        u = SCM.build_u0_2nd_order(Mbin, fill(4.0, Mbin), Float64, :equator)
        s, L, c, o = SCM.dense_to_shards(u, Mbin, 1)
        @test length(L) == 1
        @test length(L[1]) == 5 * Mbin * Mbin
        @test SCM.resolve_cpu_nshards(48) == 1
        @test occursin("pair-interleaved", src)
        @test occursin("mg_pair(M, jl, j, f) = 5 * ((jl - 1) * M + (j - 1)) + f", src)
        @test !occursin("bs = M * mloc", src)
        println("[rebench] B5 mg_pair(M,jl,j,f)=5*((jl-1)*M+(j-1))+f  large_len=",
                SCM.mg_large_length(Mbin, Mbin), "  shards=", length(L),
                "  nshards_default=", SCM.resolve_cpu_nshards(48))
    end

    @testset "B7/B8 ensemble API as claimed" begin
        err7 = try
            SCM.prepare_derived(_cfg(; fk=:lorentzian, gk=:gaussian, Md=4, Mg=3);
                                ensemble_method=:histogram)
            ""
        catch e
            sprint(showerror, e)
        end
        println("[rebench] B7 forced hist+nonconst g error: ", err7)
        @test occursin("only :constant g", err7)
        @test occursin(":gaussian", err7)
        @test !occursin("or quad-friendly kinds", err7)

        c_uni = (
            C_ens=0.6, M_delta=4, M_g=1, Ttotal=1e-5, Nt_save=3,
            delta0=0.0, kappa_e=2π * 1e6, kappa_i=0.0,
            freq_inhomogeneity=(kind=:uniform, FWHM=2π * 1e6, span_gamma=1.0, renormalize=false),
            g_inhomogeneity=(kind=:constant, g_value=2π * 100),
        )
        plan = SCM.resolve_ensemble_method(c_uni, :auto)
        duni = SCM.prepare_derived(c_uni; ensemble_method=:auto)
        println("[rebench] B8 :auto+:uniform plan=", plan.method,
                "  prepare.method=", duni.ensemble_method, "  M=", duni.M)
        @test plan.method === :histogram
        @test duni.ensemble_method === :histogram
        @test duni.M == 4

        err8 = try
            SCM.prepare_derived((
                C_ens=0.6, M_delta=3, M_g=1, Ttotal=1e-5, Nt_save=3,
                delta0=0.0, kappa_e=2π * 1e6, kappa_i=0.0,
                freq_inhomogeneity=(kind=:notakind, FWHM=2π * 1e6, span_gamma=2.5, renormalize=false),
                g_inhomogeneity=(kind=:constant, g_value=2π * 100),
            ); ensemble_method=:auto)
            ""
        catch e
            sprint(showerror, e)
        end
        println("[rebench] B8 :auto+unknown Δ error: ", err8)
        @test occursin("no quadrature rule", err8)
        @test occursin("notakind", err8)
    end

    @testset "B10 docs invoke --project=. --startup-file=no" begin
        readme = read(joinpath(@__DIR__, "..", "README.md"), String)
        mono = read(joinpath(@__DIR__, "..", "MONOLITH.md"), String)
        req = read(joinpath(@__DIR__, "..", "REQUIREMENTS.md"), String)
        testh = read(joinpath(@__DIR__, "spin_cavity_monolith.jl"), String)
        inv = "julia --project=. --startup-file=no test/spin_cavity_monolith.jl"
        @test occursin(inv, readme)
        @test occursin(inv, mono)
        @test occursin(inv, testh)
        @test occursin("--project=. --startup-file=no", req)
        bare = occursin("julia --startup-file=no test/spin_cavity_monolith.jl", readme) &&
               !occursin(inv, readme)
        println("[rebench] B10 README/MONOLITH/REQUIREMENTS use --project=. --startup-file=no")
        @test !bare
    end
end

println("Re-bench finished (tip ", TIP_SHA, ").")
