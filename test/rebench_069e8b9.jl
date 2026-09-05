# Re-bench tip 069e8b9ac127c42885cb044c151b0e3a7caba610 (PR #4 / monolith).
# Harness only — no src/ product changes.
#   julia --project=. --startup-file=no -t 1 test/rebench_069e8b9.jl
#
# Product path: `integrate_order2_sharded!` / `_rk6_order2!` / `Order2Pool`.
# Does not replay a sliced `kS[1:n]` call site (that was PR #8's 640 B leftover).

using Test
using LinearAlgebra
using Random
using ForwardDiff

if !isdefined(Main, :SpinCavityMonolith)
    include(joinpath(@__DIR__, "..", "src", "SpinCavityMonolith.jl"))
    using .SpinCavityMonolith
end
const SCM = Main.SpinCavityMonolith

const TIP_SHA = "069e8b9ac127c42885cb044c151b0e3a7caba610"

_bench_Et(t) = ComplexF64(0.1, 0.05)

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

@testset "rebench tip $TIP_SHA" begin
    nt = Threads.nthreads()
    println("[rebench] tip=", TIP_SHA, "  nthreads=", nt)
    src = read(joinpath(@__DIR__, "..", "src", "SpinCavityMonolith.jl"), String)

    @testset "B1 0 B/RHS @1 thread (product rhs2_sharded!)" begin
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
        # Product killed the leftover; the phrase remains only in the comment.
        @test occursin("no kS[1:n] slices", src)
        @test occursin("_lincomb_n!", src)
        @test !occursin("_lincomb_shards!", src)

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

        # False-positive leftover: Vector `kS[1:n]` at a harness call site.
        kS = [zero(s) for _ in 1:6]
        kL = [[zero(L[1])] for _ in 1:6]
        _ = kS[1:2]; _ = kL[1:2]
        slice_b = @allocated begin
            _ = kS[1:2]; _ = kL[1:2]
            _ = kS[1:3]; _ = kL[1:3]
            _ = kS[1:4]; _ = kL[1:4]
            _ = kS[1:5]; _ = kL[1:5]
        end
        println("[rebench] harness-only kS[1:n] Vector slice leftover bytes = ", slice_b,
                "  (NOT product; PR #8 pitfall)")
        @test slice_b > 0
    end

    @testset "B6 :ck45 is Cash–Karp; GPU/adjoint error; no silent Tsit5" begin
        @test SCM._canon_integrator(:ck45) === :ck45
        @test SCM._canon_integrator(:cash_karp) === :ck45
        @test_throws ErrorException SCM._canon_integrator(:dp5)
        @test_throws ErrorException SCM._canon_integrator(:auto)
        @test SCM._integrator_from_compute((method=:ck45,)) === :ck45
        @test SCM._integrator_from_compute((integrator=:tsit5, method=:ck45)) === :tsit5
        tab = SCM.CK45Tab(Float64)
        tsit = SCM.Tsit5Tab(Float64)
        @test tab.c[1] ≈ 1 / 5
        @test tab.c[5] ≈ 7 / 8
        @test tab.b[1] ≈ 37 / 378
        @test tab.b[2] == 0
        @test tab.b[6] ≈ 512 / 1771
        @test tab.c[1] != tsit.c[1]
        d1 = (
            timespan = (0.0, 2e-8),
            M = 2,
            Nj = [3.0, 5.0],
            delta_b = [0.0, 1e5],
            g_b = [2π * 100, 2π * 90],
            delta0 = 0.0,
            kappa_e = 2π * 1e6,
            kappa_i = 0.0,
        )
        a5, _, _, i5 = SCM.solve_1st_order(d1, _bench_Et, :equator; integrator=:tsit5, backend=:cpu)
        ack, _, _, ick = SCM.solve_1st_order(d1, _bench_Et, :equator; integrator=:ck45, backend=:cpu)
        @test i5.integrator === :tsit5
        @test ick.integrator === :ck45
        @test i5.nsteps != ick.nsteps
        @test a5 ≈ ack rtol=1e-8 atol=1e-12
        _, info5 = SCM.solve_2nd_order(d1, _bench_Et, :equator; integrator=:tsit5, backend=:cpu)
        _, infock = SCM.solve_2nd_order(d1, _bench_Et, :equator; integrator=:ck45, backend=:cpu)
        @test info5.integrator === :tsit5
        @test infock.integrator === :ck45
        @test info5.nsteps != infock.nsteps
        println("[rebench] B6 1st nsteps tsit5=", i5.nsteps, " ck45=", ick.nsteps)
        println("[rebench] B6 2nd nsteps tsit5=", info5.nsteps, " ck45=", infock.nsteps)

        pulse, d, u = _tiny_pulse()
        err_adj = try
            SCM.pulse_cost_grad_adjoint(u, pulse, d; integrator=:ck45, reltol=1e2, abstol=1e2, dt0=4e-8)
            ""
        catch e
            sprint(showerror, e)
        end
        println("[rebench] B6 adjoint :ck45 error: ", err_adj)
        @test occursin("Tsit5-only", err_adj)
        @test !occursin("silent", lowercase(err_adj))

        println("[rebench] B6 cuda_functional=", SCM.cuda_functional(),
                " gpu_count=", SCM.gpu_count())
        @test occursin("GPU 1st-order is Tsit5-only", src)
        @test occursin("GPU order-2 is Tsit5-only", src)
        if SCM.cuda_functional()
            @test_throws ErrorException SCM.solve_1st_order(d1, _bench_Et, :equator; integrator=:ck45, backend=:gpu)
            @test_throws ErrorException SCM.solve_2nd_order(d1, _bench_Et, :equator; integrator=:ck45, backend=:gpu)
        else
            # No CUDA: backend=:gpu does not enter the GPU branch; must still honor :ck45 on CPU.
            _, i_gpu_kw = SCM.solve_2nd_order(d1, _bench_Et, :equator; integrator=:ck45, backend=:gpu)
            @test i_gpu_kw.integrator === :ck45
            println("[rebench] B6 no-CUDA backend=:gpu+:ck45 fell through to CPU Cash–Karp (not silent Tsit5)")
        end
    end

    @testset "B9 Project.toml stdlibs + Pkg.precompile + method/integrator" begin
        proj = read(joinpath(@__DIR__, "..", "Project.toml"), String)
        @test occursin("LinearAlgebra", proj)
        @test occursin("Random", proj)
        @test occursin("Printf", proj)
        @test occursin("ForwardDiff", proj)
        println("[rebench] B9 Project.toml lists LinearAlgebra/Random/Printf (Pkg.precompile run separately)")
        settings = SCM.load_settings(joinpath(@__DIR__, "..", "examples", "monolith_order2.jl"))
        prep_ck = SCM.prepare(merge(settings, (compute=(backend=:cpu, method=:ck45),)))
        @test prep_ck.integrator === :ck45
    end

    @testset "B3/B5/B7/B8/B10 still open or not (report only)" begin
        d = (timespan=(0.0, 1e-4), FWHM=1e6, kappa_t=2π * 1e6, g_mean=2π * 100,
             sqrt_kappa_e=sqrt(2π * 1e6))
        pulse = SCM.CompositePulse(1, 4, 4, d)
        uu = SCM.initial_guess(pulse; seed=1)
        EE = SCM.build_E_of_t(pulse, uu)
        ts, te, = SCM.decode(pulse, uu)
        tmid = (ts[1] + te[1]) / 2
        EE(tmid)
        nE = @allocated EE(tmid)
        println("[rebench] B3 bspline E(t) bytes/eval = ", nE, "  (OPEN if >0)")

        @test SCM.mg_large_length(8, 8) == 5 * 8 * 8
        @test occursin("5 × M × mloc", src)
        println("[rebench] B5 large layout still 5×M×mloc (claimed packing; prior OPEN)")

        c_gauss_g = _cfg(; fk=:lorentzian, gk=:gaussian, Md=5, Mg=3)
        err7 = try
            SCM.prepare_derived(c_gauss_g; ensemble_method=:histogram)
            ""
        catch e
            sprint(showerror, e)
        end
        println("[rebench] B7 forced hist+nonconst g error: ", err7)

        c_unk = (
            C_ens=0.6, M_delta=3, M_g=1, Ttotal=1e-5, Nt_save=3,
            delta0=0.0, kappa_e=2π * 1e6, kappa_i=0.0,
            freq_inhomogeneity=(kind=:uniform, FWHM=2π * 1e6, span_gamma=2.5, renormalize=false),
            g_inhomogeneity=(kind=:constant, g_value=2π * 100),
        )
        plan = SCM.resolve_ensemble_method(c_unk, :auto)
        err8 = try
            SCM.prepare_derived(c_unk; ensemble_method=:auto)
            ""
        catch e
            sprint(showerror, e)
        end
        println("[rebench] B8 :auto+unknown Δ kind plan=", plan.method, " error=", err8)

        readme = read(joinpath(@__DIR__, "..", "README.md"), String)
        mono = read(joinpath(@__DIR__, "..", "MONOLITH.md"), String)
        doc_bare = occursin("julia --startup-file=no test/spin_cavity_monolith.jl", readme) ||
                   occursin("julia --startup-file=no test/spin_cavity_monolith.jl", mono)
        println("[rebench] B10 documented bare test invoke (needs --project=.)? ", doc_bare)
        println("[rebench] B4 nshards default=", SCM.resolve_cpu_nshards(48),
                "  @threads present=", occursin("@threads", src))
    end
end

println("Re-bench finished (tip ", TIP_SHA, ").")
