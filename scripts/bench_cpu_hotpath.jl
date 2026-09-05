# Product-path CPU hot-path bench for tip 069e8b9 (PR #4 / monolith).
#   julia --project=. --startup-file=no -t 1 scripts/bench_cpu_hotpath.jl
#
# Measures the production step (`_rk6_order2!` + reused `Order2Pool`),
# not a harness that reintroduces `kS[1:n]` slices (PR #8 false-positive).

using Printf

include(joinpath(@__DIR__, "..", "src", "SpinCavityMonolith.jl"))
using .SpinCavityMonolith
const SCM = SpinCavityMonolith

const TIP = "069e8b9ac127c42885cb044c151b0e3a7caba610"

_bench_Et(t) = ComplexF64(0.1, 0.05)

function setup_rhs(Mb)
    Nj = fill(4.0, Mb)
    delta_b = [1e5 * sin(0.7 * j) for j in 1:Mb]
    g_b = fill(2π * 100.0, Mb)
    u = SCM.build_u0_2nd_order(Mb, Nj, Float64, :equator)
    u[1] = 0.01 + 0.002im
    s, L, c, o = SCM.dense_to_shards(u, Mb, 1)
    ds = zero(s); dL = [zero(x) for x in L]
    return (ds, dL, s, L, c, o, 0.0, 2π * 1e6, 2π * 2e5, delta_b, g_b, Mb, 0.1 + 0.05im)
end

function setup_order2(Mb)
    Nj = fill(4.0, Mb)
    delta_b = [2π * 1e4 * (j - Mb / 2) for j in 1:Mb]
    g_b = fill(2π * 100.0, Mb)
    u = SCM.build_u0_2nd_order(Mb, Nj, Float64, :equator)
    u[1] = 0.01 + 0.002im
    s, L, c, o = SCM.dense_to_shards(u, Mb, 1)
    pool = SCM.Order2Pool(s, L)
    return (; Mb, Nj, delta_b, g_b, s, L, c, o, pool,
            ke=2π * 1e6, ki=2π * 1e5)
end

function time_rhs(args; nwarm=20, niter=200)
    for _ in 1:nwarm
        SCM.rhs2_sharded!(args...)
    end
    t0 = time_ns()
    for _ in 1:niter
        SCM.rhs2_sharded!(args...)
    end
    t1 = time_ns()
    return (t1 - t0) / niter / 1e6
end

function harness_slice_bytes()
    # Recreates the PR #8 call-site leftover: Vector slices `kS[1:n]`.
    # This is NOT the product path. Product uses NTuple + unrolled `_lincomb_n!`.
    kS = [zeros(ComplexF64, 8) for _ in 1:6]
    kL = [[zeros(ComplexF64, 8)] for _ in 1:6]
    bytes = @allocated begin
        _ = kS[1:2]; _ = kL[1:2]
        _ = kS[1:3]; _ = kL[1:3]
        _ = kS[1:4]; _ = kL[1:4]
        _ = kS[1:5]; _ = kL[1:5]
    end
    return bytes
end

function main()
    nt = Threads.nthreads()
    println("=== bench_cpu_hotpath tip=", TIP, " nthreads=", nt, " ===")
    println("path: product `_rk6_order2!` / `integrate_order2_sharded!` / Order2Pool")
    println("nshards default M=48: ", SCM.resolve_cpu_nshards(48))

    for Mb in (8, 48)
        args = setup_rhs(Mb)
        SCM.rhs2_sharded!(args...)
        SCM.rhs2_sharded!(args...)
        bytes = @allocated SCM.rhs2_sharded!(args...)
        ms = time_rhs(args)
        @printf("[bench] B1 rhs2_sharded! M=%d  bytes/RHS=%d  time=%.4f ms  nthreads=%d\n",
                Mb, bytes, ms, nt)
    end

    ctx = setup_order2(8)
    tab5 = SCM.Tsit5Tab(Float64)
    tabck = SCM.CK45Tab(Float64)
    SCM._rk6_order2!(ctx.pool, ctx.s, ctx.L, ctx.c, ctx.o, 0.0, ctx.ke, ctx.ki,
                     ctx.delta_b, ctx.g_b, ctx.Mb, _bench_Et, 0.0, 1e-8, tab5)
    a2 = @allocated SCM._rk6_order2!(ctx.pool, ctx.s, ctx.L, ctx.c, ctx.o, 0.0, ctx.ke, ctx.ki,
                                    ctx.delta_b, ctx.g_b, ctx.Mb, _bench_Et, 0.0, 1e-8, tab5)
    SCM._rk6_order2!(ctx.pool, ctx.s, ctx.L, ctx.c, ctx.o, 0.0, ctx.ke, ctx.ki,
                     ctx.delta_b, ctx.g_b, ctx.Mb, _bench_Et, 0.0, 1e-8, tabck)
    a2c = @allocated SCM._rk6_order2!(ctx.pool, ctx.s, ctx.L, ctx.c, ctx.o, 0.0, ctx.ke, ctx.ki,
                                     ctx.delta_b, ctx.g_b, ctx.Mb, _bench_Et, 0.0, 1e-8, tabck)
    @printf("[bench] B2 product `_rk6_order2!` Tsit5 bytes/step=%d\n", a2)
    @printf("[bench] B2 product `_rk6_order2!` CK45  bytes/step=%d\n", a2c)

    function _ass_integrate(tspan, pool)
        s0, L0, c0, o0 = SCM.build_u0_2nd_mgpu(ctx.Mb, ctx.Nj, :equator, 1)
        _, _, nst = SCM.integrate_order2_sharded!(
            s0, L0, c0, o0, 0.0, ctx.ke, ctx.ki, ctx.delta_b, ctx.g_b, ctx.Mb,
            _bench_Et, tspan; reltol=1e-6, abstol=1e-6, dt0=1e-7, pool=pool)
        return nst
    end
    _ass_integrate((0.0, 2e-7), ctx.pool)
    nst_s = _ass_integrate((0.0, 2e-7), ctx.pool)
    b_s = @allocated _ass_integrate((0.0, 2e-7), ctx.pool)
    nst_l = _ass_integrate((0.0, 8e-7), ctx.pool)
    b_l = @allocated _ass_integrate((0.0, 8e-7), ctx.pool)
    per = (b_l - b_s) / max(nst_l - nst_s, 1)
    @printf("[bench] B2 integrate_order2_sharded! reused pool short nsteps=%d bytes=%d\n", nst_s, b_s)
    @printf("[bench] B2 integrate_order2_sharded! reused pool long  nsteps=%d bytes=%d\n", nst_l, b_l)
    @printf("[bench] B2 incremental bytes/step=%.2f  TARGET=0  (product path, reused Order2Pool)\n", per)

    sl = harness_slice_bytes()
    sl2 = harness_slice_bytes()
    @printf("[bench] PR#8-style kS[1:n] call-site leftover bytes=%d (warmup) / %d (2nd) — NOT product\n",
            sl, sl2)

    d = (timespan=(0.0, 1e-4), FWHM=1e6, kappa_t=2π * 1e6, g_mean=2π * 100,
         sqrt_kappa_e=sqrt(2π * 1e6))
    pulse = SCM.CompositePulse(1, 4, 4, d)
    uu = SCM.initial_guess(pulse; seed=1)
    EE = SCM.build_E_of_t(pulse, uu)
    ts, te, = SCM.decode(pulse, uu)
    tmid = (ts[1] + te[1]) / 2
    EE(tmid)
    nE = @allocated EE(tmid)
    println("[bench] B3 bspline E(t) bytes/eval=", nE, "  TARGET=0 (still open if >0)")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
