# CPU hot-path bench for tip 05bed84.
#   julia --project=. --startup-file=no -t 1 scripts/bench_cpu_hotpath.jl
#   julia --project=. --startup-file=no -t 4 scripts/bench_cpu_hotpath.jl
#
# Reports bytes/RHS, bytes/step, and M=48 rhs2_sharded! timing.

using Printf

include(joinpath(@__DIR__, "..", "src", "SpinCavityMonolith.jl"))
using .SpinCavityMonolith
const SCM = SpinCavityMonolith

const TIP = "05bed847a8fa7e5b8bf3e92aca2d2977796e9d73"

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

function time_rhs(args; nwarm=20, niter=200)
    for _ in 1:nwarm
        SCM.rhs2_sharded!(args...)
    end
    t0 = time_ns()
    for _ in 1:niter
        SCM.rhs2_sharded!(args...)
    end
    t1 = time_ns()
    return (t1 - t0) / niter / 1e6   # ms
end

function integrate_once(Mb, tspan; dt0=0.0)
    Nj = fill(4.0, Mb)
    delta_b = [1e5 * sin(0.7 * j) for j in 1:Mb]
    g_b = fill(2π * 100.0, Mb)
    s, L, c, o = SCM.build_u0_2nd_mgpu(Mb, Nj, :equator, 1)
    s, L, nst = SCM.integrate_order2_sharded!(
        s, L, c, o, 0.0, 2π * 1e6, 2π * 2e5, delta_b, g_b, Mb,
        t -> 0.05 + 0.02im, tspan; reltol=1e-6, abstol=1e-6, dt0=dt0)
    return nst
end

function main()
    nt = Threads.nthreads()
    println("=== bench_cpu_hotpath tip=", TIP, " nthreads=", nt, " ===")
    println("nshards default M=48: ", SCM.resolve_cpu_nshards(48))

    for Mb in (8, 48)
        args = setup_rhs(Mb)
        SCM.rhs2_sharded!(args...)
        SCM.rhs2_sharded!(args...)
        bytes = @allocated SCM.rhs2_sharded!(args...)
        ms = time_rhs(args)
        @printf("[bench] rhs2_sharded! M=%d  bytes/RHS=%d  time=%.4f ms  nthreads=%d\n",
                Mb, bytes, ms, nt)
    end

    # B2 incremental bytes/step
    integrate_once(8, (0.0, 2e-7); dt0=1e-7)
    nst_s = integrate_once(8, (0.0, 2e-7); dt0=1e-7)
    b_s = @allocated integrate_once(8, (0.0, 2e-7); dt0=1e-7)
    nst_l = integrate_once(8, (0.0, 8e-7); dt0=1e-7)
    b_l = @allocated integrate_once(8, (0.0, 8e-7); dt0=1e-7)
    per = (b_l - b_s) / max(nst_l - nst_s, 1)
    @printf("[bench] integrate M=8 short nsteps=%d bytes=%d\n", nst_s, b_s)
    @printf("[bench] integrate M=8 long  nsteps=%d bytes=%d\n", nst_l, b_l)
    @printf("[bench] B2 incremental bytes/step=%.2f  PRIOR~19456\n", per)

    # B3
    d = (timespan=(0.0, 1e-4), FWHM=1e6, kappa_t=2π * 1e6, g_mean=2π * 100,
         sqrt_kappa_e=sqrt(2π * 1e6))
    pulse = SCM.CompositePulse(1, 4, 4, d)
    uu = SCM.initial_guess(pulse; seed=1)
    EE = SCM.build_E_of_t(pulse, uu)
    ts, te, = SCM.decode(pulse, uu)
    tmid = (ts[1] + te[1]) / 2
    EE(tmid)
    nE = @allocated EE(tmid)
    println("[bench] B3 bspline E(t) bytes/eval=", nE, "  PRIOR=704")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
