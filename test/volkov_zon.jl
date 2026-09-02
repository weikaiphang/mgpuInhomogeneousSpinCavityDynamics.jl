# ============================================================
# Tests for the standalone analytical Volkov-Zon solver
# (src/volkov_zon.jl).
#
# Runs on its own -- no CUDA, no DifferentialEquations:
#
#     julia --project=. test/volkov_zon.jl
#
# It is also included from test/runtests.jl.
# ============================================================

using Test

if !isdefined(@__MODULE__, :VolkovZon)
    include(joinpath(@__DIR__, "..", "src", "volkov_zon.jl"))
end
using .VolkovZon

const VZ = VolkovZon
# NOTE: `gaussian_drive` / `wurst_drive` are also defined by src/pulses.jl,
# which test/runtests.jl loads into Main, so the drive constructors are
# qualified here. Everything else the module exports is unique.
const TWOPI = 2 * pi

# ------------------------------------------------------------
# Numerics toolbox
# ------------------------------------------------------------

@testset "VolkovZon: Faddeeva and erfc" begin
    @test abs(faddeeva(0.0 + 0.0im) - 1.0) < 1e-12
    # w(z) + w(-z) = 2 exp(-z^2)
    z = 1.3 + 0.4im
    @test abs(faddeeva(z) + faddeeva(-z) - 2 * exp(-z * z)) < 1e-12
    # w(i y) = exp(y^2) erfc(y)  ->  the real-argument erfc used for the
    # truncated-Gaussian coupling moments
    @test abs(VZ._erfc(0.5) - 0.4795001221869535) < 1e-12
    @test abs(VZ._erfc(1.0) - 0.15729920705028513) < 1e-12
    @test abs(VZ._erfc(2.0) - 0.004677734981047266) < 1e-12
    @test abs(VZ._erfc(-1.5) - 1.9661051464753107) < 1e-12
    @test abs(VZ._erf(0.0)) < 1e-12
end

@testset "VolkovZon: FFT" begin
    n = 64
    x = ComplexF64[cos(0.7k) + im * sin(0.13k^2) for k in 0:(n - 1)]
    X = VZ._fft(x)
    naive = ComplexF64[sum(x[k + 1] * cis(-2pi * m * k / n) for k in 0:(n - 1))
                       for m in 0:(n - 1)]
    @test maximum(abs, X .- naive) < 1e-11
    @test maximum(abs, VZ._ifft(X) .- x) < 1e-13
    @test VZ._next_pow2(1000) == 1024
    @test VZ._next_pow2(1024) == 1024
end

@testset "VolkovZon: polynomial roots" begin
    # coefficients spanning ~1e27, as they do for a real system: the degree
    # must survive, and every root must actually be a root
    c = ComplexF64[1.1761353261999338e27, 6.723796045017493e20,
                   1.7634685569329034e14, 2.162790987310664e7, 1.0]
    r = VZ._polyroots(c)
    @test length(r) == 4
    scale = maximum(abs, c)
    @test maximum(abs(VZ._polyval(c, rk)) for rk in r) < 1e-12 * scale

    q = 1.3 - 0.7im
    @test maximum(abs, VZ._shift_pow(q, 1) .- ComplexF64[q, 1]) < 1e-15
    @test abs(VZ._polyval(VZ._shift_pow(q, 4), 0.2 + 0.1im) -
              (0.2 + 0.1im + q)^4) < 1e-12
end

@testset "VolkovZon: Gauss-Legendre and cumulative integral" begin
    x, w = VZ._gauss_legendre(10)
    @test abs(sum(w) - 2) < 1e-14
    @test abs(sum(w .* x .^ 4) - 2 / 5) < 1e-14      # exact through degree 19
    t = collect(range(0.0, 3.0; length = 401))
    h = t[2] - t[1]
    f = exp.(-t) .* sin.(3 .* t)
    exact = (3 .- exp.(-t) .* (3 .* cos.(3 .* t) .+ sin.(3 .* t))) ./ 10
    @test maximum(abs, VZ._cumint(f, h) .- exact) < 1e-7
end

# ------------------------------------------------------------
# Distributions
# ------------------------------------------------------------

@testset "VolkovZon: detuning distributions" begin
    FWHM = TWOPI * 1e6
    for det in (DetuningLorentzian(FWHM), DetuningPowerLaw(FWHM, 2),
                DetuningPowerLaw(FWHM, 4), DetuningGaussian(FWHM))
        # FWHM really is the full width at half maximum
        @test abs(detuning_pdf(det, FWHM / 2) / detuning_pdf(det, 0.0) - 0.5) < 1e-12

        # the quadrature grid integrates the density to one
        sys = VZSystem(kappa_e = TWOPI * 1e6, detuning = det,
                       coupling = CouplingConstant(TWOPI * 100), C_ens = 0.5)
        d, wd, _, _ = vz_ensemble_grid(sys;
                                       M_delta = VolkovZon.is_rational(det) ? 4000 : 600)
        @test abs(sum(wd) - 1) < 1e-10
        @test all(wd .>= 0)

        # K(0) = 1, and Ktilde is the Laplace transform of K
        @test abs(char_fn(det, 0.0) - 1) < 1e-14
        decay = det isa DetuningGaussian ? det.sigma : det.w
        tq, wq = VZ._gl_on(4000, 0.0, 60.0 / decay)
        for s in (3e6 + 1.1e6im, 1e5 - 4e6im, 2e7 + 0.0im)
            ref = sum(wq .* [char_fn(det, tv) for tv in tq] .* exp.(-s .* tq))
            @test abs(laplace_kernel(det, s) - ref) < 1e-10 * abs(ref)
        end
    end

    # n = 1 power law IS the Lorentzian
    lor = DetuningLorentzian(TWOPI * 1e6)
    pl1 = DetuningPowerLaw(TWOPI * 1e6, 1)
    @test abs(pl1.w - lor.w) < 1e-9 * lor.w
    for x in (0.0, 1e6, -4e6)
        @test abs(detuning_pdf(pl1, x) - detuning_pdf(lor, x)) < 1e-9 * detuning_pdf(lor, 0.0)
    end

    @test VolkovZon.is_rational(lor)
    @test VolkovZon.is_rational(DetuningPowerLaw(TWOPI * 1e6, 3))
    @test !VolkovZon.is_rational(DetuningGaussian(TWOPI * 1e6))
end

@testset "VolkovZon: coupling moments" begin
    cases = (
        (CouplingGaussian(TWOPI * 100, TWOPI * 30), g -> exp(-((g - TWOPI * 100) / (TWOPI * 30))^2 / 2),
         TWOPI * 100 - 6 * TWOPI * 30, TWOPI * 100 + 6 * TWOPI * 30),
        (CouplingLorentzian(TWOPI * 100, TWOPI * 40), g -> 1 / ((g - TWOPI * 100)^2 + (TWOPI * 20)^2),
         TWOPI * 100 - 20 * TWOPI * 20, TWOPI * 100 + 20 * TWOPI * 20),
        (CouplingPowerLaw(5 / 3, TWOPI * 1.0, TWOPI * 1000.0), g -> g^(-5 / 3),
         TWOPI * 1.0, TWOPI * 1000.0),
    )
    for (cpl, shape, lo, hi) in cases
        g, w = VZ._gl_on(4000, lo, hi)
        p = w .* shape.(g)
        p ./= sum(p)
        @test abs(coupling_moment(cpl, 1) - sum(p .* g)) < 1e-10 * abs(sum(p .* g))
        @test abs(coupling_moment(cpl, 2) - sum(p .* g .^ 2)) < 1e-10 * sum(p .* g .^ 2)
    end
    @test coupling_moment(CouplingConstant(7.0), 2) == 49.0
end

# ------------------------------------------------------------
# System bookkeeping and the polariton poles
# ------------------------------------------------------------

@testset "VolkovZon: cooperativity and poles" begin
    ke = TWOPI * 1e6
    det = DetuningLorentzian(TWOPI * 1e6)
    cpl = CouplingConstant(TWOPI * 100)

    sys = VZSystem(kappa_e = ke, detuning = det, coupling = cpl, C_ens = 0.6)
    @test abs(sys.C_ens - 0.6) < 1e-12
    sys2 = VZSystem(kappa_e = ke, detuning = det, coupling = cpl, N = sys.N)
    @test abs(sys2.C_ens - 0.6) < 1e-12
    # matches the main package's Lorentzian relation N = C kappa_t FWHM / (4 <g^2>)
    @test abs(sys.N - 0.6 * ke * det.FWHM / (4 * coupling_moment(cpl, 2))) <
          1e-9 * sys.N

    # Vieta on chi(s) = (s + lambda)(s + w) - s0 Omega^2
    roots = VZ._residue_data(sys).roots
    @test length(roots) == 2
    lam = sys.lambda
    @test abs(sum(roots) + (lam + det.w)) < 1e-12 * abs(lam + det.w)
    @test abs(prod(roots) - (lam * det.w + sys.Omega2)) <
          1e-12 * abs(lam * det.w + sys.Omega2)
    @test all(real.(roots) .< 0)                       # C < 1: below threshold
    # D(s) really does vanish there
    for r in roots
        @test abs(transfer_denominator(sys, r)) < 1e-6 * abs(r)
    end

    # bad-cavity limit: a very broad line gives kappa_eff = kappa_t (1 + C)
    broad = VZSystem(kappa_e = ke, detuning = DetuningLorentzian(TWOPI * 1e9),
                     coupling = cpl, C_ens = 0.6)
    rb = VZ._residue_data(broad).roots
    slow = rb[argmin(abs.(real.(rb)))]
    @test abs(real(slow) + (ke / 2) * 1.6) < 2e-3 * (ke / 2) * 1.6
end

# ------------------------------------------------------------
# Solver correctness
# ------------------------------------------------------------

@testset "VolkovZon: empty ensemble reduces to the bare cavity" begin
    ke, E0 = TWOPI * 1e6, 2.5
    sys = VZSystem(kappa_e = ke, kappa_i = TWOPI * 2e5, delta0 = TWOPI * 3e5,
                   detuning = DetuningLorentzian(TWOPI * 1e6),
                   coupling = CouplingConstant(TWOPI * 100), N = 0.0,
                   a0 = 0.4 - 0.2im, drive = VZ.constant_drive(E0))
    Ttotal = 5e-6
    sol = vz_solve(sys; Ttotal = Ttotal, Nt_save = 401, method = :residue)
    lam = sys.lambda
    exact = sys.a0 .* exp.(-lam .* sol.t) .+
            sqrt(ke) * E0 .* (1 .- exp.(-lam .* sol.t)) ./ lam
    @test maximum(abs, sol.a .- exact) < 1e-12 * maximum(abs, exact)
    @test abs(sol.a[1] - sys.a0) < 1e-14
    @test maximum(abs, sol.Sigma_m) < 1e-6   # exactly zero analytically
    @test maximum(abs, sol.a_out .- (sol.E .- sqrt(ke) .* sol.a)) == 0.0

    # the Bromwich path only converges as O(h) for a drive that is still on at
    # the window edges -- documented behaviour, checked here so it stays bounded
    solf = vz_solve(sys; Ttotal = Ttotal, Nt_save = 401, method = :fft)
    @test maximum(abs, solf.a .- exact) < 1e-4 * maximum(abs, exact)
end

@testset "VolkovZon: residue and Bromwich paths agree" begin
    ke = TWOPI * 1e6
    drive = VZ.gaussian_drive(t0 = 6e-6, sigma = 1e-6, amp = 1.0)
    for det in (DetuningLorentzian(TWOPI * 1e6), DetuningPowerLaw(TWOPI * 1e6, 3))
        sys = VZSystem(kappa_e = ke, delta0 = TWOPI * 1e5, detuning = det,
                       coupling = CouplingPowerLaw(5 / 3, TWOPI * 1.0, TWOPI * 1000.0),
                       C_ens = 0.6, drive = drive)
        r = vz_solve(sys; Ttotal = 20e-6, Nt_save = 2001, method = :residue)
        f = vz_solve(sys; Ttotal = 20e-6, Nt_save = 2001, method = :fft)
        @test r.method === :residue
        @test f.method === :fft
        @test maximum(abs, r.a .- f.a) < 1e-8 * maximum(abs, r.a)
        @test maximum(abs, r.Sigma_m .- f.Sigma_m) < 1e-8 * maximum(abs, r.Sigma_m)
        @test r.poles !== nothing && f.poles === nothing
        # residues of Q/chi sum to one
        @test abs(sum(r.residues) - 1) < 1e-10
    end
end

@testset "VolkovZon: analytic solution vs direct binned integration" begin
    # A power-law line with n = 4 has |delta|^-8 tails, so a bounded bin grid
    # captures it and a plain RK4 integration of the binned linear model is a
    # legitimate independent reference.
    ke = TWOPI * 1e6
    det = DetuningPowerLaw(TWOPI * 1e5, 4)
    cpl = CouplingConstant(TWOPI * 100)
    drive = VZ.gaussian_drive(t0 = 0.6e-6, sigma = 0.12e-6, amp = 1.0)
    sys = VZSystem(kappa_e = ke, detuning = det, coupling = cpl,
                   C_ens = 0.6, drive = drive)

    Ttotal, Nt = 2e-6, 201
    sol = vz_solve(sys; Ttotal = Ttotal, Nt_save = Nt)

    nd = 400
    x, wq = VZ._gl_on(nd, -30 * det.w, 30 * det.w)
    p = wq .* [detuning_pdf(det, v) for v in x]
    p ./= sum(p)

    lam, ske, N, g, s0 = sys.lambda, sqrt(ke), sys.N, sys.g1, sys.s0
    nsub = 8
    h = (Ttotal / (Nt - 1)) / nsub
    a = 0.0 + 0.0im
    sm = zeros(ComplexF64, nd)
    ref = zeros(ComplexF64, Nt)
    rhs(tt, av, smv) = (-lam * av + ske * drive(tt) - im * N * g * sum(p .* smv),
                        -im .* x .* smv .+ (im * s0 * g) .* av)
    tt = 0.0
    for n in 1:(Nt - 1)
        for _ in 1:nsub
            k1a, k1s = rhs(tt, a, sm)
            k2a, k2s = rhs(tt + h / 2, a + h / 2 * k1a, sm .+ (h / 2) .* k1s)
            k3a, k3s = rhs(tt + h / 2, a + h / 2 * k2a, sm .+ (h / 2) .* k2s)
            k4a, k4s = rhs(tt + h, a + h * k3a, sm .+ h .* k3s)
            a += h / 6 * (k1a + 2k2a + 2k3a + k4a)
            sm .+= (h / 6) .* (k1s .+ 2 .* k2s .+ 2 .* k3s .+ k4s)
            tt += h
        end
        ref[n + 1] = a
    end
    @test maximum(abs, sol.a .- ref) < 1e-7 * maximum(abs, ref)
end

@testset "VolkovZon: resolved bins reproduce the collective coherence" begin
    ke = TWOPI * 1e6
    drive = VZ.gaussian_drive(t0 = 6e-6, sigma = 1e-6, amp = 1.0)
    sys = VZSystem(kappa_e = ke, detuning = DetuningPowerLaw(TWOPI * 1e6, 4),
                   coupling = CouplingPowerLaw(5 / 3, TWOPI * 1.0, TWOPI * 1000.0),
                   C_ens = 0.6, a0 = 0.2 + 0.1im, drive = drive)
    Nt = 201
    d, wd, g, wg = vz_ensemble_grid(sys; M_delta = 1200, M_g = 8)
    sol = vz_solve(sys; Ttotal = 20e-6, Nt_save = Nt, bin_deltas = d, bin_gs = g)

    @test size(sol.sigma_minus) == (length(d), length(g), Nt)
    Sm = [sys.N * sum(wd[i] * wg[j] * sol.sigma_minus[i, j, n]
                      for i in eachindex(d), j in eachindex(g))
          for n in 1:Nt]
    @test maximum(abs, Sm .- sol.Sigma_m) < 1e-5 * maximum(abs, sol.Sigma_m)
    @test maximum(abs, sol.sigma_plus .- conj.(sol.sigma_minus)) == 0.0
    # every spin starts in the ground state with no coherence
    @test maximum(abs, sol.sigma_minus[:, :, 1]) <
          1e-10 * maximum(abs, sol.sigma_minus)
end

@testset "VolkovZon: inversion bookkeeping and validity flag" begin
    ke = TWOPI * 1e6
    sys = VZSystem(kappa_e = ke, detuning = DetuningLorentzian(TWOPI * 1e6),
                   coupling = CouplingConstant(TWOPI * 100), C_ens = 0.6,
                   drive = VZ.gaussian_drive(t0 = 6e-6, sigma = 1e-6, amp = 1.0))
    sol = vz_solve(sys; Ttotal = 20e-6, Nt_save = 2001)
    @test abs(sol.Sigma_z[1] + sys.N / 2) < 1e-9 * sys.N       # ground state
    @test all(sol.Sigma_z .<= 0)                               # stays below zero
    @test sol.linear_validity < 1e-6
    @test sol.N_total > 0
    @test sol.detuning.kind === :lorentzian
    @test sol.coupling.kind === :constant

    # an inverted ensemble radiates: inversion decreases from +N/2
    inv = VZSystem(kappa_e = ke, detuning = DetuningLorentzian(TWOPI * 1e6),
                   coupling = CouplingConstant(TWOPI * 100), C_ens = 0.6,
                   initial_condition = :inverted, coherence0 = 1e-2 + 0.0im)
    isol = vz_solve(inv; Ttotal = 20e-6, Nt_save = 2001)
    @test abs(isol.Sigma_z[1] - inv.N / 2) < 1e-9 * inv.N
    @test isol.Sigma_z[end] < isol.Sigma_z[1]          # it radiates
    @test all(diff(isol.Sigma_z) .<= 1e-9 * inv.N)     # monotonically
    @test isol.linear_validity < 1e-3
    @test maximum(abs, isol.a) > 0
end

@testset "VolkovZon: argument validation" begin
    det = DetuningLorentzian(TWOPI * 1e6)
    cpl = CouplingConstant(TWOPI * 100)
    @test_throws ErrorException VZSystem(kappa_e = TWOPI * 1e6, detuning = det,
                                         coupling = cpl)
    @test_throws ErrorException VZSystem(kappa_e = TWOPI * 1e6, detuning = det,
                                         coupling = cpl, C_ens = 0.6, N = 1e6)
    @test_throws ErrorException VZSystem(kappa_e = -1.0, detuning = det,
                                         coupling = cpl, C_ens = 0.6)
    @test_throws ErrorException VZSystem(kappa_e = TWOPI * 1e6, detuning = det,
                                         coupling = cpl, C_ens = 0.6,
                                         initial_condition = :sideways)
    @test_throws ErrorException DetuningLorentzian(-1.0)
    @test_throws ErrorException DetuningPowerLaw(TWOPI * 1e6, 0)
    @test_throws ErrorException CouplingPowerLaw(5 / 3, 10.0, 1.0)

    sys = VZSystem(kappa_e = TWOPI * 1e6, detuning = DetuningGaussian(TWOPI * 1e6),
                   coupling = cpl, C_ens = 0.6)
    @test_throws ErrorException vz_solve(sys; Ttotal = 1e-6, method = :residue)
    @test_throws ErrorException vz_solve(sys; Ttotal = -1.0)
    @test vz_solve(sys; Ttotal = 1e-6, Nt_save = 101).method === :fft
end
