# ============================================================
# Analytical Volkov-Zon method -- demonstration
#
#     julia --project=. examples/volkov_zon_demo.jl
#
# Runs entirely on the standalone solver in src/volkov_zon.jl:
# no GPU, no ODE integration, no package dependencies beyond the
# Julia standard library.
#
# All frequencies are angular (rad/s) and all times are seconds,
# matching the rest of the repository.
# ============================================================

using Printf

include(joinpath(@__DIR__, "..", "src", "volkov_zon.jl"))
using .VolkovZon

const TWOPI = 2 * pi

# ------------------------------------------------------------
# Shared cavity / pulse setup (mirrors examples/run_demo.jl)
# ------------------------------------------------------------

const KAPPA_E = TWOPI * 1e6
const KAPPA_I = 0.0
const DELTA0 = 0.0
const C_ENS = 0.6
const FWHM = TWOPI * 1e6

const TTOTAL = 60e-6
const NT_SAVE = 6001

signal = gaussian_drive(t0 = 20e-6, sigma = (10 / 6) * 1e-6, amp = 1.0)

function report(label, sol)
    ipk = argmax(abs.(sol.a))
    @printf("%-34s  method=%-8s N=%.3e\n", label, String(sol.method), sol.N_total)
    @printf("    peak |<a>|      = %.6e  at t = %.3f us\n",
            abs(sol.a[ipk]), sol.t[ipk] * 1e6)
    @printf("    peak |a_out|    = %.6e\n", maximum(abs, sol.a_out))
    @printf("    peak |Sigma^+|  = %.6e\n", maximum(abs, sol.Sigma_p))
    @printf("    linear_validity = %.3e\n", sol.linear_validity)
    if sol.poles !== nothing
        for (k, p) in enumerate(sol.poles)
            @printf("    pole %d: decay %.4e rad/s, shift %+.4e rad/s\n",
                    k, -real(p), imag(p))
        end
    end
    println()
end

println("=" ^ 64)
println("1.  The three detuning line shapes, constant coupling")
println("=" ^ 64)
println()

for (label, det) in (
    ("Lorentzian detuning", DetuningLorentzian(FWHM)),
    ("Power-law detuning (n = 2)", DetuningPowerLaw(FWHM, 2)),
    ("Power-law detuning (n = 4)", DetuningPowerLaw(FWHM, 4)),
    ("Gaussian detuning", DetuningGaussian(FWHM)),
)
    sys = VZSystem(kappa_e = KAPPA_E, kappa_i = KAPPA_I, delta0 = DELTA0,
                   detuning = det, coupling = CouplingConstant(TWOPI * 100),
                   C_ens = C_ENS, drive = signal)
    report(label, vz_solve(sys; Ttotal = TTOTAL, Nt_save = NT_SAVE))
end

println("=" ^ 64)
println("2.  The three coupling distributions, Lorentzian line")
println("=" ^ 64)
println()

for (label, cpl) in (
    ("Constant coupling", CouplingConstant(TWOPI * 100)),
    ("Gaussian coupling", CouplingGaussian(TWOPI * 100, TWOPI * 30)),
    ("Lorentzian coupling", CouplingLorentzian(TWOPI * 100, TWOPI * 40)),
    ("Power-law coupling (alpha=5/3)", CouplingPowerLaw(5 / 3, TWOPI * 1.0, TWOPI * 1000.0)),
)
    sys = VZSystem(kappa_e = KAPPA_E, detuning = DetuningLorentzian(FWHM),
                   coupling = cpl, C_ens = C_ENS, drive = signal)
    sol = vz_solve(sys; Ttotal = TTOTAL, Nt_save = NT_SAVE)
    @printf("%-34s  <g> = %.4e  <g^2> = %.4e\n", label,
            sol.g_moments.g1, sol.g_moments.g2)
    @printf("    peak |<a>| = %.6e   (only <g^2> enters the cavity dynamics)\n\n",
            maximum(abs, sol.a))
end

println("=" ^ 64)
println("3.  Free induction decay of an inverted ensemble")
println("=" ^ 64)
println()

for (label, det) in (
    ("Lorentzian", DetuningLorentzian(FWHM)),
    ("Power-law n = 3", DetuningPowerLaw(FWHM, 3)),
    ("Gaussian", DetuningGaussian(FWHM)),
)
    sys = VZSystem(kappa_e = KAPPA_E, detuning = det,
                   coupling = CouplingConstant(TWOPI * 100), C_ens = C_ENS,
                   initial_condition = :inverted, coherence0 = 1e-2 + 0.0im)
    sol = vz_solve(sys; Ttotal = 20e-6, Nt_save = 20001)
    ipk = argmax(abs.(sol.a))
    @printf("%-18s  burst |<a>|max = %.6e at t = %.4f us, ", label,
            abs(sol.a[ipk]), sol.t[ipk] * 1e6)
    @printf("emitted inversion = %.4e spins\n", sol.Sigma_z[1] - sol.Sigma_z[end])
end
println()

println("=" ^ 64)
println("4.  The two inversion routes agree (independent algorithms)")
println("=" ^ 64)
println()

sys = VZSystem(kappa_e = KAPPA_E, delta0 = TWOPI * 1e5,
               detuning = DetuningPowerLaw(FWHM, 3),
               coupling = CouplingPowerLaw(5 / 3, TWOPI * 1.0, TWOPI * 1000.0),
               C_ens = C_ENS, drive = signal)
r = vz_solve(sys; Ttotal = TTOTAL, Nt_save = NT_SAVE, method = :residue)
f = vz_solve(sys; Ttotal = TTOTAL, Nt_save = NT_SAVE, method = :fft)
@printf("    exact residues vs Bromwich inversion: |da|/max|a| = %.3e\n",
        maximum(abs, r.a .- f.a) / maximum(abs, r.a))
@printf("                                          |dSigma|    = %.3e\n\n",
        maximum(abs, r.Sigma_m .- f.Sigma_m) / maximum(abs, r.Sigma_m))

println("=" ^ 64)
println("5.  Resolved spins: sigma^-(delta, g, t) on a quadrature grid")
println("=" ^ 64)
println()

d, wd, g, wg = vz_ensemble_grid(sys; M_delta = 800, M_g = 8)
sol = vz_solve(sys; Ttotal = TTOTAL, Nt_save = 601, bin_deltas = d, bin_gs = g)
Sm = [sys.N * sum(wd[i] * wg[j] * sol.sigma_minus[i, j, n]
                  for i in eachindex(d), j in eachindex(g))
      for n in eachindex(sol.t)]
@printf("    grid: %d detunings x %d couplings\n", length(d), length(g))
@printf("    ensemble sum of resolved bins vs collective Sigma^-: %.3e\n",
        maximum(abs, Sm .- sol.Sigma_m) / maximum(abs, sol.Sigma_m))
ires = argmin(abs.(d))
@printf("    most resonant bin (delta = %+.3e rad/s): peak |sigma^-| = %.4e\n\n",
        d[ires], maximum(abs, sol.sigma_minus[ires, end, :]))

println("=" ^ 64)
println("6.  Cost: the analytic solution is independent of ensemble size")
println("=" ^ 64)
println()

sysL = VZSystem(kappa_e = KAPPA_E, detuning = DetuningLorentzian(FWHM),
                coupling = CouplingConstant(TWOPI * 100), C_ens = C_ENS,
                drive = signal)
vz_solve(sysL; Ttotal = TTOTAL, Nt_save = 101)          # warm up
for nt in (1001, 10001, 100001)
    el = @elapsed vz_solve(sysL; Ttotal = TTOTAL, Nt_save = nt)
    @printf("    Nt_save = %6d : %.4f s\n", nt, el)
end
println()
println("There is no ensemble discretisation to converge: the detuning and")
println("coupling averages are evaluated in closed form, so the cost depends")
println("only on how many time points are asked for.")
