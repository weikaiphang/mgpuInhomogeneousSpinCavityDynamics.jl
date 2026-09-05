# Small-system validation oracle owner: QuantumToolbox.jl (QuTiP-like, Quantum 2025).
# Used for tiny Hilbert truncations (1–2 spins, low Fock cutoff) against the
# cumulant solver. QuTiP via PythonCall is an optional extra, not required.

if !@isdefined(HAVE_QUANTUMTOOLBOX)
    const HAVE_QUANTUMTOOLBOX = false
end

function oracle_available()
    return HAVE_QUANTUMTOOLBOX
end

function _oracle_error()
    error("QuantumToolbox.jl is not available. Install it to run the exact-space oracle.")
end

"""
    oracle_tc_mesolve(; Ncut, Δ0, δ, g, κt, η, tlist, ψ0)

Exact Lindblad evolution of one spin + cavity for the chimera Hamiltonian.
Returns expectation values ⟨a⟩, ⟨S⁺⟩, ⟨Sᶻ⟩.
"""
function oracle_tc_mesolve(;
        Ncut::Integer=8,
        Δ0=0.0, δ=0.0, g=0.05, κt=0.2, η=0.02,
        tlist=range(0.0, 2.0; length=41),
        excited::Bool=false)
    oracle_available() || _oracle_error()
    QT = QuantumToolbox
    a = QT.tensor(QT.destroy(Ncut), QT.qeye(2))
    σm = QT.tensor(QT.qeye(Ncut), QT.sigmam())
    σp = QT.tensor(QT.qeye(Ncut), QT.sigmap())
    σz = QT.tensor(QT.qeye(Ncut), QT.sigmaz())
    H = Δ0 * a' * a + (δ / 2) * σz + g * (a * σp + a' * σm) + 1im * η * (a' - a)
    c_ops = [sqrt(κt) * a]
    spin = excited ? QT.basis(2, 1) : QT.basis(2, 0)
    ψ0 = QT.tensor(QT.fock(Ncut, 0), spin)
    sol = QT.mesolve(H, ψ0, collect(tlist), c_ops; e_ops=[a, σp, σz / 2])
    return (
        t = collect(tlist),
        a = sol.expect[1, :],
        Sp = sol.expect[2, :],
        Sz = sol.expect[3, :],
    )
end

function compare_oracle_to_cumulant_1st(;
        Ncut=8, Δ0=0.0, δ=0.0, g=0.02, κe=0.2, κi=0.0, E=0.01,
        tspan=(0.0, 1.0), nsave=21, reltol=1e-8, abstol=1e-8)
    oracle_available() || _oracle_error()
    tlist = range(tspan[1], tspan[2]; length=nsave)
    η = sqrt(κe) * real(E)
    exact = oracle_tc_mesolve(; Ncut=Ncut, Δ0=Δ0, δ=δ, g=g, κt=κe + κi, η=η, tlist=tlist)
    u0 = ComplexF64[0.0, 0.0, -0.5]
    p = (Δ0, κe, κi, [δ], [g], 1, t -> ComplexF64(E))
    prob = chimera_ode_problem(rhs_1st_order!, u0, tspan, p)
    saved = ComplexF64[]
    function affect!(integrator)
        push!(saved, integrator.u[1])
    end
    cb = PresetTimeCallback(collect(tlist), affect!; save_positions=(false, false))
    sol = solve(prob, OrdinaryDiffEq.Tsit5(); reltol=reltol, abstol=abstol, callback=cb,
                save_on=false, save_everystep=false, dense=false)
    a_c = saved
    length(a_c) == length(exact.a) || error("oracle / cumulant save-length mismatch")
    err = maximum(abs.(a_c .- exact.a))
    return (err=err, a_exact=exact.a, a_cumulant=a_c, t=exact.t, sol=sol)
end
