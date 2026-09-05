# Derive 1st- and 2nd-order inhomogeneous TC equations with QuantumCumulants.
# Usage: julia --project=. scripts/derive_qc_eoms.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using QuantumCumulants
using SecondQuantizedAlgebra

include(joinpath(@__DIR__, "..", "src", "chimera", "eoms", "hamiltonian.jl"))
include(joinpath(@__DIR__, "..", "src", "chimera", "eoms", "quantum_cumulants_impl.jl"))

println("Hamiltonian: ", chimera_hamiltonian_symbols())

eqs1 = _derive_tc_meanfield_impl(1, 1)
eqs2 = _derive_tc_meanfield_impl(1, 2)
eqsM = _derive_tc_meanfield_impl(2, 1)

outdir = joinpath(@__DIR__, "..", "src", "chimera", "eoms", "generated")
mkpath(outdir)
open(joinpath(outdir, "tc_equations.txt"), "w") do io
    println(io, "=== order 1, M=1 ===")
    println(io, eqs1)
    println(io, "\n=== order 2, M=1 ===")
    println(io, eqs2)
    println(io, "\n=== order 1, M=2 indexed ===")
    println(io, eqsM)
end
println("Wrote ", joinpath(outdir, "tc_equations.txt"))
println("order-1 M=1 states: ", length(eqs1.states))
println("order-2 M=1 states: ", length(eqs2.states))
println("order-1 M=2 states: ", length(eqsM.states))
