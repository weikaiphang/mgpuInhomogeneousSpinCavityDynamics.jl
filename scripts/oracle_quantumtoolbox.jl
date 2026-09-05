# Tiny Hilbert-space oracle (QuantumToolbox.jl) vs 1st-order cumulant.
# Usage: julia --project=. scripts/oracle_quantumtoolbox.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using QuantumToolbox
using OrdinaryDiffEq
using DifferentialEquations
using DiffEqCallbacks

include(joinpath(@__DIR__, "..", "src", "chimera", "eoms", "state_1st.jl"))
include(joinpath(@__DIR__, "..", "src", "chimera", "eoms", "closure_1st.jl"))
include(joinpath(@__DIR__, "..", "src", "chimera", "integrate", "sciml.jl"))
const HAVE_QUANTUMTOOLBOX = true
include(joinpath(@__DIR__, "..", "src", "chimera", "oracle", "quantumtoolbox.jl"))

cmp = compare_oracle_to_cumulant_1st(; Ncut=8, g=0.015, κe=0.25, E=0.008, tspan=(0.0, 0.8), nsave=17)
println("max |⟨a⟩_exact − ⟨a⟩_cumulant| = ", cmp.err)
cmp.err < 0.05 || error("oracle disagreement too large: $(cmp.err)")
