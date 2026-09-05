#!/usr/bin/env julia
#   julia --project=. scripts/run_monolith.jl --settings examples/monolith_forward.jl
#   julia --project=. scripts/run_monolith.jl -s examples/monolith_optimizer.jl --grad adjoint

include(joinpath(@__DIR__, "..", "src", "SpinCavityMonolith.jl"))
using .SpinCavityMonolith
SpinCavityMonolith._load_optional_stacks!()
exit(SpinCavityMonolith.main(ARGS))
