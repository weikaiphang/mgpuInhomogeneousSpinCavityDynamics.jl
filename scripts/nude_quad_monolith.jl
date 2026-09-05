#!/usr/bin/env julia
# CLI for the nude-quad multi-GPU monolith.
#   julia --project=. scripts/nude_quad_monolith.jl --settings examples/monolith_forward.jl
#   julia --project=. scripts/nude_quad_monolith.jl -s examples/monolith_optimizer.jl -m optimizer

include(joinpath(@__DIR__, "..", "src", "monolith_mgpu.jl"))
using .NudeQuadMonolith
NudeQuadMonolith._load_optional_stacks!()
exit(NudeQuadMonolith.main(ARGS))
