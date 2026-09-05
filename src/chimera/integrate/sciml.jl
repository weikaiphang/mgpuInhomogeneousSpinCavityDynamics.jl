# Time integration owner: DifferentialEquations.jl / OrdinaryDiffEq (SciML).
# Production stepper is Tsit5 via OrdinaryDiffEq for single-device and
# multi-GPU. Multi-GPU only shards correlators and NCCL-Allreduces the
# 3M row-sums *inside* this same vector field — there is no homemade RK.

const CHIMERA_SCIML_ALG = OrdinaryDiffEq.Tsit5
const CHIMERA_INTEGRATOR = :sciml_ordinarydiffeq

function chimera_ode_problem(rhs, u0, tspan, p)
    return ODEProblem(rhs, u0, tspan, p)
end

function chimera_solve(prob; reltol, abstol, callback=nothing, kwargs...)
    return solve(
        prob,
        OrdinaryDiffEq.Tsit5();
        reltol = reltol,
        abstol = abstol,
        callback = callback,
        save_on = false,
        save_everystep = false,
        dense = false,
        kwargs...,
    )
end
