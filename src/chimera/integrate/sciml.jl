# Time integration owner: DifferentialEquations.jl / OrdinaryDiffEq (SciML).
# Single-device production path is Tsit5 via OrdinaryDiffEq.
# The custom MGPU Tsit5/CK45 stepper is only the multi-GPU performance path
# and must match this SciML trajectory within solver tolerances.

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
