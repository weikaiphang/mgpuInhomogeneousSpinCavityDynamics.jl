using Test
using OrdinaryDiffEq
using DifferentialEquations
using DiffEqCallbacks

const _QT_OK = try
    @eval using QuantumToolbox
    true
catch
    false
end

if !_QT_OK
    @testset "QuantumToolbox oracle (optional)" begin
        @test_skip "QuantumToolbox.jl not installed"
    end
else
    if !@isdefined(HAVE_QUANTUMTOOLBOX)
        const HAVE_QUANTUMTOOLBOX = true
    end
    if !@isdefined(oracle_tc_mesolve)
        include(joinpath(@__DIR__, "..", "src", "chimera", "oracle", "quantumtoolbox.jl"))
    end
    if !@isdefined(rhs_1st_order!)
        include(joinpath(@__DIR__, "..", "src", "chimera", "eoms", "state_1st.jl"))
        include(joinpath(@__DIR__, "..", "src", "chimera", "eoms", "closure_1st.jl"))
    end
    if !@isdefined(chimera_ode_problem)
        include(joinpath(@__DIR__, "..", "src", "chimera", "integrate", "sciml.jl"))
    end

    @testset "QuantumToolbox weak-drive oracle vs 1st-order cumulant" begin
        cmp = compare_oracle_to_cumulant_1st(;
            Ncut=6, g=0.01, κe=0.3, E=0.005, tspan=(0.0, 0.4), nsave=9,
        )
        @test isfinite(cmp.err)
        @test cmp.err < 5e-2
    end
end
