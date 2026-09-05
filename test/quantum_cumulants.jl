using Test

const _QC_OK = try
    @eval using QuantumCumulants
    true
catch
    false
end

if !_QC_OK
    @testset "QuantumCumulants algebra (required)" begin
        @test false || error("QuantumCumulants.jl must be installed for chimera CI")
    end
else
    if !@isdefined(CHIMERA_HAMILTONIAN)
        include(joinpath(@__DIR__, "..", "src", "chimera", "eoms", "hamiltonian.jl"))
    end
    if !isdefined(@__MODULE__, :_derive_tc_meanfield_impl)
        include(joinpath(@__DIR__, "..", "src", "chimera", "eoms", "quantum_cumulants_impl.jl"))
    end
    if !@isdefined(qc_equation_count)
        include(joinpath(@__DIR__, "..", "src", "chimera", "eoms", "quantum_cumulants.jl"))
    end

    @testset "QuantumCumulants derives 1st- and 2nd-order TC EOMs" begin
        eqs1 = _derive_tc_meanfield_impl(1, 1)
        eqs2 = _derive_tc_meanfield_impl(1, 2)
        eqsM2 = _derive_tc_meanfield_impl(2, 1)
        n1 = length(eqs1.states)
        n2 = length(eqs2.states)
        @test n1 >= 3
        @test n2 > n1
        @test length(eqsM2.states) >= 3
        @test CHIMERA_HAMILTONIAN.interaction == "sum_j g_j * (a * Sp_j + a' * Sm_j)"
        @test qc_available()
        gen = read(joinpath(@__DIR__, "..", "src", "chimera", "eoms", "generated", "tc_equations.txt"), String)
        @test occursin("=== order 2, M=1 ===", gen)
        @test occursin("⟨a' * σ₂₂⟩", gen)
        @test occursin(string(eqs2), gen) || occursin("∂ₜ(⟨a⟩)", gen)
    end
end
