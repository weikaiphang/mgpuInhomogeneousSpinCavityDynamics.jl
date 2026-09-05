# Package-owned Tavis–Cummings / Dicke Hamiltonian.
# QuantumCumulants.jl (Plankensteiner et al., Quantum 6, 617 (2022)) and
# SecondQuantizedAlgebra derive the cumulant EOMs from this operator.
#
# Convention (matches the 1st- and 2nd-order closures and the QRT Jacobian):
#
#   H = Δ₀ a†a + Σⱼ δⱼ Sᶻⱼ + Σⱼ gⱼ (a S⁺ⱼ + a† S⁻ⱼ)
#     + i √κₑ (E(t) a† − E(t)* a)
#
#   Ṡ = −i [S, H] + κₜ 𝒟[a] S
#   Sᶻ = ±1/2 per spin; bin operators scale with Nⱼ
#   κₜ = κₑ + κᵢ
#
# QuantumCumulants uses Transition σ(2,1)=S⁺, σ(1,2)=S⁻, σ(2,2)=Sᶻ+1/2
# on NLevelSpace(:atom, 2) with level 1 = ground.

const CHIMERA_HAMILTONIAN = (
    name = :driven_lossy_tavis_cummings,
    cavity = "Δ0 * a' * a",
    spins = "sum_j δ_j * Sz_j",
    interaction = "sum_j g_j * (a * Sp_j + a' * Sm_j)",
    drive = "im * sqrt(κe) * (E * a' - conj(E) * a)",
    jump = "sqrt(κt) * a",
    spin_convention = "Sz = ±1/2 per spin",
)

function chimera_hamiltonian_symbols()
    return CHIMERA_HAMILTONIAN
end
