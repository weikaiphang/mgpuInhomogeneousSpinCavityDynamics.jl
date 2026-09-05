# Cumulant EOM owner: QuantumCumulants.jl + SecondQuantizedAlgebra.
# Small-M reference: derive → complete → ModelingToolkit/ODESystem → OrdinaryDiffEq.
# Large-M production RHS implements this closure (closure_1st / closure_2nd).

if @isdefined(HAVE_QUANTUMCUMULANTS) && HAVE_QUANTUMCUMULANTS &&
   !isdefined(@__MODULE__, :_derive_tc_meanfield_impl)
    include("quantum_cumulants_impl.jl")
end

function qc_available()
    return isdefined(@__MODULE__, :_derive_tc_meanfield_impl)
end

function _qc_error()
    error("QuantumCumulants.jl is required for chimera EOM derivation. " *
          "Add it with Pkg.add(\"QuantumCumulants\") and instantiate the project.")
end

"""
    derive_tc_meanfield(; M=1, order=1)

Build the driven lossy Tavis–Cummings Hamiltonian in QuantumCumulants and
return the completed mean-field equations at the requested cumulant order.

`M` is the number of inhomogeneous spins (each Nⱼ = 1). Large-M symbolic
expansion is intentionally not attempted here; the structured closures are
the backend for that limit.
"""
function derive_tc_meanfield(; M::Integer=1, order::Integer=1)
    qc_available() || _qc_error()
    return _derive_tc_meanfield_impl(Int(M), Int(order))
end

function qc_equation_count(eqs)
    return length(eqs.states)
end

function qc_to_odesystem(eqs; name=:chimera_qc)
    qc_available() || _qc_error()
    if HAVE_MODELINGTOOLKIT
        return ModelingToolkit.ODESystem(eqs; name=name)
    end
    return try
        QuantumCumulants.ODESystem(eqs; name=name)
    catch
        System(eqs; name=name)
    end
end

"""
    qc_rhs_1st_M1!(du, u, p, t)

Evaluate the QuantumCumulants 1st-order RHS for one spin, mapped onto the
chimera 1st-order packing `[⟨a⟩, ⟨S⁺⟩, ⟨Sᶻ⟩]` with `⟨σ22⟩ = ⟨Sᶻ⟩ + 1/2`.

`p = (Δ0, κt, δ, g, η)` with real drive `η = √κₑ E`.
"""
function qc_rhs_1st_M1!(du, u, p, t)
    Δ0, κt, δ, g, η = p
    a = u[1]
    Sp = u[2]
    Sz = u[3]
    σ22 = Sz + 0.5
    # Heisenberg / Lindblad of H in hamiltonian.jl, order-1 factorization.
    # Kept as the numeric image of derive_tc_meanfield(; M=1, order=1).
    du[1] = η - 1im * Δ0 * a - 1im * g * conj(Sp) - 0.5 * κt * a
    du[2] = 1im * δ * Sp - 2im * g * conj(a) * Sz
    du[3] = -1im * g * a * Sp + 1im * g * conj(a) * conj(Sp)
    return nothing
end

function qc_verify_1st_against_closure(eqs=nothing)
    qc_available() || _qc_error()
    if eqs === nothing
        eqs = derive_tc_meanfield(; M=1, order=1)
    end
    qc_equation_count(eqs) >= 1 || error("QuantumCumulants returned an empty equation set")
    return eqs
end

"""
    compare_qc_to_closure_1st(; kwargs...) -> (maxabs, du_qc, du_cl)

Evaluate the chimera 1st-order closure against the QC-mapped 1st-order RHS
for M=1 (Nⱼ unused). Both implement the same Hamiltonian.
"""
function compare_qc_to_closure_1st(;
        a=0.2 + 0.1im, Sp=0.05 + 0.02im, Sz=-0.4,
        Δ0=0.3, κe=0.4, κi=0.1, δ=0.25, g=0.15, E=0.08)
    u = ComplexF64[a, Sp, Sz]
    η = sqrt(κe) * E
    du_qc = zero(u)
    qc_rhs_1st_M1!(du_qc, u, (Δ0, κe + κi, δ, g, η), 0.0)
    du_cl = zero(u)
    p = (Δ0, κe, κi, [δ], [g], 1, t -> ComplexF64(E))
    rhs_1st_order!(du_cl, u, p, 0.0)
    return maximum(abs.(du_qc .- du_cl)), du_qc, du_cl
end

function compare_qc_to_closure_1st_M(M::Integer;
        Δ0=0.0, κe=2π * 1e6, κi=0.0, E=0.1)
    M >= 1 || error("M must be >= 1")
    delta = collect(range(-0.2, 0.2; length=M))
    g = fill(0.15, M)
    u = zeros(ComplexF64, 1 + 2M)
    u[1] = 0.05 + 0.02im
    u[2:1+M] .= 0.01 .+ 0.002im .* (1:M)
    u[2+M:1+2M] .= -0.5
    du_cl = zero(u)
    rhs_1st_order!(du_cl, u, (Δ0, κe, κi, delta, g, M, t -> ComplexF64(E)), 0.0)
    du_qc = zero(u)
    η = sqrt(κe) * E
    a = u[1]
    Sp = @view u[2:1+M]
    Sz = @view u[2+M:1+2M]
    du_qc[1] = η - 1im * Δ0 * a - 1im * sum(g .* conj.(Sp)) - 0.5 * (κe + κi) * a
    du_qc[2:1+M] .= 1im .* delta .* Sp .- 2im .* g .* conj(a) .* Sz
    du_qc[2+M:1+2M] .= -1im .* g .* a .* Sp .+ 1im .* g .* conj(a) .* conj.(Sp)
    return maximum(abs.(du_qc .- du_cl)), du_qc, du_cl
end

function solve_qc_sciml_1st_M1(;
        Δ0=0.0, κe=0.2, κi=0.0, δ=0.1, g=0.05, E=0.02,
        tspan=(0.0, 2.0), u0=ComplexF64[0.0, 0.0, -0.5],
        reltol=1e-8, abstol=1e-8)
    η = sqrt(κe) * real(E)
    p_qc = (Δ0, κe + κi, δ, g, η)
    p_cl = (Δ0, κe, κi, [δ], [g], 1, t -> ComplexF64(E))
    prob_qc = chimera_ode_problem(qc_rhs_1st_M1!, copy(u0), tspan, p_qc)
    prob_cl = chimera_ode_problem(rhs_1st_order!, copy(u0), tspan, p_cl)
    sol_qc = solve(prob_qc, OrdinaryDiffEq.Tsit5(); reltol=reltol, abstol=abstol, save_everystep=false)
    sol_cl = solve(prob_cl, OrdinaryDiffEq.Tsit5(); reltol=reltol, abstol=abstol, save_everystep=false)
    return sol_qc, sol_cl
end

function qc_algebra_selftest()
    qc_available() || _qc_error()
    eqs1 = derive_tc_meanfield(; M=1, order=1)
    n1 = qc_equation_count(eqs1)
    n1 >= 3 || error("expected at least ⟨a⟩, ⟨σ21⟩, ⟨σ22⟩ from QuantumCumulants order-1")
    eqs2 = derive_tc_meanfield(; M=1, order=2)
    n2 = qc_equation_count(eqs2)
    n2 > n1 || error("order-2 QuantumCumulants set must be larger than order-1")
    return (order1=n1, order2=n2, eqs1=eqs1, eqs2=eqs2)
end

# Numeric image of the checked-in QuantumCumulants order-2 M=1 set
# (`eoms/generated/tc_equations.txt`). Packing is the QC state
#   [⟨a⟩, ⟨σ21⟩, ⟨σ22⟩, ⟨a'σ22⟩, ⟨a σ21⟩, ⟨a'a⟩, ⟨a'σ21⟩, ⟨a'a'⟩]
# with ⟨a'⟩ = conj(⟨a⟩) and ⟨σ12⟩ = conj(⟨σ21⟩). This is the derivation
# reference, not a runtime backend.
function qc_rhs_2nd_M1_image!(du, u, p, t)
    Δ0, κt, δ, g, η = p
    a, σ21, σ22, adσ22, aσ21, ada, adσ21, adad = u
    ca = conj(a)
    σ12 = conj(σ21)
    adσ12 = conj(aσ21)
    aσ22 = conj(adσ22)
    du[1] = η - 1im * g * σ12 + a * (-0.5 * κt - 1im * Δ0)
    du[2] = 1im * g * ca - 2im * g * adσ22 + 1im * δ * σ21
    du[3] = 1im * g * adσ12 - 1im * g * aσ21
    du[4] = σ22 * η + adσ22 * (-0.5 * κt + 1im * Δ0) +
            1im * g * (σ12 * adad + 2 * ca * adσ12 - 2 * σ12 * ca^2) -
            1im * g * (a * adσ21 + ca * aσ21 + ada * σ21 - 2 * ca * σ21 * a)
    du[5] = σ21 * η + 1im * g * ada - 1im * g * σ22 +
            aσ21 * (-0.5 * κt + 1im * (-Δ0 + δ)) -
            2im * g * (a * adσ22 + ca * aσ22 + ada * σ22 - 2 * ca * σ22 * a)
    du[6] = a * η + ca * η - ada * κt + 1im * g * aσ21 - 1im * g * adσ12
    du[7] = σ21 * η + 1im * g * adad + adσ21 * (-0.5 * κt + 1im * (Δ0 + δ)) -
            2im * g * (σ22 * adad + 2 * ca * adσ22 - 2 * σ22 * ca^2)
    du[8] = 2 * ca * η + 2im * g * adσ21 + adad * (-κt + 2im * Δ0)
    return nothing
end

function chimera_u_to_qc_2nd_M1(u)
    a = u[1]
    ad_ad = u[2]
    ad_a = u[3]
    Sp = u[4]
    Sz = u[5]
    adSp = u[6]
    adSm = u[7]
    adSz = u[8]
    return ComplexF64[
        a,
        Sp,
        Sz + 0.5,
        adSz + 0.5 * conj(a),
        conj(adSm),
        ad_a,
        adSp,
        ad_ad,
    ]
end

function chimera_du_to_qc_2nd_M1(du, u)
    a = u[1]
    da = du[1]
    return ComplexF64[
        da,
        du[4],
        du[5],
        du[8] + 0.5 * conj(da),
        conj(du[7]),
        du[3],
        du[6],
        du[2],
    ]
end

# QC image indices that map onto chimera without a correlator-basis change:
# ⟨a⟩, ⟨σ21⟩, ⟨σ22⟩=Sz+1/2, ⟨a'a⟩, ⟨a'σ21⟩, ⟨a'a'⟩.
# ⟨a'σ22⟩ and ⟨a σ21⟩ need Sz↔σ22 / Hermitian pairing; that naive map is
# not the production backend (see compare_qc_to_closure_2nd_M1).
const QC_2ND_OVERLAP = (1, 2, 3, 6, 7, 8)
const QC_2ND_SIGMA22_SHIFT = (4, 5)

"""
    compare_qc_to_closure_2nd_M1(; kwargs...) -> NamedTuple

Map a chimera M=1 2nd-order state onto the QC order-2 M=1 coordinates and
compare the generated-equation image to `rhs_2nd_order!`. Drive `E` is real
so `η = √κₑ E` matches the QC derivation.

`overlap` is the maxabs error on the shared sector. `shift_residual` is the
maxabs error on the naive ⟨a'σ22⟩ / ⟨a σ21⟩ map — expected nonzero.
"""
function compare_qc_to_closure_2nd_M1(;
        a=0.2 + 0.1im, Sp=0.05 + 0.02im, Sz=-0.4,
        ad_ad=0.03 + 0.01im, ad_a=0.08 + 0.0im,
        adSp=0.02 - 0.01im, adSm=0.01 + 0.03im, adSz=-0.05 + 0.02im,
        Δ0=0.3, κe=0.4, κi=0.1, δ=0.25, g=0.15, E=0.08)
    u = zeros(ComplexF64, state_length_2nd_order(1))
    u[1] = a
    u[2] = ad_ad
    u[3] = ad_a
    u[4] = Sp
    u[5] = Sz
    u[6] = adSp
    u[7] = adSm
    u[8] = adSz
    η = sqrt(κe) * real(E)
    u_qc = chimera_u_to_qc_2nd_M1(u)
    du_qc = zero(u_qc)
    qc_rhs_2nd_M1_image!(du_qc, u_qc, (Δ0, κe + κi, δ, g, η), 0.0)
    du_cl = zero(u)
    mask = reshape(ComplexF64[0], 1, 1)
    p = (Δ0, κe, κi, [δ], [g], 1, mask, Returns(ComplexF64(real(E))))
    rhs_2nd_order!(du_cl, u, p, 0.0)
    du_from_cl = chimera_du_to_qc_2nd_M1(du_cl, u)
    overlap = maximum(abs(du_qc[i] - du_from_cl[i]) for i in QC_2ND_OVERLAP)
    shift_residual = maximum(abs(du_qc[i] - du_from_cl[i]) for i in QC_2ND_SIGMA22_SHIFT)
    return (; overlap, shift_residual, du_qc, du_from_cl)
end
