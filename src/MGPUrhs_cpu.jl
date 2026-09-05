
# CPU reference for the MGPU path. Packing matches the monolith
# (3+9M+4M²): after the replicated small block, four M×M cross blocks
# only (SpSp, SzSp, SmSp, SzSz). No SzSpT — that copy exists only in the
# sharded GPU layout for coalescing.
#
# There is ONE 2nd-order RHS: `rhs_2nd_order!`. This function is a thin
# wrapper so MGPU CPU checks cannot drift from the monolith. Device
# kernels in MGPUkernels.jl remain the fused sharded replica.
function rhs_cpu!(du::AbstractVector{Complex{T}}, u::AbstractVector{Complex{T}},
                  delta0::T, kappa_e::T, kappa_i::T,
                  delta_b::AbstractVector{T}, g_b::AbstractVector{T},
                  Et::Complex{T}) where {T}
    M = length(delta_b)
    mask = Complex{T}.(.!Matrix{Bool}(I, M, M))
    p = (delta0, kappa_e, kappa_i, delta_b, g_b, M, mask, Returns(Et))
    rhs_2nd_order!(du, u, p, zero(T))
    return nothing
end
