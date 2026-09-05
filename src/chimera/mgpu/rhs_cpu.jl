
# CPU / SciML stand-in for the multi-GPU path.
#
# There is ONE 2nd-order vector field: `rhs_2nd_order!`. This file
# (1) wraps it as `rhs_cpu!` so callers cannot drift, and
# (2) evaluates the same VF after assembling the 3M cross row-sums
#     from an `EnsemblePartition` (`rhs_2nd_order_sharded!`).
# Device NCCL/P2P/host Allreduce is the same 3M buffer contract.

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

"""
    fill_local_cross_rowsums!(rowsum, u, g, M, jrange)

Owned slots of the 3M interleaved buffer:
    rowsum[3(j-1)+1] = Σ_{k≠j} SpSp_cross[j,k] * g[k]
    rowsum[3(j-1)+2] = Σ_{k≠j} SmSp_cross[j,k] * g[k]
    rowsum[3(j-1)+3] = Σ_{k≠j} SzSp_cross[j,k] * g[k]
Non-owned slots are left at zero so NCCL Allreduce is a true sum.
"""
function fill_local_cross_rowsums!(rowsum::AbstractVector{C},
                                   u::AbstractVector{C},
                                   g::AbstractVector,
                                   M::Integer,
                                   jrange) where {C<:Complex}
    fill!(rowsum, zero(C))
    st = unpack_state_2nd_order_u(u, M)
    SpSp_cross = st[13]
    SzSp_cross = st[14]
    SmSp_cross = st[15]
    @inbounds for j in jrange
        accP = zero(C)
        accM = zero(C)
        accZ = zero(C)
        for k in 1:M
            k == j && continue
            gk = g[k]
            accP += SpSp_cross[j, k] * gk
            accM += SmSp_cross[j, k] * gk
            accZ += SzSp_cross[j, k] * gk
        end
        rb = 3 * (j - 1)
        rowsum[rb + 1] = accP
        rowsum[rb + 2] = accM
        rowsum[rb + 3] = accZ
    end
    return rowsum
end

function unpack_rowsums_3M(rs::AbstractVector{C}, M::Integer) where {C}
    rsP = similar(rs, M)
    rsM = similar(rs, M)
    rsZ = similar(rs, M)
    @inbounds for j in 1:M
        rb = 3 * (j - 1)
        rsP[j] = rs[rb + 1]
        rsM[j] = rs[rb + 2]
        rsZ[j] = rs[rb + 3]
    end
    return rsP, rsM, rsZ
end

"""
    rhs_2nd_order_sharded!(du, u, p, t, part, mode)

Same VF as `rhs_2nd_order!`. The 3M cross row-sums are computed per
shard and assembled with `assemble_rowsums!` (`:nccl` / `:p2p` / `:host`
CPU mirrors, or `:none` for a single shard). OrdinaryDiffEq calls this
(or `rhs_2nd_order_mgpu!`) — there is no homemade Runge–Kutta.
"""
function rhs_2nd_order_sharded!(du, u, p, t,
                               part::EnsemblePartition,
                               mode::Symbol)
    M = part.M
    g = p[5]
    ns = nshards(part)
    C = eltype(u)
    bufs = [zeros(C, 3M) for _ in 1:ns]
    for i in 1:ns
        fill_local_cross_rowsums!(bufs[i], u, g, M, part[i])
    end
    assemble_rowsums!(mode, bufs, part)
    rsP, rsM, rsZ = unpack_rowsums_3M(bufs[1], M)
    rhs_2nd_order!(du, u, p, t, (rsP, rsM, rsZ))
    return nothing
end

function rhs_2nd_order_sharded!(du, u, p, t; nshards::Integer = 1,
                               exchange::Symbol = :none)
    M = p[6]
    part = EnsemblePartition(M, nshards)
    mode = nshards == 1 ? :none : exchange
    return rhs_2nd_order_sharded!(du, u, p, t, part, mode)
end
