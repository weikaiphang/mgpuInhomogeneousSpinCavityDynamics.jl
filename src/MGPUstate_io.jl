# ============================================================
# PACK / UNPACK BETWEEN THE ORIGINAL FLAT LAYOUT AND THE SHARDS
#
# The original single-GPU state is
#
#     [small block | SpSp_cross | SzSp_cross | SmSp_cross | SzSz_cross]
#
# with each cross matrix stored column-major, index [j,k] = pair (j,k).
# A shard stores the same pair as column jl, row k of an M × mloc matrix.
# ============================================================

"""
    scatter_state!(prob, ireg, u_full)

Write a host vector in the original layout into register `ireg` of every
shard.  The transpose replica of `SzSp` is filled from `transpose(SzSp)`.
"""
function scatter_state!(prob::MGPUProblem{T}, ireg::Int,
                        u_full::AbstractVector{Complex{T}}) where {T}
    M = prob.M
    nsmall = small_length(M)
    length(u_full) == global_state_length(M) ||
        error("Expected state of length $(global_state_length(M)), got $(length(u_full)).")

    small = u_full[1:nsmall]
    oP  = nsmall
    oZ  = nsmall + M * M
    oM  = nsmall + 2 * M * M
    oZZ = nsmall + 3 * M * M

    each_shard(prob.shards, prob.exec) do s
        u = s.regs[ireg]
        mloc = s.mloc
        joff = s.joff
        lo = s.lo
        bs = M * mloc

        host = Vector{Complex{T}}(undef, 5 * bs)
        @inbounds for jl in 1:mloc
            j = joff + jl
            col = (jl - 1) * M
            for k in 1:M
                lin = (k - 1) * M + j          # original [j,k]
                host[col + k]           = u_full[oP  + lin]
                host[bs + col + k]      = u_full[oZ  + lin]
                host[2bs + col + k]     = u_full[oZ  + (j - 1) * M + k]  # SzSp[k,j]
                host[3bs + col + k]     = u_full[oM  + lin]
                host[4bs + col + k]     = u_full[oZZ + lin]
            end
        end

        # Plain copyto!/fill! target CUDA.jl's task-local *default* stream,
        # not s.stream -- a different stream object than the one every RHS
        # kernel below is launched on.  Without pinning these to s.stream,
        # the synchronize below waits on the wrong stream, and rhs!'s first
        # kernels can start reading u before this upload has landed (a real,
        # observed race under virtual sharding: multiple shards' scatters
        # serialize on the shared default stream while each shard's own
        # kernels run on its own stream, so the race window grows with
        # nshards). CUDA.stream! makes these operations targets s.stream
        # too, so the final synchronize(s.stream) is actually sufficient.
        CUDA.stream!(s.stream) do
            CUDA.fill!(u, zero(Complex{T}))
            copyto!(u, 1, small, 1, nsmall)
            copyto!(u, lo + 1, host, 1, length(host))
        end
        CUDA.synchronize(s.stream)
    end
    return nothing
end

"""
    gather_state(prob, ireg) -> Vector

Reassemble register `ireg` into a host vector in the original layout.  The
transpose replica is not written out.
"""
function gather_state(prob::MGPUProblem{T}, ireg::Int) where {T}
    M = prob.M
    nsmall = small_length(M)
    u_full = zeros(Complex{T}, global_state_length(M))

    s1 = prob.shards[1]
    CUDA.device!(s1.dev)
    CUDA.synchronize(s1.stream)
    copyto!(u_full, 1, s1.regs[ireg], 1, nsmall)

    oP  = nsmall
    oZ  = nsmall + M * M
    oM  = nsmall + 2 * M * M
    oZZ = nsmall + 3 * M * M

    for s in prob.shards
        CUDA.device!(s.dev)
        CUDA.synchronize(s.stream)
        mloc = s.mloc
        joff = s.joff
        bs = M * mloc
        host = Vector{Complex{T}}(undef, 4 * bs)
        # skip the SzSpT block (offset 2bs); copy the other four matrices
        copyto!(host, 1,                      s.regs[ireg], s.lo + 1,           bs)
        copyto!(host, bs + 1,                 s.regs[ireg], s.lo + bs + 1,      bs)
        copyto!(host, 2bs + 1,                s.regs[ireg], s.lo + 3bs + 1,     bs)
        copyto!(host, 3bs + 1,                s.regs[ireg], s.lo + 4bs + 1,     bs)

        @inbounds for jl in 1:mloc
            j = joff + jl
            col = (jl - 1) * M
            for k in 1:M
                lin = (k - 1) * M + j
                u_full[oP  + lin] = host[col + k]
                u_full[oZ  + lin] = host[bs + col + k]
                u_full[oM  + lin] = host[2bs + col + k]
                u_full[oZZ + lin] = host[3bs + col + k]
            end
        end
    end
    return u_full
end
