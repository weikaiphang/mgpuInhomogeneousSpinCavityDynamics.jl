# ============================================================
# INITIAL CONDITIONS
#
# The small block is tiny, so it is built on the host and broadcast to every
# shard.  The O(M²) block is written directly on the device: the only
# non-zero entry is <Sz_j Sz_k> = Nj_j Nj_k / 4 off the diagonal, so no host
# buffer of that size is ever allocated.
# ============================================================

"""
    small_block_initial(M, Nj, kind, T)

Host vector of length `3 + 9M` holding the replicated part of the initial
state for `kind ∈ (:ground, :inverted, :custom)`.
"""
function small_block_initial(M::Int, Nj::AbstractVector, kind::Symbol, ::Type{T}) where {T}
    u = zeros(Complex{T}, small_length(M))

    Sz0 = if kind === :ground
        -Nj ./ 2
    elseif kind === :inverted
        Nj ./ 2
    elseif kind === :custom
        zeros(length(Nj))
    else
        error("Unknown initial_condition = $(kind). Use :ground, :inverted or :custom.")
    end

    SzSz0 = kind === :custom ? zeros(length(Nj)) : (Nj .^ 2) ./ 4

    u[small_range(M, F_Sz)]     .= Complex{T}.(Sz0)
    u[small_range(M, F_SzSz_s)] .= Complex{T}.(SzSz0)

    return u
end

"""
    set_initial_condition!(prob, Nj, kind)

Write the initial state into register 1 on every shard.
"""
function set_initial_condition!(prob::MGPUProblem{T}, Nj::AbstractVector,
                                kind::Symbol) where {T}
    M = prob.M
    small = small_block_initial(M, Nj, kind, T)
    fill_diag = kind !== :custom

    each_shard(prob.shards, prob.exec) do s
        u = s.regs[1]
        # See scatter_state!'s comment: fill!/copyto! must be pinned to
        # s.stream, the same stream init_szsz_cross_kernel! below runs on --
        # otherwise fill! (on the untargeted default stream) can race with
        # (and even overwrite, since it zeros the whole register) the
        # kernel's writes.
        CUDA.stream!(s.stream) do
            CUDA.fill!(u, zero(Complex{T}))
            copyto!(u, 1, small, 1, length(small))
        end

        if fill_diag
            n = M * s.mloc
            @cuda threads=256 blocks=min(cld(n, 256), 4096) stream=s.stream init_szsz_cross_kernel!(
                u, s.Nj, M, s.mloc, s.joff, s.lo)
        end
        CUDA.synchronize(s.stream)
    end

    return nothing
end
