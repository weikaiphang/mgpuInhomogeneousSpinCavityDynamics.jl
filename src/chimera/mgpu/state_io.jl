

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
                lin = (k - 1) * M + j
                host[col + k]           = u_full[oP  + lin]
                host[bs + col + k]      = u_full[oZ  + lin]
                host[2bs + col + k]     = u_full[oZ  + (j - 1) * M + k]
                host[3bs + col + k]     = u_full[oM  + lin]
                host[4bs + col + k]     = u_full[oZZ + lin]
            end
        end











        CUDA.stream!(s.stream) do
            CUDA.fill!(u, zero(Complex{T}))
            copyto!(u, 1, small, 1, nsmall)
            copyto!(u, lo + 1, host, 1, length(host))
        end
        CUDA.synchronize(s.stream)
    end
    return nothing
end


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
