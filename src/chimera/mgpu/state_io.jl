
# Scatter owned columns of the three cross blocks that enter the 3M
# row-sum. OrdinaryDiffEq keeps the assembled monolith state.

function scatter_cross_columns!(s::Shard{T}, u_full::AbstractVector, M::Int) where {T}
    nsmall = small_length(M)
    oP = nsmall
    oZ = nsmall + M * M
    oM = nsmall + 2 * M * M
    mloc = s.mloc
    joff = s.joff
    hostP = Vector{Complex{T}}(undef, M * mloc)
    hostM = Vector{Complex{T}}(undef, M * mloc)
    hostZ = Vector{Complex{T}}(undef, M * mloc)
    uhost = u_full isa CuArray ? Array(u_full) : u_full

    @inbounds for jl in 1:mloc
        j = joff + jl
        col = (jl - 1) * M
        for k in 1:M
            lin = (k - 1) * M + j
            hostP[col + k] = Complex{T}(uhost[oP + lin])
            hostZ[col + k] = Complex{T}(uhost[oZ + lin])
            hostM[col + k] = Complex{T}(uhost[oM + lin])
        end
    end

    CUDA.device!(s.dev)
    CUDA.stream!(s.stream) do
        copyto!(s.SpSp, hostP)
        copyto!(s.SzSp, hostZ)
        copyto!(s.SmSp, hostM)
    end
    return nothing
end


function scatter_state!(prob::MGPUProblem{T}, u_full::AbstractVector{Complex{T}}) where {T}
    M = prob.M
    length(u_full) == global_state_length(M) ||
        error("Expected state of length $(global_state_length(M)), got $(length(u_full)).")
    isempty(prob.shards) && return nothing
    each_shard(prob.shards, prob.exec) do s
        scatter_cross_columns!(s, u_full, M)
        CUDA.synchronize(s.stream)
    end
    return nothing
end

# Kept for API compatibility: the assembled VF *is* the monolith state.
gather_state(::MGPUProblem{T}, u_full::AbstractVector{Complex{T}}) where {T} = u_full
