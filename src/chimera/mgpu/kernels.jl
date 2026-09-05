
# Device helpers for the 3M cross row-sum only.
# The homemade Tsit5/CK45 combine / errnorm / low-storage kernels and the
# fused `cross_rhs_kernel!` / `small_rhs_kernel!` ODE engine are gone.
# OrdinaryDiffEq evaluates `rhs_2nd_order!`; these kernels fill owned
# slots of the Allreduce buffer.

@inline function warp_reduce3(v1::Complex{T}, v2::Complex{T}, v3::Complex{T}) where {T}
    o = 16
    while o > 0
        v1 += CUDA.shfl_down_sync(0xffffffff, v1, o)
        v2 += CUDA.shfl_down_sync(0xffffffff, v2, o)
        v3 += CUDA.shfl_down_sync(0xffffffff, v3, o)
        o >>= 1
    end
    return v1, v2, v3
end

@inline function block_reduce3(sh, v1::Complex{T}, v2::Complex{T}, v3::Complex{T}) where {T}
    tid  = threadIdx().x
    lane = (tid - 1) & 31
    wid  = (tid - 1) >> 5

    v1, v2, v3 = warp_reduce3(v1, v2, v3)

    @inbounds if lane == 0
        sh[3 * wid + 1] = v1
        sh[3 * wid + 2] = v2
        sh[3 * wid + 3] = v3
    end
    sync_threads()

    if tid == 1
        nw = (blockDim().x + 31) >> 5
        @inbounds begin
            s1 = sh[1]
            s2 = sh[2]
            s3 = sh[3]
            for w in 1:(nw - 1)
                s1 += sh[3 * w + 1]
                s2 += sh[3 * w + 2]
                s3 += sh[3 * w + 3]
            end
        end
        return s1, s2, s3
    end
    return v1, v2, v3
end


# Owned-column partials of (SpSp, SmSp, SzSp) .* diag_mask * g.
# Layout: SpSp[k + (jl-1)*M] = SpSp_cross[j, k] for local column j.
function rowsum_partial_kernel!(part, SpSp, SmSp, SzSp, g_b,
                                M::Int, mloc::Int, joff::Int,
                                chunk_len::Int, nchunk::Int)
    CT = eltype(SpSp)
    sh = CuStaticSharedArray(CT, 96)

    tid = threadIdx().x
    jl  = blockIdx().y
    ch  = blockIdx().x
    j = joff + jl

    kstart = (ch - 1) * chunk_len + 1
    kend   = min(ch * chunk_len, M)
    colbase = (jl - 1) * M

    accP = zero(CT)
    accM = zero(CT)
    accZ = zero(CT)

    k = kstart + (tid - 1)
    @inbounds while k <= kend
        if k != j
            gk = g_b[k]
            i = colbase + k
            accP += gk * SpSp[i]
            accM += gk * SmSp[i]
            accZ += gk * SzSp[i]
        end
        k += blockDim().x
    end

    r1, r2, r3 = block_reduce3(sh, accP, accM, accZ)

    @inbounds if tid == 1
        pb = 3 * ((jl - 1) * nchunk + (ch - 1))
        part[pb + 1] = r1
        part[pb + 2] = r2
        part[pb + 3] = r3
    end
    return nothing
end


function rowsum_finalize_kernel!(rowsum, part, mloc::Int, nchunk::Int, joff::Int)
    jl = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    jl > mloc && return nothing

    @inbounds begin
        pb = 3 * (jl - 1) * nchunk
        s1 = part[pb + 1]
        s2 = part[pb + 2]
        s3 = part[pb + 3]
        for c in 1:(nchunk - 1)
            q = pb + 3 * c
            s1 += part[q + 1]
            s2 += part[q + 2]
            s3 += part[q + 3]
        end

        rb = 3 * (joff + jl - 1)
        rowsum[rb + 1] = s1
        rowsum[rb + 2] = s2
        rowsum[rb + 3] = s3
    end
    return nothing
end


function fill_kernel!(x, v, n::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds while i <= n
        x[i] = v
        i += stride
    end
    return nothing
end
