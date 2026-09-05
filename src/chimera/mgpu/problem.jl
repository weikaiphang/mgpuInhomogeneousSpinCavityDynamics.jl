
# Sharding + NCCL context. Not a stepper: OrdinaryDiffEq owns time.
# Device shards hold local cross-block columns and the 3M row-sum buffer.

mutable struct Shard{T}
    id::Int
    dev::CuDevice
    stream::CuStream

    joff::Int
    mloc::Int

    g_b::CuVector{T}
    SpSp::CuVector{Complex{T}}
    SmSp::CuVector{Complex{T}}
    SzSp::CuVector{Complex{T}}

    rowsum::Vector{CuVector{Complex{T}}}
    part::CuVector{Complex{T}}
    ev::CuEvent

    nthreads_cross::Int
    nchunk::Int
    chunk_len::Int
end


mutable struct MGPUProblem{T,F}
    M::Int
    part::EnsemblePartition
    shards::Vector{Shard{T}}
    exec::Executor

    delta0::T
    kappa_e::T
    kappa_i::T
    E_of_t::F
    delta_b::Vector{T}
    g_b::Vector{T}

    hostbuf::Vector{Complex{T}}
    comms::Any
    exchange::Symbol
    nrhs::Base.RefValue{Int}
end

nshards(prob::MGPUProblem) = nshards(prob.part)

function cuda_available()
    try
        return CUDA.functional()
    catch
        return false
    end
end


function launch_config(dev::CuDevice, M::Int, mloc::Int)
    CUDA.device!(dev)
    nsm = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)

    nthreads = 256
    if M < nthreads
        nthreads = max(32, 32 * cld(M, 32))
    end

    want = 8 * nsm
    nchunk = clamp(cld(want, max(mloc, 1)), 1, max(1, cld(M, nthreads)))
    chunk_len = cld(M, nchunk)
    nchunk = cld(M, chunk_len)

    return nthreads, nchunk, chunk_len
end


function build_shards(::Type{T}, M, part, devs, g_b) where {T}
    ns = length(devs)
    shards = Vector{Shard{T}}(undef, ns)

    for p in 1:ns
        dev = devs[p]
        CUDA.device!(dev)

        mloc = part.counts[p]
        joff = part.offsets[p]
        nthreads, nchunk, chunk_len = launch_config(dev, M, mloc)
        nloc = M * mloc

        shards[p] = Shard{T}(
            p, dev, CuStream(),
            joff, mloc,
            CuArray(T.(g_b)),
            CUDA.zeros(Complex{T}, nloc),
            CUDA.zeros(Complex{T}, nloc),
            CUDA.zeros(Complex{T}, nloc),
            [CUDA.zeros(Complex{T}, 3M), CUDA.zeros(Complex{T}, 3M)],
            CUDA.zeros(Complex{T}, 3 * nchunk * mloc),
            CuEvent(CUDA.EVENT_DISABLE_TIMING),
            nthreads, nchunk, chunk_len,
        )
    end

    return shards
end

function free_shards!(shards)
    isempty(shards) && return nothing
    for s in shards
        CUDA.device!(s.dev)
        empty!(s.rowsum)
    end
    GC.gc()
    for d in unique(getfield.(shards, :dev))
        CUDA.device!(d)
        CUDA.reclaim()
    end
    return nothing
end


function exchange_rowsums!(prob::MGPUProblem{T}, rb) where {T}
    mode = prob.exchange
    if mode === :nccl
        exchange_rowsums_nccl!(prob, rb)
    elseif mode === :p2p
        exchange_rowsums_p2p!(prob, rb)
    else
        exchange_rowsums_host!(prob, rb)
    end
    return nothing
end

function exchange_rowsums_nccl!(prob::MGPUProblem{T}, rb) where {T}
    shards = prob.shards
    comms = prob.comms
    NCCL.group() do
        for (s, comm) in zip(shards, comms)
            CUDA.device!(s.dev)
            buf = reinterpret(real(eltype(rb(s))), rb(s))
            NCCL.Allreduce!(buf, +, comm; stream = s.stream)
        end
    end
    return nothing
end

function exchange_rowsums_p2p!(prob::MGPUProblem{T}, rb) where {T}
    shards = prob.shards
    for s in shards
        CUDA.device!(s.dev)
        CUDA.synchronize(s.stream)
    end
    for dst in shards
        CUDA.device!(dst.dev)
        for src in shards
            src.id == dst.id && continue
            off = 3 * src.joff
            copyto!(rb(dst), off + 1, rb(src), off + 1, 3 * src.mloc)
        end
    end
    return nothing
end

function exchange_rowsums_host!(prob::MGPUProblem{T}, rb) where {T}
    shards = prob.shards
    M = prob.M
    host = prob.hostbuf

    for s in shards
        CUDA.device!(s.dev)
        CUDA.synchronize(s.stream)
    end

    for s in shards
        CUDA.device!(s.dev)
        off = 3 * s.joff
        copyto!(host, off + 1, rb(s), off + 1, 3 * s.mloc)
    end
    for s in shards
        CUDA.device!(s.dev)
        dst = rb(s)
        unsafe_copyto!(pointer(dst), pointer(host), 3M;
                       stream = s.stream, async = true)
    end
    return nothing
end


function device_cross_rowsums!(prob::MGPUProblem{T}, u) where {T}
    shards = prob.shards
    M = prob.M
    par = 0
    rb(s) = s.rowsum[par + 1]

    each_shard(shards, prob.exec) do s
        scatter_cross_columns!(s, u, M)
        st = s.stream
        @cuda threads=256 blocks=1 stream=st fill_kernel!(
            rb(s), zero(eltype(rb(s))), length(rb(s)))
        @cuda threads=s.nthreads_cross blocks=(s.nchunk, s.mloc) stream=st rowsum_partial_kernel!(
            s.part, s.SpSp, s.SmSp, s.SzSp, s.g_b,
            M, s.mloc, s.joff, s.chunk_len, s.nchunk)
        @cuda threads=256 blocks=cld(s.mloc, 256) stream=st rowsum_finalize_kernel!(
            rb(s), s.part, s.mloc, s.nchunk, s.joff)
    end

    length(shards) > 1 && exchange_rowsums!(prob, rb)

    s1 = shards[1]
    CUDA.device!(s1.dev)
    CUDA.synchronize(s1.stream)
    host = Vector{Complex{T}}(undef, 3M)
    copyto!(host, rb(s1))
    return unpack_rowsums_3M(host, M)
end


# SciML signature. One VF: injected NCCL/P2P/host row-sums into rhs_2nd_order!.
function rhs_2nd_order_mgpu!(du, u, prob::MGPUProblem, t)
    prob.nrhs[] += 1
    M = prob.M
    C = eltype(u)
    mask = C.(.!Matrix{Bool}(I, M, M))
    p = (prob.delta0, prob.kappa_e, prob.kappa_i, prob.delta_b, prob.g_b,
         M, mask, prob.E_of_t)
    if nshards(prob) == 1
        if u isa CuArray
            CUDA.device!(first(CUDA.devices()))
            p = (prob.delta0, prob.kappa_e, prob.kappa_i,
                 CuArray(prob.delta_b), CuArray(prob.g_b), M,
                 CuArray(mask), prob.E_of_t)
        end
        rhs_2nd_order!(du, u, p, t)
        return nothing
    end
    if isempty(prob.shards) || !cuda_available()
        mode = prob.exchange
        mode === :nccl || mode === :p2p || mode === :host || (mode = :host)
        rhs_2nd_order_sharded!(du, u, p, t, prob.part, mode)
        return nothing
    end
    rsP, rsM, rsZ = device_cross_rowsums!(prob, u)
    if u isa CuArray
        CUDA.device!(prob.shards[1].dev)
        rsP, rsM, rsZ = CuArray(C.(rsP)), CuArray(C.(rsM)), CuArray(C.(rsZ))
        mask = CuArray(mask)
        p = (prob.delta0, prob.kappa_e, prob.kappa_i,
             CuArray(prob.delta_b), CuArray(prob.g_b), M, mask, prob.E_of_t)
    else
        rsP, rsM, rsZ = C.(rsP), C.(rsM), C.(rsZ)
    end
    rhs_2nd_order!(du, u, p, t, (rsP, rsM, rsZ))
    return nothing
end
