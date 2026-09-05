

mutable struct Shard{T}
    id::Int
    dev::CuDevice
    stream::CuStream

    joff::Int
    mloc::Int
    n::Int
    nsmall::Int
    lo::Int

    delta_b::CuVector{T}
    g_b::CuVector{T}
    Nj::CuVector{T}

    regs::Vector{CuVector{Complex{T}}}

    rowsum::Vector{CuVector{Complex{T}}}
    part::CuVector{Complex{T}}
    gsums::CuVector{Complex{T}}
    normpart::CuVector{T}
    normout::CuVector{T}
    ev::CuEvent

    nthreads_cross::Int
    nchunk::Int
    chunk_len::Int
    nblocks_small::Int
    nblocks_vec::Int
    nblocks_norm::Int

    nccl_send::Union{Nothing,CuVector{Complex{T}}}
    nccl_recv::Union{Nothing,CuVector{Complex{T}}}
end


struct MGPUProblem{T,F}
    M::Int
    part::EnsemblePartition
    shards::Vector{Shard{T}}
    exec::Executor
    nreg::Int

    delta0::T
    kappa_e::T
    kappa_t::T
    sqrt_ke::T
    E_of_t::F

    atol::T
    rtol::T

    rowparity::Base.RefValue{Int}
    nrhs::Base.RefValue{Int}








    hostbuf::Vector{Complex{T}}
    comms::Any
    peer_ok::Bool
end

nshards(prob::MGPUProblem) = length(prob.shards)



function launch_config(dev::CuDevice, M::Int, mloc::Int, n::Int)
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

    nblocks_small = cld(M, 256)
    nblocks_vec   = min(cld(n, 256), 32 * nsm)
    nblocks_norm  = min(cld(n, 256), 8 * nsm)

    return nthreads, nchunk, chunk_len, nblocks_small, nblocks_vec, nblocks_norm
end



function build_shards(::Type{T}, M, part, devs, delta_b, g_b, Nj, nreg) where {T}
    ns = length(devs)
    shards = Vector{Shard{T}}(undef, ns)
    maxmloc = maximum(part.counts)
    equal = all(==(part.counts[1]), part.counts)
    need_pad = ns > 1 && !equal

    for p in 1:ns
        dev = devs[p]
        CUDA.device!(dev)

        mloc = part.counts[p]
        joff = part.offsets[p]
        nsmall = small_length(M)
        n = shard_length(M, mloc)

        nthreads, nchunk, chunk_len, nbs, nbv, nbn = launch_config(dev, M, mloc, n)

        regs = [CUDA.zeros(Complex{T}, n) for _ in 1:nreg]
        nccl_send = need_pad ? CUDA.zeros(Complex{T}, 3 * maxmloc) : nothing
        nccl_recv = need_pad ? CUDA.zeros(Complex{T}, 3 * maxmloc * ns) : nothing

        shards[p] = Shard{T}(
            p, dev, CuStream(),
            joff, mloc, n, nsmall, nsmall,
            CuArray(T.(delta_b)), CuArray(T.(g_b)), CuArray(T.(Nj)),
            regs,
            [CUDA.zeros(Complex{T}, 3M), CUDA.zeros(Complex{T}, 3M)],
            CUDA.zeros(Complex{T}, 3 * nchunk * mloc),
            CUDA.zeros(Complex{T}, 3),
            CUDA.zeros(T, nbn),
            CUDA.zeros(T, 1),
            CuEvent(CUDA.EVENT_DISABLE_TIMING),
            nthreads, nchunk, chunk_len, nbs, nbv, nbn,
            nccl_send, nccl_recv,
        )
    end

    return shards
end

function free_shards!(shards)
    for s in shards
        CUDA.device!(s.dev)
        empty!(s.regs)
        empty!(s.rowsum)
    end
    GC.gc()
    for d in unique(getfield.(shards, :dev))
        CUDA.device!(d)
        CUDA.reclaim()
    end
    return nothing
end



function rhs!(prob::MGPUProblem{T}, iu::Int, idu::Int, t::Real) where {T}
    shards = prob.shards
    ns = length(shards)
    M = prob.M

    Et = Complex{T}(prob.E_of_t(t))

    par = prob.rowparity[]
    prob.rowparity[] = 1 - par
    rb(s) = s.rowsum[par + 1]

    prob.nrhs[] += 1

    # Small-state RHS and the three global cavity sums are O(M) and run on
    # every shard. The small block is replicated in each shard register so
    # the next RK stage can read it without an O(M) broadcast of 3+9M
    # complexes, which is more expensive than the local kernels. Cross
    # terms remain sharded; only the O(M) row-sum vector is exchanged.
    each_shard(shards, prob.exec) do s
        u  = s.regs[iu]
        du = s.regs[idu]
        st = s.stream

        @cuda threads=256 blocks=1 stream=st global_sums_kernel!(
            s.gsums, u, s.g_b, M)

        @cuda threads=s.nthreads_cross blocks=(s.nchunk, s.mloc) stream=st cross_rhs_kernel!(
            du, u, s.part, s.delta_b, s.g_b,
            M, s.mloc, s.joff, s.lo, s.chunk_len, s.nchunk)

        @cuda threads=256 blocks=cld(s.mloc, 256) stream=st rowsum_finalize_kernel!(
            rb(s), s.part, s.mloc, s.nchunk, s.joff)
        CUDA.record(s.ev, st)
    end

    ns > 1 && exchange_rowsums!(prob, rb)

    each_shard(shards, prob.exec) do s
        @cuda threads=256 blocks=s.nblocks_small stream=s.stream small_rhs_kernel!(
            s.regs[idu], s.regs[iu], rb(s), s.gsums, s.delta_b, s.g_b,
            Et, prob.delta0, prob.kappa_t, prob.sqrt_ke, M)
    end

    return nothing
end


# 3M packed row-sums (∑_k g_k ⟨S⁺S⁺⟩_{jk}, ⟨S⁻S⁺⟩_{jk}, ⟨SᶻS⁺⟩_{jk}).
# Prefer this over the standalone opt script's 5-field gather.
function exchange_rowsums!(prob::MGPUProblem{T}, rb) where {T}
    shards = prob.shards
    length(shards) <= 1 && return nothing

    if _nccl_allgather_rowsums!(prob, rb)
        return nothing
    elseif prob.peer_ok
        _p2p_allgather_rowsums!(prob, rb)
        return nothing
    end

    @warn "exchange_rowsums!: NCCL and P2P unavailable; staging 3M row-sums through the HOST" maxlog=1
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

function _nccl_allgather_rowsums!(prob::MGPUProblem{T}, rb) where {T}
    comms = prob.comms
    comms === nothing && return false
    shards = prob.shards
    try
        for s in shards
            CUDA.device!(s.dev)
            CUDA.synchronize(s.stream)
        end
        equal = all(s -> s.mloc == shards[1].mloc, shards)
        _NCCL.group() do
            for (s, comm) in zip(shards, comms)
                CUDA.device!(s.dev)
                buf = rb(s)
                if equal
                    nloc = 3 * s.mloc
                    send = view(buf, 3 * s.joff + 1 : 3 * s.joff + nloc)
                    _NCCL.Allgather!(send, buf, comm)
                else
                    s.nccl_send === nothing && error("padded NCCL buffers missing")
                    nloc = 3 * s.mloc
                    copyto!(s.nccl_send, 1, buf, 3 * s.joff + 1, nloc)
                    _NCCL.Allgather!(s.nccl_send, s.nccl_recv, comm)
                end
            end
        end
        if !equal
            maxmloc = maximum(s -> s.mloc, shards)
            stride = 3 * maxmloc
            for dst in shards
                CUDA.device!(dst.dev)
                dbuf = rb(dst)
                for src in shards
                    off = 3 * src.joff
                    nloc = 3 * src.mloc
                    copyto!(dbuf, off + 1, dst.nccl_recv, (src.id - 1) * stride + 1, nloc)
                end
            end
        end
        return true
    catch err
        @debug "NCCL row-sum Allgather failed; trying P2P/host" err
        return false
    end
end

function _p2p_allgather_rowsums!(prob::MGPUProblem{T}, rb) where {T}
    shards = prob.shards
    for dst in shards
        CUDA.device!(dst.dev)
        dbuf = rb(dst)
        for src in shards
            src.id == dst.id && continue
            CUDA.wait(dst.stream, src.ev)
            off = 3 * src.joff
            n = 3 * src.mloc
            n == 0 && continue
            sbuf = rb(src)
            GC.@preserve dbuf sbuf begin
                CUDA.unsafe_copyto!(pointer(dbuf, off + 1), pointer(sbuf, off + 1), n;
                                    async = true, stream = dst.stream)
            end
        end
    end
    return nothing
end




function combine!(prob::MGPUProblem{T}, iout::Int, ibase::Int, iks, coeffs,
                  dt::T; n::Symbol = :full) where {T}
    ks_idx = Tuple(iks)
    cs = Tuple(T.(coeffs))
    each_shard(prob.shards, prob.exec) do s
        len = n === :full ? s.n : save_prefix_length(prob.M)
        ks = map(i -> s.regs[i], ks_idx)
        @cuda threads=256 blocks=s.nblocks_vec stream=s.stream combine_kernel!(
            s.regs[iout], s.regs[ibase], ks, cs, dt, len)
    end
    return nothing
end

function copy_register!(prob::MGPUProblem, idst::Int, isrc::Int)
    each_shard(prob.shards, prob.exec) do s
        copyto!(s.regs[idst], s.regs[isrc])
    end
    return nothing
end

function swap_registers!(prob::MGPUProblem, i::Int, j::Int)
    for s in prob.shards
        s.regs[i], s.regs[j] = s.regs[j], s.regs[i]
    end
    return nothing
end

function fill_register!(prob::MGPUProblem{T}, i::Int, v) where {T}
    val = Complex{T}(v)
    each_shard(prob.shards, prob.exec) do s
        @cuda threads=256 blocks=s.nblocks_vec stream=s.stream fill_kernel!(
            s.regs[i], val, s.n)
    end
    return nothing
end

function sync_shards!(prob::MGPUProblem)
    each_shard(prob.shards, prob.exec) do s
        CUDA.synchronize(s.stream)
    end
    return nothing
end


function error_norm(prob::MGPUProblem{T}, iuprev::Int, iu::Int, iks, es,
                    dt::T) where {T}
    ks_idx = Tuple(iks)
    cs = Tuple(T.(es))

    each_shard(prob.shards, prob.exec) do s
        istart = s.id == 1 ? 1 : s.nsmall + 1
        sa, sb = szspt_range(s)
        ks = map(i -> s.regs[i], ks_idx)
        @cuda threads=256 blocks=s.nblocks_norm stream=s.stream errnorm_kernel!(
            s.normpart, s.regs[iuprev], s.regs[iu], ks, cs, dt,
            prob.atol, prob.rtol, istart, s.n, sa, sb)
        @cuda threads=256 blocks=1 stream=s.stream sum_partials_kernel!(
            s.normout, s.normpart, s.nblocks_norm)
    end

    return gather_norm(prob)
end


function diff_norm(prob::MGPUProblem{T}, ix::Int, iy::Int, iref::Int) where {T}
    each_shard(prob.shards, prob.exec) do s
        istart = s.id == 1 ? 1 : s.nsmall + 1
        sa, sb = szspt_range(s)
        y = iy == 0 ? nothing : s.regs[iy]
        @cuda threads=256 blocks=s.nblocks_norm stream=s.stream diffnorm_kernel!(
            s.normpart, s.regs[ix], y, s.regs[iref],
            prob.atol, prob.rtol, istart, s.n, sa, sb)
        @cuda threads=256 blocks=1 stream=s.stream sum_partials_kernel!(
            s.normout, s.normpart, s.nblocks_norm)
    end
    return gather_norm(prob)
end

function szspt_range(s::Shard)
    bs = s.nsmall == 0 ? 0 : (s.n - s.nsmall) ÷ NBLOCK

    a = s.lo + 2 * bs + 1
    b = s.lo + 3 * bs
    return a, b
end

function gather_norm(prob::MGPUProblem{T}) where {T}
    total = zero(T)
    for s in prob.shards
        CUDA.device!(s.dev)
        CUDA.synchronize(s.stream)
        total += Array(s.normout)[1]
    end
    return sqrt(total / global_state_length(prob.M))
end
