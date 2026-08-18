# ============================================================
# SHARDS, PROBLEM ASSEMBLY AND THE DISTRIBUTED RIGHT-HAND SIDE
# ============================================================

"""
    Shard{T}

Everything that lives on one GPU: the owned slice of the ensemble, the
register set, scratch buffers and the launch configuration tuned for that
device.  All registers are single contiguous vectors holding the replicated
small block followed by the sharded large block.
"""
mutable struct Shard{T}
    id::Int
    dev::CuDevice
    stream::CuStream

    joff::Int          # 0-based global offset of the first owned bin
    mloc::Int          # owned bins
    n::Int             # register length
    nsmall::Int
    lo::Int            # offset of the large block inside a register

    delta_b::CuVector{T}
    g_b::CuVector{T}
    Nj::CuVector{T}

    regs::Vector{CuVector{Complex{T}}}

    rowsum::Vector{CuVector{Complex{T}}}   # double buffered
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
end

"""
    MGPUProblem{T}

Immutable description of one sharded simulation: the shards, the physical
constants, the drive, and the solver tolerances.
"""
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

    # Pinned, permanently-referenced row-sum staging buffer (see
    # exchange_rowsums!). Living for the whole problem's lifetime is what
    # makes the async H2D copy safe: a buffer that was freshly `Vector`-
    # allocated per call could be collected by the GC before CUDA actually
    # finished reading it, since `GC.@preserve` only protects it for the
    # (near-instant) duration of *issuing* the async copy, not until the
    # transfer completes.
    hostbuf::Vector{Complex{T}}
end

nshards(prob::MGPUProblem) = length(prob.shards)


# ------------------------------------------------------------
# Launch configuration
# ------------------------------------------------------------

function launch_config(dev::CuDevice, M::Int, mloc::Int, n::Int)
    CUDA.device!(dev)
    nsm = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)

    nthreads = 256
    if M < nthreads
        nthreads = max(32, 32 * cld(M, 32))
    end

    # split the k range so that the grid keeps every SM busy even when a
    # shard owns few bins
    want = 8 * nsm
    nchunk = clamp(cld(want, max(mloc, 1)), 1, max(1, cld(M, nthreads)))
    chunk_len = cld(M, nchunk)
    nchunk = cld(M, chunk_len)

    nblocks_small = cld(M, 256)
    nblocks_vec   = min(cld(n, 256), 32 * nsm)
    nblocks_norm  = min(cld(n, 256), 8 * nsm)

    return nthreads, nchunk, chunk_len, nblocks_small, nblocks_vec, nblocks_norm
end


# ------------------------------------------------------------
# Construction
# ------------------------------------------------------------

function build_shards(::Type{T}, M, part, devs, delta_b, g_b, Nj, nreg) where {T}
    ns = length(devs)
    shards = Vector{Shard{T}}(undef, ns)

    for p in 1:ns
        dev = devs[p]
        CUDA.device!(dev)

        mloc = part.counts[p]
        joff = part.offsets[p]
        nsmall = small_length(M)
        n = shard_length(M, mloc)

        nthreads, nchunk, chunk_len, nbs, nbv, nbn = launch_config(dev, M, mloc, n)

        regs = [CUDA.zeros(Complex{T}, n) for _ in 1:nreg]

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


# ------------------------------------------------------------
# Right-hand side
#
#   phase 1 (per shard, fully parallel)
#       * global coupling sums from the replicated small block
#       * the fused O(M²) cross kernel, which also produces this shard's
#         row sums
#       * push the row-sum slice to every peer, then record an event
#
#   phase 2 (per shard)
#       * wait on the peers' events
#       * evaluate the small block redundantly, giving bit-identical
#         replicated state on every shard
#
# The only bytes that cross the interconnect are 3 complex numbers per
# ensemble bin.  Everything of size M² stays on its own GPU.
# ------------------------------------------------------------

function rhs!(prob::MGPUProblem{T}, iu::Int, idu::Int, t::Real) where {T}
    shards = prob.shards
    ns = length(shards)
    M = prob.M

    Et = Complex{T}(prob.E_of_t(t))

    par = prob.rowparity[]
    prob.rowparity[] = 1 - par
    rb(s) = s.rowsum[par + 1]

    prob.nrhs[] += 1

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
    end

    ns > 1 && exchange_rowsums!(prob, rb)

    each_shard(shards, prob.exec) do s
        @cuda threads=256 blocks=s.nblocks_small stream=s.stream small_rhs_kernel!(
            s.regs[idu], s.regs[iu], rb(s), s.gsums, s.delta_b, s.g_b,
            Et, prob.delta0, prob.kappa_t, prob.sqrt_ke, M)
    end

    return nothing
end

"""
    exchange_rowsums!(prob, rb)

All-gather the per-shard row-sum slices.  The payload is `3M` complex
numbers, so a host staging buffer is both simple and cheap next to the
O(M²) kernel.

Uses `prob.hostbuf`, a pinned buffer allocated once in [`assemble_problem`](@ref)
and kept alive for the problem's whole lifetime, rather than a fresh
`Vector` per call: the async H2D copy below needs its source memory to
stay valid (and page-locked, for the transfer to be a genuine DMA rather
than a driver-internal staged copy) until CUDA has actually finished
reading it, which a per-call throwaway buffer cannot guarantee.
"""
function exchange_rowsums!(prob::MGPUProblem{T}, rb) where {T}
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


# ------------------------------------------------------------
# Distributed vector operations
# ------------------------------------------------------------

"""
    combine!(prob, iout, ibase, iks, coeffs, dt; n=:full)

`u[iout] = u[ibase] + dt * Σ coeffs[s] * u[iks[s]]` on every shard.  `n` may
be `:save` to restrict the update to the observable prefix, which is what the
dense-output path uses.
"""
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

"""
    error_norm(prob, iuprev, iu, iks, es, dt)

RMS of the embedded error estimate over the *global* state.  Each shard
reduces its own large block; the replicated small block is counted once, on
shard 1, so the norm is exactly what a single-GPU run would produce.
"""
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

"""
    diff_norm(prob, ix, iy, iref)

RMS of `(x - y)` (or of `x` alone when `iy == 0`) in the tolerance-weighted
norm.  Used by the low-storage stepper and the initial step-size heuristic.
"""
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
    # SzSpT is block 3 of the large region
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
