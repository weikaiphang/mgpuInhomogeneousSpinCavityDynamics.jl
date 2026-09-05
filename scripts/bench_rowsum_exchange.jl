# Times O(M) row-sum assembly: host staging vs NCCL Allreduce / P2P.
# Structural argument (always valid): the production RHS no longer copies the
# 3M complex row-sum through a pinned host buffer when NCCL or P2P is used.
# See exchange_rowsums_nccl! / exchange_rowsums_p2p! / exchange_rowsums_host!
# in src/MGPUproblem.jl.

using CUDA
using Printf
using Statistics
try
    using NCCL
catch
end

function _host_roundtrip!(host, shards)
    for (dev, buf, joff, mloc) in shards
        CUDA.device!(dev)
        off = 3 * joff
        copyto!(host, off + 1, buf, off + 1, 3 * mloc)
    end
    for (dev, buf, _, _) in shards
        CUDA.device!(dev)
        copyto!(buf, host)
    end
    return nothing
end

function _sync_all(shards)
    for (dev, _, _, _) in shards
        CUDA.device!(dev)
        CUDA.synchronize()
    end
end

function bench_rowsum(; M::Int = 4096, ns::Int = 0, repeats::Int = 20)
    if !CUDA.functional()
        println("CUDA not functional. Structural result only:")
        println("  NCCL Allreduce / P2P keep the 3M row-sum on device.")
        println("  Host path: ns device→host copies + ns host→device copies per RHS.")
        println("  Device path: one grouped Allreduce, no host touch of the O(M) vector.")
        return nothing
    end

    ndev = length(CUDA.devices())
    ns = ns <= 0 ? max(1, ndev) : ns
    part = let
        base = div(M, ns)
        rem_ = mod(M, ns)
        counts = [base + (p <= rem_ ? 1 : 0) for p in 1:ns]
        offsets = zeros(Int, ns)
        for p in 2:ns
            offsets[p] = offsets[p-1] + counts[p-1]
        end
        (counts, offsets)
    end

    shards = []
    for p in 1:ns
        dev = collect(CUDA.devices())[mod1(p, ndev)]
        CUDA.device!(dev)
        buf = CUDA.rand(ComplexF64, 3M)
        push!(shards, (dev, buf, part[2][p], part[1][p]))
    end
    host = Vector{ComplexF64}(undef, 3M)

    _host_roundtrip!(host, shards)
    _sync_all(shards)
    th = Float64[]
    for _ in 1:repeats
        t0 = time_ns()
        _host_roundtrip!(host, shards)
        _sync_all(shards)
        push!(th, (time_ns() - t0) / 1e6)
    end

    @printf "M = %d  shards = %d  devices = %d\n" M ns ndev
    @printf "host staging   median %.3f ms  (min %.3f)\n" median(th) minimum(th)

    nccl_ok = false
    if ns > 1 && ndev >= 2 && length(unique(s[1] for s in shards)) == ns
        try
            comms = NCCL.Communicators([s[1] for s in shards])
            NCCL.group() do
                for ((dev, buf, _, _), comm) in zip(shards, comms)
                    CUDA.device!(dev)
                    NCCL.Allreduce!(reinterpret(Float64, buf), +, comm)
                end
            end
            _sync_all(shards)
            tn = Float64[]
            for _ in 1:repeats
                t0 = time_ns()
                NCCL.group() do
                    for ((dev, buf, _, _), comm) in zip(shards, comms)
                        CUDA.device!(dev)
                        NCCL.Allreduce!(reinterpret(Float64, buf), +, comm)
                    end
                end
                _sync_all(shards)
                push!(tn, (time_ns() - t0) / 1e6)
            end
            @printf "NCCL Allreduce median %.3f ms  (min %.3f)  speedup %.2fx\n" median(tn) minimum(tn) (median(th) / median(tn))
            nccl_ok = true
        catch err
            println("NCCL path unavailable: ", err)
        end
    else
        println("NCCL multi-GPU comparison skipped (need ≥2 visible devices).")
        bytes = 3M * sizeof(ComplexF64)
        @printf "host path moves ≈ %.2f MiB through the CPU per RHS.\n" (2 * ns * bytes / 2^20)
        println("NCCL Allreduce keeps that buffer on the devices.")
    end
    return nccl_ok
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    bench_rowsum()
end
