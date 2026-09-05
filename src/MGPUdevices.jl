

struct Executor
    threaded::Bool
end

function Executor(ns::Integer; threaded::Union{Nothing,Bool} = nothing)
    th = threaded === nothing ? (ns > 1 && Threads.nthreads() >= ns) : threaded
    if th && Threads.nthreads() < ns
        @warn """
        Threaded executor requested with fewer Julia threads than shards.
        Start Julia with `-t $(ns)` (or more) for one thread per GPU.
        """ nthreads=Threads.nthreads() nshards=ns
    end
    return Executor(th)
end


function each_shard(f::F, shards, exec::Executor) where {F}
    if exec.threaded && length(shards) > 1
        @sync for s in shards
            Threads.@spawn begin
                CUDA.device!(s.dev)
                f(s)
            end
        end
    else
        for s in shards
            CUDA.device!(s.dev)
            f(s)
        end
    end
    return nothing
end


function resolve_devices(ns::Union{Nothing,Integer}, device_ids::Union{Nothing,AbstractVector})
    CUDA.functional() || error("CUDA is not functional; no GPU available.")

    all_devs = collect(CUDA.devices())
    ndev = length(all_devs)
    ndev >= 1 || error("No CUDA devices found.")

    if device_ids === nothing
        n = ns === nothing ? ndev : Int(ns)
        n >= 1 || error("nshards must be positive.")
        if n > ndev
            @warn """
            Requested more shards than visible GPUs; shards will share devices.
            """ nshards=n ndevices=ndev
        end

        return [all_devs[mod1(p, ndev)] for p in 1:n]
    else
        ids = collect(device_ids)
        if ns !== nothing && length(ids) != ns
            error("length(device_ids) = $(length(ids)) does not match nshards = $(ns).")
        end
        return [begin
                    0 <= i < ndev || error("Device id $i out of range 0:$(ndev-1).")
                    all_devs[i+1]
                end for i in ids]
    end
end


function enable_peer_access!(devs)
    uniq = unique(devs)
    length(uniq) <= 1 && return 1.0

    npair = 0
    nok = 0
    for src in uniq, dst in uniq
        src == dst && continue
        npair += 1
        ok = false
        try
            if CUDA.can_access_peer(src, dst)
                CUDA.device!(src) do
                    CUDA.enable_peer_access(CUDA.context(dst))
                end
                ok = true
            end
        catch err
            @debug "Peer access setup failed" src dst err
        end
        nok += ok ? 1 : 0
    end
    return npair == 0 ? 1.0 : nok / npair
end

function device_summary(devs)
    io = IOBuffer()
    for (p, d) in enumerate(devs)
        CUDA.device!(d)
        free, total = CUDA.free_memory(), CUDA.total_memory()
        println(io, "  shard $p -> device $(CUDA.deviceid(d)) ",
                "[$(CUDA.name(d))] ",
                "$(round(free/2^30, digits=2)) / $(round(total/2^30, digits=2)) GiB free")
    end
    return String(take!(io))
end
