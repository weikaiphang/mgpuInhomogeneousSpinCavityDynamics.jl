
const IDX_a     = 1
const IDX_ad_ad = 2
const IDX_ad_a  = 3
const NSCALAR   = 3

const F_Sp     = 1
const F_Sz     = 2
const F_adSp   = 3
const F_adSm   = 4
const F_adSz   = 5
const F_SpSp_s = 6
const F_SzSp_s = 7
const F_SmSp_s = 8
const F_SzSz_s = 9
const NSMALLFIELD = 9

const B_SpSp  = 1
const B_SzSp  = 2
const B_SzSpT = 3
const B_SmSp  = 4
const B_SzSz  = 5
const NBLOCK  = 5

@inline small_offset(M::Integer, f::Integer) = NSCALAR + (f - 1) * M
@inline small_range(M::Integer, f::Integer) =
    (small_offset(M, f) + 1):(small_offset(M, f) + M)

small_length(M::Integer) = NSCALAR + NSMALLFIELD * M
large_length(M::Integer, mloc::Integer) = NBLOCK * M * mloc
shard_length(M::Integer, mloc::Integer) = small_length(M) + large_length(M, mloc)


save_prefix_length(M::Integer) = NSCALAR + 5 * M

global_state_length(M::Integer) = NSCALAR + NSMALLFIELD * M + 4 * M * M




struct EnsemblePartition
    M::Int
    counts::Vector{Int}
    offsets::Vector{Int}
end

function EnsemblePartition(M::Integer, nshards::Integer)
    M >= 1 || error("M must be positive, got $M.")
    nshards >= 1 || error("nshards must be positive, got $nshards.")
    nshards <= M || error(
        "Cannot split M = $M ensemble bins over $nshards shards; " *
        "use at most $M shards."
    )

    base = div(M, nshards)
    rem_ = mod(M, nshards)

    counts = [base + (p <= rem_ ? 1 : 0) for p in 1:nshards]
    offsets = zeros(Int, nshards)
    for p in 2:nshards
        offsets[p] = offsets[p-1] + counts[p-1]
    end

    @assert sum(counts) == M
    return EnsemblePartition(Int(M), counts, offsets)
end

nshards(part::EnsemblePartition) = length(part.counts)
Base.getindex(part::EnsemblePartition, p::Integer) =
    (part.offsets[p] + 1):(part.offsets[p] + part.counts[p])




function register_count(integrator::Symbol)
    if integrator === :tsit5

        return 9
    elseif integrator === :ck45

        return 5
    else
        error("Unknown integrator = $(integrator). Use :tsit5 or :ck45.")
    end
end


function shard_bytes(M::Integer, mloc::Integer, integrator::Symbol, ::Type{T}) where {T}
    elt = sizeof(Complex{T})
    state = shard_length(M, mloc) * elt
    regs  = register_count(integrator) * state

    scratch = (2 * 3 * M + 3 * 8 * mloc + 3 + save_prefix_length(M)) * elt
    consts  = 2 * M * sizeof(T)
    return regs + scratch + consts
end


function memory_report(M::Integer, ns::Integer; integrator::Symbol = :tsit5,
                       T::Type = Float64)
    part = EnsemblePartition(M, ns)
    mloc = maximum(part.counts)
    per = shard_bytes(M, mloc, integrator, T)

    gib(x) = x / 2^30

    io = IOBuffer()
    println(io, "Ensemble bins M            : $M")
    println(io, "Shards (GPUs)              : $ns")
    println(io, "Bins per shard (max)       : $mloc")
    println(io, "Element type               : Complex{$T}")
    println(io, "Integrator                 : $integrator ($(register_count(integrator)) registers)")
    println(io, "State per shard            : $(round(gib(shard_length(M, mloc)*sizeof(Complex{T})), digits=3)) GiB")
    println(io, "Device memory per shard    : $(round(gib(per), digits=3)) GiB")
    println(io, "Device memory total        : $(round(gib(per*ns), digits=3)) GiB")
    return String(take!(io))
end


function max_bins(bytes_per_gpu::Real, ns::Integer; integrator::Symbol = :tsit5,
                  T::Type = Float64, safety::Real = 0.85)
    budget = safety * bytes_per_gpu
    lo, hi = 1, 1
    while hi < 10^7 && shard_bytes(hi, cld(hi, ns), integrator, T) < budget
        lo = hi
        hi *= 2
    end
    while lo < hi
        mid = (lo + hi + 1) ÷ 2
        if shard_bytes(mid, cld(mid, ns), integrator, T) < budget
            lo = mid
        else
            hi = mid - 1
        end
    end
    return lo
end
