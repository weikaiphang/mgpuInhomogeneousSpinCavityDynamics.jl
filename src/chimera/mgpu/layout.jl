
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

# Row-sum shards store local columns of the three cross blocks that
# enter the 3M Allreduce (SpSp, SmSp, SzSp). No SzSpT transpose copy —
# that existed only for the deleted fused RHS kernel.
const B_SpSp  = 1
const B_SmSp  = 2
const B_SzSp  = 3
const NROWSUM_BLOCK = 3

@inline small_offset(M::Integer, f::Integer) = NSCALAR + (f - 1) * M
@inline small_range(M::Integer, f::Integer) =
    (small_offset(M, f) + 1):(small_offset(M, f) + M)

# Logical small prefix (3 + 9M) of the monolith state. Multi-GPU no
# longer keeps a private register file; OrdinaryDiffEq holds one u.
small_length(M::Integer) = NSCALAR + NSMALLFIELD * M
large_length(M::Integer, mloc::Integer) = NROWSUM_BLOCK * M * mloc
shard_length(M::Integer, mloc::Integer) = small_length(M) + large_length(M, mloc)


save_prefix_length(M::Integer) = NSCALAR + 5 * M

# Logical global state still 3 + 9M + 4M²; do not count SzSpT.
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




# Workspace only: local columns of 3 cross blocks + 3M row-sum buffer.
# OrdinaryDiffEq owns the assembled 3+9M+4M² state (one copy).
function shard_bytes(M::Integer, mloc::Integer, ::Type{T}) where {T}
    elt = sizeof(Complex{T})
    cols = large_length(M, mloc) * elt
    rowsum = 2 * 3 * M * elt
    scratch = (3 * 8 * mloc + 3) * elt
    consts = 2 * M * sizeof(T)
    return cols + rowsum + scratch + consts
end


function memory_report(M::Integer, ns::Integer; T::Type = Float64)
    part = EnsemblePartition(M, ns)
    mloc = maximum(part.counts)
    per = shard_bytes(M, mloc, T)
    assembled = global_state_length(M) * sizeof(Complex{T})

    gib(x) = x / 2^30

    io = IOBuffer()
    println(io, "Ensemble bins M            : $M")
    println(io, "Shards (GPUs)              : $ns")
    println(io, "Bins per shard (max)       : $mloc")
    println(io, "Element type               : Complex{$T}")
    println(io, "Stepper                    : OrdinaryDiffEq Tsit5 (one assembled VF)")
    println(io, "Assembled state            : $(round(gib(assembled), digits=3)) GiB")
    println(io, "Row-sum workspace / shard  : $(round(gib(per), digits=3)) GiB")
    println(io, "Device workspace total     : $(round(gib(per * ns), digits=3)) GiB")
    return String(take!(io))
end


function max_bins(bytes_per_gpu::Real, ns::Integer;
                  T::Type = Float64, safety::Real = 0.85)
    budget = safety * bytes_per_gpu
    lo, hi = 1, 1
    while hi < 10^7 && shard_bytes(hi, cld(hi, ns), T) < budget
        lo = hi
        hi *= 2
    end
    while lo < hi
        mid = (lo + hi + 1) ÷ 2
        if shard_bytes(mid, cld(mid, ns), T) < budget
            lo = mid
        else
            hi = mid - 1
        end
    end
    return lo
end


# Backend picker used by `_setup_rowsum_exchange`. Kept pure so the
# NCCL → P2P → host ladder can be stress-tested without a GPU.
function choose_rowsum_exchange(ns::Integer;
                                have_nccl::Bool,
                                nunique_devices::Integer,
                                nccl_ok::Bool,
                                p2p_ok::Bool)
    ns <= 1 && return :none
    if have_nccl && nunique_devices == ns && nccl_ok
        return :nccl
    elseif p2p_ok
        return :p2p
    else
        return :host
    end
end

rowsum_owned_range(part::EnsemblePartition, p::Integer) =
    (3 * part.offsets[p] + 1):(3 * part.offsets[p] + 3 * part.counts[p])

# CPU mirrors of exchange_rowsums_nccl! / _p2p! / _host!.
# NCCL Allreduces the full 3M buffer (non-owned slots must be zero).
# P2P and host copy each shard's owned 3*mloc slice (assignment, not sum).
function assemble_rowsums_nccl!(buffers::Vector{V}) where {V<:AbstractVector}
    isempty(buffers) && return buffers
    acc = copy(buffers[1])
    for b in @view buffers[2:end]
        acc .+= b
    end
    for b in buffers
        copyto!(b, acc)
    end
    return buffers
end

function assemble_rowsums_p2p!(buffers::Vector{V}, part::EnsemblePartition) where {V<:AbstractVector}
    ns = nshards(part)
    length(buffers) == ns || error("expected $(ns) shard buffers, got $(length(buffers))")
    for dst in 1:ns
        for src in 1:ns
            src == dst && continue
            r = rowsum_owned_range(part, src)
            copyto!(buffers[dst], first(r), buffers[src], first(r), length(r))
        end
    end
    return buffers
end

function assemble_rowsums_host!(buffers::Vector{V}, part::EnsemblePartition) where {V<:AbstractVector}
    ns = nshards(part)
    length(buffers) == ns || error("expected $(ns) shard buffers, got $(length(buffers))")
    host = similar(buffers[1], 3 * part.M)
    fill!(host, zero(eltype(host)))
    for p in 1:ns
        r = rowsum_owned_range(part, p)
        copyto!(host, first(r), buffers[p], first(r), length(r))
    end
    for b in buffers
        copyto!(b, host)
    end
    return buffers
end

function assemble_rowsums!(mode::Symbol, buffers::Vector{V},
                           part::EnsemblePartition) where {V<:AbstractVector}
    if mode === :nccl
        return assemble_rowsums_nccl!(buffers)
    elseif mode === :p2p
        return assemble_rowsums_p2p!(buffers, part)
    elseif mode === :host
        return assemble_rowsums_host!(buffers, part)
    elseif mode === :none
        return buffers
    else
        error("Unknown row-sum exchange mode = $(mode). Use :nccl, :p2p, :host, or :none.")
    end
end
