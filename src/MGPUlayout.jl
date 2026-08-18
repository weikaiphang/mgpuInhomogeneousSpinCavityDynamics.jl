# ============================================================
# SHARDED STATE LAYOUT (2nd-order cumulant)
#
# The state is split into two parts with very different sizes:
#
#   SMALL block  (O(M),   replicated bit-identically on every shard)
#
#       [ a, a†a†, a†a,
#         S⁺, Sz, a†S⁺, a†S⁻, a†Sz,
#         S⁺S⁺|same, SzS⁺|same, S⁻S⁺|same, SzSz|same ]
#
#       length = 3 + 9M, identical ordering to the single-GPU package,
#       so saved observables keep the same index ranges.
#
#   LARGE block  (O(M²),  sharded over the ensemble index j)
#
#       Five M × m_loc matrices, column jl holds the owned bin
#       j = joff + jl and runs over all partner bins k = 1..M:
#
#           1  SpSp [k, jl] = <S⁺_j S⁺_k>
#           2  SzSp [k, jl] = <Sz_j S⁺_k>
#           3  SzSpT[k, jl] = <Sz_k S⁺_j>      (transpose replica)
#           4  SmSp [k, jl] = <S⁻_j S⁺_k>
#           5  SzSz [k, jl] = <Sz_j Sz_k>
#
# Storing the transpose replica SzSpT costs one extra M × m_loc matrix
# but removes the only term in the equations of motion that reaches
# across shards.  Because
#
#       SpSp is symmetric, SmSp is Hermitian, SzSz is symmetric,
#
# the derivative of SzSpT at (j,k) can be written entirely in terms of
# quantities the shard already owns, so the O(M²) part of the right-hand
# side needs *no* inter-GPU communication at all.
#
# Both blocks live in ONE contiguous device vector per shard, so every
# Runge-Kutta vector operation is a single fused kernel over the whole
# state.
# ============================================================

# ---- small block: scalars ----
const IDX_a     = 1
const IDX_ad_ad = 2
const IDX_ad_a  = 3
const NSCALAR   = 3

# ---- small block: vector field identifiers ----
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

# ---- large block: matrix identifiers ----
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

"""
    save_prefix_length(M)

Number of leading entries of the small block that hold every observable the
solver writes out (scalars plus S⁺, Sz, a†S⁺, a†S⁻, a†Sz).  Interpolation and
device-to-host transfers only ever touch this prefix.
"""
save_prefix_length(M::Integer) = NSCALAR + 5 * M

# Global (unsharded) state length, used only for the error-norm denominator
# so that tolerances mean the same thing as in the single-GPU package.
global_state_length(M::Integer) = NSCALAR + NSMALLFIELD * M + 4 * M * M


# ============================================================
# ENSEMBLE PARTITION
# ============================================================

"""
    EnsemblePartition(M, nshards)

Contiguous, load-balanced split of the `M` ensemble bins over `nshards`
shards.  Shard `p` owns bins `offsets[p]+1 : offsets[p]+counts[p]`.
"""
struct EnsemblePartition
    M::Int
    counts::Vector{Int}
    offsets::Vector{Int}   # 0-based
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


# ============================================================
# MEMORY MODEL
# ============================================================

"""
    register_count(integrator)

Number of full-state device registers an integrator needs.  This is what sets
the maximum reachable ensemble size, so it is reported by [`memory_report`](@ref).
"""
function register_count(integrator::Symbol)
    if integrator === :tsit5
        # uprev, tmp, k1..k7
        return 9
    elseif integrator === :ck45
        # uprev, stage, accumulator, k, error
        return 5
    else
        error("Unknown integrator = $(integrator). Use :tsit5 or :ck45.")
    end
end

"""
    shard_bytes(M, mloc, integrator, T)

Device bytes required by one shard: `register_count` copies of the shard state
plus the small O(M) scratch buffers.
"""
function shard_bytes(M::Integer, mloc::Integer, integrator::Symbol, ::Type{T}) where {T}
    elt = sizeof(Complex{T})
    state = shard_length(M, mloc) * elt
    regs  = register_count(integrator) * state
    # rowsum double buffer, partial reductions, global sums, saved prefix scratch
    scratch = (2 * 3 * M + 3 * 8 * mloc + 3 + save_prefix_length(M)) * elt
    consts  = 2 * M * sizeof(T)
    return regs + scratch + consts
end

"""
    memory_report(M, nshards; integrator=:tsit5, T=Float64)

Human-readable summary of the device memory needed to run an `M`-bin ensemble
split over `nshards` GPUs.
"""
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

"""
    max_bins(bytes_per_gpu, nshards; integrator=:tsit5, T=Float64, safety=0.85)

Largest ensemble size `M` that fits in `bytes_per_gpu` on each of `nshards`
GPUs.  Useful for sizing a run before launching it.
"""
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
