# ============================================================
# FUSED CUDA KERNELS
#
# The single-GPU package expresses the right-hand side as several dozen
# broadcast statements over M × M arrays, plus three GEMVs and four mask
# multiplications.  Every one of those is a separate pass over device memory,
# and the problem is entirely bandwidth bound.
#
# Here the whole O(M²) block is one kernel: five loads and five stores per
# ensemble pair, with the row sums that the vector equations need accumulated
# in registers during the same pass (so the GEMVs are free), and the
# zero-diagonal handled by a branch instead of a stored mask array.
# ============================================================

@inline muli(z::Complex) = Complex(-imag(z), real(z))   # i * z, without a multiply

# ------------------------------------------------------------
# Deterministic block reduction of three complex accumulators.
# Warp shuffles, then a serial pass over at most 32 warp results, so the
# summation order is fixed and results are reproducible run to run.
# ------------------------------------------------------------

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

@inline function warp_reduce1(v::T) where {T}
    o = 16
    while o > 0
        v += CUDA.shfl_down_sync(0xffffffff, v, o)
        o >>= 1
    end
    return v
end

@inline function block_reduce1(sh, v::T) where {T}
    tid  = threadIdx().x
    lane = (tid - 1) & 31
    wid  = (tid - 1) >> 5

    v = warp_reduce1(v)
    @inbounds if lane == 0
        sh[wid + 1] = v
    end
    sync_threads()

    if tid == 1
        nw = (blockDim().x + 31) >> 5
        @inbounds begin
            s = sh[1]
            for w in 1:(nw - 1)
                s += sh[w + 1]
            end
        end
        return s
    end
    return v
end


# ============================================================
# 1) CROSS-BIN BLOCK  (the O(M²) work)
#
# grid  = (nchunk, mloc):  one block per (k-chunk, owned bin)
# Column jl of the shard is contiguous in k, so the loads are coalesced and
# the row-sum reduction runs over contiguous memory.
# ============================================================

function cross_rhs_kernel!(du, u, part, delta_b, g_b,
                           M::Int, mloc::Int, joff::Int, lo::Int,
                           chunk_len::Int, nchunk::Int)
    CT = eltype(u)
    T  = real(CT)
    sh = CuStaticSharedArray(CT, 96)

    tid = threadIdx().x
    jl  = blockIdx().y
    ch  = blockIdx().x

    j = joff + jl

    @inbounds a = u[IDX_a]

    oSp   = NSCALAR
    oSz   = NSCALAR + M
    oadSp = NSCALAR + 2M
    oadSm = NSCALAR + 3M
    oadSz = NSCALAR + 4M

    @inbounds begin
        gj  = g_b[j]
        dj  = delta_b[j]
        Spj   = u[oSp + j]
        Szj   = u[oSz + j]
        adSpj = u[oadSp + j]
        adSmj = u[oadSm + j]
        adSzj = u[oadSz + j]
    end

    ca     = conj(a)
    cSpj   = conj(Spj)
    cadSmj = conj(adSmj)
    cadSzj = conj(adSzj)

    bs      = M * mloc                    # distance between the five matrices
    colbase = lo + (jl - 1) * M           # + k -> element (k, jl) of matrix 1

    kstart = (ch - 1) * chunk_len + 1
    kend   = min(ch * chunk_len, M)

    accP = zero(CT)
    accM = zero(CT)
    accZ = zero(CT)

    k = kstart + (tid - 1)
    @inbounds while k <= kend
        i1 = colbase + k

        P  = u[i1]
        Z  = u[i1 + bs]
        ZT = u[i1 + 2bs]
        Mm = u[i1 + 3bs]
        ZZ = u[i1 + 4bs]

        if k == j
            z = zero(CT)
            du[i1]       = z
            du[i1 + bs]  = z
            du[i1 + 2bs] = z
            du[i1 + 3bs] = z
            du[i1 + 4bs] = z
        else
            gk = g_b[k]
            dk = delta_b[k]

            Spk   = u[oSp + k]
            Szk   = u[oSz + k]
            adSpk = u[oadSp + k]
            adSmk = u[oadSm + k]
            adSzk = u[oadSz + k]

            cSpk   = conj(Spk)
            cadSmk = conj(adSmk)
            cadSzk = conj(adSzk)

            # ---- <S⁺_j S⁺_k> ----
            W  = Spk * adSzj + ca * Z  + adSpk * Szj - 2 * Spk * ca * Szj
            Ws = Spj * adSzk + ca * ZT + adSpj * Szk - 2 * Spj * ca * Szk
            du[i1] = muli((dj + dk) * P - 2 * gj * W - 2 * gk * Ws)

            # ---- <Sz_j S⁺_k> and its transpose replica <Sz_k S⁺_j> ----
            U1  = Spk * cadSmj + Spj * cadSmk + a * P - 2 * Spk * Spj * a
            U2  = Spk * adSmj + ca * Mm       + adSpk * cSpj - 2 * Spk * ca * cSpj
            U2s = Spj * adSmk + ca * conj(Mm) + adSpj * cSpk - 2 * Spj * ca * cSpk
            U3  = ZZ * ca + Szk * adSzj + Szj * adSzk - 2 * ca * Szk * Szj

            du[i1 + bs]  = muli(dk * Z  - gj * U1 + gj * U2  - 2 * gk * U3)
            du[i1 + 2bs] = muli(dj * ZT - gk * U1 + gk * U2s - 2 * gj * U3)

            # ---- <S⁻_j S⁺_k> and <Sz_j Sz_k> ----
            Y1  = cadSzj * Spk + cadSmk * Szj + a * Z  - 2 * Spk * a * Szj
            Y1s = cadSzk * Spj + cadSmj * Szk + a * ZT - 2 * Spj * a * Szk
            Y2  = ca * conj(ZT) + Szk * adSmj + cSpj * adSzk - 2 * ca * Szk * cSpj
            Y2s = ca * conj(Z)  + Szj * adSmk + cSpk * adSzj - 2 * ca * Szj * cSpk

            du[i1 + 3bs] = muli((dk - dj) * Mm + 2 * gj * Y1 - 2 * gk * Y2)
            du[i1 + 4bs] = muli(gj * (Y2 - Y1s) + gk * (Y2s - Y1))

            # ---- row sums for the vector equations (free: data already loaded) ----
            accP += gk * P
            accM += gk * Mm
            accZ += gk * Z
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


# ============================================================
# 2) ROW-SUM FINALISATION
# Reduce the per-chunk partials into this shard's slice of the globally
# indexed row-sum buffer (3 consecutive entries per bin, so the owned slice
# is contiguous and ships to peers in a single copy).
# ============================================================

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


# ============================================================
# 3) GLOBAL COUPLING SUMS
#     Σ₁ = Σ g_j conj(S⁺_j),  Σ₂ = Σ g_j <a†S⁺_j>,  Σ₃ = Σ g_j <a†S⁻_j>
# Computed redundantly on every shard from the replicated small block, so no
# communication is needed for the cavity equations.
# ============================================================

function global_sums_kernel!(gsums, u, g_b, M::Int)
    sh = CuStaticSharedArray(Complex{eltype(g_b)}, 96)

    oSp   = NSCALAR
    oadSp = NSCALAR + 2M
    oadSm = NSCALAR + 3M

    s1 = zero(eltype(gsums))
    s2 = zero(eltype(gsums))
    s3 = zero(eltype(gsums))

    j = threadIdx().x
    @inbounds while j <= M
        g = g_b[j]
        s1 += g * conj(u[oSp + j])
        s2 += g * u[oadSp + j]
        s3 += g * u[oadSm + j]
        j += blockDim().x
    end

    r1, r2, r3 = block_reduce3(sh, s1, s2, s3)

    @inbounds if threadIdx().x == 1
        gsums[1] = r1
        gsums[2] = r2
        gsums[3] = r3
    end
    return nothing
end


# ============================================================
# 4) SMALL BLOCK (scalars, first-order spins, same-bin correlations)
#
# Evaluated redundantly on every shard.  It costs O(M) work, needs the
# gathered row sums, and produces bit-identical output everywhere, which is
# what keeps the replicated part of the state exactly in sync without any
# further communication.
# ============================================================

function small_rhs_kernel!(du, u, rowsum, gsums, delta_b, g_b, Et::Complex{T},
                           delta0::T, kappa_t::T, sqrt_ke::T, M::Int) where {T}
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j > M && return nothing

    oSp     = NSCALAR
    oSz     = NSCALAR + M
    oadSp   = NSCALAR + 2M
    oadSm   = NSCALAR + 3M
    oadSz   = NSCALAR + 4M
    oSpSp_s = NSCALAR + 5M
    oSzSp_s = NSCALAR + 6M
    oSmSp_s = NSCALAR + 7M
    oSzSz_s = NSCALAR + 8M

    @inbounds begin
        a     = u[IDX_a]
        ad_ad = u[IDX_ad_ad]
        ad_a  = u[IDX_ad_a]
    end
    ca  = conj(a)
    cEt = conj(Et)
    half = T(0.5)

    # ---- cavity (one thread does the three scalars) ----
    if j == 1
        @inbounds begin
            S1 = gsums[1]
            S2 = gsums[2]
            S3 = gsums[3]

            du[IDX_a] = sqrt_ke * Et - muli(delta0 * a) - muli(S1) - half * kappa_t * a

            du[IDX_ad_ad] = 2 * muli(delta0 * ad_ad) + 2 * muli(S2) -
                            kappa_t * ad_ad + 2 * sqrt_ke * ca * cEt

            du[IDX_ad_a] = muli(conj(S3)) - muli(S3) - kappa_t * ad_a +
                           sqrt_ke * Et * ca + sqrt_ke * cEt * a
        end
    end

    @inbounds begin
        gj = g_b[j]
        dj = delta_b[j]

        Sp   = u[oSp + j]
        Sz   = u[oSz + j]
        adSp = u[oadSp + j]
        adSm = u[oadSm + j]
        adSz = u[oadSz + j]

        SpSp_s = u[oSpSp_s + j]
        SzSp_s = u[oSzSp_s + j]
        SmSp_s = u[oSmSp_s + j]
        SzSz_s = u[oSzSz_s + j]

        rb = 3 * (j - 1)
        sum_gSpSp = gj * SpSp_s + rowsum[rb + 1]
        sum_gSmSp = gj * SmSp_s + rowsum[rb + 2]
        sum_gSzSp = gj * SzSp_s + rowsum[rb + 3]
    end

    cSp   = conj(Sp)
    cadSm = conj(adSm)
    cadSz = conj(adSz)

    @inbounds begin
        # ---- first-order spin ----
        du[oSp + j] = muli(dj * Sp - 2 * gj * adSz)
        du[oSz + j] = muli(gj * adSm - gj * cadSm)

        # ---- cavity-spin correlations ----
        du[oadSp + j] = muli(delta0 * adSp + dj * adSp + sum_gSpSp -
                             2 * gj * (2 * ca * adSz + ad_ad * Sz - 2 * ca * ca * Sz)) -
                        half * kappa_t * adSp + sqrt_ke * cEt * Sp

        du[oadSm + j] = muli(delta0 * adSm - dj * adSm + 2 * gj * Sz + sum_gSmSp +
                             2 * gj * (cadSz * ca + a * adSz + Sz * ad_a - 2 * ca * a * Sz)) -
                        half * kappa_t * adSm + sqrt_ke * cEt * cSp

        du[oadSz + j] = muli(delta0 * adSz + sum_gSzSp -
                             gj * (Sp + Sp * ad_a + ca * cadSm + a * adSp - 2 * Sp * ca * a) +
                             gj * (2 * ca * adSm + ad_ad * cSp - 2 * ca * ca * cSp)) -
                        half * kappa_t * adSz + sqrt_ke * cEt * Sz

        # ---- same-bin spin correlations ----
        du[oSpSp_s + j] = muli(2 * dj * SpSp_s + 2 * gj * adSp -
                               4 * gj * (Sp * adSz + SzSp_s * ca + adSp * Sz -
                                         2 * Sp * ca * Sz))

        du[oSzSp_s + j] = muli(dj * SzSp_s -
                               gj * (2 * Sp * cadSm + a * SpSp_s - 2 * Sp * Sp * a) +
                               gj * (Sp * adSm + ca * SmSp_s + adSp * cSp -
                                     2 * Sp * ca * cSp) -
                               2 * gj * (SzSz_s * ca + 2 * adSz * Sz - 2 * ca * Sz * Sz))

        Q1 = cadSz * Sp + SzSp_s * a + cadSm * Sz - 2 * Sp * a * Sz
        Q2 = ca * conj(SzSp_s) + cSp * adSz + adSm * Sz - 2 * ca * cSp * Sz

        du[oSmSp_s + j] = muli(2 * gj * Q1 - 2 * gj * Q2)
        du[oSzSz_s + j] = muli(gj * cadSm - gj * adSm - 2 * gj * Q1 + 2 * gj * Q2)
    end

    return nothing
end


# ============================================================
# 5) RUNGE-KUTTA VECTOR OPERATIONS
#
# All registers are single contiguous vectors covering the small and large
# blocks at once, so each stage combination is one kernel over the whole
# shard state instead of one per state component.
# ============================================================

# Compile-time unrolled Σ_s cs[s]*ks[s][i]; recursion keeps the tuple in
# registers instead of spilling it to local memory.
@inline function combo(ks::Tuple{Any}, cs::Tuple{Any}, i)
    @inbounds return cs[1] * ks[1][i]
end
@inline function combo(ks::Tuple, cs::Tuple, i)
    @inbounds return cs[1] * ks[1][i] + combo(Base.tail(ks), Base.tail(cs), i)
end

"""
    combine_kernel!(uout, ubase, ks, cs, dt, n)

`uout[i] = ubase[i] + dt * Σ_s cs[s] * ks[s][i]` for `i in 1:n`.
"""
function combine_kernel!(uout, ubase, ks::NTuple{NK,Any}, cs::NTuple{NK,T},
                         dt::T, n::Int) where {NK,T}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds while i <= n
        uout[i] = ubase[i] + dt * combo(ks, cs, i)
        i += stride
    end
    return nothing
end

"""
    lowstorage_kernel!(stage, accum, err, k, base, dt, cA, cB, cE, n, write_stage, first)

The three register updates of a 2R+ low-storage stage, fused into one pass:

    stage = base + dt*cA*k       (skipped on the final stage)
    accum = base + dt*cB*k
    err   = err  + dt*cE*k

`base` is the running solution accumulator, which on the first stage is the
previous step's state, so no register has to be pre-copied.
"""
function lowstorage_kernel!(stage, accum, err, k, base, dt::T, cA::T, cB::T, cE::T,
                            n::Int, write_stage::Bool, first::Bool) where {T}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds while i <= n
        ki = k[i]
        ai = base[i]
        if write_stage
            stage[i] = ai + (dt * cA) * ki
        end
        accum[i] = ai + (dt * cB) * ki
        err[i]   = (first ? zero(ki) : err[i]) + (dt * cE) * ki
        i += stride
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

"""
    errnorm_kernel!(partial, uprev, u, ks, es, dt, atol, rtol, istart, iend)

Sum of `abs2(err_i / (atol + rtol*max(|uprev_i|,|u_i|)))` with the error
estimate `err = dt * Σ es[s] * ks[s]` formed on the fly, so the embedded
solution never has to be materialised.
"""
function errnorm_kernel!(partial, uprev, u, ks::NTuple{NK,Any}, es::NTuple{NK,T},
                         dt::T, atol::T, rtol::T, istart::Int, iend::Int,
                         skip_a::Int, skip_b::Int) where {NK,T}
    sh = CuStaticSharedArray(T, 32)

    i = istart + (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    stride = blockDim().x * gridDim().x
    acc = zero(T)

    @inbounds while i <= iend
        if skip_a <= i <= skip_b
            i += stride
            continue
        end
        e = dt * combo(ks, es, i)
        sc = atol + rtol * max(abs(uprev[i]), abs(u[i]))
        acc += abs2(e) / (sc * sc)
        i += stride
    end

    r = block_reduce1(sh, acc)
    @inbounds if threadIdx().x == 1
        partial[blockIdx().x] = r
    end
    return nothing
end

"""
    diffnorm_kernel!(partial, x, y, uref, atol, rtol, istart, iend)

Sum of `abs2((x_i - y_i) / (atol + rtol*|uref_i|))`, used by the low-storage
stepper (whose error register is explicit) and by the initial step-size
heuristic.  Passing `y = nothing` measures `x` itself.
"""
function diffnorm_kernel!(partial, x, y, uref, atol::T, rtol::T,
                          istart::Int, iend::Int, skip_a::Int, skip_b::Int) where {T}
    sh = CuStaticSharedArray(T, 32)

    i = istart + (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    stride = blockDim().x * gridDim().x
    acc = zero(T)

    @inbounds while i <= iend
        if skip_a <= i <= skip_b
            i += stride
            continue
        end
        e = y === nothing ? x[i] : x[i] - y[i]
        sc = atol + rtol * abs(uref[i])
        acc += abs2(e) / (sc * sc)
        i += stride
    end

    r = block_reduce1(sh, acc)
    @inbounds if threadIdx().x == 1
        partial[blockIdx().x] = r
    end
    return nothing
end

"""
    init_szsz_cross_kernel!(u, Nj, M, mloc, joff, lo)

Fill the sharded `<Sz_j Sz_k>` block with `Nj[j]*Nj[k]/4` off the diagonal.
Done on device so that no O(M²) host buffer is ever allocated.
"""
function init_szsz_cross_kernel!(u, Nj, M::Int, mloc::Int, joff::Int, lo::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    n = M * mloc
    base = lo + 4 * n
    @inbounds while i <= n
        jl = (i - 1) ÷ M + 1
        k  = (i - 1) % M + 1
        j  = joff + jl
        u[base + i] = j == k ? zero(eltype(u)) :
                      eltype(u)(Nj[j] * Nj[k] / 4)
        i += stride
    end
    return nothing
end

function sum_partials_kernel!(out, partial, n::Int)
    sh = CuStaticSharedArray(eltype(out), 32)
    acc = zero(eltype(out))
    i = threadIdx().x
    @inbounds while i <= n
        acc += partial[i]
        i += blockDim().x
    end
    r = block_reduce1(sh, acc)
    @inbounds if threadIdx().x == 1
        out[1] = r
    end
    return nothing
end
