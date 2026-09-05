
using Base.Threads: @threads, nthreads

# CPU twin of the 2nd-order RHS. Conventions match rhs_2nd_order! / kernels:
#   ∂t a  = √κ_e E − iδ₀ a − i∑ g S⁻ − (κ/2) a     (κ = κ_e + κ_i)
#   ∂t S⁺ = iΔ S⁺ − 2i g ⟨a† Sᶻ⟩
#   ∂t Sᶻ = −i g ⟨a S⁺⟩ + i g ⟨a† S⁻⟩
# No spin T₁/T₂ on this RHS.
# threaded=nothing → auto (nthreads()>1 && M≥16). Same math as serial.
const _RHS_CPU_THREAD_MIN_M = 16

@inline function _rhs_cpu_use_threads(threaded, M)
    want = threaded === nothing ? M >= _RHS_CPU_THREAD_MIN_M : threaded
    return want && nthreads() > 1 && M >= 2
end

function rhs_cpu!(du::AbstractVector{Complex{T}}, u::AbstractVector{Complex{T}},
                  delta0::T, kappa_e::T, kappa_i::T,
                  delta_b::AbstractVector{T}, g_b::AbstractVector{T},
                  Et::Complex{T}; threaded::Union{Nothing,Bool}=nothing) where {T}
    M = length(delta_b)
    kappa_t = kappa_e + kappa_i
    sqrt_ke = sqrt(kappa_e)
    half = T(1) / 2

    a     = u[IDX_a]
    ad_ad = u[IDX_ad_ad]
    ad_a  = u[IDX_ad_a]
    ca    = conj(a)
    cEt   = conj(Et)

    Sp   = @view u[small_range(M, F_Sp)]
    Sz   = @view u[small_range(M, F_Sz)]
    adSp = @view u[small_range(M, F_adSp)]
    adSm = @view u[small_range(M, F_adSm)]
    adSz = @view u[small_range(M, F_adSz)]
    SpSp_s = @view u[small_range(M, F_SpSp_s)]
    SzSp_s = @view u[small_range(M, F_SzSp_s)]
    SmSp_s = @view u[small_range(M, F_SmSp_s)]
    SzSz_s = @view u[small_range(M, F_SzSz_s)]

    nsmall = small_length(M)
    SpSp = reshape(@view(u[nsmall + 1           : nsmall + M*M]),     M, M)
    SzSp = reshape(@view(u[nsmall + M*M + 1     : nsmall + 2M*M]),    M, M)
    SmSp = reshape(@view(u[nsmall + 2M*M + 1    : nsmall + 3M*M]),    M, M)
    SzSz = reshape(@view(u[nsmall + 3M*M + 1    : nsmall + 4M*M]),    M, M)

    dSp   = @view du[small_range(M, F_Sp)]
    dSz   = @view du[small_range(M, F_Sz)]
    dadSp = @view du[small_range(M, F_adSp)]
    dadSm = @view du[small_range(M, F_adSm)]
    dadSz = @view du[small_range(M, F_adSz)]
    dSpSp_s = @view du[small_range(M, F_SpSp_s)]
    dSzSp_s = @view du[small_range(M, F_SzSp_s)]
    dSmSp_s = @view du[small_range(M, F_SmSp_s)]
    dSzSz_s = @view du[small_range(M, F_SzSz_s)]

    dSpSp = reshape(@view(du[nsmall + 1          : nsmall + M*M]),    M, M)
    dSzSp = reshape(@view(du[nsmall + M*M + 1    : nsmall + 2M*M]),   M, M)
    dSmSp = reshape(@view(du[nsmall + 2M*M + 1   : nsmall + 3M*M]),   M, M)
    dSzSz = reshape(@view(du[nsmall + 3M*M + 1   : nsmall + 4M*M]),   M, M)

    S1 = zero(Complex{T})
    S2 = zero(Complex{T})
    S3 = zero(Complex{T})
    @inbounds for j in 1:M
        gj = g_b[j]
        S1 += gj * conj(Sp[j])
        S2 += gj * adSp[j]
        S3 += gj * adSm[j]
    end

    du[IDX_a]     = sqrt_ke * Et - im * delta0 * a - im * S1 - half * kappa_t * a
    du[IDX_ad_ad] = 2im * delta0 * ad_ad + 2im * S2 - kappa_t * ad_ad + 2 * sqrt_ke * ca * cEt
    du[IDX_ad_a]  = im * conj(S3) - im * S3 - kappa_t * ad_a +
                    sqrt_ke * Et * ca + sqrt_ke * cEt * a

    use_thr = _rhs_cpu_use_threads(threaded, M)
    if use_thr
        @threads for j in 1:M
            _rhs_cpu_small_j!(dSp, dSz, dadSp, dadSm, dadSz,
                              dSpSp_s, dSzSp_s, dSmSp_s, dSzSz_s,
                              Sp, Sz, adSp, adSm, adSz,
                              SpSp_s, SzSp_s, SmSp_s, SzSz_s,
                              SpSp, SmSp, SzSp, g_b, delta_b,
                              a, ad_ad, ad_a, ca, cEt, delta0, kappa_t, sqrt_ke, half, M, j)
        end
        @threads for k in 1:M
            _rhs_cpu_cross_k!(dSpSp, dSzSp, dSmSp, dSzSz,
                              Sp, Sz, adSp, adSm, adSz,
                              SpSp, SzSp, SmSp, SzSz,
                              g_b, delta_b, a, ca, M, k)
        end
    else
        @inbounds for j in 1:M
            _rhs_cpu_small_j!(dSp, dSz, dadSp, dadSm, dadSz,
                              dSpSp_s, dSzSp_s, dSmSp_s, dSzSz_s,
                              Sp, Sz, adSp, adSm, adSz,
                              SpSp_s, SzSp_s, SmSp_s, SzSz_s,
                              SpSp, SmSp, SzSp, g_b, delta_b,
                              a, ad_ad, ad_a, ca, cEt, delta0, kappa_t, sqrt_ke, half, M, j)
        end
        @inbounds for k in 1:M
            _rhs_cpu_cross_k!(dSpSp, dSzSp, dSmSp, dSzSz,
                              Sp, Sz, adSp, adSm, adSz,
                              SpSp, SzSp, SmSp, SzSz,
                              g_b, delta_b, a, ca, M, k)
        end
    end

    return nothing
end

@inline function _rhs_cpu_small_j!(dSp, dSz, dadSp, dadSm, dadSz,
                                   dSpSp_s, dSzSp_s, dSmSp_s, dSzSz_s,
                                   Sp, Sz, adSp, adSm, adSz,
                                   SpSp_s, SzSp_s, SmSp_s, SzSz_s,
                                   SpSp, SmSp, SzSp, g_b, delta_b,
                                   a, ad_ad, ad_a, ca, cEt, delta0, kappa_t, sqrt_ke, half, M, j)
    @inbounds begin
        gj = g_b[j]
        dj = delta_b[j]

        dSp[j] = im * dj * Sp[j] - 2im * gj * adSz[j]
        dSz[j] = -im * gj * conj(adSm[j]) + im * gj * adSm[j]

        # Same-bin + off-diagonal cross only (k==j is unused in the cross block).
        sumP = gj * SpSp_s[j]
        sumM = gj * SmSp_s[j]
        sumZ = gj * SzSp_s[j]
        for k in 1:M
            k == j && continue
            gk = g_b[k]
            sumP += gk * SpSp[j, k]
            sumM += gk * SmSp[j, k]
            sumZ += gk * SzSp[j, k]
        end

        dadSp[j] = (im * delta0 * adSp[j] + im * dj * adSp[j] + im * sumP
                    - half * kappa_t * adSp[j] + sqrt_ke * cEt * Sp[j]
                    - 2im * gj * (2 * ca * adSz[j] + ad_ad * Sz[j] - 2 * ca^2 * Sz[j]))

        dadSm[j] = (im * delta0 * adSm[j] - im * dj * adSm[j]
                    + 2im * gj * Sz[j] + im * sumM
                    - half * kappa_t * adSm[j] + sqrt_ke * cEt * conj(Sp[j])
                    + 2im * gj * (conj(adSz[j]) * ca + a * adSz[j] + Sz[j] * ad_a
                                  - 2 * ca * a * Sz[j]))

        dadSz[j] = (im * delta0 * adSz[j] + im * sumZ
                    - half * kappa_t * adSz[j] + sqrt_ke * cEt * Sz[j]
                    - im * gj * (Sp[j] + Sp[j] * ad_a + ca * conj(adSm[j]) + a * adSp[j]
                                 - 2 * Sp[j] * ca * a)
                    + im * gj * (2 * ca * adSm[j] + ad_ad * conj(Sp[j])
                                 - 2 * ca^2 * conj(Sp[j])))

        dSpSp_s[j] = (2im * dj * SpSp_s[j] + 2im * gj * adSp[j]
                      - 4im * gj * (Sp[j] * adSz[j] + SzSp_s[j] * ca + adSp[j] * Sz[j]
                                    - 2 * Sp[j] * ca * Sz[j]))

        dSzSp_s[j] = (im * dj * SzSp_s[j]
                      - im * gj * (2 * Sp[j] * conj(adSm[j]) + a * SpSp_s[j] - 2 * Sp[j]^2 * a)
                      + im * gj * (Sp[j] * adSm[j] + ca * SmSp_s[j] + adSp[j] * conj(Sp[j])
                                   - 2 * Sp[j] * ca * conj(Sp[j]))
                      - 2im * gj * (SzSz_s[j] * ca + 2 * adSz[j] * Sz[j] - 2 * ca * Sz[j]^2))

        Q1 = conj(adSz[j]) * Sp[j] + SzSp_s[j] * a + conj(adSm[j]) * Sz[j] - 2 * Sp[j] * a * Sz[j]
        Q2 = ca * conj(SzSp_s[j]) + conj(Sp[j]) * adSz[j] + adSm[j] * Sz[j] - 2 * ca * conj(Sp[j]) * Sz[j]
        dSmSp_s[j] = 2im * gj * Q1 - 2im * gj * Q2
        dSzSz_s[j] = im * gj * conj(adSm[j]) - im * gj * adSm[j] - 2im * gj * Q1 + 2im * gj * Q2
    end
    return nothing
end

# Column-major: inner j, thread over k. Same algebra as the old (j,k) nest.
@inline function _rhs_cpu_cross_k!(dSpSp, dSzSp, dSmSp, dSzSz,
                                   Sp, Sz, adSp, adSm, adSz,
                                   SpSp, SzSp, SmSp, SzSz,
                                   g_b, delta_b, a, ca, M, k)
    @inbounds begin
        gk = g_b[k]
        dk = delta_b[k]
        for j in 1:M
            if j == k
                dSpSp[j, k] = 0
                dSzSp[j, k] = 0
                dSmSp[j, k] = 0
                dSzSz[j, k] = 0
                continue
            end

            gj = g_b[j]
            dj = delta_b[j]

            W  = Sp[k] * adSz[j] + ca * SzSp[j, k] + adSp[k] * Sz[j] - 2 * Sp[k] * ca * Sz[j]
            Ws = Sp[j] * adSz[k] + ca * SzSp[k, j] + Sz[k] * adSp[j] - 2 * Sp[j] * ca * Sz[k]
            dSpSp[j, k] = im * (dj + dk) * SpSp[j, k] - 2im * gj * W - 2im * gk * Ws

            U1 = Sp[k] * conj(adSm[j]) + Sp[j] * conj(adSm[k]) + a * SpSp[j, k] - 2 * Sp[k] * Sp[j] * a
            U2 = Sp[k] * adSm[j] + ca * SmSp[j, k] + adSp[k] * conj(Sp[j]) - 2 * Sp[k] * ca * conj(Sp[j])
            U3 = SzSz[j, k] * ca + Sz[k] * adSz[j] + Sz[j] * adSz[k] - 2 * ca * Sz[k] * Sz[j]
            dSzSp[j, k] = im * dk * SzSp[j, k] - im * gj * U1 + im * gj * U2 - 2im * gk * U3

            Y1 = conj(adSz[j]) * Sp[k] + conj(adSm[k]) * Sz[j] + a * SzSp[j, k] - 2 * Sp[k] * a * Sz[j]
            Y2 = ca * conj(SzSp[k, j]) + Sz[k] * adSm[j] + conj(Sp[j]) * adSz[k] - 2 * ca * Sz[k] * conj(Sp[j])
            dSmSp[j, k] = -im * dj * SmSp[j, k] + im * dk * SmSp[j, k] + 2im * gj * Y1 - 2im * gk * Y2

            Y1s = conj(adSz[k]) * Sp[j] + conj(adSm[j]) * Sz[k] + a * SzSp[k, j] - 2 * Sp[j] * a * Sz[k]
            Y2s = ca * conj(SzSp[j, k]) + Sz[j] * adSm[k] + conj(Sp[k]) * adSz[j] - 2 * ca * Sz[j] * conj(Sp[k])
            dSzSz[j, k] = -im * gj * Y1s + im * gj * Y2 - im * gk * Y1 + im * gk * Y2s
        end
    end
    return nothing
end

# CPU transcription of MGPUkernels.jl (muli(z) = i*z). SzSpT := SzSp[k,j].
# One physics truth with rhs_cpu! / mulpath; this locks the device formulas
# without requiring a GPU.
@inline _muli_kernel(z) = Complex(-imag(z), real(z))

function rhs_kernel_replica!(du::AbstractVector{Complex{T}}, u::AbstractVector{Complex{T}},
                             delta0::T, kappa_e::T, kappa_i::T,
                             delta_b::AbstractVector{T}, g_b::AbstractVector{T},
                             Et::Complex{T}) where {T}
    fill!(du, zero(Complex{T}))
    M = length(delta_b)
    kappa_t = kappa_e + kappa_i
    sqrt_ke = sqrt(kappa_e)
    half = T(1) / 2
    a     = u[IDX_a]
    ad_ad = u[IDX_ad_ad]
    ad_a  = u[IDX_ad_a]
    ca    = conj(a)
    cEt   = conj(Et)

    Sp   = @view u[small_range(M, F_Sp)]
    Sz   = @view u[small_range(M, F_Sz)]
    adSp = @view u[small_range(M, F_adSp)]
    adSm = @view u[small_range(M, F_adSm)]
    adSz = @view u[small_range(M, F_adSz)]
    SpSp_s = @view u[small_range(M, F_SpSp_s)]
    SzSp_s = @view u[small_range(M, F_SzSp_s)]
    SmSp_s = @view u[small_range(M, F_SmSp_s)]
    SzSz_s = @view u[small_range(M, F_SzSz_s)]
    nsmall = small_length(M)
    SpSp = reshape(@view(u[nsmall + 1           : nsmall + M*M]),     M, M)
    SzSp = reshape(@view(u[nsmall + M*M + 1     : nsmall + 2M*M]),    M, M)
    SmSp = reshape(@view(u[nsmall + 2M*M + 1    : nsmall + 3M*M]),    M, M)
    SzSz = reshape(@view(u[nsmall + 3M*M + 1    : nsmall + 4M*M]),    M, M)

    S1 = sum(g_b .* conj.(Sp))
    S2 = sum(g_b .* adSp)
    S3 = sum(g_b .* adSm)
    du[IDX_a]     = sqrt_ke * Et - _muli_kernel(delta0 * a) - _muli_kernel(S1) - half * kappa_t * a
    du[IDX_ad_ad] = 2 * _muli_kernel(delta0 * ad_ad) + 2 * _muli_kernel(S2) -
                    kappa_t * ad_ad + 2 * sqrt_ke * ca * cEt
    du[IDX_ad_a]  = _muli_kernel(conj(S3)) - _muli_kernel(S3) - kappa_t * ad_a +
                    sqrt_ke * Et * ca + sqrt_ke * cEt * a

    dSp   = @view du[small_range(M, F_Sp)]
    dSz   = @view du[small_range(M, F_Sz)]
    dadSp = @view du[small_range(M, F_adSp)]
    dadSm = @view du[small_range(M, F_adSm)]
    dadSz = @view du[small_range(M, F_adSz)]
    dSpSp_s = @view du[small_range(M, F_SpSp_s)]
    dSzSp_s = @view du[small_range(M, F_SzSp_s)]
    dSmSp_s = @view du[small_range(M, F_SmSp_s)]
    dSzSz_s = @view du[small_range(M, F_SzSz_s)]
    dSpSp = reshape(@view(du[nsmall + 1          : nsmall + M*M]),    M, M)
    dSzSp = reshape(@view(du[nsmall + M*M + 1    : nsmall + 2M*M]),   M, M)
    dSmSp = reshape(@view(du[nsmall + 2M*M + 1   : nsmall + 3M*M]),   M, M)
    dSzSz = reshape(@view(du[nsmall + 3M*M + 1   : nsmall + 4M*M]),   M, M)

    @inbounds for j in 1:M
        gj = g_b[j]; dj = delta_b[j]
        Spj = Sp[j]; Szj = Sz[j]
        adSpj = adSp[j]; adSmj = adSm[j]; adSzj = adSz[j]
        cSpj = conj(Spj); cadSmj = conj(adSmj); cadSzj = conj(adSzj)
        sumP = gj * SpSp_s[j]; sumM = gj * SmSp_s[j]; sumZ = gj * SzSp_s[j]
        for k in 1:M
            k == j && continue
            gk = g_b[k]; dk = delta_b[k]
            P = SpSp[j, k]; Z = SzSp[j, k]; ZT = SzSp[k, j]; Mm = SmSp[j, k]; ZZ = SzSz[j, k]
            sumP += gk * P; sumM += gk * Mm; sumZ += gk * Z
            Spk = Sp[k]; Szk = Sz[k]
            adSpk = adSp[k]; adSmk = adSm[k]; adSzk = adSz[k]
            cSpk = conj(Spk); cadSmk = conj(adSmk); cadSzk = conj(adSzk)

            W  = Spk * adSzj + ca * Z  + adSpk * Szj - 2 * Spk * ca * Szj
            Ws = Spj * adSzk + ca * ZT + adSpj * Szk - 2 * Spj * ca * Szk
            dSpSp[j, k] = _muli_kernel((dj + dk) * P - 2 * gj * W - 2 * gk * Ws)

            U1  = Spk * cadSmj + Spj * cadSmk + a * P - 2 * Spk * Spj * a
            U2  = Spk * adSmj + ca * Mm + adSpk * cSpj - 2 * Spk * ca * cSpj
            U3  = ZZ * ca + Szk * adSzj + Szj * adSzk - 2 * ca * Szk * Szj
            dSzSp[j, k] = _muli_kernel(dk * Z - gj * U1 + gj * U2 - 2 * gk * U3)

            Y1  = cadSzj * Spk + cadSmk * Szj + a * Z  - 2 * Spk * a * Szj
            Y1s = cadSzk * Spj + cadSmj * Szk + a * ZT - 2 * Spj * a * Szk
            Y2  = ca * conj(ZT) + Szk * adSmj + cSpj * adSzk - 2 * ca * Szk * cSpj
            Y2s = ca * conj(Z)  + Szj * adSmk + cSpk * adSzj - 2 * ca * Szj * cSpk
            dSmSp[j, k] = _muli_kernel((dk - dj) * Mm + 2 * gj * Y1 - 2 * gk * Y2)
            dSzSz[j, k] = _muli_kernel(gj * (Y2 - Y1s) + gk * (Y2s - Y1))
        end
        dSpSp[j, j] = 0; dSzSp[j, j] = 0; dSmSp[j, j] = 0; dSzSz[j, j] = 0

        dSp[j] = _muli_kernel(dj * Spj - 2 * gj * adSzj)
        dSz[j] = _muli_kernel(gj * adSmj - gj * cadSmj)
        dadSp[j] = _muli_kernel(delta0 * adSpj + dj * adSpj + sumP -
                                2 * gj * (2 * ca * adSzj + ad_ad * Szj - 2 * ca * ca * Szj)) -
                   half * kappa_t * adSpj + sqrt_ke * cEt * Spj
        dadSm[j] = _muli_kernel(delta0 * adSmj - dj * adSmj + 2 * gj * Szj + sumM +
                                2 * gj * (cadSzj * ca + a * adSzj + Szj * ad_a - 2 * ca * a * Szj)) -
                   half * kappa_t * adSmj + sqrt_ke * cEt * cSpj
        dadSz[j] = _muli_kernel(delta0 * adSzj + sumZ -
                                gj * (Spj + Spj * ad_a + ca * cadSmj + a * adSpj - 2 * Spj * ca * a) +
                                gj * (2 * ca * adSmj + ad_ad * cSpj - 2 * ca * ca * cSpj)) -
                   half * kappa_t * adSzj + sqrt_ke * cEt * Szj
        dSpSp_s[j] = _muli_kernel(2 * dj * SpSp_s[j] + 2 * gj * adSpj -
                                  4 * gj * (Spj * adSzj + SzSp_s[j] * ca + adSpj * Szj - 2 * Spj * ca * Szj))
        dSzSp_s[j] = _muli_kernel(dj * SzSp_s[j] -
                                  gj * (2 * Spj * cadSmj + a * SpSp_s[j] - 2 * Spj * Spj * a) +
                                  gj * (Spj * adSmj + ca * SmSp_s[j] + adSpj * cSpj - 2 * Spj * ca * cSpj) -
                                  2 * gj * (SzSz_s[j] * ca + 2 * adSzj * Szj - 2 * ca * Szj * Szj))
        Q1 = cadSzj * Spj + SzSp_s[j] * a + cadSmj * Szj - 2 * Spj * a * Szj
        Q2 = ca * conj(SzSp_s[j]) + cSpj * adSzj + adSmj * Szj - 2 * ca * cSpj * Szj
        dSmSp_s[j] = _muli_kernel(2 * gj * Q1 - 2 * gj * Q2)
        dSzSz_s[j] = _muli_kernel(gj * cadSmj - gj * adSmj - 2 * gj * Q1 + 2 * gj * Q2)
    end
    return nothing
end
