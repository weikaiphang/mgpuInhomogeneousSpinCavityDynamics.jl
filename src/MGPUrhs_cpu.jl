
function rhs_cpu!(du::AbstractVector{Complex{T}}, u::AbstractVector{Complex{T}},
                  delta0::T, kappa_e::T, kappa_i::T,
                  delta_b::AbstractVector{T}, g_b::AbstractVector{T},
                  Et::Complex{T}) where {T}
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

    S1 = sum(g_b .* conj.(Sp))
    S2 = sum(g_b .* adSp)
    S3 = sum(g_b .* adSm)

    du[IDX_a]     = sqrt_ke * Et - im * delta0 * a - im * S1 - half * kappa_t * a
    du[IDX_ad_ad] = 2im * delta0 * ad_ad + 2im * S2 - kappa_t * ad_ad + 2 * sqrt_ke * ca * cEt
    du[IDX_ad_a]  = im * conj(S3) - im * S3 - kappa_t * ad_a +
                    sqrt_ke * Et * ca + sqrt_ke * cEt * a

    @inbounds for j in 1:M
        gj = g_b[j]
        dj = delta_b[j]

        dSp[j] = im * dj * Sp[j] - 2im * gj * adSz[j]
        dSz[j] = -im * gj * conj(adSm[j]) + im * gj * adSm[j]

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

    @inbounds for j in 1:M, k in 1:M
        if j == k
            dSpSp[j, k] = 0
            dSzSp[j, k] = 0
            dSmSp[j, k] = 0
            dSzSz[j, k] = 0
            continue
        end

        gj = g_b[j]; gk = g_b[k]
        dj = delta_b[j]; dk = delta_b[k]

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

    return nothing
end
