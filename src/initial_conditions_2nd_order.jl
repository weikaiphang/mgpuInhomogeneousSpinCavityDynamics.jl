# Uncorrelated product-state same-bin moments (authoritative):
#   SmSp_same = |Sp|²(1 − 1/Nj) + Nj/2 − Sz
#   SzSz_same = Sz²(1 − 1/Nj) + Nj/4
#   SpSp_same = Sp²(1 − 1/Nj)
#   SzSp_same = Sz*Sp*(1 − 1/Nj)
# Cross j≠k are mean products. Ground (Sp=0, Sz=−Nj/2) ⇒ SmSp_same = Nj
# (equivalently Nj/2 − Sz), so vacuum⊗ground is a 2nd-order fixed point.
# :weak / :weak_inverted use _WEAK_SEED = 1e-3, which sits outside the
# Bloch ball |⟨S⟩| = Nj/2 by design (a seed, not a physical CSS).

@inline function _uncorrelated_same_moments(Sp, Sz, Nj)
    n = float(real(Nj))
    invN = n > 0 ? (1 - inv(n)) : zero(n)
    SpSp = Sp * Sp * invN
    SzSp = Sz * Sp * invN
    SmSp = abs2(Sp) * invN + n / 2 - Sz
    SzSz = Sz * Sz * invN + n / 4
    return SpSp, SzSp, SmSp, SzSz
end

function build_u0_cpu_2nd_order(M, Nj, initial_condition)
    if initial_condition == :ground
        return _u0_2nd_order_from_means(M, Nj, zero.(Nj), .-Nj ./ 2)
    elseif initial_condition == :inverted
        return _u0_2nd_order_from_means(M, Nj, zero.(Nj), Nj ./ 2)
    elseif initial_condition == :equator
        return _u0_2nd_order_from_means(M, Nj, Nj ./ 2, zero.(Nj))
    elseif initial_condition == :weak
        return _u0_2nd_order_from_means(M, Nj, _WEAK_SEED .* Nj ./ 2, .-Nj ./ 2)
    elseif initial_condition == :weak_inverted
        return _u0_2nd_order_from_means(M, Nj, _WEAK_SEED .* Nj ./ 2, Nj ./ 2)
    elseif initial_condition == :custom
        return zeros(ComplexF64, state_length_2nd_order(M))
    else
        _unknown_initial_condition(initial_condition)
    end
end

function build_u0_gpu_2nd_order(M, Nj, initial_condition)
    return CuArray(build_u0_cpu_2nd_order(M, Nj, initial_condition))
end

const build_u0_gpu_2nd_order_ground = (M, Nj) -> build_u0_gpu_2nd_order(M, Nj, :ground)
const build_u0_gpu_2nd_order_inverted = (M, Nj) -> build_u0_gpu_2nd_order(M, Nj, :inverted)
const build_u0_gpu_2nd_order_equator = (M, Nj) -> build_u0_gpu_2nd_order(M, Nj, :equator)
const build_u0_gpu_2nd_order_weak = (M, Nj) -> build_u0_gpu_2nd_order(M, Nj, :weak)
const build_u0_gpu_2nd_order_weak_inverted = (M, Nj) -> build_u0_gpu_2nd_order(M, Nj, :weak_inverted)
const build_u0_gpu_2nd_order_custom = (M, Nj) -> build_u0_gpu_2nd_order(M, Nj, :custom)

function _u0_2nd_order_from_means(M, Nj, Sp0, Sz0)
    length(Nj) == M || error("length(Nj) = $(length(Nj)) ≠ M = $M")
    length(Sp0) == M || error("length(Sp0) = $(length(Sp0)) ≠ M = $M")
    length(Sz0) == M || error("length(Sz0) = $(length(Sz0)) ≠ M = $M")

    u0 = zeros(ComplexF64, state_length_2nd_order(M))
    idx = 1

    u0[idx] = 0; idx += 1
    u0[idx] = 0; idx += 1
    u0[idx] = 0; idx += 1

    u0[idx:idx+M-1] .= ComplexF64.(Sp0); idx += M
    u0[idx:idx+M-1] .= ComplexF64.(Sz0); idx += M

    u0[idx:idx+M-1] .= 0; idx += M
    u0[idx:idx+M-1] .= 0; idx += M
    u0[idx:idx+M-1] .= 0; idx += M

    @inbounds for j in 1:M
        SpSp, SzSp, SmSp, SzSz = _uncorrelated_same_moments(Sp0[j], Sz0[j], Nj[j])
        u0[idx + j - 1]         = ComplexF64(SpSp)
        u0[idx + M + j - 1]     = ComplexF64(SzSp)
        u0[idx + 2M + j - 1]    = ComplexF64(SmSp)
        u0[idx + 3M + j - 1]    = ComplexF64(SzSz)
    end
    idx += 4M

    mat = zeros(ComplexF64, M, M)
    @inbounds for k in 1:M, j in 1:M
        j == k && continue
        mat[j, k] = ComplexF64(Sp0[j] * Sp0[k])
    end
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M * M

    fill!(mat, 0)
    @inbounds for k in 1:M, j in 1:M
        j == k && continue
        mat[j, k] = ComplexF64(Sz0[j] * Sp0[k])
    end
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M * M

    fill!(mat, 0)
    @inbounds for k in 1:M, j in 1:M
        j == k && continue
        mat[j, k] = ComplexF64(conj(Sp0[j]) * Sp0[k])
    end
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M * M

    fill!(mat, 0)
    @inbounds for k in 1:M, j in 1:M
        j == k && continue
        mat[j, k] = ComplexF64(Sz0[j] * Sz0[k])
    end
    u0[idx:idx+M*M-1] .= vec(mat)

    return u0
end
