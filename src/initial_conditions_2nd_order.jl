if !@isdefined(_WEAK_SEED)
    const _WEAK_SEED = 1.0e-3
end


@inline function _samebin_factor(Nj::Real)
    Nj > 0 ? (1 - 1 / Nj) : zero(Nj)
end

@inline function product_SpSp_same(Nj, Sp)
    Sp * Sp * _samebin_factor(real(Nj))
end

@inline function product_SzSp_same(Nj, Sz, Sp)
    Sz * Sp * _samebin_factor(real(Nj))
end

@inline function product_SmSp_same(Nj, Sp, Sz)
    abs2(Sp) * _samebin_factor(real(Nj)) + Nj / 2 - Sz
end

@inline function product_SzSz_same(Nj, Sz)
    abs2(Sz) * _samebin_factor(real(Nj)) + Nj / 4
end

function product_state_samebin(Nj, Sp, Sz)
    return (
        product_SpSp_same(Nj, Sp),
        product_SzSp_same(Nj, Sz, Sp),
        product_SmSp_same(Nj, Sp, Sz),
        product_SzSz_same(Nj, Sz),
    )
end

function _spin_means_2nd_order(Nj, kind::Symbol)
    if kind === :ground
        return zero.(Nj), -Nj ./ 2
    elseif kind === :inverted
        return zero.(Nj), Nj ./ 2
    elseif kind === :equator
        return Nj ./ 2, zero.(Nj)
    elseif kind === :weak
        return _WEAK_SEED .* Nj ./ 2, -Nj ./ 2
    elseif kind === :weak_inverted
        return _WEAK_SEED .* Nj ./ 2, Nj ./ 2
    else
        _unknown_initial_condition(kind)
    end
end

function build_u0_2nd_order(M, Nj, initial_condition)
    if initial_condition === :custom
        return zeros(ComplexF64, state_length_2nd_order(M))
    elseif initial_condition in (:ground, :inverted, :equator, :weak, :weak_inverted)
        Sp0, Sz0 = _spin_means_2nd_order(Nj, initial_condition)
        return _u0_2nd_order_from_means(M, Nj, Sp0, Sz0)
    else
        _unknown_initial_condition(initial_condition)
    end
end

function build_u0_gpu_2nd_order(M, Nj, initial_condition)
    return CuArray(build_u0_2nd_order(M, Nj, initial_condition))
end

build_u0_gpu_2nd_order_ground(M, Nj)         = build_u0_gpu_2nd_order(M, Nj, :ground)
build_u0_gpu_2nd_order_inverted(M, Nj)       = build_u0_gpu_2nd_order(M, Nj, :inverted)
build_u0_gpu_2nd_order_equator(M, Nj)        = build_u0_gpu_2nd_order(M, Nj, :equator)
build_u0_gpu_2nd_order_weak(M, Nj)           = build_u0_gpu_2nd_order(M, Nj, :weak)
build_u0_gpu_2nd_order_weak_inverted(M, Nj)  = build_u0_gpu_2nd_order(M, Nj, :weak_inverted)
build_u0_gpu_2nd_order_custom(M, Nj)         = build_u0_gpu_2nd_order(M, Nj, :custom)

function _u0_2nd_order_from_means(M, Nj, Sp0, Sz0)
    length(Nj) == M || error("length(Nj) = $(length(Nj)) ≠ M = $M")
    length(Sp0) == M || error("length(Sp0) ≠ M")
    length(Sz0) == M || error("length(Sz0) ≠ M")

    u0 = zeros(ComplexF64, state_length_2nd_order(M))
    idx = 1

    u0[idx] = 0
    idx += 1
    u0[idx] = 0
    idx += 1
    u0[idx] = 0
    idx += 1

    u0[idx:idx+M-1] .= ComplexF64.(Sp0)
    idx += M

    u0[idx:idx+M-1] .= ComplexF64.(Sz0)
    idx += M

    u0[idx:idx+M-1] .= 0
    idx += M
    u0[idx:idx+M-1] .= 0
    idx += M
    u0[idx:idx+M-1] .= 0
    idx += M

    @inbounds for j in 1:M
        spsp, szsp, smsp, szsz = product_state_samebin(Nj[j], Sp0[j], Sz0[j])
        u0[idx + j - 1]     = ComplexF64(spsp)
        u0[idx + M + j - 1] = ComplexF64(szsp)
        u0[idx + 2M + j - 1] = ComplexF64(smsp)
        u0[idx + 3M + j - 1] = ComplexF64(szsz)
    end
    idx += 4M

    mat = zeros(ComplexF64, M, M)
    @inbounds for k in 1:M, j in 1:M
        j == k && continue
        mat[j, k] = ComplexF64(Sp0[j] * Sp0[k])
    end
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    fill!(mat, 0)
    @inbounds for k in 1:M, j in 1:M
        j == k && continue
        mat[j, k] = ComplexF64(Sz0[j] * Sp0[k])
    end
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    fill!(mat, 0)
    @inbounds for k in 1:M, j in 1:M
        j == k && continue
        mat[j, k] = ComplexF64(conj(Sp0[j]) * Sp0[k])
    end
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    fill!(mat, 0)
    @inbounds for k in 1:M, j in 1:M
        j == k && continue
        mat[j, k] = ComplexF64(Sz0[j] * Sz0[k])
    end
    u0[idx:idx+M*M-1] .= vec(mat)

    return u0
end
