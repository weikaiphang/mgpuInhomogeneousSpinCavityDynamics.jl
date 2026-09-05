function build_u0_gpu_2nd_order(M, Nj, initial_condition)
    if initial_condition == :ground
        return build_u0_gpu_2nd_order_ground(M, Nj)

    elseif initial_condition == :inverted
        return build_u0_gpu_2nd_order_inverted(M, Nj)

    elseif initial_condition == :equator
        return build_u0_gpu_2nd_order_equator(M, Nj)

    elseif initial_condition == :weak
        return build_u0_gpu_2nd_order_weak(M, Nj)

    elseif initial_condition == :weak_inverted
        return build_u0_gpu_2nd_order_weak_inverted(M, Nj)

    elseif initial_condition == :custom
        return build_u0_gpu_2nd_order_custom(M, Nj)

    else
        _unknown_initial_condition(initial_condition)
    end
end

function build_u0_gpu_2nd_order_ground(M, Nj)
    u0 = zeros(ComplexF64, state_length_2nd_order(M))
    idx = 1

    u0[idx] = 0.0 + 0.0im
    idx += 1

    u0[idx] = 0.0 + 0.0im
    idx += 1

    u0[idx] = 0.0 + 0.0im
    idx += 1

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    u0[idx:idx+M-1] .= - Nj ./ 2
    idx += M

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    u0[idx:idx+M-1] .= Nj.^2 ./ 4
    idx += M

    mat = zeros(ComplexF64, M, M)
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    mat .= 0.0
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    mat .= 0.0
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    mat .= 0.0
    for j in 1:M
        for k in 1:M
            if j != k
                mat[j, k] = Nj[j] * Nj[k] / 4
            end
        end
    end

    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    return CuArray(u0)
end

function build_u0_gpu_2nd_order_inverted(M, Nj)
    u0 = zeros(ComplexF64, state_length_2nd_order(M))
    idx = 1

    u0[idx] = 0.0 + 0.0im
    idx += 1

    u0[idx] = 0.0 + 0.0im
    idx += 1

    u0[idx] = 0.0 + 0.0im
    idx += 1

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    u0[idx:idx+M-1] .= Nj ./ 2
    idx += M

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    u0[idx:idx+M-1] .= Nj.^2 ./ 4
    idx += M

    mat = zeros(ComplexF64, M, M)
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    mat .= 0.0
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    mat .= 0.0
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    mat .= 0.0
    for j in 1:M
        for k in 1:M
            if j != k
                mat[j, k] = Nj[j] * Nj[k] / 4
            end
        end
    end

    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    return CuArray(u0)
end

function build_u0_gpu_2nd_order_custom(M, Nj)
    u0 = zeros(ComplexF64, state_length_2nd_order(M))
    idx = 1

    u0[idx] = 0.0 + 0.0im
    idx += 1

    u0[idx] = 0.0 + 0.0im
    idx += 1

    u0[idx] = 0.0 + 0.0im
    idx += 1

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    mat = zeros(ComplexF64, M, M)
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    mat .= 0.0
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    mat .= 0.0
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    mat .= 0.0
    for j in 1:M
        for k in 1:M
            if j != k
                mat[j, k] = 0.0 + 0.0im
            end
        end
    end

    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    return CuArray(u0)
end

function _u0_2nd_order_from_means(M, Nj, Sp0, Sz0)
    u0 = zeros(ComplexF64, state_length_2nd_order(M))
    idx = 1

    u0[idx] = 0.0 + 0.0im
    idx += 1
    u0[idx] = 0.0 + 0.0im
    idx += 1
    u0[idx] = 0.0 + 0.0im
    idx += 1

    u0[idx:idx+M-1] .= ComplexF64.(Sp0)
    idx += M

    u0[idx:idx+M-1] .= ComplexF64.(Sz0)
    idx += M

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    u0[idx:idx+M-1] .= ComplexF64.(Sp0 .* Sp0)
    idx += M
    u0[idx:idx+M-1] .= ComplexF64.(Sz0 .* Sp0)
    idx += M
    u0[idx:idx+M-1] .= ComplexF64.(abs2.(Sp0))
    idx += M
    u0[idx:idx+M-1] .= ComplexF64.(Sz0 .* Sz0)
    idx += M

    mat = zeros(ComplexF64, M, M)
    @inbounds for k in 1:M
        for j in 1:M
            if j != k
                mat[j, k] = ComplexF64(Sp0[j] * Sp0[k])
            end
        end
    end
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    fill!(mat, 0)
    @inbounds for k in 1:M
        for j in 1:M
            if j != k
                mat[j, k] = ComplexF64(Sz0[j] * Sp0[k])
            end
        end
    end
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    fill!(mat, 0)
    @inbounds for k in 1:M
        for j in 1:M
            if j != k
                mat[j, k] = ComplexF64(conj(Sp0[j]) * Sp0[k])
            end
        end
    end
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    fill!(mat, 0)
    @inbounds for k in 1:M
        for j in 1:M
            if j != k
                mat[j, k] = ComplexF64(Sz0[j] * Sz0[k])
            end
        end
    end
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    return u0
end

function build_u0_gpu_2nd_order_equator(M, Nj)
    return CuArray(_u0_2nd_order_from_means(M, Nj, Nj ./ 2, zero.(Nj)))
end

function build_u0_gpu_2nd_order_weak(M, Nj)
    return CuArray(_u0_2nd_order_from_means(
        M, Nj, _WEAK_SEED .* Nj ./ 2, -Nj ./ 2,
    ))
end

function build_u0_gpu_2nd_order_weak_inverted(M, Nj)
    return CuArray(_u0_2nd_order_from_means(
        M, Nj, _WEAK_SEED .* Nj ./ 2, Nj ./ 2,
    ))
end