function _unknown_initial_condition(initial_condition)
    error(
        "Unknown initial_condition = $(initial_condition). " *
        "Use :ground, :inverted, :equator, :weak, :weak_inverted, or :custom."
    )
end

function build_u0_gpu_1st_order(M, Nj, initial_condition)
    if initial_condition == :ground
        return build_u0_gpu_1st_order_ground(M, Nj)

    elseif initial_condition == :inverted
        return build_u0_gpu_1st_order_inverted(M, Nj)

    elseif initial_condition == :equator
        return build_u0_gpu_1st_order_equator(M, Nj)

    elseif initial_condition == :weak
        return build_u0_gpu_1st_order_weak(M, Nj)

    elseif initial_condition == :weak_inverted
        return build_u0_gpu_1st_order_weak_inverted(M, Nj)

    elseif initial_condition == :custom
        return build_u0_gpu_1st_order_custom(M, Nj)

    else
        _unknown_initial_condition(initial_condition)
    end
end

function build_u0_gpu_1st_order_ground(M, Nj)
    u0 = zeros(ComplexF64, state_length_1st_order(M))

    idx = 1

    u0[idx] = 0.0 + 0.0im
    idx += 1

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    u0[idx:idx+M-1] .= - Nj ./ 2
    idx += M

    return CuArray(u0)
end

function build_u0_gpu_1st_order_inverted(M, Nj)
    u0 = zeros(ComplexF64, state_length_1st_order(M))

    idx = 1

    u0[idx] = 0.0 + 0.0im
    idx += 1

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    u0[idx:idx+M-1] .= Nj ./ 2
    idx += M

    return CuArray(u0)
end

function build_u0_gpu_1st_order_custom(M, Nj)
    u0 = zeros(ComplexF64, state_length_1st_order(M))

    idx = 1

    u0[idx] = 0.0 + 0.0im
    idx += 1

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    return CuArray(u0)
end

function build_u0_gpu_1st_order_equator(M, Nj)
    u0 = zeros(ComplexF64, state_length_1st_order(M))
    idx = 1
    u0[idx] = 0.0 + 0.0im
    idx += 1
    u0[idx:idx+M-1] .= Nj ./ 2
    idx += M
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    return CuArray(u0)
end

function build_u0_gpu_1st_order_weak(M, Nj)
    u0 = zeros(ComplexF64, state_length_1st_order(M))
    idx = 1
    u0[idx] = 0.0 + 0.0im
    idx += 1
    u0[idx:idx+M-1] .= _WEAK_SEED .* Nj ./ 2
    idx += M
    u0[idx:idx+M-1] .= - Nj ./ 2
    return CuArray(u0)
end

function build_u0_gpu_1st_order_weak_inverted(M, Nj)
    u0 = zeros(ComplexF64, state_length_1st_order(M))
    idx = 1
    u0[idx] = 0.0 + 0.0im
    idx += 1
    u0[idx:idx+M-1] .= _WEAK_SEED .* Nj ./ 2
    idx += M
    u0[idx:idx+M-1] .= Nj ./ 2
    return CuArray(u0)
end
