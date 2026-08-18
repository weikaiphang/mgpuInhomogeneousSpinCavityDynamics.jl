# ============================================================
# INITIAL CONDITIONS
# ============================================================

function build_u0_gpu_1st_order(M, Nj, initial_condition)
    if initial_condition == :ground
        return build_u0_gpu_1st_order_ground(M, Nj)

    elseif initial_condition == :inverted
        return build_u0_gpu_1st_order_inverted(M, Nj)

    elseif initial_condition == :custom
        return build_u0_gpu_1st_order_custom(M, Nj)

    else
        error(
            "Unknown initial_condition = $(initial_condition). " *
            "Use :ground, :inverted, :empty, or :custom."
        )
    end
end

function build_u0_gpu_1st_order_ground(M, Nj)
    u0 = zeros(ComplexF64, state_length_1st_order(M))

    idx = 1

    # a
    u0[idx] = 0.0 + 0.0im
    idx += 1

    # S+
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # Sz
    u0[idx:idx+M-1] .= - Nj ./ 2
    idx += M

    return CuArray(u0)
end

function build_u0_gpu_1st_order_inverted(M, Nj)
    u0 = zeros(ComplexF64, state_length_1st_order(M))

    idx = 1

    # a
    u0[idx] = 0.0 + 0.0im
    idx += 1

    # S+
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # Sz
    u0[idx:idx+M-1] .= Nj ./ 2
    idx += M

    return CuArray(u0)
end

function build_u0_gpu_1st_order_custom(M, Nj)
    u0 = zeros(ComplexF64, state_length_1st_order(M))

    idx = 1

    # a
    u0[idx] = 0.0 + 0.0im
    idx += 1

    # S+
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # Sz
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    return CuArray(u0)
end