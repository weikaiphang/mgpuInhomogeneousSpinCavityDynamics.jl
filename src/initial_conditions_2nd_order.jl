# ============================================================
# INITIAL CONDITIONS
# ============================================================

function build_u0_gpu_2nd_order(M, Nj, initial_condition)
    if initial_condition == :ground
        return build_u0_gpu_2nd_order_ground(M, Nj)

    elseif initial_condition == :inverted
        return build_u0_gpu_2nd_order_inverted(M, Nj)

    elseif initial_condition == :custom
        return build_u0_gpu_2nd_order_custom(M, Nj)

    else
        error(
            "Unknown initial_condition = $(initial_condition). " *
            "Use :ground, :inverted, :empty, or :custom."
        )
    end
end

function build_u0_gpu_2nd_order_ground(M, Nj)
    u0 = zeros(ComplexF64, state_length_2nd_order(M))
    idx = 1

    # a
    u0[idx] = 0.0 + 0.0im
    idx += 1

    # a†a†
    u0[idx] = 0.0 + 0.0im
    idx += 1

    # a†a
    u0[idx] = 0.0 + 0.0im
    idx += 1

    # S+
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # Sz
    u0[idx:idx+M-1] .= - Nj ./ 2
    idx += M

    # a†S+
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # a†S-
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # a†Sz
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # S+S+
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # SzS+
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # S-S+
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # SzSz
    u0[idx:idx+M-1] .= Nj.^2 ./ 4
    idx += M

    # S+S+ cross
    mat = zeros(ComplexF64, M, M)
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    # SzS+ cross
    mat .= 0.0
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    # S-S+ cross
    mat .= 0.0
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    # SzSz cross
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

    # a
    u0[idx] = 0.0 + 0.0im
    idx += 1

    # a†a†
    u0[idx] = 0.0 + 0.0im
    idx += 1

    # a†a
    u0[idx] = 0.0 + 0.0im
    idx += 1

    # S+
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # Sz
    u0[idx:idx+M-1] .= Nj ./ 2
    idx += M

    # a†S+
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # a†S-
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # a†Sz
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # S+S+
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # SzS+
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # S-S+
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # SzSz
    u0[idx:idx+M-1] .= Nj.^2 ./ 4
    idx += M

    # S+S+ cross
    mat = zeros(ComplexF64, M, M)
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    # SzS+ cross
    mat .= 0.0
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    # S-S+ cross
    mat .= 0.0
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    # SzSz cross
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

    # a
    u0[idx] = 0.0 + 0.0im
    idx += 1

    # a†a†
    u0[idx] = 0.0 + 0.0im
    idx += 1

    # a†a
    u0[idx] = 0.0 + 0.0im
    idx += 1

    # S+
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # Sz
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # a†S+
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # a†S-
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # a†Sz
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # S+S+
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # SzS+
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # S-S+
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # SzSz
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # S+S+ cross
    mat = zeros(ComplexF64, M, M)
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    # SzS+ cross
    mat .= 0.0
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    # S-S+ cross
    mat .= 0.0
    u0[idx:idx+M*M-1] .= vec(mat)
    idx += M*M

    # SzSz cross
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