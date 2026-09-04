const IDX1_a = 1
const IDX1_Sp_start = 2

idx1_Sz_start(M) = IDX1_Sp_start + M

state_length_1st_order(M) = 1 + 2M

function unpack_state_1st_order_u(u, M)
    idx = 1

    a = u[idx]
    idx += 1

    Sp = @view u[idx:idx+M-1]
    idx += M

    Sz = @view u[idx:idx+M-1]
    idx += M

    return a, Sp, Sz
end

# `unpack_state_1st_order_u` returns the cavity amplitude as the scalar
# `u[IDX1_a]`. That is what every post-solve host caller
# (`pulse_adjoint.jl`, `pulse_optimizer*.jl`, `rhs_1st_order_real.jl`)
# wants. `rhs_1st_order!`'s GPU branch instead takes `@view u[IDX1_a:IDX1_a]`
# inline -- a 1-element view avoids the `CuArray` scalar load/store that
# would sync every RK stage -- while its CPU/Dual branch keeps the scalar
# form (arithmetically identical, and allocation-free).

function unpack_state_1st_order_du(du, M)
    idx = 1

    # skip scalar derivative du[IDX1_a]
    idx += 1

    dSp = @view du[idx:idx+M-1]
    idx += M

    dSz = @view du[idx:idx+M-1]
    idx += M

    return dSp, dSz
end