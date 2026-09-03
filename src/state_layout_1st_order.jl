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

# View-only unpack for the hot RHS path (`rhs_1st_order!`).
#
# Identical slicing to `unpack_state_1st_order_u`, EXCEPT the cavity
# amplitude is returned as a 1-element `@view u[IDX1_a:IDX1_a]` instead of
# the scalar `u[IDX1_a]`. On a `CuArray` backend the scalar load `u[1]`
# (and the matching scalar store `du[1] = ...`) are device<->host
# round-trips that force a synchronisation on every RK stage; a 1-element
# view keeps the value on-device and lets the cavity equation be written
# as a length-1 broadcast. On a plain `Vector`/`Dual` backend a 1-element
# `SubArray` is arithmetically transparent, so the vector field is
# unchanged bit-for-bit (the broadcast reduces to the same scalar ops in
# the same order).
#
# `unpack_state_1st_order_u` (scalar `a`) is deliberately left as-is: its
# other callers (`pulse_adjoint.jl`, `pulse_optimizer*.jl`,
# `rhs_1st_order_real.jl`) all operate on host arrays AFTER the solve and
# want `a::Number`.
function unpack_state_1st_order_u_views(u, M)
    idx = 1

    a = @view u[idx:idx]
    idx += 1

    Sp = @view u[idx:idx+M-1]
    idx += M

    Sz = @view u[idx:idx+M-1]
    idx += M

    return a, Sp, Sz
end

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