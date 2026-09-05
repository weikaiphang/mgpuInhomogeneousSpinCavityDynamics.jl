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


function unpack_state_1st_order_du(du, M)
    idx = 1


    idx += 1

    dSp = @view du[idx:idx+M-1]
    idx += M

    dSz = @view du[idx:idx+M-1]
    idx += M

    return dSp, dSz
end