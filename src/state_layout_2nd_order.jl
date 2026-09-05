const IDX2_a     = 1
const IDX2_ad_ad = 2
const IDX2_ad_a  = 3

const IDX2_Sp_start = 4

idx2_Sz_start(M)   = IDX2_Sp_start + M
idx2_adSp_start(M) = idx2_Sz_start(M) + M
idx2_adSm_start(M) = idx2_adSp_start(M) + M
idx2_adSz_start(M) = idx2_adSm_start(M) + M

state_length_2nd_order(M) =
    3 +
    2M +
    3M +
    4M +
    4M*M

function make_diag_mask(M)
    return CuArray(ComplexF64.(.!Matrix(I, M, M)))
end

function unpack_state_2nd_order_u(u, M)
    idx = 1

    a     = u[idx]; idx += 1
    ad_ad = u[idx]; idx += 1
    ad_a  = u[idx]; idx += 1

    Sp = @view u[idx:idx+M-1]; idx += M
    Sz = @view u[idx:idx+M-1]; idx += M

    adSp = @view u[idx:idx+M-1]; idx += M
    adSm = @view u[idx:idx+M-1]; idx += M
    adSz = @view u[idx:idx+M-1]; idx += M

    SpSp_same = @view u[idx:idx+M-1]; idx += M
    SzSp_same = @view u[idx:idx+M-1]; idx += M
    SmSp_same = @view u[idx:idx+M-1]; idx += M
    SzSz_same = @view u[idx:idx+M-1]; idx += M

    SpSp_cross = reshape(@view(u[idx:idx+M*M-1]), M, M); idx += M*M
    SzSp_cross = reshape(@view(u[idx:idx+M*M-1]), M, M); idx += M*M
    SmSp_cross = reshape(@view(u[idx:idx+M*M-1]), M, M); idx += M*M
    SzSz_cross = reshape(@view(u[idx:idx+M*M-1]), M, M); idx += M*M

    return (
        a, ad_ad, ad_a,
        Sp, Sz,
        adSp, adSm, adSz,
        SpSp_same, SzSp_same, SmSp_same, SzSz_same,
        SpSp_cross, SzSp_cross, SmSp_cross, SzSz_cross
    )
end

function unpack_state_2nd_order_du(du, M)
    idx = 1

    idx += 3

    dSp = @view du[idx:idx+M-1]; idx += M
    dSz = @view du[idx:idx+M-1]; idx += M

    dadSp = @view du[idx:idx+M-1]; idx += M
    dadSm = @view du[idx:idx+M-1]; idx += M
    dadSz = @view du[idx:idx+M-1]; idx += M

    dSpSp_same = @view du[idx:idx+M-1]; idx += M
    dSzSp_same = @view du[idx:idx+M-1]; idx += M
    dSmSp_same = @view du[idx:idx+M-1]; idx += M
    dSzSz_same = @view du[idx:idx+M-1]; idx += M

    dSpSp_cross = reshape(@view(du[idx:idx+M*M-1]), M, M); idx += M*M
    dSzSp_cross = reshape(@view(du[idx:idx+M*M-1]), M, M); idx += M*M
    dSmSp_cross = reshape(@view(du[idx:idx+M*M-1]), M, M); idx += M*M
    dSzSz_cross = reshape(@view(du[idx:idx+M*M-1]), M, M); idx += M*M

    return (
        dSp, dSz,
        dadSp, dadSm, dadSz,
        dSpSp_same, dSzSp_same, dSmSp_same, dSzSz_same,
        dSpSp_cross, dSzSp_cross, dSmSp_cross, dSzSz_cross
    )
end