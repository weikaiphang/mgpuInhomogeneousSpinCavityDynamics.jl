
# Canonical 2nd-order cumulant RHS (monolith packing 3+9M+4M²).
# `rhs_cpu!` is a thin wrapper around this function. MGPU device kernels
# are a fused sharded replica, not a second equation set.
function rhs_2nd_order!(du, u, p, t)
    delta0, kappa_e, kappa_i, delta_b_gpu, g_b_gpu, M, diag_mask, E_of_t = p

    (
        a, ad_ad, ad_a,
        Sp, Sz,
        adSp, adSm, adSz,
        SpSp_same, SzSp_same, SmSp_same, SzSz_same,
        SpSp_cross, SzSp_cross, SmSp_cross, SzSz_cross
    ) = unpack_state_2nd_order_u(u, M)

    (
        dSp, dSz,
        dadSp, dadSm, dadSz,
        dSpSp_same, dSzSp_same, dSmSp_same, dSzSz_same,
        dSpSp_cross, dSzSp_cross, dSmSp_cross, dSzSz_cross
    ) = unpack_state_2nd_order_du(du, M)

    κe = kappa_e
    κt = kappa_e + kappa_i
    E_t = E_of_t(t)




    du[IDX2_a] = sqrt(κe) * E_t - 1im * delta0 * a - 1im * sum(g_b_gpu .* conj.(Sp)) - 0.5 * κt * a




    dSp .= 1im .* delta_b_gpu .* Sp - 2im .* g_b_gpu .* adSz
    dSz .= -1im .* g_b_gpu .* conj.(adSm) + 1im .* g_b_gpu .* adSm




    du[IDX2_ad_ad] = 2im * delta0 * ad_ad + 2im * sum(g_b_gpu .* adSp) - κt * ad_ad + 2 * sqrt(κe) * conj(a) * conj(E_t)
    du[IDX2_ad_a] = 1im * sum(g_b_gpu .* conj.(adSm)) - 1im * sum(g_b_gpu .* adSm) - κt * ad_a + sqrt(κe) * E_t * conj(a) + sqrt(κe) * conj(E_t) * a




    # Same-bin moments live in *_same, not on the cross-block diagonal
    # (diag_mask = .!I). The MGPU kernels skip k == j in the row-sum;
    # the monolith must do the same or custom / random states diverge.
    sumgSpSp_jk = SpSp_same .* g_b_gpu .+ (SpSp_cross .* diag_mask) * g_b_gpu
    sumgSmSp_jk = SmSp_same .* g_b_gpu .+ (SmSp_cross .* diag_mask) * g_b_gpu
    sumgSzSp_jk = SzSp_same .* g_b_gpu .+ (SzSp_cross .* diag_mask) * g_b_gpu

    dadSp .= (
    1im * delta0 .* adSp .+ 1im .* delta_b_gpu .* adSp .+ 1im .* sumgSpSp_jk
    .- 0.5 .* κt .* adSp .+ sqrt(κe) .* conj(E_t) .* Sp
    .- 2im .* g_b_gpu .* (2 .* conj(a) .* adSz .+ ad_ad .* Sz .- 2 .* conj(a).^2 .* Sz)
    )

    dadSm .= (
    1im * delta0 .* adSm .- 1im .* delta_b_gpu .* adSm
    .+ 2im .* g_b_gpu .* Sz .+ 1im .* sumgSmSp_jk
    .- 0.5 .* κt .* adSm .+ sqrt(κe) .* conj(E_t) .* conj.(Sp)
    .+ 2im .* g_b_gpu .* (conj.(adSz) .* conj(a) .+ a .* adSz .+ Sz .* ad_a .- 2 .* conj(a) .* a .* Sz)
    )

    dadSz .= (
    1im * delta0 .* adSz .+ 1im .* sumgSzSp_jk
    .- 0.5 .* κt .* adSz .+ sqrt(κe) .* conj(E_t) .* Sz
    .- 1im .* g_b_gpu .* (Sp .+ Sp .* ad_a .+ conj(a) .* conj.(adSm) .+ a .* adSp .- 2 .* Sp .* conj(a) .* a)
    .+ 1im .* g_b_gpu .* (2 .* conj(a) .* adSm .+ ad_ad .* conj.(Sp) .- 2 .* conj(a).^2 .* conj.(Sp))
    )




    dSpSp_same .= (
    2im .* delta_b_gpu .* SpSp_same .+ 2im .* g_b_gpu .* adSp
    .- 4im .* g_b_gpu .* (Sp .* adSz .+ SzSp_same .* conj(a) .+ adSp .* Sz .- 2 .* Sp .* conj(a) .* Sz)
    )

    dSzSp_same .= (
    1im .* delta_b_gpu .* SzSp_same
    .- 1im .* g_b_gpu .* (2 .* Sp .* conj.(adSm) .+ a .* SpSp_same .- 2 .* Sp.^2 .* a)
    .+ 1im .* g_b_gpu .* (Sp .* adSm .+ conj(a) .* SmSp_same .+ adSp .* conj.(Sp) .- 2 .* Sp .* conj(a) .* conj.(Sp))
    .- 2im .* g_b_gpu .* (SzSz_same .* conj(a) .+ 2 .* adSz .* Sz .- 2 .* conj(a) .* Sz.^2)
    )

    dSmSp_same .= (
    2im .* g_b_gpu .* (conj.(adSz) .* Sp .+ SzSp_same .* a .+ conj.(adSm) .* Sz .- 2 .* Sp .* a .* Sz)
    .- 2im .* g_b_gpu .* (conj(a) .* conj.(SzSp_same) .+ conj.(Sp) .* adSz .+ adSm .* Sz .- 2 .* conj(a) .* conj.(Sp) .* Sz)
    )

    dSzSz_same .= (
    1im .* g_b_gpu .* conj.(adSm) .- 1im .* g_b_gpu .* adSm
    .- 2im .* g_b_gpu .* (conj.(adSz) .* Sp .+ SzSp_same .* a .+ conj.(adSm) .* Sz .- 2 .* Sp .* a .* Sz)
    .+ 2im .* g_b_gpu .* (conj(a) .* conj.(SzSp_same) .+ conj.(Sp) .* adSz .+ adSm .* Sz .- 2 .* conj(a) .* conj.(Sp) .* Sz)
    )




    Δ_col = reshape(delta_b_gpu, M, 1)
    Δ_row = reshape(delta_b_gpu, 1, M)

    G_col = reshape(g_b_gpu, M, 1)
    G_row = reshape(g_b_gpu, 1, M)

    Sp_col = reshape(Sp, M, 1)
    Sp_row = reshape(Sp, 1, M)

    Sz_col = reshape(Sz, M, 1)
    Sz_row = reshape(Sz, 1, M)

    adSp_col = reshape(adSp, M, 1)
    adSp_row = reshape(adSp, 1, M)

    adSm_col = reshape(adSm, M, 1)
    adSm_row = reshape(adSm, 1, M)

    adSz_col = reshape(adSz, M, 1)
    adSz_row = reshape(adSz, 1, M)

    dSpSp_cross .= (
    1im .* (Δ_col .+ Δ_row) .* SpSp_cross
    .- 2im .* G_col .* (Sp_row .* adSz_col .+ conj(a) .* SzSp_cross .+ adSp_row .* Sz_col .- 2 .* Sp_row .* conj(a) .* Sz_col)
    .- 2im .* G_row .* (Sp_col .* adSz_row .+ conj(a) .* transpose(SzSp_cross) .+ Sz_row .* adSp_col .- 2 .* Sp_col .* conj(a) .* Sz_row)
    )
    dSpSp_cross .*= diag_mask

    dSzSp_cross .= (
    1im .* Δ_row .* SzSp_cross
    .- 1im .* G_col .* (Sp_row .* conj.(adSm_col) .+ Sp_col .* conj.(adSm_row) .+ a .* SpSp_cross .- 2 .* Sp_row .* Sp_col .* a)
    .+ 1im .* G_col .* (Sp_row .* adSm_col .+ conj(a) .* SmSp_cross .+ adSp_row .* conj.(Sp_col) .- 2 .* Sp_row .* conj(a) .* conj.(Sp_col))
    .- 2im .* G_row .* (SzSz_cross .* conj(a) .+ Sz_row .* adSz_col .+ Sz_col .* adSz_row .- 2 .* conj(a) .* Sz_row .* Sz_col)
    )
    dSzSp_cross .*= diag_mask

    dSmSp_cross .= (
    .- 1im .* Δ_col .* SmSp_cross .+ 1im .* Δ_row .* SmSp_cross
    .+ 2im .* G_col .* (conj.(adSz_col) .* Sp_row .+ conj.(adSm_row) .* Sz_col .+ a .* SzSp_cross .- 2 .* Sp_row .* a .* Sz_col)
    .- 2im .* G_row .* (conj(a) .* conj.(transpose(SzSp_cross)) .+ Sz_row .* adSm_col .+ conj.(Sp_col) .* adSz_row .- 2 .* conj(a) .* Sz_row .* conj.(Sp_col))
    )
    dSmSp_cross .*= diag_mask

    dSzSz_cross .= (
    .- 1im .* G_col .* (Sp_col .* conj.(adSz_row) .+ conj.(adSm_col) .* Sz_row .+ a .* transpose(SzSp_cross) .- 2 .* Sp_col .* Sz_row .* a)
    .+ 1im .* G_col .* (conj(a) .* conj.(transpose(SzSp_cross)) .+ Sz_row .* adSm_col .+ conj.(Sp_col) .* adSz_row .- 2 .* conj(a) .* Sz_row .* conj.(Sp_col))
    .- 1im .* G_row .* (conj.(adSz_col) .* Sp_row .+ conj.(adSm_row) .* Sz_col .+ a .* SzSp_cross .- 2 .* Sp_row .* a .* Sz_col)
    .+ 1im .* G_row .* (conj.(SzSp_cross) .* conj(a) .+ adSm_row .* Sz_col .+ adSz_col .* conj.(Sp_row) .- 2 .* conj(a) .* Sz_col .* conj.(Sp_row))
    )
    dSzSz_cross .*= diag_mask

    return nothing
end

