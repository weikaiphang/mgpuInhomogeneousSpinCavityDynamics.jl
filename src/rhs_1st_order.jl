# ============================================================
# 1st-order RHS
# ============================================================

function rhs_1st_order!(du, u, p, t)
    delta0, kappa_e, kappa_i, delta_b_gpu, g_b_gpu, M, E_of_t = p

    a, Sp, Sz = unpack_state_1st_order_u(u, M)
    dSp, dSz = unpack_state_1st_order_du(du, M)

    κe = kappa_e
    κt = kappa_e + kappa_i
    E_t = E_of_t(t)

    # --------------------------------------------------------
    # Cavity equation
    # --------------------------------------------------------
    du[IDX1_a] =
        sqrt(κe) * E_t -
        1im * delta0 * a -
        1im * sum(g_b_gpu .* conj.(Sp)) -
        0.5 * κt * a

    # --------------------------------------------------------
    # Spin equations
    #
    # Mean-field replacement:
    #     ⟨a† Sz⟩ ≈ conj(a) Sz
    #     ⟨a† S-⟩ ≈ conj(a) conj(S+)
    # --------------------------------------------------------

    dSp .=
        1im .* delta_b_gpu .* Sp .-
        2im .* g_b_gpu .* conj(a) .* Sz

    dSz .=
        -1im .* g_b_gpu .* a .* Sp .+
         1im .* g_b_gpu .* conj(a) .* conj.(Sp)

    return nothing
end