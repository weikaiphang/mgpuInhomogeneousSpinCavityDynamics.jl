# Factorized 1st-order Jacobian of the package-owned Tavis–Cummings EOMs.
# This is the QRT / quantum-regression drift: the same mean-field vector field
# as `rhs_1st_order!`, linearized about a saved trajectory (Gardiner & Zoller).
# It is NOT the Jacobian of the 2nd-order cumulant RHS.

function factorized_first_order_eom!(du, u, p, t)
    return rhs_1st_order!(du, u, p, t)
end

function factorized_first_order_jacobian_action!(
    da,
    dadag,
    dSp,
    dSm,
    dSz,
    a_col,
    adag_col,
    Sp_col,
    Sm_col,
    Sz_col,
    a_mean,
    Sp_mean,
    Sz_mean,
    delta0,
    delta_b,
    g_b,
    kappa_t,
)
    g_col = reshape(g_b, :, 1)
    delta_col = reshape(delta_b, :, 1)

    sum_g_Sm = vec(sum(g_col .* Sm_col; dims=1))
    sum_g_Sp = vec(sum(g_col .* Sp_col; dims=1))

    da .= ((-kappa_t / 2 - 1im * delta0) .* a_col) .- (1im .* sum_g_Sm)
    dadag .= ((-kappa_t / 2 + 1im * delta0) .* adag_col) .+ (1im .* sum_g_Sp)

    adag_mean = conj(a_mean)
    a_row = reshape(a_col, 1, :)
    adag_row = reshape(adag_col, 1, :)

    dSp .= 1im .* delta_col .* Sp_col .- 2im .* g_col .* (adag_mean .* Sz_col .+ Sz_mean .* adag_row)
    dSm .= -1im .* delta_col .* Sm_col .+ 2im .* g_col .* (a_mean .* Sz_col .+ Sz_mean .* a_row)
    dSz .= (
        -1im .* g_col .* (a_mean .* Sp_col .+ Sp_mean .* a_row) .+
        1im .* g_col .* (adag_mean .* Sm_col .+ conj.(Sp_mean) .* adag_row)
    )
    return nothing
end
