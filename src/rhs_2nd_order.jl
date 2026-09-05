# 2nd-order RHS conventions (do not change):
#   ∂t a  = √κ_e E − iδ₀ a − i∑ g S⁻ − (κ/2) a     with κ = κ_e + κ_i, S⁻ = (S⁺)*
#   ∂t S⁺ = iΔ S⁺ − 2i g ⟨a† Sᶻ⟩
#   ∂t Sᶻ = −i g ⟨a S⁺⟩ + i g ⟨a† S⁻⟩
#   a_out = E − √κ_e a;  Sᶻ = ±½ per spin; no spin T₁/T₂ on this RHS.
# Workspace kills per-call O(M)/O(M²) temporaries (g .* conj(Sp), matvecs, …).
mutable struct RHS2Workspace{V}
    gconjSp::V
    g_adSp::V
    g_adSm::V
    sumgSpSp::V
    sumgSmSp::V
    sumgSzSp::V
end

function _rhs2_workspace(u, M)
    CT = eltype(u)
    alloc = u isa Array ? (n -> zeros(CT, n)) : (n -> CUDA.zeros(CT, n))
    return RHS2Workspace(alloc(M), alloc(M), alloc(M), alloc(M), alloc(M), alloc(M))
end

# Cross[j,j] is unused (same-bin lives in *_same). Kernels skip k==j.
# mul!(sumg, cross, g) + same .* g would double-count a dirty diagonal.
@inline function _rowsum_same_plus_cross!(sumg, same, cross, g, diag_mask)
    diag_mask === nothing && error("diag_mask is required for monolith row-sums")
    mul!(sumg, cross .* diag_mask, g)
    @. sumg = same * g + sumg
    return nothing
end

function rhs_2nd_order!(du, u, p, t)
    delta0, kappa_e, kappa_i, delta_b_gpu, g_b_gpu, M = p[1], p[2], p[3], p[4], p[5], p[6]
    E_of_t = p[8]
    E_t = E_of_t(t)

    # CPU / Array path: same equations as MGPUrhs_cpu! (parity tests).
    # Dispatch on Array so this file can be tested without CUDA loaded.
    if u isa Array
        T = real(eltype(u))
        return rhs_cpu!(du, u, T(delta0), T(kappa_e), T(kappa_i),
                        delta_b_gpu, g_b_gpu, Complex{T}(E_t))
    end
    return _rhs_2nd_order_mulpath!(du, u, p, t)
end

# Broadcast / mul! monolith body (CuArray in production; Array in dirty-diag tests).
function _rhs_2nd_order_mulpath!(du, u, p, t)
    delta0, kappa_e, kappa_i, delta_b_gpu, g_b_gpu, M = p[1], p[2], p[3], p[4], p[5], p[6]
    E_t = p[8](t)

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

    diag_mask = p[7]
    ws = length(p) >= 9 ? p[9] : _rhs2_workspace(u, M)

    κe = kappa_e
    κt = kappa_e + kappa_i

    @. ws.gconjSp = g_b_gpu * conj(Sp)
    @. ws.g_adSp = g_b_gpu * adSp
    @. ws.g_adSm = g_b_gpu * adSm

    du[IDX2_a] = sqrt(κe) * E_t - 1im * delta0 * a - 1im * sum(ws.gconjSp) - 0.5 * κt * a

    @. dSp = 1im * delta_b_gpu * Sp - 2im * g_b_gpu * adSz
    @. dSz = -1im * g_b_gpu * conj(adSm) + 1im * g_b_gpu * adSm

    du[IDX2_ad_ad] = 2im * delta0 * ad_ad + 2im * sum(ws.g_adSp) - κt * ad_ad +
                     2 * sqrt(κe) * conj(a) * conj(E_t)
    du[IDX2_ad_a] = 1im * sum(conj, ws.g_adSm) - 1im * sum(ws.g_adSm) - κt * ad_a +
                    sqrt(κe) * E_t * conj(a) + sqrt(κe) * conj(E_t) * a

    _rowsum_same_plus_cross!(ws.sumgSpSp, SpSp_same, SpSp_cross, g_b_gpu, diag_mask)
    _rowsum_same_plus_cross!(ws.sumgSmSp, SmSp_same, SmSp_cross, g_b_gpu, diag_mask)
    _rowsum_same_plus_cross!(ws.sumgSzSp, SzSp_same, SzSp_cross, g_b_gpu, diag_mask)

    @. dadSp = (
        1im * delta0 * adSp + 1im * delta_b_gpu * adSp + 1im * ws.sumgSpSp
        - 0.5 * κt * adSp + sqrt(κe) * conj(E_t) * Sp
        - 2im * g_b_gpu * (2 * conj(a) * adSz + ad_ad * Sz - 2 * conj(a)^2 * Sz)
    )

    @. dadSm = (
        1im * delta0 * adSm - 1im * delta_b_gpu * adSm
        + 2im * g_b_gpu * Sz + 1im * ws.sumgSmSp
        - 0.5 * κt * adSm + sqrt(κe) * conj(E_t) * conj(Sp)
        + 2im * g_b_gpu * (conj(adSz) * conj(a) + a * adSz + Sz * ad_a - 2 * conj(a) * a * Sz)
    )

    @. dadSz = (
        1im * delta0 * adSz + 1im * ws.sumgSzSp
        - 0.5 * κt * adSz + sqrt(κe) * conj(E_t) * Sz
        - 1im * g_b_gpu * (Sp + Sp * ad_a + conj(a) * conj(adSm) + a * adSp - 2 * Sp * conj(a) * a)
        + 1im * g_b_gpu * (2 * conj(a) * adSm + ad_ad * conj(Sp) - 2 * conj(a)^2 * conj(Sp))
    )

    @. dSpSp_same = (
        2im * delta_b_gpu * SpSp_same + 2im * g_b_gpu * adSp
        - 4im * g_b_gpu * (Sp * adSz + SzSp_same * conj(a) + adSp * Sz - 2 * Sp * conj(a) * Sz)
    )

    @. dSzSp_same = (
        1im * delta_b_gpu * SzSp_same
        - 1im * g_b_gpu * (2 * Sp * conj(adSm) + a * SpSp_same - 2 * Sp^2 * a)
        + 1im * g_b_gpu * (Sp * adSm + conj(a) * SmSp_same + adSp * conj(Sp) - 2 * Sp * conj(a) * conj(Sp))
        - 2im * g_b_gpu * (SzSz_same * conj(a) + 2 * adSz * Sz - 2 * conj(a) * Sz^2)
    )

    @. dSmSp_same = (
        2im * g_b_gpu * (conj(adSz) * Sp + SzSp_same * a + conj(adSm) * Sz - 2 * Sp * a * Sz)
        - 2im * g_b_gpu * (conj(a) * conj(SzSp_same) + conj(Sp) * adSz + adSm * Sz - 2 * conj(a) * conj(Sp) * Sz)
    )

    @. dSzSz_same = (
        1im * g_b_gpu * conj(adSm) - 1im * g_b_gpu * adSm
        - 2im * g_b_gpu * (conj(adSz) * Sp + SzSp_same * a + conj(adSm) * Sz - 2 * Sp * a * Sz)
        + 2im * g_b_gpu * (conj(a) * conj(SzSp_same) + conj(Sp) * adSz + adSm * Sz - 2 * conj(a) * conj(Sp) * Sz)
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
    # Hoist the matrix transpose. `@. transpose(SzSp)` becomes transpose.(scalars)
    # (a no-op) and was the full-du leak vs rhs_cpu! / kernels (ZT = SzSp[k,j]).
    SzSp_T = transpose(SzSp_cross)

    @. dSpSp_cross = diag_mask * (
        1im * (Δ_col + Δ_row) * SpSp_cross
        - 2im * G_col * (Sp_row * adSz_col + conj(a) * SzSp_cross + adSp_row * Sz_col - 2 * Sp_row * conj(a) * Sz_col)
        - 2im * G_row * (Sp_col * adSz_row + conj(a) * SzSp_T + Sz_row * adSp_col - 2 * Sp_col * conj(a) * Sz_row)
    )

    @. dSzSp_cross = diag_mask * (
        1im * Δ_row * SzSp_cross
        - 1im * G_col * (Sp_row * conj(adSm_col) + Sp_col * conj(adSm_row) + a * SpSp_cross - 2 * Sp_row * Sp_col * a)
        + 1im * G_col * (Sp_row * adSm_col + conj(a) * SmSp_cross + adSp_row * conj(Sp_col) - 2 * Sp_row * conj(a) * conj(Sp_col))
        - 2im * G_row * (SzSz_cross * conj(a) + Sz_row * adSz_col + Sz_col * adSz_row - 2 * conj(a) * Sz_row * Sz_col)
    )

    @. dSmSp_cross = diag_mask * (
        -1im * Δ_col * SmSp_cross + 1im * Δ_row * SmSp_cross
        + 2im * G_col * (conj(adSz_col) * Sp_row + conj(adSm_row) * Sz_col + a * SzSp_cross - 2 * Sp_row * a * Sz_col)
        - 2im * G_row * (conj(a) * conj(SzSp_T) + Sz_row * adSm_col + conj(Sp_col) * adSz_row - 2 * conj(a) * Sz_row * conj(Sp_col))
    )

    @. dSzSz_cross = diag_mask * (
        -1im * G_col * (Sp_col * conj(adSz_row) + conj(adSm_col) * Sz_row + a * SzSp_T - 2 * Sp_col * Sz_row * a)
        + 1im * G_col * (conj(a) * conj(SzSp_T) + Sz_row * adSm_col + conj(Sp_col) * adSz_row - 2 * conj(a) * Sz_row * conj(Sp_col))
        - 1im * G_row * (conj(adSz_col) * Sp_row + conj(adSm_row) * Sz_col + a * SzSp_cross - 2 * Sp_row * a * Sz_col)
        + 1im * G_row * (conj(SzSp_cross) * conj(a) + adSm_row * Sz_col + adSz_col * conj(Sp_row) - 2 * conj(a) * Sz_col * conj(Sp_row))
    )

    return nothing
end
