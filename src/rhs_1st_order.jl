_is_gpu(::AbstractArray) = false

# GPU-only cavity source. Host lab/ip paths sum in a scalar loop (0-alloc).
@inline function _cavity_source_dev(g_b, Sp)
    bc = Base.Broadcast.instantiate(
        Base.broadcasted(*, g_b, Base.broadcasted(conj, Sp))
    )
    return Base.mapreduce(identity, +, bc; dims = 1)
end

function rhs_1st_order!(du, u, p, t)
    length(p) >= 8 && p[8] === :ip && return _rhs_1st_order_ip!(du, u, p, t)
    delta0, kappa_e, kappa_i, delta_b_gpu, g_b_gpu, M, E_of_t = p

    κe = kappa_e
    κt = kappa_e + kappa_i
    E_t = E_of_t(t)

    if _is_gpu(u)
        Sp = @view u[IDX1_Sp_start:IDX1_Sp_start + M - 1]
        Sz = @view u[idx1_Sz_start(M):idx1_Sz_start(M) + M - 1]
        dSp, dSz = unpack_state_1st_order_du(du, M)
        a = @view u[IDX1_a:IDX1_a]
        da = @view du[IDX1_a:IDX1_a]
        s = _cavity_source_dev(g_b_gpu, Sp)

        da .=
            (sqrt(κe) * E_t) .-
            (1im * delta0) .* a .-
            (1im .* s) .-
            (0.5 * κt) .* a

        dSp .=
            1im .* delta_b_gpu .* Sp .-
            2im .* g_b_gpu .* conj.(a) .* Sz

        dSz .=
            -1im .* g_b_gpu .* a .* Sp .+
             1im .* g_b_gpu .* conj.(a) .* conj.(Sp)
    else
        # CPU lab: scalar loop, no TLS work buffer, no conj.(Sp) / broadcast
        # temps, no SubArray views. Same arithmetic as the pre-opt broadcast
        # form (adjoint bit-identical test). Warm call is 0-alloc.
        a = u[IDX1_a]
        ac = conj(a)
        s = zero(typeof(a))
        isp = IDX1_Sp_start
        isz = idx1_Sz_start(M)
        @inbounds for j in 1:M
            Spj = u[isp + j - 1]
            Szj = u[isz + j - 1]
            gj = g_b_gpu[j]
            s += gj * conj(Spj)
            du[isp + j - 1] = 1im * delta_b_gpu[j] * Spj - 2im * gj * ac * Szj
            du[isz + j - 1] = -1im * gj * a * Spj + 1im * gj * ac * conj(Spj)
        end
        du[IDX1_a] =
            sqrt(κe) * E_t -
            1im * delta0 * a -
            1im * s -
            0.5 * κt * a
    end

    return nothing
end
