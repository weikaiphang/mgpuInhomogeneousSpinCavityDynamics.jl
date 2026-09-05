
@inline _ip_frame_of(p) = (length(p) >= 8 ? p[8] : :lab)


function ip_spins_to_lab!(Sp::AbstractVector, delta_b::AbstractVector, t::Real)
    @inbounds for j in eachindex(Sp)
        Sp[j] *= cis(delta_b[j] * t)
    end
    return Sp
end


function lab_spins_to_ip!(Sp::AbstractVector, delta_b::AbstractVector, t::Real)
    @inbounds for j in eachindex(Sp)
        Sp[j] *= cis(-delta_b[j] * t)
    end
    return Sp
end


function _rotate_real_sp_block!(x::AbstractVector, delta_b::AbstractVector, t::Real, M::Integer; sgn::Int=1)
    @inbounds for j in 1:M
        θ = sgn * delta_b[j] * t
        c = cos(θ); s = sin(θ)
        pr = x[_real_idx_pr(j, M)]
        pI = x[_real_idx_pi(j, M)]
        x[_real_idx_pr(j, M)] = pr * c - pI * s
        x[_real_idx_pi(j, M)] = pr * s + pI * c
    end
    return x
end

function _rhs_1st_order_ip!(du, u, p, t)
    delta0, kappa_e, kappa_i, delta_b, g_b, M, E_of_t = p[1], p[2], p[3], p[4], p[5], p[6], p[7]
    _is_gpu(u) && error(
        "rhs_1st_order! frame=:ip is CPU-only (pass compute=:cpu); got a GPU array."
    )

    Sp = @view u[IDX1_Sp_start:IDX1_Sp_start + M - 1]
    Sz = @view u[idx1_Sz_start(M):idx1_Sz_start(M) + M - 1]
    dSp, dSz = unpack_state_1st_order_du(du, M)

    a  = u[IDX1_a]
    κe = kappa_e
    κt = kappa_e + kappa_i
    E_t = E_of_t(t)
    ac = conj(a)




    s = zero(eltype(u))
    @inbounds for j in 1:M
        gj = g_b[j]
        φ  = cis(delta_b[j] * t)
        φc = conj(φ)
        Splab = Sp[j] * φ
        s += gj * conj(Splab)
        dSp[j] = -2im * gj * ac * Sz[j] * φc
        dSz[j] = -1im * gj * a * Splab + 1im * gj * ac * conj(Splab)
    end

    du[IDX1_a] =
        sqrt(κe) * E_t -
        1im * delta0 * a -
        1im * s -
        0.5 * κt * a
    return nothing
end
