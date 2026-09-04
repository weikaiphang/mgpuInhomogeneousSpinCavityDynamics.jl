# ============================================================
# INTERACTION-PICTURE (co-rotating) 1st-order RHS.
#
# Stored spin coherence is  S̃⁺_j = S⁺_j · e^{-iδ_j t}  (" trans" / tilde ).
# a and Sz are frame-invariant (only the spins are rotated).
#
# The transform removes the stiff free-precession term:
#
#   lab :  Ṡ⁺_j = i δ_j S⁺_j − 2i g_j a* Sz_j
#   IP  :  Ṡ̃⁺_j =            − 2i g_j a* Sz_j · e^{-iδ_j t}
#
# The cavity source and the Sz equation still use the LAB coherence
# S⁺_j = S̃⁺_j · e^{+iδ_j t}, so the fast timescale is not gone from the RHS
# coefficients -- but it is gone from the *solution* of every far-detuned,
# physically-idle bin (its S̃⁺_j is ~constant), which is what forces the tiny
# Tsit5 steps in the lab frame. See rhs_1st_order.jl for the opt-in hook.
#
# ForwardDiff-safe: t / δ_j / g_j are Float64, only the state carries Dual,
# so cis(δ_j t) is an ordinary ComplexF64 phase factor.
#
# CPU only. The GPU forward path (solver_1st_order.jl) stays lab-frame; a
# GPU array here is a hard error.
# ============================================================

@inline _ip_frame_of(p) = (length(p) >= 8 ? p[8] : :lab)

"""
    ip_spins_to_lab!(Sp, delta_b, t) -> Sp

In place: `Sp[j] *= exp(+i·delta_b[j]·t)`  (S̃⁺ -> lab S⁺). No-op building
block for every observable/metric read on an IP-frame state.
"""
function ip_spins_to_lab!(Sp::AbstractVector, delta_b::AbstractVector, t::Real)
    @inbounds for j in eachindex(Sp)
        Sp[j] *= cis(delta_b[j] * t)
    end
    return Sp
end

"""    lab_spins_to_ip!(Sp, delta_b, t) -> Sp   (`Sp[j] *= exp(-i·delta_b[j]·t)`)."""
function lab_spins_to_ip!(Sp::AbstractVector, delta_b::AbstractVector, t::Real)
    @inbounds for j in eachindex(Sp)
        Sp[j] *= cis(-delta_b[j] * t)
    end
    return Sp
end

"""
    _rotate_real_sp_block!(x, delta_b, t, M; sgn=1) -> x

Rotate the real-split S⁺ block of `x` (`_real_idx_pr`/`_real_idx_pi` per
bin) by `sgn·δ_j·t`. `sgn=+1` is `P` (S̃⁺→S⁺, and the covector transport);
`sgn=-1` is `Q = P⁻¹`. Used to rotate the discrete-adjoint terminal seed
from lab-`S⁺` coordinates into the IP frame the recorded mesh lives in.
"""
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

    Sp = @view u[IDX1_Sp_start:IDX1_Sp_start + M - 1]        # S̃⁺
    Sz = @view u[idx1_Sz_start(M):idx1_Sz_start(M) + M - 1]
    dSp, dSz = unpack_state_1st_order_du(du, M)

    a  = u[IDX1_a]
    κe = kappa_e
    κt = kappa_e + kappa_i
    E_t = E_of_t(t)
    ac = conj(a)

    # Single bin sweep: one cis(δ_j t) per bin (the sincos is the dominant
    # per-bin cost). `dSp[j]` / `dSz[j]` do not depend on the collective
    # source `s`, so they are written here and `du[a]` is finished after.
    s = zero(eltype(u))
    @inbounds for j in 1:M
        gj = g_b[j]
        φ  = cis(delta_b[j] * t)          # e^{+iδt}
        φc = conj(φ)
        Splab = Sp[j] * φ                 # lab S⁺_j
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
