# ============================================================
# Real isomorphic split of the 1st-order mean-field state, and the
# Jacobian-transpose (VJP) of rhs_1st_order! as an R^{2N} → R^{2N} map.
#
# Layout (N = 1+2M complex = a, Sp[1:M], Sz[1:M]):
#   x = [Re(a), Im(a), Re(Sp)..., Im(Sp)..., Re(Sz)..., Im(Sz)...]
# length 2N = 2+4M. This matches the stored Complex state, including
# Im(Sz), which the Complex layout carries even though dSz is real for
# real g. Do NOT drop Im(Sz) here — that would be a different map than
# the production RHS.
#
# rhs_1st_order_real! is a pack/call/unpack wrapper around the unmodified
# rhs_1st_order!, so F_x is algebraically the production vector field.
# rhs_1st_order_vjp! is the analytic J_x^T (real g, real detunings — the
# ensemble's g_b/delta_b). It is verified against ForwardDiff of F_x on
# the toy ensemble; it is NOT used to replace rhs_1st_order! on the
# forward solve.
# ============================================================

real_state_length_1st_order(M::Integer) = 2 * state_length_1st_order(M)

@inline _real_idx_ar() = 1
@inline _real_idx_ai() = 2
@inline _real_idx_pr(j, M) = 2 + j
@inline _real_idx_pi(j, M) = 2 + M + j
@inline _real_idx_zr(j, M) = 2 + 2M + j
@inline _real_idx_zi(j, M) = 2 + 3M + j

function pack_state_real(a::Number, Sp::AbstractVector, Sz::AbstractVector)
    M = length(Sp)
    length(Sz) == M || error("pack_state_real: Sp/Sz lengths $(length(Sp))/$(length(Sz)) must match.")
    T = promote_type(real(typeof(a)), real(eltype(Sp)), real(eltype(Sz)))
    x = Vector{T}(undef, real_state_length_1st_order(M))
    pack_state_real!(x, a, Sp, Sz)
    return x
end

function pack_state_real!(x::AbstractVector, a::Number, Sp::AbstractVector, Sz::AbstractVector)
    M = length(Sp)
    length(x) == real_state_length_1st_order(M) || error(
        "pack_state_real!: x has length $(length(x)), expected $(real_state_length_1st_order(M))."
    )
    length(Sz) == M || error("pack_state_real!: Sp/Sz lengths must match.")
    x[1] = real(a)
    x[2] = imag(a)
    @inbounds for j in 1:M
        x[_real_idx_pr(j, M)] = real(Sp[j])
        x[_real_idx_pi(j, M)] = imag(Sp[j])
        x[_real_idx_zr(j, M)] = real(Sz[j])
        x[_real_idx_zi(j, M)] = imag(Sz[j])
    end
    return x
end

function pack_state_real!(x::AbstractVector, u::AbstractVector, M::Integer)
    a, Sp, Sz = unpack_state_1st_order_u(u, M)
    return pack_state_real!(x, a, Sp, Sz)
end

function unpack_state_real(x::AbstractVector, M::Integer)
    length(x) == real_state_length_1st_order(M) || error(
        "unpack_state_real: x has length $(length(x)), expected $(real_state_length_1st_order(M))."
    )
    T = eltype(x)
    a = complex(x[1], x[2])
    Sp = Vector{Complex{T}}(undef, M)
    Sz = Vector{Complex{T}}(undef, M)
    @inbounds for j in 1:M
        Sp[j] = complex(x[_real_idx_pr(j, M)], x[_real_idx_pi(j, M)])
        Sz[j] = complex(x[_real_idx_zr(j, M)], x[_real_idx_zi(j, M)])
    end
    return a, Sp, Sz
end

function real_to_complex!(u::AbstractVector, x::AbstractVector, M::Integer)
    length(u) == state_length_1st_order(M) || error(
        "real_to_complex!: u has length $(length(u)), expected $(state_length_1st_order(M))."
    )
    length(x) == real_state_length_1st_order(M) || error(
        "real_to_complex!: x has length $(length(x)), expected $(real_state_length_1st_order(M))."
    )
    u[IDX1_a] = complex(x[1], x[2])
    @inbounds for j in 1:M
        u[IDX1_Sp_start + j - 1] = complex(x[_real_idx_pr(j, M)], x[_real_idx_pi(j, M)])
        u[idx1_Sz_start(M) + j - 1] = complex(x[_real_idx_zr(j, M)], x[_real_idx_zi(j, M)])
    end
    return u
end

function complex_to_real!(x::AbstractVector, u::AbstractVector, M::Integer)
    return pack_state_real!(x, u, M)
end

"""
    rhs_1st_order_real!(dx, x, p, t)

Real-split image of `rhs_1st_order!`: unpack `x` → complex `u`, call the
production RHS, pack `du` → `dx`. `p` is the same 7-tuple the complex
RHS uses. Allocates a pair of complex work vectors (used in tests and
the algebraic identity, not on the adjoint hot path).
"""
function rhs_1st_order_real!(dx, x, p, t)
    M = p[6]
    N = state_length_1st_order(M)
    length(x) == length(dx) == real_state_length_1st_order(M) || error(
        "rhs_1st_order_real!: x/dx length $(length(x))/$(length(dx)), expected $(real_state_length_1st_order(M))."
    )
    Tx = eltype(x)
    u = Vector{Complex{Tx}}(undef, N)
    du = Vector{Complex{Tx}}(undef, N)
    real_to_complex!(u, x, M)
    rhs_1st_order!(du, u, p, t)
    pack_state_real!(dx, du, M)
    return nothing
end

"""
    rhs_1st_order_vjp!(x̄, λ, x, p, t)

`x̄ = J_x(x,t)^T λ` for the real-split production RHS, with real `g_b`
and real `delta_b` (the ensemble). `p` is the same 7-tuple; `E_of_t` is
ignored (θ-dependence of the drive is accumulated separately from the
cavity components of `λ`). `x`, `λ`, `x̄` are length-`2N` real vectors.
"""
function rhs_1st_order_vjp!(x̄::AbstractVector, λ::AbstractVector, x::AbstractVector, p, t)
    delta0, kappa_e, kappa_i, delta_b, g_b, M = p[1], p[2], p[3], p[4], p[5], p[6]
    frame = length(p) >= 8 ? p[8] : :lab
    n = real_state_length_1st_order(M)
    length(x) == length(λ) == length(x̄) == n || error(
        "rhs_1st_order_vjp!: x/λ/x̄ lengths $(length(x))/$(length(λ))/$(length(x̄)), expected $n."
    )
    length(delta_b) == M || error("rhs_1st_order_vjp!: delta_b length $(length(delta_b)) != M=$M.")
    length(g_b) == M || error("rhs_1st_order_vjp!: g_b length $(length(g_b)) != M=$M.")

    κt = kappa_e + kappa_i
    halfκ = 0.5 * κt
    δ0 = real(delta0)
    ip = frame === :ip

    ar = x[1]
    ai = x[2]
    λar = λ[1]
    λai = λ[2]

    x̄ar = -halfκ * λar - δ0 * λai
    x̄ai = δ0 * λar - halfκ * λai

    @inbounds for j in 1:M
        gj = Float64(real(g_b[j]))
        δj = Float64(real(delta_b[j]))
        pr = x[_real_idx_pr(j, M)]
        pi_ = x[_real_idx_pi(j, M)]
        zr = x[_real_idx_zr(j, M)]
        zi = x[_real_idx_zi(j, M)]
        λpr = λ[_real_idx_pr(j, M)]
        λpi = λ[_real_idx_pi(j, M)]
        λzr = λ[_real_idx_zr(j, M)]
        # dzi/dt ≡ 0 for real g: λzi does not enter J^T.

        # Interaction picture: J_ip^T = Q · J_G^T · P, where P rotates the
        # S̃⁺ (pr,pi) block by +θ_j (=> lab S⁺, and the covector), J_G^T is
        # the lab VJP with the iδ·S⁺ free-precession self-term dropped, and
        # Q rotates the S̃⁺ rows of the result back by -θ_j. One sincos/bin.
        local c::Float64, s::Float64
        if ip
            s, c = sincos(δj * t)
            pr, pi_ = pr * c - pi_ * s, pr * s + pi_ * c          # P: S̃⁺ -> S⁺
            λpr, λpi = λpr * c - λpi * s, λpr * s + λpi * c        # P: covector
        end

        two_g = 2 * gj
        x̄ar += two_g * (λpr * zi - λpi * zr + λzr * pi_)
        x̄ai += two_g * (-λpr * zr - λpi * zi + λzr * pr)

        if ip
            gpr = -gj * λai + two_g * ai * λzr
            gpi = -gj * λar + two_g * ar * λzr
            x̄[_real_idx_pr(j, M)] = gpr * c + gpi * s             # Q: rotate S̃⁺ rows by -θ_j
            x̄[_real_idx_pi(j, M)] = -gpr * s + gpi * c
        else
            # lab: unchanged term order (bit-identical to the pre-IP version)
            x̄[_real_idx_pr(j, M)] = -gj * λai + δj * λpi + two_g * ai * λzr
            x̄[_real_idx_pi(j, M)] = -gj * λar - δj * λpr + two_g * ar * λzr
        end
        x̄[_real_idx_zr(j, M)] = -two_g * ai * λpr - two_g * ar * λpi
        x̄[_real_idx_zi(j, M)] = two_g * ar * λpr - two_g * ai * λpi
    end
    x̄[1] = x̄ar
    x̄[2] = x̄ai
    return nothing
end
