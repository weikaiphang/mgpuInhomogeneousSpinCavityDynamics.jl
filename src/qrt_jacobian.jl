# QRT Jacobian for output-mode noise / ASE–RASE correlations.
#
# Product path (`QRT_PRODUCT = :star_samebin_cavity`, Niels):
#   Linearize the 2nd-order cumulant RHS on the 3+9M block only —
#   cavity (a, ⟨a†a†⟩, ⟨a†a⟩) + first moments + cavity–spin moments +
#   same-bin 2nd-order correlators. Inter-bin coupling is rank-1 through
#   the cavity (star truncated cluster). Cross-bin 4M² are not dynamical.
#   Cost is O(M) per apply. This replaces the factorized 1st-order
#   Jacobian ⟨a† Sᶻ⟩ ≈ ⟨a†⟩⟨Sᶻ⟩ that undercuts figs 3d/4c/4d.
#
# Oracle (`QRT_ORACLE = :full_dense`):
#   Real-linear finite difference of `rhs_cpu!` on the full
#   3+9M+4M² state. Apply is O(M²); dense J is for small-M tests only.
#   Never used as the product path in `compute_output_mode_noise` or
#   `compute_ase_rase_correlations_gpu`.
#
# When product matches the oracle:
#   For a tangent with zero cross-bin components, J_small,small is the
#   same for star and full (cross enters the small-field EOMs only as
#   an inhomogeneous row-sum). Tests report abs/rel error via
#   `qrt_relabs_err`. When the tangent has cross components, product
#   and oracle differ — that is the star cut, not a bug.
#
# Saved jld2 trajectories store a, n, adad, Sp, Sz, adSp, adSm, adSz
# but not same-bin or cross. `qrt_pack_product_state!` reconstructs
# same-bin with the H1 product-state closure (`_uncorrelated_same_moments`).
# Drive E(t) is inhomogeneous in f and drops out of J; Et is kept only
# so dad_a / dadSp coefficients that multiply E match the window.

const QRT_PRODUCT = :star_samebin_cavity
const QRT_ORACLE = :full_dense

qrt_product_length(M::Integer) = small_length(M)
qrt_oracle_length(M::Integer) = global_state_length(M)

function qrt_product_params(delta0, kappa_e, kappa_i, delta_b, g_b, Et=0)
    T = promote_type(typeof(delta0), typeof(kappa_e), typeof(kappa_i),
                     eltype(delta_b), eltype(g_b), real(typeof(Et)))
    return (T(delta0), T(kappa_e), T(kappa_i), delta_b, g_b, Complex{T}(Et))
end

# Star RHS on the 3+9M block: same-bin row-sums only (no cross loop).
function qrt_star_rhs!(du::AbstractVector{Complex{T}}, u::AbstractVector{Complex{T}},
                       delta0::T, kappa_e::T, kappa_i::T,
                       delta_b::AbstractVector{T}, g_b::AbstractVector{T},
                       Et::Complex{T}) where {T}
    M = length(delta_b)
    length(u) == length(du) == small_length(M) || error(
        "qrt_star_rhs!: expected length $(small_length(M)), got $(length(u))/$(length(du))."
    )
    kappa_t = kappa_e + kappa_i
    sqrt_ke = sqrt(kappa_e)
    half = T(1) / 2
    a     = u[IDX_a]
    ad_ad = u[IDX_ad_ad]
    ad_a  = u[IDX_ad_a]
    ca    = conj(a)
    cEt   = conj(Et)

    Sp   = @view u[small_range(M, F_Sp)]
    Sz   = @view u[small_range(M, F_Sz)]
    adSp = @view u[small_range(M, F_adSp)]
    adSm = @view u[small_range(M, F_adSm)]
    adSz = @view u[small_range(M, F_adSz)]
    SpSp_s = @view u[small_range(M, F_SpSp_s)]
    SzSp_s = @view u[small_range(M, F_SzSp_s)]
    SmSp_s = @view u[small_range(M, F_SmSp_s)]
    SzSz_s = @view u[small_range(M, F_SzSz_s)]

    dSp   = @view du[small_range(M, F_Sp)]
    dSz   = @view du[small_range(M, F_Sz)]
    dadSp = @view du[small_range(M, F_adSp)]
    dadSm = @view du[small_range(M, F_adSm)]
    dadSz = @view du[small_range(M, F_adSz)]
    dSpSp_s = @view du[small_range(M, F_SpSp_s)]
    dSzSp_s = @view du[small_range(M, F_SzSp_s)]
    dSmSp_s = @view du[small_range(M, F_SmSp_s)]
    dSzSz_s = @view du[small_range(M, F_SzSz_s)]

    S1 = zero(Complex{T})
    S2 = zero(Complex{T})
    S3 = zero(Complex{T})
    @inbounds for j in 1:M
        gj = g_b[j]
        S1 += gj * conj(Sp[j])
        S2 += gj * adSp[j]
        S3 += gj * adSm[j]
    end
    du[IDX_a]     = sqrt_ke * Et - im * delta0 * a - im * S1 - half * kappa_t * a
    du[IDX_ad_ad] = 2im * delta0 * ad_ad + 2im * S2 - kappa_t * ad_ad + 2 * sqrt_ke * ca * cEt
    du[IDX_ad_a]  = im * conj(S3) - im * S3 - kappa_t * ad_a +
                    sqrt_ke * Et * ca + sqrt_ke * cEt * a

    # Same-bin + cavity-mediated only. `sumP = gj * SpSp_s[j]` — the k≠j
    # row-sum is omitted so apply is O(M), not O(M²).
    @inbounds for j in 1:M
        gj = g_b[j]
        dj = delta_b[j]
        dSp[j] = im * dj * Sp[j] - 2im * gj * adSz[j]
        dSz[j] = -im * gj * conj(adSm[j]) + im * gj * adSm[j]
        sumP = gj * SpSp_s[j]
        sumM = gj * SmSp_s[j]
        sumZ = gj * SzSp_s[j]
        dadSp[j] = (im * delta0 * adSp[j] + im * dj * adSp[j] + im * sumP
                    - half * kappa_t * adSp[j] + sqrt_ke * cEt * Sp[j]
                    - 2im * gj * (2 * ca * adSz[j] + ad_ad * Sz[j] - 2 * ca^2 * Sz[j]))
        dadSm[j] = (im * delta0 * adSm[j] - im * dj * adSm[j]
                    + 2im * gj * Sz[j] + im * sumM
                    - half * kappa_t * adSm[j] + sqrt_ke * cEt * conj(Sp[j])
                    + 2im * gj * (conj(adSz[j]) * ca + a * adSz[j] + Sz[j] * ad_a
                                  - 2 * ca * a * Sz[j]))
        dadSz[j] = (im * delta0 * adSz[j] + im * sumZ
                    - half * kappa_t * adSz[j] + sqrt_ke * cEt * Sz[j]
                    - im * gj * (Sp[j] + Sp[j] * ad_a + ca * conj(adSm[j]) + a * adSp[j]
                                 - 2 * Sp[j] * ca * a)
                    + im * gj * (2 * ca * adSm[j] + ad_ad * conj(Sp[j])
                                 - 2 * ca^2 * conj(Sp[j])))
        dSpSp_s[j] = (2im * dj * SpSp_s[j] + 2im * gj * adSp[j]
                      - 4im * gj * (Sp[j] * adSz[j] + SzSp_s[j] * ca + adSp[j] * Sz[j]
                                    - 2 * Sp[j] * ca * Sz[j]))
        dSzSp_s[j] = (im * dj * SzSp_s[j]
                      - im * gj * (2 * Sp[j] * conj(adSm[j]) + a * SpSp_s[j] - 2 * Sp[j]^2 * a)
                      + im * gj * (Sp[j] * adSm[j] + ca * SmSp_s[j] + adSp[j] * conj(Sp[j])
                                   - 2 * Sp[j] * ca * conj(Sp[j]))
                      - 2im * gj * (SzSz_s[j] * ca + 2 * adSz[j] * Sz[j] - 2 * ca * Sz[j]^2))
        Q1 = conj(adSz[j]) * Sp[j] + SzSp_s[j] * a + conj(adSm[j]) * Sz[j] - 2 * Sp[j] * a * Sz[j]
        Q2 = ca * conj(SzSp_s[j]) + conj(Sp[j]) * adSz[j] + adSm[j] * Sz[j] - 2 * ca * conj(Sp[j]) * Sz[j]
        dSmSp_s[j] = 2im * gj * Q1 - 2im * gj * Q2
        dSzSz_s[j] = im * gj * conj(adSm[j]) - im * gj * adSm[j] - 2im * gj * Q1 + 2im * gj * Q2
    end
    return nothing
end

function _qrt_fd_apply!(dg, f!, u, v, up, um, fp, fm; ε::Float64=1e-7)
    @. up = u + ε * v
    @. um = u - ε * v
    f!(fp, up)
    f!(fm, um)
    @. dg = (fp - fm) / (2 * ε)
    return dg
end

# Scratch for product apply / RK4. Allocated once per QRT window.
struct QRTProductScratch{T}
    up::Vector{Complex{T}}
    um::Vector{Complex{T}}
    fp::Vector{Complex{T}}
    fm::Vector{Complex{T}}
    u::Vector{Complex{T}}
    k1::Vector{Complex{T}}
    k2::Vector{Complex{T}}
    k3::Vector{Complex{T}}
    k4::Vector{Complex{T}}
    gtmp::Vector{Complex{T}}
    u1::Vector{Complex{T}}
    u2::Vector{Complex{T}}
    umid::Vector{Complex{T}}
end

function QRTProductScratch(::Type{T}, M::Integer) where {T}
    n = small_length(M)
    z() = zeros(Complex{T}, n)
    return QRTProductScratch{T}(z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z())
end

# Product J·g : real-linear FD of the O(M) star RHS at the trajectory.
function qrt_product_apply!(dg::AbstractVector, g::AbstractVector, u_traj::AbstractVector,
                            delta0, kappa_e, kappa_i, delta_b, g_b, Et=0;
                            ε::Float64=1e-7, scratch=nothing)
    M = length(delta_b)
    n = small_length(M)
    length(dg) == length(g) == n || error(
        "qrt_product_apply!: g/dg length $(length(g))/$(length(dg)), expected $n."
    )
    u0 = u_traj
    if length(u_traj) == global_state_length(M)
        u0 = view(u_traj, 1:n)
    elseif length(u_traj) != n
        error("qrt_product_apply!: u_traj length $(length(u_traj)) is not $n or $(global_state_length(M)).")
    end
    p = qrt_product_params(delta0, kappa_e, kappa_i, delta_b, g_b, Et)
    if scratch === nothing
        up = similar(g); um = similar(g); fp = similar(g); fm = similar(g)
        u = copy(u0)
    else
        up = scratch.up; um = scratch.um; fp = scratch.fp; fm = scratch.fm
        u = scratch.u
        copyto!(u, u0)
    end
    f!(du, uu) = qrt_star_rhs!(du, uu, p[1], p[2], p[3], p[4], p[5], p[6])
    _qrt_fd_apply!(dg, f!, u, g, up, um, fp, fm; ε=ε)
    return dg
end

# Full-J oracle: real-linear FD of rhs_cpu! on the 3+9M+4M² state.
# Small-M validation only — not the product path.
function qrt_oracle_apply!(dg::AbstractVector, g::AbstractVector, u_traj::AbstractVector,
                           delta0, kappa_e, kappa_i, delta_b, g_b, Et=0;
                           ε::Float64=1e-7)
    M = length(delta_b)
    n = global_state_length(M)
    length(dg) == length(g) == n || error(
        "qrt_oracle_apply!: g/dg length $(length(g))/$(length(dg)), expected $n."
    )
    length(u_traj) == n || error(
        "qrt_oracle_apply!: u_traj length $(length(u_traj)), expected $n."
    )
    p = qrt_product_params(delta0, kappa_e, kappa_i, delta_b, g_b, Et)
    up = similar(g); um = similar(g); fp = similar(g); fm = similar(g)
    u = copy(u_traj)
    f!(du, uu) = rhs_cpu!(du, uu, p[1], p[2], p[3], p[4], p[5], p[6]; threaded=false)
    _qrt_fd_apply!(dg, f!, u, g, up, um, fp, fm; ε=ε)
    return dg
end

# Dense full-J for small-M tests. Columns are J e_k via the oracle apply.
function qrt_oracle_dense(u_traj::AbstractVector, delta0, kappa_e, kappa_i,
                          delta_b, g_b, Et=0; ε::Float64=1e-7)
    n = length(u_traj)
    J = Matrix{eltype(u_traj)}(undef, n, n)
    g = zeros(eltype(u_traj), n)
    dg = similar(g)
    @inbounds for k in 1:n
        fill!(g, 0)
        g[k] = 1
        qrt_oracle_apply!(dg, g, u_traj, delta0, kappa_e, kappa_i, delta_b, g_b, Et; ε=ε)
        J[:, k] .= dg
    end
    return J
end

function qrt_relabs_err(a, b)
    num = maximum(abs, a .- b)
    den = max(maximum(abs, a), maximum(abs, b), 1e-30)
    return (abs=num, rel=num / den)
end

# Pad a product tangent with zeros in the 4M² cross blocks.
function qrt_pad_product_tangent(g_small::AbstractVector, M::Integer)
    g = zeros(eltype(g_small), global_state_length(M))
    n = small_length(M)
    length(g_small) == n || error("qrt_pad_product_tangent: length $(length(g_small)) != $n.")
    copyto!(g, 1, g_small, 1, n)
    return g
end

# Connected ⟨· a†⟩ seed on the product state (bosonic +1 on ⟨aa†⟩).
# Same-bin / higher moments use stored 2nd-order fields minus the factorized
# product; leftover 3rd-order pieces are truncated-cluster (star).
function qrt_seed_adag_column!(g::AbstractVector, u::AbstractVector, M::Integer)
    n = small_length(M)
    length(g) == n || error("qrt_seed_adag_column!: g length $(length(g)) != $n.")
    fill!(g, 0)
    a = u[IDX_a]
    adag = conj(a)
    ad_ad = u[IDX_ad_ad]
    ad_a = u[IDX_ad_a]
    g[IDX_a] = ad_a - abs2(a) + 1
    g[IDX_ad_ad] = 2 * adag * (ad_ad - adag^2)
    g[IDX_ad_a] = adag * (ad_a - abs2(a)) + adag
    Sp = @view u[small_range(M, F_Sp)]
    Sz = @view u[small_range(M, F_Sz)]
    adSp = @view u[small_range(M, F_adSp)]
    adSm = @view u[small_range(M, F_adSm)]
    adSz = @view u[small_range(M, F_adSz)]
    gSp = @view g[small_range(M, F_Sp)]
    gSz = @view g[small_range(M, F_Sz)]
    gadSp = @view g[small_range(M, F_adSp)]
    gadSm = @view g[small_range(M, F_adSm)]
    gadSz = @view g[small_range(M, F_adSz)]
    @inbounds for j in 1:M
        gSp[j] = adSp[j] - adag * Sp[j]
        gSz[j] = adSz[j] - adag * Sz[j]
        gadSp[j] = adag * (adSp[j] - adag * Sp[j])
        gadSm[j] = adag * (adSm[j] - adag * conj(Sp[j]))
        gadSz[j] = adag * (adSz[j] - adag * Sz[j])
    end
    return g
end

# Connected ⟨· a⟩ seed. Equal-time ⟨aa⟩_c = conj(⟨a†a†⟩) − ⟨a⟩², so
# conj(g[a]) matches the old anomalous ⟨a† a†⟩_c seed.
function qrt_seed_a_column!(g::AbstractVector, u::AbstractVector, M::Integer)
    n = small_length(M)
    length(g) == n || error("qrt_seed_a_column!: g length $(length(g)) != $n.")
    fill!(g, 0)
    a = u[IDX_a]
    ad_ad = u[IDX_ad_ad]
    ad_a = u[IDX_ad_a]
    g[IDX_a] = conj(ad_ad) - a^2
    g[IDX_ad_ad] = a * (ad_ad - conj(a)^2)
    g[IDX_ad_a] = a * (ad_a - abs2(a))
    Sp = @view u[small_range(M, F_Sp)]
    Sz = @view u[small_range(M, F_Sz)]
    adSp = @view u[small_range(M, F_adSp)]
    adSm = @view u[small_range(M, F_adSm)]
    adSz = @view u[small_range(M, F_adSz)]
    gSp = @view g[small_range(M, F_Sp)]
    gSz = @view g[small_range(M, F_Sz)]
    gadSp = @view g[small_range(M, F_adSp)]
    gadSm = @view g[small_range(M, F_adSm)]
    gadSz = @view g[small_range(M, F_adSz)]
    @inbounds for j in 1:M
        gSp[j] = conj(adSm[j]) - a * Sp[j]
        gSz[j] = conj(adSz[j]) - a * Sz[j]
        gadSp[j] = a * (adSp[j] - conj(a) * Sp[j])
        gadSm[j] = a * (adSm[j] - conj(a) * conj(Sp[j]))
        gadSz[j] = a * (adSz[j] - conj(a) * Sz[j])
    end
    return g
end

# Pack a saved-trajectory snapshot onto the 3+9M product state.
# Same-bin moments are reconstructed with the H1 product-state closure.
function qrt_pack_product_state!(u::AbstractVector, a, adad, n, Sp, Sz, adSp, adSm, adSz, Nj)
    M = length(Nj)
    length(u) == small_length(M) || error(
        "qrt_pack_product_state!: u length $(length(u)) != $(small_length(M))."
    )
    u[IDX_a] = a
    u[IDX_ad_ad] = adad
    u[IDX_ad_a] = n
    @inbounds for j in 1:M
        u[small_offset(M, F_Sp) + j] = Sp[j]
        u[small_offset(M, F_Sz) + j] = Sz[j]
        u[small_offset(M, F_adSp) + j] = adSp[j]
        u[small_offset(M, F_adSm) + j] = adSm[j]
        u[small_offset(M, F_adSz) + j] = adSz[j]
        SpSp, SzSp, SmSp, SzSz = _uncorrelated_same_moments(Sp[j], Sz[j], Nj[j])
        u[small_offset(M, F_SpSp_s) + j] = SpSp
        u[small_offset(M, F_SzSp_s) + j] = SzSp
        u[small_offset(M, F_SmSp_s) + j] = SmSp
        u[small_offset(M, F_SzSz_s) + j] = SzSz
    end
    return u
end

function qrt_estimate_Nj(Sp, Sz)
    M = size(Sp, 1)
    Nj = zeros(Float64, M)
    @inbounds for j in 1:M
        Nj[j] = max(1.0, 2 * sqrt(abs2(Sp[j, 1]) + abs2(Sz[j, 1])))
    end
    return Nj
end

function qrt_rk4_product_step!(g, u1, umid, u2, h, delta0, kappa_e, kappa_i,
                               delta_b, g_b, Et, scratch; ε::Float64=1e-7)
    k1 = scratch.k1; k2 = scratch.k2; k3 = scratch.k3; k4 = scratch.k4
    gtmp = scratch.gtmp
    qrt_product_apply!(k1, g, u1, delta0, kappa_e, kappa_i, delta_b, g_b, Et;
                       ε=ε, scratch=scratch)
    @. gtmp = g + (h / 2) * k1
    qrt_product_apply!(k2, gtmp, umid, delta0, kappa_e, kappa_i, delta_b, g_b, Et;
                       ε=ε, scratch=scratch)
    @. gtmp = g + (h / 2) * k2
    qrt_product_apply!(k3, gtmp, umid, delta0, kappa_e, kappa_i, delta_b, g_b, Et;
                       ε=ε, scratch=scratch)
    @. gtmp = g + h * k3
    qrt_product_apply!(k4, gtmp, u2, delta0, kappa_e, kappa_i, delta_b, g_b, Et;
                       ε=ε, scratch=scratch)
    @. g += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return g
end

# Adaptive Tsit5 on ġ = J(ū(t)) g with ū linearly interpolated on [0, dt].
function qrt_tsit5_product_step!(g, u1, u2, dt, delta0, kappa_e, kappa_i,
                                 delta_b, g_b, Et;
                                 reltol::Float64=1e-8, abstol::Float64=1e-8,
                                 ε::Float64=1e-7)
    T = real(eltype(g))
    scratch = QRTProductScratch(T, length(delta_b))
    function rhs!(dg, gg, p, t)
        θ = dt == 0 ? zero(t) : t / dt
        @. scratch.umid = (1 - θ) * u1 + θ * u2
        qrt_product_apply!(dg, gg, scratch.umid, delta0, kappa_e, kappa_i,
                           delta_b, g_b, Et; ε=ε, scratch=scratch)
        return nothing
    end
    problem = ODEProblem(rhs!, g, (0.0, dt))
    solution = solve(problem, Tsit5(); reltol=reltol, abstol=abstol, save_everystep=false)
    copyto!(g, solution.u[end])
    return g
end
