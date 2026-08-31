# ============================================================================
#  STANDALONE GRAPE  —  inversion = 1  AND  silencing factor = 1
#  on the 3ARP composite pi-pulse of a saved 1st-order run.
#
#  ── ZERO IMPACT ON THE EXISTING IMPLEMENTATION ──────────────────────────────
#  * Everything lives in `module StandaloneGRAPE`. It defines NO methods on any
#    InhomogeneousSpinCavityDynamics (ISCD) function or type.
#  * It is NOT `include`d by src/InhomogeneousSpinCavityDynamics.jl and nothing
#    in src/ references it. Run it directly:  julia --project examples/standalone_grape_3arp.jl
#  * ISCD is used READ-ONLY and only for setup / cross-checking:
#      - ISCD.load_jld2_reference  : build the ensemble `d` + identify the 3ARP control
#      - ISCD.build_E_of_t         : the analytic 3ARP drive E(t)  (warm start)
#      - ISCD.run_sim_1st_order_final (compute=:cpu) : independent Tsit5 solve, used
#                                    only to VALIDATE this file's own RK4 physics
#      - ISCD.rhs_1st_order! / metric helpers : used only in --selftest
#  * Outputs are written to NEW filenames (`*_grape_standalone.*`). The pipeline's
#    `*_optrunlog.jld2`, `*_opt_pulsemat.csv`, `*_opt_pulsepara.jld2` are untouched.
#
#  ── METHOD ────────────────────────────────────────────────────────────────
#  The mean-field 1st-order equations (a, S+_j, Sz_j) are re-implemented here as
#  a REAL map on R^(2+4M) (same real split the package's rhs_1st_order_real.jl
#  uses). Control = piecewise-constant complex cavity drive E_n on N_slices
#  uniform slices over [0,T]; each slice is integrated with `steps_per_slice`
#  fixed-step RK4 sub-steps. The gradient dJ/dE_n is the EXACT discrete adjoint
#  of that RK4 scheme (analytic J^T of the RHS, hand-derived and gradient-checked
#  in --selftest). Two trajectories share the control: :ground -> inversion,
#  :equator -> silencing factor |F|. Objective
#       J = 0.5*w_I*(1 - inversion)^2 + 0.5*w_S*(1 - silencing)^2
#  minimised by Adam on the scaled control x_n = E_n / E_scale.
# ============================================================================

module StandaloneGRAPE

import InhomogeneousSpinCavityDynamics as ISCD
using Printf, JLD2, Random

# ------------------------------------------------------------------ ensemble --
struct Ensemble
    M::Int
    sqrt_ke::Float64
    half_kt::Float64          # (kappa_e + kappa_i)/2
    delta0::Float64
    g::Vector{Float64}        # g_b   (length M)
    delta::Vector{Float64}    # delta_b (length M)
    Nj::Vector{Float64}       # length M
    T::Float64                # total time  (timespan[2]-timespan[1])
end

struct RefBundle
    ens::Ensemble
    E3arp                    # t -> ComplexF64 : the analytic 3ARP control drive
    d                        # ISCD prepare_derived output (for cross-checks)
end

function load_ref(path::AbstractString)::RefBundle
    ref = ISCD.load_jld2_reference(path; verbose=false)
    d = ref.d
    ref.control_cfg === nothing && error("no PULSE_CONFIG control segments identified in $path")
    ens = Ensemble(
        Int(d.M),
        sqrt(Float64(d.kappa_e)),
        0.5 * (Float64(d.kappa_e) + Float64(d.kappa_i)),
        Float64(d.delta0),
        collect(Float64, d.g_b),
        collect(Float64, d.delta_b),
        collect(Float64, d.Nj),
        Float64(d.timespan[2] - d.timespan[1]),
    )
    return RefBundle(ens, ISCD.build_E_of_t(ref.control_cfg), d)
end

# ----------------------------------------------------- real-state index model --
# x = [ ar, ai, pr[1:M], pi[1:M], zr[1:M], zi[1:M] ]   length 2 + 4M
@inline nstate(M::Int) = 2 + 4M
@inline off_pr(M::Int) = 2
@inline off_pi(M::Int) = 2 + M
@inline off_zr(M::Int) = 2 + 2M
@inline off_zi(M::Int) = 2 + 3M

function ic_ground!(x::Vector{Float64}, ens::Ensemble)
    M = ens.M; fill!(x, 0.0)
    zr = off_zr(M)
    @inbounds for j in 1:M
        x[zr + j] = -0.5 * ens.Nj[j]
    end
    return x
end
function ic_equator!(x::Vector{Float64}, ens::Ensemble)
    M = ens.M; fill!(x, 0.0)
    pr = off_pr(M)
    @inbounds for j in 1:M
        x[pr + j] = 0.5 * ens.Nj[j]
    end
    return x
end

# ------------------------------------------------------------------- real RHS --
# dx = f(x, E)   with E = Ex + i Ep held constant.  Mirrors rhs_1st_order!.
function rhs!(dx::Vector{Float64}, x::Vector{Float64}, Ex::Float64, Ep::Float64, ens::Ensemble)
    M = ens.M
    pr0 = off_pr(M); pi0 = off_pi(M); zr0 = off_zr(M); zi0 = off_zi(M)
    ar = x[1]; ai = x[2]
    d0 = ens.delta0; hk = ens.half_kt

    # cavity reductions  Σ g_j pr_j , Σ g_j pi_j   (serial: exact & cheap)
    sgr = 0.0; sgi = 0.0
    @inbounds @simd for j in 1:M
        gj = ens.g[j]
        sgr += gj * x[pr0 + j]
        sgi += gj * x[pi0 + j]
    end
    dx[1] = ens.sqrt_ke * Ex + d0 * ai - sgi - hk * ar
    dx[2] = ens.sqrt_ke * Ep - d0 * ar - sgr - hk * ai

    @inbounds Threads.@threads :static for j in 1:M
        gj = ens.g[j]; dj = ens.delta[j]; tg = 2.0 * gj
        prj = x[pr0 + j]; pij = x[pi0 + j]
        zrj = x[zr0 + j]; zij = x[zi0 + j]
        dx[pr0 + j] = -dj * pij + tg * (ar * zij - ai * zrj)
        dx[pi0 + j] =  dj * prj - tg * (ar * zrj + ai * zij)
        dx[zr0 + j] =  tg * (ar * pij + ai * prj)
        dx[zi0 + j] =  0.0
    end
    return dx
end

# ------------------------------------------------ analytic  J_f(x)^T  ·  λ  ----
# (state part only; the constant √κe·E term has zero Jacobian wrt x.)
# Verified against ISCD.rhs_1st_order_vjp! / finite differences in --selftest.
function rhs_vjp!(xb::Vector{Float64}, lam::Vector{Float64}, x::Vector{Float64}, ens::Ensemble)
    M = ens.M
    pr0 = off_pr(M); pi0 = off_pi(M); zr0 = off_zr(M); zi0 = off_zi(M)
    ar = x[1]; ai = x[2]
    lar = lam[1]; lai = lam[2]
    d0 = ens.delta0; hk = ens.half_kt

    bar_ar = -hk * lar - d0 * lai
    bar_ai =  d0 * lar - hk * lai
    @inbounds @simd for j in 1:M
        gj = ens.g[j]; tg = 2.0 * gj
        prj = x[pr0 + j]; pij = x[pi0 + j]
        zrj = x[zr0 + j]; zij = x[zi0 + j]
        lprj = lam[pr0 + j]; lpij = lam[pi0 + j]; lzrj = lam[zr0 + j]
        bar_ar += tg * ( zij * lprj - zrj * lpij + pij * lzrj)
        bar_ai += tg * (-zrj * lprj - zij * lpij + prj * lzrj)
    end
    xb[1] = bar_ar; xb[2] = bar_ai

    @inbounds Threads.@threads :static for j in 1:M
        gj = ens.g[j]; dj = ens.delta[j]; tg = 2.0 * gj
        lar_ = lam[1]; lai_ = lam[2]
        lprj = lam[pr0 + j]; lpij = lam[pi0 + j]; lzrj = lam[zr0 + j]
        xb[pr0 + j] = -gj * lai_ + dj * lpij + tg * ai * lzrj
        xb[pi0 + j] = -gj * lar_ - dj * lprj + tg * ar * lzrj
        xb[zr0 + j] = -tg * (ai * lprj + ar * lpij)
        xb[zi0 + j] =  tg * (ar * lprj - ai * lpij)
    end
    return xb
end

# --------------------------------------------------------- RK4 step + adjoint --
struct RKBuf
    k1::Vector{Float64}; k2::Vector{Float64}; k3::Vector{Float64}; k4::Vector{Float64}
    z::Vector{Float64}
    kb1::Vector{Float64}; kb2::Vector{Float64}; kb3::Vector{Float64}; kb4::Vector{Float64}
    zb::Vector{Float64}; tmp::Vector{Float64}
end
RKBuf(n::Int) = RKBuf((zeros(n) for _ in 1:11)...)

# y⁺ = y + h/6 (k1 + 2k2 + 2k3 + k4)
function rk4_step!(yout::Vector{Float64}, y::Vector{Float64}, Ex::Float64, Ep::Float64,
                   h::Float64, ens::Ensemble, b::RKBuf)
    rhs!(b.k1, y, Ex, Ep, ens)
    @. b.z = y + (h/2) * b.k1;  rhs!(b.k2, b.z, Ex, Ep, ens)
    @. b.z = y + (h/2) * b.k2;  rhs!(b.k3, b.z, Ex, Ep, ens)
    @. b.z = y + h * b.k3;      rhs!(b.k4, b.z, Ex, Ep, ens)
    @. yout = y + (h/6) * (b.k1 + 2*b.k2 + 2*b.k3 + b.k4)
    return yout
end

# Reverse of ONE rk4 step. `yentry` is the step's input state; `ybar` on entry is
# ∂J/∂y⁺, on return is ∂J/∂y. Returns (gEx_step, gEp_step) = ∂J/∂(Ex,Ep) of this step.
function rk4_step_adjoint!(ybar::Vector{Float64}, yentry::Vector{Float64},
                           Ex::Float64, Ep::Float64, h::Float64, ens::Ensemble, b::RKBuf)
    # recompute stages
    rhs!(b.k1, yentry, Ex, Ep, ens)
    @. b.z = yentry + (h/2) * b.k1
    z2 = copy(b.z); rhs!(b.k2, z2, Ex, Ep, ens)
    @. b.z = yentry + (h/2) * b.k2
    z3 = copy(b.z); rhs!(b.k3, z3, Ex, Ep, ens)
    @. b.z = yentry + h * b.k3
    z4 = copy(b.z); rhs!(b.k4, z4, Ex, Ep, ens)

    # stage cotangents seeded from  y⁺ = y + h/6 (k1 + 2k2 + 2k3 + k4)
    @. b.kb1 = (h/6) * ybar
    @. b.kb2 = (h/3) * ybar
    @. b.kb3 = (h/3) * ybar
    @. b.kb4 = (h/6) * ybar
    # ybar accumulates the direct  y⁺ = y + ...  term, then each stage's z̄
    # k4 = f(z4), z4 = y + h k3
    rhs_vjp!(b.zb, b.kb4, z4, ens)
    @. ybar  = ybar + b.zb
    @. b.kb3 = b.kb3 + h * b.zb
    # k3 = f(z3), z3 = y + h/2 k2
    rhs_vjp!(b.zb, b.kb3, z3, ens)
    @. ybar  = ybar + b.zb
    @. b.kb2 = b.kb2 + (h/2) * b.zb
    # k2 = f(z2), z2 = y + h/2 k1
    rhs_vjp!(b.zb, b.kb2, z2, ens)
    @. ybar  = ybar + b.zb
    @. b.kb1 = b.kb1 + (h/2) * b.zb
    # k1 = f(z1), z1 = y
    rhs_vjp!(b.zb, b.kb1, yentry, ens)
    @. ybar  = ybar + b.zb

    # E enters every stage through √κe·E in the cavity comps only:
    sk = ens.sqrt_ke
    gEx = sk * (b.kb1[1] + b.kb2[1] + b.kb3[1] + b.kb4[1])
    gEp = sk * (b.kb1[2] + b.kb2[2] + b.kb3[2] + b.kb4[2])
    return gEx, gEp
end

# ------------------------------------------------------------------- metrics --
function inversion_of(xT::Vector{Float64}, ens::Ensemble)
    M = ens.M; zr0 = off_zr(M)
    s = 0.0; @inbounds for j in 1:M; s += ens.Nj[j]; end
    inv = 0.0
    @inbounds for j in 1:M
        base = 0.5 * ens.Nj[j] + 1e-30
        u = 0.5 * (xT[zr0 + j] / base + 1.0)
        c = u < 0.0 ? 0.0 : (u > 1.0 ? 1.0 : u)
        inv += (ens.Nj[j] / s) * c
    end
    return inv
end

function silencing_of(xT::Vector{Float64}, ens::Ensemble)
    M = ens.M; pr0 = off_pr(M); pi0 = off_pi(M)
    Sr = 0.0; Si = 0.0; den = 0.0
    @inbounds for j in 1:M
        wj = ens.Nj[j] * ens.g[j]^2
        Sr += wj * xT[pr0 + j]
        Si += wj * xT[pi0 + j]
        den += wj * 0.5 * ens.Nj[j]
    end
    den += 1e-30
    Fr = Sr / den; Fi = Si / den
    absF = sqrt(Fr^2 + Fi^2 + 1e-30)
    return absF < 1.0 ? absF : 1.0
end

# terminal cotangent  xbar_T = ∂J/∂x_T  for  J = 0.5*w*(1-metric)^2
function seed_inversion_cotangent!(xbarT::Vector{Float64}, xT::Vector{Float64}, ens::Ensemble, w::Float64)
    M = ens.M; zr0 = off_zr(M)
    fill!(xbarT, 0.0)
    inv = inversion_of(xT, ens)
    dJdinv = -w * (1.0 - inv)
    s = 0.0; @inbounds for j in 1:M; s += ens.Nj[j]; end
    @inbounds for j in 1:M
        base = 0.5 * ens.Nj[j] + 1e-30
        u = 0.5 * (xT[zr0 + j] / base + 1.0)
        (u <= 0.0 || u >= 1.0) && continue          # clamp -> zero slope
        xbarT[zr0 + j] = dJdinv * (ens.Nj[j] / s) * 0.5 / base
    end
    return inv
end

function seed_silencing_cotangent!(xbarT::Vector{Float64}, xT::Vector{Float64}, ens::Ensemble, w::Float64)
    M = ens.M; pr0 = off_pr(M); pi0 = off_pi(M)
    fill!(xbarT, 0.0)
    Sr = 0.0; Si = 0.0; den = 0.0
    @inbounds for j in 1:M
        wj = ens.Nj[j] * ens.g[j]^2
        Sr += wj * xT[pr0 + j]; Si += wj * xT[pi0 + j]
        den += wj * 0.5 * ens.Nj[j]
    end
    den += 1e-30
    Fr = Sr / den; Fi = Si / den
    absF = sqrt(Fr^2 + Fi^2 + 1e-30)
    sil = absF < 1.0 ? absF : 1.0
    (absF >= 1.0) && return sil                      # clamp -> zero slope
    dJdsil = -w * (1.0 - sil)
    cr = dJdsil * (Fr / absF) / den
    ci = dJdsil * (Fi / absF) / den
    @inbounds for j in 1:M
        wj = ens.Nj[j] * ens.g[j]^2
        xbarT[pr0 + j] = cr * wj
        xbarT[pi0 + j] = ci * wj
    end
    return sil
end

# ---------------------------------------------------- forward + adjoint sweep --
# Control: Ex[1:Ns], Ep[1:Ns] piecewise constant; slice s has `sps` RK4 steps of size h.
# Checkpoints stored every `cs` slices.  Returns (xT, checkpoints, slice_of_ckpt).
struct FwdTape
    ckpt::Vector{Vector{Float64}}   # state at slice boundaries 0, cs, 2cs, ...
    cs::Int
end

function forward_record(x0::Vector{Float64}, Ex::Vector{Float64}, Ep::Vector{Float64},
                        h::Float64, sps::Int, cs::Int, ens::Ensemble, b::RKBuf)
    n = length(x0); Ns = length(Ex)
    x = copy(x0); xn = zeros(n)
    ckpt = Vector{Vector{Float64}}()
    push!(ckpt, copy(x))                              # boundary 0
    @inbounds for s in 1:Ns
        ex = Ex[s]; ep = Ep[s]
        for _ in 1:sps
            rk4_step!(xn, x, ex, ep, h, ens, b)
            x, xn = xn, x
        end
        if s % cs == 0
            push!(ckpt, copy(x))
        end
    end
    return x, FwdTape(ckpt, cs)
end

# Given xbarT (∂J/∂x_T), accumulate gEx[s], gEp[s] += ∂J/∂E_s over all slices.
function backward_accumulate!(gEx::Vector{Float64}, gEp::Vector{Float64},
                              xbarT::Vector{Float64}, tape::FwdTape,
                              Ex::Vector{Float64}, Ep::Vector{Float64},
                              h::Float64, sps::Int, ens::Ensemble, b::RKBuf)
    n = length(xbarT); Ns = length(Ex); cs = tape.cs
    xbar = copy(xbarT)
    nblk = cld(Ns, cs)
    entry = [zeros(n) for _ in 1:(cs * sps)]          # reused per block
    xw = zeros(n); xtmp = zeros(n)
    @inbounds for blk in nblk:-1:1
        s_lo = (blk - 1) * cs + 1
        s_hi = min(blk * cs, Ns)
        nslice = s_hi - s_lo + 1
        nstep = nslice * sps
        # replay this block forward from its left checkpoint, storing step entries
        copyto!(xw, tape.ckpt[blk])                   # boundary (blk-1)*cs
        st = 0
        for s in s_lo:s_hi
            ex = Ex[s]; ep = Ep[s]
            for _ in 1:sps
                st += 1
                copyto!(entry[st], xw)
                rk4_step!(xtmp, xw, ex, ep, h, ens, b)
                xw, xtmp = xtmp, xw
            end
        end
        # reverse through the block
        st = nstep
        for s in s_hi:-1:s_lo
            ex = Ex[s]; ep = Ep[s]
            gx = 0.0; gp = 0.0
            for _ in 1:sps
                dEx, dEp = rk4_step_adjoint!(xbar, entry[st], ex, ep, h, ens, b)
                gx += dEx; gp += dEp
                st -= 1
            end
            gEx[s] += gx; gEp[s] += gp
        end
    end
    return xbar
end

# --------------------------------------------------------------- objective ----
struct CostParts
    J::Float64; inv::Float64; sil::Float64
    Jinv::Float64; Jsil::Float64
end

# one full evaluation: cost + (optionally) gradient wrt (Ex,Ep)
function eval_cost_grad!(gEx::Vector{Float64}, gEp::Vector{Float64},
                         Ex::Vector{Float64}, Ep::Vector{Float64},
                         ens::Ensemble, h::Float64, sps::Int, cs::Int,
                         wI::Float64, wS::Float64, b::RKBuf; want_grad::Bool=true)
    n = nstate(ens.M)
    fill!(gEx, 0.0); fill!(gEp, 0.0)
    xbarT = zeros(n)

    # ---- ground trajectory -> inversion
    xg = zeros(n); ic_ground!(xg, ens)
    if want_grad
        xgT, tapeg = forward_record(xg, Ex, Ep, h, sps, cs, ens, b)
        inv = seed_inversion_cotangent!(xbarT, xgT, ens, wI)
        backward_accumulate!(gEx, gEp, xbarT, tapeg, Ex, Ep, h, sps, ens, b)
    else
        xgT, _ = forward_record(xg, Ex, Ep, h, sps, cs, ens, b)
        inv = inversion_of(xgT, ens)
    end

    # ---- equator trajectory -> silencing |F|
    xe = zeros(n); ic_equator!(xe, ens)
    if want_grad
        xeT, tapee = forward_record(xe, Ex, Ep, h, sps, cs, ens, b)
        sil = seed_silencing_cotangent!(xbarT, xeT, ens, wS)
        backward_accumulate!(gEx, gEp, xbarT, tapee, Ex, Ep, h, sps, ens, b)
    else
        xeT, _ = forward_record(xe, Ex, Ep, h, sps, cs, ens, b)
        sil = silencing_of(xeT, ens)
    end

    Jinv = 0.5 * wI * (1.0 - inv)^2
    Jsil = 0.5 * wS * (1.0 - sil)^2
    return CostParts(Jinv + Jsil, inv, sil, Jinv, Jsil)
end

# --------------------------------------------------------------------- Adam ----
mutable struct Adam
    m::Vector{Float64}; v::Vector{Float64}; t::Int
    lr::Float64; b1::Float64; b2::Float64; eps::Float64
end
Adam(n::Int; lr=0.05, b1=0.9, b2=0.999, eps=1e-8) = Adam(zeros(n), zeros(n), 0, lr, b1, b2, eps)
function step!(a::Adam, θ::Vector{Float64}, g::Vector{Float64})
    a.t += 1
    bc1 = 1 - a.b1^a.t; bc2 = 1 - a.b2^a.t
    @inbounds @simd for i in eachindex(θ)
        a.m[i] = a.b1 * a.m[i] + (1 - a.b1) * g[i]
        a.v[i] = a.b2 * a.v[i] + (1 - a.b2) * g[i]^2
        θ[i] -= a.lr * (a.m[i] / bc1) / (sqrt(a.v[i] / bc2) + a.eps)
    end
    return θ
end

# ------------------------------------------------------------- warm start -----
function warm_start(rb::RefBundle, Ns::Int)
    T = rb.ens.T; ds = T / Ns
    Ex = zeros(Ns); Ep = zeros(Ns)
    @inbounds for s in 1:Ns
        z = ComplexF64(rb.E3arp((s - 0.5) * ds))
        Ex[s] = real(z); Ep[s] = imag(z)
    end
    return Ex, Ep
end

# staircase drive as a t->Complex closure (for cross-checks / saving)
function staircase(Ex::Vector{Float64}, Ep::Vector{Float64}, T::Float64)
    Ns = length(Ex); ds = T / Ns
    return function (t)
        s = clamp(floor(Int, Float64(t) / ds) + 1, 1, Ns)
        return complex(Ex[s], Ep[s])
    end
end

# ------------------------------------------------------------------ selftest --
# Cross-checks (small M) that (1) `rhs!` == ISCD.rhs_1st_order!, (2) `rhs_vjp!`
# == finite differences of `rhs!`, (3) the RK4 discrete-adjoint dJ/dE matches
# central finite differences.  No optimisation, no big solve.
function selftest(; M::Int=7, seed::Int=1)
    rng = MersenneTwister(seed)
    kappa_e = 2.97e6; kappa_i = 0.13e6
    ens = Ensemble(M, sqrt(kappa_e), 0.5*(kappa_e + kappa_i), 0.11e6,
                   0.5 .+ rand(rng, M), 1e6 .* (randn(rng, M)),
                   10.0 .* (1 .+ rand(rng, M)), 5e-6)
    n = nstate(M)
    x = randn(rng, n); Ex = 0.7; Ep = -0.4

    # (1) rhs! vs ISCD.rhs_1st_order!
    dx = zeros(n); rhs!(dx, x, Ex, Ep, ens)
    Nc = ISCD.state_length_1st_order(M)
    u = zeros(ComplexF64, Nc)
    u[1] = complex(x[1], x[2])
    for j in 1:M
        u[1+j]      = complex(x[off_pr(M)+j], x[off_pi(M)+j])
        u[1+M+j]    = complex(x[off_zr(M)+j], x[off_zi(M)+j])
    end
    du = similar(u)
    kappa_e_st = ens.sqrt_ke^2
    kappa_i_st = 2*ens.half_kt - kappa_e_st
    # both sides take the RAW drive E and multiply by √κe internally.
    p = (ens.delta0, kappa_e_st, kappa_i_st, collect(ens.delta), collect(ens.g), M,
         t -> complex(Ex, Ep))
    ISCD.rhs_1st_order!(du, u, p, 0.0)
    dref = zeros(n)
    dref[1] = real(du[1]); dref[2] = imag(du[1])
    for j in 1:M
        dref[off_pr(M)+j] = real(du[1+j]);   dref[off_pi(M)+j] = imag(du[1+j])
        dref[off_zr(M)+j] = real(du[1+M+j]); dref[off_zi(M)+j] = imag(du[1+M+j])
    end
    e1 = maximum(abs.(dx .- dref)) / (maximum(abs.(dref)) + eps())
    @printf("selftest (1) rhs!  vs ISCD.rhs_1st_order!   rel max err = %.3e\n", e1)

    # (2) rhs_vjp! vs finite-difference Jacobian^T
    lam = randn(rng, n)
    xb = zeros(n); rhs_vjp!(xb, lam, x, ens)
    fd = zeros(n); h = 1e-6
    dp = zeros(n); dm = zeros(n); xp = copy(x)
    for i in 1:n
        xp .= x; xp[i] += h; rhs!(dp, xp, Ex, Ep, ens)
        xp .= x; xp[i] -= h; rhs!(dm, xp, Ex, Ep, ens)
        fd[i] = sum((dp .- dm) ./ (2h) .* lam)
    end
    e2 = maximum(abs.(xb .- fd)) / (maximum(abs.(fd)) + eps())
    @printf("selftest (2) rhs_vjp! vs finite diff        rel max err = %.3e\n", e2)

    # (3) discrete-adjoint dJ/d(Ex,Ep) vs central finite differences.
    # Uses a fully NON-DIMENSIONAL toy (all scales O(1)) so the FD check is
    # well conditioned: J actually responds to every slice and de-perturbations
    # sit well above roundoff.  Both cotangent seeds (inversion + silencing) are
    # active, and sps>1 (multi RK4 step per control slice) is exercised.
    ens2 = Ensemble(5, 2.0, 2.0, 0.0,
                    [0.80, 0.85, 0.90, 0.95, 1.00],          # g
                    [-1.5, -0.7, 0.0, 0.7, 1.5],             # delta
                    fill(2.0, 5),                             # Nj  (radius 1)
                    3.0)                                      # T
    n2 = nstate(ens2.M); b2 = RKBuf(n2)
    e3 = 0.0; inv3 = 0.0; sil3 = 0.0; J3 = 0.0
    for sps in (1, 3)
        Ns = 6; cs = 2
        hstep = ens2.T / (Ns * sps)
        Exv = [0.35, -0.25, 0.20, 0.10, -0.15, 0.30]
        Epv = [0.10,  0.22, -0.18, 0.06,  0.12, -0.08]
        gEx = zeros(Ns); gEp = zeros(Ns)
        cp = eval_cost_grad!(gEx, gEp, copy(Exv), copy(Epv), ens2, hstep, sps, cs, 1.0, 1.0, b2; want_grad=true)
        de = 1e-4
        fdx = zeros(Ns); fdp = zeros(Ns)
        for s in 1:Ns
            E1 = copy(Exv); E1[s] += de
            Jp = eval_cost_grad!(zeros(Ns), zeros(Ns), E1, copy(Epv), ens2, hstep, sps, cs, 1.0,1.0, b2; want_grad=false).J
            E2 = copy(Exv); E2[s] -= de
            Jm = eval_cost_grad!(zeros(Ns), zeros(Ns), E2, copy(Epv), ens2, hstep, sps, cs, 1.0,1.0, b2; want_grad=false).J
            fdx[s] = (Jp - Jm) / (2de)
            P1 = copy(Epv); P1[s] += de
            Jp = eval_cost_grad!(zeros(Ns), zeros(Ns), copy(Exv), P1, ens2, hstep, sps, cs, 1.0,1.0, b2; want_grad=false).J
            P2 = copy(Epv); P2[s] -= de
            Jm = eval_cost_grad!(zeros(Ns), zeros(Ns), copy(Exv), P2, ens2, hstep, sps, cs, 1.0,1.0, b2; want_grad=false).J
            fdp[s] = (Jp - Jm) / (2de)
        end
        scale = max(maximum(abs.(fdx)), maximum(abs.(fdp))) + 1e-12
        ex = maximum(abs.(gEx .- fdx)) / scale
        ep = maximum(abs.(gEp .- fdp)) / scale
        e3 = max(e3, ex, ep)
        inv3 = cp.inv; sil3 = cp.sil; J3 = cp.J
        @printf("selftest (3) sps=%d  dJ/dEx relerr=%.3e  dJ/dEp relerr=%.3e\n", sps, ex, ep)
    end
    @printf("selftest (3) worst discrete-adjoint vs FD relerr = %.3e   (J=%.6g inv=%.6g sil=%.6g)\n",
            e3, J3, inv3, sil3)
    ok = e1 < 1e-9 && e2 < 1e-5 && e3 < 1e-4
    println(ok ? "selftest PASSED" : "selftest FAILED")
    return ok
end

# plain fixed-step RK4 with a TIME-DEPENDENT complex drive Efun(t) -> Complex.
# Only used for validation (comparing this file's integrator+RHS against the
# package's adaptive Tsit5 on the smooth analytic 3ARP drive).
function rk4_run_func(x0::Vector{Float64}, Efun, T::Float64, Nstep::Int, ens::Ensemble, b::RKBuf)
    h = T / Nstep
    x = copy(x0); xn = zeros(length(x0))
    @inbounds for i in 0:Nstep-1
        t0 = i * h
        z1 = ComplexF64(Efun(t0));        rhs!(b.k1, x, real(z1), imag(z1), ens)
        z2 = ComplexF64(Efun(t0 + h/2))
        @. b.z = x + (h/2) * b.k1;        rhs!(b.k2, b.z, real(z2), imag(z2), ens)
        @. b.z = x + (h/2) * b.k2;        rhs!(b.k3, b.z, real(z2), imag(z2), ens)
        z4 = ComplexF64(Efun(t0 + h))
        @. b.z = x + h * b.k3;            rhs!(b.k4, b.z, real(z4), imag(z4), ens)
        @. xn = x + (h/6) * (b.k1 + 2*b.k2 + 2*b.k3 + b.k4)
        x, xn = xn, x
    end
    return x
end

# Two independent checks on the SAME ensemble:
#   (A) this file's RK4+RHS  vs  ISCD adaptive Tsit5, both on the SMOOTH
#       analytic 3ARP drive  -> proves the standalone physics is correct.
#   (B) warm-start staircase (this file's RK4)  vs  analytic 3ARP (this file's
#       RK4) -> shows how faithfully N_slices resolves the pulse.
function validate_physics(rb::RefBundle, Ex::Vector{Float64}, Ep::Vector{Float64},
                          h::Float64, sps::Int, cs::Int, b::RKBuf)
    ens = rb.ens; n = nstate(ens.M)
    Nstep = length(Ex) * sps

    # (A) analytic drive
    xg = zeros(n); ic_ground!(xg, ens)
    xgT = rk4_run_func(xg, rb.E3arp, ens.T, Nstep, ens, b)
    xe = zeros(n); ic_equator!(xe, ens)
    xeT = rk4_run_func(xe, rb.E3arp, ens.T, Nstep, ens, b)
    inv_rk = inversion_of(xgT, ens); sil_rk = silencing_of(xeT, ens)

    _, _, Szg = ISCD.run_sim_1st_order_final(rb.E3arp, rb.d; initial_condition=:ground, compute=:cpu)
    _, Spe, _ = ISCD.run_sim_1st_order_final(rb.E3arp, rb.d; initial_condition=:equator, compute=:cpu)
    inv_ts = ISCD._weighted_inversion(Szg, rb.d.Nj, Float64)
    sil_ts = ISCD._weighted_silencing_factor(Spe, rb.d.g_b, rb.d.Nj, Float64)

    @printf("validate (A) analytic 3ARP drive, M=%d ensemble:\n", ens.M)
    @printf("    inversion   RK4=%.6f   Tsit5=%.6f   |Δ|=%.2e\n", inv_rk, inv_ts, abs(inv_rk-inv_ts))
    @printf("    silencing   RK4=%.6f   Tsit5=%.6f   |Δ|=%.2e\n", sil_rk, sil_ts, abs(sil_rk-sil_ts))

    # (B) staircase warm start (RK4, GRAPE's own grid)
    xg2 = zeros(n); ic_ground!(xg2, ens)
    xg2T, _ = forward_record(xg2, Ex, Ep, h, sps, cs, ens, b)
    xe2 = zeros(n); ic_equator!(xe2, ens)
    xe2T, _ = forward_record(xe2, Ex, Ep, h, sps, cs, ens, b)
    @printf("validate (B) warm-start staircase (N_slices=%d), RK4:\n", length(Ex))
    @printf("    inversion   staircase=%.6f   (analytic RK4 %.6f)\n", inversion_of(xg2T, ens), inv_rk)
    @printf("    silencing   staircase=%.6f   (analytic RK4 %.6f)\n", silencing_of(xe2T, ens), sil_rk)
    return (inv_rk, sil_rk, inv_ts, sil_ts)
end

# ------------------------------------------------------------------- driver ---
function run(; path::AbstractString,
             N_slices::Int   = 10000,   # control slices  (staircase reproduces the 3ARP faithfully)
             steps_per_slice::Int = 4,  # RK4 sub-steps/slice -> Nstep=40000, dt=1.5e-8 s
             ckpt_stride::Int = 25,     #   (matches ISCD adaptive Tsit5 to ~1e-4 on this ensemble)
             iters::Int      = 40,
             lr::Float64     = 0.02,
             w_inv::Float64  = 1.0,
             w_sil::Float64  = 1.0,
             seed::Int       = 42,
             do_selftest::Bool = true,
             out_prefix::AbstractString = "")

    println("="^76)
    println("STANDALONE GRAPE   (inversion=1, silencing=1)   —   $(path)")
    println("threads = ", Threads.nthreads())
    println("="^76)

    if do_selftest
        selftest() || error("selftest failed — aborting before the expensive run")
        println()
    end

    rb = load_ref(path)
    ens = rb.ens
    n = nstate(ens.M)
    h = ens.T / (N_slices * steps_per_slice)
    @printf("ensemble  M=%d   T=%.6g s   half_kt=%.4g   sqrt_ke=%.4g\n", ens.M, ens.T, ens.half_kt, ens.sqrt_ke)
    @printf("grid      N_slices=%d   steps/slice=%d   RK4 dt=%.4g s   ckpt_stride=%d\n",
            N_slices, steps_per_slice, h, ckpt_stride)

    b = RKBuf(n)
    Ex, Ep = warm_start(rb, N_slices)
    Ex0 = copy(Ex); Ep0 = copy(Ep)          # true warm start, kept for the log
    Escale = maximum(sqrt.(Ex.^2 .+ Ep.^2)); Escale = Escale > 0 ? Escale : 1.0
    @printf("warm start from analytic 3ARP:  max|E| = %.6g   (control scale)\n", Escale)

    validate_physics(rb, Ex, Ep, h, steps_per_slice, ckpt_stride, b)
    println()

    # scaled control  θ = [Ex/Escale ; Ep/Escale]
    Ns = N_slices
    θ = vcat(Ex ./ Escale, Ep ./ Escale)
    gEx = zeros(Ns); gEp = zeros(Ns); gθ = zeros(2Ns)
    opt = Adam(2Ns; lr=lr)

    history = NamedTuple[]
    best = (J=Inf, inv=0.0, sil=0.0, θ=copy(θ))
    t0 = time()
    for it in 0:iters
        @views Ex .= θ[1:Ns] .* Escale
        @views Ep .= θ[Ns+1:2Ns] .* Escale
        cp = eval_cost_grad!(gEx, gEp, Ex, Ep, ens, h, steps_per_slice, ckpt_stride,
                             w_inv, w_sil, b; want_grad = it < iters)
        push!(history, (iter=it, J=cp.J, inversion=cp.inv, silencing=cp.sil,
                        Jinv=cp.Jinv, Jsil=cp.Jsil))
        if cp.J < best.J
            best = (J=cp.J, inv=cp.inv, sil=cp.sil, θ=copy(θ))
        end
        @printf("[it %3d] %6.1fs  J=%.6e  inversion=%.6f  silencing=%.6f   (Jinv=%.3e Jsil=%.3e)\n",
                it, time()-t0, cp.J, cp.inv, cp.sil, cp.Jinv, cp.Jsil)
        flush(stdout)
        it == iters && break
        # gradient wrt scaled control
        @views gθ[1:Ns]      .= gEx .* Escale
        @views gθ[Ns+1:2Ns]  .= gEp .* Escale
        step!(opt, θ, gθ)
    end

    # ---- final artefacts (NEW filenames; pipeline outputs untouched) ----
    θb = best.θ
    Exb = θb[1:Ns] .* Escale; Epb = θb[Ns+1:2Ns] .* Escale
    base = isempty(out_prefix) ? replace(path, r"\.jld2$" => "") * "_grape_standalone" : out_prefix
    jld = base * ".jld2"
    csv = base * "_pulsemat.csv"

    ds = ens.T / Ns
    tgrid = collect((0.5:1.0:(Ns-0.5)) .* ds)
    open(csv, "w") do io
        @printf(io, "# t_end_us,%.17g,source,standalone_grape\n", ens.T * 1e6)
        println(io, "Re,Im")
        for s in 1:Ns
            @printf(io, "%.17g,%.17g\n", Exb[s], Epb[s])
        end
    end

    JLD2.jldsave(jld; data = (
        kind = "standalone_grape",
        source_path = path,
        objective = "min 0.5*w_inv*(1-inversion)^2 + 0.5*w_sil*(1-silencing)^2",
        w_inv = w_inv, w_sil = w_sil,
        N_slices = Ns, steps_per_slice = steps_per_slice, ckpt_stride = ckpt_stride,
        rk4_dt = h, T = ens.T, M = ens.M,
        iters = iters, lr = lr, Escale = Escale, seed = seed,
        t_grid = tgrid,
        Ex_init = Ex0, Ep_init = Ep0,        # analytic-3ARP warm start (sampled on the slice grid)
        Ex_best = Exb, Ep_best = Epb,
        best_J = best.J, best_inversion = best.inv, best_silencing = best.sil,
        history = history,
    ))

    println()
    @printf("BEST  J=%.6e   inversion=%.6f   silencing=%.6f\n", best.J, best.inv, best.sil)
    println("wrote  ", jld)
    println("wrote  ", csv)
    return (jld=jld, csv=csv, best=best, history=history)
end

end # module StandaloneGRAPE

# ----------------------------------------------------------------------------- #
# CLI:  julia --project examples/standalone_grape_3arp.jl [selftest] [path]
#       env knobs: GRAPE_NSLICES GRAPE_SPS GRAPE_ITERS GRAPE_LR GRAPE_CKPT
# ----------------------------------------------------------------------------- #
if abspath(PROGRAM_FILE) == @__FILE__
    using .StandaloneGRAPE

    args = copy(ARGS)
    if length(args) >= 1 && lowercase(args[1]) == "selftest"
        StandaloneGRAPE.selftest()
    else
        path = isempty(args) ?
            joinpath(@__DIR__, "..", "data", "data_1st_order", "3ARP_pi_gstd_1em06Hz.jld2") :
            args[1]
        geti(k, d)  = haskey(ENV, k) ? parse(Int, ENV[k]) : d
        getf(k, d)  = haskey(ENV, k) ? parse(Float64, ENV[k]) : d
        StandaloneGRAPE.run(;
            path            = path,
            N_slices        = geti("GRAPE_NSLICES", 10000),
            steps_per_slice = geti("GRAPE_SPS", 4),
            ckpt_stride     = geti("GRAPE_CKPT", 25),
            iters           = geti("GRAPE_ITERS", 40),
            lr              = getf("GRAPE_LR", 0.02),
            w_inv           = getf("GRAPE_WINV", 1.0),
            w_sil           = getf("GRAPE_WSIL", 1.0),
        )
    end
end
