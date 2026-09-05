# Nude-quad multi-GPU monolith for inhomogeneous spin–cavity cumulant dynamics.
#
# One primary implementation: settings → ensemble (quad/:auto) → 1st/2nd-order
# RHS → CK45/Tsit5 → NCCL/P2P collectives → discrete-adjoint optimizer.
# Does NOT use Volkov–Zon / volkov_zon.jl (that path is not in this repo).
#
# Equation map (lab frame, ħ = 1). Collective operators per bin j of Nⱼ spins:
#   Sⱼ⁺ = Σᵢ σᵢ⁺,   Sⱼᶻ = Σᵢ σᵢᶻ / 2-convention used by this package (⟨Sᶻ⟩_ground = −Nⱼ/2)
# Cavity (driven, κₜ = κₑ + κᵢ):
#   ȧ = √κₑ E(t) − i δ₀ a − i Σⱼ gⱼ ⟨Sⱼ⁻⟩ − (κₜ/2) a          (1)
# First-order cumulants (⟨S⁻⟩ = ⟨S⁺⟩*):
#   ⟨Ṡ⁺⟩ⱼ = i δⱼ ⟨S⁺⟩ⱼ − 2i gⱼ a* ⟨Sᶻ⟩ⱼ                         (2)
#   ⟨Ṡᶻ⟩ⱼ = −i gⱼ a ⟨S⁺⟩ⱼ + i gⱼ a* ⟨S⁻⟩ⱼ                       (3)
# Second-order same-bin identities used for ICs (product state, finite Nⱼ):
#   ⟨S⁻S⁺⟩ⱼ = |⟨S⁺⟩|² (1 − 1/Nⱼ) + Nⱼ/2 − ⟨Sᶻ⟩                  (4)
#   ⟨SᶻSᶻ⟩ⱼ = ⟨Sᶻ⟩² (1 − 1/Nⱼ) + Nⱼ/4                           (5)
# Ground + vacuum is a fixed point of the 2nd-order RHS iff (4) holds:
#   ⟨S⁺⟩=0, ⟨Sᶻ⟩=−Nⱼ/2  ⇒  ⟨S⁻S⁺⟩ⱼ = Nⱼ  (NOT 0).
#
# Loss (optimizer, matches src/pulse_optimizer2.jl):
#   ss = 1 − (silencing − target_F)²
#   fid = inversion * ss
#   J_phys = (1−fid)² + (κ_I/2) max(I_min−I,0)² + (κ_S/2) max(S_min−ss,0)²
#   J = J_phys + w_time (duration/T_max) + w_tmax (max(t_end−T_max,0)/T_max)²
#         + w_power mean(|cA|/amp_scale)²

module NudeQuadMonolith

using LinearAlgebra
using Random
using Printf

const MONOLITH_SRC = @__DIR__
const _HAVE_FORWARDDIFF = try
    @eval using ForwardDiff
    true
catch
    false
end

# Optional stacks — loaded if present. CUDA/NCCL are never required to *include* this file.
const _HAVE_CUDA = Ref(false)
const _HAVE_NCCL = Ref(false)
const _HAVE_JSON3 = Ref(false)
const _HAVE_ODE = Ref(false)
const _HAVE_PKG = Ref(false)

function _try_using(mod::Symbol)
    try
        @eval using $(mod)
        return true
    catch
        return false
    end
end

function _load_optional_stacks!()
    _HAVE_CUDA[] = _try_using(:CUDA)
    _HAVE_NCCL[] = _HAVE_CUDA[] && _try_using(:NCCL)
    _HAVE_JSON3[] = _try_using(:JSON3)
    _HAVE_ODE[] = _try_using(:OrdinaryDiffEq)
    return nothing
end

# =============================================================================
# §1  Package helpers (B-spline / composite pulse / 1st-order layout+RHS)
# =============================================================================

include(joinpath(MONOLITH_SRC, "bspline.jl"))
include(joinpath(MONOLITH_SRC, "composite_pulse.jl"))
include(joinpath(MONOLITH_SRC, "state_layout_1st_order.jl"))
include(joinpath(MONOLITH_SRC, "rhs_1st_order.jl"))

# 2nd-order layout without CuArray (package state_layout_2nd_order.jl calls CUDA).
const IDX2_a     = 1
const IDX2_ad_ad = 2
const IDX2_ad_a  = 3
const IDX2_Sp_start = 4
idx2_Sz_start(M)   = IDX2_Sp_start + M
idx2_adSp_start(M) = idx2_Sz_start(M) + M
idx2_adSm_start(M) = idx2_adSp_start(M) + M
idx2_adSz_start(M) = idx2_adSm_start(M) + M
state_length_2nd_order(M) = 3 + 2M + 3M + 4M + 4M * M

function unpack_state_2nd_order_u(u, M)
    idx = 1
    a     = u[idx]; idx += 1
    ad_ad = u[idx]; idx += 1
    ad_a  = u[idx]; idx += 1
    Sp = @view u[idx:idx+M-1]; idx += M
    Sz = @view u[idx:idx+M-1]; idx += M
    adSp = @view u[idx:idx+M-1]; idx += M
    adSm = @view u[idx:idx+M-1]; idx += M
    adSz = @view u[idx:idx+M-1]; idx += M
    SpSp_same = @view u[idx:idx+M-1]; idx += M
    SzSp_same = @view u[idx:idx+M-1]; idx += M
    SmSp_same = @view u[idx:idx+M-1]; idx += M
    SzSz_same = @view u[idx:idx+M-1]; idx += M
    SpSp_cross = reshape(@view(u[idx:idx+M*M-1]), M, M); idx += M*M
    SzSp_cross = reshape(@view(u[idx:idx+M*M-1]), M, M); idx += M*M
    SmSp_cross = reshape(@view(u[idx:idx+M*M-1]), M, M); idx += M*M
    SzSz_cross = reshape(@view(u[idx:idx+M*M-1]), M, M); idx += M*M
    return (a, ad_ad, ad_a, Sp, Sz, adSp, adSm, adSz,
            SpSp_same, SzSp_same, SmSp_same, SzSz_same,
            SpSp_cross, SzSp_cross, SmSp_cross, SzSz_cross)
end

function unpack_state_2nd_order_du(du, M)
    idx = 4
    dSp = @view du[idx:idx+M-1]; idx += M
    dSz = @view du[idx:idx+M-1]; idx += M
    dadSp = @view du[idx:idx+M-1]; idx += M
    dadSm = @view du[idx:idx+M-1]; idx += M
    dadSz = @view du[idx:idx+M-1]; idx += M
    dSpSp_same = @view du[idx:idx+M-1]; idx += M
    dSzSp_same = @view du[idx:idx+M-1]; idx += M
    dSmSp_same = @view du[idx:idx+M-1]; idx += M
    dSzSz_same = @view du[idx:idx+M-1]; idx += M
    dSpSp_cross = reshape(@view(du[idx:idx+M*M-1]), M, M); idx += M*M
    dSzSp_cross = reshape(@view(du[idx:idx+M*M-1]), M, M); idx += M*M
    dSmSp_cross = reshape(@view(du[idx:idx+M*M-1]), M, M); idx += M*M
    dSzSz_cross = reshape(@view(du[idx:idx+M*M-1]), M, M); idx += M*M
    return (dSp, dSz, dadSp, dadSm, dadSz,
            dSpSp_same, dSzSp_same, dSmSp_same, dSzSz_same,
            dSpSp_cross, dSzSp_cross, dSmSp_cross, dSzSz_cross)
end

make_diag_mask_host(M) = ComplexF64.(.!Matrix(I, M, M))

# =============================================================================
# §2  Settings schema
# =============================================================================
#
# A settings file is either:
#   (A) Julia (.jl) that assigns NamedTuples (package spirit):
#         MODE, SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG,
#         optional BSPLINE, OPTIMIZER, COMPUTE
#   (B) JSON with the same keys (see MONOLITH.md).
#
# SIM_SETTING : M_delta, M_g, Ttotal, Nt_save, reltol, abstol,
#               initial_condition, ensemble_method (:auto/:quadrature/:histogram),
#               optional ensemble_M_delta, ensemble_M_g, saved_file_name
# SYSTEM_CONFIG : C_ens, delta0, kappa_e, kappa_i,
#               freq_inhomogeneity (kind, FWHM, span_gamma|span_sigma, renormalize),
#               g_inhomogeneity (kind, g_value | mean/std/span_sigma | alpha/g_min/g_max, renormalize)
# PULSE_CONFIG : tuple/vector of gaussian | wurst | custom | bspline pulses
# BSPLINE : k, n_coeff_A, n_coeff_f, degree, taper_frac
# OPTIMIZER : num_epochs, learning_rate, w_time, w_power, w_tmax, target_F,
#             I_min, kappa_I, S_min, kappa_S, track, seed, grad (:adjoint|:forward),
#             checkpoint_stride
# COMPUTE : backend (:auto/:cpu/:gpu), integrator (:tsit5/:ck45), nshards
# MODE : forward | forward_bspline | order2 | order2_bspline | optimizer
#        (hyphenated aliases forward-bspline / order2-bspline are accepted)

const MODES = (:forward, :forward_bspline, Symbol("forward-bspline"),
               :order2, :order2_bspline, Symbol("order2-bspline"), :optimizer)

function _canon_mode(mode)
    s = _sym(mode)
    s === Symbol("forward-bspline") && return :forward_bspline
    s === Symbol("order2-bspline") && return :order2_bspline
    s in (:forward, :forward_bspline, :order2, :order2_bspline, :optimizer) && return s
    error("unknown mode $mode; expected forward | forward_bspline | order2 | order2_bspline | optimizer")
end

_sym(x) = x isa Symbol ? x : Symbol(string(x))
_get(nt, k, default) = hasproperty(nt, k) ? getproperty(nt, k) : default

function default_bspline(; k=1, n_coeff_A=4, n_coeff_f=4, degree=3, taper_frac=0.1)
    return (k=Int(k), n_coeff_A=Int(n_coeff_A), n_coeff_f=Int(n_coeff_f),
            degree=Int(degree), taper_frac=Float64(taper_frac))
end

function default_optimizer()
    return (num_epochs=20, learning_rate=0.05, patience=8, seed=42,
            w_time=0.15, w_power=0.05, w_tmax=1.0, target_F=1.0,
            I_min=0.85, kappa_I=50.0, S_min=0.85, kappa_S=50.0,
            track=:weak, cf_lr_scale=0.25, grad=:adjoint,
            checkpoint_stride=typemax(Int))
end

function default_compute()
    return (backend=:auto, integrator=:tsit5, nshards=nothing, device_ids=nothing)
end

# =============================================================================
# §3  Ensemble: quadrature when quad-friendly, else histogram  (:auto)
# =============================================================================
#
# Frequency:
#   Lorentzian  FWHM → γ=FWHM/2,  δ = γ tan(θ), θ ∈ [−atan(span_γ), atan(span_γ)]
#               Gauss–Legendre on θ; weight p ∝ w θ_max / π
#               MODELING: truncated Lorentzian. renormalize=false (default) keeps
#               the truncated mass < 1, so N from cooperativity uses the analytic
#               C_ens formula while the discrete Σ p_δ may be < 1.
#   Gaussian    Gauss–Legendre on [−span_σ σ, span_σ σ] times Normal pdf
# Coupling:
#   constant    single node; M_g forced to 1
#   gaussian    GL on [max(0,μ−span σ), μ+span σ] times Normal pdf
#               MODELING: default renormalize=true (package coupling default)
#   powerlaw_g  GL on log g
# Cooperativity → total spin number (package formula):
#   Lorentzian: N = C_ens κₜ FWHM / (4 ⟨g²⟩)
#   Gaussian:   N = C_ens κₜ FWHM / (4 √(π ln 2) ⟨g²⟩)
# κₜ = κₑ + κᵢ  is used here (internal loss included), matching the main package.

function _gauss_legendre_pts(n::Integer)
    n >= 1 || error("Gauss–Legendre n must be >= 1")
    n == 1 && return ([0.0], [2.0])
    k = collect(1.0:(n - 1))
    beta = k ./ sqrt.(4 .* k .^ 2 .- 1.0)
    E = eigen(SymTridiagonal(zeros(Float64, n), beta))
    x = E.values
    w = 2.0 .* (E.vectors[1, :] .^ 2)
    p = sortperm(x)
    return x[p], w[p]
end

function _gauss_legendre_on(n::Integer, a::Real, b::Real)
    x, w = _gauss_legendre_pts(n)
    half = 0.5 * (b - a)
    return half .* x .+ 0.5 * (a + b), half .* w
end

gaussian_sigma_from_FWHM(FWHM) = FWHM / (2 * sqrt(2 * log(2)))
lorentzian_gamma_from_FWHM(FWHM) = FWHM / 2

function _norm_pdf(x, mu, sd)
    return exp(-0.5 * ((x - mu) / sd)^2) / (sd * sqrt(2π))
end

function ensemble_method_for(freq_cfg, g_cfg)
    fk = _get(freq_cfg, :kind, :unknown) |> _sym
    gk = _get(g_cfg, :kind, :unknown) |> _sym
    freq_ok = fk === :lorentzian || fk === :gaussian
    g_ok = gk === :constant || gk === :gaussian || gk === :powerlaw_g
    reason = freq_ok ? (g_ok ? "" : "g_inhomogeneity.kind=$gk has no quadrature rule") :
             "freq_inhomogeneity.kind=$fk has no quadrature rule"
    return (; method = (freq_ok && g_ok) ? :quadrature : :histogram,
              freq_kind = fk, g_kind = gk, reason = reason)
end

function resolve_ensemble_method(CONFIG, want::Symbol=:config)
    if want === :config
        want = _sym(_get(CONFIG, :ensemble_method, :auto))
    end
    want in (:histogram, :quadrature, :auto) ||
        error("ensemble_method must be :histogram, :quadrature, or :auto; got :$want")
    want === :histogram && return (; method=:histogram, reason="forced")
    plan = ensemble_method_for(CONFIG.freq_inhomogeneity, CONFIG.g_inhomogeneity)
    if want === :quadrature && plan.method !== :quadrature
        error("ensemble_method=:quadrature requested but $(plan.reason)")
    end
    return plan
end

function _quad_frequency_nodes(freq_cfg, M_delta::Integer)
    kind = _sym(freq_cfg.kind)
    FWHM = Float64(freq_cfg.FWHM)
    if kind === :lorentzian
        gamma = lorentzian_gamma_from_FWHM(FWHM)
        span = Float64(freq_cfg.span_gamma)
        theta_max = atan(span)
        x, w = _gauss_legendre_pts(M_delta)
        theta = theta_max .* x
        delta = gamma .* tan.(theta)
        p = w .* (theta_max / π)
        return delta, p
    elseif kind === :gaussian
        sigma = gaussian_sigma_from_FWHM(FWHM)
        span = Float64(freq_cfg.span_sigma)
        L = span * sigma
        delta, w = _gauss_legendre_on(M_delta, -L, L)
        p = w .* _norm_pdf.(delta, 0.0, sigma)
        return delta, p
    else
        error("no quadrature rule for freq kind $kind")
    end
end

function _as_erf(z)
    ax = abs(z)
    t = 1 / (1 + 0.3275911 * ax)
    y = 1 - (((((1.061405429t - 1.453152027)t + 1.421413741)t - 0.284496736)t + 0.254829592)t) * exp(-z * z)
    return copysign(y, z)
end

function _hist_frequency_nodes(freq_cfg, M_delta::Integer)
    kind = _sym(freq_cfg.kind)
    FWHM = Float64(freq_cfg.FWHM)
    if kind === :lorentzian
        γ = lorentzian_gamma_from_FWHM(FWHM)
        span = Float64(_get(freq_cfg, :span_gamma, 2.5))
        edges = range(-span * γ, span * γ; length=M_delta + 1)
        Fcdf = x -> (1 / π) * atan(x / γ) + 0.5
    elseif kind === :gaussian
        σ = gaussian_sigma_from_FWHM(FWHM)
        span = Float64(_get(freq_cfg, :span_sigma, 3.0))
        edges = range(-span * σ, span * σ; length=M_delta + 1)
        Fcdf = x -> 0.5 * (1 + _as_erf(x / (σ * sqrt(2))))
    else
        error("unknown freq kind $kind")
    end
    means = zeros(M_delta)
    probs = zeros(M_delta)
    for j in 1:M_delta
        lo, hi = Float64(edges[j]), Float64(edges[j+1])
        pj = Fcdf(hi) - Fcdf(lo)
        probs[j] = max(pj, 0.0)
        if pj > 0
            # midpoint fallback (histogram means; quad path is preferred)
            means[j] = 0.5 * (lo + hi)
        end
    end
    return collect(means), probs
end

function _maybe_renorm!(p, enabled)
    if enabled
        s = sum(p)
        s > 0 || error("cannot renormalize: sum(p)=0")
        p ./= s
    end
    return p
end

function _quad_coupling_bins(g_cfg, M_g::Integer)
    kind = _sym(g_cfg.kind)
    renorm = Bool(_get(g_cfg, :renormalize, true))
    if kind === :constant
        gv = Float64(g_cfg.g_value)
        return [gv], [1.0], gv, 0.0, abs2(gv)
    elseif kind === :gaussian
        mu = Float64(g_cfg.mean)
        sd = Float64(g_cfg.std)
        span = Float64(_get(g_cfg, :span_sigma, 3.0))
        lo = max(0.0, mu - span * sd)
        hi = mu + span * sd
        g, w = _gauss_legendre_on(M_g, lo, hi)
        p = w .* _norm_pdf.(g, mu, sd)
        _maybe_renorm!(p, renorm)
        return g, p, _g_stats(g, p)...
    elseif kind === :powerlaw_g
        alpha = Float64(g_cfg.alpha)
        gmin = Float64(g_cfg.g_min)
        gmax = Float64(g_cfg.g_max)
        lx, w = _gauss_legendre_on(M_g, log(gmin), log(gmax))
        g = exp.(lx)
        p = w .* g .* (g .^ (-alpha))
        _maybe_renorm!(p, renorm)
        return g, p, _g_stats(g, p)...
    else
        error("no quadrature rule for g kind $kind")
    end
end

function _g_stats(g, p)
    ps = sum(p)
    w = p ./ ps
    gm = sum(w .* g)
    g2 = sum(w .* g .^ 2)
    gs = sqrt(sum(w .* (g .- gm) .^ 2))
    return gm, gs, g2
end

function total_spin_number_from_cooperativity(C_ens, kappa_t, g2_avg, freq_cfg)
    FWHM = Float64(freq_cfg.FWHM)
    kind = _sym(freq_cfg.kind)
    C_ens >= 0 || error("C_ens must be >= 0")
    kappa_t > 0 || error("kappa_t must be > 0")
    g2_avg > 0 || error("g2_avg must be > 0")
    if kind === :lorentzian
        return C_ens * kappa_t * FWHM / (4 * g2_avg)
    elseif kind === :gaussian
        return C_ens * kappa_t * FWHM / (4 * sqrt(π * log(2)) * g2_avg)
    else
        error("unknown freq kind $kind")
    end
end

function build_2d_bins(N, delta_1d, p_delta, g_1d, p_g)
    Md, Mg = length(delta_1d), length(g_1d)
    Nj = zeros(Md * Mg)
    δ = zeros(Md * Mg)
    g = zeros(Md * Mg)
    t = 0
    for k in 1:Mg, i in 1:Md
        t += 1
        pij = p_delta[i] * p_g[k]
        Nj[t] = N * pij
        δ[t] = delta_1d[i]
        g[t] = g_1d[k]
    end
    return Nj, δ, g, sum(Nj)
end

function prepare_derived(CONFIG; ensemble_method::Symbol=:config)
    plan = resolve_ensemble_method(CONFIG, ensemble_method)
    # Order-2 uses the same :auto defaults — do not silently fall back to histogram
    # when the distributions are quadrature-friendly.
    M_delta_req = Int(CONFIG.M_delta)
    M_g_req = Int(CONFIG.M_g)
    M_delta = Int(_get(CONFIG, :ensemble_M_delta, M_delta_req))
    M_g = Int(_get(CONFIG, :ensemble_M_g, M_g_req))
    kappa_e = Float64(CONFIG.kappa_e)
    kappa_i = Float64(CONFIG.kappa_i)
    kappa_t = kappa_e + kappa_i
    freq = CONFIG.freq_inhomogeneity
    gcfg = CONFIG.g_inhomogeneity
    use_quad = plan.method === :quadrature

    if use_quad
        delta_1d, p_delta = _quad_frequency_nodes(freq, M_delta)
    else
        delta_1d, p_delta = _hist_frequency_nodes(freq, M_delta)
    end
    _maybe_renorm!(p_delta, Bool(_get(freq, :renormalize, false)))

    if use_quad
        g_1d, p_g, g_mean, g_std, g2_avg = _quad_coupling_bins(gcfg, M_g)
    else
        gk = _sym(gcfg.kind)
        gk === :constant || error("histogram path in monolith supports :constant g or quad-friendly kinds")
        g_1d, p_g, g_mean, g_std, g2_avg = _quad_coupling_bins(gcfg, 1)
    end
    M_g = length(g_1d)
    M_delta = length(delta_1d)

    N = total_spin_number_from_cooperativity(CONFIG.C_ens, kappa_t, g2_avg, freq)
    Nj, delta_b, g_b, N_total = build_2d_bins(N, delta_1d, p_delta, g_1d, p_g)
    M = M_delta * M_g
    timespan = (0.0, Float64(CONFIG.Ttotal))
    t_save = collect(range(0.0, Float64(CONFIG.Ttotal); length=Int(CONFIG.Nt_save)))
    return (
        C_ens=Float64(CONFIG.C_ens), M_delta=M_delta, M_g=M_g, M=M,
        freq_inhomogeneity=freq, FWHM=Float64(freq.FWHM),
        g_inhomogeneity=gcfg, g_mean=g_mean, g_std=g_std, g2_avg=g2_avg,
        kappa_e=kappa_e, kappa_i=kappa_i, kappa_t=kappa_t, sqrt_kappa_e=sqrt(kappa_e),
        delta0=Float64(CONFIG.delta0), N=N, N_total=N_total,
        Nj=Nj, delta_b=delta_b, g_b=g_b,
        delta_b_1d=delta_1d, p_delta=p_delta, g_b_1d=g_1d, p_g=p_g,
        timespan=timespan, t_save=t_save, Nt=length(t_save),
        ensemble_method=use_quad ? :quadrature : :histogram,
        ensemble_plan=plan,
    )
end

# =============================================================================
# §4  Pulses
# =============================================================================

function gaussian_drive(; t0, sigma, amp, omega=0.0, phase=0.0)
    t0f, sf, of, pf = Float64(t0), Float64(sigma), Float64(omega), Float64(phase)
    ampc = ComplexF64(amp)
    return function (t)
        τ = t - t0f
        return ampc * exp(-(τ^2) / (2 * sf^2)) * exp(1im * (of * τ + pf))
    end
end

function wurst_drive(; t_center, duration, amp, bandwidth, n=20.0, omega0=0.0,
                     chirp_sign=+1.0, phase0=0.0, edge_frac=1e-4)
    tc, dur = Float64(t_center), Float64(duration)
    ampc = ComplexF64(amp)
    bw, nf, o0 = Float64(bandwidth), Float64(n), Float64(omega0)
    cs, p0, ef = Float64(chirp_sign), Float64(phase0), Float64(edge_frac)
    t_start = tc - dur / 2
    edge = max(dur * ef, eps(Float64))
    return function (t)
        τ = t - t_start
        gate = 0.5 * (tanh((t - t_start) / edge) - tanh((t - (t_start + dur)) / edge))
        envelope = ampc * (1 - abs(sin(pi * (τ - dur / 2) / dur))^nf)
        phase = p0 + (o0 - cs * bw / 2) * τ + 0.5 * cs * (bw / dur) * τ^2
        return gate * envelope * exp(1im * phase)
    end
end

function build_drive_pulse(cfg)
    k = _sym(cfg.kind)
    k === :gaussian && return gaussian_drive(t0=cfg.t0, sigma=cfg.sigma, amp=cfg.amp,
        omega=_get(cfg, :omega, 0.0), phase=_get(cfg, :phase, 0.0))
    k === :wurst && return wurst_drive(
        t_center=cfg.t_center, duration=cfg.duration, amp=cfg.amp, bandwidth=cfg.bandwidth,
        n=_get(cfg, :n, 20.0), omega0=_get(cfg, :omega0, 0.0),
        chirp_sign=_get(cfg, :chirp_sign, +1.0), phase0=_get(cfg, :phase0, 0.0),
        edge_frac=_get(cfg, :edge_frac, 1e-4))
    k === :custom && return t -> ComplexF64(cfg.f(t))
    error("Unknown pulse kind: $k")
end

function build_E_of_t_raw(PULSE_CONFIG)
    drives = Tuple(build_drive_pulse(cfg) for cfg in PULSE_CONFIG)
    return function (t)
        s = 0.0 + 0.0im
        @inbounds for d in drives
            s += d(t)
        end
        return s
    end
end

# =============================================================================
# §5  Initial conditions  (CORRECTED — do not copy nude-quad 2nd-order bugs)
# =============================================================================

const WEAK_SEED = 1.0e-3

function _spin_means(Nj, kind::Symbol)
    kind === :ground && return zero.(Nj), -Nj ./ 2
    kind === :inverted && return zero.(Nj), Nj ./ 2
    kind === :equator && return Nj ./ 2, zero.(Nj)
    kind === :weak && return WEAK_SEED .* Nj ./ 2, -Nj ./ 2
    kind === :weak_inverted && return WEAK_SEED .* Nj ./ 2, Nj ./ 2
    kind === :custom && return zero.(Nj), zero.(Nj)
    error("Unknown initial_condition=$kind. Use :ground,:inverted,:equator,:weak,:weak_inverted,:custom.")
end

"""
Same-bin product-state closures (Reader / finite-N algebra).
  SpSp_same = Sp² (1 − 1/Nⱼ)
  SzSp_same = Sz Sp (1 − 1/Nⱼ)
  SmSp_same = |Sp|² (1 − 1/Nⱼ) + Nⱼ/2 − Sz     # ground ⇒ Nⱼ  (package IC = 0 is WRONG)
  SzSz_same = Sz² (1 − 1/Nⱼ) + Nⱼ/4
Cross j≠k = products of means. Nⱼ≤0 → 0.
"""
@inline function _invN(Nj)
    N = real(Nj)
    return N > 0 ? (1 - inv(N)) : zero(N)
end
@inline spsp_same_product(Sp, Nj) = Sp * Sp * _invN(Nj)
@inline szsp_same_product(Sz, Sp, Nj) = Sz * Sp * _invN(Nj)
@inline function smsp_same_product(Sp, Sz, Nj)
    N = real(Nj)
    N <= 0 && return zero(Sp)
    return abs2(Sp) * (1 - inv(N)) + N / 2 - Sz
end
@inline function szsz_same_product(Sz, Nj)
    N = real(Nj)
    N <= 0 && return zero(Sz)
    return Sz * Sz * (1 - inv(N)) + N / 4
end

function build_u0_1st_order(M, Nj, ::Type{T}, kind::Symbol=:ground) where {T}
    u0 = zeros(Complex{T}, state_length_1st_order(M))
    Sp0, Sz0 = _spin_means(Nj, kind)
    u0[IDX1_Sp_start:IDX1_Sp_start+M-1] .= Complex{T}.(Sp0)
    u0[idx1_Sz_start(M):idx1_Sz_start(M)+M-1] .= Complex{T}.(Sz0)
    return u0
end

function build_u0_2nd_order(M, Nj, ::Type{T}, kind::Symbol=:ground) where {T}
    u0 = zeros(Complex{T}, state_length_2nd_order(M))
    Sp0, Sz0 = _spin_means(collect(Float64, Nj), kind)
    # cavity vacuum: a = ad_ad = ad_a = 0
    @inbounds for j in 1:M
        Sp = Complex{T}(Sp0[j])
        Sz = Complex{T}(Sz0[j])
        N = Float64(Nj[j])
        u0[IDX2_Sp_start+j-1] = Sp
        u0[idx2_Sz_start(M)+j-1] = Sz
        # cavity–spin correlators start at 0 (product vacuum ⊗ spins)
        u0[idx2_adSp_start(M)+j-1] = 0
        u0[idx2_adSm_start(M)+j-1] = 0
        u0[idx2_adSz_start(M)+j-1] = 0
        u0[idx2_adSz_start(M)+M+j-1] = spsp_same_product(Sp, N)          # Sp²(1-1/N)
        u0[idx2_adSz_start(M)+2M+j-1] = szsp_same_product(Sz, Sp, N)     # Sz Sp (1-1/N)
        # CORRECTED vs package: ⟨S⁻S⁺⟩ = |Sp|²(1-1/N)+N/2-Sz   (ground → N, not 0)
        u0[idx2_adSz_start(M)+3M+j-1] = smsp_same_product(Sp, Sz, N)
        u0[idx2_adSz_start(M)+4M+j-1] = szsz_same_product(Sz, N)
    end
    # Cross-bin product means (j≠k). Layout after same-bin block of 4M.
    base = 3 + 9M
    @inbounds for k in 1:M, j in 1:M
        if j != k
            u0[base + (k-1)*M + j] = Complex{T}(Sp0[j] * Sp0[k])                 # SpSp
            u0[base + M*M + (k-1)*M + j] = Complex{T}(Sz0[j] * Sp0[k])           # SzSp
            u0[base + 2M*M + (k-1)*M + j] = Complex{T}(conj(Sp0[j]) * Sp0[k])    # SmSp
            u0[base + 3M*M + (k-1)*M + j] = Complex{T}(Sz0[j] * Sz0[k])          # SzSz
        end
    end
    return u0
end

# =============================================================================
# §6  First-order RHS  (eqs. 1–3)  — matches package rhs_1st_order!
# =============================================================================

function rhs1!(du, u, p, t)
    delta0, kappa_e, kappa_i, delta_b, g_b, M, E_of_t = p
    a = u[IDX1_a]
    Sp = @view u[IDX1_Sp_start:IDX1_Sp_start+M-1]
    Sz = @view u[idx1_Sz_start(M):idx1_Sz_start(M)+M-1]
    dSp = @view du[IDX1_Sp_start:IDX1_Sp_start+M-1]
    dSz = @view du[idx1_Sz_start(M):idx1_Sz_start(M)+M-1]
    κt = kappa_e + kappa_i
    Et = E_of_t(t)
    # (1) ȧ = √κₑ E − iδ₀ a − i Σ gⱼ Sⱼ⁻ − (κₜ/2) a     with S⁻ = (S⁺)*
    s = zero(a)
    @inbounds for j in 1:M
        s += g_b[j] * conj(Sp[j])
    end
    du[IDX1_a] = sqrt(kappa_e) * Et - 1im * delta0 * a - 1im * s - (0.5 * κt) * a
    # (2)(3)
    @inbounds for j in 1:M
        gj = g_b[j]
        dSp[j] = 1im * delta_b[j] * Sp[j] - 2im * gj * conj(a) * Sz[j]
        dSz[j] = -1im * gj * a * Sp[j] + 1im * gj * conj(a) * conj(Sp[j])
    end
    return nothing
end

# =============================================================================
# §7  Second-order RHS  (package rhs_2nd_order!, with κₜ = κₑ+κᵢ)
# =============================================================================
#
# Extra cavity moments:
#   d⟨a†a†⟩, d⟨a†a⟩ couple to Σ gⱼ ⟨a† Sⱼ⁺⟩ and drive √κₑ E(t).
# Spin–cavity (adSp, adSm, adSz) and same/cross bin pairs follow the
# standard 2nd-order cumulant closure used in rhs_2nd_order.jl (nude-quad).
# Cross-bin diagonals are projected out (diag_mask).

function rhs2!(du, u, p, t)
    delta0, kappa_e, kappa_i, delta_b, g_b, M, diag_mask, E_of_t = p
    (a, ad_ad, ad_a, Sp, Sz, adSp, adSm, adSz,
     SpSp_same, SzSp_same, SmSp_same, SzSz_same,
     SpSp_cross, SzSp_cross, SmSp_cross, SzSz_cross) = unpack_state_2nd_order_u(u, M)
    (dSp, dSz, dadSp, dadSm, dadSz,
     dSpSp_same, dSzSp_same, dSmSp_same, dSzSz_same,
     dSpSp_cross, dSzSp_cross, dSmSp_cross, dSzSz_cross) = unpack_state_2nd_order_du(du, M)

    κe = kappa_e
    κt = kappa_e + kappa_i
    Et = E_of_t(t)

    du[IDX2_a] = sqrt(κe) * Et - 1im * delta0 * a - 1im * sum(g_b .* conj.(Sp)) - 0.5 * κt * a
    dSp .= 1im .* delta_b .* Sp .- 2im .* g_b .* adSz
    dSz .= -1im .* g_b .* conj.(adSm) .+ 1im .* g_b .* adSm
    du[IDX2_ad_ad] = 2im * delta0 * ad_ad + 2im * sum(g_b .* adSp) - κt * ad_ad + 2 * sqrt(κe) * conj(a) * conj(Et)
    du[IDX2_ad_a] = 1im * sum(g_b .* conj.(adSm)) - 1im * sum(g_b .* adSm) - κt * ad_a +
                    sqrt(κe) * Et * conj(a) + sqrt(κe) * conj(Et) * a

    sumgSpSp_jk = SpSp_same .* g_b .+ SpSp_cross * g_b
    sumgSmSp_jk = SmSp_same .* g_b .+ SmSp_cross * g_b
    sumgSzSp_jk = SzSp_same .* g_b .+ SzSp_cross * g_b

    dadSp .= (
        1im * delta0 .* adSp .+ 1im .* delta_b .* adSp .+ 1im .* sumgSpSp_jk
        .- 0.5 .* κt .* adSp .+ sqrt(κe) .* conj(Et) .* Sp
        .- 2im .* g_b .* (2 .* conj(a) .* adSz .+ ad_ad .* Sz .- 2 .* conj(a) .^ 2 .* Sz)
    )
    dadSm .= (
        1im * delta0 .* adSm .- 1im .* delta_b .* adSm
        .+ 2im .* g_b .* Sz .+ 1im .* sumgSmSp_jk
        .- 0.5 .* κt .* adSm .+ sqrt(κe) .* conj(Et) .* conj.(Sp)
        .+ 2im .* g_b .* (conj.(adSz) .* conj(a) .+ a .* adSz .+ Sz .* ad_a .- 2 .* conj(a) .* a .* Sz)
    )
    dadSz .= (
        1im * delta0 .* adSz .+ 1im .* sumgSzSp_jk
        .- 0.5 .* κt .* adSz .+ sqrt(κe) .* conj(Et) .* Sz
        .- 1im .* g_b .* (Sp .+ Sp .* ad_a .+ conj(a) .* conj.(adSm) .+ a .* adSp .- 2 .* Sp .* conj(a) .* a)
        .+ 1im .* g_b .* (2 .* conj(a) .* adSm .+ ad_ad .* conj.(Sp) .- 2 .* conj(a) .^ 2 .* conj.(Sp))
    )

    dSpSp_same .= (
        2im .* delta_b .* SpSp_same .+ 2im .* g_b .* adSp
        .- 4im .* g_b .* (Sp .* adSz .+ SzSp_same .* conj(a) .+ adSp .* Sz .- 2 .* Sp .* conj(a) .* Sz)
    )
    dSzSp_same .= (
        1im .* delta_b .* SzSp_same
        .- 1im .* g_b .* (2 .* Sp .* conj.(adSm) .+ a .* SpSp_same .- 2 .* Sp .^ 2 .* a)
        .+ 1im .* g_b .* (Sp .* adSm .+ conj(a) .* SmSp_same .+ adSp .* conj.(Sp) .- 2 .* Sp .* conj(a) .* conj.(Sp))
        .- 2im .* g_b .* (SzSz_same .* conj(a) .+ 2 .* adSz .* Sz .- 2 .* conj(a) .* Sz .^ 2)
    )
    dSmSp_same .= (
        2im .* g_b .* (conj.(adSz) .* Sp .+ SzSp_same .* a .+ conj.(adSm) .* Sz .- 2 .* Sp .* a .* Sz)
        .- 2im .* g_b .* (conj(a) .* conj.(SzSp_same) .+ conj.(Sp) .* adSz .+ adSm .* Sz .- 2 .* conj(a) .* conj.(Sp) .* Sz)
    )
    dSzSz_same .= (
        1im .* g_b .* conj.(adSm) .- 1im .* g_b .* adSm
        .- 2im .* g_b .* (conj.(adSz) .* Sp .+ SzSp_same .* a .+ conj.(adSm) .* Sz .- 2 .* Sp .* a .* Sz)
        .+ 2im .* g_b .* (conj(a) .* conj.(SzSp_same) .+ conj.(Sp) .* adSz .+ adSm .* Sz .- 2 .* conj(a) .* conj.(Sp) .* Sz)
    )

    Δ_col = reshape(delta_b, M, 1); Δ_row = reshape(delta_b, 1, M)
    G_col = reshape(g_b, M, 1);     G_row = reshape(g_b, 1, M)
    Sp_col = reshape(Sp, M, 1);     Sp_row = reshape(Sp, 1, M)
    Sz_col = reshape(Sz, M, 1);     Sz_row = reshape(Sz, 1, M)
    adSp_col = reshape(adSp, M, 1); adSp_row = reshape(adSp, 1, M)
    adSm_col = reshape(adSm, M, 1); adSm_row = reshape(adSm, 1, M)
    adSz_col = reshape(adSz, M, 1); adSz_row = reshape(adSz, 1, M)

    dSpSp_cross .= (
        1im .* (Δ_col .+ Δ_row) .* SpSp_cross
        .- 2im .* G_col .* (Sp_row .* adSz_col .+ conj(a) .* SzSp_cross .+ adSp_row .* Sz_col .- 2 .* Sp_row .* conj(a) .* Sz_col)
        .- 2im .* G_row .* (Sp_col .* adSz_row .+ conj(a) .* transpose(SzSp_cross) .+ Sz_row .* adSp_col .- 2 .* Sp_col .* conj(a) .* Sz_row)
    ) .* diag_mask
    dSzSp_cross .= (
        1im .* Δ_row .* SzSp_cross
        .- 1im .* G_col .* (Sp_row .* conj.(adSm_col) .+ Sp_col .* conj.(adSm_row) .+ a .* SpSp_cross .- 2 .* Sp_row .* Sp_col .* a)
        .+ 1im .* G_col .* (Sp_row .* adSm_col .+ conj(a) .* SmSp_cross .+ adSp_row .* conj.(Sp_col) .- 2 .* Sp_row .* conj(a) .* conj.(Sp_col))
        .- 2im .* G_row .* (SzSz_cross .* conj(a) .+ Sz_row .* adSz_col .+ Sz_col .* adSz_row .- 2 .* conj(a) .* Sz_row .* Sz_col)
    ) .* diag_mask
    dSmSp_cross .= (
        .- 1im .* Δ_col .* SmSp_cross .+ 1im .* Δ_row .* SmSp_cross
        .+ 2im .* G_col .* (conj.(adSz_col) .* Sp_row .+ conj.(adSm_row) .* Sz_col .+ a .* SzSp_cross .- 2 .* Sp_row .* a .* Sz_col)
        .- 2im .* G_row .* (conj(a) .* conj.(transpose(SzSp_cross)) .+ Sz_row .* adSm_col .+ conj.(Sp_col) .* adSz_row .- 2 .* conj(a) .* Sz_row .* conj.(Sp_col))
    ) .* diag_mask
    dSzSz_cross .= (
        .- 1im .* G_col .* (Sp_col .* conj.(adSz_row) .+ conj.(adSm_col) .* Sz_row .+ a .* transpose(SzSp_cross) .- 2 .* Sp_col .* Sz_row .* a)
        .+ 1im .* G_col .* (conj(a) .* conj.(transpose(SzSp_cross)) .+ Sz_row .* adSm_col .+ conj.(Sp_col) .* adSz_row .- 2 .* conj(a) .* Sz_row .* conj.(Sp_col))
        .- 1im .* G_row .* (conj.(adSz_col) .* Sp_row .+ conj.(adSm_row) .* Sz_col .+ a .* SzSp_cross .- 2 .* Sp_row .* a .* Sz_col)
        .+ 1im .* G_row .* (conj.(SzSp_cross) .* conj(a) .+ adSm_row .* Sz_col .+ adSz_col .* conj.(Sp_row) .- 2 .* conj(a) .* Sz_col .* conj.(Sp_row))
    ) .* diag_mask
    return nothing
end

# =============================================================================
# §7b  Order-2 multi-GPU layout  (package MGPUlayout.jl — NCCL/P2P, not host)
# =============================================================================
#
# small (replicated, length 3+9M):  a, a†a†, a†a | Sp, Sz, adSp, adSm, adSz |
#                                   SpSp_s, SzSp_s, SmSp_s, SzSz_s
#   Small RHS is evaluated ONCE (rank 0 / a single fused kernel), then broadcast.
#   Do not re-integrate ȧ independently on every GPU.
#
# large (sharded by contiguous columns k ∈ [lo:hi], mloc = hi-lo+1):
#   5 × M × mloc blocks, column-major per block:
#     B_SpSp=1, B_SzSp=2, B_SzSpT=3, B_SmSp=4, B_SzSz=5
#   SzSpT is the *explicit* transpose of SzSp (avoids a device transpose).
#   Diagonals of large blocks are zero (same-bin lives in small).
#
# Collectives each RHS (on-device):
#   Allreduce(SUM)  of O(1) cavity sums  Σ g Sp*, Σ g adSp, Σ g adSm
#   Allgather       of O(M) row-sums     (SpSp, SmSp, SzSp)×g  over column shards
# Host `exchange_rowsums!` (MGPUproblem.jl) is intentionally NOT used.
# κₜ = κₑ + κᵢ always (do not copy sim_2nd_multi_gpu_opt.jl which drops κᵢ).

const MG_NSCALAR = 3
const MG_NSMALLFIELD = 9
const MG_B_SpSp, MG_B_SzSp, MG_B_SzSpT, MG_B_SmSp, MG_B_SzSz = 1, 2, 3, 4, 5
const MG_NBLOCK = 5
mg_small_length(M) = MG_NSCALAR + MG_NSMALLFIELD * M
mg_large_length(M, mloc) = MG_NBLOCK * M * mloc
mg_shard_length(M, mloc) = mg_small_length(M) + mg_large_length(M, mloc)

function mg_column_partition(M::Integer, nshards::Integer)
    nshards = clamp(Int(nshards), 1, M)
    base, rem_ = divrem(M, nshards)
    counts = [base + (p <= rem_ ? 1 : 0) for p in 1:nshards]
    offsets = zeros(Int, nshards)
    for p in 2:nshards
        offsets[p] = offsets[p-1] + counts[p-1]
    end
    return counts, offsets
end

"""Fill small + 5-block large ICs with the corrected product-state algebra."""
function build_u0_2nd_mgpu(M, Nj, kind::Symbol, nshards::Integer=1)
    Sp0, Sz0 = _spin_means(collect(Float64, Nj), kind)
    small = zeros(ComplexF64, mg_small_length(M))
    @inbounds for j in 1:M
        Sp, Sz, N = ComplexF64(Sp0[j]), ComplexF64(Sz0[j]), Float64(Nj[j])
        small[MG_NSCALAR + 0M + j] = Sp
        small[MG_NSCALAR + 1M + j] = Sz
        small[MG_NSCALAR + 5M + j] = spsp_same_product(Sp, N)
        small[MG_NSCALAR + 6M + j] = szsp_same_product(Sz, Sp, N)
        small[MG_NSCALAR + 7M + j] = smsp_same_product(Sp, Sz, N)
        small[MG_NSCALAR + 8M + j] = szsz_same_product(Sz, N)
    end
    counts, offsets = mg_column_partition(M, nshards)
    larges = Vector{Vector{ComplexF64}}(undef, length(counts))
    for p in eachindex(counts)
        mloc = counts[p]; lo = offsets[p]
        L = zeros(ComplexF64, mg_large_length(M, mloc))
        @inbounds for jl in 1:mloc
            k = lo + jl
            for j in 1:M
                j == k && continue
                i = (jl - 1) * M + j
                L[0 * M * mloc + i] = Sp0[j] * Sp0[k]                 # SpSp
                L[1 * M * mloc + i] = Sz0[j] * Sp0[k]                 # SzSp
                L[2 * M * mloc + i] = Sz0[k] * Sp0[j]                 # SzSpT
                L[3 * M * mloc + i] = conj(Sp0[j]) * Sp0[k]           # SmSp
                L[4 * M * mloc + i] = Sz0[j] * Sz0[k]                 # SzSz
            end
        end
        larges[p] = L
    end
    return small, larges, counts, offsets
end

# =============================================================================
# §8  Real-state pack + analytic VJP of the 1st-order RHS (discrete adjoint)
# =============================================================================

real_state_length_1st_order(M::Integer) = 2 * state_length_1st_order(M)
@inline _ri_ar() = 1
@inline _ri_ai() = 2
@inline _ri_pr(j, M) = 2 + j
@inline _ri_pi(j, M) = 2 + M + j
@inline _ri_zr(j, M) = 2 + 2M + j
@inline _ri_zi(j, M) = 2 + 3M + j

function pack_state_real!(x, u, M)
    a = u[IDX1_a]
    x[1] = real(a); x[2] = imag(a)
    @inbounds for j in 1:M
        sp = u[IDX1_Sp_start+j-1]
        sz = u[idx1_Sz_start(M)+j-1]
        x[_ri_pr(j, M)] = real(sp); x[_ri_pi(j, M)] = imag(sp)
        x[_ri_zr(j, M)] = real(sz); x[_ri_zi(j, M)] = imag(sz)
    end
    return x
end

function real_to_complex!(u, x, M)
    u[IDX1_a] = complex(x[1], x[2])
    @inbounds for j in 1:M
        u[IDX1_Sp_start+j-1] = complex(x[_ri_pr(j, M)], x[_ri_pi(j, M)])
        u[idx1_Sz_start(M)+j-1] = complex(x[_ri_zr(j, M)], x[_ri_zi(j, M)])
    end
    return u
end

"""
Analytic VJP of eqs. (1)–(3) in real coordinates (drive E(t) treated separately).
Matches package `rhs_1st_order_vjp!` (lab frame). κₜ = κₑ+κᵢ.
"""
function rhs1_vjp!(x̄, λ, x, p, t)
    delta0, kappa_e, kappa_i, delta_b, g_b, M = p[1], p[2], p[3], p[4], p[5], p[6]
    κt = kappa_e + kappa_i
    halfκ = 0.5 * κt
    δ0 = real(delta0)
    ar, ai = x[1], x[2]
    λar, λai = λ[1], λ[2]
    x̄ar = -halfκ * λar - δ0 * λai
    x̄ai =  δ0 * λar - halfκ * λai
    @inbounds for j in 1:M
        gj = Float64(real(g_b[j]))
        δj = Float64(real(delta_b[j]))
        pr = x[_ri_pr(j, M)]; pi_ = x[_ri_pi(j, M)]
        zr = x[_ri_zr(j, M)]; zi  = x[_ri_zi(j, M)]
        λpr = λ[_ri_pr(j, M)]; λpi = λ[_ri_pi(j, M)]
        λzr = λ[_ri_zr(j, M)]
        two_g = 2 * gj
        x̄ar += two_g * (λpr * zi - λpi * zr + λzr * pi_)
        x̄ai += two_g * (-λpr * zr - λpi * zi + λzr * pr)
        x̄[_ri_pr(j, M)] = -gj * λai + δj * λpi + two_g * ai * λzr
        x̄[_ri_pi(j, M)] = -gj * λar - δj * λpr + two_g * ar * λzr
        x̄[_ri_zr(j, M)] = -two_g * ai * λpr - two_g * ar * λpi
        x̄[_ri_zi(j, M)] =  two_g * ar * λpr - two_g * ai * λpi
    end
    x̄[1] = x̄ar
    x̄[2] = x̄ai
    return nothing
end

# =============================================================================
# §9  Tsit5 / CK45 with pooled stage buffers (CPU)
# =============================================================================

struct Tsit5Tab{T}
    c::NTuple{5,T}
    a21::T
    a31::T; a32::T
    a41::T; a42::T; a43::T
    a51::T; a52::T; a53::T; a54::T
    a61::T; a62::T; a63::T; a64::T; a65::T
    b::NTuple{6,T}
    e::NTuple{6,T}
end

function Tsit5Tab(::Type{T}=Float64) where {T}
    b = (T(0.09646076681806523), T(0.01), T(0.4798896504144996),
         T(1.379008574103742), T(-3.290069515436081), T(2.324710524099774))
    # embedded error weights ≈ b − b̂ (Tsitouras 2011 / OrdinaryDiffEq)
    e = (T(0.00178001105222577714), T(0.0008164344596567469),
         T(-0.007880878010261995), T(0.1447110071732629),
         T(-0.5823571654525552), T(0.45808210592918697))
    return Tsit5Tab{T}(
        (T(0.161), T(0.327), T(0.9), T(0.9800255409045097), T(1.0)),
        T(0.161),
        T(-0.008480655492356989), T(0.335480655492357),
        T(2.8971530571054935), T(-6.359448489975075), T(4.3622954328695815),
        T(5.325864828439257), T(-11.748883564062828), T(7.4955393428898365), T(-0.09249506636175525),
        T(5.86145544294642), T(-12.92096931784711), T(8.159367898576159),
        T(-0.071584973281401), T(-0.028269050394068383),
        b, e)
end

struct CK45Tab{T}
    A::NTuple{4,T}
    b::NTuple{5,T}
    e::NTuple{5,T}
    c::NTuple{5,T}
end

function CK45Tab(::Type{T}=Float64) where {T}
    A = (T(970286171893 / 4311952581923),
         T(6584761158862 / 12103376702013),
         T(2251764453980 / 15575788980749),
         T(26877169314380 / 34165994151039))
    b = (T(1153189308089 / 22510343858157),
         T(1772645290293 / 4653164025191),
         T(-1672844663538 / 4480602732383),
         T(2114624349019 / 3568978502595),
         T(5198255086312 / 14908931495163))
    bh = (T(1016888040809 / 7410784769900),
          T(11231460423587 / 58533540763752),
          T(-1563879915014 / 6823010717585),
          T(606302364029 / 971179775848),
          T(1097981568119 / 3980877426909))
    e = ntuple(i -> b[i] - bh[i], 5)
    c = (A[1], A[1] + A[2], A[1] + A[2] + A[3], A[1] + A[2] + A[3] + A[4], one(T))
    return CK45Tab{T}(A, b, e, c)
end

mutable struct StagePool{T}
    u::Vector{T}
    k::Vector{Vector{T}}
    y::Vector{T}
    u1::Vector{T}
    err::Vector{T}
end

function StagePool(u0::AbstractVector, nstages::Int)
    T = eltype(u0)
    n = length(u0)
    return StagePool{T}(copy(u0), [zeros(T, n) for _ in 1:nstages],
                        zeros(T, n), zeros(T, n), zeros(T, n))
end

@inline _primal(x::Real) = Float64(x)
@inline _primal(x::Complex) = hypot(Float64(real(x)), Float64(imag(x)))
if _HAVE_FORWARDDIFF
    @inline _primal(x::ForwardDiff.Dual) = Float64(ForwardDiff.value(x))
    @inline _primal(x::Complex{<:ForwardDiff.Dual}) = hypot(_primal(real(x)), _primal(imag(x)))
end

function _errnorm(u, u1, err, atol, rtol)
    acc = 0.0
    @inbounds for i in eachindex(err)
        sc = atol + rtol * max(_primal(abs(u[i])), _primal(abs(u1[i])))
        acc += _primal(abs2(err[i] / sc))
    end
    return sqrt(acc / length(err))
end

function tsit5_step!(pool::StagePool, rhs!, p, t, dt, tab::Tsit5Tab)
    u, k, y, u1, err = pool.u, pool.k, pool.y, pool.u1, pool.err
    rhs!(k[1], u, p, t)
    @inbounds for i in eachindex(u)
        y[i] = u[i] + dt * tab.a21 * k[1][i]
    end
    rhs!(k[2], y, p, t + tab.c[1] * dt)
    @inbounds for i in eachindex(u)
        y[i] = u[i] + dt * (tab.a31 * k[1][i] + tab.a32 * k[2][i])
    end
    rhs!(k[3], y, p, t + tab.c[2] * dt)
    @inbounds for i in eachindex(u)
        y[i] = u[i] + dt * (tab.a41 * k[1][i] + tab.a42 * k[2][i] + tab.a43 * k[3][i])
    end
    rhs!(k[4], y, p, t + tab.c[3] * dt)
    @inbounds for i in eachindex(u)
        y[i] = u[i] + dt * (tab.a51 * k[1][i] + tab.a52 * k[2][i] + tab.a53 * k[3][i] + tab.a54 * k[4][i])
    end
    rhs!(k[5], y, p, t + tab.c[4] * dt)
    @inbounds for i in eachindex(u)
        y[i] = u[i] + dt * (tab.a61 * k[1][i] + tab.a62 * k[2][i] + tab.a63 * k[3][i] +
                            tab.a64 * k[4][i] + tab.a65 * k[5][i])
    end
    rhs!(k[6], y, p, t + dt)
    @inbounds for i in eachindex(u)
        u1[i] = u[i] + dt * (tab.b[1]*k[1][i] + tab.b[2]*k[2][i] + tab.b[3]*k[3][i] +
                             tab.b[4]*k[4][i] + tab.b[5]*k[5][i] + tab.b[6]*k[6][i])
        err[i] = dt * (tab.e[1]*k[1][i] + tab.e[2]*k[2][i] + tab.e[3]*k[3][i] +
                       tab.e[4]*k[4][i] + tab.e[5]*k[5][i] + tab.e[6]*k[6][i])
    end
    return u1, err
end

function ck45_step!(pool::StagePool, rhs!, p, t, dt, tab::CK45Tab)
    # Low-storage CK45: k stored in pool.k[1], accumulator in u1, stage in y.
    u, k1, y, u1, err = pool.u, pool.k[1], pool.y, pool.u1, pool.err
    fill!(u1, 0); fill!(err, 0)
    copyto!(y, u)
    rhs!(k1, y, p, t)
    @inbounds for i in eachindex(u)
        u1[i] = u[i] + dt * tab.b[1] * k1[i]
        err[i] = dt * tab.e[1] * k1[i]
        y[i] = u[i] + dt * tab.A[1] * k1[i]
    end
    for s in 2:5
        rhs!(k1, y, p, t + tab.c[s-1] * dt)
        @inbounds for i in eachindex(u)
            u1[i] += dt * tab.b[s] * k1[i]
            err[i] += dt * tab.e[s] * k1[i]
            if s < 5
                y[i] = u[i] + dt * tab.A[s] * k1[i]   # FSAL-style low-storage update
            end
        end
    end
    return u1, err
end

mutable struct PICtrl{T}
    qold::T
end
PICtrl(::Type{T}) where {T} = PICtrl{T}(T(1e-4))

function integrate!(rhs!, u0, p, tspan; integrator::Symbol=:tsit5,
                    reltol=1e-8, abstol=1e-8, dt0=0.0, dtmax=Inf,
                    maxiters=10_000_000, tsave=nothing, save_states::Bool=false)
    T = Float64
    t, tfinal = T(tspan[1]), T(tspan[2])
    pool = StagePool(u0, integrator === :ck45 ? 1 : 6)
    tab5 = Tsit5Tab(T)
    tabck = CK45Tab(T)
    ctrl = PICtrl(T)
    if dt0 > 0
        dt = T(dt0)
    else
        rhs!(pool.k[1], pool.u, p, t)
        d1 = _primal(sqrt(sum(abs2, pool.k[1]) / length(pool.u)))
        dt = d1 < 1e-8 ? T(1e-6) * (tfinal - t) : T(0.01) / max(d1, 1e-12)
        dt = min(dt, T(tfinal - t), T(dtmax))
    end
    ts = T[]
    us = Vector{typeof(u0)}()
    function maybe_save(tt)
        if save_states || tsave !== nothing
            push!(ts, tt)
            push!(us, copy(pool.u))
        end
    end
    save_states && maybe_save(t)
    if tsave !== nothing && !isempty(tsave) && abs(tsave[1] - t) <= 1e-14
        !save_states && maybe_save(t)
    end
    isave = tsave === nothing ? 1 : (isempty(tsave) ? 1 : (abs(tsave[1]-t)<=1e-14 ? 2 : 1))
    nsteps = 0
    while t < tfinal && nsteps < maxiters
        dt = min(dt, T(dtmax), tfinal - t)
        forced = false
        if tsave !== nothing && isave <= length(tsave)
            tnext = T(tsave[isave])
            if t + dt >= tnext - 1e-14
                dt = tnext - t
                forced = true
            end
        end
        dt <= 0 && break
        nsteps += 1
        if integrator === :ck45
            u1, err = ck45_step!(pool, rhs!, p, t, dt, tabck)
        else
            u1, err = tsit5_step!(pool, rhs!, p, t, dt, tab5)
        end
        EEst = _errnorm(pool.u, u1, err, T(abstol), T(reltol))
        if EEst <= 1
            copyto!(pool.u, u1)
            t = forced && tsave !== nothing ? T(tsave[isave]) : t + dt
            if save_states
                maybe_save(t)
            elseif forced && tsave !== nothing
                maybe_save(t)
            end
            if forced && tsave !== nothing
                isave += 1
            end
            q = EEst == 0 ? T(10) : T(0.9) * EEst^(-0.2)
            q = clamp(q, T(0.2), T(10))
            dt = forced ? max(dt, dt * q) : dt * q
            ctrl.qold = max(EEst, T(1e-4))
        else
            q = T(0.9) * EEst^(-0.25)
            dt *= clamp(q, T(0.2), T(1))
            dt < 1e-18 * (tfinal - tspan[1]) && error("step-size underflow at t=$t")
        end
    end
    save_states && (isempty(ts) || ts[end] != t) && maybe_save(t)
    return pool.u, ts, us, nsteps
end

# =============================================================================
# §10  Multi-GPU: shard ensemble, on-device reductions (NCCL / P2P)
# =============================================================================
#
# 1st-order sharding: bins [lo:hi] live on GPU p; cavity amplitude `a` is
# replicated. Each RHS:
#   local S = Σ_{j∈shard} gⱼ Sⱼ⁻
#   Allreduce(SUM) → global S          (NCCL, else P2P, never per-RHS D2H)
#   then fused kernel for ȧ, ⟨Ṡ⁺⟩, ⟨Ṡᶻ⟩
# 2nd-order: Allreduce of O(1) cavity sums + Allgather of O(M) row-sums
# (SpSp_cross*g, SmSp_cross*g, SzSp_cross*g). Host-staged row-sum exchange
# is a last-resort fallback and is logged.

function cuda_functional()
    _HAVE_CUDA[] || return false
    try
        return CUDA.functional()
    catch
        return false
    end
end

function gpu_count()
    cuda_functional() || return 0
    try
        return length(CUDA.devices())
    catch
        return 0
    end
end

function _enable_p2p!(devs)
    length(devs) <= 1 && return 1.0
    np = nk = 0
    for src in unique(devs), dst in unique(devs)
        src == dst && continue
        np += 1
        try
            if CUDA.can_access_peer(src, dst)
                CUDA.device!(src) do
                    CUDA.enable_peer_access(CUDA.context(dst))
                end
                nk += 1
            end
        catch
        end
    end
    return np == 0 ? 1.0 : nk / np
end

mutable struct Collective
    kind::Symbol          # :nccl | :p2p | :host | :single
    comms
    devices
end

function build_collectives(ndev::Int)
    ndev <= 1 && return Collective(:single, nothing, nothing)
    cuda_functional() || return Collective(:host, nothing, nothing)
    devs = collect(CUDA.devices())[1:min(ndev, length(CUDA.devices()))]
    if _HAVE_NCCL[]
        try
            comms = NCCL.Communicators(devs)
            return Collective(:nccl, comms, devs)
        catch e
            @warn "NCCL communicator setup failed; trying P2P" exception=e
        end
    end
    pfrac = _enable_p2p!(devs)
    pfrac > 0 && return Collective(:p2p, nothing, devs)
    @warn "NCCL/P2P unavailable — collectives will stage through the host (not the preferred path)."
    return Collective(:host, nothing, devs)
end

"""
Allreduce-sum a length-1 Complex buffer that already lives on each device.
`bufs[p]` is the in/out CuVector on device p.
"""
function allreduce_sum!(col::Collective, bufs)
    col.kind === :single && return nothing
    if col.kind === :nccl
        for (p, buf) in enumerate(bufs)
            CUDA.device!(col.devices[p])
            # NCCL.jl Allreduce! on the in-place buffer (sum).
            try
                NCCL.Allreduce!(buf, col.comms[p]; op=NCCL.sum)
            catch
                NCCL.Allreduce!(buf, buf, col.comms[p])
            end
        end
        return nothing
    end
    if col.kind === :p2p
        # Ring reduce-then-broadcast via device pointers (no host).
        n = length(bufs)
        CUDA.device!(col.devices[1])
        acc = copy(bufs[1])
        for p in 2:n
            acc .+= bufs[p]   # P2P load
        end
        for p in 1:n
            bufs[p] .= acc
        end
        return nothing
    end
    # last-resort host
    s = sum(Array(b) for b in bufs)
    for b in bufs
        copyto!(b, s)
    end
    return nothing
end

"""
Allgather shards of a vector. `locals[p]` is mloc_p * width; `fulls[p]` is M*width.
"""
function allgather_shards!(col::Collective, locals, fulls, counts, offsets, width::Int)
    col.kind === :single && (copyto!(fulls[1], locals[1]); return nothing)
    if col.kind === :nccl
        try
            for (p, loc) in enumerate(locals)
                CUDA.device!(col.devices[p])
                NCCL.Allgather!(loc, fulls[p], col.comms[p])
            end
            return nothing
        catch e
            @warn "NCCL Allgather failed; using P2P/host" exception=e
        end
    end
    if col.kind === :p2p || col.kind === :nccl
        for p in eachindex(fulls)
            CUDA.device!(col.devices[p])
            for q in eachindex(locals)
                n = counts[q] * width
                copyto!(fulls[p], offsets[q]*width + 1, locals[q], 1, n)
            end
        end
        return nothing
    end
    host = vcat((Array(l) for l in locals)...)
    for f in fulls
        copyto!(f, host)
    end
    return nothing
end

# Fused 1st-order kernel (one thread per bin). Register reuse: a, g, δ, Sp, Sz.
function _rhs1_fused_kernel!(da, dSp, dSz, a, Sp, Sz, delta_b, g_b, src,
                             Et, delta0, kappa_e, kappa_t, M, lo, hi)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nloc = hi - lo + 1
    if j == 1
        @inbounds da[1] = sqrt(kappa_e) * Et - 1im * delta0 * a[1] - 1im * src[1] - (0.5 * kappa_t) * a[1]
    end
    if 1 <= j <= nloc
        gj = g_b[j]; dj = delta_b[j]
        sp = Sp[j]; sz = Sz[j]; av = a[1]
        @inbounds dSp[j] = 1im * dj * sp - 2im * gj * conj(av) * sz
        @inbounds dSz[j] = -1im * gj * av * sp + 1im * gj * conj(av) * conj(sp)
    end
    return nothing
end

function _src_kernel!(src, g_b, Sp, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    s = zero(eltype(src))
    @inbounds while i <= n
        s += g_b[i] * conj(Sp[i])
        i += stride
    end
    # block reduction via atomic (n is modest for 1st-order shards)
    if threadIdx().x == 1 && blockIdx().x == 1
        # serial fallback inside one thread-block launch recommended from host:
        # host uses mapreduce; this kernel is a simple strided sum with shfl
    end
    # write via atomicAdd on real/imag
    CUDA.atomic_add!(pointer(src), s)
    return nothing
end

function _gpu_local_source(g_b, Sp)
    return sum(g_b .* conj.(Sp))
end

function solve_1st_gpu(u0, p, tspan; reltol=1e-8, abstol=1e-8, integrator=:tsit5,
                       nshards=nothing, tsave=nothing)
    delta0, kappa_e, kappa_i, delta_b, g_b, M, E_of_t = p
    ndev = gpu_count()
    ndev >= 1 || error("solve_1st_gpu: no CUDA device")
    ns = nshards === nothing ? ndev : min(Int(nshards), M, ndev)
    col = build_collectives(ns)
    part_counts = fill(M ÷ ns, ns)
    part_counts[1] += M - sum(part_counts)
    offsets = zeros(Int, ns)
    for i in 2:ns
        offsets[i] = offsets[i-1] + part_counts[i-1]
    end
    # Replicated full state on GPU 1 + shard views. For 1st-order the state is
    # O(M); we keep a full copy per device (cheap vs 2nd-order O(M²)) so the
    # Allreduce is a single complex and there is no per-RHS D2H.
    C = CUDA
    prev = C.device()
    us = Vector{Any}(undef, ns)
    srcs = Vector{Any}(undef, ns)
    try
        for p in 1:ns
            C.device!(C.devices()[p])
            us[p] = C.CuArray(u0)
            srcs[p] = C.CuArray([zero(eltype(u0))])
        end
        # Host Tsit5 driving device RHS (stage buffers stay on device).
        function rhs_dev!(du, u, _, t)
            # u/du here are HOST vectors used by integrate!; we copy once per
            # stage — for the *hot* multi-GPU path we instead run a device-
            # resident integrator below when ns>=1 and we own the loop.
            error("internal: use solve_1st_gpu_resident")
        end
        return solve_1st_gpu_resident(u0, p, tspan, col, ns, part_counts, offsets;
                                      reltol=reltol, abstol=abstol, integrator=integrator,
                                      tsave=tsave)
    finally
        C.device!(prev)
    end
end

function solve_1st_gpu_resident(u0, p, tspan, col, ns, counts, offsets;
                                reltol=1e-8, abstol=1e-8, integrator=:tsit5,
                                tsave=nothing, dt0=0.0, dtmax=Inf, maxiters=10_000_000)
    delta0, kappa_e, kappa_i, delta_b, g_b, M, E_of_t = p
    κt = kappa_e + kappa_i
    C = CUDA
    # Full state replicated (1st-order is O(M)). Reductions stay on-device.
    U = Vector{Any}(undef, ns)
    K = [Vector{Any}(undef, 6) for _ in 1:ns]
    Y = Vector{Any}(undef, ns)
    U1 = Vector{Any}(undef, ns)
    ERR = Vector{Any}(undef, ns)
    dlt = Vector{Any}(undef, ns)
    gg = Vector{Any}(undef, ns)
    SRC = Vector{Any}(undef, ns)
    for p in 1:ns
        C.device!(C.devices()[p])
        U[p] = C.CuArray(u0)
        Y[p] = C.similar(U[p]); U1[p] = C.similar(U[p]); ERR[p] = C.similar(U[p])
        for s in 1:6
            K[p][s] = C.similar(U[p])
        end
        dlt[p] = C.CuArray(Float64.(delta_b))
        gg[p] = C.CuArray(Float64.(g_b))
        SRC[p] = C.CuArray(eltype(u0)[0])
    end

    function rhs_on!(dests, srcs, t)
        Et = ComplexF64(E_of_t(t))
        for p in 1:ns
            C.device!(C.devices()[p])
            u = srcs[p]
            Sp = view(u, IDX1_Sp_start:IDX1_Sp_start+M-1)
            SRC[p] .= _gpu_local_source(view(gg[p], :), Sp)  # device reduction
        end
        allreduce_sum!(col, SRC)
        for p in 1:ns
            C.device!(C.devices()[p])
            u = srcs[p]; du = dests[p]
            a = u[1]
            Sp = view(u, IDX1_Sp_start:IDX1_Sp_start+M-1)
            Sz = view(u, idx1_Sz_start(M):idx1_Sz_start(M)+M-1)
            dSp = view(du, IDX1_Sp_start:IDX1_Sp_start+M-1)
            dSz = view(du, idx1_Sz_start(M):idx1_Sz_start(M)+M-1)
            s = SRC[p][1]
            du[1] = sqrt(kappa_e) * Et - 1im * delta0 * a - 1im * s - (0.5 * κt) * a
            dSp .= 1im .* dlt[p] .* Sp .- 2im .* gg[p] .* conj(a) .* Sz
            dSz .= -1im .* gg[p] .* a .* Sp .+ 1im .* gg[p] .* conj(a) .* conj.(Sp)
        end
        return nothing
    end

    tab = Tsit5Tab(Float64)
    t, tfinal = Float64(tspan[1]), Float64(tspan[2])
    rhs_on!(K[1], U, t)  # warm
    dt = dt0 > 0 ? Float64(dt0) : 1e-3 * (tfinal - t)
    nsteps = 0
    while t < tfinal && nsteps < maxiters
        dt = min(dt, Float64(dtmax), tfinal - t)
        dt <= 0 && break
        nsteps += 1
        # Tsit5 stages, all on-device
        rhs_on!([K[p][1] for p in 1:ns], U, t)
        for p in 1:ns
            C.device!(C.devices()[p])
            Y[p] .= U[p] .+ dt * tab.a21 .* K[p][1]
        end
        rhs_on!([K[p][2] for p in 1:ns], Y, t + tab.c[1]*dt)
        for p in 1:ns
            C.device!(C.devices()[p])
            Y[p] .= U[p] .+ dt .* (tab.a31 .* K[p][1] .+ tab.a32 .* K[p][2])
        end
        rhs_on!([K[p][3] for p in 1:ns], Y, t + tab.c[2]*dt)
        for p in 1:ns
            C.device!(C.devices()[p])
            Y[p] .= U[p] .+ dt .* (tab.a41 .* K[p][1] .+ tab.a42 .* K[p][2] .+ tab.a43 .* K[p][3])
        end
        rhs_on!([K[p][4] for p in 1:ns], Y, t + tab.c[3]*dt)
        for p in 1:ns
            C.device!(C.devices()[p])
            Y[p] .= U[p] .+ dt .* (tab.a51 .* K[p][1] .+ tab.a52 .* K[p][2] .+
                                   tab.a53 .* K[p][3] .+ tab.a54 .* K[p][4])
        end
        rhs_on!([K[p][5] for p in 1:ns], Y, t + tab.c[4]*dt)
        for p in 1:ns
            C.device!(C.devices()[p])
            Y[p] .= U[p] .+ dt .* (tab.a61 .* K[p][1] .+ tab.a62 .* K[p][2] .+
                                   tab.a63 .* K[p][3] .+ tab.a64 .* K[p][4] .+ tab.a65 .* K[p][5])
        end
        rhs_on!([K[p][6] for p in 1:ns], Y, t + dt)
        nstate = length(u0)
        C.device!(C.devices()[1])
        U1[1] .= U[1] .+ dt .* (tab.b[1].*K[1][1] .+ tab.b[2].*K[1][2] .+ tab.b[3].*K[1][3] .+
                                tab.b[4].*K[1][4] .+ tab.b[5].*K[1][5] .+ tab.b[6].*K[1][6])
        ERR[1] .= dt .* (tab.e[1].*K[1][1] .+ tab.e[2].*K[1][2] .+ tab.e[3].*K[1][3] .+
                         tab.e[4].*K[1][4] .+ tab.e[5].*K[1][5] .+ tab.e[6].*K[1][6])
        sc = abstol .+ reltol .* max.(abs.(Array(U[1])), abs.(Array(U1[1])))
        EEst = sqrt(sum(abs2, Array(ERR[1]) ./ sc) / nstate)
        if EEst <= 1
            C.device!(C.devices()[1])
            U[1] .= U1[1]
            for p in 2:ns
                C.device!(C.devices()[p])
                copyto!(U[p], U[1])
            end
            t += dt
            q = EEst == 0 ? 10.0 : 0.9 * EEst^(-0.2)
            dt *= clamp(q, 0.2, 10.0)
        else
            dt *= clamp(0.9 * EEst^(-0.25), 0.2, 1.0)
        end
    end
    C.device!(C.devices()[1])
    return Array(U[1]), nsteps, col.kind
end

# =============================================================================
# §11  Discrete adjoint (Tsit5 VJP along the recorded mesh)
# =============================================================================
#
# Forward: record (t_n, Δt_n, u_n). Reverse: λ_n = (∂Φ/∂u_n)ᵀ λ_{n+1}
# through the 6 Tsit5 stages using analytic rhs1_vjp!. Drive gradient
# ∂E/∂θ is the only Dual/ForwardDiff use (cheap; θ is the B-spline vector).

struct AdjTab{T}
    c1::T; c2::T; c3::T; c4::T
    a21::T
    a31::T; a32::T
    a41::T; a42::T; a43::T
    a51::T; a52::T; a53::T; a54::T
    a61::T; a62::T; a63::T; a64::T; a65::T
    b1::T; b2::T; b3::T; b4::T; b5::T; b6::T
end
function AdjTab(::Type{T}=Float64) where {T}
    return AdjTab{T}(
        T(0.161), T(0.327), T(0.9), T(0.9800255409045097),
        T(0.161),
        T(-0.008480655492356989), T(0.335480655492357),
        T(2.8971530571054935), T(-6.359448489975075), T(4.3622954328695815),
        T(5.325864828439257), T(-11.748883564062828), T(7.4955393428898365), T(-0.09249506636175525),
        T(5.86145544294642), T(-12.92096931784711), T(8.159367898576159),
        T(-0.071584973281401), T(-0.028269050394068383),
        T(0.09646076681806523), T(0.01), T(0.4798896504144996),
        T(1.379008574103742), T(-3.290069515436081), T(2.324710524099774))
end
const ADJTAB = AdjTab(Float64)

@inline function _adj_c(tab::AdjTab, i::Int)
    i == 2 && return tab.c1
    i == 3 && return tab.c2
    i == 4 && return tab.c3
    i == 5 && return tab.c4
    return one(tab.c1)
end
@inline function _adj_a(tab::AdjTab, i::Int, j::Int)
    i == 2 && j == 1 && return tab.a21
    i == 3 && j == 1 && return tab.a31
    i == 3 && j == 2 && return tab.a32
    i == 4 && j == 1 && return tab.a41
    i == 4 && j == 2 && return tab.a42
    i == 4 && j == 3 && return tab.a43
    i == 5 && j == 1 && return tab.a51
    i == 5 && j == 2 && return tab.a52
    i == 5 && j == 3 && return tab.a53
    i == 5 && j == 4 && return tab.a54
    i == 6 && j == 1 && return tab.a61
    i == 6 && j == 2 && return tab.a62
    i == 6 && j == 3 && return tab.a63
    i == 6 && j == 4 && return tab.a64
    i == 6 && j == 5 && return tab.a65
    return zero(tab.a21)
end
@inline function _adj_b(tab::AdjTab, s::Int)
    s == 1 && return tab.b1
    s == 2 && return tab.b2
    s == 3 && return tab.b3
    s == 4 && return tab.b4
    s == 5 && return tab.b5
    return tab.b6
end

function _accumulate_drive_grad!(gθ, λx, t, pulse, u_pulse, sqrt_κe)
    λar, λai = λx[1], λx[2]
    (λar == 0 && λai == 0) && return nothing
    _HAVE_FORWARDDIFF || error("drive VJP needs ForwardDiff")
    function s(uu)
        E = build_E_of_t(pulse, uu)(t)
        return sqrt_κe * (λar * real(E) + λai * imag(E))
    end
    gθ .+= ForwardDiff.gradient(s, u_pulse)
    return nothing
end

function tsit5_step_vjp!(λ_n, gθ, λ_np1, u_n, t, dt, p, pulse, u_pulse, tab=ADJTAB)
    M = p[6]
    sqrt_κe = sqrt(real(p[2]))
    nR = length(λ_np1)
    N = state_length_1st_order(M)
    k = ntuple(_ -> zeros(ComplexF64, N), 6)
    y = ntuple(_ -> zeros(ComplexF64, N), 6)
    copyto!(y[1], u_n)
    rhs1!(k[1], y[1], p, t)
    for i in 2:6
        @inbounds for j in 1:N
            acc = k[1][j] * _adj_a(tab, i, 1)
            for s in 2:(i - 1)
                acc += k[s][j] * _adj_a(tab, i, s)
            end
            y[i][j] = u_n[j] + dt * acc
        end
        rhs1!(k[i], y[i], p, t + (i == 6 ? dt : _adj_c(tab, i) * dt))
    end
    λu = copy(λ_np1)
    λk = [(_adj_b(tab, s) * dt) .* λ_np1 for s in 1:6]
    x = zeros(Float64, nR)
    λy = zeros(Float64, nR)
    for s in 6:-1:1
        pack_state_real!(x, y[s], M)
        ts = t + (s == 1 ? 0.0 : (s == 6 ? dt : _adj_c(tab, s) * dt))
        rhs1_vjp!(λy, λk[s], x, p, ts)
        _accumulate_drive_grad!(gθ, λk[s], ts, pulse, u_pulse, sqrt_κe)
        λu .+= λy
        if s >= 2
            for j in 1:(s - 1)
                λk[j] .+= (dt * _adj_a(tab, s, j)) .* λy
            end
        end
    end
    copyto!(λ_n, λu)
    return λ_n
end

function _checkpoint_indices(n::Integer, stride::Integer)
    idxs = Int[1]
    if stride < n
        k = 1 + stride
        while k < n
            push!(idxs, k)
            k += stride
        end
    end
    idxs[end] != n && push!(idxs, n)
    return idxs
end

function tsit5_forced_step_copy(u, p, t, dt)
    tab = ADJTAB
    N = length(u)
    k = ntuple(_ -> zeros(eltype(u), N), 6)
    y = ntuple(_ -> zeros(eltype(u), N), 6)
    copyto!(y[1], u)
    rhs1!(k[1], y[1], p, t)
    for i in 2:6
        @inbounds for j in 1:N
            acc = k[1][j] * _adj_a(tab, i, 1)
            for s in 2:(i - 1)
                acc += k[s][j] * _adj_a(tab, i, s)
            end
            y[i][j] = u[j] + dt * acc
        end
        rhs1!(k[i], y[i], p, t + (i == 6 ? dt : _adj_c(tab, i) * dt))
    end
    u1 = similar(u)
    @inbounds for j in 1:N
        u1[j] = u[j] + dt * (_adj_b(tab, 1)*k[1][j] + _adj_b(tab, 2)*k[2][j] +
                             _adj_b(tab, 3)*k[3][j] + _adj_b(tab, 4)*k[4][j] +
                             _adj_b(tab, 5)*k[5][j] + _adj_b(tab, 6)*k[6][j])
    end
    return u1
end

function replay_tsit5_window(u_start, p, t_start, dts)
    u = copy(u_start)
    us = Vector{typeof(u)}(undef, length(dts) + 1)
    us[1] = copy(u)
    t = Float64(t_start)
    @inbounds for n in eachindex(dts)
        Δt = Float64(dts[n])
        u = tsit5_forced_step_copy(u, p, t, Δt)
        t += Δt
        us[n + 1] = copy(u)
    end
    return us
end

function record_tsit5_mesh(u0, p, tspan; reltol=1e-8, abstol=1e-8, tstops=Float64[],
                           checkpoint_stride::Integer=typemax(Int), dt0=0.0)
    u, ts, us, nsteps = integrate!(rhs1!, u0, p, tspan; integrator=:tsit5,
                                   reltol=reltol, abstol=abstol, save_states=true,
                                   tsave=isempty(tstops) ? nothing : tstops, dt0=dt0)
    dts = diff(ts)
    stride = max(Int(checkpoint_stride), 1)
    idxs = _checkpoint_indices(length(us), stride)
    stack_u = [copy(us[i]) for i in idxs]
    return ts, dts, us, u, nsteps, idxs, stack_u
end

function reverse_tsit5!(gθ, λx, states, t, dts, p, pulse, u_pulse)
    nstep = length(dts)
    @inbounds for n in nstep:-1:1
        Δt = Float64(dts[n])
        Δt == 0 && continue
        tsit5_step_vjp!(λx, gθ, λx, states[n], t[n], Δt, p, pulse, u_pulse)
    end
    return gθ
end

"""Checkpointed reverse (package `reverse_tsit5_on_checkpoints!`): replay each
window from a stored checkpoint, then VJP. Primary optimizer path."""
function reverse_tsit5_checkpoints!(gθ, λx, ts, dts, idxs, stack_u, p, pulse, u_pulse)
    nchk = length(idxs)
    nchk >= 2 || return reverse_tsit5!(gθ, λx, stack_u, ts, dts, p, pulse, u_pulse)
    @inbounds for w in (nchk - 1):-1:1
        i0 = idxs[w]
        i1 = idxs[w + 1]
        dtsw = @view dts[i0:(i1 - 1)]
        us = replay_tsit5_window(stack_u[w], p, ts[i0], dtsw)
        tloc = ts[i0:i1]
        reverse_tsit5!(gθ, λx, us, collect(tloc), dtsw, p, pulse, u_pulse)
    end
    return gθ
end

# =============================================================================
# §12  Observables + pulse_cost  (current pulse_optimizer2.jl formulation)
# =============================================================================

function _weighted_inversion(Sz, g_b, Nj, ::Type{T}) where {T}
    w = Nj .* abs2.(g_b)
    weight = w ./ (sum(w) + T(1e-30))
    Sz_fraction = real.(Sz) ./ (Nj ./ 2 .+ 1e-30)
    Ij = clamp.((Sz_fraction .+ 1) ./ 2, zero(T), one(T))
    return sum(weight .* Ij)
end

function _frequency_slice_indices(delta_b)
    slices = Vector{Int}[]
    slot = Dict{Float64,Int}()
    @inbounds for j in eachindex(delta_b)
        key = Float64(delta_b[j])
        s = get(slot, key, 0)
        if s == 0
            push!(slices, Int[j])
            slot[key] = length(slices)
        else
            push!(slices[s], j)
        end
    end
    return slices
end

function _weighted_silencing_factor(Sp, g_b, Nj, delta_b, ::Type{T}; eps_seed=WEAK_SEED) where {T}
    slices = _frequency_slice_indices(delta_b)
    num_acc = zero(T)
    den_acc = zero(T)
    for idx in slices
        wg = abs2.(g_b[idx])
        F_num = sum(wg .* Sp[idx])
        F_den = sum(wg .* (convert(T, eps_seed) .* (Nj[idx] ./ 2)))
        F_omega = F_num / (F_den + 1e-30)
        abs_F = sqrt(abs2(F_omega) + 1e-30)
        n_omega = sum(Nj[idx] .* wg)
        num_acc += n_omega * abs_F
        den_acc += n_omega
    end
    return clamp(num_acc / (den_acc + 1e-30), zero(T), one(T))
end

function _fidelity_physics_cost(inversion::T, silencing::T, target_F, I_min, kappa_I, S_min, kappa_S) where {T}
    ss = one(T) - (silencing - convert(T, target_F))^2
    fid = inversion * ss
    J_base = (one(T) - fid)^2
    pen_I = inversion < convert(T, I_min) ? (convert(T, I_min) - inversion) : zero(T)
    pen_S = ss < convert(T, S_min) ? (convert(T, S_min) - ss) : zero(T)
    J_pen = convert(T, 0.5 * kappa_I) * pen_I^2 + convert(T, 0.5 * kappa_S) * pen_S^2
    return J_base + J_pen, fid, ss
end

function _fidelity_gradient_coefficients(inversion, silencing_success, fidelity_phys, I_min, kappa_I, S_min, kappa_S)
    base_I = -2.0 * (1.0 - fidelity_phys) * silencing_success
    base_S = -2.0 * (1.0 - fidelity_phys) * inversion
    pen_I = inversion < I_min ? -Float64(kappa_I) * (Float64(I_min) - inversion) : 0.0
    pen_S = silencing_success < S_min ? -Float64(kappa_S) * (Float64(S_min) - silencing_success) : 0.0
    return base_I + pen_I, base_S + pen_S
end

function inversion_pullback!(λx, Sz, g_b, Nj)
    M = length(Nj)
    fill!(λx, 0)
    w = Nj .* abs2.(g_b)
    wsum = sum(w) + 1e-30
    @inbounds for j in 1:M
        den = Nj[j] / 2 + 1e-30
        frac = real(Sz[j]) / den
        s = (frac + 1) / 2
        ds = (0 <= s <= 1) ? 1.0 : 0.0
        λx[_ri_zr(j, M)] = (w[j] / wsum) * ds * (0.5 / den)
    end
    return λx
end

function silencing_pullback!(λx, Sp, g_b, Nj, delta_b; eps_seed=WEAK_SEED)
    M = length(Nj)
    fill!(λx, 0)
    slices = _frequency_slice_indices(delta_b)
    nω = [sum(Nj[idx] .* abs2.(g_b[idx])) for idx in slices]
    Nsum = sum(nω) + 1e-30
    Fr = zeros(length(slices)); Fi = zeros(length(slices)); Fden = zeros(length(slices))
    absF_star = 0.0
    @inbounds for (s, idx) in enumerate(slices)
        wg = abs2.(g_b[idx])
        fd = eps_seed * sum(wg .* (Nj[idx] ./ 2)) + 1e-30
        fr = 0.0; fi = 0.0
        for (tt, j) in enumerate(idx)
            fr += wg[tt] * real(Sp[j]); fi += wg[tt] * imag(Sp[j])
        end
        Fr[s] = fr / fd; Fi[s] = fi / fd; Fden[s] = fd
        absF_star += nω[s] * sqrt(Fr[s]^2 + Fi[s]^2 + 1e-30)
    end
    absF_star /= Nsum
    dstar = (0 <= absF_star <= 1) ? 1.0 : 0.0
    @inbounds for (s, idx) in enumerate(slices)
        absF = sqrt(Fr[s]^2 + Fi[s]^2 + 1e-30)
        pref = dstar * (nω[s] / Nsum) / absF
        for j in idx
            gj2 = abs2(g_b[j])
            λx[_ri_pr(j, M)] = pref * Fr[s] * gj2 / Fden[s]
            λx[_ri_pi(j, M)] = pref * Fi[s] * gj2 / Fden[s]
        end
    end
    return λx
end

function _ode_p(d, E_of_t)
    return (Float64(d.delta0), Float64(d.kappa_e), Float64(d.kappa_i),
            collect(Float64, real.(d.delta_b)), collect(Float64, real.(d.g_b)),
            Int(d.M), E_of_t)
end

function _pulse_tstops(t_start, t_end, tspan)
    t0, t1 = Float64(tspan[1]), Float64(tspan[2])
    out = Float64[]
    for x in Iterators.flatten((t_start, t_end))
        xv = _primal(x)
        (t0 + 1e-15 < xv <= t1 + 1e-15) && push!(out, min(xv, t1))
    end
    return unique!(sort!(out))
end

function _widen_u0(u0, E_of_t)
    probe = first(u0) + zero(E_of_t(0.0))
    typeof(probe) === eltype(u0) && return u0
    return map(x -> x + zero(E_of_t(0.0)), u0)
end

function solve_1st_order(d, E_of_t, kind::Symbol=:ground; reltol=1e-8, abstol=1e-8,
                         integrator=:tsit5, backend=:auto, nshards=nothing, tsave=nothing,
                         dt0=0.0)
    M = Int(d.M)
    u0 = _widen_u0(build_u0_1st_order(M, d.Nj, Float64, kind), E_of_t)
    p = _ode_p(d, E_of_t)
    tspan = d.timespan
    want_gpu = backend === :gpu || (backend === :auto && cuda_functional() && M >= 64)
    if want_gpu && cuda_functional()
        u, nsteps, how = solve_1st_gpu(u0, p, tspan; reltol=reltol, abstol=abstol,
                                       integrator=integrator, nshards=nshards, tsave=tsave)
        a, Sp, Sz = unpack_state_1st_order_u(u, M)
        return a, collect(Sp), collect(Sz), (backend=:gpu, collective=how, nsteps=nsteps)
    end
    u, ts, us, nsteps = integrate!(rhs1!, u0, p, tspan; integrator=integrator,
                                   reltol=reltol, abstol=abstol, tsave=tsave, dt0=dt0)
    a, Sp, Sz = unpack_state_1st_order_u(u, M)
    return a, collect(Sp), collect(Sz), (backend=:cpu, collective=:none, nsteps=nsteps, t=ts, u=us)
end

function solve_2nd_order(d, E_of_t, kind::Symbol=:ground; reltol=1e-8, abstol=1e-8,
                         integrator=:tsit5, tsave=nothing)
    M = Int(d.M)
    u0 = build_u0_2nd_order(M, d.Nj, Float64, kind)
    mask = make_diag_mask_host(M)
    p = (Float64(d.delta0), Float64(d.kappa_e), Float64(d.kappa_i),
         collect(Float64, d.delta_b), collect(Float64, d.g_b), M, mask, E_of_t)
    u, ts, us, nsteps = integrate!(rhs2!, u0, p, d.timespan; integrator=integrator,
                                   reltol=reltol, abstol=abstol, tsave=tsave)
    st = unpack_state_2nd_order_u(u, M)
    return st, (backend=:cpu, nsteps=nsteps, t=ts, u=us, u_end=u)
end

function pulse_metrics_from_state(Sp, Sz, d)
    T = Float64
    inv = _weighted_inversion(Sz, d.g_b, d.Nj, T)
    sil = _weighted_silencing_factor(Sp, d.g_b, d.Nj, d.delta_b, T)
    return inv, sil
end

function pulse_cost_theta(u, pulse, d; target_F=1.0, w_time=0.15, w_power=0.05, w_tmax=1.0,
                          I_min=0.85, kappa_I=50.0, S_min=0.85, kappa_S=50.0,
                          track=:weak, reltol=1e-8, abstol=1e-8, backend=:auto,
                          integrator=:tsit5, nshards=nothing, dt0=0.0)
    T = Float64
    E = build_E_of_t(pulse, u)
    duration = pulse_duration(pulse, u)
    t_start, t_end, _, cA, _ = decode(pulse, u)
    tstops = _pulse_tstops(t_start, t_end, d.timespan)
    tmax_excess = max(t_end[end] - pulse.T_max, zero(eltype(t_end)))
    tmax_penalty = w_tmax * (tmax_excess / pulse.T_max)^2
    n_cA = length(cA)
    power_penalty = w_power * (sum(abs2, cA ./ pulse.amp_scale) / n_cA)
    if track === :dual
        _, _, Sz, _ = solve_1st_order(d, E, :ground; reltol=reltol, abstol=abstol,
                                      integrator=integrator, backend=backend, nshards=nshards,
                                      dt0=dt0, tsave=tstops)
        inversion = _weighted_inversion(Sz, d.g_b, d.Nj, T)
        _, Sp, _, _ = solve_1st_order(d, E, :weak; reltol=reltol, abstol=abstol,
                                      integrator=integrator, backend=backend, nshards=nshards,
                                      dt0=dt0, tsave=tstops)
    else
        _, Sp, Sz_w, _ = solve_1st_order(d, E, :weak; reltol=reltol, abstol=abstol,
                                         integrator=integrator, backend=backend, nshards=nshards,
                                         dt0=dt0, tsave=tstops)
        inversion = _weighted_inversion(Sz_w, d.g_b, d.Nj, T)
    end
    silencing = _weighted_silencing_factor(Sp, d.g_b, d.Nj, d.delta_b, T)
    physics, fid, ss = _fidelity_physics_cost(inversion, silencing, target_F, I_min, kappa_I, S_min, kappa_S)
    cost = physics + w_time * (duration / pulse.T_max) + tmax_penalty + power_penalty
    return cost, inversion, silencing, duration, fid, ss
end

function pulse_cost_grad_adjoint(u, pulse, d; target_F=1.0, w_time=0.15, w_power=0.05, w_tmax=1.0,
                                 I_min=0.85, kappa_I=50.0, S_min=0.85, kappa_S=50.0,
                                 track=:weak, reltol=1e-8, abstol=1e-8,
                                 checkpoint_stride::Integer=typemax(Int), dt0=0.0)
    M = Int(d.M)
    E = build_E_of_t(pulse, u)
    p = _ode_p(d, E)
    t_start, t_end, _, cA, _ = decode(pulse, u)
    tstops = _pulse_tstops(t_start, t_end, d.timespan)
    uθ = collect(Float64, u)
    gθ = zeros(Float64, length(uθ))

    function _rev!(gθ, λ, ts, dts, us, idxs, stack)
        if checkpoint_stride < typemax(Int) && length(idxs) >= 2 && length(idxs) < length(us)
            reverse_tsit5_checkpoints!(gθ, λ, ts, dts, idxs, stack, p, pulse, uθ)
        else
            reverse_tsit5!(gθ, λ, us, ts, dts, p, pulse, uθ)
        end
    end

    if track === :dual
        u0g = build_u0_1st_order(M, d.Nj, Float64, :ground)
        tsg, dtsg, usg, u_endg, _, idxg, stkg = record_tsit5_mesh(
            u0g, p, d.timespan; reltol=reltol, abstol=abstol, tstops=tstops,
            checkpoint_stride=checkpoint_stride, dt0=dt0)
        _, _, Sz = unpack_state_1st_order_u(u_endg, M)
        Sz = collect(Sz)
        inversion = _weighted_inversion(Sz, d.g_b, d.Nj, Float64)
        u0w = build_u0_1st_order(M, d.Nj, Float64, :weak)
        tsw, dtsw, usw, u_endw, _, idxw, stkw = record_tsit5_mesh(
            u0w, p, d.timespan; reltol=reltol, abstol=abstol, tstops=tstops,
            checkpoint_stride=checkpoint_stride, dt0=dt0)
        _, Sp, _ = unpack_state_1st_order_u(u_endw, M)
        Sp = collect(Sp)
        silencing = _weighted_silencing_factor(Sp, d.g_b, d.Nj, d.delta_b, Float64)
        physics, fid, ss = _fidelity_physics_cost(inversion, silencing, target_F, I_min, kappa_I, S_min, kappa_S)
        cI, cS = _fidelity_gradient_coefficients(inversion, ss, fid, I_min, kappa_I, S_min, kappa_S)
        λI = zeros(Float64, real_state_length_1st_order(M))
        inversion_pullback!(λI, Sz, d.g_b, d.Nj); λI .*= cI
        _rev!(gθ, λI, tsg, dtsg, usg, idxg, stkg)
        λS = zeros(Float64, real_state_length_1st_order(M))
        silencing_pullback!(λS, Sp, d.g_b, d.Nj, d.delta_b); λS .*= cS
        _rev!(gθ, λS, tsw, dtsw, usw, idxw, stkw)
    else
        u0 = build_u0_1st_order(M, d.Nj, Float64, :weak)
        ts, dts, us, u_end, _, idxs, stack = record_tsit5_mesh(
            u0, p, d.timespan; reltol=reltol, abstol=abstol, tstops=tstops,
            checkpoint_stride=checkpoint_stride, dt0=dt0)
        _, Sp, Sz = unpack_state_1st_order_u(u_end, M)
        Sp = collect(Sp); Sz = collect(Sz)
        inversion = _weighted_inversion(Sz, d.g_b, d.Nj, Float64)
        silencing = _weighted_silencing_factor(Sp, d.g_b, d.Nj, d.delta_b, Float64)
        physics, fid, ss = _fidelity_physics_cost(inversion, silencing, target_F, I_min, kappa_I, S_min, kappa_S)
        cI, cS = _fidelity_gradient_coefficients(inversion, ss, fid, I_min, kappa_I, S_min, kappa_S)
        λ = zeros(Float64, real_state_length_1st_order(M))
        inversion_pullback!(λ, Sz, d.g_b, d.Nj); λ .*= cI
        λS = zeros(Float64, length(λ))
        silencing_pullback!(λS, Sp, d.g_b, d.Nj, d.delta_b)
        λ .+= cS .* λS
        _rev!(gθ, λ, ts, dts, us, idxs, stack)
    end

    if _HAVE_FORWARDDIFF
        function direct(uu)
            dur = pulse_duration(pulse, uu)
            _, te, _, cA2, _ = decode(pulse, uu)
            ex = max(te[end] - pulse.T_max, zero(eltype(uu)))
            tmax_pen = w_tmax * (ex / pulse.T_max)^2
            pow = w_power * (sum(abs2, cA2 ./ pulse.amp_scale) / length(cA2))
            return w_time * (dur / pulse.T_max) + tmax_pen + pow
        end
        gθ .+= ForwardDiff.gradient(direct, uθ)
    end
    duration = pulse_duration(pulse, u)
    tmax_excess = max(t_end[end] - pulse.T_max, 0.0)
    power_penalty = w_power * (sum(abs2, cA ./ pulse.amp_scale) / length(cA))
    cost = physics + w_time * (duration / pulse.T_max) + w_tmax * (tmax_excess / pulse.T_max)^2 + power_penalty
    return gθ, cost, inversion, silencing, duration
end

"""Dual-through-solve gradient (non-hot path). Kept for parity tests vs adjoint."""
function pulse_cost_grad_forward(u, pulse, d; kwargs...)
    _HAVE_FORWARDDIFF || error("grad=:forward requires ForwardDiff")
    uθ = collect(Float64, u)
    cost, inv, sil, dur, _, _ = pulse_cost_theta(uθ, pulse, d; kwargs...)
    g = ForwardDiff.gradient(uu -> pulse_cost_theta(uu, pulse, d; kwargs...)[1], uθ)
    return g, cost, inv, sil, dur
end

# =============================================================================
# §13  Adam on B-spline parameters only
# =============================================================================

mutable struct AdamState
    m::Vector{Float64}
    v::Vector{Float64}
    t::Int
end
AdamState(n::Integer) = AdamState(zeros(n), zeros(n), 0)

function adam_step!(u, grad, state; lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, lr_scale=nothing)
    state.t += 1
    @inbounds for i in eachindex(u)
        state.m[i] = beta1 * state.m[i] + (1 - beta1) * grad[i]
        state.v[i] = beta2 * state.v[i] + (1 - beta2) * grad[i]^2
        m_hat = state.m[i] / (1 - beta1^state.t)
        v_hat = state.v[i] / (1 - beta2^state.t)
        step = lr * m_hat / (sqrt(v_hat) + eps)
        u[i] -= lr_scale === nothing ? step : lr_scale[i] * step
    end
    return u
end

function optimize_bspline!(u, pulse, d; num_epochs=20, learning_rate=0.05, patience=8,
                           w_time=0.15, w_power=0.05, w_tmax=1.0, target_F=1.0,
                           I_min=0.85, kappa_I=50.0, S_min=0.85, kappa_S=50.0,
                           track=:weak, reltol=1e-8, abstol=1e-8, cf_lr_scale=0.25,
                           grad::Symbol=:adjoint, checkpoint_stride::Integer=typemax(Int))
    n = length(u)
    n == n_params(pulse) || error("optimizer: length(u)=$(length(u)) != n_params=$(n_params(pulse))")
    grad in (:adjoint, :forward) || error("grad must be :adjoint or :forward, got $grad")
    adam = AdamState(n)
    lr_scale = ones(n)
    k = pulse.k
    nf = k * pulse.n_coeff_f
    lr_scale[end-nf+1:end] .= cf_lr_scale
    best_u = copy(u)
    best_cost = Inf
    wait = 0
    hist = NamedTuple[]
    for epoch in 1:num_epochs
        if grad === :forward
            g, cost, inv, sil, dur = pulse_cost_grad_forward(
                u, pulse, d; target_F=target_F, w_time=w_time, w_power=w_power, w_tmax=w_tmax,
                I_min=I_min, kappa_I=kappa_I, S_min=S_min, kappa_S=kappa_S, track=track,
                reltol=reltol, abstol=abstol)
        else
            g, cost, inv, sil, dur = pulse_cost_grad_adjoint(
                u, pulse, d; target_F=target_F, w_time=w_time, w_power=w_power, w_tmax=w_tmax,
                I_min=I_min, kappa_I=kappa_I, S_min=S_min, kappa_S=kappa_S, track=track,
                reltol=reltol, abstol=abstol, checkpoint_stride=checkpoint_stride)
        end
        push!(hist, (epoch=epoch, cost=cost, inversion=inv, silencing=sil, duration=dur))
        @printf("[optimizer] epoch %d  cost=%.6g  I=%.4f  S=%.4f  T=%.3g\n", epoch, cost, inv, sil, dur)
        if cost < best_cost
            best_cost = cost
            best_u = copy(u)
            wait = 0
        else
            wait += 1
            wait >= patience && break
        end
        adam_step!(u, g, adam; lr=learning_rate, lr_scale=lr_scale)
    end
    copyto!(u, best_u)
    return u, best_cost, hist
end

# =============================================================================
# §14  B-spline parameterization of a raw pulse
# =============================================================================

function _instantaneous_frequency(t, I, Q)
    n = length(t)
    phi = atan.(Q, I)
    @inbounds for j in 2:n
        d = phi[j] - phi[j-1]
        while d > pi
            phi[j] -= 2pi
            d = phi[j] - phi[j-1]
        end
        while d < -pi
            phi[j] += 2pi
            d = phi[j] - phi[j-1]
        end
    end
    f = zeros(n)
    f[1] = (phi[2] - phi[1]) / (t[2] - t[1])
    f[n] = (phi[n] - phi[n-1]) / (t[n] - t[n-1])
    @inbounds for j in 2:n-1
        f[j] = (phi[j+1] - phi[j-1]) / (t[j+1] - t[j-1])
    end
    return phi, f
end

function _detect_subpulse_segments(A; rel_thresh=1e-3, min_active=5, min_silence=3)
    n = length(A)
    thresh = rel_thresh * maximum(A)
    active = A .>= thresh
    j = 1
    while j <= n
        if !active[j]
            j0 = j
            while j <= n && !active[j]
                j += 1
            end
            if (j - j0) < min_silence && j0 > 1 && j <= n
                active[j0:j-1] .= true
            end
        else
            j += 1
        end
    end
    segs = Tuple{Int,Int}[]
    j = 1
    while j <= n
        if active[j]
            j0 = j
            while j <= n && active[j]
                j += 1
            end
            (j - j0) >= min_active && push!(segs, (j0, j - 1))
        else
            j += 1
        end
    end
    return segs
end

function _encode_scaled_softplus(val, scale)
    y = val / max(scale, 1e-30)
    y <= 1e-12 && return _softplus_inv(1e-12)
    return _softplus_inv(y)
end

function _split_k_segments(n, k)
    k = max(1, min(k, n))
    counts = fill(n ÷ k, k)
    counts[1] += n - sum(counts)
    segs = Tuple{Int,Int}[]
    o = 0
    for c in counts
        push!(segs, (o + 1, o + c))
        o += c
    end
    return segs
end

function fit_raw_pulse_bspline(E_of_t, d, bsp; N_fit=2000, force_k=nothing)
    t = collect(range(0.0, d.timespan[2]; length=N_fit))
    Et = ComplexF64[E_of_t(tt) for tt in t]
    I = real.(Et); Q = imag.(Et)
    A = hypot.(I, Q)
    phi, f = _instantaneous_frequency(t, I, Q)
    segs = _detect_subpulse_segments(A)
    k_want = force_k === nothing ? Int(bsp.k) : Int(force_k)
    if isempty(segs)
        segs = _split_k_segments(N_fit, k_want)
    elseif length(segs) != k_want
        i0, i1 = segs[1][1], segs[end][2]
        segs = [(i0 + s[1] - 1, i0 + s[2] - 1) for s in _split_k_segments(i1 - i0 + 1, k_want)]
    end
    k = length(segs)
    nA = Int(bsp.n_coeff_A); nf = Int(bsp.n_coeff_f); deg = Int(bsp.degree)
    pulse = CompositePulse(k, nA, nf, d; degree=deg, taper_frac=bsp.taper_frac)
    raw_gap = zeros(k); raw_dur = zeros(k); raw_phi0 = zeros(k)
    raw_cA = zeros(nA, k); raw_cf = zeros(nf, k)
    t_prev = 0.0
    for (idx, (i0, i1)) in enumerate(segs)
        ts, te = t[i0], t[i1]
        gap = max(ts - t_prev, 0.0)
        dur = max(te - ts - pulse.dur_floor, 0.0)
        raw_gap[idx] = _encode_scaled_softplus(gap, pulse.gap_scale)
        raw_dur[idx] = _encode_scaled_softplus(dur, pulse.dur_scale)
        ts = t_prev + pulse.gap_scale * _softplus(raw_gap[idx])
        te = ts + pulse.dur_scale * _softplus(raw_dur[idx]) + pulse.dur_floor
        tseg = view(t, i0:i1); Aseg = view(A, i0:i1); fseg = view(f, i0:i1); phis = view(phi, i0:i1)
        nseg = length(tseg)
        knotsA = make_clamped_knots(nA, ts, te, deg)
        BA = zeros(nseg, nA)
        @inbounds for j in 1:nseg
            BA[j, :] .= bspline_basis(tseg[j], knotsA, deg)
        end
        tw = [_taper_window(tseg[j], ts, te, pulse.taper_frac) for j in 1:nseg]
        cA = max.((BA .* tw) \ collect(Aseg), 1e-12 * pulse.amp_scale)
        raw_cA[:, idx] .= _softplus_inv.(cA ./ pulse.amp_scale)
        knotsf = make_clamped_knots(nf, ts, te, deg)
        Bf = zeros(nseg, nf)
        @inbounds for j in 1:nseg
            Bf[j, :] .= bspline_basis(tseg[j], knotsf, deg)
        end
        cf = Bf \ collect(fseg)
        raw_cf[:, idx] .= clamp.(cf ./ pulse.freq_scale, -20.0, 20.0)
        raw_phi0[idx] = phis[1]
        t_prev = te
    end
    u = pack(pulse, raw_gap, raw_dur, raw_phi0, raw_cA, raw_cf)
    return pulse, u, segs
end

# =============================================================================
# §15  Settings loader + modes
# =============================================================================

function _named(x)
    x isa NamedTuple && return x
    x isa AbstractDict && return (; (Symbol(k) => _named(v) for (k, v) in x)...)
    return x
end

function _deep_named(x)
    if x isa AbstractDict
        return (; (Symbol(k) => _deep_named(v) for (k, v) in x)...)
    elseif x isa AbstractVector && !isempty(x) && (first(x) isa AbstractDict || first(x) isa NamedTuple)
        return Tuple(_deep_named(v) for v in x)
    else
        return x
    end
end

function load_settings_jl(path::AbstractString)
    env = Module(:MonolithSettings)
    Core.eval(env, :(include(p) = Base.include($env, p)))
    Base.include(env, abspath(path))
    gete(s, default) = isdefined(env, s) ? getfield(env, s) : default
    mode = _sym(gete(:MODE, :forward))
    sim = gete(:SIM_SETTING, nothing)
    sys = gete(:SYSTEM_CONFIG, nothing)
    pulse = gete(:PULSE_CONFIG, nothing)
    sim === nothing && error("$path must define SIM_SETTING")
    sys === nothing && error("$path must define SYSTEM_CONFIG")
    pulse === nothing && error("$path must define PULSE_CONFIG")
    bsp = gete(:BSPLINE, default_bspline())
    opt = gete(:OPTIMIZER, default_optimizer())
    cmp = gete(:COMPUTE, default_compute())
    return (mode=mode, sim=sim, sys=sys, pulse=pulse, bspline=_named(bsp),
            optimizer=_named(opt), compute=_named(cmp))
end

function load_settings_json(path::AbstractString)
    _HAVE_JSON3[] || _try_using(:JSON3) || error("JSON settings require JSON3")
    raw = JSON3.read(read(path, String))
    dd = _deep_named(raw)
    mode = _sym(_get(dd, :mode, :forward))
    return (mode=mode, sim=_named(dd.sim), sys=_named(dd.system), pulse=dd.pulse,
            bspline=_named(_get(dd, :bspline, default_bspline())),
            optimizer=_named(_get(dd, :optimizer, default_optimizer())),
            compute=_named(_get(dd, :compute, default_compute())))
end

function load_settings(path::AbstractString)
    ext = lowercase(splitext(path)[2])
    ext == ".json" && return load_settings_json(path)
    return load_settings_jl(path)
end

function merge_full_config(sim, sys)
    return merge(sim, sys)
end

function summarize_result(mode, d, extra)
    println("mode            : ", mode)
    println("ensemble        : ", d.ensemble_method, "  M=", d.M,
            "  (M_δ×M_g = ", d.M_delta, "×", d.M_g, ")")
    println("κₑ, κᵢ, κₜ      : ", d.kappa_e, ", ", d.kappa_i, ", ", d.kappa_t)
    println("N (cooperativity): ", d.N)
    for (k, v) in pairs(extra)
        println(rpad(string(k), 16), ": ", v)
    end
    return nothing
end

# =============================================================================
# §16  Public API
#   settings → prepare(ensemble_method=:auto|:quadrature|:histogram)
#            → forward / forward_bspline(u) / order2 / order2_bspline(u)
#            → optimize(u; grad=:adjoint|:forward)
# =============================================================================

struct Prepared
    settings
    d
    kind::Symbol
    backend::Symbol
    integrator::Symbol
    nshards
    reltol::Float64
    abstol::Float64
    E_raw
    bspline
end

function prepare(src; ensemble_method::Symbol=:auto)
    settings = src isa AbstractString ? load_settings(src) : src
    CONFIG = merge_full_config(settings.sim, settings.sys)
    d = prepare_derived(CONFIG; ensemble_method=ensemble_method)
    cmp = settings.compute
    return Prepared(
        settings, d,
        _sym(_get(settings.sim, :initial_condition, :ground)),
        _sym(_get(cmp, :backend, :auto)),
        _sym(_get(cmp, :integrator, :tsit5)),
        _get(cmp, :nshards, nothing),
        Float64(_get(settings.sim, :reltol, 1e-8)),
        Float64(_get(settings.sim, :abstol, 1e-8)),
        build_E_of_t_raw(settings.pulse),
        settings.bspline)
end

function _ensure_pulse(prep::Prepared, u=nothing)
    pulse, uθ, segs = fit_raw_pulse_bspline(prep.E_raw, prep.d, prep.bspline; force_k=prep.bspline.k)
    println("[bspline] k=", pulse.k, " n_params=", n_params(pulse),
            " layout=3k + k*nA + k*nf = ", 3 * pulse.k, "+", pulse.k * pulse.n_coeff_A,
            "+", pulse.k * pulse.n_coeff_f, "  segments=", segs)
    if u !== nothing
        length(u) == n_params(pulse) ||
            error("u length $(length(u)) != n_params=$(n_params(pulse))")
        uθ = collect(Float64, u)
    end
    return pulse, uθ
end

function _solve_first(prep::Prepared, E, kind)
    return solve_1st_order(prep.d, E, kind; reltol=prep.reltol, abstol=prep.abstol,
                           integrator=prep.integrator, backend=prep.backend,
                           nshards=prep.nshards, tsave=prep.d.t_save)
end

function _solve_second(prep::Prepared, E, kind)
    return solve_2nd_order(prep.d, E, kind; reltol=prep.reltol, abstol=prep.abstol,
                           integrator=prep.integrator, tsave=prep.d.t_save)
end

function forward(prep::Prepared; kind=prep.kind)
    a, Sp, Sz, info = _solve_first(prep, prep.E_raw, kind)
    inv, sil = pulse_metrics_from_state(Sp, Sz, prep.d)
    extra = (a=a, inversion=inv, silencing=sil, backend=info.backend,
             collective=get(info, :collective, :none), nsteps=info.nsteps)
    summarize_result(:forward, prep.d, extra)
    return (mode=:forward, d=prep.d, a=a, Sp=Sp, Sz=Sz, inversion=inv, silencing=sil,
            pulse=nothing, u=nothing, info=info)
end

function forward_bspline(prep::Prepared, u=nothing; kind=prep.kind)
    pulse, uθ = _ensure_pulse(prep, u)
    E = build_E_of_t(pulse, uθ)
    a, Sp, Sz, info = _solve_first(prep, E, kind)
    inv, sil = pulse_metrics_from_state(Sp, Sz, prep.d)
    extra = (a=a, inversion=inv, silencing=sil, backend=info.backend,
             collective=get(info, :collective, :none), nsteps=info.nsteps)
    summarize_result(:forward_bspline, prep.d, extra)
    return (mode=:forward_bspline, d=prep.d, a=a, Sp=Sp, Sz=Sz, inversion=inv, silencing=sil,
            pulse=pulse, u=uθ, info=info)
end

function order2(prep::Prepared; kind=prep.kind)
    st, info = _solve_second(prep, prep.E_raw, kind)
    a = st[1]; Sp = st[4]; Sz = st[5]; SmSp = st[11]
    inv, sil = pulse_metrics_from_state(collect(Sp), collect(Sz), prep.d)
    extra = (a=a, inversion=inv, silencing=sil, SmSp_mean=sum(real, SmSp) / length(SmSp),
             nsteps=info.nsteps)
    summarize_result(:order2, prep.d, extra)
    return (mode=:order2, d=prep.d, state=st, inversion=inv, silencing=sil,
            pulse=nothing, u=nothing, info=info)
end

function order2_bspline(prep::Prepared, u=nothing; kind=prep.kind)
    pulse, uθ = _ensure_pulse(prep, u)
    E = build_E_of_t(pulse, uθ)
    st, info = _solve_second(prep, E, kind)
    a = st[1]; Sp = st[4]; Sz = st[5]; SmSp = st[11]
    inv, sil = pulse_metrics_from_state(collect(Sp), collect(Sz), prep.d)
    extra = (a=a, inversion=inv, silencing=sil, SmSp_mean=sum(real, SmSp) / length(SmSp),
             nsteps=info.nsteps)
    summarize_result(:order2_bspline, prep.d, extra)
    return (mode=:order2_bspline, d=prep.d, state=st, inversion=inv, silencing=sil,
            pulse=pulse, u=uθ, info=info)
end

function optimize(prep::Prepared, u=nothing; grad::Symbol=:adjoint, kwargs...)
    pulse, uθ = _ensure_pulse(prep, u)
    opt = prep.settings.optimizer
    gsym = _sym(grad)
    uθ, best, hist = optimize_bspline!(
        uθ, pulse, prep.d;
        num_epochs=Int(_get(opt, :num_epochs, 20)),
        learning_rate=Float64(_get(opt, :learning_rate, 0.05)),
        patience=Int(_get(opt, :patience, 8)),
        w_time=Float64(_get(opt, :w_time, 0.15)),
        w_power=Float64(_get(opt, :w_power, 0.05)),
        w_tmax=Float64(_get(opt, :w_tmax, 1.0)),
        target_F=Float64(_get(opt, :target_F, 1.0)),
        I_min=Float64(_get(opt, :I_min, 0.85)),
        kappa_I=Float64(_get(opt, :kappa_I, 50.0)),
        S_min=Float64(_get(opt, :S_min, 0.85)),
        kappa_S=Float64(_get(opt, :kappa_S, 50.0)),
        track=_sym(_get(opt, :track, :weak)),
        reltol=prep.reltol, abstol=prep.abstol,
        cf_lr_scale=Float64(_get(opt, :cf_lr_scale, 0.25)),
        grad=gsym,
        checkpoint_stride=Int(_get(opt, :checkpoint_stride, typemax(Int))),
        kwargs...)
    extra = (best_cost=best, n_params=n_params(pulse), epochs=length(hist), grad=gsym)
    summarize_result(:optimizer, prep.d, extra)
    return (mode=:optimizer, d=prep.d, pulse=pulse, u=uθ, best_cost=best, history=hist, grad=gsym)
end

function run_mode(settings; mode_override=nothing, grad=nothing)
    mode = _canon_mode(mode_override === nothing ? settings.mode : mode_override)
    want = _sym(_get(merge_full_config(settings.sim, settings.sys), :ensemble_method, :auto))
    prep = prepare(settings; ensemble_method=want)
    if mode === :forward
        return forward(prep)
    elseif mode === :forward_bspline
        return forward_bspline(prep)
    elseif mode === :order2
        return order2(prep)
    elseif mode === :order2_bspline
        return order2_bspline(prep)
    elseif mode === :optimizer
        g = grad === nothing ? _sym(_get(settings.optimizer, :grad, :adjoint)) : _sym(grad)
        return optimize(prep; grad=g)
    end
end

function parse_cli(args)
    settings = nothing
    mode = nothing
    grad = nothing
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--settings" || a == "-s"
            i += 1
            settings = args[i]
        elseif a == "--mode" || a == "-m"
            i += 1
            mode = Symbol(args[i])
        elseif a == "--grad" || a == "-g"
            i += 1
            grad = Symbol(args[i])
        elseif a == "--help" || a == "-h"
            println("Usage: julia --project=. scripts/nude_quad_monolith.jl --settings FILE [--mode MODE] [--grad adjoint|forward]")
            println("Modes: forward | forward_bspline | order2 | order2_bspline | optimizer")
            return nothing
        elseif settings === nothing && !startswith(a, "-")
            settings = a
        else
            error("unknown CLI arg: $a")
        end
        i += 1
    end
    settings === nothing && error("pass --settings PATH")
    return (settings=settings, mode=mode, grad=grad)
end

function main(args=ARGS)
    _load_optional_stacks!()
    cli = parse_cli(args)
    cli === nothing && return 0
    settings = load_settings(cli.settings)
    run_mode(settings; mode_override=cli.mode, grad=cli.grad)
    return 0
end

export pack, unpack, decode, n_params, CompositePulse, initial_guess
export build_u0_1st_order, build_u0_2nd_order, build_u0_2nd_mgpu
export smsp_same_product, szsz_same_product, spsp_same_product, szsp_same_product
export rhs1!, rhs2!, rhs_1st_order!
export prepare_derived, ensemble_method_for, resolve_ensemble_method
export prepare, forward, forward_bspline, order2, order2_bspline, optimize
export pulse_cost_theta, pulse_cost_grad_adjoint, pulse_cost_grad_forward, optimize_bspline!
export fit_raw_pulse_bspline, run_mode, load_settings, main
export state_length_1st_order, state_length_2nd_order

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    NudeQuadMonolith._load_optional_stacks!()
    NudeQuadMonolith.main(ARGS)
end
