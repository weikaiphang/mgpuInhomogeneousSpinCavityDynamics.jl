
module AccelSolver1stOrder

using CUDA
using LinearAlgebra
using OrdinaryDiffEq


struct System
    C_ens::Float64
    delta0::Float64
    kappa_e::Float64
    kappa_i::Float64
    FWHM::Float64
    freq_kind::Symbol
    freq_n::Int
    g_kind::Symbol
    g_mean::Float64
    g_std::Float64
    g_value::Float64
    g_hwhm::Float64
    g_span::Float64
    g_alpha::Float64
    g_min::Float64
    g_max::Float64
    gamma_perp::Float64
    gamma_par::Float64
    w_eq::Float64
end

function System(;
    C_ens=0.6,
    delta0=0.0,
    kappa_e=2π * 1e6,
    kappa_i=0.0,
    FWHM=2π * 1e6,
    freq_kind=:lorentzian,
    freq_n=1,
    g_kind=:gaussian,
    g_mean=2π * 100.0,
    g_std=2π * 0.5,
    g_value=2π * 100.0,
    g_hwhm=2π * 20.0,
    g_span=20.0,
    g_alpha=1.0,
    g_min=2π * 1.0,
    g_max=2π * 1000.0,
    gamma_perp=0.0,
    gamma_par=0.0,
    w_eq=-1.0,
)
    freq_kind in (:lorentzian, :gaussian, :powerlaw) ||
        error("freq_kind must be :lorentzian, :gaussian, or :powerlaw")
    g_kind in (:constant, :gaussian, :lorentzian, :powerlaw) ||
        error("g_kind must be :constant, :gaussian, :lorentzian, or :powerlaw")
    C_ens >= 0 || error("C_ens must be non-negative")
    kappa_e > 0 || error("kappa_e must be positive")
    kappa_i >= 0 || error("kappa_i must be non-negative")
    FWHM > 0 || error("FWHM must be positive")
    gamma_perp >= 0 || error("gamma_perp must be non-negative")
    gamma_par >= 0 || error("gamma_par must be non-negative")
    -1 <= w_eq <= 1 || error("w_eq (equilibrium ⟨σᶻ⟩) must lie in [-1, 1]")
    if freq_kind === :powerlaw
        (isa(freq_n, Integer) && freq_n >= 1) ||
            error("freq_n (Pearson-VII exponent) must be an integer ≥ 1")
    end
    if g_kind === :powerlaw
        0 < g_min < g_max || error("g_kind=:powerlaw needs 0 < g_min < g_max")
    end
    if g_kind === :lorentzian
        g_hwhm > 0 || error("g_kind=:lorentzian needs g_hwhm > 0")
    end
    (g_kind === :constant || g_span > 0) || error("g_span must be positive")
    return System(
        C_ens, delta0, kappa_e, kappa_i, FWHM, freq_kind, Int(freq_n), g_kind,
        g_mean, g_std, g_value, g_hwhm, g_span, g_alpha, g_min, g_max,
        gamma_perp, gamma_par, w_eq,
    )
end

kappa_t(sys::System) = sys.kappa_e + sys.kappa_i
gammaL(sys::System) = sys.FWHM / 2
gaussian_sigma_from_FWHM(FWHM) = FWHM / (2 * sqrt(2 * log(2)))


pearson_w(FWHM, n) = FWHM / (2 * sqrt(2.0^(1 / n) - 1))
pearson_norm(w, n) = 4.0^(n - 1) * factorial(big(n - 1))^2 / (π * w * factorial(big(2n - 2))) |> Float64


function freq_scale(sys::System)
    sys.freq_kind === :lorentzian && return gammaL(sys)
    sys.freq_kind === :gaussian && return gaussian_sigma_from_FWHM(sys.FWHM)
    return pearson_w(sys.FWHM, sys.freq_n)
end


function freq_density_at_zero(sys::System)
    if sys.freq_kind === :lorentzian
        return 1.0 / (π * gammaL(sys))
    elseif sys.freq_kind === :gaussian
        return 1.0 / (sqrt(2π) * gaussian_sigma_from_FWHM(sys.FWHM))
    else
        return pearson_norm(pearson_w(sys.FWHM, sys.freq_n), sys.freq_n)
    end
end


function _pow_integral(lo::Float64, hi::Float64, q::Float64)
    abs(q + 1) < 1e-14 && return log(hi / lo)
    return (hi^(q + 1) - lo^(q + 1)) / (q + 1)
end

function _truncated_gaussian_second_moment(μ::Float64, σ::Float64, span::Float64)
    σ <= 0 && return μ^2
    lo = max(0.0, μ - span * σ)
    hi = μ + span * σ
    hi > lo || return μ^2
    x, w = _gl_on(64, lo, hi)
    invs = 1 / (σ * sqrt(2π))
    p = w .* (invs .* exp.(-0.5 .* ((x .- μ) ./ σ).^2))
    s = sum(p)
    s <= 0 && return μ^2 + σ^2
    return sum((p ./ s) .* (x .^ 2))
end

function g2_avg(sys::System)
    if sys.g_kind === :constant
        return abs2(sys.g_value)
    elseif sys.g_kind === :gaussian
        return _truncated_gaussian_second_moment(sys.g_mean, sys.g_std, sys.g_span)
    elseif sys.g_kind === :lorentzian
        z = sys.g_span
        return sys.g_mean^2 + sys.g_hwhm^2 * (z - atan(z)) / atan(z)
    else
        num = _pow_integral(sys.g_min, sys.g_max, 2 - sys.g_alpha)
        den = _pow_integral(sys.g_min, sys.g_max, -sys.g_alpha)
        return num / den
    end
end


function N_spins(sys::System)
    g2 = g2_avg(sys)
    g2 > 0 || error("g2_avg must be positive")
    ρ0 = freq_density_at_zero(sys)
    ρ0 > 0 || error("detuning density underflows at δ = 0; N cannot be set from C_ens")
    return sys.C_ens * kappa_t(sys) / (2π * g2 * ρ0)
end

has_relaxation(sys::System) = sys.gamma_perp != 0 || sys.gamma_par != 0

function default_system(; kwargs...)
    return System(; kwargs...)
end


abstract type AbstractDrive end

drive_omega(::AbstractDrive, t) = 0.0
drive_phase(::AbstractDrive, t) = 0.0
drive_tstops(::AbstractDrive) = Float64[]
has_sweep(::AbstractDrive) = false
drive_chirp_rate(::AbstractDrive) = 0.0
drive_peak_amp(d::AbstractDrive) = abs(drive_E(d, 0.0))
sweep_range(d::AbstractDrive; pad=0.0) = (-pad, pad)


struct WurstPulse <: AbstractDrive
    t_center::Float64
    duration::Float64
    amp::ComplexF64
    bandwidth::Float64
    n::Float64
    omega0::Float64
    chirp_sign::Float64
    phase0::Float64
    edge_frac::Float64
end

function WurstPulse(;
    t_center,
    duration,
    amp,
    bandwidth,
    n=20.0,
    omega0=0.0,
    chirp_sign=+1.0,
    phase0=0.0,
    edge_frac=1e-4,
)
    duration > 0 || error("WURST duration must be positive")
    bandwidth != 0 || error("WURST bandwidth must be nonzero")
    return WurstPulse(
        Float64(t_center), Float64(duration), ComplexF64(amp), Float64(bandwidth),
        Float64(n), Float64(omega0), Float64(chirp_sign), Float64(phase0), Float64(edge_frac),
    )
end

t_start(p::WurstPulse) = p.t_center - p.duration / 2
t_end(p::WurstPulse) = p.t_center + p.duration / 2
chirp_rate(p::WurstPulse) = p.chirp_sign * p.bandwidth / p.duration
omega_start(p::WurstPulse) = p.omega0 - p.chirp_sign * p.bandwidth / 2
omega_end(p::WurstPulse) = p.omega0 + p.chirp_sign * p.bandwidth / 2

has_sweep(::WurstPulse) = true
drive_chirp_rate(p::WurstPulse) = chirp_rate(p)
drive_peak_amp(p::WurstPulse) = abs(p.amp)
drive_tstops(p::WurstPulse) = [t_start(p), t_end(p)]

function _gate(p::WurstPulse, t)
    ts = t_start(p)
    te = t_end(p)
    edge = max(p.duration * p.edge_frac, eps(Float64))
    return 0.5 * (tanh((t - ts) / edge) - tanh((t - te) / edge))
end

function _envelope(p::WurstPulse, τ)
    return p.amp * (1 - abs(sin(π * (τ - p.duration / 2) / p.duration))^p.n)
end

function _poly_phase(p::WurstPulse, τ)
    return p.phase0 + (p.omega0 - p.chirp_sign * p.bandwidth / 2) * τ +
           0.5 * p.chirp_sign * (p.bandwidth / p.duration) * τ^2
end

function _poly_omega(p::WurstPulse, τ)
    return p.omega0 - p.chirp_sign * p.bandwidth / 2 +
           p.chirp_sign * (p.bandwidth / p.duration) * τ
end


function drive_E(p::WurstPulse, t)
    ts = t_start(p)
    τ = t - ts
    return _gate(p, t) * _envelope(p, τ) * cis(_poly_phase(p, τ))
end


function drive_omega(p::WurstPulse, t)
    ts = t_start(p)
    te = t_end(p)
    (t < ts || t > te) && return 0.0
    return _poly_omega(p, t - ts)
end


function drive_phase(p::WurstPulse, t)
    ts = t_start(p)
    te = t_end(p)
    t <= ts && return p.phase0
    t >= te && return _poly_phase(p, p.duration)
    return _poly_phase(p, t - ts)
end

function sweep_range(p::WurstPulse; pad=0.0)
    lo = min(omega_start(p), omega_end(p)) - pad
    hi = max(omega_start(p), omega_end(p)) + pad
    return lo, hi
end

function default_wurst(sys::System;
    t_center=50e-6,
    duration=50e-6,
    amp=nothing,
    bandwidth=nothing,
    n=20.0,
    omega0=0.0,
    chirp_sign=+1.0,
)
    A = amp === nothing ? 0.5 * sqrt(sys.kappa_e) * 2.0e4 : amp
    bw = bandwidth === nothing ? 5.0 * sys.FWHM : bandwidth
    return WurstPulse(;
        t_center=t_center, duration=duration, amp=A, bandwidth=bw,
        n=n, omega0=omega0, chirp_sign=chirp_sign,
    )
end


struct GaussianDrive <: AbstractDrive
    t0::Float64
    sigma::Float64
    amp::ComplexF64
    omega::Float64
    phase::Float64
    n_sigma::Float64
end

function GaussianDrive(; t0, sigma, amp, omega=0.0, phase=0.0, n_sigma=8.0)
    sigma > 0 || error("GaussianDrive sigma must be positive")
    return GaussianDrive(Float64(t0), Float64(sigma), ComplexF64(amp),
                         Float64(omega), Float64(phase), Float64(n_sigma))
end

function drive_E(d::GaussianDrive, t)
    τ = t - d.t0
    return d.amp * exp(-τ^2 / (2 * d.sigma^2)) * cis(d.omega * τ + d.phase)
end
drive_omega(d::GaussianDrive, t) = d.omega
drive_phase(d::GaussianDrive, t) = d.omega * (t - d.t0) + d.phase
drive_tstops(d::GaussianDrive) = [d.t0 - d.n_sigma * d.sigma, d.t0, d.t0 + d.n_sigma * d.sigma]
drive_peak_amp(d::GaussianDrive) = abs(d.amp)


struct ConstantDrive <: AbstractDrive
    amp::ComplexF64
    omega::Float64
end
ConstantDrive(; amp, omega=0.0) = ConstantDrive(ComplexF64(amp), Float64(omega))
ConstantDrive(amp) = ConstantDrive(ComplexF64(amp), 0.0)

drive_E(d::ConstantDrive, t) = d.amp * cis(d.omega * t)
drive_omega(d::ConstantDrive, t) = d.omega
drive_phase(d::ConstantDrive, t) = d.omega * t
drive_peak_amp(d::ConstantDrive) = abs(d.amp)


struct SechDrive <: AbstractDrive
    t0::Float64
    beta::Float64
    amp::ComplexF64
    mu::Float64
    omega0::Float64
    phase::Float64
    n_width::Float64
end

function SechDrive(; t0, beta, amp, mu=0.0, omega0=0.0, phase=0.0, n_width=12.0)
    beta > 0 || error("SechDrive beta must be positive")
    return SechDrive(Float64(t0), Float64(beta), ComplexF64(amp), Float64(mu),
                     Float64(omega0), Float64(phase), Float64(n_width))
end

function drive_phase(d::SechDrive, t)
    τ = t - d.t0
    return d.phase + d.mu * log(cosh(d.beta * τ)) + d.omega0 * τ
end
drive_omega(d::SechDrive, t) = d.omega0 + d.mu * d.beta * tanh(d.beta * (t - d.t0))
function drive_E(d::SechDrive, t)
    τ = t - d.t0
    return d.amp * sech(d.beta * τ) * cis(drive_phase(d, t))
end
has_sweep(d::SechDrive) = d.mu != 0
drive_chirp_rate(d::SechDrive) = d.mu * d.beta^2
drive_peak_amp(d::SechDrive) = abs(d.amp)
drive_tstops(d::SechDrive) = [d.t0 - d.n_width / d.beta, d.t0, d.t0 + d.n_width / d.beta]
function sweep_range(d::SechDrive; pad=0.0)
    w = abs(d.mu) * d.beta
    return d.omega0 - w - pad, d.omega0 + w + pad
end


struct FuncDrive{F,G} <: AbstractDrive
    f::F
    omega_f::G
    tstops::Vector{Float64}
    chirped::Bool
end


function FuncDrive(f; tstops=Float64[], omega=nothing)
    ts = collect(Float64, tstops)
    omega === nothing && return FuncDrive(f, (t -> 0.0), ts, false)
    return FuncDrive(f, omega, ts, true)
end

drive_E(d::FuncDrive, t) = ComplexF64(d.f(t))
drive_omega(d::FuncDrive, t) = d.chirped ? Float64(d.omega_f(t)) : 0.0
drive_tstops(d::FuncDrive) = d.tstops
has_sweep(d::FuncDrive) = d.chirped
function drive_phase(d::FuncDrive, t)
    d.chirped || return 0.0
    t0 = isempty(d.tstops) ? 0.0 : first(d.tstops)
    n = 1024
    h = (t - t0) / n
    s = 0.5 * (Float64(d.omega_f(t0)) + Float64(d.omega_f(t)))
    @inbounds for k in 1:(n - 1)
        s += Float64(d.omega_f(t0 + k * h))
    end
    return s * h
end


struct DriveSum{T<:Tuple} <: AbstractDrive
    drives::T
end
DriveSum(ds::AbstractDrive...) = DriveSum(ds)

drive_E(d::DriveSum, t) = sum(drive_E(x, t) for x in d.drives)
drive_tstops(d::DriveSum) = sort!(unique!(reduce(vcat, (drive_tstops(x) for x in d.drives); init=Float64[])))
has_sweep(::DriveSum) = false
drive_peak_amp(d::DriveSum) = sum(drive_peak_amp(x) for x in d.drives)


struct Ensemble
    M::Int
    M_delta::Int
    M_g::Int
    method::Symbol
    delta_1d::Vector{Float64}
    g_1d::Vector{Float64}
    p_delta::Vector{Float64}
    p_g::Vector{Float64}
    delta_b::Vector{Float64}
    g_b::Vector{Float64}
    Nj::Vector{Float64}
    N_total::Float64
    g_mean::Float64
    g_std::Float64
    g2_avg::Float64
end



function gausslegendre(n::Int)
    n >= 1 || error("gausslegendre needs n >= 1")
    n == 1 && return [0.0], [2.0]
    k = 1.0:(n - 1)
    β = k ./ sqrt.(4 .* k .^ 2 .- 1.0)
    E = eigen(SymTridiagonal(zeros(n), collect(β)))
    x = E.values
    w = 2.0 .* (E.vectors[1, :] .^ 2)
    p = sortperm(x)
    return x[p], w[p]
end


function gausshermite(n::Int)
    n >= 1 || error("gausshermite needs n >= 1")
    n == 1 && return [0.0], [sqrt(π)]
    k = 1.0:(n - 1)
    β = sqrt.(k ./ 2.0)
    E = eigen(SymTridiagonal(zeros(n), collect(β)))
    x = E.values
    w = sqrt(π) .* (E.vectors[1, :] .^ 2)
    p = sortperm(x)
    return x[p], w[p]
end



function lorentzian_tan_nodes(M_delta::Int, γ::Float64; trim=1e-3)
    M_delta >= 1 || error("M_delta must be positive")
    γ > 0 || error("Lorentzian γ must be positive")
    0 < trim < 0.5 || error("trim must be in (0, 1/2)")
    x, w = gausslegendre(M_delta)
    θmax = (1 - trim) * (π / 2)
    θ = θmax .* x
    p = w .* (θmax / π)
    p ./= sum(p)
    δ = γ .* tan.(θ)
    return δ, p
end


function gaussian_hermite_nodes(n::Int, μ::Float64, σ::Float64)
    n >= 1 || error("n must be positive")
    σ > 0 || error("Gaussian σ must be positive")
    x, w = gausshermite(n)
    p = w ./ sqrt(π)
    p ./= sum(p)
    g = μ .+ σ * sqrt(2) .* x
    return g, p
end


function pearson_tan_nodes(M_delta::Int, w::Float64, n::Int; trim=1e-3)
    M_delta >= 1 || error("M_delta must be positive")
    w > 0 || error("Pearson-VII w must be positive")
    n >= 1 || error("Pearson-VII n must be ≥ 1")
    n == 1 && return lorentzian_tan_nodes(M_delta, w; trim=trim)
    0 < trim < 0.5 || error("trim must be in (0, 1/2)")
    x, wl = gausslegendre(M_delta)
    θmax = (1 - trim) * (π / 2)
    θ = θmax .* x
    p = wl .* (θmax) .* (cos.(θ) .^ (2 * (n - 1)))
    p ./= sum(p)
    δ = w .* tan.(θ)
    return δ, p
end


function _gl_on(n::Int, a::Float64, b::Float64)
    x, w = gausslegendre(n)
    return 0.5 * (b - a) .* x .+ 0.5 * (a + b), 0.5 * (b - a) .* w
end


function lorentzian_trunc_nodes(M::Int, center::Float64, hwhm::Float64, span::Float64)
    M >= 1 || error("M must be positive")
    hwhm > 0 || error("Lorentzian hwhm must be positive")
    span > 0 || error("Lorentzian span must be positive")
    θmax = atan(span)
    x, w = gausslegendre(M)
    θ = θmax .* x
    p = w .* θmax
    p ./= sum(p)
    g = center .+ hwhm .* tan.(θ)
    return g, p
end


function powerlaw_log_nodes(M::Int, g_min::Float64, g_max::Float64, α::Float64)
    M >= 1 || error("M must be positive")
    0 < g_min < g_max || error("powerlaw needs 0 < g_min < g_max")
    lx, w = _gl_on(M, log(g_min), log(g_max))
    g = exp.(lx)
    p = w .* g .* g .^ (-α)
    p ./= sum(p)
    return g, p
end


function _product(delta_1d, p_delta, g_1d, p_g, N)
    M_delta = length(delta_1d)
    M_g = length(g_1d)
    M = M_delta * M_g
    delta_b = Vector{Float64}(undef, M)
    g_b = Vector{Float64}(undef, M)
    Nj = Vector{Float64}(undef, M)
    @inbounds for ig in 1:M_g
        for iδ in 1:M_delta
            j = iδ + (ig - 1) * M_delta
            delta_b[j] = delta_1d[iδ]
            g_b[j] = g_1d[ig]
            Nj[j] = N * p_delta[iδ] * p_g[ig]
        end
    end
    N_total = sum(Nj)
    w = Nj ./ N_total
    gm = sum(w .* g_b)
    g2 = sum(w .* abs2.(g_b))
    gs = sqrt(max(g2 - gm^2, 0.0))
    return M, M_delta, M_g, delta_b, g_b, Nj, N_total, gm, gs, g2
end

function _g_nodes(sys::System, M_g; method, span_sigma, renormalize)
    if sys.g_kind === :constant
        return [Float64(sys.g_value)], [1.0]
    elseif sys.g_kind === :gaussian
        method === :quadrature ||
            error("accel_solver_1st_order: histogram g nodes unavailable in this port")
        μ, σ, span = Float64(sys.g_mean), Float64(sys.g_std), Float64(sys.g_span)
        lo = max(0.0, μ - span * σ)
        hi = μ + span * σ
        g, w = _gl_on(M_g, lo, hi)
        p = w .* exp.(-0.5 .* ((g .- μ) ./ σ) .^ 2) ./ (σ * sqrt(2π))
        renormalize && (p ./= sum(p))
        return g, p
    elseif sys.g_kind === :lorentzian
        return lorentzian_trunc_nodes(M_g, sys.g_mean, sys.g_hwhm, sys.g_span)
    else
        return powerlaw_log_nodes(M_g, sys.g_min, sys.g_max, sys.g_alpha)
    end
end

function _delta_nodes(sys::System, M_delta; method, span, trim, renormalize)
    method === :quadrature ||
        error("accel_solver_1st_order: histogram δ nodes unavailable in this port")
    if sys.freq_kind === :lorentzian
        γ = gammaL(sys)
        δ, p = lorentzian_tan_nodes(M_delta, γ; trim=trim)
        return δ, p, 1.0
    elseif sys.freq_kind === :gaussian
        σ = gaussian_sigma_from_FWHM(sys.FWHM)
        δ, p = gaussian_hermite_nodes(M_delta, 0.0, σ)
        return δ, p, 1.0
    else
        δ, p = pearson_tan_nodes(M_delta, pearson_w(sys.FWHM, sys.freq_n), sys.freq_n; trim=trim)
        return δ, p, 1.0
    end
end

function _finish_ensemble(method, sys, M_delta, M_g, delta_1d, p_delta, g_1d, p_g; n_scale=1.0)
    N = N_spins(sys) * Float64(n_scale)
    M, Md, Mg, delta_b, g_b, Nj, N_total, gm, gs, g2 =
        _product(delta_1d, p_delta, g_1d, p_g, N)
    @assert Md == M_delta
    return Ensemble(
        M, Md, Mg, method,
        delta_1d, g_1d, p_delta, p_g,
        delta_b, g_b, Nj, N_total, gm, gs, g2,
    )
end


function quadrature_ensemble(sys::System; M_delta=48, M_g=8, trim=1e-3)
    δ, pδ, _ = _delta_nodes(sys, M_delta; method=:quadrature, span=2.5, trim=trim, renormalize=true)
    g, pg = _g_nodes(sys, M_g; method=:quadrature, span_sigma=4.0, renormalize=true)
    if any(<=(0), g)
        @warn "quadrature produced non-positive g nodes; the coupling distribution " *
              "is too wide relative to its centre (reduce g_std / g_span, or use g_kind=:powerlaw)" extrema(g)
    end
    return _finish_ensemble(:quadrature, sys, M_delta, M_g, δ, pδ, g, pg)
end

function build_ensemble(sys::System; method=:quadrature, kwargs...)
    method === :quadrature && return quadrature_ensemble(sys; kwargs...)
    error("accel_solver_1st_order: only method=:quadrature is available in this " *
          "drop-in; the :histogram path uses prepare_derived's bins directly.")
end


const WEAK_SEED = 1.0e-3

function unpack_1st(u::AbstractVector, M::Int)
    a = u[1]
    Sp = view(u, 2:1 + M)
    Sz = view(u, 2 + M:1 + 2M)
    return a, Sp, Sz
end

function _frequency_slices(delta_b::AbstractVector)
    n = length(delta_b)
    n == 0 && return Vector{Int}[]
    perm = sortperm(delta_b; by=real)
    scale = maximum(abs, delta_b) + 1.0
    tol = 1e-9 * scale
    slices = Vector{Vector{Int}}()
    ref = real(delta_b[perm[1]])
    cur = Int[]
    for k in perm
        v = real(delta_b[k])
        if isempty(cur) || abs(v - ref) <= tol
            push!(cur, k)
        else
            push!(slices, cur)
            cur = Int[k]
            ref = v
        end
    end
    isempty(cur) || push!(slices, cur)
    return slices
end


function weighted_inversion(Sz, g_b, Nj)
    W = 0.0
    acc = 0.0
    @inbounds for j in eachindex(Nj)
        w = Nj[j] * abs2(g_b[j])
        W += w
        ι = real(Sz[j]) / (Nj[j] / 2 + 1e-30)
        acc += w * clamp((ι + 1) / 2, 0.0, 1.0)
    end
    return acc / (W + 1e-30)
end


function weighted_silencing(Sp, g_b, Nj, delta_b; eps_seed=WEAK_SEED, slices=nothing)
    slices === nothing && (slices = _frequency_slices(delta_b))
    num = 0.0
    den = 0.0
    for idx in slices
        F_num = 0.0 + 0.0im
        F_den = 0.0
        nω = 0.0
        @inbounds for j in idx
            gj2 = abs2(g_b[j])
            F_num += gj2 * Sp[j]
            F_den += gj2 * (eps_seed * Nj[j] / 2)
            nω += Nj[j] * gj2
        end
        Fω = F_num / (F_den + 1e-30)
        absF = sqrt(abs2(Fω) + 1e-30)
        num += nω * absF
        den += nω
    end
    return clamp(num / (den + 1e-30), 0.0, 1.0)
end


const IDX_A = 1
state_length_1st(M) = 1 + 2M

mutable struct IntegratorCache
    live::Vector{Bool}
    eligible::Vector{Bool}
    n_live::Int
    n_eval::Int
    n_rhs_bins::Int
    halfwidth::Float64
end

function IntegratorCache(ens::Ensemble, drive::AbstractDrive; halfwidth, pad)
    M = ens.M


    ωlo, ωhi = sweep_range(drive; pad=0.0)
    lo, hi = -ωhi - pad, -ωlo + pad
    eligible = Vector{Bool}(undef, M)
    @inbounds for j in 1:M
        δ = ens.delta_b[j]
        eligible[j] = (δ >= lo) & (δ <= hi)
    end
    live = fill(false, M)
    return IntegratorCache(live, eligible, 0, 0, 0, Float64(halfwidth))
end

function default_halfwidth(sys::System, drive::AbstractDrive; n_win=4.0)
    k = abs(drive_chirp_rate(drive))



    a_ss = 2 * sqrt(sys.kappa_e) * drive_peak_amp(drive) / max(kappa_t(sys), eps(Float64))
    g0 = sys.g_kind === :constant ? abs(sys.g_value) :
         sys.g_kind === :powerlaw ? sqrt(g2_avg(sys)) : abs(sys.g_mean)
    Ω = 2 * g0 * a_ss
    return n_win * max(sqrt(k), Ω, eps(Float64))
end

_activation_start(drive::AbstractDrive) = isempty(drive_tstops(drive)) ? -Inf : first(drive_tstops(drive))

function _activate!(cache::IntegratorCache, ens::Ensemble, drive::AbstractDrive, t)



    t < _activation_start(drive) && return nothing
    ω = drive_omega(drive, t)
    hw = cache.halfwidth
    @inbounds for j in 1:ens.M
        cache.eligible[j] || continue
        cache.live[j] && continue
        if abs(ens.delta_b[j] + ω) <= hw
            cache.live[j] = true
            cache.n_live += 1
        end
    end
    return nothing
end

function build_u0_1st(ens::Ensemble, initial_condition::Symbol)
    M = ens.M
    u0 = zeros(ComplexF64, state_length_1st(M))
    Sz = view(u0, 2 + M:1 + 2M)
    Sp = view(u0, 2:1 + M)
    if initial_condition === :ground
        Sz .= -ens.Nj ./ 2
    elseif initial_condition === :inverted
        Sz .= ens.Nj ./ 2
    elseif initial_condition === :weak
        Sz .= -ens.Nj ./ 2
        Sp .= WEAK_SEED .* ens.Nj ./ 2
    elseif initial_condition === :equator
        Sp .= ens.Nj ./ 2
    else
        error("unknown initial_condition $initial_condition")
    end
    return u0
end


function frame_from_lab(a, Sp, Sz, ens::Ensemble, drive::AbstractDrive, t, frame::Symbol)
    M = ens.M
    u = Vector{ComplexF64}(undef, state_length_1st(M))
    _bcast!(v, x) = (x isa Number ? fill!(v, ComplexF64(x)) : copyto!(v, x))
    us = view(u, 2:1 + M)
    uz = view(u, 2 + M:1 + 2M)
    _bcast!(us, Sp)
    _bcast!(uz, Sz)
    if frame === :lab
        u[1] = a
    elseif frame === :ip
        u[1] = a
        @inbounds for j in 1:M
            us[j] *= cis(-ens.delta_b[j] * t)
        end
    elseif frame === :chirp
        φ = drive_phase(drive, t)
        u[1] = a * cis(-φ)
        @inbounds for j in 1:M
            us[j] *= cis(φ)
        end
    else
        error("unknown frame $frame")
    end
    return u
end


function rhs_lab!(du, u, p, t)
    sys, ens, drive, cache, active = p
    M = ens.M
    a = u[1]
    Sp = view(u, 2:1 + M)
    Sz = view(u, 2 + M:1 + 2M)
    dSp = view(du, 2:1 + M)
    dSz = view(du, 2 + M:1 + 2M)
    cache.n_eval += 1
    active && _activate!(cache, ens, drive, t)

    E = drive_E(drive, t)
    κt = kappa_t(sys)
    gperp = sys.gamma_perp
    gpar = sys.gamma_par
    weq = sys.w_eq
    ac = conj(a)

    src = 0.0 + 0.0im
    @inbounds for j in 1:M
        δj = ens.delta_b[j]
        Spj = Sp[j]
        Szj = Sz[j]

        dSp[j] = 1im * δj * Spj - gperp * Spj
        dSz[j] = -gpar * (Szj - weq * ens.Nj[j] / 2)
        (active && !cache.live[j]) && continue
        gj = ens.g_b[j]
        src += gj * conj(Spj)
        dSp[j] += -2im * gj * ac * Szj
        dSz[j] += -1im * gj * a * Spj + 1im * gj * ac * conj(Spj)
        cache.n_rhs_bins += 1
    end
    du[1] = sqrt(sys.kappa_e) * E - 1im * sys.delta0 * a - 1im * src - 0.5 * κt * a
    return nothing
end


function rhs_ip!(du, u, p, t)
    sys, ens, drive, cache, active = p
    M = ens.M
    a = u[1]
    St = view(u, 2:1 + M)
    Sz = view(u, 2 + M:1 + 2M)
    dSt = view(du, 2:1 + M)
    dSz = view(du, 2 + M:1 + 2M)
    cache.n_eval += 1
    active && _activate!(cache, ens, drive, t)

    E = drive_E(drive, t)
    κt = kappa_t(sys)
    gperp = sys.gamma_perp
    gpar = sys.gamma_par
    weq = sys.w_eq
    ac = conj(a)

    src = 0.0 + 0.0im
    @inbounds for j in 1:M
        Stj = St[j]
        Szj = Sz[j]

        dSt[j] = -gperp * Stj
        dSz[j] = -gpar * (Szj - weq * ens.Nj[j] / 2)
        (active && !cache.live[j]) && continue
        gj = ens.g_b[j]
        phase = cis(ens.delta_b[j] * t)
        Spj = Stj * phase
        src += gj * conj(Spj)
        dSt[j] += -2im * gj * ac * Szj * conj(phase)
        dSz[j] += -1im * gj * a * Spj + 1im * gj * ac * conj(Spj)
        cache.n_rhs_bins += 1
    end
    du[1] = sqrt(sys.kappa_e) * E - 1im * sys.delta0 * a - 1im * src - 0.5 * κt * a
    return nothing
end


function rhs_chirp!(du, u, p, t)
    sys, ens, drive, cache, active = p
    M = ens.M
    ã = u[1]
    Sh = view(u, 2:1 + M)
    Sz = view(u, 2 + M:1 + 2M)
    dSh = view(du, 2:1 + M)
    dSz = view(du, 2 + M:1 + 2M)
    cache.n_eval += 1
    active && _activate!(cache, ens, drive, t)

    φ = drive_phase(drive, t)
    ω = drive_omega(drive, t)
    E_lab = drive_E(drive, t)
    Ẽ = E_lab * cis(-φ)
    κt = kappa_t(sys)
    gperp = sys.gamma_perp
    gpar = sys.gamma_par
    weq = sys.w_eq
    ac = conj(ã)

    src = 0.0 + 0.0im
    @inbounds for j in 1:M
        Spj = Sh[j]
        Szj = Sz[j]
        dSh[j] = 1im * (ens.delta_b[j] + ω) * Spj - gperp * Spj
        dSz[j] = -gpar * (Szj - weq * ens.Nj[j] / 2)
        (active && !cache.live[j]) && continue
        gj = ens.g_b[j]
        src += gj * conj(Spj)
        dSh[j] += -2im * gj * ac * Szj
        dSz[j] += -1im * gj * ã * Spj + 1im * gj * ac * conj(Spj)
        cache.n_rhs_bins += 1
    end
    du[1] = sqrt(sys.kappa_e) * Ẽ - 1im * (sys.delta0 + ω) * ã - 1im * src - 0.5 * κt * ã
    return nothing
end


function lab_state(u, ens::Ensemble, drive::AbstractDrive, t, frame::Symbol)
    M = ens.M
    a = u[1]
    Sp_s = view(u, 2:1 + M)
    Sz = view(u, 2 + M:1 + 2M)
    Sp = Vector{ComplexF64}(undef, M)
    if frame === :lab
        copyto!(Sp, Sp_s)
        return a, Sp, copy(Sz)
    elseif frame === :ip
        @inbounds for j in 1:M
            Sp[j] = Sp_s[j] * cis(ens.delta_b[j] * t)
        end
        return a, Sp, copy(Sz)
    elseif frame === :chirp

        φ = drive_phase(drive, t)
        za = cis(φ)
        zs = cis(-φ)
        @inbounds for j in 1:M
            Sp[j] = Sp_s[j] * zs
        end
        return a * za, Sp, copy(Sz)
    else
        error("unknown frame $frame")
    end
end


const MAX_GPUS = 16

struct GPUPlan
    functional::Bool
    ndev::Int
    nshards::Int
    devices::Vector{Any}
end

gpu_functional() = isdefined(@__MODULE__, :CUDA) && CUDA.functional() && CUDA.ndevices() >= 1

function detect_gpus()
    gpu_functional() || return GPUPlan(false, 0, 0, Any[])
    ndev = min(Int(CUDA.ndevices()), MAX_GPUS)
    return GPUPlan(true, ndev, ndev, Any[dev for (i, dev) in enumerate(CUDA.devices()) if i <= ndev])
end


function gpu_info()
    if !gpu_functional()
        return (functional=false, ndevices=0, names=String[], cap=MAX_GPUS)
    end
    names = [CUDA.name(dev) for dev in CUDA.devices()]
    return (functional=true, ndevices=length(names), names=names, cap=MAX_GPUS)
end

const MIN_BINS_PER_SHARD = 1024


function make_gpu_plan(M::Integer; gpus=:auto, nshards=nothing,
                       min_bins::Integer=MIN_BINS_PER_SHARD)
    gpu_functional() || return GPUPlan(false, 0, 0, Any[])
    all_devs = collect(CUDA.devices())
    ndev = min(length(all_devs), MAX_GPUS)
    ndev >= 1 || return GPUPlan(false, 0, 0, Any[])

    if gpus === :auto
        ns = nshards === nothing ?
             clamp(fld(Int(M), max(Int(min_bins), 1)), 1, ndev) : Int(nshards)
    else
        ns = Int(gpus)
    end
    ns = clamp(ns, 1, MAX_GPUS)
    ns = min(ns, Int(M))
    devices = Any[all_devs[mod1(p, ndev)] for p in 1:ns]
    return GPUPlan(true, ndev, ns, devices)
end

function enable_peer_access!(devices)
    uniq = unique(devices)
    length(uniq) <= 1 && return 1.0
    npair = 0
    nok = 0
    for src in uniq, dst in uniq
        src == dst && continue
        npair += 1
        try
            if CUDA.can_access_peer(src, dst)
                CUDA.device!(src) do
                    CUDA.enable_peer_access(CUDA.context(dst))
                end
                nok += 1
            end
        catch

        end
    end
    return npair == 0 ? 1.0 : nok / npair
end

struct BinPartition
    M::Int
    nshards::Int
    counts::Vector{Int}
    offsets::Vector{Int}
end

function BinPartition(M::Integer, nshards::Integer)
    M >= 1 || error("M must be positive")
    ns = clamp(Int(nshards), 1, Int(M))
    base = div(M, ns)
    rem_ = mod(M, ns)
    counts = [base + (p <= rem_ ? 1 : 0) for p in 1:ns]
    offsets = zeros(Int, ns)
    for p in 2:ns
        offsets[p] = offsets[p - 1] + counts[p - 1]
    end
    return BinPartition(Int(M), ns, counts, offsets)
end

shard_range(part::BinPartition, p::Integer) =
    (part.offsets[p] + 1):(part.offsets[p] + part.counts[p])


const GPU_M_MIN_NATIVE = 8_192
const GPU_M_MIN_WSL    = 32_768
const GPU_BYTES_PER_BIN = 320
const _RES_TTL = 5.0

struct Resources
    ncores::Int
    nthreads::Int
    ngpu::Int
    gpu_free_bytes::Int
    gpu_names::Vector{String}
    is_wsl::Bool
end

const _RES_LOCK = ReentrantLock()
const _RES = Ref{Union{Resources,Nothing}}(nothing)
const _RES_T = Ref(0.0)

function _detect_wsl()
    try
        Sys.islinux() || return false
        haskey(ENV, "WSL_DISTRO_NAME") && return true
        isfile("/proc/version") || return false
        return occursin("microsoft", lowercase(read("/proc/version", String)))
    catch
        return false
    end
end


function detect_resources(; force::Bool=false)
    lock(_RES_LOCK) do
        if !force && _RES[] !== nothing && (time() - _RES_T[]) < _RES_TTL
            return _RES[]
        end
        r = _detect_resources_uncached()
        _RES[] = r
        _RES_T[] = time()
        return r
    end
end

function _detect_resources_uncached()
    ncores = Sys.CPU_THREADS
    nthreads = Threads.nthreads()
    ngpu = 0
    free_bytes = 0
    names = String[]
    if gpu_functional()
        prev = try
            CUDA.device()
        catch
            nothing
        end
        try
            ngpu = min(Int(CUDA.ndevices()), MAX_GPUS)
            frees = Int[]
            for dev in CUDA.devices()
                CUDA.device!(dev)
                push!(names, CUDA.name(dev))
                push!(frees, Int(CUDA.free_memory()))
            end
            free_bytes = isempty(frees) ? 0 : minimum(frees)
        catch
            ngpu = 0
        finally
            try
                prev !== nothing && CUDA.device!(prev)
            catch
            end
        end
    end
    return Resources(ncores, nthreads, ngpu, free_bytes, names, _detect_wsl())
end


function gpu_compute_worthwhile(M::Integer, res::Resources; gpu_M_min::Union{Int,Nothing}=nothing)
    res.ngpu < 1 && return false
    gmin = gpu_M_min !== nothing ? Int(gpu_M_min) : (res.is_wsl ? GPU_M_MIN_WSL : GPU_M_MIN_NATIVE)
    Int(M) >= gmin || return false
    Int(M) ÷ max(res.ngpu, 1) >= MIN_BINS_PER_SHARD || return false
    dev_bytes = Int(M) * GPU_BYTES_PER_BIN
    return dev_bytes < 0.7 * max(res.gpu_free_bytes, 1)
end


const GPU_THREADS = 128
const GPU_N_WARPS = GPU_THREADS ÷ 32

@inline muli(z::Complex{Float64}) = Complex{Float64}(-imag(z), real(z))

@inline function _warp_sum(val::Float64)
    offset = 16
    while offset > 0
        val += CUDA.shfl_down_sync(0xffffffff % UInt32, val, offset)
        offset ÷= 2
    end
    return val
end

@inline function _blockreduce_add!(partial, tid, acc_re::Float64, acc_im::Float64)
    acc_re = _warp_sum(acc_re)
    acc_im = _warp_sum(acc_im)
    sh_re = CuStaticSharedArray(Float64, GPU_N_WARPS)
    sh_im = CuStaticSharedArray(Float64, GPU_N_WARPS)
    lane = (tid - 1) & 31
    wid = (tid - 1) >> 5
    if lane == 0
        @inbounds sh_re[wid + 1] = acc_re
        @inbounds sh_im[wid + 1] = acc_im
    end
    sync_threads()
    if wid == 0
        acc_re = lane < GPU_N_WARPS ? sh_re[lane + 1] : 0.0
        acc_im = lane < GPU_N_WARPS ? sh_im[lane + 1] : 0.0
        acc_re = _warp_sum(acc_re)
        acc_im = _warp_sum(acc_im)
        if lane == 0
            @inbounds partial[Int(blockIdx().x)] = Complex{Float64}(acc_re, acc_im)
        end
    end
    return nothing
end

@inline function _blockreduce_real!(partial, tid, acc::Float64)
    acc = _warp_sum(acc)
    sh = CuStaticSharedArray(Float64, GPU_N_WARPS)
    lane = (tid - 1) & 31
    wid = (tid - 1) >> 5
    if lane == 0
        @inbounds sh[wid + 1] = acc
    end
    sync_threads()
    if wid == 0
        acc = lane < GPU_N_WARPS ? sh[lane + 1] : 0.0
        acc = _warp_sum(acc)
        if lane == 0
            @inbounds partial[Int(blockIdx().x)] = acc
        end
    end
    return nothing
end

@inline function _stage_a(a_eval, a_acc, ka, a_re, a_im, use_dev_a, dt,
                          c1, c2, c3, c4, c5, c6, nused, tid, bid)
    if use_dev_a != Int32(0)
        a = a_acc[1]
        nu = Int(nused)
        nu >= 1 && (a += dt * c1 * ka[1])
        nu >= 2 && (a += dt * c2 * ka[2])
        nu >= 3 && (a += dt * c3 * ka[3])
        nu >= 4 && (a += dt * c4 * ka[4])
        nu >= 5 && (a += dt * c5 * ka[5])
        nu >= 6 && (a += dt * c6 * ka[6])
        if tid == 1 && bid == 1
            a_eval[1] = a
        end
        return a
    end
    return Complex{Float64}(a_re, a_im)
end

function _stage_kernel!(kSp, kSz, ySp, ySz, stage, partial, uSp, uSz,
                        delta, g, Nj, live, eligible,
                        a_eval, a_acc, ka, a_re, a_im, use_dev_a,
                        ω, t, t_start, hw, gperp, gpar, weq,
                        dt, c1, c2, c3, c4, c5, c6, nused, write_y,
                        frame, use_live, n)
    tid = Int(threadIdx().x)
    bid = Int(blockIdx().x)
    gid = (bid - 1) * Int(blockDim().x) + tid
    a = _stage_a(a_eval, a_acc, ka, a_re, a_im, use_dev_a, dt,
                 c1, c2, c3, c4, c5, c6, nused, tid, bid)
    ac = conj(a)
    acc_re = 0.0
    acc_im = 0.0

    if gid <= n
        yp = uSp[gid]
        yz = uSz[gid]
        if nused >= Int32(1)
            yp += dt * c1 * kSp[gid, 1]; yz += dt * c1 * kSz[gid, 1]
        end
        if nused >= Int32(2)
            yp += dt * c2 * kSp[gid, 2]; yz += dt * c2 * kSz[gid, 2]
        end
        if nused >= Int32(3)
            yp += dt * c3 * kSp[gid, 3]; yz += dt * c3 * kSz[gid, 3]
        end
        if nused >= Int32(4)
            yp += dt * c4 * kSp[gid, 4]; yz += dt * c4 * kSz[gid, 4]
        end
        if nused >= Int32(5)
            yp += dt * c5 * kSp[gid, 5]; yz += dt * c5 * kSz[gid, 5]
        end
        if nused >= Int32(6)
            yp += dt * c6 * kSp[gid, 6]; yz += dt * c6 * kSz[gid, 6]
        end
        if write_y != Int32(0)
            ySp[gid] = yp
            ySz[gid] = yz
        end

        if use_live != Int32(0) && t >= t_start && eligible[gid] && !live[gid]
            if abs(delta[gid] + ω) <= hw
                live[gid] = true
            end
        end
        alive = (use_live == Int32(0)) | live[gid]
        gj = g[gid]
        dj = delta[gid]
        relax_z = -gpar * (yz - weq * Nj[gid] * 0.5)

        if frame == Int32(2)
            phase = cis(dj * t)
            Splab = yp * phase
            kp = -gperp * yp
        elseif frame == Int32(1)
            phase = ComplexF64(1.0, 0.0)
            Splab = yp
            kp = muli(dj * yp) - gperp * yp
        else
            phase = ComplexF64(1.0, 0.0)
            Splab = yp
            kp = muli((dj + ω) * yp) - gperp * yp
        end
        kz = relax_z

        if alive
            if frame == Int32(2)
                kp += -2im * gj * ac * yz * conj(phase)
            else
                kp += -2im * gj * ac * yz
            end
            kz += -1im * gj * a * Splab + 1im * gj * ac * conj(Splab)
            sSrc = gj * conj(Splab)
            acc_re = real(sSrc)
            acc_im = imag(sSrc)
        end
        kSp[gid, stage] = kp
        kSz[gid, stage] = kz
    end

    _blockreduce_add!(partial, tid, acc_re, acc_im)
    return nothing
end

@inline function _cavity_da(a, s, E_re, E_im, φ, ω, delta0, halfκt, ske, frame)
    E = Complex{Float64}(E_re, E_im)
    if frame == Int32(3)
        return ske * E * cis(-φ) - 1im * (delta0 + ω) * a - 1im * s - halfκt * a
    end
    return ske * E - 1im * delta0 * a - 1im * s - halfκt * a
end

function _reduce_complex!(out, partial, n)
    tid = Int(threadIdx().x)
    acc_re = 0.0
    acc_im = 0.0
    i = tid
    while i <= n
        z = partial[i]
        acc_re += real(z)
        acc_im += imag(z)
        i += Int(blockDim().x)
    end
    acc_re = _warp_sum(acc_re)
    acc_im = _warp_sum(acc_im)
    sh_re = CuStaticSharedArray(Float64, GPU_N_WARPS)
    sh_im = CuStaticSharedArray(Float64, GPU_N_WARPS)
    lane = (tid - 1) & 31
    wid = (tid - 1) >> 5
    if lane == 0
        sh_re[wid + 1] = acc_re
        sh_im[wid + 1] = acc_im
    end
    sync_threads()
    if wid == 0
        acc_re = lane < GPU_N_WARPS ? sh_re[lane + 1] : 0.0
        acc_im = lane < GPU_N_WARPS ? sh_im[lane + 1] : 0.0
        acc_re = _warp_sum(acc_re)
        acc_im = _warp_sum(acc_im)
        if lane == 0
            out[1] = Complex{Float64}(acc_re, acc_im)
        end
    end
    return nothing
end

function _reduce_complex_cavity!(out, partial, n, ka, stage, a_eval,
                                 E_re, E_im, φ, ω, delta0, halfκt, ske, frame)
    tid = Int(threadIdx().x)
    acc_re = 0.0
    acc_im = 0.0
    i = tid
    while i <= n
        z = partial[i]
        acc_re += real(z)
        acc_im += imag(z)
        i += Int(blockDim().x)
    end
    acc_re = _warp_sum(acc_re)
    acc_im = _warp_sum(acc_im)
    sh_re = CuStaticSharedArray(Float64, GPU_N_WARPS)
    sh_im = CuStaticSharedArray(Float64, GPU_N_WARPS)
    lane = (tid - 1) & 31
    wid = (tid - 1) >> 5
    if lane == 0
        sh_re[wid + 1] = acc_re
        sh_im[wid + 1] = acc_im
    end
    sync_threads()
    if wid == 0
        acc_re = lane < GPU_N_WARPS ? sh_re[lane + 1] : 0.0
        acc_im = lane < GPU_N_WARPS ? sh_im[lane + 1] : 0.0
        acc_re = _warp_sum(acc_re)
        acc_im = _warp_sum(acc_im)
        if lane == 0
            s = Complex{Float64}(acc_re, acc_im)
            out[1] = s
            ka[Int(stage)] = _cavity_da(a_eval[1], s, E_re, E_im, φ, ω, delta0, halfκt, ske, frame)
        end
    end
    return nothing
end

function _reduce_real!(out, partial, n)
    tid = Int(threadIdx().x)
    acc = 0.0
    i = tid
    while i <= n
        acc += partial[i]
        i += Int(blockDim().x)
    end
    acc = _warp_sum(acc)
    sh = CuStaticSharedArray(Float64, GPU_N_WARPS)
    lane = (tid - 1) & 31
    wid = (tid - 1) >> 5
    if lane == 0
        sh[wid + 1] = acc
    end
    sync_threads()
    if wid == 0
        acc = lane < GPU_N_WARPS ? sh[lane + 1] : 0.0
        acc = _warp_sum(acc)
        if lane == 0
            out[1] = acc
        end
    end
    return nothing
end

function _reduce_real_cavity_err!(out, partial, n, a_new, err0, a_acc, ka, dt,
                                  b1, b2, b3, b4, b5, b6,
                                  bt1, bt2, bt3, bt4, bt5, bt6, bt7,
                                  atol, reltol)
    tid = Int(threadIdx().x)
    acc = 0.0
    i = tid
    while i <= n
        acc += partial[i]
        i += Int(blockDim().x)
    end
    acc = _warp_sum(acc)
    sh = CuStaticSharedArray(Float64, GPU_N_WARPS)
    lane = (tid - 1) & 31
    wid = (tid - 1) >> 5
    if lane == 0
        sh[wid + 1] = acc
    end
    sync_threads()
    if wid == 0
        acc = lane < GPU_N_WARPS ? sh[lane + 1] : 0.0
        acc = _warp_sum(acc)
        if lane == 0
            out[1] = acc
            z = a_acc[1]
            z += dt * b1 * ka[1]
            z += dt * b2 * ka[2]
            z += dt * b3 * ka[3]
            z += dt * b4 * ka[4]
            z += dt * b5 * ka[5]
            z += dt * b6 * ka[6]
            a_new[1] = z
            a_err = dt * (bt1 * ka[1] + bt2 * ka[2] + bt3 * ka[3] + bt4 * ka[4] +
                          bt5 * ka[5] + bt6 * ka[6] + bt7 * ka[7])
            sc = atol + reltol * abs(z)
            err0[1] = abs2(a_err) / (sc * sc + 1e-30)
        end
    end
    return nothing
end

function _err_kernel!(partial, kSp, kSz, ySp, ySz,
                      dt, b1, b2, b3, b4, b5, b6, b7, atol, reltol, n)
    tid = Int(threadIdx().x)
    gid = (Int(blockIdx().x) - 1) * Int(blockDim().x) + tid
    acc = 0.0
    if gid <= n
        ep = dt * (b1 * kSp[gid, 1] + b2 * kSp[gid, 2] + b3 * kSp[gid, 3] +
                   b4 * kSp[gid, 4] + b5 * kSp[gid, 5] + b6 * kSp[gid, 6] + b7 * kSp[gid, 7])
        ez = dt * (b1 * kSz[gid, 1] + b2 * kSz[gid, 2] + b3 * kSz[gid, 3] +
                   b4 * kSz[gid, 4] + b5 * kSz[gid, 5] + b6 * kSz[gid, 6] + b7 * kSz[gid, 7])
        scp = atol + reltol * abs(ySp[gid])
        scz = atol + reltol * abs(ySz[gid])
        acc = abs2(ep) / (scp * scp + 1e-30) + abs2(ez) / (scz * scz + 1e-30)
    end
    _blockreduce_real!(partial, tid, acc)
    return nothing
end

function _kcol_copy!(kSp, kSz, dst, src, n, a_acc, a_new, ka, do_cav)
    i = (Int(blockIdx().x) - 1) * Int(blockDim().x) + Int(threadIdx().x)
    if i <= n
        kSp[i, dst] = kSp[i, src]
        kSz[i, dst] = kSz[i, src]
    end
    if do_cav != Int32(0) && i == 1
        a_acc[1] = a_new[1]
        ka[1] = ka[7]
    end
    return nothing
end

function _count_kernel!(partial, live, n)
    tid = Int(threadIdx().x)
    gid = (Int(blockIdx().x) - 1) * Int(blockDim().x) + tid
    acc = (gid <= n && live[gid]) ? 1.0 : 0.0
    _blockreduce_real!(partial, tid, acc)
    return nothing
end


mutable struct GPUShard
    id::Int
    dev::Any
    j0::Int
    mloc::Int
    nblocks::Int
    uSp::Any
    uSz::Any
    ySp::Any
    ySz::Any
    kSp::Any
    kSz::Any
    delta::Any
    g::Any
    Nj::Any
    live::Any
    eligible::Any
    partial::Any
    errpartial::Any
    a_eval::Any
    src1::Any
    err1::Any
    ka::Any
    a_acc::Any
    a_new::Any
    err0::Any
    src_h::Vector{ComplexF64}
    err_h::Vector{Float64}
    stream::Any
    root_src::Any
    root_err::Any
end

struct Tsit5Tab{T}
    c2::T; c3::T; c4::T; c5::T; c6::T
    a21::T
    a31::T; a32::T
    a41::T; a42::T; a43::T
    a51::T; a52::T; a53::T; a54::T
    a61::T; a62::T; a63::T; a64::T; a65::T
    a71::T; a72::T; a73::T; a74::T; a75::T; a76::T
    bt1::T; bt2::T; bt3::T; bt4::T; bt5::T; bt6::T; bt7::T
end

function Tsit5Tab(::Type{T}) where {T}
    Tsit5Tab{T}(
        T(0.161), T(0.327), T(0.9), T(0.9800255409045097), T(1.0),
        T(0.161),
        T(-0.008480655492356989), T(0.335480655492357),
        T(2.8971530571054935), T(-6.359448489975075), T(4.3622954328695815),
        T(5.325864828439257), T(-11.748883564062828), T(7.4955393428898365),
        T(-0.09249506636175525),
        T(5.86145544294642), T(-12.92096931784711), T(8.159367898576159),
        T(-0.071584973281401), T(-0.028269050394068383),
        T(0.09646076681806523), T(0.01), T(0.4798896504144996),
        T(1.379008574103742), T(-3.290069515436081), T(2.324710524099774),
        T(-0.00178001105222577714), T(-0.0008164344596567469),
        T(0.007880878010261995), T(-0.1447110071732629),
        T(0.5823571654525552), T(-0.45808210592918697),
        T(0.015151515151515152),
    )
end

mutable struct PICtrl
    gamma::Float64
    qmin::Float64
    qmax::Float64
    beta1::Float64
    beta2::Float64
    qoldinit::Float64
    qold::Float64
end
PICtrl() = PICtrl(0.9, 0.2, 10.0, 7 / 50, 2 / 25, 1e-4, 1e-4)

_unique_devices(shards) = unique(s.dev for s in shards)

function _pin_host(::Type{T}, n::Int) where {T}
    v = Vector{T}(undef, n)
    try
        return CUDA.pin(v)
    catch
        return v
    end
end

@inline _dense_host(src::Vector) = src
@inline _dense_host(src) = copy(src)

function _cu_copy(src)
    return CuArray(_dense_host(src))
end

function _cu_undef(::Type{T}, dims...) where {T}
    return CuArray{T}(undef, dims...)
end

@inline function _maybe_device!(dev)
    try
        CUDA.device() == dev && return
    catch
    end
    CUDA.device!(dev)
    return
end

function _launch_kernel(s::GPUShard, kernel::F, args...; threads::Int, blocks::Int) where {F}
    st = s.stream
    if st === nothing
        @cuda threads=threads blocks=blocks kernel(args...)
    else
        @cuda threads=threads blocks=blocks stream=st kernel(args...)
    end
    return nothing
end

function _sync_shard(s::GPUShard)
    if s.stream === nothing
        CUDA.synchronize()
    else
        CUDA.synchronize(s.stream)
    end
    return nothing
end

function _for_each_shard(body, shards)
    n = length(shards)
    if n == 1
        body(shards[1])
        return nothing
    end
    prev = nothing
    for s in shards
        if prev != s.dev
            _maybe_device!(s.dev)
            prev = s.dev
        end
        body(s)
    end
    return nothing
end

function _sync_all(shards)
    prev = nothing
    for s in shards
        if prev != s.dev
            _maybe_device!(s.dev)
            prev = s.dev
        end
        _sync_shard(s)
    end
    for d in _unique_devices(shards)
        CUDA.device!(d)
        CUDA.synchronize()
    end
    return nothing
end

function _fill_shard_state!(s::GPUShard, Sp0, Sz0, ens::Ensemble, cache::IntegratorCache, r)
    copyto!(s.uSp, _dense_host(view(Sp0, r)))
    copyto!(s.uSz, _dense_host(view(Sz0, r)))
    copyto!(s.delta, _dense_host(view(ens.delta_b, r)))
    copyto!(s.g, _dense_host(view(ens.g_b, r)))
    copyto!(s.Nj, _dense_host(view(ens.Nj, r)))
    copyto!(s.live, _dense_host(view(cache.live, r)))
    copyto!(s.eligible, _dense_host(view(cache.eligible, r)))
    return nothing
end

function _alloc_shard(p::Int, dev, r, Sp0, Sz0, ens::Ensemble, cache::IntegratorCache;
                      with_stream::Bool=false)
    mloc = length(r)
    nblocks = max(1, cld(mloc, GPU_THREADS))
    CUDA.device!(dev)
    st = nothing
    if with_stream
        st = try
            CUDA.CuStream()
        catch
            nothing
        end
    end
    s = GPUShard(
        p, dev, first(r) - 1, mloc, nblocks,
        _cu_copy(view(Sp0, r)),
        _cu_copy(view(Sz0, r)),
        _cu_undef(ComplexF64, mloc),
        _cu_undef(ComplexF64, mloc),
        _cu_undef(ComplexF64, mloc, 7),
        _cu_undef(ComplexF64, mloc, 7),
        _cu_copy(view(ens.delta_b, r)),
        _cu_copy(view(ens.g_b, r)),
        _cu_copy(view(ens.Nj, r)),
        _cu_copy(view(cache.live, r)),
        _cu_copy(view(cache.eligible, r)),
        _cu_undef(ComplexF64, nblocks),
        _cu_undef(Float64, nblocks),
        _cu_undef(ComplexF64, 1),
        _cu_undef(ComplexF64, 1),
        _cu_undef(Float64, 1),
        _cu_undef(ComplexF64, 7),
        _cu_undef(ComplexF64, 1),
        _cu_undef(ComplexF64, 1),
        _cu_undef(Float64, 1),
        _pin_host(ComplexF64, 1),
        _pin_host(Float64, 1),
        st,
        nothing,
        nothing,
    )
    return s
end

function _build_shards(ens::Ensemble, ::AbstractDrive, u0::Vector{ComplexF64},
                       plan::GPUPlan, cache::IntegratorCache)
    M = ens.M
    part = BinPartition(M, plan.nshards)
    enable_peer_access!(plan.devices)
    Sp0 = view(u0, 2:1 + M)
    Sz0 = view(u0, 2 + M:1 + 2M)
    shards = GPUShard[]
    multi = plan.nshards > 1
    for p in 1:plan.nshards
        r = shard_range(part, p)
        push!(shards, _alloc_shard(p, plan.devices[p], r, Sp0, Sz0, ens, cache;
                                   with_stream=multi))
    end
    if plan.nshards > 1
        root = shards[1]
        CUDA.device!(root.dev)
        root.root_src = CUDA.zeros(ComplexF64, plan.nshards)
        root.root_err = CUDA.zeros(Float64, plan.nshards)
    end
    return shards, part
end

function _reload_shards!(shards, ens::Ensemble, u0::Vector{ComplexF64}, cache::IntegratorCache)
    M = ens.M
    part = BinPartition(M, length(shards))
    Sp0 = view(u0, 2:1 + M)
    Sz0 = view(u0, 2 + M:1 + 2M)
    for (p, s) in enumerate(shards)
        _maybe_device!(s.dev)
        r = shard_range(part, p)
        _fill_shard_state!(s, Sp0, Sz0, ens, cache, r)
    end
    return part
end


mutable struct GPUWorkspace
    nshards::Int
    M::Int
    mlocs::Vector{Int}
    devs::Vector{Any}
    shards::Vector{GPUShard}
end

const _WS_LOCK = ReentrantLock()
const _WS = Ref{Union{GPUWorkspace,Nothing}}(nothing)

function _ws_matches(ws::GPUWorkspace, plan::GPUPlan, M::Int)
    ws.nshards == plan.nshards && ws.M == M || return false
    length(ws.shards) == plan.nshards || return false
    @inbounds for p in 1:plan.nshards
        ws.devs[p] == plan.devices[p] || return false
    end
    return true
end


function free_gpu_workspace!()
    lock(_WS_LOCK) do
        ws = _WS[]
        ws === nothing && return nothing
        free_shards!(ws.shards)
        _WS[] = nothing
        return nothing
    end
end

function _acquire_shards(ens::Ensemble, drive::AbstractDrive, u0::Vector{ComplexF64},
                         plan::GPUPlan, cache::IntegratorCache; reuse::Bool)
    if reuse
        taken = lock(_WS_LOCK) do
            ws = _WS[]
            if ws !== nothing && _ws_matches(ws, plan, ens.M)
                _WS[] = nothing
                return ws.shards
            end
            nothing
        end
        if taken !== nothing
            _reload_shards!(taken, ens, u0, cache)
            return taken, BinPartition(ens.M, length(taken))
        end

        free_gpu_workspace!()
    end
    return _build_shards(ens, drive, u0, plan, cache)
end

function _release_shards!(shards; keep::Bool, reclaim::Bool)
    if keep
        lock(_WS_LOCK) do
            if _WS[] !== nothing

                free_shards!(shards)
            else
                _WS[] = GPUWorkspace(length(shards),
                                     sum(s.mloc for s in shards),
                                     [s.mloc for s in shards],
                                     [s.dev for s in shards],
                                     shards)
            end
        end
    else
        free_shards!(shards)
    end
    if reclaim
        try
            GC.gc(false)
            for d in _unique_devices(shards)
                CUDA.device!(d)
                CUDA.reclaim()
            end
        catch
        end
    end
    return nothing
end


function free_shards!(shards)
    for s in shards
        try
            CUDA.device!(s.dev)
            _sync_shard(s)
            for A in (s.uSp, s.uSz, s.ySp, s.ySz, s.kSp, s.kSz,
                      s.delta, s.g, s.Nj, s.live, s.eligible, s.partial, s.errpartial,
                      s.a_eval, s.src1, s.err1, s.ka, s.a_acc, s.a_new, s.err0)
                A isa CUDA.CuArray && CUDA.unsafe_free!(A)
            end
            s.stream = nothing
        catch
        end
    end
    return nothing
end

function _frame_id(frame::Symbol)
    frame === :lab && return Int32(1)
    frame === :ip && return Int32(2)
    frame === :chirp && return Int32(3)
    error("unknown frame $frame")
end

function _stage!(shards, stage::Int, a_eval::ComplexF64, tstage::Float64,
                 ω::Float64, sys::System, drive::AbstractDrive, frame::Symbol,
                 active::Bool, cache::IntegratorCache, dt::Float64,
                 coeffs::NTuple{6,Float64}, nused::Int, write_y::Bool)
    fid = _frame_id(frame)
    ts0 = _activation_start(drive)
    ts = isfinite(ts0) ? ts0 : -floatmax(Float64) / 4
    hw = cache.halfwidth
    gperp = sys.gamma_perp
    gpar = sys.gamma_par
    weq = sys.w_eq
    st = Int32(stage)
    nu = Int32(nused)
    wy = write_y ? Int32(1) : Int32(0)
    ul = active ? Int32(1) : Int32(0)
    are, aim = real(a_eval), imag(a_eval)
    c1, c2, c3, c4, c5, c6 = coeffs
    z0 = 0.0

    _for_each_shard(shards) do s
        _launch_kernel(s, _stage_kernel!,
            s.kSp, s.kSz, s.ySp, s.ySz, st, s.partial, s.uSp, s.uSz,
            s.delta, s.g, s.Nj, s.live, s.eligible,
            s.a_eval, s.a_acc, s.ka, are, aim, Int32(0),
            ω, tstage, ts, hw, gperp, gpar, weq,
            dt, c1, c2, c3, c4, c5, c6, nu, wy, fid, ul, s.mloc;
            threads=GPU_THREADS, blocks=s.nblocks)
        _launch_kernel(s, _reduce_complex!, s.src1, s.partial, s.nblocks;
                       threads=GPU_THREADS, blocks=1)
    end
    src = _device_reduce_sources(shards, z0 + 0.0im)
    cache.n_eval += 1
    return src
end

function _device_reduce_sources(shards, z0::ComplexF64)
    n = length(shards)
    if n == 1
        s = shards[1]
        _sync_shard(s)
        copyto!(s.src_h, s.src1)
        return z0 + s.src_h[1]
    end
    _for_each_shard(shards) do s
        _sync_shard(s)
    end
    root = shards[1]
    _maybe_device!(root.dev)
    slots = root.root_src
    for s in shards
        if s.dev == root.dev
            copyto!(slots, s.id, s.src1, 1, 1)
        else
            copyto!(view(slots, s.id:s.id), s.src1)
        end
    end
    return z0 + sum(slots)
end

function _error_est!(shards, a_err::ComplexF64, a_new::ComplexF64,
                     tab::Tsit5Tab{Float64}, dt::Float64,
                     atol::Float64, reltol::Float64, M::Int)
    b1, b2, b3, b4, b5, b6, b7 = tab.bt1, tab.bt2, tab.bt3, tab.bt4, tab.bt5, tab.bt6, tab.bt7
    _for_each_shard(shards) do s
        _launch_kernel(s, _err_kernel!,
            s.errpartial, s.kSp, s.kSz, s.ySp, s.ySz,
            dt, b1, b2, b3, b4, b5, b6, b7, atol, reltol, s.mloc;
            threads=GPU_THREADS, blocks=s.nblocks)
        _launch_kernel(s, _reduce_real!, s.err1, s.errpartial, s.nblocks;
                       threads=GPU_THREADS, blocks=1)
    end
    acc = abs2(a_err) / ((atol + reltol * abs(a_new))^2 + 1e-30)
    acc += _device_reduce_errors(shards)
    return sqrt(acc / (1 + 2 * M))
end

function _device_reduce_errors(shards)
    n = length(shards)
    if n == 1
        s = shards[1]
        _sync_shard(s)
        copyto!(s.err_h, s.err1)
        return s.err_h[1]
    end
    _for_each_shard(shards) do s
        _sync_shard(s)
    end
    root = shards[1]
    _maybe_device!(root.dev)
    slots = root.root_err
    for s in shards
        if s.dev == root.dev
            copyto!(slots, s.id, s.err1, 1, 1)
        else
            copyto!(view(slots, s.id:s.id), s.err1)
        end
    end
    return sum(slots)
end

function _accept!(shards; do_cav::Bool=false)
    dc = do_cav ? Int32(1) : Int32(0)
    _for_each_shard(shards) do s
        s.uSp, s.ySp = s.ySp, s.uSp
        s.uSz, s.ySz = s.ySz, s.uSz
        _launch_kernel(s, _kcol_copy!,
            s.kSp, s.kSz, Int32(1), Int32(7), s.mloc, s.a_acc, s.a_new, s.ka, dc;
            threads=GPU_THREADS, blocks=s.nblocks)
    end
    return nothing
end

function _gather!(Sp_h, Sz_h, shards)
    for s in shards
        _maybe_device!(s.dev)
        _sync_shard(s)
        copyto!(Sp_h, s.j0 + 1, s.uSp, 1, s.mloc)
        copyto!(Sz_h, s.j0 + 1, s.uSz, 1, s.mloc)
    end
    return nothing
end

function _n_live(shards)
    _for_each_shard(shards) do s
        _launch_kernel(s, _count_kernel!, s.errpartial, s.live, s.mloc;
                       threads=GPU_THREADS, blocks=s.nblocks)
        _launch_kernel(s, _reduce_real!, s.err1, s.errpartial, s.nblocks;
                       threads=GPU_THREADS, blocks=1)
    end
    n = 0
    _for_each_shard(shards) do s
        _sync_shard(s)
        copyto!(s.err_h, s.err1)
        n += round(Int, s.err_h[1])
    end
    return n
end

function _devcav_params(sys::System, drive::AbstractDrive, frame::Symbol, active::Bool, cache::IntegratorCache)
    ts0 = _activation_start(drive)
    ts = isfinite(ts0) ? ts0 : -floatmax(Float64) / 4
    return (
        fid = _frame_id(frame),
        ts = ts,
        hw = cache.halfwidth,
        gperp = sys.gamma_perp,
        gpar = sys.gamma_par,
        weq = sys.w_eq,
        ul = active ? Int32(1) : Int32(0),
        halfκt = 0.5 * kappa_t(sys),
        delta0 = sys.delta0,
        ske = sqrt(sys.kappa_e),
    )
end

function _queue_fused_stage!(s::GPUShard, P, stage::Int, ω::Float64, tstage::Float64,
                             dt::Float64, coeffs::NTuple{6,Float64}, nused::Int, write_y::Bool,
                             E::ComplexF64, φ::Float64)
    c1, c2, c3, c4, c5, c6 = coeffs
    @cuda threads=GPU_THREADS blocks=s.nblocks _stage_kernel!(
        s.kSp, s.kSz, s.ySp, s.ySz, Int32(stage), s.partial, s.uSp, s.uSz,
        s.delta, s.g, s.Nj, s.live, s.eligible,
        s.a_eval, s.a_acc, s.ka, 0.0, 0.0, Int32(1),
        ω, tstage, P.ts, P.hw, P.gperp, P.gpar, P.weq,
        dt, c1, c2, c3, c4, c5, c6, Int32(nused), write_y ? Int32(1) : Int32(0),
        P.fid, P.ul, s.mloc)
    @cuda threads=GPU_THREADS blocks=1 _reduce_complex_cavity!(
        s.src1, s.partial, s.nblocks, s.ka, Int32(stage), s.a_eval,
        real(E), imag(E), φ, ω, P.delta0, P.halfκt, P.ske, P.fid)
    return nothing
end


struct SolveResult
    sol
    sys::System
    ens::Ensemble
    drive::AbstractDrive
    frame::Symbol
    initial_condition::Symbol
    cache::IntegratorCache
    t::Vector{Float64}
    a::Vector{ComplexF64}
    inversion::Vector{Float64}
    silencing::Vector{Float64}
    elapsed_s::Float64
    n_eval::Int
    n_live_final::Int
    compute::Symbol
    ngpus::Int
    nshards::Int
end

function _rhs_for(frame::Symbol)
    frame === :lab && return rhs_lab!
    frame === :ip && return rhs_ip!
    frame === :chirp && return rhs_chirp!
    error("frame must be :lab, :ip, or :chirp; got $frame")
end

function _want_gpu(compute::Symbol, M::Int)
    compute === :cpu && return false
    compute === :gpu && return true
    compute === :auto && return gpu_functional() && gpu_compute_worthwhile(M, detect_resources())
    error("compute must be :auto, :cpu, or :gpu; got $compute")
end

function _resolve_saveat(saveat, t0, tfinal)
    saveat === nothing && return collect(range(t0, tfinal; length=201))
    saveat isa Number && return collect(range(t0, tfinal; step=Float64(saveat)))
    return collect(Float64.(saveat))
end

const _IC_ACTIVE_OK = (:ground, :inverted)

function _initial_state(ens, drive, frame, t0, initial_condition, initial_state)
    M = ens.M
    initial_state === nothing && return build_u0_1st_framed(ens, drive, frame, t0, initial_condition)
    if initial_state isa SolveResult
        prev = initial_state
        prev.ens.M == M || error("restart ensemble size mismatch ($(prev.ens.M) vs $M)")
        a, Sp, Sz = lab_state(prev.sol.u[end], prev.ens, prev.drive, prev.t[end], prev.frame)
        return frame_from_lab(a, Sp, Sz, ens, drive, t0, frame)
    elseif initial_state isa AbstractVector
        length(initial_state) == state_length_1st(M) ||
            error("initial_state vector must have length $(state_length_1st(M)) ([a; Sp(1:M); Sz(1:M)], lab frame)")
        a = initial_state[1]
        Sp = collect(ComplexF64, @view initial_state[2:1 + M])
        Sz = collect(ComplexF64, @view initial_state[2 + M:1 + 2M])
        return frame_from_lab(a, Sp, Sz, ens, drive, t0, frame)
    elseif initial_state isa NamedTuple
        a = get(initial_state, :a, 0.0 + 0.0im)
        Sp = get(initial_state, :Sp, 0.0 + 0.0im)
        Sz = get(initial_state, :Sz, -ens.Nj ./ 2)
        return frame_from_lab(a, Sp, Sz, ens, drive, t0, frame)
    end
    error("initial_state must be nothing, a lab-frame Vector, a NamedTuple (; a, Sp, Sz), or a SolveResult")
end

function build_u0_1st_framed(ens, drive, frame, t0, initial_condition)
    u_lab = build_u0_1st(ens, initial_condition)
    frame === :lab && return u_lab
    M = ens.M
    a = u_lab[1]
    Sp = collect(ComplexF64, @view u_lab[2:1 + M])
    Sz = collect(ComplexF64, @view u_lab[2 + M:1 + 2M])
    return frame_from_lab(a, Sp, Sz, ens, drive, t0, frame)
end


function solve_1st(sys::System, drive::AbstractDrive, ens::Ensemble;
                   frame::Symbol=:chirp,
                   active::Bool=true,
                   initial_condition::Symbol=:ground,
                   initial_state=nothing,
                   tspan=(0.0, 100e-6),
                   saveat=nothing,
                   reltol=1e-8,
                   abstol=1e-8,
                   halfwidth=nothing,
                   pad=nothing,
                   n_win=4.0,
                   alg=Tsit5(),
                   compute::Symbol=:auto,
                   gpus=:auto,
                   nshards=nothing,
                   free_gpu::Bool=true,
                   reclaim_gpu::Bool=false,
                   save_states::Bool=true)
    frame in (:lab, :ip, :chirp) || error("frame must be :lab, :ip, or :chirp; got $frame")
    t0f = Float64(tspan[1])

    hw = halfwidth === nothing ? default_halfwidth(sys, drive; n_win=n_win) : Float64(halfwidth)
    pd = pad === nothing ? default_halfwidth(sys, drive; n_win=1.0) : Float64(pad)
    cache = IntegratorCache(ens, drive; halfwidth=hw, pad=pd)

    use_active = active && has_sweep(drive) && initial_state === nothing &&
                 initial_condition in _IC_ACTIVE_OK
    if active && !use_active
        why = !has_sweep(drive) ? "the drive has no frequency sweep" :
              initial_state !== nothing ? "an explicit initial_state was given" :
              "initial_condition=$initial_condition is not one of $_IC_ACTIVE_OK"
        @warn "solve_1st: active set disabled ($why); integrating every bin"
    end
    if !use_active
        fill!(cache.live, true)
        cache.eligible .= true
        cache.n_live = ens.M
    end

    if _want_gpu(compute, ens.M)
        gpu_functional() || error("solve_1st: compute=:gpu but CUDA.functional() is false")
        return solve_1st_gpu(sys, drive, ens, cache;
                             frame=frame, active=use_active,
                             initial_condition=initial_condition,
                             initial_state=initial_state,
                             tspan=tspan, saveat=saveat,
                             reltol=reltol, abstol=abstol,
                             gpus=gpus, nshards=nshards, free_gpu=free_gpu,
                             reclaim_gpu=reclaim_gpu, save_states=save_states)
    end

    u0 = _initial_state(ens, drive, frame, t0f, initial_condition, initial_state)
    p = (sys, ens, drive, cache, use_active)
    rhs! = _rhs_for(frame)
    t_save = _resolve_saveat(saveat, tspan[1], tspan[2])
    tstops = drive_tstops(drive)

    prob = ODEProblem(rhs!, u0, tspan, p)
    tw = time()
    sol = solve(prob, alg; reltol=reltol, abstol=abstol, saveat=t_save, tstops=tstops)
    elapsed = time() - tw
    string(sol.retcode) == "Success" || @warn "solve_1st retcode = $(sol.retcode)"

    Nt = length(sol.t)
    a_path = Vector{ComplexF64}(undef, Nt)
    inv_path = Vector{Float64}(undef, Nt)
    sil_path = Vector{Float64}(undef, Nt)
    for i in 1:Nt
        a, Sp, Sz = lab_state(sol.u[i], ens, drive, sol.t[i], frame)
        a_path[i] = a
        inv_path[i] = weighted_inversion(Sz, ens.g_b, ens.Nj)
        sil_path[i] = weighted_silencing(Sp, ens.g_b, ens.Nj, ens.delta_b)
    end

    return SolveResult(
        sol, sys, ens, drive, frame, initial_condition, cache,
        collect(sol.t), a_path, inv_path, sil_path,
        elapsed, cache.n_eval, cache.n_live,
        :cpu, 0, 1,
    )
end


struct GPUSol
    t::Vector{Float64}
    u::Vector{Vector{ComplexF64}}
    retcode::Symbol
    naccept::Int
    nreject::Int
    nsteps::Int
    ngpus::Int
    nshards::Int
end

@inline function _a_lab(a_stored::ComplexF64, drive::AbstractDrive, t, frame::Symbol)
    frame === :chirp || return a_stored
    return a_stored * cis(drive_phase(drive, t))
end

function _cavity_rhs(sys::System, frame::Symbol, a_eval::ComplexF64, src::ComplexF64,
                     E_lab::ComplexF64, φ::Float64, ω::Float64)
    κt = kappa_t(sys)
    sk = sqrt(sys.kappa_e)
    if frame === :chirp
        return sk * E_lab * cis(-φ) - 1im * (sys.delta0 + ω) * a_eval - 1im * src - 0.5 * κt * a_eval
    else
        return sk * E_lab - 1im * sys.delta0 * a_eval - 1im * src - 0.5 * κt * a_eval
    end
end

function solve_1st_gpu(sys::System, drive::AbstractDrive, ens::Ensemble,
                       cache::IntegratorCache;
                       frame::Symbol=:chirp,
                       active::Bool=true,
                       initial_condition::Symbol=:ground,
                       initial_state=nothing,
                       tspan=(0.0, 100e-6),
                       saveat=nothing,
                       reltol=1e-8,
                       abstol=1e-8,
                       gpus=:auto,
                       nshards=nothing,
                       free_gpu::Bool=true,
                       reclaim_gpu::Bool=false,
                       save_states::Bool=true,
                       maxiters=10_000_000)
    plan = make_gpu_plan(ens.M; gpus=gpus, nshards=nshards)
    plan.functional || error("solve_1st_gpu: no CUDA GPU (CUDA.functional() is false)")

    u0 = _initial_state(ens, drive, frame, Float64(tspan[1]), initial_condition, initial_state)
    a = u0[1]
    shards, _ = _acquire_shards(ens, drive, u0, plan, cache; reuse=!free_gpu && !reclaim_gpu)

    result = try
        _run_gpu_stepper!(shards, sys, drive, ens, cache, plan, a, frame, active,
                          initial_condition, tspan, saveat, Float64(reltol),
                          Float64(abstol), Int(maxiters), save_states)
    finally
        _sync_all(shards)
        _release_shards!(shards; keep=(!free_gpu && !reclaim_gpu), reclaim=reclaim_gpu)
    end
    return result
end

function _run_gpu_stepper!(shards, sys, drive, ens, cache, plan, a0, frame, active,
                           initial_condition, tspan, saveat, reltol, abstol, maxiters,
                           save_states::Bool)
    if length(shards) == 1
        return _run_gpu_stepper_devcav!(shards[1], sys, drive, ens, cache, plan, a0, frame, active,
                                        initial_condition, tspan, saveat, reltol, abstol, maxiters,
                                        save_states)
    end
    _run_gpu_stepper_hostcav!(shards, sys, drive, ens, cache, plan, a0, frame, active,
                              initial_condition, tspan, saveat, reltol, abstol, maxiters,
                              save_states)
end

function _run_gpu_stepper_hostcav!(shards, sys, drive, ens, cache, plan, a0, frame, active,
                                   initial_condition, tspan, saveat, reltol, abstol, maxiters,
                                   save_states::Bool)
    tab = Tsit5Tab(Float64)
    ctrl = PICtrl()
    M = ens.M
    t0, tfinal = Float64(tspan[1]), Float64(tspan[2])
    t_save = _resolve_saveat(saveat, t0, tfinal)
    tstops = sort!(unique!([t0, tfinal, drive_tstops(drive)..., t_save...]))
    filter!(τ -> (τ >= t0 - 1e-18) && (τ <= tfinal + 1e-18), tstops)

    Z6 = ntuple(_ -> 0.0, 6)
    stagecoeffs = (
        (tab.a21, 0.0, 0.0, 0.0, 0.0, 0.0),
        (tab.a31, tab.a32, 0.0, 0.0, 0.0, 0.0),
        (tab.a41, tab.a42, tab.a43, 0.0, 0.0, 0.0),
        (tab.a51, tab.a52, tab.a53, tab.a54, 0.0, 0.0),
        (tab.a61, tab.a62, tab.a63, tab.a64, tab.a65, 0.0),
        (tab.a71, tab.a72, tab.a73, tab.a74, tab.a75, tab.a76),
    )
    cnode = (tab.c2, tab.c3, tab.c4, tab.c5, tab.c6, 1.0)

    ka = zeros(ComplexF64, 7)
    a = ComplexF64(a0)


    ω = drive_omega(drive, t0)
    src = _stage!(shards, 1, a, t0, ω, sys, drive, frame, active, cache, 0.0, Z6, 0, false)
    ka[1] = _cavity_rhs(sys, frame, a, src, drive_E(drive, t0), drive_phase(drive, t0), ω)

    span = abs(tfinal - t0)
    dt = max(min(1e-6 * span, 0.01 / (abs(ka[1]) + 1.0), span / 10), 1e-16)

    Nt = length(t_save)
    a_path = Vector{ComplexF64}(undef, Nt)
    inv_path = Vector{Float64}(undef, Nt)
    sil_path = Vector{Float64}(undef, Nt)
    u_saves = save_states ? Vector{Vector{ComplexF64}}(undef, Nt) : Vector{ComplexF64}[]
    Sp_h = _pin_host(ComplexF64, M)
    Sz_h = _pin_host(ComplexF64, M)
    Sp_work = frame === :ip ? Vector{ComplexF64}(undef, M) : Sp_h
    slices = _frequency_slices(ens.delta_b)
    ust = save_states ? Vector{ComplexF64}(undef, 1 + 2M) : ComplexF64[]

    function snapshot!(idx, tnow, aval)
        _gather!(Sp_h, Sz_h, shards)
        if save_states
            ust[1] = aval
            copyto!(view(ust, 2:1 + M), Sp_h)
            copyto!(view(ust, 2 + M:1 + 2M), Sz_h)
            u_saves[idx] = copy(ust)
        end
        a_path[idx] = _a_lab(ComplexF64(aval), drive, tnow, frame)
        inv_path[idx] = weighted_inversion(Sz_h, ens.g_b, ens.Nj)
        if frame === :ip
            @inbounds for j in 1:M
                Sp_work[j] = Sp_h[j] * cis(ens.delta_b[j] * tnow)
            end
            sil_path[idx] = weighted_silencing(Sp_work, ens.g_b, ens.Nj, ens.delta_b; slices=slices)
        else
            sil_path[idx] = weighted_silencing(Sp_h, ens.g_b, ens.Nj, ens.delta_b; slices=slices)
        end
        return nothing
    end

    isave = 1
    t = t0
    if abs(t_save[1] - t0) <= 1e-18
        snapshot!(1, t0, a)
        isave = 2
    end

    naccept = nreject = nsteps = 0
    t_wall = time()
    while t < tfinal && nsteps < maxiters
        nsteps += 1
        tnext = tfinal
        for τ in tstops
            if τ > t + 1e-18
                tnext = τ
                break
            end
        end
        gap = tnext - t
        if gap <= 1e-18
            t = tnext
            continue
        end
        dt = min(dt, gap)


        @inbounds for st in 2:7
            cf = stagecoeffs[st - 1]
            tstage = st == 7 ? t + dt : t + dt * cnode[st - 1]
            aev = a
            for i in 1:(st - 1)
                aev += dt * cf[i] * ka[i]
            end
            ωs = drive_omega(drive, tstage)
            srcs = _stage!(shards, st, aev, tstage, ωs, sys, drive, frame, active,
                           cache, dt, cf, st - 1, st == 7)
            ka[st] = _cavity_rhs(sys, frame, aev, srcs, drive_E(drive, tstage),
                                 drive_phase(drive, tstage), ωs)
        end

        a_new = a
        @inbounds for i in 1:6
            a_new += dt * stagecoeffs[6][i] * ka[i]
        end
        a_err = dt * (tab.bt1 * ka[1] + tab.bt2 * ka[2] + tab.bt3 * ka[3] + tab.bt4 * ka[4] +
                      tab.bt5 * ka[5] + tab.bt6 * ka[6] + tab.bt7 * ka[7])
        EEst = _error_est!(shards, a_err, a_new, tab, dt, abstol, reltol, M)

        e = max(EEst, eps(Float64))
        q = (e^ctrl.beta1) / ctrl.qold^ctrl.beta2
        q = max(1 / ctrl.qmax, min(1 / ctrl.qmin, q / ctrl.gamma))

        if EEst <= 1.0
            naccept += 1
            ctrl.qold = max(EEst, ctrl.qoldinit)
            t += dt
            a = a_new
            _accept!(shards)
            ka[1] = ka[7]
            dt = max(min(dt / q, tfinal - t, 10 * dt), 1e-18)
            while isave <= Nt && t_save[isave] <= t + 1e-18
                snapshot!(isave, t_save[isave], a)
                isave += 1
            end
        else
            nreject += 1
            dt = max(dt / q, 1e-18)
            ω = drive_omega(drive, t)
            src = _stage!(shards, 1, a, t, ω, sys, drive, frame, active, cache, 0.0, Z6, 0, false)
            ka[1] = _cavity_rhs(sys, frame, a, src, drive_E(drive, t), drive_phase(drive, t), ω)
        end
    end
    _sync_all(shards)
    elapsed = time() - t_wall

    nsteps >= maxiters && @warn "solve_1st_gpu hit maxiters=$maxiters at t=$t"
    while isave <= Nt
        snapshot!(isave, t_save[isave], a)
        isave += 1
    end

    cache.n_live = _n_live(shards)
    retcode = nsteps >= maxiters ? :MaxIters : :Success
    gpu_sol = GPUSol(t_save, u_saves, retcode, naccept, nreject, nsteps,
                     plan.ndev, plan.nshards)

    return SolveResult(
        gpu_sol, sys, ens, drive, frame, initial_condition, cache,
        t_save, a_path, inv_path, sil_path,
        elapsed, cache.n_eval, cache.n_live,
        :gpu, plan.ndev, plan.nshards,
    )
end

function _run_gpu_stepper_devcav!(s, sys, drive, ens, cache, plan, a0, frame, active,
                                  initial_condition, tspan, saveat, reltol, abstol, maxiters,
                                  save_states::Bool)
    CUDA.device!(s.dev)
    tab = Tsit5Tab(Float64)
    ctrl = PICtrl()
    M = ens.M
    t0, tfinal = Float64(tspan[1]), Float64(tspan[2])
    t_save = _resolve_saveat(saveat, t0, tfinal)
    tstops = sort!(unique!([t0, tfinal, drive_tstops(drive)..., t_save...]))
    filter!(τ -> (τ >= t0 - 1e-18) && (τ <= tfinal + 1e-18), tstops)

    Z6 = ntuple(_ -> 0.0, 6)
    stagecoeffs = (
        (tab.a21, 0.0, 0.0, 0.0, 0.0, 0.0),
        (tab.a31, tab.a32, 0.0, 0.0, 0.0, 0.0),
        (tab.a41, tab.a42, tab.a43, 0.0, 0.0, 0.0),
        (tab.a51, tab.a52, tab.a53, tab.a54, 0.0, 0.0),
        (tab.a61, tab.a62, tab.a63, tab.a64, tab.a65, 0.0),
        (tab.a71, tab.a72, tab.a73, tab.a74, tab.a75, tab.a76),
    )
    cnode = (tab.c2, tab.c3, tab.c4, tab.c5, tab.c6, 1.0)
    P = _devcav_params(sys, drive, frame, active, cache)

    s.src_h[1] = ComplexF64(a0)
    copyto!(s.a_acc, s.src_h)
    ω = drive_omega(drive, t0)
    E0 = drive_E(drive, t0)
    φ0 = drive_phase(drive, t0)
    _queue_fused_stage!(s, P, 1, ω, t0, 0.0, Z6, 0, false, E0, φ0)
    cache.n_eval += 1
    copyto!(s.src_h, 1, s.ka, 1, 1)
    span = abs(tfinal - t0)
    dt = max(min(1e-6 * span, 0.01 / (abs(s.src_h[1]) + 1.0), span / 10), 1e-16)

    Nt = length(t_save)
    a_path = Vector{ComplexF64}(undef, Nt)
    inv_path = Vector{Float64}(undef, Nt)
    sil_path = Vector{Float64}(undef, Nt)
    u_saves = save_states ? Vector{Vector{ComplexF64}}(undef, Nt) : Vector{ComplexF64}[]
    Sp_h = _pin_host(ComplexF64, M)
    Sz_h = _pin_host(ComplexF64, M)
    Sp_work = frame === :ip ? Vector{ComplexF64}(undef, M) : Sp_h
    slices = _frequency_slices(ens.delta_b)
    ust = save_states ? Vector{ComplexF64}(undef, 1 + 2M) : ComplexF64[]
    a_h = s.src_h
    err_h = s.err_h
    err0_h = _pin_host(Float64, 1)

    function snapshot!(idx, tnow)
        copyto!(a_h, s.a_acc)
        _gather!(Sp_h, Sz_h, (s,))
        if save_states
            ust[1] = a_h[1]
            copyto!(view(ust, 2:1 + M), Sp_h)
            copyto!(view(ust, 2 + M:1 + 2M), Sz_h)
            u_saves[idx] = copy(ust)
        end
        a_path[idx] = _a_lab(a_h[1], drive, tnow, frame)
        inv_path[idx] = weighted_inversion(Sz_h, ens.g_b, ens.Nj)
        if frame === :ip
            @inbounds for j in 1:M
                Sp_work[j] = Sp_h[j] * cis(ens.delta_b[j] * tnow)
            end
            sil_path[idx] = weighted_silencing(Sp_work, ens.g_b, ens.Nj, ens.delta_b; slices=slices)
        else
            sil_path[idx] = weighted_silencing(Sp_h, ens.g_b, ens.Nj, ens.delta_b; slices=slices)
        end
        return nothing
    end

    isave = 1
    t = t0
    if abs(t_save[1] - t0) <= 1e-18
        snapshot!(1, t0)
        isave = 2
    end

    naccept = nreject = nsteps = 0
    t_wall = time()
    while t < tfinal && nsteps < maxiters
        nsteps += 1
        tnext = tfinal
        for τ in tstops
            if τ > t + 1e-18
                tnext = τ
                break
            end
        end
        gap = tnext - t
        if gap <= 1e-18
            t = tnext
            continue
        end
        dt = min(dt, gap)

        @inbounds for st in 2:7
            cf = stagecoeffs[st - 1]
            tstage = st == 7 ? t + dt : t + dt * cnode[st - 1]
            ωs = drive_omega(drive, tstage)
            Es = drive_E(drive, tstage)
            φs = drive_phase(drive, tstage)
            _queue_fused_stage!(s, P, st, ωs, tstage, dt, cf, st - 1, st == 7, Es, φs)
            cache.n_eval += 1
        end
        @cuda threads=GPU_THREADS blocks=s.nblocks _err_kernel!(
            s.errpartial, s.kSp, s.kSz, s.ySp, s.ySz,
            dt, tab.bt1, tab.bt2, tab.bt3, tab.bt4, tab.bt5, tab.bt6, tab.bt7,
            abstol, reltol, s.mloc)
        @cuda threads=GPU_THREADS blocks=1 _reduce_real_cavity_err!(
            s.err1, s.errpartial, s.nblocks, s.a_new, s.err0, s.a_acc, s.ka, dt,
            tab.a71, tab.a72, tab.a73, tab.a74, tab.a75, tab.a76,
            tab.bt1, tab.bt2, tab.bt3, tab.bt4, tab.bt5, tab.bt6, tab.bt7,
            abstol, reltol)
        copyto!(err0_h, s.err0)
        copyto!(err_h, s.err1)
        EEst = sqrt((err0_h[1] + err_h[1]) / (1 + 2 * M))

        e = max(EEst, eps(Float64))
        q = (e^ctrl.beta1) / ctrl.qold^ctrl.beta2
        q = max(1 / ctrl.qmax, min(1 / ctrl.qmin, q / ctrl.gamma))

        if EEst <= 1.0
            naccept += 1
            ctrl.qold = max(EEst, ctrl.qoldinit)
            t += dt
            _accept!((s,); do_cav=true)
            dt = max(min(dt / q, tfinal - t, 10 * dt), 1e-18)
            while isave <= Nt && t_save[isave] <= t + 1e-18
                snapshot!(isave, t_save[isave])
                isave += 1
            end
        else
            nreject += 1
            dt = max(dt / q, 1e-18)
            ω = drive_omega(drive, t)
            _queue_fused_stage!(s, P, 1, ω, t, 0.0, Z6, 0, false,
                                drive_E(drive, t), drive_phase(drive, t))
            cache.n_eval += 1
        end
    end
    CUDA.synchronize()
    elapsed = time() - t_wall

    nsteps >= maxiters && @warn "solve_1st_gpu hit maxiters=$maxiters at t=$t"
    while isave <= Nt
        snapshot!(isave, t_save[isave])
        isave += 1
    end

    cache.n_live = _n_live((s,))
    retcode = nsteps >= maxiters ? :MaxIters : :Success
    gpu_sol = GPUSol(t_save, u_saves, retcode, naccept, nreject, nsteps,
                     plan.ndev, plan.nshards)

    return SolveResult(
        gpu_sol, sys, ens, drive, frame, initial_condition, cache,
        t_save, a_path, inv_path, sil_path,
        elapsed, cache.n_eval, cache.n_live,
        :gpu, plan.ndev, plan.nshards,
    )
end

end


_is_gpu(::CUDA.AnyCuArray) = true

const _ACCEL_WEAK_SEED = 1.0e-3

_accel_get(nt, k, default) = hasproperty(nt, k) ? getproperty(nt, k) : default

_accel_freq_kind_safe(SYSTEM_CONFIG) =
    SYSTEM_CONFIG.freq_inhomogeneity.kind === :gaussian ? :gaussian : :lorentzian

function _accel_quadrature_plan(SYSTEM_CONFIG)
    fk = SYSTEM_CONFIG.freq_inhomogeneity.kind
    freq_rule =
        fk === :lorentzian ? :tan_gauss_legendre       :
        fk === :gaussian   ? :gauss_hermite            :
        fk === :powerlaw   ? :pearson_vii_tan_gauss_legendre :
        nothing
    g  = SYSTEM_CONFIG.g_inhomogeneity
    gk = g.kind
    g_rule =
        gk === :constant   ? :single_node                :
        gk === :gaussian   ? :gauss_hermite              :
        gk === :powerlaw_g ? :log_gauss_legendre         :
        gk === :lorentzian ? :truncated_cauchy_tan       :
        nothing
    return (; freq_kind = fk, freq_rule, g_kind = gk, g_rule,
              quadrature_ok = (freq_rule !== nothing && g_rule !== nothing))
end

function _accel_system(SYSTEM_CONFIG, d; for_quadrature::Bool)
    if !for_quadrature



        return AccelSolver1stOrder.System(;
            C_ens = d.C_ens, delta0 = d.delta0,
            kappa_e = d.kappa_e, kappa_i = d.kappa_i,
            FWHM = d.FWHM, freq_kind = _accel_freq_kind_safe(SYSTEM_CONFIG),
            g_kind = :constant, g_value = sqrt(max(d.g2_avg, eps(Float64))),
            gamma_perp = 0.0, gamma_par = 0.0, w_eq = -1.0,
        )
    end
    fq = SYSTEM_CONFIG.freq_inhomogeneity
    fk = fq.kind
    fk in (:lorentzian, :gaussian, :powerlaw) || error(
        "accel_solver_1st_order: freq_inhomogeneity.kind=$fk has no quadrature " *
        "rule (use accel_ensemble=:histogram, or :auto which falls back).")
    freq_n = fk === :powerlaw ?
        Int(_accel_get(fq, :n, _accel_get(fq, :freq_n, 2))) : 1
    g = SYSTEM_CONFIG.g_inhomogeneity
    base = (; C_ens = d.C_ens, delta0 = d.delta0, kappa_e = d.kappa_e,
              kappa_i = d.kappa_i, FWHM = d.FWHM, freq_kind = fk, freq_n = freq_n,
              gamma_perp = 0.0, gamma_par = 0.0, w_eq = -1.0)
    if g.kind === :constant
        return AccelSolver1stOrder.System(; base..., g_kind = :constant,
            g_value = Float64(g.g_value))
    elseif g.kind === :gaussian
        return AccelSolver1stOrder.System(; base..., g_kind = :gaussian,
            g_mean = Float64(g.mean), g_std = Float64(g.std),
            g_span = Float64(g.span_sigma))
    elseif g.kind === :powerlaw_g
        return AccelSolver1stOrder.System(; base..., g_kind = :powerlaw,
            g_alpha = Float64(g.alpha), g_min = Float64(g.g_min),
            g_max = Float64(g.g_max))
    elseif g.kind === :lorentzian
        return AccelSolver1stOrder.System(; base..., g_kind = :lorentzian,
            g_mean = Float64(_accel_get(g, :mean, _accel_get(g, :center, 0.0))),
            g_hwhm = Float64(_accel_get(g, :hwhm, _accel_get(g, :g_hwhm, 0.0))),
            g_span = Float64(_accel_get(g, :span, _accel_get(g, :span_hwhm, 20.0))))
    else
        error("accel_solver_1st_order: g_inhomogeneity.kind=$(g.kind) has no " *
              "quadrature rule (use accel_ensemble=:histogram, or :auto which " *
              "falls back).")
    end
end

function _accel_ensemble_from_derived(d)
    M_delta = length(d.delta_b_1d)
    M_g     = length(d.g_b_1d)
    return AccelSolver1stOrder.Ensemble(
        M_delta * M_g, M_delta, M_g, :histogram,
        collect(Float64, d.delta_b_1d), collect(Float64, d.g_b_1d),
        collect(Float64, d.p_delta), collect(Float64, d.p_g),
        collect(Float64, d.delta_b), collect(Float64, d.g_b),
        collect(Float64, d.Nj),
        Float64(d.N_total), Float64(d.g_mean), Float64(d.g_std), Float64(d.g2_avg),
    )
end

function _accel_build_drive(PULSE_CONFIG, E_of_t, tspan; structured::Bool)
    if !structured


        ts = Float64[]
        for cfg in PULSE_CONFIG
            if cfg.kind === :wurst
                tc = Float64(cfg.t_center); dur = Float64(cfg.duration)
                push!(ts, tc - dur / 2, tc + dur / 2)
            elseif cfg.kind === :gaussian
                t0 = Float64(cfg.t0); s = Float64(cfg.sigma)
                push!(ts, t0 - 8s, t0, t0 + 8s)
            end
        end
        lo, hi = Float64(tspan[1]), Float64(tspan[2])
        filter!(t -> lo < t < hi, ts)
        ts = sort!(unique!(ts))
        return AccelSolver1stOrder.FuncDrive(E_of_t; tstops = ts)
    end
    drives = AccelSolver1stOrder.AbstractDrive[]
    for cfg in PULSE_CONFIG
        if cfg.kind === :gaussian
            push!(drives, AccelSolver1stOrder.GaussianDrive(;
                t0 = cfg.t0, sigma = cfg.sigma, amp = cfg.amp,
                omega = _accel_get(cfg, :omega, 0.0),
                phase = _accel_get(cfg, :phase, 0.0)))
        elseif cfg.kind === :wurst
            push!(drives, AccelSolver1stOrder.WurstPulse(;
                t_center = cfg.t_center, duration = cfg.duration, amp = cfg.amp,
                bandwidth = cfg.bandwidth, n = _accel_get(cfg, :n, 20.0),
                omega0 = _accel_get(cfg, :omega0, 0.0),
                chirp_sign = _accel_get(cfg, :chirp_sign, 1.0),
                phase0 = _accel_get(cfg, :phase0, 0.0),
                edge_frac = _accel_get(cfg, :edge_frac, 1e-4)))
        elseif cfg.kind === :custom
            push!(drives, AccelSolver1stOrder.FuncDrive(cfg.f))
        else
            error("Unknown pulse kind: $(cfg.kind)")
        end
    end
    isempty(drives) && error("PULSE_CONFIG produced no drives")
    return length(drives) == 1 ? drives[1] : AccelSolver1stOrder.DriveSum(drives...)
end


function run_sim_1st_order(SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG; clean_gpu = true)
    CONFIG = build_full_config(SIM_SETTING, SYSTEM_CONFIG)
    validate_config(CONFIG)
    validate_pulse_config(PULSE_CONFIG)
    mkpath(dirname(CONFIG.saved_file_name))

    d = prepare_derived(CONFIG)
    t_saved = collect(d.t_save)
    Nt = length(t_saved)
    @assert Nt == d.Nt (
        "Inconsistent saved-time dimensions: length(d.t_save) = $Nt, d.Nt = $(d.Nt)."
    )

    E_of_t = build_E_of_t(PULSE_CONFIG)


    accel_ensemble = _accel_get(SIM_SETTING, :accel_ensemble, :histogram)
    accel_frame    = _accel_get(SIM_SETTING, :accel_frame, :lab)
    accel_active   = _accel_get(SIM_SETTING, :accel_active, false) === true
    accel_compute  = _accel_get(SIM_SETTING, :accel_compute, :default)
    accel_gpus     = _accel_get(SIM_SETTING, :accel_gpus, :auto)
    accel_nshards  = _accel_get(SIM_SETTING, :accel_nshards, nothing)
    q_M_delta      = Int(_accel_get(SIM_SETTING, :accel_M_delta, CONFIG.M_delta))
    q_M_g          = Int(_accel_get(SIM_SETTING, :accel_M_g, CONFIG.M_g))

    accel_ensemble in (:histogram, :quadrature, :auto) || error(
        "accel_solver_1st_order: accel_ensemble=$accel_ensemble " *
        "(expected :histogram, :quadrature, or :auto).")

    compute = accel_compute === :default ?
        (AccelSolver1stOrder.gpu_functional() ? :gpu : :cpu) : accel_compute



    qplan = _accel_quadrature_plan(SYSTEM_CONFIG)
    if accel_ensemble === :auto
        if qplan.quadrature_ok
            use_quad = true
            println("accel_solver_1st_order: accel_ensemble=:auto -> :quadrature  " *
                    "(freq $(qplan.freq_kind)→$(qplan.freq_rule), " *
                    "g $(qplan.g_kind)→$(qplan.g_rule))")
        else
            use_quad = false
            missing_axis = qplan.freq_rule === nothing ?
                "freq_inhomogeneity.kind=$(qplan.freq_kind)" :
                "g_inhomogeneity.kind=$(qplan.g_kind)"
            println("accel_solver_1st_order: accel_ensemble=:auto -> :histogram  " *
                    "(no quadrature rule for $missing_axis)")
        end
    else
        use_quad = accel_ensemble === :quadrature
    end

    structured = use_quad || accel_frame !== :lab || accel_active
    frame  = structured ? accel_frame : :lab
    active = structured ? accel_active : false

    sys = _accel_system(SYSTEM_CONFIG, d; for_quadrature = use_quad)
    ens = use_quad ?
        AccelSolver1stOrder.build_ensemble(sys; method = :quadrature,
            M_delta = q_M_delta, M_g = q_M_g) :
        _accel_ensemble_from_derived(d)

    M       = ens.M
    M_delta = ens.M_delta
    M_g     = ens.M_g

    drive = _accel_build_drive(PULSE_CONFIG, E_of_t, d.timespan; structured = structured)


    ic = get_initial_condition(CONFIG)
    ic_sym = :ground
    ic_vec = nothing
    if ic in (:ground, :inverted, :equator, :weak)
        ic_sym = ic
    elseif ic === :custom
        ic_vec = zeros(ComplexF64, 1 + 2M)
    elseif ic === :weak_inverted
        ic_vec = zeros(ComplexF64, 1 + 2M)
        @views ic_vec[2:1 + M]      .= _ACCEL_WEAK_SEED .* ens.Nj ./ 2
        @views ic_vec[2 + M:1 + 2M] .= ens.Nj ./ 2
    else
        error("Unknown initial_condition = $ic. " *
              "Use :ground, :inverted, :equator, :weak, :weak_inverted, or :custom.")
    end

    ens_desc = use_quad ?
        "quadrature[δ:$(qplan.freq_rule), g:$(qplan.g_rule)]" : "histogram"
    println("accel_solver_1st_order: ensemble=$ens_desc " *
            "M=$M (M_delta=$M_delta, M_g=$M_g) frame=$frame active=$active compute=$compute")

    r = AccelSolver1stOrder.solve_1st(sys, drive, ens;
        frame = frame, active = active,
        initial_condition = ic_sym, initial_state = ic_vec,
        tspan = d.timespan, saveat = t_saved,
        reltol = CONFIG.reltol, abstol = CONFIG.abstol,
        compute = compute, gpus = accel_gpus, nshards = accel_nshards,
        free_gpu = true, reclaim_gpu = clean_gpu, save_states = true)

    rc = r.sol.retcode
    (rc === :Success || occursin("Success", string(rc))) ||
        @warn "accel_solver_1st_order: solver retcode = $rc"

    length(r.t) == Nt || error(
        "accel solver returned $(length(r.t)) save points, expected $Nt.")
    u_saves = r.sol.u
    length(u_saves) == Nt || error(
        "accel solver saved $(length(u_saves)) state vectors, expected $Nt.")

    elapsed_seconds = r.elapsed_s
    println("Time taken: $elapsed_seconds seconds " *
            "(n_eval=$(r.n_eval), compute=$(r.compute), ngpus=$(r.ngpus), nshards=$(r.nshards))")


    delta_b_1d = use_quad ? collect(Float64, ens.delta_1d) : collect(d.delta_b_1d)
    g_b_1d     = use_quad ? collect(Float64, ens.g_1d)     : collect(d.g_b_1d)
    Nj_2d      = use_quad ? reshape(collect(Float64, ens.Nj), M_delta, M_g) : d.Nj_2d
    N_total    = use_quad ? ens.N_total : d.N_total

    idelta_res = argmin(abs.(delta_b_1d))
    delta_res  = delta_b_1d[idelta_res]
    keep_range = idelta_res:M_delta:M
    keep_bins  = collect(keep_range)
    @assert length(keep_bins) == M_g
    g_keep     = collect(g_b_1d)
    delta_keep = fill(delta_res, M_g)

    println("Selected resonant-detuning bin: idelta_res=$idelta_res, " *
            "delta_res/2π=$(delta_res / (2π)) Hz, number of g bins=$M_g")

    peak_config = prepare_peak_detection(SIM_SETTING, t_saved)
    Npeaks = peak_config === nothing ? 0 : length(peak_config.labels)
    println("Automatic peak detection: ", peak_config === nothing ? "disabled" : "enabled")

    needed_idx = Set{Int}()
    if peak_config !== nothing
        for widx in peak_config.window_indices, k in widx
            push!(needed_idx, k)
        end
    end


    a_save  = Vector{ComplexF64}(undef, Nt)
    Σp_save = Vector{ComplexF64}(undef, Nt)
    Σz_save = Vector{ComplexF64}(undef, Nt)
    Sp_keep = Matrix{ComplexF64}(undef, M_g, Nt)
    Sz_keep = Matrix{ComplexF64}(undef, M_g, Nt)
    Sp_at   = Dict{Int,Vector{ComplexF64}}()
    Sz_at   = Dict{Int,Vector{ComplexF64}}()

    for k in 1:Nt
        a_k, Sp_k, Sz_k = AccelSolver1stOrder.lab_state(u_saves[k], ens, drive, t_saved[k], r.frame)
        a_save[k]  = a_k
        Σp_save[k] = sum(Sp_k)
        Σz_save[k] = sum(Sz_k)
        @inbounds for (row, j) in enumerate(keep_range)
            Sp_keep[row, k] = Sp_k[j]
            Sz_keep[row, k] = Sz_k[j]
        end
        if k in needed_idx
            Sp_at[k] = Sp_k
            Sz_at[k] = Sz_k
        end
    end

    E_of_t_arr = [E_of_t(t) for t in t_saved]
    a_out     = E_of_t_arr .- d.sqrt_kappa_e .* a_save
    a_out_x   = real.(a_out)
    a_out_p   = imag.(a_out)
    a_out_abs = abs.(a_out)


    peak_detection_config = nothing
    peak_detection_results = nothing
    if peak_config !== nothing
        peak_detection_config = (
            labels = copy(peak_config.labels),
            times = copy(peak_config.times),
            half_windows = copy(peak_config.half_windows),
        )
        peak_detection_results = NamedTuple[]
        for ip in 1:Npeaks
            widx = peak_config.window_indices[ip]
            local_peak_index  = argmax(view(a_out_abs, widx))
            global_peak_index = widx[local_peak_index]
            expected_time = peak_config.times[ip]
            half_window   = peak_config.half_windows[ip]
            detected_time = t_saved[global_peak_index]
            detected_amplitude = a_out_abs[global_peak_index]

            Sp_at_peak_2d = reshape(copy(Sp_at[global_peak_index]), M_delta, M_g)
            Sz_at_peak_2d = reshape(copy(Sz_at[global_peak_index]), M_delta, M_g)

            push!(peak_detection_results, (
                label = peak_config.labels[ip],
                expected_time = expected_time,
                half_window = half_window,
                window_start = expected_time - half_window,
                window_end = expected_time + half_window,
                window_indices = copy(widx),
                window_times = t_saved[widx],
                window_a_out_abs = a_out_abs[widx],
                local_peak_index = local_peak_index,
                global_peak_index = global_peak_index,
                detected_time = detected_time,
                detected_amplitude = detected_amplitude,
                a_out_at_peak = a_out[global_peak_index],
                a_out_x_at_peak = a_out_x[global_peak_index],
                a_out_p_at_peak = a_out_p[global_peak_index],
                Sp_at_peak_2d = Sp_at_peak_2d,
                Sz_at_peak_2d = Sz_at_peak_2d,
            ))

            println("Detected peak: $(peak_config.labels[ip])")
            println("  expected time = $(expected_time * 1e6) μs")
            println("  detected time = $(detected_time * 1e6) μs")
            println("  time shift    = $((detected_time - expected_time) * 1e6) μs")
            println("  |a_out| at peak = $detected_amplitude")
        end
    end

    data = (
        SIM_SETTING = SIM_SETTING,
        SYSTEM_CONFIG = SYSTEM_CONFIG,
        PULSE_CONFIG = PULSE_CONFIG,

        t_saved = t_saved,

        a_sol = a_save,

        Σp_sol = Σp_save,
        Σz_sol = Σz_save,

        E_of_t_arr = E_of_t_arr,

        M_delta = M_delta,
        M_g = M_g,
        M_total = M,

        delta_b_1d = delta_b_1d,
        g_b_1d = g_b_1d,
        Nj_2d = Nj_2d,

        idelta_res = idelta_res,
        delta_res = delta_res,

        keep_bins = keep_bins,
        g_keep = g_keep,
        delta_keep = delta_keep,

        Sp_keep = Sp_keep,
        Sz_keep = Sz_keep,

        peak_detection_config = peak_detection_config,
        peak_detection_results = peak_detection_results,

        N_total = N_total,
        C_ens = d.C_ens,
        C_eff = d.C_eff,
        p_delta_sum = d.p_delta_sum,
        p_g_sum = d.p_g_sum,
        elapsed_seconds = elapsed_seconds,
    )

    filename = CONFIG.saved_file_name
    save_run_data(filename, data)
    println()
    println("Saving to: ", filename)

    return data
end
