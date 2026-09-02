# ============================================================
# ANALYTICAL VOLKOV-ZON METHOD
#
# Standalone forward solver for an inhomogeneous ensemble of
# two-level systems (spins) coupled to a single cavity mode.
#
# This file is FULLY INDEPENDENT of the rest of
# InhomogeneousSpinCavityDynamics.jl: it uses only the Julia
# standard library (no CUDA, no DifferentialEquations, no
# Distributions, no QuadGK), defines its own module, and can be
# dropped into any project:
#
#     include("volkov_zon.jl")
#     using .VolkovZon
#
# (If you also load `src/pulses.jl` into the same namespace, qualify
# the drive constructors as `VolkovZon.gaussian_drive` /
# `VolkovZon.wurst_drive`: those two names -- and only those two --
# are shared with the main package, and are deliberately identical
# in signature and formula.)
#
# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
#
# Same first-order (mean-field / cumulant) equations as
# `src/rhs_1st_order.jl`, written per spin. With
#
#     lambda = kappa_t/2 + i*delta0,   kappa_t = kappa_e + kappa_i
#
# the cavity amplitude a = <a> and the single-spin coherence
# sigma^-(delta, g) = <S^-> obey
#
#     da/dt     = -lambda a + sqrt(kappa_e) E(t)
#                 - i N <g sigma^->
#     dsigma^-/dt = -i delta sigma^- + 2i g a sigma^z
#     dsigma^z/dt = 2 g Im(a conj(sigma^-))
#
# where <.> averages over the joint detuning/coupling density
# rho(delta) p(g), and N is the total spin number.
#
# ------------------------------------------------------------
# THE METHOD
# ------------------------------------------------------------
#
# The Volkov-Zon construction linearises the inversion,
# sigma^z -> s0/2 (s0 = -1 ground, +1 inverted), which makes the
# spin equation linear and solvable in closed form:
#
#     sigma^-(delta,g,t) = sigma^-_0 e^{-i delta t}
#                          + i s0 g int_0^t e^{-i delta (t-tau)} a(tau) dtau
#
# Substituting back into the cavity equation, the ENTIRE
# inhomogeneous ensemble collapses onto two scalars that are
# available in closed form:
#
#   * the second coupling moment  <g^2>   (the coupling
#     distribution enters the cavity dynamics ONLY through this
#     moment -- its shape is degenerate at linear order), and
#
#   * the detuning memory kernel
#
#         K(t) = int rho(delta) e^{-i delta t} d delta
#
#     i.e. the characteristic function of rho, whose Laplace
#     transform
#
#         Ktilde(s) = int rho(delta)/(s + i delta) d delta
#
#     is obtained by CONTOUR INTEGRATION.
#
# The cavity then satisfies a scalar Volterra equation whose
# Laplace transform is algebraic:
#
#     ahat(s) = [ a(0) + sqrt(kappa_e) Ehat(s) - i N Jhat(s) ] / D(s)
#     D(s)    = s + lambda - s0 Omega^2 Ktilde(s)
#     Omega^2 = N <g^2>
#     Jhat(s) = coherence0 * <g> * Ktilde(s)
#
# NO ODE IS EVER INTEGRATED. Two inversion routes are provided:
#
#   :residue  -- exact.  For a Lorentzian detuning distribution
#                Ktilde is RATIONAL, so D(s) = 0 is a polynomial
#                (quadratic for a Lorentzian) and a(t) is an exact
#                finite sum of exponentials -- the cavity/spin
#                polariton branches -- plus an exponential
#                convolution with the drive.  The same holds for
#                the power-law (Pearson VII) family below.
#
#   :fft      -- Bromwich inversion on the line Re s = c > 0,
#                evaluated with a DFT.  Used for the Gaussian
#                detuning distribution, whose Ktilde is the
#                (entire, but non-rational) Faddeeva function.
#                The t = 0 jump/kink of ahat is subtracted
#                analytically so the inversion stays high order.
#
# ------------------------------------------------------------
# DISTRIBUTIONS
# ------------------------------------------------------------
#
# Detuning rho(delta)  --  Lorentzian, Gaussian, power law:
#
#   DetuningLorentzian(FWHM)      rho ~ 1/((d-c)^2 + w^2)
#   DetuningGaussian(FWHM)        rho ~ exp(-(d-c)^2/2 sigma^2)
#   DetuningPowerLaw(FWHM, n)     rho ~ [1 + ((d-c)/w)^2]^(-n)
#
# `DetuningPowerLaw` is the Pearson-VII / generalised-Lorentzian
# family: an algebraically decaying (power-law) line shape with
# tail exponent rho ~ |delta|^(-2n).  n = 1 reproduces the
# Lorentzian exactly and n -> infinity approaches the Gaussian, so
# the three supported shapes form one continuous family.  Integer
# n keeps Ktilde rational, which is what makes the exact residue
# solution available for the power-law case too.
#
# Coupling p(g)  --  Lorentzian, Gaussian, power law (plus a
# constant):
#
#   CouplingConstant(g)
#   CouplingGaussian(mean, std; span_sigma)      truncated normal
#   CouplingLorentzian(center, FWHM; span_gamma) truncated Cauchy
#   CouplingPowerLaw(alpha, g_min, g_max)        p(g) ~ g^(-alpha)
#
# (Coupling distributions are truncated and renormalised; an
# untruncated Cauchy has no second moment, and <g^2> is exactly
# what the dynamics needs.)
#
# ------------------------------------------------------------
# UNITS
# ------------------------------------------------------------
#
# As in the rest of the repository: times in seconds, all
# frequencies / detunings / couplings / decay rates are ANGULAR
# frequencies in rad/s, phases in radians.
#
# ------------------------------------------------------------
# VALIDITY
# ------------------------------------------------------------
#
# The method is exact for the LINEARISED ensemble, i.e. as long as
# the spins stay close to their initial inversion.  Every solution
# reports `linear_validity`, the largest fractional change of the
# total inversion computed to leading order.  It is an ENSEMBLE
# AVERAGE -- resonant bins deplete much more than the average -- so
# keep it well below 1e-3 for quantitative work.  Measured against
# a direct nonlinear integration, a `linear_validity` of 2.5e-7
# gave ~2e-5 relative error in <a>.
# ============================================================

module VolkovZon

using LinearAlgebra: eigvals

export DetuningLorentzian, DetuningGaussian, DetuningPowerLaw,
       CouplingConstant, CouplingGaussian, CouplingLorentzian, CouplingPowerLaw,
       VZSystem, vz_solve, vz_ensemble_grid,
       detuning_pdf, char_fn, laplace_kernel, is_rational, kernel_polys,
       coupling_pdf, coupling_moment, coupling_support,
       transfer_denominator,
       gaussian_drive, wurst_drive, constant_drive, zero_drive, sum_drives,
       faddeeva


# ============================================================
# 1. SMALL NUMERICS TOOLBOX (stdlib only)
# ============================================================

# ---------------- radix-2 FFT ----------------

function _fft!(x::Vector{ComplexF64}, sgn::Int)
    n = length(x)
    n <= 1 && return x
    (n & (n - 1)) == 0 ||
        error("VolkovZon: FFT length must be a power of two, got $n.")

    # bit-reversal permutation
    j = 0
    @inbounds for i in 1:(n - 1)
        bit = n >> 1
        while j & bit != 0
            j ⊻= bit
            bit >>= 1
        end
        j ⊻= bit
        if i < j
            x[i + 1], x[j + 1] = x[j + 1], x[i + 1]
        end
    end

    len = 2
    while len <= n
        half = len >> 1
        ang = sgn * 2 * pi / len
        @inbounds for i in 1:len:n
            for k in 0:(half - 1)
                w = cis(ang * k)
                u = x[i + k]
                v = x[i + k + half] * w
                x[i + k] = u + v
                x[i + k + half] = u - v
            end
        end
        len <<= 1
    end
    return x
end

_fft(x::Vector{ComplexF64}) = _fft!(copy(x), -1)

function _ifft(x::Vector{ComplexF64})
    y = _fft!(copy(x), +1)
    y ./= length(y)
    return y
end

function _next_pow2(n::Integer)
    p = 1
    while p < n
        p <<= 1
    end
    return p
end

# ---------------- Faddeeva function ----------------
#
# w(z) = exp(-z^2) erfc(-i z), via Weideman's rational
# approximation.  Accurate to ~1e-13 in the upper half plane,
# which is where every evaluation in this module lands (the
# Bromwich contour has Re s > 0).

const _W_N = 32

function _weideman_setup(N::Int)
    M = 2N
    M2 = 2M
    L = sqrt(N / sqrt(2.0))

    f = zeros(Float64, M2)
    for j in 1:(2M - 1)          # f[1] stays 0, mirroring the reference code
        k = -M + j
        th = k * pi / M
        t = L * tan(th / 2)
        f[j + 1] = exp(-t^2) * (L^2 + t^2)
    end

    g = Vector{ComplexF64}(undef, M2)   # fftshift, M2 even
    @inbounds for i in 1:M
        g[i] = f[i + M]
        g[i + M] = f[i]
    end

    A = _fft!(g, -1)
    a = Vector{Float64}(undef, N)
    @inbounds for i in 1:N
        a[i] = real(A[N + 2 - i]) / M2   # flipud(real(fft(...))[2:N+1])/M2
    end
    return L, a
end

const _WEIDEMAN = _weideman_setup(_W_N)
const _WEIDEMAN_L = _WEIDEMAN[1]
const _WEIDEMAN_A = _WEIDEMAN[2]

"""
    faddeeva(z) -> ComplexF64

Scaled complementary error function `w(z) = exp(-z^2) erfc(-i z)`.
Accurate for `imag(z) >= 0`; the lower half plane is reached
through `w(z) = 2 exp(-z^2) - w(-z)` (which overflows for strongly
negative `imag(z)`, a regime this module never enters).
"""
function faddeeva(z::ComplexF64)
    if imag(z) < 0
        return 2 * exp(-z * z) - faddeeva(-z)
    end
    L = _WEIDEMAN_L
    d = L - im * z
    Z = (L + im * z) / d
    p = zero(ComplexF64)
    @inbounds for k in eachindex(_WEIDEMAN_A)
        p = p * Z + _WEIDEMAN_A[k]
    end
    return 2 * p / (d * d) + (1 / sqrt(pi)) / d
end

faddeeva(z::Number) = faddeeva(ComplexF64(z))

function _erfc(x::Float64)
    x < 0 && return 2.0 - _erfc(-x)
    return real(exp(-x * x) * faddeeva(complex(0.0, x)))
end

_erf(x::Float64) = 1.0 - _erfc(x)

# ---------------- polynomials (ascending coefficients) ----------------

function _polyval(c::AbstractVector{ComplexF64}, x::ComplexF64)
    v = zero(ComplexF64)
    @inbounds for k in length(c):-1:1
        v = v * x + c[k]
    end
    return v
end

function _polymul(a::Vector{ComplexF64}, b::Vector{ComplexF64})
    out = zeros(ComplexF64, length(a) + length(b) - 1)
    @inbounds for i in eachindex(a), j in eachindex(b)
        out[i + j - 1] += a[i] * b[j]
    end
    return out
end

function _polysub(a::Vector{ComplexF64}, b::Vector{ComplexF64})
    out = zeros(ComplexF64, max(length(a), length(b)))
    @inbounds for i in eachindex(a)
        out[i] += a[i]
    end
    @inbounds for i in eachindex(b)
        out[i] -= b[i]
    end
    return out
end

function _polyder(c::Vector{ComplexF64})
    length(c) <= 1 && return ComplexF64[0.0 + 0.0im]
    return ComplexF64[(k) * c[k + 1] for k in 1:(length(c) - 1)]
end

"""(s + q)^m as ascending coefficients."""
function _shift_pow(q::ComplexF64, m::Int)
    out = ComplexF64[1.0 + 0.0im]
    for _ in 1:m
        out = _polymul(out, ComplexF64[q, 1.0 + 0.0im])
    end
    return out
end

"""
Roots of an ascending-coefficient polynomial, via a companion matrix.

The variable is rescaled to `s = S x` first, with `S` the geometric mean
root magnitude, so that coefficients spanning many orders of magnitude
(they routinely span 1e28 here, since the natural frequencies are ~1e7
rad/s) do not wreck the companion matrix.
"""
function _polyroots(c::Vector{ComplexF64})
    n = length(c)
    # Only EXACT zeros reduce the degree: a relative test against the largest
    # coefficient would strip a perfectly good monic leading term whenever the
    # low-order coefficients are numerically huge.
    while n > 1 && c[n] == 0
        n -= 1
    end
    n <= 1 && return ComplexF64[]
    deg = n - 1

    S = 1.0
    if c[1] != 0
        S = abs(c[1] / c[n])^(1 / deg)
        (isfinite(S) && S > 0) || (S = 1.0)
    end

    cs = ComplexF64[c[k] * S^(k - 1) for k in 1:n]

    A = zeros(ComplexF64, deg, deg)
    @inbounds for k in 1:(deg - 1)
        A[k + 1, k] = one(ComplexF64)
    end
    @inbounds for k in 1:deg
        A[k, deg] = -cs[k] / cs[n]
    end
    return S .* eigvals(A)
end

# ---------------- Gauss-Legendre nodes on [-1, 1] ----------------

function _gauss_legendre(n::Int)
    n >= 1 || error("VolkovZon: need at least one quadrature node.")
    x = zeros(Float64, n)
    w = zeros(Float64, n)
    m = (n + 1) ÷ 2
    for i in 1:m
        z = cos(pi * (i - 0.25) / (n + 0.5))
        pp = 0.0
        for _ in 1:200
            p1 = 1.0
            p2 = 0.0
            for j in 1:n
                p3 = p2
                p2 = p1
                p1 = ((2j - 1.0) * z * p2 - (j - 1.0) * p3) / j
            end
            pp = n * (z * p1 - p2) / (z * z - 1.0)
            z1 = z
            z = z1 - p1 / pp
            abs(z - z1) <= 1e-15 && break
        end
        x[i] = -z
        x[n + 1 - i] = z
        w[i] = 2.0 / ((1.0 - z * z) * pp * pp)
        w[n + 1 - i] = w[i]
    end
    return x, w
end

"""Gauss-Legendre nodes/weights mapped onto [a, b]."""
function _gl_on(n::Int, a::Float64, b::Float64)
    x, w = _gauss_legendre(n)
    return 0.5 * (b - a) .* x .+ 0.5 * (a + b), 0.5 * (b - a) .* w
end

# ---------------- cumulative integral, 4th order ----------------

function _cumint(f::Vector{Float64}, h::Float64)
    n = length(f)
    out = zeros(Float64, n)
    if n < 4
        for k in 1:(n - 1)
            out[k + 1] = out[k] + 0.5 * h * (f[k] + f[k + 1])
        end
        return out
    end
    @inbounds for k in 1:(n - 1)
        if k == 1
            step = (9f[1] + 19f[2] - 5f[3] + f[4]) / 24
        elseif k == n - 1
            step = (f[n - 3] - 5f[n - 2] + 19f[n - 1] + 9f[n]) / 24
        else
            step = (-f[k - 1] + 13f[k] + 13f[k + 1] - f[k + 2]) / 24
        end
        out[k + 1] = out[k] + h * step
    end
    return out
end

_fact(k::Integer) = prod(1.0:k; init = 1.0)


# ============================================================
# 2. DETUNING DISTRIBUTIONS
# ============================================================

abstract type DetuningDistribution end

"""
    DetuningLorentzian(FWHM; center = 0.0)

Lorentzian (Cauchy) detuning density
`rho(d) = (w/pi) / ((d - center)^2 + w^2)` with half width
`w = FWHM/2`.  Its Laplace kernel is rational, so `vz_solve`
returns an exact closed-form solution.
"""
struct DetuningLorentzian <: DetuningDistribution
    FWHM::Float64
    center::Float64
    w::Float64
end

function DetuningLorentzian(FWHM::Real; center::Real = 0.0)
    FWHM > 0 || error("VolkovZon: DetuningLorentzian FWHM must be positive.")
    return DetuningLorentzian(Float64(FWHM), Float64(center), Float64(FWHM) / 2)
end

"""
    DetuningGaussian(FWHM; center = 0.0)

Gaussian detuning density with `sigma = FWHM / (2 sqrt(2 log 2))`.
Its Laplace kernel is the Faddeeva function -- entire but not
rational -- so `vz_solve` inverts the exact transform numerically
(`method = :fft`).
"""
struct DetuningGaussian <: DetuningDistribution
    FWHM::Float64
    center::Float64
    sigma::Float64
end

function DetuningGaussian(FWHM::Real; center::Real = 0.0)
    FWHM > 0 || error("VolkovZon: DetuningGaussian FWHM must be positive.")
    return DetuningGaussian(Float64(FWHM), Float64(center),
                            Float64(FWHM) / (2 * sqrt(2 * log(2.0))))
end

"""
    DetuningPowerLaw(FWHM, n; center = 0.0)

Power-law (Pearson VII / generalised Lorentzian) detuning density

    rho(d) ~ [1 + ((d - center)/w)^2]^(-n),     rho ~ |d|^(-2n)

with `w = FWHM / (2 sqrt(2^(1/n) - 1))`.  `n` must be a positive
integer; `n = 1` is exactly `DetuningLorentzian`, and larger `n`
interpolates towards the Gaussian while keeping algebraic tails.
Integer `n` keeps the Laplace kernel rational, so `vz_solve`
returns an exact closed-form solution here as well.
"""
struct DetuningPowerLaw <: DetuningDistribution
    FWHM::Float64
    center::Float64
    n::Int
    w::Float64
    norm::Float64
    ck::Vector{Float64}
end

function DetuningPowerLaw(FWHM::Real, n::Integer; center::Real = 0.0)
    FWHM > 0 || error("VolkovZon: DetuningPowerLaw FWHM must be positive.")
    n >= 1 || error("VolkovZon: DetuningPowerLaw needs an integer n >= 1.")
    nn = Int(n)
    w = Float64(FWHM) / (2 * sqrt(2.0^(1 / nn) - 1))
    norm = 4.0^(nn - 1) * _fact(nn - 1)^2 / (pi * w * _fact(2nn - 2))
    ck = [_fact(nn - 1) * _fact(2nn - 2 - k) * 2.0^k /
          (_fact(2nn - 2) * _fact(k) * _fact(nn - 1 - k)) for k in 0:(nn - 1)]
    return DetuningPowerLaw(Float64(FWHM), Float64(center), nn, w, norm, ck)
end

"""
    detuning_pdf(d, delta) -> Float64

Normalised detuning probability density.
"""
detuning_pdf(d::DetuningLorentzian, x::Real) =
    (d.w / pi) / ((x - d.center)^2 + d.w^2)

detuning_pdf(d::DetuningGaussian, x::Real) =
    exp(-((x - d.center) / d.sigma)^2 / 2) / (sqrt(2pi) * d.sigma)

detuning_pdf(d::DetuningPowerLaw, x::Real) =
    d.norm * (1 + ((x - d.center) / d.w)^2)^(-d.n)

"""
    char_fn(d, t) -> ComplexF64

Memory kernel `K(t) = int rho(delta) exp(-i delta t) d delta`,
i.e. the characteristic function of the detuning density, in
closed form.
"""
char_fn(d::DetuningLorentzian, t::Real) =
    exp(-im * d.center * t - d.w * abs(t))

char_fn(d::DetuningGaussian, t::Real) =
    exp(-im * d.center * t - (d.sigma * t)^2 / 2)

function char_fn(d::DetuningPowerLaw, t::Real)
    x = d.w * abs(t)
    poly = 0.0
    @inbounds for k in (d.n - 1):-1:0
        poly = poly * x + d.ck[k + 1]
    end
    return exp(-im * d.center * t - x) * poly
end

"""
    laplace_kernel(d, s) -> ComplexF64

`Ktilde(s) = int rho(delta) / (s + i delta) d delta`, the Laplace
transform of [`char_fn`](@ref), in closed form (by contour
integration for the Lorentzian and power-law families, and through
the Faddeeva function for the Gaussian).  Valid as the analytic
continuation for any `s` off the imaginary axis.
"""
laplace_kernel(d::DetuningLorentzian, s::ComplexF64) =
    1.0 / (s + im * d.center + d.w)

function laplace_kernel(d::DetuningGaussian, s::ComplexF64)
    u = s + im * d.center
    return sqrt(pi / 2) / d.sigma * faddeeva(im * u / (d.sigma * sqrt(2.0)))
end

function laplace_kernel(d::DetuningPowerLaw, s::ComplexF64)
    u = s + im * d.center
    acc = zero(ComplexF64)
    @inbounds for k in 0:(d.n - 1)
        acc += d.ck[k + 1] * d.w^k * _fact(k) / (u + d.w)^(k + 1)
    end
    return acc
end

laplace_kernel(d::DetuningDistribution, s::Number) =
    laplace_kernel(d, ComplexF64(s))

"""
    is_rational(d) -> Bool

Whether `Ktilde` is a rational function of `s`, which is what
enables the exact `:residue` solution.
"""
is_rational(::DetuningLorentzian) = true
is_rational(::DetuningPowerLaw) = true
is_rational(::DetuningGaussian) = false

"""
    kernel_polys(d) -> (P, Q)

Ascending-order coefficient vectors with `Ktilde(s) = P(s)/Q(s)`,
`deg Q = n`, `deg P = n - 1`.
"""
function kernel_polys(d::DetuningLorentzian)
    q = im * d.center + d.w
    return ComplexF64[1.0 + 0.0im], ComplexF64[q, 1.0 + 0.0im]
end

function kernel_polys(d::DetuningPowerLaw)
    q = ComplexF64(im * d.center + d.w)
    Q = _shift_pow(q, d.n)
    P = zeros(ComplexF64, d.n)
    for k in 0:(d.n - 1)
        term = _shift_pow(q, d.n - 1 - k)
        coef = d.ck[k + 1] * d.w^k * _fact(k)
        @inbounds for i in eachindex(term)
            P[i] += coef * term[i]
        end
    end
    return P, Q
end

kernel_polys(d::DetuningGaussian) =
    error("VolkovZon: the Gaussian detuning kernel is not rational; " *
          "use method = :fft.")

_detuning_info(d::DetuningLorentzian) =
    (kind = :lorentzian, FWHM = d.FWHM, center = d.center, half_width = d.w)
_detuning_info(d::DetuningGaussian) =
    (kind = :gaussian, FWHM = d.FWHM, center = d.center, sigma = d.sigma)
_detuning_info(d::DetuningPowerLaw) =
    (kind = :powerlaw, FWHM = d.FWHM, center = d.center, n = d.n,
     half_width_scale = d.w, tail_exponent = 2 * d.n)


# ============================================================
# 3. COUPLING DISTRIBUTIONS
#
# At linear order the coupling density enters the CAVITY dynamics
# only through <g^2>; <g> additionally sets the collective
# coherence.  Both are available in closed form below.
# ============================================================

abstract type CouplingDistribution end

"""
    CouplingConstant(g)

Every spin has the same coupling `g`.
"""
struct CouplingConstant <: CouplingDistribution
    g::Float64
    function CouplingConstant(g::Real)
        g > 0 || error("VolkovZon: CouplingConstant g must be positive.")
        return new(Float64(g))
    end
end

"""
    CouplingGaussian(mean, std; span_sigma = 6.0)

Normal coupling density truncated to `mean ± span_sigma*std` and
renormalised.
"""
struct CouplingGaussian <: CouplingDistribution
    mean::Float64
    std::Float64
    span_sigma::Float64
    function CouplingGaussian(mean::Real, std::Real; span_sigma::Real = 6.0)
        std > 0 || error("VolkovZon: CouplingGaussian std must be positive.")
        span_sigma > 0 ||
            error("VolkovZon: CouplingGaussian span_sigma must be positive.")
        return new(Float64(mean), Float64(std), Float64(span_sigma))
    end
end

"""
    CouplingLorentzian(center, FWHM; span_gamma = 20.0)

Cauchy coupling density truncated to `center ± span_gamma*(FWHM/2)`
and renormalised.  Truncation is mandatory: an untruncated Cauchy
density has no second moment, and `<g^2>` is exactly what drives
the cavity.
"""
struct CouplingLorentzian <: CouplingDistribution
    center::Float64
    FWHM::Float64
    span_gamma::Float64
    function CouplingLorentzian(center::Real, FWHM::Real; span_gamma::Real = 20.0)
        FWHM > 0 || error("VolkovZon: CouplingLorentzian FWHM must be positive.")
        span_gamma > 0 ||
            error("VolkovZon: CouplingLorentzian span_gamma must be positive.")
        return new(Float64(center), Float64(FWHM), Float64(span_gamma))
    end
end

"""
    CouplingPowerLaw(alpha, g_min, g_max)

Power-law coupling density `p(g) ~ g^(-alpha)` on
`[g_min, g_max]`, matching the `:powerlaw_g` option of the main
package.  All moments are closed form.
"""
struct CouplingPowerLaw <: CouplingDistribution
    alpha::Float64
    g_min::Float64
    g_max::Float64
    function CouplingPowerLaw(alpha::Real, g_min::Real, g_max::Real)
        g_min > 0 || error("VolkovZon: CouplingPowerLaw g_min must be positive.")
        g_max > g_min ||
            error("VolkovZon: CouplingPowerLaw needs g_max > g_min.")
        return new(Float64(alpha), Float64(g_min), Float64(g_max))
    end
end

"""Integral of `g^q` from `lo` to `hi`."""
function _pow_integral(lo::Float64, hi::Float64, q::Float64)
    if abs(q + 1) < 1e-14
        return log(hi / lo)
    end
    return (hi^(q + 1) - lo^(q + 1)) / (q + 1)
end

"""
    coupling_moment(c, m) -> Float64

Closed-form `<g^m>` of the (truncated, renormalised) coupling
density.
"""
coupling_moment(c::CouplingConstant, m::Integer) = c.g^m

function coupling_moment(c::CouplingGaussian, m::Integer)
    m == 0 && return 1.0
    z = c.span_sigma
    Z = _erf(z / sqrt(2.0))
    phi = exp(-z^2 / 2) / sqrt(2pi)
    var = c.std^2 * (1 - 2 * z * phi / Z)
    m == 1 && return c.mean
    m == 2 && return c.mean^2 + var
    error("VolkovZon: CouplingGaussian moments are implemented for m <= 2.")
end

function coupling_moment(c::CouplingLorentzian, m::Integer)
    m == 0 && return 1.0
    m == 1 && return c.center
    if m == 2
        z = c.span_gamma
        gam = c.FWHM / 2
        # conditional second central moment of Cauchy truncated to |u| <= z
        return c.center^2 + gam^2 * (z - atan(z)) / atan(z)
    end
    error("VolkovZon: CouplingLorentzian moments are implemented for m <= 2.")
end

function coupling_moment(c::CouplingPowerLaw, m::Integer)
    num = _pow_integral(c.g_min, c.g_max, m - c.alpha)
    den = _pow_integral(c.g_min, c.g_max, -c.alpha)
    return num / den
end

"""
    coupling_pdf(c, g) -> Float64

Normalised coupling density (zero outside the truncation range;
a Dirac mass is reported as `Inf` at `g == c.g` for
`CouplingConstant`).
"""
coupling_pdf(c::CouplingConstant, g::Real) = g == c.g ? Inf : 0.0

function coupling_pdf(c::CouplingGaussian, g::Real)
    lo, hi = c.mean - c.span_sigma * c.std, c.mean + c.span_sigma * c.std
    (g < lo || g > hi) && return 0.0
    Z = _erf(c.span_sigma / sqrt(2.0))
    return exp(-((g - c.mean) / c.std)^2 / 2) / (sqrt(2pi) * c.std * Z)
end

function coupling_pdf(c::CouplingLorentzian, g::Real)
    gam = c.FWHM / 2
    lo, hi = c.center - c.span_gamma * gam, c.center + c.span_gamma * gam
    (g < lo || g > hi) && return 0.0
    return (gam / (2 * atan(c.span_gamma))) / ((g - c.center)^2 + gam^2)
end

function coupling_pdf(c::CouplingPowerLaw, g::Real)
    (g < c.g_min || g > c.g_max) && return 0.0
    return g^(-c.alpha) / _pow_integral(c.g_min, c.g_max, -c.alpha)
end

"""Truncation range actually represented by the distribution."""
coupling_support(c::CouplingConstant) = (c.g, c.g)
coupling_support(c::CouplingGaussian) =
    (c.mean - c.span_sigma * c.std, c.mean + c.span_sigma * c.std)
function coupling_support(c::CouplingLorentzian)
    gam = c.FWHM / 2
    return (c.center - c.span_gamma * gam, c.center + c.span_gamma * gam)
end
coupling_support(c::CouplingPowerLaw) = (c.g_min, c.g_max)

_coupling_info(c::CouplingConstant) = (kind = :constant, g = c.g)
_coupling_info(c::CouplingGaussian) =
    (kind = :gaussian, mean = c.mean, std = c.std, span_sigma = c.span_sigma)
_coupling_info(c::CouplingLorentzian) =
    (kind = :lorentzian, center = c.center, FWHM = c.FWHM,
     span_gamma = c.span_gamma)
_coupling_info(c::CouplingPowerLaw) =
    (kind = :powerlaw, alpha = c.alpha, g_min = c.g_min, g_max = c.g_max)


# ============================================================
# 4. DRIVES
#
# Same conventions as `src/pulses.jl` of the main package, so a
# PULSE_CONFIG entry translates one-to-one.
# ============================================================

"""
    gaussian_drive(; t0, sigma, amp, omega = 0.0, phase = 0.0)

`E(t) = amp * exp(-(t-t0)^2 / 2 sigma^2) * exp(i (omega (t-t0) + phase))`.
"""
function gaussian_drive(; t0, sigma, amp, omega = 0.0, phase = 0.0)
    t0f, sf = Float64(t0), Float64(sigma)
    ampc = ComplexF64(amp)
    omf, phf = Float64(omega), Float64(phase)
    return function (t)
        tau = t - t0f
        return ampc * exp(-(tau^2) / (2 * sf^2)) * exp(im * (omf * tau + phf))
    end
end

"""
    wurst_drive(; t_center, duration, amp, bandwidth, n = 20.0, omega0 = 0.0,
                  chirp_sign = +1.0, phase0 = 0.0, edge_frac = 1e-4)

WURST chirped pulse, identical to `wurst_drive` in `src/pulses.jl`.
"""
function wurst_drive(; t_center, duration, amp, bandwidth, n = 20.0,
                     omega0 = 0.0, chirp_sign = +1.0, phase0 = 0.0,
                     edge_frac = 1e-4)
    tc, dur = Float64(t_center), Float64(duration)
    ampc = ComplexF64(amp)
    bw, nf = Float64(bandwidth), Float64(n)
    om0, cs, ph0 = Float64(omega0), Float64(chirp_sign), Float64(phase0)
    t_start = tc - dur / 2
    edge = max(dur * Float64(edge_frac), eps(Float64))
    return function (t)
        tau = t - t_start
        gate = 0.5 * (tanh((t - t_start) / edge) -
                      tanh((t - (t_start + dur)) / edge))
        envelope = ampc * (1 - abs(sin(pi * (tau - dur / 2) / dur))^nf)
        phase = ph0 + (om0 - cs * bw / 2) * tau + 0.5 * cs * (bw / dur) * tau^2
        return gate * envelope * exp(im * phase)
    end
end

"""    constant_drive(amp)"""
constant_drive(amp) = (t -> ComplexF64(amp))

"""    zero_drive()"""
zero_drive() = (t -> 0.0 + 0.0im)

"""    sum_drives(pulses...) -> function

Superposition of several drives, matching `build_E_of_t`.
"""
function sum_drives(pulses...)
    ps = tuple(pulses...)
    return function (t)
        acc = 0.0 + 0.0im
        for p in ps
            acc += p(t)
        end
        return acc
    end
end


# ============================================================
# 5. SYSTEM
# ============================================================

"""
    VZSystem(; kappa_e, kappa_i = 0.0, delta0 = 0.0,
               detuning, coupling,
               C_ens = nothing, N = nothing,
               initial_condition = :ground,
               a0 = 0.0 + 0.0im, coherence0 = 0.0 + 0.0im,
               drive = zero_drive())

Cavity + inhomogeneous spin ensemble.

* `kappa_e`, `kappa_i` -- external / internal cavity decay rates
  (rad/s); `kappa_t = kappa_e + kappa_i`.
* `delta0` -- cavity detuning from the rotating-frame reference.
* `detuning`, `coupling` -- the two inhomogeneity distributions.
* Exactly one of `C_ens` (ensemble cooperativity) or `N` (total
  spin number) must be given.  They are related, as in the main
  package, by `C_ens = 2 pi N <g^2> rho(0) / kappa_t`, with the
  detuning density evaluated at the rotating-frame reference
  `delta = 0`.
* `initial_condition` -- `:ground` (all spins down, `s0 = -1`) or
  `:inverted` (all spins up, `s0 = +1`).
* `a0` -- initial cavity amplitude.
* `coherence0` -- initial single-spin coherence `<S^->`, uniform
  across the ensemble (use it for free-induction-decay / echo
  style problems).  Must be small compared with `1/2`.
* `drive` -- callable `E(t)` returning a complex amplitude.
"""
struct VZSystem{D <: DetuningDistribution, C <: CouplingDistribution, F}
    kappa_e::Float64
    kappa_i::Float64
    kappa_t::Float64
    delta0::Float64
    detuning::D
    coupling::C
    s0::Float64
    N::Float64
    C_ens::Float64
    g1::Float64
    g2::Float64
    Omega2::Float64
    lambda::ComplexF64
    a0::ComplexF64
    coherence0::ComplexF64
    drive::F
end

function VZSystem(; kappa_e, kappa_i = 0.0, delta0 = 0.0,
                  detuning::DetuningDistribution,
                  coupling::CouplingDistribution,
                  C_ens = nothing, N = nothing,
                  initial_condition::Symbol = :ground,
                  a0 = 0.0 + 0.0im, coherence0 = 0.0 + 0.0im,
                  drive = zero_drive())

    kappa_e > 0 || error("VolkovZon: kappa_e must be positive.")
    kappa_i >= 0 || error("VolkovZon: kappa_i must be non-negative.")
    kappa_t = Float64(kappa_e) + Float64(kappa_i)

    s0 = if initial_condition === :ground
        -1.0
    elseif initial_condition === :inverted
        +1.0
    else
        error("VolkovZon: initial_condition must be :ground or :inverted, " *
              "got $(initial_condition).")
    end

    g1 = coupling_moment(coupling, 1)
    g2 = coupling_moment(coupling, 2)
    g2 > 0 || error("VolkovZon: <g^2> must be positive.")

    rho0 = detuning_pdf(detuning, 0.0)

    Ntot = if N === nothing && C_ens === nothing
        error("VolkovZon: provide exactly one of C_ens or N.")
    elseif N !== nothing && C_ens !== nothing
        error("VolkovZon: provide exactly one of C_ens or N, not both.")
    elseif N !== nothing
        Float64(N)
    else
        rho0 > 0 ||
            error("VolkovZon: the detuning density underflows to zero at " *
                  "delta = 0, so C_ens cannot be converted to a spin " *
                  "number; pass N explicitly.")
        Float64(C_ens) * kappa_t / (2pi * g2 * rho0)
    end
    Ntot >= 0 || error("VolkovZon: the spin number must be non-negative.")

    Cens = 2pi * Ntot * g2 * rho0 / kappa_t

    return VZSystem(Float64(kappa_e), Float64(kappa_i), kappa_t,
                    Float64(delta0), detuning, coupling, s0, Ntot, Cens,
                    g1, g2, Ntot * g2,
                    ComplexF64(kappa_t / 2 + im * Float64(delta0)),
                    ComplexF64(a0), ComplexF64(coherence0), drive)
end

"""
    transfer_denominator(sys, s) -> ComplexF64

`D(s) = s + lambda - s0 Omega^2 Ktilde(s)`.  Its zeros are the
cavity/spin polariton poles.
"""
transfer_denominator(sys::VZSystem, s::ComplexF64) =
    s + sys.lambda - sys.s0 * sys.Omega2 * laplace_kernel(sys.detuning, s)


# ============================================================
# 6. EXACT RESIDUE SOLVER (rational kernels)
# ============================================================

struct _ResidueData
    P::Vector{ComplexF64}
    Q::Vector{ComplexF64}
    chi::Vector{ComplexF64}
    roots::Vector{ComplexF64}
    dchi::Vector{ComplexF64}
end

function _residue_data(sys::VZSystem)
    P, Q = kernel_polys(sys.detuning)
    chi = _polysub(_polymul(ComplexF64[sys.lambda, 1.0 + 0.0im], Q),
                   ComplexF64.(sys.s0 * sys.Omega2 .* P))
    roots = _polyroots(chi)
    isempty(roots) && error("VolkovZon: the characteristic polynomial is " *
                            "degenerate; check the system parameters.")
    # near-degenerate roots make the simple-pole partial fraction ill-posed
    scale = maximum(abs, roots)
    for i in eachindex(roots), j in (i + 1):length(roots)
        if abs(roots[i] - roots[j]) < 1e-8 * max(scale, 1.0)
            @warn "VolkovZon: nearly degenerate poles detected; the residue " *
                  "expansion is ill-conditioned here. Consider method = :fft."
            break
        end
    end
    return _ResidueData(P, Q, chi, roots, _polyder(chi))
end

"""
Panel data for the exponential convolutions: Gauss-Legendre nodes
inside one uniform step, plus the drive sampled there.
"""
struct _Panels
    dt::Float64
    u::Vector{Float64}
    w::Vector{Float64}
    E::Matrix{ComplexF64}   # (nq, Nt-1)
end

function _build_panels(t::AbstractVector{Float64}, drive, nq::Int)
    dt = t[2] - t[1]
    x, wq = _gauss_legendre(nq)
    u = 0.5 * dt .* (x .+ 1.0)
    w = 0.5 * dt .* wq
    E = Matrix{ComplexF64}(undef, nq, length(t) - 1)
    @inbounds for n in 1:(length(t) - 1)
        for j in 1:nq
            E[j, n] = ComplexF64(drive(t[n] + u[j]))
        end
    end
    return _Panels(dt, u, w, E)
end

"""
`I_z(t_n) = int_0^{t_n} exp(z (t_n - tau)) E(tau) d tau`, advanced by
an exact exponential recurrence with Gauss-Legendre panels (machine
accurate for smooth drives).
"""
function _exp_convolution(z::ComplexF64, pan::_Panels, Nt::Int)
    out = zeros(ComplexF64, Nt)
    estep = exp(z * pan.dt)
    kern = ComplexF64[exp(z * (pan.dt - pan.u[j])) * pan.w[j]
                      for j in eachindex(pan.u)]
    acc = zero(ComplexF64)
    @inbounds for n in 1:(Nt - 1)
        panel = zero(ComplexF64)
        for j in eachindex(kern)
            panel += kern[j] * pan.E[j, n]
        end
        acc = estep * acc + panel
        out[n + 1] = acc
    end
    return out
end

function _solve_residue(sys::VZSystem, t::Vector{Float64}, nq::Int,
                        deltas::Union{Nothing, Vector{Float64}})
    rd = _residue_data(sys)
    Nt = length(t)
    pan = _build_panels(t, sys.drive, nq)
    ske = sqrt(sys.kappa_e)

    # nu(s) = a0 Q(s) - i N coherence0 <g> P(s)
    nu = _polysub(ComplexF64.(sys.a0 .* rd.Q),
                  ComplexF64.((im * sys.N * sys.coherence0 * sys.g1) .* rd.P))

    a = zeros(ComplexF64, Nt)
    adot = zeros(ComplexF64, Nt)
    Ik = Vector{Vector{ComplexF64}}(undef, length(rd.roots))
    rk = Vector{ComplexF64}(undef, length(rd.roots))
    nuk = Vector{ComplexF64}(undef, length(rd.roots))

    for (k, sk) in enumerate(rd.roots)
        d = _polyval(rd.dchi, sk)
        rk[k] = _polyval(rd.Q, sk) / d
        nuk[k] = _polyval(nu, sk) / d
        Ik[k] = _exp_convolution(sk, pan, Nt)
        @inbounds for n in 1:Nt
            term = nuk[k] * exp(sk * t[n]) + ske * rk[k] * Ik[k][n]
            a[n] += term
            adot[n] += sk * term
        end
    end

    Ev = ComplexF64[ComplexF64(sys.drive(tn)) for tn in t]
    # sum_k r_k == 1 exactly, so the drive enters a' undivided
    adot .+= ske .* Ev

    # N <g sigma^-> = i (a' + lambda a - sqrt(kappa_e) E)
    Ngsm = im .* (adot .+ sys.lambda .* a .- ske .* Ev)

    bins = nothing
    if deltas !== nothing
        bins = Matrix{ComplexF64}(undef, length(deltas), Nt)   # R(delta, t)
        for (i, dl) in enumerate(deltas)
            z = ComplexF64(-im * dl)
            chiz = _polyval(rd.chi, z)
            Iz = _exp_convolution(z, pan, Nt)
            cnu = _polyval(nu, z) / chiz
            cQ = _polyval(rd.Q, z) / chiz
            @inbounds for n in 1:Nt
                bins[i, n] = cnu * exp(z * t[n]) + ske * cQ * Iz[n]
            end
            for (k, sk) in enumerate(rd.roots)
                den = _polyval(rd.dchi, sk) * (sk + im * dl)
                cn = _polyval(nu, sk) / den
                cq = _polyval(rd.Q, sk) / den
                @inbounds for n in 1:Nt
                    bins[i, n] += cn * exp(sk * t[n]) + ske * cq * Ik[k][n]
                end
            end
        end
    end

    return (a = a, Ngsm = Ngsm, E = Ev, poles = rd.roots, residues = rk,
            R = bins)
end


# ============================================================
# 7. BROMWICH / FFT SOLVER (any kernel)
#
# a(t) = (1/2 pi i) int_{c-i inf}^{c+i inf} ahat(s) e^{s t} ds,
# discretised on Re s = c > 0 with a DFT.  The t = 0 jump and kink
# of ahat -- which are known exactly -- are subtracted before the
# inversion and added back in the time domain, which lifts the
# accuracy from O(h) to O(h^3) whenever a(0) or the initial
# coherence is nonzero (and leaves the smooth case spectral).
# ============================================================

function _solve_fft(sys::VZSystem, t::Vector{Float64}, refine::Int, pad::Int,
                    damp::Float64, deltas::Union{Nothing, Vector{Float64}})
    Nt = length(t)
    h = (t[2] - t[1]) / refine
    Nfft = _next_pow2(pad * refine * (Nt - 1))
    Twin = Nfft * h
    c = damp / Twin
    lam = sys.lambda
    ske = sqrt(sys.kappa_e)

    tt = Float64[(n - 1) * h for n in 1:Nfft]
    s = Vector{ComplexF64}(undef, Nfft)
    @inbounds for m in 0:(Nfft - 1)
        kk = m < Nfft ÷ 2 ? m : m - Nfft
        s[m + 1] = c + im * (2pi * kk / Twin)
    end

    f = ComplexF64[ske * ComplexF64(sys.drive(tn)) * exp(-c * tn) for tn in tt]
    Fh = _fft(f)
    Fh .*= h
    # The DFT is a left-endpoint sum; add the trapezoid endpoint correction so
    # a drive that has not died away by the end of the window is still handled
    # accurately (exp(-i y_m Twin) == 1 on every DFT bin, hence one offset).
    fN = ske * ComplexF64(sys.drive(Twin)) * exp(-c * Twin)
    Fh .+= 0.5 * h * (fN - f[1])

    Kt = ComplexF64[laplace_kernel(sys.detuning, sv) for sv in s]
    src = im * sys.N * sys.coherence0 * sys.g1
    ah = Vector{ComplexF64}(undef, Nfft)
    @inbounds for m in 1:Nfft
        ah[m] = (sys.a0 + Fh[m] - src * Kt[m]) /
                (s[m] + lam - sys.s0 * sys.Omega2 * Kt[m])
    end

    # exact small-t behaviour: a(0) = a0, a'(0) = -lambda a0 + b2
    b2 = ske * ComplexF64(sys.drive(0.0)) - src
    dec = ComplexF64[exp(-lam * tn) for tn in tt]
    ramp = abs(lam) < 1e-300 ? tt .+ 0.0im : (1.0 .- dec) ./ lam

    function invert(hat::Vector{ComplexF64}, sub_hat::Vector{ComplexF64},
                    sub_t::Vector{ComplexF64})
        y = _ifft(hat .- sub_hat)
        out = Vector{ComplexF64}(undef, Nfft)
        @inbounds for n in 1:Nfft
            out[n] = exp(c * tt[n]) * y[n] / h + sub_t[n]
        end
        return out
    end

    sub_a_hat = Vector{ComplexF64}(undef, Nfft)
    sub_p_hat = Vector{ComplexF64}(undef, Nfft)
    @inbounds for m in 1:Nfft
        sub_a_hat[m] = sys.a0 / (s[m] + lam) + b2 / (s[m] * (s[m] + lam))
        sub_p_hat[m] = sys.a0 / (s[m] * (s[m] + lam))
    end

    a_full = invert(ah, sub_a_hat, sys.a0 .* dec .+ b2 .* ramp)
    psi_full = invert(Kt .* ah, sub_p_hat, sys.a0 .* ramp)

    pick(v) = ComplexF64[v[(n - 1) * refine + 1] for n in 1:Nt]
    a = pick(a_full)
    psi = pick(psi_full)

    Kt_time = ComplexF64[char_fn(sys.detuning, tn) for tn in t]
    Ngsm = sys.N * sys.coherence0 * sys.g1 .* Kt_time .+
           (im * sys.s0 * sys.N * sys.g2) .* psi

    Ev = ComplexF64[ComplexF64(sys.drive(tn)) for tn in t]

    bins = nothing
    if deltas !== nothing
        bins = Matrix{ComplexF64}(undef, length(deltas), Nt)
        tmp = Vector{ComplexF64}(undef, Nfft)
        for (i, dl) in enumerate(deltas)
            @inbounds for m in 1:Nfft
                tmp[m] = ah[m] / (s[m] + im * dl)
            end
            Rfull = invert(tmp, sub_p_hat, sys.a0 .* ramp)
            @inbounds for n in 1:Nt
                bins[i, n] = Rfull[(n - 1) * refine + 1]
            end
        end
    end

    # spectral-resolution diagnostic: how much of ahat sits at the
    # grid's band edge relative to its peak
    edge = 0.0
    peak = 0.0
    @inbounds for m in 1:Nfft
        v = abs(ah[m])
        peak = max(peak, v)
        kk = m - 1 < Nfft ÷ 2 ? m - 1 : m - 1 - Nfft
        if abs(kk) > Nfft ÷ 2 - Nfft ÷ 16
            edge = max(edge, v)
        end
    end

    return (a = a, Ngsm = Ngsm, E = Ev, poles = nothing, residues = nothing,
            R = bins, band_ratio = peak > 0 ? edge / peak : 0.0)
end


# ============================================================
# 8. PUBLIC SOLVER
# ============================================================

"""
    vz_solve(sys; Ttotal, Nt_save = 5001, method = :auto,
             refine = 4, pad = 2, damp = 20.0, quad_nodes = 10,
             bin_deltas = nothing, bin_gs = nothing, verbose = false)

Solve the linearised cavity + inhomogeneous spin ensemble
analytically over `[0, Ttotal]` on `Nt_save` uniformly spaced
points.  No ODE is integrated.

`method`:

* `:auto` (default) -- `:residue` when the detuning kernel is
  rational (`DetuningLorentzian`, `DetuningPowerLaw`), `:fft`
  otherwise.
* `:residue` -- exact closed-form sum of exponentials.  The poles
  and their residues are returned.
* `:fft` -- Bromwich inversion of the exact transform.  `refine`
  oversamples the internal grid, `pad` sets the inversion window
  (`pad * Ttotal`), `damp` is the contour offset `c * T_window`.
  The inversion is spectrally accurate when the solution and the
  drive rise from and return to zero inside the window.  A drive
  that is still on at `t = 0` or `t = Ttotal` (a constant or
  square drive, say) makes the drive transform converge only as
  `O(h)`; raise `refine` there, or use `:residue`, which handles
  it exactly.

Pass `bin_deltas` (and optionally `bin_gs`) to also reconstruct the
resolved single-spin coherences.

Returns a `NamedTuple` with

* `t` -- save times
* `a` -- cavity amplitude `<a>`
* `a_out` -- output field `E(t) - sqrt(kappa_e) <a>`, matching the
  main package's convention
* `E` -- drive sampled at `t`
* `Sigma_p`, `Sigma_m` -- collective coherences `sum_j <S_j^+>`
  and `sum_j <S_j^->`
* `Sigma_z` -- collective inversion `sum_j <S_j^z>` to leading
  order
* `Ngsigma_m` -- `N <g sigma^->`, the source term driving the cavity
* `poles`, `residues` -- polariton poles `s_k` and residues (only
  for `:residue`)
* `linear_validity` -- largest fractional change of the TOTAL
  inversion.  It is an ensemble average: bins near resonance
  deplete considerably more than this, so in practice the solution
  tracks the full nonlinear dynamics to a relative accuracy of
  order a few tens of times `linear_validity`.  Keep it well below
  1e-3 for quantitative work
* `sigma_minus`, `sigma_plus` -- resolved single-spin coherences of
  shape `(length(bin_deltas), length(bin_gs), Nt_save)`, returned
  when `bin_deltas` is given (`bin_gs` defaults to the mean
  coupling).  Multiply by the bin populations `N_j` to recover the
  main package's per-bin `Sp` totals
* plus `N_total`, `C_ens`, `Omega2`, `g_moments`, `detuning`,
  `coupling`, `method`
"""
function vz_solve(sys::VZSystem; Ttotal, Nt_save::Integer = 5001,
                  method::Symbol = :auto, refine::Integer = 4,
                  pad::Integer = 2, damp::Real = 20.0,
                  quad_nodes::Integer = 10,
                  bin_deltas = nothing, bin_gs = nothing,
                  verbose::Bool = false)

    Ttotal > 0 || error("VolkovZon: Ttotal must be positive.")
    Nt_save >= 2 || error("VolkovZon: Nt_save must be at least 2.")
    refine >= 1 || error("VolkovZon: refine must be at least 1.")
    pad >= 2 || error("VolkovZon: pad must be at least 2.")
    quad_nodes >= 2 || error("VolkovZon: quad_nodes must be at least 2.")

    chosen = method
    if chosen === :auto
        chosen = is_rational(sys.detuning) ? :residue : :fft
    end
    if chosen === :residue && !is_rational(sys.detuning)
        error("VolkovZon: method = :residue needs a rational detuning " *
              "kernel (DetuningLorentzian or DetuningPowerLaw); got " *
              "$(typeof(sys.detuning)). Use method = :fft.")
    end
    chosen in (:residue, :fft) ||
        error("VolkovZon: unknown method = $(method). " *
              "Use :auto, :residue or :fft.")

    t = collect(range(0.0, Float64(Ttotal); length = Int(Nt_save)))
    deltas = bin_deltas === nothing ? nothing : Float64[Float64(d) for d in bin_deltas]
    gs = bin_gs === nothing ? [sys.g1] : Float64[Float64(g) for g in bin_gs]

    raw = if chosen === :residue
        _solve_residue(sys, t, Int(quad_nodes), deltas)
    else
        _solve_fft(sys, t, Int(refine), Int(pad), Float64(damp), deltas)
    end

    a, Ngsm, Ev = raw.a, raw.Ngsm, raw.E

    # collective coherence:
    #   <sigma^-> = coherence0 K(t)
    #               + (<g>/<g^2>) ( <g sigma^-> - coherence0 <g> K(t) )
    Kt = ComplexF64[char_fn(sys.detuning, tn) for tn in t]
    free = sys.N * sys.coherence0 .* Kt
    Sigma_m = free .+ (sys.g1 / sys.g2) .* (Ngsm .- sys.g1 .* free)
    Sigma_p = conj.(Sigma_m)

    # leading-order inversion:  d/dt Sigma^z = 2 Im( a conj(N <g sigma^->) )
    integrand = Float64[2 * imag(a[n] * conj(Ngsm[n])) for n in eachindex(t)]
    Sigma_z = sys.N * sys.s0 / 2 .+ _cumint(integrand, t[2] - t[1])

    validity = sys.N > 0 ?
        maximum(abs, Sigma_z .- sys.N * sys.s0 / 2) / (sys.N / 2) : 0.0

    a_out = Ev .- sqrt(sys.kappa_e) .* a

    sigma_m = nothing
    sigma_p = nothing
    if deltas !== nothing
        R = raw.R
        sigma_m = Array{ComplexF64}(undef, length(deltas), length(gs), length(t))
        @inbounds for i in eachindex(deltas), j in eachindex(gs), n in eachindex(t)
            sigma_m[i, j, n] = sys.coherence0 * exp(-im * deltas[i] * t[n]) +
                               im * sys.s0 * gs[j] * R[i, n]
        end
        sigma_p = conj.(sigma_m)
    end

    if verbose
        println("VolkovZon: method = $(chosen)")
        println("  total spin number N = $(sys.N)")
        println("  C_ens = $(sys.C_ens),  <g> = $(sys.g1),  <g^2> = $(sys.g2)")
        if raw.poles !== nothing
            println("  poles = $(raw.poles)")
        end
        println("  linear_validity (max fractional inversion change) = $(validity)")
    end

    if validity > 0.1
        @warn "VolkovZon: the ensemble inversion changes by " *
              "$(round(validity * 100; digits = 1))% -- the linearisation " *
              "behind the Volkov-Zon method is no longer reliable. Reduce " *
              "the drive amplitude or the initial coherence."
    end
    if chosen === :fft && hasproperty(raw, :band_ratio) && raw.band_ratio > 1e-4
        @warn "VolkovZon: the transform still carries weight at the grid's " *
              "band edge; the save grid under-resolves the dynamics. " *
              "Increase Nt_save or refine."
    end

    return (t = t, a = a, a_out = a_out, E = Ev,
            Sigma_p = Sigma_p, Sigma_m = Sigma_m, Sigma_z = Sigma_z,
            Ngsigma_m = Ngsm,
            poles = raw.poles, residues = raw.residues,
            linear_validity = validity,
            sigma_minus = sigma_m, sigma_plus = sigma_p,
            bin_deltas = deltas, bin_gs = deltas === nothing ? nothing : gs,
            N_total = sys.N, C_ens = sys.C_ens, Omega2 = sys.Omega2,
            g_moments = (g1 = sys.g1, g2 = sys.g2),
            detuning = _detuning_info(sys.detuning),
            coupling = _coupling_info(sys.coupling),
            method = chosen)
end


# ============================================================
# 9. ENSEMBLE QUADRATURE GRID
# ============================================================

function _detuning_nodes(d::DetuningGaussian, M::Int)
    nodes, w = _gl_on(M, d.center - 8 * d.sigma, d.center + 8 * d.sigma)
    return nodes, w .* [detuning_pdf(d, x) for x in nodes]
end

function _detuning_nodes(d::Union{DetuningLorentzian, DetuningPowerLaw}, M::Int)
    th, w = _gl_on(M, -pi / 2 + 1e-13, pi / 2 - 1e-13)
    nodes = d.center .+ d.w .* tan.(th)
    return nodes, w .* [detuning_pdf(d, x) for x in nodes] .*
                  (d.w ./ cos.(th) .^ 2)
end

_coupling_nodes(c::CouplingConstant, ::Int) = ([c.g], [1.0])

function _coupling_nodes(c::CouplingPowerLaw, M::Int)
    lg, w = _gl_on(M, log(c.g_min), log(c.g_max))
    nodes = exp.(lg)
    return nodes, w .* nodes .* [coupling_pdf(c, g) for g in nodes]
end

function _coupling_nodes(c::Union{CouplingGaussian, CouplingLorentzian}, M::Int)
    lo, hi = coupling_support(c)
    nodes, w = _gl_on(M, lo, hi)
    return nodes, w .* [coupling_pdf(c, g) for g in nodes]
end

"""
    vz_ensemble_grid(sys; M_delta, M_g) -> (deltas, wd, gs, wg)

Quadrature nodes and normalised weights for the detuning and
coupling densities, suitable for turning the resolved coherences
of [`vz_solve`](@ref) back into ensemble sums:

    Sigma_m[n] ≈ N * sum_{i,j} wd[i] * wg[j] * sigma_minus[i, j, n]

Heavy-tailed detuning densities are handled with the substitution
`delta = center + w tan(theta)`, which maps the whole real line
onto a finite interval; the Gaussian uses a plain 8-sigma range.
"""
function vz_ensemble_grid(sys::VZSystem; M_delta::Integer = 400,
                          M_g::Integer = 16)
    deltas, wd = _detuning_nodes(sys.detuning, Int(M_delta))
    gs, wg = _coupling_nodes(sys.coupling, Int(M_g))
    return deltas, wd ./ sum(wd), gs, wg ./ sum(wg)
end

end # module VolkovZon
