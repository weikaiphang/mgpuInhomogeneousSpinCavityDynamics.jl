# ============================================================
# B-SPLINE HELPERS (differentiable composite pi-pulse optimisation)
#
# Cox-de Boor basis evaluation at a single point t, generic over the
# element type of `t`/`knots`/`c` (plain Float64 for an ordinary forward
# solve, ForwardDiff.Dual when differentiating pulse parameters through
# it) -- unlike pulses.jl's gaussian_drive/wurst_drive (which cast their
# parameters to Float64/ComplexF64 on construction), nothing here ever
# narrows a value's type, since that would silently strip ForwardDiff's
# dual partials. Ordinary `if`/ternary branching is used for the 0/0 :=
# 0 degenerate-knot-span convention below rather than a "compute both
# branches, mask, select" trick: unlike JAX's jnp.where, Julia control
# flow under ForwardDiff never evaluates the untaken branch at all, so
# there is no NaN-from-the-other-branch pitfall to guard against here.
# ============================================================

"""
    make_clamped_knots(n_coeff, t0, t1, degree=3)

Open/clamped knot vector for a degree-`degree` B-spline with `n_coeff`
coefficients: `degree+1` copies of `t0`, then `n_coeff-degree-1` interior
knots evenly spaced in `(t0, t1)`, then `degree+1` copies of `t1`.
Differentiable w.r.t. `t0`/`t1` (generic arithmetic only).
"""
function make_clamped_knots(n_coeff::Integer, t0, t1, degree::Integer=3)
    n_interior = n_coeff - degree - 1
    n_interior >= 0 || error("Need at least degree+1 = $(degree+1) coefficients, got $n_coeff.")
    T = promote_type(typeof(t0), typeof(t1))
    knots = Vector{T}(undef, n_coeff + degree + 1)
    knots[1:degree+1] .= t0
    if n_interior > 0
        step = (t1 - t0) / (n_interior + 1)
        for j in 1:n_interior
            knots[degree+1+j] = t0 + j * step
        end
    end
    knots[end-degree:end] .= t1
    return knots
end

"""
    bspline_basis(t, knots, degree) -> Vector

Cox-de Boor basis functions `B_{i,degree}(t)` for `i = 1:(length(knots)-degree-1)`,
evaluated at the single point `t`. Matches the usual extrapolate=false
convention: 0 outside `[knots[1], knots[end]]`, with the right edge
`t == knots[end]` closed (not the usual half-open `[tau_i, tau_i+1)`) so
the last basis function doesn't spuriously evaluate to 0 exactly at the
pulse's own end time -- there is exactly one non-degenerate degree-0
sub-interval whose right edge is genuinely `knots[end]`, even with
clamped/repeated end knots, and only that one is closed.
"""
function bspline_basis(t, knots::AbstractVector, degree::Integer)
    K = length(knots)
    n0 = K - 1
    T = promote_type(typeof(t), eltype(knots))
    B = zeros(T, n0)
    @inbounds for i in 1:n0
        lo, hi = knots[i], knots[i+1]
        if lo <= t < hi
            B[i] = one(T)
        elseif t == knots[end] && hi == knots[end] && lo < hi
            B[i] = one(T)
        end
    end
    for p in 1:degree
        n = K - p - 1
        Bnew = zeros(T, n)
        @inbounds for i in 1:n
            tau_i, tau_ip = knots[i], knots[i+p]
            tau_i1, tau_ip1 = knots[i+1], knots[i+p+1]
            left = tau_ip > tau_i ? (t - tau_i) / (tau_ip - tau_i) * B[i] : zero(T)
            right = tau_ip1 > tau_i1 ? (tau_ip1 - t) / (tau_ip1 - tau_i1) * B[i+1] : zero(T)
            Bnew[i] = left + right
        end
        B = Bnew
    end
    return B
end

"""
    bspline_eval(t, c, knots, degree) -> scalar

`sum(c .* bspline_basis(t, knots, degree))`, i.e. the B-spline's value at
`t`.
"""
function bspline_eval(t, c::AbstractVector, knots::AbstractVector, degree::Integer)
    B = bspline_basis(t, knots, degree)
    s = zero(promote_type(eltype(B), eltype(c)))
    @inbounds for i in eachindex(c)
        s += B[i] * c[i]
    end
    return s
end

"""
    bspline_area(c, knots, degree=3)

Exact integral of a degree-`degree` B-spline (coefficients `c`) over its
full domain `[knots[1], knots[end]]`, via the standard identity
`integral(B_i,degree) = (knots[i+degree+1]-knots[i])/(degree+1)` -- exact
even with repeated/clamped knots (a zero-width span contributes zero).
Valid whenever `c .>= 0` (guaranteed by the caller): the convex-hull
property (basis functions are non-negative and sum to 1) then makes the
spline itself non-negative on its domain, so no clipping is needed.
"""
function bspline_area(c::AbstractVector, knots::AbstractVector, degree::Integer=3)
    n = length(c)
    area = zero(promote_type(eltype(c), eltype(knots)))
    @inbounds for i in 1:n
        width = knots[i+degree+1] - knots[i]
        area += c[i] * width
    end
    return area / (degree + 1)
end

"""
    bspline_antiderivative(c, knots, degree=3) -> (knots_prime, d)

The antiderivative of the degree-`degree` B-spline `sum(c .*
bspline_basis(., knots, degree))` is itself a degree-`(degree+1)`
B-spline (de Boor's construction): one extra knot at each end, and
coefficients `d` built by cumulatively summing each basis function's own
exact integral. `d[1] = 0`, i.e. the returned antiderivative is
referenced to zero at `knots[1]`. Evaluate
`bspline_eval(t, d, knots_prime, degree+1)` to get the pointwise
antiderivative (the exact phase integral, no numerical quadrature).
"""
function bspline_antiderivative(c::AbstractVector, knots::AbstractVector, degree::Integer=3)
    n = length(c)
    T = promote_type(eltype(c), eltype(knots))
    d = Vector{T}(undef, n + 1)
    d[1] = zero(T)
    @inbounds for i in 1:n
        width = knots[i+degree+1] - knots[i]
        d[i+1] = d[i] + c[i] * width / (degree + 1)
    end
    knots_prime = vcat(knots[1:1], knots, knots[end:end])
    return knots_prime, d
end
