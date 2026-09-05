
function make_clamped_knots(n_coeff::Integer, t0, t1, degree::Integer=3)
    n_interior = n_coeff - degree - 1
    n_interior >= 0 || error("Need at least degree+1 = $(degree+1) coefficients, got $n_coeff.")
    degree >= 0 || error("degree must be >= 0, got $degree.")
    T = promote_type(typeof(t0), typeof(t1))
    n_knots = n_coeff + degree + 1
    knots = Vector{T}(undef, n_knots)
    knots[1:degree+1] .= t0
    if n_interior > 0
        step = (t1 - t0) / (n_interior + 1)
        for j in 1:n_interior
            knots[degree+1+j] = t0 + j * step
        end
    end
    knots[end-degree:end] .= t1
    length(knots) == n_knots || error(
        "internal inconsistency: clamped knot vector length $(length(knots)) != n_coeff+degree+1 = $n_knots."
    )
    return knots
end


function _bspline_knot_coeff_check(c::AbstractVector, knots::AbstractVector, degree::Integer, fname::AbstractString)
    n = length(c)
    expected = n + degree + 1
    length(knots) == expected || error(
        "$fname: knots has length $(length(knots)), but length(c)=$n and degree=$degree " *
        "require length(knots) = n+degree+1 = $expected."
    )
    return n
end

# Per-task grow-to-fit Cox–de Boor pair. No Dict get!-do (that closure
# allocated on every call). Keyed by eltype so Dual tags do not share
# Float64 buffers. Warm lookup + reuse is 0-alloc.
mutable struct _BsplineTLS{T}
    a::Vector{T}
    b::Vector{T}
end

function _bspline_tls(::Type{T}) where {T}
    tls = task_local_storage()
    bag = get(tls, :_isc_bspline_bag, nothing)
    if bag === nothing
        bag = IdDict{DataType,Any}()
        tls[:_isc_bspline_bag] = bag
    end
    s = get(bag, T, nothing)
    if s === nothing
        s = _BsplineTLS{T}(Vector{T}(undef, 0), Vector{T}(undef, 0))
        bag[T] = s
    end
    return s::_BsplineTLS{T}
end

function _bspline_scratch_pair(::Type{T}, n0::Integer) where {T}
    s = _bspline_tls(T)
    if length(s.a) < n0
        resize!(s.a, Int(n0))
        resize!(s.b, Int(n0))
    end
    return s.a, s.b
end

# Cox–de Boor into caller buffers (no views). Returns (src, n) with
# basis of length n = K - degree - 1 in src[1:n].
function _bspline_basis!(a::AbstractVector{TB}, b::AbstractVector{TB},
                         t, knots::AbstractVector, degree::Integer) where {TB}
    K = length(knots)
    K >= degree + 2 || error(
        "bspline_basis: knots has length $K, need at least degree+2 = $(degree+2) for degree=$degree."
    )
    n0 = K - 1
    length(a) >= n0 && length(b) >= n0 || error(
        "bspline scratch is shorter than n0=$n0 (a=$(length(a)), b=$(length(b)))."
    )
    tv = ForwardDiff.value(t)
    knots_end_v = ForwardDiff.value(knots[end])
    @inbounds for i in 1:n0
        lo, hi = knots[i], knots[i+1]
        lov, hiv = ForwardDiff.value(lo), ForwardDiff.value(hi)
        if lov <= tv < hiv
            a[i] = one(TB)
        elseif tv == knots_end_v && hiv == knots_end_v && lov < hiv
            a[i] = one(TB)
        else
            a[i] = zero(TB)
        end
    end
    src = a
    dst = b
    n = n0
    for p in 1:degree
        n = K - p - 1
        @inbounds for i in 1:n
            tau_i, tau_ip = knots[i], knots[i+p]
            tau_i1, tau_ip1 = knots[i+1], knots[i+p+1]
            left = tau_ip > tau_i ? (t - tau_i) / (tau_ip - tau_i) * src[i] : zero(TB)
            right = tau_ip1 > tau_i1 ? (tau_ip1 - t) / (tau_ip1 - tau_i1) * src[i+1] : zero(TB)
            dst[i] = left + right
        end
        src, dst = dst, src
    end
    return src, n
end

@inline function _bspline_eval_dot!(a, b, t, c::AbstractVector, knots, degree)
    src, n = _bspline_basis!(a, b, t, knots, degree)
    n == length(c) || error(
        "bspline_eval: basis length $n != coefficient length $(length(c))."
    )
    s = zero(promote_type(eltype(src), eltype(c)))
    @inbounds for i in 1:n
        s += src[i] * c[i]
    end
    return s
end

@inline function _bspline_eval_dot!(a, b, t, C::AbstractMatrix, col::Integer, knots, degree)
    src, n = _bspline_basis!(a, b, t, knots, degree)
    n == size(C, 1) || error(
        "bspline_eval: basis length $n != coefficient rows $(size(C, 1))."
    )
    s = zero(promote_type(eltype(src), eltype(C)))
    @inbounds for i in 1:n
        s += src[i] * C[i, col]
    end
    return s
end


function bspline_basis!(out::AbstractVector, t, knots::AbstractVector, degree::Integer)
    K = length(knots)
    n = K - degree - 1
    n >= 1 || error("bspline_basis: need at least degree+2 knots, got length $K for degree=$degree.")
    length(out) >= n || error(
        "bspline_basis!: output length $(length(out)) < basis length $n."
    )
    T = promote_type(typeof(t), eltype(knots))
    a, b = _bspline_scratch_pair(T, K - 1)
    src, nB = _bspline_basis!(a, b, t, knots, degree)
    nB == n || error("bspline_basis!: internal length $nB != $n.")
    @inbounds for i in 1:n
        out[i] = src[i]
    end
    return out
end


function bspline_basis(t, knots::AbstractVector, degree::Integer)
    n = length(knots) - degree - 1
    T = promote_type(typeof(t), eltype(knots))
    out = Vector{T}(undef, n)
    return bspline_basis!(out, t, knots, degree)
end


function bspline_eval(t, c::AbstractVector, knots::AbstractVector, degree::Integer)
    _bspline_knot_coeff_check(c, knots, degree, "bspline_eval")
    T = promote_type(typeof(t), eltype(knots))
    a, b = _bspline_scratch_pair(T, length(knots) - 1)
    return _bspline_eval_dot!(a, b, t, c, knots, degree)
end

function bspline_eval(t, C::AbstractMatrix, col::Integer, knots::AbstractVector, degree::Integer)
    n = size(C, 1)
    expected = n + degree + 1
    length(knots) == expected || error(
        "bspline_eval: knots has length $(length(knots)), but size(C,1)=$n and degree=$degree " *
        "require length(knots) = n+degree+1 = $expected."
    )
    T = promote_type(typeof(t), eltype(knots))
    a, b = _bspline_scratch_pair(T, length(knots) - 1)
    return _bspline_eval_dot!(a, b, t, C, col, knots, degree)
end


function bspline_area(c::AbstractVector, knots::AbstractVector, degree::Integer=3)
    n = _bspline_knot_coeff_check(c, knots, degree, "bspline_area")
    area = zero(promote_type(eltype(c), eltype(knots)))
    @inbounds for i in 1:n
        width = knots[i+degree+1] - knots[i]
        area += c[i] * width
    end
    return area / (degree + 1)
end


function bspline_antiderivative(c::AbstractVector, knots::AbstractVector, degree::Integer=3)
    n = _bspline_knot_coeff_check(c, knots, degree, "bspline_antiderivative")
    T = promote_type(eltype(c), eltype(knots))
    d = Vector{T}(undef, n + 1)
    d[1] = zero(T)
    @inbounds for i in 1:n
        width = knots[i+degree+1] - knots[i]
        d[i+1] = d[i] + c[i] * width / (degree + 1)
    end
    # Construct-time only (not the E(t) hot path). One Vector + copyto!;
    # a new BSpline is allocated here anyway.
    knots_prime = Vector{T}(undef, length(knots) + 2)
    knots_prime[1] = knots[1]
    copyto!(knots_prime, 2, knots, 1, length(knots))
    knots_prime[end] = knots[end]
    length(knots_prime) == length(d) + (degree + 1) + 1 || error(
        "bspline_antiderivative: internal shape mismatch " *
        "(knots_prime=$(length(knots_prime)), n_d=$(length(d)), degree=$(degree+1))."
    )
    return knots_prime, d
end
