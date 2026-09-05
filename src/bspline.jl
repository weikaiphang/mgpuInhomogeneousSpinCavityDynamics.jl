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

mutable struct _BsplineScratchPool
    caches::Vector{Dict{Tuple{DataType,Int},NTuple{2,Vector}}}
end
const _BSPLINE_SCRATCH_POOL = Ref{Union{Nothing,_BsplineScratchPool}}(nothing)
const _BSPLINE_SCRATCH_POOL_LOCK = ReentrantLock()

function _bspline_scratch_pair(::Type{T}, n0::Integer) where {T}
    pool = _BSPLINE_SCRATCH_POOL[]
    tid = Threads.threadid()
    if pool === nothing || tid > length(pool.caches)
        lock(_BSPLINE_SCRATCH_POOL_LOCK) do
            pool2 = _BSPLINE_SCRATCH_POOL[]
            needed = Threads.maxthreadid()
            if pool2 === nothing || length(pool2.caches) < needed
                _BSPLINE_SCRATCH_POOL[] = _BsplineScratchPool(
                    [Dict{Tuple{DataType,Int},NTuple{2,Vector}}() for _ in 1:needed]
                )
            end
        end
        pool = _BSPLINE_SCRATCH_POOL[]
    end
    cache = @inbounds pool.caches[tid]
    return get!(cache, (T, Int(n0))) do
        (Vector{T}(undef, n0), Vector{T}(undef, n0))
    end::NTuple{2,Vector{T}}
end

function _bspline_basis_view(t, knots::AbstractVector, degree::Integer)
    K = length(knots)
    K >= degree + 2 || error(
        "bspline_basis: knots has length $K, need at least degree+2 = $(degree+2) for degree=$degree."
    )
    n0 = K - 1
    T = promote_type(typeof(t), eltype(knots))
    tv = ForwardDiff.value(t)
    knots_end_v = ForwardDiff.value(knots[end])
    buf_a, buf_b = _bspline_scratch_pair(T, n0)
    B = view(buf_a, 1:n0)
    @inbounds for i in 1:n0
        lo, hi = knots[i], knots[i+1]
        lov, hiv = ForwardDiff.value(lo), ForwardDiff.value(hi)
        if lov <= tv < hiv
            B[i] = one(T)
        elseif tv == knots_end_v && hiv == knots_end_v && lov < hiv
            B[i] = one(T)
        else
            B[i] = zero(T)
        end
    end
    for p in 1:degree
        n = K - p - 1
        Bnew = view(isodd(p) ? buf_b : buf_a, 1:n)
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

function bspline_basis(t, knots::AbstractVector, degree::Integer)
    return collect(_bspline_basis_view(t, knots, degree))
end

function bspline_eval(t, c::AbstractVector, knots::AbstractVector, degree::Integer)
    n = _bspline_knot_coeff_check(c, knots, degree, "bspline_eval")
    B = _bspline_basis_view(t, knots, degree)
    length(B) == n || error(
        "bspline_eval: basis length $(length(B)) != coefficient length $n."
    )
    s = zero(promote_type(eltype(B), eltype(c)))
    @inbounds for i in 1:n
        s += B[i] * c[i]
    end
    return s
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
    knots_prime = vcat(knots[1:1], knots, knots[end:end])
    length(knots_prime) == length(d) + (degree + 1) + 1 || error(
        "bspline_antiderivative: internal shape mismatch " *
        "(knots_prime=$(length(knots_prime)), n_d=$(length(d)), degree=$(degree+1))."
    )
    return knots_prime, d
end
