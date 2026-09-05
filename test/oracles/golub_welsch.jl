# Optional parity oracle only. Not used in production quadrature paths.

function _gauss_legendre_pts_golub_welsch(n::Integer)
    n >= 1 || error("_gauss_legendre_pts_golub_welsch: n must be >= 1")
    n == 1 && return ([0.0], [2.0])
    k = collect(1.0:(n - 1))
    beta = k ./ sqrt.(4 .* k .^ 2 .- 1.0)
    E = eigen(SymTridiagonal(zeros(Float64, n), beta))
    x = E.values
    w = 2.0 .* (E.vectors[1, :] .^ 2)
    p = sortperm(x)
    return x[p], w[p]
end
