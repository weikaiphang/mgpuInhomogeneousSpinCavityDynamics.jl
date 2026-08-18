# ============================================================
# RUNGE-KUTTA TABLEAUS
# ============================================================

# ------------------------------------------------------------
# Tsitouras 5(4) — the method the single-GPU package uses through
# OrdinaryDiffEq, with the identical coefficients so that step sequences and
# error estimates match.  Nine full-state registers.
# ------------------------------------------------------------
struct Tsit5Tableau{T}
    c2::T; c3::T; c4::T; c5::T; c6::T
    a21::T
    a31::T; a32::T
    a41::T; a42::T; a43::T
    a51::T; a52::T; a53::T; a54::T
    a61::T; a62::T; a63::T; a64::T; a65::T
    a71::T; a72::T; a73::T; a74::T; a75::T; a76::T
    bt1::T; bt2::T; bt3::T; bt4::T; bt5::T; bt6::T; bt7::T
end

function Tsit5Tableau(::Type{T}) where {T}
    return Tsit5Tableau{T}(
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

alg_order(::Tsit5Tableau) = 5

"""
    Tsit5Interp(T)

Coefficients of the 4th-order dense-output polynomials
`b_i(Θ) = r_i1 Θ + r_i2 Θ² + r_i3 Θ³ + r_i4 Θ⁴`.  Used to sample observables
at the requested output times without forcing the integrator to step onto
them.
"""
function Tsit5Interp(::Type{T}) where {T}
    return (
        (T(1.0), T(-2.763706197274826), T(2.9132554618219126), T(-1.0530884977290216)),
        (T(0.0), T(0.13169999999999998), T(-0.2234), T(0.1017)),
        (T(0.0), T(3.9302962368947516), T(-5.941033872131505), T(2.490627285651253)),
        (T(0.0), T(-12.411077166933676), T(30.33818863028232), T(-16.548102889244902)),
        (T(0.0), T(37.50931341651104), T(-88.1789048947664), T(47.37952196281928)),
        (T(0.0), T(-27.896526289197286), T(65.09189467479366), T(-34.87065786149661)),
        (T(0.0), T(1.5), T(-4.0), T(2.5)),
    )
end

"""
    interp_weights(interp, Θ)

Evaluate the seven dense-output polynomials at `Θ ∈ [0,1]`.
"""
function interp_weights(interp::NTuple{7,NTuple{4,T}}, Θ::T) where {T}
    Θ2 = Θ * Θ
    Θ3 = Θ2 * Θ
    Θ4 = Θ2 * Θ2
    return ntuple(i -> begin
                      r = interp[i]
                      r[1] * Θ + r[2] * Θ2 + r[3] * Θ3 + r[4] * Θ4
                  end, Val(7))
end


# ------------------------------------------------------------
# RK4(3)5[2R+]C of Kennedy, Carpenter & Lewis, Appl. Numer. Math. 35 (2000)
# 177-219, Table 1.
#
# A van der Houwen two-register scheme: a[i,j] = b[j] for j <= i-2, so the
# stage values can be advanced with a running accumulator instead of keeping
# every stage derivative.  With the state itself, the accumulator, the
# derivative and an error accumulator that is five registers instead of the
# nine Tsit5 needs — i.e. roughly 1.8x more ensemble bins per GPU, at the
# cost of one order of accuracy.
#
# The coefficients below reproduce all eight fourth-order conditions and the
# embedded third-order conditions to machine precision; `test/` re-checks
# this, and dev/decode_lsrk_tableau.jl documents how they were recovered from
# the published rational forms.
# ------------------------------------------------------------
struct CK45Tableau{T}
    A::NTuple{4,T}      # subdiagonal a[i+1,i]
    b::NTuple{5,T}
    bt::NTuple{5,T}     # b - b̂
    c::NTuple{5,T}
end

function CK45Tableau(::Type{T}) where {T}
    A = (970286171893 / 4311952581923,
         6584761158862 / 12103376702013,
         2251764453980 / 15575788980749,
         26877169314380 / 34165994151039)
    b = (1153189308089 / 22510343858157,
         1772645290293 / 4653164025191,
         -1672844663538 / 4480602732383,
         2114624349019 / 3568978502595,
         5198255086312 / 14908931495163)
    bh = (1016888040809 / 7410784769900,
          11231460423587 / 58533540763752,
          -1563879915014 / 6823010717585,
          606302364029 / 971179775848,
          1097981568119 / 3980877426909)

    # c_i = Σ_j a_ij with the 2R structure
    c = (0.0,
         A[1],
         b[1] + A[2],
         b[1] + b[2] + A[3],
         b[1] + b[2] + b[3] + A[4])

    return CK45Tableau{T}(map(T, A), map(T, b), map(T, b .- bh), map(T, c))
end

alg_order(::CK45Tableau) = 4

build_tableau(alg::Symbol, ::Type{T}) where {T} =
    alg === :tsit5 ? Tsit5Tableau(T) :
    alg === :ck45  ? CK45Tableau(T)  :
    error("Unknown integrator = $(alg). Use :tsit5 or :ck45.")
