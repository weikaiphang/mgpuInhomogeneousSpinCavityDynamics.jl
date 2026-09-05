# Persistent caches above the warm serial RHS (`@allocated == 0`).
# Owns save buffers, callback host scratch, RK stages, and the PI controller
# so the CPU integrator / DiffEq callbacks do not allocate per step.
#
# Tsit5 (default): 9 registers — U, TMP, K1..K7 (MGPU Tsit5 map). Dense
# interpolant unused on this :tstops path.
# CK45: 5 registers — UPREV, STAGE, ACC, K, ERR (MGPU CK_* / Ark B6-style
# low-storage). Mapped onto u, tmp, k1, k2, k3; k4..k7 stay empty.

const _SAVE2_PREFIX_FIELDS = 5  # Sp, Sz, adSp, adSm, adSz

@inline _save2_prefix_length(M::Integer) = 3 + _SAVE2_PREFIX_FIELDS * M

@inline function _cpu_stage_count(integrator::Symbol)
    integrator === :tsit5 && return 9
    integrator === :ck45 && return 5
    error("Unknown integrator = $(integrator). Use :tsit5 or :ck45.")
end

mutable struct Solve2Stats
    nsteps::Int
    naccept::Int
    nreject::Int
    nrhs::Int
    elapsed::Float64
end

Solve2Stats() = Solve2Stats(0, 0, 0, 0, 0.0)

mutable struct Solver2Workspace{T,U,V,Tab}
    M::Int
    Nt::Int
    integrator::Symbol
    u::U
    tmp::U
    k1::U
    k2::U
    k3::U
    k4::U
    k5::U
    k6::U
    k7::U
    rhs::RHS2Workspace{V}
    host::Vector{Complex{T}}
    a::Vector{ComplexF64}
    adad::Vector{ComplexF64}
    n::Vector{Float64}
    Sp::Matrix{ComplexF64}
    Sz::Matrix{ComplexF64}
    adSp::Matrix{ComplexF64}
    adSm::Matrix{ComplexF64}
    adSz::Matrix{ComplexF64}
    tab::Tab
    ctrl::PIController{T}
    saved::Base.RefValue{Int}
    nrhs::Base.RefValue{Int}
    stats::Solve2Stats
end

function Solver2Workspace(::Type{T}, M::Integer, Nt::Integer;
                          stages::Bool = true,
                          integrator::Symbol = :tsit5) where {T}
    nreg = stages ? _cpu_stage_count(integrator) : 0
    n = state_length_2nd_order(M)
    full() = zeros(Complex{T}, n)
    emptyv() = Vector{Complex{T}}(undef, 0)
    stage(i) = (stages && i <= nreg) ? full() : emptyv()
    u = stage(1)
    rhs = _rhs2_workspace(Vector{Complex{T}}(undef, 0), M)
    tab = build_tableau(integrator, T)
    return Solver2Workspace{T,typeof(u),typeof(rhs.gconjSp),typeof(tab)}(
        Int(M), Int(Nt), integrator,
        u, stage(2), stage(3), stage(4), stage(5), stage(6), stage(7), stage(8), stage(9),
        rhs,
        Vector{Complex{T}}(undef, _save2_prefix_length(M)),
        Vector{ComplexF64}(undef, Nt),
        Vector{ComplexF64}(undef, Nt),
        Vector{Float64}(undef, Nt),
        Matrix{ComplexF64}(undef, M, Nt),
        Matrix{ComplexF64}(undef, M, Nt),
        Matrix{ComplexF64}(undef, M, Nt),
        Matrix{ComplexF64}(undef, M, Nt),
        Matrix{ComplexF64}(undef, M, Nt),
        tab,
        PIController(T, alg_order(tab)),
        Ref(0),
        Ref(0),
        Solve2Stats(),
    )
end

function attach_u0!(ws::Solver2Workspace, u0)
    length(u0) == length(ws.u) || error(
        "u0 length $(length(u0)) ≠ workspace state $(length(ws.u)).")
    copyto!(ws.u, u0)
    ws.saved[] = 0
    ws.nrhs[] = 0
    ws.ctrl.qold = ws.ctrl.qoldinit
    st = ws.stats
    st.nsteps = 0
    st.naccept = 0
    st.nreject = 0
    st.nrhs = 0
    st.elapsed = 0.0
    return ws
end

# No Array() per save. Array u copies the prefix in-place; CuArray uses copyto!.
function record_save2!(ws::Solver2Workspace, u, k::Integer)
    h = ws.host
    npref = length(h)
    if u isa Array
        @inbounds for i in 1:npref
            h[i] = u[i]
        end
    else
        copyto!(h, 1, u, 1, npref)
    end
    M = ws.M
    ws.a[k]    = ComplexF64(h[IDX2_a])
    ws.adad[k] = ComplexF64(h[IDX2_ad_ad])
    ws.n[k]    = Float64(real(h[IDX2_ad_a]))
    @inbounds for j in 1:M
        ws.Sp[j, k]   = ComplexF64(h[3 + j])
        ws.Sz[j, k]   = ComplexF64(h[3 + M + j])
        ws.adSp[j, k] = ComplexF64(h[3 + 2M + j])
        ws.adSm[j, k] = ComplexF64(h[3 + 3M + j])
        ws.adSz[j, k] = ComplexF64(h[3 + 4M + j])
    end
    ws.saved[] = k
    return nothing
end

@inline function _lincomb!(out, base, dt, k1, a1)
    @inbounds for i in eachindex(out)
        out[i] = muladd(dt * a1, k1[i], base[i])
    end
    return nothing
end

@inline function _lincomb!(out, base, dt, k1, a1, k2, a2)
    @inbounds for i in eachindex(out)
        out[i] = muladd(dt * a2, k2[i], muladd(dt * a1, k1[i], base[i]))
    end
    return nothing
end

@inline function _lincomb!(out, base, dt, k1, a1, k2, a2, k3, a3)
    @inbounds for i in eachindex(out)
        s = muladd(dt * a1, k1[i], base[i])
        s = muladd(dt * a2, k2[i], s)
        out[i] = muladd(dt * a3, k3[i], s)
    end
    return nothing
end

@inline function _lincomb!(out, base, dt, k1, a1, k2, a2, k3, a3, k4, a4)
    @inbounds for i in eachindex(out)
        s = muladd(dt * a1, k1[i], base[i])
        s = muladd(dt * a2, k2[i], s)
        s = muladd(dt * a3, k3[i], s)
        out[i] = muladd(dt * a4, k4[i], s)
    end
    return nothing
end

@inline function _lincomb!(out, base, dt, k1, a1, k2, a2, k3, a3, k4, a4, k5, a5)
    @inbounds for i in eachindex(out)
        s = muladd(dt * a1, k1[i], base[i])
        s = muladd(dt * a2, k2[i], s)
        s = muladd(dt * a3, k3[i], s)
        s = muladd(dt * a4, k4[i], s)
        out[i] = muladd(dt * a5, k5[i], s)
    end
    return nothing
end

@inline function _lincomb!(out, base, dt, k1, a1, k2, a2, k3, a3, k4, a4, k5, a5, k6, a6)
    @inbounds for i in eachindex(out)
        s = muladd(dt * a1, k1[i], base[i])
        s = muladd(dt * a2, k2[i], s)
        s = muladd(dt * a3, k3[i], s)
        s = muladd(dt * a4, k4[i], s)
        s = muladd(dt * a5, k5[i], s)
        out[i] = muladd(dt * a6, k6[i], s)
    end
    return nothing
end

@inline function _scaled_norm_vs_zero(x, xref, atol, rtol)
    acc = zero(real(eltype(x)))
    n = length(x)
    @inbounds for i in 1:n
        sc = atol + rtol * abs(xref[i])
        acc += abs2(x[i] / sc)
    end
    return sqrt(acc / n)
end

@inline function _tsit5_eest(u, y1, k1, k2, k3, k4, k5, k6, k7,
                             bt1, bt2, bt3, bt4, bt5, bt6, bt7, dt, atol, rtol)
    acc = zero(real(eltype(u)))
    n = length(u)
    @inbounds for i in 1:n
        e = dt * (bt1 * k1[i] + bt2 * k2[i] + bt3 * k3[i] + bt4 * k4[i] +
                  bt5 * k5[i] + bt6 * k6[i] + bt7 * k7[i])
        sc = atol + rtol * max(abs(u[i]), abs(y1[i]))
        acc += abs2(e / sc)
    end
    return sqrt(acc / n)
end

function _cpu_rhs!(ws::Solver2Workspace, du, u, p, t)
    rhs_2nd_order!(du, u, p, t)
    ws.nrhs[] += 1
    return nothing
end

# One Tsit5 stage sweep. k1 must already be f(u, t). Writes the 5th-order
# advance into tmp and the FSAL stage into k7. Returns the scaled error.
function tsit5_cpu_step!(ws::Solver2Workspace{T}, p, t::T, dt::T,
                         atol::T, rtol::T) where {T}
    tab = ws.tab
    u = ws.u
    tmp = ws.tmp
    k1, k2, k3, k4, k5, k6, k7 = ws.k1, ws.k2, ws.k3, ws.k4, ws.k5, ws.k6, ws.k7

    _lincomb!(tmp, u, dt, k1, tab.a21)
    _cpu_rhs!(ws, k2, tmp, p, t + tab.c2 * dt)

    _lincomb!(tmp, u, dt, k1, tab.a31, k2, tab.a32)
    _cpu_rhs!(ws, k3, tmp, p, t + tab.c3 * dt)

    _lincomb!(tmp, u, dt, k1, tab.a41, k2, tab.a42, k3, tab.a43)
    _cpu_rhs!(ws, k4, tmp, p, t + tab.c4 * dt)

    _lincomb!(tmp, u, dt, k1, tab.a51, k2, tab.a52, k3, tab.a53, k4, tab.a54)
    _cpu_rhs!(ws, k5, tmp, p, t + tab.c5 * dt)

    _lincomb!(tmp, u, dt, k1, tab.a61, k2, tab.a62, k3, tab.a63, k4, tab.a64, k5, tab.a65)
    _cpu_rhs!(ws, k6, tmp, p, t + tab.c6 * dt)

    _lincomb!(tmp, u, dt, k1, tab.a71, k2, tab.a72, k3, tab.a73, k4, tab.a74, k5, tab.a75, k6, tab.a76)
    _cpu_rhs!(ws, k7, tmp, p, t + dt)

    return _tsit5_eest(u, tmp, k1, k2, k3, k4, k5, k6, k7,
                       tab.bt1, tab.bt2, tab.bt3, tab.bt4, tab.bt5, tab.bt6, tab.bt7,
                       dt, atol, rtol)
end

# CK45 / Carpenter–Kennedy 5-stage 4th-order (MGPU lowstorage_kernel!).
# Registers: u=UPREV, tmp=STAGE, k1=ACC, k2=K, k3=ERR.
@inline function _ck45_lowstorage!(stage, accum, err, k, base, dt, cA, cB, cE,
                                   ::Val{write_stage}, ::Val{first}) where {write_stage, first}
    dA = dt * cA
    dB = dt * cB
    dE = dt * cE
    @inbounds for i in eachindex(base)
        ki = k[i]
        ai = base[i]
        write_stage && (stage[i] = muladd(dA, ki, ai))
        accum[i] = muladd(dB, ki, ai)
        err[i] = first ? (dE * ki) : muladd(dE, ki, err[i])
    end
    return nothing
end

@inline function _ck45_eest(uprev, uacc, err, atol, rtol)
    acc = zero(real(eltype(uprev)))
    n = length(uprev)
    @inbounds for i in 1:n
        sc = atol + rtol * max(abs(uprev[i]), abs(uacc[i]))
        acc += abs2(err[i] / sc)
    end
    return sqrt(acc / n)
end

function ck45_cpu_step!(ws::Solver2Workspace{T}, p, t::T, dt::T,
                        atol::T, rtol::T) where {T}
    tab = ws.tab
    uprev = ws.u
    stage = ws.tmp
    accum = ws.k1
    k = ws.k2
    err = ws.k3

    # Stage 1: k already f(uprev, t).
    _ck45_lowstorage!(stage, accum, err, k, uprev, dt, tab.A[1], tab.b[1], tab.bt[1],
                      Val(true), Val(true))
    _cpu_rhs!(ws, k, stage, p, t + tab.c[2] * dt)
    _ck45_lowstorage!(stage, accum, err, k, accum, dt, tab.A[2], tab.b[2], tab.bt[2],
                      Val(true), Val(false))
    _cpu_rhs!(ws, k, stage, p, t + tab.c[3] * dt)
    _ck45_lowstorage!(stage, accum, err, k, accum, dt, tab.A[3], tab.b[3], tab.bt[3],
                      Val(true), Val(false))
    _cpu_rhs!(ws, k, stage, p, t + tab.c[4] * dt)
    _ck45_lowstorage!(stage, accum, err, k, accum, dt, tab.A[4], tab.b[4], tab.bt[4],
                      Val(true), Val(false))
    _cpu_rhs!(ws, k, stage, p, t + tab.c[5] * dt)
    _ck45_lowstorage!(stage, accum, err, k, accum, dt, zero(T), tab.b[5], tab.bt[5],
                      Val(false), Val(false))

    EEst = _ck45_eest(uprev, accum, err, atol, rtol)
    # Restore / refresh K so the next attempt starts with f(current u).
    # Accept path: f(ACC); reject path: f(UPREV). Matches MGPU ck45_step!.
    if EEst <= one(T)
        _cpu_rhs!(ws, k, accum, p, t + dt)
    else
        _cpu_rhs!(ws, k, uprev, p, t)
    end
    return EEst
end

@inline _cpu_step!(ws::Solver2Workspace{T,U,V,<:Tsit5Tableau}, p, t::T, dt::T, atol::T, rtol::T) where {T,U,V} =
    tsit5_cpu_step!(ws, p, t, dt, atol, rtol)
@inline _cpu_step!(ws::Solver2Workspace{T,U,V,<:CK45Tableau}, p, t::T, dt::T, atol::T, rtol::T) where {T,U,V} =
    ck45_cpu_step!(ws, p, t, dt, atol, rtol)

@inline _k0(ws::Solver2Workspace{T,U,V,<:Tsit5Tableau}) where {T,U,V} = ws.k1
@inline _k0(ws::Solver2Workspace{T,U,V,<:CK45Tableau}) where {T,U,V} = ws.k2

@inline function _accept_advance!(ws::Solver2Workspace{T,U,V,<:Tsit5Tableau}) where {T,U,V}
    ws.u, ws.tmp = ws.tmp, ws.u
    ws.k1, ws.k7 = ws.k7, ws.k1
    return nothing
end

@inline function _accept_advance!(ws::Solver2Workspace{T,U,V,<:CK45Tableau}) where {T,U,V}
    ws.u, ws.k1 = ws.k1, ws.u
    return nothing
end

# Adaptive RK on the persistent workspace. Hits tsave exactly (:tstops).
# Physics is rhs_2nd_order! → rhs_cpu! (serial warm path already no-alloc).
# Writes Solve2Stats in-place (no NamedTuple) so a warm solve can be 0-alloc.
function solve_cpu_2nd!(ws::Solver2Workspace{T}, p, t0::Real, tend::Real,
                        tsave::AbstractVector;
                        integrator::Union{Nothing,Symbol} = nothing,
                        reltol::Real = 1e-8, abstol::Real = 1e-8,
                        dtmax::Real = Inf, dt0::Real = 0,
                        maxiters::Int = 10_000_000) where {T}
    if integrator !== nothing && integrator !== ws.integrator
        error("solve_cpu_2nd! integrator=$(integrator) but workspace is $(ws.integrator); " *
              "construct Solver2Workspace(; integrator=$(integrator)).")
    end
    atol = T(abstol)
    rtol = T(reltol)
    t = T(t0)
    tfinal = T(tend)
    dtmaxT = T(dtmax)
    nsave = length(tsave)
    isave = 1
    nsteps = 0
    naccept = 0
    nreject = 0
    ws.saved[] = 0
    ws.nrhs[] = 0
    ws.ctrl.qold = ws.ctrl.qoldinit

    if nsave >= 1 && abs(T(tsave[1]) - t) <= eps(T) * max(one(T), abs(t))
        record_save2!(ws, ws.u, 1)
        isave = 2
    end

    _cpu_rhs!(ws, _k0(ws), ws.u, p, t)
    span = abs(tfinal - t)
    if dt0 > 0
        dt_free = T(dt0)
    else
        d0 = _scaled_norm_vs_zero(ws.u, ws.u, atol, rtol)
        d1 = _scaled_norm_vs_zero(_k0(ws), ws.u, atol, rtol)
        dt_free = (d0 < T(1e-5) || d1 < T(1e-5)) ? T(1e-6) * span : T(0.01) * (d0 / d1)
        dt_free = min(dt_free, span, dtmaxT)
        dt_free = max(dt_free, T(1e-12) * span)
    end

    tstart = time_ns()
    while t < tfinal && nsteps < maxiters
        dt = min(dt_free, dtmaxT, tfinal - t)
        forced = false
        t_target = zero(T)
        if isave <= nsave
            tnext = T(tsave[isave])
            if t + dt >= tnext - 100 * eps(T) * max(one(T), abs(tnext))
                dt = tnext - t
                forced = true
                t_target = tnext
            end
        end
        dt <= 0 && break
        nsteps += 1

        EEst = _cpu_step!(ws, p, t, dt, atol, rtol)

        if EEst <= one(T)
            naccept += 1
            t = forced ? t_target : t + dt
            _accept_advance!(ws)
            if forced && isave <= nsave && T(tsave[isave]) == t_target
                record_save2!(ws, ws.u, isave)
                isave += 1
            end
            accept_step!(ws.ctrl, EEst)
            q, _ = controller_factors(ws.ctrl, EEst)
            dt_new = dt / q
            dt_free = forced ? max(dt_free, dt_new) : dt_new
        else
            nreject += 1
            _, q11 = controller_factors(ws.ctrl, EEst)
            dt_free = dt / min(inv(ws.ctrl.qmin), q11 / ws.ctrl.gamma)
            if dt_free < T(0) || !isfinite(dt_free)
                error("Step size underflow at t = $t (dt = $dt_free, EEst = $EEst).")
            end
        end
    end

    elapsed = (time_ns() - tstart) / 1e9
    if nsteps >= maxiters
        @warn "Reached maxiters before the end of the time span." t tfinal nsteps
    end
    st = ws.stats
    st.nsteps = nsteps
    st.naccept = naccept
    st.nreject = nreject
    st.nrhs = ws.nrhs[]
    st.elapsed = elapsed
    return st
end
