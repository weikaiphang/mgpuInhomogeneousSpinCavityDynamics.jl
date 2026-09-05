struct Tsit5DiscAdjTableau{T}
    c1::T
    c2::T
    c3::T
    c4::T
    a21::T
    a31::T
    a32::T
    a41::T
    a42::T
    a43::T
    a51::T
    a52::T
    a53::T
    a54::T
    a61::T
    a62::T
    a63::T
    a64::T
    a65::T
    b1::T
    b2::T
    b3::T
    b4::T
    b5::T
    b6::T
end

function Tsit5DiscAdjTableau(::Type{T}=Float64) where {T}
    return Tsit5DiscAdjTableau{T}(
        T(0.161),
        T(0.327),
        T(0.9),
        T(0.9800255409045097),
        T(0.161),
        T(-0.008480655492356989),
        T(0.335480655492357),
        T(2.8971530571054935),
        T(-6.359448489975075),
        T(4.3622954328695815),
        T(5.325864828439257),
        T(-11.748883564062828),
        T(7.4955393428898365),
        T(-0.09249506636175525),
        T(5.86145544294642),
        T(-12.92096931784711),
        T(8.159367898576159),
        T(-0.071584973281401),
        T(-0.028269050394068383),
        T(0.09646076681806523),
        T(0.01),
        T(0.4798896504144996),
        T(1.379008574103742),
        T(-3.290069515436081),
        T(2.324710524099774),
    )
end

const TSIT5_DISC_ADJ_TAB = Tsit5DiscAdjTableau(Float64)

mutable struct FrozenTsit5Mesh
    t::Vector{Float64}
    dt::Vector{Float64}
    u::Vector{Vector{ComplexF64}}
end

struct HostCheckpointStack
    t::Vector{Float64}
    u::Vector{Vector{ComplexF64}}
    index::Vector{Int}
    stride::Int
end

struct Tsit5DiscAdjWorkspace
    k::NTuple{6,Vector{ComplexF64}}
    y::NTuple{6,Vector{ComplexF64}}
    u_np1::Vector{ComplexF64}
    x::Vector{Float64}
    x̄::Vector{Float64}
    λu::Vector{Float64}
    λy::Vector{Float64}
    λk::NTuple{6,Vector{Float64}}
    drive_grad_n::Base.RefValue{Int}
    drive_grad_cfg::Base.RefValue{Any}
    drive_grad_result::Base.RefValue{Vector{Float64}}
end

function Tsit5DiscAdjWorkspace(M::Integer)
    N = state_length_1st_order(M)
    nR = real_state_length_1st_order(M)
    k = ntuple(_ -> Vector{ComplexF64}(undef, N), 6)
    y = ntuple(_ -> Vector{ComplexF64}(undef, N), 6)
    u_np1 = Vector{ComplexF64}(undef, N)
    x = Vector{Float64}(undef, nR)
    x̄ = Vector{Float64}(undef, nR)
    λu = Vector{Float64}(undef, nR)
    λy = Vector{Float64}(undef, nR)
    λk = ntuple(_ -> Vector{Float64}(undef, nR), 6)
    return Tsit5DiscAdjWorkspace(k, y, u_np1, x, x̄, λu, λy, λk,
                                  Ref(-1), Ref{Any}(nothing), Ref(Float64[]))
end

function _tsit5_adj_workspace(M::Integer)
    tls = task_local_storage()
    cache = get!(tls, :InhomogeneousSpinCavityDynamics_tsit5_adj_ws_cache) do
        Dict{Int,Tsit5DiscAdjWorkspace}()
    end::Dict{Int,Tsit5DiscAdjWorkspace}
    return get!(() -> Tsit5DiscAdjWorkspace(Int(M)), cache, Int(M))
end

@inline function _tsit5_stage_time(tab::Tsit5DiscAdjTableau, t, dt, i::Int)
    i == 1 && return t
    i == 2 && return t + tab.c1 * dt
    i == 3 && return t + tab.c2 * dt
    i == 4 && return t + tab.c3 * dt
    i == 5 && return t + tab.c4 * dt
    return t + dt
end

function _tsit5_combine_stage!(y, u, k, tab::Tsit5DiscAdjTableau, dt, i::Int)
    n = length(u)
    if i == 1
        @inbounds for j in 1:n
            y[j] = u[j]
        end
        return y
    end
    @inbounds for j in 1:n
        acc = k[1][j] * (i == 2 ? tab.a21 :
                         i == 3 ? tab.a31 :
                         i == 4 ? tab.a41 :
                         i == 5 ? tab.a51 : tab.a61)
        if i >= 3
            acc += k[2][j] * (i == 3 ? tab.a32 :
                              i == 4 ? tab.a42 :
                              i == 5 ? tab.a52 : tab.a62)
        end
        if i >= 4
            acc += k[3][j] * (i == 4 ? tab.a43 :
                              i == 5 ? tab.a53 : tab.a63)
        end
        if i >= 5
            acc += k[4][j] * (i == 5 ? tab.a54 : tab.a64)
        end
        if i >= 6
            acc += k[5][j] * tab.a65
        end
        y[j] = u[j] + dt * acc
    end
    return y
end

function _tsit5_combine_solution!(u_np1, u, k, tab::Tsit5DiscAdjTableau, dt)
    n = length(u)
    @inbounds for j in 1:n
        u_np1[j] = u[j] + dt * (
            tab.b1 * k[1][j] + tab.b2 * k[2][j] + tab.b3 * k[3][j] +
            tab.b4 * k[4][j] + tab.b5 * k[5][j] + tab.b6 * k[6][j]
        )
    end
    return u_np1
end

function tsit5_forced_step(u::AbstractVector, p, t, dt, tab::Tsit5DiscAdjTableau=TSIT5_DISC_ADJ_TAB)
    k = ntuple(_ -> similar(u), 6)
    y = similar(u)
    u_np1 = similar(u)
    rhs_1st_order!(k[1], u, p, t)
    for i in 2:6
        _tsit5_combine_stage!(y, u, k, tab, dt, i)
        rhs_1st_order!(k[i], y, p, _tsit5_stage_time(tab, t, dt, i))
    end
    _tsit5_combine_solution!(u_np1, u, k, tab, dt)
    return u_np1
end

function tsit5_forced_step!(u_np1, k, y_scratch, u, p, t, dt,
                           tab::Tsit5DiscAdjTableau=TSIT5_DISC_ADJ_TAB)
    rhs_1st_order!(k[1], u, p, t)
    copyto!(y_scratch[1], u)
    for i in 2:6
        _tsit5_combine_stage!(y_scratch[i], u, k, tab, dt, i)
        rhs_1st_order!(k[i], y_scratch[i], p, _tsit5_stage_time(tab, t, dt, i))
    end
    _tsit5_combine_solution!(u_np1, u, k, tab, dt)
    return u_np1
end

function _accumulate_drive_grad!(gθ::AbstractVector, λx::AbstractVector, t,
                                 pulse::CompositePulse, u_pulse::AbstractVector, sqrt_κe,
                                 ws::Tsit5DiscAdjWorkspace)
    λar = λx[1]
    λai = λx[2]
    (λar == 0 && λai == 0) && return nothing
    n = length(u_pulse)
    function s(uu)
        E = build_E_of_t(pulse, uu)(t)
        return sqrt_κe * (λar * real(E) + λai * imag(E))
    end
    if ws.drive_grad_n[] != n
        chunk = ForwardDiff.Chunk{min(60, n)}()
        ws.drive_grad_cfg[] = ForwardDiff.GradientConfig(s, u_pulse, chunk)
        ws.drive_grad_result[] = Vector{Float64}(undef, n)
        ws.drive_grad_n[] = n
    end
    cfg = ws.drive_grad_cfg[]
    result = ws.drive_grad_result[]
    ForwardDiff.gradient!(result, s, u_pulse, cfg)
    gθ .+= result
    return nothing
end

function tsit5_step_vjp!(
    λ_n::AbstractVector,
    gθ::AbstractVector,
    λ_np1::AbstractVector,
    u_n::AbstractVector,
    t, dt, p,
    pulse::CompositePulse,
    u_pulse::AbstractVector,
    ws::Tsit5DiscAdjWorkspace,
    tab::Tsit5DiscAdjTableau=TSIT5_DISC_ADJ_TAB,
)
    M = p[6]
    sqrt_κe = sqrt(real(p[2]))
    nR = length(λ_np1)

    tsit5_forced_step!(ws.u_np1, ws.k, ws.y, u_n, p, t, dt, tab)

    @inbounds for i in 1:nR
        ws.λu[i] = λ_np1[i]
    end
    @inbounds for s in 1:6
        b = s == 1 ? tab.b1 : s == 2 ? tab.b2 : s == 3 ? tab.b3 :
            s == 4 ? tab.b4 : s == 5 ? tab.b5 : tab.b6
        λks = ws.λk[s]
        @inbounds for i in 1:nR
            λks[i] = (dt * b) * λ_np1[i]
        end
    end

    for s in 6:-1:1
        pack_state_real!(ws.x, ws.y[s], M)
        rhs_1st_order_vjp!(ws.λy, ws.λk[s], ws.x, p, _tsit5_stage_time(tab, t, dt, s))
        _accumulate_drive_grad!(gθ, ws.λk[s], _tsit5_stage_time(tab, t, dt, s),
                                pulse, u_pulse, sqrt_κe, ws)
        @inbounds for i in 1:nR
            ws.λu[i] += ws.λy[i]
        end
        if s >= 2
            @inbounds for j in 1:(s - 1)
                aij = _tsit5_a(tab, s, j)
                λkj = ws.λk[j]
                @inbounds for i in 1:nR
                    λkj[i] += (dt * aij) * ws.λy[i]
                end
            end
        end
    end
    copyto!(λ_n, ws.λu)
    return λ_n
end

@inline function _tsit5_a(tab::Tsit5DiscAdjTableau, i::Int, j::Int)
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

function _host_ode_p(d, E_of_t)
    M = Int(d.M)
    all(iszero, imag.(d.delta_b)) || error(
        "_host_ode_p: d.delta_b has nonzero imaginary part(s) -- the adjoint " *
        "backend (pulse_cost_grad_adjoint) only supports real detunings; " *
        "grad_mode=:forwarddiff/threaded_grad=true handle complex delta_b correctly."
    )
    all(iszero, imag.(d.g_b)) || error(
        "_host_ode_p: d.g_b has nonzero imaginary part(s) -- the adjoint " *
        "backend (pulse_cost_grad_adjoint) only supports real couplings; " *
        "grad_mode=:forwarddiff/threaded_grad=true handle complex g_b correctly."
    )
    delta_b = collect(Float64, real.(d.delta_b))
    g_b = collect(Float64, real.(d.g_b))
    return (Float64(d.delta0), Float64(d.kappa_e), Float64(d.kappa_i), delta_b, g_b, M, E_of_t)
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

function record_adaptive_tsit5_mesh(
    u0::AbstractVector,
    p,
    tspan;
    reltol=1e-8,
    abstol=1e-8,
    tstops=Float64[],
    checkpoint_stride::Integer=typemax(Int),
    record_full_u::Bool=true,
    alg=Tsit5(),
)
    eltype(u0) <: ForwardDiff.Dual && error(
        "record_adaptive_tsit5_mesh: adjoint primal must be ComplexF64, not Dual."
    )
    nameof(typeof(alg)) === :Tsit5 || error(
        "record_adaptive_tsit5_mesh only supports Tsit5, got $(typeof(alg))."
    )
    stride = Int(checkpoint_stride)
    stride < 1 && error("checkpoint_stride must be >= 1, got $stride.")
    u0c = ComplexF64.(u0)

    if record_full_u
        prob = ODEProblem(rhs_1st_order!, u0c, tspan, p)
        sol = solve(prob, Tsit5(); reltol=reltol, abstol=abstol, tstops=tstops,
                    save_everystep=true, save_start=true, dense=false)
        _successful_solve(sol) || throw(PulseSolveFailed(sol.retcode))
        t = collect(Float64, sol.t)
        length(t) >= 2 || error("record_adaptive_tsit5_mesh: solver returned fewer than 2 nodes.")
        dt = diff(t)
        u = Vector{Vector{ComplexF64}}(undef, length(t))
        @inbounds for i in eachindex(t)
            u[i] = ComplexF64.(Array(sol.u[i]))
        end
        n = length(u)
        idxs = _checkpoint_indices(n, stride)
        stack = HostCheckpointStack(t[idxs], [copy(u[i]) for i in idxs], idxs, stride)
        mesh = FrozenTsit5Mesh(t, dt, u)
        return mesh, stack, copy(u[end])
    end

    t_log = Float64[Float64(tspan[1])]
    u_stack = Vector{ComplexF64}[ComplexF64.(u0c)]
    idx_stack = Int[1]
    step = Ref(0)
    function record!(u, t, _integrator)
        step[] += 1
        n = step[] + 1
        push!(t_log, Float64(t))
        if (n - 1) % stride == 0
            push!(u_stack, ComplexF64.(u))
            push!(idx_stack, n)
        end
        return nothing
    end
    cb = FunctionCallingCallback(record!; func_everystep=true, func_start=false)
    prob = ODEProblem(rhs_1st_order!, u0c, tspan, p)
    sol = solve(prob, Tsit5(); reltol=reltol, abstol=abstol, tstops=tstops,
                save_everystep=false, save_start=false, save_end=true, dense=false,
                callback=cb)
    _successful_solve(sol) || throw(PulseSolveFailed(sol.retcode))
    n = step[] + 1
    n >= 2 || error("record_adaptive_tsit5_mesh: solver returned fewer than 2 nodes.")
    if idx_stack[end] != n
        push!(u_stack, ComplexF64.(Array(sol.u[end])))
        push!(idx_stack, n)
    end
    t = t_log
    dt = diff(t)
    stack = HostCheckpointStack(t[idx_stack], u_stack, idx_stack, stride)
    mesh = FrozenTsit5Mesh(t, dt, Vector{ComplexF64}[])
    return mesh, stack, copy(u_stack[end])
end

function replay_tsit5_window(u_start::AbstractVector, p, t_start, dts::AbstractVector,
                             tab::Tsit5DiscAdjTableau=TSIT5_DISC_ADJ_TAB)
    M = p[6]
    N = state_length_1st_order(M)
    k = ntuple(_ -> Vector{ComplexF64}(undef, N), 6)
    y = ntuple(_ -> Vector{ComplexF64}(undef, N), 6)
    u = ComplexF64.(u_start)
    us = Vector{Vector{ComplexF64}}(undef, length(dts) + 1)
    us[1] = copy(u)
    t = Float64(t_start)
    u_np1 = Vector{ComplexF64}(undef, N)
    @inbounds for n in eachindex(dts)
        Δt = Float64(dts[n])
        tsit5_forced_step!(u_np1, k, y, u, p, t, Δt, tab)
        copyto!(u, u_np1)
        t += Δt
        us[n + 1] = copy(u)
    end
    return us
end
