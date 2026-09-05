

# Transverse seed for :weak / :weak_inverted. 1e-3 is outside the Bloch
# ball |⟨S⟩| = Nj/2 by design — it is a numerical seed, not a CSS.
const _WEAK_SEED = 1.0e-3


function build_u0_1st_order_cpu(M::Integer, Nj::AbstractVector, ::Type{T},
    initial_condition::Symbol=:ground) where {T}
    u0 = zeros(Complex{T}, state_length_1st_order(M))
    sp = IDX1_Sp_start:idx1_Sz_start(M)-1
    sz = idx1_Sz_start(M):state_length_1st_order(M)
    if initial_condition == :ground

        u0[sz] .= .-Nj ./ 2
    elseif initial_condition == :inverted

        u0[sz] .= Nj ./ 2
    elseif initial_condition == :equator



        u0[sp] .= Nj ./ 2
    elseif initial_condition == :weak











        u0[sz] .= .-Nj ./ 2
        u0[sp] .= T(_WEAK_SEED) .* Nj ./ 2
    elseif initial_condition == :weak_inverted


        u0[sz] .= Nj ./ 2
        u0[sp] .= T(_WEAK_SEED) .* Nj ./ 2
    elseif initial_condition == :custom

    else
        error("Unknown initial_condition = $(initial_condition). Use :ground, :inverted, " *
              ":equator, :weak, :weak_inverted, or :custom.")
    end
    return u0
end


_zero_drive(t) = zero(ComplexF64)

struct PulseSolveFailed <: Exception
    retcode
end

function Base.showerror(io::IO, e::PulseSolveFailed)
    print(io, "1st-order pulse ODE solve failed (retcode=$(e.retcode))")
end

function _successful_solve(sol)
    rc = sol.retcode
    name = rc isa Symbol ? rc : Symbol(string(rc))
    return name === :Success || name === :Terminated
end


const _PULSE_MAX_GPUS = 8
const _PULSE_GPU_MIN_M = 256
const _GPU_DUAL_OK = Ref{Union{Nothing,Bool}}(nothing)
const _PULSE_GPU_LOGGED = Threads.Atomic{Bool}(false)

function _cuda_mod()
    return isdefined(@__MODULE__, :CUDA) ? getfield(@__MODULE__, :CUDA) : nothing
end


function pulse_gpu_count()
    C = _cuda_mod()
    C === nothing && return 0
    try
        C.functional() || return 0
        n = length(C.devices())
        return n < 1 ? 0 : min(_PULSE_MAX_GPUS, n)
    catch
        return 0
    end
end

function _pulse_gpu_devices()
    n = pulse_gpu_count()
    n == 0 && return Any[]
    C = _cuda_mod()
    return collect(C.devices())[1:n]
end

function _resolve_compute(compute::Symbol, M::Integer)
    compute === :cpu && return :cpu
    (compute === :gpu || compute === :auto) || error(
        "compute must be :auto, :cpu, or :gpu, got $(repr(compute))."
    )
    n = pulse_gpu_count()
    if n == 0
        compute === :gpu && error(
            "compute=:gpu requested but no functional CUDA device was found."
        )
        return :cpu
    end
    compute === :auto && M < _PULSE_GPU_MIN_M && return :cpu
    return :gpu
end

function _ode_workspace_bytes(M::Integer, ::Type{T}) where {T}
    return state_length_1st_order(M) * sizeof(T) * 12
end

function _gpu_free_bytes()
    C = _cuda_mod()
    C === nothing && return 0
    try
        return Int(C.free_memory())
    catch
        return 0
    end
end

function _reclaim_current_gpu_memory()
    C = _cuda_mod()
    C === nothing && return nothing
    try
        C.functional() || return nothing
        GC.gc(false)
        C.reclaim()
    catch
    end
    return nothing
end


function _reclaim_gpu_memory()
    C = _cuda_mod()
    C === nothing && return nothing
    try
        C.functional() || return nothing
        GC.gc(false)
        devices = _pulse_gpu_devices()
        if isempty(devices)
            C.reclaim()
            return nothing
        end
        prev = C.device()
        try
            for dev in devices
                C.device!(dev)
                C.reclaim()
            end
        finally
            C.device!(prev)
        end
    catch
    end
    return nothing
end

function _log_pulse_compute_once(msg::AbstractString)
    if !Threads.atomic_cas!(_PULSE_GPU_LOGGED, false, true)
        println(msg)
    end
    return nothing
end

function _maybe_cuarray(x, compute::Symbol)
    compute === :gpu || return x
    C = _cuda_mod()
    C === nothing && error("internal: compute=:gpu but CUDA is not loaded.")
    return C.CuArray(x)
end

function _allowscalar_solve(compute::Symbol, f)
    compute === :gpu || return f()
    C = _cuda_mod()
    return C.allowscalar() do
        f()
    end
end

function _assert_ensemble_shapes(d)
    hasproperty(d, :M) || error("derived ensemble `d` is missing field M.")
    M = Int(d.M)
    M >= 1 || error("ensemble size M must be >= 1, got $M.")
    hasproperty(d, :Nj) && hasproperty(d, :delta_b) && hasproperty(d, :g_b) || error(
        "derived ensemble `d` must provide Nj, delta_b, and g_b."
    )
    length(d.Nj) == M || error("Nj length $(length(d.Nj)) != M=$M.")
    length(d.delta_b) == M || error("delta_b length $(length(d.delta_b)) != M=$M.")
    length(d.g_b) == M || error("g_b length $(length(d.g_b)) != M=$M.")
    hasproperty(d, :timespan) || error("derived ensemble `d` is missing timespan.")
    length(d.timespan) == 2 || error("timespan must have length 2, got $(length(d.timespan)).")
    return M
end

function _assert_state_shapes(Sp, Sz, Nj, M::Integer, fname::AbstractString)
    length(Sp) == M || error("$fname: Sp length $(length(Sp)) != M=$M.")
    length(Sz) == M || error("$fname: Sz length $(length(Sz)) != M=$M.")
    length(Nj) == M || error("$fname: Nj length $(length(Nj)) != M=$M.")
    return nothing
end


function _solve_1st_order_ode(
    u0, p, timespan;
    alg=Tsit5(), reltol=1e-8, abstol=1e-8, tstops=Float64[],
    save_mode::Symbol=:final, t_save=nothing, compute::Symbol=:cpu,
)
    (save_mode === :final || save_mode === :trajectory) || error(
        "save_mode must be :final or :trajectory, got $(repr(save_mode))."
    )
    prob = ODEProblem(rhs_1st_order!, u0, timespan, p)
    sol = _allowscalar_solve(compute, () -> begin
        if save_mode === :final
            solve(prob, alg; reltol=reltol, abstol=abstol, tstops=tstops,
                  save_everystep=false, save_start=false)
        else
            t_save === nothing && error("save_mode=:trajectory requires t_save.")
            solve(prob, alg; reltol=reltol, abstol=abstol, tstops=tstops, saveat=t_save)
        end
    end)
    _successful_solve(sol) || throw(PulseSolveFailed(sol.retcode))
    return sol
end


function _run_sim_1st_order_from_u0(
    u0_host::AbstractVector, E_of_t, d;
    alg=Tsit5(), reltol=1e-8, abstol=1e-8, tstops=Float64[],
    save_mode::Symbol=:final, t_save=nothing, compute::Symbol=:cpu,
    frame::Symbol=:lab,
)
    M = _assert_ensemble_shapes(d)
    length(u0_host) == state_length_1st_order(M) || error(
        "u0 has length $(length(u0_host)), expected state_length_1st_order($M) = $(state_length_1st_order(M))."
    )
    (frame === :lab || frame === :ip) || error("frame must be :lab or :ip, got $(repr(frame)).")
    if frame === :ip && compute === :gpu
        error("frame=:ip is CPU-only; pass compute=:cpu (or :auto).")
    end

    u0 = _maybe_cuarray(u0_host, compute)
    delta_b_r = collect(d.delta_b)
    delta_b = _maybe_cuarray(delta_b_r, compute)
    g_b = _maybe_cuarray(collect(d.g_b), compute)
    p = frame === :ip ?
        (d.delta0, d.kappa_e, d.kappa_i, delta_b, g_b, M, E_of_t, :ip) :
        (d.delta0, d.kappa_e, d.kappa_i, delta_b, g_b, M, E_of_t)
    tfinal = Float64(d.timespan[2])
    sol = nothing
    try
        sol = _solve_1st_order_ode(
            u0, p, d.timespan;
            alg=alg, reltol=reltol, abstol=abstol, tstops=tstops,
            save_mode=save_mode, t_save=t_save, compute=compute,
        )
        if save_mode === :final
            u_end = Array(sol.u[end])
            a, Sp, Sz = unpack_state_1st_order_u(u_end, M)
            Sp = collect(Sp)

            frame === :ip && ip_spins_to_lab!(Sp, delta_b_r, tfinal)
            Sz = collect(Sz)
            _assert_state_shapes(Sp, Sz, d.Nj, M, "_run_sim_1st_order_from_u0")
            return a, Sp, Sz
        end
        Nt = length(sol.t)
        a = Vector{eltype(sol.u[1])}(undef, Nt)
        Sp = Matrix{eltype(sol.u[1])}(undef, Nt, M)
        Sz = Matrix{eltype(sol.u[1])}(undef, Nt, M)
        @inbounds for i in 1:Nt
            ui = Array(sol.u[i])
            ai, Spi, Szi = unpack_state_1st_order_u(ui, M)
            Spi = collect(Spi)
            frame === :ip && ip_spins_to_lab!(Spi, delta_b_r, Float64(sol.t[i]))
            a[i] = ai
            Sp[i, :] .= Spi
            Sz[i, :] .= Szi
        end
        return sol.t, a, Sp, Sz
    finally
        u0 = nothing
        delta_b = nothing
        g_b = nothing
        p = nothing
        sol = nothing



        if compute === :gpu && !(eltype(u0_host) <: ForwardDiff.Dual)
            _reclaim_current_gpu_memory()
        end
    end
end

function _split_index_ranges(n::Integer, nparts::Integer)
    n >= 1 || error("n must be >= 1, got $n.")
    nparts = max(1, min(Int(nparts), n))
    base, rem = divrem(n, nparts)
    ranges = UnitRange{Int}[]
    start = 1
    for p in 1:nparts
        len = base + (p <= rem ? 1 : 0)
        push!(ranges, start:(start + len - 1))
        start += len
    end
    start == n + 1 || error("internal: index split of 1:$n into $nparts parts did not cover the range.")
    return ranges
end

function _run_pulse_jobs!(jobs, f)
    njob = length(jobs)
    njob == 0 && return nothing
    devices = _pulse_gpu_devices()
    if isempty(devices)
        Threads.@threads for i in 1:njob
            f(jobs[i], nothing)
        end
        return nothing
    end
    q = Channel{Any}(length(devices))
    for dev in devices
        put!(q, dev)
    end
    @sync for job in jobs
        Threads.@spawn begin
            dev = take!(q)
            try
                C = _cuda_mod()
                C.device!(dev)
                f(job, dev)
            finally
                put!(q, dev)
            end
        end
    end
    return nothing
end


function run_sim_1st_order_pure(
    u::AbstractVector,
    pulse::CompositePulse,
    d;
    signal_E_of_t = _zero_drive,
    initial_condition::Symbol=:ground,
    alg=Tsit5(),
    reltol=1e-8,
    abstol=1e-8,
    compute::Symbol=:auto,
    frame::Symbol=:lab,
)
    M = _assert_ensemble_shapes(d)
    length(u) == n_params(pulse) || error(
        "run_sim_1st_order_pure: u has length $(length(u)), but this CompositePulse " *
        "(k=$(pulse.k), n_coeff_A=$(pulse.n_coeff_A), n_coeff_f=$(pulse.n_coeff_f)) needs $(n_params(pulse))."
    )
    (frame === :lab || frame === :ip) || error("frame must be :lab or :ip, got $(repr(frame)).")
    T = eltype(u)
    compute_eff = _resolve_compute(compute, M)

    frame === :ip && (compute_eff = :cpu)
    if compute_eff === :gpu && T <: ForwardDiff.Dual && _GPU_DUAL_OK[] === false
        compute_eff = :cpu
    end
    if compute_eff === :gpu
        need = _ode_workspace_bytes(M, Complex{T})
        free = _gpu_free_bytes()
        if free > 0 && need > free ÷ 2
            compute === :gpu && error(
                "GPU 1st-order solve needs ~$need bytes of Dual/state workspace " *
                "but only $free bytes are free on the current device."
            )
            compute_eff = :cpu
        end
    end
    if compute_eff === :gpu
        _log_pulse_compute_once(
            "pulse ODE: GPU backend on $(pulse_gpu_count()) device(s) " *
            "(M=$M, eltype=$T, rhs_1st_order!/Tsit5 unchanged)"
        )
    end

    control_E_of_t = build_E_of_t(pulse, u)
    E_of_t(t) = control_E_of_t(t) + signal_E_of_t(t)
    u0 = build_u0_1st_order_cpu(M, d.Nj, T, initial_condition)























    t_start, t_end, _, _, _ = decode(pulse, u)
    tstops = ForwardDiff.value.(vcat(t_start, t_end))

    try
        a, Sp, Sz = _run_sim_1st_order_from_u0(
            u0, E_of_t, d;
            alg=alg, reltol=reltol, abstol=abstol, tstops=tstops,
            save_mode=:final, compute=compute_eff, frame=frame,
        )
        if compute_eff === :gpu && T <: ForwardDiff.Dual
            _GPU_DUAL_OK[] = true
        end
        return a, Sp, Sz, d.Nj
    catch e
        e isa PulseSolveFailed && rethrow()
        if compute_eff === :gpu && T <: ForwardDiff.Dual && _GPU_DUAL_OK[] !== true
            _GPU_DUAL_OK[] = false
            _reclaim_current_gpu_memory()
            @warn "GPU Dual 1st-order ODE failed; falling back to the host Dual path" exception = e
            a, Sp, Sz = _run_sim_1st_order_from_u0(
                u0, E_of_t, d;
                alg=alg, reltol=reltol, abstol=abstol, tstops=tstops,
                save_mode=:final, compute=:cpu, frame=frame,
            )
            return a, Sp, Sz, d.Nj
        end
        rethrow()
    end
end

function _forbid_initial_condition(kwargs)
    :initial_condition in keys(kwargs) && error(
        "the cost fixes its own initial conditions (:ground and/or :weak, see `track`). " *
        "Do not pass initial_condition into pulse_cost / pulse_metrics / optimise_composite_pulse."
    )
    return nothing
end


function _assert_track(track::Symbol)
    track === :weak || track === :dual || error(
        "track must be :weak (the default -- a single :weak solve, inversion " *
        "read from it too) or :dual (opt-in -- separate :ground + :weak " *
        "solves), got $(repr(track))."
    )
    return track
end

function _solver_kwargs(kwargs)
    nt = NamedTuple(kwargs)
    for k in (:compute, :threaded_grad, :grad_mode, :checkpoint_stride, :use_checkpoints)
        nt = haskey(nt, k) ? Base.structdiff(nt, NamedTuple{(k,)}) : nt
    end
    return nt
end


function _weighted_inversion(Sz, g_b, Nj, ::Type{T}) where {T}
    length(Sz) == length(Nj) == length(g_b) || error(
        "_weighted_inversion: Sz/g_b/Nj lengths $(length(Sz))/$(length(g_b))/$(length(Nj)) must match."
    )
    w = Nj .* abs2.(g_b)
    weight = w ./ (sum(w) + T(1e-30))
    Sz_fraction = real.(Sz) ./ (Nj ./ 2 .+ 1e-30)
    Ij = clamp.((Sz_fraction .+ 1) ./ 2, zero(T), one(T))
    return sum(weight .* Ij)
end


function _weak_seed_retention(Sp::AbstractVector, g_b::AbstractVector, Nj::AbstractVector,
        delta_b::AbstractVector, ::Type{T}; eps_seed::Real=_WEAK_SEED) where {T}
    length(Sp) == length(g_b) == length(Nj) == length(delta_b) || error(
        "_weak_seed_retention: Sp/g_b/Nj/delta_b lengths " *
        "$(length(Sp))/$(length(g_b))/$(length(Nj))/$(length(delta_b)) must match."
    )
    eps_seed > 0 || error("_weak_seed_retention: eps_seed must be > 0, got $eps_seed.")
    slices = _frequency_slice_indices(delta_b)
    num_acc = zero(T)
    den_acc = zero(T)
    for idx in slices
        wg = abs2.(g_b[idx])
        C_num = sum(wg .* sqrt.(abs2.(Sp[idx]) .+ 1e-30))
        C_den = sum(wg .* (convert(T, eps_seed) .* (Nj[idx] ./ 2)))
        n_omega = sum(Nj[idx] .* wg)
        num_acc += n_omega * (C_num / (C_den + 1e-30))
        den_acc += n_omega
    end
    return num_acc / (den_acc + 1e-30)
end


function _weighted_coherence(Sp::AbstractVector, g_b::AbstractVector, Nj::AbstractVector,
        delta_b::AbstractVector, ::Type{T}; eps_seed::Real=_WEAK_SEED) where {T}
    return clamp(_weak_seed_retention(Sp, g_b, Nj, delta_b, T; eps_seed=eps_seed),
                 zero(T), one(T))
end


function _frequency_slice_indices(delta_b::AbstractVector)
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


function _weighted_silencing_factor(Sp::AbstractVector, g_b::AbstractVector, Nj::AbstractVector,
        delta_b::AbstractVector, ::Type{T}; eps_seed::Real=_WEAK_SEED) where {T}
    length(Sp) == length(g_b) == length(Nj) == length(delta_b) || error(
        "_weighted_silencing_factor: Sp/g_b/Nj/delta_b lengths " *
        "$(length(Sp))/$(length(g_b))/$(length(Nj))/$(length(delta_b)) must match."
    )
    eps_seed > 0 || error("_weighted_silencing_factor: eps_seed must be > 0, got $eps_seed.")
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


function _weighted_field_amplitude(Sp::AbstractVector, g_b::AbstractVector, Nj::AbstractVector, ::Type{T};
        eps_seed::Real=_WEAK_SEED) where {T}
    length(Sp) == length(g_b) == length(Nj) || error(
        "_weighted_field_amplitude: Sp/g_b/Nj lengths $(length(Sp))/$(length(g_b))/$(length(Nj)) must match."
    )
    max_field_sum = convert(T, eps_seed) * sum(abs.(g_b) .* Nj) / 2
    E_complex = sum(g_b .* Sp) / (max_field_sum + 1e-30)
    abs_E = sqrt(abs2(E_complex) + 1e-30)
    return clamp(abs_E, zero(T), one(T))
end


function pulse_metrics(u::AbstractVector, pulse::CompositePulse, d;
                        compute::Symbol=:auto, track::Symbol=:weak, kwargs...)
    _forbid_initial_condition(kwargs)
    _assert_track(track)
    T = eltype(u)
    sk = _solver_kwargs(kwargs)
    if track === :dual
        _, _, Sz, Nj = run_sim_1st_order_pure(u, pulse, d; compute=compute, sk..., initial_condition=:ground)
        inversion = _weighted_inversion(Sz, d.g_b, Nj, T)
        _, Sp, _, Nj_eq = run_sim_1st_order_pure(u, pulse, d; compute=compute, sk..., initial_condition=:weak)
    else
        _, Sp, Sz_w, Nj_eq = run_sim_1st_order_pure(u, pulse, d; compute=compute, sk..., initial_condition=:weak)
        inversion = _weighted_inversion(Sz_w, d.g_b, Nj_eq, T)
    end
    silencing = _weighted_silencing_factor(Sp, d.g_b, Nj_eq, d.delta_b, T)
    coherence = _weighted_coherence(Sp, d.g_b, Nj_eq, d.delta_b, T)
    field_amp = _weighted_field_amplitude(Sp, d.g_b, Nj_eq, T)
    weak_seed_retention = _weak_seed_retention(Sp, d.g_b, Nj_eq, d.delta_b, T)
    return inversion, silencing, coherence, field_amp, weak_seed_retention
end



const _DEFAULT_PENALTY_MIN = 0.85
const _DEFAULT_PENALTY_KAPPA = 50.0


function _assert_penalty_params(I_min, kappa_I, S_min, kappa_S)
    (isfinite(kappa_I) && kappa_I >= 0) || error(
        "_assert_penalty_params: kappa_I must be finite and >= 0, got $kappa_I " *
        "(negative kappa_I would REWARD low inversion instead of penalising it)."
    )
    (isfinite(kappa_S) && kappa_S >= 0) || error(
        "_assert_penalty_params: kappa_S must be finite and >= 0, got $kappa_S " *
        "(negative kappa_S would REWARD low silencing_success instead of penalising it)."
    )
    (isfinite(I_min) && 0 <= I_min <= 1) || error(
        "_assert_penalty_params: I_min must be finite and in [0,1] (inversion's own " *
        "provable range), got $I_min -- a value > 1 can never be satisfied, leaving " *
        "the penalty permanently active even at inversion=1.0."
    )
    (isfinite(S_min) && 0 <= S_min <= 1) || error(
        "_assert_penalty_params: S_min must be finite and in [0,1] (silencing_success's " *
        "own provable range for target_F ∈ [0,1]), got $S_min -- a value > 1 can never " *
        "be satisfied, leaving the penalty permanently active even at silencing_success=1.0."
    )
end


function _fidelity_physics_cost(inversion::T, silencing::T, target_F, I_min, kappa_I, S_min, kappa_S) where {T}
    _assert_penalty_params(I_min, kappa_I, S_min, kappa_S)
    ss = one(T) - (silencing - convert(T, target_F))^2
    fid = inversion * ss
    J_base = (one(T) - fid)^2

    pen_I_val = inversion < convert(T, I_min) ? (convert(T, I_min) - inversion) : zero(T)
    pen_S_val = ss < convert(T, S_min) ? (convert(T, S_min) - ss) : zero(T)
    J_pen = convert(T, 0.5 * kappa_I) * pen_I_val^2 + convert(T, 0.5 * kappa_S) * pen_S_val^2

    return J_base + J_pen, fid, ss
end


function _fidelity_gradient_coefficients(inversion, silencing_success, fidelity_phys, I_min, kappa_I, S_min, kappa_S)
    _assert_penalty_params(I_min, kappa_I, S_min, kappa_S)
    base_coeff_I = -2.0 * (1.0 - fidelity_phys) * silencing_success
    base_coeff_S = -2.0 * (1.0 - fidelity_phys) * inversion

    pen_coeff_I = inversion < I_min ? -Float64(kappa_I) * (Float64(I_min) - inversion) : 0.0
    pen_coeff_S = silencing_success < S_min ? -Float64(kappa_S) * (Float64(S_min) - silencing_success) : 0.0

    return base_coeff_I + pen_coeff_I, base_coeff_S + pen_coeff_S
end


function pulse_cost(u::AbstractVector, pulse::CompositePulse, d;
                     target_F=1.0, w_time=0.15, w_power=0.05, w_tmax=1.0,
                     I_min::Real=_DEFAULT_PENALTY_MIN, kappa_I::Real=_DEFAULT_PENALTY_KAPPA,
                     S_min::Real=_DEFAULT_PENALTY_MIN, kappa_S::Real=_DEFAULT_PENALTY_KAPPA,
                     compute::Symbol=:auto, track::Symbol=:weak, kwargs...)
    _forbid_initial_condition(kwargs)
    _assert_track(track)
    T = eltype(u)
    sk = _solver_kwargs(kwargs)
    duration = pulse_duration(pulse, u)
    _, t_end, _, cA, _ = decode(pulse, u)

    tmax_excess = max(t_end[end] - pulse.T_max, zero(T))
    tmax_penalty = w_tmax * (tmax_excess / pulse.T_max)^2

    n_cA = length(cA)
    n_cA > 0 || error("pulse_cost: decoded cA is empty.")
    normalized_cA = cA ./ pulse.amp_scale
    power_penalty = w_power * (sum(abs2, normalized_cA) / n_cA)

    inversion = zero(T)
    silencing = zero(T)
    coherence = zero(T)
    field_amp = zero(T)
    weak_seed_retention = zero(T)

    try
        if track === :dual

            _, _, Sz, Nj = run_sim_1st_order_pure(u, pulse, d; compute=compute, sk..., initial_condition=:ground)
            inversion = _weighted_inversion(Sz, d.g_b, Nj, T)
            _, Sp, _, Nj_eq = run_sim_1st_order_pure(u, pulse, d; compute=compute, sk..., initial_condition=:weak)
        else

            _, Sp, Sz_w, Nj_eq = run_sim_1st_order_pure(u, pulse, d; compute=compute, sk..., initial_condition=:weak)
            inversion = _weighted_inversion(Sz_w, d.g_b, Nj_eq, T)
        end
        silencing = _weighted_silencing_factor(Sp, d.g_b, Nj_eq, d.delta_b, T)
        coherence = _weighted_coherence(Sp, d.g_b, Nj_eq, d.delta_b, T)
        field_amp = _weighted_field_amplitude(Sp, d.g_b, Nj_eq, T)
        weak_seed_retention = _weak_seed_retention(Sp, d.g_b, Nj_eq, d.delta_b, T)
    catch e
        e isa PulseSolveFailed || rethrow()
        infT = convert(T, Inf)
        nanT = convert(T, NaN)
        return infT, nanT, nanT, duration, nanT, nanT, nanT
    end

    physics_cost, _, _ = _fidelity_physics_cost(inversion, silencing, target_F, I_min, kappa_I, S_min, kappa_S)

    cost = physics_cost + w_time * (duration / pulse.T_max) + tmax_penalty + power_penalty
    return cost, inversion, silencing, duration, coherence, field_amp, weak_seed_retention
end


mutable struct AdamState
    m::Vector{Float64}
    v::Vector{Float64}
    t::Int
end

AdamState(n::Integer) = AdamState(zeros(n), zeros(n), 0)


function adam_step!(u::AbstractVector, grad::AbstractVector, state::AdamState;
                     lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, lr_scale::Union{AbstractVector,Nothing}=nothing)
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



function fit_composite_pulse(
    pulse::CompositePulse, E_target;
    N_fit::Integer=4000, num_epochs::Integer=1000,
    learning_rate::Real=0.002, seed::Integer=42,
    u_init::Union{AbstractVector,Nothing}=nothing,
)
    t_grid = range(0.0, pulse.T_max; length=N_fit)
    Ex, Ep = sample_E_of_t(E_target, pulse.T_max, N_fit)
    target = complex.(Ex, Ep)
    target_energy = sum(abs2, target) + 1e-30

    u = u_init === nothing ? initial_guess(pulse; seed=seed) : collect(Float64, u_init)
    length(u) == n_params(pulse) || error(
        "u_init has length $(length(u)), but this CompositePulse (k=$(pulse.k), " *
        "n_coeff_A=$(pulse.n_coeff_A), n_coeff_f=$(pulse.n_coeff_f)) needs $(n_params(pulse))."
    )
    n = length(u)
    adam = AdamState(n)
    history = NamedTuple[]

    function mse_only(uu)
        E_of_t = build_E_of_t(pulse, uu)
        s = zero(eltype(uu))
        @inbounds for i in eachindex(t_grid)
            s += abs2(E_of_t(t_grid[i]) - target[i])
        end
        return s / N_fit
    end

    best_u, best_mse = copy(u), Inf
    for epoch in 1:num_epochs
        grad = ForwardDiff.gradient(mse_only, u)
        mse = Float64(mse_only(u))
        if mse < best_mse
            best_mse, best_u = mse, copy(u)
        end
        push!(history, (epoch=epoch, mse=mse))
        adam_step!(u, grad, adam; lr=learning_rate)
    end

    rel_l2 = sqrt(best_mse * N_fit / target_energy)
    return best_u, (mse=best_mse, rel_l2=rel_l2, history=history)
end



function _instantaneous_frequency(t::AbstractVector, I::AbstractVector, Q::AbstractVector)
    n = length(t)
    n == length(I) == length(Q) || error(
        "t/I/Q must have the same length, got $(n)/$(length(I))/$(length(Q))."
    )
    phi = atan.(Q, I)
    @inbounds for j in 2:n
        d = phi[j] - phi[j-1]
        while d > pi
            phi[j] -= 2 * pi
            d = phi[j] - phi[j-1]
        end
        while d < -pi
            phi[j] += 2 * pi
            d = phi[j] - phi[j-1]
        end
    end
    f = Vector{Float64}(undef, n)
    f[1] = (phi[2] - phi[1]) / (t[2] - t[1])
    f[n] = (phi[n] - phi[n-1]) / (t[n] - t[n-1])
    @inbounds for j in 2:n-1
        f[j] = (phi[j+1] - phi[j-1]) / (t[j+1] - t[j-1])
    end
    return phi, f
end


function _detect_subpulse_segments(
    t::AbstractVector, A::AbstractVector;
    rel_thresh::Real=1e-3, min_active_samples::Integer=5, min_silence_samples::Integer=3,
)
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
            gap_len = j - j0
            if gap_len < min_silence_samples && j0 > 1 && j <= n
                active[j0:j-1] .= true
            end
        else
            j += 1
        end
    end

    segments = Tuple{Int,Int}[]
    j = 1
    while j <= n
        if active[j]
            j0 = j
            while j <= n && active[j]
                j += 1
            end
            if j - j0 >= min_active_samples
                push!(segments, (j0, j - 1))
            end
        else
            j += 1
        end
    end
    return segments
end


function _spline_coeff_count(n_samples::Integer; points_per_segment::Integer=22, degree::Integer=3)
    n_pieces = max(cld(n_samples, points_per_segment), 1)
    return max(n_pieces + degree, degree + 1)
end


function points_per_segment_for_budget(n_samples_max::Integer, k::Integer; degree::Integer=3, param_budget::Integer=60)
    k >= 1 || error("k must be a positive integer, got $k.")
    n_samples_max >= 1 || error("n_samples_max must be a positive integer, got $n_samples_max.")







    n_params_for(n_coeff) = 3 * k + 2 * k * n_coeff

    n_coeff_floor = degree + 1
    min_budget = n_params_for(n_coeff_floor)
    param_budget >= min_budget || error(
        "param_budget=$param_budget cannot be met for k=$k sub-pulses at degree=$degree: " *
        "even the minimum coefficient count per sub-pulse ($n_coeff_floor) already needs " *
        "$min_budget total parameters (3*k + 2*k*(degree+1)). Reduce k/degree, or raise param_budget."
    )

    n_coeff_max = (param_budget - 3 * k) ÷ (2 * k)
    n_pieces_max = n_coeff_max - degree
    pps = cld(n_samples_max, n_pieces_max)

    n_coeff_actual = _spline_coeff_count(n_samples_max; points_per_segment=pps, degree=degree)
    n_params_actual = n_params_for(n_coeff_actual)
    n_params_actual <= param_budget || error(
        "internal inconsistency: computed points_per_segment=$pps still gives " *
        "n_params=$n_params_actual > param_budget=$param_budget (n_coeff=$n_coeff_actual)."
    )

    return pps
end


function points_per_segment_for_budget(
    t::AbstractVector, I::AbstractVector, Q::AbstractVector;
    degree::Integer=3, param_budget::Integer=60,
    rel_thresh::Real=1e-3, min_active_samples::Integer=5, min_silence_samples::Integer=3,
)
    A = sqrt.(I .^ 2 .+ Q .^ 2)
    segments = _detect_subpulse_segments(
        t, A; rel_thresh=rel_thresh, min_active_samples=min_active_samples, min_silence_samples=min_silence_samples,
    )
    isempty(segments) && error(
        "No active sub-pulses detected (rel_thresh=$rel_thresh too high relative to the " *
        "trace's own peak, or the trace really is all silence)."
    )
    k = length(segments)
    n_samples_max = maximum(i_end - i_start + 1 for (i_start, i_end) in segments)
    pps = points_per_segment_for_budget(n_samples_max, k; degree=degree, param_budget=param_budget)
    return pps, segments
end


function _resolve_points_per_segment(
    points_per_segment::Integer, param_budget::Union{Nothing,Integer},
    n_samples_each, k::Integer, degree::Integer, caller_name::AbstractString,
)
    param_budget === nothing && return points_per_segment
    pps = points_per_segment_for_budget(maximum(n_samples_each), k; degree=degree, param_budget=param_budget)
    println("$caller_name: param_budget=$param_budget -> points_per_segment=$pps (k=$k)")
    return pps
end


function fit_composite_pulse_af(
    pulse::CompositePulse, t_samples::AbstractVector, A_target::AbstractVector, f_target::AbstractVector;
    weight::AbstractVector=A_target .^ 2,
    num_epochs::Integer=1000, learning_rate::Real=0.002, seed::Integer=42,
    u_init::Union{AbstractVector,Nothing}=nothing,
)
    N = length(t_samples)
    (length(A_target) == N && length(f_target) == N && length(weight) == N) || error(
        "t_samples/A_target/f_target/weight must all have the same length, got " *
        "$(N)/$(length(A_target))/$(length(f_target))/$(length(weight))."
    )
    weight_mean = sum(weight) / N + 1e-30
    A_energy = sum(abs2, A_target) + 1e-30

    u = u_init === nothing ? initial_guess(pulse; seed=seed) : collect(Float64, u_init)
    length(u) == n_params(pulse) || error(
        "u_init has length $(length(u)), but this CompositePulse (k=$(pulse.k), " *
        "n_coeff_A=$(pulse.n_coeff_A), n_coeff_f=$(pulse.n_coeff_f)) needs $(n_params(pulse))."
    )
    n = length(u)
    adam = AdamState(n)
    history = NamedTuple[]

    function loss_only(uu)
        A_of_t, f_of_t = build_A_f_of_t(pulse, uu)
        sA = zero(eltype(uu))
        sf = zero(eltype(uu))
        @inbounds for j in 1:N
            sA += abs2(A_of_t(t_samples[j]) - A_target[j])
            sf += weight[j] * abs2(f_of_t(t_samples[j]) - f_target[j])
        end
        return sA / N + (sf / N) / weight_mean
    end

    best_u, best_loss = copy(u), Inf
    for epoch in 1:num_epochs
        grad = ForwardDiff.gradient(loss_only, u)
        loss = Float64(loss_only(u))
        if loss < best_loss
            best_loss, best_u = loss, copy(u)
        end
        push!(history, (epoch=epoch, loss=loss))
        adam_step!(u, grad, adam; lr=learning_rate)
    end

    A_of_t_best, _ = build_A_f_of_t(pulse, best_u)
    A_resid = sum(j -> abs2(A_of_t_best(t_samples[j]) - A_target[j]), 1:N)
    rel_l2_A = sqrt(A_resid / A_energy)

    return best_u, (loss=best_loss, rel_l2_A=rel_l2_A, history=history)
end


const _GRAD_SAFE_FRAC = 1e-2


_encode_scaled_softplus(physical::Real, scale::Real) = _softplus_inv(max(physical, _GRAD_SAFE_FRAC * scale) / scale)


_clip_cf_raw(cf_raw, cf_clip_mult::Real) = clamp.(cf_raw, -cf_clip_mult, cf_clip_mult)


function _fit_composite_pulse_from_samples_learned(
    t::AbstractVector, I::AbstractVector, Q::AbstractVector, d;
    points_per_segment::Integer=22, degree::Integer=3, taper_frac::Real=0.1,
    rel_thresh::Real=1e-3, min_active_samples::Integer=5, min_silence_samples::Integer=3,
    cf_clip_mult::Real=20.0, num_epochs::Integer=1000, learning_rate::Real=0.002, seed::Integer=42,
    param_budget::Union{Nothing,Integer}=nothing,
)
    A = sqrt.(I .^ 2 .+ Q .^ 2)
    phi, f = _instantaneous_frequency(t, I, Q)

    segments = _detect_subpulse_segments(
        t, A; rel_thresh=rel_thresh, min_active_samples=min_active_samples, min_silence_samples=min_silence_samples,
    )
    isempty(segments) && error(
        "No active sub-pulses detected (rel_thresh=$rel_thresh too high relative to the " *
        "trace's own peak, or the trace really is all silence)."
    )
    k = length(segments)

    n_samples_each = [i_end - i_start + 1 for (i_start, i_end) in segments]
    points_per_segment = _resolve_points_per_segment(
        points_per_segment, param_budget, n_samples_each, k, degree, "fit_composite_pulse_from_samples (fit_mode=:learned)",
    )
    n_coeff = _spline_coeff_count(maximum(n_samples_each); points_per_segment=points_per_segment, degree=degree)

    pulse = CompositePulse(k, n_coeff, n_coeff, d; degree=degree, taper_frac=taper_frac)

    raw_gap = Vector{Float64}(undef, k)
    raw_dur = Vector{Float64}(undef, k)
    raw_phi0 = Vector{Float64}(undef, k)
    raw_cA = Matrix{Float64}(undef, n_coeff, k)
    raw_cf = Matrix{Float64}(undef, n_coeff, k)
    t_prev_end = 0.0
    running_seed = 0.0
    for (idx, (i_start, i_end)) in enumerate(segments)
        t_s, t_e = t[i_start], t[i_end]
        duration = t_e - t_s
        gap = max(t_s - t_prev_end, 0.0)
        dur_arg = max(duration - pulse.dur_floor, 0.0)
        raw_gap[idx] = _encode_scaled_softplus(gap, pulse.gap_scale)
        raw_dur[idx] = _encode_scaled_softplus(dur_arg, pulse.dur_scale)









        raw_phi0[idx] = phi[i_start] - running_seed
        running_seed = phi[i_end]

        peak_amp = maximum(view(A, i_start:i_end))
        raw_cA[:, idx] .= _encode_scaled_softplus(peak_amp, pulse.amp_scale)

        f_lo, f_hi = extrema(view(f, i_start:i_end))
        raw_cf[:, idx] .= _clip_cf_raw(range(f_lo, f_hi; length=n_coeff) ./ pulse.freq_scale, cf_clip_mult)

        t_prev_end = t_e
    end
    u_init = pack(pulse, raw_gap, raw_dur, raw_phi0, raw_cA, raw_cf)

    u_fit, fit_report = fit_composite_pulse_af(
        pulse, t, A, f; num_epochs=num_epochs, learning_rate=learning_rate, seed=seed, u_init=u_init,
    )
    return pulse, u_fit, fit_report, segments
end


function _fit_composite_pulse_from_samples_linear(
    t::AbstractVector, I::AbstractVector, Q::AbstractVector, d;
    points_per_segment::Integer=6, degree::Integer=3, taper_frac::Real=0.1,
    rel_thresh::Real=1e-3, min_active_samples::Integer=5, min_silence_samples::Integer=3,
    cA_floor_frac::Real=_GRAD_SAFE_FRAC, cf_clip_mult::Real=20.0,
    param_budget::Union{Nothing,Integer}=nothing,
    segments::Union{Nothing,Vector{Tuple{Int,Int}}}=nothing,
)
    A = sqrt.(I .^ 2 .+ Q .^ 2)
    phi, f = _instantaneous_frequency(t, I, Q)

    segments = segments === nothing ? _detect_subpulse_segments(
        t, A; rel_thresh=rel_thresh, min_active_samples=min_active_samples, min_silence_samples=min_silence_samples,
    ) : segments
    isempty(segments) && error(
        "No active sub-pulses detected (rel_thresh=$rel_thresh too high relative to the " *
        "trace's own peak, or the trace really is all silence)."
    )
    k = length(segments)

    n_samples_each = [i_end - i_start + 1 for (i_start, i_end) in segments]
    points_per_segment = _resolve_points_per_segment(
        points_per_segment, param_budget, n_samples_each, k, degree, "fit_composite_pulse_from_samples (fit_mode=:linear)",
    )
    n_coeff = _spline_coeff_count(maximum(n_samples_each); points_per_segment=points_per_segment, degree=degree)

    pulse = CompositePulse(k, n_coeff, n_coeff, d; degree=degree, taper_frac=taper_frac)
    cA_floor = cA_floor_frac * pulse.amp_scale

    raw_gap = Vector{Float64}(undef, k)
    raw_dur = Vector{Float64}(undef, k)
    raw_phi0 = Vector{Float64}(undef, k)
    raw_cA = Matrix{Float64}(undef, n_coeff, k)
    raw_cf = Matrix{Float64}(undef, n_coeff, k)
    n_cA_floored = 0
    n_cf_clipped = 0
    A_resid = 0.0
    f_resid = 0.0
    phi_resid = 0.0
    f_weight_sum = 0.0
    t_prev_end = 0.0
    running_phase = 0.0

    for (idx, (i_start, i_end)) in enumerate(segments)
        t_s, t_e = t[i_start], t[i_end]
        duration = t_e - t_s
        gap = max(t_s - t_prev_end, 0.0)
        dur_arg = max(duration - pulse.dur_floor, 0.0)
        raw_gap[idx] = _encode_scaled_softplus(gap, pulse.gap_scale)
        raw_dur[idx] = _encode_scaled_softplus(dur_arg, pulse.dur_scale)









































        t_s = t_prev_end + pulse.gap_scale * _softplus(raw_gap[idx])
        t_e = t_s + pulse.dur_scale * _softplus(raw_dur[idx]) + pulse.dur_floor

        t_seg = view(t, i_start:i_end)
        A_seg = view(A, i_start:i_end)
        f_seg = view(f, i_start:i_end)
        phi_seg = view(phi, i_start:i_end)
        n_seg = length(t_seg)

        knots = make_clamped_knots(n_coeff, t_s, t_e, degree)
        Basis = Matrix{Float64}(undef, n_seg, n_coeff)
        @inbounds for j in 1:n_seg
            Basis[j, :] .= bspline_basis(t_seg[j], knots, degree)
        end

        taper_w = [_taper_window(t_seg[j], t_s, t_e, taper_frac) for j in 1:n_seg]
        M_A = Basis .* taper_w
        cA_seg = M_A \ collect(A_seg)
        n_cA_floored += count(<(cA_floor), cA_seg)
        cA_seg = max.(cA_seg, cA_floor)
        raw_cA[:, idx] .= _softplus_inv.(cA_seg ./ pulse.amp_scale)
















        knots_p1 = vcat(knots[1:1], knots, knots[end:end])
        Basis_p1 = Matrix{Float64}(undef, n_seg, n_coeff + 1)
        @inbounds for j in 1:n_seg
            Basis_p1[j, :] .= bspline_basis(t_seg[j], knots_p1, degree + 1)
        end
        L = zeros(Float64, n_coeff + 1, n_coeff)
        for j in 1:n_coeff
            width = (knots[j+degree+1] - knots[j]) / (degree + 1)
            L[j+1:end, j] .= width
        end
        M_Phi_base = Basis_p1 * L
        M_Phi = hcat(M_Phi_base, ones(n_seg))

        f_weight = A_seg .^ 2 .+ 1e-30
        sw = sqrt.(f_weight)
        M_Phi_weighted = M_Phi .* sw
        phi_target_weighted = collect(phi_seg) .* sw
        sol = M_Phi_weighted \ phi_target_weighted
        cf_seg = sol[1:n_coeff]
        phase_const = sol[end]

        cf_seg_raw = cf_seg ./ pulse.freq_scale
        n_cf_clipped += count(x -> abs(x) > cf_clip_mult, cf_seg_raw)
        raw_cf[:, idx] .= _clip_cf_raw(cf_seg_raw, cf_clip_mult)

        A_pred = M_A * cA_seg
        f_pred = Basis * cf_seg
        Phi_pred = M_Phi_base * cf_seg .+ phase_const
        A_resid += sum(abs2, A_pred .- A_seg)
        f_resid += sum(f_weight .* abs2.(f_pred .- f_seg))
        phi_resid += sum(f_weight .* abs2.(Phi_pred .- phi_seg))
        f_weight_sum += sum(f_weight)




































        raw_phi0[idx] = phase_const - running_phase
        cf_seg_used = raw_cf[:, idx] .* pulse.freq_scale
        d_f_end = bspline_area(cf_seg_used, knots, degree)
        running_phase = phase_const + d_f_end

        t_prev_end = t_e
    end

    u_fit = pack(pulse, raw_gap, raw_dur, raw_phi0, raw_cA, raw_cf)

    A_energy = sum(abs2, A) + 1e-30
    rel_l2_A = sqrt(A_resid / A_energy)
    rel_l2_f = sqrt(f_resid / (f_weight_sum + 1e-30)) / pulse.freq_scale
    phi_rms_rad = sqrt(phi_resid / (f_weight_sum + 1e-30))











    E_fit = build_E_of_t(pulse, u_fit)
    E_pred = ComplexF64[E_fit(tt) for tt in t]
    E_tar = complex.(I, Q)
    complex_energy = sum(abs2, E_tar) + 1e-30
    rel_l2_complex = sqrt(sum(abs2, E_pred .- E_tar) / complex_energy)

    fit_report = (
        rel_l2_A=rel_l2_A, rel_l2_f=rel_l2_f, phi_rms_rad=phi_rms_rad,
        rel_l2_complex=rel_l2_complex, n_cA_floored=n_cA_floored, n_cf_clipped=n_cf_clipped,
    )
    return pulse, u_fit, fit_report, segments
end


function fit_composite_pulse_from_samples(
    t::AbstractVector, I::AbstractVector, Q::AbstractVector, d;
    fit_mode::Symbol=:linear,
    points_per_segment::Union{Nothing,Integer}=nothing,
    degree::Integer=3, taper_frac::Real=0.1,
    rel_thresh::Real=1e-3, min_active_samples::Integer=5, min_silence_samples::Integer=3,
    cf_clip_mult::Real=20.0,
    param_budget::Union{Nothing,Integer}=nothing,
    kwargs...,
)
    pps = points_per_segment === nothing ? (fit_mode === :linear ? 6 : 22) : points_per_segment
    if fit_mode === :linear
        return _fit_composite_pulse_from_samples_linear(
            t, I, Q, d; points_per_segment=pps, degree=degree, taper_frac=taper_frac,
            rel_thresh=rel_thresh, min_active_samples=min_active_samples, min_silence_samples=min_silence_samples,
            cf_clip_mult=cf_clip_mult, param_budget=param_budget, kwargs...,
        )
    elseif fit_mode === :learned
        return _fit_composite_pulse_from_samples_learned(
            t, I, Q, d; points_per_segment=pps, degree=degree, taper_frac=taper_frac,
            rel_thresh=rel_thresh, min_active_samples=min_active_samples, min_silence_samples=min_silence_samples,
            cf_clip_mult=cf_clip_mult, param_budget=param_budget, kwargs...,
        )
    else
        error("fit_mode must be :linear or :learned, got $(repr(fit_mode)).")
    end
end



function _interior_amp_scale_factor(inversion::Real, target_inversion::Real)
    I = Float64(inversion)
    tgt = Float64(target_inversion)
    isfinite(I) || error("_interior_amp_scale_factor: inversion must be finite, got $inversion.")
    isfinite(tgt) || error(
        "_interior_amp_scale_factor: target_inversion must be finite, got $target_inversion.",
    )
    (0.0 <= I <= 1.0) || error(
        "_interior_amp_scale_factor: inversion must be in [0, 1], got $inversion.",
    )
    (0.0 < tgt <= 1.0) || error(
        "_interior_amp_scale_factor: target_inversion must be in (0, 1], got $target_inversion.",
    )
    I_floor = 1e-12
    I < I_floor && error(
        "_interior_amp_scale_factor: inversion=$inversion is too small to map onto a " *
        "π/2-area seed (need inversion > $I_floor).",
    )
    return asin(sqrt(tgt)) / asin(sqrt(clamp(I, I_floor, 1.0)))
end


function generate_interior_seed(
    u_fit::AbstractVector,
    inversion::Real,
    silencing::Real,
    pulse::CompositePulse,
    d;
    target_inversion::Real=0.5,
    amp_scale_factor::Union{Nothing,Real}=nothing,
    chirp_bandwidth::Real=2 * pi * 1e6,
    N_samples::Integer=4000,
    param_budget::Integer=60,
    preserve_shape::Bool=false,
    degree::Union{Nothing,Integer}=nothing,
    taper_frac::Union{Nothing,Real}=nothing,
    rel_thresh::Real=1e-3,
    min_active_samples::Integer=5,
    min_silence_samples::Integer=3,
    cf_clip_mult::Real=20.0,
)
    length(u_fit) == n_params(pulse) || error(
        "generate_interior_seed: u_fit has length $(length(u_fit)), but this CompositePulse " *
        "(k=$(pulse.k), n_coeff_A=$(pulse.n_coeff_A), n_coeff_f=$(pulse.n_coeff_f)) needs " *
        "$(n_params(pulse)).",
    )
    N = Int(N_samples)
    N >= 2 || error("generate_interior_seed: N_samples must be >= 2, got $N_samples.")
    pulse.T_max > 0 || error(
        "generate_interior_seed: pulse.T_max must be positive, got $(pulse.T_max).",
    )
    S = Float64(silencing)
    isfinite(S) || error("generate_interior_seed: silencing must be finite, got $silencing.")
    (0.0 <= S <= 1.0) || error(
        "generate_interior_seed: silencing must be in [0, 1], got $silencing.",
    )
    bw = Float64(chirp_bandwidth)
    isfinite(bw) || error(
        "generate_interior_seed: chirp_bandwidth must be finite, got $chirp_bandwidth.",
    )
    bw >= 0 || error(
        "generate_interior_seed: chirp_bandwidth must be >= 0, got $chirp_bandwidth.",
    )

    scale = if amp_scale_factor === nothing
        _interior_amp_scale_factor(inversion, target_inversion)
    else
        s = Float64(amp_scale_factor)
        isfinite(s) && s > 0 || error(
            "generate_interior_seed: amp_scale_factor must be finite and > 0, got $amp_scale_factor.",
        )
        s
    end

    deg = degree === nothing ? pulse.degree : Int(degree)
    tap = taper_frac === nothing ? pulse.taper_frac : Float64(taper_frac)

    u_ref = collect(Float64, u_fit)
    t_grid = collect(range(0.0, pulse.T_max; length=N))
    E_ref = build_E_of_t(pulse, u_ref).(t_grid)
    A_mod = abs.(E_ref) .* scale

    if bw == 0.0
        I_mod = real.(E_ref) .* scale
        Q_mod = imag.(E_ref) .* scale
    else
        f_ramp = collect(range(-bw / 2, bw / 2; length=N))
        dt = pulse.T_max / (N - 1)
        Phi_mod = cumsum(f_ramp) .* dt
        I_mod = A_mod .* cos.(Phi_mod)
        Q_mod = A_mod .* sin.(Phi_mod)
    end

    if preserve_shape
        pulse.n_coeff_A == pulse.n_coeff_f || error(
            "generate_interior_seed: preserve_shape=true requires n_coeff_A == n_coeff_f " *
            "(got n_coeff_A=$(pulse.n_coeff_A), n_coeff_f=$(pulse.n_coeff_f)); pass " *
            "preserve_shape=false to AUTO-size a new shape, or rebuild `pulse` with equal " *
            "coefficient counts.",
        )
        A_det = sqrt.(I_mod .^ 2 .+ Q_mod .^ 2)
        segments_det = _detect_subpulse_segments(
            t_grid, A_det; rel_thresh=rel_thresh, min_active_samples=min_active_samples,
            min_silence_samples=min_silence_samples,
        )
        length(segments_det) == pulse.k || error(
            "generate_interior_seed: preserve_shape=true detected $(length(segments_det)) " *
            "sub-pulse(s) but pulse.k=$(pulse.k) — pass preserve_shape=false, or adjust " *
            "rel_thresh/min_active_samples/min_silence_samples so detection matches k.",
        )
        n_pieces_target = pulse.n_coeff_A - deg
        n_pieces_target >= 1 || error(
            "generate_interior_seed: n_coeff_A=$(pulse.n_coeff_A) is too small for " *
            "degree=$deg (need n_coeff_A >= degree+1 = $(deg + 1)).",
        )
        n_samples_max = maximum(i_end - i_start + 1 for (i_start, i_end) in segments_det)
        pps = cld(n_samples_max, n_pieces_target)
        n_pieces_lo = cld(n_samples_max, pps)
        n_pieces_lo == n_pieces_target || error(
            "generate_interior_seed: n_coeff_A=$(pulse.n_coeff_A) is not achievable for a " *
            "segment of $n_samples_max samples at degree=$deg (points_per_segment-based " *
            "sizing has a gap here). Pass preserve_shape=false to AUTO-size, or change " *
            "N_samples.",
        )
        pulse_new, u_new, fit_report, segments = _fit_composite_pulse_from_samples_linear(
            t_grid, I_mod, Q_mod, d; points_per_segment=pps, degree=deg, taper_frac=tap,
            segments=segments_det, cf_clip_mult=cf_clip_mult,
        )
        (pulse_new.k == pulse.k && pulse_new.n_coeff_A == pulse.n_coeff_A &&
         pulse_new.n_coeff_f == pulse.n_coeff_f) || error(
            "generate_interior_seed: preserve_shape=true expected (k=$(pulse.k), " *
            "n_coeff_A=$(pulse.n_coeff_A), n_coeff_f=$(pulse.n_coeff_f)) but the fit " *
            "produced (k=$(pulse_new.k), n_coeff_A=$(pulse_new.n_coeff_A), " *
            "n_coeff_f=$(pulse_new.n_coeff_f)).",
        )
    else
        pulse_new, u_new, fit_report, segments = fit_composite_pulse_from_samples(
            t_grid, I_mod, Q_mod, d;
            fit_mode=:linear, degree=deg, taper_frac=tap, param_budget=param_budget,
            rel_thresh=rel_thresh, min_active_samples=min_active_samples,
            min_silence_samples=min_silence_samples, cf_clip_mult=cf_clip_mult,
        )
    end

    report = merge(fit_report, (
        inversion_in=Float64(inversion),
        silencing_in=S,
        target_inversion=Float64(target_inversion),
        amp_scale_factor=scale,
        chirp_bandwidth=bw,
        N_samples=N,
        preserve_shape=preserve_shape,
    ))
    return pulse_new, u_new, report, segments
end



function _tmax_power_components(uu, pulse::CompositePulse)
    _, t_end, _, cA, _ = decode(pulse, uu)
    Tu = eltype(uu)
    n_cA = length(cA)
    n_cA > 0 || error("_tmax_power_components: decoded cA is empty.")
    tmax_excess = max(t_end[end] - pulse.T_max, zero(Tu))
    tmax_frac_sq = (tmax_excess / pulse.T_max)^2
    normalized_cA = cA ./ pulse.amp_scale
    power_mean = sum(abs2, normalized_cA) / n_cA
    return tmax_frac_sq, power_mean
end

function _direct_cost_term(uu, pulse::CompositePulse, w_time, w_power, w_tmax)
    dur = pulse_duration(pulse, uu)
    tmax_frac_sq, power_mean = _tmax_power_components(uu, pulse)
    return w_time * (dur / pulse.T_max) + w_tmax * tmax_frac_sq + w_power * power_mean
end



function _schedule_shape(x_tune::Real, fidelity_phys::Real)
    abs(x_tune) < 1e-4 && return Float64(fidelity_phys)
    return (exp(x_tune * fidelity_phys) - 1.0) / (exp(x_tune) - 1.0)
end


function _curriculum_fidelity_weight(inv::Real, sil::Real, target_F::Real, x_tune::Real)
    isfinite(x_tune) || error("_curriculum_fidelity_weight: x_tune must be finite, got $x_tune.")
    abs(x_tune) < 700 || error(
        "_curriculum_fidelity_weight: |x_tune| must be < 700 (Float64 exp overflow), got $x_tune."
    )
    inv_c = isnan(inv) ? 0.0 : clamp(Float64(inv), 0.0, 1.0)
    sil_c = isnan(sil) ? 0.0 : clamp(Float64(sil), 0.0, 1.0)
    silencing_success = 1.0 - (sil_c - Float64(target_F))^2
    fidelity_phys = inv_c * silencing_success
    return _schedule_shape(x_tune, fidelity_phys)
end


const _DEFAULT_X_TUNE_ALPHA = 0.025


function solve_optimal_x_start(F_0::Real, alpha::Real; x_max::Real=100.0, tol::Real=1e-6, max_iter::Integer=200)
    0.0 <= F_0 <= 1.0 || error("solve_optimal_x_start: F_0 must be in [0,1], got $F_0.")
    0.0 <= alpha <= 1.0 || error("solve_optimal_x_start: alpha must be in [0,1], got $alpha.")
    x_max > 0 || error("solve_optimal_x_start: x_max must be > 0, got $x_max.")
    x_max < 700 || error(
        "solve_optimal_x_start: x_max must be < 700 (Float64 exp overflow, matching " *
        "_curriculum_fidelity_weight's own |x_tune|<700 contract), got $x_max. Above " *
        "this, _schedule_shape(x_max, F_0) can silently evaluate Inf/Inf=NaN inside " *
        "the bisection loop -- caught here rather than left to a stuck/NaN search."
    )
    tol > 0 || error("solve_optimal_x_start: tol must be > 0, got $tol.")

    F_0 <= 0.0 && return 1e-4
    F_0 >= 1.0 && return 1e-4
    abs(alpha - F_0) < tol && return 1e-4

    hard_cap = 699.999
    bound = min(Float64(x_max), hard_cap)
    while true
        left, right = alpha < F_0 ? (1e-4, bound) : (-bound, -1e-4)
        for _ in 1:max_iter
            mid = (left + right) / 2.0
            val = _schedule_shape(mid, F_0)
            abs(val - alpha) < tol && return mid





            if val > alpha
                left = mid
            else
                right = mid
            end
        end









        if bound >= hard_cap
            result = (left + right) / 2.0
            achieved = _schedule_shape(result, F_0)
            if abs(achieved - alpha) >= tol
                @warn "solve_optimal_x_start: no finite x_tune within the (-700,700) exp-overflow " *
                      "range reaches alpha=$alpha at F_0=$F_0 (this pair's true root lies beyond the " *
                      "representable domain, not merely beyond x_max) -- returning the closest " *
                      "achievable boundary x=$result (achieved factor=$(round(achieved, sigdigits=4)), " *
                      "target=$alpha). Only occurs for F_0/alpha extremely close to 0 or 1."
            end
            return result
        end
        bound = min(bound * 4.0, hard_cap)
    end
end


function _reconstitute_static_direct_cost(dyn_cost::Real,
                                           base_w_time::Real, dyn_w_time::Real,
                                           base_w_tmax::Real, dyn_w_tmax::Real,
                                           base_w_power::Real, dyn_w_power::Real,
                                           duration::Real, tmax_frac_sq::Real, power_mean::Real, T_max::Real)
    return dyn_cost +
           (Float64(base_w_time) - Float64(dyn_w_time)) * (Float64(duration) / Float64(T_max)) +
           (Float64(base_w_tmax) - Float64(dyn_w_tmax)) * Float64(tmax_frac_sq) +
           (Float64(base_w_power) - Float64(dyn_w_power)) * Float64(power_mean)
end

function _gradient_on_indices(f, u::AbstractVector, idxs::Vector{Int})
    isempty(idxs) && return zeros(Float64, length(u))
    return _gradient_on_indices_val(f, u, idxs, Val{length(idxs)}())
end

function _gradient_on_indices_val(f, u::AbstractVector, idxs::Vector{Int}, ::Val{C}) where {C}
    n = length(u)
    u_host = collect(Float64, u)
    function f_reduced(u_chunk)
        T = eltype(u_chunk)
        uu = Vector{T}(undef, n)
        @inbounds for i in 1:n
            uu[i] = T(u_host[i])
        end
        @inbounds for j in 1:C
            uu[idxs[j]] = u_chunk[j]
        end
        return f(uu)
    end
    u_chunk0 = u_host[idxs]
    cfg = ForwardDiff.GradientConfig(f_reduced, u_chunk0, ForwardDiff.Chunk{C}())
    g_chunk = ForwardDiff.gradient(f_reduced, u_chunk0, cfg)
    g = zeros(Float64, n)
    @inbounds for j in 1:C
        g[idxs[j]] = g_chunk[j]
    end
    return g
end


function _pulse_cost_grad_threaded(u::AbstractVector, pulse::CompositePulse, d;
                                    target_F=1.0, w_time=0.15, w_power=0.05, w_tmax=1.0,
                                    I_min::Real=_DEFAULT_PENALTY_MIN, kappa_I::Real=_DEFAULT_PENALTY_KAPPA,
                                    S_min::Real=_DEFAULT_PENALTY_MIN, kappa_S::Real=_DEFAULT_PENALTY_KAPPA,
                                    compute::Symbol=:auto, track::Symbol=:weak, kwargs...)
    _forbid_initial_condition(kwargs)
    _assert_track(track)
    sk = _solver_kwargs(kwargs)
    n = length(u)
    length(u) == n_params(pulse) || error(
        "_pulse_cost_grad_threaded: u has length $(length(u)), expected n_params=$(n_params(pulse))."
    )
    duration = pulse_duration(pulse, u)
    M = _assert_ensemble_shapes(d)
    compute_eff = _resolve_compute(compute, M)
    use_gpu_pool = compute_eff === :gpu && pulse_gpu_count() >= 1
















    chunk = ForwardDiff.Chunk{min(60, n)}()

    function direct_only(uu)
        return _direct_cost_term(uu, pulse, w_time, w_power, w_tmax)
    end
    grad_direct = ForwardDiff.gradient(direct_only, u, ForwardDiff.GradientConfig(direct_only, u, chunk))
    direct_val = direct_only(u)

    aux_ground = Ref{Float64}(0.0)
    aux_weak = Ref{NTuple{4,Float64}}((0.0, 0.0, 0.0, 0.0))

    if track === :weak







        function weak_pair(uu)
            _, Sp, Sz_w, Nj_eq = run_sim_1st_order_pure(
                uu, pulse, d; compute=compute, sk..., initial_condition=:weak,
            )
            Tu = eltype(uu)
            inv_ = _weighted_inversion(Sz_w, d.g_b, Nj_eq, Tu)
            sil_ = _weighted_silencing_factor(Sp, d.g_b, Nj_eq, d.delta_b, Tu)
            coh_ = _weighted_coherence(Sp, d.g_b, Nj_eq, d.delta_b, Tu)
            famp_ = _weighted_field_amplitude(Sp, d.g_b, Nj_eq, Tu)
            ret_ = _weak_seed_retention(Sp, d.g_b, Nj_eq, d.delta_b, Tu)
            aux_ground[] = Float64(ForwardDiff.value(inv_))
            aux_weak[] = (Float64(ForwardDiff.value(sil_)), Float64(ForwardDiff.value(coh_)),
                          Float64(ForwardDiff.value(famp_)), Float64(ForwardDiff.value(ret_)))
            return [inv_, sil_]
        end
        local J
        try
            J = ForwardDiff.jacobian(weak_pair, u, ForwardDiff.JacobianConfig(weak_pair, u, chunk))
        catch e
            e isa PulseSolveFailed || rethrow()
            return fill(NaN, n), Inf, NaN, NaN, duration, NaN, NaN, NaN
        end
        inversion = aux_ground[]
        silencing, coherence, field_amp, weak_seed_retention = aux_weak[]
        grad_I = J[1, :]
        grad_F = J[2, :]

    elseif !use_gpu_pool
        grads = Vector{Vector{Float64}}(undef, 2)
        failed = fill(false, 2)
        Threads.@threads for i in 1:2
            try
                if i == 1

                    function ground_only(uu)
                        _, _, Sz, Nj = run_sim_1st_order_pure(
                            uu, pulse, d; compute=compute, sk..., initial_condition=:ground,
                        )
                        inv_ = _weighted_inversion(Sz, d.g_b, Nj, eltype(uu))
                        aux_ground[] = Float64(ForwardDiff.value(inv_))
                        return inv_
                    end
                    grads[1] = ForwardDiff.gradient(ground_only, u, ForwardDiff.GradientConfig(ground_only, u, chunk))
                else

                    function weak_only(uu)
                        _, Sp, _, Nj_eq = run_sim_1st_order_pure(
                            uu, pulse, d; compute=compute, sk..., initial_condition=:weak,
                        )
                        Tu = eltype(uu)
                        sil_ = _weighted_silencing_factor(Sp, d.g_b, Nj_eq, d.delta_b, Tu)
                        coh_ = _weighted_coherence(Sp, d.g_b, Nj_eq, d.delta_b, Tu)
                        famp_ = _weighted_field_amplitude(Sp, d.g_b, Nj_eq, Tu)
                        ret_ = _weak_seed_retention(Sp, d.g_b, Nj_eq, d.delta_b, Tu)
                        aux_weak[] = (Float64(ForwardDiff.value(sil_)), Float64(ForwardDiff.value(coh_)),
                                      Float64(ForwardDiff.value(famp_)), Float64(ForwardDiff.value(ret_)))
                        return sil_
                    end
                    grads[2] = ForwardDiff.gradient(weak_only, u, ForwardDiff.GradientConfig(weak_only, u, chunk))
                end
            catch e
                e isa PulseSolveFailed || rethrow()
                failed[i] = true
            end
        end
        any(failed) && return fill(NaN, n), Inf, NaN, NaN, duration, NaN, NaN, NaN

        inversion = aux_ground[]
        silencing, coherence, field_amp, weak_seed_retention = aux_weak[]
        grad_I = grads[1]
        grad_F = grads[2]

    else
        n_ic = 2
        n_gpu = pulse_gpu_count()
        n_chunks = max(1, n_gpu ÷ n_ic)
        ranges = _split_index_ranges(n, n_chunks)
        _log_pulse_compute_once(
            "pulse gradient: $n_gpu GPU(s), $(n_ic) IC track(s) × $(length(ranges)) param-chunk(s) " *
            "(M=$M, n_params=$n; start Julia with -t $n_gpu or more so device jobs overlap)"
        )

        jobs = NamedTuple[]
        for r in ranges
            push!(jobs, (kind=:ground, idxs=collect(r)))
            push!(jobs, (kind=:weak, idxs=collect(r)))
        end

        grad_I = zeros(n)
        grad_F = zeros(n)
        failed = Threads.Atomic{Bool}(false)
        ground_lock = ReentrantLock()
        weak_lock = ReentrantLock()

        _run_pulse_jobs!(jobs, (job, _dev) -> begin
            failed[] && return nothing
            try
                if job.kind === :ground
                    function ground_gpu(uu)
                        _, _, Sz, Nj = run_sim_1st_order_pure(
                            uu, pulse, d; compute=:gpu, sk..., initial_condition=:ground,
                        )
                        inv_ = _weighted_inversion(Sz, d.g_b, Nj, eltype(uu))
                        if first(job.idxs) == 1
                            aux_ground[] = Float64(ForwardDiff.value(inv_))
                        end
                        return inv_
                    end
                    g = length(job.idxs) == n ?
                        ForwardDiff.gradient(ground_gpu, u, ForwardDiff.GradientConfig(ground_gpu, u, chunk)) :
                        _gradient_on_indices(ground_gpu, u, job.idxs)
                    lock(ground_lock) do
                        grad_I .+= g
                    end
                else
                    function weak_gpu(uu)
                        _, Sp, _, Nj_eq = run_sim_1st_order_pure(
                            uu, pulse, d; compute=:gpu, sk..., initial_condition=:weak,
                        )
                        Tu = eltype(uu)
                        sil_ = _weighted_silencing_factor(Sp, d.g_b, Nj_eq, d.delta_b, Tu)
                        coh_ = _weighted_coherence(Sp, d.g_b, Nj_eq, d.delta_b, Tu)
                        famp_ = _weighted_field_amplitude(Sp, d.g_b, Nj_eq, Tu)
                        ret_ = _weak_seed_retention(Sp, d.g_b, Nj_eq, d.delta_b, Tu)
                        if first(job.idxs) == 1
                            aux_weak[] = (Float64(ForwardDiff.value(sil_)), Float64(ForwardDiff.value(coh_)),
                                          Float64(ForwardDiff.value(famp_)), Float64(ForwardDiff.value(ret_)))
                        end
                        return sil_
                    end
                    g = length(job.idxs) == n ?
                        ForwardDiff.gradient(weak_gpu, u, ForwardDiff.GradientConfig(weak_gpu, u, chunk)) :
                        _gradient_on_indices(weak_gpu, u, job.idxs)
                    lock(weak_lock) do
                        grad_F .+= g
                    end
                end
            catch e
                e isa PulseSolveFailed || rethrow()
                failed[] = true
            end
            return nothing
        end)
        _reclaim_gpu_memory()

        failed[] && return fill(NaN, n), Inf, NaN, NaN, duration, NaN, NaN, NaN
        inversion = aux_ground[]
        silencing, coherence, field_amp, weak_seed_retention = aux_weak[]
    end


    physics_cost, fidelity_phys, silencing_success =
        _fidelity_physics_cost(inversion, silencing, Float64(target_F), I_min, kappa_I, S_min, kappa_S)
    coeff_I, coeff_S = _fidelity_gradient_coefficients(inversion, silencing_success,
                                                        fidelity_phys, I_min, kappa_I, S_min, kappa_S)

    grad_S = -2.0 * (silencing - Float64(target_F)) .* grad_F
    grad_physics = coeff_I .* grad_I .+ coeff_S .* grad_S

    grad = grad_physics .+ grad_direct
    cost = physics_cost + direct_val
    return grad, cost, inversion, silencing, duration, coherence, field_amp, weak_seed_retention
end



function run_local_adam(u_start::AbstractVector, pulse::CompositePulse, d, cost_kwargs::NamedTuple;
                         hop::Integer=0, num_epochs::Integer=30, patience::Integer=5, tol::Real=1e-3,
                         learning_rate::Real=0.05, cf_lr_scale::Real=1.0, label::AbstractString="",
                         threaded_grad::Bool=false, compute::Symbol=:auto, grad_mode::Symbol=:forwarddiff,
                         anneal_direct_weights::Bool=true, hop0_phyonly::Bool=true,
                         x_tune_alpha::Union{Nothing,Real}=_DEFAULT_X_TUNE_ALPHA,
                         _precalibrated_x_tune::Union{Nothing,Real}=nothing,
                         solve_kwargs...)
    _forbid_initial_condition(solve_kwargs)
    (grad_mode === :forwarddiff || grad_mode === :adjoint) || error(
        "grad_mode must be :forwarddiff or :adjoint, got $(repr(grad_mode))."
    )
    u = copy(u_start)
    n = length(u)
    adam = AdamState(n)
    lr_scale = cf_lr_scale == 1.0 ? nothing : pack(
        pulse, ones(pulse.k), ones(pulse.k), ones(pulse.k), ones(pulse.n_coeff_A, pulse.k),
        fill(cf_lr_scale, pulse.n_coeff_f, pulse.k),
    )
    aux = Ref{NTuple{7,Float64}}((NaN, NaN, NaN, NaN, NaN, NaN, NaN))
    base_w_time = haskey(cost_kwargs, :w_time) ? Float64(cost_kwargs.w_time) : 0.15
    target_F_val = haskey(cost_kwargs, :target_F) ? Float64(cost_kwargs.target_F) : 1.0






























    seed_inv, seed_sil = NaN, NaN
    x_tune_eff = if hop == 0 && hop0_phyonly
        0.0
    elseif _precalibrated_x_tune !== nothing
        Float64(_precalibrated_x_tune)
    elseif x_tune_alpha !== nothing && anneal_direct_weights
        _, inv0, sil0, _, _, _ = pulse_cost(u_start, pulse, d; cost_kwargs..., compute=compute, solve_kwargs...)
        silencing_success0 = 1.0 - (Float64(sil0) - target_F_val)^2
        F_0 = Float64(inv0) * silencing_success0
        val = solve_optimal_x_start(F_0, Float64(x_tune_alpha))
        if isfinite(inv0) && isfinite(sil0)
            seed_inv, seed_sil = Float64(inv0), Float64(sil0)
        end
        println(
            "$label x_tune_alpha=$(x_tune_alpha): calibrated x_tune=$(round(val, sigdigits=4)) " *
            "from F_0=$(round(F_0, sigdigits=4)) at u_start"
        )
        val
    else
        0.0
    end

    best_u = copy(u_start)
    best_cost, best_inv, best_sil, best_dur = Inf, 0.0, 0.0, 0.0
    epochs_since_improve = 0
    history = NamedTuple[]
    last_good_u = copy(u_start)
    last_good_grad = zeros(n)





    last_good_aux = (NaN, seed_inv, seed_sil, NaN, NaN, NaN, NaN)
    adam_m0 = zeros(n)
    adam_v0 = zeros(n)
    adam_t0 = 0
    lr = learning_rate
    just_reverted = false
    prev_dyn_w_time = base_w_time

    for epoch in 1:num_epochs
        t_wall = time()

















        factor = if hop == 0 && hop0_phyonly
            0.0
        elseif anneal_direct_weights
            _curriculum_fidelity_weight(last_good_aux[2], last_good_aux[3], target_F_val, x_tune_eff)
        else
            1.0
        end











        dyn_w_time = (hop == 0 && hop0_phyonly) ? 0.0 : base_w_time * factor
        @assert !(hop == 0 && hop0_phyonly) || dyn_w_time == 0.0



















        epoch_cost_kwargs = merge(cost_kwargs, (w_time=dyn_w_time,))






        if epoch > 1 && dyn_w_time != prev_dyn_w_time
            w_diff = dyn_w_time - prev_dyn_w_time
            best_cost += w_diff * (best_dur / pulse.T_max)
            if isfinite(last_good_aux[1])
                shifted_last_cost = last_good_aux[1] + w_diff * (last_good_aux[4] / pulse.T_max)
                last_good_aux = (shifted_last_cost, last_good_aux[2], last_good_aux[3],
                                 last_good_aux[4], last_good_aux[5], last_good_aux[6], last_good_aux[7])
            end
        end
        prev_dyn_w_time = dyn_w_time














        if just_reverted
            grad = last_good_grad
            cost, inv_, sil_, dur_, coh_, famp_, ret_ = last_good_aux
            adam.m .= adam_m0
            adam.v .= adam_v0
            adam.t = adam_t0
        elseif grad_mode === :adjoint
            grad, cost, inv_, sil_, dur_, coh_, famp_, ret_ = pulse_cost_grad_adjoint(
                u, pulse, d; epoch_cost_kwargs..., compute=compute, solve_kwargs...,
            )
        elseif threaded_grad
            grad, cost, inv_, sil_, dur_, coh_, famp_, ret_ = _pulse_cost_grad_threaded(
                u, pulse, d; epoch_cost_kwargs..., compute=compute, solve_kwargs...,
            )
        else
            function cost_only(uu)
                c, inv_2, sil_2, dur_2, coh_2, famp_2, ret_2 = pulse_cost(uu, pulse, d; epoch_cost_kwargs..., compute=compute, solve_kwargs...)
                aux[] = (
                    Float64(ForwardDiff.value(c)),
                    Float64(ForwardDiff.value(inv_2)),
                    Float64(ForwardDiff.value(sil_2)),
                    Float64(ForwardDiff.value(dur_2)),
                    Float64(ForwardDiff.value(coh_2)),
                    Float64(ForwardDiff.value(famp_2)),
                    Float64(ForwardDiff.value(ret_2)),
                )
                return c
            end
            grad = ForwardDiff.gradient(cost_only, u)
            cost, inv_, sil_, dur_, coh_, famp_, ret_ = aux[]
        end
        if !isfinite(cost)
            epochs_since_improve += 1
            push!(history, (hop=hop, epoch=epoch, k=pulse.k, cost=cost, inversion=inv_,
                             silencing=sil_, duration=dur_, coherence=coh_, field_amp=famp_,
                             weak_seed_retention=ret_, improved=false,
                             x_tune=x_tune_eff, schedule_factor=factor))
            elapsed = time() - t_wall
            u .= last_good_u
            lr /= 2
            just_reverted = true
            println(
                "$label epoch $(lpad(epoch, 3)): $(round(elapsed, digits=1))s  " *
                "cost=Inf   ODE solve failed -- reverted to last valid point, " *
                "halved step size to $(round(lr, sigdigits=3))"
            )
            if epochs_since_improve >= patience
                println("$label early stop at epoch $epoch (no improvement > $tol for $patience epochs)")
                break
            end
            continue
        end

        lr = min(lr * 1.5, learning_rate)
        last_good_u .= u
        last_good_grad .= grad
        last_good_aux = (cost, inv_, sil_, dur_, coh_, famp_, ret_)
        adam_m0 .= adam.m
        adam_v0 .= adam.v
        adam_t0 = adam.t
        just_reverted = false

        improved = cost < best_cost - tol
        if improved
            best_cost, best_u = cost, copy(u)
            best_inv, best_sil, best_dur = inv_, sil_, dur_
            epochs_since_improve = 0
        else
            epochs_since_improve += 1
        end

        push!(history, (hop=hop, epoch=epoch, k=pulse.k, cost=cost, inversion=inv_,
                         silencing=sil_, duration=dur_, coherence=coh_, field_amp=famp_,
                         weak_seed_retention=ret_, improved=improved,
                         x_tune=x_tune_eff, schedule_factor=factor))

        adam_step!(u, grad, adam; lr=lr, lr_scale=lr_scale)

        elapsed = time() - t_wall
        mark = improved ? "*" : " "
        println(
            "$label epoch $(lpad(epoch, 3)): $(round(elapsed, digits=1))s  " *
            "cost=$(round(cost, digits=4)) $mark inversion=$(round(inv_, digits=4)) " *
            "silencing=$(round(sil_, digits=4)) duration=$(round(dur_, sigdigits=4))s"
        )

        if epochs_since_improve >= patience
            println("$label early stop at epoch $epoch (no improvement > $tol for $patience epochs)")
            break
        end
    end





    final_static_cost = if isfinite(best_cost)
        pulse_cost(best_u, pulse, d; cost_kwargs..., compute=compute, solve_kwargs...)[1]
    else
        Inf
    end

    return best_u, final_static_cost, best_inv, best_sil, best_dur, history
end


function optimise_composite_pulse(
    k::Integer, n_coeff_A::Integer, n_coeff_f::Integer, d;
    num_epochs::Integer=30, learning_rate::Real=0.05, cf_lr_scale::Real=1.0, patience::Integer=5, tol::Real=1e-3,
    n_hops::Integer=3, hop_patience::Integer=2, hop_step_size::Real=0.5, temperature::Real=1.0,
    degree::Integer=3, taper_frac::Real=0.1, w_tmax::Real=1.0, w_power::Real=0.05,
    target_F::Real=1.0, w_time::Real=0.15,
    seed::Integer=42, warm_start_u=nothing, label_prefix::AbstractString="",
    threaded_grad::Bool=true, compute::Symbol=:auto, grad_mode::Symbol=:forwarddiff,
    track::Symbol=:weak,
    anneal_direct_weights::Bool=true, hop0_phyonly::Bool=true,
    x_tune_alpha::Union{Nothing,Real}=_DEFAULT_X_TUNE_ALPHA, recalibrate_optima_x::Bool=true,
    I_min::Real=_DEFAULT_PENALTY_MIN, kappa_I::Real=_DEFAULT_PENALTY_KAPPA,
    S_min::Real=_DEFAULT_PENALTY_MIN, kappa_S::Real=_DEFAULT_PENALTY_KAPPA,
    solve_kwargs...,
)
    _forbid_initial_condition(solve_kwargs)
    _assert_track(track)
    (grad_mode === :forwarddiff || grad_mode === :adjoint) || error(
        "grad_mode must be :forwarddiff or :adjoint, got $(repr(grad_mode))."
    )
    pulse = CompositePulse(k, n_coeff_A, n_coeff_f, d; degree=degree, taper_frac=taper_frac)
    cost_kwargs = (w_tmax=w_tmax, w_power=w_power, target_F=target_F, w_time=w_time,
                   I_min=I_min, kappa_I=kappa_I, S_min=S_min, kappa_S=kappa_S, track=track)
    rng = Random.Xoshiro(seed)





    solve_settings = NamedTuple(kv for kv in pairs(solve_kwargs) if !(kv[2] isa Function))
    optimizer_settings = merge(
        (k=k, n_coeff_A=n_coeff_A, n_coeff_f=n_coeff_f, degree=degree, taper_frac=taper_frac,
         num_epochs=num_epochs, learning_rate=learning_rate, cf_lr_scale=cf_lr_scale, patience=patience, tol=tol,
         n_hops=n_hops, hop_patience=hop_patience, hop_step_size=hop_step_size, temperature=temperature,
         w_tmax=w_tmax, w_power=w_power, target_F=target_F, w_time=w_time, seed=seed,
         threaded_grad=threaded_grad, compute=compute, grad_mode=grad_mode, track=track, n_gpus=pulse_gpu_count(),
         anneal_direct_weights=anneal_direct_weights, hop0_phyonly=hop0_phyonly, x_tune_alpha=x_tune_alpha,
         recalibrate_optima_x=recalibrate_optima_x,
         I_min=I_min, kappa_I=kappa_I, S_min=S_min, kappa_S=kappa_S),
        solve_settings,
    )

    println(
        "$(label_prefix)Optimising k=$k pulses, $(n_params(pulse)) raw parameters (ForwardDiff/Adam + " *
        "basin-hopping, physics: InhomogeneousSpinCavityDynamics.jl rhs_1st_order!, " *
        "compute=$(compute), GPUs=$(pulse_gpu_count())) ..."
    )

    if warm_start_u === nothing
        u0 = initial_guess(pulse; seed=seed)
    else
        length(warm_start_u) == n_params(pulse) || error(
            "warm_start_u has length $(length(warm_start_u)), but this CompositePulse " *
            "(k=$k, n_coeff_A=$n_coeff_A, n_coeff_f=$n_coeff_f) needs $(n_params(pulse))."
        )
        u0 = collect(Float64, warm_start_u)
        println("$(label_prefix)Warm-starting hop 0 from a supplied raw vector.")
    end
    initial_metrics = pulse_cost(u0, pulse, d; cost_kwargs..., compute=compute, solve_kwargs...)
    history = NamedTuple[]







    current_u, current_cost, _, _, _, hop0_history = run_local_adam(
        u0, pulse, d, cost_kwargs; hop=0, num_epochs, patience, tol, learning_rate, cf_lr_scale,
        label="$(label_prefix)[hop 0]", threaded_grad=threaded_grad, compute=compute,
        grad_mode=grad_mode, anneal_direct_weights=anneal_direct_weights,
        hop0_phyonly=hop0_phyonly, x_tune_alpha=x_tune_alpha,
        solve_kwargs...
    )
    append!(history, hop0_history)
    global_best_u, global_best_cost = current_u, current_cost
    hops_since_improve = 0
    x_tune_seed = 0.0

    for hop in 1:(n_hops-1)
        perturbation = hop_step_size .* randn(rng, length(current_u))
        candidate_u0 = current_u .+ perturbation










        hop_x_tune_alpha, hop_precal = if hop == 1 || recalibrate_optima_x
            (x_tune_alpha, nothing)
        else
            (nothing, x_tune_seed)
        end

        cand_u, cand_cost, _, _, _, hop_history = run_local_adam(
            candidate_u0, pulse, d, cost_kwargs;
            hop, num_epochs, patience, tol, learning_rate, cf_lr_scale,
            label="$(label_prefix)[hop $hop]", threaded_grad=threaded_grad, compute=compute,
            grad_mode=grad_mode, anneal_direct_weights=anneal_direct_weights,
            x_tune_alpha=hop_x_tune_alpha, _precalibrated_x_tune=hop_precal,
            solve_kwargs...,
        )
        append!(history, hop_history)
        if hop == 1
            x_tune_seed = isempty(hop_history) ? 0.0 : hop_history[1].x_tune
        end

        if cand_cost < global_best_cost - tol
            global_best_u, global_best_cost = cand_u, cand_cost
            hops_since_improve = 0
        else
            hops_since_improve += 1
        end

        delta = cand_cost - current_cost
        accept = delta < 0.0 || rand(rng) < exp(-delta / max(temperature, 1e-12))
        if accept
            current_u, current_cost = cand_u, cand_cost
        end

        accept_str = accept ? "accepted" : "rejected"
        println(
            "$(label_prefix)hop $hop: local best cost=$(round(cand_cost, digits=4)) " *
            "($accept_str as new basin, delta=$(round(delta, digits=4))) " *
            "global best cost=$(round(global_best_cost, digits=4))"
        )

        if hops_since_improve >= hop_patience
            println("$(label_prefix)Basin-hopping stopped early after $hop hops (no global improvement > $tol for $hop_patience hops).")
            break
        end
    end

    final_metrics = pulse_cost(global_best_u, pulse, d; cost_kwargs..., compute=compute, solve_kwargs...)









    if track === :weak
        sk_final = _solver_kwargs(solve_kwargs)
        _, _, Sz_gf, Nj_gf = run_sim_1st_order_pure(
            global_best_u, pulse, d; compute=compute, sk_final..., initial_condition=:ground,
        )
        final_inversion_ground = Float64(_weighted_inversion(Sz_gf, d.g_b, Nj_gf, Float64))
        final_inv_gap = Float64(final_metrics[2]) - final_inversion_ground
        println(
            "$(label_prefix)track=:weak winner re-check: inversion(:weak)=" *
            "$(round(Float64(final_metrics[2]), sigdigits=6))  inversion(:ground)=" *
            "$(round(final_inversion_ground, sigdigits=6))  inv_gap=$(round(final_inv_gap, sigdigits=3))"
        )
    else
        final_inversion_ground = Float64(final_metrics[2])
        final_inv_gap = 0.0
    end
    optimizer_settings = merge(
        optimizer_settings,
        (final_inversion_ground=final_inversion_ground, final_inv_gap=final_inv_gap),
    )

    println("$(label_prefix)Optimisation complete. Global best cost: $(round(global_best_cost, digits=4)).")
    return global_best_u, global_best_cost, pulse, u0, initial_metrics, history, final_metrics, optimizer_settings
end

function _normalise_k_specs(kinds, specs)
    if specs !== nothing
        return collect(specs)
    end
    return [(k_of_seed_kind(kind), kind) for kind in kinds]
end


function optimise_composite_pulse_over_k(
    n_coeff_A::Integer, n_coeff_f::Integer, d;
    kinds=(:hs1, :corpse, :bb1),
    specs=nothing,
    threaded::Bool=true,
    Omega_max=nothing,
    beta=nothing,
    mu=nothing,
    seed::Integer=42,
    optimizer_kwargs...,
)
    :warm_start_u in keys(optimizer_kwargs) && error(
        "optimise_composite_pulse_over_k builds a per-k canonical seed; do not pass warm_start_u."
    )
    _forbid_initial_condition(optimizer_kwargs)

    job_specs = _normalise_k_specs(kinds, specs)
    isempty(job_specs) && error("No (k, kind) specs to optimise.")
    seen = Dict{Int,Symbol}()
    for (k, kind) in job_specs
        k isa Integer && k >= 1 || error("k must be a positive integer, got $k.")
        kind isa Symbol || error("kind must be a Symbol, got $kind.")
        haskey(seen, k) && error("Duplicate k=$k (kinds $(seen[k]) and $kind). Each k can run once.")
        seen[k] = kind
        if kind !== :random
            k_of_seed_kind(kind) == k || error(
                "kind $kind requires k=$(k_of_seed_kind(kind)), got k=$k."
            )
        end
    end

    n = length(job_specs)
    nthreads = Threads.nthreads()
    use_threads = threaded && nthreads > 1 && n > 1
    println(
        "Discrete-k search: $n independent continuous optimisations " *
        (use_threads ? "on $nthreads threads" : "serially") *
        ". kinds=$(collect(spec[2] for spec in job_specs))."
    )

    function run_spec(spec)
        k, kind = spec
        prefix = "[$kind k=$k] "
        deg = get(optimizer_kwargs, :degree, 3)
        tfrac = get(optimizer_kwargs, :taper_frac, 0.1)
        pulse_seed = CompositePulse(k, n_coeff_A, n_coeff_f, d; degree=deg, taper_frac=tfrac)
        Ω = Omega_max === nothing ? pulse_seed.amp_scale : Omega_max
        u0 = seed_canonical(pulse_seed, kind; Omega_max=Ω, beta=beta, mu=mu, seed=seed)
        best_u, best_cost, pulse_out, u0_out, initial_metrics, history, final_metrics, optimizer_settings =
            optimise_composite_pulse(
                k, n_coeff_A, n_coeff_f, d;
                optimizer_kwargs...,
                seed=seed + 1000 * Int(k),
                warm_start_u=u0,
                label_prefix=prefix,
            )
        optimizer_settings = merge(optimizer_settings, (seed_kind=kind,))
        return (
            kind=kind, k=Int(k),
            best_u=best_u, best_cost=best_cost, pulse=pulse_out,
            u0=u0_out, initial_metrics=initial_metrics, history=history,
            final_metrics=final_metrics, optimizer_settings=optimizer_settings,
        )
    end

    per_k = Vector{NamedTuple}(undef, n)
    if use_threads
        Threads.@threads for i in 1:n
            per_k[i] = run_spec(job_specs[i])
        end
    else
        for i in 1:n
            per_k[i] = run_spec(job_specs[i])
        end
    end

    best = per_k[1]
    for r in per_k
        if r.best_cost < best.best_cost
            best = r
        end
    end

    println("Discrete-k search complete.")
    for r in per_k
        mark = r.k == best.k ? "*" : " "
        println(
            "  $mark k=$(r.k) $(r.kind): cost=$(round(r.best_cost, digits=4))"
        )
    end
    println("  winner: k=$(best.k) $(best.kind)  cost=$(round(best.best_cost, digits=4))")

    return (
        best_kind=best.kind, best_k=best.k,
        best_u=best.best_u, best_cost=best.best_cost, pulse=best.pulse,
        u0=best.u0, initial_metrics=best.initial_metrics, history=best.history,
        final_metrics=best.final_metrics, optimizer_settings=best.optimizer_settings,
        per_k=per_k,
    )
end

