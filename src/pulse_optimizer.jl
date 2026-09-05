

function build_u0_1st_order_cpu(M::Integer, Nj::AbstractVector, ::Type{T},
                                 initial_condition::Symbol=:ground) where {T}
    u0 = zeros(Complex{T}, state_length_1st_order(M))
    if initial_condition == :ground
        u0[idx1_Sz_start(M):end] .= .-Nj ./ 2
    elseif initial_condition == :inverted
        u0[idx1_Sz_start(M):end] .= Nj ./ 2
    elseif initial_condition == :custom

    else
        error("Unknown initial_condition = $(initial_condition). Use :ground, :inverted, or :custom.")
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


function run_sim_1st_order_pure(
    u::AbstractVector,
    pulse::CompositePulse,
    d;
    signal_E_of_t = _zero_drive,
    initial_condition::Symbol=:ground,
    alg=Tsit5(),
    reltol=1e-8,
    abstol=1e-8,
)
    T = eltype(u)
    M = d.M
    control_E_of_t = build_E_of_t(pulse, u)
    E_of_t(t) = control_E_of_t(t) + signal_E_of_t(t)
    u0 = build_u0_1st_order_cpu(M, d.Nj, T, initial_condition)
    p = (d.delta0, d.kappa_e, d.kappa_i, d.delta_b, d.g_b, M, E_of_t)























    t_start, t_end, _, _ = decode(pulse, u)
    tstops = ForwardDiff.value.(vcat(t_start, t_end))

    prob = ODEProblem(rhs_1st_order!, u0, d.timespan, p)
    sol = solve(prob, alg; reltol=reltol, abstol=abstol, save_everystep=false, save_start=false, tstops=tstops)
    _successful_solve(sol) || throw(PulseSolveFailed(sol.retcode))
    a, Sp, Sz = unpack_state_1st_order_u(sol.u[end], M)
    return a, collect(Sp), collect(Sz), d.Nj
end


function pulse_metrics(u::AbstractVector, pulse::CompositePulse, d; kwargs...)
    _, Sp, Sz, Nj = run_sim_1st_order_pure(u, pulse, d; kwargs...)
    T = eltype(u)
    weight = Nj ./ sum(Nj)
    Sz_fraction = real.(Sz) ./ (Nj ./ 2 .+ 1e-30)
    inversion = sum(weight .* clamp.((Sz_fraction .+ 1) ./ 2, zero(T), one(T)))
    Sp_abs = sqrt.(abs2.(Sp) .+ 1e-30)
    Sp_fraction = Sp_abs ./ (Nj ./ 2 .+ 1e-30)
    coherence = sum(weight .* clamp.(Sp_fraction, zero(T), one(T)))
    return inversion, coherence
end


function pulse_cost(u::AbstractVector, pulse::CompositePulse, d;
                     w_inv=1.0, w_coh=0.7, w_time=0.15, w_tmax=1.0, kwargs...)
    T = eltype(u)
    duration = pulse_duration(pulse, u)
    _, t_end, _, _ = decode(pulse, u)
    tmax_excess = max(t_end[end] - pulse.T_max, zero(T))
    tmax_penalty = w_tmax * (tmax_excess / pulse.T_max)^2

    inversion, coherence = try
        pulse_metrics(u, pulse, d; kwargs...)
    catch e
        e isa PulseSolveFailed || rethrow()
        infT = convert(T, Inf)
        nanT = convert(T, NaN)
        return infT, nanT, nanT, duration
    end

    cost = -w_inv * inversion - w_coh * coherence + w_time * (duration / pulse.T_max) + tmax_penalty
    return cost, inversion, coherence, duration
end


mutable struct AdamState
    m::Vector{Float64}
    v::Vector{Float64}
    t::Int
end

AdamState(n::Integer) = AdamState(zeros(n), zeros(n), 0)


function adam_step!(u::AbstractVector, grad::AbstractVector, state::AdamState;
                     lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8)
    state.t += 1
    @inbounds for i in eachindex(u)
        state.m[i] = beta1 * state.m[i] + (1 - beta1) * grad[i]
        state.v[i] = beta2 * state.v[i] + (1 - beta2) * grad[i]^2
        m_hat = state.m[i] / (1 - beta1^state.t)
        v_hat = state.v[i] / (1 - beta2^state.t)
        u[i] -= lr * m_hat / (sqrt(v_hat) + eps)
    end
    return u
end



function run_local_adam(u_start::AbstractVector, pulse::CompositePulse, d, cost_kwargs::NamedTuple;
                         hop::Integer=0, num_epochs::Integer=30, patience::Integer=5, tol::Real=1e-3,
                         learning_rate::Real=0.05, label::AbstractString="", solve_kwargs...)
    u = copy(u_start)
    n = length(u)
    adam = AdamState(n)
    aux = Ref{NTuple{4,Float64}}((NaN, NaN, NaN, NaN))
    function cost_only(uu)
        c, inv_, coh_, dur_ = pulse_cost(uu, pulse, d; cost_kwargs..., solve_kwargs...)
        aux[] = (
            Float64(ForwardDiff.value(c)),
            Float64(ForwardDiff.value(inv_)),
            Float64(ForwardDiff.value(coh_)),
            Float64(ForwardDiff.value(dur_)),
        )
        return c
    end

    best_u = copy(u_start)
    best_cost, best_inv, best_coh, best_dur = Inf, 0.0, 0.0, 0.0
    epochs_since_improve = 0
    history = NamedTuple[]
    last_good_u = copy(u_start)
    last_good_grad = zeros(n)
    last_good_aux = (NaN, NaN, NaN, NaN)
    adam_m0 = zeros(n)
    adam_v0 = zeros(n)
    adam_t0 = 0
    lr = learning_rate
    just_reverted = false

    for epoch in 1:num_epochs
        t_wall = time()










        if just_reverted
            grad = last_good_grad
            cost, inv_, coh_, dur_ = last_good_aux
            adam.m .= adam_m0
            adam.v .= adam_v0
            adam.t = adam_t0
        else
            grad = ForwardDiff.gradient(cost_only, u)
            cost, inv_, coh_, dur_ = aux[]
        end

        if !isfinite(cost)
            epochs_since_improve += 1
            push!(history, (hop=hop, epoch=epoch, k=pulse.k, cost=cost, inversion=inv_,
                             coherence=coh_, duration=dur_, improved=false))
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
        last_good_aux = (cost, inv_, coh_, dur_)
        adam_m0 .= adam.m
        adam_v0 .= adam.v
        adam_t0 = adam.t
        just_reverted = false

        improved = cost < best_cost - tol
        if improved
            best_cost, best_u = cost, copy(u)
            best_inv, best_coh, best_dur = inv_, coh_, dur_
            epochs_since_improve = 0
        else
            epochs_since_improve += 1
        end

        push!(history, (hop=hop, epoch=epoch, k=pulse.k, cost=cost, inversion=inv_,
                         coherence=coh_, duration=dur_, improved=improved))

        adam_step!(u, grad, adam; lr=lr)

        elapsed = time() - t_wall
        mark = improved ? "*" : " "
        println(
            "$label epoch $(lpad(epoch, 3)): $(round(elapsed, digits=1))s  " *
            "cost=$(round(cost, digits=4)) $mark inversion=$(round(inv_, digits=4)) " *
            "coherence=$(round(coh_, digits=4)) duration=$(round(dur_, sigdigits=4))s"
        )

        if epochs_since_improve >= patience
            println("$label early stop at epoch $epoch (no improvement > $tol for $patience epochs)")
            break
        end
    end

    return best_u, best_cost, best_inv, best_coh, best_dur, history
end


function optimise_composite_pulse(
    k::Integer, n_coeff_A::Integer, n_coeff_f::Integer, d;
    num_epochs::Integer=30, learning_rate::Real=0.05, patience::Integer=5, tol::Real=1e-3,
    n_hops::Integer=3, hop_patience::Integer=2, hop_step_size::Real=0.5, temperature::Real=1.0,
    degree::Integer=3, taper_frac::Real=0.1, w_tmax::Real=1.0,
    seed::Integer=42, warm_start_u=nothing, label_prefix::AbstractString="", solve_kwargs...,
)
    pulse = CompositePulse(k, n_coeff_A, n_coeff_f, d; degree=degree, taper_frac=taper_frac)
    cost_kwargs = (w_tmax=w_tmax,)
    rng = Random.Xoshiro(seed)





    solve_settings = NamedTuple(kv for kv in pairs(solve_kwargs) if !(kv[2] isa Function))
    optimizer_settings = merge(
        (k=k, n_coeff_A=n_coeff_A, n_coeff_f=n_coeff_f, degree=degree, taper_frac=taper_frac,
         num_epochs=num_epochs, learning_rate=learning_rate, patience=patience, tol=tol,
         n_hops=n_hops, hop_patience=hop_patience, hop_step_size=hop_step_size, temperature=temperature,
         w_tmax=w_tmax, seed=seed),
        solve_settings,
    )

    println(
        "$(label_prefix)Optimising k=$k pulses, $(n_params(pulse)) raw parameters (ForwardDiff/Adam + " *
        "basin-hopping, physics: InhomogeneousSpinCavityDynamics.jl rhs_1st_order!) ..."
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
    initial_metrics = pulse_cost(u0, pulse, d; cost_kwargs..., solve_kwargs...)
    history = NamedTuple[]

    current_u, current_cost, _, _, _, hop0_history = run_local_adam(
        u0, pulse, d, cost_kwargs; hop=0, num_epochs, patience, tol, learning_rate,
        label="$(label_prefix)[hop 0]", solve_kwargs...
    )
    append!(history, hop0_history)
    global_best_u, global_best_cost = current_u, current_cost
    hops_since_improve = 0

    for hop in 1:(n_hops-1)
        perturbation = hop_step_size .* randn(rng, length(current_u))
        candidate_u0 = current_u .+ perturbation

        cand_u, cand_cost, _, _, _, hop_history = run_local_adam(
            candidate_u0, pulse, d, cost_kwargs;
            hop, num_epochs, patience, tol, learning_rate,
            label="$(label_prefix)[hop $hop]", solve_kwargs...,
        )
        append!(history, hop_history)

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

    final_metrics = pulse_cost(global_best_u, pulse, d; cost_kwargs..., solve_kwargs...)

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

