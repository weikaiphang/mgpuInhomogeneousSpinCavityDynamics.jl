


function _k_move_probabilities(k::Integer)
    p_stay, p_up, p_down = 0.6, 0.2, 0.2
    if k <= 1
        p_down = 0.0
    end
    total = p_stay + p_up + p_down
    return p_stay / total, p_up / total, p_down / total
end


function _choose_k_move(rng, k::Integer)
    p_stay, p_up, _ = _k_move_probabilities(k)
    r = rand(rng)
    r < p_stay && return :stay
    r < p_stay + p_up && return :up
    return :down
end


function _physical_gap_dur(pulse::CompositePulse, u::AbstractVector)
    t_start, t_end, _, _, _ = decode(pulse, u)
    k = pulse.k
    gap = Vector{Float64}(undef, k)
    dur = Vector{Float64}(undef, k)
    prev_end = 0.0
    @inbounds for i in 1:k
        gap[i] = t_start[i] - prev_end
        dur[i] = t_end[i] - t_start[i]
        prev_end = t_end[i]
    end
    return gap, dur
end


function _encode_gap_dur(new_pulse::CompositePulse, gap::AbstractVector, dur::AbstractVector)
    floor_gap = 1e-6 * new_pulse.gap_scale
    floor_dur = 1e-6 * new_pulse.dur_scale
    raw_gap = _softplus_inv.(max.(gap, floor_gap) ./ new_pulse.gap_scale)
    raw_dur = _softplus_inv.(max.(dur .- new_pulse.dur_floor, floor_dur) ./ new_pulse.dur_scale)
    return raw_gap, raw_dur
end


function _physical_cA(pulse::CompositePulse, u::AbstractVector)
    _, _, _, cA, _ = decode(pulse, u)
    return collect(Float64, cA)
end


function _encode_cA(new_pulse::CompositePulse, cA::AbstractMatrix)
    floor_amp = 1e-9 * new_pulse.amp_scale
    return _softplus_inv.(max.(cA, floor_amp) ./ new_pulse.amp_scale)
end


function _grow_pulse(pulse::CompositePulse, u::AbstractVector, d)
    _, _, raw_phi0, _, raw_cf = unpack(pulse, u)
    gap, dur = _physical_gap_dur(pulse, u)
    cA = _physical_cA(pulse, u)
    k = pulse.k

    new_pulse = CompositePulse(k + 1, pulse.n_coeff_A, pulse.n_coeff_f, d;
                                degree=pulse.degree, taper_frac=pulse.taper_frac)























    slot = argmax(gap)
    G = gap[slot]
    d_new = max(G / 3, new_pulse.dur_floor)
    g_new = max((G - d_new) / 2, 0.0)
    g_after = max(G - g_new - d_new, 0.0)

    new_gap = copy(gap)
    new_dur = copy(dur)
    new_gap[slot] = g_after
    insert!(new_gap, slot, g_new)
    insert!(new_dur, slot, d_new)

    new_raw_gap, new_raw_dur = _encode_gap_dur(new_pulse, new_gap, new_dur)

    silent_cA = fill(0.01 * new_pulse.amp_scale, pulse.n_coeff_A)
    zero_cf = zeros(pulse.n_coeff_f)
    new_cA = hcat(cA[:, 1:slot-1], silent_cA, cA[:, slot:end])
    new_raw_cA = _encode_cA(new_pulse, new_cA)
    raw_cf_mat = collect(Float64, raw_cf)
    new_raw_cf = hcat(raw_cf_mat[:, 1:slot-1], zero_cf, raw_cf_mat[:, slot:end])







    raw_phi0_vec = collect(Float64, raw_phi0)
    new_raw_phi0 = vcat(raw_phi0_vec[1:slot-1], 0.0, raw_phi0_vec[slot:end])

    new_u = pack(new_pulse, new_raw_gap, new_raw_dur, new_raw_phi0, new_raw_cA, new_raw_cf)
    return new_pulse, new_u
end


function _shrink_pulse(pulse::CompositePulse, u::AbstractVector, d)
    pulse.k >= 2 || error("_shrink_pulse requires k >= 2, got k=$(pulse.k).")
    _, _, raw_phi0, _, raw_cf = unpack(pulse, u)
    gap, dur = _physical_gap_dur(pulse, u)
    cA = _physical_cA(pulse, u)
    k = pulse.k

    amps = vec(sum(cA; dims=1)) ./ pulse.n_coeff_A
    areas = amps .* dur
    m = argmin(areas)

    new_gap = copy(gap)
    new_dur = copy(dur)
    if m < k
        new_gap[m+1] += new_gap[m] + new_dur[m]
    end
    deleteat!(new_gap, m)
    deleteat!(new_dur, m)

    new_pulse = CompositePulse(k - 1, pulse.n_coeff_A, pulse.n_coeff_f, d;
                                degree=pulse.degree, taper_frac=pulse.taper_frac)
    new_raw_gap, new_raw_dur = _encode_gap_dur(new_pulse, new_gap, new_dur)

    keep = [1:m-1; m+1:k]
    new_cA = cA[:, keep]
    new_raw_cA = _encode_cA(new_pulse, new_cA)
    raw_cf_mat = collect(Float64, raw_cf)
    new_raw_cf = raw_cf_mat[:, keep]
    new_raw_phi0 = collect(Float64, raw_phi0)[keep]

    new_u = pack(new_pulse, new_raw_gap, new_raw_dur, new_raw_phi0, new_raw_cA, new_raw_cf)
    return new_pulse, new_u
end


function _extract_physics_cost(cost, u::AbstractVector, pulse::CompositePulse, w_power)
    _, _, _, cA, _ = decode(pulse, u)
    normalized_cA = cA ./ pulse.amp_scale
    power_penalty = w_power * (sum(abs2, normalized_cA) / length(normalized_cA))
    return cost - power_penalty
end


function optimise_composite_pulse_rjmcmc(
    k::Integer, n_coeff_A::Integer, n_coeff_f::Integer, d;
    num_epochs::Integer=30, learning_rate::Real=0.05, patience::Integer=5, tol::Real=1e-3,
    n_hops::Integer=3, hop_patience::Integer=2, hop_step_size::Real=0.5, temperature::Real=1.0,
    degree::Integer=3, taper_frac::Real=0.1, w_tmax::Real=1.0, w_power::Real=0.05,
    target_F::Real=1.0, w_time::Real=0.15,
    seed::Integer=42, warm_start_u=nothing, label_prefix::AbstractString="",
    track::Symbol=:weak,
    anneal_direct_weights::Bool=true, hop0_phyonly::Bool=true,
    x_tune_alpha::Union{Nothing,Real}=_DEFAULT_X_TUNE_ALPHA, recalibrate_optima_x::Bool=true,
    I_min::Real=_DEFAULT_PENALTY_MIN, kappa_I::Real=_DEFAULT_PENALTY_KAPPA,
    S_min::Real=_DEFAULT_PENALTY_MIN, kappa_S::Real=_DEFAULT_PENALTY_KAPPA,
    solve_kwargs...,
)
    _forbid_initial_condition(solve_kwargs)
    _assert_track(track)
    pulse = CompositePulse(k, n_coeff_A, n_coeff_f, d; degree=degree, taper_frac=taper_frac)
    cost_kwargs = (w_tmax=w_tmax, w_power=w_power, target_F=target_F, w_time=w_time,
                   I_min=I_min, kappa_I=kappa_I, S_min=S_min, kappa_S=kappa_S, track=track)
    rng = Random.Xoshiro(seed)

    solve_settings = NamedTuple(kv for kv in pairs(solve_kwargs) if !(kv[2] isa Function))
    optimizer_settings = merge(
        (k=k, n_coeff_A=n_coeff_A, n_coeff_f=n_coeff_f, degree=degree, taper_frac=taper_frac,
         num_epochs=num_epochs, learning_rate=learning_rate, patience=patience, tol=tol,
         n_hops=n_hops, hop_patience=hop_patience, hop_step_size=hop_step_size, temperature=temperature,
         w_tmax=w_tmax, w_power=w_power, target_F=target_F, w_time=w_time, seed=seed,
         track=track,
         anneal_direct_weights=anneal_direct_weights, hop0_phyonly=hop0_phyonly, x_tune_alpha=x_tune_alpha,
         recalibrate_optima_x=recalibrate_optima_x,
         I_min=I_min, kappa_I=kappa_I, S_min=S_min, kappa_S=kappa_S),
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
        label="$(label_prefix)[hop 0]", anneal_direct_weights=anneal_direct_weights,
        hop0_phyonly=hop0_phyonly, x_tune_alpha=x_tune_alpha,
        solve_kwargs...
    )
    append!(history, hop0_history)
    x_tune_seed = 0.0
    current_pulse = pulse






    current_phys_cost = _extract_physics_cost(current_cost, current_u, current_pulse, w_power)
    global_best_pulse, global_best_u, global_best_cost = current_pulse, current_u, current_cost
    global_best_phys_cost = current_phys_cost
    hops_since_improve = 0

    for hop in 1:(n_hops-1)
        move = _choose_k_move(rng, current_pulse.k)
        if move === :stay
            candidate_pulse = current_pulse
            perturbation = hop_step_size .* randn(rng, length(current_u))
            candidate_u0 = current_u .+ perturbation
        elseif move === :up
            candidate_pulse, candidate_u0 = _grow_pulse(current_pulse, current_u, d)
        else
            candidate_pulse, candidate_u0 = _shrink_pulse(current_pulse, current_u, d)
        end











        hop_x_tune_alpha, hop_precal = if hop == 1 || recalibrate_optima_x
            (x_tune_alpha, nothing)
        else
            (nothing, x_tune_seed)
        end

        cand_u, cand_cost, _, _, _, hop_history = run_local_adam(
            candidate_u0, candidate_pulse, d, cost_kwargs;
            hop, num_epochs, patience, tol, learning_rate,
            label="$(label_prefix)[hop $hop move=$move k=$(candidate_pulse.k)]",
            anneal_direct_weights=anneal_direct_weights,
            x_tune_alpha=hop_x_tune_alpha, _precalibrated_x_tune=hop_precal,
            solve_kwargs...,
        )
        append!(history, hop_history)
        if hop == 1
            x_tune_seed = isempty(hop_history) ? 0.0 : hop_history[1].x_tune
        end
        cand_phys_cost = _extract_physics_cost(cand_cost, cand_u, candidate_pulse, w_power)

        if cand_phys_cost < global_best_phys_cost - tol
            global_best_pulse, global_best_u, global_best_cost = candidate_pulse, cand_u, cand_cost
            global_best_phys_cost = cand_phys_cost
            hops_since_improve = 0
        else
            hops_since_improve += 1
        end

        delta = cand_phys_cost - current_phys_cost
        accept = delta < 0.0 || rand(rng) < exp(-delta / max(temperature, 1e-12))
        if accept
            current_pulse, current_u, current_cost = candidate_pulse, cand_u, cand_cost
            current_phys_cost = cand_phys_cost
        end

        accept_str = accept ? "accepted" : "rejected"
        println(
            "$(label_prefix)hop $hop (move=$move, k=$(candidate_pulse.k)): local best raw_cost=$(round(cand_cost, digits=4)) " *
            "phys_cost=$(round(cand_phys_cost, digits=4)) " *
            "($accept_str as new basin, phys_delta=$(round(delta, digits=4))) " *
            "global best raw_cost=$(round(global_best_cost, digits=4)), global best k=$(global_best_pulse.k)"
        )

        if hops_since_improve >= hop_patience
            println("$(label_prefix)Basin-hopping stopped early after $hop hops (no global improvement > $tol for $hop_patience hops).")
            break
        end
    end

    final_metrics = pulse_cost(global_best_u, global_best_pulse, d; cost_kwargs..., solve_kwargs...)




    if track === :weak
        sk_final = _solver_kwargs(solve_kwargs)
        _, _, Sz_gf, Nj_gf = run_sim_1st_order_pure(
            global_best_u, global_best_pulse, d; sk_final..., initial_condition=:ground,
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

    println("$(label_prefix)Optimisation complete. Global best cost: $(round(global_best_cost, digits=4)), global best k=$(global_best_pulse.k).")
    return global_best_u, global_best_cost, global_best_pulse, u0, initial_metrics, history, final_metrics, optimizer_settings
end


function optimise_composite_pulse_over_k_rjmcmc(
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
        "optimise_composite_pulse_over_k_rjmcmc builds a per-k canonical seed; do not pass warm_start_u."
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
            optimise_composite_pulse_rjmcmc(
                k, n_coeff_A, n_coeff_f, d;
                optimizer_kwargs...,
                seed=seed + 1000 * Int(k),
                warm_start_u=u0,
                label_prefix=prefix,
            )
        optimizer_settings = merge(optimizer_settings, (seed_kind=kind,))
        return (
            kind=kind, k=pulse_out.k, start_k=Int(k),
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






        mark = r.start_k == best.start_k ? "*" : " "
        k_str = r.k == r.start_k ? "k=$(r.k)" : "k=$(r.start_k)->$(r.k)"
        println(
            "  $mark $k_str $(r.kind): cost=$(round(r.best_cost, digits=4))"
        )
    end
    println("  winner: k=$(best.k) (started k=$(best.start_k)) $(best.kind)  cost=$(round(best.best_cost, digits=4))")

    return (
        best_kind=best.kind, best_k=best.k,
        best_u=best.best_u, best_cost=best.best_cost, pulse=best.pulse,
        u0=best.u0, initial_metrics=best.initial_metrics, history=best.history,
        final_metrics=best.final_metrics, optimizer_settings=best.optimizer_settings,
        per_k=per_k,
    )
end
