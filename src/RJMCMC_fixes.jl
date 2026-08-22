function _grow_pulse(pulse::CompositePulse, u::AbstractVector, d)
    _, _, raw_cA, raw_cf = unpack(pulse, u)
    gap, dur = _physical_gap_dur(pulse, u)
    k = pulse.k

    slot = argmax(gap)
    G = gap[slot]
    g_new = G / 3
    d_new = G / 3
    g_after = G - g_new - d_new

    new_gap = copy(gap)
    new_dur = copy(dur)
    new_gap[slot] = g_after
    insert!(new_gap, slot, g_new)
    insert!(new_dur, slot, d_new)

    new_pulse = CompositePulse(k + 1, pulse.n_coeff_A, pulse.n_coeff_f, d;
                                degree=pulse.degree, taper_frac=pulse.taper_frac)
    new_raw_gap, new_raw_dur = _encode_gap_dur(new_pulse, new_gap, new_dur)

    # FIX 1: Set to -4.6 to preserve gradient (~1% amplitude)
    silent_cA = fill(-4.6, pulse.n_coeff_A)
    zero_cf = zeros(pulse.n_coeff_f)
    raw_cA_mat = collect(Float64, raw_cA)
    raw_cf_mat = collect(Float64, raw_cf)
    new_raw_cA = hcat(raw_cA_mat[:, 1:slot-1], silent_cA, raw_cA_mat[:, slot:end])
    new_raw_cf = hcat(raw_cf_mat[:, 1:slot-1], zero_cf, raw_cf_mat[:, slot:end])

    new_u = pack(new_pulse, new_raw_gap, new_raw_dur, new_raw_cA, new_raw_cf)
    return new_pulse, new_u
end

function _shrink_pulse(pulse::CompositePulse, u::AbstractVector, d)
    pulse.k >= 2 || error("_shrink_pulse requires k >= 2, got k=$(pulse.k).")
    _, _, raw_cA, raw_cf = unpack(pulse, u)
    gap, dur = _physical_gap_dur(pulse, u)
    _, _, cA, _ = decode(pulse, u)
    k = pulse.k

    # FIX 2: Weight the mean amplitude by the physical duration to target the true smallest area
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
    raw_cA_mat = collect(Float64, raw_cA)
    raw_cf_mat = collect(Float64, raw_cf)
    new_raw_cA = raw_cA_mat[:, keep]
    new_raw_cf = raw_cf_mat[:, keep]

    new_u = pack(new_pulse, new_raw_gap, new_raw_dur, new_raw_cA, new_raw_cf)
    return new_pulse, new_u
end

"""
    _extract_physics_cost(cost, u, pulse, w_power)

Strips the dimension-dependent L2 power penalty from the total cost. 
Used exclusively to evaluate trans-dimensional Metropolis hops based purely on 
inversion, coherence, and duration physics, avoiding the 1/k mean dilution ratchet.
"""
function _extract_physics_cost(cost, u, pulse::CompositePulse, w_power)
    _, _, cA, _ = decode(pulse, u)
    normalized_cA = cA ./ pulse.amp_scale
    power_penalty = w_power * (sum(abs2, normalized_cA) / length(normalized_cA))
    return cost - power_penalty
end

function optimise_composite_pulse(
    k::Integer, n_coeff_A::Integer, n_coeff_f::Integer, d;
    num_epochs::Integer=30, learning_rate::Real=0.05, patience::Integer=5, tol::Real=1e-3,
    n_hops::Integer=3, hop_patience::Integer=2, hop_step_size::Real=0.5, temperature::Real=1.0,
    degree::Integer=3, taper_frac::Real=0.1, w_tmax::Real=1.0, w_power::Real=0.05,
    w_inv::Real=1.0, w_coh::Real=0.7, w_time::Real=0.15,
    seed::Integer=42, warm_start_u=nothing, label_prefix::AbstractString="", solve_kwargs...,
)
    _forbid_initial_condition(solve_kwargs)
    pulse = CompositePulse(k, n_coeff_A, n_coeff_f, d; degree=degree, taper_frac=taper_frac)
    cost_kwargs = (w_tmax=w_tmax, w_power=w_power, w_inv=w_inv, w_coh=w_coh, w_time=w_time)
    rng = Random.Xoshiro(seed)

    solve_settings = NamedTuple(kv for kv in pairs(solve_kwargs) if !(kv[2] isa Function))
    optimizer_settings = merge(
        (k=k, n_coeff_A=n_coeff_A, n_coeff_f=n_coeff_f, degree=degree, taper_frac=taper_frac,
         num_epochs=num_epochs, learning_rate=learning_rate, patience=patience, tol=tol,
         n_hops=n_hops, hop_patience=hop_patience, hop_step_size=hop_step_size, temperature=temperature,
         w_tmax=w_tmax, w_power=w_power, w_inv=w_inv, w_coh=w_coh, w_time=w_time, seed=seed),
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
    current_pulse = pulse
    
    # FIX 3: Initialize Physics Cost tracking
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

        cand_u, cand_cost, _, _, _, hop_history = run_local_adam(
            candidate_u0, candidate_pulse, d, cost_kwargs;
            hop, num_epochs, patience, tol, learning_rate,
            label="$(label_prefix)[hop $hop move=$move k=$(candidate_pulse.k)]", solve_kwargs...,
        )
        append!(history, hop_history)

        # FIX 3: Evaluate jumps using Physics Cost
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
            "($accept_str as new basin, phys_delta=$(round(delta, digits=4))) " *
            "global best raw_cost=$(round(global_best_cost, digits=4)), global best k=$(global_best_pulse.k)"
        )

        if hops_since_improve >= hop_patience
            println("$(label_prefix)Basin-hopping stopped early after $hop hops (no global improvement > $tol for $hop_patience hops).")
            break
        end
    end

    final_metrics = pulse_cost(global_best_u, global_best_pulse, d; cost_kwargs..., solve_kwargs...)

    println("$(label_prefix)Optimisation complete. Global best cost: $(round(global_best_cost, digits=4)), global best k=$(global_best_pulse.k).")
    return global_best_u, global_best_cost, global_best_pulse, u0, initial_metrics, history, final_metrics, optimizer_settings
end

