struct BeamBranch
    id::Int
    k::Int
    pulse::CompositePulse
    u::Vector{Float64}
    cost::Float64
    phys_cost::Float64
    inversion::Float64
    silencing::Float64
    duration::Float64
    stage::Int
    move::Symbol
    parent_id::Int
end

_branch_log_row(b::BeamBranch) = (
    id=b.id, parent_id=b.parent_id, stage=b.stage, move=b.move, k=b.k,
    cost=b.cost, phys_cost=b.phys_cost, inversion=b.inversion,
    silencing=b.silencing, duration=b.duration,
)

function _with_blas_pinned(f::F, active::Bool) where {F}
    if !active
        return f()
    end
    old = LinearAlgebra.BLAS.get_num_threads()
    LinearAlgebra.BLAS.set_num_threads(1)
    try
        return f()
    finally
        LinearAlgebra.BLAS.set_num_threads(old)
    end
end

function optimise_composite_pulse_beamsearch(
    k0::Integer, n_coeff_A::Integer, n_coeff_f::Integer, d;
    beam_width::Integer=3, max_stages::Integer=6, stage_patience::Integer=2,
    extreme_factor::Real=5.0, new_branch_epoch_multiplier::Real=2.0,
    new_branch_hop_multiplier::Real=1.5, k_max::Union{Nothing,Integer}=nothing,
    num_epochs::Integer=30, learning_rate::Real=0.05, patience::Integer=5, tol::Real=1e-3,
    n_hops::Integer=3, hop_patience::Integer=2, hop_step_size::Real=0.5, temperature::Real=1.0,
    degree::Integer=3, taper_frac::Real=0.1, w_tmax::Real=1.0, w_power::Real=0.05,
    target_F::Real=1.0, w_time::Real=0.15,
    seed::Integer=42, warm_start_u=nothing, threaded::Bool=true,
    label_prefix::AbstractString="", solve_kwargs...,
)
    _forbid_initial_condition(solve_kwargs)
    k0 >= 1 || error("k0 must be a positive integer, got $k0.")
    beam_width >= 1 || error("beam_width must be >= 1, got $beam_width.")
    max_stages >= 0 || error("max_stages must be >= 0, got $max_stages.")
    stage_patience >= 1 || error("stage_patience must be >= 1, got $stage_patience.")
    k_max === nothing || k_max >= k0 || error("k_max=$k_max must be >= k0=$k0.")

    mature_budget = (epochs=num_epochs, hops=n_hops)
    newborn_budget = (
        epochs=max(1, round(Int, num_epochs * new_branch_epoch_multiplier)),
        hops=max(1, round(Int, n_hops * new_branch_hop_multiplier)),
    )

    _id_counter = Ref(0)
    next_id() = (_id_counter[] += 1; _id_counter[])

    function _run_child(job, stage::Integer)
        move, parent, id = job.move, job.parent, job.id
        if move === :stay
            k_child = parent.k
            rng_local = Random.Xoshiro(seed + 1000 * id)
            seed_u = parent.u .+ hop_step_size .* randn(rng_local, length(parent.u))
            budget = mature_budget
        elseif move === :up
            _, seed_u = _grow_pulse(parent.pulse, parent.u, d)
            k_child = parent.k + 1
            budget = newborn_budget
        else
            _, seed_u = _shrink_pulse(parent.pulse, parent.u, d)
            k_child = parent.k - 1
            budget = newborn_budget
        end
        prefix = "$(label_prefix)[stage $stage id=$id move=$move k=$k_child <- id=$(parent.id) k=$(parent.k)] "
        best_u, best_cost, pulse_out, _, _, history, final_metrics, _ = optimise_composite_pulse(
            k_child, n_coeff_A, n_coeff_f, d;
            num_epochs=budget.epochs, learning_rate=learning_rate, patience=patience, tol=tol,
            n_hops=budget.hops, hop_patience=hop_patience, hop_step_size=hop_step_size, temperature=temperature,
            degree=degree, taper_frac=taper_frac, w_tmax=w_tmax, w_power=w_power,
            target_F=target_F, w_time=w_time,
            seed=seed + 1000 * id, warm_start_u=seed_u, label_prefix=prefix, solve_kwargs...,
        )
        phys_cost = _extract_physics_cost(best_cost, best_u, pulse_out, w_power)
        branch = BeamBranch(
            id, k_child, pulse_out, best_u, best_cost, phys_cost,
            final_metrics[2], final_metrics[3], final_metrics[4], stage, move, parent.id,
        )
        tagged_history = [merge(row, (stage=stage, move=move, branch_id=id, parent_id=parent.id)) for row in history]
        return branch, tagged_history
    end

    println(
        "$(label_prefix)Beam search starting k0=$k0, beam_width=$beam_width, max_stages=$max_stages " *
        "(ForwardDiff/Adam + basin-hopping per branch, physics: InhomogeneousSpinCavityDynamics.jl rhs_1st_order!) ..."
    )

    root_prefix = "$(label_prefix)[stage 0 root k0=$k0] "
    root_u, root_cost, root_pulse, u0, initial_metrics, history0, root_metrics, _ = optimise_composite_pulse(
        k0, n_coeff_A, n_coeff_f, d;
        num_epochs=mature_budget.epochs, learning_rate=learning_rate, patience=patience, tol=tol,
        n_hops=mature_budget.hops, hop_patience=hop_patience, hop_step_size=hop_step_size, temperature=temperature,
        degree=degree, taper_frac=taper_frac, w_tmax=w_tmax, w_power=w_power,
        target_F=target_F, w_time=w_time,
        seed=seed, warm_start_u=warm_start_u, label_prefix=root_prefix, solve_kwargs...,
    )
    root_id = next_id()
    root_phys_cost = _extract_physics_cost(root_cost, root_u, root_pulse, w_power)
    root = BeamBranch(
        root_id, k0, root_pulse, root_u, root_cost, root_phys_cost,
        root_metrics[2], root_metrics[3], root_metrics[4], 0, :root, 0,
    )

    all_history = [merge(row, (stage=0, move=:root, branch_id=root_id, parent_id=0)) for row in history0]
    branch_log = [_branch_log_row(root)]
    global_best = root
    alive = BeamBranch[root]
    stages_since_improve = 0

    for stage in 1:max_stages
        jobs = NamedTuple[]
        for branch in alive
            push!(jobs, (id=next_id(), move=:stay, parent=branch))
        end
        minb = alive[argmin([b.k for b in alive])]
        maxb = alive[argmax([b.k for b in alive])]
        if minb.k >= 2
            push!(jobs, (id=next_id(), move=:down, parent=minb))
        end
        if k_max === nothing || maxb.k < k_max
            push!(jobs, (id=next_id(), move=:up, parent=maxb))
        end

        njobs = length(jobs)
        use_threads = threaded && Threads.nthreads() > 1 && njobs > 1
        results = Vector{Tuple{BeamBranch,Vector{NamedTuple}}}(undef, njobs)
        _with_blas_pinned(use_threads) do
            if use_threads
                Threads.@threads for i in 1:njobs
                    results[i] = _run_child(jobs[i], stage)
                end
            else
                for i in 1:njobs
                    results[i] = _run_child(jobs[i], stage)
                end
            end
        end

        stay_improved = false
        extreme_phys_costs = Float64[]
        new_alive = BeamBranch[]
        for (job, (branch, hist)) in zip(jobs, results)
            append!(all_history, hist)
            push!(branch_log, _branch_log_row(branch))
            push!(new_alive, branch)
            if job.move === :stay
                branch.phys_cost < job.parent.phys_cost - tol && (stay_improved = true)
            else
                push!(extreme_phys_costs, branch.phys_cost)
            end
            branch.phys_cost < global_best.phys_cost - tol && (global_best = branch)
        end

        sort!(new_alive, by=b -> b.phys_cost)
        alive = length(new_alive) > beam_width ? new_alive[1:beam_width] : new_alive
        ks_check = [b.k for b in alive]
        length(unique(ks_check)) == length(ks_check) || error(
            "Internal invariant violated: duplicate k among alive beam branches at stage $stage: $ks_check."
        )

        extremes_bad = isempty(extreme_phys_costs) ? true :
                       all(pc -> pc > global_best.phys_cost + extreme_factor * tol, extreme_phys_costs)
        stages_since_improve = stay_improved ? 0 : stages_since_improve + 1

        println(
            "$(label_prefix)stage $stage: $njobs children, alive after pruning: " *
            join(["k=$(b.k)(phys=$(round(b.phys_cost, digits=4)))" for b in alive], ", ") *
            "; global best phys_cost=$(round(global_best.phys_cost, digits=4)) at k=$(global_best.k)."
        )

        if extremes_bad && stages_since_improve >= stage_patience
            println(
                "$(label_prefix)Beam search stopped early after stage $stage (extreme branches not " *
                "beating global best by > $(extreme_factor)*tol=$(extreme_factor*tol), and no stay-branch " *
                "improvement for $stages_since_improve stages >= stage_patience=$stage_patience)."
            )
            break
        end
    end

    final_ids = Set(b.id for b in alive)
    branch_log = [merge(row, (alive_at_end=(row.id in final_ids),)) for row in branch_log]

    cost_kwargs = (w_tmax=w_tmax, w_power=w_power, target_F=target_F, w_time=w_time)
    final_metrics = pulse_cost(global_best.u, global_best.pulse, d; cost_kwargs..., solve_kwargs...)

    solve_settings = NamedTuple(kv for kv in pairs(solve_kwargs) if !(kv[2] isa Function))
    optimizer_settings = merge(
        (k0=k0, n_coeff_A=n_coeff_A, n_coeff_f=n_coeff_f, degree=degree, taper_frac=taper_frac,
         beam_width=beam_width, max_stages=max_stages, stage_patience=stage_patience,
         extreme_factor=extreme_factor, new_branch_epoch_multiplier=new_branch_epoch_multiplier,
         new_branch_hop_multiplier=new_branch_hop_multiplier, k_max=k_max,
         num_epochs=num_epochs, learning_rate=learning_rate, patience=patience, tol=tol,
         n_hops=n_hops, hop_patience=hop_patience, hop_step_size=hop_step_size, temperature=temperature,
         w_tmax=w_tmax, w_power=w_power, target_F=target_F, w_time=w_time, seed=seed),
        solve_settings,
    )

    println(
        "$(label_prefix)Beam search complete. Global best cost=$(round(global_best.cost, digits=4)) " *
        "at k=$(global_best.k) (branch id=$(global_best.id))."
    )

    return (
        best_u=global_best.u, best_cost=global_best.cost, best_k=global_best.k, pulse=global_best.pulse,
        u0=u0, initial_metrics=initial_metrics, final_metrics=final_metrics,
        history=all_history, branch_log=branch_log, optimizer_settings=optimizer_settings,
    )
end

function optimise_composite_pulse_over_k_beamsearch(
    n_coeff_A::Integer, n_coeff_f::Integer, d;
    kinds=(:hs1, :corpse, :bb1),
    specs=nothing,
    threaded::Bool=true,
    Omega_max=nothing,
    beta=nothing,
    mu=nothing,
    seed::Integer=42,
    beamsearch_kwargs...,
)
    :warm_start_u in keys(beamsearch_kwargs) && error(
        "optimise_composite_pulse_over_k_beamsearch builds a per-k0 canonical seed; do not pass warm_start_u."
    )
    _forbid_initial_condition(beamsearch_kwargs)

    job_specs = _normalise_k_specs(kinds, specs)
    isempty(job_specs) && error("No (k0, kind) specs to optimise.")
    seen = Dict{Int,Symbol}()
    for (k, kind) in job_specs
        k isa Integer && k >= 1 || error("k0 must be a positive integer, got $k.")
        kind isa Symbol || error("kind must be a Symbol, got $kind.")
        haskey(seen, k) && error("Duplicate k0=$k (kinds $(seen[k]) and $kind). Each starting k0 can run once.")
        seen[k] = kind
        if kind !== :random
            k_of_seed_kind(kind) == k || error("kind $kind requires k=$(k_of_seed_kind(kind)), got k0=$k.")
        end
    end

    n = length(job_specs)
    nthreads = Threads.nthreads()
    use_threads = threaded && nthreads > 1 && n > 1
    println(
        "Beam search over starting k0: $n independent beam search(es) " *
        (use_threads ? "on $nthreads threads" : "serially") *
        ". kinds=$(collect(spec[2] for spec in job_specs))."
    )

    function run_spec(spec)
        k0, kind = spec
        prefix = "[$kind k0=$k0] "
        deg = get(beamsearch_kwargs, :degree, 3)
        tfrac = get(beamsearch_kwargs, :taper_frac, 0.1)
        pulse_seed = CompositePulse(k0, n_coeff_A, n_coeff_f, d; degree=deg, taper_frac=tfrac)
        Ω = Omega_max === nothing ? pulse_seed.amp_scale : Omega_max
        u0 = seed_canonical(pulse_seed, kind; Omega_max=Ω, beta=beta, mu=mu, seed=seed)
        result = optimise_composite_pulse_beamsearch(
            k0, n_coeff_A, n_coeff_f, d;
            beamsearch_kwargs...,
            seed=seed + 1000 * Int(k0),
            warm_start_u=u0,
            label_prefix=prefix,
            threaded=use_threads ? false : threaded,
        )
        return merge(result, (start_k0=Int(k0), kind=kind))
    end

    per_k0 = Vector{NamedTuple}(undef, n)
    _with_blas_pinned(use_threads) do
        if use_threads
            Threads.@threads for i in 1:n
                per_k0[i] = run_spec(job_specs[i])
            end
        else
            for i in 1:n
                per_k0[i] = run_spec(job_specs[i])
            end
        end
    end

    best = per_k0[1]
    for r in per_k0
        if r.best_cost < best.best_cost
            best = r
        end
    end

    println("Beam search over k0 complete.")
    for r in per_k0
        mark = r.start_k0 == best.start_k0 ? "*" : " "
        println("  $mark k0=$(r.start_k0) $(r.kind) -> best k=$(r.best_k): cost=$(round(r.best_cost, digits=4))")
    end
    println("  winner: k0=$(best.start_k0) $(best.kind) -> k=$(best.best_k)  cost=$(round(best.best_cost, digits=4))")

    return merge(best, (per_k0=per_k0,))
end
