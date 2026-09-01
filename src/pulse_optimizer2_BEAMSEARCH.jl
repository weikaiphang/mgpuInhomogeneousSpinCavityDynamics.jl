# ============================================================
# DETERMINISTIC BEAM-SEARCH K-HOPPING EXTENSION
#
# Like pulse_optimizer2_RJMCMC.jl, this file is a THIN, ADDITIVE extension
# that does NOT redefine any shared physics/local-descent machinery. It
# reuses, by direct reference:
#   from pulse_optimizer2.jl -- `optimise_composite_pulse` (the fixed-k
#     Adam + basin-hopping continuous optimiser), `pulse_cost`,
#     `CompositePulse`, `_forbid_initial_condition`, `_normalise_k_specs`,
#     `k_of_seed_kind`;
#   from pulse_optimizer2_RJMCMC.jl -- `_grow_pulse`/`_shrink_pulse` (the
#     longest-silence-gap / smallest-area k-change seeding heuristics) and
#     `_extract_physics_cost` (the k-invariant cost used for all cross-k
#     comparisons, for exactly the power-penalty-dilution reason documented
#     on `_extract_physics_cost`'s own docstring in that file).
# Must therefore be loaded AFTER both of those files, into the SAME
# namespace, e.g. `using InhomogeneousSpinCavityDynamics` (which already
# includes pulse_optimizer2.jl and pulse_optimizer2_RJMCMC.jl) followed by
# `Base.include(InhomogeneousSpinCavityDynamics, "pulse_optimizer2_BEAMSEARCH.jl")`.
# This file is NOT added to the package's own include list and does not
# modify pulse_optimizer2.jl, pulse_optimizer2_RJMCMC.jl,
# multi_seed_pulse_optimizer.jl, or InhomogeneousSpinCavityDynamics.jl --
# every piece of logic a deterministic beam search needs beyond what those
# files already provide lives entirely in THIS file.
#
# WHAT THIS FILE ADDS: a genuinely different outer search strategy from
# pulse_optimizer2_RJMCMC.jl's single-walker, Metropolis-accepted k-hopping.
# Here, every hop-equivalent ("stage") deterministically tries ALL of:
# stay (re-optimise every currently live branch at its own k), grow (spawn
# one new k+1 branch from whichever live branch currently has the HIGHEST
# k), and shrink (spawn one new k-1 branch from whichever live branch
# currently has the LOWEST k) -- never a probabilistic choice between them,
# and never a Metropolis accept/reject on the result. Instead, after every
# stage, all live branches (refreshed "stay" branches plus any newly grown/
# shrunk ones) are ranked by `_extract_physics_cost` and only the best
# `beam_width` survive to the next stage -- classic beam search, with `k`
# as the dimension being searched breadth-first rather than depth-first.
# See `optimise_composite_pulse_beamsearch`'s own docstring for the full
# per-stage mechanics, stopping rule, and the k-distinctness invariant
# that makes pruning safe without any branch-merging logic.
#
# WHY CPU-THREAD PARALLEL, NOT GPU: exactly the same reason
# multi_seed_pulse_optimizer.jl's own module docstring gives at length --
# every branch's own continuous optimisation runs through
# `run_sim_1st_order_pure` (pulse_optimizer2.jl), which is deliberately,
# explicitly CPU-only (plain `Vector`/`Array` state, no `CuArray`
# anywhere) so `ForwardDiff.gradient` can differentiate through the ODE
# solve at all. There is no GPU work anywhere in this call chain for a
# multi-GPU dispatcher to schedule, on this machine or any other -- this
# was confirmed explicitly before writing this file, rather than adding
# CUDA device-pinning scaffolding that would sit inertly on top of a
# CPU-only pipeline. The real, already-CPU-bound parallelism lever is
# across INDEPENDENT branches, and this file exploits it at TWO levels:
#   (1) within one beam search, across the several sibling branches
#       (stay/grow/shrink children) that must each run their own
#       `optimise_composite_pulse` call in a given stage -- see
#       `optimise_composite_pulse_beamsearch`'s own `threaded` kwarg;
#   (2) across several independent beam searches started from different
#       starting `k0` -- see `optimise_composite_pulse_over_k_beamsearch`,
#       mirroring `optimise_composite_pulse_over_k`/`_rjmcmc`'s own
#       per-k `Threads.@threads` dispatch one level further out.
# Nesting two `Threads.@threads` loops (level 2 outer, level 1 inner)
# would oversubscribe `Threads.nthreads()`, so `optimise_composite_pulse_over_k_beamsearch`
# disables level 1 inside each of its own dispatched calls whenever IT is
# the level actually spreading work across threads (and vice versa) --
# see that function's own docstring. Both levels pin `LinearAlgebra.BLAS`
# to 1 thread for their own duration while dispatching in parallel, same
# discipline multi_seed_pulse_optimizer.jl already applies, via this
# file's own small `_with_blas_pinned` helper (kept local rather than
# reused from that file, since duplicating one ~10-line try/finally idiom
# carries none of the ~500-line-copy drift risk pulse_optimizer2_RJMCMC.jl's
# own header warns about, and avoids adding a load-order dependency on
# multi_seed_pulse_optimizer.jl for something this small).
# ============================================================

"""
    BeamBranch

One live (or formerly-live, once logged) node in the beam-search tree.
`id` is a globally unique, monotonically increasing integer assigned the
moment a branch is CREATED (including every "stay" refinement -- a stay
child gets a fresh `id`, not its parent's, so `branch_log` rows form a
proper parent-pointer chain regardless of move type; walk `parent_id`
backward to reconstruct any branch's full stage-by-stage trajectory,
stay-refinements included). `k`/`pulse`/`u`/`cost` are this branch's own
best point from its OWN `optimise_composite_pulse` call (fixed-`k` Adam +
basin-hopping); `phys_cost` is `_extract_physics_cost` of that same point
-- ALL cross-branch comparisons (pruning, extreme-badness, global best)
use `phys_cost`, never raw `cost`, for the same k-dilution reason
`pulse_optimizer2_RJMCMC.jl`'s `_extract_physics_cost` docstring explains
at length. `stage`/`move`/`parent_id` record how this branch's CURRENT
state was produced (`move ∈ (:root, :stay, :up, :down)`, `parent_id=0`
only for the root).
"""
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

"""
    _with_blas_pinned(f, active::Bool)

Runs `f()` with `LinearAlgebra.BLAS` pinned to 1 thread for the duration
(restored afterward, even on error) when `active`; otherwise just runs
`f()` unchanged. `active` should be true exactly when the caller is about
to dispatch more than one concurrently-running `ForwardDiff`/ODE job across
Julia threads -- each such job could otherwise ALSO spawn BLAS's own
(multi-threaded by default) pool underneath it, oversubscribing the
machine's actual cores, the same concern `multi_seed_pulse_optimizer.jl`
already documents and handles for its own single level of parallelism.
"""
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

"""
    optimise_composite_pulse_beamsearch(k0, n_coeff_A, n_coeff_f, d;
        beam_width=3, max_stages=6, stage_patience=2, extreme_factor=5.0,
        new_branch_epoch_multiplier=2.0, new_branch_hop_multiplier=1.5,
        k_max=nothing, num_epochs=30, learning_rate=0.05, patience=5,
        tol=1e-3, n_hops=3, hop_patience=2, hop_step_size=0.5,
        temperature=1.0, degree=3, taper_frac=0.1, w_tmax=1.0, w_power=0.05,
        target_F=1.0, w_time=0.15, seed=42,
        warm_start_u=nothing, threaded=true, label_prefix="", solve_kwargs...)
        -> NamedTuple

Deterministic beam search over sub-pulse count `k`, breadth-first rather
than pulse_optimizer2_RJMCMC.jl's single-walker Metropolis hopping:

  0. STAGE 0 (root): one branch at the input `k0`, seeded from
     `warm_start_u` if given, else a fresh `initial_guess` (exactly as
     `optimise_composite_pulse` itself defaults), run through ONE
     `optimise_composite_pulse` call (fixed-`k0` Adam + basin-hopping,
     `num_epochs`/`n_hops` budget below).

  1. EVERY subsequent stage, from the current set of live branches
     (initially just the root), deterministically builds this stage's
     children -- no probabilistic move choice, no Metropolis test:
       - STAY: every live branch spawns exactly one stay child, seeded by
         perturbing that branch's own current `u` with isotropic Gaussian
         noise (`hop_step_size`, in raw/pre-softplus units -- same noise
         scale pulse_optimizer2_RJMCMC.jl's own "stay" hop uses), same `k`.
       - GROW (k+1): the live branch with the CURRENTLY HIGHEST `k` spawns
         one up child, seeded via `_grow_pulse` (splice a near-zero
         sub-pulse into the longest silence gap; no noise added on top).
         Skipped if `k_max !== nothing` and that branch is already at
         `k_max`.
       - SHRINK (k-1): the live branch with the CURRENTLY LOWEST `k`
         spawns one down child, seeded via `_shrink_pulse` (drop the
         smallest-area sub-pulse; no noise added on top). Skipped when
         that branch's `k < 2` (`_shrink_pulse` requires `k >= 2`).
     Every child then runs its OWN `optimise_composite_pulse` call at its
     own (possibly new) `k`, warm-started from that seed -- i.e. each
     child gets the FULL "Adam + local basin-hopping" continuous
     optimiser pulse_optimizer2.jl already implements, not a single Adam
     descent; this file adds no new local-descent code of its own. STAY
     children run at the "mature" budget (`num_epochs`/`n_hops`); GROW/
     SHRINK children run at a BOOSTED budget
     (`round(num_epochs*new_branch_epoch_multiplier)`,
     `round(n_hops*new_branch_hop_multiplier)`) since they start farther
     from a local optimum than a branch merely continuing its own
     refinement.

  2. PRUNING: after a stage's children all finish, every live branch is
     REPLACED by its own stay child (same lineage, new `id` -- see
     [`BeamBranch`](@ref)), and any grow/shrink children join as brand-new
     branches; the combined set is ranked by `_extract_physics_cost` and
     only the best `beam_width` survive into the next stage. Because GROW
     always targets `(current max k)+1` and SHRINK always targets
     `(current min k)-1` -- strictly outside every live branch's own `k`
     -- no two live branches can ever collide on the same `k`, at any
     stage, with or without pruning in between (pruning can only ever
     shrink the live-k set, never create a k that already existed
     elsewhere); this file relies on that invariant to prune by simple
     top-`beam_width` truncation, with no branch-merging logic, and
     asserts it after every stage as a loud, cheap sanity check rather
     than trusting it silently.

  3. GLOBAL BEST: tracked continuously across every branch ever evaluated
     (by `phys_cost`, independent of pruning -- a branch that gets pruned
     away this stage is still eligible to have been the global best), and
     is what gets returned; its own `k` need not match `k0`, nor the `k`
     of any branch still alive at the end.

  4. STOPPING: after each stage, this stage's newly created grow/shrink
     children (if any were possible to create) are checked against the
     global best: "significantly worse" means EVERY grow/shrink child
     this stage has `phys_cost > global_best.phys_cost +
     extreme_factor*tol` (vacuously true if neither move was possible,
     e.g. every live branch already at `k=1` with none `>= 2` and no
     `k_max` headroom). Separately, a stage counts as a "stay
     improvement" if ANY stay child improved on ITS OWN parent's
     `phys_cost` by more than `tol`; `stages_since_improve` counts
     consecutive stages without one. Stops (before `max_stages`) once
     BOTH: this stage's extremes are significantly worse AND
     `stages_since_improve >= stage_patience` -- mirroring the
     `hop_patience`/`tol` early-stop pattern already used by
     `optimise_composite_pulse`/`optimise_composite_pulse_rjmcmc`, just
     with two conditions instead of one since a beam has both a "still
     finding better k" axis and a "still improving within k" axis to
     exhaust. `max_stages` is the hard backstop regardless (analogous to
     `n_hops` there), so a run always terminates even if the heuristic
     stopping rule never fires.

`d`, `degree`/`taper_frac`, and every `w_*`/`target_F` cost-weight keyword
are forwarded to `CompositePulse`/`pulse_cost` exactly as in
`optimise_composite_pulse`/`optimise_composite_pulse_rjmcmc` -- same
defaults, same meaning. `num_epochs`/`learning_rate`/`patience`/`tol`/
`n_hops`/`hop_patience`/`hop_step_size`/`temperature` are this run's
MATURE (stay-child) budget; see point 1 above for the newborn multiplier
on `num_epochs`/`n_hops`. Do not pass `initial_condition` -- forbidden the
same way as elsewhere in this package.

`threaded`: when true (default) and `Threads.nthreads() > 1` and a given
stage has more than one child to run, that stage's children are dispatched
across Julia threads (`BLAS` pinned to 1 thread meanwhile, see
[`_with_blas_pinned`](@ref)) -- see this file's own module docstring for
why this is CPU-thread, not GPU, parallelism, and why
`optimise_composite_pulse_over_k_beamsearch` (the level above this one)
passes `threaded=false` into calls of THIS function whenever it is itself
the level spreading work across threads.

Returns a `NamedTuple`: `best_u`/`best_cost`/`best_k`/`pulse` (the global
best, `pulse` matching `best_u`'s own `k`), `u0`/`initial_metrics` (the
root branch's own starting point/metrics -- the full `pulse_cost` return
`(cost, inversion, silencing, duration, coherence, field_amp,
weak_seed_retention)`, everything after `silencing` diagnostic-only),
`final_metrics` (same shape,
recomputed fresh at `best_u`, NOT necessarily equal to the global best
branch's own cached `cost` since that branch's own `optimise_composite_pulse`
call may have used different `solve_kwargs` -- mirrors
`optimise_composite_pulse_rjmcmc`'s own `final_metrics` convention),
`history` (every branch's own `run_local_adam` per-epoch log, concatenated,
each row additionally tagged `stage`/`move`/`branch_id`/`parent_id`),
`branch_log` (one row per branch EVER created --
`id`/`parent_id`/`stage`/`move`/`k`/`cost`/`phys_cost`/`inversion`/
`silencing`/`duration`/`alive_at_end` -- the full beam tree, independent of
`history`'s finer per-epoch granularity), and `optimizer_settings` (every
setting that affected this run, same spirit as the other two entry
points' own `optimizer_settings`).
"""
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
        else # :down
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

"""
    optimise_composite_pulse_over_k_beamsearch(n_coeff_A, n_coeff_f, d;
        kinds=(:hs1, :corpse, :bb1), specs=nothing, threaded=true,
        Omega_max=nothing, beta=nothing, mu=nothing, seed=42,
        beamsearch_kwargs...)
        -> NamedTuple

Second, OUTER level of CPU-thread parallelism (see this file's own module
docstring): launches one independent
[`optimise_composite_pulse_beamsearch`](@ref) run per starting `k0`, each
warm-started from a canonical [`seed_canonical`](@ref) seed for that `k0`
-- same `kinds`/`specs`/duplicate-`k0`/`Omega_max`/`beta`/`mu` semantics as
[`optimise_composite_pulse_over_k`](@ref) and
[`optimise_composite_pulse_over_k_rjmcmc`](@ref) (`specs=((k0, kind), ...)`
to choose both explicitly, `:random` for a fresh `initial_guess`). Each
spec's own beam search can still drift its OWN branches away from `k0` via
grow/shrink stages, exactly as `optimise_composite_pulse_over_k_rjmcmc`'s
per-spec RJMCMC runs can -- `k0` only fixes where THAT spec's search tree
starts growing from, not where its winning branch ends up; `per_k0[i].best_k`
reports the winning branch's own actual `k`.

Threading is dispatched at EXACTLY ONE of the two available levels per
call, never both nested: when `threaded=true`, `Threads.nthreads() > 1`,
and more than one spec is given, THIS function spreads the `n` independent
beam searches across threads (`BLAS` pinned to 1 thread meanwhile) and
passes `threaded=false` into every one of those `optimise_composite_pulse_beamsearch`
calls, so each spec's own per-stage sibling-branch dispatch runs serially
within its own thread. Otherwise (a single spec, `threaded=false`, or
`Threads.nthreads() == 1`), no parallelism happens at this level and each
spec's OWN `threaded` behaviour (forwarded from `beamsearch_kwargs` as
ordinary keywords, since `threaded` is this function's own named
parameter and therefore never present in `beamsearch_kwargs` to begin
with) is left as `optimise_composite_pulse_beamsearch`'s own default,
i.e. that single run can still spread ITS sibling branches across
threads. Either way, only one `Threads.@threads` loop is ever active at
once, avoiding the oversubscription two nested loops would cause.

Do not pass `warm_start_u` (the seed is built per `k0`, same restriction as
the two existing `_over_k` entry points). Returns the winning spec's own
`optimise_composite_pulse_beamsearch` NamedTuple, merged with `start_k0`
and `kind`, plus `per_k0` (one such merged NamedTuple per spec, in input
order).
"""
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
