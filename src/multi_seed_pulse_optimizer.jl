# ============================================================
# MULTI-SEED RJMCMC WRAPPER
#
# A thin convenience layer over optimise_composite_pulse_over_k_rjmcmc
# (pulse_optimizer2_RJMCMC.jl): launches one independent trans-dimensional
# RJMCMC search per starting sub-pulse count k, each warm-started from the
# best available CANONICAL seed for that k, against the SAME ensemble `d`,
# then compiles every seed's own optimum into one table plus the overall
# best. No new optimisation logic lives here -- this file only builds the
# `specs` argument optimise_composite_pulse_over_k_rjmcmc already accepts
# and forwards everything else unchanged. Must be loaded after
# pulse_optimizer2_RJMCMC.jl (which must itself be loaded after
# pulse_optimizer2.jl -- see that file's own header for why).
#
# WHY THIS IS CPU-PARALLEL, NOT GPU-PARALLEL: every seed's own optimisation
# runs entirely through run_sim_1st_order_pure (pulse_optimizer2.jl), which
# is deliberately, explicitly CPU-only -- plain `Vector`/`Array` state, no
# `CuArray` anywhere, so `ForwardDiff.gradient` can differentiate through
# the ODE solve at all (see that function's own docstring: "No callback, no
# file I/O, no CUDA"). `rhs_1st_order!`, the shared RHS, has no CUDA
# references either. The ONLY CUDA-touching solver in this package,
# `run_sim_1st_order` (solver_1st_order.jl), is a different, non-
# differentiable, forward-only production path that ForwardDiff cannot run
# through and that the optimiser never calls -- so there is no GPU work
# for a multi-GPU dispatcher to schedule here, on THIS machine or any
# other; adding one would just detect however many GPUs are present and
# then use none of them. The real, already-CPU-bound parallelism lever is
# across SEEDS (one independent ForwardDiff+ODE optimisation per starting
# k), which `optimise_composite_pulse_over_k_rjmcmc` already dispatches
# via `Threads.@threads` when `threaded=true` (the default) and
# `Threads.nthreads() > 1` -- i.e. it already auto-detects and uses
# whatever Julia thread count the process was started with, no code
# change needed for that part either. What THIS wrapper adds on top,
# below, is the two things that were still manual: (1) telling the caller
# clearly whether that parallelism is actually active for the CURRENT
# process (Julia defaults to `-t 1`, i.e. serial, unless started with
# `-t auto`/`-t N`) rather than silently running serially with no
# explanation, and (2) automatically pinning BLAS to one thread per call
# while seeds run concurrently, since `Threads.nthreads()` Julia threads
# each potentially spawning BLAS's own (multi-threaded, `BLAS.
# get_num_threads()` by default) thread pool oversubscribes the machine's
# actual cores -- `optimise_composite_pulse_over_k_rjmcmc`'s own docstring
# already flags this as something callers should do by hand
# (`BLAS.set_num_threads(1)`); this wrapper just does it for them,
# automatically, only while it's actually running seeds in parallel, and
# restores the prior setting afterward.
# ============================================================

"""
    _canonical_kind_of_k(k::Integer) -> Symbol

Best available canonical [`seed_canonical`](@ref) kind for `k` sub-pulses:
`:hs1` for `k=1`, `:corpse` for `k=5`, `:bb1` for `k=7` -- the only k
values this package has a NAMED composite-pulse form for (see
canon_pulses.jl). Every other `k`, including EVERY EVEN `k` (`2`, `4`,
`6`, ...), falls back to `:random`: `seed_composite_with_ghosts`'
ghost-interleaved construction requires `k = 2*length(areas)-1`, which is
odd for any positive integer number of active pulses, so it can
structurally never produce an even-k seed at all, and `seed_canonical`
itself only ever dispatches to `:hs1`/`:corpse`/`:bb1`/`:random` -- there
is no other canonical form available to fall back to for an odd `k`
without a name either (e.g. `k=3`).
"""
function _canonical_kind_of_k(k::Integer)
    k == 1 && return :hs1
    k == 5 && return :corpse
    k == 7 && return :bb1
    return :random
end

"""
    multi_seed_optimise_pulse_rjmcmc(n_coeff_A, n_coeff_f, d;
        ks=1:7, threaded=true, manage_blas_threads=true,
        Omega_max=nothing, beta=nothing, mu=nothing,
        seed=42, optimizer_kwargs...)
        -> NamedTuple

Multi-seed wrapper over [`optimise_composite_pulse_over_k_rjmcmc`](@ref):
for the SAME ensemble `d`, launches one independent trans-dimensional
[`optimise_composite_pulse_rjmcmc`](@ref) run per starting sub-pulse count
`k` in `ks` (default `1:7`), each warm-started from the best available
CANONICAL seed for that `k` -- see [`_canonical_kind_of_k`](@ref) for
exactly which `k` get a genuine named seed (`1`, `5`, `7`) versus a
`:random` fallback (every other `k` in `ks`, including all even `k`) --
then compiles every run's own optimum into one table, plus the single
overall best.

Each seed's OWN run is itself trans-dimensional: `k` can drift away from
its starting value via `_grow_pulse`/`_shrink_pulse` during that run, so
`ks` sets the STARTING points explored, not a guarantee on the final `k`
values seen in the result (exactly as in
[`optimise_composite_pulse_over_k_rjmcmc`](@ref), which this wrapper
calls directly with `specs = [(k, _canonical_kind_of_k(k)) for k in ks]`
-- everything else, including `threaded`/`Omega_max`/`beta`/`mu`/`seed`/
`optimizer_kwargs`, is forwarded unchanged).

**Parallelism is CPU-thread-based, across seeds, and fully automatic --
see this file's own module docstring for why there is no GPU dispatch
here.** With `threaded=true` (the default), before delegating to
`optimise_composite_pulse_over_k_rjmcmc` (which does the actual
`Threads.@threads` dispatch) this function:
  - Compares `Threads.nthreads()` (how many Julia threads this PROCESS
    was started with) against `Sys.CPU_THREADS` (how many this MACHINE
    has) and `length(ks)` (how many seeds there are to run), and prints
    which regime applies -- parallel across `Threads.nthreads()` threads,
    or (if Julia was started without `-t`) serial, with the exact launch
    flag (`-t auto` or `-t N`) that would parallelise it. No code change
    is ever required to use however many threads ARE available: this
    detection re-runs every call, so the SAME call scales automatically
    from a 1-core laptop to a many-core server.
  - If `manage_blas_threads=true` (default) AND parallelism is actually
    active (`Threads.nthreads() > 1`), pins `LinearAlgebra.BLAS`
    to 1 thread per call for the duration of this call (restored
    afterward, even on error), since each of the `Threads.nthreads()`
    concurrently-running seeds could otherwise ALSO spawn BLAS's own
    (multi-threaded by default) pool underneath it, oversubscribing the
    machine's actual cores. Pass `manage_blas_threads=false` to manage
    this yourself instead (e.g. if you're already running other
    BLAS-heavy work in the same session and want to control this
    directly).

Returns the same NamedTuple shape as
[`optimise_composite_pulse_over_k_rjmcmc`](@ref): `best_kind`, `best_k`,
`best_u`, `best_cost`, `pulse`, `u0`, `initial_metrics`, `history`,
`final_metrics`, `optimizer_settings` (the winning seed's own run), and
`per_k` (one row per starting `k` in `ks`, each carrying its own `k`
(final, possibly hopped-to), `start_k` (the `ks` value it began from),
`pulse`, `best_u`, `best_cost`, etc. -- see that function's own docstring
for the full per-row shape).

`ks` need not be `1:7`, need not be contiguous, and need not be sorted --
any collection of distinct positive integers works (duplicates raise the
same "Duplicate k" error `optimise_composite_pulse_over_k_rjmcmc` already
raises, since two entries would build the exact same starting spec twice).
"""
function multi_seed_optimise_pulse_rjmcmc(
    n_coeff_A::Integer, n_coeff_f::Integer, d;
    ks=1:7,
    threaded::Bool=true,
    manage_blas_threads::Bool=true,
    Omega_max=nothing,
    beta=nothing,
    mu=nothing,
    seed::Integer=42,
    optimizer_kwargs...,
)
    ks_vec = collect(ks)
    n_seeds = length(ks_vec)
    n_julia_threads = Threads.nthreads()
    n_cpus = Sys.CPU_THREADS
    parallel_active = threaded && n_julia_threads > 1 && n_seeds > 1

    if parallel_active
        println(
            "multi_seed_optimise_pulse_rjmcmc: $n_seeds seed(s) requested, " *
            "$n_julia_threads Julia thread(s) available (this machine has $n_cpus CPU core(s)) " *
            "-- dispatching seeds across threads."
        )
    elseif threaded && n_seeds > 1
        println(
            "multi_seed_optimise_pulse_rjmcmc: $n_seeds seed(s) requested but Julia is running " *
            "with only 1 thread (this machine has $n_cpus CPU core(s) available) -- seeds will run " *
            "SERIALLY. Restart Julia with `-t auto` (or `-t $n_cpus`) to parallelise across seeds; " *
            "no code change is needed, this function auto-detects whatever thread count is active."
        )
    end

    manage_blas = manage_blas_threads && parallel_active
    old_blas_threads = manage_blas ? LinearAlgebra.BLAS.get_num_threads() : nothing
    if manage_blas
        println(
            "multi_seed_optimise_pulse_rjmcmc: pinning BLAS to 1 thread per call while $n_julia_threads " *
            "seeds run concurrently (was $old_blas_threads) to avoid oversubscribing $n_cpus cores; " *
            "pass manage_blas_threads=false to manage this yourself."
        )
        LinearAlgebra.BLAS.set_num_threads(1)
    end

    try
        specs = [(k, _canonical_kind_of_k(k)) for k in ks_vec]
        return optimise_composite_pulse_over_k_rjmcmc(
            n_coeff_A, n_coeff_f, d;
            specs=specs, threaded=threaded, Omega_max=Omega_max, beta=beta, mu=mu, seed=seed,
            optimizer_kwargs...,
        )
    finally
        manage_blas && LinearAlgebra.BLAS.set_num_threads(old_blas_threads)
    end
end
