

function _canonical_kind_of_k(k::Integer)
    k == 1 && return :hs1
    k == 5 && return :corpse
    k == 7 && return :bb1
    return :random
end


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
