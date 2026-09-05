

function assemble_problem(M::Integer,
                          delta_b::AbstractVector,
                          g_b::AbstractVector,
                          Nj::AbstractVector,
                          E_of_t,
                          delta0::Real, kappa_e::Real, kappa_i::Real;
                          nshards::Union{Nothing,Integer} = nothing,
                          device_ids::Union{Nothing,AbstractVector} = nothing,
                          integrator::Symbol = :ck45,
                          atol::Real = 1e-8,
                          rtol::Real = 1e-8,
                          threaded::Union{Nothing,Bool} = nothing,
                          T::Type = Float64,
                          verbose::Bool = true)

    length(delta_b) == M || error("length(delta_b) = $(length(delta_b)) ≠ M = $M.")
    length(g_b)     == M || error("length(g_b) = $(length(g_b)) ≠ M = $M.")
    length(Nj)      == M || error("length(Nj) = $(length(Nj)) ≠ M = $M.")

    devs = resolve_devices(nshards, device_ids)
    ns = length(devs)
    part = EnsemblePartition(M, ns)
    exec = Executor(ns; threaded = threaded)
    nreg = register_count(integrator)

    pfrac = enable_peer_access!(devs)
    peer_ok = ns <= 1 || pfrac >= 1
    if ns > 1 && !peer_ok && verbose
        @warn "Peer-to-peer access is not available on every GPU pair; " *
              "row-sum exchange will use NCCL if possible, else the host." pfrac
    end

    comms = nothing
    if ns > 1 && _NCCL !== nothing
        try
            comms = _NCCL.Communicators(devs)
        catch err
            verbose && @warn "NCCL communicators unavailable; using P2P or host row-sum exchange" exception=err
            comms = nothing
        end
    end

    if verbose
        println("Assembling $ns shard(s) over M = $M ensemble bins:")
        print(device_summary(devs))
        println(memory_report(M, ns; integrator = integrator, T = T))
    end

    shards = build_shards(T, M, part, devs, delta_b, g_b, Nj, nreg)





    local hostbuf
    try
        CUDA.device!(devs[1])
        hostbuf = CUDA.pin(Vector{Complex{T}}(undef, 3M))
    catch


        free_shards!(shards)
        rethrow()
    end

    return MGPUProblem{T,typeof(E_of_t)}(
        Int(M), part, shards, exec, nreg,
        T(delta0), T(kappa_e), T(kappa_e + kappa_i), T(sqrt(kappa_e)),
        E_of_t, T(atol), T(rtol), Ref(0), Ref(0), hostbuf, comms, peer_ok,
    )
end



function mgpu_run_sim_2nd_order(SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG;
                           nshards = UNSET,
                           device_ids = UNSET,
                           integrator = UNSET,
                           save_mode = UNSET,
                           save_spins = UNSET,
                           threaded = UNSET,
                           progress_every::Int = 0,
                           dt0::Real = 0,
                           dtmax::Real = Inf,
                           maxiters::Int = 10_000_000,
                           verbose::Bool = true,
                           clean_gpu::Bool = true)

    SIM_SETTING = _with_default_ensemble_method(SIM_SETTING, :second_order)
    CONFIG = mgpu_build_full_config(SIM_SETTING, SYSTEM_CONFIG)
    validate_config(CONFIG)
    validate_pulse_config(PULSE_CONFIG)

    ns_kw     = pick(SIM_SETTING, :nshards, nshards, nothing)
    dev_kw    = pick(SIM_SETTING, :device_ids, device_ids, nothing)
    integ     = pick(SIM_SETTING, :integrator, integrator, :ck45)
    smode     = pick(SIM_SETTING, :save_mode, save_mode, :tstops)
    s_spins   = pick(SIM_SETTING, :save_spins, save_spins, true)
    th_kw     = pick(SIM_SETTING, :threaded, threaded, nothing)

    mkpath(dirname(CONFIG.saved_file_name))

    d = prepare_derived(CONFIG)
    M  = d.M
    Nt = d.Nt
    E_of_t = build_E_of_t(PULSE_CONFIG)
    kind = get_initial_condition(CONFIG)

    T = Float64
    prob = assemble_problem(
        M, d.delta_b, d.g_b, d.Nj, E_of_t, d.delta0, d.kappa_e, d.kappa_i;
        nshards = ns_kw, device_ids = dev_kw, integrator = integ,
        atol = CONFIG.abstol, rtol = CONFIG.reltol, threaded = th_kw, T = T,
        verbose = verbose,
    )













    data = try
        set_initial_condition!(prob, d.Nj, kind)

        store = ObservableStore(T, M, Nt; save_spins = s_spins)

        opts = SolverOptions{T}(
            integrator = integ,
            save_mode = smode,
            reltol = T(CONFIG.reltol),
            abstol = T(CONFIG.abstol),
            dtmax = T(dtmax),
            dt0 = T(dt0),
            maxiters = maxiters,
            verbose = verbose,
            progress_every = progress_every,
        )

        verbose && println("Integrating with $(integ), save_mode = $(smode)...")

        stats = solve_mgpu!(prob, d.timespan[1], d.timespan[2], d.t_save,
                            (k, t, ireg) -> record!(store, prob, ireg, k, t),
                            opts)

        verbose && begin
            println("Callback saved $(store.saved[]) / $Nt requested time points")
            store.saved[] == Nt || error(
                "Callback saved $(store.saved[]) points, but expected $Nt."
            )
            println("Time taken: $(stats.elapsed) seconds")
            println("  accepted steps : $(stats.naccept)")
            println("  rejected steps : $(stats.nreject)")
            println("  RHS evaluations: $(stats.nrhs)")
            println("  dt range       : $(stats.dtmin_used) … $(stats.dtmax_used)")
        end

        t_saved = d.t_save
        E_of_t_arr = [E_of_t(t) for t in t_saved]

        (
            SIM_SETTING = SIM_SETTING,
            SYSTEM_CONFIG = SYSTEM_CONFIG,
            PULSE_CONFIG = PULSE_CONFIG,

            t_saved = t_saved,
            a_sol = store.a,
            n_sol = store.n,
            adad_sol = store.adad,
            Sp_sol = store.Sp,
            Sz_sol = store.Sz,
            adSp_sol = store.adSp,
            adSm_sol = store.adSm,
            adSz_sol = store.adSz,
            E_of_t_arr = E_of_t_arr,

            delta_b_1d = d.delta_b_1d,
            g_b_1d = d.g_b_1d,
            Nj_2d = d.Nj_2d,
            N_total = d.N_total,
            elapsed_seconds = stats.elapsed,
            solver_stats = stats,
            nshards = length(prob.shards),
            partition = prob.part,
        )
    finally
        if clean_gpu
            verbose && println("Cleaning GPU memory...")
            free_shards!(prob.shards)
            verbose && println("GPU memory cleanup finished.")
        end
    end



    filename = CONFIG.saved_file_name
    save_run_data(filename, data)
    verbose && println("Saving to: ", filename)

    return data
end

function mgpu_run_simulation(SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG; kwargs...)
    order = get_simulation_order(SIM_SETTING)
    SIM_SETTING = _with_default_ensemble_method(SIM_SETTING, order)
    if order in (:first_order, :order1, :first, 1)
        println("Start running 1st-order cumulant spin-cavity simulation...")




        kw1 = haskey(kwargs, :clean_gpu) ? (clean_gpu = kwargs[:clean_gpu],) : NamedTuple()
        return run_sim_1st_order(SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG; kw1...)
    elseif order in (:second_order, :order2, :second, 2)
        println("Start running 2nd-order cumulant spin-cavity simulation (multi-GPU)...")
        return mgpu_run_sim_2nd_order(SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG; kwargs...)
    else
        error("Unknown simulation_order = $(order). Use :order1 or :order2.")
    end
end

mgpu_build_full_config(SIM_SETTING, SYSTEM_CONFIG) = merge(SIM_SETTING, SYSTEM_CONFIG)

const UNSET = :__unset__

pick(cfg, name::Symbol, kw, default) =
    kw !== UNSET ? kw : (hasproperty(cfg, name) ? getproperty(cfg, name) : default)
