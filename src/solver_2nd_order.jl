
function _cuda_2nd_functional()
    try
        return CUDA.functional()
    catch
        return false
    end
end

function _want_gpu_2nd(backend::Symbol)
    backend === :cpu && return false
    backend === :gpu && return true
    backend === :auto && return _cuda_2nd_functional()
    error("backend must be :auto, :cpu, or :gpu; got $(backend).")
end

function run_sim_2nd_order(SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG;
                           clean_gpu=true, backend::Symbol=:auto)
    SIM_SETTING = _with_default_ensemble_method(SIM_SETTING, :second_order)
    CONFIG = build_full_config(SIM_SETTING, SYSTEM_CONFIG)

    validate_config(CONFIG)
    validate_pulse_config(PULSE_CONFIG)

    mkpath(dirname(CONFIG.saved_file_name))

    d = prepare_derived(CONFIG)

    M  = d.M
    Nt = d.Nt

    E_of_t = build_E_of_t(PULSE_CONFIG)

    initial_condition = get_initial_condition(CONFIG)
    use_gpu = _want_gpu_2nd(backend)

    t_saved = d.t_save
    E_of_t_arr = Vector{ComplexF64}(undef, Nt)
    @inbounds for i in 1:Nt
        E_of_t_arr[i] = E_of_t(t_saved[i])
    end

    if !use_gpu
        u0 = build_u0_cpu_2nd_order(M, d.Nj, initial_condition)
        ws = Solver2Workspace(Float64, M, Nt; stages = true)
        attach_u0!(ws, u0)
        p = (
            d.delta0,
            d.kappa_e,
            d.kappa_i,
            Float64.(d.delta_b),
            Float64.(d.g_b),
            M,
            nothing,
            E_of_t,
            ws.rhs,
        )
        t0 = time_ns()
        stats = solve_cpu_2nd!(ws, p, d.timespan[1], d.timespan[2], t_saved;
                               reltol = CONFIG.reltol, abstol = CONFIG.abstol)
        elapsed_seconds = (time_ns() - t0) / 1e9
        println("Callback saved $(ws.saved[]) / $Nt requested time points")
        ws.saved[] == Nt || error(
            "Callback saved $(ws.saved[]) points, but expected $Nt."
        )
        println("Time taken: $elapsed_seconds seconds")
        println("  accepted steps : $(stats.naccept)")
        println("  rejected steps : $(stats.nreject)")
        println("  RHS evaluations: $(stats.nrhs)")

        data = (
            SIM_SETTING = SIM_SETTING,
            SYSTEM_CONFIG = SYSTEM_CONFIG,
            PULSE_CONFIG = PULSE_CONFIG,
            t_saved = t_saved,
            a_sol = ws.a,
            n_sol = ws.n,
            adad_sol = ws.adad,
            Sp_sol = ws.Sp,
            Sz_sol = ws.Sz,
            adSp_sol = ws.adSp,
            adSm_sol = ws.adSm,
            adSz_sol = ws.adSz,
            E_of_t_arr = E_of_t_arr,
            delta_b_1d = d.delta_b_1d,
            g_b_1d = d.g_b_1d,
            Nj_2d = d.Nj_2d,
            N_total = d.N_total,
            elapsed_seconds = elapsed_seconds,
            solver_stats = stats,
            backend = :cpu,
        )
        filename = CONFIG.saved_file_name
        save_run_data(filename, data)
        println("Saving to: ", filename)
        return data
    end

    u0_gpu = build_u0_gpu_2nd_order(M, d.Nj, initial_condition)

    delta_b_gpu = CuArray(Float64.(d.delta_b))
    g_b_gpu     = CuArray(Float64.(d.g_b))
    diag_mask   = make_diag_mask(M)

    rhs_ws = _rhs2_workspace(u0_gpu, M)
    save_ws = Solver2Workspace(Float64, M, Nt; stages = false)
    p_gpu = (
        d.delta0,
        d.kappa_e,
        d.kappa_i,
        delta_b_gpu,
        g_b_gpu,
        M,
        diag_mask,
        E_of_t,
        rhs_ws,
    )

    prob_gpu = ODEProblem(rhs_2nd_order!, u0_gpu, d.timespan, p_gpu)

    sol_gpu = nothing
    cb = nothing

    try
    kref = Ref(0)

    function affect!(integrator)
        kref[] += 1
        record_save2!(save_ws, integrator.u, kref[])
    end

    cb = PresetTimeCallback(d.t_save, affect!; save_positions=(false, false))

    t0 = time_ns()

    CUDA.allowscalar(false)
    integrator = init(prob_gpu, Tsit5();
        reltol = CONFIG.reltol,
        abstol = CONFIG.abstol,
        callback = cb,
        save_on = false,
        save_everystep = false,
        dense = false,
    )
    solve!(integrator)
    sol_gpu = integrator.sol
    CUDA.synchronize()

    elapsed_seconds = (time_ns() - t0) / 1e9

    println("Callback saved $(kref[]) / $Nt requested time points")
    kref[] == Nt || error(
        "Callback saved $(kref[]) points, but expected $Nt."
    )

    println("Time taken: $elapsed_seconds seconds")

    data = (
        SIM_SETTING = SIM_SETTING,
        SYSTEM_CONFIG = SYSTEM_CONFIG,
        PULSE_CONFIG = PULSE_CONFIG,

        t_saved = t_saved,

        a_sol = save_ws.a,
        n_sol = save_ws.n,
        adad_sol = save_ws.adad,

        Sp_sol = save_ws.Sp,
        Sz_sol = save_ws.Sz,

        adSp_sol = save_ws.adSp,
        adSm_sol = save_ws.adSm,
        adSz_sol = save_ws.adSz,

        E_of_t_arr = E_of_t_arr,

        delta_b_1d = d.delta_b_1d,
        g_b_1d = d.g_b_1d,
        Nj_2d = d.Nj_2d,

        N_total = d.N_total,
        elapsed_seconds = elapsed_seconds,
        backend = :gpu,
    )

    filename = CONFIG.saved_file_name
    save_run_data(filename, data)

    println("Saving to: ", filename)

    return data
    finally
        if clean_gpu
            println("Cleaning GPU memory...")

            u0_gpu = nothing
            delta_b_gpu = nothing
            g_b_gpu = nothing
            diag_mask = nothing

            p_gpu = nothing
            prob_gpu = nothing
            sol_gpu = nothing
            cb = nothing

            GC.gc()
            CUDA.reclaim()

            println("GPU memory cleanup finished.")
        end
    end
end
