# ============================================================
# MAIN RUN FUNCTION FOR SECOND-ORDER SIMULATION
# ============================================================

function run_sim_2nd_order(SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG; clean_gpu=true)
    CONFIG = build_full_config(SIM_SETTING, SYSTEM_CONFIG)

    validate_config(CONFIG)
    validate_pulse_config(PULSE_CONFIG)

    mkpath(dirname(CONFIG.saved_file_name))

    d = prepare_derived(CONFIG)

    M  = d.M
    Nt = d.Nt

    E_of_t = build_E_of_t(PULSE_CONFIG)

    initial_condition = get_initial_condition(CONFIG)
    u0_gpu = build_u0_gpu_2nd_order(M, d.Nj, initial_condition)

    delta_b_gpu = CuArray(Float64.(d.delta_b))
    g_b_gpu     = CuArray(Float64.(d.g_b))
    diag_mask   = make_diag_mask(M)

    p_gpu = (
        d.delta0,
        d.kappa_e,
        d.kappa_i,
        delta_b_gpu,
        g_b_gpu,
        M,
        diag_mask,
        E_of_t,
    )

    prob_gpu = ODEProblem(rhs_2nd_order!, u0_gpu, d.timespan, p_gpu)

    sol_gpu = nothing
    cb = nothing

    try
    # --------------------------------------------------------
    # Save arrays
    # --------------------------------------------------------

    a_save    = Vector{ComplexF64}(undef, Nt)
    adad_save = Vector{ComplexF64}(undef, Nt)
    n_save    = Vector{Float64}(undef, Nt)

    Sp_save = Matrix{ComplexF64}(undef, M, Nt)
    Sz_save = Matrix{ComplexF64}(undef, M, Nt)

    adSp_save = Matrix{ComplexF64}(undef, M, Nt)
    adSm_save = Matrix{ComplexF64}(undef, M, Nt)
    adSz_save = Matrix{ComplexF64}(undef, M, Nt)

    range_Sp = IDX2_Sp_start : IDX2_Sp_start + M - 1
    range_Sz = idx2_Sz_start(M) : idx2_Sz_start(M) + M - 1

    range_adSp = idx2_adSp_start(M) : idx2_adSp_start(M) + M - 1
    range_adSm = idx2_adSm_start(M) : idx2_adSm_start(M) + M - 1
    range_adSz = idx2_adSz_start(M) : idx2_adSz_start(M) + M - 1

    kref = Ref(0)

    function affect!(integrator)
        kref[] += 1
        k = kref[]
        u = integrator.u

        a_save[k]    = Array(@view u[IDX2_a:IDX2_a])[1]
        adad_save[k] = Array(@view u[IDX2_ad_ad:IDX2_ad_ad])[1]
        n_save[k]    = real(Array(@view u[IDX2_ad_a:IDX2_ad_a])[1])

        Sp_save[:, k] .= Array(@view u[range_Sp])
        Sz_save[:, k] .= Array(@view u[range_Sz])

        adSp_save[:, k] .= Array(@view u[range_adSp])
        adSm_save[:, k] .= Array(@view u[range_adSm])
        adSz_save[:, k] .= Array(@view u[range_adSz])
    end

    cb = PresetTimeCallback(d.t_save, affect!; save_positions=(false, false))

    t0 = time_ns()

    sol_gpu = CUDA.allowscalar() do
        solve(prob_gpu, Tsit5();
            reltol = CONFIG.reltol,
            abstol = CONFIG.abstol,
            callback = cb,
            save_on = false,
            save_everystep = false,
            dense = false,
        )
    end

    elapsed_seconds = (time_ns() - t0) / 1e9

    println("Callback saved $(kref[]) / $Nt requested time points")
    if kref[] != Nt
        @warn "Callback did not save all requested time points" saved=kref[] expected=Nt
    end

    println("Time taken: $elapsed_seconds seconds")

    # --------------------------------------------------------
    # Post-process
    # --------------------------------------------------------

    t_saved = d.t_save

    E_of_t_arr = [E_of_t(t) for t in t_saved]

    data = (
        SIM_SETTING = SIM_SETTING,
        SYSTEM_CONFIG = SYSTEM_CONFIG,
        PULSE_CONFIG = PULSE_CONFIG,

        t_saved = t_saved,

        a_sol = a_save,
        n_sol = n_save,
        adad_sol = adad_save,

        Sp_sol = Sp_save,
        Sz_sol = Sz_save,

        adSp_sol = adSp_save,
        adSm_sol = adSm_save,
        adSz_sol = adSz_save,

        E_of_t_arr = E_of_t_arr,

        delta_b_1d = d.delta_b_1d,
        g_b_1d = d.g_b_1d,
        Nj_2d = d.Nj_2d,

        N_total = d.N_total,
        elapsed_seconds = elapsed_seconds,
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