# ============================================================
# MAIN RUN FUNCTION FOR FIRST-ORDER SIMULATION
# ============================================================

function run_sim_1st_order(
    SIM_SETTING,
    SYSTEM_CONFIG,
    PULSE_CONFIG;
    clean_gpu = true,
)
    # ========================================================
    # BUILD AND VALIDATE CONFIGURATION
    # ========================================================

    CONFIG = build_full_config(
        SIM_SETTING,
        SYSTEM_CONFIG,
    )

    validate_config(CONFIG)
    validate_pulse_config(PULSE_CONFIG)

    mkpath(dirname(CONFIG.saved_file_name))

    d = prepare_derived(CONFIG)

    M = d.M

    t_saved = collect(d.t_save)
    Nt = length(t_saved)

    @assert Nt == d.Nt (
        "Inconsistent saved-time dimensions: " *
        "length(d.t_save) = $Nt, while d.Nt = $(d.Nt)."
    )

    E_of_t = build_E_of_t(PULSE_CONFIG)

    # ========================================================
    # INITIAL CONDITION AND GPU PARAMETERS
    # ========================================================

    initial_condition = get_initial_condition(CONFIG)

    u0_gpu = build_u0_gpu_1st_order(
        M,
        d.Nj,
        initial_condition,
    )

    delta_b_gpu = CuArray(Float64.(d.delta_b))
    g_b_gpu     = CuArray(Float64.(d.g_b))

    p_gpu = (
        d.delta0,
        d.kappa_e,
        d.kappa_i,
        delta_b_gpu,
        g_b_gpu,
        M,
        E_of_t,
    )

    prob_gpu = ODEProblem(
        rhs_1st_order!,
        u0_gpu,
        d.timespan,
        p_gpu,
    )

    peak_Sp_gpu = nothing
    peak_Sz_gpu = nothing
    sol_gpu = nothing
    cb = nothing

    try
    # ========================================================
    # ENSEMBLE DIMENSIONS
    # ========================================================

    M_delta = length(d.delta_b_1d)
    M_g     = length(d.g_b_1d)

    @assert M == M_delta * M_g (
        "The flattened ensemble size is inconsistent. " *
        "M = $M, while M_delta × M_g = $(M_delta * M_g)."
    )

    # ========================================================
    # RESONANT-DELTA BIN FOR EVERY g VALUE
    # ========================================================

    # Select the detuning bin nearest delta = 0.
    idelta_res = argmin(abs.(d.delta_b_1d))
    delta_res  = d.delta_b_1d[idelta_res]

    # The flattened spin arrays follow Julia column-major order:
    #
    # flat_index = idelta + (ig - 1) * M_delta
    #
    # Therefore, the resonant-delta indices form this strided
    # range:
    keep_range = idelta_res:M_delta:M
    keep_bins  = collect(keep_range)

    @assert length(keep_bins) == M_g

    g_keep     = collect(d.g_b_1d)
    delta_keep = fill(delta_res, M_g)

    println("Selected resonant-detuning bin:")
    println("  idelta_res = $idelta_res")
    println("  delta_res / 2π = $(delta_res / (2π)) Hz")
    println("  number of g bins = $M_g")

    # ========================================================
    # OPTIONAL AUTOMATIC PEAK DETECTION
    # ========================================================

    peak_config = prepare_peak_detection(
        SIM_SETTING,
        t_saved,
    )

    Npeaks = peak_config === nothing ? 0 : length(peak_config.labels)

    if peak_config === nothing
        println("Automatic peak detection: disabled")
    else
        println("Automatic peak detection: enabled")

        for ip in 1:Npeaks
            println(
                "  $(peak_config.labels[ip]): expected time = " *
                "$(peak_config.times[ip] * 1e6) μs, " *
                "search window = ±" *
                "$(peak_config.half_windows[ip] * 1e6) μs"
            )
        end
    end

    # Since the saved-time points are ordered, every search
    # window corresponds to a contiguous range of callback
    # indices.
    peak_window_start_indices = if peak_config === nothing
        Int[]
    else
        [first(indices) for indices in peak_config.window_indices]
    end

    peak_window_end_indices = if peak_config === nothing
        Int[]
    else
        [last(indices) for indices in peak_config.window_indices]
    end

    # ========================================================
    # MAIN SAVE ARRAYS
    # ========================================================

    # Cavity amplitude at every saved time.
    a_save = Vector{ComplexF64}(undef, Nt)

    # Collective spin values at every saved time.
    Σp_save = Vector{ComplexF64}(undef, Nt)
    Σz_save = Vector{ComplexF64}(undef, Nt)

    # Resonant-delta trajectories.
    #
    # Rows:    g bins
    # Columns: saved times
    Sp_keep = Matrix{ComplexF64}(undef, M_g, Nt)
    Sz_keep = Matrix{ComplexF64}(undef, M_g, Nt)

    range_Sp = (
        IDX1_Sp_start :
        IDX1_Sp_start + M - 1
    )

    range_Sz = (
        idx1_Sz_start(M) :
        idx1_Sz_start(M) + M - 1
    )

    # ========================================================
    # PEAK-DETECTION BUFFERS
    # ========================================================

    # These buffers hold only one full spin state per requested
    # peak: the best state found so far inside that peak's
    # search window.
    #
    # They are GPU arrays, so updating the best peak does not
    # require transferring every bin to the CPU during solving.
    peak_Sp_gpu = if peak_config === nothing
        nothing
    else
        [
            CUDA.zeros(ComplexF64, M)
            for _ in 1:Npeaks
        ]
    end

    peak_Sz_gpu = if peak_config === nothing
        nothing
    else
        [
            CUDA.zeros(ComplexF64, M)
            for _ in 1:Npeaks
        ]
    end

    # Best peak information found so far.
    peak_best_amplitudes = if peak_config === nothing
        nothing
    else
        fill(-Inf, Npeaks)
    end

    peak_best_global_indices = if peak_config === nothing
        nothing
    else
        zeros(Int, Npeaks)
    end

    # ========================================================
    # CALLBACK
    # ========================================================

    kref = Ref(0)

    function affect!(integrator)
        kref[] += 1
        k = kref[]

        u = integrator.u

        # ----------------------------------------------------
        # Save cavity amplitude
        # ----------------------------------------------------

        a_save[k] = Array(
            @view u[IDX1_a:IDX1_a]
        )[1]

        # GPU views of all spin bins.
        #
        # Creating these views does not transfer the complete
        # arrays to the CPU.
        Sp_gpu = @view u[range_Sp]
        Sz_gpu = @view u[range_Sz]

        # ----------------------------------------------------
        # Collective spin quantities
        # ----------------------------------------------------

        Σp_save[k] = sum(Sp_gpu)
        Σz_save[k] = sum(Sz_gpu)

        # ----------------------------------------------------
        # Resonant-delta bins for every g
        # ----------------------------------------------------

        # Only M_g selected values are transferred to the CPU,
        # rather than all M spin bins.
        Sp_keep[:, k] .= Array(
            @view Sp_gpu[keep_range]
        )

        Sz_keep[:, k] .= Array(
            @view Sz_gpu[keep_range]
        )

        # ----------------------------------------------------
        # OPTIONAL ONLINE PEAK DETECTION
        # ----------------------------------------------------

        if peak_config !== nothing
            # Compute the output amplitude only when the current
            # saved time belongs to at least one search window.
            a_out_abs_now = 0.0
            a_out_computed = false

            for ip in 1:Npeaks
                inside_window = (
                    peak_window_start_indices[ip] <= k <=
                    peak_window_end_indices[ip]
                )

                if inside_window
                    if !a_out_computed
                        E_now = E_of_t(integrator.t)

                        a_out_now = (
                            E_now -
                            d.sqrt_kappa_e * a_save[k]
                        )

                        a_out_abs_now = abs(a_out_now)
                        a_out_computed = true
                    end

                    # If this is the largest output amplitude
                    # encountered so far in this search window,
                    # replace the saved peak state.
                    if a_out_abs_now > peak_best_amplitudes[ip]
                        peak_best_amplitudes[ip] = a_out_abs_now
                        peak_best_global_indices[ip] = k

                        # Device-to-device copies.
                        #
                        # These save the complete spin state only
                        # for the current best peak candidate.
                        peak_Sp_gpu[ip] .= Sp_gpu
                        peak_Sz_gpu[ip] .= Sz_gpu
                    end
                end
            end
        end

        return nothing
    end

    cb = PresetTimeCallback(
        t_saved,
        affect!;
        save_positions = (false, false),
    )

    # ========================================================
    # SOLVE
    # ========================================================

    t0 = time_ns()

    sol_gpu = CUDA.allowscalar() do
        solve(
            prob_gpu,
            Tsit5();
            reltol = CONFIG.reltol,
            abstol = CONFIG.abstol,
            callback = cb,
            save_on = false,
            save_everystep = false,
            dense = false,
        )
    end

    CUDA.synchronize()

    elapsed_seconds = (time_ns() - t0) / 1e9

    println(
        "Callback saved $(kref[]) / $Nt requested time points"
    )

    kref[] == Nt || error(
        "Callback saved $(kref[]) points, but expected $Nt."
    )

    println("Time taken: $elapsed_seconds seconds")

    # ========================================================
    # POST-PROCESS CAVITY OUTPUT
    # ========================================================

    E_of_t_arr = [
        E_of_t(t)
        for t in t_saved
    ]

    a_out = (
        E_of_t_arr .-
        d.sqrt_kappa_e .* a_save
    )

    a_out_x   = real.(a_out)
    a_out_p   = imag.(a_out)
    a_out_abs = abs.(a_out)

    # ========================================================
    # BUILD PEAK-DETECTION RESULTS
    # ========================================================

    peak_detection_config = nothing
    peak_detection_results = nothing

    if peak_config !== nothing
        peak_detection_config = (
            labels = copy(peak_config.labels),
            times = copy(peak_config.times),
            half_windows = copy(peak_config.half_windows),
        )

        peak_detection_results = NamedTuple[]

        for ip in 1:Npeaks
            global_peak_index = peak_best_global_indices[ip]

            global_peak_index != 0 || error(
                "No peak candidate was saved for " *
                "$(peak_config.labels[ip])."
            )

            window_indices = peak_config.window_indices[ip]

            local_peak_index = findfirst(
                ==(global_peak_index),
                window_indices,
            )

            local_peak_index !== nothing || error(
                "Detected peak index is not inside its search window."
            )

            expected_time = peak_config.times[ip]
            half_window   = peak_config.half_windows[ip]

            detected_time      = t_saved[global_peak_index]
            detected_amplitude = a_out_abs[global_peak_index]

            # Transfer only the final best full-bin state from
            # GPU to CPU, and directly arrange it as:
            #
            # rows    = delta bins
            # columns = g bins
            Sp_at_peak_2d = Array(
                reshape(
                    peak_Sp_gpu[ip],
                    M_delta,
                    M_g,
                )
            )

            Sz_at_peak_2d = Array(
                reshape(
                    peak_Sz_gpu[ip],
                    M_delta,
                    M_g,
                )
            )

            result = (
                label = peak_config.labels[ip],

                expected_time = expected_time,
                half_window = half_window,

                window_start = expected_time - half_window,
                window_end = expected_time + half_window,

                window_indices = copy(window_indices),
                window_times = t_saved[window_indices],
                window_a_out_abs = a_out_abs[window_indices],

                local_peak_index = local_peak_index,
                global_peak_index = global_peak_index,

                detected_time = detected_time,
                detected_amplitude = detected_amplitude,

                a_out_at_peak = a_out[global_peak_index],
                a_out_x_at_peak = a_out_x[global_peak_index],
                a_out_p_at_peak = a_out_p[global_peak_index],

                # Full ensemble state only at the detected peak.
                #
                # Shape: M_delta × M_g
                Sp_at_peak_2d = Sp_at_peak_2d,
                Sz_at_peak_2d = Sz_at_peak_2d,
            )

            push!(peak_detection_results, result)

            println()
            println(
                "Detected peak: $(peak_config.labels[ip])"
            )

            println(
                "  expected time = " *
                "$(expected_time * 1e6) μs"
            )

            println(
                "  detected time = " *
                "$(detected_time * 1e6) μs"
            )

            println(
                "  time shift = " *
                "$((detected_time - expected_time) * 1e6) μs"
            )

            println(
                "  |a_out| at peak = $detected_amplitude"
            )
        end
    end

    # ========================================================
    # SAVE DATA
    # ========================================================

    data = (
        SIM_SETTING = SIM_SETTING,
        SYSTEM_CONFIG = SYSTEM_CONFIG,
        PULSE_CONFIG = PULSE_CONFIG,

        t_saved = t_saved,

        a_sol = a_save,

        Σp_sol = Σp_save,
        Σz_sol = Σz_save,

        E_of_t_arr = E_of_t_arr,

        M_delta = M_delta,
        M_g = M_g,
        M_total = M,

        delta_b_1d = d.delta_b_1d,
        g_b_1d = d.g_b_1d,
        Nj_2d = d.Nj_2d,

        # ----------------------------------------------------
        # Resonant-delta trajectories for every g
        # ----------------------------------------------------

        idelta_res = idelta_res,
        delta_res = delta_res,

        keep_bins = keep_bins,
        g_keep = g_keep,
        delta_keep = delta_keep,

        # Shape: M_g × Nt
        Sp_keep = Sp_keep,
        Sz_keep = Sz_keep,

        # ----------------------------------------------------
        # Optional peak detection
        # ----------------------------------------------------

        peak_detection_config = peak_detection_config,
        peak_detection_results = peak_detection_results,

        N_total = d.N_total,
        elapsed_seconds = elapsed_seconds,
    )

    filename = CONFIG.saved_file_name
    save_run_data(filename, data)

    println()
    println("Saving to: ", filename)

    return data
    finally
        if clean_gpu
            println("Cleaning GPU memory...")

            u0_gpu = nothing
            delta_b_gpu = nothing
            g_b_gpu = nothing

            peak_Sp_gpu = nothing
            peak_Sz_gpu = nothing

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