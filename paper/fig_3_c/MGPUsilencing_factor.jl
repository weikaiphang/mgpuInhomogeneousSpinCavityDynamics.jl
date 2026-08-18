using InhomogeneousSpinCavityDynamics
using JLD2
using Printf: @sprintf, @printf

# ============================================================
# 1) OUTPUT DIRECTORY
# ============================================================

OUTDIR = normpath(
    joinpath(
        @__DIR__,
        "..",
        "..",
        "data",
        "fig_3_c_mgpu",
    ),
)

mkpath(OUTDIR)

# ============================================================
# 2) USER SETTINGS: DURATION AND g-STD SCANS
# ============================================================

# For each WURST duration, specify the g standard deviations
# that should be scanned.
#
# duration_us is entered in μs.
# g_std_Hz is entered in Hz.

SWEEP_PLAN = (
    (
        duration_us = 100.0,
        g_std_Hz = [
            1e-6,
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
        ],
    ),
    
    (
        duration_us = 30.0,
        g_std_Hz = [
            1e-6,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
        ],
    ),
    
    (
        duration_us = 10.0,
        g_std_Hz = [
            1e-6,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
            1.2,
            1.4,
            1.6,
            1.8,
            2.0,
            2.2,
            2.4,
        ],
    ),
)

# ============================================================
# 3) PULSE AND ANALYSIS TIMES
# ============================================================

SIGNAL_CENTER_S = 15e-6
SIGNAL_SIGMA_S  = 3e-6
WURST_CENTER_S  = 120e-6

# Signal center in μs.
TSIG_US = SIGNAL_CENTER_S * 1e6

# The same half-span is used for input and Echo1 analysis.
ANALYSIS_HALF_SPAN_US = 10.0

# Expected Echo1 center:
#
# t_echo1 = 2*t_WURST - t_signal

ECHO1_CENTER_S =
    2 * WURST_CENTER_S - SIGNAL_CENTER_S

ECHO1_CENTER_US =
    ECHO1_CENTER_S * 1e6

println("Input-pulse center:    $TSIG_US μs")
println("Expected Echo1 center: $ECHO1_CENTER_US μs")
println("Analysis half-span:    $ANALYSIS_HALF_SPAN_US μs")
println("Output directory:      $OUTDIR")

# ============================================================
# 4) BASE SIMULATION SETTINGS
# ============================================================

SIM_SETTING_BASE = (
    # --- simulation order ---
    simulation_order = :order1,

    # --- discretization ---
    M_delta = 1000,
    M_g     = 20,

    # --- initial condition ---
    initial_condition = :ground,

    # --- simulation time ---
    Ttotal = 300e-6,

    # --- solver ---
    Nt_save = 5001,
    reltol  = 1e-8,
    abstol  = 1e-8,

    # Replaced separately for every sweep point.
    saved_file_name =
        joinpath(OUTDIR, "temporary.jld2"),

    # --- Echo1 peak detection for phase recording ---
    peak_detection = (
        labels = [:echo1],
        times = [ECHO1_CENTER_S],
        half_window =
            ANALYSIS_HALF_SPAN_US * 1e-6,
    ),
)

# ============================================================
# 5) BASE SYSTEM CONFIGURATION
# ============================================================

SYSTEM_CONFIG_BASE = (
    # --- cooperativity ---
    C_ens = 0.6,

    # --- cavity ---
    delta0 = 0.0,
    kappa_e = 2 * pi * 1e6,
    kappa_i = 2 * pi * 0,

    # --- detuning distribution ---
    freq_inhomogeneity = (
        kind = :lorentzian,
        FWHM = 2 * pi * 1e6,
        span_gamma = 2.5,
        renormalize = false,
    ),

    # --- coupling distribution ---
    #
    # std is replaced for every sweep point.
    g_inhomogeneity = (
        kind = :gaussian,
        mean = 2 * pi * 100,
        std  = 2 * pi * 1,
        span_sigma = 3.0,
        renormalize = true,
    ),
)

# ============================================================
# 6) BASE PULSE CONFIGURATION
# ============================================================

PULSE_CONFIG_BASE = (
    (
        name = "Gaussian input signal",
        kind = :gaussian,

        t0    = SIGNAL_CENTER_S,
        sigma = SIGNAL_SIGMA_S,

        amp =
            0.5 *
            sqrt(SYSTEM_CONFIG_BASE.kappa_e) *
            0.332,

        omega = 0.0,
        phase = 0.0,
    ),

    (
        name = "First WURST pulse",
        kind = :wurst,

        t_center = WURST_CENTER_S,

        # Replaced for every duration scan.
        duration = 100e-6,

        amp =
            0.5 *
            sqrt(SYSTEM_CONFIG_BASE.kappa_e) *
            2.0e4,

        bandwidth =
            5.0 *
            SYSTEM_CONFIG_BASE.freq_inhomogeneity.FWHM,

        n          = 20.0,
        omega0     = 0.0,
        chirp_sign = +1.0,
        phase0     = 0.0,
        edge_frac  = 1e-4,
    ),
)

# ============================================================
# 7) TIME-WINDOW INDICES
# ============================================================

function time_window_indices(
    t,
    center,
    half_span,
)
    idx = findall(
        x -> center - half_span <= x <= center + half_span,
        t,
    )

    isempty(idx) && error(
        "No saved time points were found in the requested window. " *
        "center = $center, half_span = $half_span"
    )

    return idx
end

# ============================================================
# 8) PULSE-WINDOW ANALYSIS
# ============================================================

function pulse_window_analysis(
    t_us,
    center_us,
    half_span_us,
    y,
)
    idx = time_window_indices(
        t_us,
        center_us,
        half_span_us,
    )

    t_zoom = t_us[idx]
    y_zoom = y[idx]

    length(t_zoom) >= 2 || error(
        "Pulse-analysis window needs at least two points."
    )

    imax = argmax(abs.(y_zoom))

    ymax = y_zoom[imax]
    tmax = t_zoom[imax]

    dt = diff(t_zoom)

    integral = sum(
        0.5 .* (
            y_zoom[1:end-1] .+
            y_zoom[2:end]
        ) .* dt
    )

    norm_factor = sqrt(
        sum(
            0.5 .* (
                abs.(y_zoom[1:end-1]).^2 .+
                abs.(y_zoom[2:end]).^2
            ) .* dt
        )
    )

    return (
        indices = idx,
        times = t_zoom,
        values = y_zoom,
        maximum = ymax,
        time_at_maximum = tmax,
        integral = integral,
        norm = norm_factor,
    )
end

# ============================================================
# 9) FILENAME LABEL
# ============================================================

function filename_label(x::Real)
    text = @sprintf("%.6g", Float64(x))

    return replace(
        text,
        "." => "p",
        "-" => "m",
        "+" => "",
    )
end

# ============================================================
# 10) DURATION AND g-STD SWEEP
# ============================================================

function run_duration_gstd_sweep(
    sweep_plan,
    sim_setting_base,
    system_config_base,
    pulse_config_base,
    outdir;
    tsig_us,
    echo1_center_us,
    analysis_half_span_us,
)
    total_runs = sum(
        length(scan.g_std_Hz)
        for scan in sweep_plan
    )

    # The final summary contains only these three quantities.
    duration_us_results = Float64[]
    g_std_Hz_results    = Float64[]
    ACE_amp_results     = Float64[]

    run_index = 0

    println()
    println("Starting duration and g-std sweep.")
    println("Total number of simulations: $total_runs")
    println()

    # --------------------------------------------------------
    # Scan WURST durations
    # --------------------------------------------------------

    for scan in sweep_plan
        duration_us =
            Float64(scan.duration_us)

        duration_s =
            duration_us * 1e-6

        wurst_center_us =
            pulse_config_base[2].t_center * 1e6

        wurst_start_us =
            wurst_center_us - duration_us / 2

        wurst_end_us =
            wurst_center_us + duration_us / 2

        input_window_end_us =
            tsig_us + analysis_half_span_us

        echo_window_start_us =
            echo1_center_us - analysis_half_span_us

        # Warn when the WURST pulse approaches the input window.
        if wurst_start_us <= input_window_end_us
            @warn(
                "The WURST pulse overlaps or approaches the input " *
                "analysis window.",
                duration_us = duration_us,
                wurst_start_us = wurst_start_us,
                input_window_end_us = input_window_end_us,
            )
        end

        # Warn when the WURST pulse approaches the Echo1 window.
        if wurst_end_us >= echo_window_start_us
            @warn(
                "The WURST pulse overlaps or approaches the Echo1 " *
                "analysis window.",
                duration_us = duration_us,
                wurst_end_us = wurst_end_us,
                echo_window_start_us = echo_window_start_us,
            )
        end

        # ----------------------------------------------------
        # Scan g-distribution standard deviations
        # ----------------------------------------------------

        for g_std_Hz_value in scan.g_std_Hz
            run_index += 1

            g_std_Hz =
                Float64(g_std_Hz_value)

            # Convert Hz to rad/s for SYSTEM_CONFIG.
            g_std_rad_s =
                2 * pi * g_std_Hz

            duration_label =
                filename_label(duration_us)

            std_label =
                filename_label(g_std_Hz)

            saved_file_name = joinpath(
                outdir,
                "duration_$(duration_label)us_" *
                "gstd_$(std_label)Hz.jld2",
            )

            # =================================================
            # UPDATE g-INHOMOGENEITY
            # =================================================

            g_inhomogeneity_run = merge(
                system_config_base.g_inhomogeneity,
                (
                    std = g_std_rad_s,
                ),
            )

            system_config_run = merge(
                system_config_base,
                (
                    g_inhomogeneity =
                        g_inhomogeneity_run,
                ),
            )

            # =================================================
            # UPDATE WURST DURATION
            # =================================================

            wurst_pulse_run = merge(
                pulse_config_base[2],
                (
                    duration = duration_s,
                ),
            )

            pulse_config_run = (
                pulse_config_base[1],
                wurst_pulse_run,
            )

            # =================================================
            # UPDATE SAVED FILENAME
            # =================================================

            sim_setting_run = merge(
                sim_setting_base,
                (
                    saved_file_name =
                        saved_file_name,
                ),
            )

            println(
                "============================================================"
            )

            println("Run $run_index / $total_runs")
            println("WURST duration = $duration_us μs")
            println("g std          = $g_std_Hz Hz")
            println("Saving to      = $saved_file_name")

            println(
                "============================================================"
            )

            # =================================================
            # RUN SIMULATION
            # =================================================

            mgpu_run_simulation(
                sim_setting_run,
                system_config_run,
                pulse_config_run,
            )

            # =================================================
            # LOAD SAVED DATA
            # =================================================
            #
            # The simulation saves one top-level object named
            # `data`, which is a NamedTuple.
            #

            @load saved_file_name data

            t_saved =
                vec(data.t_saved)

            a_sol =
                vec(data.a_sol)

            E_of_t_arr =
                vec(data.E_of_t_arr)

            n_time =
                length(t_saved)

            (
                length(a_sol) == n_time &&
                length(E_of_t_arr) == n_time
            ) || error(
                "t_saved, a_sol, and E_of_t_arr have different " *
                "lengths in:\n$saved_file_name"
            )

            # =================================================
            # CONSTRUCT INPUT AND OUTPUT FIELDS
            # =================================================

            kappa_e =
                data.SYSTEM_CONFIG.kappa_e

            # Input field:
            #
            # a_in(t) = E(t)
            #
            # Input-pulse analysis uses its absolute value.

            a_in_abs =
                abs.(E_of_t_arr)

            # Output field:
            #
            # a_out(t) = E(t) - sqrt(kappa_e)*a(t)

            a_out =
                E_of_t_arr .-
                sqrt(kappa_e) .* a_sol

            # pulse_window_analysis uses a time axis in μs.
            t_us =
                t_saved .* 1e6

            # =================================================
            # INPUT-PULSE ANALYSIS FROM a_in_abs
            # =================================================

            input_analysis = pulse_window_analysis(
                t_us,
                tsig_us,
                analysis_half_span_us,
                a_in_abs,
            )

            input_analysis.norm > 0 || error(
                "The input-pulse norm is zero for:\n" *
                "duration = $duration_us μs\n" *
                "g std = $g_std_Hz Hz"
            )

            # =================================================
            # ECHO1 ANALYSIS FROM a_out
            # =================================================

            echo1_analysis = pulse_window_analysis(
                t_us,
                echo1_center_us,
                analysis_half_span_us,
                a_out,
            )

            # =================================================
            # NORMALIZED ACE AMPLITUDE
            # =================================================
            #
            # Input norm: calculated from a_in_abs.
            # Echo1 norm: calculated from a_out.
            #

            ACE_amp = Float64(
                echo1_analysis.norm /
                input_analysis.norm
            )

            # =================================================
            # RECORD SUMMARY RESULT
            # =================================================

            push!(
                duration_us_results,
                duration_us,
            )

            push!(
                g_std_Hz_results,
                g_std_Hz,
            )

            push!(
                ACE_amp_results,
                ACE_amp,
            )

            println("Input norm from a_in  = $(input_analysis.norm)")
            println("Echo1 norm from a_out = $(echo1_analysis.norm)")
            println("ACE_amp               = $ACE_amp")
            println()
        end
    end

    # ========================================================
    # 11) SAVE FINAL JLD2 SUMMARY
    # ========================================================
    #
    # This file contains only:
    #
    #   duration_us
    #   g_std_Hz
    #   ACE_amp
    #

    summary_file = joinpath(
        outdir,
        "fig_3_c_ACE_summary.jld2",
    )

    JLD2.jldsave(
        summary_file;

        duration_us =
            duration_us_results,

        g_std_Hz =
            g_std_Hz_results,

        ACE_amp =
            ACE_amp_results,
    )

    # ========================================================
    # 12) SAVE FINAL CSV SUMMARY
    # ========================================================
    #
    # This file also contains only:
    #
    #   duration_us
    #   g_std_Hz
    #   ACE_amp
    #

    csv_file = joinpath(
        outdir,
        "fig_3_c_ACE_summary.csv",
    )

    open(csv_file, "w") do io
        println(
            io,
            "duration_us,g_std_Hz,ACE_amp",
        )

        for i in eachindex(duration_us_results)
            @printf(
                io,
                "%.9g,%.9g,%.16e\n",
                duration_us_results[i],
                g_std_Hz_results[i],
                ACE_amp_results[i],
            )
        end
    end

    println(
        "============================================================"
    )

    println("All simulations finished.")
    println("JLD2 summary: $summary_file")
    println("CSV summary:  $csv_file")

    println(
        "============================================================"
    )

    return (
        duration_us =
            duration_us_results,

        g_std_Hz =
            g_std_Hz_results,

        ACE_amp =
            ACE_amp_results,
    )
end

# ============================================================
# 13) RUN THE SWEEP
# ============================================================

results = Base.invokelatest(
    run_duration_gstd_sweep,
    SWEEP_PLAN,
    SIM_SETTING_BASE,
    SYSTEM_CONFIG_BASE,
    PULSE_CONFIG_BASE,
    OUTDIR;

    tsig_us =
        TSIG_US,

    echo1_center_us =
        ECHO1_CENTER_US,

    analysis_half_span_us =
        ANALYSIS_HALF_SPAN_US,
)

nothing