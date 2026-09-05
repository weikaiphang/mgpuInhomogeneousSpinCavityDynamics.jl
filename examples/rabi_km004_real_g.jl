using InhomogeneousSpinCavityDynamics
using JLD2
using Plots
using Measures
using Printf

default(
    size = (900, 600),
    dpi = 300,
    lw = 2.5,
    guidefont = font(16),
    tickfont = font(13),
    legendfont = font(11),
    titlefont = font(18),
    grid = true,
    left_margin = 3mm,
    bottom_margin = 3mm,
)


SWEEP_OUTDIR = joinpath("/scratch", ENV["USER"], "rose_runs", "rabi_sweep_km004_real_g")
mkpath(SWEEP_OUTDIR)

SECOND_PULSE_ALPHA_MIN = 0.0
SECOND_PULSE_ALPHA_MAX = 10_000.0
N_SWEEP_POINTS = 101

SECOND_PULSE_ALPHA_NORMALIZATION = 9523.0

PULSE_TIME_INTERVAL_S = 80e-6

PULSE_6SIGMA_DURATION_S = 20e-6
FIRST_PULSE_CENTER_S = 10e-6

SECOND_PULSE_CENTER_S = (
    PULSE_6SIGMA_DURATION_S +
    PULSE_TIME_INTERVAL_S +
    FIRST_PULSE_CENTER_S
)

ECHO_CENTER_S = (
    2 * PULSE_6SIGMA_DURATION_S +
    FIRST_PULSE_CENTER_S +
    2 * PULSE_TIME_INTERVAL_S
)

T_TOTAL_S = (
    3 * PULSE_6SIGMA_DURATION_S +
    3 * FIRST_PULSE_CENTER_S +
    2 * PULSE_TIME_INTERVAL_S
)

ECHO_WINDOW_HALF_SPAN_S = 40e-6

SKIP_EXISTING = true



BASE_SIM_SETTING = (
    simulation_order = :order1,

    M_delta = 3500,
    M_g     = 4157,

    initial_condition = :ground,
    Ttotal = T_TOTAL_S,
    Nt_save = 2001,
    reltol  = 1e-8,
    abstol  = 1e-8,
)



SYSTEM_CONFIG = (
    C_ens = 0.1,

    delta0 = 0,
    kappa_e = 2pi * 1.28e6,
    kappa_i = 2pi * 0.0577e6,

    freq_inhomogeneity = (
        kind = :gaussian,
        FWHM = 2pi * 5.0e6,
        span_sigma = 3.0,
        renormalize = false,
    ),


    g_inhomogeneity = (
        kind = :user_defined,
        filename = expanduser(
            "~/spin_cavity_project/g_dist/g_dist_cpw_in_g_1hz_to_15khz.jld2",
        ),
        renormalize = true,
    ),
)



FIRST_PULSE_SIGMA_S = PULSE_6SIGMA_DURATION_S / 6
FIRST_PULSE_OMEGA_RAD_S = 0.0
FIRST_PULSE_PHASE_RAD = 0.0

SECOND_PULSE_SIGMA_S = PULSE_6SIGMA_DURATION_S / 6
SECOND_PULSE_OMEGA_RAD_S = 0.0
SECOND_PULSE_PHASE_RAD = 0.0


function build_pulse_config(system_config, second_pulse_alpha)
    first_pulse = (
        name = "First Gaussian pulse",
        kind = :gaussian,
        t0 = FIRST_PULSE_CENTER_S,
        sigma = FIRST_PULSE_SIGMA_S,
        amp = 0.5 * sqrt(system_config.kappa_e) * (second_pulse_alpha / 2),
        omega = FIRST_PULSE_OMEGA_RAD_S,
        phase = FIRST_PULSE_PHASE_RAD,
    )

    second_pulse_amp = (
        0.5 * sqrt(system_config.kappa_e) * second_pulse_alpha
    )

    second_pulse = (
        name = "Swept Gaussian pulse",
        kind = :gaussian,
        t0 = SECOND_PULSE_CENTER_S,
        sigma = SECOND_PULSE_SIGMA_S,
        amp = second_pulse_amp,
        omega = SECOND_PULSE_OMEGA_RAD_S,
        phase = SECOND_PULSE_PHASE_RAD,
    )

    return (first_pulse, second_pulse)
end



function make_alpha_tag(alpha)
    text = @sprintf("%.6g", Float64(alpha))
    return replace(text, "." => "p", "-" => "m", "+" => "")
end



trapz(t, y) = sum(
    0.5 .* (y[1:end-1] .+ y[2:end]) .* diff(t)
)


function echo_box_mode_amplitude(data, echo_center_s, half_span_s)
    t = Float64.(data.t_saved)


    a_out = (
        data.E_of_t_arr .-
        sqrt(data.SYSTEM_CONFIG.kappa_e) .* data.a_sol
    )

    indices = findall(
        x -> echo_center_s - half_span_s <= x <= echo_center_s + half_span_s,
        t,
    )

    length(indices) >= 2 || error(
        "The echo window contains fewer than two saved time points.",
    )

    t_window = t[indices]
    a_out_window = a_out[indices]


    mode_f = ones(ComplexF64, length(t_window))
    mode_f ./= sqrt(real(trapz(t_window, abs2.(mode_f))))


    return abs(trapz(t_window, conj.(mode_f) .* a_out_window))
end



function check_todo_settings(n_sweep_points, base_sim_setting, system_config)
    required = Pair{String, Any}[
        "N_SWEEP_POINTS" => n_sweep_points,
        "PULSE_TIME_INTERVAL_S" => PULSE_TIME_INTERVAL_S,
        "ECHO_CENTER_S" => ECHO_CENTER_S,
        "ECHO_WINDOW_HALF_SPAN_S" => ECHO_WINDOW_HALF_SPAN_S,
        "BASE_SIM_SETTING.M_delta" => base_sim_setting.M_delta,
        "BASE_SIM_SETTING.M_g" => base_sim_setting.M_g,
        "BASE_SIM_SETTING.initial_condition" => base_sim_setting.initial_condition,
        "BASE_SIM_SETTING.Ttotal" => base_sim_setting.Ttotal,
        "BASE_SIM_SETTING.Nt_save" => base_sim_setting.Nt_save,
        "BASE_SIM_SETTING.reltol" => base_sim_setting.reltol,
        "BASE_SIM_SETTING.abstol" => base_sim_setting.abstol,
        "SYSTEM_CONFIG.delta0" => system_config.delta0,
        "freq_inhomogeneity.span_sigma" => system_config.freq_inhomogeneity.span_sigma,
        "freq_inhomogeneity.renormalize" => system_config.freq_inhomogeneity.renormalize,
        "g_inhomogeneity.filename" => system_config.g_inhomogeneity.filename,
        "g_inhomogeneity.renormalize" => system_config.g_inhomogeneity.renormalize,
        "FIRST_PULSE_CENTER_S" => FIRST_PULSE_CENTER_S,
        "FIRST_PULSE_OMEGA_RAD_S" => FIRST_PULSE_OMEGA_RAD_S,
        "FIRST_PULSE_PHASE_RAD" => FIRST_PULSE_PHASE_RAD,
        "SECOND_PULSE_CENTER_S" => SECOND_PULSE_CENTER_S,
    ]

    append!(required, Pair{String, Any}[
        "SECOND_PULSE_SIGMA_S" => SECOND_PULSE_SIGMA_S,
        "SECOND_PULSE_OMEGA_RAD_S" => SECOND_PULSE_OMEGA_RAD_S,
        "SECOND_PULSE_PHASE_RAD" => SECOND_PULSE_PHASE_RAD,
    ])

    unfilled = first.(filter(pair -> last(pair) === missing, required))
    isempty(unfilled) || error(
        "Complete these TODO settings before running:\n  " *
        join(unfilled, "\n  "),
    )

    Int(n_sweep_points) > 1 || error(
        "N_SWEEP_POINTS must be greater than one.",
    )

    PULSE_TIME_INTERVAL_S isa Real &&
        isfinite(PULSE_TIME_INTERVAL_S) &&
        PULSE_TIME_INTERVAL_S >= 0 || error(
            "PULSE_TIME_INTERVAL_S must be finite and non-negative.",
        )

    isapprox(base_sim_setting.Ttotal, T_TOTAL_S; rtol=0.0, atol=0.0) || error(
        "BASE_SIM_SETTING.Ttotal must use the derived T_TOTAL_S.",
    )

    echo_window_start = ECHO_CENTER_S - ECHO_WINDOW_HALF_SPAN_S
    echo_window_end = ECHO_CENTER_S + ECHO_WINDOW_HALF_SPAN_S

    echo_window_start >= 0 || error(
        "The derived echo window starts before the simulation.",
    )
    echo_window_end <= base_sim_setting.Ttotal || error(
        "The derived echo window ends at $(echo_window_end * 1e6) us, " *
        "after Ttotal = $(base_sim_setting.Ttotal * 1e6) us.",
    )

    return nothing
end



function run_rabi_sweep(
    alpha_min,
    alpha_max,
    n_sweep_points,
    alpha_normalization,
    base_sim_setting,
    system_config,
    sweep_outdir;
    skip_existing = true,
)
    check_todo_settings(n_sweep_points, base_sim_setting, system_config)

    alpha_values = collect(
        range(alpha_min, alpha_max; length = Int(n_sweep_points)),
    )
    normalized_alpha_values = alpha_values ./ alpha_normalization
    echo_amplitudes = Vector{Float64}(undef, length(alpha_values))

    N_runs = length(alpha_values)
    println("============================================================")
    println("Starting second-pulse amplitude sweep")
    println("Number of simulations: $N_runs")
    println("Fixed C_ens:           $(system_config.C_ens)")
    print_cooperativity_honesty(system_config)
    println("alpha range:           [$alpha_min, $alpha_max]")
    println("Pulse interval:        $(PULSE_TIME_INTERVAL_S * 1e6) us")
    println("First pulse center:    $(FIRST_PULSE_CENTER_S * 1e6) us")
    println("Second pulse center:   $(SECOND_PULSE_CENTER_S * 1e6) us")
    println("Echo center:           $(ECHO_CENTER_S * 1e6) us")
    println("Total simulation time: $(T_TOTAL_S * 1e6) us")
    println("Output directory:      $sweep_outdir")
    println("============================================================")

    for (run_index, second_pulse_alpha) in enumerate(alpha_values)
        alpha_tag = make_alpha_tag(second_pulse_alpha)
        filename = "rabi_alpha2_$(alpha_tag).jld2"
        saved_file_name = joinpath(sweep_outdir, filename)

        println()
        println("------------------------------------------------------------")
        println("Run $run_index / $N_runs")
        println("Second-pulse alpha = $second_pulse_alpha")
        println("Normalized alpha   = $(second_pulse_alpha / alpha_normalization)")
        println("Output file       = $saved_file_name")
        println("------------------------------------------------------------")

        data = if skip_existing && isfile(saved_file_name)
            println("Output exists. Loading it for echo analysis.")
            JLD2.load(saved_file_name, "data")
        else
            sim_setting = merge(
                base_sim_setting,
                (saved_file_name = saved_file_name,),
            )
            pulse_config = build_pulse_config(
                system_config,
                second_pulse_alpha,
            )
            run_simulation(sim_setting, system_config, pulse_config)
        end

        echo_amplitudes[run_index] = echo_box_mode_amplitude(
            data,
            ECHO_CENTER_S,
            ECHO_WINDOW_HALF_SPAN_S,
        )
        println("Echo box-mode amplitude = $(echo_amplitudes[run_index])")
    end

    summary_file = joinpath(sweep_outdir, "rabi_summary.jld2")
    JLD2.jldsave(
        summary_file;
        alpha_values,
        normalized_alpha_values,
        echo_amplitudes,
        pulse_time_interval_s = PULSE_TIME_INTERVAL_S,
        first_pulse_center_s = FIRST_PULSE_CENTER_S,
        second_pulse_center_s = SECOND_PULSE_CENTER_S,
        echo_center_s = ECHO_CENTER_S,
        total_time_s = T_TOTAL_S,
        echo_window_half_span_s = ECHO_WINDOW_HALF_SPAN_S,
    )

    plt = plot(
        normalized_alpha_values,
        echo_amplitudes;
        marker = :circle,
        xlabel = "Second-pulse normalized amplitude",
        ylabel = "Echo box-mode amplitude",
        title = "Echo amplitude versus second-pulse amplitude",
        legend = false,
        grid = true,
    )

    figure_file = joinpath(
        sweep_outdir,
        "rabi_echo_vs_normalized_amp.png",
    )
    savefig(plt, figure_file)
    display(plt)

    println()
    println("============================================================")
    println("Second-pulse amplitude sweep finished.")
    println("Summary saved to: $summary_file")
    println("Figure saved to:  $figure_file")
    println("============================================================")

    return (
        alpha_values = alpha_values,
        normalized_alpha_values = normalized_alpha_values,
        echo_amplitudes = echo_amplitudes,
        summary_file = summary_file,
        figure_file = figure_file,
    )
end



run_rabi_sweep(
    SECOND_PULSE_ALPHA_MIN,
    SECOND_PULSE_ALPHA_MAX,
    N_SWEEP_POINTS,
    SECOND_PULSE_ALPHA_NORMALIZATION,
    BASE_SIM_SETTING,
    SYSTEM_CONFIG,
    SWEEP_OUTDIR;
    skip_existing = SKIP_EXISTING,
)

nothing
