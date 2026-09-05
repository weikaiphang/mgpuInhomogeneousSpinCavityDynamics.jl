
using JLD2
using Plots
using Measures
using Statistics
using Printf


load_file = "data/demo.jld2"

figdir = "fig"
mkpath(figdir)

echo1_zoom_half_span_us = 30.0
echo2_zoom_half_span_us = 30.0

whole_plot_ymin = -2000.0
whole_plot_ymax =  2000.0

delta_plot_min = -0.5e6
delta_plot_max =  0.5e6

g_plot_min = 0.0
g_plot_max = 15000.0

delta_window_MHz = 1.0

selected_g_values_Hz = [
    1.0,
    10.0,
    100.0,
    1000.0,
    5000.0,
    10000.0,
    15000.0,
]

wurst_zoom_padding_us = 5.0



function as_symbol(x)
    return Symbol(lowercase(String(x)))
end


function pulse_kind(pulse)
    hasproperty(pulse, :kind) || error("A PULSE_CONFIG entry has no field :kind")
    return as_symbol(getproperty(pulse, :kind))
end


function find_peak_result(results, target_label::Symbol, fallback_index::Int)
    results === nothing && return nothing
    isempty(results) && return nothing

    i = findfirst(results) do r
        hasproperty(r, :label) && as_symbol(getproperty(r, :label)) == target_label
    end

    if i !== nothing
        return results[i]
    end

    return length(results) >= fallback_index ? results[fallback_index] : nothing
end


function ensure_2d(A, M_delta::Int, M_g::Int, name::AbstractString)
    if ndims(A) == 2
        size(A) == (M_delta, M_g) || error(
            "$name has size $(size(A)); expected ($M_delta, $M_g)."
        )
        return Array(A)
    elseif ndims(A) == 1
        length(A) == M_delta * M_g || error(
            "$name has length $(length(A)); expected $(M_delta * M_g)."
        )
        return reshape(Array(A), M_delta, M_g)
    else
        error("$name must be a vector or matrix, but ndims($name) = $(ndims(A)).")
    end
end


function normalize_resonant_rows(A, Nj_keep)
    M_g_local, Nt_local = size(A)
    length(Nj_keep) == M_g_local || error("Nj_keep length does not match A rows.")

    T = eltype(A) <: Complex ? ComplexF64 : Float64
    out = Matrix{T}(undef, M_g_local, Nt_local)

    for ig in 1:M_g_local
        denom = Nj_keep[ig] / 2

        if denom > 0
            out[ig, :] .= A[ig, :] ./ denom
        else
            if T <: Complex
                out[ig, :] .= NaN + NaN * im
            else
                out[ig, :] .= NaN
            end
        end
    end

    return out
end


function time_window_indices(t_us, center_us, half_span_us)
    idx = findall(
        (t_us .>= center_us - half_span_us) .&
        (t_us .<= center_us + half_span_us)
    )

    isempty(idx) && error(
        "The time window centered at $center_us μs with half-span " *
        "$half_span_us μs contains no saved points."
    )

    return idx
end


function display_and_save(p, filename, figdir)
    display(p)
    savefig(p, joinpath(figdir, filename))
    return nothing
end


function add_sequence_markers!(
    p;
    tsig_us,
    tw1_center_us,
    techo1_expected_us,
    tw2_center_us,
    techo2_expected_us,
)
    vline!(p, [tsig_us];
        ls=:dash, lc=:gray, lw=1.8, label="signal")

    vline!(p, [tw1_center_us];
        ls=:dash, lc=:red, lw=1.8, label="WURST1")

    vline!(p, [techo1_expected_us];
        ls=:dash, lc=:green, lw=1.8, label="Echo1 expected")

    vline!(p, [tw2_center_us];
        ls=:dash, lc=:red, lw=1.8, label="WURST2")

    vline!(p, [techo2_expected_us];
        ls=:dash, lc=:green, lw=1.8, label="Echo2 expected")

    return p
end


function pulse_window_analysis(t_us, center_us, half_span_us, y)
    idx = time_window_indices(t_us, center_us, half_span_us)

    t_zoom = t_us[idx]
    y_zoom = y[idx]

    length(t_zoom) >= 2 || error("Pulse-analysis window needs at least two points.")

    imax = argmax(abs.(y_zoom))
    ymax = y_zoom[imax]
    tmax = t_zoom[imax]

    dt = diff(t_zoom)

    integral = sum(
        0.5 .* (y_zoom[1:end-1] .+ y_zoom[2:end]) .* dt
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


@load load_file data

SIM_SETTING   = data.SIM_SETTING
SYSTEM_CONFIG = data.SYSTEM_CONFIG
PULSE_CONFIG  = data.PULSE_CONFIG

t_s  = collect(data.t_saved)
t_us = t_s .* 1e6
Nt   = length(t_s)

a_sol = collect(data.a_sol)

Σp = collect(data.Σp_sol)
Σz = real.(collect(data.Σz_sol))

E_of_t_arr = collect(data.E_of_t_arr)


hasproperty(SYSTEM_CONFIG, :kappa_e) || error(
    "SYSTEM_CONFIG contains no kappa_e."
)

kappa_e = Float64(SYSTEM_CONFIG.kappa_e)

a_out = E_of_t_arr .- sqrt(kappa_e) .* a_sol

a_out_x   = real.(a_out)
a_out_p   = imag.(a_out)
a_out_abs = abs.(a_out)


M_delta = Int(data.M_delta)
M_g     = Int(data.M_g)
M_total = Int(data.M_total)

delta_b_1d = collect(data.delta_b_1d)
g_b_1d     = collect(data.g_b_1d)
Nj_2d      = Array(data.Nj_2d)

idelta_res = Int(data.idelta_res)
delta_res  = data.delta_res

keep_bins  = Int.(collect(data.keep_bins))
g_keep     = collect(data.g_keep)
delta_keep = collect(data.delta_keep)

Sp_keep = ComplexF64.(Array(data.Sp_keep))
Sz_keep = Float64.(real.(Array(data.Sz_keep)))

peak_detection_config  = data.peak_detection_config
peak_detection_results = data.peak_detection_results

N_total        = data.N_total
elapsed_seconds = data.elapsed_seconds


@assert M_total == M_delta * M_g
@assert length(delta_b_1d) == M_delta
@assert length(g_b_1d) == M_g
@assert size(Nj_2d) == (M_delta, M_g)

@assert length(t_s) == length(a_sol)
@assert length(t_s) == length(Σp)
@assert length(t_s) == length(Σz)
@assert length(t_s) == length(E_of_t_arr)
@assert length(t_s) == length(a_out)
@assert length(t_s) == length(a_out_x)
@assert length(t_s) == length(a_out_p)
@assert length(t_s) == length(a_out_abs)

@assert length(keep_bins) == M_g
@assert length(g_keep) == M_g
@assert length(delta_keep) == M_g
@assert size(Sp_keep) == (M_g, Nt)
@assert size(Sz_keep) == (M_g, Nt)

expected_keep_bins = [
    idelta_res + (ig - 1) * M_delta
    for ig in 1:M_g
]

@assert keep_bins == expected_keep_bins


gaussian_pulses = [p for p in PULSE_CONFIG if pulse_kind(p) == :gaussian]
wurst_pulses    = [p for p in PULSE_CONFIG if pulse_kind(p) == :wurst]

isempty(gaussian_pulses) && error("No Gaussian signal pulse found in PULSE_CONFIG.")
length(wurst_pulses) < 2 && error("At least two WURST pulses are required for ROSE plotting.")

signal_pulse = sort(gaussian_pulses; by=p -> p.t0)[1]
wurst_pulses = sort(wurst_pulses; by=p -> p.t_center)

wurst1 = wurst_pulses[1]
wurst2 = wurst_pulses[2]

tsig       = signal_pulse.t0
sigma_sig  = signal_pulse.sigma

tw1_center = wurst1.t_center
Tw1        = wurst1.duration
t0w1       = tw1_center - Tw1 / 2

tw2_center = wurst2.t_center
Tw2        = wurst2.duration
t0w2       = tw2_center - Tw2 / 2

echo1 = find_peak_result(peak_detection_results, :echo1, 1)
echo2 = find_peak_result(peak_detection_results, :echo2, 2)

has_peak_results = echo1 !== nothing && echo2 !== nothing

if has_peak_results
    techo1_expected = echo1.expected_time
    techo2_expected = echo2.expected_time

    t_echo1_detected = echo1.detected_time
    t_echo2_detected = echo2.detected_time

    amp_echo1_detected = echo1.detected_amplitude
    amp_echo2_detected = echo2.detected_amplitude
else

    techo1_expected = 2 * tw1_center - tsig
    techo2_expected = 2 * tw2_center - techo1_expected

    t_echo1_detected = NaN
    t_echo2_detected = NaN

    amp_echo1_detected = NaN
    amp_echo2_detected = NaN
end

Ttotal = hasproperty(SIM_SETTING, :Ttotal) ? SIM_SETTING.Ttotal : last(t_s)

tsig_us             = tsig * 1e6
sigma_sig_us        = sigma_sig * 1e6

tw1_center_us       = tw1_center * 1e6
tw2_center_us       = tw2_center * 1e6

techo1_expected_us  = techo1_expected * 1e6
techo2_expected_us  = techo2_expected * 1e6

t_echo1_detected_us = t_echo1_detected * 1e6
t_echo2_detected_us = t_echo2_detected * 1e6


sp_avg = Σp ./ (N_total / 2)
sx_avg = real.(sp_avg)
sy_avg = imag.(sp_avg)
sz_avg = Σz ./ (N_total / 2)

a_in_x   = real.(E_of_t_arr)
a_in_p   = imag.(E_of_t_arr)
a_in_abs = abs.(E_of_t_arr)

Nj_flat = vec(Nj_2d)
Nj_keep = Nj_flat[keep_bins]

sigma_p_keep = normalize_resonant_rows(Sp_keep, Nj_keep)
sigma_z_keep = normalize_resonant_rows(Sz_keep, Nj_keep)

g_keep_Hz = g_keep ./ (2π)

delta_axis_Hz  = delta_b_1d ./ (2π)
g_axis_Hz      = g_b_1d ./ (2π)
delta_axis_MHz = delta_axis_Hz ./ 1e6

println()
println("============================================================")
println("Loaded first-order ROSE data")
println("============================================================")
println("File                 = $load_file")
println("Nt                   = $Nt")
println("M_delta              = $M_delta")
println("M_g                  = $M_g")
println("M_total              = $M_total")
println("N_total              = $N_total")
println("Sp_keep size         = $(size(Sp_keep))")
println("Sz_keep size         = $(size(Sz_keep))")
println("idelta_res           = $idelta_res")
println("delta_res / 2π       = $(delta_res / (2π)) Hz")
println("number of kept bins  = $(length(keep_bins))")
println("elapsed_seconds      = $elapsed_seconds")
println("peak results present = $has_peak_results")
println("============================================================")
println()


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


idx_delta_plot = findall(
    (delta_axis_Hz .>= delta_plot_min) .&
    (delta_axis_Hz .<= delta_plot_max)
)

idx_g_plot = findall(
    (g_axis_Hz .>= g_plot_min) .&
    (g_axis_Hz .<= g_plot_max)
)

isempty(idx_delta_plot) && error(
    "No detuning bins lie inside delta_plot_min/max."
)

isempty(idx_g_plot) && error(
    "No g bins lie inside g_plot_min/max."
)

delta_plot_Hz = delta_axis_Hz[idx_delta_plot]
g_plot_Hz     = g_axis_Hz[idx_g_plot]

println("2D plot region:")
println("δ / 2π = [$(minimum(delta_plot_Hz)), $(maximum(delta_plot_Hz))] Hz")
println("g / 2π = [$(minimum(g_plot_Hz)), $(maximum(g_plot_Hz))] Hz")
println()


plot_delta = bar(
    delta_axis_Hz,
    vec(sum(Nj_2d; dims=2));
    legend=false,
    xlabel="δ / 2π (Hz)",
    ylabel="Number of spins",
    title="Detuning histogram",
)

display_and_save(plot_delta, "plot_delta_histogram.png", figdir)


plot_g = bar(
    g_axis_Hz,
    vec(sum(Nj_2d; dims=1));
    legend=false,
    xlabel="g / 2π (Hz)",
    ylabel="Number of spins",
    title="g histogram",
)

display_and_save(plot_g, "plot_g_histogram.png", figdir)


plot_spins = plot(t_us, sx_avg; label="⟨σx⟩")
plot!(plot_spins, t_us, sy_avg; label="⟨σy⟩")
plot!(plot_spins, t_us, sz_avg; label="⟨σz⟩")

add_sequence_markers!(
    plot_spins;
    tsig_us=tsig_us,
    tw1_center_us=tw1_center_us,
    techo1_expected_us=techo1_expected_us,
    tw2_center_us=tw2_center_us,
    techo2_expected_us=techo2_expected_us,
)

xlabel!(plot_spins, "time (μs)")
ylabel!(plot_spins, "mean value")
title!(plot_spins, "Global spin dynamics")

display_and_save(plot_spins, "plot_spins.png", figdir)


plot_drive = plot(t_us, a_in_abs; label="|E(t)|")

add_sequence_markers!(
    plot_drive;
    tsig_us=tsig_us,
    tw1_center_us=tw1_center_us,
    techo1_expected_us=techo1_expected_us,
    tw2_center_us=tw2_center_us,
    techo2_expected_us=techo2_expected_us,
)

xlabel!(plot_drive, "time (μs)")
ylabel!(plot_drive, "|E(t)|")
title!(plot_drive, "Input + WURST drive amplitude")

display_and_save(plot_drive, "plot_drive_amplitude.png", figdir)


signal_zoom_half_span_us = 5 * sigma_sig_us
idx_signal = time_window_indices(t_us, tsig_us, signal_zoom_half_span_us)

plot_signal = plot(t_us[idx_signal], a_in_x[idx_signal]; label="Re[a_in]")
plot!(plot_signal, t_us[idx_signal], a_in_p[idx_signal]; label="Im[a_in]")
plot!(plot_signal, t_us[idx_signal], a_out_x[idx_signal]; label="Re[a_out]")
plot!(plot_signal, t_us[idx_signal], a_out_p[idx_signal]; label="Im[a_out]")
plot!(plot_signal, t_us[idx_signal], a_out_abs[idx_signal]; label="|a_out|")

vline!(plot_signal, [tsig_us]; ls=:dash, lc=:gray, label="signal center")

xlabel!(plot_signal, "time (μs)")
ylabel!(plot_signal, "Amplitude (s⁻¹ᐟ²)")
title!(plot_signal, "Input pulse zoom")

display_and_save(plot_signal, "plot_input_zoom.png", figdir)


idx_echo1_zoom = time_window_indices(
    t_us,
    techo1_expected_us,
    echo1_zoom_half_span_us,
)

plot_echo1 = plot(t_us[idx_echo1_zoom], a_out_x[idx_echo1_zoom]; label="Re[a_out]")
plot!(plot_echo1, t_us[idx_echo1_zoom], a_out_p[idx_echo1_zoom]; label="Im[a_out]")
plot!(plot_echo1, t_us[idx_echo1_zoom], a_out_abs[idx_echo1_zoom]; label="|a_out|")

vline!(plot_echo1, [techo1_expected_us];
    ls=:dash, lc=:green, label="expected echo1")

if has_peak_results
    vline!(plot_echo1, [t_echo1_detected_us];
        ls=:dot, lc=:black, lw=2.0, label="detected peak")
end

xlabel!(plot_echo1, "time (μs)")
ylabel!(plot_echo1, "Amplitude (s⁻¹ᐟ²)")
title!(plot_echo1, "First echo zoom")

display_and_save(plot_echo1, "plot_echo1_zoom.png", figdir)


idx_echo2_zoom = time_window_indices(
    t_us,
    techo2_expected_us,
    echo2_zoom_half_span_us,
)

plot_echo2 = plot(t_us[idx_echo2_zoom], a_out_x[idx_echo2_zoom]; label="Re[a_out]")
plot!(plot_echo2, t_us[idx_echo2_zoom], a_out_p[idx_echo2_zoom]; label="Im[a_out]")
plot!(plot_echo2, t_us[idx_echo2_zoom], a_out_abs[idx_echo2_zoom]; label="|a_out|")

vline!(plot_echo2, [techo2_expected_us];
    ls=:dash, lc=:green, label="expected echo2")

if has_peak_results
    vline!(plot_echo2, [t_echo2_detected_us];
        ls=:dot, lc=:black, lw=2.0, label="detected peak")
end

xlabel!(plot_echo2, "time (μs)")
ylabel!(plot_echo2, "Amplitude (s⁻¹ᐟ²)")
title!(plot_echo2, "Second echo zoom")

display_and_save(plot_echo2, "plot_echo2_zoom.png", figdir)


selected_g_indices = unique([
    argmin(abs.(g_keep_Hz .- g_target))
    for g_target in selected_g_values_Hz
])

println("Selected resonant g-bin trajectories:")
for ig in selected_g_indices
    println(
        "ig = $ig, " *
        "g / 2π = $(g_keep_Hz[ig]) Hz, " *
        "flat bin = $(keep_bins[ig]), " *
        "Nj = $(Nj_keep[ig])"
    )
end
println()

ig_first = first(selected_g_indices)

plot_selected_sz = plot(
    t_us,
    vec(sigma_z_keep[ig_first, :]);
    label="g/2π = $(round(g_keep_Hz[ig_first], sigdigits=5)) Hz",
    xlabel="time (μs)",
    ylabel="⟨σz⟩",
    title="Resonant-spin Sz trajectories for selected g bins",
    linewidth=2,
)

for ig in Iterators.drop(selected_g_indices, 1)
    plot!(
        plot_selected_sz,
        t_us,
        vec(sigma_z_keep[ig, :]);
        label="g/2π = $(round(g_keep_Hz[ig], sigdigits=5)) Hz",
        linewidth=2,
    )
end

add_sequence_markers!(
    plot_selected_sz;
    tsig_us=tsig_us,
    tw1_center_us=tw1_center_us,
    techo1_expected_us=techo1_expected_us,
    tw2_center_us=tw2_center_us,
    techo2_expected_us=techo2_expected_us,
)

display_and_save(
    plot_selected_sz,
    "resonant_Sz_selected_g_trajectories.png",
    figdir,
)

plot_selected_sp_abs = plot(
    t_us,
    abs.(vec(sigma_p_keep[ig_first, :]));
    label="g/2π = $(round(g_keep_Hz[ig_first], sigdigits=5)) Hz",
    xlabel="time (μs)",
    ylabel="|⟨σ+⟩|",
    title="Resonant-spin |S⁺| trajectories for selected g bins",
    linewidth=2,
)

for ig in Iterators.drop(selected_g_indices, 1)
    plot!(
        plot_selected_sp_abs,
        t_us,
        abs.(vec(sigma_p_keep[ig, :]));
        label="g/2π = $(round(g_keep_Hz[ig], sigdigits=5)) Hz",
        linewidth=2,
    )
end

add_sequence_markers!(
    plot_selected_sp_abs;
    tsig_us=tsig_us,
    tw1_center_us=tw1_center_us,
    techo1_expected_us=techo1_expected_us,
    tw2_center_us=tw2_center_us,
    techo2_expected_us=techo2_expected_us,
)

display_and_save(
    plot_selected_sp_abs,
    "resonant_abs_Sp_selected_g_trajectories.png",
    figdir,
)

plot_selected_sp_phase = plot(
    t_us,
    angle.(vec(sigma_p_keep[ig_first, :]));
    label="g/2π = $(round(g_keep_Hz[ig_first], sigdigits=5)) Hz",
    xlabel="time (μs)",
    ylabel="arg(⟨σ+⟩) (rad)",
    title="Resonant-spin phase trajectories for selected g bins",
    linewidth=2,
    ylims=(-π, π),
)

for ig in Iterators.drop(selected_g_indices, 1)
    plot!(
        plot_selected_sp_phase,
        t_us,
        angle.(vec(sigma_p_keep[ig, :]));
        label="g/2π = $(round(g_keep_Hz[ig], sigdigits=5)) Hz",
        linewidth=2,
    )
end

add_sequence_markers!(
    plot_selected_sp_phase;
    tsig_us=tsig_us,
    tw1_center_us=tw1_center_us,
    techo1_expected_us=techo1_expected_us,
    tw2_center_us=tw2_center_us,
    techo2_expected_us=techo2_expected_us,
)

display_and_save(
    plot_selected_sp_phase,
    "resonant_phase_Sp_selected_g_trajectories.png",
    figdir,
)


plot_io = plot(t_us, a_in_x; label="Re[a_in]")
plot!(plot_io, t_us, a_in_p; label="Im[a_in]")
plot!(plot_io, t_us, a_out_x; label="Re[a_out]")
plot!(plot_io, t_us, a_out_p; label="Im[a_out]")
plot!(plot_io, t_us, a_out_abs; label="|a_out|")

add_sequence_markers!(
    plot_io;
    tsig_us=tsig_us,
    tw1_center_us=tw1_center_us,
    techo1_expected_us=techo1_expected_us,
    tw2_center_us=tw2_center_us,
    techo2_expected_us=techo2_expected_us,
)

xlabel!(plot_io, "time (μs)")
ylabel!(plot_io, "Amplitude (s⁻¹ᐟ²)")
title!(plot_io, "Whole-time input-output field")

display_and_save(plot_io, "plot_input_output.png", figdir)


if whole_plot_ymin !== nothing && whole_plot_ymax !== nothing
    plot_io_limited = deepcopy(plot_io)
    ylims!(plot_io_limited, whole_plot_ymin, whole_plot_ymax)

    display_and_save(
        plot_io_limited,
        "plot_input_output_limited.png",
        figdir,
    )
end


w1_start_us = t0w1 * 1e6
w1_end_us   = (t0w1 + Tw1) * 1e6

idx_w1 = findall(
    (t_us .>= w1_start_us - wurst_zoom_padding_us) .&
    (t_us .<= w1_end_us + wurst_zoom_padding_us)
)

isempty(idx_w1) && error("WURST1 zoom window is empty.")

plot_w1 = plot(t_us[idx_w1], real.(E_of_t_arr[idx_w1]); label="Re[E(t)]")
plot!(plot_w1, t_us[idx_w1], imag.(E_of_t_arr[idx_w1]); label="Im[E(t)]")
plot!(plot_w1, t_us[idx_w1], abs.(E_of_t_arr[idx_w1]); label="|E(t)|")

vline!(plot_w1, [tw1_center_us]; ls=:dash, lc=:red, label="WURST1 center")

xlabel!(plot_w1, "time (μs)")
ylabel!(plot_w1, "Drive amplitude")
title!(plot_w1, "WURST1 zoom")

display_and_save(plot_w1, "plot_wurst1_zoom.png", figdir)


w2_start_us = t0w2 * 1e6
w2_end_us   = (t0w2 + Tw2) * 1e6

idx_w2 = findall(
    (t_us .>= w2_start_us - wurst_zoom_padding_us) .&
    (t_us .<= w2_end_us + wurst_zoom_padding_us)
)

isempty(idx_w2) && error("WURST2 zoom window is empty.")

plot_w2 = plot(t_us[idx_w2], real.(E_of_t_arr[idx_w2]); label="Re[E(t)]")
plot!(plot_w2, t_us[idx_w2], imag.(E_of_t_arr[idx_w2]); label="Im[E(t)]")
plot!(plot_w2, t_us[idx_w2], abs.(E_of_t_arr[idx_w2]); label="|E(t)|")

vline!(plot_w2, [tw2_center_us]; ls=:dash, lc=:red, label="WURST2 center")

xlabel!(plot_w2, "time (μs)")
ylabel!(plot_w2, "Drive amplitude")
title!(plot_w2, "WURST2 zoom")

display_and_save(plot_w2, "plot_wurst2_zoom.png", figdir)


if has_peak_results
    Sp_echo1_2d = ensure_2d(
        echo1.Sp_at_peak_2d,
        M_delta,
        M_g,
        "echo1.Sp_at_peak_2d",
    )

    Sz_echo1_2d = real.(ensure_2d(
        echo1.Sz_at_peak_2d,
        M_delta,
        M_g,
        "echo1.Sz_at_peak_2d",
    ))

    Sp_echo2_2d = ensure_2d(
        echo2.Sp_at_peak_2d,
        M_delta,
        M_g,
        "echo2.Sp_at_peak_2d",
    )

    Sz_echo2_2d = real.(ensure_2d(
        echo2.Sz_at_peak_2d,
        M_delta,
        M_g,
        "echo2.Sz_at_peak_2d",
    ))

    phase_echo1_2d = angle.(Sp_echo1_2d)
    phase_echo2_2d = angle.(Sp_echo2_2d)

    sp_clim_max = maximum([
        maximum(abs.(Sp_echo1_2d[idx_delta_plot, idx_g_plot])),
        maximum(abs.(Sp_echo2_2d[idx_delta_plot, idx_g_plot])),
    ])
    sp_clim_max = max(sp_clim_max, eps(Float64))

    sz_clim_min = minimum([
        minimum(Sz_echo1_2d[idx_delta_plot, idx_g_plot]),
        minimum(Sz_echo2_2d[idx_delta_plot, idx_g_plot]),
    ])

    sz_clim_max = maximum([
        maximum(Sz_echo1_2d[idx_delta_plot, idx_g_plot]),
        maximum(Sz_echo2_2d[idx_delta_plot, idx_g_plot]),
    ])

    if sz_clim_min == sz_clim_max
        pad = max(abs(sz_clim_min), 1.0) * 1e-12
        sz_clim_min -= pad
        sz_clim_max += pad
    end


    plot_phase1 = heatmap(
        delta_plot_Hz,
        g_plot_Hz,
        phase_echo1_2d[idx_delta_plot, idx_g_plot]';
        xlabel="δ / 2π (Hz)",
        ylabel="g / 2π (Hz)",
        title="Spin phase at detected Echo1 peak",
        colorbar_title="Phase (rad)",
        clims=(-π, π),
    )

    display_and_save(
        plot_phase1,
        "phase_echo1_delta_x_g_y.png",
        figdir,
    )


    plot_phase2 = heatmap(
        delta_plot_Hz,
        g_plot_Hz,
        phase_echo2_2d[idx_delta_plot, idx_g_plot]';
        xlabel="δ / 2π (Hz)",
        ylabel="g / 2π (Hz)",
        title="Spin phase at detected Echo2 peak",
        colorbar_title="Phase (rad)",
        clims=(-π, π),
    )

    display_and_save(
        plot_phase2,
        "phase_echo2_delta_x_g_y.png",
        figdir,
    )






    idx_delta_phase = findall(abs.(delta_axis_MHz) .<= delta_window_MHz)

    isempty(idx_delta_phase) && error(
        "No detuning bins lie inside ±$delta_window_MHz MHz."
    )

    println(
        "Phase-versus-g: keeping $(length(idx_delta_phase)) of " *
        "$(length(delta_axis_MHz)) detuning bins."
    )
    println("Detuning range = ±$delta_window_MHz MHz")
    println()

    Sp_sum_g_echo1 = vec(sum(Sp_echo1_2d[idx_delta_phase, :]; dims=1))
    Sp_sum_g_echo2 = vec(sum(Sp_echo2_2d[idx_delta_phase, :]; dims=1))

    phase_vs_g_echo1 = angle.(Sp_sum_g_echo1)
    phase_vs_g_echo2 = angle.(Sp_sum_g_echo2)

    plot_phase_g = plot(
        g_axis_Hz,
        phase_vs_g_echo1;
        label="Echo1",
        xlabel="g / 2π (Hz)",
        ylabel="Phase of ∑δ S⁺ (rad)",
        title="Spin phase vs g (|δ| ≤ $delta_window_MHz MHz)",
        linewidth=2,
        ylims=(-π, π),
    )

    plot!(
        plot_phase_g,
        g_axis_Hz,
        phase_vs_g_echo2;
        label="Echo2",
        linewidth=2,
    )

    xlims!(plot_phase_g, g_plot_min, g_plot_max)

    display_and_save(plot_phase_g, "phase_vs_g.png", figdir)
else
    @warn(
        "peak_detection_results is empty or incomplete. " *
        "Skipping peak-window plots, echo heatmaps, and phase-versus-g."
    )
end


analysis_half_span_us = 15 * sigma_sig_us

input_analysis = pulse_window_analysis(
    t_us,
    tsig_us,
    analysis_half_span_us,
    a_in_abs,
)

echo1_analysis = pulse_window_analysis(
    t_us,
    techo1_expected_us,
    analysis_half_span_us,
    a_out_abs,
)

echo2_analysis = pulse_window_analysis(
    t_us,
    techo2_expected_us,
    analysis_half_span_us,
    a_out_abs,
)

ain_max   = input_analysis.maximum
echo1_max = echo1_analysis.maximum
echo2_max = echo2_analysis.maximum

result_peak_echo1 = echo1_max / ain_max
result_peak_echo2 = echo2_max / ain_max

result_integral_echo1 = echo1_analysis.integral / input_analysis.integral
result_integral_echo2 = echo2_analysis.integral / input_analysis.integral

result_norm_echo1 = echo1_analysis.norm / input_analysis.norm
result_norm_echo2 = echo2_analysis.norm / input_analysis.norm


println()
println("============================================================")
println("ROSE retrieve summary")
println("============================================================")
println("File = $load_file")
println("M_delta = $M_delta")
println("M_g = $M_g")
println("M_total = $M_total")
println("N_total = $N_total")
println("Ttotal = $(Ttotal * 1e6) μs")
println()
println("Signal center = $tsig_us μs")
println("Signal sigma  = $sigma_sig_us μs")
println("WURST1 center = $tw1_center_us μs")
println("WURST1 duration = $(Tw1 * 1e6) μs")
println("WURST2 center = $tw2_center_us μs")
println("WURST2 duration = $(Tw2 * 1e6) μs")
println()
println("Echo1 expected time = $techo1_expected_us μs")
println("Echo2 expected time = $techo2_expected_us μs")

if has_peak_results
    println("Echo1 detected time = $t_echo1_detected_us μs")
    println("Echo2 detected time = $t_echo2_detected_us μs")
    println("Echo1 detected amplitude = $amp_echo1_detected")
    println("Echo2 detected amplitude = $amp_echo2_detected")
end

println()
println("2D plot region:")
println("δ / 2π = [$delta_plot_min, $delta_plot_max] Hz")
println("g / 2π = [$g_plot_min, $g_plot_max] Hz")
println()
println("Input max time = $(input_analysis.time_at_maximum) μs")
println("Echo1 max time = $(echo1_analysis.time_at_maximum) μs")
println("Echo2 max time = $(echo2_analysis.time_at_maximum) μs")
println()
println("Echo1 result from peak      = $result_peak_echo1")
println("Echo1 result from uniform f = $result_integral_echo1")
println("Echo1 result from a_out f   = $result_norm_echo1")
println()
println("Echo2 result from peak      = $result_peak_echo2")
println("Echo2 result from uniform f = $result_integral_echo2")
println("Echo2 result from a_out f   = $result_norm_echo2")
println()
println("Simulation elapsed time = $elapsed_seconds seconds")
println("Figures saved in: $(abspath(figdir))")
println("============================================================")
