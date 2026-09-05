
using JLD2
using Plots
using Measures
using Printf


load_file = "data/demo.jld2"

figdir = "fig"
mkpath(figdir)

time_plot_min_us = nothing
time_plot_max_us = nothing

whole_plot_ymin = -2000.0
whole_plot_ymax =  2000.0

plot_drive_components = true

plot_cavity_components = true

plot_output_components = true



function display_and_save(plt, filename, figdir)
    display(plt)
    savefig(plt, joinpath(figdir, filename))
    return plt
end


function select_time_indices(
    t_us,
    time_min_us,
    time_max_us,
)
    lower = time_min_us === nothing ? first(t_us) : time_min_us
    upper = time_max_us === nothing ? last(t_us)  : time_max_us

    lower <= upper || error(
        "time_plot_min_us must be smaller than or equal to " *
        "time_plot_max_us."
    )

    idx = findall(
        (t_us .>= lower) .&
        (t_us .<= upper)
    )

    isempty(idx) && error(
        "No saved time points lie inside the selected plotting range."
    )

    return idx
end


function optional_field(data, field_name::Symbol, default_value)
    if hasproperty(data, field_name)
        return getproperty(data, field_name)
    end

    return default_value
end


data = JLD2.load(load_file, "data")

SIM_SETTING = optional_field(
    data,
    :SIM_SETTING,
    nothing,
)

SYSTEM_CONFIG = optional_field(
    data,
    :SYSTEM_CONFIG,
    nothing,
)

PULSE_CONFIG = optional_field(
    data,
    :PULSE_CONFIG,
    nothing,
)

t_s  = Float64.(collect(data.t_saved))
t_us = t_s .* 1e6

Nt = length(t_s)

a_sol = ComplexF64.(collect(data.a_sol))

Σp = ComplexF64.(collect(data.Σp_sol))
Σz = real.(ComplexF64.(collect(data.Σz_sol)))

E_of_t_arr = ComplexF64.(collect(data.E_of_t_arr))

delta_b_1d = Float64.(collect(data.delta_b_1d))
g_b_1d     = Float64.(collect(data.g_b_1d))
Nj_2d      = Float64.(Array(data.Nj_2d))

M_delta = hasproperty(data, :M_delta) ?
          Int(data.M_delta) :
          length(delta_b_1d)

M_g = hasproperty(data, :M_g) ?
      Int(data.M_g) :
      length(g_b_1d)

M_total = hasproperty(data, :M_total) ?
          Int(data.M_total) :
          M_delta * M_g

N_total = hasproperty(data, :N_total) ?
          Float64(data.N_total) :
          sum(Nj_2d)

elapsed_seconds = hasproperty(data, :elapsed_seconds) ?
                  Float64(data.elapsed_seconds) :
                  NaN



hasproperty(SYSTEM_CONFIG, :kappa_e) || error(
    "SYSTEM_CONFIG contains no kappa_e."
)

kappa_e = Float64(SYSTEM_CONFIG.kappa_e)

a_out = E_of_t_arr .- sqrt(kappa_e) .* a_sol

a_out_x   = real.(a_out)
a_out_p   = imag.(a_out)
a_out_abs = abs.(a_out)


M_total == M_delta * M_g || error(
    "M_total = $M_total, but M_delta * M_g = $(M_delta * M_g)."
)

length(delta_b_1d) == M_delta || error(
    "delta_b_1d has length $(length(delta_b_1d)), " *
    "but M_delta = $M_delta."
)

length(g_b_1d) == M_g || error(
    "g_b_1d has length $(length(g_b_1d)), but M_g = $M_g."
)

size(Nj_2d) == (M_delta, M_g) || error(
    "Nj_2d has size $(size(Nj_2d)); " *
    "expected ($M_delta, $M_g)."
)

for (name, values) in (
    ("a_sol", a_sol),
    ("Σp_sol", Σp),
    ("Σz_sol", Σz),
    ("E_of_t_arr", E_of_t_arr),
    ("a_out", a_out),
    ("a_out_x", a_out_x),
    ("a_out_p", a_out_p),
    ("a_out_abs", a_out_abs),
)
    length(values) == Nt || error(
        "$name has length $(length(values)); expected $Nt."
    )
end

N_total > 0 || error(
    "N_total must be positive."
)

N_from_bins = sum(Nj_2d)

if !isapprox(N_from_bins, N_total; rtol=1e-8, atol=1e-8)
    @warn(
        "sum(Nj_2d) does not match the saved N_total.",
        sum_Nj_2d=N_from_bins,
        saved_N_total=N_total,
    )
end


delta_axis_Hz = delta_b_1d ./ (2π)
g_axis_Hz     = g_b_1d ./ (2π)

N_vs_delta = vec(sum(Nj_2d; dims=2))

N_vs_g = vec(sum(Nj_2d; dims=1))

normalization = N_total / 2

sx_avg = real.(Σp) ./ normalization
sy_avg = imag.(Σp) ./ normalization
sz_avg = Σz ./ normalization

a_in_x   = real.(E_of_t_arr)
a_in_p   = imag.(E_of_t_arr)
a_in_abs = abs.(E_of_t_arr)

a_x   = real.(a_sol)
a_p   = imag.(a_sol)
a_abs = abs.(a_sol)

n_coh = abs2.(a_sol)

a_out_intensity = abs2.(a_out)

idx_time = select_time_indices(
    t_us,
    time_plot_min_us,
    time_plot_max_us,
)

t_plot = t_us[idx_time]


println()
println("============================================================")
println("Loaded general first-order simulation data")
println("============================================================")
println("File                 = $load_file")
println("Nt                   = $Nt")
println("M_delta              = $M_delta")
println("M_g                  = $M_g")
println("M_total              = $M_total")
println("N_total              = $N_total")
println("Nj_2d size           = $(size(Nj_2d))")

if !isnan(elapsed_seconds)
    println("elapsed_seconds      = $elapsed_seconds")
end

if SIM_SETTING !== nothing &&
   hasproperty(SIM_SETTING, :simulation_order)

    println(
        "simulation_order     = " *
        "$(SIM_SETTING.simulation_order)"
    )
end

if PULSE_CONFIG !== nothing
    println("number of pulses     = $(length(PULSE_CONFIG))")
end

println(
    "plotted time range   = " *
    "[$(first(t_plot)), $(last(t_plot))] μs"
)

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


plot_delta = bar(
    delta_axis_Hz,
    N_vs_delta;
    legend=false,
    xlabel="δ / 2π (Hz)",
    ylabel="Number of spins",
    title="Detuning distribution",
)

display_and_save(
    plot_delta,
    "plot_delta_histogram.png",
    figdir,
)


plot_g = bar(
    g_axis_Hz,
    N_vs_g;
    legend=false,
    xlabel="g / 2π (Hz)",
    ylabel="Number of spins",
    title="Coupling-strength distribution",
)

display_and_save(
    plot_g,
    "plot_g_histogram.png",
    figdir,
)


plot_spins = plot(
    t_plot,
    sx_avg[idx_time];
    label="⟨σx⟩",
)

plot!(
    plot_spins,
    t_plot,
    sy_avg[idx_time];
    label="⟨σy⟩",
)

plot!(
    plot_spins,
    t_plot,
    sz_avg[idx_time];
    label="⟨σz⟩",
)

xlabel!(plot_spins, "time (μs)")
ylabel!(plot_spins, "mean value")
title!(plot_spins, "Collective spin dynamics")

display_and_save(
    plot_spins,
    "plot_spin_dynamics.png",
    figdir,
)


plot_drive = plot(
    t_plot,
    a_in_abs[idx_time];
    label="|E(t)|",
)

if plot_drive_components
    plot!(
        plot_drive,
        t_plot,
        a_in_x[idx_time];
        label="Re[E(t)]",
    )

    plot!(
        plot_drive,
        t_plot,
        a_in_p[idx_time];
        label="Im[E(t)]",
    )
end

xlabel!(plot_drive, "time (μs)")
ylabel!(plot_drive, "Drive amplitude")
title!(plot_drive, "Applied drive")

display_and_save(
    plot_drive,
    "plot_drive.png",
    figdir,
)


plot_cavity = plot(
    t_plot,
    a_abs[idx_time];
    label="|⟨a⟩|",
)

if plot_cavity_components
    plot!(
        plot_cavity,
        t_plot,
        a_x[idx_time];
        label="Re⟨a⟩",
    )

    plot!(
        plot_cavity,
        t_plot,
        a_p[idx_time];
        label="Im⟨a⟩",
    )
end

xlabel!(plot_cavity, "time (μs)")
ylabel!(plot_cavity, "Cavity amplitude")
title!(plot_cavity, "Intracavity field")

display_and_save(
    plot_cavity,
    "plot_cavity_field.png",
    figdir,
)


plot_n_coh = plot(
    t_plot,
    n_coh[idx_time];
    label="|⟨a⟩|²",
)

xlabel!(plot_n_coh, "time (μs)")
ylabel!(plot_n_coh, "Coherent photon number")
title!(plot_n_coh, "Coherent intracavity photon number")

display_and_save(
    plot_n_coh,
    "plot_coherent_photon_number.png",
    figdir,
)


plot_io = plot(
    t_plot,
    a_in_abs[idx_time];
    label="|a_in|",
)

plot!(
    plot_io,
    t_plot,
    a_out_abs[idx_time];
    label="|a_out|",
)

if plot_drive_components
    plot!(
        plot_io,
        t_plot,
        a_in_x[idx_time];
        label="Re[a_in]",
    )

    plot!(
        plot_io,
        t_plot,
        a_in_p[idx_time];
        label="Im[a_in]",
    )
end

if plot_output_components
    plot!(
        plot_io,
        t_plot,
        a_out_x[idx_time];
        label="Re[a_out]",
    )

    plot!(
        plot_io,
        t_plot,
        a_out_p[idx_time];
        label="Im[a_out]",
    )
end

xlabel!(plot_io, "time (μs)")
ylabel!(plot_io, "Amplitude (s⁻¹ᐟ²)")
title!(plot_io, "Input-output field")

display_and_save(
    plot_io,
    "plot_input_output.png",
    figdir,
)


if whole_plot_ymin !== nothing &&
   whole_plot_ymax !== nothing

    whole_plot_ymin < whole_plot_ymax || error(
        "whole_plot_ymin must be smaller than whole_plot_ymax."
    )

    plot_io_limited = deepcopy(plot_io)

    ylims!(
        plot_io_limited,
        whole_plot_ymin,
        whole_plot_ymax,
    )

    display_and_save(
        plot_io_limited,
        "plot_input_output_limited.png",
        figdir,
    )
end


plot_output_intensity = plot(
    t_plot,
    a_out_intensity[idx_time];
    label="|a_out|²",
)

xlabel!(plot_output_intensity, "time (μs)")
ylabel!(plot_output_intensity, "Output intensity")
title!(plot_output_intensity, "Coherent output-field intensity")

display_and_save(
    plot_output_intensity,
    "plot_output_intensity.png",
    figdir,
)

println("All general first-order plots are complete.")
println("Figures were saved in: $figdir")