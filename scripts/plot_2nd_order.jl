using JLD2
using Plots
using Printf
using Measures


FILE = "data/demo.jld2"

FIG_DIR = "fig"


data = JLD2.load(FILE, "data")

SIM_SETTING   = data.SIM_SETTING
SYSTEM_CONFIG = data.SYSTEM_CONFIG
PULSE_CONFIG = data.PULSE_CONFIG

t_saved = Float64.(data.t_saved)

times = t_saved .* 1e6

a_sol = ComplexF64.(data.a_sol)
n_sol = real.(ComplexF64.(data.n_sol))

adad_sol = ComplexF64.(data.adad_sol)

E_of_t_arr = ComplexF64.(data.E_of_t_arr)

delta_b_1d = Float64.(data.delta_b_1d)
g_b_1d     = Float64.(data.g_b_1d)
Nj_2d      = Float64.(data.Nj_2d)

N_total = Float64(data.N_total)

elapsed_seconds = Float64(data.elapsed_seconds)

M_delta = hasproperty(SIM_SETTING, :M_delta) ?
          Int(SIM_SETTING.M_delta) :
          size(Nj_2d, 1)

M_g = hasproperty(SIM_SETTING, :M_g) ?
      Int(SIM_SETTING.M_g) :
      size(Nj_2d, 2)

Nj = vec(Nj_2d)

M = length(Nj)

Nt = length(t_saved)


function as_bin_time_matrix(A, M, Nt, name)
    B = ComplexF64.(Array(A))

    if ndims(B) != 2
        error(
            "$name must be a two-dimensional array, " *
            "but its size is $(size(B))."
        )
    end

    if size(B) == (M, Nt)
        return B

    elseif size(B) == (Nt, M)
        @warn "$name was stored as time × bin. Transposing it."
        return permutedims(B)

    else
        error(
            "Unexpected size for $name: $(size(B)).\n" *
            "Expected either ($M, $Nt) or ($Nt, $M)."
        )
    end
end

Sp = as_bin_time_matrix(
    data.Sp_sol,
    M,
    Nt,
    "Sp_sol",
)

Sz = as_bin_time_matrix(
    data.Sz_sol,
    M,
    Nt,
    "Sz_sol",
)

adSp = as_bin_time_matrix(
    data.adSp_sol,
    M,
    Nt,
    "adSp_sol",
)

adSm = as_bin_time_matrix(
    data.adSm_sol,
    M,
    Nt,
    "adSm_sol",
)

adSz = as_bin_time_matrix(
    data.adSz_sol,
    M,
    Nt,
    "adSz_sol",
)

for (name, array) in (
    ("a_sol", a_sol),
    ("n_sol", n_sol),
    ("adad_sol", adad_sol),
    ("E_of_t_arr", E_of_t_arr),
)
    if length(array) != Nt
        error(
            "$name has length $(length(array)), " *
            "but t_saved has length $Nt."
        )
    end
end


Σp = vec(sum(Sp, dims=1))
Σz = vec(sum(Sz, dims=1))

Σx = real.(Σp)
Σy = imag.(Σp)
Σz_real = real.(Σz)

sx_avg = Σx ./ (N_total / 2)
sy_avg = Σy ./ (N_total / 2)
sz_avg = Σz_real ./ (N_total / 2)


kappa_e = Float64(SYSTEM_CONFIG.kappa_e)
kappa_i = Float64(SYSTEM_CONFIG.kappa_i)
kappa_t = kappa_e + kappa_i

sqrt_kappa_e = sqrt(kappa_e)

a_out = E_of_t_arr .- sqrt_kappa_e .* a_sol

E_in_x = real.(E_of_t_arr)
E_in_p = imag.(E_of_t_arr)

a_out_x = real.(a_out)
a_out_p = imag.(a_out)
a_out_abs = abs.(a_out)


println("Loaded file: $FILE")
println("Simulation order: $(SIM_SETTING.simulation_order)")
println("M_delta: $M_delta")
println("M_g: $M_g")
println("M_total: $M")
println("N_total: $N_total")
println("Number of saved time points: $Nt")
println("κe / 2π: $(kappa_e / (2π)) Hz")
println("κi / 2π: $(kappa_i / (2π)) Hz")
println("κt / 2π: $(kappa_t / (2π)) Hz")
println("Simulation runtime: $(round(elapsed_seconds; digits=3)) s")

N_from_bins = sum(Nj)

if !isapprox(N_from_bins, N_total; rtol=1e-8, atol=1e-8)
    @warn(
        "sum(Nj_2d) does not exactly match the saved N_total.",
        sum_Nj=N_from_bins,
        saved_N_total=N_total,
    )
end


default(
    size = (900, 600),
    dpi = 300,
    lw = 2.5,
    guidefont = font(16),
    tickfont = font(13),
    legendfont = font(13),
    titlefont = font(18),
    grid = true,
    left_margin = 3mm,
)

mkpath(FIG_DIR)


function trapz(x, y)
    length(x) == length(y) ||
        error("x and y must have the same length.")

    length(x) >= 2 ||
        error("At least two points are required for integration.")

    return sum(
        0.5 .* (y[1:end-1] .+ y[2:end]) .* diff(x)
    )
end

function zoom_plot(
    times,
    x,
    p;
    center_us,
    span_us,
    title_str,
    save_name=nothing,
)
    tmin = center_us - span_us
    tmax = center_us + span_us

    mask = (times .>= tmin) .& (times .<= tmax)

    any(mask) ||
        error("The selected zoom window contains no saved points.")

    plt = plot(
        times[mask],
        x[mask];
        lw=2,
        label="Re⟨a_out⟩",
    )

    plot!(
        plt,
        times[mask],
        p[mask];
        lw=2,
        label="Im⟨a_out⟩",
    )

    xlabel!(plt, "time (μs)")
    ylabel!(plt, "Amplitude (s⁻¹ᐟ²)")
    title!(plt, title_str)

    display(plt)

    if save_name !== nothing
        savefig(plt, save_name)
    end

    return plt, mask
end

function zoom_scalar(
    times,
    y;
    center_us,
    span_us,
    title_str,
    save_name=nothing,
)
    tmin = center_us - span_us
    tmax = center_us + span_us

    mask = (times .>= tmin) .& (times .<= tmax)

    any(mask) ||
        error("The selected zoom window contains no saved points.")

    plt = plot(
        times[mask],
        y[mask];
        lw=2,
        label="value",
    )

    xlabel!(plt, "time (μs)")
    ylabel!(plt, "value")
    title!(plt, title_str)

    display(plt)

    if save_name !== nothing
        savefig(plt, save_name)
    end

    return plt, mask
end


plot_spins = plot(
    times,
    sx_avg;
    lw=2,
    label="⟨σx⟩ = Re⟨S⁺⟩/(N/2)",
)

plot!(
    plot_spins,
    times,
    sy_avg;
    lw=2,
    label="⟨σy⟩ = Im⟨S⁺⟩/(N/2)",
)

plot!(
    plot_spins,
    times,
    sz_avg;
    lw=2,
    label="⟨σz⟩",
)

xlabel!(plot_spins, "time (μs)")
ylabel!(plot_spins, "mean value")

title!(
    plot_spins,
    "Spin dynamics " *
    "(N_total=$(round(N_total)), Mδ=$M_delta, Mg=$M_g)",
)

display(plot_spins)

savefig(
    plot_spins,
    joinpath(FIG_DIR, "plot_spins_pm.png"),
)


plot_io = plot(
    times,
    E_in_x;
    lw=2,
    label="Re[E(t)]",
)

if maximum(abs.(E_in_p)) > 1e-12
    plot!(
        plot_io,
        times,
        E_in_p;
        lw=2,
        label="Im[E(t)]",
    )
end

plot!(
    plot_io,
    times,
    a_out_x;
    lw=2,
    label="Re⟨a_out⟩",
)

plot!(
    plot_io,
    times,
    a_out_p;
    lw=2,
    label="Im⟨a_out⟩",
)

xlabel!(plot_io, "time (μs)")
ylabel!(plot_io, "Amplitude (s⁻¹ᐟ²)")
title!(plot_io, "Input / output field")

display(plot_io)

savefig(
    plot_io,
    joinpath(FIG_DIR, "plot_io_pm.png"),
)


mid_left  = max(1, fld(M, 2))
mid_right = min(M, mid_left + 1)

candidate_bins = unique([
    1,
    mid_left,
    mid_right,
    M,
])

keep_bins = [
    b for b in candidate_bins
    if Nj[b] > 0
]

if isempty(keep_bins)
    keep_bins = [argmax(Nj)]
end

println("Selected bins: $keep_bins")


first_bin = keep_bins[1]

plot_bins = plot(
    times,
    real.(Sz[first_bin, :]) ./ (Nj[first_bin] / 2);
    lw=2,
    label="⟨σz⟩ bin=$first_bin",
)

for b in keep_bins[2:end]
    plot!(
        plot_bins,
        times,
        real.(Sz[b, :]) ./ (Nj[b] / 2);
        lw=2,
        label="⟨σz⟩ bin=$b",
    )
end

xlabel!(plot_bins, "time (μs)")
ylabel!(plot_bins, "mean value")
title!(plot_bins, "Selected-bin Sz dynamics (M_total=$M)")

display(plot_bins)

savefig(
    plot_bins,
    joinpath(FIG_DIR, "plot_selected_bins_pm.png"),
)


plot_sp_bins = plot(
    times,
    real.(Sp[first_bin, :]) ./ (Nj[first_bin] / 2);
    lw=2,
    label="Re⟨S⁺⟩ bin=$first_bin",
)

for b in keep_bins[2:end]
    plot!(
        plot_sp_bins,
        times,
        real.(Sp[b, :]) ./ (Nj[b] / 2);
        lw=2,
        label="Re⟨S⁺⟩ bin=$b",
    )
end

xlabel!(plot_sp_bins, "time (μs)")
ylabel!(plot_sp_bins, "mean value")
title!(plot_sp_bins, "Selected-bin Re⟨S⁺⟩ dynamics")

display(plot_sp_bins)

savefig(
    plot_sp_bins,
    joinpath(FIG_DIR, "plot_selected_bins_Sp_real_pm.png"),
)


plot_n = plot(
    times,
    n_sol;
    lw=2,
    label="⟨a†a⟩",
)

xlabel!(plot_n, "time (μs)")
ylabel!(plot_n, "photon number")
title!(plot_n, "Photon number (M_total=$M)")

display(plot_n)

savefig(
    plot_n,
    joinpath(FIG_DIR, "photon_number_pm.png"),
)


Dx = 0.25 .* (
    adad_sol
    .+ conj.(adad_sol)
    .+ 2 .* n_sol
    .+ 1
    .- (a_sol .+ conj.(a_sol)).^2
)

Dy = 0.25 .* (
    .-adad_sol
    .- conj.(adad_sol)
    .+ 2 .* n_sol
    .+ 1
    .+ (a_sol .- conj.(a_sol)).^2
)

Dx_plot = real.(Dx)
Dy_plot = real.(Dy)

Dt = Dx_plot .+ Dy_plot

plot_var = plot(
    times,
    Dx_plot;
    lw=2,
    label="Var(X)",
)

plot!(
    plot_var,
    times,
    Dy_plot;
    lw=2,
    label="Var(Y)",
)

xlabel!(plot_var, "time (μs)")
ylabel!(plot_var, "Quadrature variance")
title!(plot_var, "Quadrature variance (M_total=$M)")

display(plot_var)

savefig(
    plot_var,
    joinpath(FIG_DIR, "quadrature_variance_pm.png"),
)

plot_total_var = plot(
    times,
    Dt;
    lw=2,
    label="Var(X) + Var(Y)",
)

xlabel!(plot_total_var, "time (μs)")
ylabel!(plot_total_var, "Total variance")
title!(plot_total_var, "Total variance (M_total=$M)")

display(plot_total_var)

savefig(
    plot_total_var,
    joinpath(FIG_DIR, "total_variance_pm.png"),
)

println("Plots done.")