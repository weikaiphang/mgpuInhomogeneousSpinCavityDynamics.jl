# ============================================================
# PULSE CONSTRUCTORS
# ============================================================

function gaussian_drive(; t0, sigma, amp, omega=0.0, phase=0.0)
    t0_f    = Float64(t0)
    sigma_f = Float64(sigma)
    amp_c   = ComplexF64(amp)
    omega_f = Float64(omega)
    phase_f = Float64(phase)

    return function (t)
        τ = t - t0_f

        return amp_c *
               exp(-(τ^2) / (2*sigma_f^2)) *
               exp(1im * (omega_f * τ + phase_f))
    end
end

function wurst_drive(;
    t_center,
    duration,
    amp,
    bandwidth,
    n=20.0,
    omega0=0.0,
    chirp_sign=+1.0,
    phase0=0.0,
    edge_frac=1e-4,
)
    t_center_f  = Float64(t_center)
    duration_f  = Float64(duration)
    amp_c       = ComplexF64(amp)
    bandwidth_f = Float64(bandwidth)
    n_f         = Float64(n)
    omega0_f    = Float64(omega0)
    chirp_s     = Float64(chirp_sign)
    phase0_f    = Float64(phase0)
    edge_frac_f = Float64(edge_frac)

    t_start = t_center_f - duration_f/2
    edge    = max(duration_f * edge_frac_f, eps(Float64))

    return function (t)
        τ = t - t_start

        gate = 0.5 * (
            tanh((t - t_start) / edge) -
            tanh((t - (t_start + duration_f)) / edge)
        )

        envelope = amp_c * (
            1 - abs(sin(pi * (τ - duration_f/2) / duration_f))^n_f
        )

        phase = phase0_f +
                (omega0_f - chirp_s * bandwidth_f/2) * τ +
                0.5 * chirp_s * (bandwidth_f / duration_f) * τ^2

        return gate * envelope * exp(1im * phase)
    end
end

function custom_drive(f)
    return t -> ComplexF64(f(t))
end

function build_drive_pulse(cfg)
    if cfg.kind == :gaussian
        return gaussian_drive(
            t0    = cfg.t0,
            sigma = cfg.sigma,
            amp   = cfg.amp,
            omega = cfg.omega,
            phase = cfg.phase,
        )

    elseif cfg.kind == :wurst
        return wurst_drive(
            t_center   = cfg.t_center,
            duration   = cfg.duration,
            amp        = cfg.amp,
            bandwidth  = cfg.bandwidth,
            n          = cfg.n,
            omega0     = cfg.omega0,
            chirp_sign = cfg.chirp_sign,
            phase0     = cfg.phase0,
            edge_frac  = cfg.edge_frac,
        )

    elseif cfg.kind == :custom
        return custom_drive(cfg.f)

    else
        error("Unknown pulse kind: $(cfg.kind)")
    end
end

function build_E_of_t(PULSE_CONFIG)
    drive_pulses = tuple((build_drive_pulse(cfg) for cfg in PULSE_CONFIG)...)

    return function E_of_t(t)
        Et = 0.0 + 0.0im

        @inbounds for pulse in drive_pulses
            Et += pulse(t)
        end

        return Et
    end
end

"""
Samples E_of_t at `N` evenly spaced points over `[0, t_end]` and returns
`(Re[E(t)], Im[E(t)])` as two vectors, unchanged from before.

If `savepath` is given, the same samples are additionally written to
`savepath` as an (N, 2) matrix `[Re[E(t)] Im[E(t)]]` -- one row per sample,
in sampling order -- via a plain, dependency-free CSV (see `save_E_samples`),
directly readable from Python. The total pulse length `t_end` is recorded
in the file too (see `save_E_samples` for the exact format/read pattern).
"""
function sample_E_of_t(E_of_t, t_end, N=10000; savepath=nothing)
    t_points = range(0.0, t_end; length=N)
    Et = ComplexF64[E_of_t(t) for t in t_points]
    Ex = real.(Et)
    Ep = imag.(Et)

    if savepath !== nothing
        save_E_samples(hcat(Ex, Ep), t_end, savepath)
    end

    return Ex, Ep
end

"""
Writes an (N, 2) matrix `[Re[E(t)] Im[E(t)]]` (as produced by
`sample_E_of_t`) to `path` as a plain CSV -- no Julia-specific type tags
(unlike JLD2, whose compound HDF5 types aren't readable via plain h5py/
numpy), so it opens directly from Python.

`t_end` (seconds -- the same value passed to `sample_E_of_t`) is recorded
as the total pulse length in microseconds, on its own `#` comment line
before the column header:

    # t_end_us,1100.0
    Re,Im
    1.234e-05,0.0
    ...

Read the sample matrix back with e.g.
`numpy.loadtxt(path, delimiter=",", skiprows=2)` (skiprows=2 to skip both
the metadata line and the "Re,Im" column header); read `t_end_us` back
with e.g. `float(open(path).readline().split(",")[1])`.
"""
function save_E_samples(E_matrix::AbstractMatrix{<:Real}, t_end::Real, path::AbstractString)
    size(E_matrix, 2) == 2 || error(
        "save_E_samples expects an (N, 2) matrix [Re[E(t)] Im[E(t)]], " *
        "got size $(size(E_matrix))."
    )

    dir = dirname(path)
    isempty(dir) || mkpath(dir)

    open(path, "w") do io
        @printf(io, "# t_end_us,%.17g\n", t_end * 1e6)
        println(io, "Re,Im")
        for row in eachrow(E_matrix)
            @printf(io, "%.17g,%.17g\n", row[1], row[2])
        end
    end

    return path
end

"""
    load_E_samples(path) -> (t_end, Ex, Ep)

Reads a `_pulsemat.csv` file back in -- the exact read-side counterpart of
[`save_E_samples`](@ref)/[`sample_E_of_t`](@ref): the `# t_end_us,<value>`
metadata line, then the `Re,Im` header, then one `Re,Im` row per sample, in
the same order they were written. `t_end` is returned in SECONDS (the file
itself stores microseconds, matching `save_E_samples`'s own `t_end*1e6`
convention) -- the corresponding sample TIMES are `range(0.0, t_end;
length=length(Ex))`, since that is exactly the grid `sample_E_of_t` wrote
them on; this function does not reconstruct that vector itself; a caller
needing it just calls `collect(range(0.0, t_end; length=length(Ex)))`.
Dependency-free (no CSV.jl/DelimitedFiles), matching the writer's own
hand-rolled, Python-readable format.
"""
function load_E_samples(path::AbstractString)
    lines = readlines(path)
    length(lines) >= 2 || error("$path has fewer than the expected 2 header lines.")
    startswith(lines[1], "# t_end_us,") || error(
        "$path's first line does not match the expected \"# t_end_us,<value>\" " *
        "metadata format written by save_E_samples; got: $(lines[1])"
    )
    t_end_us = parse(Float64, split(lines[1], ",")[2])
    lines[2] == "Re,Im" || error(
        "$path's second line is not the expected \"Re,Im\" column header; got: $(lines[2])"
    )
    N = length(lines) - 2
    Ex = Vector{Float64}(undef, N)
    Ep = Vector{Float64}(undef, N)
    @inbounds for j in 1:N
        parts = split(lines[2+j], ',')
        length(parts) == 2 || error(
            "$path line $(2+j) does not have exactly 2 comma-separated fields: $(lines[2+j])"
        )
        Ex[j] = parse(Float64, parts[1])
        Ep[j] = parse(Float64, parts[2])
    end
    return t_end_us * 1e-6, Ex, Ep
end

"""
Saves `data` to `filename` via JLD2 -- exactly `@save filename data`, the
convention every `run_sim_*` function already uses -- and, alongside it,
regenerates that run's pulse-sample matrix: rebuilds `E_of_t` from
`data.PULSE_CONFIG` and calls `sample_E_of_t` over
`data.SIM_SETTING.Ttotal` at `data.SIM_SETTING.Nt_save` points, saving the
result next to `filename` as `<basename>_pulsemat.csv` (same naming
convention as `scripts/sample_pulse_from_run.jl`, which does this in batch
for already-saved runs -- a run saved through this function no longer needs
to be picked up by that script's gap-filling scan, since its
`*_pulsemat.csv` is written right here, at save time).

Requires `data.PULSE_CONFIG` and `data.SIM_SETTING` (with `Ttotal`/
`Nt_save`) -- i.e. any `data` NamedTuple built the way `run_sim_1st_order`/
`run_sim_2nd_order`/`mgpu_run_sim_2nd_order` already build theirs.
"""
function save_run_data(filename::AbstractString, data)
    dir = dirname(filename)
    isempty(dir) || mkpath(dir)

    @save filename data

    E_of_t = build_E_of_t(data.PULSE_CONFIG)

    pulsemat_path = filename[1:end-length(".jld2")] * "_pulsemat.csv"
    sample_E_of_t(
        E_of_t,
        data.SIM_SETTING.Ttotal,
        data.SIM_SETTING.Nt_save;
        savepath = pulsemat_path,
    )

    return filename
end

# ============================================================
# PULSE PLOTTING
# ============================================================

"""
Plot the amplitude of a generated pulse E_of_t versus time.

`t_end` and `freq` are used to sample E_of_t (see `sample_E_of_t`).
Set `plot_components=true` to overlay Re[E(t)] and Im[E(t)] alongside |E(t)|.
Set `savepath` to a file path to also save the figure.
"""
function plot_E_of_t(
    E_of_t,
    t_end,
    freq;
    plot_components=true,
    time_unit=:us,
    title="Applied drive",
    savepath=nothing,
)
    Ex, Ep = sample_E_of_t(E_of_t, t_end, freq)
    N = length(Ex)
    t_points = range(0.0, t_end; length=N)

    scale = time_unit == :us ? 1e6 :
            time_unit == :ns ? 1e9 :
            time_unit == :s  ? 1.0 :
            error("Unknown time_unit: $time_unit (use :s, :us, or :ns)")

    xlab = time_unit == :us ? "time (μs)" :
           time_unit == :ns ? "time (ns)" : "time (s)"

    t_plot = t_points .* scale
    E_abs  = sqrt.(Ex.^2 .+ Ep.^2)

    plt = plot(
        t_plot,
        E_abs;
        label="|E(t)|",
        xlabel=xlab,
        ylabel="Amplitude",
        title=title,
    )

    if plot_components
        plot!(plt, t_plot, Ex; label="Re[E(t)]")
        plot!(plt, t_plot, Ep; label="Im[E(t)]")
    end

    if savepath !== nothing
        savefig(plt, savepath)
    end

    return plt
end