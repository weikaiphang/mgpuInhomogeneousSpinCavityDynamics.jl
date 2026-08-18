using InhomogeneousSpinCavityDynamics
using Plots
using Measures

# ============================================================
# 1) USER SETTINGS
# ============================================================

# A file, a folder, or multiple files/folders.
INPUTS = joinpath(@__DIR__, "..", "data")

# Examples:
# INPUTS = joinpath(@__DIR__, "..", "data", "fig_3_d")
# INPUTS = ["data/run1.jld2", "data/run2.jld2"]
# INPUTS = ["data/folder1", "data/folder2", "data/run3.jld2"]

Tpi_us = 75.0
T1_us = 25.0
T2_us = 75.0

PLOT_CORRELATIONS = true
SAVE_PLOTS = false

FIGDIR = joinpath(@__DIR__, "..", "fig", "correlations")

# ============================================================
# 2) CORRELATION SETTINGS
# ============================================================

settings = CorrelationSettings(
    tpi_us = Tpi_us,

    ASE_window_rel_us = (-T2_us, -T1_us),
    RASE_window_rel_us = (T1_us, T2_us),

    Nkeep_ASE = 5000,
    Nkeep_RASE = 5000,

    tA_choices_rel_us = [-50.0],
    tR_choices_rel_us = [50.0],

    tau_min_us = -20.0,
    tau_max_us = 20.0,
    dtau_us = 0.01,

    # Used only for adaptive Tsit5 in ASE-RASE.
    reltol = 1e-8,
    abstol = 1e-8,
)

# ============================================================
# 3) FILE HELPERS
# ============================================================

function find_input_files(inputs)
    paths = inputs isa AbstractString ? (inputs,) : inputs
    files = String[]

    for path in paths
        if isfile(path)
            endswith(lowercase(path), ".jld2") ||
                error("Input file must have extension .jld2: $path")

            push!(files, abspath(path))

        elseif isdir(path)
            append!(
                files,
                filter(
                    file -> isfile(file) &&
                            endswith(lowercase(file), ".jld2"),
                    readdir(path; join = true),
                ),
            )

        else
            error("Input does not exist: $path")
        end
    end

    # Do not process previously generated correlation files.
    filter!(
        file -> !endswith(lowercase(file), "_correlation.jld2"),
        files,
    )

    files = sort!(unique(files))
    isempty(files) && error("No input .jld2 files found.")

    return files
end

function correlation_output_file(input_file)
    stem = splitext(basename(input_file))[1]

    return joinpath(
        dirname(input_file),
        "correlation_data",
        stem * "_correlation.jld2",
    )
end

# ============================================================
# 4) PLOTTING
# ============================================================

if PLOT_CORRELATIONS
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
        bottom_margin = 3mm,
    )
end

function plot_correlation(
    x,
    correlations,
    fixed_times;
    scan_name,
    fixed_name,
    correlation_name,
    title,
)
    plt = plot(
        xlabel = "$scan_name time relative to π pulse (μs)",
        ylabel = "Re[$correlation_name]",
        title = title,
        legend = :bottomright,
    )

    for i in eachindex(fixed_times)
        plot!(
            plt,
            x,
            real.(correlations[:, i]);
            label = "fixed $fixed_name t = " *
                    "$(round(fixed_times[i]; digits = 3)) μs",
        )
    end

    return plt
end

function plot_all_correlations(data, input_file)
    stem = splitext(basename(input_file))[1]
    output_dir = joinpath(FIGDIR, stem)

    SAVE_PLOTS && mkpath(output_dir)

    specifications = (
        (
            data.tR_scan_rel_us,
            data.ASE_RASE.Cxx,
            data.tA_fixed_rel_us,
            "RASE",
            "ASE",
            "Cxx",
            "ASE–RASE xx correlation",
            "ASE_RASE_Cxx.png",
        ),
        (
            data.tR_scan_rel_us,
            data.ASE_RASE.Cpp,
            data.tA_fixed_rel_us,
            "RASE",
            "ASE",
            "Cpp",
            "ASE–RASE pp correlation",
            "ASE_RASE_Cpp.png",
        ),
        (
            data.tA_scan_rel_us,
            data.ASE_ASE.Cxx,
            data.tA_fixed_rel_us,
            "ASE",
            "ASE",
            "Cxx",
            "ASE–ASE xx correlation",
            "ASE_ASE_Cxx.png",
        ),
        (
            data.tA_scan_rel_us,
            data.ASE_ASE.Cpp,
            data.tA_fixed_rel_us,
            "ASE",
            "ASE",
            "Cpp",
            "ASE–ASE pp correlation",
            "ASE_ASE_Cpp.png",
        ),
        (
            data.tR_scan_rel_us,
            data.RASE_RASE.Cxx,
            data.tR_fixed_rel_us,
            "RASE",
            "RASE",
            "Cxx",
            "RASE–RASE xx correlation",
            "RASE_RASE_Cxx.png",
        ),
        (
            data.tR_scan_rel_us,
            data.RASE_RASE.Cpp,
            data.tR_fixed_rel_us,
            "RASE",
            "RASE",
            "Cpp",
            "RASE–RASE pp correlation",
            "RASE_RASE_Cpp.png",
        ),
    )

    for (
        x,
        correlations,
        fixed_times,
        scan_name,
        fixed_name,
        correlation_name,
        title,
        filename,
    ) in specifications

        plt = plot_correlation(
            x,
            correlations,
            fixed_times;
            scan_name,
            fixed_name,
            correlation_name,
            title,
        )

        display(plt)

        if SAVE_PLOTS
            output_path = joinpath(output_dir, filename)
            savefig(plt, output_path)
            println("Saved plot to: ", output_path)
        end
    end

    return nothing
end

# ============================================================
# 5) PROCESS ONE FILE
# ============================================================

function process_file(input_file, settings)
    println("\nProcessing: ", input_file)

    workspace = load_correlation_workspace(input_file)

    correlation_data = compute_ase_rase_correlations_gpu(
        workspace,
        settings;
        verbose = true,
    )

    output_file = correlation_output_file(input_file)

    save_correlation_results(
        output_file,
        correlation_data,
    )

    println("Saved correlation data to: ", output_file)

    PLOT_CORRELATIONS &&
        plot_all_correlations(correlation_data, input_file)

    return nothing
end

# ============================================================
# 6) RUN ALL FILES
# ============================================================

input_files = find_input_files(INPUTS)

println("Found $(length(input_files)) input file(s).")

for input_file in input_files
    process_file(input_file, settings)
end

println("\nFinished all correlation calculations.")
