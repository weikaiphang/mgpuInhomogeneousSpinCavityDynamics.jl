using InhomogeneousSpinCavityDynamics

############################################################
# USER SETTINGS
############################################################

INPUTS = joinpath(@__DIR__, "..", "data")      # Input path: one .jld2 file, one folder, or a list of files/folders.

Tc_us = 65.0                                              # Width of each noise-analysis time window, in μs.
Nkeep = 5000                                              # Maximum number of saved time points retained in each window.

centers_us = [35.0, 115.0]                                # Center times of the noise windows, in μs.
window_names = ["ASE window", "RASE window"]              # Names used when printing the results for each window.

batch_size = 1000                                         # Number of starting-time points processed simultaneously on the GPU.

############################################################
# FIND JLD2 FILES
############################################################

function find_jld2_files(inputs)
    paths = inputs isa AbstractString ? [inputs] : inputs
    files = String[]

    for path in paths
        if isfile(path) && endswith(lowercase(path), ".jld2")
            push!(files, path)
        elseif isdir(path)
            append!(
                files,
                filter(
                    file -> endswith(lowercase(file), ".jld2"),
                    readdir(path; join=true),
                ),
            )
        else
            @warn "Skipping invalid path" path
        end
    end

    return sort!(unique!(files))
end

files = find_jld2_files(INPUTS)
isempty(files) && error("No .jld2 files were found.")

############################################################
# COMPUTE NOISE
############################################################

for file in files
    println("\n", "="^60)
    println("File: ", basename(file))
    println("="^60)

    noise_data = load_noise_data(file)

    results = compute_noise_windows(
        noise_data,
        centers_us,
        Tc_us;
        Nkeep=Nkeep,
        batch_size=batch_size,
    )

    for (name, result) in zip(window_names, results)
        print_noise_result(name, result)
    end
end