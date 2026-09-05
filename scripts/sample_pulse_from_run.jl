using InhomogeneousSpinCavityDynamics
using JLD2

ROOT_DIR   = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "data")
N_OVERRIDE = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : nothing

isdir(ROOT_DIR) || error("No such directory: $ROOT_DIR")


function find_jld2_files(root)
    paths = String[]

    for (dir, _, files) in walkdir(root)
        for f in files
            endswith(f, ".jld2") && push!(paths, joinpath(dir, f))
        end
    end

    return sort(paths)
end

pulsemat_path(jld2_path) = jld2_path[1:end-length(".jld2")] * "_pulsemat.csv"

jld2_files = find_jld2_files(ROOT_DIR)

println("Found $(length(jld2_files)) .jld2 file(s) under $ROOT_DIR")
println()


function generate_missing_pulse_matrices(jld2_files, N_OVERRIDE)
    n_generated = 0
    n_skipped   = 0
    n_failed    = 0

    for jld2_path in jld2_files
        outpath = pulsemat_path(jld2_path)

        if isfile(outpath)
            println("[skip] $outpath already exists")
            n_skipped += 1
            continue
        end

        try
            @load jld2_path data

            E_of_t = build_E_of_t(data.PULSE_CONFIG)
            Ttotal = data.SIM_SETTING.Ttotal
            N = N_OVERRIDE === nothing ? data.SIM_SETTING.Nt_save : N_OVERRIDE

            sample_E_of_t(E_of_t, Ttotal, N; savepath=outpath)

            println("[ok]   $jld2_path -> $outpath  (N=$N)")
            n_generated += 1

        catch err
            println("[fail] $jld2_path : $(sprint(showerror, err))")
            n_failed += 1
        end
    end

    println()
    println(
        "Done: $n_generated generated, $n_skipped skipped " *
        "(already existed), $n_failed failed " *
        "(out of $(length(jld2_files)) total).",
    )

    return nothing
end

generate_missing_pulse_matrices(jld2_files, N_OVERRIDE)

nothing
