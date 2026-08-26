# ============================================================
# Datagen paths, manifest, and JLD2 helpers.
# ============================================================

const DATAGEN_ROOT = normpath(joinpath(@__DIR__, "..", "..", "data", "datagen"))
const DATAGEN_CONFIG_DIR = joinpath(DATAGEN_ROOT, "configs")
const DATAGEN_RESULT_DIR = joinpath(DATAGEN_ROOT, "results")
const DATAGEN_MANIFEST = joinpath(DATAGEN_ROOT, "manifest.json")

const RUN_RULES_VERSION = "2"

function ensure_datagen_dirs()
    mkpath(DATAGEN_CONFIG_DIR)
    mkpath(DATAGEN_RESULT_DIR)
    return nothing
end

function git_commit_stamp()
    repo = normpath(joinpath(@__DIR__, "..", ".."))
    try
        return readchomp(`git -C $repo rev-parse HEAD`)
    catch
        return "unknown"
    end
end

function simulconfig_path(run_id::Integer)
    return joinpath(DATAGEN_CONFIG_DIR, @sprintf("run_%06d_simulconfig.jld2", run_id))
end

function result_path(run_id::Integer, ic::Symbol)
    return joinpath(DATAGEN_RESULT_DIR, @sprintf("run_%06d_%s.jld2", run_id, ic))
end

function pulsemat_path(run_id::Integer, ic::Symbol)
    return joinpath(DATAGEN_RESULT_DIR, @sprintf("run_%06d_%s_pulsemat.csv", run_id, ic))
end

function load_manifest()
    if isfile(DATAGEN_MANIFEST)
        return Dict{String, Any}(
            String(k) => v
            for (k, v) in JSON3.read(read(DATAGEN_MANIFEST, String))
        )
    else
        return Dict{String, Any}()
    end
end

function save_manifest(manifest)
    ensure_datagen_dirs()
    open(DATAGEN_MANIFEST, "w") do io
        JSON3.write(io, manifest)
    end
    return nothing
end

function save_simulconfig(path, SYSTEM_CONFIG, PULSE_SPEC, metadata)
    ensure_datagen_dirs()
    JLD2.jldsave(
        path;
        SYSTEM_CONFIG = SYSTEM_CONFIG,
        PULSE_SPEC = PULSE_SPEC,
        metadata = metadata,
    )
    return path
end

function load_simulconfig(path)
    raw = JLD2.load(path)
    return (
        SYSTEM_CONFIG = raw["SYSTEM_CONFIG"],
        PULSE_SPEC = raw["PULSE_SPEC"],
        metadata = raw["metadata"],
    )
end
