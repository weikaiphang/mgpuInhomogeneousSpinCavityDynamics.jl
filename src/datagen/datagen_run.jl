# PINN first-order datagen orchestrator.
# Does not modify package physics. Run with:
#
#   julia --project=. src/datagen/datagen_run.jl --phase configs
#   julia --project=. src/datagen/datagen_run.jl --phase configs --dry-run
#   julia --project=. src/datagen/datagen_run.jl --phase simulate
#   julia --project=. src/datagen/datagen_run.jl --phase all
#
# Optional: --limit N   --start IDX   --stop IDX   --no-skip
# (--start/--stop are 1-based indices into the sorted configs listing)

module DataGen

using InhomogeneousSpinCavityDynamics
using JLD2
using JSON3
using Printf
using LinearAlgebra

const ISC = InhomogeneousSpinCavityDynamics

include("datagen_io.jl")
include("datagen_system.jl")
include("datagen_pulse.jl")
include("datagen_execute.jl")
include("datagen_catalog.jl")

function parse_args(args)
    phase = "configs"
    dry_run = false
    limit = 0
    start_id = 1
    stop_id = 0
    skip_existing = true

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--phase" && i < length(args)
            phase = args[i + 1]
            i += 2
        elseif a == "--dry-run"
            dry_run = true
            i += 1
        elseif a == "--limit" && i < length(args)
            limit = parse(Int, args[i + 1])
            i += 2
        elseif a == "--start" && i < length(args)
            start_id = parse(Int, args[i + 1])
            i += 2
        elseif a == "--stop" && i < length(args)
            stop_id = parse(Int, args[i + 1])
            i += 2
        elseif a == "--no-skip"
            skip_existing = false
            i += 1
        else
            error("Unknown argument: $a")
        end
    end

    phase in ("configs", "simulate", "all") || error(
        "--phase must be configs, simulate, or all, got $(phase)."
    )
    return (
        phase = phase,
        dry_run = dry_run,
        limit = limit,
        start_id = start_id,
        stop_id = stop_id,
        skip_existing = skip_existing,
    )
end

function main(args)
    opt = parse_args(args)
    println("PINN datagen  run_rules_version=$(RUN_RULES_VERSION)")
    println("Output root: $DATAGEN_ROOT")

    if opt.phase in ("configs", "all")
        phase_configs(; dry_run = opt.dry_run, limit = opt.limit)
    end
    if opt.phase in ("simulate", "all")
        opt.dry_run && opt.phase == "simulate" && error(
            "--dry-run is only for --phase configs."
        )
        opt.dry_run && return
        phase_simulate(;
            skip_existing = opt.skip_existing,
            start_id = opt.start_id,
            stop_id = opt.stop_id,
        )
    end
    return nothing
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    DataGen.main(ARGS)
end
