# PINN first-order datagen orchestrator.
# Does not modify package physics. Run with:
#
#   julia --project=. src/datagen/datagen_run.jl --phase configs
#   julia --project=. src/datagen/datagen_run.jl --phase configs --dry-run
#   julia --project=. src/datagen/datagen_run.jl --phase simulate
#   julia --project=. src/datagen/datagen_run.jl --phase all
#
# Configs:  --dry-run   --limit N
# Simulate: --start IDX --stop IDX --limit N --no-skip
#   --default-conditions ground|equatorial|both   (default both)
#   --M-cap N          (default 60000)
#   --M-g-cap N        (default 30)
#   --M-sizing default|N
#   --NT-save N        (default 5001)
# (--start/--stop are 1-based indices into the sorted configs listing)

module DataGen

using InhomogeneousSpinCavityDynamics
using JLD2
using JSON3
using Printf

const ISC = InhomogeneousSpinCavityDynamics

include("datagen_io.jl")
include("datagen_system.jl")
include("datagen_pulse.jl")
include("datagen_execute.jl")
include("datagen_catalog.jl")

function _need_value(args, i, flag)
    i < length(args) || error("$flag requires a value.")
    return args[i + 1], i + 2
end

function parse_args(args)
    phase = "configs"
    dry_run = false
    limit = 0
    start_id = 1
    stop_id = 0
    skip_existing = true
    conditions = "both"
    M_cap = RULE_M_CAP
    M_g_max = RULE_M_G_MAX
    m_sizing = "default"
    Nt_save = RULE_NT_SAVE

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--phase"
            phase, i = _need_value(args, i, a)
        elseif a == "--dry-run"
            dry_run = true
            i += 1
        elseif a == "--limit"
            val, i = _need_value(args, i, a)
            limit = parse(Int, val)
        elseif a == "--start"
            val, i = _need_value(args, i, a)
            start_id = parse(Int, val)
        elseif a == "--stop"
            val, i = _need_value(args, i, a)
            stop_id = parse(Int, val)
        elseif a == "--no-skip"
            skip_existing = false
            i += 1
        elseif a == "--default-conditions" || a == "--conditions"
            conditions, i = _need_value(args, i, a)
        elseif a == "--M-cap"
            val, i = _need_value(args, i, a)
            M_cap = parse(Int, val)
        elseif a == "--M-g-cap"
            val, i = _need_value(args, i, a)
            M_g_max = parse(Int, val)
        elseif a == "--M-sizing"
            m_sizing, i = _need_value(args, i, a)
        elseif a == "--NT-save"
            val, i = _need_value(args, i, a)
            Nt_save = parse(Int, val)
        else
            error("Unknown argument: $a")
        end
    end

    phase in ("configs", "simulate", "all") || error(
        "--phase must be configs, simulate, or all, got $(phase)."
    )
    dry_run && phase == "simulate" && error(
        "--dry-run is only for --phase configs."
    )
    limit >= 0 || error("--limit must be non-negative, got $limit.")
    start_id >= 1 || error("--start must be >= 1, got $start_id.")
    stop_id >= 0 || error("--stop must be >= 0 (0 = no stop), got $stop_id.")
    stop_id == 0 || stop_id >= start_id || error(
        "--stop=$stop_id is before --start=$start_id."
    )
    return (
        phase = phase,
        dry_run = dry_run,
        limit = limit,
        start_id = start_id,
        stop_id = stop_id,
        skip_existing = skip_existing,
        run = make_run_params(;
            ics = parse_default_conditions(conditions),
            M_cap = M_cap,
            M_g_max = M_g_max,
            n_sizes = parse_m_sizing(m_sizing),
            Nt_save = Nt_save,
        ),
    )
end

function main(args)
    opt = parse_args(args)
    println("PINN datagen  run_rules_version=$(RUN_RULES_VERSION)")
    println("Output root: $DATAGEN_ROOT")
    if opt.phase in ("simulate", "all") && !(opt.dry_run && opt.phase == "all")
        r = opt.run
        println(
            "Run params: conditions=$(r.ics)  M-cap=$(r.M_cap)  M-g-cap=$(r.M_g_max)  " *
            "M-sizing=$(r.n_sizes == 1 ? "default" : r.n_sizes)  NT-save=$(r.Nt_save)"
        )
    end

    if opt.phase in ("configs", "all")
        phase_configs(; dry_run = opt.dry_run, limit = opt.limit)
    end
    if opt.phase in ("simulate", "all")
        opt.dry_run && return
        phase_simulate(
            opt.run;
            skip_existing = opt.skip_existing,
            start_id = opt.start_id,
            stop_id = opt.stop_id,
            limit = opt.limit,
        )
    end
    return nothing
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    DataGen.main(ARGS)
end
